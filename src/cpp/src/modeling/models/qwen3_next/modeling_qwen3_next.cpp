// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_next/modeling_qwen3_next.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_set>

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>
#include <openvino/op/tensor_iterator.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>
#include <ov_ops/rms.hpp>

#include "modeling/ops/kv_cache.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int extract_layer_index_from_weight_name(const std::string& name) {
    static const std::string bracket_prefix = "model.layers[";
    static const std::string dot_prefix = "model.layers.";

    if (starts_with(name, bracket_prefix)) {
        size_t pos = bracket_prefix.size();
        size_t end = pos;
        while (end < name.size() && std::isdigit(static_cast<unsigned char>(name[end]))) {
            end++;
        }
        if (end > pos && end < name.size() && name[end] == ']') {
            return std::stoi(name.substr(pos, end - pos));
        }
    }

    if (starts_with(name, dot_prefix)) {
        size_t pos = dot_prefix.size();
        size_t end = pos;
        while (end < name.size() && std::isdigit(static_cast<unsigned char>(name[end]))) {
            end++;
        }
        if (end > pos && end < name.size() && name[end] == '.') {
            return std::stoi(name.substr(pos, end - pos));
        }
    }

    return -1;
}

std::unordered_set<int32_t> make_index_set(const std::vector<int32_t>& idx) {
    return std::unordered_set<int32_t>(idx.begin(), idx.end());
}

std::vector<std::string> build_default_layer_types(int32_t num_layers, int32_t interval) {
    const int32_t safe_interval = interval > 0 ? interval : 4;
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i) {
        out.push_back(((i + 1) % safe_interval) == 0 ? "full_attention" : "linear_attention");
    }
    return out;
}

std::optional<int32_t> get_qwen3_next_layer_limit_from_env() {
    static constexpr const char* kEnvName = "OV_GENAI_QWEN3_NEXT_NUM_LAYERS";
    const char* raw = std::getenv(kEnvName);
    if (!raw || raw[0] == '\0') {
        return std::nullopt;
    }

    try {
        const int64_t parsed = std::stoll(raw);
        if (parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
            OPENVINO_THROW(kEnvName, " must be an integer in [0, INT32_MAX], got: ", raw);
        }
        return static_cast<int32_t>(parsed);
    } catch (const std::exception&) {
        OPENVINO_THROW(kEnvName, " must be an integer in [0, INT32_MAX], got: ", raw);
    }
}

bool use_fused_conv_op() {
    const char* raw = std::getenv("OV_GENAI_USE_FUSED_CONV_OP");
    if (!raw || raw[0] == '\0')
        return true;  // enabled by default
    return std::string(raw) != "0";
}

ov::genai::modeling::models::Qwen3NextConfig apply_qwen3_next_layer_limit(
    const ov::genai::modeling::models::Qwen3NextConfig& input_cfg) {
    auto cfg = input_cfg;
    const auto env_limit = get_qwen3_next_layer_limit_from_env();
    if (!env_limit.has_value()) {
        return cfg;
    }

    const int32_t limited_layers = std::clamp(*env_limit, 0, cfg.num_hidden_layers);
    cfg.num_hidden_layers = limited_layers;

    if (!cfg.layer_types.empty() && cfg.layer_types.size() >= static_cast<size_t>(limited_layers)) {
        cfg.layer_types.resize(static_cast<size_t>(limited_layers));
    }

    if (!cfg.mlp_only_layers.empty()) {
        std::vector<int32_t> filtered;
        filtered.reserve(cfg.mlp_only_layers.size());
        for (int32_t idx : cfg.mlp_only_layers) {
            if (idx >= 0 && idx < limited_layers) {
                filtered.push_back(idx);
            }
        }
        cfg.mlp_only_layers = std::move(filtered);
    }
    return cfg;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

Qwen3NextRMSNorm::Qwen3NextRMSNorm(BuilderContext& ctx, const std::string& name, float eps, Module* parent)
    : Module(name, ctx, parent),
      eps_(eps) {
    weight_param_ = &register_parameter("weight");
}

const Tensor& Qwen3NextRMSNorm::weight() const {
    if (!weight_param_) {
        OPENVINO_THROW("Qwen3NextRMSNorm weight parameter is not registered");
    }
    return weight_param_->value();
}

Tensor Qwen3NextRMSNorm::forward(const Tensor& x) const {
    auto x_f32 = x.to(ov::element::f32);
    auto gamma = 1.0f + weight().to(ov::element::f32);
    auto rms_node = std::make_shared<ov::op::internal::RMS>(
        x_f32.output(), gamma.output(), static_cast<double>(eps_), x.dtype());
    return Tensor(rms_node->output(0), x.context());
}

std::pair<Tensor, Tensor> Qwen3NextRMSNorm::forward(const Tensor& x, const Tensor& residual) const {
    auto sum = x + residual;
    return {forward(sum), sum};
}

Qwen3NextAttention::Qwen3NextAttention(BuilderContext& ctx,
                                       const std::string& name,
                                       const Qwen3NextConfig& cfg,
                                       Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads),
      head_dim_(cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / cfg.num_attention_heads)),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      rope_theta_(cfg.rope_theta),
      q_norm_(ctx, "q_norm", cfg.rms_norm_eps, this),
      k_norm_(ctx, "k_norm", cfg.rms_norm_eps, this) {
    if (num_heads_ <= 0 || head_dim_ <= 0 || num_kv_heads_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3Next attention configuration");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
    }
    rotary_dim_ = static_cast<int32_t>(std::floor(static_cast<float>(head_dim_) * cfg.partial_rotary_factor));
    rotary_dim_ = std::max<int32_t>(0, std::min<int32_t>(rotary_dim_, head_dim_));
    if ((rotary_dim_ % 2) != 0) {
        rotary_dim_ -= 1;
    }

    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");
}

const Tensor& Qwen3NextAttention::q_proj_weight() const {
    if (!q_proj_param_) {
        OPENVINO_THROW("Qwen3NextAttention q_proj parameter is not registered");
    }
    return q_proj_param_->value();
}

const Tensor& Qwen3NextAttention::k_proj_weight() const {
    if (!k_proj_param_) {
        OPENVINO_THROW("Qwen3NextAttention k_proj parameter is not registered");
    }
    return k_proj_param_->value();
}

const Tensor& Qwen3NextAttention::v_proj_weight() const {
    if (!v_proj_param_) {
        OPENVINO_THROW("Qwen3NextAttention v_proj parameter is not registered");
    }
    return v_proj_param_->value();
}

const Tensor& Qwen3NextAttention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("Qwen3NextAttention o_proj parameter is not registered");
    }
    return o_proj_param_->value();
}

Tensor Qwen3NextAttention::forward(const Tensor& hidden_states,
                                   const Tensor& beam_idx,
                                   const Tensor& rope_cos,
                                   const Tensor& rope_sin,
                                   const Tensor* attention_mask,
                                   const Tensor* precomputed_sdpa_mask) const {
    auto* policy = &ctx().op_policy();
    auto* op_ctx = hidden_states.context();

    auto q_linear = ops::linear(hidden_states, q_proj_weight()).reshape({0, 0, num_heads_, head_dim_ * 2});
    auto q_states = ops::slice(q_linear, 0, head_dim_, 1, 3);
    auto gate = ops::slice(q_linear, head_dim_, head_dim_ * 2, 1, 3).reshape({0, 0, num_heads_ * head_dim_});

    auto k_states = ops::linear(hidden_states, k_proj_weight()).reshape({0, 0, num_kv_heads_, head_dim_});
    auto v_states = ops::linear(hidden_states, v_proj_weight()).reshape({0, 0, num_kv_heads_, head_dim_});

    auto q_heads = q_norm_.forward(q_states).permute({0, 2, 1, 3});
    auto k_heads = k_norm_.forward(k_states).permute({0, 2, 1, 3});
    auto v_heads = v_states.permute({0, 2, 1, 3});

    if (rotary_dim_ > 0) {
        q_heads = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, rotary_dim_, policy, head_dim_);
        k_heads = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, rotary_dim_, policy, head_dim_);
    }

    const std::string cache_prefix = full_path().empty() ? name() : full_path();
    auto cached = ops::append_kv_cache(k_heads, v_heads, beam_idx, num_kv_heads_, head_dim_, cache_prefix, ctx());
    auto k_expanded = ops::llm::repeat_kv(cached.first, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(cached.second, num_heads_, num_kv_heads_, head_dim_);

    const Tensor* sdpa_mask = precomputed_sdpa_mask;
    std::optional<Tensor> local_mask;
    if (!sdpa_mask) {
        local_mask = attention_mask
                         ? ops::llm::build_kv_causal_mask_with_attention(q_heads, cached.first, *attention_mask)
                         : ops::llm::build_kv_causal_mask(q_heads, cached.first);
        sdpa_mask = &(*local_mask);
    }
    auto attn = ops::llm::sdpa(q_heads, k_expanded, v_expanded, scaling_, 3, sdpa_mask, false, policy);

    const int64_t attn_hidden = static_cast<int64_t>(num_heads_) * static_cast<int64_t>(head_dim_);
    auto merged = attn.permute({0, 2, 1, 3}).reshape({0, 0, attn_hidden});
    auto gate_sigmoid = Tensor(std::make_shared<ov::op::v0::Sigmoid>(gate.output()), op_ctx);
    auto gated = merged * gate_sigmoid;
    return ops::linear(gated, o_proj_weight());
}

Qwen3NextMLP::Qwen3NextMLP(BuilderContext& ctx,
                           const std::string& name,
                           const Qwen3NextConfig& cfg,
                           int32_t intermediate_size,
                           Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3Next MLP activation: ", cfg.hidden_act);
    }
    if (intermediate_size <= 0) {
        OPENVINO_THROW("Qwen3NextMLP intermediate size must be > 0");
    }
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& Qwen3NextMLP::gate_proj_weight() const {
    if (!gate_proj_param_) {
        OPENVINO_THROW("Qwen3NextMLP gate_proj parameter is not registered");
    }
    return gate_proj_param_->value();
}

const Tensor& Qwen3NextMLP::up_proj_weight() const {
    if (!up_proj_param_) {
        OPENVINO_THROW("Qwen3NextMLP up_proj parameter is not registered");
    }
    return up_proj_param_->value();
}

const Tensor& Qwen3NextMLP::down_proj_weight() const {
    if (!down_proj_param_) {
        OPENVINO_THROW("Qwen3NextMLP down_proj parameter is not registered");
    }
    return down_proj_param_->value();
}

Tensor Qwen3NextMLP::forward(const Tensor& x) const {
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

Qwen3NextSparseMoeBlock::Qwen3NextSparseMoeBlock(BuilderContext& ctx,
                                                 const std::string& name,
                                                 const Qwen3NextConfig& cfg,
                                                 Module* parent)
    : Module(name, ctx, parent),
      hidden_size_(cfg.hidden_size),
      expert_intermediate_size_(cfg.moe_intermediate_size),
      shared_intermediate_size_(cfg.shared_expert_intermediate_size),
      num_experts_(cfg.num_experts),
      top_k_(cfg.num_experts_per_tok > 0 ? cfg.num_experts_per_tok : 1),
      norm_topk_prob_(cfg.norm_topk_prob),
      group_size_((cfg.group_size > 0) ? static_cast<size_t>(cfg.group_size) : 128) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3Next MoE activation: ", cfg.hidden_act);
    }
    if (hidden_size_ <= 0 || expert_intermediate_size_ <= 0 || shared_intermediate_size_ <= 0 || num_experts_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3Next MoE configuration");
    }
    if (top_k_ <= 0 || top_k_ > num_experts_) {
        OPENVINO_THROW("Invalid Qwen3Next MoE top-k configuration");
    }

    gate_param_ = &register_parameter("gate.weight");
    shared_expert_gate_param_ = &register_parameter("shared_expert_gate.weight");
    shared_gate_proj_param_ = &register_parameter("shared_expert.gate_proj.weight");
    shared_up_proj_param_ = &register_parameter("shared_expert.up_proj.weight");
    shared_down_proj_param_ = &register_parameter("shared_expert.down_proj.weight");

    // Initialize expert parameter vectors
    gate_experts_param_.resize(static_cast<size_t>(num_experts_));
    up_experts_param_.resize(static_cast<size_t>(num_experts_));
    down_experts_param_.resize(static_cast<size_t>(num_experts_));
    
    // Initialize scales and zps vectors for quantized MoE weights
    gate_exps_scales_.resize(static_cast<size_t>(num_experts_));
    gate_exps_zps_.resize(static_cast<size_t>(num_experts_));
    up_exps_scales_.resize(static_cast<size_t>(num_experts_));
    up_exps_zps_.resize(static_cast<size_t>(num_experts_));
    down_exps_scales_.resize(static_cast<size_t>(num_experts_));
    down_exps_zps_.resize(static_cast<size_t>(num_experts_));

    for (int32_t i = 0; i < num_experts_; ++i) {
        const std::string prefix = "experts." + std::to_string(i) + ".";
        const size_t idx = static_cast<size_t>(i);
        
        gate_experts_param_[idx] = &register_parameter(prefix + "gate_proj.weight");
        up_experts_param_[idx] = &register_parameter(prefix + "up_proj.weight");
        down_experts_param_[idx] = &register_parameter(prefix + "down_proj.weight");
        
        // Set custom weight_loader for gate_proj to capture scales/zps
        gate_experts_param_[idx]->set_weight_loader([this, idx](WeightParameter& param,
                                                   weights::WeightSource& source,
                                                   weights::WeightFinalizer& finalizer,
                                                   const std::string& weight_name,
                                                   const std::optional<int>& shard_id) {
            (void)shard_id;
            if (!param.context()) {
                OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
            }
            auto weight = finalizer.finalize(weight_name, source, *param.context());
            param.bind(weight);
            if (weight.get_auxiliary("scales") != std::nullopt &&
                weight.get_auxiliary("zps") != std::nullopt) {
                gate_exps_scales_[idx] = weight.auxiliary.at("scales");
                gate_exps_zps_[idx] = weight.auxiliary.at("zps");
            }
        });

        // Set custom weight_loader for up_proj
        up_experts_param_[idx]->set_weight_loader([this, idx](WeightParameter& param,
                                                 weights::WeightSource& source,
                                                 weights::WeightFinalizer& finalizer,
                                                 const std::string& weight_name,
                                                 const std::optional<int>& shard_id) {
            (void)shard_id;
            if (!param.context()) {
                OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
            }
            auto weight = finalizer.finalize(weight_name, source, *param.context());
            param.bind(weight);
            if (weight.get_auxiliary("scales") != std::nullopt &&
                weight.get_auxiliary("zps") != std::nullopt) {
                up_exps_scales_[idx] = weight.auxiliary.at("scales");
                up_exps_zps_[idx] = weight.auxiliary.at("zps");
            }
        });

        // Set custom weight_loader for down_proj
        down_experts_param_[idx]->set_weight_loader([this, idx](WeightParameter& param,
                                                   weights::WeightSource& source,
                                                   weights::WeightFinalizer& finalizer,
                                                   const std::string& weight_name,
                                                   const std::optional<int>& shard_id) {
            (void)shard_id;
            if (!param.context()) {
                OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
            }
            auto weight = finalizer.finalize(weight_name, source, *param.context());
            param.bind(weight);
            if (weight.get_auxiliary("scales") != std::nullopt &&
                weight.get_auxiliary("zps") != std::nullopt) {
                down_exps_scales_[idx] = weight.auxiliary.at("scales");
                down_exps_zps_[idx] = weight.auxiliary.at("zps");
            }
        });
    }
}

const Tensor& Qwen3NextSparseMoeBlock::gate_weight() const {
    if (!gate_param_) {
        OPENVINO_THROW("Qwen3NextSparseMoeBlock gate parameter is not registered");
    }
    return gate_param_->value();
}

const Tensor& Qwen3NextSparseMoeBlock::shared_expert_gate_weight() const {
    if (!shared_expert_gate_param_) {
        OPENVINO_THROW("Qwen3NextSparseMoeBlock shared_expert_gate parameter is not registered");
    }
    return shared_expert_gate_param_->value();
}

const Tensor& Qwen3NextSparseMoeBlock::shared_gate_proj_weight() const {
    if (!shared_gate_proj_param_) {
        OPENVINO_THROW("Qwen3NextSparseMoeBlock shared gate_proj parameter is not registered");
    }
    return shared_gate_proj_param_->value();
}

const Tensor& Qwen3NextSparseMoeBlock::shared_up_proj_weight() const {
    if (!shared_up_proj_param_) {
        OPENVINO_THROW("Qwen3NextSparseMoeBlock shared up_proj parameter is not registered");
    }
    return shared_up_proj_param_->value();
}

const Tensor& Qwen3NextSparseMoeBlock::shared_down_proj_weight() const {
    if (!shared_down_proj_param_) {
        OPENVINO_THROW("Qwen3NextSparseMoeBlock shared down_proj parameter is not registered");
    }
    return shared_down_proj_param_->value();
}

Tensor Qwen3NextSparseMoeBlock::gate_expert_weights() const {
    std::vector<Tensor> ws;
    ws.reserve(gate_experts_param_.size());
    for (auto* p : gate_experts_param_) {
        ws.push_back(p->value());
    }
    auto result = ops::concat(ws, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::up_expert_weights() const {
    std::vector<Tensor> ws;
    ws.reserve(up_experts_param_.size());
    for (auto* p : up_experts_param_) {
        ws.push_back(p->value());
    }
    auto result = ops::concat(ws, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::down_expert_weights() const {
    std::vector<Tensor> ws;
    ws.reserve(down_experts_param_.size());
    for (auto* p : down_experts_param_) {
        ws.push_back(p->value());
    }
    auto result = ops::concat(ws, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::gate_exps_scales() const {
    auto result = ops::concat(gate_exps_scales_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::gate_exps_zps() const {
    auto result = ops::concat(gate_exps_zps_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::up_exps_scales() const {
    auto result = ops::concat(up_exps_scales_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::up_exps_zps() const {
    auto result = ops::concat(up_exps_zps_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::down_exps_scales() const {
    auto result = ops::concat(down_exps_scales_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::down_exps_zps() const {
    auto result = ops::concat(down_exps_zps_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::forward(const Tensor& hidden_states) const {
    auto* op_ctx = hidden_states.context();
    auto input_dtype = hidden_states.dtype();

    auto flat = hidden_states.reshape({-1, hidden_size_});
    auto flat_f32 = flat.to(ov::element::f32);
    
    // Use optimized MoE kernel for routed experts
    auto routed_out = ops::moe3gemm_fused_compressed(
        flat_f32,
        gate_weight(),
        gate_expert_weights(),
        gate_exps_scales(),
        gate_exps_zps(),
        up_expert_weights(),
        up_exps_scales(),
        up_exps_zps(),
        down_expert_weights(),
        down_exps_scales(),
        down_exps_zps(),
        hidden_size_,
        expert_intermediate_size_,
        num_experts_,
        top_k_,
        group_size_,
        ov::element::f16);

    // Shared expert computation (using standard linear ops)
    auto shared_gate = ops::linear(flat_f32, shared_gate_proj_weight().to(ov::element::f32));
    auto shared_up = ops::linear(flat_f32, shared_up_proj_weight().to(ov::element::f32));
    auto shared_hidden = ops::silu(shared_gate) * shared_up;
    auto shared_out = ops::linear(shared_hidden, shared_down_proj_weight().to(ov::element::f32));

    // Gated combination of routed and shared expert outputs
    auto shared_gate_logits = ops::linear(flat_f32, shared_expert_gate_weight().to(ov::element::f32));
    auto shared_gate_sigmoid = Tensor(std::make_shared<ov::op::v0::Sigmoid>(shared_gate_logits.output()), op_ctx);
    auto combined = routed_out + (shared_out * shared_gate_sigmoid);
    auto restored = combined.reshape(shape::of(hidden_states), false);
    return restored.to(input_dtype);
}

Qwen3NextGatedDeltaNet::Qwen3NextGatedDeltaNet(BuilderContext& ctx,
                                               const std::string& name,
                                               const Qwen3NextConfig& cfg,
                                               int32_t layer_idx,
                                               Module* parent)
    : Module(name, ctx, parent),
      layer_idx_(layer_idx),
      hidden_size_(cfg.hidden_size),
      num_v_heads_(cfg.linear_num_value_heads),
      num_k_heads_(cfg.linear_num_key_heads),
      head_k_dim_(cfg.linear_key_head_dim),
      head_v_dim_(cfg.linear_value_head_dim),
      key_dim_(head_k_dim_ * num_k_heads_),
      value_dim_(head_v_dim_ * num_v_heads_),
      conv_dim_(key_dim_ * 2 + value_dim_),
      conv_kernel_size_(cfg.linear_conv_kernel_dim),
      conv_state_size_(cfg.linear_conv_kernel_dim),
      eps_(cfg.rms_norm_eps) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3Next linear attention activation: ", cfg.hidden_act);
    }
    if (hidden_size_ <= 0 || num_v_heads_ <= 0 || num_k_heads_ <= 0 || head_k_dim_ <= 0 || head_v_dim_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3Next linear attention configuration");
    }
    if (conv_kernel_size_ <= 0) {
        OPENVINO_THROW("Qwen3Next linear_conv_kernel_dim must be > 0");
    }
    if ((num_v_heads_ % num_k_heads_) != 0) {
        OPENVINO_THROW("Qwen3Next linear_num_value_heads must be divisible by linear_num_key_heads");
    }

    in_proj_qkvz_param_ = &register_parameter("in_proj_qkvz.weight");
    in_proj_ba_param_ = &register_parameter("in_proj_ba.weight");
    conv1d_param_ = &register_parameter("conv1d.weight");
    a_log_param_ = &register_parameter("A_log");
    dt_bias_param_ = &register_parameter("dt_bias");
    norm_param_ = &register_parameter("norm.weight");
    out_proj_param_ = &register_parameter("out_proj.weight");
}

const Tensor& Qwen3NextGatedDeltaNet::in_proj_qkvz_weight() const {
    if (!in_proj_qkvz_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet in_proj_qkvz parameter is not registered");
    }
    return in_proj_qkvz_param_->value();
}

const Tensor& Qwen3NextGatedDeltaNet::in_proj_ba_weight() const {
    if (!in_proj_ba_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet in_proj_ba parameter is not registered");
    }
    return in_proj_ba_param_->value();
}

const Tensor& Qwen3NextGatedDeltaNet::conv1d_weight() const {
    if (!conv1d_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet conv1d parameter is not registered");
    }
    return conv1d_param_->value();
}

const Tensor& Qwen3NextGatedDeltaNet::a_log() const {
    if (!a_log_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet A_log parameter is not registered");
    }
    return a_log_param_->value();
}

const Tensor& Qwen3NextGatedDeltaNet::dt_bias() const {
    if (!dt_bias_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet dt_bias parameter is not registered");
    }
    return dt_bias_param_->value();
}

const Tensor& Qwen3NextGatedDeltaNet::out_proj_weight() const {
    if (!out_proj_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet out_proj parameter is not registered");
    }
    return out_proj_param_->value();
}

Tensor Qwen3NextGatedDeltaNet::apply_depthwise_causal_conv(const Tensor& mixed_qkv,
                                                           const Tensor& prev_conv_state,
                                                           Tensor* next_conv_state) const {
    auto* op_ctx = mixed_qkv.context();
    auto input_with_state = ops::concat({prev_conv_state, mixed_qkv}, 2);

    auto conv_weight = conv1d_weight().reshape({conv_dim_, 1, 1, conv_kernel_size_}, false);
    auto conv = std::make_shared<ov::op::v1::GroupConvolution>(input_with_state.output(),
                                                                conv_weight.output(),
                                                                ov::Strides{1},
                                                                ov::CoordinateDiff{0},
                                                                ov::CoordinateDiff{0},
                                                                ov::Strides{1});
    auto conv_act = ops::silu(Tensor(conv, op_ctx));

    auto seq_len = shape::dim(mixed_qkv, 2);
    auto conv_len = shape::dim(conv_act, 2);
    auto out_start = std::make_shared<ov::op::v1::Subtract>(conv_len, seq_len);
    auto out_slice = std::make_shared<ov::op::v8::Slice>(conv_act.output(),
                                                         out_start,
                                                         conv_len,
                                                         ops::const_vec(op_ctx, std::vector<int64_t>{1}),
                                                         ops::const_vec(op_ctx, std::vector<int64_t>{2}));

    auto total_len = shape::dim(input_with_state, 2);
    auto kernel = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(conv_state_size_)});
    auto state_start = std::make_shared<ov::op::v1::Subtract>(total_len, kernel);
    auto state_slice = std::make_shared<ov::op::v8::Slice>(input_with_state.output(),
                                                           state_start,
                                                           total_len,
                                                           ops::const_vec(op_ctx, std::vector<int64_t>{1}),
                                                           ops::const_vec(op_ctx, std::vector<int64_t>{2}));
    if (next_conv_state) {
        *next_conv_state = Tensor(state_slice, op_ctx);
    }
    return Tensor(out_slice, op_ctx);
}

Tensor Qwen3NextGatedDeltaNet::rms_norm_gated(const Tensor& x, const Tensor& z) const {
    auto x_f32 = x.to(ov::element::f32);
    auto z_f32 = z.to(ov::element::f32);
    auto var = x_f32.pow(2.0f).mean(-1, true);
    auto norm = x_f32 * (var + eps_).rsqrt();
    auto gate = ops::silu(z_f32);
    auto weighted = norm * norm_param_->value().to(ov::element::f32);
    return (weighted * gate).to(x.dtype());
}

Tensor Qwen3NextGatedDeltaNet::forward(const Tensor& hidden_states,
                                       const Tensor& beam_idx,
                                       const Tensor* attention_mask,
                                       const Tensor* cache_position) const {
    (void)cache_position;
    auto* op_ctx = hidden_states.context();

    Tensor masked_hidden = hidden_states;
    if (attention_mask) {
        masked_hidden = hidden_states * attention_mask->to(hidden_states.dtype()).unsqueeze(2);
    }

    const int32_t ratio = num_v_heads_ / num_k_heads_;

    auto projected_qkvz = ops::linear(masked_hidden, in_proj_qkvz_weight());
    auto projected_ba = ops::linear(masked_hidden, in_proj_ba_weight());

    auto mixed_qkvz = projected_qkvz.reshape({0, 0, num_k_heads_, 2 * head_k_dim_ + 2 * ratio * head_v_dim_});
    auto mixed_ba = projected_ba.reshape({0, 0, num_k_heads_, 2 * ratio});

    auto q_pre = ops::slice(mixed_qkvz, 0, head_k_dim_, 1, 3);
    auto k_pre = ops::slice(mixed_qkvz, head_k_dim_, 2 * head_k_dim_, 1, 3);
    auto v_pre = ops::slice(mixed_qkvz, 2 * head_k_dim_, 2 * head_k_dim_ + ratio * head_v_dim_, 1, 3);
    auto z = ops::slice(mixed_qkvz,
                        2 * head_k_dim_ + ratio * head_v_dim_,
                        2 * head_k_dim_ + 2 * ratio * head_v_dim_,
                        1,
                        3);
    auto b = ops::slice(mixed_ba, 0, ratio, 1, 3);
    auto a = ops::slice(mixed_ba, ratio, 2 * ratio, 1, 3);

    auto value_pre = v_pre.reshape({0, 0, num_v_heads_, head_v_dim_});
    z = z.reshape({0, 0, num_v_heads_, head_v_dim_});
    b = b.reshape({0, 0, num_v_heads_});
    a = a.reshape({0, 0, num_v_heads_});

    auto q_flat = q_pre.reshape({0, 0, key_dim_});
    auto k_flat = k_pre.reshape({0, 0, key_dim_});
    auto v_flat = value_pre.reshape({0, 0, value_dim_});

    auto mixed_qkv = ops::concat({q_flat, k_flat, v_flat}, 2).permute({0, 2, 1});

    auto batch = shape::dim(masked_hidden, 0);
    auto conv_shape = shape::make({batch,
                                   ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(conv_dim_)}),
                                   ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(conv_state_size_)})});
    auto conv_init = shape::broadcast_to(Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(masked_hidden.dtype()), conv_shape);

    auto state_prefix = "linear_states." + std::to_string(layer_idx_);
    ov::op::util::VariableInfo conv_info{ov::PartialShape{-1, conv_dim_, conv_state_size_},
                                         masked_hidden.dtype(),
                                         state_prefix + ".conv"};
    auto conv_var = std::make_shared<ov::op::util::Variable>(conv_info);
    Tensor mixed_after_conv;
    if (use_fused_conv_op()) {
        auto conv_w_2d = conv1d_weight().reshape({conv_dim_, conv_kernel_size_}, false);
        auto fused_result = ops::fused_conv(
            mixed_qkv,
            conv_w_2d,
            beam_idx,
            conv_init,
            conv_var);
        mixed_after_conv = fused_result.first;
    } else {
        auto conv_read = std::make_shared<ov::op::v6::ReadValue>(conv_init.output(), conv_var);
        auto conv_cached = ops::gather(Tensor(conv_read->output(0), op_ctx), beam_idx, 0);

        Tensor next_conv_state;
        mixed_after_conv = apply_depthwise_causal_conv(mixed_qkv, conv_cached, &next_conv_state);
        auto conv_assign = std::make_shared<ov::opset13::Assign>(next_conv_state.output(), conv_var);
        ctx().register_sink(conv_assign);
    }

    auto mixed_bt = mixed_after_conv.permute({0, 2, 1});
    auto q_conv = ops::slice(mixed_bt, 0, key_dim_, 1, 2);
    auto k_conv = ops::slice(mixed_bt, key_dim_, key_dim_ * 2, 1, 2);
    auto v_conv = ops::slice(mixed_bt, key_dim_ * 2, key_dim_ * 2 + value_dim_, 1, 2);

    auto q_heads = q_conv.reshape({0, 0, num_k_heads_, head_k_dim_});
    auto k_heads = k_conv.reshape({0, 0, num_k_heads_, head_k_dim_});
    auto v_heads = v_conv.reshape({0, 0, num_v_heads_, head_v_dim_});

    if (ratio > 1) {
        q_heads = ops::llm::repeat_kv(q_heads.permute({0, 2, 1, 3}), num_v_heads_, num_k_heads_, head_k_dim_)
                      .permute({0, 2, 1, 3});
        k_heads = ops::llm::repeat_kv(k_heads.permute({0, 2, 1, 3}), num_v_heads_, num_k_heads_, head_k_dim_)
                      .permute({0, 2, 1, 3});
    }

    auto reduce_kdim = ops::const_vec(op_ctx, std::vector<int64_t>{-1});
    auto q_f32 = q_heads.to(ov::element::f32);
    auto k_f32 = k_heads.to(ov::element::f32);
    auto v_f32 = v_heads.to(ov::element::f32);
    auto q_ss = Tensor(std::make_shared<ov::op::v1::ReduceSum>(q_f32.pow(2.0f).output(), reduce_kdim, true), op_ctx);
    auto k_ss = Tensor(std::make_shared<ov::op::v1::ReduceSum>(k_f32.pow(2.0f).output(), reduce_kdim, true), op_ctx);
    auto q_normed = q_f32 * (q_ss + 1e-6f).rsqrt();
    auto k_normed = k_f32 * (k_ss + 1e-6f).rsqrt();
    auto q_scaled = q_normed * (1.0f / std::sqrt(static_cast<float>(head_k_dim_)));

    auto beta = Tensor(std::make_shared<ov::op::v0::Sigmoid>(b.to(ov::element::f32).output()), op_ctx);
    auto softplus_in = a.to(ov::element::f32) + dt_bias().to(ov::element::f32);
    auto softplus = Tensor(std::make_shared<ov::op::v4::SoftPlus>(softplus_in.output()), op_ctx);
    auto g = -(a_log().to(ov::element::f32).exp() * softplus);

    auto recurrent_shape = shape::make({batch,
                                        ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(num_v_heads_)}),
                                        ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_k_dim_)}),
                                        ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_v_dim_)})});
    auto recurrent_init = shape::broadcast_to(Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx), recurrent_shape);

    ov::op::util::VariableInfo recurrent_info{ov::PartialShape{-1, num_v_heads_, head_k_dim_, head_v_dim_},
                                              ov::element::f32,
                                              state_prefix + ".recurrent"};
    auto recurrent_var = std::make_shared<ov::op::util::Variable>(recurrent_info);
    auto recurrent_read = std::make_shared<ov::op::v6::ReadValue>(recurrent_init.output(), recurrent_var);
    auto recurrent_cached = ops::gather(Tensor(recurrent_read->output(0), op_ctx), beam_idx, 0);

    auto q_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, num_v_heads_, head_k_dim_});
    auto k_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, num_v_heads_, head_k_dim_});
    auto v_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, num_v_heads_, head_v_dim_});
    auto g_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, num_v_heads_});
    auto b_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, num_v_heads_});
    auto state_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, num_v_heads_, head_k_dim_, head_v_dim_});

    auto axis_seq = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto q_s = std::make_shared<ov::op::v0::Squeeze>(q_t, axis_seq);
    auto k_s = std::make_shared<ov::op::v0::Squeeze>(k_t, axis_seq);
    auto v_s = std::make_shared<ov::op::v0::Squeeze>(v_t, axis_seq);
    auto g_s = std::make_shared<ov::op::v0::Squeeze>(g_t, axis_seq);
    auto b_s = std::make_shared<ov::op::v0::Squeeze>(b_t, axis_seq);

    auto g_exp = std::make_shared<ov::op::v0::Exp>(g_s);
    auto g_exp_u1 = std::make_shared<ov::op::v0::Unsqueeze>(g_exp, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
    auto g_exp_e = std::make_shared<ov::op::v0::Unsqueeze>(g_exp_u1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
    auto state_decay = std::make_shared<ov::op::v1::Multiply>(state_t, g_exp_e, ov::op::AutoBroadcastType::NUMPY);

    auto k_uns = std::make_shared<ov::op::v0::Unsqueeze>(k_s, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
    auto state_k = std::make_shared<ov::op::v1::Multiply>(state_decay, k_uns, ov::op::AutoBroadcastType::NUMPY);
    auto kv_mem = std::make_shared<ov::op::v1::ReduceSum>(state_k,
                                                           ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
                                                           false);

    auto b_uns = std::make_shared<ov::op::v0::Unsqueeze>(b_s, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
    auto delta = std::make_shared<ov::op::v1::Multiply>(
        std::make_shared<ov::op::v1::Subtract>(v_s, kv_mem, ov::op::AutoBroadcastType::NUMPY),
        b_uns,
        ov::op::AutoBroadcastType::NUMPY);

    auto delta_uns = std::make_shared<ov::op::v0::Unsqueeze>(delta, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-2}));
    auto outer = std::make_shared<ov::op::v1::Multiply>(k_uns, delta_uns, ov::op::AutoBroadcastType::NUMPY);
    auto state_new = std::make_shared<ov::op::v1::Add>(state_decay, outer, ov::op::AutoBroadcastType::NUMPY);

    auto q_uns = std::make_shared<ov::op::v0::Unsqueeze>(q_s, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
    auto y_state = std::make_shared<ov::op::v1::Multiply>(state_new, q_uns, ov::op::AutoBroadcastType::NUMPY);
    auto y_t_step = std::make_shared<ov::op::v1::ReduceSum>(y_state,
                                                            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
                                                            false);
    auto y_t_unsq = std::make_shared<ov::op::v0::Unsqueeze>(y_t_step, axis_seq);

    auto state_result = std::make_shared<ov::op::v0::Result>(state_new);
    auto y_result = std::make_shared<ov::op::v0::Result>(y_t_unsq);
    auto body = std::make_shared<ov::Model>(ov::OutputVector{state_result->output(0), y_result->output(0)},
                                            ov::ParameterVector{q_t, k_t, v_t, g_t, b_t, state_t});

    auto ti = std::make_shared<ov::op::v0::TensorIterator>();
    ti->set_body(body);
    ti->set_sliced_input(q_t, q_scaled.output(), 0, 1, 1, -1, 1);
    ti->set_sliced_input(k_t, k_normed.output(), 0, 1, 1, -1, 1);
    ti->set_sliced_input(v_t, v_f32.output(), 0, 1, 1, -1, 1);
    ti->set_sliced_input(g_t, g.output(), 0, 1, 1, -1, 1);
    ti->set_sliced_input(b_t, beta.output(), 0, 1, 1, -1, 1);
    ti->set_merged_input(state_t, recurrent_cached.output(), state_result);

    auto recurrent_final = ti->get_iter_value(state_result, -1);
    auto core_attn = ti->get_concatenated_slices(y_result, 0, 1, 1, -1, 1);
    auto recurrent_assign = std::make_shared<ov::opset13::Assign>(recurrent_final, recurrent_var);
    ctx().register_sink(recurrent_assign);

    auto core_attn_tensor = Tensor(core_attn, op_ctx);
    auto core_flat = core_attn_tensor.reshape({-1, head_v_dim_});
    auto z_flat = z.reshape({-1, head_v_dim_});
    auto gated = rms_norm_gated(core_flat, z_flat);
    auto gated_4d = gated.reshape(shape::of(z), false);
    auto merged = gated_4d.reshape({0, 0, value_dim_});
    return ops::linear(merged, out_proj_weight()).to(masked_hidden.dtype());
}

// ─────────────────────────────────────────────────────────────────────────────
// Qwen3NextGatedDeltaNet2 — uses ops::linear_attention instead of TensorIterator
// ─────────────────────────────────────────────────────────────────────────────

Qwen3NextGatedDeltaNet2::Qwen3NextGatedDeltaNet2(BuilderContext& ctx,
                                                 const std::string& name,
                                                 const Qwen3NextConfig& cfg,
                                                 int32_t layer_idx,
                                                 Module* parent)
    : Module(name, ctx, parent),
      layer_idx_(layer_idx),
      hidden_size_(cfg.hidden_size),
      num_v_heads_(cfg.linear_num_value_heads),
      num_k_heads_(cfg.linear_num_key_heads),
      head_k_dim_(cfg.linear_key_head_dim),
      head_v_dim_(cfg.linear_value_head_dim),
      key_dim_(head_k_dim_ * num_k_heads_),
      value_dim_(head_v_dim_ * num_v_heads_),
      conv_dim_(key_dim_ * 2 + value_dim_),
      conv_kernel_size_(cfg.linear_conv_kernel_dim),
      conv_state_size_(cfg.linear_conv_kernel_dim),
      eps_(cfg.rms_norm_eps) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3Next linear attention activation: ", cfg.hidden_act);
    }
    if (hidden_size_ <= 0 || num_v_heads_ <= 0 || num_k_heads_ <= 0 || head_k_dim_ <= 0 || head_v_dim_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3Next linear attention configuration");
    }
    if (conv_kernel_size_ <= 0) {
        OPENVINO_THROW("Qwen3Next linear_conv_kernel_dim must be > 0");
    }
    if ((num_v_heads_ % num_k_heads_) != 0) {
        OPENVINO_THROW("Qwen3Next linear_num_value_heads must be divisible by linear_num_key_heads");
    }

    in_proj_qkvz_param_ = &register_parameter("in_proj_qkvz.weight");
    in_proj_ba_param_ = &register_parameter("in_proj_ba.weight");
    conv1d_param_ = &register_parameter("conv1d.weight");
    a_log_param_ = &register_parameter("A_log");
    dt_bias_param_ = &register_parameter("dt_bias");
    norm_param_ = &register_parameter("norm.weight");
    out_proj_param_ = &register_parameter("out_proj.weight");
}

const Tensor& Qwen3NextGatedDeltaNet2::in_proj_qkvz_weight() const {
    if (!in_proj_qkvz_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet2 in_proj_qkvz parameter is not registered");
    }
    return in_proj_qkvz_param_->value();
}

const Tensor& Qwen3NextGatedDeltaNet2::in_proj_ba_weight() const {
    if (!in_proj_ba_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet2 in_proj_ba parameter is not registered");
    }
    return in_proj_ba_param_->value();
}

const Tensor& Qwen3NextGatedDeltaNet2::conv1d_weight() const {
    if (!conv1d_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet2 conv1d parameter is not registered");
    }
    return conv1d_param_->value();
}

const Tensor& Qwen3NextGatedDeltaNet2::a_log() const {
    if (!a_log_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet2 A_log parameter is not registered");
    }
    return a_log_param_->value();
}

const Tensor& Qwen3NextGatedDeltaNet2::dt_bias() const {
    if (!dt_bias_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet2 dt_bias parameter is not registered");
    }
    return dt_bias_param_->value();
}

const Tensor& Qwen3NextGatedDeltaNet2::out_proj_weight() const {
    if (!out_proj_param_) {
        OPENVINO_THROW("Qwen3NextGatedDeltaNet2 out_proj parameter is not registered");
    }
    return out_proj_param_->value();
}

Tensor Qwen3NextGatedDeltaNet2::apply_depthwise_causal_conv(const Tensor& mixed_qkv,
                                                            const Tensor& prev_conv_state,
                                                            Tensor* next_conv_state) const {
    auto* op_ctx = mixed_qkv.context();
    auto input_with_state = ops::concat({prev_conv_state, mixed_qkv}, 2);

    auto conv_weight = conv1d_weight().reshape({conv_dim_, 1, 1, conv_kernel_size_}, false);
    auto conv = std::make_shared<ov::op::v1::GroupConvolution>(input_with_state.output(),
                                                                conv_weight.output(),
                                                                ov::Strides{1},
                                                                ov::CoordinateDiff{0},
                                                                ov::CoordinateDiff{0},
                                                                ov::Strides{1});
    auto conv_act = ops::silu(Tensor(conv, op_ctx));

    auto seq_len = shape::dim(mixed_qkv, 2);
    auto conv_len = shape::dim(conv_act, 2);
    auto out_start = std::make_shared<ov::op::v1::Subtract>(conv_len, seq_len);
    auto out_slice = std::make_shared<ov::op::v8::Slice>(conv_act.output(),
                                                         out_start,
                                                         conv_len,
                                                         ops::const_vec(op_ctx, std::vector<int64_t>{1}),
                                                         ops::const_vec(op_ctx, std::vector<int64_t>{2}));

    auto total_len = shape::dim(input_with_state, 2);
    auto kernel = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(conv_state_size_)});
    auto state_start = std::make_shared<ov::op::v1::Subtract>(total_len, kernel);
    auto state_slice = std::make_shared<ov::op::v8::Slice>(input_with_state.output(),
                                                           state_start,
                                                           total_len,
                                                           ops::const_vec(op_ctx, std::vector<int64_t>{1}),
                                                           ops::const_vec(op_ctx, std::vector<int64_t>{2}));
    if (next_conv_state) {
        *next_conv_state = Tensor(state_slice, op_ctx);
    }
    return Tensor(out_slice, op_ctx);
}

Tensor Qwen3NextGatedDeltaNet2::rms_norm_gated(const Tensor& x, const Tensor& z) const {
    auto x_f32 = x.to(ov::element::f32);
    auto z_f32 = z.to(ov::element::f32);
    auto var = x_f32.pow(2.0f).mean(-1, true);
    auto norm = x_f32 * (var + eps_).rsqrt();
    auto gate = ops::silu(z_f32);
    auto weighted = norm * norm_param_->value().to(ov::element::f32);
    return (weighted * gate).to(x.dtype());
}

Tensor Qwen3NextGatedDeltaNet2::forward(const Tensor& hidden_states,
                                        const Tensor& beam_idx,
                                        const Tensor* attention_mask,
                                        const Tensor* cache_position) const {
    (void)cache_position;
    auto* op_ctx = hidden_states.context();

    Tensor masked_hidden = hidden_states;
    if (attention_mask) {
        masked_hidden = hidden_states * attention_mask->to(hidden_states.dtype()).unsqueeze(2);
    }

    const int32_t ratio = num_v_heads_ / num_k_heads_;

    // ── linear projections ──
    auto projected_qkvz = ops::linear(masked_hidden, in_proj_qkvz_weight());
    auto projected_ba = ops::linear(masked_hidden, in_proj_ba_weight());

    auto mixed_qkvz = projected_qkvz.reshape({0, 0, num_k_heads_, 2 * head_k_dim_ + 2 * ratio * head_v_dim_});
    auto mixed_ba = projected_ba.reshape({0, 0, num_k_heads_, 2 * ratio});

    auto q_pre = ops::slice(mixed_qkvz, 0, head_k_dim_, 1, 3);
    auto k_pre = ops::slice(mixed_qkvz, head_k_dim_, 2 * head_k_dim_, 1, 3);
    auto v_pre = ops::slice(mixed_qkvz, 2 * head_k_dim_, 2 * head_k_dim_ + ratio * head_v_dim_, 1, 3);
    auto z = ops::slice(mixed_qkvz,
                        2 * head_k_dim_ + ratio * head_v_dim_,
                        2 * head_k_dim_ + 2 * ratio * head_v_dim_,
                        1,
                        3);
    auto b = ops::slice(mixed_ba, 0, ratio, 1, 3);
    auto a = ops::slice(mixed_ba, ratio, 2 * ratio, 1, 3);

    auto value_pre = v_pre.reshape({0, 0, num_v_heads_, head_v_dim_});
    z = z.reshape({0, 0, num_v_heads_, head_v_dim_});
    b = b.reshape({0, 0, num_v_heads_});
    a = a.reshape({0, 0, num_v_heads_});

    // ── causal depthwise convolution ──
    auto q_flat = q_pre.reshape({0, 0, key_dim_});
    auto k_flat = k_pre.reshape({0, 0, key_dim_});
    auto v_flat = value_pre.reshape({0, 0, value_dim_});

    auto mixed_qkv = ops::concat({q_flat, k_flat, v_flat}, 2).permute({0, 2, 1});

    auto batch = shape::dim(masked_hidden, 0);
    auto conv_shape = shape::make({batch,
                                   ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(conv_dim_)}),
                                   ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(conv_state_size_)})});
    auto conv_init = shape::broadcast_to(Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(masked_hidden.dtype()), conv_shape);

    auto state_prefix = "linear_states." + std::to_string(layer_idx_);
    ov::op::util::VariableInfo conv_info{ov::PartialShape{-1, conv_dim_, conv_state_size_},
                                         masked_hidden.dtype(),
                                         state_prefix + ".conv"};
    auto conv_var = std::make_shared<ov::op::util::Variable>(conv_info);
    Tensor mixed_after_conv;
    if (use_fused_conv_op()) {
        auto conv_w_2d = conv1d_weight().reshape({conv_dim_, conv_kernel_size_}, false);
        auto fused_result = ops::fused_conv(
            mixed_qkv,
            conv_w_2d,
            beam_idx,
            conv_init,
            conv_var);
        mixed_after_conv = fused_result.first;
    } else {
        auto conv_read = std::make_shared<ov::op::v6::ReadValue>(conv_init.output(), conv_var);
        auto conv_cached = ops::gather(Tensor(conv_read->output(0), op_ctx), beam_idx, 0);

        Tensor next_conv_state;
        mixed_after_conv = apply_depthwise_causal_conv(mixed_qkv, conv_cached, &next_conv_state);
        auto conv_assign = std::make_shared<ov::opset13::Assign>(next_conv_state.output(), conv_var);
        ctx().register_sink(conv_assign);
    }

    auto mixed_bt = mixed_after_conv.permute({0, 2, 1});
    auto q_conv = ops::slice(mixed_bt, 0, key_dim_, 1, 2);
    auto k_conv = ops::slice(mixed_bt, key_dim_, key_dim_ * 2, 1, 2);
    auto v_conv = ops::slice(mixed_bt, key_dim_ * 2, key_dim_ * 2 + value_dim_, 1, 2);

    auto q_heads = q_conv.reshape({0, 0, num_k_heads_, head_k_dim_});
    auto k_heads = k_conv.reshape({0, 0, num_k_heads_, head_k_dim_});
    auto v_heads = v_conv.reshape({0, 0, num_v_heads_, head_v_dim_});

    // if (ratio > 1) {
    //     q_heads = ops::llm::repeat_kv(q_heads.permute({0, 2, 1, 3}), num_v_heads_, num_k_heads_, head_k_dim_)
    //                   .permute({0, 2, 1, 3});
    //     k_heads = ops::llm::repeat_kv(k_heads.permute({0, 2, 1, 3}), num_v_heads_, num_k_heads_, head_k_dim_)
    //                   .permute({0, 2, 1, 3});
    // }

    // ── beta and g computation (identical to v1) ──
    auto beta = Tensor(std::make_shared<ov::op::v0::Sigmoid>(b.to(ov::element::f32).output()), op_ctx);
    auto softplus_in = a.to(ov::element::f32) + dt_bias().to(ov::element::f32);
    auto softplus = Tensor(std::make_shared<ov::op::v4::SoftPlus>(softplus_in.output()), op_ctx);
    auto g = -(a_log().to(ov::element::f32).exp() * softplus);

    // ── recurrent state via ReadValue / Assign ──
    auto recurrent_shape = shape::make({batch,
                                        ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(num_v_heads_)}),
                                        ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_k_dim_)}),
                                        ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_v_dim_)})});
    auto recurrent_init = shape::broadcast_to(Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx), recurrent_shape);

    ov::op::util::VariableInfo recurrent_info{ov::PartialShape{-1, num_v_heads_, head_k_dim_, head_v_dim_},
                                              ov::element::f32,
                                              state_prefix + ".recurrent"};
    auto recurrent_var = std::make_shared<ov::op::util::Variable>(recurrent_info);
    auto recurrent_read = std::make_shared<ov::op::v6::ReadValue>(recurrent_init.output(), recurrent_var);
    auto recurrent_cached = ops::gather(Tensor(recurrent_read->output(0), op_ctx), beam_idx, 0);

    // ── ops::linear_attention replaces the TensorIterator loop ──
    // linear_attention expects: q [B, S, H, K], k [B, S, H, K], v [B, S, H, V],
    //                           beta [B, S, H], g [B, S, H], state [B, H, K, V]
    // and returns: (output [B, S, H, V], new_state [B, H, K, V])
    // Note: q/k/v are passed in their original dtype (f16/bf16); the OCL kernel
    //       converts to float internally. Normalization is also done inside the kernel.
    auto la_result = ops::linear_attention(q_heads, k_heads, v_heads, beta, g, recurrent_cached);
    auto core_attn_tensor = la_result.first;   // [B, S, num_v_heads, head_v_dim]
    auto recurrent_final = la_result.second;    // [B, num_v_heads, head_k_dim, head_v_dim]

    auto recurrent_assign = std::make_shared<ov::opset13::Assign>(recurrent_final.output(), recurrent_var);
    ctx().register_sink(recurrent_assign);

    // ── post-processing: gated RMS norm + out projection ──
    auto core_flat = core_attn_tensor.reshape({-1, head_v_dim_});
    auto z_flat = z.reshape({-1, head_v_dim_});
    auto gated = rms_norm_gated(core_flat, z_flat);
    auto gated_4d = gated.reshape(shape::of(z), false);
    auto merged = gated_4d.reshape({0, 0, value_dim_});
    return ops::linear(merged, out_proj_weight()).to(masked_hidden.dtype());
}

Qwen3NextDecoderLayer::Qwen3NextDecoderLayer(BuilderContext& ctx,
                                             const std::string& name,
                                             const Qwen3NextConfig& cfg,
                                             int32_t layer_idx,
                                             Module* parent)
    : Module(name, ctx, parent),
      layer_type_(cfg.layer_types.at(static_cast<size_t>(layer_idx))),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {
    if (layer_type_ == "full_attention") {
        self_attn_ = std::make_unique<Qwen3NextAttention>(ctx, "self_attn", cfg, this);
    } else if (layer_type_ == "linear_attention") {
        linear_attn_ = std::make_unique<Qwen3NextGatedDeltaNet2>(ctx, "linear_attn", cfg, layer_idx, this);
    } else {
        OPENVINO_THROW("Unsupported Qwen3Next layer type: ", layer_type_);
    }

    const auto mlp_only = make_index_set(cfg.mlp_only_layers);
    const int32_t sparse_step = cfg.decoder_sparse_step > 0 ? cfg.decoder_sparse_step : 1;
    const bool use_moe = (cfg.num_experts > 0) &&
                         (mlp_only.count(layer_idx) == 0) &&
                         (((layer_idx + 1) % sparse_step) == 0);
    if (use_moe) {
        moe_mlp_ = std::make_unique<Qwen3NextSparseMoeBlock>(ctx, "mlp", cfg, this);
    } else {
        dense_mlp_ = std::make_unique<Qwen3NextMLP>(ctx, "mlp", cfg, cfg.intermediate_size, this);
    }
}

std::pair<Tensor, Tensor> Qwen3NextDecoderLayer::forward(const Tensor& hidden_states,
                                                         const Tensor& beam_idx,
                                                         const Tensor& rope_cos,
                                                         const Tensor& rope_sin,
                                                         const Tensor* full_attention_mask,
                                                         const Tensor* linear_attention_mask,
                                                         const Tensor* cache_position,
                                                         const std::optional<Tensor>& residual,
                                                         const Tensor* precomputed_full_attn_sdpa_mask) const {
    Tensor normed;
    Tensor next_residual;
    if (residual) {
        auto norm_out = input_layernorm_.forward(hidden_states, *residual);
        normed = norm_out.first;
        next_residual = norm_out.second;
    } else {
        normed = input_layernorm_.forward(hidden_states);
        next_residual = hidden_states;
    }

    Tensor mixed;
    if (layer_type_ == "full_attention") {
        mixed = self_attn_->forward(
            normed,
            beam_idx,
            rope_cos,
            rope_sin,
            full_attention_mask,
            precomputed_full_attn_sdpa_mask);
    } else {
        mixed = linear_attn_->forward(normed, beam_idx, linear_attention_mask, cache_position);
    }

    auto post = post_attention_layernorm_.forward(mixed, next_residual);
    Tensor mlp_out = dense_mlp_ ? dense_mlp_->forward(post.first) : moe_mlp_->forward(post.first);
    return {mlp_out, post.second};
}

Qwen3NextModel::Qwen3NextModel(BuilderContext& ctx, const Qwen3NextConfig& cfg, Module* parent)
    : Module("model", ctx, parent),
      embed_tokens_(ctx, "embed_tokens", this),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      head_dim_(cfg.head_dim > 0
                    ? cfg.head_dim
                    : (cfg.num_attention_heads > 0 ? (cfg.hidden_size / cfg.num_attention_heads) : 0)),
      rope_theta_(cfg.rope_theta) {
    if (head_dim_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3Next head dimension");
    }
    rotary_dim_ = static_cast<int32_t>(std::floor(static_cast<float>(head_dim_) * cfg.partial_rotary_factor));
    rotary_dim_ = std::max<int32_t>(0, std::min<int32_t>(rotary_dim_, head_dim_));
    if ((rotary_dim_ % 2) != 0) {
        rotary_dim_ -= 1;
    }
    OPENVINO_ASSERT(rotary_dim_ > 0, "Qwen3Next rotary dimension must be > 0");

    Qwen3NextConfig layer_cfg = cfg;
    if (layer_cfg.layer_types.empty()) {
        layer_cfg.layer_types = build_default_layer_types(cfg.num_hidden_layers, cfg.full_attention_interval);
    }
    if (layer_cfg.layer_types.size() != static_cast<size_t>(cfg.num_hidden_layers)) {
        OPENVINO_THROW("Qwen3Next layer_types size mismatch with num_hidden_layers");
    }

    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", layer_cfg, i, this);
    }
}

Tensor Qwen3NextModel::forward(const Tensor& input_ids,
                               const Tensor& position_ids,
                               const Tensor& beam_idx,
                               const Tensor& full_attention_mask,
                               const Tensor* linear_attention_mask,
                               const Tensor* cache_position) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, rotary_dim_, rope_theta_, policy);

    std::optional<Tensor> linear_mask_view;
    const Tensor* linear_mask = nullptr;
    if (linear_attention_mask) {
        auto* op_ctx = input_ids.context();
        auto q_len = shape::dim(input_ids, 1);
        auto mask_len = shape::dim(*linear_attention_mask, 1);
        auto start = std::make_shared<ov::op::v1::Subtract>(mask_len, q_len);
        auto sliced = std::make_shared<ov::op::v8::Slice>(
            linear_attention_mask->output(),
            start,
            mask_len,
            ops::const_vec(op_ctx, std::vector<int64_t>{1}),
            ops::const_vec(op_ctx, std::vector<int64_t>{1}));
        linear_mask_view = Tensor(sliced, op_ctx);
        linear_mask = &(*linear_mask_view);
    }

    auto* op_ctx = input_ids.context();
    auto q_len_1d = Tensor(shape::dim(input_ids, 1), op_ctx);
    auto shared_full_attn_sdpa_mask =
        ops::llm::build_kv_causal_mask_with_attention_from_q_len(q_len_1d, full_attention_mask);

    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto out = layer.forward(hidden_states,
                                 beam_idx,
                                 cos_sin.first,
                                 cos_sin.second,
                                 &full_attention_mask,
                                 linear_mask,
                                 cache_position,
                                 residual,
                                 &shared_full_attn_sdpa_mask);
        hidden_states = out.first;
        residual = out.second;
    }

    if (residual) {
        return norm_.forward(hidden_states, *residual).first;
    }
    return norm_.forward(hidden_states);
}

VocabEmbedding& Qwen3NextModel::embed_tokens() {
    return embed_tokens_;
}

Qwen3NextForCausalLM::Qwen3NextForCausalLM(BuilderContext& ctx, const Qwen3NextConfig& cfg, Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_head_(ctx, "lm_head", this) {
    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}

Tensor Qwen3NextForCausalLM::forward(const Tensor& input_ids,
                                     const Tensor& position_ids,
                                     const Tensor& beam_idx,
                                     const Tensor& full_attention_mask,
                                     const Tensor* linear_attention_mask,
                                     const Tensor* cache_position) {
    auto hidden = model_.forward(input_ids,
                                 position_ids,
                                 beam_idx,
                                 full_attention_mask,
                                 linear_attention_mask,
                                 cache_position);
    return lm_head_.forward(hidden);
}

std::shared_ptr<ov::Model> create_qwen3_next_model(
    const Qwen3NextConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    const auto effective_cfg = apply_qwen3_next_layer_limit(cfg);

    BuilderContext ctx;
    Qwen3NextForCausalLM model(ctx, effective_cfg);

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_missing = false;
    options.allow_unmatched = true;
    options.report_missing = true;
    options.report_unmatched = true;
    auto report = ov::genai::modeling::weights::load_model(model, source, finalizer, options);

    for (const auto& name : report.unmatched) {
        const int layer_idx = extract_layer_index_from_weight_name(name);
        if (layer_idx >= effective_cfg.num_hidden_layers) {
            // Allow unused deeper layers when we intentionally build a truncated layer stack.
            continue;
        }
        if (starts_with(name, "mtp.") || ends_with(name, "_scale_inv")) {
            continue;
        }
        OPENVINO_THROW("Unsupported unmatched Qwen3Next weight: ", name);
    }

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    input_ids.output().get_node()->output(0).set_names({"input_ids"});
    attention_mask.output().get_node()->output(0).set_names({"attention_mask"});
    position_ids.output().get_node()->output(0).set_names({"position_ids"});
    beam_idx.output().get_node()->output(0).set_names({"beam_idx"});

    auto logits = model.forward(input_ids, position_ids, beam_idx, attention_mask, &attention_mask, nullptr);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");
    auto ov_model = ctx.build_model({result->output(0)});
    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
    return ov_model;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

#include "loaders/model_builder.hpp"

namespace {

ov::genai::modeling::models::Qwen3NextConfig to_qwen3_next_config(const ov::genai::loaders::ModelConfig& config) {
    using ov::genai::modeling::models::Qwen3NextConfig;

    Qwen3NextConfig cfg;
    cfg.architecture = "qwen3_next";
    cfg.hidden_size = config.hidden_size;
    cfg.num_attention_heads = config.num_attention_heads;
    cfg.num_key_value_heads = config.num_key_value_heads > 0 ? config.num_key_value_heads : config.num_attention_heads;
    cfg.head_dim = config.head_dim > 0 ? config.head_dim : (config.hidden_size / config.num_attention_heads);
    cfg.intermediate_size = config.intermediate_size;
    cfg.num_hidden_layers = config.num_hidden_layers;
    cfg.vocab_size = config.vocab_size;
    cfg.max_position_embeddings = config.max_position_embeddings;
    cfg.rms_norm_eps = config.rms_norm_eps;
    cfg.rope_theta = config.rope_theta;
    cfg.partial_rotary_factor = config.partial_rotary_factor > 0.0f ? config.partial_rotary_factor : 0.25f;
    cfg.hidden_act = config.hidden_act;
    cfg.attention_bias = config.attention_bias;
    cfg.tie_word_embeddings = config.tie_word_embeddings;

    cfg.full_attention_interval = config.full_attention_interval > 0 ? config.full_attention_interval : 4;
    auto all_layer_types = config.layer_types.empty()
                               ? build_default_layer_types(config.num_hidden_layers, cfg.full_attention_interval)
                               : config.layer_types;
    if (all_layer_types.size() < static_cast<size_t>(cfg.num_hidden_layers)) {
        OPENVINO_THROW("Qwen3Next layer_types size is smaller than requested layer count. layer_types=",
                       all_layer_types.size(),
                       ", requested_layers=",
                       cfg.num_hidden_layers);
    }
    cfg.layer_types.assign(all_layer_types.begin(), all_layer_types.begin() + cfg.num_hidden_layers);

    cfg.linear_conv_kernel_dim = config.linear_conv_kernel_dim > 0 ? config.linear_conv_kernel_dim : 4;
    cfg.linear_key_head_dim = config.linear_key_head_dim > 0 ? config.linear_key_head_dim : 128;
    cfg.linear_value_head_dim = config.linear_value_head_dim > 0 ? config.linear_value_head_dim : 128;
    cfg.linear_num_key_heads = config.linear_num_key_heads > 0 ? config.linear_num_key_heads : cfg.num_attention_heads;
    cfg.linear_num_value_heads = config.linear_num_value_heads > 0 ? config.linear_num_value_heads : cfg.num_attention_heads;

    cfg.decoder_sparse_step = config.decoder_sparse_step > 0 ? config.decoder_sparse_step : 1;
    cfg.moe_intermediate_size = config.moe_intermediate_size > 0 ? config.moe_intermediate_size : 512;
    cfg.shared_expert_intermediate_size =
        config.shared_expert_intermediate_size > 0 ? config.shared_expert_intermediate_size : 512;
    cfg.num_experts = config.num_experts > 0 ? config.num_experts : 0;
    cfg.num_experts_per_tok = config.num_experts_per_tok > 0 ? config.num_experts_per_tok : 1;
    cfg.norm_topk_prob = config.norm_topk_prob;
    cfg.output_router_logits = config.output_router_logits;
    cfg.router_aux_loss_coef = config.router_aux_loss_coef;
    for (int32_t layer_idx : config.mlp_only_layers) {
        if (layer_idx >= 0 && layer_idx < cfg.num_hidden_layers) {
            cfg.mlp_only_layers.push_back(layer_idx);
        }
    }
    return cfg;
}

std::shared_ptr<ov::Model> build_qwen3_next_model(
    const ov::genai::loaders::ModelConfig& config,
    ov::genai::modeling::weights::WeightSource& weight_source,
    ov::genai::modeling::weights::WeightFinalizer& weight_finalizer) {
    auto cfg = to_qwen3_next_config(config);
    return ov::genai::modeling::models::create_qwen3_next_model(cfg, weight_source, weight_finalizer);
}

static bool qwen3_next_registered = []() {
    ov::genai::loaders::ModelBuilder::instance().register_architecture("qwen3_next", build_qwen3_next_model);
    ov::genai::loaders::ModelBuilder::instance().register_architecture("qwen3next", build_qwen3_next_model);
    return true;
}();

}  // namespace
