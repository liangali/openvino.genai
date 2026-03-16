// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_moe/modeling_qwen3_moe.hpp"

#include <cmath>
#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/kv_cache.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "modeling/weights/weight_source.hpp"

namespace {

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
}

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

std::string base_key_from_weight(const std::string& name) {
    constexpr std::string_view suffix = ".weight";
    if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return name.substr(0, name.size() - suffix.size());
    }
    return name;
}

void load_raw_weight(ov::genai::modeling::WeightParameter& param,
                     ov::genai::modeling::weights::WeightSource& source,
                     const std::string& weight_name) {
    if (!param.context()) {
        OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
    }
    if (!source.has(weight_name)) {
        OPENVINO_THROW("Missing weight tensor: ", weight_name);
    }
    param.bind(ov::genai::modeling::ops::constant(source.get_tensor(weight_name), param.context()));
}

ov::genai::modeling::Tensor dequantize_packed_moe_weight(const ov::genai::modeling::Tensor& packed_weight,
                                                          const ov::genai::modeling::Tensor& scales_e_g_o,
                                                          const ov::genai::modeling::Tensor& zps_e_g_o,
                                                          int32_t num_experts,
                                                          int32_t out_features,
                                                          int32_t in_features) {
    auto* op_ctx = packed_weight.context();

    // Convert [E, G, O] -> [E, O, G] to align with packed [E, O, G, GS].
    auto perm = ov::genai::modeling::ops::const_vec(op_ctx, std::vector<int64_t>{0, 2, 1});
    auto scales_e_o_g = ov::genai::modeling::Tensor(
        std::make_shared<ov::op::v1::Transpose>(scales_e_g_o.output(), perm),
        op_ctx);
    auto zps_e_o_g = ov::genai::modeling::Tensor(
        std::make_shared<ov::op::v1::Transpose>(zps_e_g_o.output(), perm),
        op_ctx);

    auto packed_f32 = packed_weight.to(ov::element::f32);
    auto zps_f32 = zps_e_o_g.unsqueeze(-1).to(ov::element::f32);
    auto scales_f32 = scales_e_o_g.unsqueeze(-1).to(ov::element::f32);
    auto dequant_grouped = (packed_f32 - zps_f32) * scales_f32;

    return dequant_grouped.reshape({num_experts, out_features, in_features}, false);
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

Qwen3MoeAttention::Qwen3MoeAttention(BuilderContext& ctx,
                                     const std::string& name,
                                     const Qwen3MoeConfig& cfg,
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
    if (num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid attention head configuration");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
    }
    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");

    q_bias_param_ = &register_parameter("q_proj.bias");
    k_bias_param_ = &register_parameter("k_proj.bias");
    v_bias_param_ = &register_parameter("v_proj.bias");
    o_bias_param_ = &register_parameter("o_proj.bias");
}

const Tensor& Qwen3MoeAttention::q_proj_weight() const {
    if (!q_proj_param_) {
        OPENVINO_THROW("Qwen3MoeAttention q_proj parameter not registered");
    }
    return q_proj_param_->value();
}

const Tensor& Qwen3MoeAttention::k_proj_weight() const {
    if (!k_proj_param_) {
        OPENVINO_THROW("Qwen3MoeAttention k_proj parameter not registered");
    }
    return k_proj_param_->value();
}

const Tensor& Qwen3MoeAttention::v_proj_weight() const {
    if (!v_proj_param_) {
        OPENVINO_THROW("Qwen3MoeAttention v_proj parameter not registered");
    }
    return v_proj_param_->value();
}

const Tensor& Qwen3MoeAttention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("Qwen3MoeAttention o_proj parameter not registered");
    }
    return o_proj_param_->value();
}

const Tensor* Qwen3MoeAttention::q_proj_bias() const {
    return (q_bias_param_ && q_bias_param_->is_bound()) ? &q_bias_param_->value() : nullptr;
}

const Tensor* Qwen3MoeAttention::k_proj_bias() const {
    return (k_bias_param_ && k_bias_param_->is_bound()) ? &k_bias_param_->value() : nullptr;
}

const Tensor* Qwen3MoeAttention::v_proj_bias() const {
    return (v_bias_param_ && v_bias_param_->is_bound()) ? &v_bias_param_->value() : nullptr;
}

const Tensor* Qwen3MoeAttention::o_proj_bias() const {
    return (o_bias_param_ && o_bias_param_->is_bound()) ? &o_bias_param_->value() : nullptr;
}

Tensor Qwen3MoeAttention::forward(const Tensor& positions, const Tensor& hidden_states, const Tensor& beam_idx) const {
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(positions, head_dim_, rope_theta_, policy);
    auto seq_len = Tensor(shape::dim(positions, 1), positions.context()).squeeze(0);
    auto causal_mask = ops::llm::causal_mask_from_seq_len(seq_len);
    return forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, causal_mask);
}

Tensor Qwen3MoeAttention::forward(const Tensor& hidden_states,
                                  const Tensor& beam_idx,
                                  const Tensor& rope_cos,
                                  const Tensor& rope_sin,
                                  const Tensor& causal_mask) const {
    auto q = add_bias_if_present(ops::linear(hidden_states, q_proj_weight()), q_proj_bias());
    auto k = add_bias_if_present(ops::linear(hidden_states, k_proj_weight()), k_proj_bias());
    auto v = add_bias_if_present(ops::linear(hidden_states, v_proj_weight()), v_proj_bias());

    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    if (q_norm_.weight_param().is_bound()) {
        q_heads = q_norm_.forward(q_heads);
    }
    if (k_norm_.weight_param().is_bound()) {
        k_heads = k_norm_.forward(k_heads);
    }

    auto* policy = &ctx().op_policy();
    auto q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    const std::string cache_prefix = full_path().empty() ? name() : full_path();
    auto cached = ops::append_kv_cache(k_rot, v_heads, beam_idx, num_kv_heads_, head_dim_, cache_prefix, ctx());
    auto k_expanded = ops::llm::repeat_kv(cached.first, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(cached.second, num_heads_, num_kv_heads_, head_dim_);

    auto mask = ops::llm::build_kv_causal_mask(q_rot, k_expanded);
    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &mask, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    auto out = add_bias_if_present(ops::linear(merged, o_proj_weight()), o_proj_bias());
    return out;
}

Qwen3MoE::Qwen3MoE(BuilderContext& ctx, const std::string& name, const Qwen3MoeConfig& cfg, Module* parent)
    : Module(name, ctx, parent),
      hidden_size_(cfg.hidden_size),
      inter_size_(cfg.moe_intermediate_size > 0 ? cfg.moe_intermediate_size : cfg.intermediate_size),
      num_experts_(cfg.expert_count),
      top_k_(cfg.expert_used_count > 0 ? cfg.expert_used_count : 1),
      group_size_((cfg.group_size > 0) ? static_cast<size_t>(cfg.group_size) : std::numeric_limits<size_t>::max()) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3 MoE activation: ", cfg.hidden_act);
    }
    if (hidden_size_ <= 0 || inter_size_ <= 0 || num_experts_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3 MoE configuration");
    }

    if (cfg.architecture == "qwen3_moe") {
        has_gate_quant_ = true;
        has_up_quant_ = true;
        has_down_quant_ = true;

        gate_exps_param_.resize(num_experts_);
        up_exps_param_.resize(num_experts_);
        down_exps_param_.resize(num_experts_);
        gate_exps_scales_.resize(num_experts_);
        gate_exps_zps_.resize(num_experts_);
        up_exps_scales_.resize(num_experts_);
        up_exps_zps_.resize(num_experts_);
        down_exps_scales_.resize(num_experts_);
        down_exps_zps_.resize(num_experts_);

        gate_inp_param_ = &register_parameter("gate.weight");
        for (int32_t i = 0; i < num_experts_; ++i) {
            std::string prefix = "experts." + std::to_string(i) + ".";
            gate_exps_param_[i] = &register_parameter(prefix + "gate_proj.weight");
            up_exps_param_[i] = &register_parameter(prefix + "up_proj.weight");
            down_exps_param_[i] = &register_parameter(prefix + "down_proj.weight");

            gate_exps_param_[i]->set_weight_loader([this, i](WeightParameter& param,
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
                    gate_exps_scales_[i] = weight.auxiliary.at("scales");
                    gate_exps_zps_[i] = weight.auxiliary.at("zps");
                } else {
                    has_gate_quant_ = false;
                }
            });

            up_exps_param_[i]->set_weight_loader([this, i](WeightParameter& param,
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
                    up_exps_scales_[i] = weight.auxiliary.at("scales");
                    up_exps_zps_[i] = weight.auxiliary.at("zps");
                } else {
                    has_up_quant_ = false;
                }
            });

            down_exps_param_[i]->set_weight_loader([this, i](WeightParameter& param,
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
                    down_exps_scales_[i] = weight.auxiliary.at("scales");
                    down_exps_zps_[i] = weight.auxiliary.at("zps");
                } else {
                    has_down_quant_ = false;
                }
            });
        }
    } else {
        has_gate_quant_ = true;
        has_up_quant_ = true;
        has_down_quant_ = true;

        gate_exps_param_.resize(1);
        up_exps_param_.resize(1);
        down_exps_param_.resize(1);
        gate_exps_scales_.resize(1);
        gate_exps_zps_.resize(1);
        up_exps_scales_.resize(1);
        up_exps_zps_.resize(1);
        down_exps_scales_.resize(1);
        down_exps_zps_.resize(1);

        gate_inp_param_ = &register_parameter("gate_inp.weight");
        gate_exps_param_[0] = &register_parameter("gate_exps.weight");
        up_exps_param_[0] = &register_parameter("up_exps.weight");
        down_exps_param_[0] = &register_parameter("down_exps.weight");
        gate_exps_param_[0]->set_weight_loader([this](WeightParameter& param,
                                                   weights::WeightSource& source,
                                                   weights::WeightFinalizer& finalizer,
                                                   const std::string& weight_name,
                                                   const std::optional<int>& shard_id) {
            (void)finalizer;
            (void)shard_id;
            load_raw_weight(param, source, weight_name);
            const std::string base_key = base_key_from_weight(weight_name);
            const std::string scales_key = base_key + ".scales";
            const std::string zps_key = base_key + ".zps";
            if (source.has(scales_key) && source.has(zps_key)) {
                gate_exps_scales_[0] = ops::constant(source.get_tensor(scales_key), param.context());
                gate_exps_zps_[0] = ops::constant(source.get_tensor(zps_key), param.context());
            } else {
                has_gate_quant_ = false;
            }
        });

        up_exps_param_[0]->set_weight_loader([this](WeightParameter& param,
                                                 weights::WeightSource& source,
                                                 weights::WeightFinalizer& finalizer,
                                                 const std::string& weight_name,
                                                 const std::optional<int>& shard_id) {
            (void)finalizer;
            (void)shard_id;
            load_raw_weight(param, source, weight_name);
            const std::string base_key = base_key_from_weight(weight_name);
            const std::string scales_key = base_key + ".scales";
            const std::string zps_key = base_key + ".zps";
            if (source.has(scales_key) && source.has(zps_key)) {
                up_exps_scales_[0] = ops::constant(source.get_tensor(scales_key), param.context());
                up_exps_zps_[0] = ops::constant(source.get_tensor(zps_key), param.context());
            } else {
                has_up_quant_ = false;
            }
        });

        down_exps_param_[0]->set_weight_loader([this](WeightParameter& param,
                                                   weights::WeightSource& source,
                                                   weights::WeightFinalizer& finalizer,
                                                   const std::string& weight_name,
                                                   const std::optional<int>& shard_id) {
            (void)finalizer;
            (void)shard_id;
            load_raw_weight(param, source, weight_name);
            const std::string base_key = base_key_from_weight(weight_name);
            const std::string scales_key = base_key + ".scales";
            const std::string zps_key = base_key + ".zps";
            if (source.has(scales_key) && source.has(zps_key)) {
                down_exps_scales_[0] = ops::constant(source.get_tensor(scales_key), param.context());
                down_exps_zps_[0] = ops::constant(source.get_tensor(zps_key), param.context());
            } else {
                has_down_quant_ = false;
            }
        });
    }
}

const Tensor& Qwen3MoE::gate_inp_weight() const {
    if (!gate_inp_param_) {
        OPENVINO_THROW("Qwen3MoE gate input parameter not registered");
    }
    return gate_inp_param_->value();
}

Tensor Qwen3MoE::gate_exps_weight() const {
    if (gate_exps_param_.empty() || !gate_exps_param_[0]) {
        OPENVINO_THROW("Qwen3MoE gate expert parameter not registered");
    }
    std::vector<Tensor> valid;
    valid.reserve(gate_exps_param_.size());
    for (auto* p : gate_exps_param_) {
        valid.push_back(p->value().unsqueeze(0));
    }
    auto result = ops::concat(valid, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3MoE::up_exps_weight() const {
    if (up_exps_param_.empty() || !up_exps_param_[0]) {
        OPENVINO_THROW("Qwen3MoE up expert parameter not registered");
    }
    std::vector<Tensor> valid;
    valid.reserve(up_exps_param_.size());
    for (auto* p : up_exps_param_) {
        valid.push_back(p->value().unsqueeze(0));
    }
    auto result = ops::concat(valid, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3MoE::down_exps_weight() const {
    if (down_exps_param_.empty() || !down_exps_param_[0]) {
        OPENVINO_THROW("Qwen3MoE down expert parameter not registered");
    }
    std::vector<Tensor> valid;
    valid.reserve(down_exps_param_.size());
    for (auto* p : down_exps_param_) {
        valid.push_back(p->value().unsqueeze(0));
    }
    auto result = ops::concat(valid, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3MoE::gate_exps_scales() const {
    std::vector<Tensor> expanded;
    expanded.reserve(gate_exps_scales_.size());
    for (const auto& t : gate_exps_scales_) {
        expanded.push_back(t.unsqueeze(0));
    }
    auto result = ops::concat(expanded, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3MoE::gate_exps_zps() const {
    std::vector<Tensor> expanded;
    expanded.reserve(gate_exps_zps_.size());
    for (const auto& t : gate_exps_zps_) {
        expanded.push_back(t.unsqueeze(0));
    }
    auto result = ops::concat(expanded, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3MoE::up_exps_scales() const {
    std::vector<Tensor> expanded;
    expanded.reserve(up_exps_scales_.size());
    for (const auto& t : up_exps_scales_) {
        expanded.push_back(t.unsqueeze(0));
    }
    auto result = ops::concat(expanded, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3MoE::up_exps_zps() const {
    std::vector<Tensor> expanded;
    expanded.reserve(up_exps_zps_.size());
    for (const auto& t : up_exps_zps_) {
        expanded.push_back(t.unsqueeze(0));
    }
    auto result = ops::concat(expanded, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3MoE::down_exps_scales() const {
    std::vector<Tensor> expanded;
    expanded.reserve(down_exps_scales_.size());
    for (const auto& t : down_exps_scales_) {
        expanded.push_back(t.unsqueeze(0));
    }
    auto result = ops::concat(expanded, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3MoE::down_exps_zps() const {
    std::vector<Tensor> expanded;
    expanded.reserve(down_exps_zps_.size());
    for (const auto& t : down_exps_zps_) {
        expanded.push_back(t.unsqueeze(0));
    }
    auto result = ops::concat(expanded, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

bool Qwen3MoE::can_use_fused_path() const {
    if (!(has_gate_quant_ && has_up_quant_ && has_down_quant_)) {
        return false;
    }

    // Fused compressed MoE kernel expects packed expert weights.
    const auto gate_rank = gate_exps_weight().output().get_shape().size();
    const auto up_rank = up_exps_weight().output().get_shape().size();
    const auto down_rank = down_exps_weight().output().get_shape().size();
    return gate_rank == 4 && up_rank == 4 && down_rank == 4;
}

size_t Qwen3MoE::infer_group_size() const {
    const auto shape = gate_exps_weight().output().get_shape();
    if (shape.size() == 4 && shape[3] > 0) {
        return shape[3];
    }
    return group_size_;
}

Tensor Qwen3MoE::routed_fused(const Tensor& flat_f32) const {
    return ops::moe3gemm_fused_compressed(
        flat_f32,
        gate_inp_weight(),
        gate_exps_weight(),
        gate_exps_scales(),
        gate_exps_zps(),
        up_exps_weight(),
        up_exps_scales(),
        up_exps_zps(),
        down_exps_weight(),
        down_exps_scales(),
        down_exps_zps(),
        hidden_size_,
        inter_size_,
        num_experts_,
        top_k_,
        infer_group_size(),
        ov::element::f16);
}

Tensor Qwen3MoE::routed_fallback(const Tensor& flat_f32) const {
    auto* op_ctx = flat_f32.context();

    auto logits = ops::linear(flat_f32, gate_inp_weight().to(ov::element::f32));
    auto scores = logits.softmax(1);

    auto k_node = ops::const_scalar(op_ctx, static_cast<int64_t>(top_k_));
    auto topk = std::make_shared<ov::op::v11::TopK>(scores.output(),
                                                    k_node,
                                                    -1,
                                                    ov::op::v11::TopK::Mode::MAX,
                                                    ov::op::v11::TopK::SortType::SORT_VALUES,
                                                    ov::element::i64);

    Tensor topk_vals(topk->output(0), op_ctx);
    Tensor topk_idx(topk->output(1), op_ctx);

    auto zeros = shape::broadcast_to(Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx), shape::of(scores));
    auto scatter_axis = ops::const_scalar(op_ctx, static_cast<int64_t>(1));
    auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(zeros.output(),
                                                                         topk_idx.output(),
                                                                         topk_vals.output(),
                                                                         scatter_axis);
    Tensor routing(scatter, op_ctx);  // [T, E]
    auto perm = ops::const_vec(op_ctx, std::vector<int64_t>{1, 0});
    auto routing_t = Tensor(std::make_shared<ov::op::v1::Transpose>(routing.output(), perm), op_ctx);
    auto routing_3d = routing_t.unsqueeze(-1);  // [E, T, 1]

    auto flat_3d = flat_f32.unsqueeze(0);
    auto tiled = ops::tensor::tile(flat_3d, {num_experts_, 1, 1});  // [E, T, H]

    Tensor gate_w;
    Tensor up_w;
    Tensor down_w;

    if (has_gate_quant_ && gate_exps_weight().output().get_shape().size() == 4) {
        gate_w = dequantize_packed_moe_weight(gate_exps_weight(),
                                              gate_exps_scales(),
                                              gate_exps_zps(),
                                              num_experts_,
                                              inter_size_,
                                              hidden_size_);
        up_w = dequantize_packed_moe_weight(up_exps_weight(),
                                            up_exps_scales(),
                                            up_exps_zps(),
                                            num_experts_,
                                            inter_size_,
                                            hidden_size_);
    } else {
        gate_w = gate_exps_weight().to(ov::element::f32);
        up_w = up_exps_weight().to(ov::element::f32);
    }

    if (has_down_quant_ && down_exps_weight().output().get_shape().size() == 4) {
        down_w = dequantize_packed_moe_weight(down_exps_weight(),
                                              down_exps_scales(),
                                              down_exps_zps(),
                                              num_experts_,
                                              hidden_size_,
                                              inter_size_);
    } else {
        down_w = down_exps_weight().to(ov::element::f32);
    }

    auto gate_bmm = ops::matmul(tiled, gate_w, false, true);
    auto up_bmm = ops::matmul(tiled, up_w, false, true);
    auto swiglu = ops::silu(gate_bmm) * up_bmm;
    auto down_bmm = ops::matmul(swiglu, down_w, false, true);

    auto weighted = down_bmm.to(ov::element::f32) * routing_3d;
    auto reduce_axis = ops::const_vec(op_ctx, std::vector<int64_t>{0});
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(weighted.output(), reduce_axis, false);
    return Tensor(reduced, op_ctx);
}

Tensor Qwen3MoE::forward(const Tensor& x) const {
    auto flat = x.reshape({-1, hidden_size_});
    auto flat_f32 = flat.to(ov::element::f32);
    auto routed = can_use_fused_path() ? routed_fused(flat_f32) : routed_fallback(flat_f32);
    auto restored = routed.reshape(shape::of(x), false);
    return restored.to(x.dtype());
}

Qwen3MoeDecoderLayer::Qwen3MoeDecoderLayer(BuilderContext& ctx,
                                           const std::string& name,
                                           const Qwen3MoeConfig& cfg,
                                           Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, this),
      moe_(ctx, cfg.architecture == "qwen3_moe" ? "mlp" : "moe", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {}

std::pair<Tensor, Tensor> Qwen3MoeDecoderLayer::forward(const Tensor& hidden_states,
                                                        const Tensor& beam_idx,
                                                        const Tensor& rope_cos,
                                                        const Tensor& rope_sin,
                                                        const Tensor& causal_mask,
                                                        const std::optional<Tensor>& residual) const {
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
    auto attn_out = self_attn_.forward(normed, beam_idx, rope_cos, rope_sin, causal_mask);
    auto post_norm = post_attention_layernorm_.forward(attn_out, next_residual);
    auto moe_out = moe_.forward(post_norm.first);
    return {moe_out, post_norm.second};
}

Qwen3MoeModel::Qwen3MoeModel(BuilderContext& ctx, const Qwen3MoeConfig& cfg, Module* parent)
    : Module("model", ctx, parent),
      embed_tokens_(ctx, "embed_tokens", this),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      head_dim_(cfg.head_dim > 0
                    ? cfg.head_dim
                    : (cfg.num_attention_heads > 0 ? (cfg.hidden_size / cfg.num_attention_heads) : 0)),
      rope_theta_(cfg.rope_theta) {
    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", cfg, this);
    }
}

Tensor Qwen3MoeModel::forward(const Tensor& input_ids, const Tensor& position_ids, const Tensor& beam_idx) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    auto seq_len = Tensor(shape::dim(position_ids, 1), position_ids.context()).squeeze(0);
    auto causal_mask = ops::llm::causal_mask_from_seq_len(seq_len);
    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto layer_out = layer.forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, causal_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
    }
    if (residual) {
        return norm_.forward(hidden_states, *residual).first;
    }
    return norm_.forward(hidden_states);
}

VocabEmbedding& Qwen3MoeModel::embed_tokens() {
    return embed_tokens_;
}

RMSNorm& Qwen3MoeModel::norm() {
    return norm_;
}

Qwen3MoeForCausalLM::Qwen3MoeForCausalLM(BuilderContext& ctx, const Qwen3MoeConfig& cfg, Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_head_(ctx, "lm_head", this) {
    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}

Tensor Qwen3MoeForCausalLM::forward(const Tensor& input_ids,
                                    const Tensor& position_ids,
                                    const Tensor& beam_idx) {
    auto hidden = model_.forward(input_ids, position_ids, beam_idx);
    return lm_head_.forward(hidden);
}

Qwen3MoeModel& Qwen3MoeForCausalLM::model() {
    return model_;
}

LMHead& Qwen3MoeForCausalLM::lm_head() {
    return lm_head_;
}

std::shared_ptr<ov::Model> create_qwen3_moe_model(
    const Qwen3MoeConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3MoeForCausalLM model(ctx, cfg);

    ov::genai::modeling::weights::load_model(model, source, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    (void)attention_mask;
    auto logits = model.forward(input_ids, position_ids, beam_idx);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
