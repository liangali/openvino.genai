// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <tuple>

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>
#include <openvino/op/tensor_iterator.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>
#include <ov_ops/rms.hpp>

#include "modeling/models/dflash_draft/dflash_draft.hpp"
#include "modeling/ops/kv_cache.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/rope.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

std::vector<std::string> build_default_layer_types(int32_t num_layers, int32_t interval) {
    const int32_t safe_interval = interval > 0 ? interval : 4;
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i) {
        out.push_back(((i + 1) % safe_interval) == 0 ? "full_attention" : "linear_attention");
    }
    return out;
}

std::optional<int32_t> get_qwen3_5_layer_limit_from_env() {
    static constexpr const char* kEnvName = "OV_GENAI_qwen3_5_NUM_LAYERS";
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

bool use_linear_attention_op() {
    const char* raw = std::getenv("OV_GENAI_USE_LINEAR_ATTENTION_OP");
    if (!raw || raw[0] == '\0')
        return true;  // enabled by default
    return std::string(raw) != "0";
}

bool use_fused_conv_op() {
    const char* raw = std::getenv("OV_GENAI_USE_FUSED_CONV_OP");
    if (!raw || raw[0] == '\0')
        return true;  // enabled by default
    return std::string(raw) != "0";
}

bool use_state_snapshots() {
    const char* raw = std::getenv("OV_GENAI_DISABLE_STATE_SNAPSHOTS");
    if (raw && std::string(raw) == "1")
        return false;
    return true;  // enabled by default
}

// When set, forces snapshot outputs even in the normal (non-DFlash) model builder.
// Used for benchmarking snapshot kernel overhead in isolation.
bool force_state_snapshots() {
    const char* raw = std::getenv("OV_GENAI_FORCE_STATE_SNAPSHOTS");
    return raw && std::string(raw) == "1";
}

// Accumulator for snapshot outputs during model construction.
// GatedDeltaNet::forward() pushes {name, output} pairs here when snapshots are enabled.
struct SnapshotOutputAccumulator {
    std::vector<std::pair<std::string, ov::Output<ov::Node>>> entries;
    bool active = false;
    int64_t snapshot_max_seq = 0; 
};
static SnapshotOutputAccumulator g_snapshot_accumulator;

ov::genai::modeling::models::Qwen3_5TextModelConfig apply_qwen3_5_layer_limit(
    const ov::genai::modeling::models::Qwen3_5TextModelConfig& input_cfg) {
    auto cfg = input_cfg;
    const auto env_limit = get_qwen3_5_layer_limit_from_env();
    if (!env_limit.has_value()) {
        return cfg;
    }

    const int32_t limited_layers = std::clamp(*env_limit, 0, cfg.num_hidden_layers);
    cfg.num_hidden_layers = limited_layers;

    if (!cfg.layer_types.empty() && cfg.layer_types.size() >= static_cast<size_t>(limited_layers)) {
        cfg.layer_types.resize(static_cast<size_t>(limited_layers));
    }

    return cfg;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

Qwen3_5EmbeddingInjector::Qwen3_5EmbeddingInjector(BuilderContext& ctx, const std::string& name, Module* parent)
    : Module(name, ctx, parent) {}

Tensor Qwen3_5EmbeddingInjector::forward(const Tensor& inputs_embeds,
                                         const Tensor& visual_embeds,
                                         const Tensor& visual_pos_mask) const {
    auto mask = visual_pos_mask.unsqueeze(2);
    auto updates = visual_embeds.to(inputs_embeds.dtype());
    return ops::tensor::masked_scatter(inputs_embeds, mask, updates);
}

Qwen3_5RMSNorm::Qwen3_5RMSNorm(BuilderContext& ctx, const std::string& name, float eps, Module* parent)
    : Module(name, ctx, parent),
      eps_(eps) {
    weight_param_ = &register_parameter("weight");
}

const Tensor& Qwen3_5RMSNorm::weight() const {
    if (!weight_param_) {
        OPENVINO_THROW("Qwen3_5RMSNorm weight parameter is not registered");
    }
    return weight_param_->value();
}

Tensor Qwen3_5RMSNorm::forward(const Tensor& x) const {
    auto x_f32 = x.to(ov::element::f32);
    // Qwen3.5 uses (1 + weight) as the RMS scale factor.
    auto gamma = 1.0f + weight().to(ov::element::f32);
    auto rms_node = std::make_shared<ov::op::internal::RMS>(
        x_f32.output(), gamma.output(), static_cast<double>(eps_), x.dtype());
    return Tensor(rms_node->output(0), x.context());
}

std::pair<Tensor, Tensor> Qwen3_5RMSNorm::forward(const Tensor& x, const Tensor& residual) const {
    auto sum = x + residual;
    return {forward(sum), sum};
}

Qwen3_5Attention::Qwen3_5Attention(BuilderContext& ctx,
                                       const std::string& name,
                                       const Qwen3_5TextModelConfig& cfg,
                                       Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads),
      head_dim_(cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / cfg.num_attention_heads)),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      q_norm_(ctx, "q_norm", cfg.rms_norm_eps, this),
      k_norm_(ctx, "k_norm", cfg.rms_norm_eps, this) {
    if (num_heads_ <= 0 || head_dim_ <= 0 || num_kv_heads_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3_5 attention configuration");
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

const Tensor& Qwen3_5Attention::q_proj_weight() const {
    if (!q_proj_param_) {
        OPENVINO_THROW("Qwen3_5Attention q_proj parameter is not registered");
    }
    return q_proj_param_->value();
}

const Tensor& Qwen3_5Attention::k_proj_weight() const {
    if (!k_proj_param_) {
        OPENVINO_THROW("Qwen3_5Attention k_proj parameter is not registered");
    }
    return k_proj_param_->value();
}

const Tensor& Qwen3_5Attention::v_proj_weight() const {
    if (!v_proj_param_) {
        OPENVINO_THROW("Qwen3_5Attention v_proj parameter is not registered");
    }
    return v_proj_param_->value();
}

const Tensor& Qwen3_5Attention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("Qwen3_5Attention o_proj parameter is not registered");
    }
    return o_proj_param_->value();
}

Tensor Qwen3_5Attention::forward(const Tensor& hidden_states,
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
        local_mask = attention_mask ? ops::llm::build_kv_causal_mask_with_attention(q_heads, cached.first, *attention_mask)
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

Qwen3_5MLP::Qwen3_5MLP(BuilderContext& ctx,
                           const std::string& name,
                           const Qwen3_5TextModelConfig& cfg,
                           int32_t intermediate_size,
                           Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3_5 MLP activation: ", cfg.hidden_act);
    }
    if (intermediate_size <= 0) {
        OPENVINO_THROW("Qwen3_5MLP intermediate size must be > 0");
    }
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& Qwen3_5MLP::gate_proj_weight() const {
    if (!gate_proj_param_) {
        OPENVINO_THROW("Qwen3_5MLP gate_proj parameter is not registered");
    }
    return gate_proj_param_->value();
}

const Tensor& Qwen3_5MLP::up_proj_weight() const {
    if (!up_proj_param_) {
        OPENVINO_THROW("Qwen3_5MLP up_proj parameter is not registered");
    }
    return up_proj_param_->value();
}

const Tensor& Qwen3_5MLP::down_proj_weight() const {
    if (!down_proj_param_) {
        OPENVINO_THROW("Qwen3_5MLP down_proj parameter is not registered");
    }
    return down_proj_param_->value();
}

Tensor Qwen3_5MLP::forward(const Tensor& x) const {
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

Qwen3_5GatedDeltaNet::Qwen3_5GatedDeltaNet(BuilderContext& ctx,
                                               const std::string& name,
                                               const Qwen3_5TextModelConfig& cfg,
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
        OPENVINO_THROW("Unsupported Qwen3_5 linear attention activation: ", cfg.hidden_act);
    }
    if (hidden_size_ <= 0 || num_v_heads_ <= 0 || num_k_heads_ <= 0 || head_k_dim_ <= 0 || head_v_dim_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3_5 linear attention configuration");
    }
    if (conv_kernel_size_ <= 0) {
        OPENVINO_THROW("Qwen3_5 linear_conv_kernel_dim must be > 0");
    }
    if ((num_v_heads_ % num_k_heads_) != 0) {
        OPENVINO_THROW("Qwen3_5 linear_num_value_heads must be divisible by linear_num_key_heads");
    }

    in_proj_qkv_param_ = &register_parameter("in_proj_qkv.weight");
    in_proj_z_param_ = &register_parameter("in_proj_z.weight");
    in_proj_b_param_ = &register_parameter("in_proj_b.weight");
    in_proj_a_param_ = &register_parameter("in_proj_a.weight");
    conv1d_param_ = &register_parameter("conv1d.weight");
    a_log_param_ = &register_parameter("A_log");
    dt_bias_param_ = &register_parameter("dt_bias");
    norm_param_ = &register_parameter("norm.weight");
    out_proj_param_ = &register_parameter("out_proj.weight");
}

const Tensor& Qwen3_5GatedDeltaNet::in_proj_qkv_weight() const {
    if (!in_proj_qkv_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet in_proj_qkv parameter is not registered");
    }
    return in_proj_qkv_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::in_proj_z_weight() const {
    if (!in_proj_z_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet in_proj_z parameter is not registered");
    }
    return in_proj_z_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::in_proj_b_weight() const {
    if (!in_proj_b_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet in_proj_b parameter is not registered");
    }
    return in_proj_b_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::in_proj_a_weight() const {
    if (!in_proj_a_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet in_proj_a parameter is not registered");
    }
    return in_proj_a_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::conv1d_weight() const {
    if (!conv1d_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet conv1d parameter is not registered");
    }
    return conv1d_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::a_log() const {
    if (!a_log_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet A_log parameter is not registered");
    }
    return a_log_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::dt_bias() const {
    if (!dt_bias_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet dt_bias parameter is not registered");
    }
    return dt_bias_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::out_proj_weight() const {
    if (!out_proj_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet out_proj parameter is not registered");
    }
    return out_proj_param_->value();
}

Tensor Qwen3_5GatedDeltaNet::apply_depthwise_causal_conv(const Tensor& mixed_qkv,
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

Tensor Qwen3_5GatedDeltaNet::rms_norm_gated(const Tensor& x, const Tensor& z) const {
    auto x_f32 = x.to(ov::element::f32);
    auto z_f32 = z.to(ov::element::f32);
    auto var = x_f32.pow(2.0f).mean(-1, true);
    auto norm = x_f32 * (var + eps_).rsqrt();
    auto gate = ops::silu(z_f32);
    auto weighted = norm * norm_param_->value().to(ov::element::f32);
    return (weighted * gate).to(x.dtype());
}

Tensor Qwen3_5GatedDeltaNet::forward(const Tensor& hidden_states,
                                       const Tensor& beam_idx,
                                       const Tensor* attention_mask,
                                       const Tensor* cache_position,
                                       const Tensor* state_update_mode) const {
    (void)cache_position;
    auto* op_ctx = hidden_states.context();
    Tensor state_update_mode_tensor = state_update_mode
        ? *state_update_mode
        : Tensor(ops::const_vec(op_ctx, std::vector<int32_t>{1}), op_ctx);

    Tensor masked_hidden = hidden_states;
    if (attention_mask) {
        masked_hidden = hidden_states * attention_mask->to(hidden_states.dtype()).unsqueeze(2);
    }

    const int32_t ratio = num_v_heads_ / num_k_heads_;

    auto projected_qkv = ops::linear(masked_hidden, in_proj_qkv_weight());
    auto projected_z = ops::linear(masked_hidden, in_proj_z_weight());
    auto projected_b = ops::linear(masked_hidden, in_proj_b_weight());
    auto projected_a = ops::linear(masked_hidden, in_proj_a_weight());

    // Keep Z in head layout; b/a are already [B, S, num_v_heads].
    auto z = projected_z.reshape({0, 0, num_v_heads_, head_v_dim_});
    auto b = projected_b;
    auto a = projected_a;

    auto mixed_qkv = projected_qkv.permute({0, 2, 1});

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
        // ── FusedConv op path: fuses Gather + Concat + GroupConv + SiLU + Slice ──
        auto conv_w_2d = conv1d_weight().reshape({conv_dim_, conv_kernel_size_}, false);

        if (g_snapshot_accumulator.active) {
            auto [conv_out, conv_state, conv_snap] = ops::fused_conv_with_snapshots(
                mixed_qkv, conv_w_2d, beam_idx, conv_init, conv_var, state_update_mode_tensor,
                g_snapshot_accumulator.snapshot_max_seq);
            mixed_after_conv = conv_out;
            g_snapshot_accumulator.entries.push_back(
                {"snapshot." + conv_info.variable_id, conv_snap.output()});
        } else {
            auto fused_result = ops::fused_conv(
                mixed_qkv, conv_w_2d, beam_idx, conv_init, conv_var, state_update_mode_tensor);
            mixed_after_conv = fused_result.first;
        }
    } else {
        // ── Fallback: original decomposed path ──
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

    if (ratio > 1 && !use_linear_attention_op()) {
        q_heads = ops::llm::repeat_kv(q_heads.permute({0, 2, 1, 3}), num_v_heads_, num_k_heads_, head_k_dim_)
                      .permute({0, 2, 1, 3});
        k_heads = ops::llm::repeat_kv(k_heads.permute({0, 2, 1, 3}), num_v_heads_, num_k_heads_, head_k_dim_)
                      .permute({0, 2, 1, 3});
    }

    auto reduce_kdim = ops::const_vec(op_ctx, std::vector<int64_t>{-1});
    auto q_f32 = q_heads.to(ov::element::f32);
    auto k_f32 = k_heads.to(ov::element::f32);
    auto v_f32 = v_heads.to(ov::element::f32);

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

    Tensor core_attn_tensor;  // [B, S, num_v_heads, head_v_dim]

    if (use_linear_attention_op()) {
        // ── Fused LinearAttention op path (mirrors FusedConv pattern) ──
        // No ReadValue/Assign — LinearAttention manages the variable exclusively.
        // The GPU impl reads from variable memory (if set) or from recurrent_init (first iteration),
        // and writes updated state directly to variable memory.
        if (g_snapshot_accumulator.active) {
            auto [attn_out, recur_state, recur_snap] = ops::linear_attention_with_snapshots(
                q_f32, k_f32, v_f32, beta, g, recurrent_init, recurrent_var, state_update_mode_tensor,
                g_snapshot_accumulator.snapshot_max_seq);
            core_attn_tensor = attn_out;
            g_snapshot_accumulator.entries.push_back(
                {"snapshot." + recurrent_info.variable_id, recur_snap.output()});
        } else {
            auto la_result = ops::linear_attention(q_f32, k_f32, v_f32, beta, g, recurrent_init, recurrent_var, state_update_mode_tensor);
            core_attn_tensor = la_result.first;
        }
    } else {
        // ── TensorIterator path (default) ──
        // Traditional ReadValue + Gather + Assign pattern for variable state management.
        auto recurrent_read = std::make_shared<ov::op::v6::ReadValue>(recurrent_init.output(), recurrent_var);
        auto recurrent_cached = ops::gather(Tensor(recurrent_read->output(0), op_ctx), beam_idx, 0);

        auto q_ss = Tensor(std::make_shared<ov::op::v1::ReduceSum>(q_f32.pow(2.0f).output(), reduce_kdim, true), op_ctx);
        auto k_ss = Tensor(std::make_shared<ov::op::v1::ReduceSum>(k_f32.pow(2.0f).output(), reduce_kdim, true), op_ctx);
        auto q_normed = q_f32 * (q_ss + 1e-6f).rsqrt();
        auto k_normed = k_f32 * (k_ss + 1e-6f).rsqrt();
        auto q_scaled = q_normed * (1.0f / std::sqrt(static_cast<float>(head_k_dim_)));

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

        core_attn_tensor = Tensor(core_attn, op_ctx);
    }
    auto gated_4d = rms_norm_gated(core_attn_tensor, z);
    auto merged = gated_4d.reshape({0, 0, value_dim_});
    return ops::linear(merged, out_proj_weight()).to(masked_hidden.dtype());
}

Qwen3_5DecoderLayer::Qwen3_5DecoderLayer(BuilderContext& ctx,
                                             const std::string& name,
                                             const Qwen3_5TextModelConfig& cfg,
                                             int32_t layer_idx,
                                             Module* parent)
    : Module(name, ctx, parent),
      layer_type_(cfg.layer_types.at(static_cast<size_t>(layer_idx))),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {
    if (layer_type_ == "full_attention") {
        self_attn_ = std::make_unique<Qwen3_5Attention>(ctx, "self_attn", cfg, this);
    } else if (layer_type_ == "linear_attention") {
        linear_attn_ = std::make_unique<Qwen3_5GatedDeltaNet>(ctx, "linear_attn", cfg, layer_idx, this);
    } else {
        OPENVINO_THROW("Unsupported Qwen3_5 layer type: ", layer_type_);
    }

    if (cfg.is_moe_enabled()) {
        moe_mlp_ = std::make_unique<Qwen3_5SparseMoeBlock>(ctx, "mlp", cfg, this);
    } else {
        dense_mlp_ = std::make_unique<Qwen3_5MLP>(ctx, "mlp", cfg, cfg.intermediate_size, this);
    }
}

std::pair<Tensor, Tensor> Qwen3_5DecoderLayer::forward(const Tensor& hidden_states,
                                                         const Tensor& beam_idx,
                                                         const Tensor& rope_cos,
                                                         const Tensor& rope_sin,
                                                         const Tensor* full_attention_mask,
                                                         const Tensor* linear_attention_mask,
                                                         const Tensor* cache_position,
                                                         const std::optional<Tensor>& residual,
                                                         const Tensor* state_update_mode,
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
        mixed = self_attn_->forward(normed,
                                    beam_idx,
                                    rope_cos,
                                    rope_sin,
                                    full_attention_mask,
                                    precomputed_full_attn_sdpa_mask);
    } else {
        mixed = linear_attn_->forward(normed, beam_idx, linear_attention_mask, cache_position, state_update_mode);
    }

    auto post = post_attention_layernorm_.forward(mixed, next_residual);
    Tensor mlp_out = dense_mlp_ ? dense_mlp_->forward(post.first) : moe_mlp_->forward(post.first);
    return {mlp_out, post.second};
}

Qwen3_5Model::Qwen3_5Model(BuilderContext& ctx, const Qwen3_5TextModelConfig& cfg, Module* parent)
    : Module("model", ctx, parent),
      cfg_(cfg),
      embed_tokens_(ctx, "embed_tokens", this),
      embedding_injector_(ctx, "embedding_injector", this),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      head_dim_(cfg.head_dim > 0
                    ? cfg.head_dim
                    : (cfg.num_attention_heads > 0 ? (cfg.hidden_size / cfg.num_attention_heads) : 0)),
      rope_theta_(cfg.rope_theta) {
    if (head_dim_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3_5 head dimension");
    }
    rotary_dim_ = static_cast<int32_t>(std::floor(static_cast<float>(head_dim_) * cfg.partial_rotary_factor));
    rotary_dim_ = std::max<int32_t>(0, std::min<int32_t>(rotary_dim_, head_dim_));
    if ((rotary_dim_ % 2) != 0) {
        rotary_dim_ -= 1;
    }
    OPENVINO_ASSERT(rotary_dim_ > 0, "Qwen3_5 rotary dimension must be > 0");

    Qwen3_5TextModelConfig layer_cfg = cfg;
    if (layer_cfg.layer_types.empty()) {
        layer_cfg.layer_types = build_default_layer_types(cfg.num_hidden_layers, cfg.full_attention_interval);
    }
    if (layer_cfg.layer_types.size() != static_cast<size_t>(cfg.num_hidden_layers)) {
        OPENVINO_THROW("Qwen3_5 layer_types size mismatch with num_hidden_layers");
    }

    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", layer_cfg, i, this);
    }
}

std::pair<Tensor, Tensor> Qwen3_5Model::build_mrope_cos_sin(const Tensor& position_ids) const {
    auto* ctx = position_ids.context();
    const int32_t half_dim = rotary_dim_ / 2;
    std::vector<float> inv_freq(static_cast<size_t>(half_dim));
    for (int32_t i = 0; i < half_dim; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(rotary_dim_);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(rope_theta_, exponent);
    }

    auto inv_freq_const = ops::const_vec(ctx, inv_freq);
    Tensor inv_freq_tensor(inv_freq_const, ctx);
    auto inv_freq_reshaped =
        inv_freq_tensor.reshape({1, 1, static_cast<int64_t>(half_dim)}, false);

    const auto pos_rank = position_ids.output().get_partial_shape().rank();
    if (pos_rank.is_static() && pos_rank.get_length() == 2) {
        auto pos_f = position_ids.to(ov::element::f32);
        auto freqs = pos_f.unsqueeze(2) * inv_freq_reshaped;
        return {freqs.cos(), freqs.sin()};
    }

    auto pos_t = ops::slice(position_ids, 0, 1, 1, 0).squeeze(0).to(ov::element::f32);
    auto pos_h = ops::slice(position_ids, 1, 2, 1, 0).squeeze(0).to(ov::element::f32);
    auto pos_w = ops::slice(position_ids, 2, 3, 1, 0).squeeze(0).to(ov::element::f32);

    auto freqs_t = pos_t.unsqueeze(2) * inv_freq_reshaped;
    if (!cfg_.mrope_interleaved) {
        return {freqs_t.cos(), freqs_t.sin()};
    }

    auto freqs_h = pos_h.unsqueeze(2) * inv_freq_reshaped;
    auto freqs_w = pos_w.unsqueeze(2) * inv_freq_reshaped;
    auto freqs_all = ops::tensor::stack({freqs_t, freqs_h, freqs_w}, 0);
    auto freqs = ops::rope::mrope_interleaved(freqs_all, cfg_.mrope_section);
    return {freqs.cos(), freqs.sin()};
}

Tensor Qwen3_5Model::forward_impl(const Tensor* input_ids,
                                  const Tensor* inputs_embeds,
                                  const Tensor& position_ids,
                                  const Tensor& beam_idx,
                                  const Tensor& full_attention_mask,
                                  const Tensor* linear_attention_mask,
                                  const Tensor* cache_position,
                                  const Tensor* visual_embeds,
                                  const Tensor* visual_pos_mask,
                                  const Tensor* state_update_mode) {
    OPENVINO_ASSERT((input_ids != nullptr) || (inputs_embeds != nullptr),
                    "Either input_ids or inputs_embeds must be provided");
    OPENVINO_ASSERT(!(input_ids != nullptr && inputs_embeds != nullptr),
                    "input_ids and inputs_embeds are mutually exclusive");
    OPENVINO_ASSERT((visual_embeds == nullptr) == (visual_pos_mask == nullptr),
                    "visual_embeds and visual_pos_mask must be provided together");

    const auto source_ps = (inputs_embeds ? inputs_embeds->output() : input_ids->output()).get_partial_shape();
    const auto source_rank = source_ps.rank();
    OPENVINO_ASSERT(!source_rank.is_static() || source_rank.get_length() >= 2,
                    "input source must have rank >= 2");

    const auto position_ps = position_ids.output().get_partial_shape();
    const auto position_rank = position_ps.rank();
    OPENVINO_ASSERT(!position_rank.is_static() || position_rank.get_length() == 3,
                    "position_ids must have rank 3 [3, B, S]");

    if (source_rank.is_static() && source_rank.get_length() >= 2 && position_rank.is_static() &&
        position_rank.get_length() == 3 && source_ps[0].is_static() && source_ps[1].is_static() &&
        position_ps[1].is_static() && position_ps[2].is_static()) {
        OPENVINO_ASSERT(position_ps[1].get_length() == source_ps[0].get_length() &&
                        position_ps[2].get_length() == source_ps[1].get_length(),
                        "position_ids shape mismatch with input source");
    }

    const auto full_mask_ps = full_attention_mask.output().get_partial_shape();
    const auto full_mask_rank = full_mask_ps.rank();
    OPENVINO_ASSERT(!full_mask_rank.is_static() || full_mask_rank.get_length() >= 2,
                    "full_attention_mask must have rank >= 2");

    if (source_rank.is_static() && source_rank.get_length() >= 2 && full_mask_rank.is_static() &&
        full_mask_rank.get_length() >= 2 && source_ps[0].is_static() && source_ps[1].is_static() &&
        full_mask_ps[full_mask_rank.get_length() - 2].is_static() &&
        full_mask_ps[full_mask_rank.get_length() - 1].is_static()) {
        OPENVINO_ASSERT(full_mask_ps[full_mask_rank.get_length() - 2].get_length() == source_ps[0].get_length(),
                        "full_attention_mask batch dimension mismatch");
        OPENVINO_ASSERT(full_mask_ps[full_mask_rank.get_length() - 1].get_length() == source_ps[1].get_length(),
                        "full_attention_mask sequence dimension mismatch");
    }

    Tensor hidden_states = inputs_embeds ? *inputs_embeds : embed_tokens_.forward(*input_ids);
    if (visual_embeds && visual_pos_mask) {
        hidden_states = embedding_injector_.forward(hidden_states, *visual_embeds, *visual_pos_mask);
    }
    auto cos_sin = build_mrope_cos_sin(position_ids);
    const Tensor& seq_source = inputs_embeds ? *inputs_embeds : *input_ids;
    auto* op_ctx = seq_source.context();
    auto q_len_1d = Tensor(shape::dim(seq_source, 1), op_ctx);
    auto shared_full_attn_sdpa_mask =
        ops::llm::build_kv_causal_mask_with_attention_from_q_len(q_len_1d, full_attention_mask);

    std::optional<Tensor> linear_mask_view;
    const Tensor* linear_mask = nullptr;
    if (linear_attention_mask) {
        auto q_len = shape::dim(seq_source, 1);
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

    // Sort capture IDs for efficient lookup during the layer loop
    auto sorted_capture_ids = capture_layer_ids_;
    std::sort(sorted_capture_ids.begin(), sorted_capture_ids.end());
    size_t capture_idx = 0;

    std::optional<Tensor> residual;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto out = layers_[i].forward(hidden_states,
                                 beam_idx,
                                 cos_sin.first,
                                 cos_sin.second,
                                 &full_attention_mask,
                                 linear_mask,
                                 cache_position,
                                 residual,
                                 state_update_mode,
                                 &shared_full_attn_sdpa_mask);
        hidden_states = out.first;
        residual = out.second;

        // Capture intermediate hidden state at selected layer indices
        if (capture_idx < sorted_capture_ids.size() &&
            static_cast<int32_t>(i) == sorted_capture_ids[capture_idx]) {
            Tensor pre_norm = residual ? (hidden_states + *residual) : hidden_states;
            captured_hidden_.push_back(pre_norm);
            ++capture_idx;
        }
    }

    if (residual) {
        return norm_.forward(hidden_states, *residual).first;
    }
    return norm_.forward(hidden_states);
}

Tensor Qwen3_5Model::forward(const Tensor& input_ids,
                             const Tensor& position_ids,
                             const Tensor& beam_idx,
                             const Tensor& full_attention_mask,
                             const Tensor* linear_attention_mask,
                             const Tensor* cache_position,
                             const Tensor* visual_embeds,
                             const Tensor* visual_pos_mask,
                             const Tensor* state_update_mode) {
    return forward_impl(&input_ids,
                        nullptr,
                        position_ids,
                        beam_idx,
                        full_attention_mask,
                        linear_attention_mask,
                        cache_position,
                        visual_embeds,
                        visual_pos_mask,
                        state_update_mode);
}

Tensor Qwen3_5Model::forward_embeds(const Tensor& inputs_embeds,
                                    const Tensor& position_ids,
                                    const Tensor& beam_idx,
                                    const Tensor& full_attention_mask,
                                    const Tensor* linear_attention_mask,
                                    const Tensor* cache_position,
                                    const Tensor* visual_embeds,
                                    const Tensor* visual_pos_mask,
                                    const Tensor* state_update_mode) {
    return forward_impl(nullptr,
                        &inputs_embeds,
                        position_ids,
                        beam_idx,
                        full_attention_mask,
                        linear_attention_mask,
                        cache_position,
                        visual_embeds,
                        visual_pos_mask,
                        state_update_mode);
}

VocabEmbedding& Qwen3_5Model::embed_tokens() {
    return embed_tokens_;
}

Qwen3_5ForCausalLM::Qwen3_5ForCausalLM(BuilderContext& ctx, const Qwen3_5TextModelConfig& cfg, Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_head_(ctx, "lm_head", this) {
    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}

Tensor Qwen3_5ForCausalLM::forward(const Tensor& input_ids,
                                   const Tensor& position_ids,
                                   const Tensor& beam_idx,
                                   const Tensor& full_attention_mask,
                                   const Tensor* linear_attention_mask,
                                   const Tensor* cache_position,
                                   const Tensor* visual_embeds,
                                   const Tensor* visual_pos_mask,
                                   const Tensor* state_update_mode) {
    auto hidden = model_.forward(input_ids,
                                 position_ids,
                                 beam_idx,
                                 full_attention_mask,
                                 linear_attention_mask,
                                 cache_position,
                                 visual_embeds,
                                 visual_pos_mask,
                                 state_update_mode);
    return lm_head_.forward(hidden);
}

Tensor Qwen3_5ForCausalLM::forward_embeds(const Tensor& inputs_embeds,
                                          const Tensor& position_ids,
                                          const Tensor& beam_idx,
                                          const Tensor& full_attention_mask,
                                          const Tensor* linear_attention_mask,
                                          const Tensor* cache_position,
                                          const Tensor* visual_embeds,
                                          const Tensor* visual_pos_mask,
                                          const Tensor* state_update_mode) {
    auto hidden = model_.forward_embeds(inputs_embeds,
                                        position_ids,
                                        beam_idx,
                                        full_attention_mask,
                                        linear_attention_mask,
                                        cache_position,
                                        visual_embeds,
                                        visual_pos_mask,
                                        state_update_mode);
    return lm_head_.forward(hidden);
}

std::pair<Tensor, Tensor> Qwen3_5Model::forward_with_selected_layers(
    const Tensor& input_ids,
    const Tensor& position_ids,
    const Tensor& beam_idx,
    const Tensor& full_attention_mask,
    const Tensor* linear_attention_mask,
    const Tensor* cache_position,
    const Tensor* state_update_mode,
    const std::vector<int32_t>& layer_ids,
    const Tensor* visual_embeds,
    const Tensor* visual_pos_mask) {
    // Set up captures, then delegate to the shared forward_impl path
    capture_layer_ids_ = layer_ids;
    captured_hidden_.clear();

    auto final_out = forward(input_ids, position_ids, beam_idx, full_attention_mask,
                             linear_attention_mask, cache_position,
                             visual_embeds, visual_pos_mask, state_update_mode);

    capture_layer_ids_.clear();

    if (captured_hidden_.empty()) {
        return {final_out, final_out};
    }
    auto concat_hidden = ops::concat(captured_hidden_, 2);
    captured_hidden_.clear();
    return {final_out, concat_hidden};
}

namespace {

Qwen3_5TextModelConfig make_text_model_config(const Qwen3_5Config& cfg) {
    Qwen3_5TextModelConfig text_cfg;
    text_cfg.architecture = "qwen3_5";
    text_cfg.hidden_size = cfg.text.hidden_size;
    text_cfg.num_attention_heads = cfg.text.num_attention_heads;
    text_cfg.num_key_value_heads = cfg.text.num_key_value_heads > 0 ? cfg.text.num_key_value_heads : cfg.text.num_attention_heads;
    text_cfg.head_dim = cfg.text.resolved_head_dim();
    text_cfg.intermediate_size = cfg.text.intermediate_size;
    text_cfg.num_hidden_layers = cfg.text.num_hidden_layers;
    text_cfg.vocab_size = cfg.text.vocab_size;
    text_cfg.max_position_embeddings = cfg.text.max_position_embeddings;
    text_cfg.rms_norm_eps = cfg.text.rms_norm_eps;
    text_cfg.rope_theta = cfg.text.rope_theta;
    text_cfg.partial_rotary_factor = cfg.text.partial_rotary_factor;
    text_cfg.hidden_act = cfg.text.hidden_act;
    text_cfg.attention_bias = cfg.text.attention_bias;
    text_cfg.tie_word_embeddings = cfg.text.tie_word_embeddings;
    text_cfg.layer_types = cfg.text.layer_types;
    text_cfg.full_attention_interval = cfg.text.full_attention_interval;
    text_cfg.linear_conv_kernel_dim = cfg.text.linear_conv_kernel_dim;
    text_cfg.linear_key_head_dim = cfg.text.linear_key_head_dim;
    text_cfg.linear_value_head_dim = cfg.text.linear_value_head_dim;
    text_cfg.linear_num_key_heads = cfg.text.linear_num_key_heads;
    text_cfg.linear_num_value_heads = cfg.text.linear_num_value_heads;
    text_cfg.moe_intermediate_size = cfg.text.moe_intermediate_size;
    text_cfg.shared_expert_intermediate_size = cfg.text.shared_expert_intermediate_size;
    text_cfg.num_experts = cfg.text.num_experts;
    text_cfg.num_experts_per_tok = cfg.text.num_experts_per_tok;
    text_cfg.norm_topk_prob = cfg.text.norm_topk_prob;
    text_cfg.output_router_logits = cfg.text.output_router_logits;
    text_cfg.router_aux_loss_coef = cfg.text.router_aux_loss_coef;
    text_cfg.mrope_interleaved = cfg.text.rope.mrope_interleaved;
    text_cfg.mrope_section = cfg.text.rope.mrope_section;
    return text_cfg;
}

}  // namespace

std::shared_ptr<ov::Model> create_qwen3_5_text_model(
    const Qwen3_5Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds,
    bool enable_visual_inputs) {
    auto text_cfg = make_text_model_config(cfg);

    const auto effective_cfg = apply_qwen3_5_layer_limit(text_cfg);

    BuilderContext ctx;
    Qwen3_5ForCausalLM model(ctx, effective_cfg);
    // HF Qwen3.5-MoE checkpoints store text weights under
    // model.language_model.layers.N.*, while this model registers
    // model.layers[N].* parameters.
    // Add per-layer rules first so generic prefix rules don't consume them.
    for (int32_t i = 0; i < effective_cfg.num_hidden_layers; ++i) {
        const std::string idx = std::to_string(i);
        model.packed_mapping().rules.push_back(
            {"model.language_model.layers." + idx + ".", "model.layers[" + idx + "].", 0});
        model.packed_mapping().rules.push_back(
            {"language_model.layers." + idx + ".", "model.layers[" + idx + "].", 0});
    }
    model.packed_mapping().rules.push_back({"model.language_model.", "model.", 0});
    model.packed_mapping().rules.push_back({"language_model.", "model.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_missing = false;
    options.allow_unmatched = true;
    options.report_missing = true;
    options.report_unmatched = false;
    (void)ov::genai::modeling::weights::load_model(model, source, finalizer, options);

    const auto float_type = ov::element::f32;
    auto attention_mask = ctx.parameter(Qwen3_5TextIO::kAttentionMask, ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter(Qwen3_5TextIO::kPositionIds, ov::element::i64, ov::PartialShape{3, -1, -1});
    auto beam_idx = ctx.parameter(Qwen3_5TextIO::kBeamIdx, ov::element::i32, ov::PartialShape{-1});

    Tensor input_ids;
    Tensor inputs_embeds;
    if (use_inputs_embeds) {
        inputs_embeds = ctx.parameter(Qwen3_5TextIO::kInputsEmbeds, float_type, ov::PartialShape{-1, -1, cfg.text.hidden_size});
    } else {
        input_ids = ctx.parameter(Qwen3_5TextIO::kInputIds, ov::element::i64, ov::PartialShape{-1, -1});
    }

    const Tensor* visual_embeds_ptr = nullptr;
    const Tensor* visual_pos_mask_ptr = nullptr;
    Tensor visual_embeds;
    Tensor visual_pos_mask;
    if (enable_visual_inputs) {
        visual_embeds = ctx.parameter(Qwen3_5TextIO::kVisualEmbeds, float_type, ov::PartialShape{-1, -1, cfg.text.hidden_size});
        visual_pos_mask = ctx.parameter(Qwen3_5TextIO::kVisualPosMask, ov::element::boolean, ov::PartialShape{-1, -1});
        visual_embeds_ptr = &visual_embeds;
        visual_pos_mask_ptr = &visual_pos_mask;
    }

    // Optionally activate snapshot accumulation for kernel overhead benchmarking
    const bool do_snapshots = force_state_snapshots() && use_state_snapshots()
                              && use_fused_conv_op() && use_linear_attention_op();
    if (do_snapshots) {
        g_snapshot_accumulator.entries.clear();
        g_snapshot_accumulator.active = true;
    }

    Tensor logits;
    if (use_inputs_embeds) {
        logits = model.forward_embeds(inputs_embeds,
                                      position_ids,
                                      beam_idx,
                                      attention_mask,
                                      &attention_mask,
                                      nullptr,
                                      visual_embeds_ptr,
                                      visual_pos_mask_ptr);
    } else {
        logits = model.forward(input_ids, position_ids, beam_idx, attention_mask, &attention_mask, nullptr, visual_embeds_ptr, visual_pos_mask_ptr);
    }

    if (do_snapshots) {
        g_snapshot_accumulator.active = false;
    }

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, Qwen3_5TextIO::kLogits);

    ov::OutputVector model_outputs = {result->output(0)};
    if (do_snapshots) {
        for (auto& [name, output] : g_snapshot_accumulator.entries) {
            auto snap_result = std::make_shared<ov::op::v0::Result>(output);
            set_name(snap_result, name);
            model_outputs.push_back(snap_result->output(0));
        }
        g_snapshot_accumulator.entries.clear();
    }
    auto ov_model = ctx.build_model(model_outputs);
    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
    return ov_model;
}

std::shared_ptr<ov::Model> create_qwen3_5_dflash_target_model(
    const Qwen3_5Config& cfg,
    const std::vector<int32_t>& target_layer_ids,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    int32_t snapshot_block_size,
    bool enable_visual_inputs) {
    auto text_cfg = make_text_model_config(cfg);
    const auto effective_cfg = apply_qwen3_5_layer_limit(text_cfg);

    BuilderContext ctx;
    Qwen3_5ForCausalLM model(ctx, effective_cfg);

    for (int32_t i = 0; i < effective_cfg.num_hidden_layers; ++i) {
        const std::string idx = std::to_string(i);
        model.packed_mapping().rules.push_back(
            {"model.language_model.layers." + idx + ".", "model.layers[" + idx + "].", 0});
        model.packed_mapping().rules.push_back(
            {"language_model.layers." + idx + ".", "model.layers[" + idx + "].", 0});
    }
    model.packed_mapping().rules.push_back({"model.language_model.", "model.", 0});
    model.packed_mapping().rules.push_back({"language_model.", "model.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_missing = false;
    options.allow_unmatched = true;
    options.report_missing = true;
    options.report_unmatched = false;
    (void)ov::genai::modeling::weights::load_model(model, source, finalizer, options);

    auto input_ids = ctx.parameter(Qwen3_5TextIO::kInputIds, ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter(Qwen3_5TextIO::kAttentionMask, ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter(Qwen3_5TextIO::kPositionIds, ov::element::i64, ov::PartialShape{3, -1, -1});
    auto beam_idx = ctx.parameter(Qwen3_5TextIO::kBeamIdx, ov::element::i32, ov::PartialShape{-1});
    auto state_update_mode = ctx.parameter("state_update_mode", ov::element::i32, ov::PartialShape{1});

    const Tensor* visual_embeds_ptr = nullptr;
    const Tensor* visual_pos_mask_ptr = nullptr;
    Tensor visual_embeds;
    Tensor visual_pos_mask;
    if (enable_visual_inputs) {
        visual_embeds = ctx.parameter(Qwen3_5TextIO::kVisualEmbeds, ov::element::f32,
                                       ov::PartialShape{-1, -1, cfg.text.hidden_size});
        visual_pos_mask = ctx.parameter(Qwen3_5TextIO::kVisualPosMask, ov::element::boolean,
                                         ov::PartialShape{-1, -1});
        visual_embeds_ptr = &visual_embeds;
        visual_pos_mask_ptr = &visual_pos_mask;
    }

    // Enable snapshot accumulation during model construction
    g_snapshot_accumulator.entries.clear();
    g_snapshot_accumulator.active = use_state_snapshots() && use_fused_conv_op() && use_linear_attention_op();
    g_snapshot_accumulator.snapshot_max_seq = snapshot_block_size;

    auto outputs = model.model().forward_with_selected_layers(
        input_ids, position_ids, beam_idx, attention_mask, &attention_mask, nullptr, &state_update_mode, target_layer_ids,
        visual_embeds_ptr, visual_pos_mask_ptr);

    g_snapshot_accumulator.active = false;

    auto logits = model.lm_head().forward(outputs.first);
    auto hidden_out = outputs.second;

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits.output());
    auto hidden_result = std::make_shared<ov::op::v0::Result>(hidden_out.output());
    set_name(logits_result, "logits");
    set_name(hidden_result, "target_hidden");

    ov::OutputVector model_outputs = {logits_result->output(0), hidden_result->output(0)};

    // Add snapshot outputs as named model results
    for (auto& [name, output] : g_snapshot_accumulator.entries) {
        auto result = std::make_shared<ov::op::v0::Result>(output);
        set_name(result, name);
        model_outputs.push_back(result->output(0));
    }
    g_snapshot_accumulator.entries.clear();

    auto ov_model = ctx.build_model(model_outputs);

    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
    return ov_model;
}

std::shared_ptr<ov::Model> create_qwen3_5_embedding_model(
    const Qwen3_5Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    (void)cfg;
    BuilderContext ctx;

    Module root("model", ctx);
    VocabEmbedding embed(ctx, "embed_tokens", &root);

    // Add HF weight mapping rules for Qwen3.5
    root.packed_mapping().rules.push_back({"model.language_model.", "model.", 0});
    root.packed_mapping().rules.push_back({"language_model.", "model.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_unmatched = false;
    options.report_missing = true;
    ov::genai::modeling::weights::load_model(root, source, finalizer, options);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto output = embed.forward(input_ids);

    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "embeddings");
    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_5_lm_head_model(
    const Qwen3_5Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    const ov::element::Type& input_type) {
    BuilderContext ctx;

    Module root("", ctx);
    LMHead head(ctx, "lm_head", &root);

    if (!source.has("lm_head.weight") && cfg.tie_word_embeddings) {
        // Qwen3.5 safetensors may use several naming conventions depending on
        // whether the checkpoint is text-only or VLM.
        const std::vector<std::string> embed_candidates = {
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
        };
        std::string embed_weight;
        for (const auto& name : embed_candidates) {
            if (source.has(name)) {
                embed_weight = name;
                break;
            }
        }
        if (embed_weight.empty()) {
            OPENVINO_THROW("Missing lm_head.weight and no embedding weight available to tie.");
        }
        auto tied = finalizer.finalize(embed_weight, source, ctx.op_context());
        head.weight_param().bind(tied);
    }

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_unmatched = false;
    options.report_missing = true;
    ov::genai::modeling::weights::load_model(root, source, finalizer, options);

    auto hidden = ctx.parameter("hidden_states", input_type, ov::PartialShape{-1, -1, cfg.text.hidden_size});
    auto logits = head.forward(hidden);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");
    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_5_draft_helper_model(
    const Qwen3_5Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    const ov::element::Type& lm_head_input_type) {
    BuilderContext ctx;

    // ── Embedding path (under "model" prefix) ──
    Module embed_root("model", ctx);
    VocabEmbedding embed(ctx, "embed_tokens", &embed_root);

    embed_root.packed_mapping().rules.push_back({"model.language_model.", "model.", 0});
    embed_root.packed_mapping().rules.push_back({"language_model.", "model.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_unmatched = false;
    options.report_missing = true;
    ov::genai::modeling::weights::load_model(embed_root, source, finalizer, options);

    // ── LM head path (tied to embed_tokens) ──
    Module lm_root("", ctx);
    LMHead head(ctx, "lm_head", &lm_root);

    if (cfg.tie_word_embeddings) {
        head.tie_to(embed.weight_param());
    } else if (!source.has("lm_head.weight")) {
        const std::vector<std::string> embed_candidates = {
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
        };
        std::string embed_weight;
        for (const auto& name : embed_candidates) {
            if (source.has(name)) { embed_weight = name; break; }
        }
        if (!embed_weight.empty()) {
            auto tied = finalizer.finalize(embed_weight, source, ctx.op_context());
            head.weight_param().bind(tied);
        }
    }

    ov::genai::modeling::weights::load_model(lm_root, source, finalizer, options);

    // ── Inputs ──
    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto hidden_states = ctx.parameter("hidden_states", lm_head_input_type,
                                        ov::PartialShape{-1, -1, cfg.text.hidden_size});

    // ── Forward: both paths always execute (GPU overhead < 1ms for unused path) ──
    auto embeddings = embed.forward(input_ids);
    auto logits = head.forward(hidden_states);

    // ── Outputs ──
    auto embed_result = std::make_shared<ov::op::v0::Result>(embeddings.output());
    set_name(embed_result, "embeddings");
    auto logits_result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(logits_result, "logits");

    return ctx.build_model({embed_result->output(0), logits_result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_5_dflash_combined_draft_model(
    const Qwen3_5Config& qwen_cfg,
    const DFlashDraftConfig& draft_cfg,
    ov::genai::modeling::weights::WeightSource& target_source,
    ov::genai::modeling::weights::WeightFinalizer& target_finalizer,
    ov::genai::modeling::weights::WeightSource& draft_source,
    ov::genai::modeling::weights::WeightFinalizer& draft_finalizer) {
    BuilderContext ctx;

    // ── Embedding path (target weights under "model" prefix) ──
    Module embed_root("model", ctx);
    VocabEmbedding embed(ctx, "embed_tokens", &embed_root);
    embed_root.packed_mapping().rules.push_back({"model.language_model.", "model.", 0});
    embed_root.packed_mapping().rules.push_back({"language_model.", "model.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_unmatched = false;
    options.report_missing = true;
    ov::genai::modeling::weights::load_model(embed_root, target_source, target_finalizer, options);

    // ── Draft layers (draft weights) ──
    DFlashDraftModel draft_model(ctx, draft_cfg);
    ov::genai::modeling::weights::load_model(draft_model, draft_source, draft_finalizer);

    // ── LM head (target weights, tied to embed_tokens when configured) ──
    Module lm_root("", ctx);
    LMHead head(ctx, "lm_head", &lm_root);
    if (qwen_cfg.tie_word_embeddings) {
        head.tie_to(embed.weight_param());
    } else if (!target_source.has("lm_head.weight")) {
        const std::vector<std::string> embed_candidates = {
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
        };
        std::string embed_weight;
        for (const auto& name : embed_candidates) {
            if (target_source.has(name)) { embed_weight = name; break; }
        }
        if (!embed_weight.empty()) {
            auto tied = target_finalizer.finalize(embed_weight, target_source, ctx.op_context());
            head.weight_param().bind(tied);
        }
    }
    ov::genai::modeling::weights::load_model(lm_root, target_source, target_finalizer, options);

    // ── Inputs ──
    const ov::element::Type dtype = ov::element::f32;
    const int64_t ctx_dim = static_cast<int64_t>(draft_cfg.hidden_size) *
                            static_cast<int64_t>(draft_cfg.num_hidden_layers);
    auto target_hidden = ctx.parameter("target_hidden", dtype, ov::PartialShape{-1, -1, ctx_dim});
    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    // ── Forward: embed → draft → lm_head  (single graph) ──
    auto noise_embedding = embed.forward(input_ids);
    auto draft_hidden = draft_model.forward(target_hidden, noise_embedding, position_ids);
    auto logits = head.forward(draft_hidden);

    // ── Output ──
    auto logits_result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(logits_result, "logits");

    return ctx.build_model({logits_result->output(0)});
}

// ============================================================================
// DFlash Context KV Preprocessing Model
// ============================================================================
// Computes fc(target_hidden) → RMSNorm → K,V projections + KNorm + RoPE
// for all draft layers.  Runs once per verify cycle on newly-accepted tokens.
// Inputs:  target_hidden [1, A, ctx_dim], position_ids [1, A]
// Outputs: k_0..k_N [1, kv_heads, A, head_dim], v_0..v_N (same shape)
std::shared_ptr<ov::Model> create_qwen3_5_dflash_context_kv_model(
    const DFlashDraftConfig& draft_cfg,
    ov::genai::modeling::weights::WeightSource& draft_source,
    ov::genai::modeling::weights::WeightFinalizer& draft_finalizer) {
    BuilderContext ctx;

    DFlashDraftModel draft_model(ctx, draft_cfg);
    ov::genai::modeling::weights::load_model(draft_model, draft_source, draft_finalizer);

    const ov::element::Type dtype = ov::element::f32;
    const int64_t ctx_dim = static_cast<int64_t>(draft_cfg.hidden_size) *
                            static_cast<int64_t>(draft_cfg.num_hidden_layers);
    auto target_hidden = ctx.parameter("target_hidden", dtype, ov::PartialShape{-1, -1, ctx_dim});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    auto kv_pairs = draft_model.build_context_kv(target_hidden, position_ids);

    ov::OutputVector outputs;
    for (size_t i = 0; i < kv_pairs.size(); ++i) {
        auto k_result = std::make_shared<ov::op::v0::Result>(kv_pairs[i].first.output());
        set_name(k_result, "context_k_" + std::to_string(i));
        auto v_result = std::make_shared<ov::op::v0::Result>(kv_pairs[i].second.output());
        set_name(v_result, "context_v_" + std::to_string(i));
        outputs.push_back(k_result->output(0));
        outputs.push_back(v_result->output(0));
    }

    return ctx.build_model(outputs);
}

// ============================================================================
// DFlash Lightweight Draft Step Model
// ============================================================================
// Embed → attention (with pre-computed context KV) → MLP → lm_head.
// Runs once per draft cycle.  Skips fc + context KV computation entirely.
// Inputs:  input_ids [1, B], position_ids [1, B],
//          context_k_0..N [1, kv_heads, T, head_dim], context_v_0..N (same)
// Output:  logits [1, B, vocab_size]
std::shared_ptr<ov::Model> create_qwen3_5_dflash_step_model(
    const Qwen3_5Config& qwen_cfg,
    const DFlashDraftConfig& draft_cfg,
    ov::genai::modeling::weights::WeightSource& target_source,
    ov::genai::modeling::weights::WeightFinalizer& target_finalizer,
    ov::genai::modeling::weights::WeightSource& draft_source,
    ov::genai::modeling::weights::WeightFinalizer& draft_finalizer) {
    BuilderContext ctx;

    // ── Embedding (target weights) ──
    Module embed_root("model", ctx);
    VocabEmbedding embed(ctx, "embed_tokens", &embed_root);
    embed_root.packed_mapping().rules.push_back({"model.language_model.", "model.", 0});
    embed_root.packed_mapping().rules.push_back({"language_model.", "model.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_unmatched = false;
    options.report_missing = true;
    ov::genai::modeling::weights::load_model(embed_root, target_source, target_finalizer, options);

    // ── Draft layers ──
    DFlashDraftModel draft_model(ctx, draft_cfg);
    ov::genai::modeling::weights::load_model(draft_model, draft_source, draft_finalizer);

    // ── LM head (target weights) ──
    Module lm_root("", ctx);
    LMHead head(ctx, "lm_head", &lm_root);
    if (qwen_cfg.tie_word_embeddings) {
        head.tie_to(embed.weight_param());
    } else if (!target_source.has("lm_head.weight")) {
        const std::vector<std::string> embed_candidates = {
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
        };
        std::string embed_weight;
        for (const auto& name : embed_candidates) {
            if (target_source.has(name)) { embed_weight = name; break; }
        }
        if (!embed_weight.empty()) {
            auto tied = target_finalizer.finalize(embed_weight, target_source, ctx.op_context());
            head.weight_param().bind(tied);
        }
    }
    ov::genai::modeling::weights::load_model(lm_root, target_source, target_finalizer, options);

    // ── Inputs ──
    const ov::element::Type dtype = ov::element::f32;
    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    // Context KV inputs: one K,V pair per draft layer.
    const int32_t kv_heads = draft_cfg.num_key_value_heads > 0
                                 ? draft_cfg.num_key_value_heads
                                 : draft_cfg.num_attention_heads;
    const int32_t head_dim = draft_cfg.head_dim > 0
                                 ? draft_cfg.head_dim
                                 : (draft_cfg.hidden_size / draft_cfg.num_attention_heads);
    std::vector<std::pair<Tensor, Tensor>> context_kv;
    for (int32_t i = 0; i < draft_cfg.num_hidden_layers; ++i) {
        auto ck = ctx.parameter("context_k_" + std::to_string(i), dtype,
                                ov::PartialShape{-1, kv_heads, -1, head_dim});
        auto cv = ctx.parameter("context_v_" + std::to_string(i), dtype,
                                ov::PartialShape{-1, kv_heads, -1, head_dim});
        context_kv.push_back({ck, cv});
    }

    // ── Forward: embed → draft_with_cached_kv → lm_head ──
    auto noise_embedding = embed.forward(input_ids);
    auto draft_hidden = draft_model.forward_with_cached_kv(noise_embedding, position_ids, context_kv);
    auto logits = head.forward(draft_hidden);

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(logits_result, "logits");

    return ctx.build_model({logits_result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

