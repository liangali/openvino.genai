// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/dflash_draft/dflash_draft.hpp"

#include <cmath>

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/weights/weight_loader.hpp"

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

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

DFlashAttention::DFlashAttention(BuilderContext& ctx,
                                 const std::string& name,
                                 const DFlashDraftConfig& cfg,
                                 Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads),
      head_dim_(cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / cfg.num_attention_heads)),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      rope_theta_(cfg.rope_theta),
      attention_bias_(cfg.attention_bias),
      q_norm_(ctx, "q_norm", cfg.rms_norm_eps, this),
      k_norm_(ctx, "k_norm", cfg.rms_norm_eps, this) {
    if (num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid DFlash attention head configuration");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
    }

    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");

    if (attention_bias_) {
        q_bias_param_ = &register_parameter("q_proj.bias");
        k_bias_param_ = &register_parameter("k_proj.bias");
        v_bias_param_ = &register_parameter("v_proj.bias");
        o_bias_param_ = &register_parameter("o_proj.bias");
        q_bias_param_->set_optional(true);
        k_bias_param_->set_optional(true);
        v_bias_param_->set_optional(true);
        o_bias_param_->set_optional(true);
    }
}

const Tensor& DFlashAttention::q_proj_weight() const {
    if (!q_proj_param_) {
        OPENVINO_THROW("DFlashAttention q_proj parameter not registered");
    }
    return q_proj_param_->value();
}

const Tensor& DFlashAttention::k_proj_weight() const {
    if (!k_proj_param_) {
        OPENVINO_THROW("DFlashAttention k_proj parameter not registered");
    }
    return k_proj_param_->value();
}

const Tensor& DFlashAttention::v_proj_weight() const {
    if (!v_proj_param_) {
        OPENVINO_THROW("DFlashAttention v_proj parameter not registered");
    }
    return v_proj_param_->value();
}

const Tensor& DFlashAttention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("DFlashAttention o_proj parameter not registered");
    }
    return o_proj_param_->value();
}

const Tensor* DFlashAttention::q_proj_bias() const {
    return (q_bias_param_ && q_bias_param_->is_bound()) ? &q_bias_param_->value() : nullptr;
}

const Tensor* DFlashAttention::k_proj_bias() const {
    return (k_bias_param_ && k_bias_param_->is_bound()) ? &k_bias_param_->value() : nullptr;
}

const Tensor* DFlashAttention::v_proj_bias() const {
    return (v_bias_param_ && v_bias_param_->is_bound()) ? &v_bias_param_->value() : nullptr;
}

const Tensor* DFlashAttention::o_proj_bias() const {
    return (o_bias_param_ && o_bias_param_->is_bound()) ? &o_bias_param_->value() : nullptr;
}

Tensor DFlashAttention::forward(const Tensor& target_hidden,
                                const Tensor& hidden_states,
                                const Tensor& rope_cos,
                                const Tensor& rope_sin) const {
    auto kv_input = ops::concat({target_hidden, hidden_states}, 1);
    auto q = add_bias_if_present(ops::linear(hidden_states, q_proj_weight()), q_proj_bias());
    auto k = add_bias_if_present(ops::linear(kv_input, k_proj_weight()), k_proj_bias());
    auto v = add_bias_if_present(ops::linear(kv_input, v_proj_weight()), v_proj_bias());

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
    auto cos_tail = ops::llm::rope_tail(rope_cos, q_heads);
    auto sin_tail = ops::llm::rope_tail(rope_sin, q_heads);
    auto q_rot = ops::llm::apply_rope(q_heads, cos_tail, sin_tail, head_dim_, policy);
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    auto k_expanded = ops::llm::repeat_kv(k_rot, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(v_heads, num_heads_, num_kv_heads_, head_dim_);

    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, nullptr, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    auto out = add_bias_if_present(ops::linear(merged, o_proj_weight()), o_proj_bias());
    return out;
}

std::pair<Tensor, Tensor> DFlashAttention::build_context_kv(const Tensor& context_hidden,
                                                            const Tensor& rope_cos,
                                                            const Tensor& rope_sin) const {
    // K,V projection on context_hidden only (no draft tokens).
    auto k = add_bias_if_present(ops::linear(context_hidden, k_proj_weight()), k_proj_bias());
    auto v = add_bias_if_present(ops::linear(context_hidden, v_proj_weight()), v_proj_bias());

    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    if (k_norm_.weight_param().is_bound()) {
        k_heads = k_norm_.forward(k_heads);
    }

    auto* policy = &ctx().op_policy();
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    return {k_rot, v_heads};  // [1, num_kv_heads, T, head_dim] each
}

std::pair<Tensor, Tensor> DFlashAttention::post_process_context_kv(const Tensor& k_proj,
                                                                    const Tensor& v_proj,
                                                                    const Tensor& rope_cos,
                                                                    const Tensor& rope_sin) const {
    // Apply reshape, KNorm, RoPE to raw K projection; just reshape V.
    auto k_heads = k_proj.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v_proj.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    if (k_norm_.weight_param().is_bound()) {
        k_heads = k_norm_.forward(k_heads);
    }

    auto* policy = &ctx().op_policy();
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    return {k_rot, v_heads};
}

Tensor DFlashAttention::forward_with_cached_kv(const Tensor& hidden_states,
                                               const Tensor& context_k,
                                               const Tensor& context_v,
                                               const Tensor& rope_cos,
                                               const Tensor& rope_sin) const {
    // Q from draft tokens only.
    auto q = add_bias_if_present(ops::linear(hidden_states, q_proj_weight()), q_proj_bias());
    // K,V from draft tokens only (context K,V already pre-computed).
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
    // RoPE for draft positions only (cos/sin already scoped to draft positions).
    auto q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    // Concat cached context K,V with fresh draft K,V.
    // context_k: [1, num_kv_heads, T, head_dim], k_rot: [1, num_kv_heads, B, head_dim]
    auto k_full = ops::concat({context_k, k_rot}, 2);
    auto v_full = ops::concat({context_v, v_heads}, 2);

    auto k_expanded = ops::llm::repeat_kv(k_full, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(v_full, num_heads_, num_kv_heads_, head_dim_);

    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, nullptr, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    auto out = add_bias_if_present(ops::linear(merged, o_proj_weight()), o_proj_bias());
    return out;
}

DFlashMLP::DFlashMLP(BuilderContext& ctx, const std::string& name, const DFlashDraftConfig& cfg, Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported DFlash MLP activation: ", cfg.hidden_act);
    }
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& DFlashMLP::gate_proj_weight() const {
    if (!gate_proj_param_) {
        OPENVINO_THROW("DFlashMLP gate projection parameter not registered");
    }
    return gate_proj_param_->value();
}

const Tensor& DFlashMLP::up_proj_weight() const {
    if (!up_proj_param_) {
        OPENVINO_THROW("DFlashMLP up projection parameter not registered");
    }
    return up_proj_param_->value();
}

const Tensor& DFlashMLP::down_proj_weight() const {
    if (!down_proj_param_) {
        OPENVINO_THROW("DFlashMLP down projection parameter not registered");
    }
    return down_proj_param_->value();
}

Tensor DFlashMLP::forward(const Tensor& x) const {
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

DFlashDecoderLayer::DFlashDecoderLayer(BuilderContext& ctx,
                                       const std::string& name,
                                       const DFlashDraftConfig& cfg,
                                       Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, this),
      mlp_(ctx, "mlp", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {}

Tensor DFlashDecoderLayer::forward(const Tensor& target_hidden,
                                   const Tensor& hidden_states,
                                   const Tensor& rope_cos,
                                   const Tensor& rope_sin) const {
    auto residual = hidden_states;
    auto normed = input_layernorm_.forward(hidden_states);
    auto attn_out = self_attn_.forward(target_hidden, normed, rope_cos, rope_sin);
    auto attn_residual = residual + attn_out;
    auto post_norm = post_attention_layernorm_.forward(attn_residual);
    auto mlp_out = mlp_.forward(post_norm);
    return attn_residual + mlp_out;
}

std::pair<Tensor, Tensor> DFlashDecoderLayer::build_context_kv(const Tensor& context_hidden,
                                                               const Tensor& rope_cos,
                                                               const Tensor& rope_sin) const {
    // context_hidden does NOT go through input_layernorm (it's a separate path).
    return self_attn_.build_context_kv(context_hidden, rope_cos, rope_sin);
}

Tensor DFlashDecoderLayer::forward_with_cached_kv(const Tensor& hidden_states,
                                                  const Tensor& context_k,
                                                  const Tensor& context_v,
                                                  const Tensor& rope_cos,
                                                  const Tensor& rope_sin) const {
    auto residual = hidden_states;
    auto normed = input_layernorm_.forward(hidden_states);
    auto attn_out = self_attn_.forward_with_cached_kv(normed, context_k, context_v, rope_cos, rope_sin);
    auto attn_residual = residual + attn_out;
    auto post_norm = post_attention_layernorm_.forward(attn_residual);
    auto mlp_out = mlp_.forward(post_norm);
    return attn_residual + mlp_out;
}

DFlashDraftModel::DFlashDraftModel(BuilderContext& ctx, const DFlashDraftConfig& cfg, Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      hidden_norm_(ctx, "hidden_norm", cfg.rms_norm_eps, this),
      head_dim_(cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / cfg.num_attention_heads)),
      rope_theta_(cfg.rope_theta) {
    fc_weight_param_ = &register_parameter("fc.weight");
    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, "layers." + std::to_string(i), cfg, this);
    }
}

const Tensor& DFlashDraftModel::fc_weight() const {
    if (!fc_weight_param_) {
        OPENVINO_THROW("DFlashDraftModel fc parameter not registered");
    }
    return fc_weight_param_->value();
}

Tensor DFlashDraftModel::forward(const Tensor& target_hidden,
                                 const Tensor& noise_embedding,
                                 const Tensor& position_ids) const {
    auto hidden_states = noise_embedding;
    auto* policy = &ctx().op_policy();
    auto conditioned = ops::linear(target_hidden, fc_weight());
    auto context_hidden = hidden_norm_.forward(conditioned);
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    for (const auto& layer : layers_) {
        hidden_states = layer.forward(context_hidden, hidden_states, cos_sin.first, cos_sin.second);
    }
    return norm_.forward(hidden_states);
}

std::vector<std::pair<Tensor, Tensor>> DFlashDraftModel::build_context_kv(
    const Tensor& target_hidden,
    const Tensor& position_ids) const {
    auto* policy = &ctx().op_policy();
    auto conditioned = ops::linear(target_hidden, fc_weight());
    auto context_hidden = hidden_norm_.forward(conditioned);
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);

    // ── Batched KV projection (SGLang-inspired) ──
    // Stack all K,V weights: [k_w0, v_w0, k_w1, v_w1, ...] along dim 0.
    // Single large MatMul replaces 2*num_layers separate MatMuls.
    const int32_t kv_dim = layers_[0].attn().kv_dim();
    const int32_t num_layers = static_cast<int32_t>(layers_.size());

    std::vector<Tensor> all_kv_weights;
    all_kv_weights.reserve(static_cast<size_t>(num_layers) * 2);
    for (const auto& layer : layers_) {
        all_kv_weights.push_back(layer.attn().get_k_proj_weight());
        all_kv_weights.push_back(layer.attn().get_v_proj_weight());
    }
    // batched_weight: [num_layers * 2 * kv_dim, hidden_size]
    auto batched_weight = ops::concat(all_kv_weights, 0);

    // Single MatMul: context_hidden[1,A,H] × batched_weight^T → [1,A, num_layers*2*kv_dim]
    auto all_kv = ops::linear(context_hidden, batched_weight);

    // Slice per-layer K,V and apply post-processing (reshape + KNorm + RoPE).
    std::vector<std::pair<Tensor, Tensor>> kv_pairs;
    kv_pairs.reserve(layers_.size());
    for (int32_t i = 0; i < num_layers; ++i) {
        const int64_t k_start = static_cast<int64_t>(i) * 2 * kv_dim;
        const int64_t v_start = k_start + kv_dim;
        const int64_t v_end = v_start + kv_dim;

        auto k_proj = ops::slice(all_kv, k_start, v_start, 1, 2);   // [1,A,kv_dim]
        auto v_proj = ops::slice(all_kv, v_start, v_end, 1, 2);     // [1,A,kv_dim]

        // Add bias if present (rare, but handle for completeness).
        if (layers_[i].attn().has_kv_bias()) {
            const auto* k_bias = layers_[i].attn().get_k_proj_bias();
            const auto* v_bias = layers_[i].attn().get_v_proj_bias();
            if (k_bias) k_proj = k_proj + *k_bias;
            if (v_bias) v_proj = v_proj + *v_bias;
        }

        kv_pairs.push_back(layers_[i].attn().post_process_context_kv(
            k_proj, v_proj, cos_sin.first, cos_sin.second));
    }
    return kv_pairs;
}

Tensor DFlashDraftModel::forward_with_cached_kv(
    const Tensor& noise_embedding,
    const Tensor& position_ids,
    const std::vector<std::pair<Tensor, Tensor>>& context_kv) const {
    auto hidden_states = noise_embedding;
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);

    for (size_t i = 0; i < layers_.size(); ++i) {
        hidden_states = layers_[i].forward_with_cached_kv(
            hidden_states, context_kv[i].first, context_kv[i].second,
            cos_sin.first, cos_sin.second);
    }
    return norm_.forward(hidden_states);
}

std::shared_ptr<ov::Model> create_dflash_draft_model(
    const DFlashDraftConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    const ov::element::Type& /*input_type*/) {
    BuilderContext ctx;
    DFlashDraftModel model(ctx, cfg);

    ov::genai::modeling::weights::load_model(model, source, finalizer);

    // Force FP32 compute for draft path to avoid BF16 instability.
    const ov::element::Type dtype = ov::element::f32;

    const int64_t ctx_dim = static_cast<int64_t>(cfg.hidden_size) *
                            static_cast<int64_t>(cfg.num_hidden_layers);
    auto target_hidden = ctx.parameter("target_hidden", dtype, ov::PartialShape{-1, -1, ctx_dim});
    auto noise_embedding = ctx.parameter("noise_embedding", dtype, ov::PartialShape{-1, -1, cfg.hidden_size});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    auto hidden = model.forward(target_hidden, noise_embedding, position_ids);

    auto result = std::make_shared<ov::op::v0::Result>(hidden.output());
    set_name(result, "draft_hidden");
    auto ov_model = ctx.build_model({result->output(0)});
    ov_model->set_rt_info(true, {"modeling", "dflash_draft"});
    return ov_model;
}

std::vector<int32_t> build_target_layer_ids(int32_t num_target_layers, int32_t num_draft_layers) {
    std::vector<int32_t> ids;
    if (num_target_layers <= 0 || num_draft_layers <= 0) {
        return ids;
    }
    if (num_draft_layers == 1) {
        ids.push_back(num_target_layers / 2);
        return ids;
    }
    const int32_t start = 1;
    const int32_t end = num_target_layers - 3;
    const int32_t span = end - start;
    ids.reserve(static_cast<size_t>(num_draft_layers));
    for (int32_t i = 0; i < num_draft_layers; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(num_draft_layers - 1);
        const int32_t val = static_cast<int32_t>(std::lround(start + t * span));
        ids.push_back(val);
    }
    return ids;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

// ============================================================================
// Model Builder Registration
// ============================================================================

#include "loaders/model_builder.hpp"

namespace {

std::shared_ptr<ov::Model> build_dflash_model(
    const ov::genai::loaders::ModelConfig& config,
    ov::genai::modeling::weights::WeightSource& weight_source,
    ov::genai::modeling::weights::WeightFinalizer& weight_finalizer) {
    using namespace ov::genai::modeling;
    using namespace ov::genai::modeling::models;

    DFlashDraftConfig cfg;
    cfg.architecture = "dflash";
    cfg.hidden_size = config.hidden_size;
    cfg.intermediate_size = config.intermediate_size;
    cfg.num_hidden_layers = config.num_hidden_layers;
    cfg.num_target_layers = config.num_target_layers;
    cfg.num_attention_heads = config.num_attention_heads;
    cfg.num_key_value_heads = config.num_key_value_heads > 0 ? config.num_key_value_heads : config.num_attention_heads;
    cfg.head_dim = config.head_dim > 0 ? config.head_dim : (config.hidden_size / config.num_attention_heads);
    cfg.block_size = config.block_size;
    cfg.rms_norm_eps = config.rms_norm_eps;
    cfg.rope_theta = config.rope_theta;
    cfg.hidden_act = config.hidden_act;
    cfg.attention_bias = config.attention_bias;

    return create_dflash_draft_model(cfg, weight_source, weight_finalizer, ov::element::f32);
}

static bool dflash_registered = []() {
    ov::genai::loaders::ModelBuilder::instance().register_architecture("dflash", build_dflash_model);
    return true;
}();

}  // namespace
