// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3/modeling_qwen3.hpp"

#include <cmath>
#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/kv_cache.hpp"
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

Qwen3Attention::Qwen3Attention(BuilderContext& ctx,
                               const std::string& name,
                               const Qwen3DenseConfig& cfg,
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

    if (cfg.attention_bias) {
        q_bias_param_ = &register_parameter("q_proj.bias");
        k_bias_param_ = &register_parameter("k_proj.bias");
        v_bias_param_ = &register_parameter("v_proj.bias");
        o_bias_param_ = &register_parameter("o_proj.bias");
    }
}

const Tensor& Qwen3Attention::q_proj_weight() const {
    if (!q_proj_param_) {
        OPENVINO_THROW("Qwen3Attention q_proj parameter not registered");
    }
    return q_proj_param_->value();
}

const Tensor& Qwen3Attention::k_proj_weight() const {
    if (!k_proj_param_) {
        OPENVINO_THROW("Qwen3Attention k_proj parameter not registered");
    }
    return k_proj_param_->value();
}

const Tensor& Qwen3Attention::v_proj_weight() const {
    if (!v_proj_param_) {
        OPENVINO_THROW("Qwen3Attention v_proj parameter not registered");
    }
    return v_proj_param_->value();
}

const Tensor& Qwen3Attention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("Qwen3Attention o_proj parameter not registered");
    }
    return o_proj_param_->value();
}

const Tensor* Qwen3Attention::q_proj_bias() const {
    return (q_bias_param_ && q_bias_param_->is_bound()) ? &q_bias_param_->value() : nullptr;
}

const Tensor* Qwen3Attention::k_proj_bias() const {
    return (k_bias_param_ && k_bias_param_->is_bound()) ? &k_bias_param_->value() : nullptr;
}

const Tensor* Qwen3Attention::v_proj_bias() const {
    return (v_bias_param_ && v_bias_param_->is_bound()) ? &v_bias_param_->value() : nullptr;
}

const Tensor* Qwen3Attention::o_proj_bias() const {
    return (o_bias_param_ && o_bias_param_->is_bound()) ? &o_bias_param_->value() : nullptr;
}

Tensor Qwen3Attention::forward(const Tensor& positions, const Tensor& hidden_states, const Tensor& beam_idx) const {
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(positions, head_dim_, rope_theta_, policy);
    auto* op_ctx = positions.context();
    auto batch = shape::dim(positions, 0);
    auto seq_len = shape::dim(positions, 1);
    auto mask_shape = shape::make({batch, seq_len});
    auto one = Tensor(ops::const_scalar(op_ctx, int64_t{1}), op_ctx);
    auto attention_mask = shape::broadcast_to(one, mask_shape);
    return forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, attention_mask);
}

Tensor Qwen3Attention::forward(const Tensor& hidden_states,
                               const Tensor& beam_idx,
                               const Tensor& rope_cos,
                               const Tensor& rope_sin,
                               const Tensor& attention_mask) const {
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

    // Build causal mask with attention_mask integration for NPU/NPUW compatibility
    auto mask = ops::llm::build_kv_causal_mask_with_attention(q_rot, k_expanded, attention_mask);
    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &mask, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    auto out = add_bias_if_present(ops::linear(merged, o_proj_weight()), o_proj_bias());
    return out;
}

Tensor Qwen3Attention::forward_no_cache(const Tensor& hidden_states,
                                        const Tensor& rope_cos,
                                        const Tensor& rope_sin,
                                        const Tensor& attention_mask) const {
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

    auto k_expanded = ops::llm::repeat_kv(k_rot, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(v_heads, num_heads_, num_kv_heads_, head_dim_);

    auto mask = ops::llm::build_kv_causal_mask_with_attention(q_rot, k_expanded, attention_mask);
    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &mask, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    auto out = add_bias_if_present(ops::linear(merged, o_proj_weight()), o_proj_bias());
    return out;
}

Qwen3MLP::Qwen3MLP(BuilderContext& ctx, const std::string& name, const Qwen3DenseConfig& cfg, Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3 MLP activation: ", cfg.hidden_act);
    }
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& Qwen3MLP::gate_proj_weight() const {
    if (!gate_proj_param_) {
        OPENVINO_THROW("Qwen3MLP gate projection parameter not registered");
    }
    return gate_proj_param_->value();
}

const Tensor& Qwen3MLP::up_proj_weight() const {
    if (!up_proj_param_) {
        OPENVINO_THROW("Qwen3MLP up projection parameter not registered");
    }
    return up_proj_param_->value();
}

const Tensor& Qwen3MLP::down_proj_weight() const {
    if (!down_proj_param_) {
        OPENVINO_THROW("Qwen3MLP down projection parameter not registered");
    }
    return down_proj_param_->value();
}

Tensor Qwen3MLP::forward(const Tensor& x) const {
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

Qwen3DecoderLayer::Qwen3DecoderLayer(BuilderContext& ctx,
                                     const std::string& name,
                                     const Qwen3DenseConfig& cfg,
                                     Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, this),
      mlp_(ctx, "mlp", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {}

std::pair<Tensor, Tensor> Qwen3DecoderLayer::forward(const Tensor& hidden_states,
                                                     const Tensor& beam_idx,
                                                     const Tensor& rope_cos,
                                                     const Tensor& rope_sin,
                                                     const Tensor& attention_mask,
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
    auto attn_out = self_attn_.forward(normed, beam_idx, rope_cos, rope_sin, attention_mask);
    auto post_norm = post_attention_layernorm_.forward(attn_out, next_residual);
    auto mlp_out = mlp_.forward(post_norm.first);
    return {mlp_out, post_norm.second};
}

std::pair<Tensor, Tensor> Qwen3DecoderLayer::forward_no_cache(const Tensor& hidden_states,
                                                              const Tensor& rope_cos,
                                                              const Tensor& rope_sin,
                                                              const Tensor& attention_mask,
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
    auto attn_out = self_attn_.forward_no_cache(normed, rope_cos, rope_sin, attention_mask);
    auto post_norm = post_attention_layernorm_.forward(attn_out, next_residual);
    auto mlp_out = mlp_.forward(post_norm.first);
    return {mlp_out, post_norm.second};
}

Qwen3Model::Qwen3Model(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent)
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

Tensor Qwen3Model::forward(const Tensor& input_ids, const Tensor& position_ids, const Tensor& beam_idx, const Tensor& attention_mask) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto layer_out = layer.forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, attention_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
    }
    if (residual) {
        return norm_.forward(hidden_states, *residual).first;
    }
    return norm_.forward(hidden_states);
}

Tensor Qwen3Model::forward_embeds(const Tensor& inputs_embeds,
                                  const Tensor& position_ids,
                                  const Tensor& beam_idx,
                                  const Tensor& attention_mask) {
    auto hidden_states = inputs_embeds;
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto layer_out = layer.forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, attention_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
    }
    if (residual) {
        return norm_.forward(hidden_states, *residual).first;
    }
    return norm_.forward(hidden_states);
}

std::pair<Tensor, Tensor> Qwen3Model::forward_with_penultimate(const Tensor& input_ids,
                                                               const Tensor& position_ids,
                                                               const Tensor& beam_idx) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    Tensor penultimate = hidden_states;
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    auto* op_ctx = input_ids.context();
    auto batch = shape::dim(input_ids, 0);
    auto seq_len = shape::dim(input_ids, 1);
    auto mask_shape = shape::make({batch, seq_len});
    auto one = Tensor(ops::const_scalar(op_ctx, int64_t{1}), op_ctx);
    auto attention_mask = shape::broadcast_to(one, mask_shape);
    std::optional<Tensor> residual;
    const int32_t capture_idx = static_cast<int32_t>(layers_.size()) - 2;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto layer_out = layers_[i].forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, attention_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
        if (static_cast<int32_t>(i) == capture_idx) {
            penultimate = hidden_states;
        }
    }
    Tensor final_out = residual ? norm_.forward(hidden_states, *residual).first : norm_.forward(hidden_states);
    return {final_out, penultimate};
}

Tensor Qwen3Model::forward_no_cache(const Tensor& input_ids,
                                    const Tensor& position_ids,
                                    const Tensor& attention_mask) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto layer_out = layer.forward_no_cache(hidden_states, cos_sin.first, cos_sin.second, attention_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
    }
    if (residual) {
        return norm_.forward(hidden_states, *residual).first;
    }
    return norm_.forward(hidden_states);
}

std::pair<Tensor, Tensor> Qwen3Model::forward_with_penultimate_no_cache(const Tensor& input_ids,
                                                                        const Tensor& position_ids,
                                                                        const Tensor& attention_mask) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    Tensor penultimate = hidden_states;
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    std::optional<Tensor> residual;
    const int32_t capture_idx = static_cast<int32_t>(layers_.size()) - 2;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto layer_out = layers_[i].forward_no_cache(hidden_states, cos_sin.first, cos_sin.second, attention_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
        if (static_cast<int32_t>(i) == capture_idx) {
            penultimate = hidden_states;
        }
    }
    Tensor final_out = residual ? norm_.forward(hidden_states, *residual).first : norm_.forward(hidden_states);
    return {final_out, penultimate};
}

std::pair<Tensor, Tensor> Qwen3Model::forward_with_pre_norm_no_cache(const Tensor& input_ids,
                                                                     const Tensor& position_ids,
                                                                     const Tensor& attention_mask) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto layer_out = layer.forward_no_cache(hidden_states, cos_sin.first, cos_sin.second, attention_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
    }
    Tensor pre_norm = residual ? (hidden_states + *residual) : hidden_states;
    Tensor final_out = residual ? norm_.forward(hidden_states, *residual).first : norm_.forward(hidden_states);
    return {final_out, pre_norm};
}

std::pair<Tensor, Tensor> Qwen3Model::forward_with_selected_layers(const Tensor& input_ids,
                                                                   const Tensor& position_ids,
                                                                   const Tensor& beam_idx,
                                                                   const Tensor& attention_mask,
                                                                   const std::vector<int32_t>& layer_ids) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    std::optional<Tensor> residual;

    std::vector<int32_t> sorted_ids = layer_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    size_t capture_idx = 0;
    std::vector<Tensor> captures;
    captures.reserve(sorted_ids.size());

    for (size_t i = 0; i < layers_.size(); ++i) {
        auto layer_out = layers_[i].forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, attention_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
        if (capture_idx < sorted_ids.size() && static_cast<int32_t>(i) == sorted_ids[capture_idx]) {
            Tensor pre_norm = residual ? (hidden_states + *residual) : hidden_states;
            captures.push_back(pre_norm);
            ++capture_idx;
        }
    }

    Tensor final_out = residual ? norm_.forward(hidden_states, *residual).first : norm_.forward(hidden_states);
    if (captures.empty()) {
        return {final_out, final_out};
    }
    auto concat_hidden = ops::concat(captures, 2);
    return {final_out, concat_hidden};
}

std::pair<Tensor, Tensor> Qwen3Model::forward_with_selected_layers_no_cache(const Tensor& input_ids,
                                                                            const Tensor& position_ids,
                                                                            const Tensor& attention_mask,
                                                                            const std::vector<int32_t>& layer_ids) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    std::optional<Tensor> residual;

    std::vector<int32_t> sorted_ids = layer_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    size_t capture_idx = 0;
    std::vector<Tensor> captures;
    captures.reserve(sorted_ids.size());

    for (size_t i = 0; i < layers_.size(); ++i) {
        auto layer_out = layers_[i].forward_no_cache(hidden_states, cos_sin.first, cos_sin.second, attention_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
        if (capture_idx < sorted_ids.size() && static_cast<int32_t>(i) == sorted_ids[capture_idx]) {
            Tensor pre_norm = residual ? (hidden_states + *residual) : hidden_states;
            captures.push_back(pre_norm);
            ++capture_idx;
        }
    }

    Tensor final_out = residual ? norm_.forward(hidden_states, *residual).first : norm_.forward(hidden_states);
    if (captures.empty()) {
        return {final_out, final_out};
    }
    auto concat_hidden = ops::concat(captures, 2);
    return {final_out, concat_hidden};
}

VocabEmbedding& Qwen3Model::embed_tokens() {
    return embed_tokens_;
}

RMSNorm& Qwen3Model::norm() {
    return norm_;
}

Qwen3ForCausalLM::Qwen3ForCausalLM(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_head_(ctx, "lm_head", this) {
    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}

Tensor Qwen3ForCausalLM::forward(const Tensor& input_ids,
                                 const Tensor& position_ids,
                                 const Tensor& beam_idx,
                                 const Tensor& attention_mask) {
    auto hidden = model_.forward(input_ids, position_ids, beam_idx, attention_mask);
    return lm_head_.forward(hidden);
}

Tensor Qwen3ForCausalLM::forward_embeds(const Tensor& inputs_embeds,
                                        const Tensor& position_ids,
                                        const Tensor& beam_idx,
                                        const Tensor& attention_mask) {
    auto hidden = model_.forward_embeds(inputs_embeds, position_ids, beam_idx, attention_mask);
    return lm_head_.forward(hidden);
}

Qwen3Model& Qwen3ForCausalLM::model() {
    return model_;
}

LMHead& Qwen3ForCausalLM::lm_head() {
    return lm_head_;
}

std::shared_ptr<ov::Model> create_qwen3_dense_model(
    const Qwen3DenseConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3ForCausalLM model(ctx, cfg);

    ov::genai::modeling::weights::load_model(model, source, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    // Use attention_mask in the forward pass for NPU/NPUW compatibility
    auto logits = model.forward(input_ids, position_ids, beam_idx, attention_mask);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");
    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_text_encoder_model(
    const Qwen3DenseConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3Model model(ctx, cfg);

    ov::genai::modeling::weights::load_model(model, source, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    auto outputs = model.forward_with_pre_norm_no_cache(input_ids, position_ids, attention_mask);

    auto result = std::make_shared<ov::op::v0::Result>(outputs.second.output());
    set_name(result, "hidden_states");
    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_dflash_target_model(
    const Qwen3DenseConfig& cfg,
    const std::vector<int32_t>& target_layer_ids,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3ForCausalLM model(ctx, cfg);

    ov::genai::modeling::weights::load_model(model, source, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    // Use attention_mask
    auto outputs = model.model().forward_with_selected_layers(input_ids, position_ids, beam_idx, attention_mask, target_layer_ids);
    auto logits = model.lm_head().forward(outputs.first);
    auto hidden_out = outputs.second;
    if (hidden_out.dtype() != logits.dtype()) {
        hidden_out = hidden_out.to(logits.dtype());
    }

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(logits_result, "logits");
    auto hidden_result = std::make_shared<ov::op::v0::Result>(hidden_out.output());
    set_name(hidden_result, "target_hidden");

    return ctx.build_model({logits_result->output(0), hidden_result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_dflash_target_model_no_cache(
    const Qwen3DenseConfig& cfg,
    const std::vector<int32_t>& target_layer_ids,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3ForCausalLM model(ctx, cfg);

    ov::genai::modeling::weights::load_model(model, source, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});

    auto outputs = model.model().forward_with_selected_layers_no_cache(input_ids, position_ids, attention_mask, target_layer_ids);
    auto logits = model.lm_head().forward(outputs.first);
    auto hidden_out = outputs.second;
    if (hidden_out.dtype() != logits.dtype()) {
        hidden_out = hidden_out.to(logits.dtype());
    }

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(logits_result, "logits");
    auto hidden_result = std::make_shared<ov::op::v0::Result>(hidden_out.output());
    set_name(hidden_result, "target_hidden");

    return ctx.build_model({logits_result->output(0), hidden_result->output(0)});
}

std::shared_ptr<ov::Model> create_qwen3_embedding_model(
    const Qwen3DenseConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    (void)cfg;
    BuilderContext ctx;

    Module root("model", ctx);
    VocabEmbedding embed(ctx, "embed_tokens", &root);

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

std::shared_ptr<ov::Model> create_qwen3_lm_head_model(
    const Qwen3DenseConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    const ov::element::Type& input_type) {
    BuilderContext ctx;

    Module root("", ctx);
    LMHead head(ctx, "lm_head", &root);

    // Some HF shards omit lm_head.weight when embeddings are tied.
    if (!source.has("lm_head.weight") && cfg.tie_word_embeddings) {
        const std::string embed_weight = "model.embed_tokens.weight";
        if (!source.has(embed_weight)) {
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

    auto hidden = ctx.parameter("hidden_states", input_type, ov::PartialShape{-1, -1, cfg.hidden_size});
    auto logits = head.forward(hidden);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

// ============================================================================
// Model Builder Registration
// ============================================================================
// The build function and registration are placed here (in the model file)
// rather than in model_builder.cpp, following the pattern from vLLM where
// each model file is self-contained with its own registration.
// This allows adding new models without modifying model_builder.cpp.
// ============================================================================

#include "loaders/model_builder.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

/**
 * @brief Build Qwen3/Qwen2 model using modeling API
 *
 * This function converts ModelConfig to Qwen3DenseConfig, creates the model
 * structure, loads weights, and builds the final OpenVINO model.
 */
std::shared_ptr<ov::Model> build_qwen3_model(
    const ov::genai::loaders::ModelConfig& config,
    ov::genai::modeling::weights::WeightSource& weight_source,
    ov::genai::modeling::weights::WeightFinalizer& weight_finalizer) {

    using namespace ov::genai::modeling;
    using namespace ov::genai::modeling::models;

    BuilderContext ctx;

    // Convert ModelConfig to Qwen3DenseConfig
    Qwen3DenseConfig cfg;
    cfg.architecture = config.architecture;
    cfg.hidden_size = config.hidden_size;
    cfg.num_hidden_layers = config.num_hidden_layers;
    cfg.num_attention_heads = config.num_attention_heads;
    cfg.num_key_value_heads = config.num_key_value_heads > 0
        ? config.num_key_value_heads : config.num_attention_heads;
    cfg.head_dim = config.head_dim > 0
        ? config.head_dim : (config.hidden_size / config.num_attention_heads);
    cfg.rope_theta = config.rope_theta;
    cfg.attention_bias = config.attention_bias;
    cfg.rms_norm_eps = config.rms_norm_eps;
    cfg.tie_word_embeddings = config.tie_word_embeddings;
    cfg.hidden_act = config.hidden_act;

    // Create model
    Qwen3ForCausalLM model(ctx, cfg);

    // Load weights
    weights::load_model(model, weight_source, weight_finalizer);

    // Helper to set tensor name
    auto set_name = [](auto node, const std::string& name) {
        node->output(0).set_names({name});
    };

    // Create input parameters with dynamic shapes
    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    // Set tensor names for inputs
    input_ids.output().get_node()->output(0).set_names({"input_ids"});
    attention_mask.output().get_node()->output(0).set_names({"attention_mask"});
    position_ids.output().get_node()->output(0).set_names({"position_ids"});
    beam_idx.output().get_node()->output(0).set_names({"beam_idx"});

    // Use attention_mask in the forward pass for NPU/NPUW compatibility
    auto logits = model.forward(input_ids, position_ids, beam_idx, attention_mask);

    // Build model with named output
    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");
    auto ov_model = ctx.build_model({result->output(0)});

    // Set runtime options for optimal performance
    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});

    return ov_model;
}

// Self-registration: Register Qwen3 builder at static initialization
// Note: Qwen2 support is pending validation and will be added separately
static bool qwen3_registered = []() {
    ov::genai::loaders::ModelBuilder::instance().register_architecture("qwen3", build_qwen3_model);
    return true;
}();

}  // namespace