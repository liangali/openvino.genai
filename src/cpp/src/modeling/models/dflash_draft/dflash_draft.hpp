// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ov {
class Model;
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {
namespace weights {
class WeightFinalizer;
class WeightSource;
}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov

#include "modeling/builder_context.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct DFlashDraftConfig {
    std::string architecture = "dflash";
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_target_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t block_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
};

class DFlashAttention : public Module {
public:
    DFlashAttention(BuilderContext& ctx, const std::string& name, const DFlashDraftConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& target_hidden,
                   const Tensor& hidden_states,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin) const;

    /// Build context K,V for preprocessing model.
    /// Returns {K, V} at [1, num_kv_heads, seq, head_dim], K is post-KNorm + post-RoPE.
    std::pair<Tensor, Tensor> build_context_kv(const Tensor& context_hidden,
                                               const Tensor& rope_cos,
                                               const Tensor& rope_sin) const;

    /// Build attention using pre-computed context K,V (no fc/context recomputation).
    /// context_k/v: [1, num_kv_heads, T, head_dim], already post-norm/RoPE.
    /// rope_cos/sin: for draft positions only [1, B, ...].
    Tensor forward_with_cached_kv(const Tensor& hidden_states,
                                  const Tensor& context_k,
                                  const Tensor& context_v,
                                  const Tensor& rope_cos,
                                  const Tensor& rope_sin) const;

    /// Post-process raw K,V projections: reshape + KNorm + RoPE (for K), reshape (for V).
    /// k_proj, v_proj: [1, A, kv_dim].  Returns {K, V} at [1, kv_heads, A, head_dim].
    std::pair<Tensor, Tensor> post_process_context_kv(const Tensor& k_proj,
                                                      const Tensor& v_proj,
                                                      const Tensor& rope_cos,
                                                      const Tensor& rope_sin) const;

    /// Public weight accessors for batched KV projection.
    const Tensor& get_k_proj_weight() const { return k_proj_weight(); }
    const Tensor& get_v_proj_weight() const { return v_proj_weight(); }
    bool has_kv_bias() const { return attention_bias_; }
    const Tensor* get_k_proj_bias() const { return k_proj_bias(); }
    const Tensor* get_v_proj_bias() const { return v_proj_bias(); }
    int32_t kv_dim() const { return num_kv_heads_ * head_dim_; }

private:
    const Tensor& q_proj_weight() const;
    const Tensor& k_proj_weight() const;
    const Tensor& v_proj_weight() const;
    const Tensor& o_proj_weight() const;

    const Tensor* q_proj_bias() const;
    const Tensor* k_proj_bias() const;
    const Tensor* v_proj_bias() const;
    const Tensor* o_proj_bias() const;

    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t hidden_size_ = 0;
    float scaling_ = 1.0f;
    float rope_theta_ = 10000.0f;
    bool attention_bias_ = false;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    WeightParameter* q_bias_param_ = nullptr;
    WeightParameter* k_bias_param_ = nullptr;
    WeightParameter* v_bias_param_ = nullptr;
    WeightParameter* o_bias_param_ = nullptr;

    RMSNorm q_norm_;
    RMSNorm k_norm_;
};

class DFlashMLP : public Module {
public:
    DFlashMLP(BuilderContext& ctx, const std::string& name, const DFlashDraftConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

class DFlashDecoderLayer : public Module {
public:
    DFlashDecoderLayer(BuilderContext& ctx, const std::string& name, const DFlashDraftConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& target_hidden,
                   const Tensor& hidden_states,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin) const;

    /// Build context K,V for this layer (delegates to self_attn_).
    std::pair<Tensor, Tensor> build_context_kv(const Tensor& context_hidden,
                                               const Tensor& rope_cos,
                                               const Tensor& rope_sin) const;

    /// Forward with pre-computed context K,V (no context recomputation).
    Tensor forward_with_cached_kv(const Tensor& hidden_states,
                                  const Tensor& context_k,
                                  const Tensor& context_v,
                                  const Tensor& rope_cos,
                                  const Tensor& rope_sin) const;

    /// Access the attention module (for batched KV weight collection).
    const DFlashAttention& attn() const { return self_attn_; }

private:
    DFlashAttention self_attn_;
    DFlashMLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

class DFlashDraftModel : public Module {
public:
    DFlashDraftModel(BuilderContext& ctx, const DFlashDraftConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& target_hidden,
                   const Tensor& noise_embedding,
                   const Tensor& position_ids) const;

    /// Build context KV preprocessing outputs for all layers.
    /// Returns vector of {K, V} pairs, one per layer.
    /// K: [1, num_kv_heads, seq, head_dim] post-KNorm + post-RoPE.
    /// V: [1, num_kv_heads, seq, head_dim].
    std::vector<std::pair<Tensor, Tensor>> build_context_kv(
        const Tensor& target_hidden,
        const Tensor& position_ids) const;

    /// Forward with pre-computed context K,V caches (skips fc + context KV computation).
    /// context_kv: vector of {K_i, V_i} pairs from build_context_kv.
    /// position_ids: draft positions only [1, B].
    Tensor forward_with_cached_kv(
        const Tensor& noise_embedding,
        const Tensor& position_ids,
        const std::vector<std::pair<Tensor, Tensor>>& context_kv) const;

private:
    const Tensor& fc_weight() const;

    DFlashDraftConfig cfg_;
    std::vector<DFlashDecoderLayer> layers_;
    RMSNorm norm_;
    RMSNorm hidden_norm_;
    WeightParameter* fc_weight_param_ = nullptr;
    int32_t head_dim_ = 0;
    float rope_theta_ = 10000.0f;
};

std::shared_ptr<ov::Model> create_dflash_draft_model(
    const DFlashDraftConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    // NOTE: Draft is always constructed in FP32 to avoid BF16 instabilities; the
    // type argument is kept for API compatibility only.
    const ov::element::Type& input_type = ov::element::f32);

std::vector<int32_t> build_target_layer_ids(int32_t num_target_layers, int32_t num_draft_layers);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
