// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
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
#include "modeling/layers/lm_head.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3NextConfig {
    std::string architecture = "qwen3_next";
    int32_t hidden_size = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t vocab_size = 0;
    int32_t max_position_embeddings = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    float partial_rotary_factor = 0.25f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
    bool tie_word_embeddings = false;
    std::vector<std::string> layer_types;
    int32_t full_attention_interval = 4;

    int32_t linear_conv_kernel_dim = 4;
    int32_t linear_key_head_dim = 128;
    int32_t linear_value_head_dim = 128;
    int32_t linear_num_key_heads = 16;
    int32_t linear_num_value_heads = 32;

    int32_t decoder_sparse_step = 1;
    int32_t moe_intermediate_size = 512;
    int32_t shared_expert_intermediate_size = 512;
    int32_t num_experts = 512;
    int32_t num_experts_per_tok = 10;
    bool norm_topk_prob = true;
    bool output_router_logits = false;
    float router_aux_loss_coef = 0.0f;
    std::vector<int32_t> mlp_only_layers;
    int32_t group_size = 128;  // MoE quantization group size
};

class Qwen3NextRMSNorm : public Module {
public:
    Qwen3NextRMSNorm(BuilderContext& ctx, const std::string& name, float eps, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;
    std::pair<Tensor, Tensor> forward(const Tensor& x, const Tensor& residual) const;

private:
    const Tensor& weight() const;

    WeightParameter* weight_param_ = nullptr;
    float eps_ = 1e-6f;
};

class Qwen3NextAttention : public Module {
public:
    Qwen3NextAttention(BuilderContext& ctx, const std::string& name, const Qwen3NextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin,
                   const Tensor* attention_mask,
                   const Tensor* precomputed_sdpa_mask = nullptr) const;

private:
    const Tensor& q_proj_weight() const;
    const Tensor& k_proj_weight() const;
    const Tensor& v_proj_weight() const;
    const Tensor& o_proj_weight() const;

    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t hidden_size_ = 0;
    int32_t rotary_dim_ = 0;
    float scaling_ = 1.0f;
    float rope_theta_ = 10000.0f;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    Qwen3NextRMSNorm q_norm_;
    Qwen3NextRMSNorm k_norm_;
};

class Qwen3NextGatedDeltaNet : public Module {
public:
    Qwen3NextGatedDeltaNet(BuilderContext& ctx, const std::string& name, const Qwen3NextConfig& cfg, int32_t layer_idx, Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor* attention_mask,
                   const Tensor* cache_position) const;

private:
    const Tensor& in_proj_qkvz_weight() const;
    const Tensor& in_proj_ba_weight() const;
    const Tensor& conv1d_weight() const;
    const Tensor& a_log() const;
    const Tensor& dt_bias() const;
    const Tensor& out_proj_weight() const;

    Tensor apply_depthwise_causal_conv(const Tensor& mixed_qkv,
                                       const Tensor& prev_conv_state,
                                       Tensor* next_conv_state) const;

    Tensor rms_norm_gated(const Tensor& x, const Tensor& z) const;

    int32_t layer_idx_ = 0;
    int32_t hidden_size_ = 0;
    int32_t num_v_heads_ = 0;
    int32_t num_k_heads_ = 0;
    int32_t head_k_dim_ = 0;
    int32_t head_v_dim_ = 0;
    int32_t key_dim_ = 0;
    int32_t value_dim_ = 0;
    int32_t conv_dim_ = 0;
    int32_t conv_kernel_size_ = 4;
    int32_t conv_state_size_ = 0;
    float eps_ = 1e-6f;

    WeightParameter* in_proj_qkvz_param_ = nullptr;
    WeightParameter* in_proj_ba_param_ = nullptr;
    WeightParameter* conv1d_param_ = nullptr;
    WeightParameter* a_log_param_ = nullptr;
    WeightParameter* dt_bias_param_ = nullptr;
    WeightParameter* norm_param_ = nullptr;
    WeightParameter* out_proj_param_ = nullptr;
};

/// Qwen3NextGatedDeltaNet2 — equivalent to Qwen3NextGatedDeltaNet but replaces
/// the TensorIterator-based recurrent loop with the fused `ops::linear_attention`
/// operation.  The pre-processing (projection, conv1d, normalization, gating) and
/// post-processing (rms_norm_gated, out projection) remain identical.
class Qwen3NextGatedDeltaNet2 : public Module {
public:
    Qwen3NextGatedDeltaNet2(BuilderContext& ctx, const std::string& name, const Qwen3NextConfig& cfg, int32_t layer_idx, Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor* attention_mask,
                   const Tensor* cache_position) const;

private:
    const Tensor& in_proj_qkvz_weight() const;
    const Tensor& in_proj_ba_weight() const;
    const Tensor& conv1d_weight() const;
    const Tensor& a_log() const;
    const Tensor& dt_bias() const;
    const Tensor& out_proj_weight() const;

    Tensor apply_depthwise_causal_conv(const Tensor& mixed_qkv,
                                       const Tensor& prev_conv_state,
                                       Tensor* next_conv_state) const;

    Tensor rms_norm_gated(const Tensor& x, const Tensor& z) const;

    int32_t layer_idx_ = 0;
    int32_t hidden_size_ = 0;
    int32_t num_v_heads_ = 0;
    int32_t num_k_heads_ = 0;
    int32_t head_k_dim_ = 0;
    int32_t head_v_dim_ = 0;
    int32_t key_dim_ = 0;
    int32_t value_dim_ = 0;
    int32_t conv_dim_ = 0;
    int32_t conv_kernel_size_ = 4;
    int32_t conv_state_size_ = 0;
    float eps_ = 1e-6f;

    WeightParameter* in_proj_qkvz_param_ = nullptr;
    WeightParameter* in_proj_ba_param_ = nullptr;
    WeightParameter* conv1d_param_ = nullptr;
    WeightParameter* a_log_param_ = nullptr;
    WeightParameter* dt_bias_param_ = nullptr;
    WeightParameter* norm_param_ = nullptr;
    WeightParameter* out_proj_param_ = nullptr;
};

class Qwen3NextMLP : public Module {
public:
    Qwen3NextMLP(BuilderContext& ctx,
                 const std::string& name,
                 const Qwen3NextConfig& cfg,
                 int32_t intermediate_size,
                 Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

class Qwen3NextSparseMoeBlock : public Module {
public:
    Qwen3NextSparseMoeBlock(BuilderContext& ctx, const std::string& name, const Qwen3NextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& gate_weight() const;
    const Tensor& shared_expert_gate_weight() const;
    const Tensor& shared_gate_proj_weight() const;
    const Tensor& shared_up_proj_weight() const;
    const Tensor& shared_down_proj_weight() const;
    
    // MoE expert weights (stacked)
    Tensor gate_expert_weights() const;
    Tensor up_expert_weights() const;
    Tensor down_expert_weights() const;
    
    // MoE quantization scales and zero-points (stacked)
    Tensor gate_exps_scales() const;
    Tensor gate_exps_zps() const;
    Tensor up_exps_scales() const;
    Tensor up_exps_zps() const;
    Tensor down_exps_scales() const;
    Tensor down_exps_zps() const;

    int32_t hidden_size_ = 0;
    int32_t expert_intermediate_size_ = 0;
    int32_t shared_intermediate_size_ = 0;
    int32_t num_experts_ = 0;
    int32_t top_k_ = 1;
    bool norm_topk_prob_ = true;
    size_t group_size_ = 128;

    WeightParameter* gate_param_ = nullptr;
    WeightParameter* shared_expert_gate_param_ = nullptr;
    WeightParameter* shared_gate_proj_param_ = nullptr;
    WeightParameter* shared_up_proj_param_ = nullptr;
    WeightParameter* shared_down_proj_param_ = nullptr;
    std::vector<WeightParameter*> gate_experts_param_;
    std::vector<WeightParameter*> up_experts_param_;
    std::vector<WeightParameter*> down_experts_param_;
    
    // Quantization scales and zero-points for each expert
    std::vector<Tensor> gate_exps_scales_;
    std::vector<Tensor> gate_exps_zps_;
    std::vector<Tensor> up_exps_scales_;
    std::vector<Tensor> up_exps_zps_;
    std::vector<Tensor> down_exps_scales_;
    std::vector<Tensor> down_exps_zps_;
};

class Qwen3NextDecoderLayer : public Module {
public:
    Qwen3NextDecoderLayer(BuilderContext& ctx,
                          const std::string& name,
                          const Qwen3NextConfig& cfg,
                          int32_t layer_idx,
                          Module* parent = nullptr);

    std::pair<Tensor, Tensor> forward(const Tensor& hidden_states,
                                      const Tensor& beam_idx,
                                      const Tensor& rope_cos,
                                      const Tensor& rope_sin,
                                      const Tensor* full_attention_mask,
                                      const Tensor* linear_attention_mask,
                                      const Tensor* cache_position,
                                      const std::optional<Tensor>& residual,
                                      const Tensor* precomputed_full_attn_sdpa_mask = nullptr) const;

private:
    std::string layer_type_;
    std::unique_ptr<Qwen3NextAttention> self_attn_;
    std::unique_ptr<Qwen3NextGatedDeltaNet2> linear_attn_;
    std::unique_ptr<Qwen3NextMLP> dense_mlp_;
    std::unique_ptr<Qwen3NextSparseMoeBlock> moe_mlp_;
    Qwen3NextRMSNorm input_layernorm_;
    Qwen3NextRMSNorm post_attention_layernorm_;
};

class Qwen3NextModel : public Module {
public:
    Qwen3NextModel(BuilderContext& ctx, const Qwen3NextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor& full_attention_mask,
                   const Tensor* linear_attention_mask,
                   const Tensor* cache_position);

    VocabEmbedding& embed_tokens();

private:
    VocabEmbedding embed_tokens_;
    std::vector<Qwen3NextDecoderLayer> layers_;
    Qwen3NextRMSNorm norm_;
    int32_t head_dim_ = 0;
    int32_t rotary_dim_ = 0;
    float rope_theta_ = 10000.0f;
};

class Qwen3NextForCausalLM : public Module {
public:
    Qwen3NextForCausalLM(BuilderContext& ctx, const Qwen3NextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor& full_attention_mask,
                   const Tensor* linear_attention_mask,
                   const Tensor* cache_position);

private:
    Qwen3NextConfig cfg_;
    Qwen3NextModel model_;
    LMHead lm_head_;
};

std::shared_ptr<ov::Model> create_qwen3_next_model(
    const Qwen3NextConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

