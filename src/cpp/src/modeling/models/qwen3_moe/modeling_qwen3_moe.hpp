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
#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3MoeConfig {
    std::string architecture = "qwen3moe";
    int32_t hidden_size = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = true;
    bool tie_word_embeddings = false;

    int32_t expert_count = 0;
    int32_t expert_used_count = 0;
    int32_t moe_intermediate_size = 0;
    int32_t group_size = 128;
};

class Qwen3MoeAttention : public Module {
public:
    Qwen3MoeAttention(BuilderContext& ctx, const std::string& name, const Qwen3MoeConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& positions, const Tensor& hidden_states, const Tensor& beam_idx) const;
    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin,
                   const Tensor& causal_mask) const;

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

class Qwen3MoE : public Module {
public:
    Qwen3MoE(BuilderContext& ctx, const std::string& name, const Qwen3MoeConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_inp_weight() const;
    Tensor gate_exps_weight() const;
    Tensor up_exps_weight() const;
    Tensor down_exps_weight() const;

    Tensor gate_exps_scales() const;
    Tensor gate_exps_zps() const;
    Tensor up_exps_scales() const;
    Tensor up_exps_zps() const;
    Tensor down_exps_scales() const;
    Tensor down_exps_zps() const;

    bool can_use_fused_path() const;
    size_t infer_group_size() const;
    Tensor routed_fused(const Tensor& flat_f32) const;
    Tensor routed_fallback(const Tensor& flat_f32) const;

    WeightParameter* gate_inp_param_ = nullptr;
    std::vector<WeightParameter*> gate_exps_param_;
    std::vector<WeightParameter*> up_exps_param_;
    std::vector<WeightParameter*> down_exps_param_;

    std::vector<Tensor> gate_exps_scales_;
    std::vector<Tensor> gate_exps_zps_;
    std::vector<Tensor> up_exps_scales_;
    std::vector<Tensor> up_exps_zps_;
    std::vector<Tensor> down_exps_scales_;
    std::vector<Tensor> down_exps_zps_;

    int32_t hidden_size_ = 0;
    int32_t inter_size_ = 0;
    int32_t num_experts_ = 0;
    int32_t top_k_ = 1;
    size_t  group_size_ = 128;
    bool has_gate_quant_ = false;
    bool has_up_quant_ = false;
    bool has_down_quant_ = false;
};

class Qwen3MoeDecoderLayer : public Module {
public:
    Qwen3MoeDecoderLayer(BuilderContext& ctx, const std::string& name, const Qwen3MoeConfig& cfg, Module* parent = nullptr);

    std::pair<Tensor, Tensor> forward(const Tensor& hidden_states,
                                      const Tensor& beam_idx,
                                      const Tensor& rope_cos,
                                      const Tensor& rope_sin,
                                      const Tensor& causal_mask,
                                      const std::optional<Tensor>& residual) const;

private:
    Qwen3MoeAttention self_attn_;
    Qwen3MoE moe_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

class Qwen3MoeModel : public Module {
public:
    Qwen3MoeModel(BuilderContext& ctx, const Qwen3MoeConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx);

    VocabEmbedding& embed_tokens();
    RMSNorm& norm();

private:
    VocabEmbedding embed_tokens_;
    std::vector<Qwen3MoeDecoderLayer> layers_;
    RMSNorm norm_;
    int32_t head_dim_ = 0;
    float rope_theta_ = 10000.0f;
};

class Qwen3MoeForCausalLM : public Module {
public:
    Qwen3MoeForCausalLM(BuilderContext& ctx, const Qwen3MoeConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx);

    Qwen3MoeModel& model();
    LMHead& lm_head();

private:
    Qwen3MoeConfig cfg_;
    Qwen3MoeModel model_;
    LMHead lm_head_;
};

std::shared_ptr<ov::Model> create_qwen3_moe_model(
    const Qwen3MoeConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
