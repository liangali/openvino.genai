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

struct Qwen3DenseConfig {
    std::string architecture = "qwen3";
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
};

class Qwen3Attention : public Module {
public:
    Qwen3Attention(BuilderContext& ctx, const std::string& name, const Qwen3DenseConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& positions, const Tensor& hidden_states, const Tensor& beam_idx) const;
    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin,
                   const Tensor& attention_mask) const;
    Tensor forward_no_cache(const Tensor& hidden_states,
                            const Tensor& rope_cos,
                            const Tensor& rope_sin,
                            const Tensor& attention_mask) const;

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

class Qwen3MLP : public Module {
public:
    Qwen3MLP(BuilderContext& ctx, const std::string& name, const Qwen3DenseConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

class Qwen3DecoderLayer : public Module {
public:
    Qwen3DecoderLayer(BuilderContext& ctx, const std::string& name, const Qwen3DenseConfig& cfg, Module* parent = nullptr);

    std::pair<Tensor, Tensor> forward(const Tensor& hidden_states,
                                      const Tensor& beam_idx,
                                      const Tensor& rope_cos,
                                      const Tensor& rope_sin,
                                      const Tensor& attention_mask,
                                      const std::optional<Tensor>& residual) const;
    std::pair<Tensor, Tensor> forward_no_cache(const Tensor& hidden_states,
                                               const Tensor& rope_cos,
                                               const Tensor& rope_sin,
                                               const Tensor& attention_mask,
                                               const std::optional<Tensor>& residual) const;

private:
    Qwen3Attention self_attn_;
    Qwen3MLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

class Qwen3Model : public Module {
public:
    Qwen3Model(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent = nullptr);
    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor& attention_mask);
    std::pair<Tensor, Tensor> forward_with_penultimate(const Tensor& input_ids,
                                                       const Tensor& position_ids,
                                                       const Tensor& beam_idx);
    std::pair<Tensor, Tensor> forward_with_selected_layers(const Tensor& input_ids,
                                                           const Tensor& position_ids,
                                                           const Tensor& beam_idx,
                                                           const Tensor& attention_mask,
                                                           const std::vector<int32_t>& layer_ids);
    Tensor forward_no_cache(const Tensor& input_ids,
                            const Tensor& position_ids,
                            const Tensor& attention_mask);
    Tensor forward_embeds(const Tensor& inputs_embeds,
                          const Tensor& position_ids,
                          const Tensor& beam_idx,
                          const Tensor& attention_mask);
    std::pair<Tensor, Tensor> forward_with_penultimate_no_cache(const Tensor& input_ids,
                                                                const Tensor& position_ids,
                                                                const Tensor& attention_mask);
    std::pair<Tensor, Tensor> forward_with_pre_norm_no_cache(const Tensor& input_ids,
                                                             const Tensor& position_ids,
                                                             const Tensor& attention_mask);
    std::pair<Tensor, Tensor> forward_with_selected_layers_no_cache(const Tensor& input_ids,
                                                                    const Tensor& position_ids,
                                                                    const Tensor& attention_mask,
                                                                    const std::vector<int32_t>& layer_ids);

    VocabEmbedding& embed_tokens();
    RMSNorm& norm();

private:
    VocabEmbedding embed_tokens_;
    std::vector<Qwen3DecoderLayer> layers_;
    RMSNorm norm_;
    int32_t head_dim_ = 0;
    float rope_theta_ = 10000.0f;
};

class Qwen3ForCausalLM : public Module {
public:
    Qwen3ForCausalLM(BuilderContext& ctx, const Qwen3DenseConfig& cfg, Module* parent = nullptr);
    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor& attention_mask);
    Tensor forward_embeds(const Tensor& inputs_embeds,
                          const Tensor& position_ids,
                          const Tensor& beam_idx,
                          const Tensor& attention_mask);

    Qwen3Model& model();
    LMHead& lm_head();

private:
    Qwen3DenseConfig cfg_;
    Qwen3Model model_;
    LMHead lm_head_;
};

std::shared_ptr<ov::Model> create_qwen3_dense_model(
    const Qwen3DenseConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_text_encoder_model(
    const Qwen3DenseConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_dflash_target_model(
    const Qwen3DenseConfig& cfg,
    const std::vector<int32_t>& target_layer_ids,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_dflash_target_model_no_cache(
    const Qwen3DenseConfig& cfg,
    const std::vector<int32_t>& target_layer_ids,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_embedding_model(
    const Qwen3DenseConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_lm_head_model(
    const Qwen3DenseConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    const ov::element::Type& input_type);

std::shared_ptr<ov::Model> create_qwen3_dflash_target_model(
    const Qwen3DenseConfig& cfg,
    const std::vector<int32_t>& target_layer_ids,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_dflash_target_model_no_cache(
    const Qwen3DenseConfig& cfg,
    const std::vector<int32_t>& target_layer_ids,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_embedding_model(
    const Qwen3DenseConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_qwen3_lm_head_model(
    const Qwen3DenseConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    const ov::element::Type& input_type = ov::element::f32);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

