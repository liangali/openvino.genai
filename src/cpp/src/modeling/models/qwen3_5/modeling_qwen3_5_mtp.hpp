// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/module.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"  // for component types

namespace ov {
namespace genai {
namespace modeling {
namespace models {

// MTP draft model for Qwen3.5 (dense and MoE).
// Predicts one draft token per call using:
//   embed_tokens -> (norm+concat+fc) -> DecoderLayer (full_attention) -> norm -> lm_head
// hidden_states input is the main model's last-token post-norm output.
class Qwen3_5MtpForDraft : public Module {
public:
    Qwen3_5MtpForDraft(BuilderContext& ctx,
                        const std::string& name,
                        const Qwen3_5TextModelConfig& cfg,
                        Module* parent = nullptr);

    // Returns logits [B, 1, vocab_size].
    // hidden_states: main model's post-norm last-token output [B, 1, hidden_size].
    // No attention_mask parameter: MTP always processes exactly 1 decode token.
    // Implementation passes nullptr as full_attention_mask to Qwen3_5DecoderLayer.
    // If the build compiles but crashes with a null-mask assertion at runtime, add:
    //   auto trivial_mask = ops::ones({1,1,1,1}, ov::element::f16);
    // and pass &trivial_mask in place of nullptr.
    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& hidden_states,
                   const Tensor& beam_idx) const;

private:
    Qwen3_5TextModelConfig cfg_;
    int32_t rotary_dim_ = 0;  // computed from cfg_ in constructor
    // Embed + fusion
    ov::genai::modeling::VocabEmbedding   embed_tokens_;
    Qwen3_5RMSNorm                        pre_fc_norm_embed_;
    Qwen3_5RMSNorm                        pre_fc_norm_hidden_;
    WeightParameter*                      fc_param_ = nullptr;  // weight [H, 2H]
    // Single transformer layer (full_attention, reuses existing class)
    Qwen3_5DecoderLayer                   layer_;
    // Output
    Qwen3_5RMSNorm                        norm_;
    ov::genai::modeling::LMHead           lm_head_;
};

// Builds and returns a compiled OV model for the MTP draft.
// Inputs:  input_ids [B,1,i64], position_ids [B,1,i64],
//          hidden_states [B,1,H,f16], beam_idx [B,i32]
// Outputs: logits [B,1,V]
std::shared_ptr<ov::Model> create_qwen3_5_mtp_model(
    const Qwen3_5Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
