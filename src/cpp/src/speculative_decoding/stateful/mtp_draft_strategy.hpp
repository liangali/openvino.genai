// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "sampling/sampler.hpp"
#include "utils.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"

#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "speculative_decoding/stateful/fast_draft_strategy.hpp"  // for LLMInferWrapper, StatefulSpeculativePipelineBase

namespace ov {
namespace genai {

// ── MtpDraftRunner ──────────────────────────────────────────────────────────
// Wraps the compiled MTP OV model (from create_qwen3_5_mtp_model()).
class MtpDraftRunner {
public:
    MtpDraftRunner(ov::InferRequest mtp_request,
                   LLMInferWrapper& main_runner);

    // Draft one token. Returns argmax of MTP logits.
    int64_t infer_next(int64_t prev_token_id, int64_t position);

    // Roll back MTP KV cache by trim_count positions.
    void trim_kv_cache(std::size_t trim_count);

    // Reset all KV state (new sequence).
    void reset_state();

    std::size_t get_num_processed_tokens() const { return num_processed_tokens_; }

private:
    ov::InferRequest          mtp_runner_;
    LLMInferWrapper&          main_runner_ref_;
    ov::genai::utils::KVAxesPosition kv_pos_;

    ov::Tensor input_ids_buf_;
    ov::Tensor position_ids_buf_;
    ov::Tensor beam_idx_buf_;

    std::size_t num_processed_tokens_ = 0;
};

// ── MtpSpeculativeLLMPipeline ───────────────────────────────────────────────
class MtpSpeculativeLLMPipeline : public StatefulSpeculativePipelineBase {
public:
    MtpSpeculativeLLMPipeline(const ov::genai::ModelDesc& main_model_desc,
                               std::shared_ptr<ov::Model>  mtp_ov_model,
                               const std::string&           device,
                               const ov::AnyMap&            properties = {});

    // Construct from a pre-compiled main model runner (avoids re-compilation of main model).
    // Used when the main model OV graph has already been freed to reduce peak memory.
    MtpSpeculativeLLMPipeline(std::unique_ptr<LLMInferWrapper> main_runner,
                               std::shared_ptr<ov::Model>        mtp_ov_model,
                               const std::string&                 device,
                               const ov::AnyMap&                  properties,
                               const ov::genai::Tokenizer&        tokenizer,
                               const ov::genai::GenerationConfig& generation_config);

    ~MtpSpeculativeLLMPipeline() override = default;

    void finish_chat() override;

protected:
    GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config) override;

    EncodedResults generate_tokens(const EncodedInputs& inputs,
                                   const GenerationConfig& config,
                                   StreamerVariant streamer) override;

private:
    std::unique_ptr<LLMInferWrapper> m_main_runner;
    std::unique_ptr<MtpDraftRunner>  m_mtp_runner;
};

}  // namespace genai
}  // namespace ov
