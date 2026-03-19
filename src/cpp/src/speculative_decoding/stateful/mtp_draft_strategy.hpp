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
#include "modeling/models/qwen3_5/mtp_draft_runner.hpp"

namespace ov {
namespace genai {

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
