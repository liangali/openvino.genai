// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <map>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "llm/pipeline_base.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

/**
 * DFlash speculative decoding pipeline for Qwen3.5 hybrid-attention models.
 *
 * Builds four sub-models from safetensors:
 *   1. Target model (Qwen3.5 CausalLM with selected-layer hidden output)
 *   2. DFlash draft model
 *   3. Embedding model (shared with target)
 *   4. LM-head model (shared with target)
 *
 * The generate loop implements:
 *   - Prefill via target model
 *   - Draft: embed block → draft model → lm-head → argmax
 *   - Verify: target model batch-verify → acceptance
 *   - Linear-attention state save/restore around verification
 */
class StatefulDFlashPipeline : public LLMPipelineImplBase {
public:
    StatefulDFlashPipeline(const ModelDesc& main_model_desc,
                           const utils::DFlashModelConfig& dflash_cfg,
                           const std::filesystem::path& main_model_path);

    ~StatefulDFlashPipeline() override;

    DecodedResults generate(StringInputs inputs,
                            OptionalGenerationConfig generation_config,
                            StreamerVariant streamer) override;

    DecodedResults generate(const ChatHistory& history,
                            OptionalGenerationConfig generation_config,
                            StreamerVariant streamer) override;

    EncodedResults generate(const EncodedInputs& inputs,
                            OptionalGenerationConfig generation_config,
                            StreamerVariant streamer) override;

    void start_chat(const std::string& system_message = "") override;
    void finish_chat() override;

private:
    // Sub-model infer requests
    ov::InferRequest m_target_request;
    ov::InferRequest m_draft_request;  // Combined: embed + draft + lm_head

    // Config
    int32_t m_block_size = 16;
    int64_t m_mask_token_id = -1;
    int64_t m_eos_token_id = -1;
    std::vector<int32_t> m_target_layer_ids;
    size_t m_hidden_dim = 0;

    // KV cache state
    ov::genai::utils::KVCacheState m_target_kv_state;

    // Snapshot-based state selection (eliminates replay)
    bool m_has_snapshots = false;
    bool m_gpu_snapshots = false;  // true when GPU RemoteTensors are bound to snapshot outputs
    bool m_has_state_update_mode_input = false;
    bool m_use_deferred_state_commit = false;
    int32_t m_pending_snapshot_commit_index = -1;  // -1 = no pending commit
    ov::RemoteContext m_remote_context;
    // Pre-allocated GPU buffers for snapshot outputs (keyed by snapshot port name)
    std::map<std::string, ov::RemoteTensor> m_snapshot_remote_tensors;

    // Model paths (for debug)
    std::filesystem::path m_main_model_path;
    std::filesystem::path m_draft_model_path;

    // Helpers
    std::vector<std::pair<std::string, ov::Tensor>> save_linear_states();
    void restore_linear_states(const std::vector<std::pair<std::string, ov::Tensor>>& saved);

    ov::Tensor make_ids_tensor(const std::vector<int64_t>& ids) const;
    ov::Tensor make_attention_mask(size_t len) const;
    ov::Tensor make_mrope_position_ids(size_t start, size_t count) const;
    ov::Tensor make_beam_idx() const;
    ov::Tensor make_state_update_mode_tensor(int32_t mode) const;

    int64_t argmax_last_token(const ov::Tensor& logits) const;
    std::vector<int64_t> argmax_logits_slice(const ov::Tensor& logits, size_t start, size_t count) const;
};

}  // namespace genai
}  // namespace ov
