// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/runtime/infer_request.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

// Wraps the compiled MTP OV draft model (from create_qwen3_5_mtp_model()).
// Takes a non-owning reference to the main model InferRequest to read
// hidden_states from its output after each main-model inference.
class MtpDraftRunner {
public:
    // mtp_request: owned.
    // main_request: non-owning reference — must outlive this object.
    //   Used only to read "hidden_states" output after main model infer.
    MtpDraftRunner(ov::InferRequest mtp_request,
                   ov::InferRequest& main_request);

    // Draft one token.
    // prev_token_id: the last confirmed output token (fed as input_ids to MTP).
    // position: 0-based KV length of the main model BEFORE this call
    //           (i.e. the position_id the draft token will occupy).
    // Reads hidden_states from main_request, runs MTP infer, returns argmax token.
    int64_t infer_next(int64_t prev_token_id, int64_t position);

    // Roll back MTP KV cache by trim_count positions.
    void trim_kv_cache(std::size_t trim_count);

    // Reset all KV state (new sequence).
    void reset_state();

    /// Draft N tokens sequentially.
    /// Before calling: caller must call inject_hidden_state_slice() to set
    /// main_runner_ref_ hidden_states to the single-token slice for position j
    /// (or leave it as-is if coming from a single-token main-model step).
    /// After return: MTP KV is at past_len + N.
    std::vector<int64_t> draft_n(int64_t next_id, int64_t past_len, int N);

    /// Extract position slice_idx from a [1, M, H] hidden_states tensor and
    /// store it as a [1, 1, H] staging tensor (staged_hs_) so the next
    /// infer_next / draft_n call feeds the correct hidden state to MTP.
    /// Does NOT modify main_runner_ref_ — avoids corrupting the main model's
    /// output tensor whose shape must remain [1, M, H] for subsequent verify passes.
    void inject_hidden_state_slice(const ov::Tensor& multi_hs, size_t slice_idx);

    std::size_t get_num_processed_tokens() const { return num_processed_tokens_; }

private:
    ov::InferRequest               mtp_runner_;
    ov::InferRequest&              main_runner_ref_;   // non-owning
    ov::genai::utils::KVAxesPosition kv_pos_;

    ov::Tensor input_ids_buf_;
    ov::Tensor position_ids_buf_;
    ov::Tensor beam_idx_buf_;

    // Staged single-token hidden state ([1, 1, H]) for the next infer_next call.
    // Set by inject_hidden_state_slice() and consumed (cleared) by infer_next().
    // Using a dedicated staging tensor avoids writing into the main model's output
    // tensor (which would corrupt its shape for subsequent verify passes).
    ov::Tensor staged_hs_;

    std::size_t num_processed_tokens_ = 0;
};

}  // namespace genai
}  // namespace ov
