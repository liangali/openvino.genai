// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_5/mtp_draft_runner.hpp"

#include <algorithm>

#include "openvino/core/type/float16.hpp"

namespace ov {
namespace genai {

MtpDraftRunner::MtpDraftRunner(ov::InferRequest mtp_request,
                               ov::InferRequest& main_request)
    : mtp_runner_(std::move(mtp_request)),
      main_runner_ref_(main_request) {
    // Get KV cache axis positions from the MTP model's runtime model
    kv_pos_ = ov::genai::utils::get_kv_axes_pos(
        mtp_runner_.get_compiled_model().get_runtime_model());

    // Pre-allocate scalar inputs
    input_ids_buf_    = ov::Tensor(ov::element::i64, {1, 1});
    position_ids_buf_ = ov::Tensor(ov::element::i64, {1, 1});
    beam_idx_buf_     = ov::Tensor(ov::element::i32, {1});
    beam_idx_buf_.data<int32_t>()[0] = 0;

    mtp_runner_.set_tensor("input_ids",    input_ids_buf_);
    mtp_runner_.set_tensor("position_ids", position_ids_buf_);
    mtp_runner_.set_tensor("beam_idx",     beam_idx_buf_);
}

int64_t MtpDraftRunner::infer_next(int64_t prev_token_id, int64_t position) {
    input_ids_buf_.data<int64_t>()[0]    = prev_token_id;
    position_ids_buf_.data<int64_t>()[0] = position;

    // Copy hidden_states from main model's output to avoid aliasing
    // KEY CHANGE vs old MtpDraftRunner: direct get_tensor() instead of
    // main_runner_ref_.get_infer_request().get_tensor()
    const ov::Tensor src = main_runner_ref_.get_tensor("hidden_states");
    ov::Tensor hidden_copy(src.get_element_type(), src.get_shape());
    src.copy_to(hidden_copy);
    mtp_runner_.set_tensor("hidden_states", hidden_copy);

    mtp_runner_.infer();
    ++num_processed_tokens_;

    // Argmax over vocabulary: logits shape [1, 1, V]
    const ov::Tensor logits = mtp_runner_.get_tensor("logits");
    const auto vocab_size   = logits.get_shape()[2];

    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<ov::float16>();
        std::size_t best = 0;
        float best_val = static_cast<float>(data[0]);
        for (std::size_t i = 1; i < vocab_size; ++i) {
            float v = static_cast<float>(data[i]);
            if (v > best_val) { best_val = v; best = i; }
        }
        return static_cast<int64_t>(best);
    } else {
        const auto* data = logits.data<float>();
        return static_cast<int64_t>(
            std::max_element(data, data + vocab_size) - data);
    }
}

void MtpDraftRunner::trim_kv_cache(std::size_t trim_count) {
    if (trim_count == 0 || num_processed_tokens_ < trim_count) return;

    ov::genai::utils::KVCacheState to_trim_state;
    to_trim_state.num_tokens_to_trim = trim_count;
    to_trim_state.seq_length_axis    = kv_pos_.seq_len;
    to_trim_state.reset_mem_state    = false;
    ov::genai::utils::trim_kv_cache(mtp_runner_, to_trim_state, {});

    num_processed_tokens_ -= trim_count;
}

void MtpDraftRunner::reset_state() {
    mtp_runner_.reset_state();
    num_processed_tokens_ = 0;
}

}  // namespace genai
}  // namespace ov
