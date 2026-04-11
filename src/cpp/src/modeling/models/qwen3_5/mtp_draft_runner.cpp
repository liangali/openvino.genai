// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_5/mtp_draft_runner.hpp"

#include <algorithm>
#include <cstring>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace genai {

MtpDraftRunner::MtpDraftRunner(ov::InferRequest mtp_request,
                               ov::InferRequest& main_request)
    : mtp_runner_(std::move(mtp_request)),
      main_runner_ref_(main_request) {
    // Get KV cache axis positions from the MTP model's runtime model.
    // get_runtime_model() returns nullptr for GPU-compiled models loaded from cached IR,
    // so fall back to the standard layout {batch=0, seq_len=2} in that case.
    {
        auto runtime_model = mtp_runner_.get_compiled_model().get_runtime_model();
        if (runtime_model) {
            kv_pos_ = ov::genai::utils::get_kv_axes_pos(runtime_model);
        } else {
            kv_pos_ = ov::genai::utils::KVAxesPosition{0u, 2u};
        }
    }

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
    // Allocate fresh scalar tensors each call to avoid page-lock issues after GPU inference.
    // The Intel GPU driver may pin CPU pages after mtp_runner_.infer() making subsequent
    // writes to the pre-allocated buffers fault with 0xC0000005.
    ov::Tensor fresh_ids(ov::element::i64, {1, 1});
    ov::Tensor fresh_pos(ov::element::i64, {1, 1});
    fresh_ids.data<int64_t>()[0] = prev_token_id;
    fresh_pos.data<int64_t>()[0] = position;
    mtp_runner_.set_tensor("input_ids",    fresh_ids);
    mtp_runner_.set_tensor("position_ids", fresh_pos);

    // Use the staged single-token hidden state if it was set by inject_hidden_state_slice()
    // or by the inter-step copy in draft_n(). Otherwise fall back to the main model's last
    // hidden state (the main model now outputs [1, M, H] for all positions — extract last).
    if (staged_hs_ && staged_hs_.get_size() > 0) {
        mtp_runner_.set_tensor("hidden_states", staged_hs_);
        staged_hs_ = ov::Tensor{};  // consume — cleared so the next call falls through if not re-staged
    } else {
        // main_runner_ref_ outputs hidden_states as [1, M, H] (all positions).
        // MTP model expects [1, 1, H] — extract the last position using inject helper.
        const ov::Tensor src = main_runner_ref_.get_tensor("hidden_states");
        const auto& src_shape = src.get_shape();
        const size_t M = src_shape[1];
        inject_hidden_state_slice(src, M - 1);  // stages last row into staged_hs_ as [1,1,H]
        mtp_runner_.set_tensor("hidden_states", staged_hs_);
        staged_hs_ = ov::Tensor{};  // consume
    }

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

void MtpDraftRunner::inject_hidden_state_slice(const ov::Tensor& multi_hs,
                                                size_t            slice_idx) {
    // multi_hs shape: [1, M, H]
    const auto&  shape     = multi_hs.get_shape();
    const size_t M         = shape[1];
    const size_t H         = shape[2];
    const size_t row_bytes = multi_hs.get_byte_size() / M;  // dtype-agnostic
    // Allocate (or reuse) the staging buffer as [1, 1, H].
    // Do NOT write into main_runner_ref_'s hidden_states — that is the main model's
    // output tensor whose shape ([1, M, H] after a verify pass) must not be altered.
    staged_hs_ = ov::Tensor(multi_hs.get_element_type(), {1, 1, H});
    const auto* src = static_cast<const uint8_t*>(multi_hs.data()) + slice_idx * row_bytes;
    std::memcpy(staged_hs_.data(), src, row_bytes);
}

std::vector<int64_t> MtpDraftRunner::draft_n(int64_t next_id,
                                               int64_t past_len,
                                               int     N) {
    std::vector<int64_t> drafts;
    drafts.reserve(static_cast<size_t>(N));
    int64_t cur = next_id;
    for (int i = 0; i < N; ++i) {
        int64_t d = infer_next(cur, past_len + static_cast<int64_t>(i));
        drafts.push_back(d);
        cur = d;
        // For intermediate steps: stage MTP's output hidden_states so the next
        // infer_next reads MTP's own hidden state (not a stale main-model tensor).
        // Only needed for i < N-1 (last step's output is not read by draft_n itself).
        // Write into staged_hs_ — never modify main_runner_ref_'s output tensor.
        if (i + 1 < N) {
            const ov::Tensor mtp_out = mtp_runner_.get_tensor("hidden_states");
            staged_hs_ = ov::Tensor(mtp_out.get_element_type(), mtp_out.get_shape());
            mtp_out.copy_to(staged_hs_);
        }
    }
    return drafts;
}

}  // namespace genai
}  // namespace ov
