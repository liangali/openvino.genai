// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include "openvino/genai/perf_metrics.hpp"

namespace ov {
namespace genai {

/**
 * DFlash-specific performance metrics, returned via EncodedResults/DecodedResults
 * extended_perf_metrics field.  Python can read these after generate().
 */
struct DFlashPerfMetrics : public ov::genai::ExtendedPerfMetrics {
    // Total number of draft speculation steps
    size_t draft_steps         = 0;
    // Total draft tokens accepted (not counting the one posterior token per step)
    size_t accepted_draft_tokens = 0;
    // Total tokens generated (prompt excluded)
    size_t generated_tokens    = 0;
    // Average accepted draft tokens per step
    double avg_accepted_per_step = 0.0;
    // Ratio: accepted_draft_tokens / (generated_tokens - draft_steps) ~ how many came from draft
    double draft_acceptance_rate = 0.0;
    // Per-step accepted count (length == draft_steps)
    std::vector<size_t> accepted_per_step;
    // Per-step accepted token IDs (each inner vector = accepted drafts + posterior_next)
    std::vector<std::vector<int64_t>> accepted_tokens_per_step;

    // Total wall-clock time for full small-model decode path per step
    // (embed + draft + lm_head + argmax), ms
    double draft_total_ms = 0.0;
    // Average full small-model decode time per speculation step, ms/step
    double avg_draft_step_ms = 0.0;
    // Average full small-model decode time per accepted draft token, ms/token
    double avg_accepted_draft_token_ms = 0.0;

    // Small model (draft) full decode timing
    size_t draft_decode_count = 0;
    double avg_draft_decode_ms = 0.0;

    // Large model (target) full decode timing inside DFlash loop
    // verify: target forward on [last + drafted block]
    // replay: target forward on accepted block
    // target_decode_count = target_verify_count + target_replay_count
    size_t target_verify_count = 0;
    size_t target_replay_count = 0;
    size_t target_decode_count = 0;

    double target_verify_total_ms = 0.0;
    double target_replay_total_ms = 0.0;
    double target_decode_total_ms = 0.0;

    double avg_target_verify_ms = 0.0;
    double avg_target_replay_ms = 0.0;
    double avg_target_decode_ms = 0.0;

    // Per-step trace lines mirrored from console logging so benchmark reports
    // can persist the detailed verify / snapshot timings.
    std::vector<std::string> verify_trace_lines;
    std::vector<std::string> snapshot_restore_trace_lines;
};

}  // namespace genai
}  // namespace ov
