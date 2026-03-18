// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mtp_draft_strategy.hpp"
#include "continuous_batching/timer.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/genai/text_streamer.hpp"

#include <algorithm>
#include <numeric>

namespace {

ov::genai::StreamingStatus stream_generated_tokens(std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

}  // anonymous namespace

namespace ov {
namespace genai {

// ── MtpDraftRunner ───────────────────────────────────────────────────────────

MtpDraftRunner::MtpDraftRunner(ov::InferRequest mtp_request,
                               LLMInferWrapper& main_runner)
    : mtp_runner_(std::move(mtp_request)),
      main_runner_ref_(main_runner) {
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
    const ov::Tensor src = main_runner_ref_.get_infer_request().get_tensor("hidden_states");
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

// ── MtpSpeculativeLLMPipeline ────────────────────────────────────────────────

MtpSpeculativeLLMPipeline::MtpSpeculativeLLMPipeline(
    const ov::genai::ModelDesc& main_model_desc,
    std::shared_ptr<ov::Model>  mtp_ov_model,
    const std::string&           device,
    const ov::AnyMap&            properties)
    : StatefulSpeculativePipelineBase(main_model_desc.tokenizer,
                                      main_model_desc.generation_config) {
    OPENVINO_ASSERT(main_model_desc.model != nullptr, "Main model cannot be null");
    OPENVINO_ASSERT(mtp_ov_model != nullptr, "MTP model cannot be null");

    m_tokenizer = main_model_desc.tokenizer;

    // Main model runner
    m_main_runner = std::make_unique<LLMInferWrapper>(main_model_desc);
    OPENVINO_ASSERT(m_main_runner != nullptr, "Failed to create main model inference wrapper");

    // Compile MTP model and build runner
    auto mtp_compiled = ov::genai::utils::singleton_core().compile_model(
        mtp_ov_model, device.empty() ? main_model_desc.device : device,
        properties.empty() ? main_model_desc.properties : properties);
    auto mtp_request = mtp_compiled.create_infer_request();

    m_mtp_runner = std::make_unique<MtpDraftRunner>(
        std::move(mtp_request), *m_main_runner);
    OPENVINO_ASSERT(m_mtp_runner != nullptr, "Failed to create MTP draft runner");
}

GenerationConfig MtpSpeculativeLLMPipeline::resolve_generation_config(
    OptionalGenerationConfig generation_config) {
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(generation_config);
    config.validate();
    return config;
}

EncodedResults MtpSpeculativeLLMPipeline::generate_tokens(const EncodedInputs& inputs,
                                                          const GenerationConfig& config,
                                                          StreamerVariant streamer) {
    ManualTimer generate_timer("MtpSpeculativeLLMPipeline::generate_tokens(EncodedInputs)");
    generate_timer.start();

    ov::Tensor input_ids;
    ov::Tensor attention_mask;

    if (auto data = std::get_if<ov::Tensor>(&inputs)) {
        input_ids     = *data;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto data = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids      = data->input_ids;
        attention_mask = data->attention_mask;
    }

    ov::Shape prompt_shape = input_ids.get_shape();
    const size_t batch_size = prompt_shape[0];
    OPENVINO_ASSERT(batch_size == 1u, "Currently only batch size=1 is supported");

    OPENVINO_ASSERT(config.is_greedy_decoding(), "Currently only greedy decoding is supported");
    OPENVINO_ASSERT(config.num_return_sequences == 1u,
        "Currently only \"num_return_sequences\" equal to 1 is supported!");

    m_main_runner->set_generation_config(config);

    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    ov::genai::EncodedResults results;
    auto& raw_perf_counters = m_sd_perf_metrics.raw_metrics;
    results.scores.resize(1u);
    results.scores[0] = 0u;
    results.tokens.resize(1u);

    ov::Tensor position_ids{ov::element::i64, input_ids.get_shape()};
    utils::initialize_position_ids(position_ids, attention_mask);

    // ── Prefill ──────────────────────────────────────────────────────────────
    ManualTimer first_token_timer("MTP speculative decode: first token timer");
    first_token_timer.start();

    int64_t out_token = m_main_runner->infer_first(input_ids, attention_mask, position_ids);

    first_token_timer.end();
    raw_perf_counters.m_token_infer_durations.emplace_back(first_token_timer.get_duration_microsec());
    raw_perf_counters.m_inference_durations[0] += ov::genai::MicroSeconds(first_token_timer.get_duration_microsec());
    raw_perf_counters.m_new_token_times.emplace_back(first_token_timer.get_end_time());
    raw_perf_counters.m_batch_sizes.emplace_back(1u);

    auto streaming_status = stream_generated_tokens(streamer_ptr, std::vector<int64_t>{out_token});
    results.tokens[0].push_back(out_token);

    int64_t mtp_prefix_token = -1;

    // ── Decode loop ───────────────────────────────────────────────────────────
    ManualTimer iteration_timer("MTP speculative decode: iteration");

    while (m_main_runner->can_infer() && (streaming_status == ov::genai::StreamingStatus::RUNNING)) {
        iteration_timer.start();

        int64_t current_pos = static_cast<int64_t>(m_main_runner->get_num_processed_tokens());

        // MTP KV sync after ACCEPT: run MTP on the previously accepted draft token
        if (mtp_prefix_token >= 0) {
            m_mtp_runner->infer_next(mtp_prefix_token, current_pos - 1);
            mtp_prefix_token = -1;
        }

        // Draft one token using MTP
        int64_t draft_token = m_mtp_runner->infer_next(out_token, current_pos);

        // Verify: ONE main call for both tokens
        auto ref_tokens = m_main_runner->infer_next_return_all({out_token, draft_token});
        // ref_tokens[0] = prediction for out_token position (= what main thinks follows out_token)
        // ref_tokens[1] = prediction for draft_token position

        if (ref_tokens[0] == draft_token) {
            // ACCEPT: draft was correct
            streaming_status = stream_generated_tokens(streamer_ptr,
                std::vector<int64_t>{draft_token, ref_tokens[1]});
            results.tokens[0].push_back(draft_token);
            results.tokens[0].push_back(ref_tokens[1]);
            out_token        = ref_tokens[1];
            mtp_prefix_token = draft_token;

            m_sd_metrics.update_acceptance_rate(0, 100.f);
            m_sd_metrics.update_draft_accepted_tokens(0, 1u);
            m_sd_metrics.update_generated_len(2u);
        } else {
            // REJECT: trim ONLY main KV cache, not MTP
            m_main_runner->trim_kv_cache(1);
            streaming_status = stream_generated_tokens(streamer_ptr,
                std::vector<int64_t>{ref_tokens[0]});
            results.tokens[0].push_back(ref_tokens[0]);
            out_token        = ref_tokens[0];
            // mtp_prefix_token stays -1

            m_sd_metrics.update_acceptance_rate(0, 0.f);
            m_sd_metrics.update_draft_accepted_tokens(0, 0u);
            m_sd_metrics.update_generated_len(1u);
        }

        m_sd_metrics.update_draft_generated_len(0, 1u);

        iteration_timer.end();
        raw_perf_counters.m_token_infer_durations.emplace_back(iteration_timer.get_duration_microsec());
        raw_perf_counters.m_inference_durations[0] += ov::genai::MicroSeconds(iteration_timer.get_duration_microsec());
        raw_perf_counters.m_new_token_times.emplace_back(iteration_timer.get_end_time());
        raw_perf_counters.m_batch_sizes.emplace_back(
            (mtp_prefix_token >= 0) ? 2u : 1u);
        iteration_timer.clear();
    }

    m_streaming_was_cancelled = (streaming_status == ov::genai::StreamingStatus::CANCEL);
    if (streamer_ptr) {
        streamer_ptr->end();
    }

    // Reset state if not in a chat session
    if (!m_is_chat_active) {
        m_main_runner->reset_state();
        m_mtp_runner->reset_state();
    }

    generate_timer.end();

    // Perf metrics
    m_sd_perf_metrics.num_input_tokens = input_ids.get_shape().at(1);
    m_sd_perf_metrics.load_time = this->m_load_time_ms;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());

    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_main_runner->raw_perf_metrics;

    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());

    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);

    generate_timer.clear();
    iteration_timer.clear();
    return results;
}

void MtpSpeculativeLLMPipeline::finish_chat() {
    StatefulSpeculativePipelineBase::finish_chat();
    m_main_runner->reset_state();
    m_mtp_runner->reset_state();
}

}  // namespace genai
}  // namespace ov
