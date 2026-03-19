// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mtp_draft_strategy.hpp"
#include "continuous_batching/timer.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "utils.hpp"

#include <algorithm>
#include <iostream>
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

    // Compile MTP model and build runner.
    // The MTP model is built in F32 and does not need quantization.
    // Strip QUANTIZATION_CONFIG (and any GGUF properties) from the compile properties
    // since GPU plugin rejects unknown options.
    const ov::AnyMap& base_props = properties.empty() ? main_model_desc.properties : properties;
    auto [props_no_quant, _qcfg] = ov::genai::utils::extract_quantization_config(base_props);
    auto [mtp_props, _gguf] = ov::genai::utils::extract_gguf_properties(props_no_quant);
    auto mtp_compiled = ov::genai::utils::singleton_core().compile_model(
        mtp_ov_model, device.empty() ? main_model_desc.device : device, mtp_props);
    auto mtp_request = mtp_compiled.create_infer_request();

    m_mtp_runner = std::make_unique<MtpDraftRunner>(
        std::move(mtp_request), m_main_runner->get_infer_request());
    OPENVINO_ASSERT(m_mtp_runner != nullptr, "Failed to create MTP draft runner");
}

MtpSpeculativeLLMPipeline::MtpSpeculativeLLMPipeline(
    std::unique_ptr<LLMInferWrapper> main_runner,
    std::shared_ptr<ov::Model>        mtp_ov_model,
    const std::string&                 device,
    const ov::AnyMap&                  properties,
    const ov::genai::Tokenizer&        tokenizer,
    const ov::genai::GenerationConfig& generation_config)
    : StatefulSpeculativePipelineBase(tokenizer, generation_config) {
    OPENVINO_ASSERT(main_runner != nullptr, "Main model runner cannot be null");
    OPENVINO_ASSERT(mtp_ov_model != nullptr, "MTP model cannot be null");

    m_tokenizer    = tokenizer;
    m_main_runner  = std::move(main_runner);

    // Compile MTP model; strip quantization/GGUF properties.
    // The MTP model itself is always built in F32 (small 1-layer draft head);
    // quantization properties are not needed here.
    auto [props_no_quant, _qcfg] = ov::genai::utils::extract_quantization_config(properties);
    auto [mtp_props, _gguf]      = ov::genai::utils::extract_gguf_properties(props_no_quant);

    // Try compiling MTP on the same device as the main model.
    // If that fails (e.g. GPU VRAM exhausted by the large main model), fall back to CPU.
    ov::InferRequest mtp_request;
    bool compiled_on_gpu = false;
    try {
        auto mtp_compiled = ov::genai::utils::singleton_core().compile_model(
            mtp_ov_model, device, mtp_props);
        mtp_request = mtp_compiled.create_infer_request();
        compiled_on_gpu = true;
    } catch (const std::exception& e) {
        if (device != "CPU") {
            std::cerr << "[MTP] " << device << " compile failed (" << e.what()
                      << "), falling back to CPU for MTP draft model." << std::endl;
            auto mtp_compiled_cpu = ov::genai::utils::singleton_core().compile_model(
                mtp_ov_model, "CPU");
            mtp_request = mtp_compiled_cpu.create_infer_request();
        } else {
            throw;
        }
    }
    (void)compiled_on_gpu;

    m_mtp_runner = std::make_unique<MtpDraftRunner>(
        std::move(mtp_request), m_main_runner->get_infer_request());
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
    std::size_t accept_count = 0, reject_count = 0;

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

        // Step 1: verify out_token — one main model call at position current_pos.
        // Main model takes out_token as input and predicts the next token.
        int64_t ref0 = m_main_runner->infer_next(out_token);

        if (ref0 == draft_token) {
            // ACCEPT: draft was correct. Run main model on draft_token to get next token.
            int64_t ref1 = m_main_runner->infer_next(draft_token);
            streaming_status = stream_generated_tokens(streamer_ptr,
                std::vector<int64_t>{draft_token, ref1});
            results.tokens[0].push_back(draft_token);
            results.tokens[0].push_back(ref1);
            out_token        = ref1;
            mtp_prefix_token = draft_token;

            ++accept_count;
            m_sd_metrics.update_acceptance_rate(0, 100.f);
            m_sd_metrics.update_draft_accepted_tokens(0, 1u);
            m_sd_metrics.update_generated_len(2u);
        } else {
            // REJECT: use ref0 as the output token.
            // MTP already processed out_token in its KV; main also processed out_token.
            // Both are in sync at current_pos+1. No KV trim needed.
            streaming_status = stream_generated_tokens(streamer_ptr,
                std::vector<int64_t>{ref0});
            results.tokens[0].push_back(ref0);
            out_token        = ref0;
            // mtp_prefix_token stays -1

            ++reject_count;
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

    // Print accept/reject rate
    const std::size_t total = accept_count + reject_count;
    if (total > 0) {
        std::cerr << "[MTP] Accept rate: " << accept_count << "/" << total
                  << " = " << (100.0 * accept_count / total) << "%" << std::endl;
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
