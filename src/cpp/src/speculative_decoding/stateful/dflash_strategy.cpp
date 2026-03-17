// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "loaders/model_config.hpp"
#include "modeling/models/dflash_draft/dflash_draft.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

namespace ov {
namespace genai {

namespace {

int64_t resolve_mask_token_id(const Tokenizer& tokenizer) {
    const auto vocab = tokenizer.get_vocab();
    auto it = vocab.find("<|MASK|>");
    if (it != vocab.end()) return it->second;
    auto it2 = vocab.find("<|fim_pad|>");
    if (it2 != vocab.end()) return it2->second;
    auto it3 = vocab.find("<|vision_pad|>");
    if (it3 != vocab.end()) return it3->second;
    const int64_t pad = tokenizer.get_pad_token_id();
    if (pad != -1) return pad;
    return tokenizer.get_eos_token_id();
}

template <typename T>
int64_t argmax_row(const T* data, size_t vocab) {
    T max_val = data[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (data[i] > max_val) { max_val = data[i]; max_idx = i; }
    }
    return static_cast<int64_t>(max_idx);
}

ov::Tensor ensure_f32_copy(const ov::Tensor& t) {
    ov::Tensor out(ov::element::f32, t.get_shape());
    t.copy_to(out);
    return out;
}

}  // namespace

// ── Construction ──────────────────────────────────────────────────────────────

StatefulDFlashPipeline::StatefulDFlashPipeline(
    const ModelDesc& main_model_desc,
    const utils::DFlashModelConfig& dflash_cfg,
    const std::filesystem::path& main_model_path)
    : LLMPipelineImplBase(main_model_desc.tokenizer, main_model_desc.generation_config),
      m_main_model_path(main_model_path),
      m_draft_model_path(dflash_cfg.draft_model_path) {

    const std::string device = dflash_cfg.device.empty() ? main_model_desc.device : dflash_cfg.device;

    // Load configs
    auto target_qwen35_cfg = modeling::models::Qwen3_5Config::from_json_file(main_model_path);
    auto draft_cfg = loaders::ModelConfig::from_hf_json(dflash_cfg.draft_model_path / "config.json");

    // DFlash draft config
    modeling::models::DFlashDraftConfig dc;
    dc.hidden_size = draft_cfg.hidden_size;
    dc.intermediate_size = draft_cfg.intermediate_size;
    dc.num_hidden_layers = draft_cfg.num_hidden_layers;
    dc.num_target_layers = draft_cfg.num_target_layers > 0
                               ? draft_cfg.num_target_layers
                               : target_qwen35_cfg.text.num_hidden_layers;
    dc.num_attention_heads = draft_cfg.num_attention_heads;
    dc.num_key_value_heads = draft_cfg.num_key_value_heads > 0
                                 ? draft_cfg.num_key_value_heads
                                 : draft_cfg.num_attention_heads;
    dc.head_dim = draft_cfg.head_dim > 0
                      ? draft_cfg.head_dim
                      : (draft_cfg.hidden_size / draft_cfg.num_attention_heads);
    dc.block_size = draft_cfg.block_size > 0 ? draft_cfg.block_size : 16;
    dc.rms_norm_eps = draft_cfg.rms_norm_eps;
    dc.rope_theta = draft_cfg.rope_theta;
    dc.hidden_act = draft_cfg.hidden_act;
    dc.attention_bias = draft_cfg.attention_bias;

    m_block_size = dc.block_size;
    m_target_layer_ids = modeling::models::build_target_layer_ids(
        dc.num_target_layers, dc.num_hidden_layers);

    // Load safetensors weights
    auto target_data = safetensors::load_safetensors(main_model_path);
    safetensors::SafetensorsWeightSource target_source(std::move(target_data));
    safetensors::SafetensorsWeightFinalizer target_finalizer;

    auto draft_data = safetensors::load_safetensors(dflash_cfg.draft_model_path);
    safetensors::SafetensorsWeightSource draft_source(std::move(draft_data));
    safetensors::SafetensorsWeightFinalizer draft_finalizer;

    // Build 4 sub-models
    auto target_model = modeling::models::create_qwen3_5_dflash_target_model(
        target_qwen35_cfg, m_target_layer_ids, target_source, target_finalizer);
    auto draft_model = modeling::models::create_dflash_draft_model(
        dc, draft_source, draft_finalizer, ov::element::f32);
    auto embed_model = modeling::models::create_qwen3_5_embedding_model(
        target_qwen35_cfg, target_source, target_finalizer);
    auto lm_head_model = modeling::models::create_qwen3_5_lm_head_model(
        target_qwen35_cfg, target_source, target_finalizer, ov::element::f32);

    // Compile
    ov::Core core;
    ov::AnyMap compile_cfg = {
        {ov::hint::inference_precision.name(), ov::element::f16},
        {ov::hint::kv_cache_precision.name(), ov::element::f16},
        {ov::hint::activations_scale_factor.name(), 8.0f}};

    auto compiled_target = core.compile_model(target_model, device, compile_cfg);
    auto compiled_draft = core.compile_model(draft_model, device, compile_cfg);
    auto compiled_embed = core.compile_model(embed_model, device, compile_cfg);
    auto compiled_lm_head = core.compile_model(lm_head_model, device, compile_cfg);

    m_target_request = compiled_target.create_infer_request();
    m_draft_request = compiled_draft.create_infer_request();
    m_embed_request = compiled_embed.create_infer_request();
    m_lm_head_request = compiled_lm_head.create_infer_request();

    m_target_request.reset_state();

    auto kv_pos = ov::genai::utils::get_kv_axes_pos(compiled_target.get_runtime_model());
    m_target_kv_state.seq_length_axis = kv_pos.seq_len;

    // Token IDs
    m_mask_token_id = resolve_mask_token_id(m_tokenizer);
    m_eos_token_id = m_tokenizer.get_eos_token_id();
}

StatefulDFlashPipeline::~StatefulDFlashPipeline() = default;

// ── Helpers ───────────────────────────────────────────────────────────────────

ov::Tensor StatefulDFlashPipeline::make_ids_tensor(const std::vector<int64_t>& ids) const {
    ov::Tensor t(ov::element::i64, {1, ids.size()});
    std::memcpy(t.data(), ids.data(), ids.size() * sizeof(int64_t));
    return t;
}

ov::Tensor StatefulDFlashPipeline::make_attention_mask(size_t len) const {
    ov::Tensor mask(ov::element::i64, {1, len});
    auto* data = mask.data<int64_t>();
    for (size_t i = 0; i < len; ++i) data[i] = 1;
    return mask;
}

ov::Tensor StatefulDFlashPipeline::make_mrope_position_ids(size_t start, size_t count) const {
    ov::Tensor ids(ov::element::i64, {3, 1, count});
    auto* data = ids.data<int64_t>();
    for (size_t dim = 0; dim < 3; ++dim)
        for (size_t i = 0; i < count; ++i)
            data[dim * count + i] = static_cast<int64_t>(start + i);
    return ids;
}

ov::Tensor StatefulDFlashPipeline::make_beam_idx() const {
    ov::Tensor b(ov::element::i32, {1});
    b.data<int32_t>()[0] = 0;
    return b;
}

int64_t StatefulDFlashPipeline::argmax_last_token(const ov::Tensor& logits) const {
    auto shape = logits.get_shape();
    size_t seq_len = shape[1], vocab = shape[2];
    return argmax_logits_slice(logits, seq_len - 1, 1).front();
}

std::vector<int64_t> StatefulDFlashPipeline::argmax_logits_slice(
    const ov::Tensor& logits, size_t start, size_t count) const {
    auto shape = logits.get_shape();
    size_t vocab = shape[2];
    std::vector<int64_t> tokens;
    tokens.reserve(count);
    if (logits.get_element_type() == ov::element::f16) {
        auto* d = logits.data<const ov::float16>();
        for (size_t i = 0; i < count; ++i)
            tokens.push_back(argmax_row(d + (start + i) * vocab, vocab));
    } else if (logits.get_element_type() == ov::element::bf16) {
        auto* d = logits.data<const ov::bfloat16>();
        for (size_t i = 0; i < count; ++i)
            tokens.push_back(argmax_row(d + (start + i) * vocab, vocab));
    } else {
        auto* d = logits.data<const float>();
        for (size_t i = 0; i < count; ++i)
            tokens.push_back(argmax_row(d + (start + i) * vocab, vocab));
    }
    return tokens;
}

std::vector<std::pair<std::string, ov::Tensor>> StatefulDFlashPipeline::save_linear_states() {
    std::vector<std::pair<std::string, ov::Tensor>> saved;
    for (auto& state : m_target_request.query_state()) {
        if (state.get_name().find("linear_states.") != std::string::npos) {
            auto src = state.get_state();
            ov::Tensor copy(src.get_element_type(), src.get_shape());
            src.copy_to(copy);
            saved.emplace_back(state.get_name(), std::move(copy));
        }
    }
    return saved;
}

void StatefulDFlashPipeline::restore_linear_states(
    const std::vector<std::pair<std::string, ov::Tensor>>& saved) {
    for (auto& state : m_target_request.query_state()) {
        for (const auto& [name, tensor] : saved) {
            if (state.get_name() == name) {
                state.set_state(tensor);
                break;
            }
        }
    }
}

// ── Generate (string inputs) ─────────────────────────────────────────────────

DecodedResults StatefulDFlashPipeline::generate(
    StringInputs inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {

    // Tokenize
    std::string prompt;
    if (auto* str = std::get_if<std::string>(&inputs)) {
        prompt = *str;
    } else {
        auto& vec = std::get<std::vector<std::string>>(inputs);
        OPENVINO_ASSERT(vec.size() == 1, "DFlash pipeline supports batch size 1 only");
        prompt = vec[0];
    }

    // Apply chat template
    std::string formatted_prompt = prompt;
    bool add_special = true;
    try {
        if (!m_tokenizer.get_chat_template().empty()) {
            ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            formatted_prompt = m_tokenizer.apply_chat_template(history, true);
            add_special = false;
        }
    } catch (...) {}

    auto encoded = m_tokenizer.encode(formatted_prompt, {ov::genai::add_special_tokens(add_special)});
    EncodedInputs enc_inputs = encoded;
    auto enc_results = generate(enc_inputs, generation_config, streamer);

    // Decode
    DecodedResults decoded;
    decoded.texts.resize(enc_results.tokens.size());
    for (size_t i = 0; i < enc_results.tokens.size(); ++i) {
        decoded.texts[i] = m_tokenizer.decode(enc_results.tokens[i]);
    }
    decoded.scores = enc_results.scores;
    decoded.perf_metrics = enc_results.perf_metrics;
    return decoded;
}

// ── Generate (chat history) ──────────────────────────────────────────────────

DecodedResults StatefulDFlashPipeline::generate(
    const ChatHistory& history,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {
    std::string prompt = m_tokenizer.apply_chat_template(history, true);
    return generate(StringInputs(prompt), generation_config, streamer);
}

// ── Generate (encoded inputs) ────────────────────────────────────────────────

EncodedResults StatefulDFlashPipeline::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {

    auto config = generation_config.has_value() ? *generation_config : m_generation_config;
    if (config.stop_token_ids.empty()) {
        config.stop_token_ids = m_generation_config.stop_token_ids;
    }
    if (config.eos_token_id == -1) {
        if (m_generation_config.eos_token_id != -1) {
            config.set_eos_token_id(m_generation_config.eos_token_id);
        } else {
            config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }
    }
    config.validate();

    const auto should_stop_on_token = [&](int64_t token) {
        if (!config.ignore_eos && token == config.eos_token_id) {
            return true;
        }
        return config.stop_token_ids.find(token) != config.stop_token_ids.end();
    };

    const int max_new_tokens = config.max_new_tokens;

    // Extract input_ids
    std::vector<int64_t> prompt_ids;
    if (auto* tokenized = std::get_if<TokenizedInputs>(&inputs)) {
        auto shape = tokenized->input_ids.get_shape();
        auto* data = tokenized->input_ids.data<const int64_t>();
        prompt_ids.assign(data, data + shape[1]);
    } else {
        auto& tensor = std::get<ov::Tensor>(inputs);
        auto shape = tensor.get_shape();
        auto* data = tensor.data<const int64_t>();
        prompt_ids.assign(data, data + shape[1]);
    }

    std::vector<int64_t> output_ids = prompt_ids;
    const size_t prompt_len = prompt_ids.size();
    const size_t max_length = prompt_len + static_cast<size_t>(max_new_tokens);

    // Setup streamer
    std::shared_ptr<StreamerBase> streamer_ptr;
    std::function<StreamingStatus(std::string)> streamer_fn;
    if (auto* fn = std::get_if<std::function<StreamingStatus(std::string)>>(&streamer)) {
        streamer_fn = *fn;
    } else if (auto* base = std::get_if<std::shared_ptr<StreamerBase>>(&streamer)) {
        streamer_ptr = *base;
    }

    auto stream_token = [&](int64_t token) -> bool {
        if (streamer_fn) {
            auto text = m_tokenizer.decode({token}, {ov::genai::skip_special_tokens(true)});
            return streamer_fn(text) != StreamingStatus::RUNNING;
        }
        if (streamer_ptr) {
            return streamer_ptr->write(token) != StreamingStatus::RUNNING;
        }
        return false;
    };

    auto beam_idx = make_beam_idx();

    // Reset target state
    m_target_request.reset_state();

    // ── Prefill ──
    m_target_request.set_tensor("input_ids", make_ids_tensor(output_ids));
    m_target_request.set_tensor("attention_mask", make_attention_mask(output_ids.size()));
    m_target_request.set_tensor("position_ids", make_mrope_position_ids(0, output_ids.size()));
    m_target_request.set_tensor("beam_idx", beam_idx);
    m_target_request.infer();

    auto logits = m_target_request.get_tensor("logits");
    auto target_hidden_block = ensure_f32_copy(m_target_request.get_tensor("target_hidden"));
    m_hidden_dim = target_hidden_block.get_shape()[2];

    // Hidden state storage
    ov::Tensor target_hidden_storage(ov::element::f32, {1, max_length + m_block_size, m_hidden_dim});
    {
        ov::Tensor init_slice(target_hidden_storage, {0, 0, 0}, {1, prompt_len, m_hidden_dim});
        target_hidden_block.copy_to(init_slice);
    }
    size_t target_hidden_len = prompt_len;

    int64_t next_token = argmax_last_token(logits);
    output_ids.push_back(next_token);
    bool stopped = stream_token(next_token) || should_stop_on_token(next_token);

    // ── Decode loop ──
    while (output_ids.size() < max_length && !stopped) {
        if (should_stop_on_token(next_token)) break;

        // Build draft block: [last_token, MASK, ...]
        std::vector<int64_t> block_ids(static_cast<size_t>(m_block_size), m_mask_token_id);
        block_ids[0] = output_ids.back();

        // Embed
        m_embed_request.set_tensor("input_ids", make_ids_tensor(block_ids));
        m_embed_request.infer();
        auto noise_embedding = m_embed_request.get_tensor("embeddings");

        // Draft position_ids (2D)
        const size_t total_pos = target_hidden_len + block_ids.size();
        ov::Tensor draft_pos(ov::element::i64, {1, total_pos});
        {
            auto* pd = draft_pos.data<int64_t>();
            for (size_t i = 0; i < target_hidden_len; ++i)
                pd[i] = static_cast<int64_t>(i);
            for (size_t i = target_hidden_len; i < total_pos; ++i)
                pd[i] = static_cast<int64_t>(target_hidden_len - 1 + (i - target_hidden_len));
        }

        // Run draft
        ov::Tensor hidden_view(target_hidden_storage, {0, 0, 0}, {1, target_hidden_len, m_hidden_dim});
        m_draft_request.set_tensor("target_hidden", hidden_view);
        m_draft_request.set_tensor("noise_embedding", noise_embedding);
        m_draft_request.set_tensor("position_ids", draft_pos);
        m_draft_request.infer();

        auto draft_hidden = m_draft_request.get_tensor("draft_hidden");
        // Convert dtype if needed
        const auto& lm_port = m_lm_head_request.get_compiled_model().input("hidden_states");
        if (draft_hidden.get_element_type() != lm_port.get_element_type()) {
            ov::Tensor converted(lm_port.get_element_type(), draft_hidden.get_shape());
            draft_hidden.copy_to(converted);
            draft_hidden = converted;
        }

        // LM head
        m_lm_head_request.set_tensor("hidden_states", draft_hidden);
        m_lm_head_request.infer();
        auto draft_logits = m_lm_head_request.get_tensor("logits");

        // Argmax draft tokens (skip pos 0)
        const size_t draft_len = block_ids.size() - 1;
        auto draft_tokens = argmax_logits_slice(draft_logits, 1, draft_len);

        // Build verify block
        std::vector<int64_t> block_output_ids;
        block_output_ids.reserve(block_ids.size());
        block_output_ids.push_back(output_ids.back());
        block_output_ids.insert(block_output_ids.end(), draft_tokens.begin(), draft_tokens.end());

        const size_t verify_len = block_output_ids.size();

        // Save linear states before verification
        auto saved_linear = save_linear_states();

        // Verify
        m_target_request.set_tensor("input_ids", make_ids_tensor(block_output_ids));
        m_target_request.set_tensor("attention_mask", make_attention_mask(target_hidden_len + verify_len));
        m_target_request.set_tensor("position_ids", make_mrope_position_ids(target_hidden_len, verify_len));
        m_target_request.set_tensor("beam_idx", beam_idx);
        m_target_request.infer();

        logits = m_target_request.get_tensor("logits");
        auto posterior = argmax_logits_slice(logits, 0, verify_len);

        // Acceptance
        size_t accepted = 0;
        for (size_t i = 0; i < draft_tokens.size(); ++i) {
            if (draft_tokens[i] == posterior[i]) ++accepted;
            else break;
        }
        int64_t posterior_next = posterior[accepted];
        const size_t num_accepted = accepted + 1;

        // Restore linear states, trim ALL verify tokens from KV, reprocess accepted only
        restore_linear_states(saved_linear);
        m_target_kv_state.num_tokens_to_trim = verify_len;
        ov::genai::utils::trim_kv_cache(m_target_request, m_target_kv_state, std::nullopt);
        m_target_kv_state.num_tokens_to_trim = 0;

        {
            std::vector<int64_t> accepted_block(block_output_ids.begin(),
                                                block_output_ids.begin() + static_cast<ptrdiff_t>(num_accepted));
            m_target_request.set_tensor("input_ids", make_ids_tensor(accepted_block));
            m_target_request.set_tensor("attention_mask", make_attention_mask(target_hidden_len + num_accepted));
            m_target_request.set_tensor("position_ids", make_mrope_position_ids(target_hidden_len, num_accepted));
            m_target_request.set_tensor("beam_idx", beam_idx);
            m_target_request.infer();
        }

        // Update hidden storage
        target_hidden_block = ensure_f32_copy(m_target_request.get_tensor("target_hidden"));
        if (num_accepted > 0 && target_hidden_len + num_accepted <= max_length + m_block_size) {
            ov::Tensor src_slice(target_hidden_block, {0, 0, 0}, {1, num_accepted, m_hidden_dim});
            ov::Tensor dst_slice(target_hidden_storage,
                                 {0, target_hidden_len, 0},
                                 {1, target_hidden_len + num_accepted, m_hidden_dim});
            src_slice.copy_to(dst_slice);
            target_hidden_len += num_accepted;
        }

        // Push accepted tokens
        for (size_t i = 0; i < accepted && output_ids.size() < max_length; ++i) {
            output_ids.push_back(draft_tokens[i]);
            if (stream_token(draft_tokens[i]) || should_stop_on_token(draft_tokens[i])) {
                stopped = true;
                break;
            }
        }
        if (stopped) break;
        if (output_ids.size() >= max_length) break;

        next_token = posterior_next;
        output_ids.push_back(next_token);
        if (stream_token(next_token) || should_stop_on_token(next_token)) break;
    }

    if (streamer_ptr) streamer_ptr->end();

    // Build results
    EncodedResults results;
    results.tokens.resize(1);
    results.tokens[0].assign(output_ids.begin() + prompt_len, output_ids.end());
    results.scores = {0.0f};
    return results;
}

void StatefulDFlashPipeline::start_chat(const std::string& system_message) {
    (void)system_message;
}

void StatefulDFlashPipeline::finish_chat() {
    m_target_request.reset_state();
}

}  // namespace genai
}  // namespace ov
