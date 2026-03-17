// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <random>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "openvino/genai/tokenizer.hpp"
#include "loaders/model_config.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"
#include "utils.hpp"

#include "modeling/models/dflash_draft/dflash_draft.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct StageStats {
    double total_ms = 0.0;
    double max_ms = 0.0;
    size_t count = 0;

    void add(double ms) {
        total_ms += ms;
        max_ms = std::max(max_ms, ms);
        ++count;
    }

    double avg() const {
        return count ? total_ms / static_cast<double>(count) : 0.0;
    }
};

struct PerfStats {
    StageStats prefill_wall;
    StageStats target_ctx_wall;
    StageStats embed_wall;
    StageStats draft_wall;
    StageStats lm_head_wall;
    StageStats verify_wall;
    StageStats other_wall;
    StageStats prep_wall;
    StageStats postproc_wall;
    StageStats kv_trim_wall;
    StageStats hidden_append_wall;
    StageStats set_tensor_wall;
    StageStats get_tensor_wall;
    StageStats argmax_wall;
    StageStats make_tensor_wall;

    size_t draft_steps = 0;
    size_t accepted_tokens = 0;
    size_t generated_tokens = 0;
    double ttft_ms = 0.0;
    double total_generate_ms = 0.0;
    std::vector<size_t> accepted_per_step;
};

struct TargetBaselineStats {
    StageStats prefill_wall;
    StageStats decode_wall;
    size_t generated_tokens = 0;
    double ttft_ms = 0.0;
    double total_generate_ms = 0.0;
};

double duration_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_stage_stats(const std::string& name, const StageStats& wall) {
    std::cout << "  " << name << ": wall " << wall.avg() << " ms (max " << wall.max_ms << ", "
              << wall.count << " runs)" << std::endl;
}

std::vector<int64_t> tensor_to_ids(const ov::Tensor& ids_tensor) {
    const auto shape = ids_tensor.get_shape();
    if (shape.size() != 2 || shape[0] != 1) {
        throw std::runtime_error("input_ids tensor must have shape [1, S]");
    }
    const size_t seq_len = shape[1];
    const auto* data = ids_tensor.data<const int64_t>();
    return std::vector<int64_t>(data, data + seq_len);
}

std::string preview_vec(const std::vector<int64_t>& v, size_t n) {
    std::ostringstream oss;
    oss << "[";
    const size_t limit = std::min(n, v.size());
    for (size_t i = 0; i < limit; ++i) {
        if (i) oss << ",";
        oss << v[i];
    }
    if (v.size() > limit) oss << "...";
    oss << "]";
    return oss.str();
}

ov::Tensor make_ids_tensor(const std::vector<int64_t>& ids) {
    ov::Tensor tensor(ov::element::i64, {1, ids.size()});
    std::memcpy(tensor.data(), ids.data(), ids.size() * sizeof(int64_t));
    return tensor;
}

ov::Tensor make_attention_mask(size_t seq_len) {
    ov::Tensor mask(ov::element::i64, {1, seq_len});
    auto* data = mask.data<int64_t>();
    for (size_t i = 0; i < seq_len; ++i) {
        data[i] = 1;
    }
    return mask;
}

// Make 3D mRoPE position_ids [3, 1, seq_len] for text-only.
// For text without vision, all three dimensions have the same values.
ov::Tensor make_mrope_position_ids(size_t start, size_t count) {
    ov::Tensor ids(ov::element::i64, {3, 1, count});
    auto* data = ids.data<int64_t>();
    for (size_t dim = 0; dim < 3; ++dim) {
        for (size_t i = 0; i < count; ++i) {
            data[dim * count + i] = static_cast<int64_t>(start + i);
        }
    }
    return ids;
}

// Make 3D mRoPE position_ids for overlap (target_hidden context + draft block).
ov::Tensor make_mrope_position_ids_with_overlap(size_t context_len, size_t draft_len) {
    const size_t total = context_len + draft_len;
    ov::Tensor ids(ov::element::i64, {3, 1, total});
    auto* data = ids.data<int64_t>();
    for (size_t dim = 0; dim < 3; ++dim) {
        const size_t offset = dim * total;
        for (size_t i = 0; i < context_len; ++i) {
            data[offset + i] = static_cast<int64_t>(i);
        }
        for (size_t i = context_len; i < total; ++i) {
            data[offset + i] = static_cast<int64_t>(context_len - 1 + (i - context_len));
        }
    }
    return ids;
}

ov::Tensor ensure_f32_copy(const ov::Tensor& t) {
    ov::Tensor out(ov::element::f32, t.get_shape());
    t.copy_to(out);
    return out;
}

template <typename T>
int64_t argmax_row(const T* data, size_t vocab) {
    T max_val = data[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return static_cast<int64_t>(max_idx);
}

std::vector<int64_t> argmax_logits_slice(const ov::Tensor& logits, size_t start, size_t count) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("logits tensor must have shape [1, S, V]");
    }
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    if (start + count > seq_len) {
        throw std::runtime_error("logits slice out of range");
    }

    std::vector<int64_t> tokens;
    tokens.reserve(count);

    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<const ov::float16>();
        for (size_t i = 0; i < count; ++i)
            tokens.push_back(argmax_row(data + (start + i) * vocab, vocab));
        return tokens;
    }
    if (logits.get_element_type() == ov::element::bf16) {
        const auto* data = logits.data<const ov::bfloat16>();
        for (size_t i = 0; i < count; ++i)
            tokens.push_back(argmax_row(data + (start + i) * vocab, vocab));
        return tokens;
    }
    if (logits.get_element_type() == ov::element::f32) {
        const auto* data = logits.data<const float>();
        for (size_t i = 0; i < count; ++i)
            tokens.push_back(argmax_row(data + (start + i) * vocab, vocab));
        return tokens;
    }
    throw std::runtime_error("Unsupported logits dtype");
}

int64_t argmax_last_token(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("logits tensor must have shape [1, S, V]");
    }
    const size_t seq_len = shape[1];
    return argmax_logits_slice(logits, seq_len - 1, 1).front();
}

int64_t resolve_mask_token_id(ov::genai::Tokenizer tokenizer) {
    const auto vocab = tokenizer.get_vocab();
    const auto it = vocab.find("<|MASK|>");
    if (it != vocab.end()) {
        return it->second;
    }
    auto try_single_token = [&](const std::string& token) -> std::optional<int64_t> {
        try {
            auto encoded = tokenizer.encode(token, {ov::genai::add_special_tokens(false)}).input_ids;
            if (encoded.get_shape() == ov::Shape{1, 1}) {
                return encoded.data<const int64_t>()[0];
            }
        } catch (...) {
        }
        return std::nullopt;
    };
    if (auto fim_pad = try_single_token("<|fim_pad|>"))
        return *fim_pad;
    if (auto vision_pad = try_single_token("<|vision_pad|>"))
        return *vision_pad;
    const int64_t pad_id = tokenizer.get_pad_token_id();
    if (pad_id != -1) return pad_id;
    const int64_t eos_id = tokenizer.get_eos_token_id();
    if (eos_id != -1) return eos_id;
    throw std::runtime_error("Tokenizer has no suitable mask/pad/eos token to use as mask placeholder");
}

ov::Tensor make_beam_idx(size_t batch) {
    ov::Tensor beam_idx(ov::element::i32, {batch});
    auto* data = beam_idx.data<int32_t>();
    for (size_t i = 0; i < batch; ++i) {
        data[i] = static_cast<int32_t>(i);
    }
    return beam_idx;
}

// Save linear attention (GatedDeltaNet) states — these cannot be "trimmed"
// like KV cache entries, so we must save/restore them around verification.
std::vector<std::pair<std::string, ov::Tensor>> save_linear_states(ov::InferRequest& req) {
    std::vector<std::pair<std::string, ov::Tensor>> saved;
    for (auto& state : req.query_state()) {
        const auto& name = state.get_name();
        if (name.find("linear_states.") != std::string::npos) {
            auto src = state.get_state();
            ov::Tensor copy(src.get_element_type(), src.get_shape());
            src.copy_to(copy);
            saved.emplace_back(name, std::move(copy));
        }
    }
    return saved;
}

void restore_linear_states(ov::InferRequest& req,
                           const std::vector<std::pair<std::string, ov::Tensor>>& saved) {
    for (auto& state : req.query_state()) {
        const auto& name = state.get_name();
        for (const auto& [saved_name, saved_tensor] : saved) {
            if (name == saved_name) {
                state.set_state(saved_tensor);
                break;
            }
        }
    }
}

struct TargetBaselineResult {
    TargetBaselineStats stats;
    std::vector<int64_t> output_ids;
};

TargetBaselineResult run_target_baseline(ov::InferRequest& target_request,
                                         const ov::Tensor& beam_idx,
                                         const std::vector<int64_t>& prompt_ids,
                                         int max_new_tokens,
                                         int64_t eos_token_id) {
    TargetBaselineResult result;
    result.output_ids = prompt_ids;
    const size_t prompt_len = prompt_ids.size();
    const size_t max_length = prompt_len + static_cast<size_t>(max_new_tokens);

    auto prefill_start = Clock::now();
    target_request.set_tensor("input_ids", make_ids_tensor(result.output_ids));
    target_request.set_tensor("attention_mask", make_attention_mask(result.output_ids.size()));
    target_request.set_tensor("position_ids", make_mrope_position_ids(0, result.output_ids.size()));
    target_request.set_tensor("beam_idx", beam_idx);
    target_request.infer();
    auto logits = target_request.get_tensor("logits");
    auto prefill_end = Clock::now();
    result.stats.prefill_wall.add(duration_ms(prefill_start, prefill_end));

    int64_t next_token = argmax_last_token(logits);
    result.stats.ttft_ms = duration_ms(prefill_start, prefill_end);
    result.output_ids.push_back(next_token);
    const auto generation_start = Clock::now();
    while (result.output_ids.size() < max_length) {
        if (next_token == eos_token_id) break;
        auto decode_start = Clock::now();
        const size_t pos = result.output_ids.size() - 1;
        target_request.set_tensor("input_ids", make_ids_tensor({next_token}));
        target_request.set_tensor("attention_mask", make_attention_mask(1));
        target_request.set_tensor("position_ids", make_mrope_position_ids(pos, 1));
        target_request.set_tensor("beam_idx", beam_idx);
        target_request.infer();
        auto decode_end = Clock::now();
        result.stats.decode_wall.add(duration_ms(decode_start, decode_end));

        logits = target_request.get_tensor("logits");
        next_token = argmax_last_token(logits);
        result.output_ids.push_back(next_token);
    }

    const auto generation_end = Clock::now();
    result.stats.total_generate_ms = duration_ms(generation_start, generation_end);
    result.stats.generated_tokens = result.output_ids.size() - prompt_len;
    return result;
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <TARGET_MODEL_DIR> <DRAFT_MODEL_DIR> [PROMPT] [DEVICE] [MAX_NEW_TOKENS] [BLOCK_SIZE]\n"
                  << "\n"
                  << "  TARGET_MODEL_DIR  Path to Qwen3.5-4B HF model directory\n"
                  << "  DRAFT_MODEL_DIR   Path to DFlash draft model directory\n"
                  << "  PROMPT            Input prompt (default: 'Tell me a short story about a robot.')\n"
                  << "  DEVICE            OpenVINO device (default: CPU)\n"
                  << "  MAX_NEW_TOKENS    Max tokens to generate (default: 64)\n"
                  << "  BLOCK_SIZE        DFlash block size override (default: from config)\n";
        return 1;
    }

    const std::filesystem::path target_dir = argv[1];
    const std::filesystem::path draft_dir = argv[2];
    const std::string prompt = (argc > 3) ? argv[3] : "Tell me a short story about a robot.";
    const std::string device = (argc > 4) ? argv[4] : "CPU";
    const int max_new_tokens = (argc > 5) ? std::stoi(argv[5]) : 500;
    const int block_size_arg = (argc > 6) ? std::stoi(argv[6]) : 0;

    std::cout << "[Qwen3.5 DFlash Sample]" << std::endl;
    std::cout << "  Target model: " << target_dir << std::endl;
    std::cout << "  Draft model:  " << draft_dir << std::endl;
    std::cout << "  Device:       " << device << std::endl;
    std::cout << "  Max tokens:   " << max_new_tokens << std::endl;
    std::cout << "  Prompt:       " << prompt << std::endl;

    // Load target model config (Qwen3.5)
    auto target_qwen35_cfg = ov::genai::modeling::models::Qwen3_5Config::from_json_file(target_dir);
    // Also load the generic ModelConfig for DFlash-related fields
    auto draft_cfg = ov::genai::loaders::ModelConfig::from_hf_json(draft_dir / "config.json");
    auto target_generic_cfg = ov::genai::loaders::ModelConfig::from_hf_json(target_dir / "config.json");

    // Set up DFlash draft config
    ov::genai::modeling::models::DFlashDraftConfig dflash_cfg;
    dflash_cfg.hidden_size = draft_cfg.hidden_size;
    dflash_cfg.intermediate_size = draft_cfg.intermediate_size;
    dflash_cfg.num_hidden_layers = draft_cfg.num_hidden_layers;
    dflash_cfg.num_target_layers = (draft_cfg.num_target_layers > 0)
                                       ? draft_cfg.num_target_layers
                                       : target_qwen35_cfg.text.num_hidden_layers;
    dflash_cfg.num_attention_heads = draft_cfg.num_attention_heads;
    dflash_cfg.num_key_value_heads = draft_cfg.num_key_value_heads > 0
                                         ? draft_cfg.num_key_value_heads
                                         : draft_cfg.num_attention_heads;
    dflash_cfg.head_dim = draft_cfg.head_dim > 0
                              ? draft_cfg.head_dim
                              : (draft_cfg.hidden_size / draft_cfg.num_attention_heads);
    dflash_cfg.block_size = block_size_arg > 0 ? block_size_arg : draft_cfg.block_size;
    dflash_cfg.rms_norm_eps = draft_cfg.rms_norm_eps;
    dflash_cfg.rope_theta = draft_cfg.rope_theta;
    dflash_cfg.hidden_act = draft_cfg.hidden_act;
    dflash_cfg.attention_bias = draft_cfg.attention_bias;

    if (dflash_cfg.block_size <= 0) {
        dflash_cfg.block_size = 16;
    }
    if (dflash_cfg.block_size < 2) {
        throw std::runtime_error("block_size must be >= 2 for DFlash decoding");
    }

    const auto target_layer_ids = ov::genai::modeling::models::build_target_layer_ids(
        dflash_cfg.num_target_layers, dflash_cfg.num_hidden_layers);
    std::cout << "dflash_cfg.block_size is " << dflash_cfg.block_size << std::endl;
    std::cout << "dflash_cfg.num_target_layers is " << dflash_cfg.num_target_layers << std::endl;
    std::cout << "target_layer_ids: ";
    for (auto id : target_layer_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // Load weights
    auto target_data = ov::genai::safetensors::load_safetensors(target_dir);
    ov::genai::safetensors::SafetensorsWeightSource target_source(std::move(target_data));
    ov::genai::safetensors::SafetensorsWeightFinalizer target_finalizer;

    auto draft_data = ov::genai::safetensors::load_safetensors(draft_dir);
    ov::genai::safetensors::SafetensorsWeightSource draft_source(std::move(draft_data));
    ov::genai::safetensors::SafetensorsWeightFinalizer draft_finalizer;

    PerfStats perf;
    double dflash_throughput = 0.0;

    // Build models
    std::cout << "[Building target model (Qwen3.5 DFlash target)...]" << std::endl;
    auto target_model = ov::genai::modeling::models::create_qwen3_5_dflash_target_model(
        target_qwen35_cfg, target_layer_ids, target_source, target_finalizer);

    std::cout << "[Building draft model...]" << std::endl;
    const auto draft_compute_type = ov::element::f32;
    auto draft_model = ov::genai::modeling::models::create_dflash_draft_model(
        dflash_cfg, draft_source, draft_finalizer, draft_compute_type);

    std::cout << "[Building embedding model...]" << std::endl;
    auto embed_model = ov::genai::modeling::models::create_qwen3_5_embedding_model(
        target_qwen35_cfg, target_source, target_finalizer);

    std::cout << "[Building lm_head model...]" << std::endl;
    auto lm_head_model = ov::genai::modeling::models::create_qwen3_5_lm_head_model(
        target_qwen35_cfg, target_source, target_finalizer, ov::element::f32);

    // Compile models
    ov::Core core;
    ov::AnyMap compile_cfg = {
        {ov::hint::inference_precision.name(), ov::element::f16},
        {ov::hint::kv_cache_precision.name(), ov::element::f16},
        {ov::hint::activations_scale_factor.name(), 8.0f}
    };

    std::cout << "[compile_cfg] ";
    for (const auto& kv : compile_cfg) {
        std::cout << kv.first << "=";
        if (kv.second.is<ov::element::Type>()) {
            std::cout << kv.second.as<ov::element::Type>().get_type_name();
        } else if (kv.second.is<float>()) {
            std::cout << kv.second.as<float>();
        } else {
            std::cout << "(unknown)";
        }
        std::cout << " ";
    }
    std::cout << std::endl;

    // Tokenizer
    ov::genai::Tokenizer tokenizer(target_dir);
    const int64_t mask_token_id = resolve_mask_token_id(tokenizer);
    const int64_t eos_token_id = tokenizer.get_eos_token_id();
    std::cout << "[DEBUG] mask_token_id=" << mask_token_id << " eos_token_id=" << eos_token_id << std::endl;

    // Apply chat template if available
    std::string formatted_prompt = prompt;
    bool add_special_tokens = true;
    try {
        if (!tokenizer.get_chat_template().empty()) {
            ov::genai::ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            formatted_prompt = tokenizer.apply_chat_template(history, true);
            add_special_tokens = false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: chat template apply failed: " << e.what() << ", using raw prompt" << std::endl;
    }

    auto encoded = tokenizer.encode(formatted_prompt, {ov::genai::add_special_tokens(add_special_tokens)});
    const std::vector<int64_t> prompt_ids = tensor_to_ids(encoded.input_ids);
    std::vector<int64_t> output_ids = prompt_ids;
    const size_t prompt_len = output_ids.size();
    const size_t max_length = prompt_len + static_cast<size_t>(max_new_tokens);

    std::cout << "[INFO] prompt_len=" << prompt_len << " max_length=" << max_length << std::endl;

    // Compile
    std::cout << "[Compiling models on " << device << "...]" << std::endl;
    auto compiled_target = core.compile_model(target_model, device, compile_cfg);
    auto compiled_embed = core.compile_model(embed_model, device, compile_cfg);
    auto compiled_lm_head = core.compile_model(lm_head_model, device, compile_cfg);
    auto compiled_draft = core.compile_model(draft_model, device, compile_cfg);

    auto target_request = compiled_target.create_infer_request();
    auto embed_request = compiled_embed.create_infer_request();
    auto lm_head_request = compiled_lm_head.create_infer_request();
    auto draft_request = compiled_draft.create_infer_request();

    target_request.reset_state();
    ov::genai::utils::KVCacheState target_kv_state;
    const auto kv_pos = ov::genai::utils::get_kv_axes_pos(compiled_target.get_runtime_model());
    target_kv_state.seq_length_axis = kv_pos.seq_len;
    const auto beam_idx = make_beam_idx(1);

    std::cout << "---------------START INFERENCE --------------------" << std::endl;

    // Prefill: run target on prompt to get first token.
    auto prefill_start = Clock::now();
    target_request.set_tensor("input_ids", make_ids_tensor(output_ids));
    target_request.set_tensor("attention_mask", make_attention_mask(output_ids.size()));
    target_request.set_tensor("position_ids", make_mrope_position_ids(0, output_ids.size()));
    target_request.set_tensor("beam_idx", beam_idx);
    target_request.infer();
    auto logits = target_request.get_tensor("logits");

    target_kv_state.add_inputs(make_ids_tensor(output_ids));
    ov::Tensor target_hidden_block = ensure_f32_copy(target_request.get_tensor("target_hidden"));
    const size_t hidden_dim = target_hidden_block.get_shape()[2];
    ov::Tensor target_hidden_storage(ov::element::f32, {1, max_length, hidden_dim});
    ov::Tensor target_hidden_init(target_hidden_storage, {0, 0, 0}, {1, prompt_len, hidden_dim});
    target_hidden_block.copy_to(target_hidden_init);
    size_t target_hidden_len = prompt_len;

    int64_t next_token = argmax_last_token(logits);
    auto prefill_end = Clock::now();
    perf.ttft_ms = duration_ms(prefill_start, prefill_end);
    output_ids.push_back(next_token);

    if (next_token == eos_token_id) {
        std::cout << "\n[Early Stop] EOS token generated after prefill" << std::endl;
    }

    const double prefill_ms = duration_ms(prefill_start, prefill_end);
    perf.prefill_wall.add(prefill_ms);

    // Print prompt and first token for streaming output
    std::cout << "\n[Streaming Output]\n";
    auto prompt_text = tokenizer.decode(prompt_ids, {ov::genai::skip_special_tokens(true)});
    std::cout << prompt_text;
    auto first_token_text = tokenizer.decode({next_token}, {ov::genai::skip_special_tokens(true)});
    std::cout << first_token_text << std::flush;

    const auto generation_start = Clock::now();

    bool stopped_by_eos = false;
    while (output_ids.size() < max_length) {
        if (next_token == eos_token_id) {
            stopped_by_eos = true;
            std::cout << "\n[Early Stop] EOS token detected at position " << output_ids.size() << std::endl;
            break;
        }
        const auto step_start = Clock::now();
        double step_tracked_ms = 0.0;

        // Build draft block inputs: [last_token, MASK, MASK, ...]
        std::vector<int64_t> block_ids(static_cast<size_t>(dflash_cfg.block_size), mask_token_id);
        block_ids[0] = output_ids.back();

        // Embed block tokens
        auto make_embed_start = Clock::now();
        auto embed_ids = make_ids_tensor(block_ids);
        auto make_embed_end = Clock::now();
        perf.make_tensor_wall.add(duration_ms(make_embed_start, make_embed_end));

        auto set_embed_start = Clock::now();
        embed_request.set_tensor("input_ids", embed_ids);
        auto set_embed_end = Clock::now();
        perf.set_tensor_wall.add(duration_ms(set_embed_start, set_embed_end));

        auto embed_start = Clock::now();
        embed_request.infer();
        auto embed_end = Clock::now();
        const double embed_ms = duration_ms(embed_start, embed_end);
        perf.embed_wall.add(embed_ms);
        step_tracked_ms += embed_ms;

        auto get_embed_start = Clock::now();
        auto noise_embedding = embed_request.get_tensor("embeddings");
        auto get_embed_end = Clock::now();
        perf.get_tensor_wall.add(duration_ms(get_embed_start, get_embed_end));

        // Build position_ids for draft model (mRoPE 3D is NOT used by draft;
        // the draft model uses 2D position_ids like original DFlash)
        auto make_pos_start = Clock::now();
        const auto position_ids = make_mrope_position_ids_with_overlap(target_hidden_len, block_ids.size());
        auto make_pos_end = Clock::now();
        perf.make_tensor_wall.add(duration_ms(make_pos_start, make_pos_end));

        // Draft model uses 2D position_ids — build a 2D version
        const size_t total_pos = target_hidden_len + block_ids.size();
        ov::Tensor draft_position_ids(ov::element::i64, {1, total_pos});
        {
            auto* pos_data = draft_position_ids.data<int64_t>();
            for (size_t i = 0; i < target_hidden_len; ++i) {
                pos_data[i] = static_cast<int64_t>(i);
            }
            for (size_t i = target_hidden_len; i < total_pos; ++i) {
                pos_data[i] = static_cast<int64_t>(target_hidden_len - 1 + (i - target_hidden_len));
            }
        }

        // Run draft model
        auto draft_start = Clock::now();
        ov::Tensor target_hidden_view(target_hidden_storage, {0, 0, 0}, {1, target_hidden_len, hidden_dim});
        auto set_draft_start = Clock::now();
        draft_request.set_tensor("target_hidden", target_hidden_view);
        draft_request.set_tensor("noise_embedding", noise_embedding);
        draft_request.set_tensor("position_ids", draft_position_ids);
        auto set_draft_end = Clock::now();
        perf.set_tensor_wall.add(duration_ms(set_draft_start, set_draft_end));

        draft_request.infer();
        auto draft_end = Clock::now();
        const double draft_ms = duration_ms(draft_start, draft_end);
        perf.draft_wall.add(draft_ms);

        auto get_draft_start = Clock::now();
        auto draft_hidden = draft_request.get_tensor("draft_hidden");
        auto get_draft_end = Clock::now();
        perf.get_tensor_wall.add(duration_ms(get_draft_start, get_draft_end));

        // Convert draft_hidden dtype if needed for lm_head
        const auto& lm_head_port = compiled_lm_head.input("hidden_states");
        if (draft_hidden.get_element_type() != lm_head_port.get_element_type()) {
            auto convert_start = Clock::now();
            ov::Tensor converted(lm_head_port.get_element_type(), draft_hidden.get_shape());
            draft_hidden.copy_to(converted);
            draft_hidden = converted;
            auto convert_end = Clock::now();
            perf.other_wall.add(duration_ms(convert_start, convert_end));
        }
        auto set_lm_start = Clock::now();
        lm_head_request.set_tensor("hidden_states", draft_hidden);
        auto set_lm_end = Clock::now();
        perf.set_tensor_wall.add(duration_ms(set_lm_start, set_lm_end));

        // LM head: draft_hidden -> draft_logits
        auto lm_head_start = Clock::now();
        lm_head_request.infer();
        auto lm_head_end = Clock::now();
        const double lm_head_ms = duration_ms(lm_head_start, lm_head_end);
        perf.lm_head_wall.add(lm_head_ms);
        step_tracked_ms += lm_head_ms;

        auto get_lm_start = Clock::now();
        auto draft_logits = lm_head_request.get_tensor("logits");
        auto get_lm_end = Clock::now();
        perf.get_tensor_wall.add(duration_ms(get_lm_start, get_lm_end));

        // Argmax draft tokens (skip position 0 which is the last accepted token)
        const size_t draft_len = block_ids.size() - 1;
        auto argmax_draft_start = Clock::now();
        auto draft_tokens = argmax_logits_slice(draft_logits, 1, draft_len);
        auto argmax_draft_end = Clock::now();
        perf.argmax_wall.add(duration_ms(argmax_draft_start, argmax_draft_end));

        // Batch verification: construct block_output_ids
        std::vector<int64_t> block_output_ids;
        block_output_ids.reserve(block_ids.size());
        block_output_ids.push_back(output_ids.back());  // Last accepted token
        block_output_ids.insert(block_output_ids.end(), draft_tokens.begin(), draft_tokens.end());

        const size_t verify_len = block_output_ids.size();
        const size_t kv_len = target_hidden_len + verify_len;

        // Qwen3.5 has linear attention (GatedDeltaNet/Mamba) layers whose
        // recurrent state cannot be "trimmed" like KV cache.  We must save
        // the linear states before verification, then restore them and
        // reprocess only the accepted tokens so that both KV cache and
        // linear states are correct.
        auto saved_linear = save_linear_states(target_request);

        auto verify_start = Clock::now();
        target_request.set_tensor("input_ids", make_ids_tensor(block_output_ids));
        target_request.set_tensor("attention_mask", make_attention_mask(kv_len));
        target_request.set_tensor("position_ids", make_mrope_position_ids(target_hidden_len, verify_len));
        target_request.set_tensor("beam_idx", beam_idx);
        target_request.infer();
        auto verify_end = Clock::now();
        const double verify_ms = duration_ms(verify_start, verify_end);
        perf.verify_wall.add(verify_ms);
        step_tracked_ms += verify_ms;

        logits = target_request.get_tensor("logits");
        auto posterior_tokens = argmax_logits_slice(logits, 0, verify_len);

        // Find acceptance length
        size_t accepted = 0;
        for (size_t i = 0; i < draft_tokens.size(); ++i) {
            if (draft_tokens[i] == posterior_tokens[i]) {
                ++accepted;
            } else {
                break;
            }
        }

        int64_t posterior_next = posterior_tokens[accepted];
        const size_t num_accepted_verify = accepted + 1;

        // Restore linear attention states to pre-verification values.
        restore_linear_states(target_request, saved_linear);

        // Trim ALL verification tokens from KV cache (full-attention layers).
        target_kv_state.num_tokens_to_trim = verify_len;
        ov::genai::utils::trim_kv_cache(target_request, target_kv_state, std::nullopt);
        target_kv_state.num_tokens_to_trim = 0;

        // Reprocess only accepted tokens.  This correctly advances both
        // the KV cache (for full-attention layers) and the linear attention
        // recurrent/conv states by exactly the accepted token count.
        {
            std::vector<int64_t> accepted_block(block_output_ids.begin(),
                                                block_output_ids.begin() + static_cast<ptrdiff_t>(num_accepted_verify));
            target_request.set_tensor("input_ids", make_ids_tensor(accepted_block));
            target_request.set_tensor("attention_mask", make_attention_mask(target_hidden_len + num_accepted_verify));
            target_request.set_tensor("position_ids", make_mrope_position_ids(target_hidden_len, num_accepted_verify));
            target_request.set_tensor("beam_idx", beam_idx);
            target_request.infer();
        }

        // Update target_hidden storage from the reprocessed (clean) inference.
        target_hidden_block = ensure_f32_copy(target_request.get_tensor("target_hidden"));
        const size_t available_space = max_length - target_hidden_len;
        const size_t num_to_append = std::min(num_accepted_verify, available_space);
        if (num_to_append > 0) {
            ov::Tensor accepted_hidden_slice(target_hidden_block, {0, 0, 0}, {1, num_to_append, hidden_dim});
            ov::Tensor hidden_append_dst(target_hidden_storage,
                                         {0, target_hidden_len, 0},
                                         {1, target_hidden_len + num_to_append, hidden_dim});
            accepted_hidden_slice.copy_to(hidden_append_dst);
            target_hidden_len += num_to_append;
        }

        auto postproc_start = Clock::now();

        const size_t before_accept = output_ids.size();
        std::vector<int64_t> newly_accepted;
        for (size_t i = 0; i < accepted && output_ids.size() < max_length; ++i) {
            output_ids.push_back(draft_tokens[i]);
            newly_accepted.push_back(draft_tokens[i]);
        }
        // Stream print accepted draft tokens
        if (!newly_accepted.empty()) {
            auto accepted_text = tokenizer.decode(newly_accepted, {ov::genai::skip_special_tokens(true)});
            std::cout << accepted_text << std::flush;
        }
        const size_t accepted_pushed = output_ids.size() - before_accept;
        ++perf.draft_steps;
        perf.accepted_tokens += accepted_pushed + 1;
        perf.accepted_per_step.push_back(accepted_pushed + 1);
        auto postproc_end = Clock::now();
        perf.postproc_wall.add(duration_ms(postproc_start, postproc_end));

        if (output_ids.size() >= max_length) break;

        next_token = posterior_next;
        output_ids.push_back(next_token);
        // Stream print posterior token
        auto posterior_text = tokenizer.decode({posterior_next}, {ov::genai::skip_special_tokens(true)});
        std::cout << posterior_text << std::flush;

        // Check if posterior token is EOS
        if (posterior_next == eos_token_id) {
            stopped_by_eos = true;
            std::cout << "\n[Early Stop] EOS token in posterior at position " << output_ids.size() << std::endl;
            break;
        }

        const double step_ms = duration_ms(step_start, Clock::now());
        const double other_ms = std::max(0.0, step_ms - step_tracked_ms);
        perf.other_wall.add(other_ms);
    }

    const auto generation_end = Clock::now();
    perf.total_generate_ms = duration_ms(generation_start, generation_end);
    perf.generated_tokens = output_ids.size() - prompt_len;

    std::cout << "\n\n[Generation Complete]" << std::endl;
    std::cout << "[Stop Reason] " << (stopped_by_eos ? "EOS token detected" : "Max length reached") << "\n" << std::endl;

    const size_t tokens_after_first = perf.generated_tokens > 0 ? perf.generated_tokens - 1 : 0;
    const double tpot_ms = tokens_after_first > 0
                               ? (perf.total_generate_ms - perf.ttft_ms) / static_cast<double>(tokens_after_first)
                               : 0.0;
    dflash_throughput = perf.total_generate_ms > 0
                            ? (static_cast<double>(perf.generated_tokens) * 1000.0) / perf.total_generate_ms
                            : 0.0;
    const double avg_accept = perf.draft_steps > 0
                                  ? static_cast<double>(perf.accepted_tokens) / static_cast<double>(perf.draft_steps)
                                  : 0.0;
    const size_t target_generated = perf.generated_tokens > perf.accepted_tokens
                                        ? perf.generated_tokens - perf.accepted_tokens
                                        : 0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[Tokens] prompt=" << prompt_len << ", generated=" << perf.generated_tokens
              << ", draft_accepted=" << perf.accepted_tokens << ", target_only=" << target_generated
              << ", avg_accept_per_block=" << avg_accept << std::endl;
    std::cout << "[Latency] TTFT=" << perf.ttft_ms << " ms, TPOT=" << tpot_ms
              << " ms/token, total_generate=" << perf.total_generate_ms
              << " ms, throughput=" << dflash_throughput << " tokens/s" << std::endl;
    std::cout << "[Verify Stats] total verify calls=" << perf.verify_wall.count
              << ", draft_steps=" << perf.draft_steps << std::endl;
    std::cout << "[Stage timings] wall-clock (ms):" << std::endl;
    print_stage_stats("prefill target", perf.prefill_wall);
    print_stage_stats("target ctx", perf.target_ctx_wall);
    print_stage_stats("embed", perf.embed_wall);
    print_stage_stats("draft", perf.draft_wall);
    print_stage_stats("lm_head", perf.lm_head_wall);
    print_stage_stats("target verify", perf.verify_wall);
    print_stage_stats("prep", perf.prep_wall);
    print_stage_stats("postproc", perf.postproc_wall);
    print_stage_stats("kv trim", perf.kv_trim_wall);
    print_stage_stats("hidden append", perf.hidden_append_wall);
    print_stage_stats("set_tensor", perf.set_tensor_wall);
    print_stage_stats("get_tensor", perf.get_tensor_wall);
    print_stage_stats("argmax", perf.argmax_wall);
    print_stage_stats("make_tensor", perf.make_tensor_wall);
    print_stage_stats("other (untracked)", perf.other_wall);
    if (!perf.accepted_per_step.empty()) {
        std::cout << "[Draft acceptance per step] [";
        for (size_t i = 0; i < perf.accepted_per_step.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << perf.accepted_per_step[i];
        }
        std::cout << "]" << std::endl;
    }

    // =====================================================================
    // Baseline: run the standard Qwen3.5 text model for comparison
    // =====================================================================
    std::cout << std::endl << "=====Run Baseline Qwen3.5 model for comparison=====" << std::endl;

    auto baseline_model = ov::genai::modeling::models::create_qwen3_5_text_model(
        target_qwen35_cfg, target_source, target_finalizer, false, false);
    auto compiled_baseline = core.compile_model(baseline_model, device, compile_cfg);
    auto baseline_request = compiled_baseline.create_infer_request();
    const auto baseline_beam_idx = make_beam_idx(1);
    baseline_request.reset_state();
    auto baseline = run_target_baseline(baseline_request, baseline_beam_idx, prompt_ids, max_new_tokens, eos_token_id);
    const size_t baseline_tokens_after_first = baseline.stats.generated_tokens > 0 ? baseline.stats.generated_tokens - 1 : 0;
    const double baseline_tpot_ms = baseline_tokens_after_first > 0
                                        ? (baseline.stats.total_generate_ms - baseline.stats.ttft_ms)
                                              / static_cast<double>(baseline_tokens_after_first)
                                        : 0.0;
    const double baseline_throughput = baseline.stats.total_generate_ms > 0
                                           ? (static_cast<double>(baseline.stats.generated_tokens) * 1000.0)
                                                 / baseline.stats.total_generate_ms
                                           : 0.0;

    std::cout << "[Target-only] generated=" << baseline.stats.generated_tokens
              << ", TTFT=" << baseline.stats.ttft_ms << " ms, TPOT=" << baseline_tpot_ms
              << " ms/token, total_generate=" << baseline.stats.total_generate_ms
              << " ms, throughput=" << baseline_throughput << " tokens/s" << std::endl;
    std::cout << "[Target-only stage timings] wall-clock (ms):" << std::endl;
    print_stage_stats("prefill target", baseline.stats.prefill_wall);
    print_stage_stats("decode target", baseline.stats.decode_wall);
    auto text_original = tokenizer.decode(baseline.output_ids, {ov::genai::skip_special_tokens(true)});
    std::cout << text_original << std::endl;

    const size_t dflash_tokens_after_first = perf.generated_tokens > 0 ? perf.generated_tokens - 1 : 0;
    const double dflash_tpot_ms = dflash_tokens_after_first > 0
                                     ? (perf.total_generate_ms - perf.ttft_ms)
                                           / static_cast<double>(dflash_tokens_after_first)
                                     : 0.0;
    if (baseline_throughput > 0.0 && dflash_throughput > 0.0) {
        std::cout << "[Compare] dflash / baseline throughput ratio=" << (dflash_throughput / baseline_throughput) << std::endl;
    }
    if (baseline_tpot_ms > 0.0 && dflash_tpot_ms > 0.0) {
        std::cout << "[Compare] Decoding speedup: " << (baseline_tpot_ms / dflash_tpot_ms) << std::endl;
    }

    return 0;
} catch (const std::exception& ex) {
    std::cerr << "Qwen3.5 DFlash sample failed: " << ex.what() << std::endl;
    return 1;
}
