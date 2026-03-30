// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
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
#include "modeling/weights/quantization_config.hpp"
#include "utils.hpp"

#include "load_image.hpp"
#include "modeling/models/dflash_draft/dflash_draft.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
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
    StageStats draft_wall;
    StageStats verify_wall;
    StageStats other_wall;
    StageStats postproc_wall;
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

ov::Tensor make_state_update_mode_tensor(int32_t mode) {
    ov::Tensor t(ov::element::i32, {1});
    t.data<int32_t>()[0] = mode;
    return t;
}

bool deferred_state_commit_enabled_by_default() {
    const char* raw = std::getenv("OV_GENAI_DISABLE_DFLASH_DEFERRED_STATE_COMMIT");
    return !(raw && std::string(raw) == "1");
}

std::string build_vl_prompt(const std::string& user_prompt, int64_t image_tokens) {
    std::string prompt = "<|im_start|>user\n<|vision_start|>";
    for (int64_t i = 0; i < image_tokens; ++i)
        prompt += "<|image_pad|>";
    prompt += "<|vision_end|>\n";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

std::string resolve_pos_embed_name(ov::genai::safetensors::SafetensorsWeightSource& source) {
    for (const auto& name : {"model.visual.pos_embed.weight", "visual.pos_embed.weight", "pos_embed.weight"}) {
        if (source.has(name)) return name;
    }
    for (const auto& name : source.keys()) {
        if (name.find("pos_embed.weight") != std::string::npos) return name;
    }
    throw std::runtime_error("Failed to find pos_embed.weight in safetensors");
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

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <TARGET_MODEL_DIR> <DRAFT_MODEL_DIR> [PROMPT] [DEVICE] [MAX_NEW_TOKENS] [BLOCK_SIZE]"
                  << " [TARGET_QUANT] [DRAFT_QUANT] [IMAGE_PATH]\n"
                  << "\n"
                  << "  TARGET_MODEL_DIR  Path to Qwen3.5-4B HF model directory\n"
                  << "  DRAFT_MODEL_DIR   Path to DFlash draft model directory\n"
                  << "  PROMPT            Input prompt (default: 'Tell me a short story about a robot.')\n"
                  << "  DEVICE            OpenVINO device (default: CPU)\n"
                  << "  MAX_NEW_TOKENS    Max tokens to generate (default: 64)\n"
                  << "  BLOCK_SIZE        DFlash block size override (default: from config)\n"
                  << "  TARGET_QUANT      Target model quantization: FP16, INT4_ASYM, INT4_SYM (default: FP16 or env)\n"
                  << "  DRAFT_QUANT       Draft model quantization: FP16, INT4_ASYM, INT4_SYM (default: FP16)\n"
                  << "  IMAGE_PATH        Path to image file for VL mode (optional, text-only if omitted)\n";
        return 1;
    }

    const std::filesystem::path target_dir = argv[1];
    const std::filesystem::path draft_dir = argv[2];
    const std::string prompt = (argc > 3) ? argv[3] : "Tell me a short story about a robot.";
    const std::string device = (argc > 4) ? argv[4] : "CPU";
    const int max_new_tokens = (argc > 5) ? std::stoi(argv[5]) : 500;
    const int block_size_arg = (argc > 6) ? std::stoi(argv[6]) : 0;
    const std::string target_quant_arg = (argc > 7) ? argv[7] : "";
    const std::string draft_quant_arg = (argc > 8) ? argv[8] : "";
    const std::string image_path_arg = (argc > 9) ? argv[9] : "";

    const bool vl_mode = !image_path_arg.empty();

    std::cout << "[Qwen3.5 DFlash Sample" << (vl_mode ? " VL" : "") << "]" << std::endl;
    std::cout << "  Target model: " << target_dir << std::endl;
    std::cout << "  Draft model:  " << draft_dir << std::endl;
    if (vl_mode) std::cout << "  Image:        " << image_path_arg << std::endl;
    std::cout << "  Device:       " << device << std::endl;
    std::cout << "  Max tokens:   " << max_new_tokens << std::endl;

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

    // Parse quantization config: CLI arg takes priority, then env vars
    auto parse_quant_mode = [](const std::string& s) -> ov::genai::modeling::weights::QuantizationConfig::Mode {
        if (s == "INT4_ASYM") return ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
        if (s == "INT4_SYM")  return ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_SYM;
        if (s == "INT8_ASYM") return ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM;
        if (s == "INT8_SYM")  return ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_SYM;
        return ov::genai::modeling::weights::QuantizationConfig::Mode::NONE;
    };
    auto quant_mode_name = [](ov::genai::modeling::weights::QuantizationConfig::Mode m) -> const char* {
        switch (m) {
            case ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM: return "INT4_ASYM";
            case ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_SYM:  return "INT4_SYM";
            case ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM: return "INT8_ASYM";
            case ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_SYM:  return "INT8_SYM";
            default: return "NONE";
        }
    };

    // Target quantization: CLI arg > env var > FP16
    ov::genai::modeling::weights::QuantizationConfig target_quant_config;
    if (!target_quant_arg.empty() && target_quant_arg != "FP16") {
        target_quant_config.mode = parse_quant_mode(target_quant_arg);
        target_quant_config.group_size = 128;
        target_quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM;
    } else if (target_quant_arg.empty()) {
        target_quant_config = ov::genai::modeling::weights::parse_quantization_config_from_env();
    }
    std::cout << "[quant] target: " << (target_quant_config.enabled() ? quant_mode_name(target_quant_config.mode) : "FP16");
    if (target_quant_config.enabled()) std::cout << ", group_size=" << target_quant_config.group_size;
    std::cout << std::endl;

    // Draft quantization: CLI arg > FP16 (no env var fallback for draft)
    ov::genai::modeling::weights::QuantizationConfig draft_quant_config;
    if (!draft_quant_arg.empty() && draft_quant_arg != "FP16") {
        draft_quant_config.mode = parse_quant_mode(draft_quant_arg);
        draft_quant_config.group_size = 128;
        draft_quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM;
    }
    std::cout << "[quant] draft:  " << (draft_quant_config.enabled() ? quant_mode_name(draft_quant_config.mode) : "FP16");
    if (draft_quant_config.enabled()) std::cout << ", group_size=" << draft_quant_config.group_size;
    std::cout << std::endl;

    // Load weights
    PerfStats perf;
    double dflash_throughput = 0.0;

    // Build models inside a scope so that weight sources are freed after model building.
    // On iGPU (shared CPU/GPU memory), this recovers ~10+ GB of safetensors data.
    std::shared_ptr<ov::Model> target_model;
    std::shared_ptr<ov::Model> context_kv_model;
    std::shared_ptr<ov::Model> draft_step_model;
    std::shared_ptr<ov::Model> vision_model;
    ov::Tensor vl_pos_embed_weight;  // Extracted in scope for VL preprocessing later
    {
        auto target_data = ov::genai::safetensors::load_safetensors(target_dir);
        ov::genai::safetensors::SafetensorsWeightSource target_source(std::move(target_data));
        ov::genai::safetensors::SafetensorsWeightFinalizer target_finalizer(
            target_quant_config.enabled() ? target_quant_config
                                          : ov::genai::modeling::weights::QuantizationConfig{});

        auto draft_data = ov::genai::safetensors::load_safetensors(draft_dir);
        ov::genai::safetensors::SafetensorsWeightSource draft_source(std::move(draft_data));
        ov::genai::safetensors::SafetensorsWeightFinalizer draft_finalizer(
            draft_quant_config.enabled() ? draft_quant_config
                                         : ov::genai::modeling::weights::QuantizationConfig{});

        std::cout << "[Building target model (Qwen3.5 DFlash target" << (vl_mode ? " VL" : "") << ")...]" << std::endl;
        target_model = ov::genai::modeling::models::create_qwen3_5_dflash_target_model(
            target_qwen35_cfg, target_layer_ids, target_source, target_finalizer, dflash_cfg.block_size, vl_mode);

        if (vl_mode) {
            std::cout << "[Building vision model...]" << std::endl;
            ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer;
            vision_model = ov::genai::modeling::models::create_qwen3_5_vision_model(
                target_qwen35_cfg, target_source, vision_finalizer);

            // Extract pos_embed weight now — needed for VL preprocessing after scope exits
            const std::string pos_embed_name = resolve_pos_embed_name(target_source);
            vl_pos_embed_weight = target_source.get_tensor(pos_embed_name);
        }

        // Build split draft models: context_kv (runs once per verify) + draft_step (lightweight)
        std::cout << "[Building context KV model (fc+KV proj for context caching)...]" << std::endl;
        context_kv_model = ov::genai::modeling::models::create_qwen3_5_dflash_context_kv_model(
            dflash_cfg, draft_source, draft_finalizer);

        std::cout << "[Building draft step model (embed+attn+MLP+lm_head with cached KV)...]" << std::endl;
        draft_step_model = ov::genai::modeling::models::create_qwen3_5_dflash_step_model(
            target_qwen35_cfg, dflash_cfg, target_source, target_finalizer,
            draft_source, draft_finalizer);
    } // Weight sources (target_source, draft_source) and their safetensors data freed here.

    // Dump IR models for inspection (operator graph)
    {
        const std::string dump_dir = "ir_dump";
        std::filesystem::create_directories(dump_dir);
        std::cout << "[IR Dump] Saving target model IR..." << std::endl;
        ov::serialize(target_model, dump_dir + "/target_model.xml", dump_dir + "/target_model.bin");
        std::cout << "[IR Dump] Saving context KV model IR..." << std::endl;
        ov::serialize(context_kv_model, dump_dir + "/context_kv_model.xml", dump_dir + "/context_kv_model.bin");
        std::cout << "[IR Dump] Saving draft step model IR..." << std::endl;
        ov::serialize(draft_step_model, dump_dir + "/draft_step_model.xml", dump_dir + "/draft_step_model.bin");
        std::cout << "[IR Dump] Models saved to " << std::filesystem::absolute(dump_dir) << std::endl;

        // Print op summary for context KV model
        std::cout << "\n[Context KV Model - Op Summary]" << std::endl;
        std::map<std::string, int> op_counts;
        for (const auto& op : context_kv_model->get_ordered_ops()) {
            op_counts[op->get_type_info().name]++;
        }
        for (const auto& [name, count] : op_counts) {
            std::cout << "  " << name << ": " << count << std::endl;
        }
        std::cout << "  Total ops: " << context_kv_model->get_ordered_ops().size() << std::endl;

        // Print op summary for draft step model
        std::cout << "\n[Draft Step Model - Op Summary]" << std::endl;
        op_counts.clear();
        for (const auto& op : draft_step_model->get_ordered_ops()) {
            op_counts[op->get_type_info().name]++;
        }
        for (const auto& [name, count] : op_counts) {
            std::cout << "  " << name << ": " << count << std::endl;
        }
        std::cout << "  Total ops: " << draft_step_model->get_ordered_ops().size() << std::endl;

        // Print op summary for target model
        std::cout << "\n[Target Model - Op Summary]" << std::endl;
        op_counts.clear();
        for (const auto& op : target_model->get_ordered_ops()) {
            op_counts[op->get_type_info().name]++;
        }
        for (const auto& [name, count] : op_counts) {
            std::cout << "  " << name << ": " << count << std::endl;
        }
        std::cout << "  Total ops: " << target_model->get_ordered_ops().size() << std::endl;
        std::cout << std::endl;
    }

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

    // ---- VL preprocessing (vision encode + prompt building) ----
    ov::Tensor visual_embeds_padded;   // VL: [1, prompt_len, hidden_size]
    ov::Tensor visual_pos_mask_tensor; // VL: [1, prompt_len] boolean
    ov::Tensor vl_position_ids;        // VL: [3, 1, prompt_len] mRoPE
    ov::Tensor vl_input_ids;           // VL: [1, prompt_len] token ids
    ov::CompiledModel compiled_vision;
    if (vl_mode) {
        std::cout << "[VL] Loading and preprocessing image: " << image_path_arg << std::endl;

        // Load preprocessor config
        ov::genai::modeling::models::Qwen3_5VisionPreprocessConfig pre_cfg;
        const auto pre_cfg_path = target_dir / "preprocessor_config.json";
        if (std::filesystem::exists(pre_cfg_path)) {
            pre_cfg = ov::genai::modeling::models::Qwen3_5VisionPreprocessConfig::from_json_file(pre_cfg_path);
        }

        // Compile and run vision encoder
        std::cout << "[VL] Compiling vision model..." << std::endl;
        compiled_vision = core.compile_model(vision_model, device, compile_cfg);
        auto vision_request = compiled_vision.create_infer_request();

        auto image = utils::load_image(image_path_arg);

        ov::genai::modeling::models::Qwen3_5VisionPreprocessor preprocessor(
            target_qwen35_cfg.vision, pre_cfg);
        auto vision_inputs = preprocessor.preprocess(image, vl_pos_embed_weight);

        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kPixelValues, vision_inputs.pixel_values);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kGridThw, vision_inputs.grid_thw);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kPosEmbeds, vision_inputs.pos_embeds);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kRotaryCos, vision_inputs.rotary_cos);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kRotarySin, vision_inputs.rotary_sin);

        std::cout << "[VL] Running vision encoder..." << std::endl;
        vision_request.infer();
        auto visual_embeds_raw = vision_request.get_tensor(
            ov::genai::modeling::models::Qwen3_5VisionIO::kVisualEmbeds);

        // Count visual tokens and build VL prompt
        const int64_t image_tokens =
            ov::genai::modeling::models::Qwen3_5VisionPreprocessor::count_visual_tokens(
                vision_inputs.grid_thw, target_qwen35_cfg.vision.spatial_merge_size);
        std::cout << "[VL] Image tokens: " << image_tokens << std::endl;

        std::string vl_prompt = build_vl_prompt(prompt, image_tokens);
        auto vl_encoded = tokenizer.encode(vl_prompt, {ov::genai::add_special_tokens(false)});

        // Build input plan for mRoPE position IDs and visual position mask
        ov::genai::modeling::models::Qwen3_5InputPlanner planner(target_qwen35_cfg);
        auto plan = planner.build_plan(vl_encoded.input_ids, &vl_encoded.attention_mask,
                                       &vision_inputs.grid_thw);

        // Scatter visual embeddings into padded sequence-length tensor
        visual_embeds_padded = ov::genai::modeling::models::Qwen3_5InputPlanner::scatter_visual_embeds(
            visual_embeds_raw, plan.visual_pos_mask);
        visual_pos_mask_tensor = plan.visual_pos_mask;
        vl_position_ids = plan.position_ids;
        vl_input_ids = vl_encoded.input_ids;

        std::cout << "[VL] Vision preprocessing complete." << std::endl;
    }

    // Tokenize prompt
    ov::Tensor prompt_input_ids;
    if (vl_mode) {
        prompt_input_ids = vl_input_ids;
    } else {
        // Apply chat template if available (text-only mode)
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
        prompt_input_ids = encoded.input_ids;
    }

    const std::vector<int64_t> prompt_ids = tensor_to_ids(prompt_input_ids);
    std::vector<int64_t> output_ids = prompt_ids;
    const size_t prompt_len = output_ids.size();
    const size_t max_length = prompt_len + static_cast<size_t>(max_new_tokens);

    std::cout << "[INFO] prompt_len=" << prompt_len << " max_length=" << max_length << std::endl;

    // Compile
    std::cout << "[Compiling models on " << device << "...]" << std::endl;
    std::cout << "[Compiling target model...]" << std::endl;
    auto compiled_target = core.compile_model(target_model, device, compile_cfg);
    std::cout << "[Compiling context KV model...]" << std::endl;
    auto compiled_context_kv = core.compile_model(context_kv_model, device, compile_cfg);
    std::cout << "[Compiling draft step model...]" << std::endl;
    auto compiled_draft_step = core.compile_model(draft_step_model, device, compile_cfg);
    std::cout << "[All 3 models compiled.]" << std::endl;

    auto target_request = compiled_target.create_infer_request();
    auto context_kv_request = compiled_context_kv.create_infer_request();
    auto draft_step_request = compiled_draft_step.create_infer_request();

    // Release model graphs and weight data — no longer needed after compilation.
    // On iGPU (shared CPU/GPU memory), this recovers ~10+ GB that would otherwise
    // compete with GPU for memory bandwidth during decode.
    target_model.reset();
    context_kv_model.reset();
    draft_step_model.reset();
    if (vision_model) vision_model.reset();

    target_request.reset_state();
    ov::genai::utils::KVCacheState target_kv_state;
    const auto kv_pos = ov::genai::utils::get_kv_axes_pos(compiled_target.get_runtime_model());
    target_kv_state.seq_length_axis = kv_pos.seq_len;
    const auto beam_idx = make_beam_idx(1);

    // Detect state_update_mode input
    bool has_state_update_mode_input = false;
    for (const auto& input : compiled_target.inputs()) {
        const auto& names = input.get_names();
        if (names.find("state_update_mode") != names.end()) {
            has_state_update_mode_input = true;
            break;
        }
    }

    auto set_target_state_update_mode = [&](int32_t mode) {
        if (!has_state_update_mode_input) return;
        target_request.set_tensor("state_update_mode", make_state_update_mode_tensor(mode));
    };

    // Detect snapshot outputs (enables replay-free verify)
    bool has_snapshots = false;
    for (auto& output : compiled_target.outputs()) {
        for (auto& name : output.get_names()) {
            if (name.find("snapshot.") == 0) {
                has_snapshots = true;
                break;
            }
        }
        if (has_snapshots) break;
    }

    // Setup GPU-side snapshot tensors if running on GPU
    bool gpu_snapshots = false;
    std::map<std::string, ov::Tensor> snapshot_remote_tensors;
    ov::RemoteContext remote_context;
    if (has_snapshots) {
        try {
            remote_context = compiled_target.get_context();
            for (auto& output : compiled_target.outputs()) {
                std::string snap_name;
                for (auto& name : output.get_names()) {
                    if (name.find("snapshot.") == 0) { snap_name = name; break; }
                }
                if (snap_name.empty()) continue;

                auto pshape = output.get_partial_shape();
                ov::Shape snap_shape;
                snap_shape.push_back(1);
                snap_shape.push_back(static_cast<size_t>(dflash_cfg.block_size));
                for (size_t d = 2; d < pshape.size(); ++d)
                    snap_shape.push_back(pshape[d].get_length());

                auto dtype = output.get_element_type();
                auto snap_remote = remote_context.create_tensor(dtype, snap_shape);
                snapshot_remote_tensors[snap_name] = snap_remote;
            }
            gpu_snapshots = !snapshot_remote_tensors.empty();
            if (gpu_snapshots) {
                std::cout << "[Snapshots] GPU-side snapshot tensors allocated: "
                          << snapshot_remote_tensors.size() << " outputs" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "[Snapshots] GPU context not available, falling back to host: "
                      << e.what() << std::endl;
            gpu_snapshots = false;
        }
    }

    const bool use_deferred_state_commit = has_snapshots &&
                                           has_state_update_mode_input &&
                                           deferred_state_commit_enabled_by_default();
    if (use_deferred_state_commit) {
        std::cout << "[DFlash] Deferred state commit enabled for snapshot verify" << std::endl;
    } else if (has_snapshots && !has_state_update_mode_input) {
        std::cout << "[DFlash] Deferred state commit unavailable: target model has no state_update_mode input" << std::endl;
    }

    int32_t pending_snapshot_commit_index = -1;

    std::cout << "[DFlash] has_state_update_mode_input=" << has_state_update_mode_input
              << " has_snapshots=" << has_snapshots
              << " gpu_snapshots=" << gpu_snapshots
              << " use_deferred_state_commit=" << use_deferred_state_commit << std::endl;

    std::cout << "---------------START INFERENCE --------------------" << std::endl;

    // Bind GPU RemoteTensors for snapshot outputs before prefill.
    if (gpu_snapshots) {
        for (auto& [snap_name, snap_remote] : snapshot_remote_tensors) {
            target_request.set_tensor(snap_name, snap_remote);
        }
    }

    target_request.get_tensor("attention_mask").set_shape({1, 0});

    // Prefill: run target on prompt to get first token.
    auto prefill_start = Clock::now();
    target_request.set_tensor("input_ids", make_ids_tensor(output_ids));
    target_request.set_tensor("attention_mask", make_attention_mask(output_ids.size()));
    if (vl_mode) {
        target_request.set_tensor("position_ids", vl_position_ids);
        target_request.set_tensor("visual_embeds", visual_embeds_padded);
        target_request.set_tensor("visual_pos_mask", visual_pos_mask_tensor);
    } else {
        target_request.set_tensor("position_ids", make_mrope_position_ids(0, output_ids.size()));
    }
    target_request.set_tensor("beam_idx", beam_idx);
    set_target_state_update_mode(1);
    target_request.infer();
    auto logits = target_request.get_tensor("logits");

    // For VL mode, prepare zero visual tensors for subsequent decode/verify steps
    ov::Tensor zero_visual_embeds;
    ov::Tensor zero_visual_pos_mask;
    if (vl_mode) {
        const size_t hidden_size = static_cast<size_t>(target_qwen35_cfg.text.hidden_size);
        zero_visual_embeds = ov::Tensor(ov::element::f32, {1, 1, hidden_size});
        std::memset(zero_visual_embeds.data(), 0, zero_visual_embeds.get_byte_size());
        zero_visual_pos_mask = ov::Tensor(ov::element::boolean, {1, 1});
        zero_visual_pos_mask.data<bool>()[0] = false;
    }

    target_kv_state.add_inputs(make_ids_tensor(output_ids));
    ov::Tensor target_hidden_block = ensure_f32_copy(target_request.get_tensor("target_hidden"));
    const size_t hidden_dim = target_hidden_block.get_shape()[2];
    const size_t hidden_storage_elems = (max_length + static_cast<size_t>(dflash_cfg.block_size)) * hidden_dim;
    std::cerr << "[DFlash] target_hidden_storage: [1," << (max_length + dflash_cfg.block_size)
              << "," << hidden_dim << "] = "
              << (hidden_storage_elems * 4 / 1024 / 1024) << " MB (f32 on CPU)" << std::endl;
    ov::Tensor target_hidden_storage(ov::element::f32, {1, max_length + static_cast<size_t>(dflash_cfg.block_size), hidden_dim});
    ov::Tensor target_hidden_init(target_hidden_storage, {0, 0, 0}, {1, prompt_len, hidden_dim});
    target_hidden_block.copy_to(target_hidden_init);
    size_t target_hidden_len = prompt_len;

    // ── Initialize context KV caches ──
    // Run context_kv_model on full prompt hidden states to pre-compute K,V for all layers.
    const int32_t num_draft_layers = dflash_cfg.num_hidden_layers;
    const int32_t kv_heads = dflash_cfg.num_key_value_heads > 0
                                 ? dflash_cfg.num_key_value_heads
                                 : dflash_cfg.num_attention_heads;
    const int32_t head_dim = dflash_cfg.head_dim > 0
                                 ? dflash_cfg.head_dim
                                 : (dflash_cfg.hidden_size / dflash_cfg.num_attention_heads);
    const size_t max_ctx_len = max_length + static_cast<size_t>(dflash_cfg.block_size);

    // Pre-allocate context KV cache storage: [1, kv_heads, max_ctx_len, head_dim] per layer
    std::vector<ov::Tensor> context_k_storage(num_draft_layers);
    std::vector<ov::Tensor> context_v_storage(num_draft_layers);
    for (int32_t i = 0; i < num_draft_layers; ++i) {
        context_k_storage[i] = ov::Tensor(ov::element::f32,
            {1, static_cast<size_t>(kv_heads), max_ctx_len, static_cast<size_t>(head_dim)});
        context_v_storage[i] = ov::Tensor(ov::element::f32,
            {1, static_cast<size_t>(kv_heads), max_ctx_len, static_cast<size_t>(head_dim)});
    }
    size_t context_kv_len = 0;  // tracks how many context positions are cached

    // Compute initial context KV from prefill hidden states
    {
        ov::Tensor init_hidden_view(target_hidden_storage, {0, 0, 0}, {1, prompt_len, hidden_dim});
        ov::Tensor init_pos_ids(ov::element::i64, {1, prompt_len});
        {
            auto* p = init_pos_ids.data<int64_t>();
            for (size_t i = 0; i < prompt_len; ++i) p[i] = static_cast<int64_t>(i);
        }
        context_kv_request.set_tensor("target_hidden", init_hidden_view);
        context_kv_request.set_tensor("position_ids", init_pos_ids);
        context_kv_request.infer();

        // Copy outputs into KV cache storage
        for (int32_t i = 0; i < num_draft_layers; ++i) {
            auto k_out = context_kv_request.get_tensor("context_k_" + std::to_string(i));
            auto v_out = context_kv_request.get_tensor("context_v_" + std::to_string(i));
            // k_out: [1, kv_heads, prompt_len, head_dim]
            ov::Tensor k_dst(context_k_storage[i], {0, 0, 0, 0},
                             {1, static_cast<size_t>(kv_heads), prompt_len, static_cast<size_t>(head_dim)});
            ov::Tensor v_dst(context_v_storage[i], {0, 0, 0, 0},
                             {1, static_cast<size_t>(kv_heads), prompt_len, static_cast<size_t>(head_dim)});
            k_out.copy_to(k_dst);
            v_out.copy_to(v_dst);
        }
        context_kv_len = prompt_len;
        std::cerr << "[DFlash] Context KV cache initialized: " << context_kv_len << " positions, "
                  << (num_draft_layers * 2 * kv_heads * context_kv_len * head_dim * 4 / 1024 / 1024) << " MB" << std::endl;
    }

    int64_t next_token = argmax_last_token(logits);
    auto prefill_end = Clock::now();
    perf.ttft_ms = duration_ms(prefill_start, prefill_end);
    output_ids.push_back(next_token);

    if (next_token == eos_token_id) {
        std::cout << "\n[Early Stop] EOS token generated after prefill" << std::endl;
    }

    const double prefill_ms = duration_ms(prefill_start, prefill_end);
    perf.prefill_wall.add(prefill_ms);

    std::cout << "\n[Generating...]" << std::flush;

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

        // Draft model uses 2D position_ids (not mRoPE 3D)
        // With context KV caching, position_ids only covers DRAFT positions [T-1, T, ..., T+B-2]
        const size_t draft_block_size = block_ids.size();
        ov::Tensor draft_position_ids(ov::element::i64, {1, draft_block_size});
        {
            auto* pos_data = draft_position_ids.data<int64_t>();
            for (size_t i = 0; i < draft_block_size; ++i) {
                pos_data[i] = static_cast<int64_t>(target_hidden_len - 1 + i);
            }
        }

        // Run draft step model with cached context KV
        auto draft_start = Clock::now();

        // Set context KV cache views (only the valid portion)
        for (int32_t li = 0; li < num_draft_layers; ++li) {
            ov::Tensor k_view(context_k_storage[li], {0, 0, 0, 0},
                              {1, static_cast<size_t>(kv_heads), context_kv_len, static_cast<size_t>(head_dim)});
            ov::Tensor v_view(context_v_storage[li], {0, 0, 0, 0},
                              {1, static_cast<size_t>(kv_heads), context_kv_len, static_cast<size_t>(head_dim)});
            draft_step_request.set_tensor("context_k_" + std::to_string(li), k_view);
            draft_step_request.set_tensor("context_v_" + std::to_string(li), v_view);
        }

        draft_step_request.set_tensor("input_ids", embed_ids);
        draft_step_request.set_tensor("position_ids", draft_position_ids);

        draft_step_request.infer();
        auto draft_end = Clock::now();
        const double draft_ms = duration_ms(draft_start, draft_end);
        perf.draft_wall.add(draft_ms);
        step_tracked_ms += draft_ms;

        auto draft_logits = draft_step_request.get_tensor("logits");

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

        // Save linear states for fallback (only when snapshots unavailable).
        std::vector<std::pair<std::string, ov::Tensor>> saved_linear;
        if (!has_snapshots) {
            saved_linear = save_linear_states(target_request);
        }

        // Verify
        auto verify_start = Clock::now();
        target_request.set_tensor("input_ids", make_ids_tensor(block_output_ids));
        target_request.set_tensor("attention_mask", make_attention_mask(target_hidden_len + verify_len));
        target_request.set_tensor("position_ids", make_mrope_position_ids(target_hidden_len, verify_len));
        target_request.set_tensor("beam_idx", beam_idx);
        if (vl_mode) {
            target_request.set_tensor("visual_embeds", zero_visual_embeds);
            target_request.set_tensor("visual_pos_mask", zero_visual_pos_mask);
        }

        if (use_deferred_state_commit && pending_snapshot_commit_index >= 0) {
            set_target_state_update_mode(-(pending_snapshot_commit_index + 1));
            pending_snapshot_commit_index = -1;
        } else if (use_deferred_state_commit) {
            set_target_state_update_mode(0);
        } else {
            set_target_state_update_mode(1);
        }

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
        const size_t num_accepted = accepted + 1;

        // Multi-path acceptance handling (matching dflash_strategy)
        const bool all_accepted = (accepted == draft_tokens.size());
        const bool has_linear_states = has_snapshots || !saved_linear.empty();
        const bool use_snapshot_restore = has_snapshots &&
                                          has_linear_states &&
                                          num_accepted > 0 &&
                                          (!all_accepted || use_deferred_state_commit);

        if (use_snapshot_restore) {
            const size_t step = num_accepted - 1;
            pending_snapshot_commit_index = static_cast<int32_t>(step);

            const size_t tokens_to_trim = verify_len - num_accepted;
            target_kv_state.num_tokens_to_trim = tokens_to_trim;
            ov::genai::utils::trim_kv_cache(target_request, target_kv_state, std::nullopt);
            target_kv_state.num_tokens_to_trim = 0;

            target_hidden_block = ensure_f32_copy(target_request.get_tensor("target_hidden"));
        } else if (all_accepted) {
            target_hidden_block = ensure_f32_copy(target_request.get_tensor("target_hidden"));
        } else if (!has_linear_states) {
            const size_t tokens_to_trim = verify_len - num_accepted;
            target_kv_state.num_tokens_to_trim = tokens_to_trim;
            ov::genai::utils::trim_kv_cache(target_request, target_kv_state, std::nullopt);
            target_kv_state.num_tokens_to_trim = 0;
            target_hidden_block = ensure_f32_copy(target_request.get_tensor("target_hidden"));
        } else {
            restore_linear_states(target_request, saved_linear);

            target_kv_state.num_tokens_to_trim = verify_len;
            ov::genai::utils::trim_kv_cache(target_request, target_kv_state, std::nullopt);
            target_kv_state.num_tokens_to_trim = 0;

            {
                std::vector<int64_t> accepted_block(block_output_ids.begin(),
                                                    block_output_ids.begin() + static_cast<ptrdiff_t>(num_accepted));
                target_request.set_tensor("input_ids", make_ids_tensor(accepted_block));
                target_request.set_tensor("attention_mask", make_attention_mask(target_hidden_len + num_accepted));
                target_request.set_tensor("position_ids", make_mrope_position_ids(target_hidden_len, num_accepted));
                target_request.set_tensor("beam_idx", beam_idx);
                if (vl_mode) {
                    target_request.set_tensor("visual_embeds", zero_visual_embeds);
                    target_request.set_tensor("visual_pos_mask", zero_visual_pos_mask);
                }
                set_target_state_update_mode(1);
                target_request.infer();
            }

            target_hidden_block = ensure_f32_copy(target_request.get_tensor("target_hidden"));
        }

        if (num_accepted > 0 && target_hidden_len + num_accepted <= max_length + static_cast<size_t>(dflash_cfg.block_size)) {
            ov::Tensor src_slice(target_hidden_block, {0, 0, 0}, {1, num_accepted, hidden_dim});
            ov::Tensor dst_slice(target_hidden_storage,
                                 {0, target_hidden_len, 0},
                                 {1, target_hidden_len + num_accepted, hidden_dim});
            src_slice.copy_to(dst_slice);

            // Incrementally update context KV caches for the newly accepted positions.
            {
                ov::Tensor delta_hidden(target_hidden_storage,
                                        {0, target_hidden_len, 0},
                                        {1, target_hidden_len + num_accepted, hidden_dim});
                ov::Tensor delta_pos_ids(ov::element::i64, {1, num_accepted});
                {
                    auto* p = delta_pos_ids.data<int64_t>();
                    for (size_t i = 0; i < num_accepted; ++i) {
                        p[i] = static_cast<int64_t>(target_hidden_len + i);
                    }
                }
                context_kv_request.set_tensor("target_hidden", delta_hidden);
                context_kv_request.set_tensor("position_ids", delta_pos_ids);
                context_kv_request.infer();

                // Append to KV cache storage
                for (int32_t li = 0; li < num_draft_layers; ++li) {
                    auto k_out = context_kv_request.get_tensor("context_k_" + std::to_string(li));
                    auto v_out = context_kv_request.get_tensor("context_v_" + std::to_string(li));
                    ov::Tensor k_dst(context_k_storage[li],
                                     {0, 0, context_kv_len, 0},
                                     {1, static_cast<size_t>(kv_heads), context_kv_len + num_accepted, static_cast<size_t>(head_dim)});
                    ov::Tensor v_dst(context_v_storage[li],
                                     {0, 0, context_kv_len, 0},
                                     {1, static_cast<size_t>(kv_heads), context_kv_len + num_accepted, static_cast<size_t>(head_dim)});
                    k_out.copy_to(k_dst);
                    v_out.copy_to(v_dst);
                }
                context_kv_len += num_accepted;
            }

            target_hidden_len += num_accepted;
        }

        auto postproc_start = Clock::now();

        const size_t before_accept = output_ids.size();
        std::vector<int64_t> newly_accepted;
        for (size_t i = 0; i < accepted && output_ids.size() < max_length; ++i) {
            output_ids.push_back(draft_tokens[i]);
            newly_accepted.push_back(draft_tokens[i]);
        }

        const size_t accepted_pushed = output_ids.size() - before_accept;
        ++perf.draft_steps;
        perf.accepted_tokens += accepted_pushed;  // draft-only (matches pipeline convention)
        perf.accepted_per_step.push_back(accepted_pushed);
        auto postproc_end = Clock::now();
        perf.postproc_wall.add(duration_ms(postproc_start, postproc_end));

        if (output_ids.size() >= max_length) break;

        next_token = posterior_next;
        output_ids.push_back(next_token);
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

    // Batch decode all generated tokens at once
    std::vector<int64_t> generated_ids(output_ids.begin() + static_cast<ptrdiff_t>(prompt_len), output_ids.end());
    auto output_text = tokenizer.decode(generated_ids, {ov::genai::skip_special_tokens(true)});
    std::cout << "\n\n[Output]\n" << output_text << std::endl;
    std::cout << "\n[Generation Complete]" << std::endl;
    std::cout << "[Stop Reason] " << (stopped_by_eos ? "EOS token detected" : "Max length reached") << "\n" << std::endl;

    dflash_throughput = perf.total_generate_ms > 0
                            ? (static_cast<double>(perf.generated_tokens) * 1000.0) / perf.total_generate_ms
                            : 0.0;
    const double avg_accept = perf.draft_steps > 0
                                  ? static_cast<double>(perf.accepted_tokens) / static_cast<double>(perf.draft_steps)
                                  : 0.0;
    const double acceptance_rate = perf.generated_tokens > 0
                                      ? static_cast<double>(perf.accepted_tokens) / static_cast<double>(perf.generated_tokens)
                                      : 0.0;
    const size_t tokens_after_first = perf.generated_tokens > 0 ? perf.generated_tokens - 1 : 0;
    const double tpot_ms = tokens_after_first > 0
                               ? perf.total_generate_ms / static_cast<double>(tokens_after_first)
                               : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    // Print summary in same format as modeling_qwen3_5 baseline for batch script parsing
    std::cout << "Mode: dflash / text" << std::endl;
    std::cout << "Prompt token size: " << prompt_len << std::endl;
    std::cout << "Output token size: " << perf.generated_tokens << std::endl;
    std::cout << "TTFT: " << perf.ttft_ms << " ms" << std::endl;
    std::cout << "Decode time: " << perf.total_generate_ms << " ms" << std::endl;
    std::cout << "TPOT: " << tpot_ms << " ms/token" << std::endl;
    std::cout << "Throughput: " << dflash_throughput << " tokens/s" << std::endl;
    std::cout << "Draft steps: " << perf.draft_steps << std::endl;
    std::cout << "Accepted draft tokens: " << perf.accepted_tokens << std::endl;
    std::cout << "Acceptance rate: " << std::setprecision(4) << acceptance_rate << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "Avg accepted per step: " << avg_accept << std::endl;

    std::cout << std::setprecision(3);
    std::cout << "[Stage timings] wall-clock (ms):" << std::endl;
    print_stage_stats("prefill target", perf.prefill_wall);
    print_stage_stats("draft (embed+draft+lm_head)", perf.draft_wall);
    print_stage_stats("target verify", perf.verify_wall);
    print_stage_stats("postproc", perf.postproc_wall);
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

    return 0;
} catch (const std::exception& ex) {
    std::cerr << "Qwen3.5 DFlash sample failed: " << ex.what() << std::endl;
    return 1;
}
