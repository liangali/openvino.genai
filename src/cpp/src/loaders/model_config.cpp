// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "loaders/model_config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ov {
namespace genai {
namespace loaders {

namespace {

// Helper to safely get value from GGUFMetaData variant
template<typename T>
T get_or_default(const std::map<std::string, GGUFMetaData>& meta,
                 const std::string& key,
                 const T& default_value) {
    auto it = meta.find(key);
    if (it == meta.end()) {
        return default_value;
    }
    if (auto* val = std::get_if<T>(&it->second)) {
        return *val;
    }
    return default_value;
}

// Simple JSON value extraction (for basic types)
// Note: For production, use a proper JSON library like nlohmann/json
std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";
    
    pos = json.find("\"", pos);
    if (pos == std::string::npos) return "";
    
    auto end = json.find("\"", pos + 1);
    if (end == std::string::npos) return "";
    
    return json.substr(pos + 1, end - pos - 1);
}

int extract_json_int(const std::string& json, const std::string& key, int default_val = 0) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_val;
    
    // Skip whitespace
    pos++;
    while (pos < json.size() && std::isspace(json[pos])) pos++;
    
    // Read number
    std::string num_str;
    while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '-')) {
        num_str += json[pos++];
    }
    
    return num_str.empty() ? default_val : std::stoi(num_str);
}

float extract_json_float(const std::string& json, const std::string& key, float default_val = 0.0f) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_val;
    
    // Skip whitespace
    pos++;
    while (pos < json.size() && std::isspace(json[pos])) pos++;
    
    // Read number
    std::string num_str;
    while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '-' || 
                                  json[pos] == '.' || json[pos] == 'e' || json[pos] == 'E' || json[pos] == '+')) {
        num_str += json[pos++];
    }
    
    return num_str.empty() ? default_val : std::stof(num_str);
}

bool extract_json_bool(const std::string& json, const std::string& key, bool default_val = false) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_val;
    
    // Skip whitespace
    pos++;
    while (pos < json.size() && std::isspace(json[pos])) pos++;
    
    if (json.substr(pos, 4) == "true") return true;
    if (json.substr(pos, 5) == "false") return false;
    
    return default_val;
}

std::vector<int32_t> extract_json_int_array(const std::string& json, const std::string& key) {
    std::vector<int32_t> result;
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return result;
    
    pos = json.find("[", pos);
    if (pos == std::string::npos) return result;
    
    auto end = json.find("]", pos);
    if (end == std::string::npos) return result;
    
    // Parse integers between [ and ]
    std::string array_content = json.substr(pos + 1, end - pos - 1);
    std::stringstream ss(array_content);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t\n\r");
        size_t finish = token.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && finish != std::string::npos) {
            std::string num_str = token.substr(start, finish - start + 1);
            if (!num_str.empty()) {
                try {
                    result.push_back(std::stoi(num_str));
                } catch (...) {
                    // Skip invalid entries
                }
            }
        }
    }
    return result;
}

std::vector<std::string> extract_json_string_array(const std::string& json, const std::string& key) {
    std::vector<std::string> result;
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return result;

    pos = json.find("[", pos);
    if (pos == std::string::npos) return result;

    auto end = json.find("]", pos);
    if (end == std::string::npos) return result;

    std::string array_content = json.substr(pos + 1, end - pos - 1);
    size_t i = 0;
    while (i < array_content.size()) {
        while (i < array_content.size() && std::isspace(static_cast<unsigned char>(array_content[i]))) i++;
        if (i >= array_content.size()) break;
        if (array_content[i] == ',') {
            i++;
            continue;
        }
        if (array_content[i] != '"') {
            i++;
            continue;
        }
        i++;  // skip opening quote
        size_t start = i;
        while (i < array_content.size() && array_content[i] != '"') i++;
        if (i > start) {
            result.push_back(array_content.substr(start, i - start));
        }
        if (i < array_content.size() && array_content[i] == '"') i++;  // skip closing quote
    }
    return result;
}

std::string extract_json_object(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";

    pos = json.find("{", pos);
    if (pos == std::string::npos) return "";

    size_t start = pos;
    int depth = 0;
    bool in_string = false;
    bool escape = false;

    for (size_t i = pos; i < json.size(); ++i) {
        char c = json[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) {
                return json.substr(start, i - start + 1);
            }
        }
    }
    return "";
}

}  // namespace

ModelConfig ModelConfig::from_gguf(const std::map<std::string, GGUFMetaData>& meta) {
    ModelConfig config;
    
    // Architecture
    config.architecture = get_or_default<std::string>(meta, "architecture", "unknown");
    config.model_type = config.architecture;  // GGUF uses architecture as model type
    
    // Dimensions
    config.hidden_size = get_or_default<int>(meta, "hidden_size", 0);
    config.intermediate_size = get_or_default<int>(meta, "intermediate_size", 0);
    config.num_hidden_layers = get_or_default<int>(meta, "layer_num", 0);
    config.num_attention_heads = get_or_default<int>(meta, "head_num", 0);
    config.num_key_value_heads = get_or_default<int>(meta, "head_num_kv", 0);
    config.head_dim = get_or_default<int>(meta, "head_size", 0);
    config.vocab_size = get_or_default<int>(meta, "vocab_size", 0);
    config.max_position_embeddings = get_or_default<int>(meta, "max_position_embeddings", 0);
    
    // Normalization
    config.rms_norm_eps = get_or_default<float>(meta, "rms_norm_eps", 1e-6f);
    
    // RoPE
    config.rope_theta = get_or_default<float>(meta, "rope_freq_base", 10000.0f);
    config.rope_scaling_factor = get_or_default<float>(meta, "rope_scaling_factor", 1.0f);
    
    // Other
    config.file_type = get_or_default<int>(meta, "file_type", 0);
    
    // Compute head_dim if not provided
    if (config.head_dim == 0 && config.hidden_size > 0 && config.num_attention_heads > 0) {
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }
    
    return config;
}

ModelConfig ModelConfig::from_hf_json(const std::filesystem::path& config_path) {
    // Read JSON file
    std::ifstream file(config_path);
    if (!file) {
        throw std::runtime_error("Cannot open config file: " + config_path.string());
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    
    ModelConfig config;
    
    // Parse JSON fields
    config.model_type = extract_json_string(json, "model_type");
    config.architecture = config.model_type;
    
    // Handle architectures array if present
    std::string arch = extract_json_string(json, "architectures");
    if (!arch.empty()) {
        // First architecture in the list
        if (arch.find("DFlashDraftModel") != std::string::npos) config.architecture = "dflash";
        // Order matters: check Qwen3Next before Qwen3 to avoid misclassification.
        else if (arch.find("Qwen3Next") != std::string::npos) config.architecture = "qwen3_next";
        else if (arch.find("Qwen3Moe") != std::string::npos) config.architecture = "qwen3_moe";

        else if (arch.find("Qwen3") != std::string::npos) config.architecture = "qwen3";
        else if (arch.find("Qwen2") != std::string::npos) config.architecture = "qwen2";
        else if (arch.find("Llama") != std::string::npos) config.architecture = "llama";
        else if (arch.find("Mistral") != std::string::npos) config.architecture = "mistral";
        else if (arch.find("SmolLM3") != std::string::npos) config.architecture = "smollm3";
    }

    if (config.model_type == "qwen3_next") {
        config.architecture = "qwen3_next";
    } else if (config.model_type == "qwen3_moe") {
        config.architecture = "qwen3_moe";
    } else if (config.model_type == "qwen3_asr_audio_encoder") {
        config.architecture = "qwen3_asr_audio_encoder";
    }
    
    // Also check model_type for SmolLM3
    if (config.model_type == "smollm3" || config.model_type == "LlamaForCausalLM") {
        // SmolLM3 may use LlamaForCausalLM as model_type, check for smollm3-specific fields
        int no_rope_interval = extract_json_int(json, "no_rope_layer_interval", 0);
        if (no_rope_interval > 0) {
            config.architecture = "smollm3";
        }
    }
    
    std::string dims_json = json;

    // Qwen3-ASR stores LM dimensions in thinker_config.text_config.
    if (config.model_type == "qwen3_asr") {
        config.architecture = "qwen3_asr";
        auto thinker_json = extract_json_object(json, "thinker_config");
        if (!thinker_json.empty()) {
            auto text_json = extract_json_object(thinker_json, "text_config");
            if (!text_json.empty()) {
                dims_json = text_json;
            }

            auto audio_json = extract_json_object(thinker_json, "audio_config");
            if (!audio_json.empty()) {
                config.audio_num_mel_bins = extract_json_int(audio_json, "num_mel_bins", 0);
                config.audio_hidden_size = extract_json_int(audio_json, "d_model", 0);
                config.audio_intermediate_size = extract_json_int(audio_json, "encoder_ffn_dim", 0);
                config.audio_num_hidden_layers = extract_json_int(audio_json, "encoder_layers", 0);
                config.audio_num_attention_heads = extract_json_int(audio_json, "encoder_attention_heads", 0);
                config.audio_max_position_embeddings = extract_json_int(audio_json, "max_source_positions", 0);
                config.audio_downsample_hidden_size = extract_json_int(audio_json, "downsample_hidden_size", 0);
                config.audio_output_dim = extract_json_int(audio_json, "output_dim", 0);
                config.audio_hidden_act = extract_json_string(audio_json, "activation_function");
            }
        }
    } else if (config.model_type == "qwen3_asr_audio_encoder") {
        // Audio encoder config uses different field names.
        // Map to unified ModelConfig dimensions:
        // hidden_size <- d_model
        // intermediate_size <- encoder_ffn_dim
        // num_hidden_layers <- encoder_layers
        // num_attention_heads <- encoder_attention_heads
        config.hidden_size = extract_json_int(json, "d_model");
        config.intermediate_size = extract_json_int(json, "encoder_ffn_dim");
        config.num_hidden_layers = extract_json_int(json, "encoder_layers");
        config.num_attention_heads = extract_json_int(json, "encoder_attention_heads");
        config.num_key_value_heads = config.num_attention_heads;
        config.head_dim = (config.hidden_size > 0 && config.num_attention_heads > 0)
            ? (config.hidden_size / config.num_attention_heads)
            : 0;
        config.max_position_embeddings = extract_json_int(json, "max_source_positions", 1500);
        config.hidden_act = extract_json_string(json, "activation_function");
        if (config.hidden_act.empty()) {
            config.hidden_act = "gelu";
        }
        return config;
    }

    // Dimensions
    config.hidden_size = extract_json_int(dims_json, "hidden_size");
    config.intermediate_size = extract_json_int(dims_json, "intermediate_size");
    config.num_hidden_layers = extract_json_int(dims_json, "num_hidden_layers");
    config.num_attention_heads = extract_json_int(dims_json, "num_attention_heads");
    config.num_key_value_heads = extract_json_int(dims_json, "num_key_value_heads", config.num_attention_heads);
    config.head_dim = extract_json_int(dims_json, "head_dim", 0);
    config.vocab_size = extract_json_int(dims_json, "vocab_size");
    config.max_position_embeddings = extract_json_int(dims_json, "max_position_embeddings");

    // DFlash-specific
    config.block_size = extract_json_int(json, "block_size", 0);
    config.num_target_layers = extract_json_int(json, "num_target_layers", 0);
    
    // Normalization
    config.rms_norm_eps = extract_json_float(dims_json, "rms_norm_eps", 1e-6f);
    
    // RoPE
    config.rope_theta = extract_json_float(dims_json, "rope_theta", 10000.0f);
    
    // Other
    config.tie_word_embeddings = extract_json_bool(dims_json, "tie_word_embeddings", false);
    config.attention_bias = extract_json_bool(dims_json, "attention_bias", false);
    config.mlp_bias = extract_json_bool(dims_json, "mlp_bias", false);
    config.hidden_act = extract_json_string(dims_json, "hidden_act");
    if (config.hidden_act.empty()) {
        config.hidden_act = "silu";
    }
    
    // SmolLM3-specific
    config.no_rope_layer_interval = extract_json_int(json, "no_rope_layer_interval", 0);
    config.no_rope_layers = extract_json_int_array(json, "no_rope_layers");

    // Qwen3-Next specific fields
    config.full_attention_interval = extract_json_int(json, "full_attention_interval", 4);
    config.partial_rotary_factor = extract_json_float(json, "partial_rotary_factor", 1.0f);
    config.linear_conv_kernel_dim = extract_json_int(json, "linear_conv_kernel_dim", 0);
    config.linear_key_head_dim = extract_json_int(json, "linear_key_head_dim", 0);
    config.linear_value_head_dim = extract_json_int(json, "linear_value_head_dim", 0);
    config.linear_num_key_heads = extract_json_int(json, "linear_num_key_heads", 0);
    config.linear_num_value_heads = extract_json_int(json, "linear_num_value_heads", 0);
    config.decoder_sparse_step = extract_json_int(json, "decoder_sparse_step", 0);
    config.moe_intermediate_size = extract_json_int(json, "moe_intermediate_size", 0);
    config.shared_expert_intermediate_size = extract_json_int(json, "shared_expert_intermediate_size", 0);
    config.num_experts = extract_json_int(json, "num_experts", 0);
    config.num_experts_per_tok = extract_json_int(json, "num_experts_per_tok", 0);
    config.norm_topk_prob = extract_json_bool(json, "norm_topk_prob", true);
    config.output_router_logits = extract_json_bool(json, "output_router_logits", false);
    config.router_aux_loss_coef = extract_json_float(json, "router_aux_loss_coef", 0.0f);
    config.mlp_only_layers = extract_json_int_array(json, "mlp_only_layers");
    config.layer_types = extract_json_string_array(json, "layer_types");
    
    // Compute head_dim if not provided
    if (config.head_dim == 0 && config.hidden_size > 0 && config.num_attention_heads > 0) {
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }
    
    // Parse torch_dtype (fallback to "dtype")
    std::string dtype_str = extract_json_string(json, "torch_dtype");
    if (dtype_str.empty()) {
        dtype_str = extract_json_string(json, "dtype");
    }
    if (dtype_str == "bfloat16") config.dtype = ov::element::bf16;
    else if (dtype_str == "float16") config.dtype = ov::element::f16;
    else if (dtype_str == "float32") config.dtype = ov::element::f32;
    
    return config;
}

std::map<std::string, GGUFMetaData> ModelConfig::to_gguf_format() const {
    std::map<std::string, GGUFMetaData> meta;
    
    meta["architecture"] = architecture;
    meta["hidden_size"] = hidden_size;
    meta["intermediate_size"] = intermediate_size;
    meta["layer_num"] = num_hidden_layers;
    meta["head_num"] = num_attention_heads;
    meta["head_num_kv"] = num_key_value_heads;
    meta["head_size"] = head_dim;
    meta["vocab_size"] = vocab_size;
    meta["max_position_embeddings"] = max_position_embeddings;
    meta["rms_norm_eps"] = rms_norm_eps;
    meta["rope_freq_base"] = rope_theta;
    meta["rope_scaling_factor"] = rope_scaling_factor;
    meta["file_type"] = file_type;
    
    return meta;
}

void ModelConfig::validate() const {
    std::vector<std::string> missing;
    
    if (architecture.empty()) missing.push_back("architecture");
    if (hidden_size <= 0) missing.push_back("hidden_size");
    if (num_hidden_layers <= 0) missing.push_back("num_hidden_layers");
    if (num_attention_heads <= 0) missing.push_back("num_attention_heads");
    if (vocab_size <= 0) missing.push_back("vocab_size");
    if (architecture == "dflash") {
        if (block_size <= 0) missing.push_back("block_size");
        if (num_target_layers <= 0) missing.push_back("num_target_layers");
    }
    
    if (!missing.empty()) {
        std::string msg = "Missing required config fields: ";
        for (size_t i = 0; i < missing.size(); ++i) {
            if (i > 0) msg += ", ";
            msg += missing[i];
        }
        throw std::runtime_error(msg);
    }
}

std::string ModelConfig::summary() const {
    std::stringstream ss;
    ss << "ModelConfig {\n"
       << "  architecture: " << architecture << "\n"
       << "  model_type: " << model_type << "\n"
       << "  hidden_size: " << hidden_size << "\n"
       << "  intermediate_size: " << intermediate_size << "\n"
       << "  num_hidden_layers: " << num_hidden_layers << "\n"
       << "  num_attention_heads: " << num_attention_heads << "\n"
       << "  num_key_value_heads: " << num_key_value_heads << "\n"
       << "  head_dim: " << head_dim << "\n"
       << "  vocab_size: " << vocab_size << "\n"
       << "  max_position_embeddings: " << max_position_embeddings << "\n"
       << "  block_size: " << block_size << "\n"
       << "  num_target_layers: " << num_target_layers << "\n"
       << "  rms_norm_eps: " << rms_norm_eps << "\n"
       << "  rope_theta: " << rope_theta << "\n"
       << "  tie_word_embeddings: " << (tie_word_embeddings ? "true" : "false") << "\n"
       << "  hidden_act: " << hidden_act << "\n"
       << "  full_attention_interval: " << full_attention_interval << "\n"
       << "  partial_rotary_factor: " << partial_rotary_factor << "\n"
       << "  linear_conv_kernel_dim: " << linear_conv_kernel_dim << "\n"
       << "  linear_key_head_dim: " << linear_key_head_dim << "\n"
       << "  linear_value_head_dim: " << linear_value_head_dim << "\n"
       << "  linear_num_key_heads: " << linear_num_key_heads << "\n"
       << "  linear_num_value_heads: " << linear_num_value_heads << "\n"
       << "  decoder_sparse_step: " << decoder_sparse_step << "\n"
       << "  moe_intermediate_size: " << moe_intermediate_size << "\n"
       << "  shared_expert_intermediate_size: " << shared_expert_intermediate_size << "\n"
       << "  num_experts: " << num_experts << "\n"
       << "  num_experts_per_tok: " << num_experts_per_tok << "\n"
       << "}";
    return ss.str();
}

}  // namespace loaders
}  // namespace genai
}  // namespace ov
