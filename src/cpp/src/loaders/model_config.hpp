// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file model_config.hpp
 * @brief Unified model configuration structure
 * 
 * ModelConfig provides a format-agnostic representation of model configuration.
 * It can be constructed from different source formats (GGUF metadata, HuggingFace JSON).
 */

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <filesystem>

#include <openvino/openvino.hpp>
#include "modeling/weights/quantization_config.hpp"

// Forward declarations for format-specific types
namespace ov {
namespace genai {

// GGUF metadata type (from gguf.hpp)
using GGUFMetaData = std::variant<
    std::monostate,
    float,
    int,
    ov::Tensor,
    std::string,
    std::vector<std::string>,
    std::vector<int32_t>
>;

}  // namespace genai
}  // namespace ov

namespace ov {
namespace genai {
namespace loaders {

/**
 * @brief Unified model configuration
 * 
 * This structure contains all configuration fields needed to build a model,
 * regardless of the source format. Factory methods convert format-specific
 * configurations to this unified format.
 */
struct ModelConfig {
    // ========== Model identification ==========
    
    /// Architecture name (e.g., "llama", "qwen3", "mistral")
    std::string architecture;
    
    /// Model type from HuggingFace (e.g., "qwen3", "llama")
    std::string model_type;
    
    // ========== Model dimensions ==========
    
    /// Hidden layer size (e.g., 2560 for Qwen3-4B)
    int32_t hidden_size = 0;
    
    /// Intermediate size in MLP (e.g., 9728 for Qwen3-4B)
    int32_t intermediate_size = 0;
    
    /// Number of transformer layers (e.g., 36 for Qwen3-4B)
    int32_t num_hidden_layers = 0;
    
    /// Number of attention heads (e.g., 32 for Qwen3-4B)
    int32_t num_attention_heads = 0;
    
    /// Number of key-value heads for GQA (e.g., 8 for Qwen3-4B)
    int32_t num_key_value_heads = 0;
    
    /// Head dimension (hidden_size / num_attention_heads)
    int32_t head_dim = 0;
    
    /// Vocabulary size
    int32_t vocab_size = 0;
    
    /// Maximum sequence length
    int32_t max_position_embeddings = 0;

    // ========== DFlash draft-specific ==========

    /// Block size for DFlash draft generation
    int32_t block_size = 0;

    /// Number of layers in target model (for DFlash conditioning)
    int32_t num_target_layers = 0;
    
    // ========== Normalization ==========
    
    /// RMS normalization epsilon
    float rms_norm_eps = 1e-6f;
    
    // ========== RoPE (Rotary Position Embedding) ==========
    
    /// RoPE base frequency
    float rope_theta = 10000.0f;
    
    /// RoPE scaling factor
    float rope_scaling_factor = 1.0f;
    
    // ========== Other configurations ==========
    
    /// Whether to tie embedding and output weights
    bool tie_word_embeddings = false;
    
    /// Whether attention has bias
    bool attention_bias = false;
    
    /// Whether MLP has bias (SmolLM3)
    bool mlp_bias = false;
    
    /// Activation function (e.g., "silu", "gelu")
    std::string hidden_act = "silu";

    // ========== Qwen3-Next specific ==========

    /// Layer types for hybrid attention (e.g., linear_attention/full_attention)
    std::vector<std::string> layer_types;

    /// Full attention insertion interval when layer_types is not explicitly provided
    int32_t full_attention_interval = 4;

    /// Partial rotary factor for RoPE (Qwen3-Next uses 0.25)
    float partial_rotary_factor = 1.0f;

    /// Linear attention Conv1D kernel size
    int32_t linear_conv_kernel_dim = 0;

    /// Linear attention key head dimension
    int32_t linear_key_head_dim = 0;

    /// Linear attention value head dimension
    int32_t linear_value_head_dim = 0;

    /// Linear attention number of key heads
    int32_t linear_num_key_heads = 0;

    /// Linear attention number of value heads
    int32_t linear_num_value_heads = 0;

    /// MoE sparsity frequency in decoder
    int32_t decoder_sparse_step = 0;

    /// MoE routed expert hidden dimension
    int32_t moe_intermediate_size = 0;

    /// MoE shared expert hidden dimension
    int32_t shared_expert_intermediate_size = 0;

    /// Number of routed experts
    int32_t num_experts = 0;

    /// Number of selected experts per token
    int32_t num_experts_per_tok = 0;

    /// Whether to normalize top-k probabilities in routing
    bool norm_topk_prob = true;

    /// Whether to output router logits
    bool output_router_logits = false;

    /// Router auxiliary loss coefficient
    float router_aux_loss_coef = 0.0f;

    /// Optional list of dense MLP-only layer indices
    std::vector<int32_t> mlp_only_layers;

    // ========== Qwen3-ASR nested audio_config ==========

    /// Audio encoder mel bins (thinker_config.audio_config.num_mel_bins)
    int32_t audio_num_mel_bins = 0;

    /// Audio encoder hidden size (thinker_config.audio_config.d_model)
    int32_t audio_hidden_size = 0;

    /// Audio encoder FFN size (thinker_config.audio_config.encoder_ffn_dim)
    int32_t audio_intermediate_size = 0;

    /// Audio encoder number of layers (thinker_config.audio_config.encoder_layers)
    int32_t audio_num_hidden_layers = 0;

    /// Audio encoder attention heads (thinker_config.audio_config.encoder_attention_heads)
    int32_t audio_num_attention_heads = 0;

    /// Audio encoder max source positions (thinker_config.audio_config.max_source_positions)
    int32_t audio_max_position_embeddings = 0;

    /// Audio encoder downsample hidden size (thinker_config.audio_config.downsample_hidden_size)
    int32_t audio_downsample_hidden_size = 0;

    /// Audio encoder output embedding size (thinker_config.audio_config.output_dim)
    int32_t audio_output_dim = 0;

    /// Audio encoder activation (thinker_config.audio_config.activation_function)
    std::string audio_hidden_act;
    
    // ========== Quantization ==========
    
    /// In-flight quantization configuration
    std::optional<ov::genai::modeling::weights::QuantizationConfig> quantization_config;

    // ========== SmolLM3-specific ==========
    
    /// Interval for layers without RoPE (SmolLM3: every 4th layer has no RoPE)
    int32_t no_rope_layer_interval = 0;
    
    /// Explicit list of layers without RoPE (alternative to interval)
    std::vector<int32_t> no_rope_layers;
    
    /// Model data type (e.g., bf16, fp16)
    ov::element::Type dtype = ov::element::bf16;
    
    /// GGUF file type (for quantization info)
    int32_t file_type = 0;
    
    // ========== Factory methods ==========
    
    /**
     * @brief Create ModelConfig from GGUF metadata
     * 
     * @param meta GGUF metadata map
     * @return ModelConfig with values from GGUF
     */
    static ModelConfig from_gguf(const std::map<std::string, GGUFMetaData>& meta);
    
    /**
     * @brief Create ModelConfig from HuggingFace config.json
     * 
     * @param config_path Path to config.json file
     * @return ModelConfig with values from JSON
     */
    static ModelConfig from_hf_json(const std::filesystem::path& config_path);
    
    /**
     * @brief Get configuration value as GGUF-compatible format
     * 
     * This is used for compatibility with existing building_blocks code.
     * 
     * @return Map in GGUF metadata format
     */
    std::map<std::string, GGUFMetaData> to_gguf_format() const;
    
    /**
     * @brief Validate configuration completeness
     * 
     * @throws std::runtime_error if required fields are missing
     */
    void validate() const;
    
    /**
     * @brief Get a human-readable summary of the configuration
     */
    std::string summary() const;
};

}  // namespace loaders
}  // namespace genai
}  // namespace ov
