// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace ov {
class Model;
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {
namespace weights {
class WeightFinalizer;
class WeightSource;
}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov

#include "modeling/models/qwen3/modeling_qwen3.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3ASRTextConfig {
    std::string architecture = "qwen3_asr";
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t max_position_embeddings = 0;

    float rms_norm_eps = 1e-6f;
    float rope_theta = 5000000.0f;

    std::string hidden_act = "silu";
    bool attention_bias = false;
    bool tie_word_embeddings = false;
};

struct Qwen3ASRAudioConfig {
    std::string architecture = "qwen3_asr_audio_encoder";
    int32_t num_mel_bins = 128;
    int32_t d_model = 1280;
    int32_t encoder_layers = 32;
    int32_t encoder_attention_heads = 20;
    int32_t encoder_ffn_dim = 5120;
    int32_t output_dim = 3584;
    int32_t max_source_positions = 1500;
    int32_t downsample_hidden_size = 480;
    std::string activation_function = "gelu";
};

struct Qwen3ASRTextIO {
    static constexpr const char* kInputIds = "input_ids";
    static constexpr const char* kInputsEmbeds = "inputs_embeds";
    static constexpr const char* kAttentionMask = "attention_mask";
    static constexpr const char* kPositionIds = "position_ids";
    static constexpr const char* kBeamIdx = "beam_idx";
    static constexpr const char* kAudioEmbeds = "audio_embeds";
    static constexpr const char* kAudioPosMask = "audio_pos_mask";
    static constexpr const char* kLogits = "logits";
};

struct Qwen3ASRAudioIO {
    static constexpr const char* kInputAudioFeatures = "input_audio_features";
    static constexpr const char* kAudioFeatureLengths = "audio_feature_lengths";
    static constexpr const char* kAudioEmbeds = "audio_embeds";
    static constexpr const char* kAudioOutputLengths = "audio_output_lengths";
};

// Same formula used by vLLM Qwen3-ASR multimodal processor.
int64_t qwen3_asr_feat_extract_output_length(int64_t input_length);

std::shared_ptr<ov::Model> create_qwen3_asr_text_model(
    const Qwen3ASRTextConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds = false,
    bool enable_audio_inputs = true);

std::shared_ptr<ov::Model> create_qwen3_asr_audio_encoder_model(
    const Qwen3ASRAudioConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
