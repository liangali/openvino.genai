// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <filesystem>

#include <gtest/gtest.h>

#include "loaders/model_builder.hpp"
#include "loaders/model_config.hpp"
#include "modeling/models/qwen3_asr/modeling_qwen3_asr.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

int64_t ref_feat_len(int64_t input_length) {
  auto floor_div = [](int64_t a, int64_t b) -> int64_t {
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r > 0) != (b > 0))) {
      --q;
    }
    return q;
  };
    const int64_t input_lengths_leave = input_length % 100;
  const int64_t feat_lengths = floor_div(input_lengths_leave - 1, 2) + 1;
  const int64_t output_lengths =
    (floor_div((floor_div(feat_lengths - 1, 2) + 1 - 1), 2) + 1 + floor_div(input_length, 100) * 13);
    return output_lengths;
}

}  // namespace

TEST(Qwen3ASRModeling, FeatureLengthFormulaBasicCases) {
    using ov::genai::modeling::models::qwen3_asr_feat_extract_output_length;

    EXPECT_EQ(qwen3_asr_feat_extract_output_length(1), 1);
    EXPECT_EQ(qwen3_asr_feat_extract_output_length(10), 2);
    EXPECT_EQ(qwen3_asr_feat_extract_output_length(100), 13);
    EXPECT_EQ(qwen3_asr_feat_extract_output_length(101), 14);
      EXPECT_EQ(qwen3_asr_feat_extract_output_length(256), 33);

    for (int64_t n : {1LL, 2LL, 3LL, 9LL, 10LL, 20LL, 99LL, 100LL, 101LL, 199LL, 200LL, 256LL, 399LL, 400LL}) {
        EXPECT_EQ(qwen3_asr_feat_extract_output_length(n), ref_feat_len(n));
    }
}

TEST(Qwen3ASRModeling, ModelConfigParsesNestedThinkerTextConfig) {
    namespace fs = std::filesystem;

    const fs::path tmp = fs::temp_directory_path() / "ov_qwen3_asr_config_test.json";
    {
        std::ofstream out(tmp);
        out << R"({
  "model_type": "qwen3_asr",
  "thinker_config": {
    "text_config": {
      "hidden_size": 4096,
      "intermediate_size": 22016,
      "num_hidden_layers": 32,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "vocab_size": 151936,
      "max_position_embeddings": 128000,
      "rms_norm_eps": 1e-6,
      "rope_theta": 5000000.0,
      "hidden_act": "silu",
      "attention_bias": false,
      "tie_word_embeddings": false
    }
  }
})";
    }

    auto cfg = ov::genai::loaders::ModelConfig::from_hf_json(tmp);

    EXPECT_EQ(cfg.model_type, "qwen3_asr");
    EXPECT_EQ(cfg.architecture, "qwen3_asr");
    EXPECT_EQ(cfg.hidden_size, 4096);
    EXPECT_EQ(cfg.intermediate_size, 22016);
    EXPECT_EQ(cfg.num_hidden_layers, 32);
    EXPECT_EQ(cfg.num_attention_heads, 32);
    EXPECT_EQ(cfg.num_key_value_heads, 8);
    EXPECT_EQ(cfg.head_dim, 128);
    EXPECT_EQ(cfg.vocab_size, 151936);
    EXPECT_EQ(cfg.max_position_embeddings, 128000);

    std::error_code ec;
    fs::remove(tmp, ec);
}

  TEST(Qwen3ASRModeling, ModelConfigParsesAudioEncoderConfig) {
    namespace fs = std::filesystem;

    const fs::path tmp = fs::temp_directory_path() / "ov_qwen3_asr_audio_config_test.json";
    {
      std::ofstream out(tmp);
      out << R"({
    "model_type": "qwen3_asr_audio_encoder",
    "num_mel_bins": 128,
    "encoder_layers": 32,
    "encoder_attention_heads": 20,
    "encoder_ffn_dim": 5120,
    "d_model": 1280,
    "max_source_positions": 1500,
    "activation_function": "gelu"
  })";
    }

    auto cfg = ov::genai::loaders::ModelConfig::from_hf_json(tmp);

    EXPECT_EQ(cfg.model_type, "qwen3_asr_audio_encoder");
    EXPECT_EQ(cfg.architecture, "qwen3_asr_audio_encoder");
    EXPECT_EQ(cfg.hidden_size, 1280);
    EXPECT_EQ(cfg.intermediate_size, 5120);
    EXPECT_EQ(cfg.num_hidden_layers, 32);
    EXPECT_EQ(cfg.num_attention_heads, 20);
    EXPECT_EQ(cfg.max_position_embeddings, 1500);
    EXPECT_EQ(cfg.hidden_act, "gelu");

    std::error_code ec;
    fs::remove(tmp, ec);
  }

  TEST(Qwen3ASRModeling, ModelBuilderRegistrationsExist) {
    auto& builder = ov::genai::loaders::ModelBuilder::instance();

    EXPECT_TRUE(builder.has_architecture("qwen3_asr"));
    EXPECT_TRUE(builder.has_architecture("qwen3asr"));
    EXPECT_TRUE(builder.has_architecture("qwen3_asr_text"));

    EXPECT_TRUE(builder.has_architecture("qwen3_asr_audio_encoder"));
    EXPECT_TRUE(builder.has_architecture("qwen3asraudioencoder"));
  }

  TEST(Qwen3ASRModeling, AudioEncoderModelBuildsWithDummyWeights) {
    // Small config for fast graph construction.
    ov::genai::modeling::models::Qwen3ASRAudioConfig cfg;
    cfg.num_mel_bins = 8;
    cfg.d_model = 4;
    cfg.encoder_layers = 1;
    cfg.encoder_attention_heads = 2;
    cfg.encoder_ffn_dim = 8;
    cfg.output_dim = 6;
    cfg.max_source_positions = 16;
    cfg.downsample_hidden_size = 2;
    cfg.activation_function = "gelu";

    // conv_out input dim formula in implementation:
    // downsample_hidden_size * ((((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
    const size_t conv_out_in = static_cast<size_t>(cfg.downsample_hidden_size) *
                   static_cast<size_t>(((((cfg.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2));

    test_utils::DummyWeightSource weights;
    // Conv stem
    weights.add("audio_tower.conv2d1.weight", test_utils::make_tensor(test_utils::make_seq(2 * 1 * 3 * 3, 0.01f, 0.001f), {2, 1, 3, 3}));
    weights.add("audio_tower.conv2d1.bias", test_utils::make_tensor(test_utils::make_seq(2, 0.0f, 0.0f), {2}));
    weights.add("audio_tower.conv2d2.weight", test_utils::make_tensor(test_utils::make_seq(2 * 2 * 3 * 3, 0.01f, 0.001f), {2, 2, 3, 3}));
    weights.add("audio_tower.conv2d2.bias", test_utils::make_tensor(test_utils::make_seq(2, 0.0f, 0.0f), {2}));
    weights.add("audio_tower.conv2d3.weight", test_utils::make_tensor(test_utils::make_seq(2 * 2 * 3 * 3, 0.01f, 0.001f), {2, 2, 3, 3}));
    weights.add("audio_tower.conv2d3.bias", test_utils::make_tensor(test_utils::make_seq(2, 0.0f, 0.0f), {2}));
    weights.add("audio_tower.conv_out.weight", test_utils::make_tensor(test_utils::make_seq(4 * conv_out_in, 0.01f, 0.001f), {4, conv_out_in}));

    // Encoder layer[0]
    weights.add("audio_tower.layers[0].self_attn.q_proj.weight", test_utils::make_tensor(test_utils::make_seq(4 * 4, 0.01f, 0.001f), {4, 4}));
    weights.add("audio_tower.layers[0].self_attn.q_proj.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
    weights.add("audio_tower.layers[0].self_attn.k_proj.weight", test_utils::make_tensor(test_utils::make_seq(4 * 4, 0.01f, 0.001f), {4, 4}));
    weights.add("audio_tower.layers[0].self_attn.k_proj.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
    weights.add("audio_tower.layers[0].self_attn.v_proj.weight", test_utils::make_tensor(test_utils::make_seq(4 * 4, 0.01f, 0.001f), {4, 4}));
    weights.add("audio_tower.layers[0].self_attn.v_proj.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
    weights.add("audio_tower.layers[0].self_attn.out_proj.weight", test_utils::make_tensor(test_utils::make_seq(4 * 4, 0.01f, 0.001f), {4, 4}));
    weights.add("audio_tower.layers[0].self_attn.out_proj.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));

    weights.add("audio_tower.layers[0].self_attn_layer_norm.weight", test_utils::make_tensor(test_utils::make_seq(4, 1.0f, 0.0f), {4}));
    weights.add("audio_tower.layers[0].self_attn_layer_norm.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
    weights.add("audio_tower.layers[0].final_layer_norm.weight", test_utils::make_tensor(test_utils::make_seq(4, 1.0f, 0.0f), {4}));
    weights.add("audio_tower.layers[0].final_layer_norm.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));

    weights.add("audio_tower.layers[0].fc1.weight", test_utils::make_tensor(test_utils::make_seq(8 * 4, 0.01f, 0.001f), {8, 4}));
    weights.add("audio_tower.layers[0].fc1.bias", test_utils::make_tensor(test_utils::make_seq(8, 0.0f, 0.0f), {8}));
    weights.add("audio_tower.layers[0].fc2.weight", test_utils::make_tensor(test_utils::make_seq(4 * 8, 0.01f, 0.001f), {4, 8}));
    weights.add("audio_tower.layers[0].fc2.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));

    // Output projection
    weights.add("audio_tower.ln_post.weight", test_utils::make_tensor(test_utils::make_seq(4, 1.0f, 0.0f), {4}));
    weights.add("audio_tower.ln_post.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
    weights.add("audio_tower.proj1.weight", test_utils::make_tensor(test_utils::make_seq(4 * 4, 0.01f, 0.001f), {4, 4}));
    weights.add("audio_tower.proj1.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
    weights.add("audio_tower.proj2.weight", test_utils::make_tensor(test_utils::make_seq(6 * 4, 0.01f, 0.001f), {6, 4}));
    weights.add("audio_tower.proj2.bias", test_utils::make_tensor(test_utils::make_seq(6, 0.0f, 0.0f), {6}));

    test_utils::DummyWeightFinalizer finalizer;

    std::shared_ptr<ov::Model> model;
    EXPECT_NO_THROW(model = ov::genai::modeling::models::create_qwen3_asr_audio_encoder_model(cfg, weights, finalizer));
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->inputs().size(), 2u);
    EXPECT_EQ(model->outputs().size(), 2u);

    EXPECT_TRUE(model->output(0).get_names().count("audio_embeds") > 0);
    EXPECT_TRUE(model->output(1).get_names().count("audio_output_lengths") > 0);
    EXPECT_EQ(model->output(0).get_partial_shape().rank().get_length(), 3);
    EXPECT_EQ(model->output(1).get_partial_shape().rank().get_length(), 1);
  }

  TEST(Qwen3ASRModeling, TextModelBuildsWithThinkerPrefixedWeightsAndAudioInputs) {
    // Tiny text config for build-time smoke test.
    ov::genai::modeling::models::Qwen3ASRTextConfig cfg;
    cfg.architecture = "qwen3_asr";
    cfg.vocab_size = 8;
    cfg.hidden_size = 4;
    cfg.intermediate_size = 6;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 2;
    cfg.max_position_embeddings = 64;
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = 10000.0f;
    cfg.hidden_act = "silu";
    cfg.attention_bias = true;
    cfg.tie_word_embeddings = false;

    const size_t vocab = static_cast<size_t>(cfg.vocab_size);
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    const size_t intermediate = static_cast<size_t>(cfg.intermediate_size);
    const size_t num_kv_heads = static_cast<size_t>(cfg.num_key_value_heads);
    const size_t head_dim = static_cast<size_t>(cfg.head_dim);
    const size_t kv_hidden = num_kv_heads * head_dim;

    test_utils::DummyWeightSource weights;

    // thinker.* names verify packed mapping in create_qwen3_asr_text_model().
    weights.add("thinker.model.embed_tokens.weight",
          test_utils::make_tensor(test_utils::make_seq(vocab * hidden, 0.01f, 0.001f), {vocab, hidden}));

    weights.add("thinker.model.layers[0].input_layernorm.weight",
          test_utils::make_tensor(test_utils::make_seq(hidden, 1.0f, 0.0f), {hidden}));
    weights.add("thinker.model.layers[0].post_attention_layernorm.weight",
          test_utils::make_tensor(test_utils::make_seq(hidden, 1.0f, 0.0f), {hidden}));

    weights.add("thinker.model.layers[0].self_attn.q_proj.weight",
          test_utils::make_tensor(test_utils::make_seq(hidden * hidden, 0.01f, 0.001f), {hidden, hidden}));
    weights.add("thinker.model.layers[0].self_attn.k_proj.weight",
          test_utils::make_tensor(test_utils::make_seq(kv_hidden * hidden, 0.01f, 0.001f), {kv_hidden, hidden}));
    weights.add("thinker.model.layers[0].self_attn.v_proj.weight",
          test_utils::make_tensor(test_utils::make_seq(kv_hidden * hidden, 0.01f, 0.001f), {kv_hidden, hidden}));
    weights.add("thinker.model.layers[0].self_attn.o_proj.weight",
          test_utils::make_tensor(test_utils::make_seq(hidden * hidden, 0.01f, 0.001f), {hidden, hidden}));

    weights.add("thinker.model.layers[0].self_attn.q_proj.bias",
          test_utils::make_tensor(test_utils::make_seq(hidden, 0.0f, 0.0f), {hidden}));
    weights.add("thinker.model.layers[0].self_attn.k_proj.bias",
          test_utils::make_tensor(test_utils::make_seq(kv_hidden, 0.0f, 0.0f), {kv_hidden}));
    weights.add("thinker.model.layers[0].self_attn.v_proj.bias",
          test_utils::make_tensor(test_utils::make_seq(kv_hidden, 0.0f, 0.0f), {kv_hidden}));
    weights.add("thinker.model.layers[0].self_attn.o_proj.bias",
          test_utils::make_tensor(test_utils::make_seq(hidden, 0.0f, 0.0f), {hidden}));

    weights.add("thinker.model.layers[0].self_attn.q_norm.weight",
          test_utils::make_tensor(test_utils::make_seq(head_dim, 1.0f, 0.0f), {head_dim}));
    weights.add("thinker.model.layers[0].self_attn.k_norm.weight",
          test_utils::make_tensor(test_utils::make_seq(head_dim, 1.0f, 0.0f), {head_dim}));

    weights.add("thinker.model.layers[0].mlp.gate_proj.weight",
          test_utils::make_tensor(test_utils::make_seq(intermediate * hidden, 0.01f, 0.001f), {intermediate, hidden}));
    weights.add("thinker.model.layers[0].mlp.up_proj.weight",
          test_utils::make_tensor(test_utils::make_seq(intermediate * hidden, 0.01f, 0.001f), {intermediate, hidden}));
    weights.add("thinker.model.layers[0].mlp.down_proj.weight",
          test_utils::make_tensor(test_utils::make_seq(hidden * intermediate, 0.01f, 0.001f), {hidden, intermediate}));

    weights.add("thinker.model.norm.weight",
          test_utils::make_tensor(test_utils::make_seq(hidden, 1.0f, 0.0f), {hidden}));
    weights.add("thinker.lm_head.weight",
          test_utils::make_tensor(test_utils::make_seq(vocab * hidden, 0.01f, 0.001f), {vocab, hidden}));

    test_utils::DummyWeightFinalizer finalizer;

    std::shared_ptr<ov::Model> model;
    EXPECT_NO_THROW(model = ov::genai::modeling::models::create_qwen3_asr_text_model(
      cfg,
      weights,
      finalizer,
      false,
      true));
    ASSERT_NE(model, nullptr);

    // input_ids + attention_mask + position_ids + beam_idx + audio_embeds + audio_pos_mask
    EXPECT_EQ(model->inputs().size(), 6u);
    EXPECT_EQ(model->outputs().size(), 1u);

    EXPECT_TRUE(model->output(0).get_names().count("logits") > 0);
    EXPECT_EQ(model->output(0).get_partial_shape().rank().get_length(), 3);
  }

TEST(Qwen3ASRModeling, AudioAndTextModelInterfacesAreCompatible) {
      // Build a tiny audio encoder.
      ov::genai::modeling::models::Qwen3ASRAudioConfig audio_cfg;
      audio_cfg.architecture = "qwen3_asr_audio_encoder";
      audio_cfg.num_mel_bins = 8;
      audio_cfg.d_model = 4;
      audio_cfg.encoder_layers = 1;
      audio_cfg.encoder_attention_heads = 2;
      audio_cfg.encoder_ffn_dim = 8;
      audio_cfg.output_dim = 4;  // Must match text hidden_size for interface check.
      audio_cfg.max_source_positions = 16;
      audio_cfg.downsample_hidden_size = 2;
      audio_cfg.activation_function = "gelu";

      const size_t conv_out_in = static_cast<size_t>(audio_cfg.downsample_hidden_size) *
                                 static_cast<size_t>(((((audio_cfg.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2));

      test_utils::DummyWeightSource audio_weights;
      audio_weights.add("audio_tower.conv2d1.weight", test_utils::make_tensor(test_utils::make_seq(2 * 1 * 3 * 3, 0.01f, 0.001f), {2, 1, 3, 3}));
      audio_weights.add("audio_tower.conv2d1.bias", test_utils::make_tensor(test_utils::make_seq(2, 0.0f, 0.0f), {2}));
      audio_weights.add("audio_tower.conv2d2.weight", test_utils::make_tensor(test_utils::make_seq(2 * 2 * 3 * 3, 0.01f, 0.001f), {2, 2, 3, 3}));
      audio_weights.add("audio_tower.conv2d2.bias", test_utils::make_tensor(test_utils::make_seq(2, 0.0f, 0.0f), {2}));
      audio_weights.add("audio_tower.conv2d3.weight", test_utils::make_tensor(test_utils::make_seq(2 * 2 * 3 * 3, 0.01f, 0.001f), {2, 2, 3, 3}));
      audio_weights.add("audio_tower.conv2d3.bias", test_utils::make_tensor(test_utils::make_seq(2, 0.0f, 0.0f), {2}));
      audio_weights.add("audio_tower.conv_out.weight", test_utils::make_tensor(test_utils::make_seq(4 * conv_out_in, 0.01f, 0.001f), {4, conv_out_in}));

      audio_weights.add("audio_tower.layers[0].self_attn.q_proj.weight", test_utils::make_tensor(test_utils::make_seq(4 * 4, 0.01f, 0.001f), {4, 4}));
      audio_weights.add("audio_tower.layers[0].self_attn.q_proj.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
      audio_weights.add("audio_tower.layers[0].self_attn.k_proj.weight", test_utils::make_tensor(test_utils::make_seq(4 * 4, 0.01f, 0.001f), {4, 4}));
      audio_weights.add("audio_tower.layers[0].self_attn.k_proj.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
      audio_weights.add("audio_tower.layers[0].self_attn.v_proj.weight", test_utils::make_tensor(test_utils::make_seq(4 * 4, 0.01f, 0.001f), {4, 4}));
      audio_weights.add("audio_tower.layers[0].self_attn.v_proj.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
      audio_weights.add("audio_tower.layers[0].self_attn.out_proj.weight", test_utils::make_tensor(test_utils::make_seq(4 * 4, 0.01f, 0.001f), {4, 4}));
      audio_weights.add("audio_tower.layers[0].self_attn.out_proj.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));

      audio_weights.add("audio_tower.layers[0].self_attn_layer_norm.weight", test_utils::make_tensor(test_utils::make_seq(4, 1.0f, 0.0f), {4}));
      audio_weights.add("audio_tower.layers[0].self_attn_layer_norm.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
      audio_weights.add("audio_tower.layers[0].final_layer_norm.weight", test_utils::make_tensor(test_utils::make_seq(4, 1.0f, 0.0f), {4}));
      audio_weights.add("audio_tower.layers[0].final_layer_norm.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));

      audio_weights.add("audio_tower.layers[0].fc1.weight", test_utils::make_tensor(test_utils::make_seq(8 * 4, 0.01f, 0.001f), {8, 4}));
      audio_weights.add("audio_tower.layers[0].fc1.bias", test_utils::make_tensor(test_utils::make_seq(8, 0.0f, 0.0f), {8}));
      audio_weights.add("audio_tower.layers[0].fc2.weight", test_utils::make_tensor(test_utils::make_seq(4 * 8, 0.01f, 0.001f), {4, 8}));
      audio_weights.add("audio_tower.layers[0].fc2.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));

      audio_weights.add("audio_tower.ln_post.weight", test_utils::make_tensor(test_utils::make_seq(4, 1.0f, 0.0f), {4}));
      audio_weights.add("audio_tower.ln_post.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
      audio_weights.add("audio_tower.proj1.weight", test_utils::make_tensor(test_utils::make_seq(4 * 4, 0.01f, 0.001f), {4, 4}));
      audio_weights.add("audio_tower.proj1.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));
      audio_weights.add("audio_tower.proj2.weight", test_utils::make_tensor(test_utils::make_seq(4 * 4, 0.01f, 0.001f), {4, 4}));
      audio_weights.add("audio_tower.proj2.bias", test_utils::make_tensor(test_utils::make_seq(4, 0.0f, 0.0f), {4}));

      test_utils::DummyWeightFinalizer audio_finalizer;
      auto audio_model = ov::genai::modeling::models::create_qwen3_asr_audio_encoder_model(audio_cfg, audio_weights, audio_finalizer);
      ASSERT_NE(audio_model, nullptr);

      // Build a tiny text model expecting audio embeds with hidden_size=4.
      ov::genai::modeling::models::Qwen3ASRTextConfig text_cfg;
      text_cfg.architecture = "qwen3_asr";
      text_cfg.vocab_size = 8;
      text_cfg.hidden_size = 4;
      text_cfg.intermediate_size = 6;
      text_cfg.num_hidden_layers = 1;
      text_cfg.num_attention_heads = 2;
      text_cfg.num_key_value_heads = 1;
      text_cfg.head_dim = 2;
      text_cfg.max_position_embeddings = 64;
      text_cfg.rms_norm_eps = 1e-6f;
      text_cfg.rope_theta = 10000.0f;
      text_cfg.hidden_act = "silu";
      text_cfg.attention_bias = true;
      text_cfg.tie_word_embeddings = false;

      const size_t vocab = static_cast<size_t>(text_cfg.vocab_size);
      const size_t hidden = static_cast<size_t>(text_cfg.hidden_size);
      const size_t intermediate = static_cast<size_t>(text_cfg.intermediate_size);
      const size_t kv_hidden = static_cast<size_t>(text_cfg.num_key_value_heads * text_cfg.head_dim);

      test_utils::DummyWeightSource text_weights;
      text_weights.add("thinker.model.embed_tokens.weight",
                               test_utils::make_tensor(test_utils::make_seq(vocab * hidden, 0.01f, 0.001f), {vocab, hidden}));
      text_weights.add("thinker.model.layers[0].input_layernorm.weight",
                               test_utils::make_tensor(test_utils::make_seq(hidden, 1.0f, 0.0f), {hidden}));
      text_weights.add("thinker.model.layers[0].post_attention_layernorm.weight",
                               test_utils::make_tensor(test_utils::make_seq(hidden, 1.0f, 0.0f), {hidden}));
      text_weights.add("thinker.model.layers[0].self_attn.q_proj.weight",
                               test_utils::make_tensor(test_utils::make_seq(hidden * hidden, 0.01f, 0.001f), {hidden, hidden}));
      text_weights.add("thinker.model.layers[0].self_attn.k_proj.weight",
                               test_utils::make_tensor(test_utils::make_seq(kv_hidden * hidden, 0.01f, 0.001f), {kv_hidden, hidden}));
      text_weights.add("thinker.model.layers[0].self_attn.v_proj.weight",
                               test_utils::make_tensor(test_utils::make_seq(kv_hidden * hidden, 0.01f, 0.001f), {kv_hidden, hidden}));
      text_weights.add("thinker.model.layers[0].self_attn.o_proj.weight",
                               test_utils::make_tensor(test_utils::make_seq(hidden * hidden, 0.01f, 0.001f), {hidden, hidden}));
      text_weights.add("thinker.model.layers[0].self_attn.q_proj.bias", test_utils::make_tensor(test_utils::make_seq(hidden, 0.0f, 0.0f), {hidden}));
      text_weights.add("thinker.model.layers[0].self_attn.k_proj.bias", test_utils::make_tensor(test_utils::make_seq(kv_hidden, 0.0f, 0.0f), {kv_hidden}));
      text_weights.add("thinker.model.layers[0].self_attn.v_proj.bias", test_utils::make_tensor(test_utils::make_seq(kv_hidden, 0.0f, 0.0f), {kv_hidden}));
      text_weights.add("thinker.model.layers[0].self_attn.o_proj.bias", test_utils::make_tensor(test_utils::make_seq(hidden, 0.0f, 0.0f), {hidden}));
      text_weights.add("thinker.model.layers[0].self_attn.q_norm.weight", test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(text_cfg.head_dim), 1.0f, 0.0f), {static_cast<size_t>(text_cfg.head_dim)}));
      text_weights.add("thinker.model.layers[0].self_attn.k_norm.weight", test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(text_cfg.head_dim), 1.0f, 0.0f), {static_cast<size_t>(text_cfg.head_dim)}));
      text_weights.add("thinker.model.layers[0].mlp.gate_proj.weight",
                               test_utils::make_tensor(test_utils::make_seq(intermediate * hidden, 0.01f, 0.001f), {intermediate, hidden}));
      text_weights.add("thinker.model.layers[0].mlp.up_proj.weight",
                               test_utils::make_tensor(test_utils::make_seq(intermediate * hidden, 0.01f, 0.001f), {intermediate, hidden}));
      text_weights.add("thinker.model.layers[0].mlp.down_proj.weight",
                               test_utils::make_tensor(test_utils::make_seq(hidden * intermediate, 0.01f, 0.001f), {hidden, intermediate}));
      text_weights.add("thinker.model.norm.weight", test_utils::make_tensor(test_utils::make_seq(hidden, 1.0f, 0.0f), {hidden}));
      text_weights.add("thinker.lm_head.weight",
                               test_utils::make_tensor(test_utils::make_seq(vocab * hidden, 0.01f, 0.001f), {vocab, hidden}));

      test_utils::DummyWeightFinalizer text_finalizer;
      auto text_model = ov::genai::modeling::models::create_qwen3_asr_text_model(text_cfg, text_weights, text_finalizer, false, true);
      ASSERT_NE(text_model, nullptr);

      auto get_input_shape_by_name = [](const std::shared_ptr<ov::Model>& m,
                                                        const std::string& tensor_name) -> ov::PartialShape {
            for (const auto& input : m->inputs()) {
                  if (input.get_names().count(tensor_name) > 0) {
                        return input.get_partial_shape();
                  }
            }
            return {};
      };

      const ov::PartialShape audio_embeds_out = audio_model->output("audio_embeds").get_partial_shape();
      const ov::PartialShape audio_embeds_in = get_input_shape_by_name(text_model, "audio_embeds");
      const ov::PartialShape audio_pos_mask_in = get_input_shape_by_name(text_model, "audio_pos_mask");

      ASSERT_TRUE(audio_embeds_out.rank().is_static());
      ASSERT_TRUE(audio_embeds_in.rank().is_static());
      ASSERT_TRUE(audio_pos_mask_in.rank().is_static());

      EXPECT_EQ(audio_embeds_out.rank().get_length(), 3);
      EXPECT_EQ(audio_embeds_in.rank().get_length(), 3);
      EXPECT_EQ(audio_pos_mask_in.rank().get_length(), 2);

      ASSERT_TRUE(audio_embeds_out[2].is_static());
      ASSERT_TRUE(audio_embeds_in[2].is_static());
      EXPECT_EQ(audio_embeds_out[2].get_length(), audio_embeds_in[2].get_length());
}
