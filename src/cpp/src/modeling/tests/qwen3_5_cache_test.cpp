// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"
#include "utils.hpp"

namespace {

ov::genai::modeling::models::Qwen3_5Config make_small_cfg() {
    using namespace ov::genai::modeling::models;
    Qwen3_5Config cfg;
    cfg.model_type = "qwen3_5";

    cfg.text.model_type = "qwen3_5_text";
    cfg.text.vocab_size = 64;
    cfg.text.hidden_size = 16;
    cfg.text.intermediate_size = 32;
    cfg.text.num_hidden_layers = 2;
    cfg.text.num_attention_heads = 4;
    cfg.text.num_key_value_heads = 2;
    cfg.text.head_dim = 4;
    cfg.text.partial_rotary_factor = 0.5f;
    cfg.text.max_position_embeddings = 256;
    cfg.text.layer_types = {"linear_attention", "full_attention"};
    cfg.text.linear_conv_kernel_dim = 2;
    cfg.text.linear_key_head_dim = 4;
    cfg.text.linear_value_head_dim = 4;
    cfg.text.linear_num_key_heads = 2;
    cfg.text.linear_num_value_heads = 2;
    cfg.text.rope.mrope_interleaved = true;
    cfg.text.rope.mrope_section = {1, 1, 0};

    cfg.vision.model_type = "qwen3_5";
    cfg.vision.depth = 1;
    cfg.vision.hidden_size = 8;
    cfg.vision.intermediate_size = 16;
    cfg.vision.num_heads = 2;
    cfg.vision.in_channels = 3;
    cfg.vision.patch_size = 2;
    cfg.vision.temporal_patch_size = 1;
    cfg.vision.spatial_merge_size = 2;
    cfg.vision.out_hidden_size = cfg.text.hidden_size;
    cfg.vision.num_position_embeddings = 16;
    cfg.vision.deepstack_visual_indexes.clear();

    cfg.image_token_id = 7;
    cfg.video_token_id = 8;
    cfg.vision_start_token_id = 9;
    cfg.vision_end_token_id = 10;

    cfg.finalize();
    cfg.validate();
    return cfg;
}

bool is_attention_kv_state(const std::string& name) {
    return name.find("past_key_values.") != std::string::npos ||
           name.find(".key_cache") != std::string::npos ||
           name.find(".value_cache") != std::string::npos;
}

}  // namespace

TEST(Qwen3_5CacheRuntime, TrimsOnlyAttentionStatesAndKeepsLinearStates) {
    namespace tests = ov::genai::modeling::tests;
    const auto cfg = make_small_cfg();

    auto specs = ov::genai::modeling::models::build_qwen3_5_text_weight_specs(cfg.text);
    ov::genai::modeling::weights::SyntheticWeightSource source(std::move(specs), 3001u, -0.02f, 0.02f);
    tests::DummyWeightFinalizer finalizer;

    auto text_model = ov::genai::modeling::models::create_qwen3_5_text_model(cfg, source, finalizer, false, false, false);
    ov::Core core;
    auto compiled = core.compile_model(text_model, "GPU");
    auto request = compiled.create_infer_request();

    ov::Tensor input_ids(ov::element::i64, {1, 3});
    auto* ids = input_ids.data<int64_t>();
    ids[0] = 11;
    ids[1] = 12;
    ids[2] = 13;

    ov::Tensor attention_mask(ov::element::i64, {1, 3});
    auto* attn = attention_mask.data<int64_t>();
    attn[0] = 1;
    attn[1] = 1;
    attn[2] = 1;

    ov::genai::modeling::models::Qwen3_5InputPlanner planner(cfg);
    auto plan = planner.build_plan(input_ids, &attention_mask, nullptr);

    ov::Tensor beam_idx(ov::element::i32, {1});
    beam_idx.data<int32_t>()[0] = 0;

    request.reset_state();
    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, input_ids);
    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, attention_mask);
    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, plan.position_ids);
    request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, beam_idx);
    request.infer();

    auto kv_axes = ov::genai::utils::get_kv_axes_pos(text_model);
    EXPECT_EQ(kv_axes.batch, 0u);
    EXPECT_EQ(kv_axes.seq_len, 2u);

    auto states = request.query_state();
    ASSERT_FALSE(states.empty());

    bool has_linear = false;
    bool has_attention = false;
    size_t linear_seq_before = 0;
    size_t attention_seq_before = 0;

    for (auto& state : states) {
        const auto name = state.get_name();
        const auto shape = state.get_state().get_shape();
        if (name.find("linear_states.") != std::string::npos && name.find(".conv") != std::string::npos) {
            has_linear = true;
            ASSERT_EQ(shape.size(), 3u);
            linear_seq_before = shape[2];
        }
        if (is_attention_kv_state(name)) {
            has_attention = true;
            ASSERT_EQ(shape.size(), 4u);
            attention_seq_before = shape[kv_axes.seq_len];
        }
    }

    EXPECT_TRUE(has_linear);
    EXPECT_TRUE(has_attention);
    ASSERT_GT(attention_seq_before, 0u);

    ov::genai::utils::KVCacheState kv_cache_state;
    kv_cache_state.seq_length_axis = kv_axes.seq_len;
    kv_cache_state.num_tokens_to_trim = 1;
    ov::genai::utils::trim_kv_cache(request, kv_cache_state, std::nullopt);

    size_t linear_seq_after = 0;
    size_t attention_seq_after = 0;
    states = request.query_state();
    for (auto& state : states) {
        const auto name = state.get_name();
        const auto shape = state.get_state().get_shape();
        if (name.find("linear_states.") != std::string::npos && name.find(".conv") != std::string::npos) {
            linear_seq_after = shape[2];
        }
        if (is_attention_kv_state(name)) {
            attention_seq_after = shape[kv_axes.seq_len];
        }
    }

    EXPECT_EQ(linear_seq_after, linear_seq_before);
    EXPECT_EQ(attention_seq_after, attention_seq_before - 1);
}
