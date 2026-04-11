// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <utility>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"

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
    cfg.vision.depth = 2;
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

}  // namespace

TEST(Qwen3_5VLME2E, DummyPrefillAndDecode) {
    namespace tests = ov::genai::modeling::tests;
    const auto cfg = make_small_cfg();
    auto specs = ov::genai::modeling::models::build_qwen3_5_vlm_weight_specs(cfg);
    ov::genai::modeling::weights::SyntheticWeightSource source(std::move(specs), 2028u, -0.02f, 0.02f);
    tests::DummyWeightFinalizer finalizer;

    auto vision_model = ov::genai::modeling::models::create_qwen3_5_vision_model(cfg, source, finalizer);
    auto text_model = ov::genai::modeling::models::create_qwen3_5_text_model(cfg, source, finalizer, false, true, false);

    ov::Core core;
    auto compiled_vision = core.compile_model(vision_model, "GPU");
    auto compiled_text = core.compile_model(text_model, "GPU");

    ov::Tensor pixel_values(ov::element::f32, {4, 3, 1, 2, 2});
    std::memset(pixel_values.data(), 0, pixel_values.get_byte_size());
    ov::Tensor grid_thw(ov::element::i64, {1, 3});
    auto* grid = grid_thw.data<int64_t>();
    grid[0] = 1;
    grid[1] = 2;
    grid[2] = 2;
    ov::Tensor pos_embeds(ov::element::f32, {4, static_cast<size_t>(cfg.vision.hidden_size)});
    std::memset(pos_embeds.data(), 0, pos_embeds.get_byte_size());
    ov::Tensor rotary_cos(ov::element::f32, {4, static_cast<size_t>(cfg.vision.head_dim())});
    ov::Tensor rotary_sin(ov::element::f32, {4, static_cast<size_t>(cfg.vision.head_dim())});
    std::memset(rotary_cos.data(), 0, rotary_cos.get_byte_size());
    std::memset(rotary_sin.data(), 0, rotary_sin.get_byte_size());

    auto vision_request = compiled_vision.create_infer_request();
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kPixelValues, pixel_values);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kGridThw, grid_thw);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kPosEmbeds, pos_embeds);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kRotaryCos, rotary_cos);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kRotarySin, rotary_sin);
    vision_request.infer();

    ov::Tensor visual_embeds = vision_request.get_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kVisualEmbeds);
    ASSERT_EQ(visual_embeds.get_shape().size(), 2u);
    ASSERT_EQ(visual_embeds.get_shape()[0], 1u);

    ov::Tensor input_ids(ov::element::i64, {1, 3});
    auto* ids = input_ids.data<int64_t>();
    ids[0] = 1;
    ids[1] = cfg.image_token_id;
    ids[2] = 2;
    ov::Tensor attention_mask(ov::element::i64, {1, 3});
    auto* attn = attention_mask.data<int64_t>();
    attn[0] = 1;
    attn[1] = 1;
    attn[2] = 1;

    ov::genai::modeling::models::Qwen3_5InputPlanner planner(cfg);
    auto plan = planner.build_plan(input_ids, &attention_mask, &grid_thw);
    auto visual_padded =
        ov::genai::modeling::models::Qwen3_5InputPlanner::scatter_visual_embeds(visual_embeds, plan.visual_pos_mask);

    ov::Tensor beam_idx(ov::element::i32, {1});
    beam_idx.data<int32_t>()[0] = 0;

    auto text_request = compiled_text.create_infer_request();
    text_request.reset_state();
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, input_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, attention_mask);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, plan.position_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, beam_idx);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, visual_padded);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, plan.visual_pos_mask);
    text_request.infer();

    ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
    ASSERT_EQ(logits.get_shape().size(), 3u);
    ASSERT_EQ(logits.get_shape()[0], 1u);
    ASSERT_EQ(logits.get_shape()[1], 3u);
    ASSERT_EQ(logits.get_shape()[2], static_cast<size_t>(cfg.text.vocab_size));

    ov::Tensor step_ids(ov::element::i64, {1, 1});
    step_ids.data<int64_t>()[0] = 3;
    ov::Tensor step_mask(ov::element::i64, {1, 1});
    step_mask.data<int64_t>()[0] = 1;
    auto decode_pos =
        ov::genai::modeling::models::Qwen3_5InputPlanner::build_decode_position_ids(plan.rope_deltas, 3, 1);

    ov::Tensor zero_visual(ov::element::f32, {1, 1, static_cast<size_t>(cfg.text.hidden_size)});
    ov::Tensor zero_visual_mask(ov::element::boolean, {1, 1});
    std::memset(zero_visual.data(), 0, zero_visual.get_byte_size());
    std::memset(zero_visual_mask.data(), 0, zero_visual_mask.get_byte_size());

    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, step_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, step_mask);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, decode_pos);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, beam_idx);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, zero_visual);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, zero_visual_mask);
    text_request.infer();

    logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
    ASSERT_EQ(logits.get_shape().size(), 3u);
    ASSERT_EQ(logits.get_shape()[0], 1u);
    ASSERT_EQ(logits.get_shape()[1], 1u);
    ASSERT_EQ(logits.get_shape()[2], static_cast<size_t>(cfg.text.vocab_size));
}
