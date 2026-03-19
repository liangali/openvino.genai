// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"

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

TEST(Qwen3_5DummyINT4Binding, BuildsTextAndVisionWithQuantizedFinalizer) {
    const auto cfg = make_small_cfg();
    auto specs = ov::genai::modeling::models::build_qwen3_5_vlm_weight_specs(cfg);
    ov::genai::modeling::weights::SyntheticWeightSource source(std::move(specs), 42u, -0.02f, 0.02f);

    ov::genai::modeling::weights::QuantizationConfig qcfg;
    qcfg.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
    qcfg.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
    qcfg.group_size = 8;
    qcfg.selection.exclude_patterns.clear();

    std::shared_ptr<ov::Model> text_model;
    std::shared_ptr<ov::Model> vision_model;
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer(qcfg);
        EXPECT_NO_THROW(text_model = ov::genai::modeling::models::create_qwen3_5_text_model(
                            cfg,
                            source,
                            text_finalizer,
                            false,
                            true,
                            false));
    }
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer(qcfg);
        EXPECT_NO_THROW(vision_model = ov::genai::modeling::models::create_qwen3_5_vision_model(
                            cfg,
                            source,
                            vision_finalizer));
    }

    ASSERT_NE(text_model, nullptr);
    ASSERT_NE(vision_model, nullptr);

    ov::Core core;
    EXPECT_NO_THROW((void)core.compile_model(text_model, "GPU"));
    EXPECT_NO_THROW((void)core.compile_model(vision_model, "GPU"));
}
