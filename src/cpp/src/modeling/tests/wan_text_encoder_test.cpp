// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/wan/modeling_wan_umt5.hpp"
#include "modeling/models/wan/processing_wan.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

void add_zero_weight(test_utils::DummyWeightSource& source,
                     const std::string& name,
                     const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::memset(tensor.data(), 0, tensor.get_byte_size());
    source.add(name, tensor);
}

void add_one_weight(test_utils::DummyWeightSource& source,
                    const std::string& name,
                    const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::fill_n(tensor.data<float>(), tensor.get_size(), 1.0f);
    source.add(name, tensor);
}

}  // namespace

TEST(WanTextEncoder, BuildsAndRuns) {
    const size_t batch = 1;
    const size_t seq_len = 512;
    const size_t vocab = 8;
    const size_t hidden = 4;
    const size_t num_heads = 2;
    const size_t head_dim = 2;
    const size_t intermediate = 8;
    const size_t num_layers = 1;
    const size_t num_buckets = 4;

    test_utils::DummyWeightSource weights;
    test_utils::DummyWeightFinalizer finalizer;

    add_zero_weight(weights, "shared.weight", {vocab, hidden});

    add_one_weight(weights, "encoder.block.0.layer.0.layer_norm.weight", {hidden});
    add_zero_weight(weights, "encoder.block.0.layer.0.SelfAttention.q.weight", {num_heads * head_dim, hidden});
    add_zero_weight(weights, "encoder.block.0.layer.0.SelfAttention.k.weight", {num_heads * head_dim, hidden});
    add_zero_weight(weights, "encoder.block.0.layer.0.SelfAttention.v.weight", {num_heads * head_dim, hidden});
    add_zero_weight(weights, "encoder.block.0.layer.0.SelfAttention.o.weight", {hidden, num_heads * head_dim});
    add_zero_weight(weights, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                    {num_buckets, num_heads});

    add_one_weight(weights, "encoder.block.0.layer.1.layer_norm.weight", {hidden});
    add_zero_weight(weights, "encoder.block.0.layer.1.DenseReluDense.wi_0.weight", {intermediate, hidden});
    add_zero_weight(weights, "encoder.block.0.layer.1.DenseReluDense.wi_1.weight", {intermediate, hidden});
    add_zero_weight(weights, "encoder.block.0.layer.1.DenseReluDense.wo.weight", {hidden, intermediate});

    add_one_weight(weights, "encoder.final_layer_norm.weight", {hidden});

    ov::genai::modeling::models::UMT5Config cfg;
    cfg.vocab_size = static_cast<int32_t>(vocab);
    cfg.d_model = static_cast<int32_t>(hidden);
    cfg.d_ff = static_cast<int32_t>(intermediate);
    cfg.d_kv = static_cast<int32_t>(head_dim);
    cfg.num_heads = static_cast<int32_t>(num_heads);
    cfg.num_layers = static_cast<int32_t>(num_layers);
    cfg.relative_attention_num_buckets = static_cast<int32_t>(num_buckets);
    cfg.relative_attention_max_distance = 8;
    cfg.layer_norm_epsilon = 1e-6f;
    cfg.dense_act_fn = "gelu_new";
    cfg.is_gated_act = true;
    cfg.finalize();
    cfg.validate();

    auto model = ov::genai::modeling::models::create_umt5_text_encoder_model(cfg, weights, finalizer);

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<int64_t> input_ids_data(seq_len, 1);
    std::vector<int64_t> attention_mask_data(seq_len, 1);

    ov::Tensor input_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(input_ids_tensor.data(), input_ids_data.data(), input_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(0, input_ids_tensor);

    ov::Tensor attention_mask_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(attention_mask_tensor.data(), attention_mask_data.data(), attention_mask_data.size() * sizeof(int64_t));
    request.set_input_tensor(1, attention_mask_tensor);

    request.infer();

    std::vector<float> expected(batch * seq_len * hidden, 0.0f);
    test_utils::expect_tensor_near(request.get_output_tensor(0), expected, test_utils::k_tol_default);
}

TEST(WanPromptClean, HandlesHtmlAndWhitespace) {
    const std::string input = "  Hello   world &amp; test  ";
    const std::string output = ov::genai::modeling::models::prompt_clean(input);
    EXPECT_EQ(output, "Hello world & test");
}
