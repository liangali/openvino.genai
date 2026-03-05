// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include "modeling/builder_context.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

// MoE tests must share a single ov::Core (and therefore a single GPU engine/
// OpenCL context).  The oneDNN library maintains a process-wide primitive cache
// that stores compiled cl_program objects bound to a specific cl_context.  When
// successive MoE tests each create their own ov::Core they get different
// cl_context instances; test N finds stale cl_program entries (compiled for an
// earlier, already-destroyed context) in the oneDNN cache and crashes with
// CL_OUT_OF_RESOURCES.  Sharing one Core keeps a single context alive for the
// lifetime of the test binary so every cache lookup returns a valid primitive.
static ov::Core& get_moe_test_core() {
    static ov::Core core;
    return core;
}

TEST(Ops, Matmul) {
    ov::genai::modeling::BuilderContext ctx;

    auto a = ctx.parameter("a", ov::element::f32, ov::Shape{2, 3});
    auto b = ctx.parameter("b", ov::element::f32, ov::Shape{3, 2});

    auto out = ov::genai::modeling::ops::matmul(a, b);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto a_data = test_utils::make_seq(6, 1.0f, 1.0f);
    auto b_data = test_utils::make_seq(6, 7.0f, 1.0f);
    ov::Tensor a_tensor(ov::element::f32, {2, 3});
    ov::Tensor b_tensor(ov::element::f32, {3, 2});
    std::memcpy(a_tensor.data(), a_data.data(), a_data.size() * sizeof(float));
    std::memcpy(b_tensor.data(), b_data.data(), b_data.size() * sizeof(float));

    request.set_input_tensor(0, a_tensor);
    request.set_input_tensor(1, b_tensor);
    request.infer();

    auto expected = test_utils::matmul_ref(a_data, b_data, 2, 3, 2);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(Ops, MatmulTransposeA) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t m = 2;
    const size_t k = 3;
    const size_t n = 4;

    auto a = ctx.parameter("a", ov::element::f32, ov::Shape{k, m});
    auto b = ctx.parameter("b", ov::element::f32, ov::Shape{k, n});

    auto out = ov::genai::modeling::ops::matmul(a, b, true, false);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto a_data = test_utils::make_seq(k * m, 0.5f, 0.5f);
    auto b_data = test_utils::make_seq(k * n, 1.0f, 0.25f);
    ov::Tensor a_tensor(ov::element::f32, {k, m});
    ov::Tensor b_tensor(ov::element::f32, {k, n});
    std::memcpy(a_tensor.data(), a_data.data(), a_data.size() * sizeof(float));
    std::memcpy(b_tensor.data(), b_data.data(), b_data.size() * sizeof(float));

    request.set_input_tensor(0, a_tensor);
    request.set_input_tensor(1, b_tensor);
    request.infer();

    auto expected = test_utils::matmul_ref_transpose_a(a_data, b_data, m, k, n);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(Ops, Linear) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t in_features = 3;
    const size_t out_features = 4;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{batch, seq_len, in_features});
    auto w = ctx.parameter("w", ov::element::f32, ov::Shape{out_features, in_features});

    auto out = ov::genai::modeling::ops::linear(x, w);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto x_data = test_utils::make_seq(batch * seq_len * in_features, 0.1f, 0.1f);
    auto w_data = test_utils::make_seq(out_features * in_features, 0.2f, 0.05f);
    ov::Tensor x_tensor(ov::element::f32, {batch, seq_len, in_features});
    ov::Tensor w_tensor(ov::element::f32, {out_features, in_features});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    std::memcpy(w_tensor.data(), w_data.data(), w_data.size() * sizeof(float));

    request.set_input_tensor(0, x_tensor);
    request.set_input_tensor(1, w_tensor);
    request.infer();

    auto expected = test_utils::linear_ref_3d(x_data, w_data, batch, seq_len, in_features, out_features);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(Ops, LinearAttention) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 2;
    const size_t seq_len = 16;
    const size_t num_heads = 16;
    const size_t head_dim = 128;

    auto q = ctx.parameter("q", ov::element::f32, ov::Shape{batch, seq_len, num_heads, head_dim});
    auto k = ctx.parameter("k", ov::element::f32, ov::Shape{batch, seq_len, num_heads, head_dim});
    auto v = ctx.parameter("v", ov::element::f32, ov::Shape{batch, seq_len, num_heads, head_dim});
    auto g = ctx.parameter("g", ov::element::f32, ov::Shape{batch, seq_len, num_heads});
    auto beta = ctx.parameter("beta", ov::element::f32, ov::Shape{batch, seq_len, num_heads});
    auto init_state = ctx.parameter("init_state", ov::element::f32, ov::Shape{batch, num_heads, head_dim, head_dim});

    auto out_pair = ov::genai::modeling::ops::linear_attention(q, k, v, beta, g, init_state);
    auto model = ctx.build_model({out_pair.first.output(), out_pair.second.output()});
    ov::serialize(model, "linear_attention_original.xml");
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    ov::serialize(compiled.get_runtime_model(), "linear_attention_compiled.xml");
    auto request = compiled.create_infer_request();

    auto q_data = test_utils::random_f32(batch * seq_len * num_heads * head_dim, -0.5f, 0.5f, 11);
    auto k_data = test_utils::random_f32(batch * seq_len * num_heads * head_dim, -0.5f, 0.5f, 23);
    auto v_data = test_utils::random_f32(batch * seq_len * num_heads * head_dim, -0.5f, 0.5f, 37);
    auto g_data = test_utils::random_f32(batch * seq_len * num_heads, -0.5f, 0.0f, 41);       // exp(g) in (0.6, 1]
    auto beta_data = test_utils::random_f32(batch * seq_len * num_heads, 0.0f, 0.3f, 53);     // small learning rate
    auto init_state_data = test_utils::random_f32(batch * num_heads * head_dim * head_dim, -0.01f, 0.01f, 59);

    ov::Tensor q_tensor(ov::element::f32, {batch, seq_len, num_heads, head_dim});
    ov::Tensor k_tensor(ov::element::f32, {batch, seq_len, num_heads, head_dim});
    ov::Tensor v_tensor(ov::element::f32, {batch, seq_len, num_heads, head_dim});
    ov::Tensor beta_tensor(ov::element::f32, {batch, seq_len, num_heads});
    ov::Tensor g_tensor(ov::element::f32, {batch, seq_len, num_heads});
    ov::Tensor init_state_tensor(ov::element::f32, {batch, num_heads, head_dim, head_dim});
    std::memcpy(q_tensor.data(), q_data.data(), q_data.size() * sizeof(float));
    std::memcpy(k_tensor.data(), k_data.data(), k_data.size() * sizeof(float));
    std::memcpy(v_tensor.data(), v_data.data(), v_data.size() * sizeof(float));
    std::memcpy(beta_tensor.data(), beta_data.data(), beta_data.size() * sizeof(float));
    std::memcpy(g_tensor.data(), g_data.data(), g_data.size() * sizeof(float));
    std::memcpy(init_state_tensor.data(), init_state_data.data(), init_state_data.size() * sizeof(float));

    request.set_input_tensor(0, q_tensor);
    request.set_input_tensor(1, k_tensor);
    request.set_input_tensor(2, v_tensor);
    request.set_input_tensor(3, beta_tensor);
    request.set_input_tensor(4, g_tensor);
    request.set_input_tensor(5, init_state_tensor);
    request.infer();

    auto expected_pair = test_utils::linear_attention_ref(q_data,
                                                          k_data,
                                                          v_data,
                                                          beta_data,
                                                          g_data,
                                                          init_state_data,
                                                          batch,
                                                          seq_len,
                                                          num_heads,
                                                          head_dim);

    test_utils::expect_tensor_near(request.get_output_tensor(0), expected_pair.first, test_utils::k_tol_linear_attn);
    test_utils::expect_tensor_near(request.get_output_tensor(1), expected_pair.second, test_utils::k_tol_linear_attn);
}

TEST(Ops, ConstHelpers) {
    ov::genai::modeling::BuilderContext ctx;
    auto* op_ctx = &ctx.op_context();

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2});

    auto two = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_scalar(op_ctx, 2.0f), op_ctx);
    auto scaled = x * two;

    auto idx =
        ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_vec(op_ctx, std::vector<int64_t>{1}), op_ctx);
    auto picked = ov::genai::modeling::ops::gather(x, idx, 0);

    auto model = ctx.build_model({scaled.output(), picked.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {3.0f, 5.0f};
    ov::Tensor x_tensor(ov::element::f32, {2});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    test_utils::expect_tensor_near(request.get_output_tensor(0), {6.0f, 10.0f}, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(1), {5.0f}, test_utils::k_tol_default);
}

TEST(Ops, ReduceMean) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});
    auto out = ov::genai::modeling::ops::reduce_mean(x, 1, false);

    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto x_data = test_utils::make_seq(6, 1.0f, 1.0f);
    ov::Tensor x_tensor(ov::element::f32, {2, 3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    auto expected = test_utils::mean_ref(x_data, 2, 3);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(Ops, GatherSliceConcat) {
    ov::genai::modeling::BuilderContext ctx;

    auto data = ctx.parameter("data", ov::element::f32, ov::Shape{2, 5});
    auto indices = ctx.parameter("indices", ov::element::i64, ov::Shape{2});

    auto gathered = ov::genai::modeling::ops::gather(data, indices, 1);
    auto sliced = ov::genai::modeling::ops::slice(data, 1, 5, 2, 1);
    auto concatenated = ov::genai::modeling::ops::concat({gathered, sliced}, 1);

    auto model = ctx.build_model({gathered.output(), sliced.output(), concatenated.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto data_values = test_utils::make_seq(10, 0.0f, 1.0f);
    ov::Tensor data_tensor(ov::element::f32, {2, 5});
    std::memcpy(data_tensor.data(), data_values.data(), data_values.size() * sizeof(float));

    std::vector<int64_t> indices_values = {0, 2};
    ov::Tensor idx_tensor(ov::element::i64, {2});
    std::memcpy(idx_tensor.data(), indices_values.data(), indices_values.size() * sizeof(int64_t));

    request.set_input_tensor(0, data_tensor);
    request.set_input_tensor(1, idx_tensor);
    request.infer();

    std::vector<float> expected_gather = {0.0f, 2.0f, 5.0f, 7.0f};
    std::vector<float> expected_slice = {1.0f, 3.0f, 6.0f, 8.0f};
    std::vector<float> expected_concat = {0.0f, 2.0f, 1.0f, 3.0f, 5.0f, 7.0f, 6.0f, 8.0f};

    test_utils::expect_tensor_near(request.get_output_tensor(0), expected_gather, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(1), expected_slice, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(2), expected_concat, test_utils::k_tol_default);
}

TEST(Ops, Rms) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});
    auto w = ctx.parameter("w", ov::element::f32, ov::Shape{3});

    const float eps = 1.0f;
    auto out = ov::genai::modeling::ops::rms(x, w, eps);

    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto x_data = test_utils::make_seq(6, 1.0f, 1.0f);
    auto w_data = std::vector<float>{0.5f, 1.0f, 1.5f};
    ov::Tensor x_tensor(ov::element::f32, {2, 3});
    ov::Tensor w_tensor(ov::element::f32, {3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    std::memcpy(w_tensor.data(), w_data.data(), w_data.size() * sizeof(float));

    request.set_input_tensor(0, x_tensor);
    request.set_input_tensor(1, w_tensor);
    request.infer();

    auto expected = test_utils::rms_ref(x_data, w_data, 2, 3, eps);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(Ops, Moe3GemmFusedCompressed) {
    ov::genai::modeling::BuilderContext ctx;

    constexpr size_t batch = 1;
    constexpr size_t seq_len = 16;
    constexpr size_t hidden_size = 1024;
    constexpr size_t inter_size = 2048;
    constexpr size_t num_experts = 8;
    constexpr size_t top_k = 4;
    constexpr size_t group_size = 128;

    static_assert(hidden_size % group_size == 0, "hidden_size must be divisible by group_size");
    static_assert(inter_size % group_size == 0, "inter_size must be divisible by group_size");

    const size_t tokens = batch * seq_len;
    auto hidden_param = ctx.parameter("hidden", ov::element::f32, ov::Shape{tokens, hidden_size});

    auto hidden_states = test_utils::random_f32(tokens * hidden_size, -0.5f, 0.5f, 11);
    auto gate_inp = test_utils::random_f32(num_experts * hidden_size, -0.5f, 0.5f, 23);

    auto gate_w_f32 = test_utils::random_f32(num_experts * inter_size * hidden_size, -0.5f, 0.5f, 31);
    auto up_w_f32 = test_utils::random_f32(num_experts * inter_size * hidden_size, -0.5f, 0.5f, 37);
    auto down_w_f32 = test_utils::random_f32(num_experts * hidden_size * inter_size, -0.5f, 0.5f, 41);
    // Scale expert FFN weights to prevent FP16 overflow in intermediate buffers.
    // Without scaling: gate_proj_max = 1024 * 0.5^2 = 256, SwiGLU_max = 256^2 = 65536 > FP16_MAX(65504).
    // With /14: gate_proj_max ~18, SwiGLU_max ~324, down_out_max ~2048*324/14 ~47k < FP16_MAX.
    for (auto& v : gate_w_f32) v /= 14.0f;
    for (auto& v : up_w_f32)   v /= 14.0f;
    for (auto& v : down_w_f32) v /= 14.0f;

    auto q_gate = test_utils::quantize_q41(gate_w_f32, num_experts, inter_size, hidden_size, group_size);
    auto q_up = test_utils::quantize_q41(up_w_f32, num_experts, inter_size, hidden_size, group_size);
    auto q_down = test_utils::quantize_q41(down_w_f32, num_experts, hidden_size, inter_size, group_size);

    auto gate_w_deq = test_utils::dequantize_q41(q_gate, num_experts, inter_size, hidden_size);
    auto up_w_deq = test_utils::dequantize_q41(q_up, num_experts, inter_size, hidden_size);
    auto down_w_deq = test_utils::dequantize_q41(q_down, num_experts, hidden_size, inter_size);

    auto* op_ctx = &ctx.op_context();
    auto gate_inp_tensor = test_utils::make_tensor(gate_inp, {num_experts, hidden_size});

    auto gate_inp_const = ov::genai::modeling::ops::constant(gate_inp_tensor, op_ctx);
    auto gate_exps_weight = ov::genai::modeling::ops::constant(q_gate.weights_u4, op_ctx);
    auto gate_exps_scales = ov::genai::modeling::ops::constant(q_gate.scales_f16, op_ctx);
    auto gate_exps_zps = ov::genai::modeling::ops::constant(q_gate.zps_u4, op_ctx);
    auto up_exps_weight = ov::genai::modeling::ops::constant(q_up.weights_u4, op_ctx);
    auto up_exps_scales = ov::genai::modeling::ops::constant(q_up.scales_f16, op_ctx);
    auto up_exps_zps = ov::genai::modeling::ops::constant(q_up.zps_u4, op_ctx);
    auto down_exps_weight = ov::genai::modeling::ops::constant(q_down.weights_u4, op_ctx);
    auto down_exps_scales = ov::genai::modeling::ops::constant(q_down.scales_f16, op_ctx);
    auto down_exps_zps = ov::genai::modeling::ops::constant(q_down.zps_u4, op_ctx);

    auto out = ov::genai::modeling::ops::moe3gemm_fused_compressed(
        hidden_param,
        gate_inp_const,
        gate_exps_weight,
        gate_exps_scales,
        gate_exps_zps,
        up_exps_weight,
        up_exps_scales,
        up_exps_zps,
        down_exps_weight,
        down_exps_scales,
        down_exps_zps,
        static_cast<int32_t>(hidden_size),
        static_cast<int32_t>(inter_size),
        static_cast<int32_t>(num_experts),
        static_cast<int32_t>(top_k),
        static_cast<int32_t>(group_size),
        ov::element::f16);

    auto model = ctx.build_model({out.output()});

    auto& core = get_moe_test_core();
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto hidden_tensor = test_utils::make_tensor(hidden_states, {tokens, hidden_size});
    request.set_input_tensor(0, hidden_tensor);
    request.infer();

    auto expected = test_utils::moe_ref(hidden_states,
                                        gate_inp,
                                        gate_w_deq,
                                        up_w_deq,
                                        down_w_deq,
                                        batch,
                                        seq_len,
                                        hidden_size,
                                        inter_size,
                                        num_experts,
                                        top_k);
                                        
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_moe);
}

TEST(Ops, Moe3GemmFusedCompressedwithInt4RouterWeights) {
    
    ov::genai::modeling::BuilderContext ctx;

    constexpr size_t batch = 1;
    constexpr size_t seq_len = 16;
    constexpr size_t hidden_size = 1024;
    constexpr size_t inter_size = 2048;
    constexpr size_t num_experts = 8;
    constexpr size_t top_k = 4;
    constexpr size_t group_size = 128;

    static_assert(hidden_size % group_size == 0, "hidden_size must be divisible by group_size");
    static_assert(inter_size % group_size == 0, "inter_size must be divisible by group_size");

    const size_t tokens = batch * seq_len;
    auto hidden_param = ctx.parameter("hidden", ov::element::f32, ov::Shape{tokens, hidden_size});

    auto hidden_states = test_utils::random_f32(tokens * hidden_size, -0.5f, 0.5f, 11);
    auto gate_inp = test_utils::random_f32(num_experts * hidden_size, -0.5f, 0.5f, 23);
    
    auto gate_w_f32 = test_utils::random_f32(num_experts * inter_size * hidden_size, -0.5f, 0.5f, 31);
    auto up_w_f32 = test_utils::random_f32(num_experts * inter_size * hidden_size, -0.5f, 0.5f, 37);
    auto down_w_f32 = test_utils::random_f32(num_experts * hidden_size * inter_size, -0.5f, 0.5f, 41);
    // Scale expert FFN weights to prevent FP16 overflow in intermediate buffers.
    for (auto& v : gate_w_f32) v /= 14.0f;
    for (auto& v : up_w_f32)   v /= 14.0f;
    for (auto& v : down_w_f32) v /= 14.0f;

    auto q_gate_inp = test_utils::quantize_q41(gate_inp, num_experts, hidden_size, group_size);
    auto q_gate = test_utils::quantize_q41(gate_w_f32, num_experts, inter_size, hidden_size, group_size);
    auto q_up = test_utils::quantize_q41(up_w_f32, num_experts, inter_size, hidden_size, group_size);
    auto q_down = test_utils::quantize_q41(down_w_f32, num_experts, hidden_size, inter_size, group_size);

    auto gate_w_deq = test_utils::dequantize_q41(q_gate, num_experts, inter_size, hidden_size);
    auto up_w_deq = test_utils::dequantize_q41(q_up, num_experts, inter_size, hidden_size);
    auto down_w_deq = test_utils::dequantize_q41(q_down, num_experts, hidden_size, inter_size);

    auto* op_ctx = &ctx.op_context();
    auto gate_inp_tensor = test_utils::make_dequant_subgraph(q_gate_inp, op_ctx);
    
    auto gate_exps_weight = ov::genai::modeling::ops::constant(q_gate.weights_u4, op_ctx);
    auto gate_exps_scales = ov::genai::modeling::ops::constant(q_gate.scales_f16, op_ctx);
    auto gate_exps_zps = ov::genai::modeling::ops::constant(q_gate.zps_u4, op_ctx);
    auto up_exps_weight = ov::genai::modeling::ops::constant(q_up.weights_u4, op_ctx);
    auto up_exps_scales = ov::genai::modeling::ops::constant(q_up.scales_f16, op_ctx);
    auto up_exps_zps = ov::genai::modeling::ops::constant(q_up.zps_u4, op_ctx);
    auto down_exps_weight = ov::genai::modeling::ops::constant(q_down.weights_u4, op_ctx);
    auto down_exps_scales = ov::genai::modeling::ops::constant(q_down.scales_f16, op_ctx);
    auto down_exps_zps = ov::genai::modeling::ops::constant(q_down.zps_u4, op_ctx);

    auto out = ov::genai::modeling::ops::moe3gemm_fused_compressed(
        hidden_param,
        gate_inp_tensor,
        gate_exps_weight,
        gate_exps_scales,
        gate_exps_zps,
        up_exps_weight,
        up_exps_scales,
        up_exps_zps,
        down_exps_weight,
        down_exps_scales,
        down_exps_zps,
        static_cast<int32_t>(hidden_size),
        static_cast<int32_t>(inter_size),
        static_cast<int32_t>(num_experts),
        static_cast<int32_t>(top_k),
        static_cast<int32_t>(group_size),
        ov::element::f16);

    auto model = ctx.build_model({out.output()});
    ov::serialize(model, "Moe3GemmFusedCompressed_original.xml");

    auto& core = get_moe_test_core();
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    ov::serialize(compiled.get_runtime_model(), "Moe3GemmFusedCompressed_compiled.xml");

    auto hidden_tensor = test_utils::make_tensor(hidden_states, {tokens, hidden_size});
    request.set_input_tensor(0, hidden_tensor);
    request.infer();
    
    auto gate_inp_deq = test_utils::dequantize_q41(q_gate_inp, num_experts, 1, hidden_size);
}

TEST(Ops, Moe3GemmFusedCompressedWithSharedExperts) {
    ov::genai::modeling::BuilderContext ctx;

    constexpr size_t batch = 1;
    constexpr size_t seq_len = 16;
    constexpr size_t hidden_size = 1024;
    constexpr size_t inter_size = 2048;
    constexpr size_t shared_inter_size = 2048;
    constexpr size_t num_experts = 8;
    constexpr size_t top_k = 4;
    constexpr size_t group_size = 128;

    static_assert(hidden_size % group_size == 0, "hidden_size must be divisible by group_size");
    static_assert(inter_size % group_size == 0, "inter_size must be divisible by group_size");
    static_assert(shared_inter_size % group_size == 0, "shared_inter_size must be divisible by group_size");

    const size_t tokens = batch * seq_len;
    auto hidden_param = ctx.parameter("hidden", ov::element::f32, ov::Shape{tokens, hidden_size});

    auto hidden_states = test_utils::random_f32(tokens * hidden_size, -0.5f, 0.5f, 11);
    auto gate_inp = test_utils::random_f32(num_experts * hidden_size, -0.5f, 0.5f, 23);

    // Scale weights to prevent FP16 SwiGLU overflow.
    // Without scaling: gate_proj_max = 1024 * 0.5^2 = 256, SwiGLU_max = 256 * 256 = 65536 > FP16_MAX(65504).
    // With /7 on gate/down and /11 on up: gate_proj_max ~36, SwiGLU_max ~36*36=1296, down_out_max ~2048*1296/7 ~379k?
    // Use a tighter scale: divide gate/up by 14, down by 14 so SwiGLU_max ~18*18=324, down_out ~2048*324/14~47k < 65504.
    auto gate_w_f32 = test_utils::random_f32(num_experts * inter_size * hidden_size, -0.5f, 0.5f, 31);
    auto up_w_f32 = test_utils::random_f32(num_experts * inter_size * hidden_size, -0.5f, 0.5f, 37);
    auto down_w_f32 = test_utils::random_f32(num_experts * hidden_size * inter_size, -0.5f, 0.5f, 41);
    // Scale expert FFN weights: gate/up ÷14, down ÷14
    for (auto& v : gate_w_f32) v /= 14.0f;
    for (auto& v : up_w_f32)   v /= 14.0f;
    for (auto& v : down_w_f32) v /= 14.0f;

    auto shared_gate_w_f32 = test_utils::random_f32(1 * shared_inter_size * hidden_size, -0.5f, 0.5f, 51);
    auto shared_up_w_f32 = test_utils::random_f32(1 * shared_inter_size * hidden_size, -0.5f, 0.5f, 52);
    auto shared_down_w_f32 = test_utils::random_f32(1 * hidden_size * shared_inter_size, -0.5f, 0.5f, 53);
    // Scale shared expert FFN weights the same way
    for (auto& v : shared_gate_w_f32) v /= 14.0f;
    for (auto& v : shared_up_w_f32)   v /= 14.0f;
    for (auto& v : shared_down_w_f32) v /= 14.0f;

    auto q_gate = test_utils::quantize_q41(gate_w_f32, num_experts, inter_size, hidden_size, group_size);
    auto q_up = test_utils::quantize_q41(up_w_f32, num_experts, inter_size, hidden_size, group_size);
    auto q_down = test_utils::quantize_q41(down_w_f32, num_experts, hidden_size, inter_size, group_size);

    auto q_shared_gate = test_utils::quantize_q41(shared_gate_w_f32, 1, shared_inter_size, hidden_size, group_size);
    auto q_shared_up = test_utils::quantize_q41(shared_up_w_f32, 1, shared_inter_size, hidden_size, group_size);
    auto q_shared_down = test_utils::quantize_q41(shared_down_w_f32, 1, hidden_size, shared_inter_size, group_size);

    auto gate_w_deq = test_utils::dequantize_q41(q_gate, num_experts, inter_size, hidden_size);
    auto up_w_deq = test_utils::dequantize_q41(q_up, num_experts, inter_size, hidden_size);
    auto down_w_deq = test_utils::dequantize_q41(q_down, num_experts, hidden_size, inter_size);

    auto shared_gate_w_deq = test_utils::dequantize_q41(q_shared_gate, 1, shared_inter_size, hidden_size);
    auto shared_up_w_deq = test_utils::dequantize_q41(q_shared_up, 1, shared_inter_size, hidden_size);
    auto shared_down_w_deq = test_utils::dequantize_q41(q_shared_down, 1, hidden_size, shared_inter_size);

    auto* op_ctx = &ctx.op_context();
    auto gate_inp_tensor = test_utils::make_tensor(gate_inp, {num_experts, hidden_size});

    auto gate_inp_const = ov::genai::modeling::ops::constant(gate_inp_tensor, op_ctx);
    auto gate_exps_weight = ov::genai::modeling::ops::constant(q_gate.weights_u4, op_ctx);
    auto gate_exps_scales = ov::genai::modeling::ops::constant(q_gate.scales_f16, op_ctx);
    auto gate_exps_zps = ov::genai::modeling::ops::constant(q_gate.zps_u4, op_ctx);
    auto up_exps_weight = ov::genai::modeling::ops::constant(q_up.weights_u4, op_ctx);
    auto up_exps_scales = ov::genai::modeling::ops::constant(q_up.scales_f16, op_ctx);
    auto up_exps_zps = ov::genai::modeling::ops::constant(q_up.zps_u4, op_ctx);
    auto down_exps_weight = ov::genai::modeling::ops::constant(q_down.weights_u4, op_ctx);
    auto down_exps_scales = ov::genai::modeling::ops::constant(q_down.scales_f16, op_ctx);
    auto down_exps_zps = ov::genai::modeling::ops::constant(q_down.zps_u4, op_ctx);

    // Prepare Shared Tensors (Need to be unsqueezed/formatted for Op if necessary, but op takes what quantize gives usually)
    // IMPORTANT: moe3gemm_fused_compressed expects explicit tensors for shared experts.
    auto shared_gate_weight = ov::genai::modeling::ops::constant(q_shared_gate.weights_u4, op_ctx);
    auto shared_gate_scales = ov::genai::modeling::ops::constant(q_shared_gate.scales_f16, op_ctx);
    auto shared_gate_zps = ov::genai::modeling::ops::constant(q_shared_gate.zps_u4, op_ctx);
    auto shared_up_weight = ov::genai::modeling::ops::constant(q_shared_up.weights_u4, op_ctx);
    auto shared_up_scales = ov::genai::modeling::ops::constant(q_shared_up.scales_f16, op_ctx);
    auto shared_up_zps = ov::genai::modeling::ops::constant(q_shared_up.zps_u4, op_ctx);
    auto shared_down_weight = ov::genai::modeling::ops::constant(q_shared_down.weights_u4, op_ctx);
    auto shared_down_scales = ov::genai::modeling::ops::constant(q_shared_down.scales_f16, op_ctx);
    auto shared_down_zps = ov::genai::modeling::ops::constant(q_shared_down.zps_u4, op_ctx);

    std::vector<float> shared_gate_inp_dummy = test_utils::random_f32(1 * hidden_size, 0.1f, 0.5f, 61);
    auto shared_gate_inp_tensor = test_utils::make_tensor(shared_gate_inp_dummy, {1, hidden_size});
    auto shared_gate_gate_f32 = ov::genai::modeling::ops::constant(shared_gate_inp_tensor, op_ctx);
    auto shared_gate_gate = shared_gate_gate_f32.to(ov::element::f16);

    auto out = ov::genai::modeling::ops::moe3gemm_fused_compressed(
        hidden_param,
        gate_inp_const,
        gate_exps_weight,
        gate_exps_scales,
        gate_exps_zps,
        up_exps_weight,
        up_exps_scales,
        up_exps_zps,
        down_exps_weight,
        down_exps_scales,
        down_exps_zps,
        static_cast<int32_t>(hidden_size),
        static_cast<int32_t>(inter_size),
        static_cast<int32_t>(num_experts),
        static_cast<int32_t>(top_k),
        static_cast<int32_t>(group_size),
        ov::element::f16,
        shared_gate_weight,
        shared_gate_scales,
        shared_gate_zps,
        shared_up_weight,
        shared_up_scales,
        shared_up_zps,
        shared_down_weight,
        shared_down_scales,
        shared_down_zps,
        shared_gate_gate);

    auto model = ctx.build_model({out.output()});

    auto& core = get_moe_test_core();
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto hidden_tensor = test_utils::make_tensor(hidden_states, {tokens, hidden_size});
    request.set_input_tensor(0, hidden_tensor);
    request.infer();

    auto expected_sparse = test_utils::moe_ref(hidden_states,
                                        gate_inp,
                                        gate_w_deq,
                                        up_w_deq,
                                        down_w_deq,
                                        batch,
                                        seq_len,
                                        hidden_size,
                                        inter_size,
                                        num_experts,
                                        top_k);

    auto expected_shared = test_utils::moe_ref(hidden_states,
                                        shared_gate_inp_dummy,
                                        shared_gate_w_deq,
                                        shared_up_w_deq,
                                        shared_down_w_deq,
                                        batch,
                                        seq_len,
                                        hidden_size,
                                        shared_inter_size,
                                        1, // num_experts
                                        1); // top_k

    // Compute Scalar Gate Sigmoid
    std::vector<float> shared_gate_sigmoid(tokens, 0.0f);
    for (size_t t = 0; t < tokens; ++t) {
        float acc = 0.0f;
        for (size_t h = 0; h < hidden_size; ++h) {
            acc += hidden_states[t * hidden_size + h] * shared_gate_inp_dummy[h];
        }
        shared_gate_sigmoid[t] = 1.0f / (1.0f + std::exp(-acc));
    }

    std::vector<float> expected(expected_sparse.size());
    for(size_t t=0; t < tokens; ++t) {
        float scalar_scale = shared_gate_sigmoid[t];
        for(size_t h=0; h < hidden_size; ++h) {
            size_t idx = t * hidden_size + h;
            expected[idx] = expected_sparse[idx] + expected_shared[idx] * scalar_scale;
        }
    }
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_moe);
}
