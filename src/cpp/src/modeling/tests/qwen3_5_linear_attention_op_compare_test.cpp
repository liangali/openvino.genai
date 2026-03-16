// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Unit test: Compare Qwen3_5GatedDeltaNet output between the fused
// LinearAttention op path and the TensorIterator fallback path.
//
// The code path in Qwen3_5GatedDeltaNet::forward() is controlled by the
// environment variable OV_GENAI_USE_LINEAR_ATTENTION_OP.
//   - unset / "1"  →  fused LinearAttention op
//   - "0"          →  TensorIterator loop
//
// We build two separate OV models, compile both on GPU, feed identical
// random inputs, and assert the outputs match within tolerance.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/op/read_value.hpp>
#include <openvino/op/tensor_iterator.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

// ── RAII helper to set / restore an environment variable ──────────────────────
class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value)
        : name_(name) {
        const char* prev = std::getenv(name);
        had_prev_ = (prev != nullptr);
        if (had_prev_) {
            prev_value_ = prev;
        }
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }

    ~ScopedEnvVar() {
#ifdef _WIN32
        if (had_prev_) {
            _putenv_s(name_.c_str(), prev_value_.c_str());
        } else {
            // Setting to "" effectively removes on MSVC
            _putenv_s(name_.c_str(), "");
        }
#else
        if (had_prev_) {
            setenv(name_.c_str(), prev_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
#endif
    }

private:
    std::string name_;
    std::string prev_value_;
    bool had_prev_ = false;
};

// ── Qwen3_5 config with small dims suitable for unit tests ────────────────────
ov::genai::modeling::models::Qwen3_5TextModelConfig make_test_cfg() {
    ov::genai::modeling::models::Qwen3_5TextModelConfig cfg;
    cfg.hidden_size = 64;
    cfg.linear_num_key_heads = 2;
    cfg.linear_num_value_heads = 4;
    cfg.linear_key_head_dim = 16;
    cfg.linear_value_head_dim = 16;
    cfg.linear_conv_kernel_dim = 4;
    cfg.rms_norm_eps = 1e-6f;
    cfg.hidden_act = "silu";
    return cfg;
}

// ── Helper: add all required weights for Qwen3_5GatedDeltaNet ─────────────────
// Uses random_f32 with small magnitudes to avoid numerical instability from
// large accumulated values flowing through exp / softplus in the recurrent core.
void add_qwen3_5_gated_delta_net_weights(
    test_utils::DummyWeightSource& weights,
    const std::string& prefix,
    const ov::genai::modeling::models::Qwen3_5TextModelConfig& cfg) {
    const int32_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    const int32_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    const int32_t conv_dim = key_dim * 2 + value_dim;
    const int32_t proj_qkv = key_dim * 2 + value_dim;

    // Use deterministic seeds (different per weight) and small range to keep
    // intermediate activations bounded and the two code-paths numerically close.
    auto rnd = [](size_t n, uint32_t seed) {
        return test_utils::random_f32(n, -0.05f, 0.05f, seed);
    };

    weights.add(
        prefix + ".in_proj_qkv.weight",
        test_utils::make_tensor(
            rnd(static_cast<size_t>(proj_qkv * cfg.hidden_size), 10),
            {static_cast<size_t>(proj_qkv), static_cast<size_t>(cfg.hidden_size)}));
    weights.add(
        prefix + ".in_proj_z.weight",
        test_utils::make_tensor(
            rnd(static_cast<size_t>(value_dim * cfg.hidden_size), 20),
            {static_cast<size_t>(value_dim), static_cast<size_t>(cfg.hidden_size)}));
    weights.add(
        prefix + ".in_proj_b.weight",
        test_utils::make_tensor(
            rnd(static_cast<size_t>(cfg.linear_num_value_heads * cfg.hidden_size), 30),
            {static_cast<size_t>(cfg.linear_num_value_heads), static_cast<size_t>(cfg.hidden_size)}));
    weights.add(
        prefix + ".in_proj_a.weight",
        test_utils::make_tensor(
            rnd(static_cast<size_t>(cfg.linear_num_value_heads * cfg.hidden_size), 40),
            {static_cast<size_t>(cfg.linear_num_value_heads), static_cast<size_t>(cfg.hidden_size)}));
    weights.add(
        prefix + ".conv1d.weight",
        test_utils::make_tensor(
            rnd(static_cast<size_t>(conv_dim * cfg.linear_conv_kernel_dim), 50),
            {static_cast<size_t>(conv_dim), static_cast<size_t>(cfg.linear_conv_kernel_dim)}));
    // A_log and dt_bias: keep small so that exp(A_log) and softplus stay bounded.
    weights.add(
        prefix + ".A_log",
        test_utils::make_tensor(
            test_utils::random_f32(static_cast<size_t>(cfg.linear_num_value_heads), -0.01f, 0.01f, 60),
            {static_cast<size_t>(cfg.linear_num_value_heads)}));
    weights.add(
        prefix + ".dt_bias",
        test_utils::make_tensor(
            test_utils::random_f32(static_cast<size_t>(cfg.linear_num_value_heads), -0.01f, 0.01f, 70),
            {static_cast<size_t>(cfg.linear_num_value_heads)}));
    weights.add(
        prefix + ".norm.weight",
        test_utils::make_tensor(
            test_utils::make_seq(static_cast<size_t>(cfg.linear_value_head_dim), 1.0f, 0.0f),
            {static_cast<size_t>(cfg.linear_value_head_dim)}));
    weights.add(
        prefix + ".out_proj.weight",
        test_utils::make_tensor(
            rnd(static_cast<size_t>(cfg.hidden_size * value_dim), 80),
            {static_cast<size_t>(cfg.hidden_size), static_cast<size_t>(value_dim)}));
}

// ── Helper: build an ov::Model from Qwen3_5GatedDeltaNet with dynamic seq ────
std::shared_ptr<ov::Model> build_qwen3_5_linear_attn_model(
    const ov::genai::modeling::models::Qwen3_5TextModelConfig& cfg,
    int64_t seq_len = -1) {
    ov::genai::modeling::BuilderContext ctx;
    ov::genai::modeling::models::Qwen3_5GatedDeltaNet linear_attn(ctx, "linear_attn", cfg, 0);

    test_utils::DummyWeightSource weights;
    test_utils::DummyWeightFinalizer finalizer;
    add_qwen3_5_gated_delta_net_weights(weights, "linear_attn", cfg);
    ov::genai::modeling::weights::load_model(linear_attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32,
                                       ov::PartialShape{1, seq_len, cfg.hidden_size});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64,
                                        ov::PartialShape{1, seq_len});

    auto out = linear_attn.forward(hidden_states, beam_idx, &attention_mask, nullptr);
    return ctx.build_model({out.output()});
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Graph structure: LinearAttention path has LinearAttention op, no TI
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Qwen3_5LinearAttentionOpCompare, LinearAttentionPathHasCorrectOps) {
    ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "1");
    auto cfg = make_test_cfg();
    auto model = build_qwen3_5_linear_attn_model(cfg, 2);

    bool has_linear_attention_op = false;
    bool has_tensor_iterator = false;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("LinearAttention")) {
            has_linear_attention_op = true;
        }
        if (ov::as_type_ptr<ov::op::v0::TensorIterator>(op)) {
            has_tensor_iterator = true;
        }
    }

    EXPECT_TRUE(has_linear_attention_op)
        << "LinearAttention op path should contain a LinearAttention node";
    EXPECT_FALSE(has_tensor_iterator)
        << "LinearAttention op path should NOT contain a TensorIterator";
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Graph structure: TensorIterator path has TI, no LinearAttention op
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Qwen3_5LinearAttentionOpCompare, TensorIteratorPathHasCorrectOps) {
    ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "0");
    auto cfg = make_test_cfg();
    auto model = build_qwen3_5_linear_attn_model(cfg, 2);

    bool has_linear_attention_op = false;
    bool has_tensor_iterator = false;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("LinearAttention")) {
            has_linear_attention_op = true;
        }
        if (ov::as_type_ptr<ov::op::v0::TensorIterator>(op)) {
            has_tensor_iterator = true;
        }
    }

    EXPECT_FALSE(has_linear_attention_op)
        << "TensorIterator path should NOT contain a LinearAttention node";
    EXPECT_TRUE(has_tensor_iterator)
        << "TensorIterator path should contain a TensorIterator";
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Both paths register identical stateful variables (conv + recurrent)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Qwen3_5LinearAttentionOpCompare, BothPathsRegisterSameStates) {
    auto cfg = make_test_cfg();

    auto check_states = [&](const std::string& env_value) {
        ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", env_value.c_str());
        auto model = build_qwen3_5_linear_attn_model(cfg, 2);

        bool has_conv = false, has_recurrent = false;
        size_t read_count = 0, assign_count = 0;

        for (const auto& op : model->get_ops()) {
            if (op->get_type_name() == std::string("FusedConv")) {
                has_conv = true;
            }
            if (auto read = ov::as_type_ptr<ov::op::v6::ReadValue>(op)) {
                read_count++;
                const auto id = read->get_variable_id();
                has_conv = has_conv || id.find("linear_states.0.conv") != std::string::npos;
                has_recurrent = has_recurrent || id.find("linear_states.0.recurrent") != std::string::npos;
            }
            if (ov::as_type_ptr<ov::op::v6::Assign>(op)) {
                assign_count++;
            }
        }

        EXPECT_GE(read_count, 1u) << "env=" << env_value;
        EXPECT_GE(assign_count, 1u) << "env=" << env_value;
        EXPECT_TRUE(has_conv) << "Missing conv state (env=" << env_value << ")";
        EXPECT_TRUE(has_recurrent) << "Missing recurrent state (env=" << env_value << ")";
    };

    check_states("1");
    check_states("0");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Numerical equivalence on GPU: LinearAttention op ≈ TensorIterator
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Qwen3_5LinearAttentionOpCompare, NumericallyMatchOnGPU) {
    auto cfg = make_test_cfg();
    const size_t seq_len = 4;
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);

    // Build model using LinearAttention op path.
    std::shared_ptr<ov::Model> model_la;
    {
        ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "1");
        model_la = build_qwen3_5_linear_attn_model(cfg, static_cast<int64_t>(seq_len));
    }

    // Build model using TensorIterator path.
    std::shared_ptr<ov::Model> model_ti;
    {
        ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "0");
        model_ti = build_qwen3_5_linear_attn_model(cfg, static_cast<int64_t>(seq_len));
    }

    ov::Core core;
    auto compiled_la = core.compile_model(model_la, "GPU");
    auto compiled_ti = core.compile_model(model_ti, "GPU");
    ov::serialize(compiled_la.get_runtime_model(), "qwen3_5_dummy_compiled_la.xml");
    ov::serialize(compiled_ti.get_runtime_model(), "qwen3_5_dummy_compiled_ti.xml");
    auto req_la = compiled_la.create_infer_request();
    auto req_ti = compiled_ti.create_infer_request();

    // Identical random input for both.
    auto input_data = test_utils::random_f32(seq_len * hidden, -0.1f, 0.1f, 42);
    ov::Tensor hs_tensor(ov::element::f32, {1, seq_len, hidden});
    std::memcpy(hs_tensor.data<float>(), input_data.data(), input_data.size() * sizeof(float));

    ov::Tensor bi_tensor(ov::element::i32, {1});
    bi_tensor.data<int32_t>()[0] = 0;

    ov::Tensor am_tensor(ov::element::i64, {1, seq_len});
    std::fill_n(am_tensor.data<int64_t>(), seq_len, 1);

    for (auto* req : {&req_la, &req_ti}) {
        req->set_tensor("hidden_states", hs_tensor);
        req->set_tensor("beam_idx", bi_tensor);
        req->set_tensor("attention_mask", am_tensor);
        req->infer();
    }

    // Compare outputs element-wise.
    auto out_la = req_la.get_output_tensor(0);
    auto out_ti = req_ti.get_output_tensor(0);
    ASSERT_EQ(out_la.get_shape(), out_ti.get_shape());
    ASSERT_EQ(out_la.get_shape(), (ov::Shape{1, seq_len, hidden}));

    const float* d_la = out_la.data<float>();
    const float* d_ti = out_ti.data<float>();
    const size_t n = out_la.get_size();
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(d_la[i] - d_ti[i]));
    }

    std::cout << "[Qwen3_5 LinearAttention vs TensorIterator] seq_len=" << seq_len
              << "  max_diff=" << max_diff << "  tol=" << test_utils::k_tol_linear_attn << "\n";

    EXPECT_LT(max_diff, test_utils::k_tol_linear_attn)
        << "LinearAttention op and TensorIterator outputs differ by " << max_diff;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Stateful prefill + decode equivalence on GPU
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Qwen3_5LinearAttentionOpCompare, StatefulPrefillDecodeMatchOnGPU) {
    auto cfg = make_test_cfg();
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);

    // Build both models with dynamic sequence length.
    std::shared_ptr<ov::Model> model_la, model_ti;
    {
        ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "1");
        model_la = build_qwen3_5_linear_attn_model(cfg);
    }
    {
        ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "0");
        model_ti = build_qwen3_5_linear_attn_model(cfg);
    }

    ov::Core core;
    auto compiled_la = core.compile_model(model_la, "GPU");
    auto compiled_ti = core.compile_model(model_ti, "GPU");
    auto req_la = compiled_la.create_infer_request();
    auto req_ti = compiled_ti.create_infer_request();

    ov::Tensor bi_tensor(ov::element::i32, {1});
    bi_tensor.data<int32_t>()[0] = 0;

    // ── Phase 1: Prefill (seq_len = 3) ──
    const size_t prefill_len = 3;
    auto prefill_data = test_utils::random_f32(prefill_len * hidden, -0.1f, 0.1f, 123);
    ov::Tensor prefill_hs(ov::element::f32, {1, prefill_len, hidden});
    std::memcpy(prefill_hs.data<float>(), prefill_data.data(), prefill_data.size() * sizeof(float));

    ov::Tensor prefill_mask(ov::element::i64, {1, prefill_len});
    std::fill_n(prefill_mask.data<int64_t>(), prefill_len, 1);

    for (auto* req : {&req_la, &req_ti}) {
        req->set_tensor("hidden_states", prefill_hs);
        req->set_tensor("beam_idx", bi_tensor);
        req->set_tensor("attention_mask", prefill_mask);
        req->infer();
    }

    // Check prefill outputs match.
    {
        auto out_la = req_la.get_output_tensor(0);
        auto out_ti = req_ti.get_output_tensor(0);
        ASSERT_EQ(out_la.get_shape(), (ov::Shape{1, prefill_len, hidden}));
        ASSERT_EQ(out_ti.get_shape(), (ov::Shape{1, prefill_len, hidden}));

        const float* d_la = out_la.data<float>();
        const float* d_ti = out_ti.data<float>();
        float max_diff = 0.0f;
        for (size_t i = 0; i < out_la.get_size(); ++i) {
            max_diff = std::max(max_diff, std::abs(d_la[i] - d_ti[i]));
        }
        std::cout << "[Prefill seq_len=" << prefill_len << "] max_diff=" << max_diff << "\n";
        EXPECT_LT(max_diff, test_utils::k_tol_linear_attn)
            << "Prefill outputs differ by " << max_diff;
    }

    // ── Phase 2: Decode (seq_len = 1), states carried over ──
    const size_t decode_len = 1;
    auto decode_data = test_utils::random_f32(decode_len * hidden, -0.05f, 0.05f, 456);
    ov::Tensor decode_hs(ov::element::f32, {1, decode_len, hidden});
    std::memcpy(decode_hs.data<float>(), decode_data.data(), decode_data.size() * sizeof(float));

    ov::Tensor decode_mask(ov::element::i64, {1, decode_len});
    decode_mask.data<int64_t>()[0] = 1;

    for (auto* req : {&req_la, &req_ti}) {
        req->set_tensor("hidden_states", decode_hs);
        req->set_tensor("beam_idx", bi_tensor);
        req->set_tensor("attention_mask", decode_mask);
        req->infer();
    }

    // Check decode outputs match.
    {
        auto out_la = req_la.get_output_tensor(0);
        auto out_ti = req_ti.get_output_tensor(0);
        ASSERT_EQ(out_la.get_shape(), (ov::Shape{1, decode_len, hidden}));
        ASSERT_EQ(out_ti.get_shape(), (ov::Shape{1, decode_len, hidden}));

        const float* d_la = out_la.data<float>();
        const float* d_ti = out_ti.data<float>();
        float max_diff = 0.0f;
        for (size_t i = 0; i < out_la.get_size(); ++i) {
            max_diff = std::max(max_diff, std::abs(d_la[i] - d_ti[i]));
        }
        std::cout << "[Decode seq_len=" << decode_len << " after prefill] max_diff=" << max_diff << "\n";
        EXPECT_LT(max_diff, test_utils::k_tol_linear_attn)
            << "Decode outputs (with carried-over state) differ by " << max_diff;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Performance comparison on GPU: LinearAttention op vs TensorIterator
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Qwen3_5LinearAttentionOpCompare, PerformanceOnGPU) {
    // Use a larger config for meaningful perf measurement.
    auto cfg = make_test_cfg();
    cfg.hidden_size = 2048;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 32;
    cfg.linear_key_head_dim = 128;
    cfg.linear_value_head_dim = 128;

    struct Scenario {
        std::string name;
        size_t seq_len;
        int warmup;
        int iters;
    };
    const std::vector<Scenario> scenarios = {
        {"prefill_seq64",  64, 1, 20},
        {"prefill_seq256", 256, 1, 10},
        {"decode_seq1",    1,  1, 50},
    };

    // Build models.
    std::shared_ptr<ov::Model> model_la, model_ti;
    {
        ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "1");
        model_la = build_qwen3_5_linear_attn_model(cfg);
    }
    {
        ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "0");
        model_ti = build_qwen3_5_linear_attn_model(cfg);
    }

    ov::Core core;
    auto compiled_la = core.compile_model(model_la, "GPU");
    auto compiled_ti = core.compile_model(model_ti, "GPU");
    auto req_la = compiled_la.create_infer_request();
    auto req_ti = compiled_ti.create_infer_request();

    const size_t hidden = static_cast<size_t>(cfg.hidden_size);

    std::cout << "\n========== Qwen3_5 GatedDeltaNet: LinearAttention op vs TensorIterator ==========\n";

    for (const auto& sc : scenarios) {
        const size_t sl = sc.seq_len;

        auto input_data = test_utils::random_f32(sl * hidden, -0.1f, 0.1f, 42);
        ov::Tensor hs_tensor(ov::element::f32, {1, sl, hidden});
        std::memcpy(hs_tensor.data<float>(), input_data.data(), input_data.size() * sizeof(float));

        ov::Tensor bi_tensor(ov::element::i32, {1});
        bi_tensor.data<int32_t>()[0] = 0;

        ov::Tensor am_tensor(ov::element::i64, {1, sl});
        std::fill_n(am_tensor.data<int64_t>(), sl, 1);

        auto bench = [&](ov::InferRequest& req, int warmup, int iters) -> double {
            req.set_tensor("hidden_states", hs_tensor);
            req.set_tensor("beam_idx", bi_tensor);
            req.set_tensor("attention_mask", am_tensor);

            for (auto& s : req.query_state()) s.reset();
            for (int i = 0; i < warmup; ++i) {
                req.infer();
                for (auto& s : req.query_state()) s.reset();
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) {
                req.infer();
                for (auto& s : req.query_state()) s.reset();
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        };

        double ms_la = bench(req_la, sc.warmup, sc.iters);
        double ms_ti = bench(req_ti, sc.warmup, sc.iters);
        double speedup = ms_ti / ms_la;

        std::cout << "  [" << sc.name << "]  seq_len=" << sl
                  << "  LA=" << std::fixed << std::setprecision(3) << ms_la << " ms"
                  << "  TI=" << ms_ti << " ms"
                  << "  speedup=" << std::setprecision(2) << speedup << "x"
                  << (speedup >= 1.0 ? " (LA faster)" : " (TI faster)")
                  << "\n";

        EXPECT_EQ(req_la.get_output_tensor(0).get_shape(), (ov::Shape{1, sl, hidden}));
        EXPECT_EQ(req_ti.get_output_tensor(0).get_shape(), (ov::Shape{1, sl, hidden}));
    }

    std::cout << "================================================================================\n";
}
