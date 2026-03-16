// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/op/read_value.hpp>
#include <openvino/op/tensor_iterator.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

ov::genai::modeling::models::Qwen3_5TextModelConfig make_qwen3_5_9b_attention_cfg() {
    const auto dense_9b = ov::genai::modeling::models::Qwen3_5Config::make_dummy_dense9b_config();

    ov::genai::modeling::models::Qwen3_5TextModelConfig cfg;
    cfg.hidden_size = dense_9b.text.hidden_size;
    cfg.num_attention_heads = dense_9b.text.num_attention_heads;
    cfg.num_key_value_heads = dense_9b.text.num_key_value_heads;
    cfg.head_dim = dense_9b.text.head_dim;
    cfg.rms_norm_eps = dense_9b.text.rms_norm_eps;
    cfg.rope_theta = dense_9b.text.rope_theta;
    cfg.partial_rotary_factor = dense_9b.text.partial_rotary_factor;
    cfg.hidden_act = dense_9b.text.hidden_act;

    cfg.linear_conv_kernel_dim = dense_9b.text.linear_conv_kernel_dim;
    cfg.linear_key_head_dim = dense_9b.text.linear_key_head_dim;
    cfg.linear_value_head_dim = dense_9b.text.linear_value_head_dim;
    cfg.linear_num_key_heads = dense_9b.text.linear_num_key_heads;
    cfg.linear_num_value_heads = dense_9b.text.linear_num_value_heads;
    return cfg;
}

ov::genai::modeling::models::Qwen3_5TextModelConfig make_linear_attention_cfg() {
    // Align with Qwen3.5-35B-A3B text_config for realistic linear-attention behavior.
    ov::genai::modeling::models::Qwen3_5TextModelConfig cfg;
    cfg.architecture = "qwen3_5_moe";
    cfg.vocab_size = 248320;
    cfg.hidden_size = 2048;
    cfg.num_hidden_layers = 40;
    cfg.num_attention_heads = 16;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 256;
    cfg.max_position_embeddings = 262144;
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = 10000000.0f;
    cfg.partial_rotary_factor = 0.25f;
    cfg.hidden_act = "silu";
    cfg.attention_bias = false;
    cfg.full_attention_interval = 4;
    cfg.linear_conv_kernel_dim = 4;
    cfg.linear_key_head_dim = 128;
    cfg.linear_value_head_dim = 128;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 32;
    cfg.moe_intermediate_size = 512;
    cfg.shared_expert_intermediate_size = 512;
    cfg.num_experts = 256;
    cfg.num_experts_per_tok = 8;
    cfg.output_router_logits = false;
    cfg.router_aux_loss_coef = 0.001f;
    cfg.mrope_interleaved = true;
    cfg.mrope_section = {11, 11, 10};
    cfg.layer_types.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        cfg.layer_types.push_back(((i + 1) % cfg.full_attention_interval) ? "linear_attention" : "full_attention");
    }
    return cfg;
}

int32_t resolve_rotary_dim(const ov::genai::modeling::models::Qwen3_5TextModelConfig& cfg) {
    int32_t rotary_dim = static_cast<int32_t>(std::floor(static_cast<float>(cfg.head_dim) * cfg.partial_rotary_factor));
    rotary_dim = std::max<int32_t>(0, std::min<int32_t>(rotary_dim, cfg.head_dim));
    if ((rotary_dim % 2) != 0) {
        rotary_dim -= 1;
    }
    return rotary_dim;
}

std::vector<ov::genai::modeling::weights::SyntheticWeightSpec> make_attention_weight_specs(
    const ov::genai::modeling::models::Qwen3_5TextModelConfig& cfg,
    const std::string& prefix) {
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    const size_t q_out = static_cast<size_t>(cfg.num_attention_heads) * static_cast<size_t>(cfg.head_dim) * 2ull;
    const size_t kv_out = static_cast<size_t>(cfg.num_key_value_heads) * static_cast<size_t>(cfg.head_dim);
    const size_t attn_hidden = static_cast<size_t>(cfg.num_attention_heads) * static_cast<size_t>(cfg.head_dim);

    return {
        {prefix + ".q_proj.weight", {q_out, hidden}, ov::element::f32},
        {prefix + ".k_proj.weight", {kv_out, hidden}, ov::element::f32},
        {prefix + ".v_proj.weight", {kv_out, hidden}, ov::element::f32},
        {prefix + ".o_proj.weight", {hidden, attn_hidden}, ov::element::f32},
        {prefix + ".q_norm.weight", {static_cast<size_t>(cfg.head_dim)}, ov::element::f32},
        {prefix + ".k_norm.weight", {static_cast<size_t>(cfg.head_dim)}, ov::element::f32},
    };
}

ov::genai::modeling::weights::QuantizationConfig make_int4_quant_config(const std::string& prefix) {
    ov::genai::modeling::weights::QuantizationConfig quant_config;
    quant_config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
    quant_config.group_size = 128;
    quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::NONE;
    quant_config.selection.only_2d_weights = true;
    quant_config.selection.include_weights = {
        prefix + ".q_proj.weight",
        prefix + ".k_proj.weight",
        prefix + ".v_proj.weight",
        prefix + ".o_proj.weight"};
    return quant_config;
}

ov::genai::modeling::weights::QuantizationConfig make_linear_attention_int4_quant_config(const std::string& prefix) {
    ov::genai::modeling::weights::QuantizationConfig quant_config;
    quant_config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
    quant_config.group_size = 128;
    quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::NONE;
    quant_config.selection.only_2d_weights = true;
    quant_config.selection.include_weights = {
        prefix + ".in_proj_qkv.weight",
        prefix + ".in_proj_z.weight",
        prefix + ".in_proj_b.weight",
        prefix + ".in_proj_a.weight",
        prefix + ".conv1d.weight",
        prefix + ".out_proj.weight"};
    return quant_config;
}

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

void add_linear_attention_weights(
    test_utils::DummyWeightSource& weights,
    const std::string& prefix,
    const ov::genai::modeling::models::Qwen3_5TextModelConfig& cfg) {
    const int32_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    const int32_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    const int32_t conv_dim = key_dim * 2 + value_dim;
    const int32_t proj_qkv = key_dim * 2 + value_dim;

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

std::shared_ptr<ov::Model> build_full_attention_model(
    const ov::genai::modeling::models::Qwen3_5TextModelConfig& cfg,
    int32_t rotary_dim) {
    ov::genai::modeling::BuilderContext ctx;
    ov::genai::modeling::models::Qwen3_5Attention attn(ctx, "self_attn", cfg);

    auto specs = make_attention_weight_specs(cfg, "self_attn");
    ov::genai::modeling::weights::SyntheticWeightSource weights(std::move(specs), 2026u, -0.02f, 0.02f);
    auto quant_config = make_int4_quant_config("self_attn");
    ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(quant_config);
    ov::genai::modeling::weights::load_model(attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{1, 2, cfg.hidden_size});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{1});
    auto rope_cos = ctx.parameter("rope_cos", ov::element::f32, ov::PartialShape{1, 2, rotary_dim / 2});
    auto rope_sin = ctx.parameter("rope_sin", ov::element::f32, ov::PartialShape{1, 2, rotary_dim / 2});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{1, 2});

    auto out = attn.forward(hidden_states, beam_idx, rope_cos, rope_sin, &attention_mask);
    return ctx.build_model({out.output()});
}

std::shared_ptr<ov::Model> build_linear_attention_model(
    const ov::genai::modeling::models::Qwen3_5TextModelConfig& cfg,
    int64_t seq_len) {
    ov::genai::modeling::BuilderContext ctx;
    ov::genai::modeling::models::Qwen3_5GatedDeltaNet linear_attn(ctx, "linear_attn", cfg, 0);

    test_utils::DummyWeightSource weights;
    auto quant_config = make_linear_attention_int4_quant_config("linear_attn");
    ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(quant_config);
    add_linear_attention_weights(weights, "linear_attn", cfg);
    ov::genai::modeling::weights::load_model(linear_attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{1, seq_len, cfg.hidden_size});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{1, seq_len});

    auto out = linear_attn.forward(hidden_states, beam_idx, &attention_mask, nullptr);
    return ctx.build_model({out.output()});
}

bool has_gpu_device(ov::Core& core) {
    const auto devices = core.get_available_devices();
    return std::any_of(devices.begin(), devices.end(), [](const std::string& dev) {
        return dev.rfind("GPU", 0) == 0;
    });
}

bool has_op_type(const std::shared_ptr<ov::Model>& model, const std::string& type_name) {
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == type_name) {
            return true;
        }
    }
    return false;
}

void dump_ir_model(const std::shared_ptr<ov::Model>& model, const std::string& base_name) {
    ov::serialize(model, base_name + ".xml");
}

void dump_ir_model(const ov::CompiledModel& compiled_model, const std::string& base_name) {
    ov::serialize(compiled_model.get_runtime_model(), base_name + ".xml");
}

struct LinearStateSummary {
    bool has_conv = false;
    bool has_recurrent = false;
    size_t read_count = 0;
    size_t assign_count = 0;
};

LinearStateSummary summarize_linear_states(const std::shared_ptr<ov::Model>& model) {
    LinearStateSummary summary;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("FusedConv")) {
            summary.has_conv = true;
        }
        if (auto read = ov::as_type_ptr<ov::op::v6::ReadValue>(op)) {
            summary.read_count++;
            const auto id = read->get_variable_id();
            summary.has_conv = summary.has_conv || id.find("linear_states.0.conv") != std::string::npos;
            summary.has_recurrent = summary.has_recurrent || id.find("linear_states.0.recurrent") != std::string::npos;
        }
        if (ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            summary.assign_count++;
        }
    }
    return summary;
}

ov::Tensor make_hidden_states_tensor(size_t seq_len, size_t hidden, float min_value, float max_value, uint32_t seed) {
    ov::Tensor hidden_states(ov::element::f32, ov::Shape{1, seq_len, hidden});
    auto data = test_utils::random_f32(seq_len * hidden, min_value, max_value, seed);
    std::memcpy(hidden_states.data<float>(), data.data(), data.size() * sizeof(float));
    return hidden_states;
}

ov::Tensor make_attention_mask_tensor(size_t seq_len) {
    ov::Tensor attention_mask(ov::element::i64, ov::Shape{1, seq_len});
    std::fill_n(attention_mask.data<int64_t>(), seq_len, 1);
    return attention_mask;
}

ov::Tensor make_attention_mask_tensor_with_padding(size_t seq_len, uint32_t seed) {
    ov::Tensor attention_mask(ov::element::i64, ov::Shape{1, seq_len});
    auto* mask = attention_mask.data<int64_t>();
    std::fill_n(mask, seq_len, 0);

    if (seq_len == 0) {
        return attention_mask;
    }
    if (seq_len == 1) {
        mask[0] = 1;
        return attention_mask;
    }

    const size_t min_valid = std::max<size_t>(1, (seq_len * 7) / 10);
    const size_t max_valid = seq_len - 1;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(min_valid, max_valid);
    const size_t valid_tokens = dist(rng);
    std::fill_n(mask, valid_tokens, 1);
    return attention_mask;
}

float max_abs_diff(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    const float* l = lhs.data<float>();
    const float* r = rhs.data<float>();
    float diff = 0.0f;
    for (size_t i = 0; i < lhs.get_size(); ++i) {
        diff = std::max(diff, std::abs(l[i] - r[i]));
    }
    return diff;
}

std::unordered_map<std::string, ov::Shape> collect_state_shapes(ov::InferRequest& request) {
    std::unordered_map<std::string, ov::Shape> shapes;
    for (const auto& state : request.query_state()) {
        shapes[state.get_name()] = state.get_state().get_shape();
    }
    return shapes;
}

}  // namespace

TEST(Qwen3_5AttentionULT, FullAttention_QuantizedGpuInfer_Smoke) {
    ov::Core core;
    if (!has_gpu_device(core)) {
        GTEST_SKIP() << "GPU device is not available";
    }

    const auto cfg = make_qwen3_5_9b_attention_cfg();
    const int32_t rotary_dim = resolve_rotary_dim(cfg);
    ASSERT_GT(rotary_dim, 0);

    auto model = build_full_attention_model(cfg, rotary_dim);

    const std::vector<std::string> expected_compressed_nodes = {
        "self_attn.q_proj.weight_compressed",
        "self_attn.k_proj.weight_compressed",
        "self_attn.v_proj.weight_compressed",
        "self_attn.o_proj.weight_compressed"};

    for (const auto& expected_name : expected_compressed_nodes) {
        bool found = false;
        for (const auto& op : model->get_ops()) {
            if (op->get_friendly_name() == expected_name) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Missing quantized node: " << expected_name;
    }

    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    ov::Tensor hidden_states(ov::element::f32, ov::Shape{1, 2, static_cast<size_t>(cfg.hidden_size)});
    ov::Tensor beam_idx(ov::element::i32, ov::Shape{1});
    ov::Tensor rope_cos(ov::element::f32, ov::Shape{1, 2, static_cast<size_t>(rotary_dim / 2)});
    ov::Tensor rope_sin(ov::element::f32, ov::Shape{1, 2, static_cast<size_t>(rotary_dim / 2)});
    ov::Tensor attention_mask(ov::element::i64, ov::Shape{1, 2});

    std::fill_n(hidden_states.data<float>(), hidden_states.get_size(), 0.1f);
    beam_idx.data<int32_t>()[0] = 0;
    std::fill_n(rope_cos.data<float>(), rope_cos.get_size(), 1.0f);
    std::fill_n(rope_sin.data<float>(), rope_sin.get_size(), 0.0f);
    std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

    request.set_tensor("hidden_states", hidden_states);
    request.set_tensor("beam_idx", beam_idx);
    request.set_tensor("rope_cos", rope_cos);
    request.set_tensor("rope_sin", rope_sin);
    request.set_tensor("attention_mask", attention_mask);
    request.infer();

    const auto output = request.get_output_tensor(0);
    EXPECT_EQ(output.get_shape(), (ov::Shape{1, 2, static_cast<size_t>(cfg.hidden_size)}));
}

TEST(Qwen3_5AttentionULT, LinearAttention_BasicOps_GraphStateAndGpuInfer) {
    ov::Core core;
    if (!has_gpu_device(core)) {
        GTEST_SKIP() << "GPU device is not available";
    }

    const auto cfg = make_linear_attention_cfg();
    std::shared_ptr<ov::Model> model;
    {
        ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "0");
        model = build_linear_attention_model(cfg, 2);
    }

    EXPECT_TRUE(has_op_type(model, "TensorIterator"));
    EXPECT_FALSE(has_op_type(model, "LinearAttention"));

    const auto states = summarize_linear_states(model);
    EXPECT_GE(states.read_count, 1u);
    EXPECT_GE(states.assign_count, 1u);
    EXPECT_TRUE(states.has_conv);
    EXPECT_TRUE(states.has_recurrent);

    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    auto hidden_states = make_hidden_states_tensor(2, hidden, -0.02f, 0.02f, 100u);
    ov::Tensor beam_idx(ov::element::i32, ov::Shape{1});
    auto attention_mask = make_attention_mask_tensor(2);
    beam_idx.data<int32_t>()[0] = 0;

    request.set_tensor("hidden_states", hidden_states);
    request.set_tensor("beam_idx", beam_idx);
    request.set_tensor("attention_mask", attention_mask);
    request.infer();

    const auto output = request.get_output_tensor(0);
    EXPECT_EQ(output.get_shape(), (ov::Shape{1, 2, hidden}));
}

TEST(Qwen3_5AttentionULT, LinearAttention_FusedOp_GraphStateAndGpuInfer) {
    ov::Core core;
    if (!has_gpu_device(core)) {
        GTEST_SKIP() << "GPU device is not available";
    }

    const auto cfg = make_linear_attention_cfg();
    std::shared_ptr<ov::Model> model;
    {
        ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "1");
        model = build_linear_attention_model(cfg, 2);
    }

    EXPECT_TRUE(has_op_type(model, "LinearAttention"));
    EXPECT_FALSE(has_op_type(model, "TensorIterator"));

    const auto states = summarize_linear_states(model);
    EXPECT_GE(states.read_count, 1u);
    EXPECT_GE(states.assign_count, 1u);
    EXPECT_TRUE(states.has_conv);
    EXPECT_TRUE(states.has_recurrent);

    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    auto hidden_states = make_hidden_states_tensor(2, hidden, -0.02f, 0.02f, 100u);
    ov::Tensor beam_idx(ov::element::i32, ov::Shape{1});
    auto attention_mask = make_attention_mask_tensor(2);
    beam_idx.data<int32_t>()[0] = 0;

    request.set_tensor("hidden_states", hidden_states);
    request.set_tensor("beam_idx", beam_idx);
    request.set_tensor("attention_mask", attention_mask);
    request.infer();

    const auto output = request.get_output_tensor(0);
    EXPECT_EQ(output.get_shape(), (ov::Shape{1, 2, hidden}));
}

TEST(Qwen3_5AttentionULT, LinearAttention_BasicOpsVsFused_OutputMatch_StatelessAndStateful) {
    ov::Core core;
    if (!has_gpu_device(core)) {
        GTEST_SKIP() << "GPU device is not available";
    }

    const auto cfg = make_linear_attention_cfg();
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);

    std::shared_ptr<ov::Model> model_basic;
    std::shared_ptr<ov::Model> model_fused;
    {
        ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "0");
        model_basic = build_linear_attention_model(cfg, -1);
    }
    {
        ScopedEnvVar env("OV_GENAI_USE_LINEAR_ATTENTION_OP", "1");
        model_fused = build_linear_attention_model(cfg, -1);
    }

    ASSERT_TRUE(has_op_type(model_basic, "TensorIterator"));
    ASSERT_FALSE(has_op_type(model_basic, "LinearAttention"));
    ASSERT_TRUE(has_op_type(model_fused, "LinearAttention"));
    ASSERT_FALSE(has_op_type(model_fused, "TensorIterator"));
    dump_ir_model(model_basic, "qwen3_5_linear_attention_basic_original");
    dump_ir_model(model_fused, "qwen3_5_linear_attention_fused_original");

    ov::Tensor beam_idx(ov::element::i32, ov::Shape{1});
    beam_idx.data<int32_t>()[0] = 0;

    auto run_once = [&](ov::InferRequest& req, size_t seq_len, float min_value, float max_value, uint32_t seed) {
        auto hidden_states = make_hidden_states_tensor(seq_len, hidden, min_value, max_value, seed);
        auto attention_mask = make_attention_mask_tensor_with_padding(seq_len, seed ^ 0x9e3779b9u);
        req.set_tensor("hidden_states", hidden_states);
        req.set_tensor("beam_idx", beam_idx);
        req.set_tensor("attention_mask", attention_mask);
        req.infer();
        req.wait();
    };

    // 1) Prefill comparison (long prompt with deterministic variable-length padding)
    const size_t prefill_len = 1024;
    {
        auto compiled_basic = core.compile_model(model_basic, "GPU");
        auto compiled_fused = core.compile_model(model_fused, "GPU");
        dump_ir_model(compiled_basic, "qwen3_5_linear_attention_basic_prefill_compiled");
        dump_ir_model(compiled_fused, "qwen3_5_linear_attention_fused_prefill_compiled");
        auto req_basic = compiled_basic.create_infer_request();
        auto req_fused = compiled_fused.create_infer_request();

        run_once(req_basic, prefill_len, -0.02f, 0.02f, 123u);
        run_once(req_fused, prefill_len, -0.02f, 0.02f, 123u);

        const auto prefill_basic = req_basic.get_output_tensor(0);
        const auto prefill_fused = req_fused.get_output_tensor(0);
        ASSERT_EQ(prefill_basic.get_shape(), (ov::Shape{1, prefill_len, hidden}));
        ASSERT_EQ(prefill_fused.get_shape(), (ov::Shape{1, prefill_len, hidden}));

        const float prefill_diff = max_abs_diff(prefill_basic, prefill_fused);
        EXPECT_LT(prefill_diff, test_utils::k_tol_linear_attn)
            << "Prefill max diff: " << prefill_diff
            << ", tol: " << test_utils::k_tol_linear_attn;
    }

    // 2) Decode comparison (single token, fresh compile/request as requested)
    const size_t decode_len = 1;
    {
        auto compiled_basic = core.compile_model(model_basic, "GPU");
        auto compiled_fused = core.compile_model(model_fused, "GPU");
        dump_ir_model(compiled_basic, "qwen3_5_linear_attention_basic_decode_compiled");
        dump_ir_model(compiled_fused, "qwen3_5_linear_attention_fused_decode_compiled");
        auto req_basic = compiled_basic.create_infer_request();
        auto req_fused = compiled_fused.create_infer_request();

        run_once(req_basic, decode_len, -0.02f, 0.02f, 456u);
        run_once(req_fused, decode_len, -0.02f, 0.02f, 456u);

        const auto decode_basic = req_basic.get_output_tensor(0);
        const auto decode_fused = req_fused.get_output_tensor(0);
        ASSERT_EQ(decode_basic.get_shape(), (ov::Shape{1, decode_len, hidden}));
        ASSERT_EQ(decode_fused.get_shape(), (ov::Shape{1, decode_len, hidden}));

        const float decode_diff = max_abs_diff(decode_basic, decode_fused);
        EXPECT_LT(decode_diff, test_utils::k_tol_linear_attn)
            << "Decode max diff: " << decode_diff
            << ", tol: " << test_utils::k_tol_linear_attn;

        const auto basic_state_shapes = collect_state_shapes(req_basic);
        const auto fused_state_shapes = collect_state_shapes(req_fused);
        ASSERT_EQ(basic_state_shapes.size(), fused_state_shapes.size());
        for (const auto& kv : basic_state_shapes) {
            const auto it = fused_state_shapes.find(kv.first);
            ASSERT_NE(it, fused_state_shapes.end()) << "Missing state in fused path: " << kv.first;
            EXPECT_EQ(kv.second, it->second) << "State shape mismatch for: " << kv.first;
        }
    }
}
