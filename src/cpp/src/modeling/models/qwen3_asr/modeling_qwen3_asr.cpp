// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_asr/modeling_qwen3_asr.hpp"

#include <cmath>
#include <limits>

#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

ov::genai::modeling::Tensor normalize_asr_position_ids(const ov::genai::modeling::Tensor& position_ids) {
    // vLLM Qwen3-ASR uses MRoPE-style position_ids with shape [3, B, S].
    // Qwen3 text stack expects [B, S], so use the first plane.
    return ov::genai::modeling::ops::slice(position_ids, 0, 1, 1, 0).squeeze(0);
}

ov::Tensor create_sinusoidal_position_embedding(size_t length, size_t channels, int64_t max_timescale = 10000) {
    if (channels % 2 != 0) {
        OPENVINO_THROW("Sinusoidal embedding channels must be even");
    }
    ov::Tensor table(ov::element::f32, ov::Shape{length, channels});
    float* data = table.data<float>();

    const size_t half = channels / 2;
    const float log_inc = std::log(static_cast<float>(max_timescale)) / static_cast<float>(half - 1);
    for (size_t pos = 0; pos < length; ++pos) {
        for (size_t i = 0; i < half; ++i) {
            const float inv = std::exp(-log_inc * static_cast<float>(i));
            const float v = static_cast<float>(pos) * inv;
            data[pos * channels + i] = std::sin(v);
            data[pos * channels + half + i] = std::cos(v);
        }
    }
    return table;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

int64_t qwen3_asr_feat_extract_output_length(int64_t input_length) {
    auto floor_div = [](int64_t a, int64_t b) -> int64_t {
        if (b == 0) {
            OPENVINO_THROW("Division by zero in qwen3_asr_feat_extract_output_length");
        }
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

namespace {

Tensor compute_cnn_output_lengths(const Tensor& input_lengths) {
    auto* ctx = input_lengths.context();
    auto one = Tensor(ops::const_scalar(ctx, int64_t{1}), ctx);
    auto two = Tensor(ops::const_scalar(ctx, int64_t{2}), ctx);

    auto lengths = input_lengths.to(ov::element::i64);
    for (int i = 0; i < 3; ++i) {
        lengths = ((lengths - one) / two) + one;
    }
    return lengths;
}

Tensor build_valid_mask(const Tensor& lengths, const Tensor& sequence_tensor) {
    auto* ctx = sequence_tensor.context();
    auto seq_len = Tensor(shape::dim(sequence_tensor, 1), ctx).squeeze(0);
    auto idx = ops::range(seq_len, 0, 1, ov::element::i64).unsqueeze(0);  // [1, T]
    auto one = Tensor(ops::const_scalar(ctx, int64_t{1}), ctx);
    auto len_minus_one = (lengths.to(ov::element::i64) - one).unsqueeze(1);  // [B, 1]
    return ops::less_equal(idx, len_minus_one);  // [B, T]
}

class Qwen3ASRAudioAttentionLite : public Module {
public:
    Qwen3ASRAudioAttentionLite(BuilderContext& ctx,
                               const std::string& name,
                               const Qwen3ASRAudioConfig& cfg,
                               Module* parent = nullptr)
        : Module(name, ctx, parent),
          num_heads_(cfg.encoder_attention_heads),
          head_dim_(cfg.d_model / cfg.encoder_attention_heads),
          scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
        q_proj_w_ = &register_parameter("q_proj.weight");
        q_proj_b_ = &register_parameter("q_proj.bias");
        k_proj_w_ = &register_parameter("k_proj.weight");
        k_proj_b_ = &register_parameter("k_proj.bias");
        v_proj_w_ = &register_parameter("v_proj.weight");
        v_proj_b_ = &register_parameter("v_proj.bias");
        out_proj_w_ = &register_parameter("out_proj.weight");
        out_proj_b_ = &register_parameter("out_proj.bias");
    }

    Tensor forward(const Tensor& hidden_states, const Tensor& valid_mask) const {
        auto* op_ctx = hidden_states.context();
        auto q = ops::linear(hidden_states, q_proj_w_->value()) + q_proj_b_->value();
        auto k = ops::linear(hidden_states, k_proj_w_->value()) + k_proj_b_->value();
        auto v = ops::linear(hidden_states, v_proj_w_->value()) + v_proj_b_->value();

        auto qh = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
        auto kh = k.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
        auto vh = v.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});

        auto zero = Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx);
        auto neg = Tensor(ops::const_scalar(op_ctx, -65504.0f), op_ctx);
        auto pad_mask = ops::where(valid_mask, zero, neg).unsqueeze({1, 2});  // [B,1,1,T]

        auto attn = ops::llm::sdpa(qh, kh, vh, scale_, 3, &pad_mask, false, &ctx().op_policy());
        auto merged = attn.permute({0, 2, 1, 3}).reshape({0, 0, static_cast<int64_t>(num_heads_ * head_dim_)});
        return ops::linear(merged, out_proj_w_->value()) + out_proj_b_->value();
    }

private:
    int32_t num_heads_ = 0;
    int32_t head_dim_ = 0;
    float scale_ = 1.0f;

    WeightParameter* q_proj_w_ = nullptr;
    WeightParameter* q_proj_b_ = nullptr;
    WeightParameter* k_proj_w_ = nullptr;
    WeightParameter* k_proj_b_ = nullptr;
    WeightParameter* v_proj_w_ = nullptr;
    WeightParameter* v_proj_b_ = nullptr;
    WeightParameter* out_proj_w_ = nullptr;
    WeightParameter* out_proj_b_ = nullptr;
};

class Qwen3ASRAudioEncoderLayerLite : public Module {
public:
    Qwen3ASRAudioEncoderLayerLite(BuilderContext& ctx,
                                  const std::string& name,
                                  const Qwen3ASRAudioConfig& cfg,
                                  Module* parent = nullptr)
        : Module(name, ctx, parent),
          self_attn_(ctx, "self_attn", cfg, this) {
        ln1_w_ = &register_parameter("self_attn_layer_norm.weight");
        ln1_b_ = &register_parameter("self_attn_layer_norm.bias");
        fc1_w_ = &register_parameter("fc1.weight");
        fc1_b_ = &register_parameter("fc1.bias");
        fc2_w_ = &register_parameter("fc2.weight");
        fc2_b_ = &register_parameter("fc2.bias");
        ln2_w_ = &register_parameter("final_layer_norm.weight");
        ln2_b_ = &register_parameter("final_layer_norm.bias");
    }

    Tensor forward(const Tensor& x, const Tensor& valid_mask, const std::string& activation) const {
        auto norm1 = ops::nn::layer_norm(x, ln1_w_->value(), &ln1_b_->value(), 1e-5f, -1);
        auto attn = self_attn_.forward(norm1, valid_mask);
        auto h = x + attn;

        auto norm2 = ops::nn::layer_norm(h, ln2_w_->value(), &ln2_b_->value(), 1e-5f, -1);
        auto ff = ops::linear(norm2, fc1_w_->value()) + fc1_b_->value();
        if (activation == "relu") {
            ff = ops::nn::relu(ff);
        } else {
            ff = ops::nn::gelu(ff, true);
        }
        ff = ops::linear(ff, fc2_w_->value()) + fc2_b_->value();
        h = h + ff;

        auto keep = valid_mask.unsqueeze(2).to(h.dtype());
        return h * keep;
    }

private:
    Qwen3ASRAudioAttentionLite self_attn_;
    WeightParameter* ln1_w_ = nullptr;
    WeightParameter* ln1_b_ = nullptr;
    WeightParameter* fc1_w_ = nullptr;
    WeightParameter* fc1_b_ = nullptr;
    WeightParameter* fc2_w_ = nullptr;
    WeightParameter* fc2_b_ = nullptr;
    WeightParameter* ln2_w_ = nullptr;
    WeightParameter* ln2_b_ = nullptr;
};

class Qwen3ASRAudioEncoderLite : public Module {
public:
    Qwen3ASRAudioEncoderLite(BuilderContext& ctx, const Qwen3ASRAudioConfig& cfg, Module* parent = nullptr)
        : Module("audio_tower", ctx, parent), cfg_(cfg) {
        conv2d1_w_ = &register_parameter("conv2d1.weight");
        conv2d1_b_ = &register_parameter("conv2d1.bias");
        conv2d2_w_ = &register_parameter("conv2d2.weight");
        conv2d2_b_ = &register_parameter("conv2d2.bias");
        conv2d3_w_ = &register_parameter("conv2d3.weight");
        conv2d3_b_ = &register_parameter("conv2d3.bias");
        conv_out_w_ = &register_parameter("conv_out.weight");

        ln_post_w_ = &register_parameter("ln_post.weight");
        ln_post_b_ = &register_parameter("ln_post.bias");
        proj1_w_ = &register_parameter("proj1.weight");
        proj1_b_ = &register_parameter("proj1.bias");
        proj2_w_ = &register_parameter("proj2.weight");
        proj2_b_ = &register_parameter("proj2.bias");

        layers_.reserve(static_cast<size_t>(cfg.encoder_layers));
        for (int32_t i = 0; i < cfg.encoder_layers; ++i) {
            layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", cfg, this);
        }

        positional_const_ = ops::constant(
            create_sinusoidal_position_embedding(static_cast<size_t>(cfg.max_source_positions),
                                                 static_cast<size_t>(cfg.d_model)),
            &ctx.op_context());
    }

    std::pair<Tensor, Tensor> forward(const Tensor& input_features, const Tensor& audio_feature_lengths) const {
        // input_features: [B, mel, T]
        auto x = input_features.unsqueeze(1);  // [B,1,mel,T]

        x = ops::nn::gelu(ops::nn::conv2d(x, conv2d1_w_->value(), conv2d1_b_->value(), {2, 2}, {1, 1}, {1, 1}), true);
        x = ops::nn::gelu(ops::nn::conv2d(x, conv2d2_w_->value(), conv2d2_b_->value(), {2, 2}, {1, 1}, {1, 1}), true);
        x = ops::nn::gelu(ops::nn::conv2d(x, conv2d3_w_->value(), conv2d3_b_->value(), {2, 2}, {1, 1}, {1, 1}), true);

        // [B,C,F,T] -> [B,T,C,F] -> [B,T,C*F]
        x = x.permute({0, 3, 1, 2});
        x = x.reshape({0, 0, -1});
        x = ops::linear(x, conv_out_w_->value());

        // Add sinusoidal positional embedding
        auto* op_ctx = x.context();
        auto t = Tensor(shape::dim(x, 1), op_ctx).squeeze(0);
        auto pos_idx = ops::range(t, 0, 1, ov::element::i64);
        auto pos = ops::gather(positional_const_, pos_idx, 0);  // [T, D]
        auto pos_b = shape::broadcast_to(pos.unsqueeze(0), shape::make({shape::dim(x, 0), shape::dim(x, 1), shape::dim(x, 2)}));
        x = x + pos_b.to(x.dtype());

        auto cnn_lengths = compute_cnn_output_lengths(audio_feature_lengths);
        auto valid_mask = build_valid_mask(cnn_lengths, x);  // [B, T]

        x = x * valid_mask.unsqueeze(2).to(x.dtype());
        for (const auto& layer : layers_) {
            x = layer.forward(x, valid_mask, cfg_.activation_function);
        }

        x = ops::nn::layer_norm(x, ln_post_w_->value(), &ln_post_b_->value(), 1e-5f, -1);
        x = ops::linear(x, proj1_w_->value()) + proj1_b_->value();
        if (cfg_.activation_function == "relu") {
            x = ops::nn::relu(x);
        } else {
            x = ops::nn::gelu(x, true);
        }
        x = ops::linear(x, proj2_w_->value()) + proj2_b_->value();
        x = x * valid_mask.unsqueeze(2).to(x.dtype());
        return {x, cnn_lengths};
    }

private:
    Qwen3ASRAudioConfig cfg_;
    std::vector<Qwen3ASRAudioEncoderLayerLite> layers_;
    Tensor positional_const_;

    WeightParameter* conv2d1_w_ = nullptr;
    WeightParameter* conv2d1_b_ = nullptr;
    WeightParameter* conv2d2_w_ = nullptr;
    WeightParameter* conv2d2_b_ = nullptr;
    WeightParameter* conv2d3_w_ = nullptr;
    WeightParameter* conv2d3_b_ = nullptr;
    WeightParameter* conv_out_w_ = nullptr;

    WeightParameter* ln_post_w_ = nullptr;
    WeightParameter* ln_post_b_ = nullptr;
    WeightParameter* proj1_w_ = nullptr;
    WeightParameter* proj1_b_ = nullptr;
    WeightParameter* proj2_w_ = nullptr;
    WeightParameter* proj2_b_ = nullptr;
};

}  // namespace

std::shared_ptr<ov::Model> create_qwen3_asr_text_model(
    const Qwen3ASRTextConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds,
    bool enable_audio_inputs) {
    Qwen3DenseConfig dense_cfg;
    dense_cfg.architecture = cfg.architecture;
    dense_cfg.hidden_size = cfg.hidden_size;
    dense_cfg.num_attention_heads = cfg.num_attention_heads;
    dense_cfg.num_key_value_heads = cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads;
    dense_cfg.head_dim = cfg.head_dim;
    dense_cfg.intermediate_size = cfg.intermediate_size;
    dense_cfg.num_hidden_layers = cfg.num_hidden_layers;
    dense_cfg.rms_norm_eps = cfg.rms_norm_eps;
    dense_cfg.rope_theta = cfg.rope_theta;
    dense_cfg.hidden_act = cfg.hidden_act;
    dense_cfg.attention_bias = cfg.attention_bias;
    dense_cfg.tie_word_embeddings = cfg.tie_word_embeddings;

    BuilderContext ctx;
    Qwen3ForCausalLM model(ctx, dense_cfg);

    // Support raw HF ASR prefixes and vLLM-remapped prefixes.
    model.packed_mapping().rules.push_back({"thinker.model.", "model.", 0});
    model.packed_mapping().rules.push_back({"thinker.lm_head.", "lm_head.", 0});
    model.packed_mapping().rules.push_back({"language_model.model.", "model.", 0});
    model.packed_mapping().rules.push_back({"language_model.lm_head.", "lm_head.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_missing = false;
    options.allow_unmatched = true;  // ASR checkpoints also contain audio tower weights.
    options.report_missing = true;
    options.report_unmatched = false;
    (void)ov::genai::modeling::weights::load_model(model, source, finalizer, options);

    auto attention_mask = ctx.parameter(Qwen3ASRTextIO::kAttentionMask, ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter(Qwen3ASRTextIO::kPositionIds, ov::element::i64, ov::PartialShape{3, -1, -1});
    auto beam_idx = ctx.parameter(Qwen3ASRTextIO::kBeamIdx, ov::element::i32, ov::PartialShape{-1});

    const auto float_type = ov::element::f32;
    Tensor input_ids;
    Tensor inputs_embeds;
    if (use_inputs_embeds) {
        inputs_embeds =
            ctx.parameter(Qwen3ASRTextIO::kInputsEmbeds, float_type, ov::PartialShape{-1, -1, cfg.hidden_size});
    } else {
        input_ids = ctx.parameter(Qwen3ASRTextIO::kInputIds, ov::element::i64, ov::PartialShape{-1, -1});
        inputs_embeds = model.model().embed_tokens().forward(input_ids);
    }

    if (enable_audio_inputs) {
        auto audio_embeds =
            ctx.parameter(Qwen3ASRTextIO::kAudioEmbeds, float_type, ov::PartialShape{-1, -1, cfg.hidden_size});
        auto audio_pos_mask =
            ctx.parameter(Qwen3ASRTextIO::kAudioPosMask, ov::element::boolean, ov::PartialShape{-1, -1});
        inputs_embeds = ops::tensor::masked_scatter(inputs_embeds, audio_pos_mask.unsqueeze(2), audio_embeds.to(inputs_embeds.dtype()));
    }

    auto position_ids_2d = normalize_asr_position_ids(position_ids);
    auto logits = model.forward_embeds(inputs_embeds, position_ids_2d, beam_idx, attention_mask);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, Qwen3ASRTextIO::kLogits);
    auto ov_model = ctx.build_model({result->output(0)});

    // Match qwen3 runtime options to reduce KV-cache bandwidth pressure during decode.
    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});

    return ov_model;
}

std::shared_ptr<ov::Model> create_qwen3_asr_audio_encoder_model(
    const Qwen3ASRAudioConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3ASRAudioEncoderLite model(ctx, cfg);

    for (int32_t i = 0; i < cfg.encoder_layers; ++i) {
        const std::string idx = std::to_string(i);
        model.packed_mapping().rules.push_back({"thinker.audio_tower.layers." + idx + ".", "audio_tower.layers[" + idx + "].", 0});
        model.packed_mapping().rules.push_back({"audio_tower.layers." + idx + ".", "audio_tower.layers[" + idx + "].", 0});
    }
    model.packed_mapping().rules.push_back({"thinker.audio_tower.", "audio_tower.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_missing = false;
    options.allow_unmatched = true;
    options.report_missing = true;
    options.report_unmatched = false;
    (void)ov::genai::modeling::weights::load_model(model, source, finalizer, options);

    auto input_audio_features =
        ctx.parameter(Qwen3ASRAudioIO::kInputAudioFeatures, ov::element::f32, ov::PartialShape{-1, -1, -1});
    auto audio_feature_lengths =
        ctx.parameter(Qwen3ASRAudioIO::kAudioFeatureLengths, ov::element::i64, ov::PartialShape{-1});

    auto out = model.forward(input_audio_features, audio_feature_lengths);

    auto embeds_res = std::make_shared<ov::op::v0::Result>(out.first.output());
    set_name(embeds_res, Qwen3ASRAudioIO::kAudioEmbeds);
    auto lens_res = std::make_shared<ov::op::v0::Result>(out.second.output());
    set_name(lens_res, Qwen3ASRAudioIO::kAudioOutputLengths);
    return ctx.build_model({embeds_res->output(0), lens_res->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

// ============================================================================
// Model Builder Registration
// ============================================================================

#include "loaders/model_builder.hpp"

namespace {

std::shared_ptr<ov::Model> build_qwen3_asr_model(
    const ov::genai::loaders::ModelConfig& config,
    ov::genai::modeling::weights::WeightSource& weight_source,
    ov::genai::modeling::weights::WeightFinalizer& weight_finalizer) {
    using namespace ov::genai::modeling::models;

    if (config.hidden_size <= 0 || config.num_hidden_layers <= 0 || config.num_attention_heads <= 0 ||
        config.intermediate_size <= 0) {
        OPENVINO_THROW("Invalid Qwen3-ASR config: hidden_size/num_hidden_layers/num_attention_heads/intermediate_size must be > 0");
    }

    Qwen3ASRTextConfig cfg;
    cfg.architecture = "qwen3_asr";
    cfg.vocab_size = config.vocab_size;
    cfg.hidden_size = config.hidden_size;
    cfg.intermediate_size = config.intermediate_size;
    cfg.num_hidden_layers = config.num_hidden_layers;
    cfg.num_attention_heads = config.num_attention_heads;
    cfg.num_key_value_heads = config.num_key_value_heads > 0 ? config.num_key_value_heads : config.num_attention_heads;
    cfg.head_dim = config.head_dim > 0 ? config.head_dim : (config.hidden_size / config.num_attention_heads);
    cfg.max_position_embeddings = config.max_position_embeddings;
    cfg.rms_norm_eps = config.rms_norm_eps;
    cfg.rope_theta = config.rope_theta;
    cfg.hidden_act = config.hidden_act;
    cfg.attention_bias = config.attention_bias;
    cfg.tie_word_embeddings = config.tie_word_embeddings;

    return create_qwen3_asr_text_model(cfg, weight_source, weight_finalizer, false, true);
}

std::shared_ptr<ov::Model> build_qwen3_asr_audio_encoder_model(
    const ov::genai::loaders::ModelConfig& config,
    ov::genai::modeling::weights::WeightSource& weight_source,
    ov::genai::modeling::weights::WeightFinalizer& weight_finalizer) {
    using namespace ov::genai::modeling::models;

    Qwen3ASRAudioConfig cfg;
    cfg.architecture = "qwen3_asr_audio_encoder";
    cfg.d_model = config.hidden_size > 0 ? config.hidden_size : 1280;
    cfg.encoder_layers = config.num_hidden_layers > 0 ? config.num_hidden_layers : 32;
    cfg.encoder_attention_heads = config.num_attention_heads > 0 ? config.num_attention_heads : 20;
    cfg.encoder_ffn_dim = config.intermediate_size > 0 ? config.intermediate_size : 5120;
    cfg.max_source_positions = config.max_position_embeddings > 0 ? config.max_position_embeddings : 1500;
    cfg.num_mel_bins = 128;
    cfg.downsample_hidden_size = 480;
    cfg.output_dim = 3584;
    cfg.activation_function = config.hidden_act.empty() ? "gelu" : config.hidden_act;
    return create_qwen3_asr_audio_encoder_model(cfg, weight_source, weight_finalizer);
}

static bool _registered_qwen3_asr = []() {
    ov::genai::loaders::ModelBuilder::instance().register_architecture("qwen3_asr", build_qwen3_asr_model);
    ov::genai::loaders::ModelBuilder::instance().register_architecture("qwen3asr", build_qwen3_asr_model);
    ov::genai::loaders::ModelBuilder::instance().register_architecture("qwen3_asr_text", build_qwen3_asr_model);
    ov::genai::loaders::ModelBuilder::instance().register_architecture("qwen3_asr_audio_encoder", build_qwen3_asr_audio_encoder_model);
    ov::genai::loaders::ModelBuilder::instance().register_architecture("qwen3asraudioencoder", build_qwen3_asr_audio_encoder_model);
    return true;
}();

}  // namespace
