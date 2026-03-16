
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/deepseek_ocr2/modeling_deepseek_sam_vit.hpp"

#include <algorithm>
#include <cmath>

#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/op/interpolate.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
}

auto set_name = [](const std::shared_ptr<ov::Node>& node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

ov::genai::modeling::Tensor to_i64(const ov::genai::modeling::Tensor& x) {
    if (x.dtype() == ov::element::i64) {
        return x;
    }
    return x.to(ov::element::i64);
}

ov::Output<ov::Node> mul_dim(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b) {
    return std::make_shared<ov::op::v1::Multiply>(a, b);
}

ov::Output<ov::Node> add_dim(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b) {
    return std::make_shared<ov::op::v1::Add>(a, b);
}

ov::Output<ov::Node> div_dim(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b) {
    return std::make_shared<ov::op::v1::Divide>(a, b, ov::op::AutoBroadcastType::NUMPY);
}

ov::Output<ov::Node> as_shape_dim(const ov::Output<ov::Node>& value, ov::genai::modeling::OpContext* ctx) {
    auto rank = value.get_partial_shape().rank();
    if (rank.is_static() && rank.get_length() == 0) {
        ov::genai::modeling::Tensor t(value, ctx);
        return t.unsqueeze(0).output();
    }
    return value;
}

ov::genai::modeling::Tensor reduce_sum(const ov::genai::modeling::Tensor& x, int64_t axis, bool keepdim) {
    auto* ctx = x.context();
    auto axis_node = ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{axis});
    auto node = std::make_shared<ov::op::v1::ReduceSum>(x.output(), axis_node, keepdim);
    return ov::genai::modeling::Tensor(node, ctx);
}

ov::genai::modeling::Tensor dynamic_pad_nhwc(const ov::genai::modeling::Tensor& x,
                                             const ov::genai::modeling::Tensor& pad_h,
                                             const ov::genai::modeling::Tensor& pad_w) {
    auto* ctx = x.context();
    auto zero_vec = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{0}), ctx);
    auto pad_h_vec = pad_h.unsqueeze(0);
    auto pad_w_vec = pad_w.unsqueeze(0);
    auto pads_begin = ov::genai::modeling::ops::concat({zero_vec, zero_vec, zero_vec, zero_vec}, 0);
    auto pads_end = ov::genai::modeling::ops::concat({zero_vec, pad_h_vec, pad_w_vec, zero_vec}, 0);
    auto pad_value = ov::genai::modeling::ops::const_scalar(ctx, 0.0f);
    auto node = std::make_shared<ov::op::v1::Pad>(x.output(),
                                                  pads_begin.output(),
                                                  pads_end.output(),
                                                  pad_value,
                                                  ov::op::PadMode::CONSTANT);
    return ov::genai::modeling::Tensor(node, ctx);
}

ov::genai::modeling::Tensor slice_nhwc(const ov::genai::modeling::Tensor& x,
                                       const ov::genai::modeling::Tensor& out_h,
                                       const ov::genai::modeling::Tensor& out_w) {
    auto* ctx = x.context();
    auto batch = ov::genai::modeling::Tensor(ov::genai::modeling::shape::dim(x, 0), ctx);
    auto channels = ov::genai::modeling::Tensor(ov::genai::modeling::shape::dim(x, 3), ctx);
    auto zero_vec = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{0}), ctx);
    auto one_vec = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{1}), ctx);

    auto begin = ov::genai::modeling::ops::concat({zero_vec, zero_vec, zero_vec, zero_vec}, 0);
    auto end = ov::genai::modeling::ops::concat({batch,
                                                 out_h.unsqueeze(0),
                                                 out_w.unsqueeze(0),
                                                 channels}, 0);
    auto step = ov::genai::modeling::ops::concat({one_vec, one_vec, one_vec, one_vec}, 0);

    auto node = std::make_shared<ov::op::v8::Slice>(x.output(), begin.output(), end.output(), step.output());
    return ov::genai::modeling::Tensor(node, ctx);
}

ov::genai::modeling::Tensor interpolate_1d(const ov::genai::modeling::Tensor& x,
                                           const ov::Output<ov::Node>& target_len) {
    auto* ctx = x.context();
    auto axes = ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{2});
    auto sizes = ov::genai::modeling::shape::make({as_shape_dim(target_len, ctx)});

    ov::op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::LINEAR;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
    attrs.nearest_mode = ov::op::v11::Interpolate::NearestMode::FLOOR;

    auto node = std::make_shared<ov::op::v11::Interpolate>(x.output(), sizes, axes, attrs);
    return ov::genai::modeling::Tensor(node, ctx);
}

ov::genai::modeling::Tensor interpolate_2d_bicubic(const ov::genai::modeling::Tensor& x,
                                                   const ov::Output<ov::Node>& out_h,
                                                   const ov::Output<ov::Node>& out_w) {
    auto* ctx = x.context();
    auto axes = ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{2, 3});
    auto sizes = ov::genai::modeling::shape::make({as_shape_dim(out_h, ctx), as_shape_dim(out_w, ctx)});

    ov::op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::CUBIC;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
    attrs.nearest_mode = ov::op::v11::Interpolate::NearestMode::FLOOR;
    attrs.antialias = true;
    attrs.cube_coeff = -0.75f;

    auto node = std::make_shared<ov::op::v11::Interpolate>(x.output(), sizes, axes, attrs);
    return ov::genai::modeling::Tensor(node, ctx);
}

ov::genai::modeling::Tensor get_abs_pos_sam(const ov::genai::modeling::Tensor& abs_pos,
                                            const ov::genai::modeling::Tensor& tgt_h,
                                            const ov::genai::modeling::Tensor& tgt_w) {
    auto orig_dtype = abs_pos.dtype();

    auto pos = abs_pos.permute({0, 3, 1, 2}).to(ov::element::f32);
    auto interp = interpolate_2d_bicubic(pos, tgt_h.output(), tgt_w.output());
    auto restored = interp.to(orig_dtype).permute({0, 2, 3, 1});
    return restored;
}

ov::genai::modeling::Tensor get_rel_pos(const ov::genai::modeling::Tensor& rel_pos,
                                        const ov::genai::modeling::Tensor& q_size,
                                        const ov::genai::modeling::Tensor& k_size) {
    auto* ctx = rel_pos.context();
    auto orig_dtype = rel_pos.dtype();

    auto q_i64 = to_i64(q_size);
    auto k_i64 = to_i64(k_size);
    auto max_qk = std::make_shared<ov::op::v1::Maximum>(q_i64.output(), k_i64.output());
    auto two = ov::genai::modeling::ops::const_scalar(ctx, static_cast<int64_t>(2));
    auto max_rel = std::make_shared<ov::op::v1::Multiply>(max_qk, two);
    auto one = ov::genai::modeling::ops::const_scalar(ctx, static_cast<int64_t>(1));
    auto max_rel_dist = std::make_shared<ov::op::v1::Subtract>(max_rel, one);

    auto rel_pos_f = rel_pos.to(ov::element::f32);
    auto len = ov::genai::modeling::shape::dim(rel_pos, 0);
    auto channels = ov::genai::modeling::shape::dim(rel_pos, 1);
    auto one_dim = ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{1});
    auto reshape_shape = ov::genai::modeling::shape::make({one_dim, len, channels});
    auto reshaped = rel_pos_f.reshape(reshape_shape, false).permute({0, 2, 1});
    auto resized = interpolate_1d(reshaped, max_rel_dist);
    auto resized_back = resized.permute({0, 2, 1}).to(orig_dtype);
    auto out_shape = ov::genai::modeling::shape::make({as_shape_dim(max_rel_dist, ctx), channels});
    auto rel_pos_resized = resized_back.reshape(out_shape, false);

    auto q_f = q_i64.to(ov::element::f32);
    auto k_f = k_i64.to(ov::element::f32);
    auto one_f = ov::genai::modeling::ops::const_scalar(ctx, 1.0f);
    auto q_scale = std::make_shared<ov::op::v1::Divide>(k_f.output(), q_f.output(), ov::op::AutoBroadcastType::NUMPY);
    auto k_scale = std::make_shared<ov::op::v1::Divide>(q_f.output(), k_f.output(), ov::op::AutoBroadcastType::NUMPY);
    auto q_scale_max = std::make_shared<ov::op::v1::Maximum>(q_scale, one_f);
    auto k_scale_max = std::make_shared<ov::op::v1::Maximum>(k_scale, one_f);

    auto q_range = ov::genai::modeling::ops::range(q_i64, 0, 1, ov::element::i32).to(ov::element::f32);
    auto k_range = ov::genai::modeling::ops::range(k_i64, 0, 1, ov::element::i32).to(ov::element::f32);

    auto q_coords = q_range.unsqueeze(1) * ov::genai::modeling::Tensor(q_scale_max, ctx);
    auto k_coords = k_range.unsqueeze(0) * ov::genai::modeling::Tensor(k_scale_max, ctx);
    auto k_minus_one = ov::genai::modeling::Tensor(std::make_shared<ov::op::v1::Subtract>(k_f.output(), one_f), ctx);
    auto rel = q_coords - k_coords + k_minus_one * ov::genai::modeling::Tensor(k_scale_max, ctx);
    auto rel_idx = rel.to(ov::element::i64);

    return ov::genai::modeling::ops::gather(rel_pos_resized, rel_idx, 0);
}

std::pair<ov::genai::modeling::Tensor, ov::genai::modeling::Tensor>
add_decomposed_rel_pos(const ov::genai::modeling::Tensor& q,
                       const ov::genai::modeling::Tensor& rel_pos_h,
                       const ov::genai::modeling::Tensor& rel_pos_w,
                       const ov::genai::modeling::Tensor& q_h,
                       const ov::genai::modeling::Tensor& q_w,
                       const ov::genai::modeling::Tensor& k_h,
                       const ov::genai::modeling::Tensor& k_w) {
    auto* ctx = q.context();
    auto q_h_i64 = to_i64(q_h);
    auto q_w_i64 = to_i64(q_w);
    auto k_h_i64 = to_i64(k_h);
    auto k_w_i64 = to_i64(k_w);

    auto rel_h = get_rel_pos(rel_pos_h, q_h_i64, k_h_i64);
    auto rel_w = get_rel_pos(rel_pos_w, q_w_i64, k_w_i64);

    auto batch = ov::genai::modeling::shape::dim(q, 0);
    auto dim = ov::genai::modeling::shape::dim(q, 2);
    auto q_shape = ov::genai::modeling::shape::make({batch,
                                                     as_shape_dim(q_h_i64.output(), ctx),
                                                     as_shape_dim(q_w_i64.output(), ctx),
                                                     dim});
    auto q_reshaped = q.reshape(q_shape, false);

    auto rel_h_exp = rel_h.unsqueeze(0).unsqueeze(2);  // [1, q_h, 1, k_h, dim]
    auto rel_w_exp = rel_w.unsqueeze(0).unsqueeze(1);  // [1, 1, q_w, k_w, dim]

    auto q_exp = q_reshaped.unsqueeze(3);  // [B, q_h, q_w, 1, dim]
    auto rel_h_mul = q_exp * rel_h_exp;
    auto rel_w_mul = q_exp * rel_w_exp;

    // Keep reduced axis, then squeeze explicitly to avoid backend rank-mismatch issues.
    auto rel_h_sum = reduce_sum(rel_h_mul, 4, true).squeeze(4);
    auto rel_w_sum = reduce_sum(rel_w_mul, 4, true).squeeze(4);

    auto one_dim = ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{1});
    auto hw = mul_dim(q_h_i64.output(), q_w_i64.output());
    auto rel_h_shape = ov::genai::modeling::shape::make({batch,
                                                         as_shape_dim(hw, ctx),
                                                         as_shape_dim(k_h_i64.output(), ctx),
                                                         one_dim});
    auto rel_w_shape = ov::genai::modeling::shape::make({batch,
                                                         as_shape_dim(hw, ctx),
                                                         one_dim,
                                                         as_shape_dim(k_w_i64.output(), ctx)});
    auto rel_h_out = rel_h_sum.reshape(rel_h_shape, false);
    auto rel_w_out = rel_w_sum.reshape(rel_w_shape, false);
    return {rel_h_out, rel_w_out};
}

struct WindowPartitionResult {
    ov::genai::modeling::Tensor windows;
    ov::genai::modeling::Tensor padded_h;
    ov::genai::modeling::Tensor padded_w;
};

WindowPartitionResult window_partition(const ov::genai::modeling::Tensor& x, int32_t window_size) {
    auto* ctx = x.context();
    auto h = ov::genai::modeling::Tensor(ov::genai::modeling::shape::dim(x, 1), ctx).squeeze(0);
    auto w = ov::genai::modeling::Tensor(ov::genai::modeling::shape::dim(x, 2), ctx).squeeze(0);
    auto window = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_scalar(ctx, static_cast<int64_t>(window_size)), ctx);

    auto h_mod = ov::genai::modeling::Tensor(std::make_shared<ov::op::v1::Mod>(h.output(),
                                                                              window.output(),
                                                                              ov::op::AutoBroadcastType::NUMPY), ctx);
    auto w_mod = ov::genai::modeling::Tensor(std::make_shared<ov::op::v1::Mod>(w.output(),
                                                                              window.output(),
                                                                              ov::op::AutoBroadcastType::NUMPY), ctx);
    auto h_pad_raw = ov::genai::modeling::Tensor(std::make_shared<ov::op::v1::Subtract>(window.output(),
                                                                                       h_mod.output(),
                                                                                       ov::op::AutoBroadcastType::NUMPY), ctx);
    auto w_pad_raw = ov::genai::modeling::Tensor(std::make_shared<ov::op::v1::Subtract>(window.output(),
                                                                                       w_mod.output(),
                                                                                       ov::op::AutoBroadcastType::NUMPY), ctx);
    auto pad_h = ov::genai::modeling::Tensor(std::make_shared<ov::op::v1::Mod>(h_pad_raw.output(),
                                                                              window.output(),
                                                                              ov::op::AutoBroadcastType::NUMPY), ctx);
    auto pad_w = ov::genai::modeling::Tensor(std::make_shared<ov::op::v1::Mod>(w_pad_raw.output(),
                                                                              window.output(),
                                                                              ov::op::AutoBroadcastType::NUMPY), ctx);

    auto padded = dynamic_pad_nhwc(x, pad_h, pad_w);

    auto hp = ov::genai::modeling::Tensor(add_dim(h.output(), pad_h.output()), ctx);
    auto wp = ov::genai::modeling::Tensor(add_dim(w.output(), pad_w.output()), ctx);

    auto hp_div = ov::genai::modeling::Tensor(div_dim(hp.output(), window.output()), ctx);
    auto wp_div = ov::genai::modeling::Tensor(div_dim(wp.output(), window.output()), ctx);
    auto hp_div_i64 = to_i64(hp_div);
    auto wp_div_i64 = to_i64(wp_div);

    auto batch = ov::genai::modeling::shape::dim(padded, 0);
    auto channels = ov::genai::modeling::shape::dim(padded, 3);
    auto win = ov::genai::modeling::ops::const_scalar(ctx, static_cast<int64_t>(window_size));

    auto shape1 = ov::genai::modeling::shape::make({batch,
                                                    as_shape_dim(hp_div_i64.output(), ctx),
                                                    as_shape_dim(win, ctx),
                                                    as_shape_dim(wp_div_i64.output(), ctx),
                                                    as_shape_dim(win, ctx),
                                                    channels});
    auto reshaped = padded.reshape(shape1, false);
    auto permuted = reshaped.permute({0, 1, 3, 2, 4, 5});

    auto num_windows = mul_dim(mul_dim(batch, hp_div_i64.output()), wp_div_i64.output());
    auto windows_shape = ov::genai::modeling::shape::make({as_shape_dim(num_windows, ctx),
                                                           as_shape_dim(win, ctx),
                                                           as_shape_dim(win, ctx),
                                                           channels});
    auto windows = permuted.reshape(windows_shape, false);

    return {windows, hp, wp};
}

ov::genai::modeling::Tensor window_unpartition(const ov::genai::modeling::Tensor& windows,
                                               int32_t window_size,
                                               const ov::genai::modeling::Tensor& padded_h,
                                               const ov::genai::modeling::Tensor& padded_w,
                                               const ov::genai::modeling::Tensor& orig_h,
                                               const ov::genai::modeling::Tensor& orig_w) {
    auto* ctx = windows.context();
    auto win = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_scalar(ctx, static_cast<int64_t>(window_size)), ctx);
    auto hp_div = ov::genai::modeling::Tensor(div_dim(padded_h.output(), win.output()), ctx);
    auto wp_div = ov::genai::modeling::Tensor(div_dim(padded_w.output(), win.output()), ctx);
    auto hp_div_i64 = to_i64(hp_div);
    auto wp_div_i64 = to_i64(wp_div);

    auto total = ov::genai::modeling::shape::dim(windows, 0);
    auto denom = mul_dim(hp_div_i64.output(), wp_div_i64.output());
    auto batch = std::make_shared<ov::op::v1::Divide>(total, denom, ov::op::AutoBroadcastType::NUMPY);
    auto batch_i64 = std::make_shared<ov::op::v0::Convert>(batch, ov::element::i64);

    auto channels = ov::genai::modeling::shape::dim(windows, 3);
    auto shape1 = ov::genai::modeling::shape::make({as_shape_dim(batch_i64, ctx),
                                                    as_shape_dim(hp_div_i64.output(), ctx),
                                                    as_shape_dim(wp_div_i64.output(), ctx),
                                                    as_shape_dim(win.output(), ctx),
                                                    as_shape_dim(win.output(), ctx),
                                                    channels});
    auto reshaped = windows.reshape(shape1, false);
    auto permuted = reshaped.permute({0, 1, 3, 2, 4, 5});

    auto shape2 = ov::genai::modeling::shape::make({as_shape_dim(batch_i64, ctx),
                                                    as_shape_dim(padded_h.output(), ctx),
                                                    as_shape_dim(padded_w.output(), ctx),
                                                    channels});
    auto merged = permuted.reshape(shape2, false);

    return slice_nhwc(merged, orig_h, orig_w);
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

DeepseekSamPatchEmbed::DeepseekSamPatchEmbed(BuilderContext& ctx,
                                             const std::string& name,
                                             const DeepseekSamConfig& cfg,
                                             Module* parent)
    : Module(name, ctx, parent),
      in_channels_(cfg.in_channels),
      patch_size_(cfg.patch_size),
      embed_dim_(cfg.embed_dim) {
    weight_param_ = &register_parameter("proj.weight");
    bias_param_ = &register_parameter("proj.bias");
}

const Tensor& DeepseekSamPatchEmbed::weight() const {
    if (!weight_param_) {
        OPENVINO_THROW("DeepseekSamPatchEmbed weight parameter not registered");
    }
    return weight_param_->value();
}

const Tensor* DeepseekSamPatchEmbed::bias() const {
    return (bias_param_ && bias_param_->is_bound()) ? &bias_param_->value() : nullptr;
}

Tensor DeepseekSamPatchEmbed::forward(const Tensor& pixel_values) const {
    auto x = pixel_values.to(weight().dtype());
    const std::vector<int64_t> strides = {patch_size_, patch_size_};
    const std::vector<int64_t> pads = {0, 0};
    Tensor conv;
    if (const auto* b = bias()) {
        conv = ops::nn::conv2d(x, weight(), *b, strides, pads, pads);
    } else {
        conv = ops::nn::conv2d(x, weight(), strides, pads, pads);
    }
    return conv.permute({0, 2, 3, 1});
}

DeepseekSamLayerNorm2d::DeepseekSamLayerNorm2d(BuilderContext& ctx,
                                               const std::string& name,
                                               float eps,
                                               Module* parent)
    : Module(name, ctx, parent),
      eps_(eps) {
    weight_param_ = &register_parameter("weight");
    bias_param_ = &register_parameter("bias");
}

const Tensor& DeepseekSamLayerNorm2d::weight() const {
    if (!weight_param_) {
        OPENVINO_THROW("DeepseekSamLayerNorm2d weight parameter not registered");
    }
    return weight_param_->value();
}

const Tensor& DeepseekSamLayerNorm2d::bias() const {
    if (!bias_param_) {
        OPENVINO_THROW("DeepseekSamLayerNorm2d bias parameter not registered");
    }
    return bias_param_->value();
}

Tensor DeepseekSamLayerNorm2d::forward(const Tensor& input) const {
    auto orig_dtype = input.dtype();
    auto x = input.to(ov::element::f32);
    auto mean = x.mean(1, true);
    auto diff = x - mean;
    auto var = diff.pow(2.0f).mean(1, true);
    auto norm = diff * (var + eps_).rsqrt();
    auto out = norm.to(orig_dtype);
    auto weight_broadcast = weight().to(orig_dtype).reshape({1, -1, 1, 1});
    auto bias_broadcast = bias().to(orig_dtype).reshape({1, -1, 1, 1});
    return out * weight_broadcast + bias_broadcast;
}

DeepseekSamMLP::DeepseekSamMLP(BuilderContext& ctx, const std::string& name, Module* parent)
    : Module(name, ctx, parent) {
    fc1_weight_param_ = &register_parameter("lin1.weight");
    fc1_bias_param_ = &register_parameter("lin1.bias");
    fc2_weight_param_ = &register_parameter("lin2.weight");
    fc2_bias_param_ = &register_parameter("lin2.bias");
}

const Tensor& DeepseekSamMLP::fc1_weight() const {
    if (!fc1_weight_param_) {
        OPENVINO_THROW("DeepseekSamMLP fc1 weight parameter not registered");
    }
    return fc1_weight_param_->value();
}

const Tensor* DeepseekSamMLP::fc1_bias() const {
    return (fc1_bias_param_ && fc1_bias_param_->is_bound()) ? &fc1_bias_param_->value() : nullptr;
}

const Tensor& DeepseekSamMLP::fc2_weight() const {
    if (!fc2_weight_param_) {
        OPENVINO_THROW("DeepseekSamMLP fc2 weight parameter not registered");
    }
    return fc2_weight_param_->value();
}

const Tensor* DeepseekSamMLP::fc2_bias() const {
    return (fc2_bias_param_ && fc2_bias_param_->is_bound()) ? &fc2_bias_param_->value() : nullptr;
}

Tensor DeepseekSamMLP::forward(const Tensor& input) const {
    auto fc1 = add_bias_if_present(ops::linear(input, fc1_weight()), fc1_bias());
    auto act = ops::nn::gelu(fc1, true);
    return add_bias_if_present(ops::linear(act, fc2_weight()), fc2_bias());
}

DeepseekSamAttention::DeepseekSamAttention(BuilderContext& ctx,
                                           const std::string& name,
                                           const DeepseekSamConfig& cfg,
                                           Module* parent)
    : Module(name, ctx, parent),
      hidden_size_(cfg.embed_dim),
      num_heads_(cfg.num_heads),
      head_dim_(cfg.embed_dim / std::max(cfg.num_heads, 1)),
      use_rel_pos_(cfg.use_rel_pos) {
    if (hidden_size_ <= 0 || num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid DeepseekSamAttention configuration");
    }
    if (hidden_size_ != num_heads_ * head_dim_) {
        OPENVINO_THROW("DeepseekSamAttention hidden_size must equal num_heads * head_dim");
    }
    qkv_weight_param_ = &register_parameter("qkv.weight");
    qkv_bias_param_ = &register_parameter("qkv.bias");
    proj_weight_param_ = &register_parameter("proj.weight");
    proj_bias_param_ = &register_parameter("proj.bias");
    if (use_rel_pos_) {
        rel_pos_h_param_ = &register_parameter("rel_pos_h");
        rel_pos_w_param_ = &register_parameter("rel_pos_w");
    }
}

const Tensor& DeepseekSamAttention::qkv_weight() const {
    if (!qkv_weight_param_) {
        OPENVINO_THROW("DeepseekSamAttention qkv weight parameter not registered");
    }
    return qkv_weight_param_->value();
}

const Tensor* DeepseekSamAttention::qkv_bias() const {
    return (qkv_bias_param_ && qkv_bias_param_->is_bound()) ? &qkv_bias_param_->value() : nullptr;
}

const Tensor& DeepseekSamAttention::proj_weight() const {
    if (!proj_weight_param_) {
        OPENVINO_THROW("DeepseekSamAttention proj weight parameter not registered");
    }
    return proj_weight_param_->value();
}

const Tensor* DeepseekSamAttention::proj_bias() const {
    return (proj_bias_param_ && proj_bias_param_->is_bound()) ? &proj_bias_param_->value() : nullptr;
}

const Tensor& DeepseekSamAttention::rel_pos_h() const {
    if (!rel_pos_h_param_) {
        OPENVINO_THROW("DeepseekSamAttention rel_pos_h parameter not registered");
    }
    return rel_pos_h_param_->value();
}

const Tensor& DeepseekSamAttention::rel_pos_w() const {
    if (!rel_pos_w_param_) {
        OPENVINO_THROW("DeepseekSamAttention rel_pos_w parameter not registered");
    }
    return rel_pos_w_param_->value();
}

Tensor DeepseekSamAttention::forward(const Tensor& hidden_states) const {
    auto* ctx = hidden_states.context();
    auto qkv = add_bias_if_present(ops::linear(hidden_states, qkv_weight()), qkv_bias());
    auto qkv_reshaped = qkv.reshape({0, -1, 3, num_heads_, head_dim_});
    auto qkv_perm = qkv_reshaped.permute({2, 0, 3, 1, 4});
    auto q = ops::slice(qkv_perm, 0, 1, 1, 0).squeeze(0);
    auto k = ops::slice(qkv_perm, 1, 2, 1, 0).squeeze(0);
    auto v = ops::slice(qkv_perm, 2, 3, 1, 0).squeeze(0);

    const auto batch = shape::dim(q, 0);
    const auto heads = shape::dim(q, 1);
    const auto seq = shape::dim(q, 2);
    const auto head_dim = shape::dim(q, 3);

    Tensor attn_bias;
    if (use_rel_pos_) {
        auto h = Tensor(shape::dim(hidden_states, 1), ctx).squeeze(0);
        auto w = Tensor(shape::dim(hidden_states, 2), ctx).squeeze(0);
        auto q_flat_shape = shape::make({as_shape_dim(mul_dim(batch, heads), ctx), seq, head_dim});
        auto q_flat = q.reshape(q_flat_shape, false);
        auto rel = add_decomposed_rel_pos(q_flat, rel_pos_h(), rel_pos_w(), h, w, h, w);
        auto rel_h = rel.first;
        auto rel_w = rel.second;

        auto rel_h_shape = shape::make({batch, heads, seq, shape::dim(rel_h, 2), shape::dim(rel_h, 3)});
        auto rel_w_shape = shape::make({batch, heads, seq, shape::dim(rel_w, 2), shape::dim(rel_w, 3)});
        auto rel_h_5d = rel_h.reshape(rel_h_shape, false);
        auto rel_w_5d = rel_w.reshape(rel_w_shape, false);

        auto bias = rel_h_5d + rel_w_5d;
        auto kw = shape::dim(rel_w, 3);
        auto kh = shape::dim(rel_h, 2);
        auto hw = mul_dim(kh, kw);
        auto bias_shape = shape::make({batch, heads, seq, as_shape_dim(hw, ctx)});
        attn_bias = bias.reshape(bias_shape, false);
    }

    auto* policy = &this->ctx().op_policy();
    const float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    auto context = ops::llm::sdpa(q, k, v, scaling, 3, use_rel_pos_ ? &attn_bias : nullptr, false, policy);
    auto merged = context.permute({0, 2, 1, 3});
    auto h = Tensor(shape::dim(hidden_states, 1), ctx).squeeze(0);
    auto w = Tensor(shape::dim(hidden_states, 2), ctx).squeeze(0);
    auto hidden_dim = ops::const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(hidden_size_)});
    auto out_shape = shape::make({batch, as_shape_dim(h.output(), ctx), as_shape_dim(w.output(), ctx), hidden_dim});
    auto out = merged.reshape(out_shape, false);
    auto proj = add_bias_if_present(ops::linear(out, proj_weight()), proj_bias());
    return proj;
}

DeepseekSamBlock::DeepseekSamBlock(BuilderContext& ctx,
                                   const std::string& name,
                                   const DeepseekSamConfig& cfg,
                                   int32_t window_size,
                                   Module* parent)
    : Module(name, ctx, parent),
      attn_(ctx, "attn", cfg, this),
      mlp_(ctx, "mlp", this),
      window_size_(window_size),
      eps_(cfg.layer_norm_eps) {
    register_module("attn", &attn_);
    register_module("mlp", &mlp_);

    norm1_weight_param_ = &register_parameter("norm1.weight");
    norm1_bias_param_ = &register_parameter("norm1.bias");
    norm2_weight_param_ = &register_parameter("norm2.weight");
    norm2_bias_param_ = &register_parameter("norm2.bias");
}

const Tensor& DeepseekSamBlock::norm1_weight() const {
    if (!norm1_weight_param_) {
        OPENVINO_THROW("DeepseekSamBlock norm1 weight parameter not registered");
    }
    return norm1_weight_param_->value();
}

const Tensor& DeepseekSamBlock::norm1_bias() const {
    if (!norm1_bias_param_) {
        OPENVINO_THROW("DeepseekSamBlock norm1 bias parameter not registered");
    }
    return norm1_bias_param_->value();
}

const Tensor& DeepseekSamBlock::norm2_weight() const {
    if (!norm2_weight_param_) {
        OPENVINO_THROW("DeepseekSamBlock norm2 weight parameter not registered");
    }
    return norm2_weight_param_->value();
}

const Tensor& DeepseekSamBlock::norm2_bias() const {
    if (!norm2_bias_param_) {
        OPENVINO_THROW("DeepseekSamBlock norm2 bias parameter not registered");
    }
    return norm2_bias_param_->value();
}

Tensor DeepseekSamBlock::forward(const Tensor& hidden_states) const {
    auto shortcut = hidden_states;
    auto norm1 = ops::nn::layer_norm(hidden_states, norm1_weight(), &norm1_bias(), eps_, -1);
    Tensor attn_in = norm1;
    WindowPartitionResult window_data;
    bool use_window = window_size_ > 0;
    if (use_window) {
        window_data = window_partition(attn_in, window_size_);
        attn_in = window_data.windows;
    }

    auto attn_out = attn_.forward(attn_in);
    if (use_window) {
        auto h = Tensor(shape::dim(norm1, 1), norm1.context()).squeeze(0);
        auto w = Tensor(shape::dim(norm1, 2), norm1.context()).squeeze(0);
        attn_out = window_unpartition(attn_out, window_size_,
                                      window_data.padded_h, window_data.padded_w,
                                      h, w);
    }
    auto resid = shortcut + attn_out;
    auto norm2 = ops::nn::layer_norm(resid, norm2_weight(), &norm2_bias(), eps_, -1);
    auto mlp_out = mlp_.forward(norm2);
    return resid + mlp_out;
}

DeepseekSamVisionModel::DeepseekSamVisionModel(BuilderContext& ctx,
                                               const DeepseekSamConfig& cfg,
                                               Module* parent)
    : Module("sam_model", ctx, parent),
      cfg_(cfg),
      patch_embed_(ctx, "patch_embed", cfg, this),
      blocks_(),
      neck_norm1_(ctx, "neck.1", cfg.layer_norm_eps, this),
      neck_norm2_(ctx, "neck.3", cfg.layer_norm_eps, this) {
    register_module("patch_embed", &patch_embed_);
    register_module("neck.1", &neck_norm1_);
    register_module("neck.3", &neck_norm2_);

    pos_embed_param_ = &register_parameter("pos_embed");
    neck0_weight_param_ = &register_parameter("neck.0.weight");
    neck1_weight_param_ = &get_parameter("sam_model.neck.1.weight");
    neck1_bias_param_ = &get_parameter("sam_model.neck.1.bias");
    neck2_weight_param_ = &register_parameter("neck.2.weight");
    neck3_weight_param_ = &get_parameter("sam_model.neck.3.weight");
    neck3_bias_param_ = &get_parameter("sam_model.neck.3.bias");
    net2_weight_param_ = &register_parameter("net_2.weight");
    net3_weight_param_ = &register_parameter("net_3.weight");

    blocks_.reserve(static_cast<size_t>(cfg.depth));
    for (int32_t i = 0; i < cfg.depth; ++i) {
        bool is_global = std::find(cfg.global_attn_indexes.begin(),
                                   cfg.global_attn_indexes.end(), i) != cfg.global_attn_indexes.end();
        int32_t window = is_global ? 0 : cfg.window_size;
        std::string name = std::string("blocks.") + std::to_string(i);
        blocks_.emplace_back(ctx, name, cfg, window, this);
        register_module(name, &blocks_.back());
    }
}

const Tensor& DeepseekSamVisionModel::pos_embed() const {
    if (!pos_embed_param_) {
        OPENVINO_THROW("DeepseekSamVisionModel pos_embed parameter not registered");
    }
    return pos_embed_param_->value();
}

const Tensor& DeepseekSamVisionModel::neck0_weight() const {
    if (!neck0_weight_param_) {
        OPENVINO_THROW("DeepseekSamVisionModel neck0 weight not registered");
    }
    return neck0_weight_param_->value();
}

const Tensor& DeepseekSamVisionModel::neck2_weight() const {
    if (!neck2_weight_param_) {
        OPENVINO_THROW("DeepseekSamVisionModel neck2 weight not registered");
    }
    return neck2_weight_param_->value();
}

const Tensor& DeepseekSamVisionModel::net2_weight() const {
    if (!net2_weight_param_) {
        OPENVINO_THROW("DeepseekSamVisionModel net2 weight not registered");
    }
    return net2_weight_param_->value();
}

const Tensor& DeepseekSamVisionModel::net3_weight() const {
    if (!net3_weight_param_) {
        OPENVINO_THROW("DeepseekSamVisionModel net3 weight not registered");
    }
    return net3_weight_param_->value();
}

Tensor DeepseekSamVisionModel::forward(const Tensor& pixel_values) const {
    auto x = patch_embed_.forward(pixel_values);
    auto h = Tensor(shape::dim(x, 1), x.context()).squeeze(0);
    auto w = Tensor(shape::dim(x, 2), x.context()).squeeze(0);
    x = x + get_abs_pos_sam(pos_embed(), h, w).to(x.dtype());

    for (const auto& block : blocks_) {
        x = block.forward(x);
    }

    auto x_nchw = x.permute({0, 3, 1, 2});
    x_nchw = x_nchw.to(neck0_weight().dtype());
    auto y = ops::nn::conv2d(x_nchw, neck0_weight(), {1, 1}, {0, 0}, {0, 0});
    y = neck_norm1_.forward(y);
    y = y.to(neck2_weight().dtype());
    y = ops::nn::conv2d(y, neck2_weight(), {1, 1}, {1, 1}, {1, 1});
    y = neck_norm2_.forward(y);

    auto x2 = y.to(net2_weight().dtype());
    x2 = ops::nn::conv2d(x2, net2_weight(), {2, 2}, {1, 1}, {1, 1});
    auto x3 = x2.to(net3_weight().dtype());
    x3 = ops::nn::conv2d(x3, net3_weight(), {2, 2}, {1, 1}, {1, 1});
    return x3;
}

std::shared_ptr<ov::Model> create_deepseek_sam_model(
    const DeepseekOCR2VisionConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    DeepseekSamConfig sam_cfg;
    sam_cfg.image_size = cfg.image_size;
    sam_cfg.embed_dim = cfg.sam_vit_b.width;
    sam_cfg.depth = cfg.sam_vit_b.layers;
    sam_cfg.num_heads = cfg.sam_vit_b.heads;
    sam_cfg.global_attn_indexes = cfg.sam_vit_b.global_attn_indexes;
    sam_cfg.net2_channels = cfg.sam_vit_b.downsample_channels.empty() ? 512 : cfg.sam_vit_b.downsample_channels[0];
    sam_cfg.net3_channels = cfg.qwen2_0_5b.dim;

    BuilderContext ctx;
    DeepseekSamVisionModel model(ctx, sam_cfg);
    model.packed_mapping().rules.push_back({DeepseekOCR2WeightNames::kSamPrefix, "sam_model.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    auto report = ov::genai::modeling::weights::load_model(model, source, finalizer, options);
    (void)report;

    auto pixel_values = ctx.parameter(DeepseekSamIO::kPixelValues,
                                      ov::element::f32,
                                      ov::PartialShape{-1, sam_cfg.in_channels, -1, -1});
    auto output = model.forward(pixel_values);

    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, DeepseekSamIO::kVisionFeats);
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
