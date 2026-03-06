// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <math.h>
#include <iostream>

#include <openvino/openvino.hpp>
#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/op/placeholder_extension.hpp"
#include "openvino/op/fused_mlp.hpp"
#include "openvino/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/op/moe.hpp"
#include <ov_ops/fully_connected.hpp>
#include <ov_ops/rms.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>

#include "gguf_utils/building_blocks.hpp"
#include "gguf_utils/gguf.hpp"  // For ov_extended_types

using namespace ov;
using namespace ov::op::v13;
using namespace ov::op;

static const size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

Output<ov::Node> causal_mask(
    const Output<ov::Node>& attention_mask,
    const Output<ov::Node>& keys,
    const Output<ov::Node>& hidden_dim,
    const Output<ov::Node>& input_shape) {

    // Extract shape of attention mask
    auto t130 = std::make_shared<v3::ShapeOf>(attention_mask, element::i64);
    auto t131 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1);
    auto t132 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t133 = std::make_shared<v8::Gather>(t130, t131, t132);

    // Reshape and construct new shapes
    auto t134 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t135 = std::make_shared<v1::Reshape>(t133, t134, false);
    auto t40 = input_shape;
    auto index_1 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1);
    auto axis_0 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t127 = std::make_shared<v8::Gather>(t40, index_1, axis_0);
    auto t129 = std::make_shared<v1::Reshape>(t127, t134, false);
    auto t136 = std::make_shared<v0::Concat>(OutputVector{t129, t135}, 0);
    auto min_shape_val = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 1});
    auto t137 = std::make_shared<v1::Maximum>(min_shape_val, t136, AutoBroadcastType::NUMPY);
    auto const_neg65504 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, -65504.0f);
    auto t138 = std::make_shared<v3::Broadcast>(const_neg65504, t137, BroadcastType::NUMPY);

    // Create upper triangular mask for causal masking
    auto t139 = std::make_shared<v3::ShapeOf>(t138, element::i32);
    auto t140 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1);
    auto t141 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t142 = std::make_shared<v8::Gather>(t139, t140, t141, 0);
    auto t143 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1);

    // Define ranges for the causal mask
    auto zero_const = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
    auto t144 = std::make_shared<v4::Range>(zero_const, t142, t143, element::i32);
    auto axes_zero = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 0);
    auto t145 = std::make_shared<v0::Unsqueeze>(t144, axes_zero);
    auto t146 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1);
    auto t147 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
    auto t148 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);

    // Broadcast causal mask
    auto t149 = std::make_shared<v8::Gather>(t139, t147, t148);
    auto t150 = std::make_shared<v1::Add>(t149, t146, AutoBroadcastType::NUMPY);
    auto t151 = std::make_shared<v4::Range>(t146, t150, t143, element::i32);
    auto t152 = std::make_shared<v0::Unsqueeze>(t151, t143);
    auto t153 = std::make_shared<v1::GreaterEqual>(t145, t152, AutoBroadcastType::NUMPY);

    // Create a causal mask using a selective operation
    auto t154 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, 0.0f);
    auto t155 = std::make_shared<v1::Select>(t153, t138, t154, AutoBroadcastType::NUMPY);

    // Next branch
    auto t156 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
    auto t157 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1);
    auto t158 = std::make_shared<v4::Range>(t156, t133, t157, element::f32);
    auto t159 = std::make_shared<v0::Convert>(t158, element::i64);
    auto t160 = std::make_shared<v0::Convert>(t159, element::f32);
    auto t161 = std::make_shared<v3::ShapeOf>(keys, element::i64);
    auto t162 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 2);
    auto t163 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t164 = std::make_shared<v8::Gather>(t161, t162, t163, 0);
    auto t165 = std::make_shared<v1::Add>(t164, t127, AutoBroadcastType::NUMPY);
    auto t166 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1);
    auto t167 = std::make_shared<v4::Range>(t164, t165, t166, element::f32);
    auto t168 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{-1, 1});
    auto t169 = std::make_shared<v1::Reshape>(t167, t168, false);
    auto t170 = std::make_shared<v1::Greater>(t160, t169, AutoBroadcastType::NUMPY);
    auto t171 = std::make_shared<v0::Convert>(t170, element::f32);

    auto t172 = std::make_shared<v1::Multiply>(t155, t171, AutoBroadcastType::NUMPY);
    auto t173 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t174 = std::make_shared<v0::Unsqueeze>(t172, t173);
    auto t48 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1);
    auto t175 = std::make_shared<v0::Unsqueeze>(t174, t48);
    auto t41 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t42 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t43 = std::make_shared<v8::Gather>(input_shape, t41, t42);
    auto t176 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t177 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t178 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t179 = std::make_shared<v0::Concat>(OutputVector{t43, t176, t177, t178}, 0);
    auto t180 = std::make_shared<v3::Broadcast>(t175, t179, BroadcastType::BIDIRECTIONAL);
    auto t181 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto t182 = std::make_shared<v1::Reshape>(t180, t181, false);
    auto t183 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t184 = std::make_shared<v3::ShapeOf>(t180, element::i64);
    auto t185 = std::make_shared<v1::ReduceProd>(t184, t183, false);
    auto t186 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1);
    auto t187 = std::make_shared<v4::Range>(t183, t185, t186, element::i64);
    auto t188 = std::make_shared<v1::Reshape>(t187, t184, false);
    auto t189 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t190 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t191 = std::make_shared<ov::opset13::Slice>(t188, t189, t135, t190, hidden_dim);
    auto t192 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{-1, 1});
    auto t193 = std::make_shared<v1::Reshape>(t191, t192, false);
    auto t194 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t195 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t196 = std::make_shared<ov::opset13::Slice>(t180, t194, t135, t195, hidden_dim);

    auto t197 = std::make_shared<v0::Unsqueeze>(attention_mask, t48);
    auto t198 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 2);
    auto t199 = std::make_shared<v0::Unsqueeze>(t197, t198);
    auto t200 = std::make_shared<v0::Convert>(t199, element::f32);
    auto t201 = std::make_shared<v1::Add>(t196, t200, AutoBroadcastType::NUMPY);
    auto t202 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1,1,1,1}, std::vector<float>{0.0f});
    auto t203 = std::make_shared<v1::Equal>(t201, t202, AutoBroadcastType::NUMPY);
    auto t204 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, -65504.0f);
    auto t205 = std::make_shared<v1::Select>(t203, t204, t196, AutoBroadcastType::NUMPY);
    auto t206 = std::make_shared<v3::ShapeOf>(t196, element::i64);
    auto t207 = std::make_shared<v3::Broadcast>(t205, t206, BroadcastModeSpec(BroadcastType::NUMPY));
    auto t208 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto t209 = std::make_shared<v1::Reshape>(t207, t208, false);
    auto t210 = std::make_shared<v15::ScatterNDUpdate>(t182, t193, t209);
    auto t211 = std::make_shared<v1::Reshape>(t210, t184, false);
    auto t212 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t213 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t214 = std::make_shared<v1::Reshape>(t164, t213, false);
    auto t215 = std::make_shared<v1::Add>(t214, t129, AutoBroadcastType::NUMPY);
    auto t216 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t217 = std::make_shared<ov::opset13::Slice>(t211, t212, t215, t216, hidden_dim);

    return t217;
}

// Rotate half the hidden dimensions of the input tensor
Output<ov::Node> rotate_half(const Output<ov::Node>& x, int64_t head_size, const Output<Node>& axis) {
    auto t58 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{head_size / 2});
    auto t59 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{9223372036854775807});
    auto t60 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});

    // Slice second half
    auto t62 = std::make_shared<ov::opset13::Slice>(x, t58, t59, t60, axis);
    
    // Multiply by -1
    auto t63 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1,1,1,1}, std::vector<float>{-1.0f});
    auto t64 = std::make_shared<v1::Multiply>(t62, t63);
    
    // Slice first half
    auto t65 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t66 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{head_size / 2});
    auto t67 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t68 = std::make_shared<ov::opset13::Slice>(x, t65, t66, t67, axis);
    auto rotated = std::make_shared<v0::Concat>(ov::OutputVector{t64, t68}, -1);

    return rotated;
}

// Apply Rotary Position Embedding to query and key tensors
std::tuple<Output<ov::Node>, Output<ov::Node>, std::pair<Output<ov::Node>, Output<ov::Node>>> 
    apply_rotary_pos_emb(
        const Output<ov::Node>& q, 
        const Output<ov::Node>& k,
        const Output<ov::Node>& cos,
        const Output<ov::Node>& sin,
        int64_t head_size,
        const Output<Node>& hidden_dim,
        const std::pair<Output<ov::Node>, Output<ov::Node>>& cos_sin_cached,
        int64_t unsqueeze_dim=1) {
    
    // Handle unsqueeze or cached values
    Output<ov::Node> cos_unsqueezed, sin_unsqueezed;
    
    if (cos_sin_cached.first.get_node() == nullptr) {
        auto unsqueeze_axes1 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, unsqueeze_dim);
        cos_unsqueezed = std::make_shared<v0::Unsqueeze>(cos, unsqueeze_axes1);
        auto unsqueeze_axes2 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, unsqueeze_dim);
        sin_unsqueezed = std::make_shared<v0::Unsqueeze>(sin, unsqueeze_axes2);
    } else {
        cos_unsqueezed = cos_sin_cached.first;
        sin_unsqueezed = cos_sin_cached.second;
    }

    op::internal::RoPE::Config config;
    config.rotary_ndims = head_size;

    // Apply rotation
    // auto q_rot = std::make_shared<v1::Add>(
    //     std::make_shared<v1::Multiply>(q, cos_unsqueezed),
    //     std::make_shared<v1::Multiply>(rotate_half(q, head_size, hidden_dim), sin_unsqueezed)
    // );

    // auto k_rot = std::make_shared<v1::Add>(
    //     std::make_shared<v1::Multiply>(k, cos_unsqueezed),
    //     std::make_shared<v1::Multiply>(rotate_half(k, head_size, hidden_dim), sin_unsqueezed)
    // );

    std::vector<Output<Node>> q_rope_args;
    q_rope_args.push_back(q);
    q_rope_args.push_back(cos_unsqueezed);
    q_rope_args.push_back(sin_unsqueezed);
    auto q_rot = std::make_shared<internal::RoPE>(q_rope_args, config);

    std::vector<Output<Node>> k_rope_args;
    k_rope_args.push_back(k);
    k_rope_args.push_back(cos_unsqueezed);
    k_rope_args.push_back(sin_unsqueezed);
    auto k_rot = std::make_shared<internal::RoPE>(k_rope_args, config);

    return {q_rot, k_rot, {cos_unsqueezed, sin_unsqueezed}};
}

// Generate Rotary Position Embedding components
std::pair<Output<ov::Node>, Output<ov::Node>> rope_emb(
    const Output<ov::Node>& x,
    const Output<ov::Node>& rope_const,
    const Output<ov::Node>& position_ids,
    const Output<ov::Node>& batch_dim) {
    
    // Process position IDs
    auto position_expanded = std::make_shared<v0::Convert>(
        std::make_shared<v0::Unsqueeze>(position_ids, 
            std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1)),
        element::f32
    );

    // Broadcast rope constants
    auto target_shape = std::make_shared<v0::Concat>(OutputVector{
        batch_dim,
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 1),
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 1)
    }, 0);

    auto inv_freq_expanded = std::make_shared<v3::Broadcast>(
        rope_const, target_shape, BroadcastType::BIDIRECTIONAL
    );

    // Compute frequencies
    auto freqs = std::make_shared<v0::MatMul>(
        inv_freq_expanded, position_expanded,
        false, false
    );

    auto freqs_transposed = std::make_shared<v1::Transpose>(
        freqs, 
        std::make_shared<ov::op::v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 2, 1})
    );

    // Concatenate and compute trigonometric values
    auto emb = std::make_shared<ov::opset13::Concat>(
        ov::NodeVector{freqs_transposed, freqs_transposed}, -1
    );

    return {
        std::make_shared<ov::opset13::Cos>(emb),
        std::make_shared<ov::opset13::Sin>(emb)
    };
}


ov::Output<ov::Node> make_rms_norm_qwen3(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& weights,
    float rms_norm_eps) {
    // auto eps_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1,1,1,1}, rms_norm_eps);
    // auto square = std::make_shared<ov::op::v1::Power>(
    //     input, 
    //     std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 2.0f));
    
    // auto variance = std::make_shared<ov::op::v1::ReduceMean>(
    //     square, 
    //     std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, -1),
    //     true);

    // auto add_eps = std::make_shared<ov::op::v1::Add>(variance, eps_node);
    // auto sqrt_node = std::make_shared<ov::op::v0::Sqrt>(add_eps);
    // auto reciprocal = std::make_shared<ov::op::v1::Divide>(
    //     std::make_shared<ov::op::v0::Constant>(
    //         ov::element::f32, ov::Shape{}, 1.0f),
    //     sqrt_node);

    // std::shared_ptr<ov::Node> mul = std::make_shared<ov::op::v1::Multiply>(
    //     reciprocal, input, AutoBroadcastType::NUMPY);

    // auto weight_tensor = weights.at(key + ".weight");
    // // Check if all elements are 1.0
    // bool all_ones = true;
    // if (weight_tensor.get_element_type() == ov::element::f32) {
    //     const float* data = weight_tensor.data<float>();
    //     for (size_t i = 0; i < weight_tensor.get_size(); ++i) {
    //         if (data[i] != 1.0f) {
    //             all_ones = false;
    //             break;
    //         }
    //     }
    // } else if (weight_tensor.get_element_type() == ov::element::f16) {
    //     const uint16_t* data = weight_tensor.data<uint16_t>();
    //     const uint16_t one_in_fp16 = 0x3C00;
    //     for (size_t i = 0; i < weight_tensor.get_size(); ++i) {
    //         if (data[i] != one_in_fp16) {
    //             all_ones = false;
    //             break;
    //         }
    //     }
    // } else {
    //     OPENVINO_THROW("Unsupported weight type ", weight_tensor.get_element_type());
    // }

    // if (!all_ones) {
    //     weight_tensor.set_shape(ov::Shape{1, 1, 1, weight_tensor.get_shape()[0]});
    //     auto weights_const = std::make_shared<ov::op::v0::Constant>(weight_tensor);
    //     auto weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
    //     mul = std::make_shared<ov::op::v1::Multiply>(mul, weights_f32, AutoBroadcastType::NUMPY);
    // }

    auto weight_tensor = weights.at(key + ".weight");
    weight_tensor.set_shape(ov::Shape{1, 1, 1, weight_tensor.get_shape()[0]});
    auto weights_const = std::make_shared<ov::op::v0::Constant>(weight_tensor);
    auto weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);

    auto rms = std::make_shared<ov::op::internal::RMS>(input, weights_f32, (double)rms_norm_eps, ov::element::f32);

    return rms;
}

// Helper function to split heads
// There are q_norm k_norm in Qwen3, if key_name + ".self_attn.q_norm" + ".weight" exists, a rms_norm will be built, if not it will go to else branch.
std::shared_ptr<v1::Transpose> split_heads(const Output<Node>& x,
                                            int num_h,
                                            int  head_dim,
                                            float rms_norm_eps,
                                            const std::string& key,
                                            const std::unordered_map<std::string, ov::Tensor>& weights) {
    auto shape = std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 0, num_h, head_dim});
    auto reshaped = std::make_shared<v1::Reshape>(x, shape, true);
    if (weights.count(key + ".weight")) { //Qwen3 rms_norm
        auto mul = make_rms_norm_qwen3(key, reshaped, weights, rms_norm_eps);
        auto transpose_order = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
        
        return std::make_shared<v1::Transpose>(mul, transpose_order);
    } else { //none-Qwen3 architecture
        auto transpose_order = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
        return std::make_shared<v1::Transpose>(reshaped, transpose_order);
    } 
};

std::tuple<Output<Node>, ov::SinkVector, std::pair<Output<Node>, Output<Node>>, Output<Node>>
multi_head_attention(
    const Output<Node>& query,
    const Output<Node>& key,
    const Output<Node>& value,
    const std::string& key_name,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const std::map<std::string, GGUFMetaData>& configs,
    const Output<Node>& batch_dim,
    int layer_idx,
    const Output<Node>& hidden_dim,
    const Output<Node>& input_shape,
    const Output<Node>& output_shape,
    const Output<Node>& attention_mask,
    const Output<Node>& mask,
    const Output<Node>& position_ids,
    const Output<Node>& rope_const,
    const Output<Node>& beam_idx,
    const std::pair<Output<Node>, Output<Node>>& cos_sin_cached) {
    int num_heads = std::get<int>(configs.at("head_num"));
    int head_dim = std::get<int>(configs.at("head_size"));
    int num_heads_kv = std::get<int>(configs.at("head_num_kv"));
    float rms_norm_eps = std::get<float>(configs.at("rms_norm_eps"));
    
    // 1. Split heads
    // There are q_norm k_norm in Qwen3, if key_name + ".self_attn.q_norm" + ".weight" exists, a rms_norm will be built.
    auto q_split = split_heads(query, num_heads, head_dim, rms_norm_eps, key_name + ".self_attn.q_norm", consts);
    auto k_split = split_heads(key, num_heads_kv, head_dim, rms_norm_eps, key_name  + ".self_attn.k_norm", consts);
    auto v_split = split_heads(value, num_heads_kv, head_dim, rms_norm_eps, key_name + ".self_attn.v_norm", consts);

    // 2. Apply rotary embeddings
    Output<Node> cos, sin;
    if (cos_sin_cached.first.get_node() == nullptr) {
        std::tie(cos, sin) = rope_emb(v_split, rope_const, position_ids, batch_dim);
    }

    auto [q_rot, k_rot, new_cos_sin] = apply_rotary_pos_emb(
        q_split, k_split, cos, sin, head_dim, hidden_dim, cos_sin_cached
    );

    // 3. Handle cache
    auto create_cache = [&](const std::string& name, const Output<Node>& init_value) {
        auto var_info = ov::op::util::VariableInfo{
                ov::PartialShape{-1, num_heads_kv, -1, head_dim},
                ov::element::f32,
                name
            };
        auto var = std::make_shared<ov::op::util::Variable>(var_info);
        auto read_value = std::make_shared<v6::ReadValue>(init_value, var);
        auto gathered = std::make_shared<v8::Gather>(read_value, beam_idx, 
            std::make_shared<v0::Constant>(element::i64, Shape{}, 0), 0);
        return std::make_pair(var, gathered);
    };

    auto zero_const = std::make_shared<v0::Constant>(element::f32, Shape{}, 0.0f);
    auto k_cache_default = std::make_shared<v3::Broadcast>(zero_const, 
        std::make_shared<ov::opset13::Concat>(OutputVector{
            batch_dim,
            std::make_shared<v0::Constant>(element::i64, Shape{1}, num_heads_kv),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim)
        }, 0));

    auto v_cache_default = std::make_shared<v3::Broadcast>(zero_const, 
        std::make_shared<ov::opset13::Concat>(OutputVector{
            batch_dim,
            std::make_shared<v0::Constant>(element::i64, Shape{1}, num_heads_kv),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim)
        }, 0));

    auto k_cache = create_cache(
        "past_key_values." + std::to_string(layer_idx) + ".keypresent." + std::to_string(layer_idx) + ".key",
        k_cache_default
    );
    auto v_cache = create_cache(
        "past_key_values." + std::to_string(layer_idx) + ".valuepresent." + std::to_string(layer_idx) + ".key",
        v_cache_default
    );

    auto k_combined = std::make_shared<ov::opset13::Concat>(OutputVector{k_cache.second, k_rot}, 2);
    auto v_combined = std::make_shared<ov::opset13::Concat>(OutputVector{v_cache.second, v_split}, 2);

    auto k_assign = std::make_shared<ov::opset13::Assign>(k_combined, k_cache.first); //->get_variable_id()
    auto v_assign = std::make_shared<ov::opset13::Assign>(v_combined, v_cache.first);

    // 4. Handle group query attention
    Output<Node> k_reshaped = k_combined;
    Output<Node> v_reshaped = v_combined;
    if (num_heads != num_heads_kv) {
        int kv_per_head = num_heads / num_heads_kv;
        auto unsqueeze_axes1 = std::make_shared<v0::Constant>(element::i64, Shape{}, 2);
        auto k_unsq = std::make_shared<v0::Unsqueeze>(k_combined, unsqueeze_axes1);
        auto unsqueeze_axes2 = std::make_shared<v0::Constant>(element::i64, Shape{}, 2);
        auto v_unsq = std::make_shared<v0::Unsqueeze>(v_combined, unsqueeze_axes2);

        auto broadcast_shape1 = std::make_shared<ov::opset13::Concat>(OutputVector{
            batch_dim,
            std::make_shared<v0::Constant>(element::i64, Shape{1}, num_heads_kv),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, kv_per_head),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim)
        }, 0);

        k_reshaped = std::make_shared<v1::Reshape>(
            std::make_shared<v3::Broadcast>(k_unsq, broadcast_shape1, BroadcastType::BIDIRECTIONAL),
            std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, num_heads, -1, head_dim}),
            true
        );


        auto broadcast_shape2 = std::make_shared<ov::opset13::Concat>(OutputVector{
            batch_dim,
            std::make_shared<v0::Constant>(element::i64, Shape{1}, num_heads_kv),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, kv_per_head),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim)
        }, 0);
        v_reshaped = std::make_shared<v1::Reshape>(
            std::make_shared<v3::Broadcast>(v_unsq, broadcast_shape2, BroadcastType::BIDIRECTIONAL),
            std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, num_heads, -1, head_dim}),
            true
        );
    }

    // 5. Create causal mask if needed
    Output<Node> final_mask = mask;
    if (mask.get_node() == nullptr) {
        final_mask = causal_mask(attention_mask, k_cache.second, hidden_dim, input_shape);
    }

    // 6. Scaled dot product attention
    auto attention = std::make_shared<ScaledDotProductAttention>(
        q_rot, k_reshaped, v_reshaped, final_mask, false);

    // 7. Reshape output
    auto transpose_order = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto context_transposed = std::make_shared<v1::Transpose>(attention, transpose_order);
    auto output = std::make_shared<v1::Reshape>(context_transposed, output_shape, false);

    return {
        output,
        {k_assign, v_assign},
        new_cos_sin,
        final_mask
    };
}

// TODO: can be issues with allocated memory
// TODO: rewrite without doubling a memory
ov::Tensor reorder_interleaved_format(const ov::Tensor& weights, int head_size) {
    ov::Shape input_shape = weights.get_shape();
    if (input_shape.empty() || input_shape[0] % head_size != 0) {
        throw std::invalid_argument("Invalid input dimensions");
    }

    size_t num_heads = input_shape[0] / head_size;
    size_t total_rows = input_shape[0];
    std::vector<size_t> permutation(total_rows);

    // Precompute permutation indices
    for (size_t i = 0; i < total_rows; ++i) {
        size_t head = i / head_size;
        size_t row_in_head = i % head_size;
        size_t new_row_in_head = (row_in_head < head_size/2)
            ? row_in_head * 2
            : (row_in_head - head_size/2) * 2 + 1;
        permutation[i] = head * head_size + new_row_in_head;
    }

    // Create output tensor
    ov::Tensor reordered(weights.get_element_type(), input_shape);

    // Calculate row size in bytes
    size_t row_size = weights.get_byte_size() / total_rows;
    const char* src_data = (const char*)weights.data();
    char* dst_data = (char*)reordered.data();

    // Perform permutation copy
    for (size_t i = 0; i < total_rows; ++i) {
        std::memcpy(dst_data + i * row_size,
                   src_data + permutation[i] * row_size,
                   row_size);
    }

    return reordered;
}

ov::Output<ov::Node> make_fp16_weights(
    const std::string& key,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    bool reorder,
    int head_size,
    bool convert_to_f32=true
) {

    auto it = consts.find(key + ".weight");
    OPENVINO_ASSERT(it != consts.end(), "Weight not found: ", key);
    ov::Tensor weight_f16 = it->second;    

    // Apply reordering
    if (reorder) {
        weight_f16 = reorder_interleaved_format(weight_f16, head_size);
    }

    // Create FP16 constant and convert to FP32
    auto weights_node = std::make_shared<v0::Constant>(weight_f16);
    weights_node->set_friendly_name(key + ".weight");
    if (convert_to_f32) {
        return std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f32);
    } else if (weights_node->get_element_type() != ov::element::f16) {
        return std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    } else {
        return weights_node;
    }
}

// Retrieve tensors
ov::Tensor get_tensor(const std::unordered_map<std::string, ov::Tensor>& consts,
                    const std::string& key) {
    auto it = consts.find(key);
    OPENVINO_ASSERT(it != consts.end(), "Missing tensor: ", key);
    return it->second;
};

ov::Output<ov::Node> make_int8_weights(
    const std::string& key,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    bool reorder,
    int head_size,
    size_t group_size = GGML_QUANTIZATION_GROUP_SIZE) {

    ov::Tensor weight = get_tensor(consts, key + ".weight");
    ov::Tensor scales = get_tensor(consts, key + ".scales");
    ov::Tensor biases = get_tensor(consts, key + ".biases");

    // Reshape weight to (num_heads, -1, group_size)
    ov::Shape orig_shape = weight.get_shape();
    orig_shape[1] *= sizeof(uint32_t) / sizeof(uint8_t);
    size_t num_groups = orig_shape[1] / group_size;

    // Expand dimensions for scales and biases
    auto scale_shape = scales.get_shape();
    scale_shape.push_back(1);
    scales.set_shape(scale_shape);
    biases.set_shape(scale_shape);

    // Apply reordering
    if (reorder) {
        weight = reorder_interleaved_format(weight, head_size);
        scales = reorder_interleaved_format(scales, head_size);
        biases = reorder_interleaved_format(biases, head_size);
    }

    // Create graph nodes
    auto weights_node = std::make_shared<v0::Constant>(ov::element::u8, ov::Shape{orig_shape[0], num_groups, group_size}, static_cast<uint8_t*>(weight.data()), nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);
    ov::Tensor biases_u8(ov::element::u8, scale_shape);

    // Calculate zero point
    const ov::float16* bias_data = biases.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const ov::float16* scale_data = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    uint8_t* bias_u8_data = biases_u8.data<uint8_t>();
    for (size_t i = 0; i < biases_u8.get_size(); ++i) {
        bias_u8_data[i] = (uint8_t)std::round(-1.f * static_cast<float>(bias_data[i]) / static_cast<float>(scale_data[i]));
    }

    auto zero_point = std::make_shared<ov::op::v0::Constant>(biases_u8);

    // Quantization operations
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);

    auto w_zp = std::make_shared<ov::op::v1::Subtract>(
        weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY
    );
    auto w_zp_s = std::make_shared<ov::op::v1::Multiply>(
        w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY
    );

    // Reshape back to original dimensions
    auto final_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape
    );
    auto w_zp_s_r = std::make_shared<ov::op::v1::Reshape>(
        w_zp_s, final_shape, false
    );

    return std::make_shared<ov::op::v0::Convert>(w_zp_s_r, ov::element::f32);
}

ov::Output<ov::Node> make_int4_weights(
    const std::string& key,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    bool reorder,
    int head_size,
    size_t group_size = 32) { // Assuming GGML_QUANTIZATION_GROUP_SIZE = 32

    ov::Tensor weight = get_tensor(consts, key + ".weight");

    // Convert weight to uint8 view and adjust shape
    ov::Shape orig_weight_shape = weight.get_shape();
    bool has_expert_dim = (orig_weight_shape.size() == 3);  // [num_experts, rows, cols_bytes] vs [rows, cols_bytes]
    
    size_t rows_idx = has_expert_dim ? 1 : 0;
    size_t cols_idx = has_expert_dim ? 2 : 1;
    
    // Unpack u32 packed format: each cols value contains 8 x 4-bit values
    orig_weight_shape[cols_idx] *= sizeof(uint32_t) / sizeof(uint8_t) * 2;

    // Retrieve scales and biases
    ov::Tensor scales = get_tensor(consts, key + ".scales");
    ov::Tensor biases = get_tensor(consts, key + ".biases");

    // Expand dimensions for scales and biases (add group_size=1 dimension at the end)
    ov::Shape scale_bias_shape = scales.get_shape();
    scale_bias_shape.push_back(1);
    scales.set_shape(scale_bias_shape);
    biases.set_shape(scale_bias_shape);

    // Apply reordering if needed
    if (reorder) {
        weight = reorder_interleaved_format(weight, head_size);
        scales = reorder_interleaved_format(scales, head_size);
        biases = reorder_interleaved_format(biases, head_size);
    }

    // Create INT4 weight tensor with proper shape
    // 2D: [rows, group_num, group_size]
    // 3D: [num_experts, rows, group_num, group_size]
    ov::Shape packed_shape;
    if (has_expert_dim) {
        packed_shape = {
            orig_weight_shape[0],  // num_experts
            orig_weight_shape[1],  // rows
            orig_weight_shape[2] / group_size,  // group_num
            group_size
        };
    } else {
        packed_shape = {
            orig_weight_shape[0],  // rows
            orig_weight_shape[1] / group_size,  // group_num
            group_size
        };
    }

    auto weights_node = std::make_shared<v0::Constant>(ov::element::u4, packed_shape, static_cast<uint8_t*>(weight.data()), nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holde"] = weight;
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);

    // Pack zero points: two subsequent values into one
    const ov::float16* bias_data = biases.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const ov::float16* scale_data = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    ov::Tensor zero_point_tensor(ov::element::u4, scale_bias_shape);
    uint8_t* zero_point_data = static_cast<uint8_t*>(zero_point_tensor.data());
    for (size_t i = 0; i < zero_point_tensor.get_byte_size(); ++i) {
        uint8_t bias1 = (uint8_t)std::round(-1.f * static_cast<float>(bias_data[i * 2]) / static_cast<float>(scale_data[i * 2]));
        uint8_t bias2 = (uint8_t)std::round(-1.f * static_cast<float>(bias_data[i * 2 + 1]) / static_cast<float>(scale_data[i * 2 + 1]));
        zero_point_data[i] = (bias2 << 4) | (bias1 & 0x0F);
    }

    auto zero_points_node = std::make_shared<ov::op::v0::Constant>(zero_point_tensor);
    auto zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_node, ov::element::f16);

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    // Perform dequantization
    auto w_zp = std::make_shared<ov::op::v1::Subtract>(
        weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);

    auto w_zp_s = std::make_shared<ov::op::v1::Multiply>(
        w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);

    // Reshape back to original shape
    auto final_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{orig_weight_shape.size()}, orig_weight_shape);

    auto w_zp_s_r = std::make_shared<ov::op::v1::Reshape>(
        w_zp_s, final_shape, false);

    return std::make_shared<ov::op::v0::Convert>(w_zp_s_r, ov::element::f32);
}

std::vector<ov::Output<ov::Node>> make_int4_weights_for_moe(
    const std::string& key,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    bool reorder,
    int head_size,
    size_t group_size = 32) { // Assuming GGML_QUANTIZATION_GROUP_SIZE = 32

    ov::Tensor weight = get_tensor(consts, key + ".weight");

    // Convert weight to uint8 view and adjust shape
    ov::element::Type weight_type = weight.get_element_type();
    ov::Shape orig_weight_shape = weight.get_shape();
    bool has_expert_dim = (orig_weight_shape.size() == 3);  // [num_experts, rows, cols_bytes] vs [rows, cols_bytes]
    size_t rows_idx = has_expert_dim ? 1 : 0;
    size_t cols_idx = has_expert_dim ? 2 : 1;
    // packed int4 (u32)
    if (weight_type == ov::element::u32) {
        // Unpack u32 packed format: each cols value contains 8 x 4-bit values
        orig_weight_shape[cols_idx] *= sizeof(uint32_t) / sizeof(uint8_t) * 2;
    } else if (weight_type == ov::element::u8) {
        orig_weight_shape[cols_idx] *= 2;
    }

    // Retrieve scales and biases
    ov::Tensor scales = get_tensor(consts, key + ".scales");
    ov::Tensor biases;
    auto bias_it = consts.find(key + ".biases");
    if (bias_it != consts.end()) {
        biases = bias_it->second;
    } else {
        // Create zero bias tensor with same shape and type as scales
        biases = ov::Tensor(scales.get_element_type(), scales.get_shape());
        std::memset(biases.data(), 0, biases.get_byte_size());
    }

    // Expand dimensions for scales and biases (add group_size=1 dimension at the end)
    ov::Shape scale_bias_shape = scales.get_shape();
    scale_bias_shape.push_back(1);
    scales.set_shape(scale_bias_shape);
    biases.set_shape(scale_bias_shape);

    // Apply reordering if needed
    if (reorder) {
        weight = reorder_interleaved_format(weight, head_size);
        scales = reorder_interleaved_format(scales, head_size);
        biases = reorder_interleaved_format(biases, head_size);
    }

    // Create INT4 weight tensor with proper shape
    // 2D: [rows, group_num, group_size]
    // 3D: [num_experts, rows, group_num, group_size]
    ov::Shape packed_shape;
    if (has_expert_dim) {
        packed_shape = {
            orig_weight_shape[0],  // num_experts
            orig_weight_shape[1],  // rows
            orig_weight_shape[2] / group_size,  // group_num
            group_size
        };
    } else {
        packed_shape = {
            orig_weight_shape[0],  // rows
            orig_weight_shape[1] / group_size,  // group_num
            group_size
        };
    }

    auto weights_node = std::make_shared<v0::Constant>(ov::element::u4, packed_shape, static_cast<uint8_t*>(weight.data()), nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holde"] = weight;

    // Pack zero points: two subsequent values into one
    ov::element::Type bias_type = biases.get_element_type();
    ov::Tensor zero_point_tensor(ov::element::u4, scale_bias_shape);
    uint8_t* zero_point_data = static_cast<uint8_t*>(zero_point_tensor.data());
    if (bias_type == ov::element::u8) {
        const uint8_t* bias_data = biases.data<uint8_t>();
        for (size_t i = 0; i < zero_point_tensor.get_byte_size(); ++i) {
            uint8_t bias1 = bias_data[i * 2];
            uint8_t bias2 = bias_data[i * 2 + 1];
            zero_point_data[i] = (bias2 << 4) | (bias1 & 0x0F);
        }
    } else if (bias_type == ov::element::f16) {
        const ov::float16* bias_data = biases.data<ov::element_type_traits<ov::element::f16>::value_type>();
        const ov::float16* scale_data = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
        for (size_t i = 0; i < zero_point_tensor.get_byte_size(); ++i) {
            uint8_t bias1 = (uint8_t)std::round(-1.f * static_cast<float>(bias_data[i * 2]) / static_cast<float>(scale_data[i * 2]));
            uint8_t bias2 = (uint8_t)std::round(-1.f * static_cast<float>(bias_data[i * 2 + 1]) / static_cast<float>(scale_data[i * 2 + 1]));
            zero_point_data[i] = (bias2 << 4) | (bias1 & 0x0F);
        }
    } else {
        throw std::runtime_error("Unsupported bias type in make_int4_weights_for_moe: only u8 and f16 supported");
    }

    auto zero_points_node = std::make_shared<ov::op::v0::Constant>(zero_point_tensor);

    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);

    // Reshape zero_points and scales from [E, I, H, 1] to [E, I, H]
    ov::Shape reshaped_scale_bias_shape = scales.get_shape();
    reshaped_scale_bias_shape.pop_back(); // Remove the last dimension (1)
    
    auto reshape_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{reshaped_scale_bias_shape.size()}, reshaped_scale_bias_shape);
    
    auto zero_points_reshaped = std::make_shared<ov::op::v1::Reshape>(
        zero_points_node, reshape_shape, false);
    
    auto scales_reshaped = std::make_shared<ov::op::v1::Reshape>(
        scales_node, reshape_shape, false);

    // Transpose from [E, I, H] to [E, H, I]
    auto transpose_order = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{reshaped_scale_bias_shape.size()}, 
        std::vector<int64_t>{0, 2, 1});
    
    auto zero_points_transposed = std::make_shared<ov::op::v1::Transpose>(
        zero_points_reshaped, transpose_order);
    
    auto scales_transposed = std::make_shared<ov::op::v1::Transpose>(
        scales_reshaped, transpose_order);

    return {weights_node, scales_transposed, zero_points_transposed};
}

/**
 * @brief Create weight subgraph for in-flight INT4 quantized weights (sym/asym)
 * 
 * Handles the RTN format produced by C++ quantization:
 * - Weight: [out_features, packed_in_features] where packed_in_features = (in_features + 1) / 2
 *           Each byte contains 2 INT4 values packed as (high nibble, low nibble)
 *           Symmetric stored as signed i4 [-8,7]; asymmetric stored as unsigned u4 [0,15]
 * - Scales: [out_features, num_groups] in FP16
 * - Zero-point: [out_features, num_groups] in U8 (asymmetric only, stored at key + ".biases")
 * 
 * Dequantization:
 *   symmetric   -> weight_fp = int4_val * scale
 *   asymmetric -> weight_fp = (u4_val - zero_point) * scale
 */
ov::Output<ov::Node> make_inflight_int4_weights(
    const std::string& key,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    int qtype_int,
    size_t group_size = 128) {

    bool symmetric = ov_extended_types::is_symmetric_type(qtype_int);

    ov::Tensor weight = get_tensor(consts, key + ".weight");
    ov::Tensor scales = get_tensor(consts, key + ".scales");

    ov::Shape weight_shape = weight.get_shape();  // [out_features, packed_in_features]
    ov::Shape scales_shape = scales.get_shape();  // [out_features, num_groups]
    
    size_t out_features = weight_shape[0];
    size_t packed_in_features = weight_shape[1];
    size_t in_features = packed_in_features * 2;  // 2 int4 values per byte
    size_t num_groups = scales_shape[1];
    
    // Shape: [out_features, num_groups, group_size] for broadcast with scales
    ov::Shape unpacked_shape = {out_features, num_groups, group_size};
    
    // Adjust if in_features doesn't divide evenly by group_size
    size_t actual_group_size = in_features / num_groups;
    if (actual_group_size != group_size) {
        unpacked_shape = {out_features, num_groups, actual_group_size};
    }
    
    // Use i4 for symmetric (-8..7) or u4 for asymmetric (0..15)
    auto elem_type = symmetric ? ov::element::i4 : ov::element::u4;
    auto weights_node = std::make_shared<v0::Constant>(
        elem_type, unpacked_shape, static_cast<uint8_t*>(weight.data()), nullptr);
    weights_node->get_rt_info()["__inflight_tensor_holder"] = weight;
    
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    
    // Reshape scales from [out_features, num_groups] to [out_features, num_groups, 1] for broadcast
    ov::Shape scales_broadcast_shape = {out_features, num_groups, 1};
    scales.set_shape(scales_broadcast_shape);
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);

    std::shared_ptr<ov::Node> dequantized;
    if (symmetric) {
        // Symmetric: weight_fp = signed_weight * scale
        dequantized = std::make_shared<ov::op::v1::Multiply>(
            weights_f16, scales_node, ov::op::AutoBroadcastType::NUMPY);
    } else {
        // Asymmetric: weight_fp = (u4_weight - zero_point) * scale
        auto zp_it = consts.find(key + ".biases");
        if (zp_it == consts.end()) {
            throw std::runtime_error("Zero point tensor not found for asymmetric INT4 weight: " + key);
        }
        ov::Tensor zero_point = zp_it->second;
        zero_point.set_shape(scales_broadcast_shape);
        auto zero_point_node = std::make_shared<ov::op::v0::Constant>(zero_point);
        auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point_node, ov::element::f16);
        auto shifted = std::make_shared<ov::op::v1::Subtract>(
            weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
        dequantized = std::make_shared<ov::op::v1::Multiply>(
            shifted, scales_node, ov::op::AutoBroadcastType::NUMPY);
    }
    
    // Reshape back to [out_features, in_features]
    ov::Shape output_shape = {out_features, in_features};
    auto final_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{output_shape.size()}, output_shape);
    auto w_reshaped = std::make_shared<ov::op::v1::Reshape>(dequantized, final_shape, false);
    
    return std::make_shared<ov::op::v0::Convert>(w_reshaped, ov::element::f32);
}

/**
 * @brief Create weight subgraph for in-flight symmetric INT8 quantized weights
 */
ov::Output<ov::Node> make_inflight_int8_weights_sym(
    const std::string& key,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    size_t group_size = 128) {

    ov::Tensor weight = get_tensor(consts, key + ".weight");
    ov::Tensor scales = get_tensor(consts, key + ".scales");

    ov::Shape weight_shape = weight.get_shape();  // [out_features, in_features]
    ov::Shape scales_shape = scales.get_shape();  // [out_features, num_groups]
    
    size_t out_features = weight_shape[0];
    size_t in_features = weight_shape[1];
    size_t num_groups = scales_shape[1];
    size_t actual_group_size = in_features / num_groups;
    
    // Reshape weights to [out_features, num_groups, group_size]
    ov::Shape grouped_shape = {out_features, num_groups, actual_group_size};
    
    auto weights_node = std::make_shared<v0::Constant>(
        ov::element::i8, grouped_shape, static_cast<int8_t*>(weight.data()), nullptr);
    weights_node->get_rt_info()["__inflight_tensor_holder"] = weight;
    
    // Convert I8 to F16 for arithmetic
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    
    // Reshape scales from [out_features, num_groups] to [out_features, num_groups, 1] for broadcast
    ov::Shape scales_broadcast_shape = {out_features, num_groups, 1};
    scales.set_shape(scales_broadcast_shape);
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    
    // Dequantize: weight_fp = int8_weight * scale
    auto w_scaled = std::make_shared<ov::op::v1::Multiply>(
        weights_f16, scales_node, ov::op::AutoBroadcastType::NUMPY);
    
    // Reshape back to [out_features, in_features]
    ov::Shape output_shape = {out_features, in_features};
    auto final_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{output_shape.size()}, output_shape);
    auto w_reshaped = std::make_shared<ov::op::v1::Reshape>(w_scaled, final_shape, false);
    
    return std::make_shared<ov::op::v0::Convert>(w_reshaped, ov::element::f32);
}

ov::Output<ov::Node> make_weights_subgraph(const std::string& key,
                                           const std::unordered_map<std::string, ov::Tensor>& consts,
                                           gguf_tensor_type qtype,
                                           bool reorder,
                                           int head_size) {
    // Check for in-flight quantization types (using int cast for comparison)
    int qtype_int = static_cast<int>(qtype);
    if (ov_extended_types::is_inflight_type(qtype_int)) {
        // For in-flight types, use dedicated functions that handle our RTN format
        // Note: reorder is not supported for in-flight quantization yet
        if (ov_extended_types::is_int4_type(qtype_int)) {
            return make_inflight_int4_weights(key, consts, qtype_int);
        } else {
            return make_inflight_int8_weights_sym(key, consts);
        }
    }
    
    switch (qtype) {
    case gguf_tensor_type::GGUF_TYPE_F16:
    case gguf_tensor_type::GGUF_TYPE_BF16:
        return make_fp16_weights(key, consts, reorder, head_size);
    case gguf_tensor_type::GGUF_TYPE_Q8_0:
        return make_int8_weights(key, consts, reorder, head_size);
    case gguf_tensor_type::GGUF_TYPE_Q4_0:
        return make_int4_weights(key, consts, reorder, head_size);
    case gguf_tensor_type::GGUF_TYPE_Q4_1:
        return make_int4_weights(key, consts, reorder, head_size);
    case gguf_tensor_type::GGUF_TYPE_Q4_K:
        return make_int4_weights(key, consts, reorder, head_size);
    case gguf_tensor_type::GGUF_TYPE_Q6_K:
        return make_int8_weights(key, consts, reorder, head_size, 16);
    default:
        OPENVINO_THROW("Unsupported quantization type: ", static_cast<int>(qtype));
    }
}

ov::Output<ov::Node> make_fused_fc(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    gguf_tensor_type qtype,
    bool reorder = false,
    int head_size = -1) {
    auto w_f32 = make_weights_subgraph(key, consts, qtype, reorder, head_size);

    ov::Output<ov::Node> bias;

    // Add post-MatMul Add operation if exists
    if (consts.count(key + ".bias")) {
        auto add_tensor = get_tensor(consts, key + ".bias");
        auto add_const = std::make_shared<v0::Constant>(add_tensor);
        bias = std::make_shared<ov::op::v0::Convert>(add_const, ov::element::f32);
    } else {
        bias = std::make_shared<ov::op::internal::PlaceholderExtension>();
    }

    auto output = std::make_shared<ov::op::internal::FullyConnected>(input, w_f32, bias);

    return output;
}

// Fused QKV projection: concatenates Q, K, V weights and does single FC operation
std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>, ov::Output<ov::Node>> 
make_qkv_fused_fc(
    const std::string& layer_prefix,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    bool reorder,
    int head_size,
    int num_heads,
    int num_heads_kv) {
    
    // Get individual weight subgraphs
    auto q_weights = make_weights_subgraph(
        layer_prefix + ".self_attn.q_proj", consts, 
        qtypes.at(layer_prefix + ".self_attn.q_proj.qtype"), reorder, head_size);
    auto k_weights = make_weights_subgraph(
        layer_prefix + ".self_attn.k_proj", consts,
        qtypes.at(layer_prefix + ".self_attn.k_proj.qtype"), reorder, head_size);
    auto v_weights = make_weights_subgraph(
        layer_prefix + ".self_attn.v_proj", consts,
        qtypes.at(layer_prefix + ".self_attn.v_proj.qtype"), false, -1);
    
    // Concatenate weights along output dimension (axis=0 typically)
    auto qkv_weights = std::make_shared<v0::Concat>(
        ov::OutputVector{q_weights, k_weights, v_weights}, 0);
    
    // Handle biases if they exist
    ov::Output<ov::Node> qkv_bias;
    bool has_bias = consts.count(layer_prefix + ".self_attn.q_proj.bias") > 0;

    if (has_bias) {
        auto q_bias_tensor = get_tensor(consts, layer_prefix + ".self_attn.q_proj.bias");
        auto k_bias_tensor = get_tensor(consts, layer_prefix + ".self_attn.k_proj.bias");
        auto v_bias_tensor = get_tensor(consts, layer_prefix + ".self_attn.v_proj.bias");
        
        auto q_bias = std::make_shared<ov::op::v0::Convert>(
            std::make_shared<v0::Constant>(q_bias_tensor), ov::element::f32);
        auto k_bias = std::make_shared<ov::op::v0::Convert>(
            std::make_shared<v0::Constant>(k_bias_tensor), ov::element::f32);
        auto v_bias = std::make_shared<ov::op::v0::Convert>(
            std::make_shared<v0::Constant>(v_bias_tensor), ov::element::f32);

        qkv_bias = std::make_shared<v0::Concat>(
            ov::OutputVector{q_bias, k_bias, v_bias}, 0);
    } else {
        qkv_bias = std::make_shared<ov::op::internal::PlaceholderExtension>();
    }

    // Single fused FC operation
    auto qkv_output = std::make_shared<ov::op::internal::FullyConnected>(
        input, qkv_weights, qkv_bias);
    
    // Split the output back into Q, K, V
    int q_dim = num_heads * head_size;
    int k_dim = num_heads_kv * head_size;
    int v_dim = num_heads_kv * head_size;
    
    // Use VariadicSplit to divide into Q, K, V portions
    auto split_lengths = std::make_shared<v0::Constant>(
        ov::element::i64, ov::Shape{3}, std::vector<int64_t>{q_dim, k_dim, v_dim});
    auto axis = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, -1);
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(qkv_output, axis, split_lengths);
    
    return {split->output(0), split->output(1), split->output(2)};
}

ov::Output<ov::Node> make_fc(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    gguf_tensor_type qtype,
    bool reorder = false,
    int head_size = -1) {
    if (qtype == gguf_tensor_type::GGUF_TYPE_BF16 || qtype == gguf_tensor_type::GGUF_TYPE_F16) {
        return make_fused_fc(key, input, consts, qtype, reorder, head_size);
    }
    auto w_f32 = make_weights_subgraph(key, consts, qtype, reorder, head_size);
    std::shared_ptr<ov::Node> output = std::make_shared<ov::op::v0::MatMul>(input, w_f32, false, true);

    // Add post-MatMul Add operation if exists
    if (consts.count(key + ".bias")) {
        auto add_tensor = get_tensor(consts, key + ".bias");
        auto add_const = std::make_shared<v0::Constant>(add_tensor);
        auto add_convert = std::make_shared<ov::op::v0::Convert>(add_const, ov::element::f32);
        output = std::make_shared<ov::op::v1::Add>(
                                    output, add_convert, ov::op::AutoBroadcastType::NUMPY);
    }
    return output;
}

ov::Output<ov::Node> make_inflight_moe(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::map<std::string, GGUFMetaData>& configs,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    std::string name_prefix,
    std::string name_suffix,
    bool reorder = false,
    int head_size = -1) {
    auto gate_inp_key = key + ".moe.gate_inp";
    auto gate_exps_key = key + ".moe.gate_exps";
    auto up_exps_key = key + ".moe.up_exps";
    auto down_exps_key = key + ".moe.down_exps";

    auto gate_inp_type = qtypes.at(gate_inp_key + ".qtype");
    auto gate_exps_type = qtypes.at(gate_exps_key + ".qtype");
    auto up_exps_type = qtypes.at(up_exps_key + ".qtype");
    auto down_exps_type = qtypes.at(down_exps_key + ".qtype");

    // TODO: support other types
    if (gate_inp_type != gguf_tensor_type::GGUF_TYPE_F32 && gate_inp_type != gguf_tensor_type::GGUF_TYPE_BF16) {
        std::cout << "gate_inp_type != f32 " << gate_inp_type << std::endl;
        exit(0);
    }

    if (gate_exps_type != up_exps_type || gate_exps_type != down_exps_type ||
        !ov_extended_types::is_int4_type(gate_exps_type)) {
        std::cout << "gate/up/down exps type should be q4" << std::endl;
        exit(0);
    }

    auto hidden_f16 = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
    auto gate_inp_w_f32 = make_fp16_weights(gate_inp_key, consts, false, -1, true);
    auto gate_inp_w_f16 = std::make_shared<ov::op::v0::Convert>(gate_inp_w_f32, ov::element::f16);
    auto router_f16 = std::make_shared<ov::op::v0::MatMul>(hidden_f16, gate_inp_w_f16, false, true);

    // Load concatenated expert weights (already in correct layout)    
    auto fused_gate_weights = make_int4_weights_for_moe(gate_exps_key, consts, false, -1, 128);
    auto fused_up_weights = make_int4_weights_for_moe(up_exps_key, consts, false, -1, 128);
    auto fused_down_weights = make_int4_weights_for_moe(down_exps_key, consts, false, -1, 128);

    std::string shared_gate_key = key + ".moe.shared_expert.gate_proj";
    std::string shared_up_key = key + ".moe.shared_expert.up_proj";
    std::string shared_down_key = key + ".moe.shared_expert.down_proj";
    std::string shared_gate_gate_key = key + ".moe.shared_expert_gate";

    int num_shared_expert = 0;
    std::vector<Output<Node>> shared_inputs;
    if (consts.count(shared_gate_key + ".weight")) {
        num_shared_expert = 1;
        auto shared_gate_w = make_int4_weights_for_moe(shared_gate_key, consts, false, -1, 128);
        auto shared_up_w = make_int4_weights_for_moe(shared_up_key, consts, false, -1, 128);
        auto shared_down_w = make_int4_weights_for_moe(shared_down_key, consts, false, -1, 128);

        shared_inputs.push_back(shared_gate_w[0]);
        shared_inputs.push_back(shared_gate_w[1]);
        shared_inputs.push_back(shared_gate_w[2]);
        shared_inputs.push_back(shared_up_w[0]);
        shared_inputs.push_back(shared_up_w[1]);
        shared_inputs.push_back(shared_up_w[2]);
        shared_inputs.push_back(shared_down_w[0]);
        shared_inputs.push_back(shared_down_w[1]);
        shared_inputs.push_back(shared_down_w[2]);

        if (consts.count(shared_gate_gate_key + ".weight")) {
             auto shared_gate_gate_w = make_weights_subgraph(shared_gate_gate_key, consts,
                qtypes.count(shared_gate_gate_key + ".qtype") ? qtypes.at(shared_gate_gate_key + ".qtype") : gguf_tensor_type::GGUF_TYPE_F16,
                false, -1);
             shared_inputs.push_back(shared_gate_gate_w);
        }
    }

    // Create MOE internal op
    ov::OutputVector moe_inputs = {
        hidden_f16,
        router_f16,
        fused_gate_weights[0], 
        fused_gate_weights[1], 
        fused_gate_weights[2], 
        fused_up_weights[0], 
        fused_up_weights[1],
        fused_up_weights[2],
        fused_down_weights[0],
        fused_down_weights[1],
        fused_down_weights[2]
    };
    
    if (num_shared_expert > 0) {
        moe_inputs.insert(moe_inputs.end(), shared_inputs.begin(), shared_inputs.end());
    }

    int cfg_num_expert = 0;
    int cfg_top_k = 0;
    int cfg_inter_size = 0;
    if (configs.count("expert_count")) {
        cfg_num_expert = std::get<int>(configs.at("expert_count"));
    }
    if (configs.count("expert_used_count")) {
        cfg_top_k = std::get<int>(configs.at("expert_used_count"));
    }
    if (configs.count("moe_inter_size")) {
        cfg_inter_size = std::get<int>(configs.at("moe_inter_size"));
    }

    ov::op::internal::MOE3GemmFusedCompressed::Config config;
    config.hidden_size = fused_down_weights[0].get_partial_shape()[1].get_length();
    config.inter_size = fused_gate_weights[0].get_partial_shape()[1].get_length();
    config.num_expert = fused_gate_weights[0].get_partial_shape()[0].get_length();
    config.top_k = cfg_top_k > 0 ? cfg_top_k : 1;
    config.group_size = 128;
    config.out_type = ov::element::f16;
    config.num_shared_expert = num_shared_expert;

    if (config.inter_size != cfg_inter_size) {
        std::cout << "inter size is not matched!" << std::endl;
        exit(0);
    }

    if (config.num_expert != cfg_num_expert) {
        std::cout << "expert num is not matched!" << std::endl;
        exit(0);
    }

    auto moe_node = std::make_shared<ov::op::internal::MOE3GemmFusedCompressed>(moe_inputs, config);
    auto moe_f32 = std::make_shared<ov::op::v0::Convert>(moe_node, ov::element::f32);
    moe_f32->set_friendly_name(name_prefix + ".moe" + name_suffix);

    return moe_f32->output(0);
}


ov::Output<ov::Node> make_moe(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::map<std::string, GGUFMetaData>& configs,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    std::string name_prefix,
    std::string name_suffix,
    bool reorder = false,
    int head_size = -1) {
    auto gate_inp_key = key + ".moe.gate_inp";
    auto gate_exps_key = key + ".moe.gate_exps";
    auto up_exps_key = key + ".moe.up_exps";
    auto down_exps_key = key + ".moe.down_exps";

    auto gate_inp_type = qtypes.at(gate_inp_key + ".qtype");
    auto gate_exps_type = qtypes.at(gate_exps_key + ".qtype");
    auto up_exps_type = qtypes.at(up_exps_key + ".qtype");
    auto down_exps_type = qtypes.at(down_exps_key + ".qtype");

    if (ov_extended_types::is_inflight_type(gate_exps_type)) {
        return make_inflight_moe(key, input, configs, consts, qtypes, name_prefix, name_suffix, reorder, head_size);
    }

    // TODO: support other types
    if (gate_inp_type != gguf_tensor_type::GGUF_TYPE_F32) {
        std::cout << "gate_inp_type != f32 " << gate_inp_type << std::endl;
        exit(0);
    }

    // TODO: support q8
    if (gate_exps_type != up_exps_type || gate_exps_type != down_exps_type || gate_exps_type != gguf_tensor_type::GGUF_TYPE_Q4_1) {
        std::cout << "gate/up/down exps type should be q4_1" << std::endl;
        exit(0);
    }

    auto hidden_f16 = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
    auto gate_inp_w_f32 = std::make_shared<ov::op::v0::Constant>(get_tensor(consts, gate_inp_key + ".weight"));
    auto gate_inp_w_f16 = std::make_shared<ov::op::v0::Convert>(gate_inp_w_f32, ov::element::f16);
    auto router_f16 = std::make_shared<ov::op::v0::MatMul>(hidden_f16, gate_inp_w_f16, false, true);

    auto gate_w = std::make_shared<ov::op::v0::Constant>(get_tensor(consts, gate_exps_key + ".weight"));
    auto gate_s = std::make_shared<ov::op::v0::Constant>(get_tensor(consts, gate_exps_key + ".scales"));
    auto gate_z = std::make_shared<ov::op::v0::Constant>(get_tensor(consts, gate_exps_key + ".zps"));

    auto up_w = std::make_shared<ov::op::v0::Constant>(get_tensor(consts, up_exps_key + ".weight"));
    auto up_s = std::make_shared<ov::op::v0::Constant>(get_tensor(consts, up_exps_key + ".scales"));
    auto up_z = std::make_shared<ov::op::v0::Constant>(get_tensor(consts, up_exps_key + ".zps"));

    auto down_w = std::make_shared<ov::op::v0::Constant>(get_tensor(consts, down_exps_key + ".weight"));
    auto down_s = std::make_shared<ov::op::v0::Constant>(get_tensor(consts, down_exps_key + ".scales"));
    auto down_z = std::make_shared<ov::op::v0::Constant>(get_tensor(consts, down_exps_key + ".zps"));

    const auto gate_shape = get_tensor(consts, gate_exps_key + ".weight").get_shape();
    const auto down_shape = get_tensor(consts, down_exps_key + ".weight").get_shape();
    const auto scales_shape = get_tensor(consts, gate_exps_key + ".scales").get_shape();

    const size_t num_experts = gate_shape[0];
    const size_t inter_size = gate_shape[1];
    const size_t hidden_size = down_shape[1];
    // TODO: channelwise
    const size_t group_size = 128;

    int cfg_num_expert = 0;
    int cfg_top_k = 0;
    int cfg_inter_size = 0;
    if (configs.count("expert_count")) {
        cfg_num_expert = std::get<int>(configs.at("expert_count"));
    }
    if (configs.count("expert_used_count")) {
        cfg_top_k = std::get<int>(configs.at("expert_used_count"));
    }
    if (configs.count("moe_inter_size")) {
        cfg_inter_size = std::get<int>(configs.at("moe_inter_size"));
    }

    std::string shared_gate_key = key + ".moe.shared_expert.gate_proj";
    std::string shared_up_key = key + ".moe.shared_expert.up_proj";
    std::string shared_down_key = key + ".moe.shared_expert.down_proj";
    std::string shared_gate_gate_key = key + ".moe.shared_expert_gate";

    int num_shared_expert = 0;
    std::vector<Output<Node>> shared_inputs;
    if (consts.count(shared_gate_key + ".weight")) {
        num_shared_expert = 1;
        auto shared_gate_w = make_int4_weights_for_moe(shared_gate_key, consts, false, -1, 128);
        auto shared_up_w = make_int4_weights_for_moe(shared_up_key, consts, false, -1, 128);
        auto shared_down_w = make_int4_weights_for_moe(shared_down_key, consts, false, -1, 128);

        shared_inputs.push_back(shared_gate_w[0]);
        shared_inputs.push_back(shared_gate_w[1]);
        shared_inputs.push_back(shared_gate_w[2]);
        shared_inputs.push_back(shared_up_w[0]);
        shared_inputs.push_back(shared_up_w[1]);
        shared_inputs.push_back(shared_up_w[2]);
        shared_inputs.push_back(shared_down_w[0]);
        shared_inputs.push_back(shared_down_w[1]);
        shared_inputs.push_back(shared_down_w[2]);

        if (consts.count(shared_gate_gate_key + ".weight")) {
             auto shared_gate_gate_w = make_weights_subgraph(shared_gate_gate_key, consts,
                qtypes.count(shared_gate_gate_key + ".qtype") ? qtypes.at(shared_gate_gate_key + ".qtype") : gguf_tensor_type::GGUF_TYPE_F16,
                false, -1);
             shared_inputs.push_back(shared_gate_gate_w);
        }
    }

    ov::op::internal::MOE3GemmFusedCompressed::Config config;
    config.hidden_size = static_cast<int>(hidden_size);
    config.inter_size = cfg_inter_size > 0 ? cfg_inter_size : static_cast<int>(inter_size);
    config.num_expert = cfg_num_expert > 0 ? cfg_num_expert : static_cast<int>(num_experts);
    config.top_k = cfg_top_k > 0 ? cfg_top_k : 1;
    config.group_size = static_cast<int>(group_size);
    config.out_type = ov::element::f16;
    // TODO: design issue
    bool is_pa = true;
    config.has_batch_dim = is_pa ? 0 : 1;
    config.num_shared_expert = num_shared_expert;

    ov::OutputVector args = {
        hidden_f16,
        router_f16,
        gate_w, gate_s, gate_z,
        up_w, up_s, up_z,
        down_w, down_s, down_z};

    if (num_shared_expert > 0) {
        args.insert(args.end(), shared_inputs.begin(), shared_inputs.end());
    }

    auto moe = std::make_shared<ov::op::internal::MOE3GemmFusedCompressed>(args, config);
    auto moe_f32 = std::make_shared<ov::op::v0::Convert>(moe, ov::element::f32);
    moe_f32->set_friendly_name(name_prefix + ".moe" + name_suffix);
    return moe_f32;
}

ov::Output<ov::Node> make_mlp(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    std::string name_prefix,
    std::string name_suffix,
    bool reorder = false,
    int head_size = -1) {
    auto gate_key = key + ".mlp.gate_proj";
    auto up_key = key + ".mlp.up_proj";
    auto down_key = key + ".mlp.down_proj";
    auto gate_type = qtypes.at(gate_key + ".qtype");
    auto up_type = qtypes.at(up_key + ".qtype");
    auto down_type = qtypes.at(down_key + ".qtype");

    bool has_bias = consts.count(gate_key + ".bias") || consts.count(up_key + ".bias") || consts.count(down_key + ".bias");

    if (gate_type == up_type && up_type == down_type
        && (gate_type == gguf_tensor_type::GGUF_TYPE_BF16
            || gate_type == gguf_tensor_type::GGUF_TYPE_F16) && !has_bias) {
        // fp16 weights, use fused mlp
        auto gate_w_f16 = make_fp16_weights(gate_key, consts, reorder, head_size, false);
        auto up_w_f16 = make_fp16_weights(up_key, consts, reorder, head_size, false);
        auto down_w_f16 = make_fp16_weights(down_key, consts, reorder, head_size, false);

        auto trans_const = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{1, 0});
        auto gate_w_f16_trans = std::make_shared<ov::op::v1::Transpose>(gate_w_f16, trans_const);
        auto up_w_f16_trans = std::make_shared<ov::op::v1::Transpose>(up_w_f16, trans_const);
        auto down_w_f16_trans = std::make_shared<ov::op::v1::Transpose>(down_w_f16, trans_const);

        auto axes = ov::op::v0::Constant::create(ov::element::i64, Shape{1}, {3});
        auto input_f16 = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
        auto input_f16_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_f16, axes);

        // auto mlp_out = std::make_shared<ov::intel_gpu::op::FusedMLP>(input_f16, gate_w_f16, up_w_f16, down_w_f16);
        auto mlp_out = std::make_shared<ov::op::internal::FusedMLP>(input_f16_unsqueeze, gate_w_f16_trans, up_w_f16_trans, down_w_f16_trans);
        auto mlp_out_f32 = std::make_shared<ov::op::v0::Convert>(mlp_out, ov::element::f32);
        auto mlp_out_f32_squeeze = std::make_shared<ov::op::v0::Squeeze>(mlp_out_f32, axes);

        return mlp_out_f32_squeeze;
    } else {
        auto gate_proj = make_fc(
            gate_key,
            input,
            consts,
            gate_type,
            reorder,
            head_size
        );
        auto silu = std::make_shared<ov::op::v4::Swish>(gate_proj);
        auto up_proj = make_fc(
            up_key,
            input,
            consts,
            up_type,
            reorder,
            head_size);
        auto mul = std::make_shared<ov::op::v1::Multiply>(
            silu, up_proj, ov::op::AutoBroadcastType::NUMPY);
        mul->set_friendly_name(name_prefix + ".mlp.mul" + name_suffix);
        auto down_proj = make_fc(
            down_key,
            mul,
            consts,
            down_type,
            reorder,
            head_size);
        return down_proj;
    }
}

ov::Output<ov::Node> make_lm_head(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const ov::Output<ov::Node>& embeddings_node,
    gguf_tensor_type qtype) {

    ov::Output<ov::Node> w_f32;
    if (consts.count(key + ".weight")) {
        gguf_tensor_type lm_qtype = qtype;
        if (!consts.count(key + ".scales")) {
            lm_qtype = gguf_tensor_type::GGUF_TYPE_F16;
        }
        w_f32 = make_weights_subgraph(key, consts, lm_qtype, false, -1);
    } else {
        w_f32 = embeddings_node;
    }
    // return std::make_shared<ov::op::v0::MatMul>(
    //     input, w_f32, false, true);

    auto no_bias = std::make_shared<ov::op::internal::PlaceholderExtension>();
    // auto matmul = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, w_f32, no_bias);
    auto matmul = std::make_shared<ov::op::internal::FullyConnected>(input, w_f32, no_bias);
    return matmul;
}

ov::Output<ov::Node> make_rms_norm(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    float epsilon) {

    // auto eps_node = std::make_shared<ov::op::v0::Constant>(
    //     ov::element::f32, ov::Shape{1,1,1}, epsilon);
    // auto square = std::make_shared<ov::op::v1::Power>(
    //     input, 
    //     std::make_shared<ov::op::v0::Constant>(
    //         ov::element::f32, ov::Shape{1,1,1}, 2.0f));
    
    // auto variance = std::make_shared<ov::op::v1::ReduceMean>(
    //     square, 
    //     std::make_shared<ov::op::v0::Constant>(
    //         ov::element::i32, ov::Shape{1}, -1),
    //     true);

    // auto add_eps = std::make_shared<ov::op::v1::Add>(variance, eps_node);
    // auto sqrt_node = std::make_shared<ov::op::v0::Sqrt>(add_eps);
    // auto reciprocal = std::make_shared<ov::op::v1::Divide>(
    //     std::make_shared<ov::op::v0::Constant>(
    //         ov::element::f32, ov::Shape{1,1,1}, 1.0f),
    //     sqrt_node);

    // std::shared_ptr<ov::Node> mul = std::make_shared<ov::op::v1::Multiply>(
    //     reciprocal, input, AutoBroadcastType::NUMPY);

    // if (consts.count(key + ".weight")) {
    //     auto weight_tensor = consts.at(key + ".weight");
    //     // Check if all elements are 1.0
    //     bool all_ones = true;
    //     if (weight_tensor.get_element_type() == ov::element::f32) {
    //         const float* data = weight_tensor.data<float>();
    //         for (size_t i = 0; i < weight_tensor.get_size(); ++i) {
    //             if (data[i] != 1.0f) {
    //                 all_ones = false;
    //                 break;
    //             }
    //         }
    //     } else if (weight_tensor.get_element_type() == ov::element::f16) {
    //         const uint16_t* data = weight_tensor.data<uint16_t>();
    //         const uint16_t one_in_fp16 = 0x3C00;
    //         for (size_t i = 0; i < weight_tensor.get_size(); ++i) {
    //             if (data[i] != one_in_fp16) {
    //                 all_ones = false;
    //                 break;
    //             }
    //         }
    //     } else {
    //         OPENVINO_THROW("Unsupported weight type ", weight_tensor.get_element_type());
    //     }

    //     if (!all_ones) {
    //         weight_tensor.set_shape(ov::Shape{1, 1, weight_tensor.get_shape()[0]});
    //         auto weights_const = std::make_shared<ov::op::v0::Constant>(
    //             weight_tensor);
    //         auto weights_f32 = std::make_shared<ov::op::v0::Convert>(
    //             weights_const, ov::element::f32);
    //         mul = std::make_shared<ov::op::v1::Multiply>(
    //             mul, weights_f32, AutoBroadcastType::NUMPY);
    //     }
    // }

    // return mul;

    if (consts.count(key + ".weight")) {
        // std::cout << "make_rms_norm, use internal rms" << std::endl;
        auto weight_tensor = consts.at(key + ".weight");
        weight_tensor.set_shape(ov::Shape{1, 1, weight_tensor.get_shape()[0]});
        auto weights_const = std::make_shared<ov::op::v0::Constant>(
            weight_tensor);
        auto weights_f32 = std::make_shared<ov::op::v0::Convert>(
            weights_const, ov::element::f32);

        auto rms = std::make_shared<ov::op::internal::RMS>(input, weights_f32, (double)epsilon, ov::element::f32);

        return rms;
    } else {
        std::cout << "make_rms_norm, no weight" << std::endl;
        auto eps_node = std::make_shared<ov::op::v0::Constant>(
            ov::element::f32, ov::Shape{1,1,1}, epsilon);
        auto square = std::make_shared<ov::op::v1::Power>(
            input, 
            std::make_shared<ov::op::v0::Constant>(
                ov::element::f32, ov::Shape{1,1,1}, 2.0f));
        
        auto variance = std::make_shared<ov::op::v1::ReduceMean>(
            square, 
            std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{1}, -1),
            true);

        auto add_eps = std::make_shared<ov::op::v1::Add>(variance, eps_node);
        auto sqrt_node = std::make_shared<ov::op::v0::Sqrt>(add_eps);
        auto reciprocal = std::make_shared<ov::op::v1::Divide>(
            std::make_shared<ov::op::v0::Constant>(
                ov::element::f32, ov::Shape{1,1,1}, 1.0f),
            sqrt_node);

        std::shared_ptr<ov::Node> mul = std::make_shared<ov::op::v1::Multiply>(
            reciprocal, input, AutoBroadcastType::NUMPY);

        return mul;
    }
}

std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>> make_embedding(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    gguf_tensor_type qtype) {
        
    auto embedding_type = qtype;
    // Detmbedding_type = qtype;
    if (consts.count(key + ".scales") == 0) {
        embedding_type = gguf_tensor_type::GGUF_TYPE_F16;
    }

    // Create embedding weights
    auto embed_f32 = make_weights_subgraph(key, consts, embedding_type, false, -1);

    // Convert input to int32 indices
    auto input_int32 = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);

    // Gather embeddings
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto embeddings = std::make_shared<ov::op::v8::Gather>(embed_f32, input_int32, axis);

    return {embeddings, embed_f32};
}

std::tuple<ov::Output<ov::Node>, 
           ov::SinkVector,
           ov::Output<ov::Node>,
           std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>,
           std::shared_ptr<ov::Node>> 
    layer(const std::map<std::string, GGUFMetaData>& configs,
        std::unordered_map<std::string, ov::Tensor>& consts,
        std::unordered_map<std::string, gguf_tensor_type>& qtypes,
        int layer_idx,
        const ov::Output<ov::Node>& hidden_states,
        const ov::Output<ov::Node>& attn_mask,
        const ov::Output<ov::Node>& causal_mask,
        const ov::Output<ov::Node>& position_ids,
        const ov::Output<ov::Node>& rope_const,
        const ov::Output<ov::Node>& beam_idx,
        const ov::Output<ov::Node>& batch_dim,
        const ov::Output<ov::Node>& hidden_dim,
        const std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>& cos_sin_cached,
        const std::shared_ptr<ov::Node>& output_shape) {

    std::string name_suffix = ".layer" + std::to_string(layer_idx);
    std::string name_prefix = "model.layers.self_attn";
    std::string layer_prefix = format("model.layers[%d]", layer_idx);

    // LayerNorm
    auto input_layernorm = make_rms_norm(layer_prefix + ".input_layernorm",
                                         hidden_states,
                                         consts,
                                         std::get<float>(configs.at("rms_norm_eps")));

    // Attention projections
    // check if it's llama structure, if so, reorder= true
    bool reorder = false;
    if (std::get<std::string>(configs.at("architecture")).find("llama") != std::string::npos) {
        reorder = true;
    }
    
    int num_heads = std::get<int>(configs.at("head_num"));
    int head_size = std::get<int>(configs.at("head_size"));
    int num_heads_kv = std::get<int>(configs.at("head_num_kv"));
    
    // Option 1: Use fused QKV FC (recommended for performance)
    // auto [q, k, v] = make_qkv_fused_fc(
    //     layer_prefix,
    //     input_layernorm,
    //     consts,
    //     qtypes,
    //     reorder,
    //     head_size,
    //     num_heads,
    //     num_heads_kv);
    
    // Option 2: Use separate FCs (original approach)
    auto q = make_fc(
        layer_prefix + ".self_attn.q_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".self_attn.q_proj.qtype"),
        reorder,
        std::get<int>(configs.at("head_size")));
    
    auto k = make_fc(
        layer_prefix + ".self_attn.k_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".self_attn.k_proj.qtype"),
        reorder,
        std::get<int>(configs.at("head_size")));

    auto v = make_fc(
        layer_prefix + ".self_attn.v_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".self_attn.v_proj.qtype"));

    // Handle output shape
    std::shared_ptr<ov::Node> final_output_shape = output_shape;
    if (!output_shape) {
        auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input_layernorm);
        auto indices = std::make_shared<ov::op::v0::Constant>(
            ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 1});
        auto gathered = std::make_shared<ov::op::v8::Gather>(
            input_shape, indices, 
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0));
        auto minus_one = std::make_shared<ov::op::v0::Constant>(
            ov::element::i64, ov::Shape{1}, -1);
        final_output_shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{gathered, minus_one}, 0);
    }

    // Multi-head attention
    auto [attn_output, sinks, new_cos_sin, new_causal_mask] = multi_head_attention(
        q, k, v,
        layer_prefix,
        consts,
        configs,
        batch_dim,
        layer_idx,
        hidden_dim,
        std::make_shared<ov::op::v3::ShapeOf>(input_layernorm),
        final_output_shape,
        attn_mask,
        causal_mask,
        position_ids,
        rope_const,
        beam_idx,
        cos_sin_cached);

    // Output projection
    auto o_proj = make_fc(
        layer_prefix + ".self_attn.o_proj",
        attn_output,
        consts,
        qtypes.at(layer_prefix + ".self_attn.o_proj.qtype"));

    // Residual connection
    auto attn_add = std::make_shared<ov::op::v1::Add>(
        hidden_states, o_proj, ov::op::AutoBroadcastType::NUMPY);
    attn_add->set_friendly_name(name_prefix + ".add0" + name_suffix);

    // Post-attention Layernorm
    auto post_attn_norm = make_rms_norm(
        layer_prefix + ".post_attention_layernorm",
        attn_add,
        consts,
        std::get<float>(configs.at("rms_norm_eps")));

    ov::Output<ov::Node> output;

    if (std::get<std::string>(configs.at("architecture")) == "qwen3moe" ||
        std::get<std::string>(configs.at("architecture")) == "qwen3_moe") {
        // MoE block
        auto moe_out = make_moe(
            layer_prefix, post_attn_norm, configs, consts, qtypes, name_prefix, name_suffix, reorder
        );
        output = std::make_shared<ov::op::v1::Add>(
            attn_add, moe_out, ov::op::AutoBroadcastType::NUMPY);
    } else {
        // MLP block
        auto mlp_out = make_mlp(
            layer_prefix, post_attn_norm, consts, qtypes, name_prefix, name_suffix, reorder
        );

        // Final residual connection
        output = std::make_shared<ov::op::v1::Add>(
            attn_add, mlp_out, ov::op::AutoBroadcastType::NUMPY);
        // output.set_friendly_name(name_prefix + ".add1" + name_suffix);
    }

    // // Final residual connection
    // auto output = std::make_shared<ov::op::v1::Add>(
    //     attn_add, mlp_out, ov::op::AutoBroadcastType::NUMPY);
    // output->set_friendly_name(name_prefix + ".add1" + name_suffix);

    return {output, sinks, new_causal_mask, new_cos_sin, final_output_shape};
}

// Current implementation creates MOE node, expecting ConvertMOEToMOECompressed to run first,
// but that transformation may not match if weights don't have the expected decompression pattern.
//
ov::Output<ov::Node> moe_layer_internal(
    const std::string& layer_prefix,
    const ov::Output<ov::Node>& hidden_states,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    int num_experts,
    int topk) {
    
    std::cout << "\n=== MOE Layer Construction Debug ===" << std::endl;
    std::cout << "hidden_states shape: " << hidden_states.get_partial_shape() << std::endl;
    std::cout << "num_experts: " << num_experts << ", topk: " << topk << std::endl;
    
    // Flatten batch*seq_len dimensions: [batch, seq, hidden] -> [batch*seq, hidden]
    auto flatten_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_states.get_partial_shape()[2].get_length())});
    auto hidden_states_flat = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
    std::cout << "hidden_states_flat shape: " << hidden_states_flat->get_output_partial_shape(0) << std::endl;

    // 1. Router MatMul - must be preserved for FuseMOE3GemmCompressed pattern
    auto router_key = layer_prefix + ".router";
    auto router_weights = make_weights_subgraph(router_key, consts,
        qtypes.count(router_key + ".qtype") ? qtypes.at(router_key + ".qtype") : gguf_tensor_type::GGUF_TYPE_F16,
        false, -1);
    
    std::cout << "router_weights shape: " << router_weights.get_partial_shape() << std::endl;
    
    auto router_matmul = std::make_shared<ov::op::v0::MatMul>(
        hidden_states_flat, router_weights, false, true);
    std::cout << "router_matmul shape: " << router_matmul->get_output_partial_shape(0) << std::endl;

    // 5. Load concatenated expert weights (already in correct layout)
    std::string gate_concat_key = layer_prefix + ".experts.gate_proj_fused";
    std::string up_concat_key = layer_prefix + ".experts.up_proj_fused";
    std::string down_concat_key = layer_prefix + ".experts.down_proj_fused";

    auto fused_gate_weights = make_int4_weights_for_moe(gate_concat_key, consts,
        false, -1);
    auto fused_up_weights = make_int4_weights_for_moe(up_concat_key, consts,
        false, -1);
    auto fused_down_weights = make_int4_weights_for_moe(down_concat_key, consts,
        false, -1);
    
    std::cout << "fused_gate_weights shape: " << fused_gate_weights[0].get_partial_shape() << std::endl;
    std::cout << "fused_up_weights shape: " << fused_up_weights[0].get_partial_shape() << std::endl;
    std::cout << "fused_down_weights shape: " << fused_down_weights[0].get_partial_shape() << std::endl;
    std::cout << "fused_gate_scaling shape: " << fused_gate_weights[1].get_partial_shape() << std::endl;
    std::cout << "fused_up_scaling shape: " << fused_up_weights[1].get_partial_shape() << std::endl;
    std::cout << "fused_down_scaling shape: " << fused_down_weights[1].get_partial_shape() << std::endl;

    // Load shared expert weights
    std::string shared_gate_key = layer_prefix + ".shared_expert.gate_proj";
    std::string shared_up_key = layer_prefix + ".shared_expert.up_proj";
    std::string shared_down_key = layer_prefix + ".shared_expert.down_proj";
    std::string shared_gate_gate_key = layer_prefix + ".shared_expert_gate";
    int num_shared_expert = 0;
    std::vector<Output<Node>> shared_inputs;
    if (consts.count(shared_gate_key + ".weight")) {
        num_shared_expert = 1;
        auto shared_gate_w = make_int4_weights_for_moe(shared_gate_key, consts, false, -1);
        auto shared_up_w = make_int4_weights_for_moe(shared_up_key, consts, false, -1);
        auto shared_down_w = make_int4_weights_for_moe(shared_down_key, consts, false, -1);

        shared_inputs.push_back(shared_gate_w[0]);
        shared_inputs.push_back(shared_gate_w[1]);
        shared_inputs.push_back(shared_gate_w[2]);
        shared_inputs.push_back(shared_up_w[0]);
        shared_inputs.push_back(shared_up_w[1]);
        shared_inputs.push_back(shared_up_w[2]);
        shared_inputs.push_back(shared_down_w[0]);
        shared_inputs.push_back(shared_down_w[1]);
        shared_inputs.push_back(shared_down_w[2]);

        if (consts.count(shared_gate_gate_key + ".weight")) {
             auto shared_gate_gate_w = make_weights_subgraph(shared_gate_gate_key, consts,
                qtypes.count(shared_gate_gate_key + ".qtype") ? qtypes.at(shared_gate_gate_key + ".qtype") : gguf_tensor_type::GGUF_TYPE_F16,
                false, -1);
             shared_inputs.push_back(shared_gate_gate_w);
        }
    }

    // 8. Create MOE internal op
    ov::OutputVector moe_inputs = {
        hidden_states_flat,
        router_matmul,
        fused_gate_weights[0], 
        fused_gate_weights[1], 
        fused_gate_weights[2], 
        fused_up_weights[0], 
        fused_up_weights[1],
        fused_up_weights[2],
        fused_down_weights[0],
        fused_down_weights[1],
        fused_down_weights[2]
    };
    
    if (num_shared_expert > 0) {
        moe_inputs.insert(moe_inputs.end(), shared_inputs.begin(), shared_inputs.end());
    }

    ov::op::internal::MOE3GemmFusedCompressed::Config config;
    config.hidden_size = hidden_states_flat->get_output_partial_shape(0)[1].get_length();
    config.inter_size = fused_gate_weights[0].get_partial_shape()[1].get_length();
    config.num_expert = num_experts;
    config.top_k = topk;
    config.group_size = 32;
    config.out_type = ov::element::f16;
    // We assume shared experts exist if we are adding them to inputs (which we did unconditionally above,
    // but in reality we should probably check if keys exist or if user wants shared experts)
    // For now, assuming this function is only called for architectures with shared experts (e.g. Qwen2-MoE / Qwen3-MoE)
    config.num_shared_expert = num_shared_expert;

    auto moe_node = std::make_shared<ov::op::internal::MOE3GemmFusedCompressed>(moe_inputs, config);
    std::cout << "moe_node output shape: " << moe_node->get_output_partial_shape(0) << std::endl;

    return moe_node->output(0);
}

// Fixed moe_layer implementation that matches GPU fusion pattern
// Key changes:
// 1. Add Reshape after Tile
// 2. Use transpose_b=true for MatMuls
// 3. Transpose weight layout to [num_experts, ff_dim, hidden_dim]
// 4. Fix routing path operation order

ov::Output<ov::Node> moe_layer_fused(
    const std::string& layer_prefix,
    const ov::Output<ov::Node>& hidden_states,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    int num_experts,
    int topk) {

    // Flatten batch*seq_len dimensions: [batch, seq, hidden] -> [batch*seq, hidden]
    auto flatten_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_states.get_partial_shape()[2].get_length())});
    auto hidden_states_flat = std::make_shared<ov::op::v1::Reshape>(hidden_states, flatten_shape, false);
    std::cout << "hidden_states_flat shape: " << hidden_states_flat->get_output_partial_shape(0) << std::endl;
    // 1. Router network
    auto router_key = layer_prefix + ".router";
    auto router_weights = make_weights_subgraph(router_key, consts, 
        qtypes.count(router_key + ".qtype") ? qtypes.at(router_key + ".qtype") : gguf_tensor_type::GGUF_TYPE_F16,
        false, -1);
    
    auto router_logits = std::make_shared<ov::op::v0::MatMul>(
        hidden_states_flat, router_weights, false, true);
    
    auto softmax = std::make_shared<ov::op::v8::Softmax>(router_logits, -1);
    
    // 2. TopK expert selection
    auto topk_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{}, topk);
    auto topk_op = std::make_shared<ov::op::v11::TopK>(
        softmax, topk_const, -1,
        ov::op::v11::TopK::Mode::MAX,
        ov::op::v11::TopK::SortType::SORT_VALUES,
        ov::element::i64);
    
    auto topk_values = topk_op->output(0);
    auto topk_indices = topk_op->output(1);
    
    // 3. Normalize routing weights
    auto reduce_axis = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto sum_reduce = std::make_shared<ov::op::v1::ReduceSum>(
        topk_values, reduce_axis, true);
    auto normalized_weights = std::make_shared<ov::op::v1::Divide>(
        topk_values, sum_reduce);
    
    // 4. Create sparse routing tensor [batch*seq_len, num_experts]
    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(hidden_states_flat, ov::element::i64);
    auto axis_0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
    auto axis_1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 1);
    
    auto seq_dim = std::make_shared<ov::op::v8::Gather>(input_shape, axis_0, axis_0);
    auto hidden_dim_from_shape = std::make_shared<ov::op::v8::Gather>(input_shape, axis_1, axis_0);

    auto seq_dim_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(
        seq_dim, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 0));
    auto hidden_dim_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(
        hidden_dim_from_shape, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 0));

    auto num_experts_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, num_experts);
    auto scatter_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{seq_dim_unsqueezed, num_experts_const}, 0);
    
    auto zeros_scalar = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{}, 0.0f);
    auto zeros_tensor = std::make_shared<ov::op::v3::Broadcast>(
        zeros_scalar, scatter_shape);
    
    auto scatter_axis = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 1);
    auto routing_sparse = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
        zeros_tensor, topk_indices, normalized_weights, scatter_axis);
    
    // 5. Load concatenated expert weights (already in correct layout)
    // Weights are in layout: gate/up [E, FF, H] (ff_dim, hidden_dim), down [E, FF, H] (ff_dim, hidden_dim)
    // For transpose_b=true MatMuls on gate/up, they become [E, H, FF] for multiplication
    // For down projection, weights are already [E, FF, H] so no transpose needed
    std::string gate_concat_key = layer_prefix + ".experts.gate_proj_fused";
    std::string up_concat_key = layer_prefix + ".experts.up_proj_fused";
    std::string down_concat_key = layer_prefix + ".experts.down_proj_fused";
    
    auto fused_gate_weights = make_weights_subgraph(gate_concat_key, consts,
        qtypes.count(gate_concat_key + ".qtype") ? qtypes.at(gate_concat_key + ".qtype") : gguf_tensor_type::GGUF_TYPE_F16,
        false, -1);
    auto fused_up_weights = make_weights_subgraph(up_concat_key, consts,
        qtypes.count(up_concat_key + ".qtype") ? qtypes.at(up_concat_key + ".qtype") : gguf_tensor_type::GGUF_TYPE_F16,
        false, -1);
    auto fused_down_weights = make_weights_subgraph(down_concat_key, consts,
        qtypes.count(down_concat_key + ".qtype") ? qtypes.at(down_concat_key + ".qtype") : gguf_tensor_type::GGUF_TYPE_F16,
        false, -1);
    
    // Weights are already in layout [E, FF, H] for all projections
    // Since we use transpose_b=true in MatMuls for gate/up:
    // - gate/up: [E, FF, H] transposed in MatMul -> [E, H, FF] for multiplication
    // For down projection, no transpose needed:
    // - down: [E, FF, H] used as-is for multiplication
    
    auto neg_one = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, -1);
    
    // ========== DENSE MOE: Process all experts (for pattern matching) ==========
    // Reshape from [B*S, H] to [1, B*S, H]
    auto one_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 1);
    auto reshape_to_3d_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{one_const, neg_one, hidden_dim_unsqueezed}, 0);
    auto input_3d = std::make_shared<ov::op::v1::Reshape>(
        hidden_states_flat, reshape_to_3d_shape, false);  // [1, B*S, H]
    
    std::cout << "input_3d shape: " << input_3d->get_output_partial_shape(0) << std::endl;

    auto tile_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{num_experts_const,
                        one_const,
                        one_const}, 0);
    auto tiled_input = std::make_shared<ov::op::v0::Tile>(input_3d, tile_shape);  // [E, B*S, H]
    
    std::cout << "tiled_input shape: " << tiled_input->get_output_partial_shape(0) << std::endl;

    // CRITICAL: Add Reshape after Tile (required by pattern)
    auto reshape_after_tile_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{num_experts_const, neg_one, hidden_dim_unsqueezed}, 0);
    auto reshaped_tiled_input = std::make_shared<ov::op::v1::Reshape>(
        tiled_input, reshape_after_tile_shape, false);  // [E, B*S, H]
    
    // 7. Expert computation with transpose_b=TRUE
    // gate: [E, B*S, H] × [E, FF, H]^T -> [E, B*S, FF]
    auto gate_bmm = std::make_shared<ov::op::v0::MatMul>(
        reshaped_tiled_input, fused_gate_weights, false, true);  // transpose_b=true
    auto gate_swish = std::make_shared<ov::op::v4::Swish>(gate_bmm);
    
    // up: [E, B*S, H] × [E, FF, H]^T -> [E, B*S, FF]
    auto up_bmm = std::make_shared<ov::op::v0::MatMul>(
        reshaped_tiled_input, fused_up_weights, false, true);  // transpose_b=true
    
    // SwiGLU
    auto swiglu_mul = std::make_shared<ov::op::v1::Multiply>(gate_swish, up_bmm);
    
    // down: [E, B*S, FF] × [E, FF, H] -> [E, B*S, H]
    // Note: down weights are in [E, FF, H] layout, so no transpose needed
    auto down_bmm = std::make_shared<ov::op::v0::MatMul>(
        swiglu_mul, fused_down_weights, false, true);  // transpose_b=false
    
    // Add Reshape after down_bmm (required by pattern)
    auto down_reshape = std::make_shared<ov::op::v1::Reshape>(
        down_bmm, reshape_after_tile_shape, false);
    
    // 8. Routing path: MATCH THE PATTERN ORDER
    // Pattern: ScatterElementsUpdate -> Transpose -> Reshape -> Unsqueeze
    auto routing_transpose_perm = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0});
    auto routing_transposed = std::make_shared<ov::op::v1::Transpose>(
        routing_sparse, routing_transpose_perm);  // [E, B*S]
    
    auto routing_reshape_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{num_experts_const, seq_dim_unsqueezed}, 0);
    auto routing_reshaped = std::make_shared<ov::op::v1::Reshape>(
        routing_transposed, routing_reshape_shape, false);  // [E, B*S]
    
    auto routing_weights_3d = std::make_shared<ov::op::v0::Unsqueeze>(
        routing_reshaped, 
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, -1));  // [E, B*S, 1]
    //routing_weights_3d->set_friendly_name("debug_routing_weights_3d");
    
    // 9. Weight expert outputs and reduce
    //down_reshape->set_friendly_name("debug_expert_outputs_before_weight");
    auto weighted_outputs = std::make_shared<ov::op::v1::Multiply>(
        down_reshape, routing_weights_3d);
    
    auto expert_reduce_axis = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 0);
    auto reduced_output = std::make_shared<ov::op::v1::ReduceSum>(
        weighted_outputs, expert_reduce_axis, false);  // [B*S, H]
    
    // 10. Reshape back to original shape
    auto original_shape = std::make_shared<ov::op::v3::ShapeOf>(hidden_states, ov::element::i64);
    auto final_output = std::make_shared<ov::op::v1::Reshape>(
        reduced_output, original_shape, false);
    
    // DEBUG: Return weighted_outputs instead of final_output
    return final_output;
}

ov::Output<ov::Node> init_rope(
    int64_t head_dim,
    int64_t max_position_embeddings,
    float base,
    float scaling_factor) {

    // Calculate inverse frequencies
    size_t num_elements = head_dim / 2;
    std::vector<float> inv_freq_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        float idx = static_cast<float>(2 * i);  // Matches Python's step=2
        float exponent = idx / static_cast<float>(head_dim);
        inv_freq_data[i] = 1.0f / std::pow(base, exponent);
        
        // Apply scaling factor if needed (from original Python signature)
        if (scaling_factor != 1.0f) {
            inv_freq_data[i] *= scaling_factor;
        }
    }

    // Create OpenVINO constant with shape [1, num_elements, 1]
    ov::Shape const_shape = {1, static_cast<unsigned long>(num_elements), 1};
    auto rope_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, const_shape, inv_freq_data);

    return rope_const;
}
