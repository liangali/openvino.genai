// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/tests/test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/transpose.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

ov::Tensor make_tensor(const std::vector<float>& data, const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::memcpy(tensor.data(), data.data(), data.size() * sizeof(float));
    return tensor;
}

std::vector<float> make_seq(size_t n, float start, float step) {
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        out[i] = start + step * static_cast<float>(i);
    }
    return out;
}

std::vector<float> matmul_ref(const std::vector<float>& a,
                              const std::vector<float>& b,
                              size_t m,
                              size_t k,
                              size_t n) {
    std::vector<float> out(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t kk = 0; kk < k; ++kk) {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    return out;
}

std::vector<float> matmul_ref_transpose_a(const std::vector<float>& a,
                                          const std::vector<float>& b,
                                          size_t m,
                                          size_t k,
                                          size_t n) {
    std::vector<float> out(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t kk = 0; kk < k; ++kk) {
                acc += a[kk * m + i] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    return out;
}

std::vector<float> linear_ref(const std::vector<float>& x,
                              const std::vector<float>& w,
                              size_t rows,
                              size_t in_features,
                              size_t out_features) {
    std::vector<float> y(rows * out_features, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t o = 0; o < out_features; ++o) {
            float acc = 0.0f;
            for (size_t i = 0; i < in_features; ++i) {
                acc += x[r * in_features + i] * w[o * in_features + i];
            }
            y[r * out_features + o] = acc;
        }
    }
    return y;
}

std::vector<float> linear_ref_3d(const std::vector<float>& x,
                                 const std::vector<float>& w,
                                 size_t batch,
                                 size_t seq_len,
                                 size_t in_features,
                                 size_t out_features) {
    std::vector<float> out(batch * seq_len * out_features, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t x_base = (b * seq_len + s) * in_features;
            const size_t y_base = (b * seq_len + s) * out_features;
            for (size_t o = 0; o < out_features; ++o) {
                float acc = 0.0f;
                for (size_t i = 0; i < in_features; ++i) {
                    acc += x[x_base + i] * w[o * in_features + i];
                }
                out[y_base + o] = acc;
            }
        }
    }
    return out;
}

std::vector<float> linear_ref_3d_bias(const std::vector<float>& x,
                                      const std::vector<float>& w,
                                      const std::vector<float>& bias,
                                      size_t batch,
                                      size_t seq_len,
                                      size_t in_features,
                                      size_t out_features) {
    auto out = linear_ref_3d(x, w, batch, seq_len, in_features, out_features);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t base = (b * seq_len + s) * out_features;
            for (size_t o = 0; o < out_features; ++o) {
                out[base + o] += bias[o];
            }
        }
    }
    return out;
}

std::vector<float> mean_ref(const std::vector<float>& x, size_t rows, size_t cols) {
    std::vector<float> out(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            acc += x[r * cols + c];
        }
        out[r] = acc / static_cast<float>(cols);
    }
    return out;
}

std::vector<float> rms_ref(const std::vector<float>& x,
                           const std::vector<float>& weight,
                           size_t rows,
                           size_t cols,
                           float eps) {
    std::vector<float> out(x.size(), 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float sumsq = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float v = x[r * cols + c];
            sumsq += v * v;
        }
        float mean = sumsq / static_cast<float>(cols);
        float inv = 1.0f / std::sqrt(mean + eps);
        for (size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = x[r * cols + c] * inv * weight[c];
        }
    }
    return out;
}

std::vector<float> embedding_ref(const std::vector<int64_t>& ids,
                                 const std::vector<float>& weight,
                                 size_t rows,
                                 size_t cols,
                                 size_t embed_dim) {
    std::vector<float> y(rows * cols * embed_dim, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            int64_t idx = ids[r * cols + c];
            for (size_t e = 0; e < embed_dim; ++e) {
                y[(r * cols + c) * embed_dim + e] = weight[static_cast<size_t>(idx) * embed_dim + e];
            }
        }
    }
    return y;
}

std::vector<float> add_ref(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size(), 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

std::vector<float> mul_ref(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size(), 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * b[i];
    }
    return out;
}

std::vector<float> silu_ref(const std::vector<float>& x) {
    std::vector<float> out(x.size(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        const float v = x[i];
        out[i] = v / (1.0f + std::exp(-v));
    }
    return out;
}

std::vector<float> mlp_ref(const std::vector<float>& x,
                           const std::vector<float>& gate_w,
                           const std::vector<float>& up_w,
                           const std::vector<float>& down_w,
                           size_t batch,
                           size_t seq_len,
                           size_t hidden,
                           size_t intermediate) {
    auto gate = linear_ref_3d(x, gate_w, batch, seq_len, hidden, intermediate);
    auto up = linear_ref_3d(x, up_w, batch, seq_len, hidden, intermediate);
    auto gated = mul_ref(silu_ref(gate), up);
    return linear_ref_3d(gated, down_w, batch, seq_len, intermediate, hidden);
}

std::vector<float> random_f32(size_t count, float low, float high, uint32_t seed) {
    std::vector<float> data(count);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::generate(data.begin(), data.end(), [&]() { return dist(gen); });
    return data;
}

namespace {

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float swish(float x) {
    return x * sigmoid(x);
}

void topk_softmax(const float* logits,
                  size_t num_experts,
                  size_t top_k,
                  std::vector<size_t>& indices,
                  std::vector<float>& weights) {
    indices.resize(top_k);
    weights.resize(top_k);
    std::vector<size_t> order(num_experts);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                      [&](size_t a, size_t b) { return logits[a] > logits[b]; });

    float max_v = logits[order[0]];
    for (size_t i = 1; i < top_k; ++i) {
        max_v = std::max(max_v, logits[order[i]]);
    }
    float sum = 0.0f;
    for (size_t i = 0; i < top_k; ++i) {
        const float v = std::exp(logits[order[i]] - max_v);
        weights[i] = v;
        sum += v;
    }
    for (size_t i = 0; i < top_k; ++i) {
        weights[i] /= sum;
        indices[i] = order[i];
    }
}

void set_u4(uint8_t* packed, size_t idx, uint8_t v) {
    const size_t byte_idx = idx / 2;
    const uint8_t val = static_cast<uint8_t>(v & 0x0F);
    if ((idx & 1) == 0) {
        packed[byte_idx] = static_cast<uint8_t>((packed[byte_idx] & 0xF0) | val);
    } else {
        packed[byte_idx] = static_cast<uint8_t>((packed[byte_idx] & 0x0F) | (val << 4));
    }
}

uint8_t get_u4(const uint8_t* packed, size_t idx) {
    const size_t byte_idx = idx / 2;
    const uint8_t byte_val = packed[byte_idx];
    return (idx & 1) == 0 ? (byte_val & 0x0F) : (byte_val >> 4);
}

template <typename T>
std::vector<T> concat_vectors(const std::vector<std::vector<T>>& vectors) {
    size_t total_size = 0;
    for (const auto& v : vectors) {
        total_size += v.size();
    }
    std::vector<T> result;
    result.reserve(total_size);
    for (const auto& v : vectors) {
        result.insert(result.end(), v.begin(), v.end());
    }
    return result;
}

} // namespace

// Correct implementation for graph construction helper
ov::genai::modeling::Tensor make_dequant_subgraph(const Q41Quantized& q_weights,
                                           ov::genai::modeling::OpContext* op_ctx) {
    // Extract metadata
    const auto& w_shape = q_weights.weights_u4.get_shape();
    const auto& s_shape = q_weights.scales_f16.get_shape();

    int64_t N = (w_shape.size() == 3) ? w_shape[1] : w_shape[0];
    int64_t K = (w_shape.size() == 3) ? w_shape[2] : w_shape[1];
    int64_t G = q_weights.group_num;
    int64_t GS = q_weights.group_size;

    // 1. Construct Weights Constant with PRE-FOLDED shape [N, G, GS]
    // This allows Reshape to be avoided before Convert, satisfying FC_COMPRESSED_WEIGHT_PATTERN
    ov::Shape weights_const_shape = {static_cast<size_t>(N), static_cast<size_t>(G), static_cast<size_t>(GS)};
    
    auto weights_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::u4,
        weights_const_shape,
        q_weights.weights_u4.data()
    );

    // 2. Convert Weights to F16 directly
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f16);

    // 3. Prepare Scales and ZPs
    // Pattern requires Multiply/Subtract inputs to be Constants (not Transpose/Reshape nodes) for perfect match.
    // We check if data layout allows direct Constant creation with broadcastable shape [N, G, 1].
    
    // Check if scales are [..., N] (Legacy/3D: [E, G, N]) or [..., G] (New 2D: [N, G])
    bool need_transpose = (s_shape.back() == static_cast<size_t>(N));

    std::shared_ptr<ov::Node> scales_node;
    std::shared_ptr<ov::Node> zps_node;
    
    ov::Shape broadcast_shape = {static_cast<size_t>(N), static_cast<size_t>(G), 1};

    if (!need_transpose) {
        // Good layout [N, G], just wrap in Constant with [N, G, 1]
        scales_node = std::make_shared<ov::op::v0::Constant>(
            ov::element::f16, broadcast_shape, q_weights.scales_f16.data()
        );
        auto zps_const = std::make_shared<ov::op::v0::Constant>(
            ov::element::u4, broadcast_shape, q_weights.zps_u4.data()
        );
        zps_node = std::make_shared<ov::op::v0::Convert>(zps_const, ov::element::f16);
    } else {
        // Legacy layout [G, N]. Must use Transpose node (might miss strict pattern match)
        // or assumes the test only uses new layout for pattern validation.
        auto sz_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{G, N});
        
        auto sc_const = std::make_shared<ov::op::v0::Constant>(
            ov::element::f16, ov::Shape{static_cast<size_t>(G), static_cast<size_t>(N)}, q_weights.scales_f16.data()
        );
        auto zp_const = std::make_shared<ov::op::v0::Constant>(
            ov::element::u4, ov::Shape{static_cast<size_t>(G), static_cast<size_t>(N)}, q_weights.zps_u4.data()
        );
        auto zp_conv = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        
        // Transpose [G, N] -> [N, G]
        auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0});
        auto sc_t = std::make_shared<ov::op::v1::Transpose>(sc_const, perm);
        auto zp_t = std::make_shared<ov::op::v1::Transpose>(zp_conv, perm);
        
        // Reshape to [N, G, 1]
        auto bc_shape_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{N, G, 1});
        scales_node = std::make_shared<ov::op::v1::Reshape>(sc_t, bc_shape_const, false);
        zps_node = std::make_shared<ov::op::v1::Reshape>(zp_t, bc_shape_const, false);
    }

    // 4. Compute (Weights - ZP) * Scale
    auto sub = std::make_shared<ov::op::v1::Subtract>(weights_f16, zps_node);
    auto mul = std::make_shared<ov::op::v1::Multiply>(sub, scales_node);
    
    // 5. Flatten back to [N, K] (Matches reshape_squeeze predicate)
    auto final_measured = std::make_shared<ov::op::v1::Reshape>(
        mul,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{N, K}),
        false
    );

    auto final_measured_f32 = std::make_shared<ov::op::v0::Convert>(final_measured, ov::element::f32);

    return ov::genai::modeling::Tensor(final_measured_f32->output(0), op_ctx);
}

Q41Quantized quantize_q41(const std::vector<float>& weights_f32,
                          size_t num_experts,
                          size_t n,
                          size_t k,
                          size_t group_size) {
    Q41Quantized q;
    q.group_size = group_size;
    q.k = k;
    q.group_num = k / group_size;

    ov::Shape weights_shape{num_experts, n, k};
    ov::Tensor weights(ov::element::u4, weights_shape);
    auto weights_ptr = static_cast<uint8_t*>(weights.data());
    std::memset(weights_ptr, 0, weights.get_byte_size());

    ov::Shape scale_zp_shape{num_experts, q.group_num, n};
    ov::Tensor scales(ov::element::f16, scale_zp_shape);
    ov::Tensor zps(ov::element::u4, scale_zp_shape);

    auto scales_ptr = scales.data<ov::float16>();
    auto zps_ptr = static_cast<uint8_t*>(zps.data());
    std::memset(zps_ptr, 0, zps.get_byte_size());

    for (size_t e = 0; e < num_experts; ++e) {
        for (size_t row = 0; row < n; ++row) {
            for (size_t g = 0; g < q.group_num; ++g) {
                const size_t start_idx = ((e * n + row) * k) + g * group_size;
                float min_v = std::numeric_limits<float>::max();
                float max_v = std::numeric_limits<float>::lowest();
                for (size_t i = 0; i < group_size; ++i) {
                    const float v = weights_f32[start_idx + i];
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                }

                const float d = (max_v - min_v) / 15.0f;
                const float inv = d ? 1.0f / d : 0.0f;
                int zp = 0;
                if (d != 0.0f) {
                    zp = static_cast<int>(std::round(-min_v / d));
                }
                zp = std::max(0, std::min(15, zp));

                const size_t scale_idx = (e * q.group_num + g) * n + row;
                scales_ptr[scale_idx] = ov::float16(d);
                set_u4(zps_ptr, scale_idx, static_cast<uint8_t>(zp));

                for (size_t i = 0; i < group_size; ++i) {
                    float q_f = (weights_f32[start_idx + i] - min_v) * inv;
                    int q_i = static_cast<int>(std::round(q_f));
                    q_i = std::max(0, std::min(15, q_i));
                    set_u4(weights_ptr, start_idx + i, static_cast<uint8_t>(q_i));
                }
            }
        }
    }

    q.weights_u4 = weights;
    q.scales_f16 = scales;
    q.zps_u4 = zps;

    q.weights_packed.resize(weights.get_byte_size());
    std::memcpy(q.weights_packed.data(), weights.data(), q.weights_packed.size());

    q.zps_packed.resize(zps.get_byte_size());
    std::memcpy(q.zps_packed.data(), zps.data(), q.zps_packed.size());

    q.scales.resize(scales.get_size());
    std::memcpy(q.scales.data(), scales.data<ov::float16>(), q.scales.size() * sizeof(ov::float16));

    return q;
}

Q41Quantized quantize_q41(const std::vector<float>& weights_f32,
                          size_t n,
                          size_t k,
                          size_t group_size) {
    Q41Quantized q;
    q.group_size = group_size;
    q.k = k;
    q.group_num = k / group_size;

    ov::Shape weights_shape{n, k};
    ov::Tensor weights(ov::element::u4, weights_shape);
    auto weights_ptr = static_cast<uint8_t*>(weights.data());
    std::memset(weights_ptr, 0, weights.get_byte_size());

    // Layout: [N, Group_Num]
    ov::Shape scale_zp_shape{n, q.group_num};
    ov::Tensor scales(ov::element::f16, scale_zp_shape);
    ov::Tensor zps(ov::element::u4, scale_zp_shape);

    auto scales_ptr = scales.data<ov::float16>();
    auto zps_ptr = static_cast<uint8_t*>(zps.data());
    std::memset(zps_ptr, 0, zps.get_byte_size());

    for (size_t row = 0; row < n; ++row) {
        for (size_t g = 0; g < q.group_num; ++g) {
            const size_t start_idx = row * k + g * group_size;
            float min_v = std::numeric_limits<float>::max();
            float max_v = std::numeric_limits<float>::lowest();
            
            for (size_t i = 0; i < group_size; ++i) {
                const float v = weights_f32[start_idx + i];
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }

            const float d = (max_v - min_v) / 15.0f;
            const float inv = d ? 1.0f / d : 0.0f;
            int zp = 0;
            if (d != 0.0f) {
                zp = static_cast<int>(std::round(-min_v / d));
            }
            zp = std::max(0, std::min(15, zp));

            // Matching logic: scale_idx = row * q.group_num + g;
            const size_t scale_idx = row * q.group_num + g;

            scales_ptr[scale_idx] = ov::float16(d);
            set_u4(zps_ptr, scale_idx, static_cast<uint8_t>(zp));

            for (size_t i = 0; i < group_size; ++i) {
                float q_f = (weights_f32[start_idx + i] - min_v) * inv;
                int q_i = static_cast<int>(std::round(q_f));
                q_i = std::max(0, std::min(15, q_i));
                set_u4(weights_ptr, start_idx + i, static_cast<uint8_t>(q_i));
            }
        }
    }

    q.weights_u4 = weights;
    q.scales_f16 = scales;
    q.zps_u4 = zps;

    q.weights_packed.resize(weights.get_byte_size());
    std::memcpy(q.weights_packed.data(), weights.data(), q.weights_packed.size());

    q.zps_packed.resize(zps.get_byte_size());
    std::memcpy(q.zps_packed.data(), zps.data(), q.zps_packed.size());

    q.scales.resize(scales.get_size());
    std::memcpy(q.scales.data(), scales.data<ov::float16>(), q.scales.size() * sizeof(ov::float16));

    return q;
}

std::vector<float> dequantize_q41(const Q41Quantized& q,
                                  size_t num_experts,
                                  size_t n,
                                  size_t k) {
    std::vector<float> deq(num_experts * n * k, 0.0f);
    const uint8_t* weights_packed = q.weights_packed.data();
    const uint8_t* zps_packed = q.zps_packed.data();

    bool is_2d_layout = q.scales_f16.get_shape().size() == 2 && 
                        q.scales_f16.get_shape()[0] == n;

    for (size_t e = 0; e < num_experts; ++e) {
        for (size_t row = 0; row < n; ++row) {
            for (size_t g = 0; g < q.group_num; ++g) {
                size_t scale_idx;
                if (is_2d_layout) {
                     scale_idx = row * q.group_num + g;
                } else {
                     scale_idx = (e * q.group_num + g) * n + row;
                }
                
                const float scale = static_cast<float>(q.scales[scale_idx]);
                const int zp = static_cast<int>(get_u4(zps_packed, scale_idx));
                for (size_t kk = 0; kk < q.group_size; ++kk) {
                    const size_t w_idx = ((e * n + row) * q.group_num + g) * q.group_size + kk;
                    const int qv = static_cast<int>(get_u4(weights_packed, w_idx));
                    const float v = (static_cast<float>(qv - zp)) * scale;
                    const size_t out_idx = ((e * n + row) * k) + g * q.group_size + kk;
                    deq[out_idx] = v;
                }
            }
        }
    }
    return deq;
}

std::vector<float> moe_ref(const std::vector<float>& hidden_states,
                           const std::vector<float>& gate_inp,
                           const std::vector<float>& gate_w,
                           const std::vector<float>& up_w,
                           const std::vector<float>& down_w,
                           size_t batch,
                           size_t seq_len,
                           size_t hidden_size,
                           size_t inter_size,
                           size_t num_experts,
                           size_t top_k) {
    const size_t tokens = batch * seq_len;
    std::vector<float> output(tokens * hidden_size, 0.0f);
    std::vector<size_t> topk_idx;
    std::vector<float> topk_w;
    std::vector<float> logits(num_experts, 0.0f);

    // hidden_states is interpreted as contiguous [B, S, H], which is equivalent
    // to flattened [T, H] with T = B * S.
    auto hidden_ptr = [&](size_t b, size_t s) -> const float* {
        const size_t t = b * seq_len + s;
        return &hidden_states[t * hidden_size];
    };
    auto out_ptr = [&](size_t b, size_t s) -> float* {
        const size_t t = b * seq_len + s;
        return &output[t * hidden_size];
    };

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const float* x = hidden_ptr(b, s);
            for (size_t e = 0; e < num_experts; ++e) {
                float acc = 0.0f;
                const size_t base = e * hidden_size;
                for (size_t h = 0; h < hidden_size; ++h) {
                    acc += x[h] * gate_inp[base + h];
                }
                logits[e] = acc;
            }
            topk_softmax(logits.data(), num_experts, top_k, topk_idx, topk_w);

            for (size_t k = 0; k < top_k; ++k) {
                const size_t e = topk_idx[k];
                const float w = topk_w[k];

                std::vector<float> gate(inter_size, 0.0f);
                std::vector<float> up(inter_size, 0.0f);
                for (size_t i = 0; i < inter_size; ++i) {
                    float acc_g = 0.0f;
                    float acc_u = 0.0f;
                    const size_t base = (e * inter_size + i) * hidden_size;
                    for (size_t h = 0; h < hidden_size; ++h) {
                        acc_g += x[h] * gate_w[base + h];
                        acc_u += x[h] * up_w[base + h];
                    }
                    gate[i] = swish(acc_g);
                    up[i] = acc_u;
                }

                std::vector<float> hidden(inter_size, 0.0f);
                for (size_t i = 0; i < inter_size; ++i) {
                    hidden[i] = gate[i] * up[i];
                }

                float* out = out_ptr(b, s);
                for (size_t h = 0; h < hidden_size; ++h) {
                    float acc = 0.0f;
                    const size_t base = (e * hidden_size + h) * inter_size;
                    for (size_t i = 0; i < inter_size; ++i) {
                        acc += hidden[i] * down_w[base + i];
                    }
                    out[h] += w * acc;
                }
            }
        }
    }
    return output;
}

std::pair<std::vector<float>, std::vector<float>> linear_attention_ref(const std::vector<float>& q,
                                                                        const std::vector<float>& k,
                                                                        const std::vector<float>& v,
                                                                        const std::vector<float>& beta,
                                                                        const std::vector<float>& g,
                                                                        const std::vector<float>& initial_state,
                                                                        size_t batch,
                                                                        size_t seq_len,
                                                                        size_t num_heads,
                                                                        size_t head_dim) {
    const size_t K = head_dim;
    const size_t V = head_dim;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(K));
    const float eps = 1e-6f;

    auto l2norm = [&](std::vector<float>& x) {
        float sum = 0.0f;
        for (float v : x) {
            sum += v * v;
        }
        float inv = 1.0f / std::sqrt(sum + eps);
        for (auto& v : x) {
            v *= inv;
        }
    };

    std::vector<float> out(batch * seq_len * num_heads * V, 0.0f);
    std::vector<float> out_state(batch * num_heads * K * V, 0.0f);

    auto q_idx = [&](size_t b, size_t t, size_t h, size_t d) {
        return ((b * seq_len + t) * num_heads + h) * K + d;
    };
    auto v_idx = [&](size_t b, size_t t, size_t h, size_t d) {
        return ((b * seq_len + t) * num_heads + h) * V + d;
    };
    auto g_idx = [&](size_t b, size_t t, size_t h) {
        return (b * seq_len + t) * num_heads + h;
    };
    auto state_idx = [&](size_t b, size_t h, size_t kdim, size_t vdim) {
        return ((b * num_heads + h) * V + vdim) * K + kdim;
    };

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t vdim = 0; vdim < V; ++vdim) {
                std::vector<float> state(K, 0.0f);
                for (size_t kdim = 0; kdim < K; ++kdim) {
                    state[kdim] = initial_state[state_idx(b, h, kdim, vdim)];
                }

                for (size_t t = 0; t < seq_len; ++t) {
                    float b_g = std::exp(g[g_idx(b, t, h)]);
                    float b_beta = beta[g_idx(b, t, h)];

                    std::vector<float> b_k(K, 0.0f);
                    std::vector<float> b_q(K, 0.0f);
                    for (size_t kdim = 0; kdim < K; ++kdim) {
                        b_k[kdim] = k[q_idx(b, t, h, kdim)];
                        b_q[kdim] = q[q_idx(b, t, h, kdim)];
                    }
                    l2norm(b_k);
                    l2norm(b_q);
                    for (auto& vq : b_q) {
                        vq *= q_scale;
                    }

                    for (auto& s : state) {
                        s *= b_g;
                    }

                    float h_k = 0.0f;
                    for (size_t kdim = 0; kdim < K; ++kdim) {
                        h_k += state[kdim] * b_k[kdim];
                    }

                    float b_v = v[v_idx(b, t, h, vdim)];
                    b_v = (b_v - h_k) * b_beta;

                    for (size_t kdim = 0; kdim < K; ++kdim) {
                        state[kdim] += b_k[kdim] * b_v;
                    }

                    float out_v = 0.0f;
                    for (size_t kdim = 0; kdim < K; ++kdim) {
                        out_v += state[kdim] * b_q[kdim];
                    }
                    out[v_idx(b, t, h, vdim)] = out_v;
                }

                for (size_t kdim = 0; kdim < K; ++kdim) {
                    out_state[state_idx(b, h, kdim, vdim)] = state[kdim];
                }
            }
        }
    }

    return {out, out_state};
}

std::vector<float> to_heads_ref(const std::vector<float>& x,
                                size_t batch,
                                size_t seq_len,
                                size_t num_heads,
                                size_t head_dim) {
    std::vector<float> out(batch * num_heads * seq_len * head_dim, 0.0f);
    const size_t hidden = num_heads * head_dim;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t in_base = (b * seq_len + s) * hidden;
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t out_base = ((b * num_heads + h) * seq_len + s) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    out[out_base + d] = x[in_base + h * head_dim + d];
                }
            }
        }
    }
    return out;
}

std::vector<float> rmsnorm_heads_ref(const std::vector<float>& x,
                                     const std::vector<float>& weight,
                                     size_t batch,
                                     size_t num_heads,
                                     size_t seq_len,
                                     size_t head_dim,
                                     float eps) {
    std::vector<float> out(x.size(), 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t base = ((b * num_heads + h) * seq_len + s) * head_dim;
                float sumsq = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    const float v = x[base + d];
                    sumsq += v * v;
                }
                const float mean = sumsq / static_cast<float>(head_dim);
                const float inv = 1.0f / std::sqrt(mean + eps);
                for (size_t d = 0; d < head_dim; ++d) {
                    out[base + d] = x[base + d] * inv * weight[d];
                }
            }
        }
    }
    return out;
}

std::vector<float> merge_heads_ref(const std::vector<float>& x,
                                   size_t batch,
                                   size_t seq_len,
                                   size_t num_heads,
                                   size_t head_dim) {
    std::vector<float> out(batch * seq_len * num_heads * head_dim, 0.0f);
    const size_t hidden = num_heads * head_dim;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t out_base = (b * seq_len + s) * hidden;
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t in_base = ((b * num_heads + h) * seq_len + s) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    out[out_base + h * head_dim + d] = x[in_base + d];
                }
            }
        }
    }
    return out;
}

std::vector<float> apply_rope_ref(const std::vector<float>& x,
                                  const std::vector<int64_t>& positions,
                                  size_t batch,
                                  size_t seq_len,
                                  size_t num_heads,
                                  size_t head_dim,
                                  float rope_theta) {
    const size_t half_dim = head_dim / 2;
    std::vector<float> inv_freq(half_dim, 0.0f);
    for (size_t i = 0; i < half_dim; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
        inv_freq[i] = 1.0f / std::pow(rope_theta, exponent);
    }

    std::vector<float> out(x.size(), 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const float pos = static_cast<float>(positions[b * seq_len + s]);
            for (size_t i = 0; i < half_dim; ++i) {
                const float angle = pos * inv_freq[i];
                const float c = std::cos(angle);
                const float si = std::sin(angle);
                for (size_t h = 0; h < num_heads; ++h) {
                    const size_t base = ((b * num_heads + h) * seq_len + s) * head_dim;
                    const float x1 = x[base + i];
                    const float x2 = x[base + i + half_dim];
                    out[base + i] = x1 * c - x2 * si;
                    out[base + i + half_dim] = x1 * si + x2 * c;
                }
            }
        }
    }
    return out;
}

std::vector<float> repeat_kv_ref(const std::vector<float>& x,
                                 size_t batch,
                                 size_t num_heads,
                                 size_t num_kv_heads,
                                 size_t seq_len,
                                 size_t head_dim) {
    if (num_heads == num_kv_heads) {
        return x;
    }
    std::vector<float> out(batch * num_heads * seq_len * head_dim, 0.0f);
    const size_t repeats = num_heads / num_kv_heads;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t kv = 0; kv < num_kv_heads; ++kv) {
            for (size_t r = 0; r < repeats; ++r) {
                const size_t h = kv * repeats + r;
                for (size_t s = 0; s < seq_len; ++s) {
                    const size_t in_base = ((b * num_kv_heads + kv) * seq_len + s) * head_dim;
                    const size_t out_base = ((b * num_heads + h) * seq_len + s) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        out[out_base + d] = x[in_base + d];
                    }
                }
            }
        }
    }
    return out;
}

std::vector<float> attention_ref(const std::vector<float>& hidden,
                                 const std::vector<float>& q_w,
                                 const std::vector<float>& q_b,
                                 const std::vector<float>& k_w,
                                 const std::vector<float>& k_b,
                                 const std::vector<float>& v_w,
                                 const std::vector<float>& v_b,
                                 const std::vector<float>& o_w,
                                 const std::vector<float>& o_b,
                                 const std::vector<float>* q_norm_w,
                                 const std::vector<float>* k_norm_w,
                                 const std::vector<int64_t>& positions,
                                 size_t batch,
                                 size_t seq_len,
                                 size_t hidden_size,
                                 size_t num_heads,
                                 size_t num_kv_heads,
                                 size_t head_dim,
                                 float rope_theta,
                                 float rms_norm_eps,
                                 bool use_rope) {
    const size_t kv_hidden = num_kv_heads * head_dim;
    auto q = linear_ref_3d_bias(hidden, q_w, q_b, batch, seq_len, hidden_size, hidden_size);
    auto k = linear_ref_3d_bias(hidden, k_w, k_b, batch, seq_len, hidden_size, kv_hidden);
    auto v = linear_ref_3d_bias(hidden, v_w, v_b, batch, seq_len, hidden_size, kv_hidden);

    auto qh = to_heads_ref(q, batch, seq_len, num_heads, head_dim);
    auto kh = to_heads_ref(k, batch, seq_len, num_kv_heads, head_dim);
    auto vh = to_heads_ref(v, batch, seq_len, num_kv_heads, head_dim);

    if (q_norm_w) {
        qh = rmsnorm_heads_ref(qh, *q_norm_w, batch, num_heads, seq_len, head_dim, rms_norm_eps);
    }
    if (k_norm_w) {
        kh = rmsnorm_heads_ref(kh, *k_norm_w, batch, num_kv_heads, seq_len, head_dim, rms_norm_eps);
    }

    if (use_rope) {
        qh = apply_rope_ref(qh, positions, batch, seq_len, num_heads, head_dim, rope_theta);
        kh = apply_rope_ref(kh, positions, batch, seq_len, num_kv_heads, head_dim, rope_theta);
    }

    auto kh_expanded = repeat_kv_ref(kh, batch, num_heads, num_kv_heads, seq_len, head_dim);
    auto vh_expanded = repeat_kv_ref(vh, batch, num_heads, num_kv_heads, seq_len, head_dim);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> attn_probs(batch * num_heads * seq_len * seq_len, 0.0f);
    std::vector<float> scores(seq_len, 0.0f);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                float max_score = -1e30f;
                for (size_t j = 0; j < seq_len; ++j) {
                    if (j > i) {
                        scores[j] = -65504.0f;
                        continue;
                    }
                    float acc = 0.0f;
                    const size_t q_base = ((b * num_heads + h) * seq_len + i) * head_dim;
                    const size_t k_base = ((b * num_heads + h) * seq_len + j) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        acc += qh[q_base + d] * kh_expanded[k_base + d];
                    }
                    acc *= scale;
                    scores[j] = acc;
                    if (acc > max_score) {
                        max_score = acc;
                    }
                }
                float sum = 0.0f;
                for (size_t j = 0; j < seq_len; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum += scores[j];
                }
                const size_t prob_base = ((b * num_heads + h) * seq_len + i) * seq_len;
                for (size_t j = 0; j < seq_len; ++j) {
                    attn_probs[prob_base + j] = scores[j] / sum;
                }
            }
        }
    }

    std::vector<float> context(batch * num_heads * seq_len * head_dim, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                const size_t prob_base = ((b * num_heads + h) * seq_len + i) * seq_len;
                const size_t ctx_base = ((b * num_heads + h) * seq_len + i) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < seq_len; ++j) {
                        const size_t v_base = ((b * num_heads + h) * seq_len + j) * head_dim;
                        acc += attn_probs[prob_base + j] * vh_expanded[v_base + d];
                    }
                    context[ctx_base + d] = acc;
                }
            }
        }
    }

    auto merged = merge_heads_ref(context, batch, seq_len, num_heads, head_dim);
    return linear_ref_3d_bias(merged, o_w, o_b, batch, seq_len, hidden_size, hidden_size);
}

void expect_tensor_near(const ov::Tensor& output, const std::vector<float>& expected, float tol) {
    const float abs_tol = tol;
    const float rel_tol = tol;
    constexpr size_t k_max_mismatch_logs = 100;

    ASSERT_EQ(output.get_size(), expected.size());
    const float* out_data = output.data<const float>();

    constexpr float k_threshold = 1.0f;

    size_t pass_count = 0;
    size_t fail_count = 0;
    size_t n_abs = 0;
    size_t n_rel = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    float max_error = 0.0f;
    std::vector<std::string> mismatch_logs;
    mismatch_logs.reserve(k_max_mismatch_logs);

    for (size_t i = 0; i < expected.size(); ++i) {
        float actual = out_data[i];
        float expected_val = expected[i];
        float diff = std::abs(actual - expected_val);
        float expected_magnitude = std::abs(expected_val);
        max_error = std::max(max_error, diff); 
        max_abs_error = std::max(max_abs_error, diff);

        bool pass = false;
        if (expected_magnitude > k_threshold) {
            float rel_error = diff / expected_magnitude;
            max_rel_error = std::max(max_rel_error, rel_error);
            ++n_rel;
            pass = (rel_error <= rel_tol);

            if (!pass) {
                if (mismatch_logs.size() < k_max_mismatch_logs) {
                    std::ostringstream oss;
                    oss << "Element " << i << ": Relative error " << (rel_error * 100.0f) << "%"
                        << " exceeds tolerance " << (rel_tol * 100.0f) << "%"
                        << " (actual=" << actual << ", expected=" << expected_val << ", abs_diff=" << diff << ")";
                    mismatch_logs.push_back(oss.str());
                }
            }
        } else {
            ++n_abs;
            pass = (diff <= abs_tol);
            if (!pass) {
                if (mismatch_logs.size() < k_max_mismatch_logs) {
                    std::ostringstream oss;
                    oss << "Element " << i << ": Absolute error " << diff << " exceeds tolerance " << abs_tol
                        << " (actual=" << actual << ", expected=" << expected_val << ")";
                    mismatch_logs.push_back(oss.str());
                }
            }
        }

        if (pass) {
            ++pass_count;
        } else {
            ++fail_count;
        }
    }

    if (fail_count > 0) {
        const size_t total_count = expected.size();
        const size_t match_percent = total_count == 0
                                         ? 0
                                         : static_cast<size_t>(std::round(100.0 * static_cast<double>(pass_count) /
                                                                           static_cast<double>(total_count)));
        const size_t mismatch_percent = total_count == 0
                                            ? 0
                                            : static_cast<size_t>(std::round(100.0 * static_cast<double>(fail_count) /
                                                                              static_cast<double>(total_count)));

        std::ostringstream oss;
        oss << "Tensor compare failed.\n"
            << "Stats: total=" << total_count
            << ", match=" << pass_count << " (" << match_percent << "%)"
            << ", mismatch=" << fail_count << " (" << mismatch_percent << "%)"
            << ", abs_mode_count=" << n_abs
            << ", rel_mode_count=" << n_rel
            << ", max_abs_error=" << max_abs_error
            << ", max_rel_error=" << max_rel_error
            << ", max_error=" << max_error << "\n"
            << "Showing first " << mismatch_logs.size() << " mismatch logs";
        if (fail_count > mismatch_logs.size()) {
            oss << " (remaining " << (fail_count - mismatch_logs.size()) << " omitted)";
        }
        oss << ":\n";

        for (const auto& line : mismatch_logs) {
            oss << line << '\n';
        }

        ADD_FAILURE() << oss.str();
    }
}

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov
