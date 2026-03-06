#include <openvino/openvino.hpp>
#include "openvino/opsets/opset13.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/op/moe_3gemm_fused_compressed.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <cfloat>
#include <vector>
#include <cfloat>

#ifndef MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static float swish(float x) {
    return x * sigmoid(x);
}

static std::vector<float> random_f32(size_t count, float low, float high) {
    std::vector<float> data(count);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(low, high);
    std::generate(data.begin(), data.end(), [&]() { return dist(gen); });
    return data;
}

static void set_u4(std::vector<uint8_t>& packed, size_t idx, uint8_t v) {
    const size_t byte_idx = idx / 2;
    const uint8_t val = static_cast<uint8_t>(v & 0x0F);
    if ((idx & 1) == 0) {
        packed[byte_idx] = static_cast<uint8_t>((packed[byte_idx] & 0xF0) | val);
    } else {
        packed[byte_idx] = static_cast<uint8_t>((packed[byte_idx] & 0x0F) | (val << 4));
    }
}

static void set_u4(uint8_t* packed, size_t idx, uint8_t v) {
    const size_t byte_idx = idx / 2;
    const uint8_t val = static_cast<uint8_t>(v & 0x0F);
    if ((idx & 1) == 0) {
        packed[byte_idx] = static_cast<uint8_t>((packed[byte_idx] & 0xF0) | val);
    } else {
        packed[byte_idx] = static_cast<uint8_t>((packed[byte_idx] & 0x0F) | (val << 4));
    }
}

static uint8_t get_u4(const std::vector<uint8_t>& packed, size_t idx) {
    const size_t byte_idx = idx / 2;
    const uint8_t byte_val = packed[byte_idx];
    return (idx & 1) == 0 ? (byte_val & 0x0F) : (byte_val >> 4);
}

struct Q41Quantized {
    ov::Tensor weights_u4;
    ov::Tensor scales_f16;
    ov::Tensor zps_u4;
    std::vector<uint8_t> weights_packed;
    std::vector<uint8_t> zps_packed;
    std::vector<ov::float16> scales;
    size_t group_num = 0;
    size_t group_size = 0;
    size_t k = 0;
};

static Q41Quantized quantize_q41(const std::vector<float>& weights_f32,
                                 size_t num_experts,
                                 size_t n,
                                 size_t k,
                                 size_t group_size) {
    Q41Quantized q;
    q.group_size = group_size;
    q.k = k;
    q.group_num = k / group_size;

    printf("quantize q41, group_size: %d, n: %d, k: %d\n", group_size, n, k);

    ov::Shape shape{num_experts, n, k};
    size_t shape_1 = shape[1];
    size_t shape_2 = shape[2];

    auto weights_shape = shape;

    ov::Tensor weights(ov::element::u4, std::move(weights_shape));     // [num_experts, shape 1, group_num, group_size // 2]
    auto weights_ptr = static_cast<uint8_t*>(weights.data());

    size_t group_num = shape_2 / group_size;

    ov::Shape scale_zp_shape{num_experts, group_num, shape_1};

    ov::Tensor scales(ov::element::f16, scale_zp_shape);
    // ov::Tensor biases(ov::element::f16, std::move(shape));
    ov::Tensor zps(ov::element::u4, std::move(scale_zp_shape));

    auto scales_ptr = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    // auto biases_ptr = biases.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto zps_ptr = static_cast<uint8_t*>(zps.data());

    for (size_t e = 0; e < num_experts; ++e) {
        for (size_t row = 0; row < shape_1; ++row) {
            for (int g = 0; g < group_num; g ++) {
                // size_t start_idx = g * group_size;
                size_t start_idx = ((e * shape_1 + row) * shape_2) + g * group_size;
                float max = -FLT_MAX;
                float min = FLT_MAX;
                for (int i = 0; i < group_size; i ++) {
                    const float v = weights_f32[start_idx + i];
                    if (v > max) {
                        max = v;
                    }
                    if (v < min) {
                        min = v;
                    }
                }

                const float d = (max - min) / ((1 << 4) - 1);
                const float id = d ? 1.0f/d : 0.0f;
                uint8_t zp = (uint8_t)std::round(-1.f * min / d);

                const size_t scale_idx = (e * group_num + g) * shape_1 + row;
                scales_ptr[scale_idx] = ov::float16(d);
                // biases_ptr[scale_idx] = ov::float16(zp);
                set_u4(zps_ptr, scale_idx, zp);

                for (int j = 0; j < group_size/2; ++ j) {
                    const float x0 = (weights_f32[start_idx + j * 2    ] - min)*id;
                    const float x1 = (weights_f32[start_idx + j * 2 + 1] - min)*id;
                    const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
                    const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));
                    weights_ptr[start_idx / 2 + j]  = xi0;
                    weights_ptr[start_idx / 2 + j] |= xi1 << 4;
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
    std::memcpy(q.scales.data(),
                scales.data<ov::float16>(),
                q.scales.size() * sizeof(ov::float16));


    // const size_t weight_elems = num_experts * n * q.group_num * group_size;
    // const size_t zp_elems = num_experts * n * q.group_num;
    // const size_t scale_elems = zp_elems;

    // q.weights_packed.assign((weight_elems + 1) / 2, 0);
    // q.zps_packed.assign((zp_elems + 1) / 2, 0);
    // q.scales.assign(scale_elems, ov::float16(0.0f));

    // for (size_t e = 0; e < num_experts; ++e) {
    //     for (size_t row = 0; row < n; ++row) {
    //         for (size_t g = 0; g < q.group_num; ++g) {
    //             const size_t base = ((e * n + row) * k) + g * group_size;
    //             float min_v = std::numeric_limits<float>::max();
    //             float max_v = -std::numeric_limits<float>::max();
    //             for (size_t kk = 0; kk < group_size; ++kk) {
    //                 const float v = weights_f32[base + kk];
    //                 min_v = std::min(min_v, v);
    //                 max_v = std::max(max_v, v);
    //             }

    //             float scale = (max_v - min_v) / 15.0f;
    //             if (scale == 0.0f) {
    //                 scale = 0.0f;
    //             }
    //             float zp_f = (scale == 0.0f) ? 0.0f : std::round(-min_v / scale);
    //             int zp_i = static_cast<int>(std::max(0.0f, std::min(15.0f, zp_f)));

    //             const size_t scale_idx = (e * q.group_num + g) * n + row;
    //             q.scales[scale_idx] = ov::float16(scale);
    //             set_u4(q.zps_packed, scale_idx, static_cast<uint8_t>(zp_i));

    //             for (size_t kk = 0; kk < group_size; ++kk) {
    //                 const float v = weights_f32[base + kk];
    //                 float q_f = (scale == 0.0f) ? 0.0f : std::round(v / scale + zp_i);
    //                 int q_i = static_cast<int>(std::max(0.0f, std::min(15.0f, q_f)));
    //                 const size_t w_idx = ((e * n + row) * q.group_num + g) * group_size + kk;
    //                 set_u4(q.weights_packed, w_idx, static_cast<uint8_t>(q_i));
    //             }
    //         }
    //     }
    // }

    // ov::Shape w_shape{num_experts, n, q.group_num, group_size};
    // ov::Shape s_shape{num_experts, q.group_num, n, 1};
    // ov::Shape z_shape{num_experts, q.group_num, n, 1};
    // q.weights_u4 = ov::Tensor(ov::element::u4, w_shape);
    // q.scales_f16 = ov::Tensor(ov::element::f16, s_shape);
    // q.zps_u4 = ov::Tensor(ov::element::u4, z_shape);

    // std::memcpy(q.weights_u4.data(), q.weights_packed.data(), q.weights_packed.size());
    // std::memcpy(q.scales_f16.data<ov::float16>(), q.scales.data(), q.scales.size() * sizeof(ov::float16));
    // std::memcpy(q.zps_u4.data(), q.zps_packed.data(), q.zps_packed.size());

    return q;
}

static std::vector<float> dequantize_q41(const Q41Quantized& q,
                                         size_t num_experts,
                                         size_t n,
                                         size_t k) {
    std::vector<float> deq(num_experts * n * k, 0.0f);
    for (size_t e = 0; e < num_experts; ++e) {
        for (size_t row = 0; row < n; ++row) {
            for (size_t g = 0; g < q.group_num; ++g) {
                const size_t scale_idx = (e * q.group_num + g) * n + row;
                const float scale = static_cast<float>(q.scales[scale_idx]);
                const int zp = static_cast<int>(get_u4(q.zps_packed, scale_idx));
                for (size_t kk = 0; kk < q.group_size; ++kk) {
                    const size_t w_idx = ((e * n + row) * q.group_num + g) * q.group_size + kk;
                    const int qv = static_cast<int>(get_u4(q.weights_packed, w_idx));
                    const float v = (static_cast<float>(qv - zp)) * scale;
                    const size_t out_idx = ((e * n + row) * k) + g * q.group_size + kk;
                    deq[out_idx] = v;
                }
            }
        }
    }
    return deq;
}

static void topk_softmax(const float* logits,
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

static std::vector<float> moe_cpu_ref(const std::vector<float>& hidden_states,
                                      const std::vector<float>& gate_inp,
                                      const std::vector<float>& gate_w,
                                      const std::vector<float>& up_w,
                                      const std::vector<float>& down_w,
                                      size_t batch,
                                      size_t seq,
                                      size_t hidden_size,
                                      size_t inter_size,
                                      size_t num_experts,
                                      size_t top_k) {
    const size_t tokens = batch * seq;
    std::vector<float> output(tokens * hidden_size, 0.0f);
    std::vector<size_t> topk_idx;
    std::vector<float> topk_w;
    std::vector<float> logits(num_experts, 0.0f);

    for (size_t t = 0; t < tokens; ++t) {
        const float* x = &hidden_states[t * hidden_size];
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

            float* out = &output[t * hidden_size];
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
    return output;
}

static void print_usage(const char* prog,
                        size_t seq,
                        size_t hidden_size,
                        size_t inter_size,
                        size_t num_experts,
                        size_t top_k) {
    std::cerr << "Usage: " << prog
              << " [--seq N]"
              << " [--hidden_size N]"
              << " [--inter_size N]"
              << " [--num_experts N]"
              << " [--top_k N]\n"
              << "Defaults: --seq " << seq
              << " --hidden_size " << hidden_size
              << " --inter_size " << inter_size
              << " --num_experts " << num_experts
              << " --top_k " << top_k
              << "\n";
}

static bool parse_size_t_arg(const char* text, size_t& out) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    try {
        size_t idx = 0;
        unsigned long long val = std::stoull(text, &idx, 10);
        if (idx != std::strlen(text)) {
            return false;
        }
        out = static_cast<size_t>(val);
        return true;
    } catch (...) {
        return false;
    }
}

int main(int argc, char* argv[]) {
    const size_t batch = 1;
    size_t seq = 1;
    size_t hidden_size = 128;
    size_t inter_size = 256;
    size_t num_experts = 64;
    size_t top_k = 2;
    const size_t group_size = 128;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            print_usage(argv[0], seq, hidden_size, inter_size, num_experts, top_k);
            return 0;
        }
        auto parse_opt = [&](const char* name, size_t& target) -> bool {
            const size_t name_len = std::strlen(name);
            if (std::strncmp(arg, name, name_len) != 0) {
                return false;
            }
            const char* value = nullptr;
            if (arg[name_len] == '=') {
                value = arg + name_len + 1;
            } else if (arg[name_len] == '\0') {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for " << name << "\n";
                    return true;
                }
                value = argv[++i];
            } else {
                return false;
            }
            if (!parse_size_t_arg(value, target)) {
                std::cerr << "Invalid value for " << name << ": " << value << "\n";
                return true;
            }
            return true;
        };

        if (parse_opt("--seq", seq) ||
            parse_opt("--hidden_size", hidden_size) ||
            parse_opt("--inter_size", inter_size) ||
            parse_opt("--num_experts", num_experts) ||
            parse_opt("--top_k", top_k)) {
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        print_usage(argv[0], seq, hidden_size, inter_size, num_experts, top_k);
        return 1;
    }

    if (seq == 0 || hidden_size == 0 || inter_size == 0 || num_experts == 0 || top_k == 0) {
        std::cerr << "seq, hidden_size, inter_size, num_experts, and top_k must be > 0\n";
        return 1;
    }
    if (top_k > num_experts) {
        std::cerr << "top_k must be <= num_experts\n";
        return 1;
    }

    if (hidden_size % group_size != 0 || inter_size % group_size != 0) {
        std::cerr << "hidden_size and inter_size must be divisible by group_size\n";
        return 1;
    }

    printf("Run moe3gemm test, seq_len: %d, num_experts: %d, hidden_size: %d, inter_size: %d, top_k: %d, group_size: %d\n",
           seq, num_experts, hidden_size, inter_size, top_k, group_size);

    const size_t tokens = batch * seq;
    const size_t gate_up_k = hidden_size;
    const size_t down_k = inter_size;

    auto hidden_states = random_f32(tokens * hidden_size, -0.5f, 0.5f);
    auto gate_inp = random_f32(num_experts * hidden_size, -0.5f, 0.5f);

    auto gate_w_f32 = random_f32(num_experts * inter_size * hidden_size, -0.5f, 0.5f);
    auto up_w_f32 = random_f32(num_experts * inter_size * hidden_size, -0.5f, 0.5f);
    auto down_w_f32 = random_f32(num_experts * hidden_size * inter_size, -0.5f, 0.5f);

    auto q_gate = quantize_q41(gate_w_f32, num_experts, inter_size, gate_up_k, group_size);
    auto q_up = quantize_q41(up_w_f32, num_experts, inter_size, gate_up_k, group_size);
    auto q_down = quantize_q41(down_w_f32, num_experts, hidden_size, down_k, group_size);

    auto gate_w_deq = dequantize_q41(q_gate, num_experts, inter_size, gate_up_k);
    auto up_w_deq = dequantize_q41(q_up, num_experts, inter_size, gate_up_k);
    auto down_w_deq = dequantize_q41(q_down, num_experts, hidden_size, down_k);

    auto print_deq_compare = [](const std::string& name,
                                const std::vector<float>& src,
                                const std::vector<float>& deq) {
        const size_t count = std::min(src.size(), deq.size());
        float max_abs_diff = 0.0f;
        float sum_abs_diff = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            const float diff = std::fabs(src[i] - deq[i]);
            max_abs_diff = std::max(max_abs_diff, diff);
            sum_abs_diff += diff;
        }
        const float mean_abs_diff = count ? (sum_abs_diff / static_cast<float>(count)) : 0.0f;
        std::cout << "\n" << name << " deq compare: size=" << count
                  << ", mean_abs_diff=" << mean_abs_diff
                  << ", max_abs_diff=" << max_abs_diff << "\n";
        const size_t print_n = std::min<size_t>(10, count);
        for (size_t i = 0; i < print_n; ++i) {
            std::cout << name << " i=" << i
                      << ", src=" << std::fixed << std::setprecision(6) << src[i]
                      << ", deq=" << std::fixed << std::setprecision(6) << deq[i]
                      << ", diff=" << std::fixed << std::setprecision(6)
                      << std::fabs(src[i] - deq[i]) << "\n";
        }
    };

    print_deq_compare("gate", gate_w_f32, gate_w_deq);
    print_deq_compare("up", up_w_f32, up_w_deq);
    print_deq_compare("down", down_w_f32, down_w_deq);

    auto ref = moe_cpu_ref(hidden_states, gate_inp, gate_w_deq, up_w_deq, down_w_deq,
                           batch, seq, hidden_size, inter_size, num_experts, top_k);
    auto ref_f32 = moe_cpu_ref(hidden_states, gate_inp, gate_w_f32, up_w_f32, down_w_f32,
                           batch, seq, hidden_size, inter_size, num_experts, top_k);

    ov::Shape hidden_shape{tokens, hidden_size};
    auto hidden_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, hidden_shape);
    auto hidden_f16 = std::make_shared<ov::op::v0::Convert>(hidden_param, ov::element::f16);
    ov::Shape gate_inp_shape{num_experts, hidden_size};
    auto gate_inp_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, gate_inp_shape, gate_inp);
    auto gate_inp_f16 = std::make_shared<ov::op::v0::Convert>(gate_inp_const, ov::element::f16);
    auto router_f16 = std::make_shared<ov::op::v0::MatMul>(hidden_f16, gate_inp_f16, false, true);

    auto w0 = std::make_shared<ov::op::v0::Constant>(q_gate.weights_u4);
    auto s0 = std::make_shared<ov::op::v0::Constant>(q_gate.scales_f16);
    auto z0 = std::make_shared<ov::op::v0::Constant>(q_gate.zps_u4);

    auto w1 = std::make_shared<ov::op::v0::Constant>(q_up.weights_u4);
    auto s1 = std::make_shared<ov::op::v0::Constant>(q_up.scales_f16);
    auto z1 = std::make_shared<ov::op::v0::Constant>(q_up.zps_u4);

    auto w2 = std::make_shared<ov::op::v0::Constant>(q_down.weights_u4);
    auto s2 = std::make_shared<ov::op::v0::Constant>(q_down.scales_f16);
    auto z2 = std::make_shared<ov::op::v0::Constant>(q_down.zps_u4);

    ov::op::internal::MOE3GemmFusedCompressed::Config config;
    config.hidden_size = hidden_size;
    config.inter_size = inter_size;
    config.num_expert = num_experts;
    config.top_k = top_k;
    config.group_size = group_size;
    // config.has_batch_dim = 1;
    config.out_type = ov::element::f16;

    ov::OutputVector args = {hidden_f16, router_f16, w0, s0, z0, w1, s1, z1, w2, s2, z2};
    auto moe = std::make_shared<ov::op::internal::MOE3GemmFusedCompressed>(args, config);
    auto moe_f32 = std::make_shared<ov::op::v0::Convert>(moe, ov::element::f32);
    auto result = std::make_shared<ov::op::v0::Result>(moe_f32);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{hidden_param},
                                             "moe_q41_moe3gemm_test");

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto req = compiled.create_infer_request();

    ov::Tensor hidden_tensor(ov::element::f32, hidden_shape);
    std::copy(hidden_states.begin(), hidden_states.end(), hidden_tensor.data<float>());

    req.set_input_tensor(0, hidden_tensor);
    req.infer();
    auto output = req.get_output_tensor(0);

    std::vector<float> actual(output.data<float>(), output.data<float>() + output.get_size());
    for (size_t i = 0; i < std::min<size_t>(10, actual.size()); ++i) {
        std::cout << "i: " << i
                  << ", gpu: " << std::fixed << std::setprecision(3) << actual[i]
                  << ", ref: " << std::fixed << std::setprecision(3) << ref[i]
                  << ", ref_f32: "<< std::fixed << std::setprecision(3) << ref_f32[i] << "\n";
    }

    const float tol = 1e-1f;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - ref[i]) > tol) {
            std::cerr << "Mismatch at " << i << ": gpu=" << actual[i] << " ref=" << ref[i] << "\n";
            return 1;
        }
    }

    std::cout << "Passed\n";
    return 0;
}
