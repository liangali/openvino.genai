// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"
#include "gguf_utils/rtn_quantize.hpp"

#include <openvino/core/except.hpp>
#include <openvino/core/type/float8_e4m3.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>

#include <iostream>
#include <chrono>
#include <iomanip>
#include <atomic>

namespace ov {
namespace genai {
namespace safetensors {

using namespace ov::genai::modeling::weights;

SafetensorsWeightFinalizer::SafetensorsWeightFinalizer() = default;

SafetensorsWeightFinalizer::SafetensorsWeightFinalizer(const QuantizationConfig& config)
    : selector_(config) {
    if (config.enabled()) {
        std::cout << "[SafetensorsWeightFinalizer] In-flight quantization enabled (NNCF-compatible):" << std::endl;
        
        // Helper lambda to print mode name
        auto mode_name = [](QuantizationConfig::Mode m) -> std::string {
            switch (m) {
                case QuantizationConfig::Mode::INT4_SYM: return "INT4_SYM";
                case QuantizationConfig::Mode::INT4_ASYM: return "INT4_ASYM";
                case QuantizationConfig::Mode::INT8_SYM: return "INT8_SYM";
                case QuantizationConfig::Mode::INT8_ASYM: return "INT8_ASYM";
                case QuantizationConfig::Mode::NONE: return "NONE";
                default: return "UNKNOWN";
            }
        };
        
        std::cout << "  Primary mode: " << mode_name(config.mode) 
                  << " (group_size=" << config.group_size << ")" << std::endl;
        std::cout << "  Backup mode: " << mode_name(config.backup_mode) 
                  << " (per-channel)" << std::endl;
        
        if (config.backup_mode == config.mode) {
            std::cout << "  -> All layers use same mode (" 
                      << mode_name(config.mode) << ", group_size=" << config.group_size << ")" << std::endl;
        } else if (config.backup_mode != QuantizationConfig::Mode::NONE) {
            std::cout << "  -> lm_head, embeddings will use backup mode (" 
                      << mode_name(config.backup_mode) << ", per-channel)" << std::endl;
        } else {
            std::cout << "  -> lm_head, embeddings will NOT be quantized (backup_mode=NONE)" << std::endl;
        }
        
        const auto& sel = config.selection;
        if (!sel.include_patterns.empty()) {
            std::cout << "  Include patterns: ";
            for (size_t i = 0; i < sel.include_patterns.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << sel.include_patterns[i];
            }
            std::cout << std::endl;
        }
        if (sel.layer_range.has_value()) {
            std::cout << "  Layer range: [" << sel.layer_range->first 
                      << ", " << sel.layer_range->second << "]" << std::endl;
        }
        if (sel.verbose) {
            std::cout << "  Verbose mode enabled" << std::endl;
        }
    }
}

SafetensorsWeightFinalizer::~SafetensorsWeightFinalizer() {
    if (total_weights_ > 0) {
        std::cout << "[SafetensorsWeightFinalizer] Statistics:" << std::endl;
        std::cout << "  Total weights processed: " << total_weights_ << std::endl;
        std::cout << "  Quantized weights: " << quantized_weights_ << std::endl;
        if (total_weights_ > 0) {
             float ratio = 100.0f * static_cast<float>(quantized_weights_) / static_cast<float>(total_weights_);
             std::cout << "  Quantization coverage: " << ratio << "%" << std::endl;
        }
        std::cout << "  Timing (ms):" << std::endl;
        std::cout << "    Fetch: " << std::fixed << std::setprecision(2) << total_fetch_time_ms_ << std::endl;
        std::cout << "    Quant: " << std::fixed << std::setprecision(2) << total_quant_time_ms_ << std::endl;
        std::cout << "    Graph: " << std::fixed << std::setprecision(2) << total_graph_time_ms_ << std::endl;
        std::cout << "    Total: " << std::fixed << std::setprecision(2) << (total_fetch_time_ms_ + total_quant_time_ms_ + total_graph_time_ms_) << std::endl;
    }
}

ov::genai::modeling::weights::FinalizedWeight SafetensorsWeightFinalizer::finalize(
    const std::string& name,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::OpContext& ctx) {
    if (!source.has(name)) {
        OPENVINO_THROW("Missing safetensors tensor: ", name);
    }

    // Check cache first
    const auto it_cached = cache_.find(name);
    if (it_cached != cache_.end()) {
        std::unordered_map<std::string, ov::genai::modeling::Tensor> empty_auxiliary;
        return ov::genai::modeling::weights::FinalizedWeight(
            ov::genai::modeling::Tensor(it_cached->second, &ctx),
            empty_auxiliary);
    }

    // Get tensor info for quantization check
    // Fetch tensor only once to avoid unnecessary caching
    auto start_fetch = std::chrono::high_resolution_clock::now();
    const ov::Tensor& tensor = source.get_tensor(name);
    auto end_fetch = std::chrono::high_resolution_clock::now();

    double fetch_ms = std::chrono::duration<double, std::milli>(end_fetch - start_fetch).count();
    total_fetch_time_ms_ += fetch_ms;

    ov::element::Type element_type = tensor.get_element_type();
    const ov::Shape& shape = tensor.get_shape();

    total_weights_++;

    // Check if in-flight quantization should be applied
    ov::Output<ov::Node> output;
    std::unordered_map<std::string, ov::genai::modeling::Tensor> auxiliary;
    
    double quant_ms = 0;
    double graph_ms = 0;
    bool logged = false;

    // Check if we should do in-flight quantization for this weight
    // For FP8 weights, we use direct FP8->INT4/INT8 quantization to save memory
    bool should_quantize_weight = selector_.enabled() && selector_.should_quantize(name, shape, element_type);
    
    if (element_type == ov::element::f8e4m3 && has_fp8_scale_inv(source, name)) {
        if (should_quantize_weight && shape.size() == 2) {
            // FP8 weight with quantization enabled: use direct FP8->INT4/INT8 quantization
            // This avoids creating intermediate F32 tensor, saving ~4x memory per weight
            quantized_weights_++;
            auto start_quant = std::chrono::high_resolution_clock::now();
            
            const ov::Tensor& scale_inv = source.get_tensor(name + "_scale_inv");
            auto quant_result = quantize_fp8_weight_direct(name, tensor, scale_inv, source, ctx);
            
            auto end_quant = std::chrono::high_resolution_clock::now();
            quant_ms = std::chrono::duration<double, std::milli>(end_quant - start_quant).count();
            total_quant_time_ms_ += quant_ms;
            
            auto start_graph = std::chrono::high_resolution_clock::now();
            // FP8 MoE models (like Qwen3-Next) now support MoE optimization with custom weight_loaders
            if (is_moe_weight(name)) {
                auto res = create_moe_subgraph(name, quant_result, shape, ctx);
                auto end_graph = std::chrono::high_resolution_clock::now();
                graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
                total_graph_time_ms_ += graph_ms;
                if (logged) {
                    std::cout << "[Weight: " << name << "] Fetch=" << std::fixed << std::setprecision(2) << fetch_ms 
                          << "ms, Quant=" << quant_ms << "ms, Graph=" << graph_ms << "ms" << std::endl;
                }
                return res;
            }
            output = create_dequant_subgraph(name, quant_result, shape);
            auto end_graph = std::chrono::high_resolution_clock::now();
            graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
            total_graph_time_ms_ += graph_ms;
        } else if (should_quantize_weight) {
            // FP8 weight with non-2D shape: fallback to F32 intermediate (less common)
            quantized_weights_++;
            auto start_quant = std::chrono::high_resolution_clock::now();
            
            const ov::Tensor& scale_inv = source.get_tensor(name + "_scale_inv");
            ov::Tensor f32_tensor = dequantize_fp8_to_f32(tensor, scale_inv);
            auto quant_result = quantize_weight(name, f32_tensor, source, ctx);
            // f32_tensor goes out of scope here and memory is released
            
            auto end_quant = std::chrono::high_resolution_clock::now();
            quant_ms = std::chrono::duration<double, std::milli>(end_quant - start_quant).count();
            total_quant_time_ms_ += quant_ms;
            
            auto start_graph = std::chrono::high_resolution_clock::now();
            // FP8 MoE weights also go through MoE subgraph for optimization
            if (is_moe_weight(name)) {
                auto res = create_moe_subgraph(name, quant_result, shape, ctx);
                auto end_graph = std::chrono::high_resolution_clock::now();
                graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
                total_graph_time_ms_ += graph_ms;
                if (logged) {
                    std::cout << "[Weight: " << name << "] Fetch=" << std::fixed << std::setprecision(2) << fetch_ms 
                          << "ms, Quant=" << quant_ms << "ms, Graph=" << graph_ms << "ms" << std::endl;
                }
                return res;
            }
            output = create_dequant_subgraph(name, quant_result, shape);
            auto end_graph = std::chrono::high_resolution_clock::now();
            graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
            total_graph_time_ms_ += graph_ms;
        } else {
            // FP8 weight without quantization: just dequantize FP8 to F32
            auto start_graph = std::chrono::high_resolution_clock::now();
            const ov::Tensor& scale_inv = source.get_tensor(name + "_scale_inv");
            output = create_fp8_dequant_subgraph(name, tensor, scale_inv, ctx);
            auto end_graph = std::chrono::high_resolution_clock::now();
            graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
            total_graph_time_ms_ += graph_ms;
        }
    } else if (should_quantize_weight) {
        quantized_weights_++;
        // Quantize the weight during finalization
        // Tensor reference is already fetched above, no need to fetch again
        auto start_quant = std::chrono::high_resolution_clock::now();
        auto quant_result = quantize_weight(name, tensor, source, ctx);
        auto end_quant = std::chrono::high_resolution_clock::now();
        quant_ms = std::chrono::duration<double, std::milli>(end_quant - start_quant).count();
        total_quant_time_ms_ += quant_ms;

        auto start_graph = std::chrono::high_resolution_clock::now();
        if (is_moe_weight(name)) {
            // For MoE weights, return FinalizedWeight with scales and zps in auxiliary
            auto res = create_moe_subgraph(name, quant_result, shape, ctx);
            auto end_graph = std::chrono::high_resolution_clock::now();
            graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
            total_graph_time_ms_ += graph_ms;
            if (logged) {
                std::cout << "[Weight: " << name << "] Fetch=" << std::fixed << std::setprecision(2) << fetch_ms 
                      << "ms, Quant=" << quant_ms << "ms, Graph=" << graph_ms << "ms" << std::endl;
            }
            return res;
        } else {
            output = create_dequant_subgraph(name, quant_result, shape);
        }
        auto end_graph = std::chrono::high_resolution_clock::now();
        graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
        total_graph_time_ms_ += graph_ms;
        logged = false;
    } else {
        auto start_graph = std::chrono::high_resolution_clock::now();
        // Only create Constant when NOT quantizing
        // This avoids keeping the original tensor in memory when we compress it
        std::shared_ptr<ov::op::v0::Constant> constant = std::make_shared<ov::op::v0::Constant>(tensor);
        constant->set_friendly_name(name);
        constant->output(0).set_names({name});

        // Convert to F32 if needed (matching GGUF behavior)
        // This ensures consistency with how GGUF weights are processed
        if (element_type == ov::element::bf16 || 
            element_type == ov::element::f16 ||
            element_type == ov::element::f8e4m3) {
            auto converted = std::make_shared<ov::op::v0::Convert>(constant, ov::element::f32);
            output = converted->output(0);
        } else {
            output = constant->output(0);
        }
        auto end_graph = std::chrono::high_resolution_clock::now();
        graph_ms = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
        total_graph_time_ms_ += graph_ms;
    }
    
    if (logged) {
        std::cout << "[Weight: " << name << "] Fetch=" << std::fixed << std::setprecision(2) << fetch_ms 
                  << "ms, Quant=" << quant_ms << "ms, Graph=" << graph_ms << "ms" << std::endl;
    }
    
    cache_.emplace(name, output);

    return ov::genai::modeling::weights::FinalizedWeight(
        ov::genai::modeling::Tensor(output, &ctx),
        auxiliary);
}

bool SafetensorsWeightFinalizer::has_fp8_scale_inv(ov::genai::modeling::weights::WeightSource& source,
                                                    const std::string& name) const {
    static const std::string weight_suffix = ".weight";
    if (name.size() < weight_suffix.size() ||
        name.compare(name.size() - weight_suffix.size(), weight_suffix.size(), weight_suffix) != 0) {
        return false;
    }
    return source.has(name + "_scale_inv");
}

ov::Tensor SafetensorsWeightFinalizer::dequantize_fp8_to_f32(
    const ov::Tensor& weight_fp8,
    const ov::Tensor& scale_inv) const {
    // Get shapes
    const ov::Shape& weight_shape = weight_fp8.get_shape();
    const ov::Shape& scale_shape = scale_inv.get_shape();
    
    // Calculate total elements
    size_t total_elements = 1;
    for (auto dim : weight_shape) {
        total_elements *= dim;
    }
    
    // Create output F32 tensor
    ov::Tensor result(ov::element::f32, weight_shape);
    float* out_ptr = result.data<float>();
    
    // Get input pointers - use raw void* for FP8 since data<uint8_t>() is not allowed
    const uint8_t* fp8_ptr = static_cast<const uint8_t*>(weight_fp8.data());
    const float* scale_ptr = scale_inv.data<float>();
    
    // Determine block size for scale expansion
    // For per-tensor scale: scale_shape == {1} or {1, 1}
    // For block-wise scale: scale_shape[i] < weight_shape[i]
    
    if (scale_shape.size() == weight_shape.size()) {
        // Block-wise or per-channel scaling
        if (weight_shape.size() == 2) {
            // 2D weight: [out_features, in_features]
            // Scale shape: [out_features, num_groups] or [out_features, 1]
            size_t out_features = weight_shape[0];
            size_t in_features = weight_shape[1];
            size_t num_groups = scale_shape[1];
            size_t group_size = in_features / num_groups;
            
            for (size_t o = 0; o < out_features; ++o) {
                for (size_t i = 0; i < in_features; ++i) {
                    size_t group_idx = i / group_size;
                    size_t fp8_idx = o * in_features + i;
                    size_t scale_idx = o * num_groups + group_idx;
                    
                    // FP8 E4M3 to F32 conversion using from_bits()
                    uint8_t fp8_val = fp8_ptr[fp8_idx];
                    float f32_val = static_cast<float>(ov::float8_e4m3::from_bits(fp8_val));
                    out_ptr[fp8_idx] = f32_val * scale_ptr[scale_idx];
                }
            }
        } else if (weight_shape.size() == 1) {
            // 1D weight: [features]
            size_t features = weight_shape[0];
            size_t num_groups = scale_shape[0];
            size_t group_size = features / num_groups;
            
            for (size_t i = 0; i < features; ++i) {
                size_t group_idx = i / group_size;
                uint8_t fp8_val = fp8_ptr[i];
                float f32_val = static_cast<float>(ov::float8_e4m3::from_bits(fp8_val));
                out_ptr[i] = f32_val * scale_ptr[group_idx];
            }
        } else {
            // Generic N-D case: assume last dim is grouped
            // For simplicity, flatten and apply per-tensor or approximate
            float scale = scale_ptr[0];
            for (size_t i = 0; i < total_elements; ++i) {
                uint8_t fp8_val = fp8_ptr[i];
                float f32_val = static_cast<float>(ov::float8_e4m3::from_bits(fp8_val));
                out_ptr[i] = f32_val * scale;
            }
        }
    } else {
        // Per-tensor scaling (scale is scalar or broadcastable)
        float scale = scale_ptr[0];
        for (size_t i = 0; i < total_elements; ++i) {
            uint8_t fp8_val = fp8_ptr[i];
            float f32_val = static_cast<float>(ov::float8_e4m3::from_bits(fp8_val));
            out_ptr[i] = f32_val * scale;
        }
    }
    
    return result;
}

ov::Output<ov::Node> SafetensorsWeightFinalizer::create_fp8_dequant_subgraph(
    const std::string& name,
    const ov::Tensor& weight_fp8,
    const ov::Tensor& scale_inv,
    ov::genai::modeling::OpContext& ctx) {
    auto weight_const = std::make_shared<ov::op::v0::Constant>(weight_fp8);
    weight_const->set_friendly_name(name);
    weight_const->output(0).set_names({name});

    auto weight_f32 = std::make_shared<ov::op::v0::Convert>(weight_const, ov::element::f32);
    auto expanded_scale = expand_block_scale_to_weight(scale_inv, weight_fp8.get_shape(), ctx);
    auto scale_f32 = std::make_shared<ov::op::v0::Convert>(expanded_scale, ov::element::f32);
    auto dequant = std::make_shared<ov::op::v1::Multiply>(
        weight_f32->output(0),
        scale_f32->output(0),
        ov::op::AutoBroadcastType::NUMPY);
    return dequant->output(0);
}

ov::Output<ov::Node> SafetensorsWeightFinalizer::expand_block_scale_to_weight(
    const ov::Tensor& scale_inv,
    const ov::Shape& weight_shape,
    ov::genai::modeling::OpContext& ctx) const {
    auto scale_const = std::make_shared<ov::op::v0::Constant>(scale_inv);
    const ov::Shape scale_shape = scale_inv.get_shape();

    if (scale_shape == weight_shape) {
        return scale_const->output(0);
    }

    OPENVINO_ASSERT(scale_shape.size() == weight_shape.size(),
                    "FP8 scale rank must match weight rank. scale_shape=",
                    scale_shape,
                    ", weight_shape=",
                    weight_shape);

    auto interleaved_scale_shape = make_interleaved_scale_shape(scale_shape);
    auto interleaved_repeats = make_interleaved_tile_repeats(scale_shape, weight_shape);

    auto reshape_scale = std::make_shared<ov::op::v1::Reshape>(
        scale_const->output(0),
        ctx.const_i64_vec(interleaved_scale_shape),
        false);
    auto tiled = std::make_shared<ov::op::v0::Tile>(
        reshape_scale->output(0),
        ctx.const_i64_vec(interleaved_repeats));

    std::vector<int64_t> weight_shape_i64(weight_shape.begin(), weight_shape.end());
    auto final_reshape = std::make_shared<ov::op::v1::Reshape>(
        tiled->output(0),
        ctx.const_i64_vec(weight_shape_i64),
        false);
    return final_reshape->output(0);
}

std::vector<int64_t> SafetensorsWeightFinalizer::make_interleaved_scale_shape(const ov::Shape& scale_shape) const {
    std::vector<int64_t> result;
    result.reserve(scale_shape.size() * 2);
    for (size_t i = 0; i < scale_shape.size(); ++i) {
        result.push_back(static_cast<int64_t>(scale_shape[i]));
        result.push_back(1);
    }
    return result;
}

std::vector<int64_t> SafetensorsWeightFinalizer::make_interleaved_tile_repeats(const ov::Shape& scale_shape,
                                                                                const ov::Shape& weight_shape) const {
    OPENVINO_ASSERT(scale_shape.size() == weight_shape.size(),
                    "Scale and weight rank mismatch while expanding FP8 scales.");
    std::vector<int64_t> repeats;
    repeats.reserve(scale_shape.size() * 2);
    for (size_t i = 0; i < scale_shape.size(); ++i) {
        const auto s = scale_shape[i];
        const auto w = weight_shape[i];
        OPENVINO_ASSERT(s > 0 && w > 0, "Scale/weight dimensions must be > 0 for FP8 block expansion.");
        OPENVINO_ASSERT(w % s == 0,
                        "Weight dimension must be divisible by scale dimension for FP8 block expansion. dim ",
                        i,
                        ": weight=",
                        w,
                        ", scale=",
                        s);
        repeats.push_back(1);
        repeats.push_back(static_cast<int64_t>(w / s));
    }
    return repeats;
}

rtn::QuantizedWeight SafetensorsWeightFinalizer::quantize_weight(
    const std::string& name,
    const ov::Tensor& tensor,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::OpContext& ctx) {
    
    using namespace ov::genai::rtn;
    using Mode = QuantizationConfig::Mode;
    
    // NNCF-style: Get the quantization mode for this specific weight
    // This may return different modes for different layers (e.g., INT8 for lm_head, INT4 for others)
    Mode quant_mode = selector_.get_quantization_mode(name, tensor.get_shape(), tensor.get_element_type());
    int group_size = selector_.get_group_size(name);
    
    // RTN quantization functions expect 2D tensors
    OPENVINO_ASSERT(tensor.get_shape().size() == 2 || tensor.get_shape().size() == 3 || tensor.get_shape().size() == 1, 
                    "Only 1D, 2D and 3D tensors can be quantized, got shape: ", tensor.get_shape());
    
    // Quantize using RTN algorithms
    rtn::QuantizedWeight quant_result;
    
    switch (quant_mode) {
        case Mode::INT4_SYM:
            quant_result = rtn::quantize_int4_sym(tensor, group_size);
            break;
            
        case Mode::INT4_ASYM:
            quant_result = rtn::quantize_int4_asym(tensor, group_size);
            break;
            
        case Mode::INT8_SYM:
            quant_result = rtn::quantize_int8_sym(tensor, group_size);
            break;
            
        case Mode::INT8_ASYM:
            quant_result = rtn::quantize_int8_asym(tensor, group_size);
            break;
            
        default:
            OPENVINO_THROW("Unsupported quantization mode");
    }
    
    return quant_result;
}

rtn::QuantizedWeight SafetensorsWeightFinalizer::quantize_fp8_weight_direct(
    const std::string& name,
    const ov::Tensor& fp8_tensor,
    const ov::Tensor& scale_inv,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::OpContext& ctx) {
    
    using namespace ov::genai::rtn;
    using Mode = QuantizationConfig::Mode;
    
    // Get quantization mode and group size for this weight
    Mode quant_mode = selector_.get_quantization_mode(name, fp8_tensor.get_shape(), fp8_tensor.get_element_type());
    int group_size = selector_.get_group_size(name);
    
    // Direct FP8 -> INT4/INT8 quantization without intermediate F32 tensor
    // This saves ~4x memory compared to creating a full F32 tensor
    rtn::QuantizedWeight quant_result;
    
    switch (quant_mode) {
        case Mode::INT4_SYM:
            quant_result = rtn::quantize_fp8_to_int4_sym(fp8_tensor, scale_inv, group_size);
            break;
            
        case Mode::INT4_ASYM:
            quant_result = rtn::quantize_fp8_to_int4_asym(fp8_tensor, scale_inv, group_size);
            break;
            
        case Mode::INT8_SYM:
            quant_result = rtn::quantize_fp8_to_int8_sym(fp8_tensor, scale_inv, group_size);
            break;
            
        case Mode::INT8_ASYM:
            quant_result = rtn::quantize_fp8_to_int8_asym(fp8_tensor, scale_inv, group_size);
            break;
            
        default:
            OPENVINO_THROW("Unsupported quantization mode for FP8 direct quantization");
    }
    
    return quant_result;
}

ov::Output<ov::Node> SafetensorsWeightFinalizer::create_dequant_subgraph(
    const std::string& name,
    const rtn::QuantizedWeight& quant_result,
    const ov::Shape& original_shape) {

    // Support 1D [in], 2D [out, in] and 3D [batch, out, in] weights
    OPENVINO_ASSERT(original_shape.size() == 2 || original_shape.size() == 3 || original_shape.size() == 1,
                    "Original shape must be 1D, 2D or 3D for dequantization");

    // Create scale constant
    auto scale_const = std::make_shared<ov::op::v0::Constant>(quant_result.scale);
    scale_const->set_friendly_name(name + "_scale");

    const ov::Shape& scale_shape = scale_const->get_shape();
    const ov::Shape& compressed_shape = quant_result.compressed.get_shape();

    // Determine element type based on compressed type from result
    ov::element::Type elem_type = quant_result.compressed_type;
    bool has_zero_point = quant_result.has_zero_point && quant_result.zero_point.get_size() > 0;

    // RTN quantization flattens 3D [batch, out, in] to 2D [batch*out, in] for processing,
    // but output shapes are preserved as 3D [batch, out, packed_in/num_groups].
    // We need to handle this correctly:
    // - For 3D original: flatten to 2D for dequant, then reshape back to 3D
    // - For 2D/1D: process directly

    size_t in_features = original_shape.back();
    size_t num_groups = scale_shape.back();

    // Calculate flattened out_features (for 3D, this is batch*out)
    size_t out_features = 1;
    for (size_t i = 0; i < original_shape.size() - 1; ++i) {
        out_features *= original_shape[i];
    }

    // Get the actual group_size used during quantization to detect non-divisible cases.
    // When in_features is not evenly divisible by group_size (e.g., 4304 % 128 = 80),
    // we cannot use a uniform 3D [out, num_groups, group_size] layout because
    // num_groups * computed_group_size != in_features. Instead, we use a flat layout
    // with Gather-based scale/zero_point expansion.
    int configured_gs = selector_.get_group_size(name);
    if (configured_gs <= 0) {
        configured_gs = static_cast<int>(in_features);
    }
    bool evenly_divisible = (in_features % static_cast<size_t>(configured_gs) == 0);

    if (!evenly_divisible && num_groups > 1) {
        // Non-divisible case: use flat dequantization with Gather-based scale expansion.
        // The compressed data is stored contiguously as [out_features, in_features] elements.
        // We expand scales/zero_points from [out_features, num_groups] to
        // [out_features, in_features] using Gather with group index mapping.

        const void* data_ptr = quant_result.compressed.data();
        auto flat_const = std::make_shared<ov::op::v0::Constant>(
            elem_type, ov::Shape{out_features, in_features}, data_ptr);
        flat_const->set_friendly_name(name + "_compressed");
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(flat_const, ov::element::f16);

        // Flatten scale to 2D [out_features, num_groups] if it's 3D
        std::shared_ptr<ov::Node> scale_2d;
        if (scale_shape.size() == 3) {
            auto flatten_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{2},
                std::vector<int64_t>{static_cast<int64_t>(out_features),
                                     static_cast<int64_t>(num_groups)});
            scale_2d = std::make_shared<ov::op::v1::Reshape>(scale_const, flatten_shape, false);
        } else {
            scale_2d = scale_const;
        }

        // Build group index mapping: indices[i] = i / configured_group_size
        // This maps each element position to its quantization group.
        size_t gs = static_cast<size_t>(configured_gs);
        std::vector<int32_t> group_indices(in_features);
        for (size_t i = 0; i < in_features; ++i) {
            group_indices[i] = static_cast<int32_t>(i / gs);
        }
        auto indices_const = std::make_shared<ov::op::v0::Constant>(
            ov::element::i32, ov::Shape{in_features}, group_indices);
        auto axis_const = std::make_shared<ov::op::v0::Constant>(
            ov::element::i32, ov::Shape{}, 1);

        // Expand scale: [out_features, num_groups] -> [out_features, in_features]
        auto scale_expanded = std::make_shared<ov::op::v8::Gather>(
            scale_2d, indices_const, axis_const);

        std::shared_ptr<ov::Node> dequantized;
        if (has_zero_point) {
            auto zp_const = std::make_shared<ov::op::v0::Constant>(quant_result.zero_point);
            zp_const->set_friendly_name(name + "_zp");

            std::shared_ptr<ov::Node> zp_2d;
            if (quant_result.zero_point.get_shape().size() == 3) {
                auto flatten_shape = std::make_shared<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{2},
                    std::vector<int64_t>{static_cast<int64_t>(out_features),
                                         static_cast<int64_t>(num_groups)});
                zp_2d = std::make_shared<ov::op::v1::Reshape>(zp_const, flatten_shape, false);
            } else {
                zp_2d = zp_const;
            }

            // Expand zero_point: [out_features, num_groups] -> [out_features, in_features]
            auto zp_expanded = std::make_shared<ov::op::v8::Gather>(
                zp_2d, indices_const, axis_const);
            auto zp_f16 = std::make_shared<ov::op::v0::Convert>(zp_expanded, ov::element::f16);
            auto shifted = std::make_shared<ov::op::v1::Subtract>(
                weights_f16, zp_f16, ov::op::AutoBroadcastType::NUMPY);
            dequantized = std::make_shared<ov::op::v1::Multiply>(
                shifted, scale_expanded, ov::op::AutoBroadcastType::NUMPY);
        } else {
            dequantized = std::make_shared<ov::op::v1::Multiply>(
                weights_f16, scale_expanded, ov::op::AutoBroadcastType::NUMPY);
        }

        // Reshape to original shape (needed for 1D and 3D; no-op for 2D)
        auto final_shape = std::make_shared<ov::op::v0::Constant>(
            ov::element::i64,
            ov::Shape{original_shape.size()},
            std::vector<int64_t>(original_shape.begin(), original_shape.end()));
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(dequantized, final_shape, false);
        return std::make_shared<ov::op::v0::Convert>(reshaped, ov::element::f32);
    }

    // Evenly-divisible case: use 3D grouped layout [out_features, num_groups, group_size]
    size_t group_size = in_features / num_groups;

    // For dequantization, we work with 2D layout: [out_features, num_groups, group_size]
    // This matches how RTN processed the data internally
    ov::Shape grouped_shape_2d = {out_features, num_groups, group_size};

    // Create compressed constant with 2D grouped shape
    const void* data_ptr = quant_result.compressed.data();
    auto grouped_const = std::make_shared<ov::op::v0::Constant>(elem_type, grouped_shape_2d, data_ptr);
    grouped_const->set_friendly_name(name + "_compressed");

    // Convert to F16 for arithmetic operations
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(grouped_const, ov::element::f16);

    // Reshape scale to 2D [out_features, num_groups] first if it's 3D
    std::shared_ptr<ov::Node> scale_2d;
    if (scale_shape.size() == 3) {
        // Flatten from [batch, out, num_groups] to [batch*out, num_groups]
        auto flatten_shape = std::make_shared<ov::op::v0::Constant>(
            ov::element::i64,
            ov::Shape{2},
            std::vector<int64_t>{static_cast<int64_t>(out_features), static_cast<int64_t>(num_groups)}
        );
        scale_2d = std::make_shared<ov::op::v1::Reshape>(scale_const, flatten_shape, false);
    } else {
        scale_2d = scale_const;
    }

    // Build scale broadcast shape: [out_features, num_groups, 1]
    ov::Shape scale_broadcast_shape = {out_features, num_groups, 1};

    auto scale_reshape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64,
        ov::Shape{scale_broadcast_shape.size()},
        std::vector<int64_t>(scale_broadcast_shape.begin(), scale_broadcast_shape.end())
    );
    auto scale_reshaped = std::make_shared<ov::op::v1::Reshape>(scale_2d, scale_reshape, false);

    // Apply dequantization formula
    std::shared_ptr<ov::Node> dequantized;
    if (has_zero_point) {
        // Asymmetric: weight_fp = (weight - zero_point) * scale
        auto zp_const = std::make_shared<ov::op::v0::Constant>(quant_result.zero_point);
        zp_const->set_friendly_name(name + "_zp");

        // Flatten zero_point if 3D
        std::shared_ptr<ov::Node> zp_2d;
        if (quant_result.zero_point.get_shape().size() == 3) {
            auto flatten_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64,
                ov::Shape{2},
                std::vector<int64_t>{static_cast<int64_t>(out_features), static_cast<int64_t>(num_groups)}
            );
            zp_2d = std::make_shared<ov::op::v1::Reshape>(zp_const, flatten_shape, false);
        } else {
            zp_2d = zp_const;
        }

        auto zp_reshaped = std::make_shared<ov::op::v1::Reshape>(zp_2d, scale_reshape, false);
        auto zp_f16 = std::make_shared<ov::op::v0::Convert>(zp_reshaped, ov::element::f16);
        auto shifted = std::make_shared<ov::op::v1::Subtract>(
            weights_f16, zp_f16, ov::op::AutoBroadcastType::NUMPY);
        dequantized = std::make_shared<ov::op::v1::Multiply>(
            shifted, scale_reshaped, ov::op::AutoBroadcastType::NUMPY);
    } else {
        // Symmetric: weight_fp = weight * scale
        dequantized = std::make_shared<ov::op::v1::Multiply>(
            weights_f16, scale_reshaped, ov::op::AutoBroadcastType::NUMPY);
    }

    // Reshape back to original shape
    auto final_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64,
        ov::Shape{original_shape.size()},
        std::vector<int64_t>(original_shape.begin(), original_shape.end())
    );
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(dequantized, final_shape, false);

    // Convert to F32
    return std::make_shared<ov::op::v0::Convert>(reshaped, ov::element::f32);
}

bool SafetensorsWeightFinalizer::is_moe_weight(const std::string& name) const {
    // Match MoE weight patterns that need special handling (return INT4 + scales + zps)
    //
    // Pattern 1: Qwen3-MoE per-expert weights (not yet fused):
    // - model.layers[X].mlp.experts.{i}.gate_proj.weight
    // - model.layers[X].mlp.experts.{i}.up_proj.weight  
    // - model.layers[X].mlp.experts.{i}.down_proj.weight
    //
    // Pattern 1b: Qwen3.5-MoE packed expert weights:
    // - model.layers[X].mlp.experts.gate_up_proj
    // - model.layers[X].mlp.experts.down_proj
    //
    // Pattern 2: Pre-fused MoE weights (GGUF-style):
    // - model.layers[X].moe.gate_exps.weight
    // - model.layers[X].moe.up_exps.weight
    // - model.layers[X].moe.down_exps.weight
    //
    // These all need create_moe_subgraph to return INT4 compressed + auxiliary scales/zps
    
    // Pattern 1: Per-expert weights (Qwen3-MoE native format)
    if (name.find(".mlp.experts.") != std::string::npos) {
        // Check if it's an expert projection weight
        if (name.find("gate_proj.weight") != std::string::npos ||
            name.find("up_proj.weight") != std::string::npos ||
            name.find("down_proj.weight") != std::string::npos) {
            return true;
        }

        // Packed format used by Qwen3.5-MoE (gate+up fused by expert).
        if (name.find(".mlp.experts.gate_up_proj") != std::string::npos ||
            name.find(".mlp.experts.down_proj") != std::string::npos) {
            return true;
        }
    }

    // Pattern 1c: Qwen3.5-MoE Shared Expert weights
    // These also need exposed scales/zps for the fused kernel
    if (name.find(".mlp.shared_expert.") != std::string::npos) {
        if (name.find("gate_proj.weight") != std::string::npos ||
            name.find("up_proj.weight") != std::string::npos ||
            name.find("down_proj.weight") != std::string::npos) {
            return true;
        }
    }
    
    // Pattern 2: Pre-fused weights  
    return name.find("moe.gate_exps") != std::string::npos ||
           name.find("moe.up_exps") != std::string::npos ||
           name.find("moe.down_exps") != std::string::npos;
}

ov::genai::modeling::weights::FinalizedWeight SafetensorsWeightFinalizer::create_moe_subgraph(
    const std::string& name,
    const rtn::QuantizedWeight& quant_result,
    const ov::Shape& original_shape,
    ov::genai::modeling::OpContext& ctx) {
    
    // MoE weights can be:
    // - 2D per-expert: [out_features, in_features] - Qwen3-MoE native format
    // - 3D pre-fused: [num_experts, out_features, in_features] - GGUF-style
    // Qwen3-MoE expects INT4 compressed weights + scales + zps in auxiliary
    OPENVINO_ASSERT(original_shape.size() == 2 || original_shape.size() == 3, 
                    "MoE original shape must be 2D or 3D, got: ", original_shape.size());
    
    // Determine element type based on compressed type
    ov::element::Type compressed_type = quant_result.compressed.get_element_type();
    bool has_zero_point = quant_result.has_zero_point && quant_result.zero_point.get_size() > 0;
    ov::element::Type elem_type;
    
    if (compressed_type == ov::element::u8) {
        // INT4 packed in U8: use u4 for asymmetric, i4 for symmetric
        elem_type = has_zero_point ? ov::element::u4 : ov::element::i4;
    } else if (compressed_type == ov::element::u4) {
        elem_type = ov::element::u4;
    } else if (compressed_type == ov::element::i4) {
        elem_type = ov::element::i4;
    } else if (compressed_type == ov::element::i8) {
        elem_type = ov::element::i8;
    } else {
        OPENVINO_THROW("Unsupported compressed type for MoE dequantization: ", compressed_type);
    }
    
    ov::Shape scale_shape = quant_result.scale.get_shape();
    
    // Handle both 2D (per-expert) and 3D (fused) shapes
    size_t out_features, in_features, num_groups, group_size;
    ov::Shape packed_shape;
    
    // Static counters to print summary only once per type
    static std::atomic<int> moe_2d_count{0};
    static std::atomic<int> moe_3d_count{0};
    
    if (original_shape.size() == 2) {
        // Per-expert 2D shape: [out_features, in_features]
        out_features = original_shape[0];
        in_features = original_shape[1];
        
        // Scale shape for 2D: [out_features, num_groups]
        OPENVINO_ASSERT(scale_shape.size() == 2, "Scale shape must be 2D for per-expert weights");
        num_groups = scale_shape[1];
        group_size = in_features / num_groups;
        
        // Packed shape: [out_features, num_groups, group_size]
        packed_shape = ov::Shape{out_features, num_groups, group_size};
        
        // Print only once for 2D MoE weights
        if (moe_2d_count.fetch_add(1) == 0) {
            std::cout << "[MoE] First 2D per-expert weight: " << name 
                      << ", shape=" << original_shape << ", type=" << elem_type << std::endl;
        }
    } else {
        // Pre-fused 3D shape: [num_experts, out_features, in_features]
        size_t num_experts = original_shape[0];
        out_features = original_shape[1];
        in_features = original_shape[2];
        
        // Scale shape for 3D: [num_experts, out_features, num_groups]
        OPENVINO_ASSERT(scale_shape.size() == 3, "Scale shape must be 3D for fused weights");
        num_groups = scale_shape[2];
        group_size = in_features / num_groups;
        
        // Packed shape: [num_experts, out_features, num_groups, group_size]
        packed_shape = ov::Shape{num_experts, out_features, num_groups, group_size};
        
        // Print only once for 3D MoE weights
        if (moe_3d_count.fetch_add(1) == 0) {
            std::cout << "[MoE] First 3D fused weight: " << name 
                      << ", shape=" << original_shape << ", type=" << elem_type << std::endl;
        }
    }

    const void* data_ptr = quant_result.compressed.data();
    auto weight_compressed = std::make_shared<ov::op::v0::Constant>(elem_type, packed_shape, data_ptr);
    weight_compressed->set_friendly_name(name + "_compressed");

    // Create scale constant 
    auto scale_const = std::make_shared<ov::op::v0::Constant>(quant_result.scale);
    scale_const->set_friendly_name(name + "_scale");

    // Handle zero-point: pack from u8 to u4 if needed
    std::shared_ptr<ov::op::v0::Constant> zp_const;
    if (has_zero_point) {
        ov::element::Type zp_type = quant_result.zero_point.get_element_type();
        if (zp_type == ov::element::u8 && elem_type == ov::element::u4) {
            // Zero-points need to be packed from u8 to u4 nibbles
            ov::Tensor zero_point_tensor(ov::element::u4, scale_shape);
            uint8_t* zero_point_data = static_cast<uint8_t*>(zero_point_tensor.data());
            const uint8_t* zp_data = quant_result.zero_point.data<uint8_t>();
            for (size_t i = 0; i < zero_point_tensor.get_byte_size(); ++i) {
                uint8_t bias1 = zp_data[i * 2];
                uint8_t bias2 = zp_data[i * 2 + 1];
                zero_point_data[i] = (bias2 << 4) | (bias1 & 0x0F);
            }
            zp_const = std::make_shared<ov::op::v0::Constant>(zero_point_tensor);
        } else {
            zp_const = std::make_shared<ov::op::v0::Constant>(quant_result.zero_point);
        }
        zp_const->set_friendly_name(name + "_zp");
    } else {
        OPENVINO_THROW("MoE weights require zero points for asymmetric quantization");
    }

    // Build auxiliary map with scales and zero-points
    std::unordered_map<std::string, ov::genai::modeling::Tensor> auxiliary;
    
    if (original_shape.size() == 2) {
        // For 2D weights, transpose from [O, G] to [G, O]
        auto transpose_order = std::make_shared<ov::op::v0::Constant>(
            ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0});
        
        // Create Transpose and immediately fold to Const for XML optimization
        auto scales_transposed = std::make_shared<ov::op::v1::Transpose>(scale_const, transpose_order);
        ov::OutputVector scales_outputs(1);
        OPENVINO_ASSERT(scales_transposed->constant_fold(scales_outputs, {scale_const, transpose_order}),
                        "Failed to constant fold scales transpose for ", name);
        auto scales_folded = std::dynamic_pointer_cast<ov::op::v0::Constant>(scales_outputs[0].get_node_shared_ptr());
        OPENVINO_ASSERT(scales_folded, "Scales constant fold did not produce Constant node for ", name);
        scales_folded->set_friendly_name(name + "_scales");
        
        auto zero_points_transposed = std::make_shared<ov::op::v1::Transpose>(zp_const, transpose_order);
        ov::OutputVector zps_outputs(1);
        OPENVINO_ASSERT(zero_points_transposed->constant_fold(zps_outputs, {zp_const, transpose_order}),
                        "Failed to constant fold zero points transpose for ", name);
        auto zps_folded = std::dynamic_pointer_cast<ov::op::v0::Constant>(zps_outputs[0].get_node_shared_ptr());
        OPENVINO_ASSERT(zps_folded, "Zero points constant fold did not produce Constant node for ", name);
        zps_folded->set_friendly_name(name + "_zps");
        
        auxiliary["scales"] = ov::genai::modeling::Tensor(scales_folded, &ctx);
        auxiliary["zps"] = ov::genai::modeling::Tensor(zps_folded, &ctx);
    } else {
        // For 3D weights, transpose from [E, O, G] to [E, G, O]
        auto transpose_order = std::make_shared<ov::op::v0::Constant>(
            ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
        
        // Create Transpose and immediately fold to Const for XML optimization
        auto scales_transposed = std::make_shared<ov::op::v1::Transpose>(scale_const, transpose_order);
        ov::OutputVector scales_outputs(1);
        OPENVINO_ASSERT(scales_transposed->constant_fold(scales_outputs, {scale_const, transpose_order}),
                        "Failed to constant fold scales transpose for ", name);
        auto scales_folded = std::dynamic_pointer_cast<ov::op::v0::Constant>(scales_outputs[0].get_node_shared_ptr());
        OPENVINO_ASSERT(scales_folded, "Scales constant fold did not produce Constant node for ", name);
        scales_folded->set_friendly_name(name + "_scales");
        
        auto zero_points_transposed = std::make_shared<ov::op::v1::Transpose>(zp_const, transpose_order);
        ov::OutputVector zps_outputs(1);
        OPENVINO_ASSERT(zero_points_transposed->constant_fold(zps_outputs, {zp_const, transpose_order}),
                        "Failed to constant fold zero points transpose for ", name);
        auto zps_folded = std::dynamic_pointer_cast<ov::op::v0::Constant>(zps_outputs[0].get_node_shared_ptr());
        OPENVINO_ASSERT(zps_folded, "Zero points constant fold did not produce Constant node for ", name);
        zps_folded->set_friendly_name(name + "_zps");
        
        auxiliary["scales"] = ov::genai::modeling::Tensor(scales_folded, &ctx);
        auxiliary["zps"] = ov::genai::modeling::Tensor(zps_folded, &ctx);
    }

    // Cache the compressed weight
    cache_.emplace(name, weight_compressed);

    return ov::genai::modeling::weights::FinalizedWeight(
        ov::genai::modeling::Tensor(weight_compressed, &ctx),
        auxiliary);
}

}  // namespace safetensors
}  // namespace genai
}  // namespace ov
