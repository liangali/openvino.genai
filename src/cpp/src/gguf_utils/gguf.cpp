// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils/gguf.hpp"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <openvino/core/parallel.hpp>
#include <cfloat>

#ifndef MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// https://github.com/antirez/gguf-tools/blob/af7d88d808a7608a33723fba067036202910acb3/gguflib.h#L102-L108
constexpr int gguf_array_header_size = 12;

template <typename... Args>
std::string format(std::string fmt, Args... args) {
    size_t bufferSize = 1000;
    char* buffer = new char[bufferSize];
    int n = sprintf(buffer, fmt.c_str(), args...);
    assert(n >= 0 && n < (int)bufferSize - 1);

    std::string fmtStr(buffer);
    delete[] buffer;
    return fmtStr;
}

std::optional<uint32_t> dtype_to_gguf_tensor_type(const ov::element::Type& dtype) {
    switch (dtype) {
    case ov::element::f64:
        return GGUF_TYPE_F64;
    case ov::element::f32:
        return GGUF_TYPE_F32;
    case ov::element::f16:
        return GGUF_TYPE_F16;
    case ov::element::bf16:
        return GGUF_TYPE_BF16;
    case ov::element::i8:
        return GGUF_TYPE_I8;
    case ov::element::i16:
        return GGUF_TYPE_I16;
    case ov::element::i32:
        return GGUF_TYPE_I32;
    case ov::element::i64:
        return GGUF_TYPE_I64;
    default:
        return std::nullopt;
    }
}

std::optional<ov::element::Type> gguf_type_to_dtype(const uint32_t& gguf_type) {
    switch (gguf_type) {
    case GGUF_TYPE_F64:
        return ov::element::f64;
    case GGUF_TYPE_F32:
        return ov::element::f32;
    case GGUF_TYPE_F16:
        return ov::element::f16;
    case GGUF_TYPE_BF16:
        return ov::element::bf16;
    case GGUF_TYPE_I8:
        return ov::element::i8;
    case GGUF_TYPE_I16:
        return ov::element::i16;
    case GGUF_TYPE_I32:
        return ov::element::i32;
    case GGUF_TYPE_I64:
        return ov::element::i64;
    default:
        return std::nullopt;
    }
}

ov::Shape get_shape(const gguf_tensor& tensor) {
    ov::Shape shape;
    // The dimension order in GGML is the reverse of the order used in MLX.
    for (int i = tensor.ndim - 1; i >= 0; i--) {
        shape.push_back(tensor.dim[i]);
    }
    return shape;
}

ov::Tensor extract_tensor_data(gguf_tensor* tensor) {
    std::optional<ov::element::Type> equivalent_dtype = gguf_type_to_dtype(tensor->type);
    // If there's an equivalent type, we can simply copy.
    if (equivalent_dtype.has_value()) {
        auto shape = get_shape(*tensor);
        ov::Tensor weights(equivalent_dtype.value(), shape);

        memcpy(weights.data(), tensor->weights_data, tensor->num_weights * equivalent_dtype.value().size());
        return weights;
    }
    // Otherwise, we convert to float16.
    // TODO: Add other dequantization options.
    int16_t* data = gguf_tensor_to_f16(tensor);
    OPENVINO_ASSERT(data != nullptr, "[load_gguf] gguf_tensor_to_f16 failed");

    auto shape = get_shape(*tensor);
    const size_t new_size = tensor->num_weights * sizeof(int16_t);
    ov::Tensor weights(ov::element::f16, shape);
    memcpy(weights.data(), data, new_size);
    free(data);

    return weights;
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

void gguf_load_moe_weights(std::unordered_map<std::string, ov::Tensor>& a,
                           std::unordered_map<std::string, gguf_tensor_type>& qtype_map,
                           gguf_tensor& tensor) {
    std::string name(tensor.name, tensor.namelen);

    auto shape = get_shape(tensor);

    float* weights_f32 = gguf_tensor_to_float(&tensor);

    size_t total_elem_num = ov::shape_size(shape);
    // TODO: channelwise?
    // size_t group_size = shape.back();
    size_t group_size = 128;
    size_t num_experts = shape[0];
    size_t shape_1 = shape[1];
    size_t shape_2 = shape[2];

    auto weights_shape = shape;
    // TODO: support q8, since MoE Q8 has not yet been merged into the OpenVINO 
    //       master branch, we are currently only supporting Q4 in this implementation.
    ov::Tensor weights(ov::element::u4, std::move(weights_shape));
    auto weights_ptr = static_cast<uint8_t*>(weights.data());

    size_t group_num = shape_2 / group_size;

    ov::Shape scale_zp_shape{num_experts, group_num, shape_1};

    ov::Tensor scales(ov::element::f16, scale_zp_shape);
    ov::Tensor zps(ov::element::u4, std::move(scale_zp_shape));

    auto scales_ptr = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto zps_ptr = static_cast<uint8_t*>(zps.data());

    const size_t rows = num_experts * shape_1;
    ov::parallel_for(rows, [&](size_t idx) {
        const size_t e = idx / shape_1;
        const size_t row = idx - e * shape_1;
        for (size_t g = 0; g < group_num; ++g) {
            const size_t start_idx = ((e * shape_1 + row) * shape_2) + g * group_size;
            float max = -FLT_MAX;
            float min = FLT_MAX;
            for (size_t i = 0; i < group_size; ++i) {
                const float v = weights_f32[start_idx + i];
                if (v > max) {
                    max = v;
                }
                if (v < min) {
                    min = v;
                }
            }

            const float d = (max - min) / ((1 << 4) - 1);
            const float id = d ? 1.0f / d : 0.0f;
            const uint8_t zp = static_cast<uint8_t>(std::round(-1.f * min / d));

            const size_t scale_idx = (e * group_num + g) * shape_1 + row;
            scales_ptr[scale_idx] = ov::float16(d);
            set_u4(zps_ptr, scale_idx, zp);

            const size_t dst_base = start_idx / 2;
            for (size_t j = 0; j < group_size / 2; ++j) {
                const float x0 = (weights_f32[start_idx + j * 2] - min) * id;
                const float x1 = (weights_f32[start_idx + j * 2 + 1] - min) * id;
                const uint8_t xi0 = MIN(15, static_cast<int8_t>(x0 + 0.5f));
                const uint8_t xi1 = MIN(15, static_cast<int8_t>(x1 + 0.5f));
                weights_ptr[dst_base + j] = xi0;
                weights_ptr[dst_base + j] |= static_cast<uint8_t>(xi1 << 4);
            }
        }
    });

    free(weights_f32);

    a.emplace(name, std::move(weights));

    auto check_insert = [](const auto& inserted) {
        OPENVINO_ASSERT(inserted.second,
                        "[load_gguf] Duplicate parameter name ",
                        inserted.first->second.data(),
                        ". This can happen when loading quantized tensors");
    };

    constexpr std::string_view weight_suffix = ".weight";
    const std::string name_prefix = name.substr(0, name.length() - weight_suffix.length());
    check_insert(a.emplace(name_prefix + ".scales", std::move(scales)));
    check_insert(a.emplace(name_prefix + ".zps", std::move(zps)));

    qtype_map.emplace(name_prefix + ".qtype", GGUF_TYPE_Q4_1);
}

void set_value_from_gguf(gguf_ctx* ctx, uint32_t type, gguf_value* val, GGUFMetaData& value) {
    switch (type) {
    case GGUF_VALUE_TYPE_UINT8:
        value = ov::Tensor(ov::element::u8, ov::Shape(0));
        *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::u8>::value_type>()) = val->uint8;
        break;
    case GGUF_VALUE_TYPE_INT8:
        value = ov::Tensor(ov::element::i8, ov::Shape(0));
        *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::i8>::value_type>()) = val->int8;
        break;
    case GGUF_VALUE_TYPE_UINT16:
        value = ov::Tensor(ov::element::u16, ov::Shape(0));
        *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::u16>::value_type>()) = val->uint16;
        break;
    case GGUF_VALUE_TYPE_INT16:
        value = ov::Tensor(ov::element::i16, ov::Shape(0));
        *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::i16>::value_type>()) = val->int16;
        break;
    case GGUF_VALUE_TYPE_UINT32:
        value = ov::Tensor(ov::element::u32, ov::Shape(0));
        *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::u32>::value_type>()) = val->uint32;
        break;
    case GGUF_VALUE_TYPE_INT32:
        value = ov::Tensor(ov::element::i32, ov::Shape(0));
        *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::i32>::value_type>()) = val->int32;
        break;
    case GGUF_VALUE_TYPE_UINT64:
        value = ov::Tensor(ov::element::u64, ov::Shape(0));
        *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::u64>::value_type>()) = val->uint64;
        break;
    case GGUF_VALUE_TYPE_INT64:
        value = ov::Tensor(ov::element::i64, ov::Shape(0));
        *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::i64>::value_type>()) = val->int64;
        break;
    case GGUF_VALUE_TYPE_FLOAT32:
        value = ov::Tensor(ov::element::f32, ov::Shape(0));
        *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::f32>::value_type>()) = val->float32;
        break;
    case GGUF_VALUE_TYPE_BOOL:
        value = ov::Tensor(ov::element::boolean, ov::Shape(0));
        *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::boolean>::value_type>()) = val->boolval;
        break;
    case GGUF_VALUE_TYPE_STRING:
        value = std::string(val->string.string, static_cast<int>(val->string.len));
        break;
    case GGUF_VALUE_TYPE_FLOAT64:
        value = ov::Tensor(ov::element::f64, ov::Shape(0));
        *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::f64>::value_type>()) = val->float64;
        break;
    case GGUF_VALUE_TYPE_ARRAY: {
        ctx->off += gguf_array_header_size;  // Skip header
        char* data = reinterpret_cast<char*>(val) + gguf_array_header_size;
        auto size = static_cast<size_t>(val->array.len);
        OPENVINO_ASSERT(val->array.type != GGUF_VALUE_TYPE_ARRAY,
                        "[load_gguf] Only supports loading 1-layer of nested arrays.");
        switch (val->array.type) {
        case GGUF_VALUE_TYPE_UINT8:
            value = ov::Tensor(ov::element::u8, ov::Shape{size});
            std::memcpy(std::get<ov::Tensor>(value).data<uint8_t>(),
                        reinterpret_cast<uint8_t*>(data),
                        size * sizeof(uint8_t));
            break;
        case GGUF_VALUE_TYPE_INT8:
            value = ov::Tensor(ov::element::i8, ov::Shape{size});
            std::memcpy(std::get<ov::Tensor>(value).data<int8_t>(),
                        reinterpret_cast<int8_t*>(data),
                        size * sizeof(int8_t));
            break;
        case GGUF_VALUE_TYPE_UINT16:
            value = ov::Tensor(ov::element::u16, ov::Shape{size});
            std::memcpy(std::get<ov::Tensor>(value).data<uint16_t>(),
                        reinterpret_cast<uint16_t*>(data),
                        size * sizeof(uint16_t));
            break;
        case GGUF_VALUE_TYPE_INT16:
            value = ov::Tensor(ov::element::i16, ov::Shape{size});
            std::memcpy(std::get<ov::Tensor>(value).data<int16_t>(),
                        reinterpret_cast<int16_t*>(data),
                        size * sizeof(int16_t));
            break;
        case GGUF_VALUE_TYPE_UINT32:
            value = ov::Tensor(ov::element::u32, ov::Shape{size});
            std::memcpy(std::get<ov::Tensor>(value).data<uint32_t>(),
                        reinterpret_cast<uint32_t*>(data),
                        size * sizeof(uint32_t));
            break;
        case GGUF_VALUE_TYPE_INT32:
            value = ov::Tensor(ov::element::i32, ov::Shape{size});
            std::memcpy(std::get<ov::Tensor>(value).data<int32_t>(),
                        reinterpret_cast<int32_t*>(data),
                        size * sizeof(int32_t));
            break;
        case GGUF_VALUE_TYPE_UINT64:
            value = ov::Tensor(ov::element::u64, ov::Shape{size});
            std::memcpy(std::get<ov::Tensor>(value).data<uint64_t>(),
                        reinterpret_cast<uint64_t*>(data),
                        size * sizeof(uint64_t));
            break;
        case GGUF_VALUE_TYPE_INT64:
            value = ov::Tensor(ov::element::i64, ov::Shape{size});
            std::memcpy(std::get<ov::Tensor>(value).data<int64_t>(),
                        reinterpret_cast<int64_t*>(data),
                        size * sizeof(int64_t));
            break;
        case GGUF_VALUE_TYPE_FLOAT32:
            value = ov::Tensor(ov::element::f32, ov::Shape{size});
            std::memcpy(std::get<ov::Tensor>(value).data<float>(),
                        reinterpret_cast<float*>(data),
                        size * sizeof(float));
            break;
        case GGUF_VALUE_TYPE_BOOL:
            value = ov::Tensor(ov::element::boolean, ov::Shape{size});
            std::memcpy(std::get<ov::Tensor>(value).data<bool>(), reinterpret_cast<bool*>(data), size * sizeof(bool));
            break;
        case GGUF_VALUE_TYPE_STRING: {
            std::vector<std::string> strs(size);
            for (auto& str : strs) {
                auto str_val = reinterpret_cast<gguf_string*>(data);
                data += (str_val->len + sizeof(gguf_string));
                str = std::string(str_val->string, static_cast<int>(str_val->len));
                ctx->off += (str_val->len + sizeof(gguf_string));
            }
            value = std::move(strs);
            break;
        }
        case GGUF_VALUE_TYPE_FLOAT64:
            value = ov::Tensor(ov::element::f64, ov::Shape{size});
            std::memcpy(std::get<ov::Tensor>(value).data<double>(),
                        reinterpret_cast<double*>(data),
                        size * sizeof(double));
            break;
        default:
            OPENVINO_THROW("[load_gguf] Multiple levels of nested arrays are not supported.");
        }
        break;
    }
    default:
        OPENVINO_THROW("[load_gguf] Received unexpected type.");
        break;
    }

    if (type == GGUF_VALUE_TYPE_STRING) {
        ctx->off += (sizeof(gguf_string) + std::get<std::string>(value).size());
    } else if (auto pv = std::get_if<ov::Tensor>(&value); pv) {
        ctx->off += pv->get_byte_size();
    }
}

std::unordered_map<std::string, GGUFMetaData> load_metadata(gguf_ctx* ctx) {
    std::unordered_map<std::string, GGUFMetaData> metadata;
    gguf_key key;
    while (gguf_get_key(ctx, &key)) {
        std::string key_name = std::string(key.name, key.namelen);
        auto& val = metadata.insert({key_name, GGUFMetaData{}}).first->second;
        set_value_from_gguf(ctx, key.type, key.val, val);
    }
    return metadata;
}

void load_arrays(gguf_ctx* ctx,
                 std::unordered_map<std::string, ov::Tensor>& array_map,
                 std::unordered_map<std::string, gguf_tensor_type>& qtype_map) {
    gguf_tensor tensor;

    auto check_insert = [](const auto& inserted) {
        OPENVINO_ASSERT(inserted.second,
                        "[load_gguf] Duplicate parameter name '",
                        inserted.first->first,
                        "'. This can happen when loading quantized tensors.");
    };

    while (gguf_get_tensor(ctx, &tensor)) {
        std::string name(tensor.name, tensor.namelen);

        bool is_moe_weights = name.find("exps") != std::string::npos;

        if (is_moe_weights) {
            gguf_load_moe_weights(array_map, qtype_map, tensor);
        } else if (tensor.type == GGUF_TYPE_Q4_0 || tensor.type == GGUF_TYPE_Q4_1 || tensor.type == GGUF_TYPE_Q8_0 ||
            tensor.type == GGUF_TYPE_Q4_K || tensor.type == GGUF_TYPE_Q6_K) {
            gguf_load_quantized(array_map, qtype_map, tensor);
        } else {
            ov::Tensor loaded_array = extract_tensor_data(&tensor);
            check_insert(array_map.emplace(name, loaded_array));

            constexpr std::string_view weight_suffix = ".weight";
            const std::string name_prefix = name.substr(0, name.length() - weight_suffix.length());
            qtype_map.emplace(name_prefix + ".qtype", static_cast<gguf_tensor_type>(tensor.type));
        }
    }
}

void check_file(std::string file) {
    bool exists;
    {
        std::ifstream f(file.c_str());
        exists = f.good();
    }
    OPENVINO_ASSERT(exists, "[load_gguf] Failed to open '", file, "'");
}

std::vector<std::string> get_all_files(std::string file, int total_num) {
    std::vector<std::string> files;
    files.push_back(file);

    size_t length = 5;
    size_t startPos = file.length() - 19;

    for (int i = 1; i < total_num; i++) {
        std::string new_number = std::to_string(i + 1);
        while (new_number.length() < length) {
            new_number = "0" + new_number;
        }
        file.replace(startPos, length, new_number);
        check_file(file);
        files.push_back(file);
    }
    return files;
}

GGUFLoad get_gguf_data(const std::string& file, bool is_tokenizer) {
    std::unordered_map<std::string, ov::Tensor> arrays;
    std::unordered_map<std::string, gguf_tensor_type> qtype;

    check_file(file);

    std::unique_ptr<gguf_ctx, decltype(&gguf_close)> ctx(gguf_open(file.data()), gguf_close);
    OPENVINO_ASSERT(ctx, "Failed to open '", file, "' with gguf_open");

    // get main config from first file or single file
    auto metadata = load_metadata(ctx.get());

    if (is_tokenizer) {
        return {metadata, arrays, qtype};
    }

    std::string split_flag = "split.count";
    auto it = metadata.find(split_flag);

    if (it == metadata.end())  // single GGUF file
    {
        load_arrays(ctx.get(), arrays, qtype);
        return {metadata, arrays, qtype};
    } else  // multi GGUF files
    {
        auto total_num_tensor = std::get<ov::Tensor>(metadata.at(split_flag));
        int total_num = *(total_num_tensor.data<ov::element_type_traits<ov::element::u16>::value_type>());

        std::vector<std::string> files = get_all_files(file, total_num);

        for (size_t i = 1; i < files.size(); i++) {
            std::unique_ptr<gguf_ctx, decltype(&gguf_close)> ctx_i(gguf_open(files.at(i).data()), gguf_close);
            OPENVINO_ASSERT(ctx_i, "Failed to open '", files.at(i), "' with gguf_open");

            auto metadata_tmp = load_metadata(ctx_i.get());

            load_arrays(ctx_i.get(), arrays, qtype);
        }
        load_arrays(ctx.get(), arrays, qtype);
        return {metadata, arrays, qtype};
    }
}

float metadata_to_float(const std::unordered_map<std::string, GGUFMetaData>& metadata, const std::string& key) {
    auto tensor = std::get<ov::Tensor>(metadata.at(key));
    return *(tensor.data<ov::element_type_traits<ov::element::f32>::value_type>());
}

int metadata_to_int(const std::unordered_map<std::string, GGUFMetaData>& metadata, const std::string& key) {
    auto tensor = std::get<ov::Tensor>(metadata.at(key));
    return *(tensor.data<ov::element_type_traits<ov::element::i32>::value_type>());
}

std::vector<int32_t> metadata_to_int_vec(const std::unordered_map<std::string, GGUFMetaData>& metadata,
                                         const std::string& key) {
    auto tensor = std::get<ov::Tensor>(metadata.at(key));
    const auto size = tensor.get_size();
    std::vector<int32_t> out(size, 0);
    const auto type = tensor.get_element_type();
    if (type == ov::element::i32) {
        const int32_t* data = tensor.data<const int32_t>();
        std::copy(data, data + size, out.begin());
        return out;
    }
    if (type == ov::element::i64) {
        const int64_t* data = tensor.data<const int64_t>();
        for (size_t i = 0; i < size; ++i) {
            out[i] = static_cast<int32_t>(data[i]);
        }
        return out;
    }
    OPENVINO_THROW("[load_gguf] Unsupported metadata array type for key: ", key);
}

std::map<std::string, GGUFMetaData> config_from_meta(const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    std::map<std::string, GGUFMetaData> config;
    auto arch = std::get<std::string>(metadata.at("general.architecture"));
    config["architecture"] = arch;
    config["layer_num"] = metadata_to_int(metadata, arch + ".block_count");
    config["head_num"] = metadata_to_int(metadata, arch + ".attention.head_count");
    config["head_size"] = metadata.count(arch + ".attention.key_length") ?
                     metadata_to_int(metadata, arch + ".attention.key_length") :
                     (metadata_to_int(metadata, arch + ".embedding_length") / 
                     metadata_to_int(metadata, arch + ".attention.head_count"));
    config["head_num_kv"] = metadata.count(arch + ".attention.head_count_kv") ?
            metadata_to_int(metadata, arch + ".attention.head_count_kv") :
            metadata_to_int(metadata, arch + ".attention.head_count");
    config["hidden_size"] = metadata_to_int(metadata, arch + ".embedding_length");
    config["max_position_embeddings"] = metadata.count(arch + ".context_length") ?
            metadata_to_int(metadata, arch + ".context_length") : 2048;
    config["rms_norm_eps"] = metadata_to_float(metadata, arch + ".attention.layer_norm_rms_epsilon");
    config["rope_freq_base"] = metadata.count(arch + ".rope.freq_base") ?
            metadata_to_float(metadata, arch + ".rope.freq_base") : 10000.0f;
    config["file_type"] = metadata_to_int(metadata, "general.file_type");
    // MoE related configs
    if (metadata.count(arch + ".expert_count")) {
        config["expert_count"] = metadata_to_int(metadata, arch + ".expert_count");
        config["expert_used_count"] = metadata_to_int(metadata, arch + ".expert_used_count");
        config["moe_inter_size"] = metadata_to_int(metadata, arch + ".expert_feed_forward_length");
    } else {
        config["expert_count"] = 0;
        config["expert_used_count"] = 0;
        config["moe_inter_size"] = 0;
    }

    if (metadata.count(arch + ".no_rope_layer_interval")) {
        config["no_rope_layer_interval"] = metadata_to_int(metadata, arch + ".no_rope_layer_interval");
    }
    if (metadata.count(arch + ".no_rope_layers")) {
        config["no_rope_layers"] = metadata_to_int_vec(metadata, arch + ".no_rope_layers");
    }
    return config;
}

std::unordered_map<std::string, ov::Tensor> consts_from_weights(
    const std::map<std::string, GGUFMetaData>& config,
    const std::unordered_map<std::string, ov::Tensor>& weights) {
    std::unordered_map<std::string, ov::Tensor> consts;

    consts["model.embed_tokens.weight"] = weights.at("token_embd.weight");
    consts["model.norm.weight"] = weights.at("output_norm.weight");
    if (weights.count("output.weight")) {
        consts["lm_head.weight"] = weights.at("output.weight");
        if (weights.count("output.bias")) {
            consts["lm_head.bias"] = weights.at("output.bias");
        }
    }

    // Handle quantization scales and biases
    if (weights.count("token_embd.scales")) {
        consts["model.embed_tokens.scales"] = weights.at("token_embd.scales");
        consts["model.embed_tokens.biases"] = weights.at("token_embd.biases");
    }
    if (weights.count("output.scales")) {
        consts["lm_head.scales"] = weights.at("output.scales");
        consts["lm_head.biases"] = weights.at("output.biases");
    }

    // Process layer weights
    for (int i = 0; i < std::get<int>(config.at("layer_num")); ++i) {
        consts[format("model.layers[%d].input_layernorm.weight", i)] = weights.at(format("blk.%d.attn_norm.weight", i));
        consts[format("model.layers[%d].post_attention_layernorm.weight", i)] = weights.at(format("blk.%d.ffn_norm.weight", i));
        
        // Attention weights
        consts[format("model.layers[%d].self_attn.q_proj.weight", i)] = weights.at(format("blk.%d.attn_q.weight", i));
        if (weights.count(format("blk.%d.attn_q.bias", i))) {
            consts[format("model.layers[%d].self_attn.q_proj.bias", i)] = weights.at(format("blk.%d.attn_q.bias", i));
        }
        consts[format("model.layers[%d].self_attn.k_proj.weight", i)] = weights.at(format("blk.%d.attn_k.weight", i));
        if (weights.count(format("blk.%d.attn_k.bias", i))) {
            consts[format("model.layers[%d].self_attn.k_proj.bias", i)] = weights.at(format("blk.%d.attn_k.bias", i));
        }
        consts[format("model.layers[%d].self_attn.v_proj.weight", i)] = weights.at(format("blk.%d.attn_v.weight", i));
        if (weights.count(format("blk.%d.attn_v.bias", i))) {
            consts[format("model.layers[%d].self_attn.v_proj.bias", i)] = weights.at(format("blk.%d.attn_v.bias", i));
        }
        consts[format("model.layers[%d].self_attn.o_proj.weight", i)] = weights.at(format("blk.%d.attn_output.weight", i));
        if (weights.count(format("blk.%d.attn_output.bias", i))) {
            consts[format("model.layers[%d].self_attn.o_proj.bias", i)] = weights.at(format("blk.%d.attn_output.bias", i));
        }

        //Qwen3
        if (weights.count(format("blk.%d.attn_k_norm.weight", i))) {
            consts[format("model.layers[%d].self_attn.k_norm.weight", i)] = weights.at(format("blk.%d.attn_k_norm.weight", i));
        }
        if (weights.count(format("blk.%d.attn_q_norm.weight", i))) {
            consts[format("model.layers[%d].self_attn.q_norm.weight", i)] = weights.at(format("blk.%d.attn_q_norm.weight", i));
        }

        // MLP weights
        if (weights.count(format("blk.%d.ffn_gate.weight", i))) {
            consts[format("model.layers[%d].mlp.gate_proj.weight", i)] = weights.at(format("blk.%d.ffn_gate.weight", i));
        }
        if (weights.count(format("blk.%d.ffn_gate.bias", i))) {
            consts[format("model.layers[%d].mlp.gate_proj.bias", i)] = weights.at(format("blk.%d.ffn_gate.bias", i));
        }
        if (weights.count(format("blk.%d.ffn_up.weight", i))) {
            consts[format("model.layers[%d].mlp.up_proj.weight", i)] = weights.at(format("blk.%d.ffn_up.weight", i));
        }
        if (weights.count(format("blk.%d.ffn_up.bias", i))) {
            consts[format("model.layers[%d].mlp.up_proj.bias", i)] = weights.at(format("blk.%d.ffn_up.bias", i));
        }
        if (weights.count(format("blk.%d.ffn_down.weight", i))) {
            consts[format("model.layers[%d].mlp.down_proj.weight", i)] = weights.at(format("blk.%d.ffn_down.weight", i));
        }
        if (weights.count(format("blk.%d.ffn_down.bias", i))) {
            consts[format("model.layers[%d].mlp.down_proj.bias", i)] = weights.at(format("blk.%d.ffn_down.bias", i));
        }

        // MoE weights
        // router weights
        if (weights.count(format("blk.%d.ffn_gate_inp.weight", i))) {
            consts[format("model.layers[%d].moe.gate_inp.weight", i)] = weights.at(format("blk.%d.ffn_gate_inp.weight", i));
        }
        if (weights.count(format("blk.%d.ffn_gate_exps.weight", i))) {
            consts[format("model.layers[%d].moe.gate_exps.weight", i)] = weights.at(format("blk.%d.ffn_gate_exps.weight", i));
        }
        if (weights.count(format("blk.%d.ffn_up_exps.weight", i))) {
            consts[format("model.layers[%d].moe.up_exps.weight", i)] = weights.at(format("blk.%d.ffn_up_exps.weight", i));
        }
        if (weights.count(format("blk.%d.ffn_down_exps.weight", i))) {
            consts[format("model.layers[%d].moe.down_exps.weight", i)] = weights.at(format("blk.%d.ffn_down_exps.weight", i));
        }

        // Quantization parameters 
        // If file_type not ALL_F32 = 0 or MOSTLY_F16 = 1, get dequant scales and biases 
        if (std::get<int>(config.at("file_type")) != 0 && std::get<int>(config.at("file_type")) != 1) { 
            if (weights.count(format("blk.%d.attn_q.scales", i))) {
                consts[format("model.layers[%d].self_attn.q_proj.scales", i)] = weights.at(format("blk.%d.attn_q.scales", i));
            }
            if (weights.count(format("blk.%d.attn_k.scales", i))) {
                consts[format("model.layers[%d].self_attn.k_proj.scales", i)] = weights.at(format("blk.%d.attn_k.scales", i));
            }
            if (weights.count(format("blk.%d.attn_v.scales", i))) {
                consts[format("model.layers[%d].self_attn.v_proj.scales", i)] = weights.at(format("blk.%d.attn_v.scales", i));
            }
            if (weights.count(format("blk.%d.attn_output.scales", i))) {
                consts[format("model.layers[%d].self_attn.o_proj.scales", i)] = weights.at(format("blk.%d.attn_output.scales", i));
            }
            if (weights.count(format("blk.%d.ffn_gate.scales", i))) {
                consts[format("model.layers[%d].mlp.gate_proj.scales", i)] = weights.at(format("blk.%d.ffn_gate.scales", i));
            }
            if (weights.count(format("blk.%d.ffn_up.scales", i))) {
                consts[format("model.layers[%d].mlp.up_proj.scales", i)] = weights.at(format("blk.%d.ffn_up.scales", i));
            }
            if (weights.count(format("blk.%d.ffn_down.scales", i))) {
                consts[format("model.layers[%d].mlp.down_proj.scales", i)] = weights.at(format("blk.%d.ffn_down.scales", i));
            }

            if (weights.count(format("blk.%d.ffn_gate_exps.scales", i))) {
                consts[format("model.layers[%d].moe.gate_exps.scales", i)] = weights.at(format("blk.%d.ffn_gate_exps.scales", i));
            }
            if (weights.count(format("blk.%d.ffn_up_exps.scales", i))) {
                consts[format("model.layers[%d].moe.up_exps.scales", i)] = weights.at(format("blk.%d.ffn_up_exps.scales", i));
            }
            if (weights.count(format("blk.%d.ffn_down_exps.scales", i))) {
                consts[format("model.layers[%d].moe.down_exps.scales", i)] = weights.at(format("blk.%d.ffn_down_exps.scales", i));
            }

            if (weights.count(format("blk.%d.attn_q.biases", i))) {
                consts[format("model.layers[%d].self_attn.q_proj.biases", i)] = weights.at(format("blk.%d.attn_q.biases", i));
            }  
            if (weights.count(format("blk.%d.attn_k.biases", i))) {
                consts[format("model.layers[%d].self_attn.k_proj.biases", i)] = weights.at(format("blk.%d.attn_k.biases", i));
            }    
            if (weights.count(format("blk.%d.attn_v.biases", i))) {
                consts[format("model.layers[%d].self_attn.v_proj.biases", i)] = weights.at(format("blk.%d.attn_v.biases", i));
            }
            if (weights.count(format("blk.%d.attn_output.biases", i))) {
                consts[format("model.layers[%d].self_attn.o_proj.biases", i)] = weights.at(format("blk.%d.attn_output.biases", i));
            }      
            if (weights.count(format("blk.%d.ffn_gate.biases", i))) {
                consts[format("model.layers[%d].mlp.gate_proj.biases", i)] = weights.at(format("blk.%d.ffn_gate.biases", i));
            }        
            if (weights.count(format("blk.%d.ffn_up.biases", i))) {
                consts[format("model.layers[%d].mlp.up_proj.biases", i)] = weights.at(format("blk.%d.ffn_up.biases", i));
            }           
            if (weights.count(format("blk.%d.ffn_down.biases", i))) {
                consts[format("model.layers[%d].mlp.down_proj.biases", i)] = weights.at(format("blk.%d.ffn_down.biases", i));
            }

            if (weights.count(format("blk.%d.ffn_gate_exps.zps", i))) {
                consts[format("model.layers[%d].moe.gate_exps.zps", i)] = weights.at(format("blk.%d.ffn_gate_exps.zps", i));
            }
            if (weights.count(format("blk.%d.ffn_up_exps.zps", i))) {
                consts[format("model.layers[%d].moe.up_exps.zps", i)] = weights.at(format("blk.%d.ffn_up_exps.zps", i));
            }
            if (weights.count(format("blk.%d.ffn_down_exps.zps", i))) {
                consts[format("model.layers[%d].moe.down_exps.zps", i)] = weights.at(format("blk.%d.ffn_down_exps.zps", i));
            }
        }
    }

    return consts;
}

std::unordered_map<std::string, gguf_tensor_type> get_qtype_map(
    const std::map<std::string, GGUFMetaData>& config,
    const std::unordered_map<std::string, gguf_tensor_type>& qtype) {
    std::unordered_map<std::string, gguf_tensor_type> qtype_map;

    if (qtype.count("token_embd.qtype")) {
        qtype_map["model.embed_tokens.qtype"] = qtype.at("token_embd.qtype");
    }
    if (qtype.count("output_norm.qtype")) {
        qtype_map["model.norm.qtype"] = qtype.at("output_norm.qtype");
    }
    if (qtype.count("output.qtype")) {
        qtype_map["lm_head.qtype"] = qtype.at("output.qtype");
    } else {
        qtype_map["lm_head.qtype"] = gguf_tensor_type::GGUF_TYPE_F16;  // To avoid that no output.weights layer
    }

    for (int i = 0; i < std::get<int>(config.at("layer_num")); ++i) {
        if (qtype.count(format("blk.%d.attn_norm.qtype", i))) {
            qtype_map[format("model.layers[%d].input_layernorm.qtype", i)] = qtype.at(format("blk.%d.attn_norm.qtype", i));
        }

        if (qtype.count(format("blk.%d.ffn_norm.qtype", i))) {
            qtype_map[format("model.layers[%d].post_attention_layernorm.qtype", i)] = qtype.at(format("blk.%d.ffn_norm.qtype", i));
        }

        // Attention weights
        if (qtype.count(format("blk.%d.attn_q.qtype", i))) {
            qtype_map[format("model.layers[%d].self_attn.q_proj.qtype", i)] = qtype.at(format("blk.%d.attn_q.qtype", i));
        }
        if (qtype.count(format("blk.%d.attn_k.qtype", i))) {
            qtype_map[format("model.layers[%d].self_attn.k_proj.qtype", i)] = qtype.at(format("blk.%d.attn_k.qtype", i));
        }
        if (qtype.count(format("blk.%d.attn_v.qtype", i))) {
            qtype_map[format("model.layers[%d].self_attn.v_proj.qtype", i)] = qtype.at(format("blk.%d.attn_v.qtype", i));
        }
        if (qtype.count(format("blk.%d.attn_output.qtype", i))) {
            qtype_map[format("model.layers[%d].self_attn.o_proj.qtype", i)] = qtype.at(format("blk.%d.attn_output.qtype", i));
        }

        // MLP weights
        if (qtype.count(format("blk.%d.ffn_gate.qtype", i))) {
            qtype_map[format("model.layers[%d].mlp.gate_proj.qtype", i)] = qtype.at(format("blk.%d.ffn_gate.qtype", i));
        }
        if (qtype.count(format("blk.%d.ffn_up.qtype", i))) {
            qtype_map[format("model.layers[%d].mlp.up_proj.qtype", i)] = qtype.at(format("blk.%d.ffn_up.qtype", i));
        }
        if (qtype.count(format("blk.%d.ffn_down.qtype", i))) {
            qtype_map[format("model.layers[%d].mlp.down_proj.qtype", i)] = qtype.at(format("blk.%d.ffn_down.qtype", i));
        }

        // MoE weights
        if (qtype.count(format("blk.%d.ffn_gate_inp.qtype", i))) {
            qtype_map[format("model.layers[%d].moe.gate_inp.qtype", i)] = qtype.at(format("blk.%d.ffn_gate_inp.qtype", i));
        }
        if (qtype.count(format("blk.%d.ffn_gate_exps.qtype", i))) {
            qtype_map[format("model.layers[%d].moe.gate_exps.qtype", i)] = qtype.at(format("blk.%d.ffn_gate_exps.qtype", i));
        }
        if (qtype.count(format("blk.%d.ffn_up_exps.qtype", i))) {
            qtype_map[format("model.layers[%d].moe.up_exps.qtype", i)] = qtype.at(format("blk.%d.ffn_up_exps.qtype", i));
        }
        if (qtype.count(format("blk.%d.ffn_down_exps.qtype", i))) {
            qtype_map[format("model.layers[%d].moe.down_exps.qtype", i)] = qtype.at(format("blk.%d.ffn_down_exps.qtype", i));
        }
    }

    return qtype_map;
}

std::tuple<std::map<std::string, GGUFMetaData>,
           std::unordered_map<std::string, ov::Tensor>,
           std::unordered_map<std::string, gguf_tensor_type>>
load_gguf(const std::string& file) {
    auto [metadata, weights, qtype] = get_gguf_data(file);

    auto config = config_from_meta(metadata);
    auto consts = consts_from_weights(config, weights);
    auto qtypes = get_qtype_map(config, qtype);

    return {config, consts, qtypes};
}
