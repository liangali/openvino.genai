// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "openvino/pass/serialize.hpp"

#include "openvino/genai/text_streamer.hpp"

#include "lm_encoding.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "sampling/logit_processor.hpp"
#include "sampling/sampler.hpp"
#include "utils.hpp"
#include "visual_language/inputs_embedder.hpp"

#ifdef ENABLE_SAFETENSORS
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"
#endif

namespace {

using Qwen3_5Config = ov::genai::modeling::models::Qwen3_5Config;
using Qwen3_5InputPlanner = ov::genai::modeling::models::Qwen3_5InputPlanner;
using Qwen3_5TextIO = ov::genai::modeling::models::Qwen3_5TextIO;
using Qwen3_5VisionIO = ov::genai::modeling::models::Qwen3_5VisionIO;
using Qwen3_5VisionPreprocessConfig = ov::genai::modeling::models::Qwen3_5VisionPreprocessConfig;
using Qwen3_5VisionPreprocessor = ov::genai::modeling::models::Qwen3_5VisionPreprocessor;

constexpr const char* kPosEmbedCacheResultName = "__pos_embed_cache__";

struct PreparedImageData {
    ov::Tensor image;
    ov::Tensor visual_embeds;
    ov::Tensor grid_thw;
    int64_t image_tokens = 0;
};

struct PromptBuildResult {
    std::string prompt;
    std::vector<size_t> image_sequence;
};

struct Qwen3_5VLModelArtifacts {
    std::shared_ptr<ov::Model> language_model;
    std::shared_ptr<ov::Model> vision_model;
    ov::Tensor pos_embed_weight;
};

struct Qwen3_5VLCachePaths {
    std::filesystem::path sample_language_xml_path;
    std::filesystem::path sample_vision_xml_path;
};

ov::Tensor clone_tensor(const ov::Tensor& src) {
    ov::Tensor dst(src.get_element_type(), src.get_shape());
    std::memcpy(dst.data(), src.data(), src.get_byte_size());
    return dst;
}

PreparedImageData clone_prepared_image_data(const PreparedImageData& src) {
    PreparedImageData dst;
    dst.image = clone_tensor(src.image);
    dst.visual_embeds = clone_tensor(src.visual_embeds);
    dst.grid_thw = clone_tensor(src.grid_thw);
    dst.image_tokens = src.image_tokens;
    return dst;
}

std::vector<ov::Tensor> split_images(const std::vector<ov::Tensor>& images) {
    std::vector<ov::Tensor> flat_images;
    for (const auto& image : images) {
        const auto shape = image.get_shape();
        switch (shape.size()) {
        case 3:
            flat_images.push_back(clone_tensor(image));
            break;
        case 4: {
            const size_t batch = shape[0];
            const size_t image_size = shape[1] * shape[2] * shape[3];
            for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
                ov::Tensor single(image.get_element_type(), {shape[1], shape[2], shape[3]});
                const auto* src_ptr = image.data<const uint8_t>() + batch_idx * image_size;
                std::memcpy(single.data(), src_ptr, single.get_byte_size());
                flat_images.push_back(std::move(single));
            }
            break;
        }
        default:
            OPENVINO_THROW("Input image must have [HWC] or [NHWC] layout, got rank ", shape.size());
        }
    }
    return flat_images;
}

ov::Tensor concat_tensors_dim0(const std::vector<ov::Tensor>& tensors) {
    OPENVINO_ASSERT(!tensors.empty(), "Cannot concatenate an empty tensor list");
    const auto& first = tensors.front();
    ov::Shape output_shape = first.get_shape();
    size_t outer = 0;
    const size_t inner = first.get_size() / first.get_shape().at(0);
    for (const auto& tensor : tensors) {
        OPENVINO_ASSERT(tensor.get_element_type() == first.get_element_type(), "Concatenated tensors must use the same type");
        OPENVINO_ASSERT(tensor.get_shape().size() == output_shape.size(), "Concatenated tensors must have the same rank");
        for (size_t dim = 1; dim < output_shape.size(); ++dim) {
            OPENVINO_ASSERT(tensor.get_shape().at(dim) == output_shape.at(dim), "Concatenated tensors must match on non-batch dimensions");
        }
        outer += tensor.get_shape().at(0);
    }
    output_shape[0] = outer;
    ov::Tensor output(first.get_element_type(), output_shape);
    auto* dst = static_cast<uint8_t*>(output.data());
    for (const auto& tensor : tensors) {
        const size_t bytes = tensor.get_shape().at(0) * inner * first.get_element_type().size();
        std::memcpy(dst, tensor.data(), bytes);
        dst += bytes;
    }
    return output;
}

ov::Tensor extract_pos_embed_from_vision_model(std::shared_ptr<ov::Model>& model) {
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() != kPosEmbedCacheResultName) {
            continue;
        }
        auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(result->input_value(0).get_node_shared_ptr());
        OPENVINO_ASSERT(const_node, "pos_embed cache result is not a Constant");
        ov::Tensor tensor(const_node->get_element_type(), const_node->get_shape());
        std::memcpy(tensor.data(), const_node->get_data_ptr(), tensor.get_byte_size());
        model->remove_result(result);
        return tensor;
    }
    OPENVINO_THROW("Cached vision IR does not contain pos_embed data. Re-export the Qwen3.5 vision IR with cached pos_embed.");
}

std::string resolve_pos_embed_name(ov::genai::modeling::weights::WeightSource& source) {
    const std::vector<std::string> candidates = {
        "model.visual.pos_embed.weight",
        "visual.pos_embed.weight",
        "pos_embed.weight",
    };
    for (const auto& name : candidates) {
        if (source.has(name)) {
            return name;
        }
    }
    for (const auto& name : source.keys()) {
        if (name.find("pos_embed.weight") != std::string::npos) {
            return name;
        }
    }
    OPENVINO_THROW("Failed to locate visual.pos_embed.weight in Qwen3.5 safetensors weights");
}

void embed_pos_embed_in_vision_model(std::shared_ptr<ov::Model>& model, const ov::Tensor& pos_embed) {
    auto constant = std::make_shared<ov::op::v0::Constant>(pos_embed);
    auto result = std::make_shared<ov::op::v0::Result>(constant);
    result->set_friendly_name(kPosEmbedCacheResultName);
    model->add_results({result});
}

void remove_pos_embed_cache_result(std::shared_ptr<ov::Model>& model) {
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() == kPosEmbedCacheResultName) {
            model->remove_result(result);
            return;
        }
    }
}

bool has_ir_model_pair(const std::filesystem::path& xml_path) {
    auto bin_path = xml_path;
    bin_path.replace_extension(".bin");
    return std::filesystem::exists(xml_path) && std::filesystem::is_regular_file(xml_path) &&
           std::filesystem::exists(bin_path) && std::filesystem::is_regular_file(bin_path);
}

std::string quant_mode_cache_token(ov::genai::modeling::weights::QuantizationConfig::Mode mode) {
    using Mode = ov::genai::modeling::weights::QuantizationConfig::Mode;
    switch (mode) {
    case Mode::INT4_SYM:
        return "4s";
    case Mode::INT4_ASYM:
        return "4a";
    case Mode::INT8_SYM:
        return "8s";
    case Mode::INT8_ASYM:
        return "8a";
    case Mode::NONE:
    default:
        return "n";
    }
}

std::string quant_cache_suffix(const ov::genai::modeling::weights::QuantizationConfig& cfg) {
    if (!cfg.enabled()) {
        return "";
    }
    return "_q" + quant_mode_cache_token(cfg.mode) + "_b" + quant_mode_cache_token(cfg.backup_mode) +
           "_g" + std::to_string(cfg.group_size);
}

Qwen3_5VLCachePaths get_qwen3_5_vl_cache_paths(
    const std::filesystem::path& models_dir,
    const ov::genai::modeling::weights::QuantizationConfig& text_quant_config,
    const ov::genai::modeling::weights::QuantizationConfig& vision_quant_config
) {
    return {
        models_dir / ("qwen3_5_text_vl" + quant_cache_suffix(text_quant_config) + ".xml"),
        models_dir / ("qwen3_5_vision" + quant_cache_suffix(vision_quant_config) + ".xml"),
    };
}

bool has_model_input_name(const std::shared_ptr<ov::Model>& model, const std::string& input_name) {
    if (!model) {
        return false;
    }
    for (const auto& input : model->inputs()) {
        if (input.get_names().count(input_name) > 0) {
            return true;
        }
    }
    return false;
}

bool is_vl_text_ir_compatible(const std::shared_ptr<ov::Model>& model) {
    return has_model_input_name(model, Qwen3_5TextIO::kVisualEmbeds) &&
           has_model_input_name(model, Qwen3_5TextIO::kVisualPosMask);
}

std::shared_ptr<ov::Model> load_text_model_from_cache(
    const std::filesystem::path& xml_path,
    std::filesystem::path* loaded_xml_path = nullptr
) {
    if (!has_ir_model_pair(xml_path)) {
        return nullptr;
    }
    auto model = utils::singleton_core().read_model(xml_path);
    if (!is_vl_text_ir_compatible(model)) {
        return nullptr;
    }
    if (loaded_xml_path) {
        *loaded_xml_path = xml_path;
    }
    return model;
}

std::pair<std::shared_ptr<ov::Model>, ov::Tensor> load_vision_model_from_cache(
    const std::filesystem::path& xml_path,
    std::filesystem::path* loaded_xml_path = nullptr
) {
    if (!has_ir_model_pair(xml_path)) {
        return {nullptr, {}};
    }
    auto model = utils::singleton_core().read_model(xml_path);
    try {
        auto pos_embed = extract_pos_embed_from_vision_model(model);
        if (loaded_xml_path) {
            *loaded_xml_path = xml_path;
        }
        return {model, pos_embed};
    } catch (const std::exception&) {
    }
    return {nullptr, {}};
}

Qwen3_5VLModelArtifacts load_qwen3_5_vl_models(
    const std::filesystem::path& models_dir,
    const Qwen3_5Config& config,
    bool enable_save_ov_model,
    const std::optional<ov::genai::modeling::weights::QuantizationConfig>& quant_config
) {
    Qwen3_5VLModelArtifacts artifacts;
    const auto shared_quant_config = quant_config.value_or(ov::genai::modeling::weights::parse_quantization_config_from_env());
    if (shared_quant_config.enabled() && shared_quant_config.group_size <= 0) {
        OPENVINO_THROW("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE must be > 0 when OV_GENAI_INFLIGHT_QUANT_MODE is enabled");
    }

    ov::genai::modeling::weights::QuantizationConfig vision_quant_config;
    ov::genai::modeling::weights::QuantizationConfig text_quant_config = shared_quant_config;
    const auto cache_paths = get_qwen3_5_vl_cache_paths(models_dir, text_quant_config, vision_quant_config);
    const auto language_bin_path = cache_paths.sample_language_xml_path.parent_path() /
                                   (cache_paths.sample_language_xml_path.stem().string() + ".bin");
    const auto vision_bin_path = cache_paths.sample_vision_xml_path.parent_path() /
                                 (cache_paths.sample_vision_xml_path.stem().string() + ".bin");

    artifacts.language_model = load_text_model_from_cache(cache_paths.sample_language_xml_path);

    std::tie(artifacts.vision_model, artifacts.pos_embed_weight) = load_vision_model_from_cache(cache_paths.sample_vision_xml_path);

    if (artifacts.language_model && artifacts.vision_model && artifacts.pos_embed_weight) {
        return artifacts;
    }

#ifdef ENABLE_SAFETENSORS
    if (!ov::genai::safetensors::is_safetensors_model(models_dir)) {
        OPENVINO_THROW(
            "Qwen3.5 VLM directory is missing compatible OpenVINO IR components and does not contain a safetensors model: ",
            models_dir
        );
    }

    auto data = ov::genai::safetensors::load_safetensors(models_dir);
    ov::genai::safetensors::SafetensorsWeightSource source(std::move(data));

    if (!artifacts.vision_model) {
        ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer(vision_quant_config);
        artifacts.vision_model = ov::genai::modeling::models::create_qwen3_5_vision_model(config, source, vision_finalizer);
        artifacts.pos_embed_weight = clone_tensor(source.get_tensor(resolve_pos_embed_name(source)));

        if (enable_save_ov_model) {
            embed_pos_embed_in_vision_model(artifacts.vision_model, artifacts.pos_embed_weight);
            ov::serialize(
                artifacts.vision_model,
                cache_paths.sample_vision_xml_path.string(),
                vision_bin_path.string());
            remove_pos_embed_cache_result(artifacts.vision_model);
        }
    }

    if (!artifacts.language_model) {
        ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer(text_quant_config);
        artifacts.language_model = ov::genai::modeling::models::create_qwen3_5_text_model(config, source, text_finalizer, false, true);

        if (enable_save_ov_model) {
            ov::serialize(
                artifacts.language_model,
                cache_paths.sample_language_xml_path.string(),
                language_bin_path.string());
        }
    }

    return artifacts;
#else
    OPENVINO_THROW(
        "Qwen3.5 VLM directory is missing compatible OpenVINO IR components and this build was compiled without safetensors support: ",
        models_dir
    );
#endif
}

std::string make_image_block(int64_t image_tokens) {
    OPENVINO_ASSERT(image_tokens > 0, "Qwen3.5 image token count must be positive");
    std::string block = "<|vision_start|>";
    block.reserve(block.size() + static_cast<size_t>(image_tokens) * 13 + 14);
    for (int64_t idx = 0; idx < image_tokens; ++idx) {
        block += "<|image_pad|>";
    }
    block += "<|vision_end|>";
    return block;
}

PromptBuildResult build_prompt_with_images(const std::string& prompt, const std::vector<PreparedImageData>& images) {
    if (images.empty()) {
        OPENVINO_ASSERT(prompt.find("<ov_genai_image_") == std::string::npos, "Prompt contains image placeholders but no images were provided");
        return {prompt, {}};
    }

    OPENVINO_ASSERT(prompt.find("<|image_pad|>") == std::string::npos &&
                        prompt.find("<|vision_start|>") == std::string::npos &&
                        prompt.find("<|vision_end|>") == std::string::npos,
        "Direct Qwen3.5 native image tokens are not supported here; use <ov_genai_image_i> placeholders or pass plain prompt text");

    auto [rewritten_prompt, image_sequence] = universal_to_native(
        prompt,
        [&](std::ostream& os, size_t idx) {
            OPENVINO_ASSERT(idx < images.size(), "Missing image with index ", idx);
            os << make_image_block(images.at(idx).image_tokens);
        }
    );

    if (!image_sequence.empty()) {
        return {std::move(rewritten_prompt), std::move(image_sequence)};
    }

    std::stringstream stream;
    image_sequence.reserve(images.size());
    for (size_t idx = 0; idx < images.size(); ++idx) {
        image_sequence.push_back(idx);
        stream << make_image_block(images.at(idx).image_tokens);
    }
    if (!prompt.empty()) {
        stream << '\n' << prompt;
    }
    return {stream.str(), std::move(image_sequence)};
}

ChatHistory clone_history(const ChatHistory& history) {
    ChatHistory out(history.get_messages().copy());
    out.set_tools(history.get_tools().copy());
    out.set_extra_context(history.get_extra_context().copy());
    return out;
}

std::set<int64_t> resolve_stop_token_ids(const GenerationConfig& generation_config, const Tokenizer& tokenizer) {
    std::set<int64_t> stop_token_ids = generation_config.stop_token_ids;
    if (stop_token_ids.empty() && !generation_config.ignore_eos) {
        const int64_t eos_token_id = generation_config.eos_token_id >= 0
            ? generation_config.eos_token_id
            : tokenizer.get_eos_token_id();
        if (eos_token_id >= 0) {
            stop_token_ids.insert(eos_token_id);
        }
    }
    return stop_token_ids;
}

void extract_last_logits_f32(const ov::Tensor& logits, std::vector<float>& out) {
    const auto shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Expected logits shape [B, S, V]");
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    const size_t offset = (seq_len - 1) * vocab;
    out.resize(vocab);
    if (logits.get_element_type() == ov::element::f32) {
        std::memcpy(out.data(), logits.data<const float>() + offset, vocab * sizeof(float));
    } else if (logits.get_element_type() == ov::element::f16) {
        const auto* src = logits.data<const ov::float16>() + offset;
        for (size_t i = 0; i < vocab; ++i) {
            out[i] = static_cast<float>(src[i]);
        }
    } else if (logits.get_element_type() == ov::element::bf16) {
        const auto* src = logits.data<const ov::bfloat16>() + offset;
        for (size_t i = 0; i < vocab; ++i) {
            out[i] = static_cast<float>(src[i]);
        }
    } else {
        OPENVINO_THROW("Unsupported logits dtype for Qwen3.5 VLM sampling loop");
    }
}

int64_t argmax_f32(const std::vector<float>& data) {
    return static_cast<int64_t>(std::max_element(data.begin(), data.end()) - data.begin());
}

struct SamplingContext {
    std::vector<std::pair<float, int64_t>> ranked;
    std::vector<std::pair<float, int64_t>> candidates;
    std::vector<float> probs;
};

int64_t sample_fast(const float* logits,
                    size_t vocab_size,
                    float temperature,
                    float top_p,
                    size_t top_k,
                    std::mt19937& rng,
                    SamplingContext& ctx) {
    OPENVINO_ASSERT(vocab_size > 0, "logits must not be empty");
    OPENVINO_ASSERT(temperature > 0.0f, "temperature must be positive for sampling");

    const bool use_top_k = (top_k > 0 && top_k < vocab_size);
    const bool use_top_p = (top_p > 0.0f && top_p < 1.0f);

    if (!use_top_k && !use_top_p) {
        const float max_logit = *std::max_element(logits, logits + vocab_size);
        const float inv_temp = 1.0f / temperature;

        ctx.probs.resize(vocab_size);
        float total = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            const float value = std::exp((logits[i] - max_logit) * inv_temp);
            ctx.probs[i] = value;
            total += value;
        }

        if (!(total > 0.0f) || !std::isfinite(total)) {
            return static_cast<int64_t>(std::max_element(logits, logits + vocab_size) - logits);
        }

        std::uniform_real_distribution<float> distribution(0.0f, total);
        const float dart = distribution(rng);
        float cumulative = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumulative += ctx.probs[i];
            if (cumulative >= dart) {
                return static_cast<int64_t>(i);
            }
        }
        return static_cast<int64_t>(vocab_size - 1);
    }

    if (use_top_k) {
        const size_t k = std::min(top_k, vocab_size);
        ctx.ranked.resize(k);
        for (size_t i = 0; i < k; ++i) {
            ctx.ranked[i] = {logits[i], static_cast<int64_t>(i)};
        }

        size_t min_pos = 0;
        float min_val = ctx.ranked[0].first;
        for (size_t i = 1; i < k; ++i) {
            if (ctx.ranked[i].first < min_val) {
                min_val = ctx.ranked[i].first;
                min_pos = i;
            }
        }

        for (size_t i = k; i < vocab_size; ++i) {
            if (logits[i] > min_val) {
                ctx.ranked[min_pos] = {logits[i], static_cast<int64_t>(i)};
                min_val = ctx.ranked[0].first;
                min_pos = 0;
                for (size_t j = 1; j < k; ++j) {
                    if (ctx.ranked[j].first < min_val) {
                        min_val = ctx.ranked[j].first;
                        min_pos = j;
                    }
                }
            }
        }

        std::sort(ctx.ranked.begin(), ctx.ranked.end(), [](const auto& left, const auto& right) {
            return left.first > right.first;
        });

        const float max_logit = ctx.ranked[0].first;
        const float inv_temp = 1.0f / temperature;
        ctx.probs.resize(k);
        float total = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            const float value = std::exp((ctx.ranked[i].first - max_logit) * inv_temp);
            ctx.probs[i] = value;
            total += value;
        }

        if (!(total > 0.0f) || !std::isfinite(total)) {
            return ctx.ranked[0].second;
        }

        size_t num_candidates = k;
        if (use_top_p) {
            const float threshold = top_p * total;
            float cumulative = 0.0f;
            for (size_t i = 0; i < k; ++i) {
                cumulative += ctx.probs[i];
                if (cumulative >= threshold) {
                    num_candidates = i + 1;
                    total = cumulative;
                    break;
                }
            }
        }
        num_candidates = std::max<size_t>(1, num_candidates);

        std::uniform_real_distribution<float> distribution(0.0f, total);
        const float dart = distribution(rng);
        float cumulative = 0.0f;
        for (size_t i = 0; i < num_candidates; ++i) {
            cumulative += ctx.probs[i];
            if (cumulative >= dart) {
                return ctx.ranked[i].second;
            }
        }
        return ctx.ranked[num_candidates - 1].second;
    }

    ctx.candidates.resize(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        ctx.candidates[i] = {logits[i], static_cast<int64_t>(i)};
    }

    const float max_logit = std::max_element(
        ctx.candidates.begin(),
        ctx.candidates.end(),
        [](const auto& left, const auto& right) { return left.first < right.first; }
    )->first;
    const float inv_temp = 1.0f / temperature;
    float total = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        ctx.candidates[i].first = std::exp((ctx.candidates[i].first - max_logit) * inv_temp);
        total += ctx.candidates[i].first;
    }

    if (!(total > 0.0f) || !std::isfinite(total)) {
        for (size_t i = 0; i < vocab_size; ++i) {
            if (logits[i] == max_logit) {
                return static_cast<int64_t>(i);
            }
        }
        return 0;
    }

    const float threshold = top_p * total;
    size_t num_candidates = vocab_size;
    for (size_t step = 16; step <= 1024; step *= 2) {
        if (vocab_size <= step) {
            break;
        }
        std::partial_sort(
            ctx.candidates.begin(),
            ctx.candidates.begin() + static_cast<ptrdiff_t>(step),
            ctx.candidates.end(),
            [](const auto& left, const auto& right) { return left.first > right.first; }
        );
        float cumulative = 0.0f;
        for (size_t i = 0; i < step; ++i) {
            cumulative += ctx.candidates[i].first;
            if (cumulative >= threshold) {
                num_candidates = i + 1;
                total = cumulative;
                goto nucleus_found;
            }
        }
    }

    std::sort(ctx.candidates.begin(), ctx.candidates.end(), [](const auto& left, const auto& right) {
        return left.first > right.first;
    });
    {
        float cumulative = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumulative += ctx.candidates[i].first;
            if (cumulative >= threshold) {
                num_candidates = i + 1;
                total = cumulative;
                break;
            }
        }
    }

nucleus_found:
    num_candidates = std::max<size_t>(1, num_candidates);
    std::uniform_real_distribution<float> distribution(0.0f, total);
    const float dart = distribution(rng);
    float cumulative = 0.0f;
    for (size_t i = 0; i < num_candidates; ++i) {
        cumulative += ctx.candidates[i].first;
        if (cumulative >= dart) {
            return ctx.candidates[i].second;
        }
    }
    return ctx.candidates[num_candidates - 1].second;
}

ov::Tensor make_beam_idx(size_t batch) {
    ov::Tensor beam_idx(ov::element::i32, {batch});
    auto* data = beam_idx.data<int32_t>();
    for (size_t i = 0; i < batch; ++i) {
        data[i] = static_cast<int32_t>(i);
    }
    return beam_idx;
}

int64_t get_active_prompt_length(const ov::Tensor& attention_mask) {
    OPENVINO_ASSERT(attention_mask.get_element_type() == ov::element::i64, "Expected i64 attention_mask");
    const auto& shape = attention_mask.get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "Expected attention_mask shape [B, S]");
    const auto* mask_data = attention_mask.data<const int64_t>();
    int64_t first_active_tokens = -1;
    for (size_t batch_idx = 0; batch_idx < shape[0]; ++batch_idx) {
        int64_t active_tokens = 0;
        for (size_t seq_idx = 0; seq_idx < shape[1]; ++seq_idx) {
            if (mask_data[batch_idx * shape[1] + seq_idx] != 0) {
                active_tokens += 1;
            }
        }
        if (first_active_tokens < 0) {
            first_active_tokens = active_tokens;
        } else {
            OPENVINO_ASSERT(
                first_active_tokens == active_tokens,
                "Batch decode with different active prompt lengths is not supported in Qwen3.5 VLM backend"
            );
        }
    }
    return std::max<int64_t>(0, first_active_tokens);
}

class Qwen3_5VLStatefulPipeline : public VLMPipeline::VLMPipelineBase {
public:
    Qwen3_5VLStatefulPipeline(
        const std::filesystem::path& models_dir,
        const std::string& device,
        const ov::AnyMap& properties
    ) :
        m_generation_config(utils::from_config_json_if_exists<GenerationConfig>(models_dir, "generation_config.json")),
        m_config(Qwen3_5Config::from_json_file(models_dir / "config.json")),
        m_preprocess_config(Qwen3_5VisionPreprocessConfig::from_json_file(models_dir / "preprocessor_config.json")),
        m_preprocessor(m_config.vision, m_preprocess_config),
        m_input_planner(m_config) {
        OPENVINO_ASSERT(device.find("NPU") == std::string::npos, "Qwen3.5 VLM stateful backend currently supports CPU/GPU devices only");

        auto [properties_without_scheduler, scheduler_config_ignored] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
        (void) scheduler_config_ignored;
        if (auto enable_thinking = utils::pop_option(properties_without_scheduler, "enable_thinking"); enable_thinking.has_value()) {
            OPENVINO_ASSERT(!enable_thinking->empty(), "Got empty ov::Any for parameter name: enable_thinking");
            m_enable_thinking = enable_thinking->as<bool>();
        }
        auto [compile_properties_without_quant, quant_config] = utils::extract_quantization_config(properties_without_scheduler);
        auto [compile_properties, enable_save_ov_model] = utils::extract_gguf_properties(compile_properties_without_quant);
        auto artifacts = load_qwen3_5_vl_models(models_dir, m_config, enable_save_ov_model, quant_config);

        auto language_model = artifacts.language_model;
        auto kv_pos = ov::genai::utils::get_kv_axes_pos(language_model);
        m_kv_cache_state.seq_length_axis = kv_pos.seq_len;
        m_language = utils::singleton_core().compile_model(language_model, device, compile_properties).create_infer_request();
        m_language.get_tensor("attention_mask").set_shape({1, 0});

        m_pos_embed_weight = artifacts.pos_embed_weight;
        m_vision = utils::singleton_core().compile_model(artifacts.vision_model, device, compile_properties).create_infer_request();

        m_tokenizer = Tokenizer(models_dir, compile_properties_without_quant);
        initialize_generation_defaults();
    }

    Qwen3_5VLStatefulPipeline(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties,
        const GenerationConfig& generation_config
    ) :
        m_generation_config(generation_config),
        m_config(Qwen3_5Config::from_json_file(config_dir_path / "config.json")),
        m_preprocess_config(Qwen3_5VisionPreprocessConfig::from_json_file(config_dir_path / "preprocessor_config.json")),
        m_preprocessor(m_config.vision, m_preprocess_config),
        m_input_planner(m_config),
        m_tokenizer(tokenizer) {
        OPENVINO_ASSERT(device.find("NPU") == std::string::npos, "Qwen3.5 VLM stateful backend currently supports CPU/GPU devices only");

        auto [properties_without_scheduler, scheduler_config_ignored] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
        (void) scheduler_config_ignored;
        if (auto enable_thinking = utils::pop_option(properties_without_scheduler, "enable_thinking"); enable_thinking.has_value()) {
            OPENVINO_ASSERT(!enable_thinking->empty(), "Got empty ov::Any for parameter name: enable_thinking");
            m_enable_thinking = enable_thinking->as<bool>();
        }
        auto [compile_properties_without_quant, qconf_ignored] = utils::extract_quantization_config(properties_without_scheduler);
        auto [compile_properties, enable_save_ignored] = utils::extract_gguf_properties(compile_properties_without_quant);
        (void) qconf_ignored;
        (void) enable_save_ignored;

        auto language_pair = utils::get_model_weights_pair(models_map, "language");
        auto language_model = utils::singleton_core().read_model(language_pair.first, language_pair.second);
        auto kv_pos = ov::genai::utils::get_kv_axes_pos(language_model);
        m_kv_cache_state.seq_length_axis = kv_pos.seq_len;
        m_language = utils::singleton_core().compile_model(language_model, device, compile_properties).create_infer_request();
        m_language.get_tensor("attention_mask").set_shape({1, 0});

        auto vision_pair = utils::get_model_weights_pair(models_map, "vision_embeddings");
        auto vision_model = utils::singleton_core().read_model(vision_pair.first, vision_pair.second);
        m_pos_embed_weight = extract_pos_embed_from_vision_model(vision_model);
        m_vision = utils::singleton_core().compile_model(vision_model, device, compile_properties).create_infer_request();

        initialize_generation_defaults();
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(prompt, images, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        OPENVINO_ASSERT(videos.empty(), "Qwen3.5 VLM backend does not support video input yet");
        const auto prepare_embeddings_start = std::chrono::steady_clock::now();
        auto prepared_images = prepare_images(images);
        const auto prepare_embeddings_end = std::chrono::steady_clock::now();
        const float prepare_embeddings_duration_us = PerfMetrics::get_microsec(prepare_embeddings_end - prepare_embeddings_start);
        auto prompt_data = build_prompt_with_images(prompt, prepared_images);

        if (m_is_chat_conversation) {
            ChatHistory next_history = clone_history(m_history);
            next_history.push_back({{"role", "user"}, {"content", prompt_data.prompt}});
            std::vector<PreparedImageData> next_chat_images = m_chat_images;
            append_ordered_chat_images(next_chat_images, prepared_images, prompt_data.image_sequence);

            auto result = generate_from_history(
                next_history,
                next_chat_images,
                std::move(generation_config),
                streamer,
                prepare_embeddings_duration_us
            );
            if (result.texts.empty()) {
                return result;
            }

            next_history.push_back({{"role", "assistant"}, {"content", result.texts.at(0)}});
            m_history = std::move(next_history);
            m_chat_images = std::move(next_chat_images);
            return result;
        }

        return generate_single_prompt(
            prompt_data.prompt,
            prepared_images,
            prompt_data.image_sequence,
            std::move(generation_config),
            streamer,
            true,
            prepare_embeddings_duration_us
        );
    }

    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(history, images, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        OPENVINO_ASSERT(videos.empty(), "Qwen3.5 VLM backend does not support video input yet");
        const auto prepare_embeddings_start = std::chrono::steady_clock::now();
        auto prepared_images = prepare_images(images);
        const auto prepare_embeddings_end = std::chrono::steady_clock::now();
        const float prepare_embeddings_duration_us = PerfMetrics::get_microsec(prepare_embeddings_end - prepare_embeddings_start);
        ChatHistory normalized_history = clone_history(history);

        if (!prepared_images.empty()) {
            OPENVINO_ASSERT(!normalized_history.empty(), "Chat history cannot be empty when images are provided");
            auto last_message = normalized_history.last();
            OPENVINO_ASSERT(last_message.contains("role") && last_message["role"].is_string() && last_message["role"].get_string() == "user",
                "Images can only be associated with the last user message");
            OPENVINO_ASSERT(last_message.contains("content") && last_message["content"].is_string(),
                "Qwen3.5 VLM backend expects string chat content for image messages");
            auto prompt_data = build_prompt_with_images(last_message["content"].get_string(), prepared_images);
            normalized_history.last()["content"] = prompt_data.prompt;

            std::vector<PreparedImageData> ordered_images;
            append_ordered_chat_images(ordered_images, prepared_images, prompt_data.image_sequence);
            return generate_from_history(
                normalized_history,
                ordered_images,
                std::move(generation_config),
                streamer,
                prepare_embeddings_duration_us
            );
        }

        return generate_from_history(normalized_history, {}, std::move(generation_config), streamer);
    }

    void start_chat(const std::string& system_message) override {
        finish_chat();
        m_is_chat_conversation = true;
        m_system_message = system_message;
        if (!system_message.empty()) {
            m_history.push_back({{"role", "system"}, {"content", system_message}});
        }
    }

    void finish_chat() override {
        m_is_chat_conversation = false;
        m_system_message.clear();
        m_history.clear();
        m_chat_images.clear();
        reset_language_state();
    }

    Tokenizer get_tokenizer() const override {
        return m_tokenizer;
    }

    void set_chat_template(const std::string& new_template) override {
        OPENVINO_ASSERT(!m_is_chat_conversation, "Chat template cannot be changed once start_chat() is called. Please, finish current chat via finish_chat()");
        m_tokenizer.set_chat_template(new_template);
    }

    GenerationConfig get_generation_config() const override {
        return m_generation_config;
    }

    void set_generation_config(const GenerationConfig& new_config) override {
        int64_t default_eos_token_id = m_generation_config.eos_token_id;
        auto default_stop_token_ids = m_generation_config.stop_token_ids;
        m_generation_config = new_config;
        if (m_generation_config.stop_token_ids.empty())
            m_generation_config.stop_token_ids = default_stop_token_ids;
        if (m_generation_config.eos_token_id == -1)
            m_generation_config.set_eos_token_id(default_eos_token_id);
        m_generation_config.validate();
    }

private:
    GenerationConfig m_generation_config;
    Tokenizer m_tokenizer;
    ov::InferRequest m_language;
    ov::InferRequest m_vision;
    Sampler m_sampler;
    Qwen3_5Config m_config;
    Qwen3_5VisionPreprocessConfig m_preprocess_config;
    Qwen3_5VisionPreprocessor m_preprocessor;
    Qwen3_5InputPlanner m_input_planner;
    utils::KVCacheState m_kv_cache_state;
    ov::Tensor m_pos_embed_weight;
    bool m_enable_thinking = true;
    bool m_is_chat_conversation = false;
    std::string m_system_message;
    ChatHistory m_history;
    std::vector<PreparedImageData> m_chat_images;

    void initialize_generation_defaults() {
        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }
        m_sampler.set_tokenizer(m_tokenizer);
        m_sampler.set_seed(m_generation_config.rng_seed);
    }

    void reset_language_state() {
        m_kv_cache_state.reset_state();
        m_language.reset_state();
        m_language.get_tensor("attention_mask").set_shape({1, 0});
    }

    ov::Tensor make_zero_prompt_visual_embeds(const ov::Shape& visual_mask_shape) const {
        OPENVINO_ASSERT(visual_mask_shape.size() == 2u, "Qwen3.5 visual_pos_mask must be rank 2");
        ov::Tensor zero_visual_embeds(
            ov::element::f32,
            {visual_mask_shape[0], visual_mask_shape[1], static_cast<size_t>(m_config.text.hidden_size)}
        );
        std::memset(zero_visual_embeds.data(), 0, zero_visual_embeds.get_byte_size());
        return zero_visual_embeds;
    }

    void setup_generation_config(GenerationConfig& generation_config) {
        OPENVINO_ASSERT(generation_config.num_return_sequences == 1u,
            "Qwen3.5 VLM backend currently supports num_return_sequences == 1 only");
        OPENVINO_ASSERT(generation_config.is_greedy_decoding() || generation_config.is_multinomial(),
            "Qwen3.5 VLM backend currently supports greedy or multinomial decoding only");
        if (generation_config.stop_token_ids.empty())
            generation_config.stop_token_ids = m_generation_config.stop_token_ids;
        if (generation_config.eos_token_id == -1)
            generation_config.set_eos_token_id(m_generation_config.eos_token_id);
        generation_config.validate();
    }

    std::vector<PreparedImageData> prepare_images(const std::vector<ov::Tensor>& images) {
        std::vector<PreparedImageData> prepared;
        auto flat_images = split_images(images);
        prepared.reserve(flat_images.size());
        for (const auto& image : flat_images) {
            auto vision_inputs = m_preprocessor.preprocess(image, m_pos_embed_weight);
            m_vision.set_tensor(Qwen3_5VisionIO::kPixelValues, vision_inputs.pixel_values);
            m_vision.set_tensor(Qwen3_5VisionIO::kGridThw, vision_inputs.grid_thw);
            m_vision.set_tensor(Qwen3_5VisionIO::kPosEmbeds, vision_inputs.pos_embeds);
            m_vision.set_tensor(Qwen3_5VisionIO::kRotaryCos, vision_inputs.rotary_cos);
            m_vision.set_tensor(Qwen3_5VisionIO::kRotarySin, vision_inputs.rotary_sin);
            m_vision.infer();

            PreparedImageData item;
            item.image = clone_tensor(image);
            item.visual_embeds = clone_tensor(m_vision.get_tensor(Qwen3_5VisionIO::kVisualEmbeds));
            item.grid_thw = clone_tensor(vision_inputs.grid_thw);
            item.image_tokens = Qwen3_5VisionPreprocessor::count_visual_tokens(item.grid_thw, m_config.vision.spatial_merge_size);
            prepared.push_back(std::move(item));
        }
        return prepared;
    }

    void append_ordered_chat_images(
        std::vector<PreparedImageData>& dst,
        const std::vector<PreparedImageData>& prepared_images,
        const std::vector<size_t>& image_sequence
    ) {
        for (size_t idx : image_sequence) {
            OPENVINO_ASSERT(idx < prepared_images.size(), "Missing image with index ", idx);
            dst.push_back(clone_prepared_image_data(prepared_images.at(idx)));
        }
    }

    std::string apply_chat_template(const ChatHistory& history) const {
        const JsonContainer extra_context({{"enable_thinking", m_enable_thinking}});
        return m_tokenizer.apply_chat_template(history, true, {}, std::nullopt, extra_context);
    }

    std::string build_inference_prompt(const std::string& prompt, bool has_images, bool force_user_template, bool& add_special_tokens) {
        if (force_user_template) {
            ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            add_special_tokens = false;
            return apply_chat_template(history);
        }
        if (m_tokenizer.get_chat_template().empty()) {
            if (has_images) {
                add_special_tokens = false;
                return std::string("<|im_start|>user\n") + prompt + "<|im_end|>\n<|im_start|>assistant\n";
            }
            add_special_tokens = true;
            return prompt;
        }

        ChatHistory history({{{"role", "user"}, {"content", prompt}}});
        add_special_tokens = false;
        return apply_chat_template(history);
    }

    VLMDecodedResults generate_single_prompt(
        const std::string& prompt,
        const std::vector<PreparedImageData>& prepared_images,
        const std::vector<size_t>& image_sequence,
        GenerationConfig generation_config,
        const StreamerVariant& streamer,
        bool allow_chat_template,
        float prepare_embeddings_duration_us = 0.0f
    ) {
        bool add_special_tokens = true;
        std::string inference_prompt = prompt;
        if (allow_chat_template && generation_config.apply_chat_template) {
            inference_prompt = build_inference_prompt(prompt, !prepared_images.empty(), true, add_special_tokens);
        } else if (!prepared_images.empty()) {
            inference_prompt = build_inference_prompt(prompt, true, false, add_special_tokens);
        }

        const auto tokenization_start = std::chrono::steady_clock::now();
        auto tokenized = m_tokenizer.encode(inference_prompt, ov::genai::add_special_tokens(add_special_tokens));
        const auto tokenization_end = std::chrono::steady_clock::now();
        return generate_tokenized(
            inference_prompt,
            tokenized.input_ids,
            tokenized.attention_mask,
            prepared_images,
            image_sequence,
            std::move(generation_config),
            streamer,
            PerfMetrics::get_microsec(tokenization_end - tokenization_start),
            prepare_embeddings_duration_us
        );
    }

    VLMDecodedResults generate_from_history(
        const ChatHistory& history,
        const std::vector<PreparedImageData>& prepared_images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer,
        float prepare_embeddings_duration_us = 0.0f
    ) {
        OPENVINO_ASSERT(!m_tokenizer.get_chat_template().empty(), "Qwen3.5 VLM chat mode requires a chat template");
        std::vector<size_t> image_sequence(prepared_images.size());
        std::iota(image_sequence.begin(), image_sequence.end(), 0);

        const std::string templated_history = apply_chat_template(history);
        const auto tokenization_start = std::chrono::steady_clock::now();
        auto tokenized = m_tokenizer.encode(templated_history, ov::genai::add_special_tokens(false));
        const auto tokenization_end = std::chrono::steady_clock::now();
        return generate_tokenized(
            templated_history,
            tokenized.input_ids,
            tokenized.attention_mask,
            prepared_images,
            image_sequence,
            std::move(generation_config),
            streamer,
            PerfMetrics::get_microsec(tokenization_end - tokenization_start),
            prepare_embeddings_duration_us
        );
    }

    VLMDecodedResults generate_tokenized(
        const std::string& prepared_prompt,
        const ov::Tensor& input_ids,
        const ov::Tensor& attention_mask,
        const std::vector<PreparedImageData>& prepared_images,
        const std::vector<size_t>& image_sequence,
        GenerationConfig generation_config,
        const StreamerVariant& streamer,
        float tokenization_duration_us,
        float prepare_embeddings_duration_us
    ) {
        (void)prepared_prompt;
        auto generate_start_time = std::chrono::steady_clock::now();
        VLMPerfMetrics perf_metrics;
        auto& raw_counters = perf_metrics.raw_metrics;
        raw_counters.tokenization_durations.emplace_back(tokenization_duration_us);
        perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.emplace_back(prepare_embeddings_duration_us);

        setup_generation_config(generation_config);
        reset_language_state();

        ov::Tensor ordered_visual_embeds;
        ov::Tensor ordered_grid_thw;
        std::vector<std::pair<std::string, ov::Tensor>> prompt_extra_inputs;
        std::vector<std::pair<std::string, ov::Tensor>> generation_extra_inputs;

        if (!prepared_images.empty()) {
            std::vector<ov::Tensor> visual_tensors;
            std::vector<ov::Tensor> grid_tensors;
            visual_tensors.reserve(image_sequence.size());
            grid_tensors.reserve(image_sequence.size());
            for (size_t idx : image_sequence) {
                OPENVINO_ASSERT(idx < prepared_images.size(), "Missing image with index ", idx);
                visual_tensors.push_back(prepared_images.at(idx).visual_embeds);
                grid_tensors.push_back(prepared_images.at(idx).grid_thw);
            }

            ordered_visual_embeds = concat_tensors_dim0(visual_tensors);
            ordered_grid_thw = concat_tensors_dim0(grid_tensors);
        }

        auto plan = m_input_planner.build_plan(input_ids, &attention_mask, prepared_images.empty() ? nullptr : &ordered_grid_thw);
        auto prompt_visual_embeds = prepared_images.empty()
            ? make_zero_prompt_visual_embeds(plan.visual_pos_mask.get_shape())
            : Qwen3_5InputPlanner::scatter_visual_embeds(ordered_visual_embeds, plan.visual_pos_mask);
        auto decode_visual_embeds = ov::Tensor(prompt_visual_embeds.get_element_type(), {1, 1, static_cast<size_t>(m_config.text.hidden_size)});
        std::memset(decode_visual_embeds.data(), 0, decode_visual_embeds.get_byte_size());
        auto decode_visual_mask = ov::Tensor(ov::element::boolean, {1, 1});
        std::memset(decode_visual_mask.data(), 0, decode_visual_mask.get_byte_size());

        prompt_extra_inputs.emplace_back(Qwen3_5TextIO::kVisualEmbeds, std::move(prompt_visual_embeds));
        prompt_extra_inputs.emplace_back(Qwen3_5TextIO::kVisualPosMask, plan.visual_pos_mask);
        generation_extra_inputs.emplace_back(Qwen3_5TextIO::kVisualEmbeds, std::move(decode_visual_embeds));
        generation_extra_inputs.emplace_back(Qwen3_5TextIO::kVisualPosMask, std::move(decode_visual_mask));

        return finalize_generation(
            input_ids,
            attention_mask,
            std::move(generation_config),
            streamer,
            perf_metrics,
            std::move(prompt_extra_inputs),
            std::move(generation_extra_inputs),
            plan.position_ids,
            plan.rope_deltas,
            generate_start_time,
            raw_counters
        );
    }

    VLMDecodedResults finalize_generation(
        const ov::Tensor& input_ids,
        const ov::Tensor& attention_mask,
        GenerationConfig generation_config,
        const StreamerVariant& streamer,
        VLMPerfMetrics& perf_metrics,
        std::vector<std::pair<std::string, ov::Tensor>> prompt_extra_inputs,
        std::vector<std::pair<std::string, ov::Tensor>> generation_extra_inputs,
        const ov::Tensor& position_ids,
        const ov::Tensor& rope_deltas,
        const std::chrono::steady_clock::time_point& generate_start_time,
        RawPerfMetrics& raw_counters
    ) {
        const size_t batch = input_ids.get_shape().at(0);
        OPENVINO_ASSERT(batch == 1u, "Qwen3.5 VLM backend currently supports batch size 1 only");
        perf_metrics.num_input_tokens = input_ids.get_size();

        std::shared_ptr<StreamerBase> streamer_ptr = utils::create_streamer(streamer, m_tokenizer);
        OPENVINO_ASSERT(streamer_ptr == nullptr || (generation_config.is_greedy_decoding() || generation_config.is_multinomial()),
            "Currently streaming is possible only for greedy or multinomial decoding");
        VLMDecodedResults decoded;
        const size_t max_new_tokens = generation_config.get_max_new_tokens(input_ids.get_shape().at(1));
        if (max_new_tokens == 0) {
            decoded.texts.push_back(std::string{});
            decoded.scores.push_back(0.0f);
            decoded.perf_metrics = perf_metrics;
            decoded.perf_metrics.load_time = this->get_load_time();
            decoded.perf_metrics.m_evaluated = false;
            decoded.perf_metrics.evaluate_statistics(generate_start_time);
            return decoded;
        }

        std::vector<int64_t> prompt_token_ids(input_ids.data<const int64_t>(), input_ids.data<const int64_t>() + input_ids.get_size());
        ov::genai::GenerationConfig penalty_config;
        penalty_config.do_sample = false;
        penalty_config.repetition_penalty = generation_config.repetition_penalty;
        penalty_config.frequency_penalty = generation_config.frequency_penalty;
        penalty_config.presence_penalty = generation_config.presence_penalty;
        ov::genai::LogitProcessor penalty_processor(penalty_config, prompt_token_ids);

        const bool use_sampling = generation_config.is_multinomial();
        std::mt19937 rng(generation_config.rng_seed != 0
            ? static_cast<std::mt19937::result_type>(generation_config.rng_seed)
            : std::random_device{}());
        SamplingContext sampling_ctx;
        std::vector<float> logit_buf;
        const auto stop_token_ids = resolve_stop_token_ids(generation_config, m_tokenizer);

        const ov::Tensor beam_idx = make_beam_idx(batch);
        m_language.set_tensor(Qwen3_5TextIO::kInputIds, input_ids);
        m_language.set_tensor(Qwen3_5TextIO::kAttentionMask, attention_mask);
        m_language.set_tensor(Qwen3_5TextIO::kPositionIds, position_ids);
        m_language.set_tensor(Qwen3_5TextIO::kBeamIdx, beam_idx);
        for (const auto& [tensor_name, tensor] : prompt_extra_inputs) {
            m_language.set_tensor(tensor_name, tensor);
        }

        const auto prefill_start = std::chrono::steady_clock::now();
        m_language.infer();
        const auto prefill_end = std::chrono::steady_clock::now();
        raw_counters.m_new_token_times.emplace_back(prefill_end);
        raw_counters.m_batch_sizes.emplace_back(1);
        raw_counters.m_token_infer_durations.emplace_back(PerfMetrics::get_microsec(prefill_end - prefill_start));
        raw_counters.m_inference_durations.emplace_back(PerfMetrics::get_microsec(prefill_end - prefill_start));

        ov::Tensor logits = m_language.get_tensor(Qwen3_5TextIO::kLogits);
        extract_last_logits_f32(logits, logit_buf);
        int64_t next_id;
        {
            ov::genai::Logits logits_wrapper(logit_buf.data(), logit_buf.size());
            penalty_processor.apply(logits_wrapper);
            next_id = use_sampling
                ? sample_fast(logit_buf.data(), logit_buf.size(), generation_config.temperature, generation_config.top_p, generation_config.top_k, rng, sampling_ctx)
                : argmax_f32(logit_buf);
        }
        penalty_processor.register_new_generated_token(next_id);
        penalty_processor.update_generated_len(1);

        std::vector<int64_t> generated;
        generated.push_back(next_id);

        bool streaming_cancelled = false;
        bool stop_requested = false;
        if (streamer_ptr) {
            const auto streaming_status = streamer_ptr->write(next_id);
            if (streaming_status == StreamingStatus::CANCEL) {
                streaming_cancelled = true;
                stop_requested = true;
            } else if (streaming_status == StreamingStatus::STOP) {
                stop_requested = true;
            }
        }

        int64_t past_len = get_active_prompt_length(attention_mask);
        const auto* rope_deltas_data = rope_deltas.get_size() > 0 ? rope_deltas.data<const int64_t>() : nullptr;
        ov::Tensor step_ids(ov::element::i64, {batch, 1});
        ov::Tensor step_mask(ov::element::i64, {batch, 1});
        std::fill_n(step_mask.data<int64_t>(), batch, int64_t{1});
        ov::Tensor decode_position_ids(ov::element::i64, {3, batch, 1});

        for (size_t step = 1; step < max_new_tokens && !stop_requested; ++step) {
            if (generated.size() >= generation_config.min_new_tokens && stop_token_ids.count(next_id) > 0) {
                break;
            }

            auto* step_data = step_ids.data<int64_t>();
            for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
                step_data[batch_idx] = next_id;
                const int64_t rope_delta = rope_deltas_data ? rope_deltas_data[batch_idx] : 0;
                const int64_t value = past_len + rope_delta;
                auto* pos_data = decode_position_ids.data<int64_t>();
                pos_data[batch_idx] = value;
                pos_data[batch + batch_idx] = value;
                pos_data[2 * batch + batch_idx] = value;
            }

            m_language.set_tensor(Qwen3_5TextIO::kInputIds, step_ids);
            m_language.set_tensor(Qwen3_5TextIO::kAttentionMask, step_mask);
            m_language.set_tensor(Qwen3_5TextIO::kPositionIds, decode_position_ids);
            m_language.set_tensor(Qwen3_5TextIO::kBeamIdx, beam_idx);
            for (const auto& [tensor_name, tensor] : generation_extra_inputs) {
                m_language.set_tensor(tensor_name, tensor);
            }

            const auto decode_step_start = std::chrono::steady_clock::now();
            m_language.infer();
            const auto decode_step_end = std::chrono::steady_clock::now();
            raw_counters.m_new_token_times.emplace_back(decode_step_end);
            raw_counters.m_batch_sizes.emplace_back(1);
            raw_counters.m_token_infer_durations.emplace_back(PerfMetrics::get_microsec(decode_step_end - decode_step_start));
            raw_counters.m_inference_durations.emplace_back(PerfMetrics::get_microsec(decode_step_end - decode_step_start));

            logits = m_language.get_tensor(Qwen3_5TextIO::kLogits);
            extract_last_logits_f32(logits, logit_buf);
            {
                ov::genai::Logits logits_wrapper(logit_buf.data(), logit_buf.size());
                penalty_processor.apply(logits_wrapper);
                next_id = use_sampling
                    ? sample_fast(logit_buf.data(), logit_buf.size(), generation_config.temperature, generation_config.top_p, generation_config.top_k, rng, sampling_ctx)
                    : argmax_f32(logit_buf);
            }
            penalty_processor.register_new_generated_token(next_id);
            penalty_processor.update_generated_len(generated.size() + 1);
            generated.push_back(next_id);
            past_len += 1;

            if (streamer_ptr) {
                const auto streaming_status = streamer_ptr->write(next_id);
                if (streaming_status == StreamingStatus::CANCEL) {
                    streaming_cancelled = true;
                    stop_requested = true;
                } else if (streaming_status == StreamingStatus::STOP) {
                    stop_requested = true;
                }
            }
        }

        if (streamer_ptr) {
            streamer_ptr->end();
        }

        if (streaming_cancelled) {
            reset_language_state();
        }

        auto decode_start_time = std::chrono::steady_clock::now();
        if (!streaming_cancelled) {
            decoded.texts.push_back(generated.empty() ? std::string{} : m_tokenizer.decode(generated));
            decoded.scores.push_back(0.0f);
        }
        auto decode_end_time = std::chrono::steady_clock::now();
        auto generate_end_time = std::chrono::steady_clock::now();

        decoded.perf_metrics = perf_metrics;
        auto& res_raw_counters = decoded.perf_metrics.raw_metrics;
        decoded.perf_metrics.load_time = this->get_load_time();
        decoded.perf_metrics.num_input_tokens = perf_metrics.num_input_tokens;
        decoded.perf_metrics.num_generated_tokens = generated.size();
        res_raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(generate_end_time - generate_start_time));
        res_raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_end_time - decode_start_time));
        decoded.perf_metrics.m_evaluated = false;
        decoded.perf_metrics.evaluate_statistics(generate_start_time);
        return decoded;
    }
};

}  // namespace

namespace ov::genai {

std::unique_ptr<VLMPipeline::VLMPipelineBase> make_qwen3_5_vl_pipeline(
    const std::filesystem::path& models_dir,
    const std::string& device,
    const ov::AnyMap& properties
) {
    return std::make_unique<Qwen3_5VLStatefulPipeline>(models_dir, device, properties);
}

std::unique_ptr<VLMPipeline::VLMPipelineBase> make_qwen3_5_vl_pipeline(
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap& properties,
    const GenerationConfig& generation_config
) {
    return std::make_unique<Qwen3_5VLStatefulPipeline>(models_map, tokenizer, config_dir_path, device, properties, generation_config);
}

}  // namespace ov::genai