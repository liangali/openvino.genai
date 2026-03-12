// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <fstream>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/openvino.hpp>

#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/generation_config.hpp"
#include "load_image.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"

namespace {

struct SampleOptions {
    std::optional<std::filesystem::path> model_dir;
    std::string mode;
    std::filesystem::path image_path;
    std::string user_prompt;
    std::optional<std::filesystem::path> prompt_file;
    std::string device = "GPU";
    int max_new_tokens = 64;
    bool cache_model = false;

    std::string dummy_model = "dense";

    std::optional<int> num_layers;
    int max_pixels = 0;
};

bool has_safetensors_file(const std::filesystem::path& model_dir) {
    if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir)) {
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".safetensors") {
            return true;
        }
    }
    return false;
}

bool has_ir_model_pair(const std::filesystem::path& xml_path, const std::filesystem::path& bin_path) {
    return std::filesystem::exists(xml_path) && std::filesystem::is_regular_file(xml_path) &&
           std::filesystem::exists(bin_path) && std::filesystem::is_regular_file(bin_path);
}

bool has_model_input_name(const std::shared_ptr<ov::Model>& model, const std::string& input_name) {
    if (!model) {
        return false;
    }
    for (const auto& input : model->inputs()) {
        const auto names = input.get_names();
        if (names.find(input_name) != names.end()) {
            return true;
        }
    }
    return false;
}

bool is_vl_text_ir_compatible(const std::shared_ptr<ov::Model>& model) {
    return has_model_input_name(model, ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds) &&
           has_model_input_name(model, ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask);
}

int parse_i32(const std::string& raw, const char* option_name) {
    try {
        return std::stoi(raw);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("Invalid integer for ") + option_name + ": " + raw);
    }
}

std::string to_lower(std::string value) {
    for (auto& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

void print_usage(const char* argv0) {
    std::cout
        << "Qwen3.5 Modeling Sample CLI Helper\n"
        << "==================================\n\n"
        << "Overview:\n"
        << "  - Run Qwen3.5 text-only or vision-language generation.\n"
        << "  - If --model is set: load real HF weights from that directory.\n"
        << "  - If --model is not set: use synthetic dummy weights.\n\n"
        << "Primary Usage:\n"
        << "  " << argv0 << " [--model PATH] [--mode text|vl] [--image IMAGE_PATH] [--prompt TEXT]\n\n"
        << "Examples:\n"
        << "  1) Dummy text:\n"
        << "     " << argv0 << " --mode text --prompt \"Hello\"\n"
        << "  2) Dummy VL (uses built-in 256x256 dummy image):\n"
        << "     " << argv0 << " --mode vl --dummy-model moe --prompt \"Describe this image\"\n"
        << "  3) Real HF text:\n"
        << "     " << argv0 << " --model C:\\models\\Qwen3.5 --mode text\n"
        << "  4) Real HF VL:\n"
        << "     " << argv0 << " --model C:\\models\\Qwen3.5 --mode vl --image C:\\img\\a.jpg\n\n"
        << "Options:\n"
        << "  --model PATH                    HF model directory (config.json + *.safetensors). Omit for dummy weights\n"
        << "  --mode text|vl                  Run path (default: text)\n"
        << "  --image PATH                    Image path for vl mode. Required with --model --mode vl\n"
        << "                                  In dummy vl mode, if omitted, built-in 256x256 dummy image is used\n"
        << "  --prompt TEXT                   User prompt (inline text)\n"
        << "  --prompt-file PATH              User prompt read from a file (use for long prompts to avoid CLI length limits)\n"
        << "  --device NAME                   OpenVINO device name (default: GPU)\n"
        << "  --output-tokens N               Number of generated tokens (default: 64)\n"
        << "  --cache-model                   Enable model caching behavior.\n"
        << "                                  HF mode: load cached IR from --model dir if exists, else build+save there\n"
        << "                                  Dummy mode: save built IR to app executable folder\n"
        << "                                  IR filename includes concise quant signature when quantization is enabled\n"
        << "  Quantization for both real and dummy mode is controlled only by environment variables:\n"
        << "    OV_GENAI_INFLIGHT_QUANT_MODE\n"
        << "    OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE\n"
        << "    OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE\n"
        << "  --dummy-model MODE              Dummy mode only: dense | moe (default: dense)\n"
    << "  --num-layers N                  Run only the first N text transformer layers (dummy + real model)\n"
    << "  --max-pixels N                  Limit vision input to N pixels (default: from preprocessor_config.json)\n"
    << "                                  Recommended: 602112 (3072 tokens * 14^2 patch) for ARL-H GPU\n"
        << "  -h, --help                      Show this helper\n";
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

SampleOptions parse_cli(int argc, char* argv[]) {
    SampleOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg.rfind("--", 0) != 0) {
            throw std::runtime_error("Positional arguments are not supported. Use --help and pass named options.");
        }

        auto take_value = [&](const char* option_name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + option_name);
            }
            return argv[++i];
        };

        if (arg == "--model") {
            opts.model_dir = take_value("--model");
        } else if (arg == "--mode") {
            opts.mode = take_value("--mode");
        } else if (arg == "--image") {
            opts.image_path = take_value("--image");
        } else if (arg == "--prompt") {
            opts.user_prompt = take_value("--prompt");
        } else if (arg == "--prompt-file") {
            opts.prompt_file = take_value("--prompt-file");
        } else if (arg == "--device") {
            opts.device = take_value("--device");
        } else if (arg == "--output-tokens") {
            opts.max_new_tokens = parse_i32(take_value("--output-tokens"), "--output-tokens");
        } else if (arg == "--cache-model") {
            opts.cache_model = true;
        } else if (arg == "--dummy-model") {
            opts.dummy_model = take_value("--dummy-model");
        } else if (arg == "--num-layers") {
            opts.num_layers = parse_i32(take_value("--num-layers"), "--num-layers");
        } else if (arg == "--max-pixels") {
            opts.max_pixels = parse_i32(take_value("--max-pixels"), "--max-pixels");
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (opts.mode.empty()) {
        opts.mode = "text";
    }
    opts.mode = to_lower(opts.mode);
    if (opts.mode != "text" && opts.mode != "vl") {
        throw std::runtime_error("Mode must be 'text' or 'vl'");
    }
    opts.dummy_model = to_lower(opts.dummy_model);
    if (opts.dummy_model != "dense" && opts.dummy_model != "moe") {
        throw std::runtime_error("dummy-model must be 'dense' or 'moe'");
    }
    if (!opts.image_path.empty() && opts.mode != "vl") {
        throw std::runtime_error("--image can only be used with --mode vl");
    }

    if (opts.max_new_tokens <= 0) {
        throw std::runtime_error("max_new_tokens must be > 0");
    }
    if (opts.prompt_file.has_value()) {
        std::ifstream ifs(*opts.prompt_file);
        if (!ifs) {
            throw std::runtime_error("Cannot open prompt file: " + opts.prompt_file->string());
        }
        opts.user_prompt = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
        if (opts.user_prompt.empty()) {
            throw std::runtime_error("Prompt file is empty: " + opts.prompt_file->string());
        }
    }
    if (opts.user_prompt.empty()) {
        opts.user_prompt = (opts.mode == "vl") ? "Describe the image." : "Write one sentence about OpenVINO.";
    }

    return opts;
}

void apply_text_config_overrides(ov::genai::modeling::models::Qwen3_5Config& cfg, const SampleOptions& opts) {
    if (!opts.num_layers.has_value()) {
        return;
    }
    if (*opts.num_layers <= 0) {
        throw std::runtime_error("--num-layers must be > 0");
    }

    if (*opts.num_layers > cfg.text.num_hidden_layers) {
        throw std::runtime_error("--num-layers must be <= model num_hidden_layers (" +
                                 std::to_string(cfg.text.num_hidden_layers) + "), got: " +
                                 std::to_string(*opts.num_layers));
    }

    cfg.text.num_hidden_layers = *opts.num_layers;
    if (!cfg.text.layer_types.empty()) {
        if (cfg.text.layer_types.size() >= static_cast<size_t>(*opts.num_layers)) {
            cfg.text.layer_types.resize(static_cast<size_t>(*opts.num_layers));
        } else {
            cfg.text.layer_types.clear();
        }
    }

    cfg.finalize();
    cfg.validate();
}

std::string build_vl_prompt(const std::string& user_prompt, int64_t image_tokens) {
    std::string prompt = "<|im_start|>user\n<|vision_start|>";
    prompt.reserve(prompt.size() + static_cast<size_t>(image_tokens) * 12 + user_prompt.size() + 64);
    for (int64_t i = 0; i < image_tokens; ++i) {
        prompt += "<|image_pad|>";
    }
    prompt += "<|vision_end|>\n";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

std::set<int64_t> resolve_stop_token_ids(const std::filesystem::path& model_dir,
                                         const ov::genai::Tokenizer* tokenizer) {
    std::set<int64_t> stop_token_ids;

    if (!model_dir.empty()) {
        const auto generation_config_path = model_dir / "generation_config.json";
        if (std::filesystem::exists(generation_config_path) && std::filesystem::is_regular_file(generation_config_path)) {
            try {
                ov::genai::GenerationConfig generation_config(generation_config_path);
                stop_token_ids = generation_config.stop_token_ids;
            } catch (const std::exception& e) {
                std::cerr << "Failed to load generation_config.json: " << e.what() << std::endl;
                std::cerr << "Falling back to tokenizer eos_token_id..." << std::endl;
            }
        }
    }

    if (stop_token_ids.empty() && tokenizer) {
        const int64_t eos_token_id = tokenizer->get_eos_token_id();
        if (eos_token_id >= 0) {
            stop_token_ids.insert(eos_token_id);
        }
    }

    return stop_token_ids;
}

int64_t argmax_last_token(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("logits must have shape [1, S, V]");
    }
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    const size_t offset = (seq_len - 1) * vocab;

    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<const ov::float16>() + offset;
        ov::float16 max_val = data[0];
        size_t max_idx = 0;
        for (size_t i = 1; i < vocab; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<int64_t>(max_idx);
    }
    if (logits.get_element_type() == ov::element::bf16) {
        const auto* data = logits.data<const ov::bfloat16>() + offset;
        ov::bfloat16 max_val = data[0];
        size_t max_idx = 0;
        for (size_t i = 1; i < vocab; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<int64_t>(max_idx);
    }
    if (logits.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Unsupported logits dtype");
    }
    const auto* data = logits.data<const float>() + offset;
    float max_val = data[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return static_cast<int64_t>(max_idx);
}

ov::Tensor make_beam_idx(size_t batch) {
    ov::Tensor beam_idx(ov::element::i32, {batch});
    auto* data = beam_idx.data<int32_t>();
    for (size_t i = 0; i < batch; ++i) {
        data[i] = static_cast<int32_t>(i);
    }
    return beam_idx;
}

ov::Tensor make_zero_tensor(const ov::element::Type& type, const ov::Shape& shape) {
    ov::Tensor tensor(type, shape);
    std::memset(tensor.data(), 0, tensor.get_byte_size());
    return tensor;
}

// ---------------------------------------------------------------------------
// USM-host tensor helpers — iGPU zero-copy optimization
// ---------------------------------------------------------------------------
// On iGPU, GPU can access USM-host memory directly without H2D copy.
// This eliminates the costly wait_for_events overhead for each input tensor.
// Falls back to standard ov::Tensor when the context doesn't support it.

/// Try to get the GPU RemoteContext from a CompiledModel.
std::optional<ov::RemoteContext> try_get_gpu_context(ov::CompiledModel& compiled) {
    try {
        return compiled.get_context();
    } catch (...) {
        return std::nullopt;
    }
}

/// Create a USM-host tensor of given type/shape via the GPU context.
ov::Tensor make_usm_host_tensor(std::optional<ov::RemoteContext>& ctx,
                                const ov::element::Type& type,
                                const ov::Shape& shape) {
    if (ctx.has_value()) {
        try {
            return ctx->create_host_tensor(type, shape);
        } catch (...) {}
    }
    return ov::Tensor(type, shape);
}

/// Create a USM-host tensor and copy data from an existing tensor.
ov::Tensor clone_as_usm_host(std::optional<ov::RemoteContext>& ctx,
                              const ov::Tensor& src) {
    auto dst = make_usm_host_tensor(ctx, src.get_element_type(), src.get_shape());
    std::memcpy(dst.data(), src.data(), src.get_byte_size());
    return dst;
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
    throw std::runtime_error("Failed to locate visual.pos_embed.weight");
}

// Name used for the extra pos_embed Result in cached vision IR
static constexpr const char* kPosEmbedCacheResultName = "__pos_embed_cache__";

// Embed pos_embed weight as a Constant->Result in the vision model so that
// it is serialized into the .bin alongside other vision weights.
void embed_pos_embed_in_vision_model(std::shared_ptr<ov::Model>& model,
                                     const ov::Tensor& pos_embed) {
    auto constant = std::make_shared<ov::op::v0::Constant>(pos_embed);
    auto result = std::make_shared<ov::op::v0::Result>(constant);
    result->set_friendly_name(kPosEmbedCacheResultName);
    model->add_results({result});
}

// Extract the cached pos_embed Constant from a vision model loaded from IR,
// then remove the extra Result so it doesn't affect inference.
ov::Tensor extract_pos_embed_from_vision_model(std::shared_ptr<ov::Model>& model) {
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() == kPosEmbedCacheResultName) {
            auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                result->input_value(0).get_node_shared_ptr());
            if (!const_node) {
                throw std::runtime_error("pos_embed cache result is not a Constant");
            }
            ov::Tensor tensor(const_node->get_element_type(), const_node->get_shape());
            std::memcpy(tensor.data(), const_node->get_data_ptr(), tensor.get_byte_size());
            model->remove_result(result);
            return tensor;
        }
    }
    throw std::runtime_error("Cached vision IR does not contain pos_embed data");
}

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<int64_t> build_dummy_prompt_ids(const ov::genai::modeling::models::Qwen3_5Config& cfg,
                                            const std::string& prompt,
                                            bool with_vision,
                                            int64_t image_tokens) {
    std::vector<int64_t> ids;
    ids.reserve(static_cast<size_t>(prompt.size()) + 16 + static_cast<size_t>(std::max<int64_t>(0, image_tokens)));

    if (with_vision) {
        ids.push_back(cfg.vision_start_token_id);
        for (int64_t i = 0; i < image_tokens; ++i) {
            ids.push_back(cfg.image_token_id);
        }
        ids.push_back(cfg.vision_end_token_id);
    }

    const int64_t vocab_base = std::max<int64_t>(100, cfg.text.vocab_size - 100);
    for (unsigned char c : prompt) {
        ids.push_back(static_cast<int64_t>(c % vocab_base));
    }
    if (ids.empty()) {
        ids = {1, 2, 3, 4, 5, 6};
    }
    return ids;
}

std::pair<ov::Tensor, ov::Tensor> to_batch_tensors(const std::vector<int64_t>& ids) {
    ov::Tensor input_ids(ov::element::i64, {1, ids.size()});
    std::memcpy(input_ids.data(), ids.data(), ids.size() * sizeof(int64_t));

    ov::Tensor attention_mask(ov::element::i64, {1, ids.size()});
    auto* mask = attention_mask.data<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) {
        mask[i] = 1;
    }
    return {input_ids, attention_mask};
}

ov::Tensor make_dummy_image() {
    ov::Tensor image(ov::element::u8, {1, 256, 256, 3});
    std::memset(image.data(), 127, image.get_byte_size());
    return image;
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc == 1) {
        print_usage(argv[0]);
        return 0;
    }

    SampleOptions opts;
    try {
        opts = parse_cli(argc, argv);
    } catch (const std::exception& cli_error) {
        std::cerr << "Command line error: " << cli_error.what() << '\n' << '\n';
        print_usage(argv[0]);
        return 1;
    }

    const bool use_vl = (opts.mode == "vl");
    const bool use_dummy_mode_flag = !opts.model_dir.has_value();
    if (!use_dummy_mode_flag && use_vl && opts.image_path.empty()) {
        std::cerr << "Command line error: Real weights + --mode vl requires --image PATH." << '\n' << '\n';
        print_usage(argv[0]);
        return 1;
    }
    const std::filesystem::path model_dir = opts.model_dir.value_or(std::filesystem::path{});
    if (!use_dummy_mode_flag) {
        if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir)) {
            std::cerr << "Command line error: --model must point to an existing directory." << '\n' << '\n';
            print_usage(argv[0]);
            return 1;
        }
        const bool has_config = std::filesystem::exists(model_dir / "config.json");
        if (!has_config || !has_safetensors_file(model_dir)) {
            std::cerr << "Command line error: --model must contain config.json and at least one *.safetensors file."
                      << '\n'
                      << '\n';
            print_usage(argv[0]);
            return 1;
        }
    }
    if (use_vl && !opts.image_path.empty() && !std::filesystem::exists(opts.image_path)) {
        std::cerr << "Command line error: Image path does not exist: " << opts.image_path.string() << '\n' << '\n';
        print_usage(argv[0]);
        return 1;
    }

    ov::genai::modeling::models::Qwen3_5Config cfg;
    if (use_dummy_mode_flag) {
        cfg = (opts.dummy_model == "moe")
                  ? ov::genai::modeling::models::Qwen3_5Config::make_dummy_moe35b_config()
                  : ov::genai::modeling::models::Qwen3_5Config::make_dummy_dense9b_config();
    } else {
        cfg = ov::genai::modeling::models::Qwen3_5Config::from_json_file(model_dir);
    }
    apply_text_config_overrides(cfg, opts);

    ov::genai::modeling::weights::QuantizationConfig vision_quant_config;
    ov::genai::modeling::weights::QuantizationConfig text_quant_config;
    const auto shared_quant_config = ov::genai::modeling::weights::parse_quantization_config_from_env();
    if (shared_quant_config.enabled() && shared_quant_config.group_size <= 0) {
        throw std::runtime_error(
            "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE must be > 0 when OV_GENAI_INFLIGHT_QUANT_MODE is enabled");
    }
    // vision_quant_config = shared_quant_config;  // disabled: GPU compile_model hangs with many INT4 dequant subgraphs in vision encoder
    text_quant_config = shared_quant_config;
    std::cout << "[quant] env mode=" << quant_mode_cache_token(shared_quant_config.mode)
              << ", backup=" << quant_mode_cache_token(shared_quant_config.backup_mode)
              << ", group_size=" << shared_quant_config.group_size << std::endl;

    ov::genai::modeling::models::Qwen3_5VisionPreprocessConfig pre_cfg;
    const auto pre_cfg_path = model_dir / "preprocessor_config.json";
    if (!use_dummy_mode_flag && std::filesystem::exists(pre_cfg_path)) {
        pre_cfg = ov::genai::modeling::models::Qwen3_5VisionPreprocessConfig::from_json_file(pre_cfg_path);
    }
    if (opts.max_pixels > 0) {
        std::cout << "[max-pixels] Overriding pre_cfg.max_pixels: " << pre_cfg.max_pixels
                  << " -> " << opts.max_pixels << std::endl;
        pre_cfg.max_pixels = static_cast<int64_t>(opts.max_pixels);
    }

    const std::filesystem::path app_dir = [&]() {
        std::filesystem::path exe_path = argv[0];
        if (exe_path.is_relative()) {
            exe_path = std::filesystem::absolute(exe_path);
        }
        if (exe_path.has_parent_path()) {
            return exe_path.parent_path();
        }
        return std::filesystem::current_path();
    }();
    const std::filesystem::path ir_dir = use_dummy_mode_flag ? app_dir : model_dir;
    std::string text_ir_stem = (use_vl ? "qwen3_5_text_vl" : "qwen3_5_text") + quant_cache_suffix(text_quant_config);
    if (opts.num_layers.has_value()) {
        text_ir_stem += "_l" + std::to_string(*opts.num_layers);
    }
    std::string vision_ir_stem = "qwen3_5_vision" + quant_cache_suffix(vision_quant_config);
    const auto text_xml_path = ir_dir / (text_ir_stem + ".xml");
    const auto text_bin_path = ir_dir / (text_ir_stem + ".bin");
    const auto vision_xml_path = ir_dir / (vision_ir_stem + ".xml");
    const auto vision_bin_path = ir_dir / (vision_ir_stem + ".bin");

    const bool load_text_from_ir = opts.cache_model && !use_dummy_mode_flag && has_ir_model_pair(text_xml_path, text_bin_path);
    const bool load_vision_from_ir =
        opts.cache_model && !use_dummy_mode_flag && use_vl && has_ir_model_pair(vision_xml_path, vision_bin_path);

    ov::Core core;
    std::unique_ptr<ov::genai::modeling::weights::WeightSource> source;
    auto ensure_weight_source = [&]() -> ov::genai::modeling::weights::WeightSource& {
        if (!source) {
            if (use_dummy_mode_flag) {
                constexpr uint32_t kDummySeed = 2026u;
                constexpr float kDummyInitRange = 0.02f;
                auto specs = use_vl ? ov::genai::modeling::models::build_qwen3_5_vlm_weight_specs(cfg)
                                    : ov::genai::modeling::models::build_qwen3_5_text_weight_specs(cfg.text);
                source = std::make_unique<ov::genai::modeling::weights::SyntheticWeightSource>(
                    std::move(specs),
                    kDummySeed,
                    -kDummyInitRange,
                    kDummyInitRange);
            } else {
                auto data = ov::genai::safetensors::load_safetensors(model_dir);
                source = std::make_unique<ov::genai::safetensors::SafetensorsWeightSource>(std::move(data));
            }
        }
        return *source;
    };

    std::shared_ptr<ov::Model> vision_model;
    std::shared_ptr<ov::Model> text_model;
    ov::Tensor cached_pos_embed;  // extracted from cached vision IR, or obtained from weight_source
    if (load_vision_from_ir) {
        std::cout << "[cache-model] Reusing cached vision IR: " << vision_xml_path << std::endl;
        vision_model = core.read_model(vision_xml_path.string(), vision_bin_path.string());
        // Extract pos_embed from the cached vision IR and remove the extra Result
        cached_pos_embed = extract_pos_embed_from_vision_model(vision_model);
        std::cout << "[cache-model] Extracted pos_embed from vision IR" << std::endl;
    } else if (use_vl) {
        auto& weight_source = ensure_weight_source();
        ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer(vision_quant_config);
        vision_model = ov::genai::modeling::models::create_qwen3_5_vision_model(cfg, weight_source, vision_finalizer);
        if (opts.cache_model) {
            // Embed pos_embed as a Constant->Result in the vision model so it's saved in the .bin
            const std::string pe_name = resolve_pos_embed_name(weight_source);
            cached_pos_embed = weight_source.get_tensor(pe_name);
            embed_pos_embed_in_vision_model(vision_model, cached_pos_embed);
            ov::serialize(vision_model, vision_xml_path.string(), vision_bin_path.string());
            std::cout << "[cache-model] Saved vision IR (with pos_embed): " << vision_xml_path << std::endl;
            // Remove the extra Result before compile_model
            for (const auto& result : vision_model->get_results()) {
                if (result->get_friendly_name() == kPosEmbedCacheResultName) {
                    vision_model->remove_result(result);
                    break;
                }
            }
        }
    }

    if (load_text_from_ir) {
        std::cout << "[cache-model] Reusing cached text IR: " << text_xml_path << std::endl;
        text_model = core.read_model(text_xml_path.string(), text_bin_path.string());
        if (use_vl && !is_vl_text_ir_compatible(text_model)) {
            std::cout << "[cache-model] Cached text IR is not VL-compatible (missing visual inputs), rebuilding: "
                      << text_xml_path << std::endl;
            text_model.reset();
        }
    }
    if (!text_model) {
        auto& weight_source = ensure_weight_source();
        ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer(text_quant_config);
        text_model = ov::genai::modeling::models::create_qwen3_5_text_model(
            cfg,
            weight_source,
            text_finalizer,
            false,
            use_vl);
        if (opts.cache_model) {
            ov::serialize(text_model, text_xml_path.string(), text_bin_path.string());
            std::cout << "[cache-model] Saved text IR: " << text_xml_path << std::endl;
        }
    }

    if (use_dummy_mode_flag && source) {
        source->release_all_cached_tensors();
        if (!use_vl) {
            source.reset();
        }
    }

    std::optional<ov::CompiledModel> compiled_vision;
    if (use_vl) {
        std::string vision_device = opts.device;
        if (const char* env_vision_dev = std::getenv("OV_GENAI_VISION_DEVICE")) {
            vision_device = env_vision_dev;
        }
        std::cout << "[vision] Compiling vision model on device: " << vision_device << std::endl;
        compiled_vision = core.compile_model(vision_model, vision_device);
    }
    auto compiled_text = core.compile_model(text_model, opts.device);

    ov::Tensor visual_embeds;
    ov::Tensor grid_thw;
    if (use_vl) {
        ov::Tensor image = opts.image_path.empty() ? make_dummy_image() : utils::load_image(opts.image_path);
        ov::Tensor pos_embed_weight;
        if (cached_pos_embed) {
            pos_embed_weight = cached_pos_embed;
        } else {
            auto& weight_source = ensure_weight_source();
            const std::string pos_embed_name = resolve_pos_embed_name(weight_source);
            pos_embed_weight = weight_source.get_tensor(pos_embed_name);
        }

        ov::genai::modeling::models::Qwen3_5VisionPreprocessor preprocessor(cfg.vision, pre_cfg);
        const auto preprocess_start = std::chrono::steady_clock::now();
        auto vision_inputs = preprocessor.preprocess(image, pos_embed_weight);
        const auto preprocess_end = std::chrono::steady_clock::now();

        auto vision_request = compiled_vision->create_infer_request();
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kPixelValues, vision_inputs.pixel_values);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kGridThw, vision_inputs.grid_thw);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kPosEmbeds, vision_inputs.pos_embeds);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kRotaryCos, vision_inputs.rotary_cos);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kRotarySin, vision_inputs.rotary_sin);
        const auto vision_start = std::chrono::steady_clock::now();
        vision_request.infer();
        const auto vision_end = std::chrono::steady_clock::now();

        visual_embeds = vision_request.get_tensor(ov::genai::modeling::models::Qwen3_5VisionIO::kVisualEmbeds);
        grid_thw = vision_inputs.grid_thw;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[vision] preprocess: " << elapsed_ms(preprocess_start, preprocess_end) << " ms" << std::endl;
        std::cout << "[vision] encode: " << elapsed_ms(vision_start, vision_end) << " ms" << std::endl;

        if (use_dummy_mode_flag && source) {
            source->release_all_cached_tensors();
            source.reset();
        }
    }

    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    if (!use_dummy_mode_flag) {
        try {
            tokenizer = std::make_unique<ov::genai::Tokenizer>(model_dir);
        } catch (const std::exception&) {
            tokenizer.reset();
        }
    }

    ov::Tensor input_ids;
    ov::Tensor attention_mask;
    bool use_dummy_tokenization = !tokenizer;
    if (tokenizer) {
        try {
            std::string prompt;
            bool add_special_tokens = true;
            if (use_vl) {
                const int64_t image_tokens =
                    ov::genai::modeling::models::Qwen3_5VisionPreprocessor::count_visual_tokens(
                        grid_thw,
                        cfg.vision.spatial_merge_size);
                prompt = build_vl_prompt(opts.user_prompt, image_tokens);
                add_special_tokens = false;
            } else {
                prompt = opts.user_prompt;
                if (!tokenizer->get_chat_template().empty()) {
                    ov::genai::ChatHistory history({{{"role", "user"}, {"content", prompt}}});
                    constexpr bool add_generation_prompt = true;
                    prompt = tokenizer->apply_chat_template(history, add_generation_prompt);
                    add_special_tokens = false;
                }
            }
            auto tokenized = tokenizer->encode(prompt, ov::genai::add_special_tokens(add_special_tokens));
            input_ids = tokenized.input_ids;
            attention_mask = tokenized.attention_mask;
            
            // For VL mode, verify token IDs match config - fall back to dummy if mismatch
            if (use_vl) {
                const int64_t* ids = input_ids.data<const int64_t>();
                size_t img_tok_count = 0;
                for (size_t i = 0; i < input_ids.get_shape()[1]; ++i) {
                    if (ids[i] == cfg.image_token_id) img_tok_count++;
                }
                if (img_tok_count == 0) {
                    std::cerr << "Tokenizer image_token_id mismatch (expected " << cfg.image_token_id 
                              << " but none found in input_ids)" << std::endl;
                    std::cerr << "Falling back to dummy tokenization for VL mode..." << std::endl;
                    use_dummy_tokenization = true;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Tokenizer encode failed: " << e.what() << std::endl;
            std::cerr << "Falling back to dummy tokenization..." << std::endl;
            use_dummy_tokenization = true;
            tokenizer.reset();
        }
    }
    if (use_dummy_tokenization) {
        const int64_t image_tokens = use_vl
                                         ? ov::genai::modeling::models::Qwen3_5VisionPreprocessor::count_visual_tokens(
                                               grid_thw,
                                               cfg.vision.spatial_merge_size)
                                         : 0;
        auto ids = build_dummy_prompt_ids(cfg, opts.user_prompt, use_vl, image_tokens);
        auto tensors = to_batch_tensors(ids);
        input_ids = tensors.first;
        attention_mask = tensors.second;
    }

    const size_t batch = input_ids.get_shape().at(0);
    const int64_t prompt_len = static_cast<int64_t>(input_ids.get_shape().at(1));

    ov::genai::modeling::models::Qwen3_5InputPlanner planner(cfg);
    auto plan = planner.build_plan(input_ids, &attention_mask, use_vl ? &grid_thw : nullptr);

    ov::Tensor visual_padded;
    if (use_vl) {
        visual_padded = ov::genai::modeling::models::Qwen3_5InputPlanner::scatter_visual_embeds(
            visual_embeds,
            plan.visual_pos_mask);
    }

    auto beam_idx = make_beam_idx(batch);
    auto text_request = compiled_text.create_infer_request();

    // Get GPU context for USM-host tensors (iGPU zero-copy optimization)
    auto gpu_ctx = try_get_gpu_context(compiled_text);

    // Clone prefill inputs as USM-host for zero-copy on iGPU
    auto usm_input_ids = clone_as_usm_host(gpu_ctx, input_ids);
    auto usm_attention_mask = clone_as_usm_host(gpu_ctx, attention_mask);
    auto usm_position_ids = clone_as_usm_host(gpu_ctx, plan.position_ids);
    auto usm_beam_idx = clone_as_usm_host(gpu_ctx, beam_idx);

    text_request.reset_state();
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, usm_input_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, usm_attention_mask);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, usm_position_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, usm_beam_idx);
    if (use_vl) {
        auto usm_visual_padded = clone_as_usm_host(gpu_ctx, visual_padded);
        auto usm_visual_pos_mask = clone_as_usm_host(gpu_ctx, plan.visual_pos_mask);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, usm_visual_padded);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, usm_visual_pos_mask);
    }

    const auto prefill_start = std::chrono::steady_clock::now();
    text_request.infer();
    ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
    int64_t next_id = argmax_last_token(logits);
    const auto prefill_end = std::chrono::steady_clock::now();

    std::vector<int64_t> generated;
    generated.reserve(static_cast<size_t>(opts.max_new_tokens));
    generated.push_back(next_id);

    const auto stop_token_ids = resolve_stop_token_ids(use_dummy_mode_flag ? std::filesystem::path{} : model_dir,
                                                       tokenizer.get());

    ov::Tensor step_ids = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, 1});
    ov::Tensor step_mask = make_usm_host_tensor(gpu_ctx, ov::element::i64, {batch, 1});
    auto* step_mask_data = step_mask.data<int64_t>();
    for (size_t b = 0; b < batch; ++b) {
        step_mask_data[b] = 1;
    }

    ov::Tensor decode_visual;
    ov::Tensor decode_visual_mask;
    if (use_vl) {
        decode_visual = make_usm_host_tensor(gpu_ctx, ov::element::f32, {batch, 1, static_cast<size_t>(cfg.text.hidden_size)});
        std::memset(decode_visual.data(), 0, decode_visual.get_byte_size());
        decode_visual_mask = make_usm_host_tensor(gpu_ctx, ov::element::boolean, {batch, 1});
        std::memset(decode_visual_mask.data(), 0, decode_visual_mask.get_byte_size());
    }

    int64_t past_len = prompt_len;
    if (attention_mask.get_element_type() == ov::element::i64) {
        const int64_t* mask_data = attention_mask.data<const int64_t>();
        const size_t mask_batch = attention_mask.get_shape().at(0);
        const size_t mask_seq = attention_mask.get_shape().at(1);
        if (mask_batch > 0 && mask_seq > 0) {
            int64_t first_active_tokens = -1;
            for (size_t b = 0; b < mask_batch; ++b) {
                int64_t active_tokens = 0;
                for (size_t s = 0; s < mask_seq; ++s) {
                    if (mask_data[b * mask_seq + s] != 0) {
                        active_tokens += 1;
                    }
                }

                if (first_active_tokens < 0) {
                    first_active_tokens = active_tokens;
                } else if (first_active_tokens != active_tokens) {
                    OPENVINO_THROW("Batch decode with different active prompt lengths is not supported in this sample: first=",
                                   first_active_tokens,
                                   ", current=",
                                   active_tokens,
                                   ".");
                }
            }

            if (first_active_tokens >= 0) {
                past_len = first_active_tokens;
            }
        }
    }
    // Pre-allocate USM-host tensor for decode position_ids - reuse across steps
    // to avoid per-step create_host_tensor() + memcpy overhead.
    // Shape: [3, batch, 1] (3 planes of identical position values per batch element)
    ov::Tensor usm_decode_pos = make_usm_host_tensor(gpu_ctx, ov::element::i64, {3, batch, 1});
    const int64_t* rope_deltas_data = plan.rope_deltas.data<const int64_t>();

    size_t decode_steps = 0;
    const auto decode_start = std::chrono::steady_clock::now();
    for (int step = 1; step < opts.max_new_tokens; ++step) {
        if (!stop_token_ids.empty() && stop_token_ids.count(next_id) > 0) {
            break;
        }
        auto* step_data = step_ids.data<int64_t>();
        for (size_t b = 0; b < batch; ++b) {
            step_data[b] = next_id;
        }

        // Fill position_ids in-place: all 3 planes get the same value per batch element
        auto* pos_data = usm_decode_pos.data<int64_t>();
        for (size_t b = 0; b < batch; ++b) {
            const int64_t value = past_len + rope_deltas_data[b];
            pos_data[b] = value;             // plane 0
            pos_data[batch + b] = value;     // plane 1
            pos_data[2 * batch + b] = value; // plane 2
        }

        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds, step_ids);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, step_mask);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds, usm_decode_pos);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx, usm_beam_idx);
        if (use_vl) {
            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualEmbeds, decode_visual);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kVisualPosMask, decode_visual_mask);
        }

        text_request.infer();
        logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
        next_id = argmax_last_token(logits);
        generated.push_back(next_id);
        decode_steps += 1;
        past_len += 1;
    }
    const auto decode_end = std::chrono::steady_clock::now();

    const double ttft_ms = elapsed_ms(prefill_start, prefill_end);
    const double decode_ms = elapsed_ms(decode_start, decode_end);
    const double tpot_ms = decode_steps > 0 ? (decode_ms / static_cast<double>(decode_steps)) : 0.0;
    const double throughput = decode_steps > 0 && decode_ms > 0.0
                                  ? (static_cast<double>(decode_steps) * 1000.0 / decode_ms)
                                  : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Mode: " << (use_dummy_mode_flag ? "dummy" : "hf") << " / " << opts.mode << std::endl;
    if (use_dummy_mode_flag) {
        std::cout << "Dummy text arch: " << (cfg.text.is_moe_enabled() ? "moe" : "dense") << std::endl;
    }
    std::cout << "Prompt token size: " << prompt_len << std::endl;
    std::cout << "Output token size: " << generated.size() << std::endl;
    std::cout << "TTFT: " << ttft_ms << " ms" << std::endl;
    std::cout << "Decode time: " << decode_ms << " ms" << std::endl;
    if (decode_steps > 0) {
        std::cout << "TPOT: " << tpot_ms << " ms/token" << std::endl;
        std::cout << "Throughput: " << throughput << " tokens/s" << std::endl;
    } else {
        std::cout << "TPOT: N/A" << std::endl;
        std::cout << "Throughput: N/A" << std::endl;
    }

    if (tokenizer) {
        std::cout << tokenizer->decode(generated, ov::genai::skip_special_tokens(true)) << std::endl;
    } else {
        std::cout << "Generated token ids:";
        for (const auto id : generated) {
            std::cout << ' ' << id;
        }
        std::cout << std::endl;
    }

    return 0;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return 1;
}
