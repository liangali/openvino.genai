// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <cxxopts.hpp>

#include "openvino/genai/llm_pipeline.hpp"
#include "read_prompt_from_file.h"
#include "../../../src/cpp/src/modeling/weights/quantization_config.hpp"

namespace {

using QuantizationConfig = ov::genai::modeling::weights::QuantizationConfig;

QuantizationConfig::Mode parse_quantization_mode(const std::string& mode_str, const std::string& option_name) {
    if (mode_str == "int4_sym") {
        return QuantizationConfig::Mode::INT4_SYM;
    }
    if (mode_str == "int4_asym") {
        return QuantizationConfig::Mode::INT4_ASYM;
    }
    if (mode_str == "int8_sym") {
        return QuantizationConfig::Mode::INT8_SYM;
    }
    if (mode_str == "int8_asym") {
        return QuantizationConfig::Mode::INT8_ASYM;
    }
    if (mode_str == "none") {
        return QuantizationConfig::Mode::NONE;
    }
    throw std::runtime_error("Unknown " + option_name + ": " + mode_str);
}

bool is_hybrid_attention_model(const std::filesystem::path& models_path) {
    const auto config_path = models_path / "config.json";
    if (!std::filesystem::is_directory(models_path) || !std::filesystem::exists(config_path)) {
        return false;
    }

    try {
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            return false;
        }

        const std::string config_content((std::istreambuf_iterator<char>(config_file)), std::istreambuf_iterator<char>());
        const std::string key = "\"model_type\"";
        const auto key_pos = config_content.find(key);
        if (key_pos == std::string::npos) {
            return false;
        }

        const auto colon_pos = config_content.find(':', key_pos + key.size());
        if (colon_pos == std::string::npos) {
            return false;
        }

        const auto value_begin = config_content.find('"', colon_pos + 1);
        if (value_begin == std::string::npos) {
            return false;
        }

        const auto value_end = config_content.find('"', value_begin + 1);
        if (value_end == std::string::npos) {
            return false;
        }

        const auto model_type = config_content.substr(value_begin + 1, value_end - value_begin - 1);
        return model_type == "qwen3_next" || model_type == "qwen3_5" || model_type == "qwen3_5_moe";
    } catch (...) {
        return false;
    }
}

}  // namespace

int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_genai", "Help command");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>())
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value(""))
    ("pf,prompt_file", "Read prompt from file", cxxopts::value<std::string>())
    ("nw,num_warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("n,num_iter", "Number of iterations", cxxopts::value<size_t>()->default_value(std::to_string(3)))
    ("mt,max_new_tokens", "Maximal number of new tokens", cxxopts::value<size_t>()->default_value(std::to_string(20)))
    ("d,device", "device", cxxopts::value<std::string>()->default_value("CPU"))
    ("compression_mode", "In-flight quantization mode for explicit safetensors modeling path (none, int4_sym, int4_asym, int8_sym, int8_asym)", cxxopts::value<std::string>()->default_value("int4_asym"))
    ("group_size", "Group size for in-flight quantization", cxxopts::value<int>()->default_value(std::to_string(128)))
    ("backup_mode", "Backup quantization mode for sensitive layers (none, int4_sym, int4_asym, int8_sym, int8_asym)", cxxopts::value<std::string>()->default_value("int4_asym"))
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }
    if (!result.count("model")) {
        std::cout << "Model path is required.\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    std::string prompt;
    if (result.count("prompt") && result.count("prompt_file")) {
        std::cout << "Prompt and prompt file should not exist together!" << std::endl;
        return EXIT_FAILURE;
    } else {
        if (result.count("prompt_file")) {
            prompt = utils::read_prompt(result["prompt_file"].as<std::string>());
        } else {
            prompt = result["prompt"].as<std::string>().empty() ? "The Sky is blue because" : result["prompt"].as<std::string>();
        }
    }
    if (prompt.empty()) {
        std::cout << "Prompt is empty!" << std::endl;
        return EXIT_FAILURE;
    }

    const std::filesystem::path models_path = result["model"].as<std::string>();
    const std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
    num_iter = std::max<size_t>(num_iter, 1);

    QuantizationConfig quant_config;
    quant_config.mode = parse_quantization_mode(result["compression_mode"].as<std::string>(), "compression mode");
    quant_config.group_size = result["group_size"].as<int>();
    quant_config.backup_mode = parse_quantization_mode(result["backup_mode"].as<std::string>(), "backup mode");

    ov::genai::GenerationConfig config;
    config.max_new_tokens = result["max_new_tokens"].as<size_t>();
    config.apply_chat_template = false;

    std::cout << ov::get_openvino_version() << std::endl;

    ov::AnyMap pipe_config;
    pipe_config[ov::genai::enable_save_ov_model.name()] = false;

    if (quant_config.enabled()) {
        pipe_config["QUANTIZATION_CONFIG"] = quant_config;
    }

    const bool use_forced_scheduler_config = device != "NPU" && !is_hybrid_attention_model(models_path);
    if (use_forced_scheduler_config) {
        ov::genai::SchedulerConfig scheduler_config;
        scheduler_config.enable_prefix_caching = false;
        scheduler_config.max_num_batched_tokens = std::numeric_limits<std::size_t>::max();
        pipe_config[ov::genai::scheduler_config.name()] = scheduler_config;
    } else if (device != "NPU") {
        std::cout << "Hybrid-attention model detected, using LLMPipeline default backend selection." << std::endl;
    }

    auto pipe = std::make_unique<ov::genai::LLMPipeline>(models_path, device, pipe_config);

    auto input_data = pipe->get_tokenizer().encode(prompt);
    size_t prompt_token_size = input_data.input_ids.get_shape()[1];
    std::cout << "Prompt token size:" << prompt_token_size << std::endl;

    for (size_t i = 0; i < num_warmup; i++)
        pipe->generate(prompt, config);

    ov::genai::DecodedResults res = pipe->generate(prompt, config);
    ov::genai::PerfMetrics metrics = res.perf_metrics;
    for (size_t i = 0; i < num_iter - 1; i++) {
        res = pipe->generate(prompt, config);
        metrics = metrics + res.perf_metrics;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Output token size:" << res.perf_metrics.get_num_generated_tokens() << std::endl;
    std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± " << metrics.get_generate_duration().std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± " << metrics.get_tokenization_duration().std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± " << metrics.get_detokenization_duration().std << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean  << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean  << " ± " << metrics.get_throughput().std << " tokens/s" << std::endl;

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
