// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

std::vector<ov::genai::modeling::weights::SyntheticWeightSpec> build_qwen3_5_text_weight_specs(const Qwen3_5TextConfig& cfg);
std::vector<ov::genai::modeling::weights::SyntheticWeightSpec> build_qwen3_5_vision_weight_specs(const Qwen3_5VisionConfig& cfg);
std::vector<ov::genai::modeling::weights::SyntheticWeightSpec> build_qwen3_5_vlm_weight_specs(const Qwen3_5Config& cfg);
std::vector<ov::genai::modeling::weights::SyntheticWeightSpec> build_qwen3_5_mtp_weight_specs(const Qwen3_5TextConfig& cfg);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

