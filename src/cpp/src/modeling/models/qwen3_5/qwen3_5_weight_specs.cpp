// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"

#include <string>
#include <utility>
#include <vector>

#include <openvino/core/except.hpp>

namespace {

using Spec = ov::genai::modeling::weights::SyntheticWeightSpec;

size_t to_sz(int32_t v, const char* field) {
    if (v <= 0) {
        OPENVINO_THROW("Qwen3_5 weight spec invalid ", field, ": ", v);
    }
    return static_cast<size_t>(v);
}

void add(std::vector<Spec>& out,
         std::string name,
         ov::Shape shape,
         ov::element::Type dtype = ov::element::f32) {
    out.push_back(Spec{std::move(name), std::move(shape), dtype});
}

std::vector<std::string> normalized_layer_types(const ov::genai::modeling::models::Qwen3_5TextConfig& cfg) {
    if (!cfg.layer_types.empty()) {
        return cfg.layer_types;
    }
    const int32_t interval = cfg.full_attention_interval > 0 ? cfg.full_attention_interval : 4;
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        out.push_back(((i + 1) % interval) == 0 ? "full_attention" : "linear_attention");
    }
    return out;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

std::vector<ov::genai::modeling::weights::SyntheticWeightSpec> build_qwen3_5_text_weight_specs(const Qwen3_5TextConfig& cfg_in) {
    auto cfg = cfg_in;
    cfg.finalize();
    cfg.validate();
    const bool is_moe = cfg.is_moe_enabled();

    const size_t hidden = to_sz(cfg.hidden_size, "hidden_size");
    const size_t vocab = to_sz(cfg.vocab_size, "vocab_size");
    const size_t num_layers = to_sz(cfg.num_hidden_layers, "num_hidden_layers");
    const size_t num_heads = to_sz(cfg.num_attention_heads, "num_attention_heads");
    const size_t num_kv_heads = to_sz(cfg.kv_heads(), "num_key_value_heads");
    const size_t head_dim = to_sz(cfg.resolved_head_dim(), "head_dim");
    const size_t linear_num_k_heads = to_sz(cfg.linear_num_key_heads, "linear_num_key_heads");
    const size_t linear_num_v_heads = to_sz(cfg.linear_num_value_heads, "linear_num_value_heads");
    const size_t linear_k_dim = to_sz(cfg.linear_key_head_dim, "linear_key_head_dim");
    const size_t linear_v_dim = to_sz(cfg.linear_value_head_dim, "linear_value_head_dim");
    const size_t conv_kernel = to_sz(cfg.linear_conv_kernel_dim, "linear_conv_kernel_dim");

    const size_t key_dim = linear_num_k_heads * linear_k_dim;
    const size_t value_dim = linear_num_v_heads * linear_v_dim;
    const size_t conv_dim = key_dim * 2 + value_dim;
    const size_t q_proj_out = num_heads * head_dim * 2;
    const size_t kv_proj_out = num_kv_heads * head_dim;
    const size_t intermediate = is_moe ? 0 : to_sz(cfg.intermediate_size, "intermediate_size");
    const size_t num_experts = is_moe ? to_sz(cfg.num_experts, "num_experts") : 0;
    const size_t moe_intermediate = is_moe ? to_sz(cfg.moe_intermediate_size, "moe_intermediate_size") : 0;
    const size_t shared_intermediate =
        is_moe ? to_sz(cfg.shared_expert_intermediate_size, "shared_expert_intermediate_size") : 0;

    std::vector<Spec> specs;
    specs.reserve(is_moe ? (3 + num_layers * 18) : (3 + num_layers * 14));

    add(specs, "model.embed_tokens.weight", {vocab, hidden});
    add(specs, "model.norm.weight", {hidden});
    add(specs, "lm_head.weight", {vocab, hidden});

    const auto layer_types = normalized_layer_types(cfg);
    if (layer_types.size() != num_layers) {
        OPENVINO_THROW("Qwen3_5 layer_types size mismatch while building weight specs");
    }

    for (size_t i = 0; i < num_layers; ++i) {
        const std::string layer_prefix = "model.layers[" + std::to_string(i) + "].";
        add(specs, layer_prefix + "input_layernorm.weight", {hidden});
        add(specs, layer_prefix + "post_attention_layernorm.weight", {hidden});

        if (layer_types[i] == "full_attention") {
            const std::string attn = layer_prefix + "self_attn.";
            add(specs, attn + "q_proj.weight", {q_proj_out, hidden});
            add(specs, attn + "k_proj.weight", {kv_proj_out, hidden});
            add(specs, attn + "v_proj.weight", {kv_proj_out, hidden});
            add(specs, attn + "o_proj.weight", {hidden, num_heads * head_dim});
            add(specs, attn + "q_norm.weight", {head_dim});
            add(specs, attn + "k_norm.weight", {head_dim});
        } else if (layer_types[i] == "linear_attention") {
            const std::string attn = layer_prefix + "linear_attn.";
            add(specs, attn + "in_proj_qkv.weight", {key_dim * 2 + value_dim, hidden});
            add(specs, attn + "in_proj_z.weight", {value_dim, hidden});
            add(specs, attn + "in_proj_b.weight", {linear_num_v_heads, hidden});
            add(specs, attn + "in_proj_a.weight", {linear_num_v_heads, hidden});
            add(specs, attn + "conv1d.weight", {conv_dim, conv_kernel});
            add(specs, attn + "A_log", {linear_num_v_heads});
            add(specs, attn + "dt_bias", {linear_num_v_heads});
            add(specs, attn + "norm.weight", {linear_v_dim});
            add(specs, attn + "out_proj.weight", {hidden, value_dim});
        } else {
            OPENVINO_THROW("Unsupported Qwen3_5 layer type in weight specs: ", layer_types[i]);
        }

        const std::string mlp = layer_prefix + "mlp.";
        if (is_moe) {
            add(specs, mlp + "gate.weight", {num_experts, hidden});
            add(specs, mlp + "experts.gate_up_proj", {num_experts, 2 * moe_intermediate, hidden});
            add(specs, mlp + "experts.down_proj", {num_experts, hidden, moe_intermediate});
            add(specs, mlp + "shared_expert.gate_proj.weight", {shared_intermediate, hidden});
            add(specs, mlp + "shared_expert.up_proj.weight", {shared_intermediate, hidden});
            add(specs, mlp + "shared_expert.down_proj.weight", {hidden, shared_intermediate});
            add(specs, mlp + "shared_expert_gate.weight", {1, hidden});
        } else {
            add(specs, mlp + "gate_proj.weight", {intermediate, hidden});
            add(specs, mlp + "up_proj.weight", {intermediate, hidden});
            add(specs, mlp + "down_proj.weight", {hidden, intermediate});
        }
    }

    return specs;
}

std::vector<ov::genai::modeling::weights::SyntheticWeightSpec> build_qwen3_5_vision_weight_specs(const Qwen3_5VisionConfig& cfg_in) {
    auto cfg = cfg_in;
    cfg.finalize();
    cfg.validate();

    const size_t hidden = to_sz(cfg.hidden_size, "vision.hidden_size");
    const size_t in_ch = to_sz(cfg.in_channels, "vision.in_channels");
    const size_t patch = to_sz(cfg.patch_size, "vision.patch_size");
    const size_t t_patch = to_sz(cfg.temporal_patch_size, "vision.temporal_patch_size");
    const size_t depth = to_sz(cfg.depth, "vision.depth");
    const size_t inter = to_sz(cfg.intermediate_size, "vision.intermediate_size");
    const size_t merge = to_sz(cfg.spatial_merge_size, "vision.spatial_merge_size");
    const size_t merged_hidden = hidden * merge * merge;
    const size_t out_hidden = to_sz(cfg.out_hidden_size, "vision.out_hidden_size");
    const size_t pos_count = to_sz(cfg.num_position_embeddings, "vision.num_position_embeddings");

    std::vector<Spec> specs;
    specs.reserve(8 + depth * 16 + cfg.deepstack_visual_indexes.size() * 6);

    add(specs, "visual.patch_embed.proj.weight", {hidden, in_ch, t_patch, patch, patch});
    add(specs, "visual.patch_embed.proj.bias", {hidden});
    add(specs, "visual.pos_embed.weight", {pos_count, hidden});

    for (size_t i = 0; i < depth; ++i) {
        const std::string block = "visual.blocks." + std::to_string(i) + ".";
        add(specs, block + "norm1.weight", {hidden});
        add(specs, block + "norm1.bias", {hidden});
        add(specs, block + "norm2.weight", {hidden});
        add(specs, block + "norm2.bias", {hidden});

        add(specs, block + "attn.qkv.weight", {hidden * 3, hidden});
        add(specs, block + "attn.qkv.bias", {hidden * 3});
        add(specs, block + "attn.proj.weight", {hidden, hidden});
        add(specs, block + "attn.proj.bias", {hidden});

        add(specs, block + "mlp.linear_fc1.weight", {inter, hidden});
        add(specs, block + "mlp.linear_fc1.bias", {inter});
        add(specs, block + "mlp.linear_fc2.weight", {hidden, inter});
        add(specs, block + "mlp.linear_fc2.bias", {hidden});
    }

    add(specs, "visual.merger.norm.weight", {hidden});
    add(specs, "visual.merger.norm.bias", {hidden});
    add(specs, "visual.merger.linear_fc1.weight", {merged_hidden, merged_hidden});
    add(specs, "visual.merger.linear_fc1.bias", {merged_hidden});
    add(specs, "visual.merger.linear_fc2.weight", {out_hidden, merged_hidden});
    add(specs, "visual.merger.linear_fc2.bias", {out_hidden});

    for (size_t i = 0; i < cfg.deepstack_visual_indexes.size(); ++i) {
        const std::string merger = "visual.deepstack_merger_list." + std::to_string(i) + ".";
        add(specs, merger + "norm.weight", {merged_hidden});
        add(specs, merger + "norm.bias", {merged_hidden});
        add(specs, merger + "linear_fc1.weight", {merged_hidden, merged_hidden});
        add(specs, merger + "linear_fc1.bias", {merged_hidden});
        add(specs, merger + "linear_fc2.weight", {out_hidden, merged_hidden});
        add(specs, merger + "linear_fc2.bias", {out_hidden});
    }

    return specs;
}

std::vector<ov::genai::modeling::weights::SyntheticWeightSpec> build_qwen3_5_vlm_weight_specs(const Qwen3_5Config& cfg) {
    std::vector<Spec> specs = build_qwen3_5_vision_weight_specs(cfg.vision);
    auto text_specs = build_qwen3_5_text_weight_specs(cfg.text);
    specs.insert(specs.end(), text_specs.begin(), text_specs.end());
    return specs;
}

std::vector<ov::genai::modeling::weights::SyntheticWeightSpec>
build_qwen3_5_mtp_weight_specs(const Qwen3_5TextConfig& cfg_in) {
    auto cfg = cfg_in;
    cfg.finalize();
    cfg.validate();
    const bool is_moe = cfg.is_moe_enabled();

    const size_t H       = to_sz(cfg.hidden_size, "hidden_size");
    const size_t V       = to_sz(cfg.vocab_size, "vocab_size");
    const size_t nh      = to_sz(cfg.num_attention_heads, "num_attention_heads");
    const size_t nkv     = to_sz(cfg.kv_heads(), "num_key_value_heads");
    const size_t hd      = to_sz(cfg.resolved_head_dim(), "head_dim");
    // q_proj is gated (attn_output_gate=true applies to MTP layer as well)
    const size_t q_proj_out = nh * hd * 2;
    const size_t kv_proj_out = nkv * hd;

    const size_t inter    = is_moe ? 0 : to_sz(cfg.intermediate_size, "intermediate_size");
    const size_t moe_inter = is_moe ? to_sz(cfg.moe_intermediate_size, "moe_intermediate_size") : 0;
    const size_t sh_inter  = is_moe ? to_sz(cfg.shared_expert_intermediate_size, "shared_expert_intermediate_size") : 0;
    const size_t nexp      = is_moe ? to_sz(cfg.num_experts, "num_experts") : 0;

    std::vector<Spec> specs;

    // Projection norms and fusion FC (always present regardless of dedicated embeddings)
    add(specs, "mtp.pre_fc_norm_embedding.weight", {H});
    add(specs, "mtp.pre_fc_norm_hidden.weight",    {H});
    add(specs, "mtp.fc.weight",                    {H, 2 * H});

    // MTP decoder layer (single layer at index 0 inside the mtp module)
    add(specs, "mtp.layers.0.input_layernorm.weight",          {H});
    add(specs, "mtp.layers.0.post_attention_layernorm.weight", {H});

    // Attention — q_proj is gated ({nh*hd*2, H}), k/v are not
    add(specs, "mtp.layers.0.self_attn.q_proj.weight", {q_proj_out, H});
    add(specs, "mtp.layers.0.self_attn.k_proj.weight", {kv_proj_out, H});
    add(specs, "mtp.layers.0.self_attn.v_proj.weight", {kv_proj_out, H});
    add(specs, "mtp.layers.0.self_attn.o_proj.weight", {H, nh * hd});
    add(specs, "mtp.layers.0.self_attn.q_norm.weight", {hd});
    add(specs, "mtp.layers.0.self_attn.k_norm.weight", {hd});

    if (is_moe) {
        // MoE MLP weights (per-expert; no fused gate_up_proj in MTP unlike main model)
        add(specs, "mtp.layers.0.mlp.gate.weight",               {nexp, H});
        add(specs, "mtp.layers.0.mlp.shared_expert_gate.weight", {1, H});
        add(specs, "mtp.layers.0.mlp.shared_expert.gate_proj.weight", {sh_inter, H});
        add(specs, "mtp.layers.0.mlp.shared_expert.up_proj.weight",   {sh_inter, H});
        add(specs, "mtp.layers.0.mlp.shared_expert.down_proj.weight", {H, sh_inter});
        for (size_t e = 0; e < nexp; ++e) {
            const std::string pfx = "mtp.layers.0.mlp.experts." + std::to_string(e) + ".";
            add(specs, pfx + "gate_proj.weight", {moe_inter, H});
            add(specs, pfx + "up_proj.weight",   {moe_inter, H});
            add(specs, pfx + "down_proj.weight", {H, moe_inter});
        }
    } else {
        // Dense MLP
        add(specs, "mtp.layers.0.mlp.gate_proj.weight", {inter, H});
        add(specs, "mtp.layers.0.mlp.up_proj.weight",   {inter, H});
        add(specs, "mtp.layers.0.mlp.down_proj.weight", {H, inter});
    }

    // Final norm
    add(specs, "mtp.norm.weight", {H});

    // LM head — same key as main model (shared weights, reloaded for MTP model)
    add(specs, "lm_head.weight", {V, H});

    return specs;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

