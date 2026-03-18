// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_5/modeling_qwen3_5_mtp.hpp"

#include <cmath>
#include <limits>
#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>

#include "modeling/ops/kv_cache.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"

namespace {
auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};
}

namespace ov {
namespace genai {
namespace modeling {
namespace models {

Qwen3_5MtpForDraft::Qwen3_5MtpForDraft(BuilderContext& ctx,
                                         const std::string& name,
                                         const Qwen3_5TextModelConfig& cfg,
                                         Module* parent)
    : Module(name, ctx, parent),
      cfg_(cfg),
      embed_tokens_(ctx, "embed_tokens", this),
      pre_fc_norm_embed_(ctx, "pre_fc_norm_embed", cfg.rms_norm_eps, this),
      pre_fc_norm_hidden_(ctx, "pre_fc_norm_hidden", cfg.rms_norm_eps, this),
      // layer_ is the single MTP transformer layer.
      // We use layer_idx=0 because:
      //   - cfg_.layer_types[0] == "full_attention"
      //   - KV cache prefix is based on module full_path() ("mtp.layer.self_attn"),
      //     which does NOT collide with the main model's "model.layers[N].self_attn".
      layer_(ctx, "layer", cfg_, 0, this),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      lm_head_(ctx, "lm_head", this) {
    fc_param_ = &register_parameter("fc.weight");

    // Compute rotary_dim the same way as Qwen3_5Model does.
    const int32_t head_dim = cfg_.head_dim > 0
                                 ? cfg_.head_dim
                                 : (cfg_.num_attention_heads > 0 ? (cfg_.hidden_size / cfg_.num_attention_heads) : 0);
    OPENVINO_ASSERT(head_dim > 0, "Qwen3_5MtpForDraft: invalid head_dim");
    rotary_dim_ = static_cast<int32_t>(
        std::floor(static_cast<float>(head_dim) * cfg_.partial_rotary_factor));
    rotary_dim_ = std::max<int32_t>(0, std::min<int32_t>(rotary_dim_, head_dim));
    if ((rotary_dim_ % 2) != 0) {
        rotary_dim_ -= 1;
    }
    OPENVINO_ASSERT(rotary_dim_ > 0, "Qwen3_5MtpForDraft: rotary_dim must be > 0");
}

Tensor Qwen3_5MtpForDraft::forward(const Tensor& input_ids,
                                    const Tensor& position_ids,
                                    const Tensor& hidden_states,
                                    const Tensor& beam_idx) const {
    // 1. Embed the candidate token: [B, 1, H]
    auto embed = embed_tokens_.forward(input_ids);

    // 2. Normalize embedding and hidden state independently
    auto e_norm = pre_fc_norm_embed_.forward(embed);
    auto h_norm = pre_fc_norm_hidden_.forward(hidden_states);

    // 3. Fuse: concat along feature axis -> [B, 1, 2H], then project -> [B, 1, H]
    auto fused     = ops::concat({e_norm, h_norm}, /*axis=*/-1);
    auto projected = ops::linear(fused, fc_param_->value());

    // 4. Compute 1D RoPE cos/sin tables.
    //    position_ids shape: [B, 1]  (simple 1-D, not MRoPE 3-D)
    //    rope_cos_sin returns ([B, 1, half_dim], [B, 1, half_dim])
    auto [rope_cos, rope_sin] = ops::llm::rope_cos_sin(
        position_ids, rotary_dim_, cfg_.rope_theta);

    // 5. Run MTP transformer layer (full_attention with KV cache).
    //    Pass nullptr for full_attention_mask: causal SDPA is handled internally
    //    via the is_causal=true flag in Qwen3_5Attention.
    auto [out, _] = layer_.forward(
        projected, beam_idx, rope_cos, rope_sin,
        /*full_attention_mask=*/nullptr,
        /*linear_attention_mask=*/nullptr,
        /*cache_position=*/nullptr,
        /*residual=*/std::nullopt,
        /*precomputed_full_attn_sdpa_mask=*/nullptr);

    // 6. Final norm + LM head -> logits [B, 1, vocab_size]
    auto normed = norm_.forward(out);
    return lm_head_.forward(normed);
}

std::shared_ptr<ov::Model> create_qwen3_5_mtp_model(
    const Qwen3_5Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {

    // Convert Qwen3_5TextConfig -> Qwen3_5TextModelConfig.
    // Copy ALL fields that create_qwen3_5_text_model() copies (verified from source).
    Qwen3_5TextModelConfig model_cfg;
    model_cfg.architecture                   = "qwen3_5";
    model_cfg.hidden_size                    = cfg.text.hidden_size;
    model_cfg.num_attention_heads            = cfg.text.num_attention_heads;
    model_cfg.num_key_value_heads            = cfg.text.num_key_value_heads > 0
                                                   ? cfg.text.num_key_value_heads
                                                   : cfg.text.num_attention_heads;
    model_cfg.head_dim                       = cfg.text.resolved_head_dim();
    model_cfg.intermediate_size              = cfg.text.intermediate_size;
    // num_hidden_layers is set to 1 so that Qwen3_5Model's size-check
    // (layer_types.size() == num_hidden_layers) passes.
    // The MTP has exactly one transformer layer.
    model_cfg.num_hidden_layers              = 1;
    model_cfg.vocab_size                     = cfg.text.vocab_size;
    model_cfg.max_position_embeddings        = cfg.text.max_position_embeddings;
    model_cfg.rms_norm_eps                   = cfg.text.rms_norm_eps;
    model_cfg.rope_theta                     = cfg.text.rope_theta;
    model_cfg.partial_rotary_factor          = cfg.text.partial_rotary_factor;
    model_cfg.hidden_act                     = cfg.text.hidden_act;
    model_cfg.attention_bias                 = cfg.text.attention_bias;
    model_cfg.tie_word_embeddings            = cfg.text.tie_word_embeddings;
    // The MTP layer is always full_attention regardless of the main model's layer types.
    model_cfg.layer_types                    = {"full_attention"};
    model_cfg.full_attention_interval        = cfg.text.full_attention_interval;
    model_cfg.linear_conv_kernel_dim         = cfg.text.linear_conv_kernel_dim;
    model_cfg.linear_key_head_dim            = cfg.text.linear_key_head_dim;
    model_cfg.linear_value_head_dim          = cfg.text.linear_value_head_dim;
    model_cfg.linear_num_key_heads           = cfg.text.linear_num_key_heads;
    model_cfg.linear_num_value_heads         = cfg.text.linear_num_value_heads;
    model_cfg.moe_intermediate_size          = cfg.text.moe_intermediate_size;
    model_cfg.shared_expert_intermediate_size = cfg.text.shared_expert_intermediate_size;
    model_cfg.num_experts                    = cfg.text.num_experts;
    model_cfg.num_experts_per_tok            = cfg.text.num_experts_per_tok;
    model_cfg.norm_topk_prob                 = cfg.text.norm_topk_prob;
    model_cfg.output_router_logits           = cfg.text.output_router_logits;
    model_cfg.router_aux_loss_coef           = cfg.text.router_aux_loss_coef;
    model_cfg.mrope_interleaved              = cfg.text.rope.mrope_interleaved;
    model_cfg.mrope_section                  = cfg.text.rope.mrope_section;
    model_cfg.mtp_num_hidden_layers          = cfg.text.mtp_num_hidden_layers;

    BuilderContext ctx;
    // Root module name "mtp" matches checkpoint prefix "mtp.*"
    Qwen3_5MtpForDraft model(ctx, "mtp", model_cfg);

    // ── PackedMapping rules ───────────────────────────────────────────────────
    // Map checkpoint keys -> module parameter paths.
    // API matches create_qwen3_5_text_model(): push_back({ckpt_substring, module_path, shard_id}).
    auto& rules = model.packed_mapping().rules;
    // 1. embed_tokens: shared with main model — checkpoint stores under "model.language_model."
    rules.push_back({"model.language_model.embed_tokens.", "mtp.embed_tokens.", 0});
    // 2. pre_fc_norm_embedding -> pre_fc_norm_embed  (name difference in checkpoint vs module)
    rules.push_back({"mtp.pre_fc_norm_embedding.", "mtp.pre_fc_norm_embed.", 0});
    // 3. layers.0 -> layer  (checkpoint uses "layers.0", module uses "layer")
    rules.push_back({"mtp.layers.0.", "mtp.layer.", 0});
    // 4. lm_head: checkpoint stores at top level "lm_head.", module is "mtp.lm_head."
    rules.push_back({"lm_head.", "mtp.lm_head.", 0});

    // ── Load weights ─────────────────────────────────────────────────────────
    ov::genai::modeling::weights::LoadOptions options;
    options.allow_missing    = true;   // tie_word_embeddings: lm_head may be missing
    options.allow_unmatched  = true;   // main model weights will be unmatched; ignore
    options.report_missing   = true;
    options.report_unmatched = false;

    (void)ov::genai::modeling::weights::load_model(model, source, finalizer, options);

    // ── Model inputs ─────────────────────────────────────────────────────────
    auto input_ids    = ctx.parameter("input_ids",     ov::element::i64,
                                       ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids",  ov::element::i64,
                                       ov::PartialShape{-1, -1});
    auto hidden_in    = ctx.parameter("hidden_states",  ov::element::f16,
                                       ov::PartialShape{-1, -1, model_cfg.hidden_size});
    auto beam_idx     = ctx.parameter("beam_idx",       ov::element::i32,
                                       ov::PartialShape{-1});

    auto logits = model.forward(input_ids, position_ids, hidden_in, beam_idx);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");

    auto ov_model = ctx.build_model({result->output(0)});
    ov_model->set_rt_info(ov::element::f16,
                           {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f,
                           {"runtime_options", ov::hint::activations_scale_factor.name()});
    return ov_model;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
