// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_5/modeling_qwen3_5_mtp.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
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

    // Tie lm_head weight to embed_tokens weight when tie_word_embeddings=true.
    // Must be done at construction time so load_model() binds both via the embed key.
    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(embed_tokens_.weight_param());
    }

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
    // Cast embed to hidden_states dtype to ensure type consistency
    // (embed weights may be bf16 while hidden_states is f16, or vice versa)
    auto embed_cast = embed.to(hidden_states.dtype());
    auto e_norm = pre_fc_norm_embed_.forward(embed_cast);
    auto h_norm = pre_fc_norm_hidden_.forward(hidden_states);

    // 3. Fuse: concat along feature axis -> [B, 1, 2H], then project -> [B, 1, H]
    auto fused     = ops::concat({e_norm, h_norm}, /*axis=*/-1);
    // Cast FC weight to match fused dtype
    auto fc_w = fc_param_->value().to(fused.dtype());
    auto projected = ops::linear(fused, fc_w);

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

namespace {

/// A WeightSource that checks a pre-populated in-memory overlay first,
/// then falls back to the real source.  Used to inject synthetic fused
/// expert tensors when the MTP checkpoint stores per-expert 2-D weights.
class OverlayWeightSource : public weights::WeightSource {
public:
    explicit OverlayWeightSource(weights::WeightSource& base) : base_(base) {}

    void put(const std::string& name, ov::Tensor tensor) {
        overlay_[name] = std::move(tensor);
    }

    std::vector<std::string> keys() const override {
        std::vector<std::string> k = base_.keys();
        for (const auto& p : overlay_)
            k.push_back(p.first);
        return k;
    }

    bool has(const std::string& name) const override {
        return overlay_.count(name) > 0 || base_.has(name);
    }

    const ov::Tensor& get_tensor(const std::string& name) const override {
        auto it = overlay_.find(name);
        if (it != overlay_.end())
            return it->second;
        return base_.get_tensor(name);
    }

    void release_tensor(const std::string& name) override {
        base_.release_tensor(name);
    }

private:
    weights::WeightSource& base_;
    std::unordered_map<std::string, ov::Tensor> overlay_;
};

/// Stack per-expert 2-D tensors [R, C] (e in [0, num_experts)) into [E, R, C].
ov::Tensor stack_experts(weights::WeightSource& source,
                         const std::string& key_prefix,
                         const std::string& key_suffix,
                         int32_t num_experts) {
    const auto& t0 = source.get_tensor(key_prefix + "0" + key_suffix);
    const auto t0_shape = t0.get_shape();
    OPENVINO_ASSERT(t0_shape.size() == 2,
                    "stack_experts: expected 2-D per-expert tensor, got rank ", t0_shape.size());
    ov::Shape out_shape = {static_cast<size_t>(num_experts), t0_shape[0], t0_shape[1]};
    ov::Tensor result(t0.get_element_type(), out_shape);
    const size_t expert_bytes = t0.get_byte_size();
    uint8_t* dst = static_cast<uint8_t*>(result.data());
    for (int32_t e = 0; e < num_experts; ++e) {
        const auto& te = source.get_tensor(key_prefix + std::to_string(e) + key_suffix);
        std::memcpy(dst + static_cast<size_t>(e) * expert_bytes, te.data(), expert_bytes);
    }
    return result;
}

/// Concatenate two [E, I, H] tensors along dim 1 to produce [E, 2*I, H].
ov::Tensor concat_dim1(const ov::Tensor& a, const ov::Tensor& b) {
    const auto sa = a.get_shape();
    OPENVINO_ASSERT(sa.size() == 3, "concat_dim1: expected rank 3, got ", sa.size());
    OPENVINO_ASSERT(a.get_element_type() == b.get_element_type(),
                    "concat_dim1: element type mismatch");
    const size_t E = sa[0], I = sa[1], H = sa[2];
    const size_t elem_size = a.get_element_type().size();
    ov::Tensor result(a.get_element_type(), {E, 2 * I, H});
    const uint8_t* src_a = static_cast<const uint8_t*>(a.data());
    const uint8_t* src_b = static_cast<const uint8_t*>(b.data());
    uint8_t* dst = static_cast<uint8_t*>(result.data());
    const size_t row_bytes = I * H * elem_size;
    for (size_t e = 0; e < E; ++e) {
        std::memcpy(dst + e * 2 * row_bytes,               src_a + e * row_bytes, row_bytes);
        std::memcpy(dst + e * 2 * row_bytes + row_bytes,   src_b + e * row_bytes, row_bytes);
    }
    return result;
}

}  // namespace

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

    // MTP checkpoints store MoE expert weights as per-expert 2-D tensors:
    //   mtp.layers.0.mlp.experts.N.gate_proj.weight  [I, H]
    //   mtp.layers.0.mlp.experts.N.up_proj.weight    [I, H]
    //   mtp.layers.0.mlp.experts.N.down_proj.weight  [H, I]
    //
    // Qwen3_5SparseMoeBlock expects fused 3-D keys in the source:
    //   mtp.layers.0.mlp.experts.gate_up_proj        [E, 2*I, H]
    //   mtp.layers.0.mlp.experts.down_proj           [E, H, I]
    //
    // Pre-stack and fuse per-expert tensors into an overlay source so the
    // existing custom loaders in Qwen3_5SparseMoeBlock work unchanged.
    const std::string expert_pfx = "mtp.layers.0.mlp.experts.";
    OverlayWeightSource overlay(source);
    if (model_cfg.num_experts > 0 && source.has(expert_pfx + "0.gate_proj.weight")) {
        const int32_t E = model_cfg.num_experts;
        auto gate_s = stack_experts(source, expert_pfx, ".gate_proj.weight", E);
        auto up_s   = stack_experts(source, expert_pfx, ".up_proj.weight",   E);
        auto down_s = stack_experts(source, expert_pfx, ".down_proj.weight", E);
        overlay.put(expert_pfx + "gate_up_proj", concat_dim1(gate_s, up_s));
        overlay.put(expert_pfx + "down_proj",    std::move(down_s));
    }

    (void)ov::genai::modeling::weights::load_model(model, overlay, finalizer, options);

    // ── Model inputs ─────────────────────────────────────────────────────────
    auto input_ids    = ctx.parameter("input_ids",     ov::element::i64,
                                       ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids",  ov::element::i64,
                                       ov::PartialShape{-1, -1});
    // Use f32 element type — the safetensors weight finalizer converts all BF16/F16
    // weights to F32, so the main model runs in F32 and outputs F32 hidden_states.
    auto hidden_in    = ctx.parameter("hidden_states",  ov::element::f32,
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
