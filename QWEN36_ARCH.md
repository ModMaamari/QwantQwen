# Qwen3.6-35B-A3B Architecture Reference

Ground-truth notes for the port, extracted from the live model repo
(`https://huggingface.co/Qwen/Qwen3.6-35B-A3B`): `config.json` and
`model.safetensors.index.json`. Text path only; vision is deferred.

`architectures: ["Qwen3_5MoeForConditionalGeneration"]`, `model_type: qwen3_5_moe`,
text `model_type: qwen3_5_moe_text`.

## Text config (the fields the engine must honor)

| Field | Value | Notes |
|---|---|---|
| `hidden_size` | 2048 | model dim |
| `num_hidden_layers` | 40 | |
| `vocab_size` | 248320 | |
| `tie_word_embeddings` | false | separate `lm_head` |
| `rms_norm_eps` | 1e-6 | |
| `hidden_act` | silu | SwiGLU MLP |
| **Full attention** | | every 4th layer (`full_attention_interval: 4`) |
| `num_attention_heads` | 16 | |
| `num_key_value_heads` | 2 | GQA, 8× group |
| `head_dim` | 256 | q/k/v head dim |
| `attn_output_gate` | true | q_proj emits 2× heads; half is a sigmoid gate |
| `partial_rotary_factor` | 0.25 | rotary dims = 0.25·256 = **64** |
| `rope_theta` | 1e7 | |
| `rope.mrope_section` | [11,11,10] | mRoPE; text-only collapses to 1-D positions |
| q/k norm | yes | `self_attn.q_norm` / `k_norm` (per-head RMSNorm) |
| **Linear attention (Gated DeltaNet)** | | the other 3 of every 4 layers |
| `linear_num_key_heads` | 16 | key/query heads |
| `linear_key_head_dim` | 128 | → q,k width 16·128 = 2048 each |
| `linear_num_value_heads` | 32 | value heads |
| `linear_value_head_dim` | 128 | → v/z width 32·128 = 4096 |
| `linear_conv_kernel_dim` | 4 | depthwise causal conv1d |
| `mamba_ssm_dtype` | float32 | state kept in fp32 |
| **MoE** | | shared by both layer types |
| `num_experts` | 256 | |
| `num_experts_per_tok` | 8 | top-8 |
| `moe_intermediate_size` | 512 | per-routed-expert SwiGLU width |
| `shared_expert_intermediate_size` | 512 | one always-on shared expert |
| `router_aux_loss_coef` | 0.001 | (training only) |
| `mtp_num_hidden_layers` | 1 | multi-token-predict head, deferred |
| BOS/EOS id | 248044 | from config |
| image/video token ids | 248056 / 248057 | parsed, unused (text-only) |

## Derived per-layer schedule (`full_attention_interval: 4`)

Full-attention layers (0-indexed): **3, 7, 11, 15, 19, 23, 27, 31, 35, 39**
(10 layers). All other 30 layers are Gated DeltaNet linear-attention. The
`layer_types` array in `config.json` confirms this exact pattern. Both layer
types are followed by the same MoE block.

## Tensor inventory (from safetensors index)

Per-layer name root: `model.language_model.layers.{i}.`

### Shared by every layer
- `input_layernorm.weight`            — pre-(attn/linear) RMSNorm
- `post_attention_layernorm.weight`   — pre-MoE RMSNorm
- `mlp.gate.weight`                    — router logits [2048 → 256]
- `mlp.experts.gate_up_proj`          — **batched** [256, 2048, 2·512] fused gate+up
- `mlp.experts.down_proj`             — **batched** [256, 512, 2048]
- `mlp.shared_expert.gate_proj.weight` / `up_proj.weight` / `down_proj.weight`
- `mlp.shared_expert_gate.weight`     — scalar gate for the shared expert

### Full-attention layers only (`self_attn.`)
- `q_proj.weight`  — [2048 → 16·256·2 = 8192] (2× for the output gate)
- `k_proj.weight`  — [2048 → 2·256 = 512]
- `v_proj.weight`  — [2048 → 2·256 = 512]
- `o_proj.weight`  — [16·256 = 4096 → 2048]
- `q_norm.weight` / `k_norm.weight` — per-head RMSNorm (dim 256)

### Linear-attention layers only (`linear_attn.`)
- `in_proj_qkv.weight` — fused q(2048)+k(2048)+v(4096) = [2048 → 8192]
- `in_proj_a.weight`   — α / dt input projection
- `in_proj_b.weight`   — β (delta-rule write strength) projection
- `in_proj_z.weight`   — output gate z [2048 → 4096]
- `conv1d.weight`      — depthwise causal conv, kernel 4, over q/k/v stream
- `A_log`              — per-value-head decay (log A), size 32
- `dt_bias`            — per-value-head dt bias, size 32
- `norm.weight`        — output RMSNorm (gated) over value dim
- `out_proj.weight`    — [4096 → 2048]

### Non-layer
- `model.language_model.embed_tokens.weight` — [248320, 2048]
- `model.language_model.norm.weight`         — final RMSNorm
- `lm_head.weight`                           — [2048 → 248320] (untied)
- `model.visual.*`                           — vision tower (**skipped**, text-only)

## Implications for the port

1. **Expert tensors are batched 3-D**, not per-expert files — the loader binds
   one big tensor and indexes expert `e` by offset, unlike DeepSeek's
   per-expert GGUF tensors. The `gate_up_proj` is fused (split into gate/up at
   `moe_intermediate_size`).
2. **`attn_output_gate`** means `q_proj` width is doubled; the second half is a
   sigmoid gate applied to the attention output before `o_proj`.
3. **QK-norm** (per-head RMSNorm on q and k) runs before RoPE.
4. **Partial RoPE**: only the first 64 of 256 head dims are rotated.
5. **Gated DeltaNet** state is per linear layer, fp32, and persists across the
   whole sequence — it is the session's recurrent state alongside the full-attn
   KV cache (only the 10 full-attn layers hold KV).
6. **GGUF**: no ds4-format GGUF exists for this model yet; a conversion that
   emits the `qwen3_5moe.*` metadata namespace and these tensors is Phase 8.
