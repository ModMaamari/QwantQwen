# Porting DwarfStar (ds4) to Qwen3.6-35B-A3B

## 0. Goal & philosophy

Take antirez's **DwarfStar** (`ds4`) — a small, self-contained, single-model
native inference engine built for **DeepSeek V4 Flash/Pro** — and apply the *same
philosophy* to **Qwen3.6-35B-A3B** (`Qwen/Qwen3.6-35B-A3B`,
`model_type: qwen3_5_moe`).

The DwarfStar philosophy we are preserving:

- **One model, done end-to-end.** Model-specific loading, prompt rendering,
  tool calling, KV/state handling, an HTTP server, and a CLI — not a generic
  GGUF runner.
- **Correctness before speed.** A CPU reference path validated against the
  official implementation, then GPU graph paths that must not drift.
- **Small, sharp, readable C.** No C++, narrow public API (`ds4.h`), no
  permanent semantic variants behind flags once a single release path exists.

### Scope decisions (locked for this effort)

| Decision | Choice |
|---|---|
| First milestone | **Text-only**, CUDA-targeted (correctness-first CPU reference, then CUDA) |
| Build/test env | **WSL2 + CUDA** on the dev PC (no native Windows build; ds4 is `uname`-based) |
| Git workflow | Feature branch `feature/qwen36-port`, **atomic commit per step, pushed** |
| Validation | **Structural correctness** for now (no weights / official logits yet); numeric logit-matching is a later phase once weights + an HF `transformers` reference are available |
| Multimodal | **Deferred.** Vision encoder + image/video tokens are out of scope for milestone 1 |

> **Reality check.** This is a ground-up port, not a config swap. The three
> hardest pieces of an inference engine all differ from DeepSeek at once:
> the attention/state engine, the tokenizer/chat format, and the weight
> layout. The Gated DeltaNet linear-attention core is research-grade and has
> **zero reuse** from ds4. Expect the numeric core (Phase 5) to be the bulk of
> the work and the part that ultimately needs weights to validate.

---

## 1. Architecture gap: DeepSeek V4 Flash vs Qwen3.6-35B-A3B

Confirmed from the live `config.json`
(`https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json`).

| Aspect | DeepSeek V4 Flash (ds4 target) | Qwen3.6-35B-A3B |
|---|---|---|
| Total / active params | 284B / 13B | ~35B / ~3B |
| Layers | 43 | **40** |
| Hidden size | 4096 | **2048** |
| Attention | MLA: compressed (CSA/HCA) softmax + indexer top-k, hyper-connections (HC), LoRA Q/O | **Hybrid**: `linear_attention` ×3 then `full_attention` ×1, repeating (`full_attention_interval: 4`) |
| Linear-attn layers | none | **Gated DeltaNet**: conv1d (`linear_conv_kernel_dim: 4`), 16 key heads / 32 value heads, key/value head dim 128, SSM state (`mamba_ssm_dtype: float32`) |
| Full-attn layers | every layer (compressed) | **GQA**: 16 q heads / 2 kv heads, `head_dim: 256`, `attn_output_gate: true`, partial RoPE (`partial_rotary_factor: 0.25`, `rope_theta: 1e7`), mRoPE sections `[11,11,10]` |
| KV cache | compressed rows on every layer (disk-streamable) | true KV **only on the 1-in-4 full-attn layers**; linear layers keep a **fixed-size recurrent state** |
| MoE | 256 experts, top-6, 1 shared, expert groups, `expert_weight_scale`, sigmoid routing | **256 experts, top-8**, 1 shared, `moe_intermediate_size: 512`, `shared_expert_intermediate_size: 512`, softmax routing, `router_aux_loss_coef` |
| Hyper-connections / indexer / Sinkhorn | yes | **none** |
| Vocab | 129280 | **248320** |
| Tokenizer / chat | DeepSeek BPE, DSML tool markup | Qwen BPE, ChatML-style (`<|im_start|>`/`<|im_end|>`), bos/eos id `248044` |
| MTP | optional speculative GGUF | `mtp_num_hidden_layers: 1` (deferred) |
| Modality | text | **multimodal** (vision encoder; deferred) |

### What transfers vs what is new

- **Reusable (structure/spirit):** GGUF reader, mmap loading, the `ds4_shape`
  parameterization, quant block formats (Q2_K/Q4_K/Q8_K/IQ2_XXS dot products),
  the MoE routing *skeleton*, the session/KV-checkpoint framework, the HTTP
  server + CLI scaffolding, sampling, `gguf-tools` quantization pipeline,
  asymmetric expert-quant idea.
- **Net-new (no reuse):** Gated DeltaNet linear-attention kernels + the hybrid
  3:1 layer scheduling; GQA full attention with partial RoPE + output gate;
  Qwen BPE tokenizer + ChatML rendering; Qwen tensor-name mapping; the
  `qwen3_5moe.*` GGUF metadata namespace.
- **Removed / dormant for Qwen:** MLA compression + indexer top-k, hyper-
  connections (HC/Sinkhorn), LoRA Q/O, DeepSeek SWA window, DSML tooling.

---

## 2. Where the seams are in the code (grounded in ds4.c)

- **Architecture constants** are centralized in `struct ds4_shape` and the
  global `g_ds4_shape` (`ds4.c` ~L144–323), exposed as `DS4_N_*` macros. A new
  variant adds fields + a `DS4_SHAPE_*` constant + a `DS4_VARIANT_*` enum.
- **Per-layer policy** uses a parallel array `g_ds4_compress_ratios[]`
  (`ds4.c` ~L288). The Qwen layer-type map (linear vs full) is the analogue.
- **GGUF metadata** is read under the `deepseek4.*` namespace and
  **hard-validated** in `config_validate_model()` (`ds4.c` ~L3888), which calls
  `ds4_select_shape_from_metadata()`. `general.architecture` is read at ~L2018.
  Qwen needs a sibling `config_validate_model_qwen()` reading `qwen3_5moe.*`.
- **CPU reference forward** lives roughly `ds4.c` L10000–14000; **Metal** graph
  ~L16000+; **CUDA** in `ds4_cuda.cu`. The per-layer compute is pure DeepSeek
  and is what Phase 5/7 replace for the Qwen variant.
- **Build**: `Makefile` is `uname`-based. CPU reference builds via `make cpu`
  (`-DDS4_NO_GPU`, `ds4.c`→`ds4_cpu.o`); CUDA via `nvcc`. No Windows path →
  build/test in WSL2.

### Engineering approach: additive variant, then consolidate

To keep the tree **buildable and reviewable at every commit**, we add a Qwen
variant *alongside* DeepSeek first (new shape, new validators, new forward
paths dispatched by `DS4_MODEL_VARIANT`). This temporarily violates the
"one model, no variants" rule on purpose. A **final cleanup phase** removes the
DeepSeek-only paths (MLA, indexer, HC, DSML) per the project's single-model
philosophy, once the Qwen path is real. Naming (`ds4_*` → a Qwen prefix, binary
rename) is an optional last step to minimize churn during the functional port.

---

## 3. Phased plan

Each phase is a group of atomic commits (see `QWEN36_PORT_COMMITS.md`).

- **Phase 0 — Docs & setup.** This plan, the commit map, an architecture
  reference extracted from `config.json`, build/`.gitignore` notes.
- **Phase 1 — Build & identity.** WSL2+CUDA build notes; Makefile target
  alias; no behavior change.
- **Phase 2 — Shape & config.** `DS4_VARIANT_QWEN36`, `DS4_SHAPE_QWEN36`, new
  `ds4_shape` fields, per-layer type map, `qwen3_5moe.*` metadata namespace,
  architecture-dispatched config validation.
- **Phase 3 — Tokenizer & chat.** Qwen BPE vocab/merges from GGUF; Qwen special
  tokens; ChatML rendering; bypass DSML for the Qwen variant.
- **Phase 4 — Weights.** Qwen tensor-name table + per-layer weight struct;
  loader mapping; skip vision tensors; shape validation.
- **Phase 5 — CPU reference forward (the core).** Embedding → per-layer
  dispatch → final norm → lm_head. Sub-commits: skeleton+norm/residual; GQA
  full attention (+partial RoPE, output gate, KV); Gated DeltaNet projections +
  conv1d + gating; Gated DeltaNet delta-rule recurrence + output norm; MoE
  (top-8 softmax router, SwiGLU experts, shared expert); output head + greedy
  single-forward test.
- **Phase 6 — Session, state & generation.** Hybrid KV + recurrent-state
  lifetime in `ds4_session` (sync/rewind/checkpoint); sampling + generation
  loop; CLI wiring for the Qwen variant.
- **Phase 7 — CUDA.** Port the hot paths to `ds4_cuda.cu`: GQA attention,
  Gated DeltaNet recurrence, MoE expert matmuls; wire the GPU graph driver.
- **Phase 8 — Tooling & consolidation.** `gguf-tools` Qwen conversion +
  asymmetric expert quant; retire DeepSeek-only code; optional rename.

### Validation strategy

- **Now (structural):** shapes, tensor wiring, token round-trips, dtype/stride
  sanity, no-NaN forward on random/zeroed weights, compile-clean per phase.
- **Later (numeric):** once weights exist, compare per-layer activations and
  final logits against HF `transformers` on fixed prompts at several context
  lengths — mirroring ds4's official-vector gate.

---

## 4. Risks & open questions

1. **Gated DeltaNet correctness** is the dominant risk; the delta-rule + gating
   math must match Qwen's reference exactly. Hard to verify without weights.
2. **No GGUF exists** in ds4's format for this model. Phase 8 must define the
   conversion; until then loading is exercised with a synthetic/converted GGUF.
3. **mRoPE** sections `[11,11,10]`: for text-only, position ids collapse to the
   standard 1-D case, but the partial-rotary split (factor 0.25 of `head_dim`
   256 → 64 rotary dims) must be implemented exactly.
4. **WSL2 CUDA arch**: set `CUDA_ARCH=sm_XX` for the dev GPU in Phase 7.
5. **Memory**: 35B params even at 2-bit is ~9–10 GB; full bf16 ~70 GB. Quant +
   (optional) expert SSD streaming matter for the dev box; reuse ds4's streaming.
