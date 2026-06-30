# Qwen3.6-35B-A3B Port — Atomic Commit Map

Companion to `QWEN36_PORT_PLAN.md`. Each commit is intended to be small,
self-contained, and leave the tree **buildable** (CPU/`make cpu` first; CUDA
once Phase 7 lands). Status legend: ⬜ todo · 🟦 in progress · ✅ done.

Commit message convention: `feat(qwen)/refactor(qwen)/docs(qwen)/chore(qwen): …`

> No local compiler on the dev box (Windows); each commit is validated
> **structurally** and is expected to be `make cpu`-built under WSL2 by the dev.

---

## Phase 0 — Docs & setup

- ✅ **C01** `docs(qwen): add port plan and atomic commit map`
  Adds `QWEN36_PORT_PLAN.md` + this file.
- ⬜ **C02** `docs(qwen): add Qwen3.6 architecture reference from config.json`
  `QWEN36_ARCH.md` with the exact config fields, tensor inventory, and the
  derived per-layer type schedule. Add build artifacts already covered by
  `.gitignore`; note WSL2 build.

## Phase 1 — Build & identity

- ⬜ **C03** `chore(qwen): document WSL2+CUDA build and add make alias`
  Build notes in README/section; `make cuda-qwen` alias (delegates to existing
  CUDA target). No behavior change.

## Phase 2 — Shape & config

- ⬜ **C04** `feat(qwen): add Qwen3.6 model variant and shape`
  `DS4_VARIANT_QWEN36` enum value; extend `ds4_shape` with Qwen fields
  (`full_attention_interval`, `n_head_dim_full=256`, `n_head_kv=2`,
  `linear_*` dims, `moe_intermediate`, `shared_intermediate`,
  `partial_rotary_factor`, `n_expert_used=8`, etc.); add `DS4_SHAPE_QWEN36`
  (40 layers, 2048 embd, 248320 vocab). DeepSeek shapes leave new fields zero.
- ⬜ **C05** `feat(qwen): per-layer attention-type schedule`
  `g_qwen_layer_is_full[]` derived from `full_attention_interval` (or GGUF
  array), analogous to `g_ds4_compress_ratios[]`; accessor helper.
- ⬜ **C06** `feat(qwen): dispatch config validation by architecture`
  Branch on `general.architecture`: `qwen3_5_moe` → new
  `config_validate_model_qwen()` reading the `qwen3_5moe.*` namespace and
  selecting `DS4_SHAPE_QWEN36`; DeepSeek arch keeps existing path.

## Phase 3 — Tokenizer & chat

- ⬜ **C07** `feat(qwen): load Qwen BPE tokenizer from GGUF`
  Vocab/merges/byte-fallback for the Qwen tokenizer; special-token id table
  (bos/eos `248044`, im_start/im_end, vision sentinels parsed but unused).
- ⬜ **C08** `feat(qwen): ChatML prompt rendering for the Qwen variant`
  `ds4_encode_chat_prompt`/append helpers render ChatML for `DS4_VARIANT_QWEN36`
  and bypass DSML; think-mode markers mapped to Qwen conventions.

## Phase 4 — Weights

- ⬜ **C09** `feat(qwen): Qwen tensor-name table and per-layer weight struct`
  Names for embeddings, input/post-attn norms, full-attn q/k/v/o(+gate),
  linear-attn (in-proj, conv1d, A/dt/decay, out-norm/gate), MoE
  (router, gate/up/down experts, shared expert), final norm, lm_head.
- ⬜ **C10** `feat(qwen): map GGUF tensors into Qwen weights`
  Loader binds tensors per layer-type; skips `vision_*`; validates dims against
  the shape; mmap-backed like the DeepSeek path.

## Phase 5 — CPU reference forward (core)

- ⬜ **C11** `feat(qwen): CPU forward skeleton (embed, dispatch, norm, head)`
  `qwen_cpu_*` scratch + forward that does embedding lookup, per-layer dispatch
  stub, final RMSNorm, lm_head. Residual-only layers so it runs end-to-end.
- ⬜ **C12** `feat(qwen): CPU full-attention layer (GQA + partial RoPE + gate)`
  16 q / 2 kv heads, head_dim 256, partial rotary (64 dims, theta 1e7), output
  gate, causal KV over full-attn layers only.
- ⬜ **C13** `feat(qwen): CPU Gated DeltaNet — projections, conv1d, gating`
  in-proj split (q/k/v/decay/gate), depthwise conv1d (kernel 4) + SiLU, gate
  activations; no recurrence yet.
- ⬜ **C14** `feat(qwen): CPU Gated DeltaNet — delta-rule recurrence + out norm`
  Per-step gated delta-rule state update over the value/key heads, output
  projection, RMS/gate. The numeric heart of the port.
- ⬜ **C15** `feat(qwen): CPU MoE block (top-8 softmax router + shared expert)`
  Softmax router, top-8 selection, SwiGLU experts (intermediate 512), shared
  expert, weighted combine.
- ⬜ **C16** `feat(qwen): greedy single-forward self-test for the Qwen variant`
  A `--qwen-forward-test` entry that runs one forward on a tiny prompt and
  dumps top logits; structural NaN/shape checks.

## Phase 6 — Session, state & generation

- ⬜ **C17** `feat(qwen): hybrid KV + recurrent-state session lifetime`
  Session holds full-attn KV *and* per-linear-layer recurrent state; sync,
  rewind, and checkpoint honor both.
- ⬜ **C18** `feat(qwen): sampling and generation loop for Qwen`
  Wire `ds4_session_argmax`/`sample` + the CLI generate path to the Qwen
  forward; ChatML stop tokens.

## Phase 7 — CUDA

- ⬜ **C19** `feat(qwen): CUDA GQA full-attention kernel`
- ⬜ **C20** `feat(qwen): CUDA Gated DeltaNet recurrence kernel`
- ⬜ **C21** `feat(qwen): CUDA MoE expert matmul + routing`
- ⬜ **C22** `feat(qwen): wire CUDA graph driver for the Qwen variant`

## Phase 8 — Tooling & consolidation

- ⬜ **C23** `feat(qwen): gguf-tools Qwen conversion + asymmetric expert quant`
- ⬜ **C24** `refactor(qwen): retire DeepSeek-only paths (MLA/indexer/HC/DSML)`
- ⬜ **C25** `chore(qwen): optional engine rename and docs sweep`

---

## Working agreement for the loop

For each commit: read its spec here → implement → stage → commit (message above)
→ push to `feature/qwen36-port` → mark ✅ here. The numeric commits (C13–C16,
C19–C21) are the ones that will ultimately need real weights + an HF reference
to validate beyond structure; they are written to be correct-by-construction
until then.
