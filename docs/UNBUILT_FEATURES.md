# QAI Framework — Focus List (What Is Genuinely Not Built Yet)

> See also: [README](../README.md) · [Module Registry](MODULE_REGISTRY.md) · [Roadmap](ROADMAP.md) · [User Manual](USER_MANUAL.md) · [Task Manual](guides/README.md)

**This is the only document that tracks remaining work.** Every feature not on
this list is built, tested, and documented — see the
[Module Registry](MODULE_REGISTRY.md) for the complete A–Z of what exists.

The old version of this file mixed two things together: a list of everything
built (which belongs in the README and the Task Manual) and a list of what was
still missing. It also carried two stale claims — `load_region()` and
peer-to-peer `connect_peer` were listed as "still missing" when both have been
built for a long time (`qai.WeightRegionLoader.load_region()` /
`unload_region()` in `qai/upgrades.py`, and `qai.PeerConnection` with
listen/connect modes in `qai/upgrades.py`). Those items are now removed. What
remains here is the real, verified focus list.

All 108 spec sections are implemented. The items below are **residual gaps
inside those implementations** — spots where a function is a documented
stand-in, a fallback path returns a placeholder, or a loader defers to an
external tool. Each item cites the file and line where the gap lives, so you
can verify every claim yourself.

> **v1.8.4 update (2026-11-27):** External review found four bugs in
> already-"built" features that this focus list didn't track because they
> weren't "unbuilt" — they were "built but wrong". All four are now fixed:
> (1) `.qmodel` signature now covers `logic.py` (security: tampered logic
> used to pass `verify_bundle` and execute at load time), (2) `TimeBudget`
> now actually interrupts long-running functions via SIGALRM, (3) module
> constants now travel with bundled logic, (4) `import qai` works without
> PyTorch. See `CHANGELOG.md` v1.8.4 for full details.
>
> **v1.8.5 update (2026-11-27):** Focus-list item 2 (real distillation loss)
> is resolved — the five scattered distillation implementations were
> consolidated into one canonical module (`qai.distillation_v2`) with the
> real Hinton KL loss and one entry point (`qai.distill(...)`).
>
> **v1.8.6 update (2026-11-27):** Focus-list item 1 (structured pruning)
> is resolved — `qai.structured_prune()` now actually removes neurons and
> rebuilds downstream layers (was a `not_implemented` stub).
>
> **v1.8.7 update (2026-11-27):** Focus-list item 3 (`bayesian_optimize`
> random stand-in) is resolved — all five scattered "Bayesian" code paths
> (the fake `qai.bayesian_optimize()`, `AutoTuner(strategy="bayesian")`,
> the two same-named `HyperparameterOptimizer` classes, and
> `HyperOptimizer(strategy="bayesian")` which was literally the random
> branch) now delegate to ONE canonical GP + Expected-Improvement engine
> in `qai.hyperparam_optimizer`, with every candidate scored by real
> cross-validation.
>
> **v1.8.8 update (2026-11-27):** Focus-list item 4 (GPU `conv2d` numpy
> fallback) is resolved — without PyTorch, `qai.gpu.conv2d()` now computes
> a real vectorized cross-correlation (`sliding_window_view` + `einsum`)
> instead of returning the input unchanged. 6 items below remain the
> genuine "not yet implemented" list.

---

## The focus list (priority order)

### 1. ~~Structured pruning — `qai.governance.extended`~~ — FIXED in v1.8.6
- **Resolved**: structured pruning was consolidated into ONE canonical
  implementation (`qai/pruning.py`) that actually removes neurons/filters
  and rebuilds downstream shapes — not just zeroing weights. The entry
  point is `qai.structured_prune(model, layer_name=..., fraction=...)`.
  `qai.governance.PruningPipeline.structured_prune`,
  `qai.compression.Pruner.structured_prune`, and
  `qai.surgery.ChannelPruner.remove` are all delegates. The old
  `ChannelPruner.prune` (zero-mask) is kept for backward compat and now
  honestly says "zeroed, not removed" in its return value. Enforced by
  `tests/test_structured_pruning.py` (17 tests).

### 2. ~~Real distillation loss in `qai.distillation_v2`~~ — FIXED in v1.8.5
- **Resolved**: distillation was consolidated into ONE canonical
  implementation (`qai/distillation_v2.py`) with the real Hinton loss
  (`alpha * CE + (1-alpha) * T² * KL`) running on the torch path, and an
  honest confidence-weighted / hard-label fallback for numpy-only
  techniques. `qai.advanced.KnowledgeDistiller`, `qai.compression.Distiller`,
  `qai.KnowledgeDistiller2`, and `qai.ModelDistiller` are now thin delegates
  with zero algorithm code of their own (enforced by
  `tests/test_distillation_consolidated.py`). The user-facing entry point
  is `qai.distill(teacher, student, X, y, strategy=..., temperature=...,
  alpha=...)` — one feature, parameters for the rest.

### 3. ~~`qai.bayesian_optimize()` is a random-search stand-in~~ — FIXED in v1.8.7
- **Resolved**: all five scattered "Bayesian" implementations were
  consolidated into ONE canonical engine (`qai.hyperparam_optimizer`
  — RBF-kernel GP posterior + Expected Improvement). The public entry
  `qai.bayesian_optimize()` now scores every candidate with real k-fold
  cross-validation of a real qai model (its `best_loss` used to be
  `np.random.uniform(0.01, 0.5)`, never touching the model).
  `AutoTuner(strategy="bayesian")`, `HyperOptimizer(strategy="bayesian")`
  and `qai.hparam_optimizer` (which carried a second class with the same
  name) are all delegates. Enforced by `tests/test_bayesian_consolidated.py`
  (15 tests, including a spy proving every entry point calls the same
  `optimize`).

### 4. ~~GPU `conv2d` fallback without PyTorch~~ — FIXED in v1.8.8
- **Resolved**: the numpy fallback of `qai.gpu.conv2d()` now computes a
  real cross-correlation — vectorized as
  `numpy.lib.stride_tricks.sliding_window_view` + `einsum` (the import
  that sat unused since the placeholder was written) — with torch
  semantics (no kernel flip). Supports (C_in,H,W) and (N,C_in,H,W)
  inputs, (C_out,C_in,kH,kW) kernels (plus two 3-dim tolerations),
  int or (h,w) stride/padding, and per-channel bias. Enforced by
  `tests/test_gpu_conv2d.py` (13 tests): parity against a naive loop
  reference across strides/paddings, a torch-parity test that activates
  automatically when torch is installed, and honest [WHAT]/[WHY]/[FIX]
  errors for channel mismatch / oversized kernels / bad ndims.

### 5. GGUF model loading — `qai.formats`
- **Gap**: `ModelIO.load(".gguf")` raises `NotImplementedError` and defers to
  llama.cpp's external `convert_hf_to_gguf.py`; GGUF *saving* is likewise a
  stub that shells out.
- **Works today**: `qai.conversion.import_model(".gguf")` already reads real
  GGUF tensors via the `gguf` pip package (name → ndarray map), and
  safetensors / onnx / torchscript / coreml load paths all work.
- **To build**: reuse the `gguf`-package reader inside `qai.formats.ModelIO`
  (and a metadata-aware writer) so one import surface handles GGUF end to end.
- **Evidence**: `qai/formats.py` — `.gguf` branch in `_load()` and `_save()`.

### 6. `import_model` for tflite / openvino / tensorrt — `qai.conversion`
- **Gap**: *exporting* to tflite / openvino / tensorrt works, but
  `import_model()` raises `NotImplementedError` for those three and tells you
  to use each format's native runtime (`tensorflow.lite.Interpreter`,
  `openvino.runtime.Core`, `tensorrt.Runtime`).
- **To build**: thin loaders that wrap those three native runtimes behind the
  same predict-like interface the other import paths return. (Alternatively,
  declare these export-only in the docs — decision recorded here so it is
  visible.)
- **Evidence**: `qai/conversion.py` — final `raise NotImplementedError` in
  `import_model()`.

### 7. OAuth callback receiver — `qai.network`
- **Gap**: `OAuthConnector` builds correct authorize URLs and exchanges codes,
  but you must host your own local HTTP server to catch the OAuth redirect
  (`?code=...`) before calling `exchange_code()`.
- **Works today**: the full code-exchange + refresh-token flow for Google,
  GitHub, Slack, Salesforce, Microsoft; HMAC request signing; the `qai.api`
  client.
- **To build**: an opt-in `connector.listen_for_code(port=...)` helper that
  runs a short-lived localhost HTTP listener and returns the auth code
  (stdlib `http.server` is enough).
- **Evidence**: `qai/network/__init__.py` — `OAuthConnector` docstring
  ("you must run a local HTTP server to receive the callback").

### 8. Unsupervised scoring in `qai.compare_techniques` / `auto_train`
- **Gap**: for unsupervised techniques (no `y`), the comparison runner sets
  `accuracy = 0.0` as a placeholder instead of a real score.
- **To build**: use silhouette score / inertia / Davies–Bouldin (all
  computable with numpy) so unsupervised technique comparison ranks
  meaningfully, and record the metric name in `TechniqueResult`.
- **Evidence**: `qai/auto_ml.py` — `compare_techniques()` unsupervised branch,
  `accuracy = 0.0  # placeholder`.

### 9. Agent planner argument extraction — `qai.advanced`
- **Gap**: the tool-using agent's `_extract_args()` is keyword/regex
  heuristic ("a real planner would use an LLM here", per the comment) and
  `_is_task_complete()` fires after 3+ tool calls.
- **Works today**: planning loop, tool selection, error recovery, execution
  log — the skeleton is real; only argument extraction is simplistic.
- **To build**: optional LLM-backed planner (works with any qai language
  model) behind the same interface, keeping the heuristic as the offline
  fallback.
- **Evidence**: `qai/advanced.py` — `_extract_args()`.

### 10. CLIP-style multimodal similarity is approximate
- **Gap**: the image–text similarity helper in `qai.advanced_metrics` uses an
  approximate embedding ("real CLIP uses learned embeddings", per comment).
- **Works today**: all classical metrics (CIDEr-style n-gram scores, BLEU,
  semantic-consistency proxies).
- **To build**: when `transformers` is installed, load a real CLIP model
  (lazy import, same pattern as `WhisperASR` / `Wav2Vec2` / `HuBERT` in
  `qai.techniques.modern`, which already do real HF loading).
- **Evidence**: `qai/advanced_metrics.py` — placeholder comment.

---

## Verification method (so this list stays honest)

1. `grep` the whole `qai/` tree for `NotImplementedError`, `TODO`, `stub`,
   `placeholder`, `not implemented`, `coming soon` — every hit is triaged:
   real gap → this list; false positive (e.g. an abstract base class raising
   `NotImplementedError` by design, or an HTML input placeholder attribute)
   → ignored.
2. Module-coverage check: every module under `qai/` is searched for across
   all docs — anything undocumented goes to the
   [Module Registry](MODULE_REGISTRY.md), not here.
3. Any item fixed is moved out of this list with a note in the
   [CHANGELOG](../CHANGELOG.md).

## Everything else is built

Everything not listed above is implemented, tested, and documented — 108/108
spec sections plus the 2026 additions. Do not look here for what the framework
can do; look at the [Module Registry](MODULE_REGISTRY.md) (complete A–Z) or
the [Task Manual](guides/README.md) (navigate by what you want to accomplish).
The items above are internal quality fixes, not missing features.
