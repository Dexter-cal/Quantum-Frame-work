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

---

## The focus list (priority order)

### 1. Structured pruning — `qai.governance.extended`
- **Gap**: `structured_prune()` returns `{"status": "not_implemented"}` — it
  removes nothing.
- **Works today**: magnitude pruning (`magnitude_prune()`), activation
  sparsity, weightHistogram reports.
- **To build**: actually remove neurons/channels — propagate a boolean mask
  through the layer and rebuild downstream shapes (row/col slicing of
  `nn.Linear` weights + bias, channel slicing of `nn.Conv2d`).
- **Evidence**: `qai/governance/extended.py` — `structured_prune()` docstring
  says "currently a stub".

### 2. Real distillation loss in `qai.distillation_v2`
- **Gap**: `Distiller._train_epoch()` trains the student normally and returns
  a placeholder loss `0.1`. The KL(soft) + CE(hard) loss described in the
  config (`temperature`, `alpha`) is never applied — so v2 currently behaves
  like "train a smaller model", not like true distillation.
- **Works today**: `qai.KnowledgeDistiller` (in `qai/advanced.py`) implements
  the *real* temperature-softened KL loss with PyTorch
  (`F.kl_div(log_softmax(student/T), softmax(teacher/T)) * T²`) — use it for
  genuine soft-target distillation today.
- **To build**: port the KnowledgeDistiller loss into
  `distillation_v2.Distiller` so the `vanilla` / `hint` / `attention` /
  `self` / `multi` strategies all train through the soft loss, and report the
  real combined loss in `DistillationResult`.
- **Evidence**: `qai/distillation_v2.py` `_train_epoch()` — `return 0.1  # placeholder loss`.

### 3. `qai.bayesian_optimize()` is a random-search stand-in
- **Gap**: `qai.core.autotune.bayesian_optimize()` samples parameters and
  scores candidates with `np.random.uniform(0.01, 0.5)` — the "loss" is not
  computed from a model at all. Its own docstring admits it: "simplified
  random-search stand-in, not real Bayesian optimization".
- **Works today**: `qai.hyperparam_optimizer.HyperparameterOptimizer`
  implements true GP-surrogate Bayesian optimization with an acquisition
  function, and `qai.AutoTuner` does real k-fold cross-validated grid search.
  Real BO exists in the package — it is just not wired into this convenience
  function.
- **To build**: delegate `bayesian_optimize()` to
  `hyperparam_optimizer.HyperparameterOptimizer` (or evaluate each candidate
  with a real CV fit) and deprecate the random loss.
- **Evidence**: `qai/core/autotune.py` — `bayesian_optimize()`.

### 4. GPU `conv2d` fallback without PyTorch
- **Gap**: when PyTorch is absent, `qai.gpu.conv2d()` returns the input
  tensor unchanged instead of computing the convolution (the numpy fallback
  is marked "simplified").
- **Works today**: every other `qai.gpu` kernel (`matmul`, `gather`, `softmax`
  …) has a working numpy fallback; `conv2d` is the one exception.
- **To build**: implement the correlation with
  `numpy.lib.stride_tricks.sliding_window_view` + einsum (the import is
  already there, unused).
- **Evidence**: `qai/gpu.py` — `conv2d()` numpy branch, comment
  "this is just a placeholder".

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
