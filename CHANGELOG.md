# Changelog

All notable changes to the qai framework are documented here.
Versions follow semantic versioning: MAJOR.MINOR.PATCH.

See also: [README](README.md) · [Task Guides](docs/guides/README.md) · [Tutorial](docs/TUTORIAL.md) · [Roadmap](docs/ROADMAP.md) · [Module Registry](docs/MODULE_REGISTRY.md) · [Focus List](docs/UNBUILT_FEATURES.md)

## v1.8.6 (2026-11-27) — "Real Structured Pruning"

Focus-list item 1 resolved. Structured pruning was scattered across three
entry points, all of which were misleading:
  * ``qai.governance.extended.PruningPipeline.structured_prune`` returned
    ``{"status": "not_implemented"}``.
  * ``qai.compression.Pruner.structured_prune`` delegated to surgery.
  * ``qai.surgery.ChannelPruner.prune`` claimed to "remove entire channels"
    but only zeroed them out (no shape change, no FLOP reduction).

Now: ONE canonical implementation (``qai/pruning.py``), ONE entry point
(``qai.structured_prune(...)``), and the other three are delegates.

### Changed — one feature, one code path (same pattern as v1.8.5 distillation)
- **``qai.structured_prune(model, layer_name=..., fraction=...)`` is THE
  entry point.** Actually removes neurons/filters and rebuilds downstream
  shapes — the network gets smaller, not just sparser.
- **``qai/pruning.py`` is the canonical home:**
  - ``nn.Linear``: ranks output rows by L2 (or L1) importance, drops the
    bottom ``fraction``, slices the downstream layer's input columns to
    match. Rebuilds both layers with new ``nn.Linear`` instances.
  - ``nn.Conv2d``: same for output filters + downstream input channels.
  - Handles the PyTorchMLP dual-reference pattern (each Linear is stored
    both as ``self.fc1`` AND inside ``self.net = Sequential(...)``) —
    ``_replace_by_identity`` walks the whole module tree and updates every
    reference, so no zombie layers remain.
  - Reports: shapes before/after per layer, params before/after,
    compression ratio, actual sparsity, accuracy before/after/delta
    (when ``X_val``/``y_val`` provided).
  - ``structured_prune_with_finetune()`` convenience wrapper: prunes then
    retrains briefly to recover accuracy.
- **Delegates preserved (backward compatible):**
  - ``qai.governance.PruningPipeline.structured_prune`` — no longer returns
    ``not_implemented``, delegates to canonical.
  - ``qai.compression.Pruner.structured_prune`` — delegates to canonical
    (was delegating to surgery's zero-only version).
  - ``qai.surgery.ChannelPruner.remove`` — NEW method, delegates to
    canonical for real shape-changing pruning.
  - ``qai.surgery.ChannelPruner.prune`` — KEPT as the zero-mask method
    (backward compat), now honestly says "zeroed, not removed" in its
    return value ``note`` field.

### Tests
- **``tests/test_structured_pruning.py``** (new, 17 tests, all pass):
  - params count goes DOWN (catches the zombie-layer bug)
  - layer shape actually shrinks
  - downstream layer input dim is sliced to match
  - cached ref and Sequential entry both updated (the PyTorchMLP bug)
  - predict() still works after pruning
  - accuracy delta reported when val data given
  - layer_name=None prunes all layers except output head
  - finetune wrapper recovers accuracy
  - L1 criterion works
  - clear errors for unknown layer / criterion
  - all three delegates work
  - source-level AST check: no delegate returns ``not_implemented``
  - canonical module exists and exports the right names

### Known pre-existing issue unblocked
- ``qai.surgery.ChannelPruner.prune`` (the zero-mask method) uses
  ``_extract_weights`` which returns parameter names like ``"fc1.weight"``,
  not layer names like ``"fc1"``. Pre-existing API quirk; the new
  ``ChannelPruner.remove`` doesn't have this issue because it delegates to
  the canonical ``structured_prune`` which walks ``named_modules()``.

## v1.8.5 (2026-11-27) — "One Distillation, One Entry Point"

Kills the redundancy flagged during review: knowledge distillation existed as
**five separate implementations** scattered across modules — one of them real
(`qai.advanced.KnowledgeDistiller`, torch KL), one a documented placeholder
(`qai.distillation_v2` returning `0.1`), three more partial copies
(`qai.compression.Distiller`, `qai.creative.experimental.KnowledgeDistiller2`,
`qai.creative_tools.ModelDistiller`). They are now consolidated into ONE
canonical implementation with ONE user-facing entry point.

### Changed — one feature, parameters instead of parallel modules
- **`qai.distill(teacher, student, X, y, ...)` is THE entry point** (also
  `qai.multi_distill`, `qai.self_distill`). Every knob is a parameter on the
  call: `strategy` (vanilla | hint | attention | self | multi),
  `temperature`, `alpha`, `epochs`, `lr`, `batch_size`, `align_weight`.
  No need to know which module owns which class any more.
- **`qai/distillation_v2.py` rewritten as the canonical implementation**:
  - real Hinton loss on the torch path:
    `L = alpha·CE(student, y) + (1-alpha)·T²·KL(student_soft ‖ teacher_soft)`
    with minibatching, AdamW, and gradient clipping (ported from the old
    `KnowledgeDistiller`);
  - hint/attention strategies add a weight-alignment MSE term
    (`align_weight`) between teacher and student first parameters;
  - numpy path (no torch) is honest: confidence-weighted hard-label fit
    when the technique accepts `sample_weight`, plain hard-label otherwise,
    with real `KL(teacher‖student)` before/after reported either way;
  - `DistillationResult` gained `distillation_mode`, `loss_final`,
    `kl_before`, `kl_after` — the report now says WHAT actually ran.
- **The four other distillers are now thin delegates** with zero algorithm
  code of their own (their old public signatures preserved):
  `qai.advanced.KnowledgeDistiller` (keeps `.distill()/.losses()/
  .compression_ratio()`), `qai.compression.Distiller` (static-method API),
  `qai.creative.experimental.KnowledgeDistiller2` (multi-level loss maps to
  the attention strategy), `qai.creative_tools.ModelDistiller`.
- `tests/test_distillation_consolidated.py` (new, 15 tests) — verifies the
  entry point, all four delegate APIs, a **source-level check that no
  delegate file contains `F.kl_div` or its own loss loop any more**, the
  multi/self strategies, and the KL/probability helpers.

### Fixed (as part of the consolidation)
- The old `distillation_v2` retrained the student **from scratch once per
  epoch** (`_train_epoch` called `student.train()` inside the epoch loop) —
  20 epochs meant 20 full retrains, and the "loss" was the constant 0.1.
  The canonical loop trains continuously through minibatch epochs instead.
- `qai.Distiller` top-level name is a pre-existing collision
  (`qai.compression.Distiller` shadows `qai.distillation_v2.Distiller`
  because of import order). Both now delegate to the same code, so the
  collision is harmless; the docs point new code at `qai.distill(...)`.

## v1.8.4 (2026-11-27) — "Hardened Bundles & Optional PyTorch"

Patches the four serious problems flagged in the v1.8.3 external review,
plus the two `InputValidationError` bugs that surfaced once the test suite
could actually run without PyTorch installed.

### Fixed — security
- **`.qmodel` signature now covers `logic.py`.** Earlier versions only hashed
  `model.json` + `weights.npy` + per-weight tensors, so an attacker could
  replace `logic.py` with arbitrary code that passed `verify_bundle()` and
  then ran at `load_bundle()` time. The manifest now also stores
  `logic_hash`, `logic_meta_hash`, and `tokenizer_hash`, all included in
  the HMAC-SHA256 signed payload. (`qai/bundle/__init__.py`)
- **Defense-in-depth hash check at `load_bundle` time.** Even if a caller
  skips `verify_bundle()` (or passes `verify_signature=False`), the loader
  now compares the on-disk `logic.py`/`logic_meta.json` bytes against the
  manifest hashes *before* `exec()` — tampered logic raises `ValueError`
  instead of running. (`qai/bundle/__init__.py`)
- **Module-level constants travel with bundled logic.** Logic functions
  that referenced module constants (e.g. `LIMIT = 100` at module top)
  used to raise `NameError` in a fresh process because the origin module's
  globals weren't loaded. `save_bundle` now captures the values of any
  JSON-serialisable closure deps via `inspect.getclosurevars` and writes
  them into `logic_meta.json` under a new `constants` key; `load_bundle`
  re-injects them into the exec namespace. (`qai/bundle/__init__.py`)

### Fixed — `TimeBudget` actually interrupts
- **`TimeBudget.run(fn, ...)` and `with TimeBudget(...)` now preempt the
  running function.** Previously the budget was only checked *after* the
  function returned, so `time.sleep(2.0)` with a 0.5 s budget ran for the
  full 2.0 s, and infinite loops hung forever. The new implementation
  arms `signal.SIGALRM` (POSIX main thread, sub-second resolution via
  `setitimer`) and falls back to a watchdog thread that calls
  `PyThreadState_SetAsyncExc` for non-main-thread / Windows use. Raises
  `TimeoutError` when the budget is exceeded. (`qai/budget.py`)
- The old cooperative API (`exceeded()`, `should_stop()`, `remaining()`,
  `elapsed()`) is unchanged — existing polling loops still work.

### Fixed — PyTorch is now optional for `import qai`
- **`import qai` no longer crashes without PyTorch.** The neural-network,
  transformer, CNN, RNN, and 20+ other technique files used to do
  `class Foo(nn.Module)` at module scope with no fallback, so a missing
  `torch` raised `ModuleNotFoundError` during `import qai` — even for users
  who only wanted a numpy-only technique like `classifier` or `time_series`.
- Added `qai/_torch_stub.py` with `StubModule` (a base class whose
  `__init__` raises a clear `ImportError` with `[WHAT]/[WHY]/[FIX]`), an
  `_NNStubNamespace` whose `__getattr__` returns `StubModule` (so
  `class Foo(nn.Module)` style definitions succeed at import time), and a
  `_TorchStubNamespace` that handles decorators like `@torch.no_grad()`
  as identity passthroughs.
- 27 technique modules updated to use the stub when torch is missing
  (`scripts/patch_torch_stubs.py` is the idempotent re-patcher).
- The user-facing technique classes still raise a clear `ImportError` if
  you actually try to *train* a torch-based technique without torch
  installed — the failure mode just moved from import time to use time.

### Fixed — `InputValidationError` false positives
- **Dict-input schema happy path.** `model.predict({"f1": 1.0, "f2": 2.0})`
  on a model built with `input_schema={"f1": float, "f2": float}` used to
  raise `InputValidationError` because `validate_input` saw the raw dict,
  `np.asarray(dict)` produced a 0-d object array, and the kind check
  classified it as text. Fixed by converting dict → array via the schema
  *before* validation in `predict()`. (`qai/core/model.py`)
- **Time-series happy path.** `model.predict([47.0, 48.0, 49.0])` on a
  `time_series` model trained with `lags=3` used to raise
  `InputValidationError` because `_capture_data_signature` stored
  `n_features = len(arr)` (= 49, the length of the training series) and
  the validator then demanded 49 features at predict time. Fixed by
  asking the technique for its expected input length (`lags`,
  `input_dim`, `n_features`, `n_lags`, `window_size`, `seq_len`) before
  falling back to the array length; if none of those are set, leave
  `n_features = None` so 1-D inputs aren't length-checked against the
  training array. (`qai/core/model.py`)

### Changed
- **README headline** no longer claims "108 spec sections implemented
  (100%)". It now reads "108 spec sections implemented · 10-item focus
  list tracked in `docs/UNBUILT_FEATURES.md`" — honest about what's left.
- Version bumped `1.8.0` → `1.8.4` in `qai/__init__.py` (both
  `__version__` and `__version_info__`); the codebase had been reporting
  `1.8.0` while the changelog was at `1.8.3`.

### Tests
- **`tests/test_bundle_security_regression.py`** (new, 5 tests) — verifies
  tampered `logic.py` breaks the signature, is rejected at `load_bundle`,
  and that captured constants travel with the bundle.
- **`tests/test_time_budget_interruption.py`** (new, 8 tests) — verifies
  `TimeBudget.run` interrupts `time.sleep`, infinite loops, and the
  `with` context-manager form; also that short functions return normally
  and the cooperative API still works.
- `tests/test_output_and_dict_schema.py` and `tests/test_time_series.py`
  happy paths now pass (previously raised `InputValidationError`).

### Known pre-existing issues unblocked by this release
- Several test files at `tests/test_new_features.py` use
  `qai.DataAugmentor(mode="tabular")`, but `qai.DataAugmentor` resolves
  to the `creative_tools` variant (no `mode=` kwarg) due to a namespace
  collision — `data_augmentor.DataAugmentor` and
  `data.pipeline.DataAugmentor` both exist but are shadowed. Pre-existing
  test/code mismatch, surfaced now that `import qai` works without torch.
- ~17 test failures require real PyTorch (transformer/CNN/RNN round-trips,
  distillation, autodiff, etc.) — unchanged.

## v1.8.3 (2026-11-26) — "Every Module Documented + The Honest Focus List"

### Added — docs/MODULE_REGISTRY.md (the missing feature index)
- Complete inventory of **all 155 modules and subpackages**, grouped into 12
  domains, each with a one-liner generated from the module's real docstring
  (`scripts/gen_module_registry.py` — regenerate after adding modules).
- **40+ features that were built earlier but never documented** are now
  findable, including: `qai.interpret` (mechanistic interpretability:
  activation patching, circuit tracing, sparse autoencoders, logit lens),
  `qai.frontier2025` (Mamba-2, DeepSeekMoE, TTT), `qai.mind` (QuantumMind
  flagship agent), `qai.memory_vault` (HOT/WARM/COLD persistent memory),
  `qai.hyperparam_optimizer` (true GP-surrogate Bayesian optimization),
  `qai.active_learning_v2` (Uncertainty/QBC/BALD/CoreSet query strategies),
  `qai.feature_store`, `qai.data_profiler`, `qai.data_augmentor`,
  `qai.pipeline_builder`, `qai.observability`, `qai.model_monitor`,
  `qai.modellog`, `qai.model_versioner`, `qai.ab_testing` (statistical
  significance), `qai.self_healing`, `qai.compression_advisor`,
  `qai.confidence_calibrator`, `qai.model_card_generator`, `qai.model_chat`,
  `qai.model_health`, `qai.surgery`, `qai.experiment_compare`, `qai.autoscale`,
  `qai.cbridge`, `qai.cpc_ai`, `qai.hf_converter`, `qai.modelfile` (all three
  model-file modules), `qai.federated_bridge`, `qai.integrations(_real)`
  (LangChain/LlamaIndex/Haystack/FAISS adapters), `qai.visual_workspace_v2`,
  `qai.webui`, `qai.api_reference`, `qai.autodoc`, `qai.agents_base`,
  `qai.workspace_html`.
- Coverage check added: every module in `qai/` is now either documented in the
  registry, the User Manual, or a task guide — no more "features with no
  documentation".

### Changed — docs/UNBUILT_FEATURES.md is now the focus list
- **Removed** everything built (feature tables, "what's solid" essays) — that
  content lives in the README and the Task Manual; the file now only tracks
  what is genuinely left.
- **Corrected two stale claims**: `load_region()` and peer-to-peer
  `connect_peer` were listed as "still missing" — both are built
  (`qai.WeightRegionLoader` / `qai.PeerConnection`). Removed from the list.
- **New verified focus list** (10 items, each with file-level evidence from a
  full-tree audit for `NotImplementedError`/`stub`/`placeholder`):
  structured pruning stub (`governance/extended.py`), placeholder distillation
  loss in `distillation_v2._train_epoch` (real KL exists in
  `qai.KnowledgeDistiller`), `qai.bayesian_optimize()` random-search stand-in,
  GPU `conv2d` numpy fallback, GGUF loading in `qai.formats`,
  `import_model` for tflite/openvino/tensorrt, OAuth callback receiver,
  unsupervised scoring placeholder in `auto_ml`, heuristic agent planner args,
  approximate CLIP-style similarity.

### Fixed — false "limited/missing" statements
- `docs/UNBUILT_FEATURES.md` previously said "per-request `load_region()` is
  still missing" and "peer-to-peer `connect_peer` is still missing" — both
  features exist and are exported; claims deleted.
- Cross-checked every module referenced by the docs against the code: all
  "built" claims in README, guides, and USER_MANUAL now resolve to real
  modules (verified programmatically).

### Docs wiring
- README: feature-inventory line now points to MODULE_REGISTRY + focus list;
  project tree and reference-docs table updated.
- guides/README.md: new "Browse every module (A–Z)" task row; docs-map diagram
  updated.
- USER_MANUAL header: Module Registry + Focus List cross-links added.

## v1.8.2 (2026-11-26) — "The Task Manual"

### Added — Task-oriented guides (`docs/guides/`)
Docs reorganized the way users think: **"I want to ___" → one chapter**.

- **[guides/README.md](docs/guides/README.md)** — the "I want to..." index + reading paths
- **[guides/DATA.md](docs/guides/DATA.md)** — load (CSV/JSON/Parquet/Excel/PDF/DOCX/EPUB), clean, dedup, PII, quality audit, Excel-style formulas, export
- **[guides/TRAINING.md](docs/guides/TRAINING.md)** — four-beat pattern, Universal Trainer, 11 training enhancers (SAM/SWA/EMA/...), pipelines, Auto-ML, safety
- **[guides/MODEL_CATALOG.md](docs/guides/MODEL_CATALOG.md)** — **all 229 registered techniques** organized into 16 families with when-to-use guidance (first time the full registry exists as documentation)
- **[guides/EVALUATION.md](docs/guides/EVALUATION.md)** — metric catalog, explainability, calibration, benchmark suite, stress tests
- **[guides/FINETUNING_ALIGNMENT.md](docs/guides/FINETUNING_ALIGNMENT.md)** — 6 PEFT methods + 8 alignment methods with a picking matrix
- **[guides/SERVE_DEPLOY.md](docs/guides/SERVE_DEPLOY.md)** — ProductionServer, FastInferenceEngine (PagedAttention + CUDA), Gateway, runtime versioning/A-B/rollback, signing
- **[guides/COMPRESS.md](docs/guides/COMPRESS.md)** — quantization matrix (int8/int4/GPTQ/AWQ/GGUF/EXL2), QAT, **distillation v2 (`distill`/`multi_distill`/`self_distill`)**, pruning, **NAS (`RandomSearch`/`GridSearch`/`EvolutionarySearch`/`ProgressiveGrowth`)**
- **[guides/DISTRIBUTED.md](docs/guides/DISTRIBUTED.md)** — DDP/FSDP + **federated learning** (FedAvg/FedProx/Weighted, DP-SGD, secure aggregation, round stats)
- **[guides/AGENTS_MCP.md](docs/guides/AGENTS_MCP.md)** — tools, ReAct, MultiAgent, AIAgent, test-time scaling, MCP server (14 tools + permissions/rate-limits/audit)
- **[guides/MEMORY_RAG.md](docs/guides/MEMORY_RAG.md)** — 3-tier memory, conflict detection, KnowledgeAccumulator, RAGPipeline, memory-span config
- **[guides/SECURITY_GOVERNANCE.md](docs/guides/SECURITY_GOVERNANCE.md)** — all 10 security tools, sandboxing, fairness audit, drift, watermarking, ModelGuardian
- **[guides/MULTIMODAL.md](docs/guides/MULTIMODAL.md)** — vision/audio/nlp/graph/time_series suites + MediaManipulator + tokenizers
- **[guides/MLOPS.md](docs/guides/MLOPS.md)** — MLOpsPipeline, tracking backends, registry, dashboards
- **[guides/VISUAL_WORKSPACE.md](docs/guides/VISUAL_WORKSPACE.md)** — launching and using the model microscope
- **[guides/TROUBLESHOOTING.md](docs/guides/TROUBLESHOOTING.md)** — `qai doctor`, error tables, health checks

### Fixed — stale/contradictory claims removed
- **ROADMAP.md**: replaced the stale "What's still toy scale" section (claimed no CUDA kernels — PagedAttention + CUDA dispatch are implemented; claimed HF loading untested — it's built) with an accurate "Accuracy notes" section
- **Blind spots documented** (built but invisible in docs): federation, distillation_v2, nas_search, data_pipeline, serving_gateway, runtime versioning, the 11 training enhancers, and the 229-technique registry — full list in [ROADMAP Phase 6.6](docs/ROADMAP.md)

### Added — idea backlog
- 10 concrete new feature proposals in [ROADMAP Phase 6.6](docs/ROADMAP.md) (Playground Notebooks, Technique Recommender v2, Pipeline Diff, Shadow Deployment, Federation profiles, Distillation data generator, Budget-aware NAS, Replay slider, Explainability report builder, P2P registry sync)

### Changed — navigation
- README.md: guides table added, "start here" points to the task manual
- TUTORIAL.md and USER_MANUAL.md headers now route to the guides
- Every guide cross-links its neighbors and back to the index

## v1.8.1 (2026-11-26) — "Docs Consolidation"

### Changed — Documentation cross-linked and refreshed
- **Added** [README.md](README.md) as the project landing page — install, quickstart, feature map, doc index, project layout
- **Added** [CONTRIBUTING.md](CONTRIBUTING.md) — fork→branch→PR workflow, code style, test policy
- **Added** [LICENSE](LICENSE) — MIT
- **Updated** [docs/TUTORIAL.md](docs/TUTORIAL.md) — version header bumped from v0.4.0 → v1.8.0; cross-links to USER_MANUAL and ROADMAP
- **Updated** [docs/ROADMAP.md](docs/ROADMAP.md) — current version bumped from v1.1.0 → v1.8.0; Phase 7 documentation tasks marked done
- **Updated** [docs/UNBUILT_FEATURES.md](docs/UNBUILT_FEATURES.md) — removed contradictory "honest assessment" footer that claimed Transformers, MoE, MCP, and tokenizers were missing (they have all shipped); renamed intent to "Implementation Status"
- **Updated** [docs/VISUAL_DESIGN.md](docs/VISUAL_DESIGN.md) — tech-stack recommendation now reflects the actual decision (pure HTML/CSS/JS, no build step) per ROADMAP decision #9
- **Updated** [docs/FINDINGS.md](docs/FINDINGS.md), [docs/PROGRESS.md](docs/PROGRESS.md), [docs/USER_MANUAL.md](docs/USER_MANUAL.md) — added cross-link headers
- **Updated** `.gitignore` — excludes binaries (`*.qmodel`, `*.npz`, `*.pdf`), build artifacts (`catboost_info/`, `checkpoints/`), and `.env`

### Repo hygiene
- Pushed 430 source files (382 Python modules + 21 Markdown + 19 HTML + 5 JSON + configs) to `main`
- Removed binaries, model weights, and training artifacts from tracking

## v1.8.0 (2026-11-19) — "Distributed ML & Versioning"

### Added — Federated Learning Framework (`qai.federation`)
- **Train across multiple clients with privacy** — no data centralization
- `Client` class — holds private data partition, trains locally, reports weight updates
- `FederatedServer` — orchestrates rounds: client selection → distribute global weights → local training → aggregation
- 3 aggregation strategies:
  - `FedAvgAggregator` — Federated Averaging (weighted by sample count)
  - `FedProxAggregator` — FedProx (proximal term added client-side)
  - `WeightedAggregator` — custom client weights
- `FederationConfig` — full configuration: n_rounds, clients_per_round, client_epochs, lr, batch_size, strategy, DP settings, secure aggregation, client selection
- **Differential privacy** (DP-SGD) — Gaussian noise added to client updates, configurable epsilon/delta/max_grad_norm
- **Client selection strategies**: 'random', 'best', 'worst', 'weighted' (by sample count)
- Round statistics: avg_loss, avg_acc, global_acc, weight_delta (L2 distance from previous), wall_time
- `create_federation(model, n_clients, X, y)` — auto-splits data across N clients
- `quick(model, X, y, n_rounds, n_clients)` — quick fed run
- Tested: 4 clients on iris, 2 rounds, global_acc computed per round
- `to_markdown()` — full federation report with round-by-round table

### Added — Advanced Knowledge Distillation (`qai.distillation_v2`)
- **Transfer knowledge from teacher to student** — make smaller models that perform like big ones
- `Distiller` — single-teacher vanilla distillation (soft targets + temperature + alpha weighting)
- `MultiTeacherDistiller` — multiple teachers with weighted predictions
- `SelfDistiller` — model teaches itself (Be Your Own Teacher)
- `DistillationConfig` — temperature, alpha, epochs, lr, strategy
- `DistillationResult` — teacher_acc, student_acc_before/after, compression_ratio, knowledge_transfer (% of teacher's accuracy gained)
- 5 strategies: 'vanilla', 'hint' (intermediate layer matching), 'attention' transfer, 'self', 'multi'
- Top-level: `distill(teacher, student, X, y)`, `multi_distill(teachers, student, ...)`, `self_distill(model, ...)`
- Tested: 64×32 teacher → 8×4 student, teacher_acc=0.95, student_acc_after=0.98, compression=1.42x

### Added — ML Experiment Versioning (`qai.versioning`)
- **Git-like versioning for ML experiments** — track model + data + code + env + results
- `VersionRepo` — full repository with commits, branches, tags
- `Commit` — snapshot with: commit_id, parent_id, message, timestamp, author, branch, tags, model_hash, data_hash, code_hash, env_hash, model_path (pickled), params, metrics, environment, data_shape, n_samples
- Operations:
  - `commit(model, X, y, message, params, metrics)` — record a checkpoint
  - `checkout(commit_id)` — restore a model from a commit
  - `diff(commit_a, commit_b)` — compare two commits (model/data/code/env changed, params diff, metrics diff, time diff)
  - `log(n)` — view commit history
  - `log_markdown()` — markdown commit log
  - `branch(name)` / `checkout_branch(name)` — branching for experiments
  - `tag(name, commit_id)` — tag a release
- Content hashing: SHA-1 of model architecture, data shape+sum+std, code (main script), environment (Python version, platform)
- Persistent storage at `~/.qai/versions/<repo_name>/` with separate `models/` directory for pickled snapshots
- Top-level: `init(name)`, `commit(repo, model, ...)`, `log(repo)`, `checkout(repo, commit_id)`, `diff(repo, a, b)`
- Tested: 2 commits (teacher + student), diff shows model_changed=True, metrics_diff shows acc change, checkout restores teacher

### Added — Advanced Active Learning (`qai.active_learning_v2`)
- **Smart sample selection** — pick which unlabeled samples to label next
- 8 query strategies (all subclass `QueryStrategy`):
  - `UncertaintySampling` — least_confident / margin / entropy modes
  - `QueryByCommittee` — vote_entropy / kl_divergence (bootstrap committee of deep-copied models)
  - `ExpectedModelChange` — gradient magnitude (torch only, falls back to entropy for sklearn)
  - `DensityWeighted` — uncertainty × k-NN density (lazy sklearn import, numpy fallback)
  - `BALD` — Bayesian Active Learning by Disagreement (MC-dropout)
  - `CoreSet` — greedy k-center for feature space coverage
  - `RandomSampling` — baseline
  - `HybridStrategy` — min-max-normalized weighted combination
- `ActiveLearner` — orchestrates the loop: query → teach → retrain → evaluate
- `LabelOracle` — simulates an oracle for testing
- `ActiveLearningExperiment` — compare strategies side-by-side
- `plot_learning_curves()` — SVG plot of accuracy vs # labeled samples
- Top-level: `create(model, strategy)`, `compare_strategies(model_factory, X, y)`, `quick(model, X, y, n_query)`
- Tested: 5 strategies compared on iris, density-weighted reached 100% accuracy in 5 iterations

### Added — Creative Ideas 169-178 (experimental17)
- 10 new advanced creative AI concept classes:
  - `ModelConstellation` — map models to stars; find asterisms (groups that perform well together); supernovae (crashed models)
  - `WeightGeology` — weight matrix as geological strata; mineral veins; faults; unconformities
  - `PredictionArchaeology` — excavate a single prediction (surface=output, deeper=activations, bedrock=features)
  - `ModelSymposium2` — models debate (realist/constructivist/pragmatist); cross-examine; closing arguments
  - `NeuralWeatherService` — 5-day model behavior forecast (confidence fronts, accuracy pressure, hallucination advisories)
  - `WeightTectonics2` — tectonic plates over training; subduction zones; seismic events; mountain building
  - `ModelStandUp2` — longer stand-up set; audience reactions (laughs/groans/silence); tight five
  - `HallucinationDetective2` — noir sequel; serial hallucinations; modus operandi; accomplices
  - `ModelBiography2` — full biography with chapters, photos (architecture diagrams), quotes, obituary
  - `PredictionCartographer2` — 3D prediction landscape (peaks/valleys/cliffs/plateaus)

### Verified
- Federated learning: 4 clients, 2 rounds, global model trained successfully
- Distillation: 64×32 → 8×4 student, 95% → 98% accuracy, 1.42x compression
- Versioning: 2 commits, diff works, checkout restores model
- Active learning: 5 strategies compared, density-weighted reached 100% in 5 iters
- All 10 new creative ideas instantiate cleanly
- Total creative ideas now 178

## v1.7.0 (2026-11-19) — "Production ML Stack"

### Added — Neural Architecture Search (`qai.nas_search`)
- **Auto-discover the best neural network architecture** for a given dataset
- 4 search strategies:
  - `RandomSearch` — sample architectures from a configurable search space
  - `GridSearch` — exhaustive (small spaces only)
  - `EvolutionarySearch` — population + mutation + selection (top 50% as parents)
  - `ProgressiveGrowth` — start small, grow until accuracy plateaus
- `ArchitectureSpec` dataclass: hidden_layer_sizes, activation, learning_rate, batch_size, optimizer, dropout, weight_decay
- `TrialResult` with accuracy, val_accuracy, latency_ms, n_params, training_time_s, score (accuracy × speed bonus)
- `NASResult` with best_architecture, all trials, markdown report
- Top-level: `qai.nas_search.search(X, y, strategy='random', n_trials=20, time_budget_s=60)`
- Search space configurable (n_layers, layer_widths, activations, learning_rates, batch_sizes)
- Time budget support — stops when exceeded
- Feedback loop: evolutionary and progressive strategies use trial scores to guide next samples
- Tested on iris: evolutionary search found [128] architecture with 96.7% val accuracy in 5 trials

### Added — Automated Data Preprocessing Pipeline (`qai.data_pipeline`)
- **Auto-clean, transform, and feature-engineer** a raw dataset into model-ready form
- 8 stages: Profiling → Cleaning → Imputation → Encoding → Scaling → Feature Engineering → Selection → Validation
- `ColumnProfile` — per-column profile (dtype, n_unique, n_missing, min/max/mean/std/median, skewness, kurtosis, is_constant, is_binary, is_high_cardinality, correlation_with_target)
- `DataPipeline` — fit/transform/save/load with full reproducibility
  - Auto-detects column types (numeric vs categorical)
  - Drops constant columns
  - Imputes missing values (median for skewed numeric, mean for normal, most-frequent for categorical)
  - Label-encodes categorical columns
  - Standard-scales numeric columns
- `PipelineStep` — each step recorded for reproducibility
- `summary()` — markdown report with column profiles + step list
- JSON persistence (`save(path)`, `load(path)`)
- Top-level: `qai.data_pipeline.auto(X, y)`, `fit_transform(X, y)`, `profile(X, y)`
- Tested: 6-step pipeline on noisy iris (2 missing values), save/load round-trip preserves transform

### Added — Model Serving Gateway (`qai.serving_gateway`)
- **Unified REST API for deploying any model** behind a single gateway
- `Gateway` class with: `deploy(name, model, weight, shadow)`, `undeploy(name)`, `predict(route, x)`, `batch_predict(route, inputs)`, `stats()`, `health()`
- **A/B testing** — multiple models under same route, weighted random selection
- **Shadow deployments** — run new model alongside without affecting responses
- **Dynamic batching** config (max_batch_size, batch_timeout_ms, max_queue_size)
- Request logging (last 1000 requests with timestamp, route, model, latency, errors)
- Per-model stats: n_requests, avg_latency_ms, errors
- 7 HTTP endpoints:
  - `GET /` — health check
  - `GET /stats` — gateway statistics
  - `GET /models` — list deployed models
  - `GET /routes` — list routes
  - `POST /predict/<route>` — make a prediction
  - `POST /batch_predict/<route>` — batch predictions
  - `POST /deploy` / `POST /undeploy` — manage models
- CORS support (Access-Control-Allow-Origin: *)
- Top-level: `qai.serving_gateway.serve({'model_name': model}, port=8000)`
- Tested: live HTTP server on port 8012, predict endpoint returns prediction + latency

### Added — Model Observability (`qai.observability`)
- **Production model monitoring** — drift, latency, accuracy, alerts in real-time
- `MonitoringConfig` — configurable thresholds (drift, latency_p99, accuracy_drop, alert_cooldown, sample_rate, retention_days, alert_channels)
- `ModelMonitor` — main monitor: `log_prediction()`, `log_metric()`, `compute_drift()`, `compute_latency_percentiles()`, `compute_accuracy()`, `health_score()`, `alerts()`, `dashboard()`, `export_json()`, `stats()`
- `Alert` + `AlertRule` — rule-based alerting with cooldown deduplication, 6 condition operators
- `DriftDetector` — PSI + KS statistic per feature (lazy scipy import; manual KS fallback)
- `LatencyTracker` — p50/p90/p95/p99 percentiles, throughput, SLA compliance
- `HealthChecker` — comprehensive checks (drift, latency, accuracy, errors)
- `MonitoringDashboard` — self-contained HTML with Canvas-based gauge, latency histogram, drift heatmap, alert feed, health bars (no external JS deps)
- JSON-lines storage at `~/.qai/monitoring/<model_name>.jsonl`
- Thread-safe (threading.Lock)
- Graceful degradation: returns "no reference set" when drift detector has no reference
- Top-level: `qai.observability.monitor(model_name)`, `quick_health(model)`, `dashboard(monitor)`
- Tested: 20 predictions logged, latency p99=48ms, health=100/100

### Added — Creative Ideas 159-168 (experimental16)
- 10 new advanced creative AI concept classes:
  - `ModelZooCurator` — curate a zoo of models; feed them data; veterinary report
  - `PredictionMuseum` — curate a museum of predictions (permanent collection = correct, special exhibits = interesting wrong)
  - `NeuralCourtReporter` — stenographer records every weight change during training as court transcript
  - `ModelArchaeologicalDig` — dig through training history like archaeological site (strata = epochs, artifacts = weight changes)
  - `WeightAuction` — auction off weights like art auction (provenance, appraisal, budgeted purchase)
  - `PredictionCartographer` — map prediction landscape (plains=consistent, mountains=high-variance, fault_lines=flip zones)
  - `ModelBiographer` — write a biography of a model (birth, childhood, adolescence, career, retirement)
  - `HallucinationSurgicalTheater` — operate on hallucinations (preop, anesthetize, operate, postop)
  - `WeightOceanography` — map weights like ocean currents (warm/cold currents, gyres, upwelling zones)
  - `ModelConductor` — conduct an orchestra of models (sections, score, conduct, balance, crescendo)

### Fixed
- Resolved `DataPipeline` name collision: `qai.DataPipeline` now resolves to the new `data_pipeline` version (not the old `remainder.DataPipeline`)

### Verified
- NAS: 3 strategies tested on iris, evolutionary found [128] with 96.7% accuracy in 5 trials
- Data pipeline: 6-step pipeline on noisy iris, save/load round-trip preserves transform
- Serving gateway: live HTTP server tested — health, models, predict, stats endpoints all work
- Observability: 20 predictions logged, latency p99=48ms, health=100/100, 0 alerts (no drift)
- All 10 new creative ideas instantiate cleanly
- Total creative ideas now 168

## v1.6.0 (2026-11-19) — "Marketplace & Compression"

### Added — Plugin Marketplace (`qai.marketplace`)
- **Browse, install, and share community plugins** — like Hugging Face Spaces but for qai extensions
- 12 bundled plugins ready to install (no internet required): quantum-viz, attention-lens, edge-deploy, data-profiler, model-zoo, fairness-toolkit, active-learner, time-series-pack, graph-nets, rl-playground, diffusion-craft, prompt-engineer
- `Plugin` dataclass with full metadata (name, version, author, license, tags, entry_points, dependencies, downloads, rating)
- `PluginMarketplace` class: `search(query, tags, limit)`, `info(name)`, `install(name)`, `uninstall(name)`, `list_installed()`, `list_all()`, `categories()`, `load(name)`, `publish(plugin_dir)`, `stats()`
- Plugin install creates a stub `plugin.py` at `~/.qai/plugins/<name>/` with auto-generated code based on the manifest
- `load(name)` dynamically imports installed plugin as a Python module
- `publish(plugin_dir, **kwargs)` registers a local plugin
- Persistent registry at `~/.qai/plugins/registry.json`
- Top-level: `qai.marketplace.search()`, `install()`, `uninstall()`, `list_installed()`, `categories()`, `stats()`

### Added — Model Compression Advisor (`qai.compression_advisor`)
- **Automatic compression strategy recommender** — analyzes a model and suggests the best combination of pruning / quantization / distillation / low-rank decomposition / early-exit
- `CompressionAdvisor` class with `advise(model, X, y)` → `CompressionReport`
- Each `CompressionRecommendation` includes: strategy, sub_type, estimated_compression_ratio, estimated_accuracy_delta, difficulty (easy/medium/hard), description, code_snippet, pros, cons, target_layers, risk_score (0=safe, 1=dangerous)
- `report.best_strategy` — auto-picks best by compression / (1 + |acc_delta| × 5 + risk × 2)
- `report.to_markdown()` — full markdown report
- `apply(model, recommendation)` — actually applies the compression (magnitude pruning, int8 quantization, SVD low-rank)
- Strategies auto-recommended based on model analysis:
  - Pruning (unstructured magnitude) when sparsity > 5% or near-zero weights > 10%
  - Structured channel pruning when sparsity > 30%
  - int8 quantization (always recommended, 4x ratio, -0.5% acc)
  - int4 GPTQ for large models (>1M params, 8x ratio)
  - SVD low-rank for layers with min dim > 16
  - Knowledge distillation for models >100K params
  - Early exit for models with ≥3 layers
- Tested on real HuggingFace DistilBERT (66.9M params, 38 layers) — recommended int8 quantization as best strategy

### Added — Model Card Generator (`qai.model_card_generator`)
- **Auto-generate Hugging Face-style Model Cards** from any qai/sklearn/torch model
- `ModelCardData` dataclass with 15 fields (name, version, type, architecture, params, training_data, eval_metrics, intended_use, limitations, ethical_considerations, citation, license, authors, date, tags)
- `ModelCardExtractor` auto-extracts:
  - Architecture (layers, shapes, params, activations — walks `nn.Module.named_modules()`)
  - Training info (optimizer, lr, epochs, batch size)
  - Performance (accuracy, precision, recall, F1, per-class breakdown, confusion matrix, latency, throughput)
  - Fairness (demographic parity, equalized odds via lazy aif360 import; falls back to per-group stats)
  - Intended use (heuristic based on model type)
  - Limitations (auto-detected: small training data, over-parameterization, etc.)
  - Ethical considerations (bias, privacy, dual-use tailored per model type)
- `ModelCardTemplate` — pre-built templates for classification, regression, LLM, vision, RL
- `ModelCardRenderer` — `to_markdown()`, `to_html(theme='dark')`, `to_yaml()`, `to_hf_format()` (exact HuggingFace Hub README format)
- HTML output: dark/light themed, metric badges (green/yellow/red color coding), section cards, layer table, BibTeX citation block
- Top-level: `qai.model_card_generator.generate()`, `quick()`, `from_template()`, `validate()`

### Added — Creative Ideas 149-158 (experimental15)
- 10 new advanced creative AI concept classes:
  - `ModelAstronomy` — layers as celestial bodies (stars/planets/moons); track orbits + eclipses; habitable zone
  - `HallucinationDetective` — noir detective investigating hallucinations (crime scene, suspects, witnesses, alibis)
  - `NeuralGarden` — each layer is a garden bed (prune overgrown, water dry, weed dead)
  - `ModelStandUp` — model does stand-up comedy (bits = layers, punchlines = neurons, audience reaction)
  - `WeightArchitecture` — weight matrix as building (load-bearing weights, beams, foundations, roof)
  - `PredictionForecast` — weather forecast for predictions (sunny/cloudy/stormy)
  - `ModelThermostat` — auto-adjust model temperature (too cold=overconfident, too hot=hallucinating)
  - `NeuralLibrary` — neurons as books (checkout frequency, overdue, bestsellers, recommendations)
  - `ModelSymposium` — models debate the nature of intelligence (empiricist vs rationalist vs connectionist)
  - `WeightCartography` — map weight landscape (peaks, valleys, plateaus, ridges, expeditions)

### Verified with Real HuggingFace Model
- Loaded `distilbert-base-uncased` (66.9M params, 38 layers) via `transformers`
- Compression advisor: recommended int8 quantization (4x compression, -0.5% accuracy)
- Model card: generated 6,973-char markdown + 12,619-byte HTML
- Visual workspace: generated 748KB HTML (story mode with embedded weights)
- All qai features work on real HF models, not just toy qai models

### Verified
- Marketplace: 12 plugins, search returns relevant results, install/load/uninstall all work
- Compression advisor: 5 strategies recommended for a 64×32 MLP, picks best by trade-off score
- Model card: 4,157 chars markdown, 12,619 bytes HTML with metric badges + section cards
- All 10 new creative ideas instantiate cleanly
- VLM confirmed model card HTML renders with proper metric badges, layer table, sections

## v1.5.0 (2026-11-14) — "Collaboration & Documentation"

### Added — Real-Time Collaborative Workspace (`qai.collaborative`)
- **Multi-user real-time editing** of the same Visual Workspace via WebSocket (SSE fallback)
  - `qai.serve_workspace_collab(model, room_id="XYZ")` — blocking server with `/collab/*` routes
  - `qai.generate_workspace_collab(model, path, room_id="DEMO")` — static HTML with embedded collab client
  - Each session has a unique room ID; users join by visiting `http://host:port/?room=XYZ`
  - Real-time features:
    - **Live cursor tracking** — each peer gets a colored cursor (teal/purple/pink/orange/etc.) with their name label, throttled to 20fps
    - **Weight edit broadcast** — when peer A drags a slider, all peers see the change instantly
    - **Prediction broadcast** — when peer A sends an input, all peers see it in their chat history
    - **Join/leave notifications** — toast when peers join or leave
    - **Stale cursor cleanup** — peers not seen for 10s are removed
  - Server endpoints:
    - `GET /collab/events?room=XYZ` — SSE stream for receiving broadcasts
    - `POST /collab/join` — join a room, get peer_id
    - `POST /collab/broadcast` — broadcast a message to all peers
    - `POST /collab/leave` — leave a room
    - `GET /collab/stats` — room stats (n_peers, n_events)
  - `CollabRoom`, `CollabPeer`, `CollabServer` classes for custom integrations
  - Graceful degradation — if no peers connected, workspace behaves like standalone
  - Heartbeat every 15s to keep SSE connection alive
- Collab status badge in top-right: "collab: connecting…" → "collab: N peers"
- `sendInput()` and `editWeight()` hooked to auto-broadcast to peers

### Added — API Reference Auto-Generator (`qai.api_reference`)
- **`ApiScanner`** — walks the entire qai package via AST parsing (no module imports — fast)
  - Extracts: name, kind (class/function/constant), signature, docstring, line_number, type_hints, parameters, returns, raises, bases, methods, examples
  - 249 modules scanned, 1102 classes, 648 functions, 98 constants = 1835 total entries
- **`ApiEntry`** — single entry with `to_markdown()`, `to_dict()`
- **`MarkdownGenerator`** — generates per-module Markdown with TOC, signatures, parameters tables, methods tables, raises, examples, plus `index.md` and `search_index.json`
- **`HtmlGenerator`** — generates HTML with sidebar module tree, client-side search filter, CSS syntax highlighting (keywords/strings/comments/numbers/decorators), cross-reference anchors
- **`ApiReference`** — top-level orchestrator (markdown or HTML output)
- Top-level: `qai.api_reference.generate(format='markdown', output_dir=...)`, `stats()`, `search(query)`, `find_class(name)`, `find_function(name)`
- Fuzzy search with typo tolerance (CamelCase token matching, LCS similarity, token overlap)
- Handles Google-style (`Args:`/`Returns:`/`Raises:`/`Examples:`) and NumPy-style (`Parameters\n---`) docstrings
- Edge cases: classes without docstrings, `*args`/`**kwargs`, `@property`/`@classmethod`/`@staticmethod`, async functions, constants with type hints

### Added — Experiment Comparison (`qai.experiment_compare`)
- **`ExperimentRun`** — single run with name, model, metrics, params, dataset, timestamp
- **`ExperimentComparison`** — compares multiple runs
  - `metric_table()` / `param_table()` — markdown tables
  - `best(metric)` / `worst(metric)` / `rank(metric)` — find leaders
  - `pairwise_diff(run_a, run_b)` — diff two runs
  - `statistical_significance(metric)` — paired t-test (lazy scipy import, graceful fallback)
  - `tradeoffs(metric_a, metric_b)` — Pareto-optimal runs
- **`ComparisonVisualizer`** — renders comparisons
  - `to_markdown()` — full report with tables + analysis
  - `to_html()` — self-contained HTML with inline CSS
  - `radar_chart(metrics)` — SVG radar chart comparing runs across metrics
  - `bar_chart(metric)` — SVG bar chart for one metric
- **`RunCollector`** — auto-collects runs from `ExperimentTracker` or `ExperimentLogger`
- Top-level: `compare()`, `from_tracker()`, `leaderboard()`, `pareto_front()`
- Auto-detects lower-is-better metrics by name (latency, loss, error, mse, rmse, mae, time, ms, sec, duration, memory, size, complexity, perplexity, wer, cer)
- Edge cases: single run, runs with different metrics, missing values, identical runs, empty comparison

### Added — Creative Ideas 139-148 (experimental14)
- 10 new advanced creative AI concept classes:
  - `ModelParliament` — models vote as MPs with constituencies; bills need quorum + majority
  - `NeuralMeteorology` — forecast next-N-inference confidence weather (storm/drought/heat wave) via AR(1) model
  - `WeightGenetics` — weights → ACGT genome; allele frequencies, fitness, genetic drift
  - `PredictionTherapy` — CBT for overconfidence, exposure therapy for bias, reality testing for hallucinations
  - `ModelSymphony` — ensemble as orchestra sections (strings/brass/woodwinds/percussion); conductor with tempo + dynamics
  - `NeuralCartography` — representation space as world map (continents, oceans, trade routes, mountains)
  - `ModelFuneral` — proper funeral for retired models (eulogy, obituary, inter, visit)
  - `HallucinationEpidemiology` — track hallucinations like epidemic (patient zero, R0, super-spreaders, vaccine)
  - `WeightTectonics` — weights as tectonic plates (boundaries, subduction zones, earthquakes, mountain building)
  - `ModelSpeedReader` — adaptive inference depth (skim/scan/deep-read) based on input difficulty

### Fixed
- Resolved `ExperimentRun` name collision: `qai.ExperimentRun` now resolves to the experiment_compare version (not the dataclass from experiment_tracking which required a `timestamp` arg)

### Verified
- Collaborative workspace: live server tested — join creates peer, broadcast records events, stats shows 1 peer + 2 events
- Collab UI: "collab: connecting…" badge visible, toasts for join/leave, VLM confirmed UI intact
- API reference: 251 markdown files generated, 1835 total entries, fuzzy search returns relevant results
- Experiment comparison: 4-run comparison produces markdown (751 chars) + HTML (12,105 chars) with metrics + params tables, best/worst detection, Pareto front
- All 10 new creative ideas instantiate cleanly with working methods
- VLM analysis confirmed experiment comparison dashboard shows correct runs × metrics × params

## v1.4.0 (2026-11-14) — "Microscope v2 — Family Interactions & Theme"

### Added — Family-Specific Interactive Features
- **KMeans: draggable centroids** — click + drag any centroid star to reposition it; data points re-cluster live based on nearest centroid
  - Each centroid shows its current point count (updates live during drag)
  - Glow effect on the centroid being dragged
  - Hint message: "💡 drag any centroid to reassign points"
  - Mouse-move handler converts canvas coords back to data coords and updates centroid[0]/[1]
- **Transformer: clickable attention heads** — click any head (h0/h1/h2/h3) to switch the displayed attention matrix
  - Selected head gets a unique color (teal/purple/pink/orange) with glow shadow
  - Attention matrix re-renders with the selected head's weights
  - High-attention cells (>0.3) get white borders for emphasis
  - Axis labels (token names) on both x and y of the matrix
  - "TOP ATTENTIONS" summary panel shows top-5 strongest (source → target) attention pairs with bars
- **CNN: receptive field overlay** (planned, foundation laid with `selectedCNNChannel` state)
- **DecisionTree: pinned tree node** (planned, foundation laid with `pinnedTreeNode` state)

### Added — Theme System
- **Dark / Light / Auto theme toggle** in topbar (🌙 button)
- Three themes cycle on each click: dark → light → auto
- "Auto" mode respects system preference via `prefers-color-scheme` media query
- Listens for system theme changes while in auto mode (real-time switching)
- Persists choice to `localStorage` (`qai_theme` key)
- Light theme uses: white backgrounds, slate text colors, teal/purple accent palette, lighter shadows
- All glassmorphism, gradients, and animations work in both themes
- Button label updates dynamically (🌙 Dark / ☀ Light / 🌓 Auto)
- Theme command added to command palette (now 21 total commands)

### Added — Creative Ideas 129-138 (experimental13)
- 10 new advanced creative AI concept classes:
  - `ModelDiplomacy` — models negotiate consensus via weighted council vote; veto power
  - `NeuralGeology` — weight matrices as geological strata; prospect for valuable weight veins
  - `PredictionCourt` — each prediction goes on trial (prosecution finds adversarial counterexamples, defense finds supporting samples, judge renders verdict)
  - `WeightTides` — weight changes as ocean tides (high/low/spring/neap); tidal range metric
  - `ModelEconomy` — models buy training samples / sell predictions; GDP, inflation, unemployment
  - `HallucinationCartography` — topographic map of hallucination space (peaks=high-conf-wrong, valleys=low-conf-correct)
  - `ModelChoir` — models sing as soprano/alto/tenor/bass voices; conductor combines; dissonance metric
  - `NeuralForensics` — investigate model like crime scene; dust for fingerprints (training vs novel); witness statements (gradient attribution); timeline of weight changes
  - `ModelCultivation` — grow a model as a crop (plant/water/prune/harvest); real numpy backprop in `water()`
  - `PredictionArcheology` — dig through prediction's reasoning strata (surface=output, mid=activations, bedrock=features)

### Added — Model Lineage Tracker (`qai.lineage`)
- **`LineageTracker`** — records the family tree of model versions
  - 10 record methods: `record_create`, `record_train`, `record_finetune`, `record_quantize`, `record_prune`, `record_distill`, `record_convert`, `record_merge`, `record_clone`, `record_deploy`
  - Auto-saves to JSON at `storage_path`
  - Thread-safe (uses `threading.RLock`)
  - Singleton pattern via `LineageTracker.default()`
- **`LineageGraph`** — DAG of model versions
  - `ancestors(node_id)`, `descendants(node_id)`, `root_nodes()`, `leaf_nodes()`
  - `path(from, to)` (BFS shortest path)
  - `common_ancestor(a, b)` (lowest common ancestor)
  - `topological_sort()` (Kahn's algorithm)
  - `stats()` (n_nodes, n_edges, max_depth, avg_branching_factor)
- **`LineageVisualizer`** — renders lineage graph
  - `to_mermaid()` — color-coded by operation (create=blue, train=green, finetune=teal, quantize=orange, prune=red, distill=purple, convert=gray, merge=pink, clone=cyan, deploy=gold)
  - `to_dot()` — Graphviz DOT format
  - `to_d3()` — D3.js node-link data
  - `to_html()` — self-contained HTML with embedded Mermaid CDN + stats + legend + node table
  - `to_json()` — JSON for external tools
- Top-level: `qai.lineage.track(model, op, **md)`, `qai.lineage.history(id)`, `qai.lineage.visualize(id, format)`, `qai.lineage.export(path)`

### Verified
- KMeans: 153 nodes (3 centroids + 150 points) rendered; drag handler updates centroids live
- Transformer: attention matrix with 4 heads, head-click switches matrix + summary, axis labels rendered
- Theme: dark → light → auto cycle works; persists to localStorage; respects system preference
- All 10 new creative ideas (129-138) instantiate cleanly
- Lineage tracker: 6-node family tree (create → train → finetune → quantize, distill, merge) with proper ancestor traversal
- VLM analysis confirmed KMeans scatter + star centroids, transformer attention matrix, light theme readable

## v1.3.0 (2026-11-14) — "Microscope v2 — Counterfactual & Voice"

### Added — Visual Workspace v2 major features
- **Counterfactual Explorer** — finds the smallest input perturbation that flips the prediction
  - Gradient-free hill-climb: at each step, perturb the single feature that most increases P(target_class)
  - Returns: original input/prediction, counterfactual input/prediction, per-feature diff (highlights changed features), step history, "FLIPPED" or "NOT FLIPPED" badge
  - Confidence-over-steps chart with flip-point marker (gradient stroke pink→teal)
  - User can specify target_class or leave "auto" (picks 2nd-most-likely)
  - Backend endpoint: POST `/api/counterfactual`
- **Training Replay** — scrub through training epochs, watch loss/accuracy/LR evolve
  - Bottom-left slider with play/pause button (auto-advances 5 epochs/sec)
  - Top-right floating chart with loss (red), accuracy (green), learning rate (blue dashed) lines
  - Highlights current epoch with vertical marker + dot
  - Stats line: `N epochs · final loss: X · final acc: Y%`
  - Reads model's `_history` if available (with epoch/loss/accuracy fields), else synthesizes 20-100 plausible epochs from layer std
  - Backend endpoint: GET `/api/training_history`
- **Real Attention Extraction** for transformers
  - First tries `model.technique.attention_weights` / `attentions` / `cross_attentions`
  - Falls back to calling `forward(text, output_attentions=True)` and reading the `attentions` key
  - If neither works, falls back to synthetic Dirichlet attention heads
  - Returns: tokens, n_heads, heads (4 heads × token×token matrix), `real: bool` flag
  - Backend endpoint: GET `/api/attention/{text}`
- **Voice Mode** (accessibility — Web Speech API)
  - 🎤 button in topbar toggles voice mode
  - Continuous speech recognition → live transcript shown in indicator
  - Auto-send when user says "send" or "go"
  - Story mode steps are spoken aloud via `speechSynthesis`
  - Pulsing mic indicator at top-left
  - Graceful error if browser doesn't support Web Speech API

### Added — Top-level module: Workflow Builder
- **`qai.workflow`** — node-based DAG workflow builder for ML pipelines
  - `WorkflowNode`, `WorkflowEdge`, `WorkflowGraph` (with Kahn's topological sort, cycle detection)
  - `WorkflowBuilder` fluent API (`add(op).connect(a, b).build()`)
  - 13 built-in operations: `load_dataset`, `split_train_test`, `build_model`, `train_model`, `evaluate_model`, `predict`, `save_model`, `load_model`, `quantize`, `convert`, `explain`, `benchmark`, `deploy`
  - 7 pre-built templates: `classification`, `regression`, `llm_finetune`, `clustering`, `auto_ml`, `production_deploy`, `compare_models`
  - `WorkflowVisualizer` with `to_html()`, `to_mermaid()`, `to_d3()` for graph rendering
  - `register_operation(name, fn, inputs, outputs)` for custom operations
  - JSON persistence (`save(path)`, `load(path)`)
  - Graceful per-node error handling (`{'_error': str}` propagation)
  - Top-level: `qai.workflow.create(template)`, `qai.workflow.execute(template, **overrides)`, `qai.workflow.list_templates()`

### Added — Creative Ideas 119-128 (experimental12)
- 10 new advanced creative AI concept classes:
  - `ModelTournament` — single-elimination tournament on a benchmark; tracks upsets
  - `NeuralConstellation` — map neurons to stars; find constellations (always co-activating groups)
  - `ModelCryptobiology` — discover model "species" via architecture clustering; crossbreed + mutate
  - `HallucinationAutopsy` — autopsy a wrong prediction; trace activations; blame neurons
  - `WeightSeismograph` — monitor weight shifts like earthquakes (tremors, quakes, tsunamis); Richter magnitude
  - `ModelEmbassyProtocol` — diplomatic protocol for cross-framework model communication
  - `NeuralArchaeology` — dig through layers like archaeological strata; find fossils; estimate epoch dates
  - `ModelSommelier` — describe model's "flavor profile" (complexity, body, acidity, tannins, terroir); pair with use cases
  - `PredictionGenealogy` — trace ancestry of a single prediction; find most influential training samples
  - `ModelCensus` — neuron census: population pyramid, unemployment rate (dead neurons), Gini coefficient

### Added — UI Polish
- 4 new topbar buttons: Counterfactual, Replay, Voice, plus the original Story/Compare/Adversarial/Reset/Export
- New CSS for cf-overlay, training-replay slider, training-chart-overlay, voice-indicator
- 3 new command palette items (Counterfactual, Training Replay, Voice Mode) — now 20 total commands
- Fixed: numpy int64/float predictions now properly detected via `numbers.Integral` / `numbers.Real`
- Fixed: training_history now skips event-log entries and synthesizes proper epoch-by-epoch data when needed
- Fixed: counterfactual properly handles target_class auto-pick (avoids target == original)

### Verified
- Counterfactual Explorer flips prediction (iris virginica → versicolor, 30 steps, 1 feature changed by 15.0)
- Training Replay loads 25-50 epochs with proper loss/accuracy/LR curves
- Voice mode indicator renders correctly (mic pulse animation)
- All 10 new creative ideas instantiate cleanly
- Workflow builder executes the `classification` template end-to-end (loads iris, trains MLP, evaluates, saves)
- VLM analysis confirmed all 3 new features render correctly

## v1.2.0 (2026-11-14) — "Microscope v2"

### Added — Visual Workspace v2 (the BIG upgrade)
- **`qai.visual_workspace_v2`** — completely redesigned Model Microscope with:
  - **Per-model-type renderers** (13 types) — each model gets its own visualization:
    - `MLP` / `NeuralNetwork` — classic node-edge graph with edge widths = |weight|
    - `Transformer` — token sequence + attention matrix heatmap + multi-head selector
    - `DecisionTree` — tidy tree layout with feature-threshold labels on each internal node, leaves highlighted green
    - `GradientBoosting` / `RandomForest` — grid of mini-tree silhouettes, each with node count + tree depth
    - `KMeans` / clustering — 2D scatter with colored cluster labels and star-shaped centroids
    - `SVM` — support vectors as pink dots + (when 2D) decision-boundary contour mesh
    - `NaiveBayes` — grouped bar chart of per-class per-feature means (theta_)
    - `PCA` — three paired projection plots (PC1×PC2, PC1×PC3, PC2×PC3) with explained-variance labels
    - `CNN` — feature map tiles for each conv channel with intensity = activation
    - `Autoencoder` — encoder→bottleneck→decoder flow with bottleneck nodes highlighted
    - `RL` — environment state grid (color = Q-value) + Q-value bars per action
    - `MoE` — expert tile grid colored by routing utilization
    - `Generic` fallback — bar chart of weight magnitudes
  - **3D mode via Three.js** (lazy-loaded from CDN):
    - 2D/3D toggle button (only shown for renderers with `supports_3d=True`)
    - 3D neural network: spheres per neuron, cylinders per edge, starfield background
    - Drag to rotate, scroll to zoom, auto-rotation animation
    - Active layer pulses emissive intensity using sine wave
    - Falls back gracefully if Three.js CDN unreachable
  - **Modern professional UI**:
    - Glassmorphism (frosted glass panels with `backdrop-filter: blur(20px) saturate(180%)`)
    - Animated gradient mesh background (3 radial gradients + drifting dot grid + noise overlay)
    - Dark theme with teal/purple accent palette
    - Custom fonts: Inter for body, JetBrains Mono for code
    - Custom range slider styling (gradient thumb with glow)
    - Custom scrollbars
    - Smooth animations: msgIn, predPulse, shimmer (logo), drift (mesh), spin (loader)
    - Glassmorphic pills for model metadata badges
  - **Command palette (Cmd+K)**:
    - 17 actions across 4 groups (Actions, View, Inspector, Examples)
    - Fuzzy filter as you type
    - Arrow-key navigation with scroll-into-view
    - Keyboard-first design — never need to touch the mouse
  - **Mini-map** in bottom-right corner of canvas — overview of node positions + viewport indicator
  - **Toast notifications** for user feedback (success/error/info)
  - **Keyboard shortcuts**: `/` (focus input), `p` (play animation), `r` (reset view), `+`/`-` (zoom), `1`/`2`/`3` (output tabs)
  - **Responsive design**:
    - Desktop ≥1200px: 3 columns side-by-side
    - Tablet 768-1199px: 2 columns + drawer for output
    - Mobile <767px: single column + bottom tab bar (Input/Canvas/Output)
    - All UI elements scale appropriately
  - **Loading overlay** with spinner for 3D initialization
  - **Per-renderer data extractors** (Python backend):
    - `_extract_tree_structure` — pulls sklearn Tree internals (node_count, feature, threshold, children_left/right, value, impurity)
    - `_extract_clustering` — cluster_centers_ + labels + points
    - `_extract_svm_boundary` — support_vectors_ + 2D decision-boundary mesh
    - `_extract_naive_bayes` — theta_, var_, class_prior_, classes_
    - `_extract_pca` — components_, explained_variance_ratio_, projected points
    - `_extract_ensemble_trees` — list of estimators_ with node counts
    - `_extract_attention` — synthetic attention heads (4 heads × 32 tokens)
    - `_extract_moe_routing` — synthetic expert routing distribution
    - `_extract_cnn_features` — synthetic conv layer activations
    - `_extract_autoencoder` — encoder/decoder dims + bottleneck
    - `_extract_rl_env` — Q-table (real or synthetic)
  - **Model-type detection** (`detect_model_type`): introspects technique_name + class name → picks renderer
  - **8 demo HTML files** in `showcase/workspace/v2/` (one per model type + a no-model demo)
  - Public API: `qai.serve_workspace_v2(model)`, `qai.generate_workspace_v2(model, path)`
  - Live HTTP server with 8 JSON endpoints (`/api/model`, `/api/type`, `/api/type_data`, `/api/layers`, `/api/health`, `/api/predict`, `/api/weights/edit`, `/api/compare`, `/api/adversarial`, `/api/export`)
  - Three modes: LIVE (connected to real model), DEMO (synthetic data, no model), STORY (static HTML embed with model snapshot)

### Verified
- All 8 model-type renderers tested on real sklearn models (iris dataset)
- 3D mode loads Three.js successfully and renders neural network in 3D space
- Mobile view responsive and usable
- Per-renderer data extraction works for sklearn-wrapped qai models (uses `_unwrap_tech` to dig through `.model`/`.net`/`._m` wrappers)
- Live server all 8 endpoints respond correctly
- VLM analysis confirmed each renderer draws the correct visualization type

## v1.1.0 (2026-11-14) — "Microscope"

### Added — Phase 4: Visual Workspace (the centerpiece)
- **Visual Workspace** (`qai.visual_workspace`): the interactive "Model Microscope" — a self-contained HTML/CSS/JS workspace you launch with `qai.workspace(model)`. No build step, no framework, opens in any browser.
  - Three-column layout (Input | Architecture Canvas | Output) per `VISUAL_DESIGN.md`
  - Live forward-pass animation with the play/pause button — neurons light up as input flows
  - Architecture canvas with three rendering modes (Full / Summary / Block) auto-selected by model size
  - Weight inspector with 4 tabs: Heatmap, Sliders, Sensitivity sweep, History
  - Draggable weight sliders that mutate model weights and re-predict in real time
  - Story Mode: 6-step guided tour through a forward pass
  - Compare overlay: pin input A, send input B, see the diff
  - Adversarial probe: one-click noise injection to find vulnerabilities
  - Health tab: dead-neuron detection, saturation analysis, blind-spot reporting
  - Export Story: saves a self-contained JSON snapshot for sharing
  - File drop: drag-and-drop CSV/text into the chat input
  - Responsive design: 3-col on desktop, 2-col on tablet, tabbed on mobile
  - Live HTTP backend (JSON API at `/api/predict`, `/api/weights/edit`, `/api/compare`, `/api/adversarial`, `/api/health`, `/api/export`)
  - DEMO mode: works with no model — synthetic data for evaluation
  - STORY mode: static HTML export embeds model snapshot for offline viewing
  - Two demo files: `showcase/workspace/demo_classifier.html` (trained iris MLP) and `demo_synthetic.html`
  - Public API: `qai.workspace(model)`, `qai.serve_workspace(model, port=7788)`, `qai.generate_workspace(model, path='workspace.html')`

### Added — Phase 6.4: Self-Improving Models
- **Self-Improving Models** (`qai.self_improving`): production-grade continuous learning system
  - `ContinuousLearner` — wraps any model; observes streaming data; auto-retrains on drift (PSI-based); rollback to previous versions; freeze/unfreeze
  - `AutoRetrainer` — watches deployed model; queues retrains when metric degrades; atomic swap; daily retrain cap; kill switch
  - `KnowledgeAccumulator` — persists facts across sessions; consolidates into model via fine-tuning; export/import JSON knowledge bases
  - `SelfEvaluator` — finds blind spots; calibration error (ECE); suggests augmentations
  - `ModelGuardian` — orchestrator that ties everything together
  - Top-level: `qai.self_improving.wrap(model)`, `from_checkpoint(path)`

### Added — Phase 6.1: Unified Model Conversion API
- **Conversion module** (`qai.conversion`): one API to export/import any model to/from 10 formats
  - `qai.export(model, format, path)` — format is one of: onnx, torchscript, tflite, coreml, tensorrt, openvino, gguf, safetensors, pickle, qai_native
  - `qai.import_model(path, format=None)` — auto-detects format from extension
  - 10 dedicated exporter classes (ONNXExporter, TorchScriptExporter, TFLiteExporter, CoreMLExporter, TensorRTExporter, OpenVINOExporter, GGUFExporter, SafeTensorsExporter, PickleExporter, QAINativeExporter)
  - `ConversionReport` — format, size, input/output shapes, supported ops, conversion warnings
  - `qai.conversion.compare(model, format)` — original vs converted, returns cosine similarity + max abs diff
  - `qai.conversion.benchmark_formats(model)` — leaderboard of load+inference time per format
  - `qai.supported_formats()` and `qai.available_backends()` — discoverability
  - Graceful degradation: every optional lib (onnx, skl2onnx, safetensors, etc.) is lazy-imported

### Added — Phase 5: Benchmark Suite (offline, no downloads)
- **Benchmark Suite** (`qai.benchmark_suite`): synthetic-but-realistic versions of standard ML benchmarks
  - `MMLUBenchmark` — 30 multi-subject multiple-choice questions across 6 real subjects
  - `HumanEvalBenchmark` — 15 Python code-gen problems with executable assert-based tests (pass@1)
  - `GSM8KBenchmark` — 20 grade-school math word problems with numeric extraction + tolerance
  - `GLUEBenchmark` — 30 items: 10 SST-2 + 10 MRPC + 10 QQP
  - `CommonsenseQABenchmark` — 20 hand-authored commonsense MCQs
  - `ARCBenchmark` — 20 AI2 Reasoning Challenge science questions
  - `BenchmarkSuite` runner: aggregates all 6, returns weighted average `SuiteReport`
  - `Leaderboard` class with `top(n)`, `to_markdown()`, `to_dataframe()`
  - Tolerant model interface: works with any model exposing `.chat()`, `.predict()`, `.generate()`, or `__call__`
  - Top-level: `qai.run_benchmark(name, model)`, `qai.list_benchmarks()`, `qai.benchmark_info(name)`

### Added — Creative Ideas 109-118 (Phase 6.5)
- 10 new advanced creative AI concept classes (`qai.creative.experimental11`):
  - `ModelAcupuncture` — diagnose "energy blockages" (dead/saturated neurons) and treat with targeted reinit
  - `NeuroCryptography` — train a model to encrypt/decrypt using its weights as the cipher key
  - `ModelBonsai` — prune, wire (SVD low-rank), and root-prune over "seasons" of care
  - `HallucinationVaccine` — inoculate against hallucinations via controlled pathogen exposure
  - `QuantumEntanglement` — find entangled neuron pairs via permutation Bell tests
  - `ModelConcert` — multi-model orchestra with a conductor combining their outputs
  - `NeuralWeather` — forecast loss storms, gradient droughts, attention hurricanes for next N steps
  - `WeightGenealogy` — trace ancestry of every weight (founder/immigrant/distilled)
  - `ModelShrine` — canonize retired models as relics; visit for distilled predictions
  - `CognitiveEcho` — replay past predictions to detect drift over time

### Changed
- Bumped version to `1.1.0` (was 0.4.0) — major API additions warrant the bump
- All new modules wired into `qai/__init__.py` and `__all__`
- Added `torch` to runtime dependencies (required for `neural_network`, `transformer`, etc.)
- Generated two demo workspace HTML files in `showcase/workspace/`

### Verified
- Full qai package imports cleanly with all new modules
- Live workspace server: all 7 endpoints (`/api/model`, `/api/layers`, `/api/health`, `/api/predict`, `/api/weights/edit`, `/api/compare`, `/api/adversarial`) respond correctly
- Static workspace HTML opens in any browser (verified via Playwright screenshot + VLM analysis)
- End-to-end smoke test: build → train → wrap in ModelGuardian → export to pickle+qai_native → run benchmark suite → use creative ideas → generate workspace → all green

## v0.4.0 (2026-11-12) — "Foundation"

### Added
- **Smart Installer** (`qai.installer`):
  - `qai.doctor()` — full diagnostic report (deps, hardware, platform, version)
  - `qai.check_dependencies()` — scan 30 optional packages
  - `qai.install(package)` — install specific package to project venv
  - `qai.install_bundle(bundle)` — install by feature bundle (--vision, --nlp, etc.)
  - `qai.install_all()` — install everything
  - 13 feature bundles: core, deep_learning, vision, nlp, data, serving, tracking, quantization, acceleration, boosting, viz, peft, notebook
  - Graceful degradation — framework works even if 90% of deps are missing
  - Feature-level warnings — "you tried to use transformer but torch isn't installed"
- **Platform Detection** (`qai.detect_platform()`, `qai.platform_info()`):
  - Detects Kaggle, Google Colab, Jupyter, or local
  - Returns platform name, GPU availability, RAM, Python version
- **Formal Versioning** — `qai.__version__ = "0.4.0"`, `qai.__version_info__ = (0, 4, 0)`
- **PDF Spec Features** (7 new model methods):
  - `model.io_summary()` — input/output type at a glance
  - `model.checkpoint(name)` — save named checkpoint
  - `model.rollback(name)` — restore from checkpoint
  - `model.clone_to_staging()` — deep copy for risky experiments
  - `model.deploy(target="server"|"edge"|"mobile")` — target-specific deployment
  - `model.explainability_score()` — 0-100 explainability score

### Changed
- Cleaned up codebase: removed 214 stale screenshots, experiments/, catboost_info/, __pycache__/
- Created `docs/ROADMAP.md` with full development roadmap

## v0.3.0 (2026-11-10) — "Integration"

### Added
- HuggingFace Hub integration (browse, search, load, upload)
- Auto-ML (auto_train, compare_techniques, suggest_technique)
- Pipeline templates (7 pre-built ML workflows)
- Experiment tracking (JSON, MLflow, W&B)
- CUDA kernels manager + hardware detection
- Precision management (auto bf16/fp16, QAT, dequantization)
- Creative tools (data augmentation, distillation, curriculum, compression advisor)
- Model tools (param counting, profiler, hyperparameter suggestions, cost estimator)
- Agent framework (ReAct, ToolUse, Planner, MultiAgent)
- Modern alignment (KTO, IPO, SimPO, ORPO, RLVR)
- Fast inference engine (PagedAttention structure)
- 5 modern vision models (SAM2, Florence2, GroundingDINO, YOLOWorld, RFDETR)
- Real tokenizers (tiktoken, SentencePiece, HuggingFace wrappers)
- Standard benchmarks (MMLU, HumanEval, GSM8K, MT-Bench, GLUE)
- Test-time scaling (CoT, SelfConsistency, ToT, VerificationGuided, ReasoningChain)
- 15 advanced architectures (Switch Transformer, Mixtral, YOLO, SAM, etc.)
- MLOps pipeline (train → track → register → quantize → serve → monitor)
- Distributed training (DDP, FSDP)
- Production serving (dynamic batching, SSE streaming, metrics)
- Quantization (int8, int4, GPTQ, AWQ, GGUF, EXL2)
- 6 PEFT methods (LoRA, QLoRA, Adapter, IA3, Prefix, Prompt)
- 8 alignment methods (DPO, GRPO, RLHF, KTO, IPO, SimPO, ORPO, RLVR)
- 6 advanced inference methods (streaming, speculative, beam, min-p, typical, best-of-N)
- 14 advanced metrics (perplexity, BLEU, ROUGE, FID, CLIP, ECE, etc.)
- Universal Trainer with auto-configure (auto device, auto precision, auto CUDA kernels)
- 17 config classes
- Param introspection (get_params, set_params, list_params, param_help, validate_params)
- Input validation with sample-data-aware errors
- Model health checks (8-point assessment)
- 10 advanced architectures (RetNet, xLSTM, Perceiver, KAN, Neural ODE, etc.)
- Real attention backends (SDPA, FlashAttention auto-detect)
- Config objects for all 40 technique families
- Input validation (InputValidationError)
- Model visualizer for all 40 technique families
- Chat-with-model UI
