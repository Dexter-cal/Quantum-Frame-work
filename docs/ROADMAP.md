# QAI Framework — Development Roadmap

*Created: 2026-11-12*
*Last Updated: 2026-11-26*
*Current Version: v1.8.0 — "Distributed ML & Versioning"*

> See also: [README](../README.md) · [Tutorial](TUTORIAL.md) · [Implementation Status](UNBUILT_FEATURES.md) · [Changelog](../CHANGELOG.md)

---

## Current State (as of Session 24)

### What works (verified):
- 248 techniques (218 base + 30 advanced) — all build/train/predict/explain/save/visualize
- Universal Trainer (grad clip, LR schedule, mixed precision, EMA, SWA, gradient checkpointing, auto-device, auto-precision, auto-CUDA kernels)
- 6 PEFT methods (LoRA, QLoRA, Adapter, IA3, Prefix, Prompt)
- 8 alignment methods (DPO, GRPO, RLHF, KTO, IPO, SimPO, ORPO, RLVR)
- 6 advanced inference methods (streaming, speculative, beam, min-p, typical, best-of-N)
- 6 test-time scaling methods (CoT, SelfConsistency, ToT, VerificationGuided, ReasoningChain)
- Agent framework (ReAct, ToolUse, Planner, MultiAgent)
- HuggingFace Hub integration (browse, search, load, upload)
- Auto-ML (auto_train, compare_techniques, suggest_technique)
- 7 pipeline templates (classification, regression, clustering, NLP, LLM, time_series, auto)
- Experiment tracking (JSON, MLflow, W&B)
- Production serving (dynamic batching, SSE streaming, metrics)
- Quantization (int8, int4, GPTQ, AWQ, GGUF, EXL2, QAT, dequantization)
- Precision management (auto-detect bf16/fp16, set/get precision, compare)
- CUDA kernels manager (FlashAttention, PagedAttention, fused ops)
- Model tools (param counting, summary, profiler, hyperparameter suggestions, cost estimator)
- Creative tools (data augmentation, feature importance, compression advisor, distillation, curriculum)
- 14 advanced metrics (perplexity, BLEU, ROUGE, FID, CLIP, ECE, Brier, NDCG, etc.)
- 5 modern vision models (SAM2, Florence2, GroundingDINO, YOLOWorld, RFDETR)
- MLOps pipeline (train → track → register → quantize → serve → monitor)
- 17 config classes
- Input validation, model health checks, token counter
- Real dataset loaders (iris, digits, wine, MNIST, CIFAR, GLUE, shakespeare)
- Distributed training (DDP, FSDP)
- Param introspection (get_params, set_params, list_params, param_help, validate_params)
- **NEW** — Visual Workspace (Phase 4): `qai.workspace(model)` launches the model microscope
- **NEW** — Self-Improving Models (Phase 6.4): ContinuousLearner, AutoRetrainer, KnowledgeAccumulator, SelfEvaluator, ModelGuardian
- **NEW** — Unified Conversion API (Phase 6.1): `qai.export(model, format, path)` for 10 formats (onnx, torchscript, tflite, coreml, tensorrt, openvino, gguf, safetensors, pickle, qai_native)
- **NEW** — Benchmark Suite (Phase 5): 6 offline benchmarks (MMLU 30q, HumanEval 15p, GSM8K 20p, GLUE 30i, CommonsenseQA 20q, ARC 20q) with tolerant model interface
- **NEW** — 10 more creative ideas (109-118): ModelAcupuncture, NeuroCryptography, ModelBonsai, HallucinationVaccine, QuantumEntanglement, ModelConcert, NeuralWeather, WeightGenealogy, ModelShrine, CognitiveEcho

### Accuracy notes (formerly "What's still toy scale" — updated 2026-11-26)

The old "toy scale" list below was stale. Current, accurate picture:

- ~~All neural architecture implementations (correct code, tiny defaults)~~ —
  implementations are real PyTorch with backprop, attention, and training
  loops. Defaults are chosen small so the test suite runs in CI on CPU; pass
  bigger kwargs (`d_model=256`, `n_layers=6`, ...) for real workloads.
- ~~No real pretrained weights~~ — TRUE and by design: qai is a framework,
  weights come from HuggingFace via `qai.from_huggingface("bert-base-uncased")`
  (browse/search/load/upload all built — see [guides](guides/MODEL_CATALOG.md)).
- ~~FastInferenceEngine has paged KV cache structure but no CUDA kernels~~ —
  PagedAttention is implemented (fixed-size pages, no fragmentation) and CUDA
  is used automatically when `torch.cuda.is_available()`; see
  `qai/fast_inference.py`. The CUDA kernels *manager* (FlashAttention /
  PagedAttention / fused-ops dispatch) is also built.
- Benchmarks use synthetic-but-realistic test items (MMLU has 30, not 14,042) —
  TRUE and documented in [guides/EVALUATION.md](guides/EVALUATION.md):
  embedded subsets for relative comparison between your models, not
  leaderboard claims. Full-dataset download remains a future item.
- ~~No real HuggingFace model loading tested end-to-end~~ — built and exposed
  (`qai.from_huggingface`, `qai.to_huggingface`, `qai.load_from_zoo`,
  `qai.stream_dataset`); see [guides/DATA.md](guides/DATA.md) streaming section.

Net: every spec feature is built (108/108 — see
[UNBUILT_FEATURES.md](UNBUILT_FEATURES.md)). The honest remaining limits are:
small benchmark subsets, defaults tuned for CI speed, and pretrained weights
living in HuggingFace rather than the repo.

### Visual Workspace status:
- Three-column layout: ✅ built
- Architecture canvas with three render modes: ✅ built
- Weight inspector with heatmap + sliders: ✅ built
- Forward-pass animation: ✅ built
- Story mode: ✅ built (6-step tour)
- Compare two inputs: ✅ built
- Adversarial probe: ✅ built
- Health tab (dead neurons, saturation): ✅ built
- Export story (JSON): ✅ built
- Live HTTP backend: ✅ built (7 endpoints)
- DEMO + STORY modes: ✅ built
- Responsive design: ✅ built (3-col/2-col/tabbed)
- Family-specific visualizations (MoE router, attention matrix, decision tree, KMeans clusters): ❌ next
- Voice mode (accessibility): ❌ planned
- Counterfactual explorer (gradient × input): ❌ planned
- Training replay slider: ❌ planned

---

## Roadmap (Priority Order)

### Phase 1: Foundation & Infrastructure ✅ DONE (v0.4.0)

#### 1.1 Smart Installer & Launcher ✅
- [x] `qai.__version__` with semantic versioning
- [x] `qai install` command — detect missing dependencies, install only what's needed
- [x] Feature bundles: `qai install --vision`, `--nlp`, `--serving`, `--all`
- [x] Install to project venv, not global
- [x] Graceful degradation — framework works even if 90% of deps missing
- [x] Feature-level warnings — "you tried to use transformer but torch isn't installed"
- [x] `qai doctor` — check what's installed, what's missing, what's recommended
- [x] `qai.launcher` — start the dashboard/chat/visual UI

#### 1.2 Formal Versioning & Changelog ✅
- [x] Add `qai.__version__` string
- [x] Create `CHANGELOG.md`
- [x] Tag releases with git tags

#### 1.3 Cleanup & Codebase Hygiene ✅
- [x] Delete 214 static visualization screenshots (will rebuild as interactive)
- [x] Remove stale `experiments/` directory
- [x] Remove `catboost_info/` directory
- [x] Clean up `__pycache__` directories
- [x] Add `requirements.txt` with optional dependencies clearly marked
- [x] Add `setup.py` / `pyproject.toml` for pip installation

#### 1.4 Re-read PDF Spec — FINDINGS ✅
All 7 PDF-spec features added (io_summary, weights.translate, deploy target, clone_to_staging, rollback, explainability_score, chainable output).

---

### Phase 2: Platform Integration & Streaming ✅ DONE (v0.4.0)

#### 2.1 HuggingFace Streaming Training ✅
- [x] `qai.stream_hf_dataset(name)` — stream dataset without full download
- [x] Connect streaming to Trainer's batch loop
- [x] `qai.search_hf_datasets(query)` — search for datasets on HF Hub
- [x] `qai.browse_hf_datasets()` — browse popular datasets

#### 2.2 Notebook / Platform Detection ✅
- [x] Detect Kaggle / Colab / Jupyter / local
- [x] Auto-configure for platform
- [x] `qai.detect_platform()`, `qai.platform_info()`

---

### Phase 3: Model Improvements (Fix Missing Features) ✅ DONE (v0.4.0)

#### 3.1 Transformer Improvements ✅
- [x] Proper weight initialization (`trunc_normal_(std=0.02)`)
- [x] Separate `attention_dropout` vs `residual_dropout` vs `embedding_dropout`
- [x] Configurable `layer_norm_eps`
- [x] Clean `tie_word_embeddings` toggle
- [x] RMSNorm as option (not just LayerNorm)
- [x] Configurable `rope_theta`
- [x] Block-level gradient checkpointing (`gradient_checkpointing_enable()`)
- [x] KV cache quantization (int8 cache)
- [x] Sliding window + global attention hybrid (Gemma 3 style)
- [x] Multi-token prediction architecture support

#### 3.2 CNN Improvements ✅
- [x] Configurable norm: `batch_norm` / `layer_norm` / `group_norm`
- [x] Configurable activation: `relu` / `gelu` / `silu` / `mish`
- [x] Grouped convolution support (`groups` parameter)
- [x] Dilated convolution support (`dilation` parameter)

#### 3.3 RL Agent Improvements ✅
- [x] Target networks for DQN
- [x] Experience replay buffer
- [x] Priority experience replay (PER)
- [x] Epsilon decay schedules (exponential, cosine)
- [x] Multi-step returns (n-step TD)
- [x] Distributional Q-learning (C51, QR-DQN)

#### 3.4 Generative Model Improvements ✅
- [x] Classifier-free guidance for conditional generation
- [x] FID/IS evaluation during training
- [x] Two-timescale update for GANs
- [x] EMA of generator weights
- [x] Latent space interpolation

#### 3.5 Graph Neural Network Improvements ✅
- [x] Edge feature support
- [x] Graph batching (multiple graphs per batch)
- [x] Readout functions (sum/mean/max pooling)

---

### Phase 4: Visual UI System ✅ CORE BUILT (v1.1.0)

#### 4.1 Document the Visual Design ✅
- [x] Create `docs/VISUAL_DESIGN.md` capturing all design decisions

#### 4.2 Build the Visual UI ✅ CORE BUILT
- [x] Choose tech stack (pure HTML/CSS/JS — no build step)
- [x] Build the three-column layout
- [x] Build the architecture canvas (Canvas API)
- [x] Build the weight inspector (heatmap + sliders)
- [x] Build the chat/textbook panel
- [x] Build the forward-pass animation
- [x] Build the status bar (tokens, latency, confidence)
- [x] Build the model health tab
- [x] Implement "focus on active nodes" rendering
- [x] Implement responsive breakpoints
- [x] Integrate with qai.predict / qai.chat_with_model()
- [x] Story mode (6-step guided tour)
- [x] Compare two inputs overlay
- [x] Adversarial probe button
- [x] Export story (JSON snapshot)
- [x] File drop (CSV/text upload)
- [x] DEMO mode (no model needed)
- [x] STORY mode (static HTML embed)
- [x] Live HTTP backend with 7 JSON endpoints

#### 4.3 Next: Family-Specific Visualizations (NEXT)
- [ ] MoE router heatmap (which experts get which tokens)
- [ ] Attention matrix as heatmap (query × key, click head → see what it attends to)
- [ ] Decision tree visualization with split thresholds
- [ ] KMeans scatter with draggable centroids
- [ ] CNN feature map tiles with receptive field overlay
- [ ] RL agent Q-value bars

#### 4.4 Future: Advanced Visual Features
- [ ] Counterfactual explorer (gradient × input → minimal perturbation to flip prediction)
- [ ] Training replay slider (scrub epochs → watch confidence evolve)
- [ ] Voice mode (accessibility — read story steps aloud)
- [ ] Collaborative mode (two cursors on same workspace via WebRTC)
- [ ] 3D architecture view (for very deep models — WebGL)

---

### Phase 5: Real Data & Benchmarks ✅ CORE BUILT (v1.1.0)

#### 5.1 Synthetic-Realistic Benchmarks ✅
- [x] `MMLUBenchmark` — 30 questions across 6 real subjects
- [x] `HumanEvalBenchmark` — 15 Python problems with executable tests
- [x] `GSM8KBenchmark` — 20 math word problems
- [x] `GLUEBenchmark` — 30 items (SST-2 + MRPC + QQP)
- [x] `CommonsenseQABenchmark` — 20 commonsense MCQs
- [x] `ARCBenchmark` — 20 ARC-Easy science questions
- [x] `BenchmarkSuite` runner with weighted average
- [x] `Leaderboard` class with markdown/dataframe export
- [x] Tolerant model interface (`.chat()` / `.predict()` / `.generate()` / `__call__`)

#### 5.2 NEXT: Real Benchmark Datasets (when online)
- [ ] Download real MMLU (14,042 questions, 57 subjects)
- [ ] Download real HumanEval (164 problems)
- [ ] Download real GSM8K (8,500 problems)
- [ ] Download real GLUE benchmark
- [ ] Store in `~/.qai/benchmarks/` with caching

#### 5.3 NEXT: Real Benchmark Runners
- [ ] `qai.benchmarks.run_mmlu(model, full=True)` — full 14K evaluation
- [ ] `qai.benchmarks.run_humaneval(model, full=True)` — full 164 evaluation
- [ ] `qai.benchmarks.run_gsm8k(model, full=True)` — full 8.5K evaluation
- [ ] `qai.benchmarks.leaderboard()` — compare to published results

---

### Phase 6: Advanced Features

#### 6.1 Model Conversion (Bidirectional) ✅ BUILT (v1.1.0)
- [x] `qai.from_huggingface()` — load HF model → qai
- [x] `qai.to_huggingface()` — push qai model → HF Hub
- [x] `qai.export(model, format)` — unified export to 10 formats
- [x] `qai.import_model(path, format=None)` — auto-detect import
- [x] 10 dedicated exporter classes
- [x] `qai.conversion.compare(model, format)` — original vs converted similarity
- [x] `qai.conversion.benchmark_formats(model)` — leaderboard

#### 6.2 Plugin System ✅ DONE (v0.4.0)
- [x] `@qai.plugin` decorator for custom techniques
- [x] `@qai.tool` decorator for custom tools
- [x] Plugin discovery (auto-load from `~/.qai/plugins/`)
- [ ] Plugin marketplace (browse and install community plugins)

#### 6.3 Model Lineage Tracking
- [x] `WeightGenealogy` (creative.experimental11) — traces ancestry of every weight
- [ ] Track which model came from which data
- [ ] Track which model was fine-tuned from which base
- [ ] Track which quantization was applied to which model
- [ ] Visualization of model lineage graph

#### 6.4 Self-Improving Models ✅ BUILT (v1.1.0)
- [x] `ContinuousLearner` — auto-detect drift, auto-retrain on new data
- [x] `AutoRetrainer` — auto-switch to backup during retraining
- [x] A/B test new vs old model automatically
- [x] `KnowledgeAccumulator` — permanent knowledge across sessions
- [x] `SelfEvaluator` — find blind spots, calibration errors
- [x] `ModelGuardian` — top-level orchestrator

#### Phase 6.6 — Blind-Spot Review & New Ideas (added 2026-11-26)

**Blind spots found while auditing the docs** (things that were built but
never surfaced anywhere users could find them — now documented in
[guides/](guides/README.md)):

1. `qai.federation` (v1.8.0) had no user-facing docs at all →
   [guides/DISTRIBUTED.md](guides/DISTRIBUTED.md)
2. `qai.distillation_v2` (`distill` / `multi_distill` / `self_distill`,
   MultiTeacherDistiller, SelfDistiller) was changelog-only →
   [guides/COMPRESS.md](guides/COMPRESS.md)
3. `qai.nas_search` (RandomSearch / GridSearch / EvolutionarySearch /
   ProgressiveGrowth) was changelog-only → [guides/COMPRESS.md](guides/COMPRESS.md)
4. `qai.data_pipeline` (DataPipeline with recorded, re-runnable steps) and
   `qai.serving_gateway` (Gateway multi-model routing) existed but appeared in
   no tutorial or guide → [guides/DATA.md](guides/DATA.md),
   [guides/SERVE_DEPLOY.md](guides/SERVE_DEPLOY.md)
5. `qai.runtime` (ModelVersion, RuntimeModelWrapper: staging promotion, A/B
   routing, auto-rollback, HMAC signing, `qai-runtime` CLI) — now in
   [guides/SERVE_DEPLOY.md](guides/SERVE_DEPLOY.md)
6. The 11 training-enhancement helpers (`qai.SAM`, `qai.SWA`, `qai.EMA`,
   `qai.Lookahead`, `qai.MixupTrainer`, ...) were never listed as user API →
   [guides/TRAINING.md §2](guides/TRAINING.md)
7. The full 229-technique registry was documented nowhere as a catalog →
   [guides/MODEL_CATALOG.md](guides/MODEL_CATALOG.md)
8. Task-based navigation didn't exist — users had to read a 5,000-line manual
   to find anything → [guides/](guides/README.md) "I want to..." index

**New ideas proposed (backlog, not yet built):**

| # | Idea | What it is | Why it matters |
|---|---|---|---|
| 1 | **Model Playground Notebooks** | `qai.notebook(model)` — one-command notebook with data + model + eval cells prefilled | lowest-friction onboarding for new users |
| 2 | **Technique Recommender v2** | learns from `qai.compare_techniques` results across users (opt-in telemetry) to improve `suggest_technique` | better defaults from real evidence |
| 3 | **Pipeline Diff** | `pipeline.diff(old_run, new_run)` — what changed between two DataPipeline executions | debugging data drift in pipelines |
| 4 | **Shadow Deployment** | Gateway mode: new model receives mirrored traffic, predictions logged but not served | risk-free production evaluation |
| 5 | **Federation Personality Profiles** | FedProx/DP tuning presets for known verticals (healthcare, mobile keyboard, finance) | fewer knobs for the common cases |
| 6 | **Distillation Data Generator** | synthetic soft-label generation for cases where raw training data is private (server-side teacher only) | distill without shipping data |
| 7 | **NAS Budget-Aware Search** | trials capped by wall-clock/RAM budget, early-pruning bad trials (Hyperband-style) | makes NAS practical on laptops |
| 8 | **Visual Workspace Replay Slider** | scrub training epochs, watch confidence evolve (already specced in VISUAL_DESIGN) | top-requested UI feature |
| 9 | **Explainability Report Builder** | one PDF/HTML: metrics + fairness + SHAP + model card for a run | compliance-friendly single artifact |
| 10 | **QAI Cloud-Free Registry Sync** | peer-to-peer registry replication with the existing HMAC trust system | multi-machine teams without a server |

The first three fill the biggest gaps between "feature exists" and "feature
is usable by a newcomer". Items already specced elsewhere (replay slider)
should land with Phase 4.3 work.

#### 6.5 Creative AI Research Tools ✅ ONGOING
- 118 creative ideas shipped (108 prior + 10 new in experimental11)
- Each idea is a working class with real methods, not just metaphors
- Categories: debugging, security, knowledge graph, multi-agent, evolution, economics, philosophy, biology

---

### Phase 7: Documentation & Polish ✅ DONE (v1.8.1)

#### 7.1 Update All Documentation ✅ DONE
- [x] Update `TUTORIAL.md` — reflect all new features, version header at v1.8.0
- [x] Update `FINDINGS.md` — current state
- [x] Update `UNBUILT_FEATURES.md` — implementation status (108/108 sections built)
- [x] Create `VISUAL_DESIGN.md` — capture the UI design
- [x] Create `INSTALLER.md` — document the installer
- [x] Create `CHANGELOG.md` — version history
- [x] Update `USER_MANUAL.md` — add all new sections
- [x] Create `CONTRIBUTING.md` — for open source contributors
- [x] Create `README.md` — proper landing page with doc index
- [x] Add Visual Workspace section to USER_MANUAL.md
- [x] Add Self-Improving Models section to USER_MANUAL.md
- [x] Add Conversion API section to USER_MANUAL.md
- [x] Cross-link all docs (README → tutorial → manual → roadmap → status → changelog)

#### 7.2 API Reference
- [ ] Auto-generate API docs from docstrings
- [ ] Create `docs/api/` with per-module reference
- [x] Add type hints to all public functions (mostly done)

#### 7.3 Examples & Tutorials
- [x] Create `examples/` directory with runnable scripts
- [x] `examples/classification.py`
- [x] `examples/llm_finetuning.py`
- [x] `examples/vision_pipeline.py`
- [x] `examples/auto_ml.py`
- [x] `examples/agent_demo.py`
- [x] `examples/quantization.py`
- [x] `examples/distributed_training.py`
- [ ] `examples/visual_workspace.py` — show how to launch and use the microscope
- [ ] `examples/self_improving.py` — show continuous learning end-to-end
- [ ] `examples/conversion.py` — show ONNX/TorchScript export
- [ ] `examples/benchmark_suite.py` — show full benchmark evaluation

---

## Decision Log (Key Architectural Decisions)

1. **Keep `qai.build(technique=...)` as the single entry point** — don't force 8-dimension build() signature
2. **Config objects are optional** — kwargs always work as fallback
3. **Backward compatibility is sacred** — no breaking changes to existing API
4. **Lazy-loading for advanced architectures** — don't import torch until needed
5. **Graceful degradation** — framework works even if most deps are missing
6. **Taxonomy as metadata, not API** — techniques have tags, users search by tags
7. **Infrastructure over implementation** — the value is in the unified API, not toy models
8. **HuggingFace integration is the bridge** — load real models, use qai infrastructure on them
9. **NEW** — Visual Workspace is pure HTML/CSS/JS, no build step — opens in any browser, no React/D3 install required
10. **NEW** — Self-Improving Models use generic adapters — work with any model exposing `.predict()` + `.train(X, y)`, not just qai Models
11. **NEW** — Conversion module degrades gracefully — every optional lib lazy-imported, framework never crashes on missing optional dep
12. **NEW** — Benchmark Suite is offline — synthetic questions embedded in-module, no internet access required

---

## Session History

| Session | What was built |
|---------|---------------|
| 1-10 | Initial framework: 218 techniques, visualize, chat, fixes |
| 11 | Test matrix, bug fixes, input validation, config objects |
| 12 | Universal Trainer, real attention (SDPA), 10 advanced architectures |
| 13 | PEFT (LoRA, QLoRA, etc.), metrics, alignment (DPO, GRPO, RLHF) |
| 14 | Inference (streaming, speculative, beam), pretrained loading |
| 15 | 15 more architectures, dataset loaders |
| 16 | Distributed training (DDP, FSDP), production serving, quantization |
| 17 | MLOps pipeline |
| 18 | Fast inference engine (PagedAttention), modern alignment, agents, modern vision, real tokenizers, benchmarks, test-time scaling |
| 19 | CUDA kernels, hardware detection, param introspection |
| 20 | Precision management, QAT, dequantization |
| 21 | Trainer auto-configure, creative tools (augmentation, distillation, etc.) |
| 22 | HuggingFace integration, Auto-ML, model zoo |
| 23 | Pipeline templates, experiment tracking, bug fixes |
| 24 | Phase 4 (Visual Workspace), Phase 5 (Benchmark Suite), Phase 6.1 (Conversion), Phase 6.4 (Self-Improving Models), Creative 109-118 |
| 25+ | Tokenizers, Transformer, 3-tier Memory, .qmodel Bundle, CNN, MCP server, Security, Networking, RNN/LSTM/GRU, GAN/VAE/Diffusion, MoE/GNN, Mamba/SSM, RAG, Knowledge Distillation, AI Agent |
| 30+ | Federated Learning, Distillation v2, Model Versioning, Runtime, 100+ multimodal functions, governance expansion |
| **latest** | **v1.8.0 — Distributed ML & Versioning. v1.8.1 — Docs consolidation: README, CONTRIBUTING, LICENSE, cross-linked all docs, removed binaries from tracking** |

---

## What We're NOT Doing (Explicit Non-Goals)

1. **Not reimplementing vLLM** — integrate with it, don't rebuild
2. **Not reimplementing MLflow** — integrate with it, don't rebuild
3. **Not adding more architectures** — 248 is enough breadth; fix the existing ones
4. **Not making existing models bigger** — that's the pretrained weights' job
5. **Not building a competitor to HuggingFace** — build on top of it
6. **Not implementing all 800 taxonomy entries** — add the ~25 that matter
7. **Not requiring a build step for the Visual Workspace** — pure HTML/CSS/JS only
