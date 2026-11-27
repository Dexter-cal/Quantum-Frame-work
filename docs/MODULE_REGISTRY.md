# QAI Module Registry — Every Module, A to Z

> See also: [README](../README.md) · [Task Manual](guides/README.md) · [User Manual](USER_MANUAL.md) · [Focus List](UNBUILT_FEATURES.md)

Complete inventory of the `qai` package: **155 modules and subpackages**.
Every one-liner below is generated from the module's real docstring in the source,
so this page can never drift from the code.

Legend: **NEW-DOC** = built earlier but only documented in v1.8.3 (these were the
"features with no documentation" — now every module is findable). `internal` = not
re-exported on the single `import qai` face; import it directly, e.g.
`from qai.workspace_html import WORKSPACE_HTML_V2`.

## Core & framework (19)

| Module | What it does | Import face |
|---|---|---|
| `qai.framework` | qai.framework -- Framework-level features (spec Sections 35, 36, 43, 51, 55, 58, 67, 79, 94). Eight production framework systems: 1. Outp... | top-level |
| `qai.types` | qai.types -- type aliases + Protocol definitions + error handling helpers. Production hardening pass: - Type aliases for common qai types... | top-level |
| `qai.configs` | qai.configs — family-specific configuration objects for qai.build(). Instead of passing 20 kwargs to qai.build(), users can pass a config... | top-level |
| `qai.core/__init__` | Core internals: Model, AutoTuner, weights plumbing — the engine behind qai.build() / qai.Model | top-level |
| `qai.techniques/__init__` | qai.techniques -- The core catalogue of built-in and user-registered ML techniques | top-level |
| `qai.mechanics/__init__` | qai.mechanics -- Internal learning mechanics, loss functions, optimizers, and schedulers | top-level |
| `qai.completion` | qai.completion -- Remaining spec sections to push past 50% completion. Ten systems covering spec Sections 6, 7, 16, 20, 22, 26, 82, 98, 1... | top-level |
| `qai.native` | qai.native -- C-extension bridge for direct memory access. This is the answer to the user's question: "can we write to memory like RAM/RO... | top-level |
| `qai.sklearn_compat` | qai.sklearn_compat -- discoverability layer for sklearn-wrapped techniques. Many qai techniques are thin wrappers around sklearn estimato... | `internal` |
| `qai.asyncio` | qai.asyncio -- async wrappers for the major qai APIs. Many qai operations are CPU/IO-bound (training, retrieval, generation, serving). Th... | top-level |
| `qai.chainable_output` | qai.chainable_output — chainable model output formatting (PDF spec feature). The PDF spec says: "model.output() as a real, chainable, ins... | `internal` |
| `qai.remainder` | qai.remainder -- Final remaining spec sections to maximize completion. Covers spec Sections 12, 13, 18, 21, 29, 34, 39, 44, 57, 62, 71, 7... | top-level |
| `qai.finalize` | qai.finalize -- Remaining spec sections to push completion past 60%. Nine systems covering spec Sections 15, 17, 40, 59, 60, 69, 72, 86, ... | top-level |
| `qai.upgrades` | qai.upgrades -- Completes the last 6 partial (⚠️) spec sections. 1. finetune() for more techniques (Section 5) 2. Federated cluster orche... | top-level |
| `qai.cbridge` **NEW-DOC** | qai.cbridge -- C bridge for bypassing Python limitations. The C bridge exists because Python has limits: - inspect.getsource() doesn't wo... | top-level |
| `qai.gpu` | qai.gpu -- GPU compute + custom CUDA kernels + device-aware tensors. Five layers, from highest to lowest: Layer 1: DeviceManager Detects ... | top-level |
| `qai.cuda_kernels` | qai.cuda_kernels — GPU/CUDA kernel management and hardware-aware inference. Provides: - HardwareDetector — detects CPU/GPU/TPU/MPS, recom... | top-level |
| `qai.system/__init__` | qai.system -- System environment, resource monitoring, RAM/ROM automation, time/date utilities, and automated memory safeguards for AI wo... | top-level |
| `qai.math/__init__` | qai.math -- Comprehensive mathematics, linear algebra, calculus, activations, loss functions, probability, statistics, information theory... | top-level |

## Data handling (12)

| Module | What it does | Import face |
|---|---|---|
| `qai.data/__init__` | qai.data -- Complete Data Manipulation, Cleaning, Universal Ingestion, Quality Auditing, Formulas, Version Control, NLP/Time-Series, and ... | top-level |
| `qai.preprocessing/__init__` | Data preprocessing, scaling, encoding, missing-value handling, and automated cleaning utilities for qai | top-level |
| `qai.datasets` | qai.datasets — real dataset loaders for evaluation. Provides: - load_iris() — Fisher's iris dataset (classification, 150 samples, 4 featu... | top-level |
| `qai.data_pipeline` | qai.data_pipeline — automated data preprocessing pipeline. Auto-detects data types, cleans, transforms, and feature-engineers a raw datas... | top-level |
| `qai.data_profiler` **NEW-DOC** | qai.data_profiler -- dataset health checker. Profiles a dataset and reports: missing values, outliers, type inference, distribution stats... | top-level |
| `qai.data_augmentor` **NEW-DOC** | qai.data_augmentor — configurable data augmentation pipeline for tabular and image data | top-level |
| `qai.feature_store` **NEW-DOC** | qai.feature_store — feature engineering + storage + reuse across models. A feature store lets you compute features once, store them, and ... | top-level |
| `qai.pipeline_builder` **NEW-DOC** | qai.pipeline_builder -- declarative DAG pipeline builder. Build multi-stage ML pipelines as a directed acyclic graph (DAG) with a declara... | top-level |
| `qai.pipeline_templates` | qai.pipeline_templates — pre-built end-to-end ML workflows. Ready-to-use pipeline templates that combine preprocessing, training, evaluat... | top-level |
| `qai.synthetic` | qai.synthetic -- synthetic data generators for testing + benchmarks. Beyond make_classification / make_regression (which are toy-grade), ... | top-level |
| `qai.formats` | qai.formats — comprehensive ML file format registry + converters. Supports ALL known ML model/weight file formats: Model formats (store a... | top-level |
| `qai.time_series/__init__` | qai.time_series -- Time-Series Forecasting & Analysis Utilities | `internal` |

## Model zoo & architectures (13)

| Module | What it does | Import face |
|---|---|---|
| `qai.advanced_architectures` | qai.advanced_architectures — production-grade modern architectures. This module adds 10 genuinely advanced architectures that were missin... | `internal` |
| `qai.advanced_architectures_2` | qai.advanced_architectures_2 — second batch of advanced architectures. Adds 15 more production-grade architectures: 1. Switch Transformer... | `internal` |
| `qai.frontier2025` **NEW-DOC** | qai.frontier2025 -- 2025-era frontier architectures built on the framework. 1. Mamba2 -- Structured State Space Duality (Dao & Gu 2024) S... | top-level |
| `qai.modern_vision` | qai.modern_vision — 2024-2026 state-of-the-art vision models. Provides: - SAM2 — Segment Anything Model 2 (video + image segmentation) - ... | `internal` |
| `qai.modern_alignment` | qai.modern_alignment — 2025-2026 alignment methods beyond DPO/GRPO/RLHF. Provides: - KTO (Kahneman-Tversky Optimization) — Ethayarajh et ... | top-level |
| `qai.pretrained` | qai.pretrained — pretrained weight loading for qai models. Provides: - qai.build(technique=..., pretrained=True) — auto-download weights ... | top-level |
| `qai.vision/__init__` | qai.vision -- Computer Vision Preprocessing & Image Utilities | `internal` |
| `qai.nlp/__init__` | qai.nlp -- Natural Language Processing & Tokenization Utilities | `internal` |
| `qai.audio/__init__` | qai.audio -- Audio Signal Processing & Spectrogram Utilities | `internal` |
| `qai.graph/__init__` | qai.graph -- Graph Neural Network & Network Topology Utilities | `internal` |
| `qai.tokenizers/__init__` | qai.tokenizers -- Text tokenizers for character-level, BPE, and word-level tokenization. This is the foundational module that unlocks seq... | top-level |
| `qai.real_tokenizers` | qai.real_tokenizers — production-grade tokenizer wrappers. Wraps the industry-standard tokenizers: - TiktokenWrapper — GPT-4/ChatGPT's BP... | top-level |
| `qai.agents_base` **NEW-DOC** | qai.agents_base -- Base class for learning agents + replay buffer + hyperparameter scheduler. This module addresses three of the framewor... | top-level |

## Training & optimization (21)

| Module | What it does | Import face |
|---|---|---|
| `qai.trainer` | qai.trainer — universal training loop for ALL qai models. Replaces the per-technique ad-hoc fit() loops with a single, feature-complete t... | top-level |
| `qai.distributed` | qai.distributed -- distributed training wrappers (2025/2026 era). 1. DistributedTrainer -- top-level coordinator 2. RayBackend -- Ray Tra... | top-level |
| `qai.distributed_train` | qai.distributed_train — multi-GPU and distributed training utilities. Provides: - DistributedTrainer — wraps qai.Trainer with DDP (Distri... | top-level |
| `qai.live_training` | qai.live_training -- real-time training dashboard. Opens a localhost web dashboard that shows training curves in real time as the model t... | top-level |
| `qai.federation` | qai.federation — federated learning framework. Train a model across multiple clients (each with private data) without centralizing the da... | top-level |
| `qai.federated_bridge` **NEW-DOC** | qai.federated_bridge — combine model pieces from different frameworks into one. Lets you build pieces of a model in PyTorch, sklearn, or ... | top-level |
| `qai.active_learning` **NEW-DOC** | qai.active_learning -- interactive labeling loop for any classifier. Active learning: instead of labeling a random sample of your data, l... | top-level |
| `qai.active_learning_v2` **NEW-DOC** | qai.active_learning_v2 -- advanced active-learning toolkit for the Quantum AI Framework. Active learning: instead of labeling a random sa... | top-level |
| `qai.finetuning` | qai.finetuning — parameter-efficient fine-tuning methods. Provides: - LoRA (Low-Rank Adaptation) — Hu et al. 2021 - QLoRA (Quantized LoRA... | top-level |
| `qai.alignment` | qai.alignment — preference alignment trainers. Provides: - DPO (Direct Preference Optimization) — Rafailov et al. 2023 - GRPO (Group Rela... | top-level |
| `qai.hyperparam_optimizer` **NEW-DOC** | qai.hyperparam_optimizer — Bayesian hyperparameter optimization. Goes beyond grid/random search with a Gaussian Process surrogate model t... | top-level |
| `qai.hparam_optimizer` **NEW-DOC** | qai.hparam_optimizer — Bayesian hyperparameter optimization with expected improvement. More sophisticated than qai.AutoTuner's "bayesian"... | `internal` |
| `qai.auto_ml` | qai.auto_ml — automatic model selection and hyperparameter optimization. Tries multiple techniques on your data, evaluates each via cross... | top-level |
| `qai.automl` | qai.automl -- automated machine learning pipeline. AutoML — end-to-end automated ML: data prep → model selection → tuning → ensemble Mode... | top-level |
| `qai.budget` | qai.budget — resource budget management for ML workloads. Manages RAM, GPU VRAM, CPU, and disk budgets so you can: - Set a global RAM bud... | top-level |
| `qai.callbacks` | qai.callbacks -- Standard training callbacks (Keras-style contract). Each callback is an object with zero or more of these methods: - on_... | top-level |
| `qai.self_improving` | qai.self_improving -- production-grade self-improving / continuously-learning models. This is the production version of the "Self-Improvi... | top-level |
| `qai.nas_search` | qai.nas_search — Neural Architecture Search. Automatically discover the best neural network architecture for a given task by trying diffe... | top-level |
| `qai.experiment_tracking` | qai.experiment_tracking — MLflow/W&B/TensorBoard integration. Provides: - ExperimentLogger — unified logging to MLflow, W&B, or JSON - lo... | top-level |
| `qai.experiment_compare` **NEW-DOC** | qai.experiment_compare -- compare multiple experiment runs side by side. This module provides a richer, side-by-side comparison toolkit f... | top-level |
| `qai.learning/__init__` | qai.learning -- the LearningTechnique registry (Section 66/67), a separate axis from qai.techniques (which is architecture/Type) | top-level |

## Compression & conversion (15)

| Module | What it does | Import face |
|---|---|---|
| `qai.distillation_v2` | THE canonical distillation implementation (v1.8.5): real Hinton KL loss (torch path), honest fallback for numpy, strategies vanilla/hint/attention/self/multi, entry point `qai.distill(...)`. All other distiller classes are delegates. | top-level |
| `qai.quantization` | qai.quantization — model quantization for deployment. Provides: - quantize_dynamic() — post-training int8 dynamic quantization (fast, no ... | top-level |
| `qai.precision` | qai.precision — precision management and quantization-aware training. Provides: - PrecisionManager — auto-detect best precision for hardw... | top-level |
| `qai.compression` | qai.compression -- model compression toolkit: quantize + prune + distill. CompressionPipeline — end-to-end compression in one call Quanti... | top-level |
| `qai.compression_advisor` **NEW-DOC** | qai.compression_advisor — automatic compression strategy recommender. Analyzes a model and recommends the best combination of: * Pruning ... | top-level |
| `qai.compress` | qai.compress -- a learned meta-compressor. Real-world files have different byte distributions, and no single compression algorithm is bes... | top-level |
| `qai.conversion` | qai.conversion -- unified bidirectional model conversion API. Convert any qai.Model (or sklearn estimator, or torch.nn.Module) to any of ... | top-level |
| `qai.hf_converter` **NEW-DOC** | qai.hf_converter -- convert HuggingFace models to/from qai format. Download a model from HuggingFace Hub → convert to .qmodel → use with qai | top-level |
| `qai.huggingface_hub` | qai.huggingface_hub — full HuggingFace Hub integration. Browse, search, download, load, and upload models from the HuggingFace Hub. This ... | top-level |
| `qai.modelfile` **NEW-DOC** | qai.modelfile -- model files with proper extensions + a full file system. Solves the user's question: "when you train a model, it should ... | top-level |
| `qai.modelfile_extended` **NEW-DOC** | qai.modelfile.extended -- extended model file features. HFModelImporter — download from HuggingFace → wrap as qai.Model → .qmodel ModelRe... | top-level |
| `qai.modelfile_extensions` **NEW-DOC** | qai.modelfile_extensions -- extra model-file features. HuggingFaceImporter — download HF models and import to .qmodel ModelRegistry — glo... | top-level |
| `qai.weight_translation` | qai.weight_translation — weight-to-concept translator (PDF spec feature). Translates raw weight vectors into human-readable concept label... | top-level |
| `qai.surgery` **NEW-DOC** | qai.surgery -- weight surgery: precise operations on model weights. WeightSurgeon — surgical operations on specific neurons/channels/laye... | top-level |
| `qai.weights` | qai.weights -- weights file system: inspect, edit, merge, visualize, compare. Weights are the "source code" of a model. This module treat... | top-level |

## Evaluation & interpretability (17)

| Module | What it does | Import face |
|---|---|---|
| `qai.evaluate` | qai.evaluate -- one-call evaluation suite. Picks the right metrics for the technique type and returns a formatted report. No more remembe... | top-level |
| `qai.advanced_metrics` | qai.advanced_metrics — real evaluation metrics for ML models. Provides production-grade metrics: - Perplexity (for language models) - BLE... | top-level |
| `qai.benchmark` | qai.benchmark -- performance + scaling benchmarks for the framework. - ModelBenchmark : throughput, latency, memory for a single model - ... | top-level |
| `qai.benchmark_suite` | qai.benchmark_suite — comprehensive, offline, synthetic-but-realistic benchmark suite. This module provides a self-contained, internet-fr... | top-level |
| `qai.benchmarks` | qai.benchmarks — standard ML benchmark runners. Provides: - MMLU runner — Massive Multitask Language Understanding (57 subjects) - HumanE... | `internal` |
| `qai.model_health` **NEW-DOC** | qai.model_health — model health checks and diagnostics. Provides: - health_check(model): runs a battery of checks on a trained model (dea... | top-level |
| `qai.confidence_calibrator` **NEW-DOC** | qai.confidence_calibrator — calibrate prediction confidence. Production models often have poorly calibrated probabilities (overconfident ... | top-level |
| `qai.interpret` **NEW-DOC** | qai.interpret -- mechanistic interpretability toolkit (2025 era). Five core tools inspired by Anthropic / OpenAI / DeepMind interpretabil... | top-level |
| `qai.model_visualizer` | qai.model_visualizer — universal model architecture visualization for ALL techniques. Generates visual diagrams (matplotlib) showing the ... | top-level |
| `qai.ab_testing` **NEW-DOC** | qai.ab_testing -- production A/B testing with statistical significance. Route traffic between two models, collect results, and determine ... | top-level |
| `qai.drift` | qai.drift -- production drift detection. In production, your input distribution shifts. A model trained on January data may behave badly ... | top-level |
| `qai.governance/__init__` | Governance & drift detection utilities for models and features | top-level |
| `qai.lineage` | qai.lineage -- model lineage / provenance tracker. Records the family tree of model versions (which model was fine-tuned from which, whic... | top-level |
| `qai.cpc` | qai.cpc -- Convergent Pattern Compression. A file-native learning system. You show it files; it studies their structure and meaning, extr... | top-level |
| `qai.cpc_ai` **NEW-DOC** | qai.cpc_ai -- Convergent Pattern Compression AI. The complete system: a compression-first intelligence that reads files, reorganizes them... | top-level |
| `qai.model_tools` | qai.model_tools — creative model utilities and introspection tools. Provides: - suggest_hyperparameters() — auto-suggest good hyperparams... | top-level |
| `qai.test_time_scaling` | qai.test_time_scaling — test-time compute scaling methods. Provides the 2025-2026 paradigm of "thinking longer at inference time": - Chai... | top-level |

## Agents & language (13)

| Module | What it does | Import face |
|---|---|---|
| `qai.agent` | qai.agent -- the flagship Quantum Agent system. This is the culmination of the entire framework: a super-intelligent multi-agent system t... | top-level |
| `qai.agent_framework` | qai.agents — AI agent framework with tool use, multi-agent, and reasoning. Provides: - Agent — single agent with tool use, memory, and re... | top-level |
| `qai.agents/__init__` | qai.agents -- agentic AI patterns and multi-agent orchestration (2025 era). Six core patterns + an A2A (Agent-to-Agent) protocol implemen... | top-level |
| `qai.mind` **NEW-DOC** | qai.mind -- the QuantumMind: a super-intelligent agent built on the framework. This is the framework's flagship creation — an AGI-like ag... | top-level |
| `qai.reasoning` | qai.reasoning -- test-time compute scaling (2024-2026 era). 1. ChainOfThought -- prompt the model to think step-by-step 2. SelfConsistenc... | top-level |
| `qai.mcp/__init__` | qai.mcp -- Model Context Protocol server for qai. Exposes any qai model (and other tools) as an MCP-compatible server that AI assistants ... | top-level |
| `qai.memory/__init__` | qai.memory -- Three-tier memory system (spec Section 41/95). Real production AI systems need three distinct kinds of "memory": model.memo... | top-level |
| `qai.memory_vault` **NEW-DOC** | qai.memory_vault — a tiered persistent memory system for AI agents. Three tiers with smart eviction, compression, and recall scoring: HOT... | top-level |
| `qai.rag` | qai.rag -- next-generation Retrieval-Augmented Generation (2024-2026 era). Five RAG paradigms in one module: 1. BaselineRAG -- standard v... | top-level |
| `qai.model_chat` **NEW-DOC** | qai.model_chat — interactive chat window for any model. Opens a web-based chat interface where you can: - Type inputs and see predictions... | top-level |
| `qai.creative/__init__` | qai.creative -- the 15 creative ideas built on top of the Quantum framework. These are experimental, opinionated, "wouldn't it be cool if... | top-level |
| `qai.creative_tools` | qai.creative_tools — advanced creative features for the ML workflow. Provides: - DataAugmentor — automatic data augmentation for any data... | top-level |
| `qai.collaborative` | qai.collaborative -- real-time collaborative Visual Workspace. Lets multiple users edit the same workspace simultaneously via WebRTC (pee... | top-level |

## Serving & production (17)

| Module | What it does | Import face |
|---|---|---|
| `qai.serving` | qai.serving -- unified LLM serving layer. Wraps the major 2025/2026 LLM serving engines behind one API: Production engines: - VLLMServer ... | top-level |
| `qai.serving_gateway` | qai.serving_gateway — unified model serving gateway. Deploy any qai/sklearn/torch model behind a single REST/gRPC-like API with: dynamic ... | top-level |
| `qai.production` | qai.production -- Production AI features (spec Sections 14, 24, 50, 54, 87). Five production-grade systems that complete the framework's ... | top-level |
| `qai.production_serving` | qai.production_serving — production-grade model serving. Provides: - ProductionServer — high-throughput HTTP server with: - Dynamic batch... | top-level |
| `qai.deployment` | qai.deployment -- production deployment pipeline. ModelDeployer — deploy models to Flask/FastAPI/Docker/serverless ModelRegistry3 — produ... | top-level |
| `qai.fast_inference` | qai.fast_inference — vLLM-style high-throughput inference engine. Implements the three key innovations that make vLLM ~100x faster than n... | top-level |
| `qai.inference` | qai.inference — advanced inference methods for generative models. Provides: - Speculative decoding (Leviathan et al. 2023) — 2-3x speedup... | top-level |
| `qai.streaming` | qai.streaming — streaming training from HuggingFace without full downloads. Train on datasets that are LARGER than your RAM/disk by pulli... | top-level |
| `qai.runtime/__init__` | qai.runtime -- Complete Production Runtime, Semantic Versioning, Staging Environments, AB Testing, Auto-Rollback, Cryptographic Security,... | top-level |
| `qai.autoscale` **NEW-DOC** | qai.autoscale — hardware-aware auto-configuration + C-accelerated training. HardwareProfiler — detect CPU/GPU/RAM/disk and score ML capab... | top-level |
| `qai.model_registry` | qai.model_registry -- local model marketplace. A local registry for publishing, discovering, versioning, and searching trained qai models... | top-level |
| `qai.marketplace` | qai.marketplace — community plugin marketplace for qai. Browse, install, and share community plugins (custom techniques, datasets, metric... | top-level |
| `qai.integrations` **NEW-DOC** | qai.integrations -- adapters to the wider AI ecosystem (2025/2026 era). Wraps (rather than reimplements) the libraries AI devs actually u... | top-level |
| `qai.integrations_real` **NEW-DOC** | qai.integrations.real -- real library integrations with actual tests. Unlike qai.integrations (which is mostly stubs that lazy-import), t... | top-level |
| `qai.installer` | qai.installer — smart dependency management and launcher. The installer that makes qai "just work" — even with missing dependencies. Feat... | top-level |
| `qai.infra` | qai.infra -- Infrastructure features (spec Sections 10, 42, 52, 63, 103). Five production infrastructure systems: 1. ModelMerger (Section... | top-level |
| `qai.environments` | qai.environments -- Standard environment interface + registry. The spec has `qai.GridWorld` as the only built-in environment, but every R... | top-level |

## MLOps & monitoring (12)

| Module | What it does | Import face |
|---|---|---|
| `qai.mlops` | qai.mlops — integrated MLOps pipeline tying together all production features. Provides: - MLOpsPipeline — end-to-end: train → track → reg... | top-level |
| `qai.tracking/__init__` | Experiment tracking and metrics logging for qai workflows | top-level |
| `qai.profiler` | qai.profiler -- model performance profiler. Measures latency, memory, parameter count, and (where possible) FLOPs estimate for any qai mo... | top-level |
| `qai.observability` **NEW-DOC** | qai.observability -- production model monitoring / observability system. Tracks drift, latency, accuracy, and alerts in real-time. Provid... | top-level |
| `qai.model_monitor` **NEW-DOC** | qai.model_monitor — production monitoring that tracks drift, latency, and accuracy over time. Combines qai.DriftDetector, latency trackin... | top-level |
| `qai.modellog` **NEW-DOC** | qai.modellog — per-model logging system for production. Every model gets a log file that tracks: - Training events (fit/train calls, loss... | top-level |
| `qai.model_versioner` **NEW-DOC** | qai.model_versioner — track model versions, compare diffs, rollback | top-level |
| `qai.versioning` | qai.versioning — git-like versioning for ML experiments. Tracks the full provenance of an ML experiment: * Code version (git SHA / file h... | top-level |
| `qai.model_card` | qai.model_card -- auto-generated model cards for responsible AI. Generates a Model Card (Mitchell et al. 2019) for any qai model, includi... | top-level |
| `qai.model_card_generator` **NEW-DOC** | qai.model_card_generator -- comprehensive auto-generation of model cards. Builds a Hugging Face-style Model Card for any model: qai Model... | top-level |
| `qai.self_healing` **NEW-DOC** | qai.self_healing — a model wrapper that detects performance degradation and automatically retrains when accuracy drops below a threshold | top-level |
| `qai.workflow` | qai.workflow -- node-based visual workflow / pipeline builder for ML. Think n8n / Node-RED / LangFlow / Microsoft Fabric: users compose a... | top-level |

## Security & networking (2)

| Module | What it does | Import face |
|---|---|---|
| `qai.security/__init__` | qai.security -- Security module (spec Section 33). Real production AI systems need defense-in-depth: detect poisoned data, guard against ... | top-level |
| `qai.network/__init__` | qai.network -- Networking module (spec Section 88). HTTP/WebSocket client utilities + OAuth connector stubs + task orchestration. Inspire... | top-level |

## UI & visualization (6)

| Module | What it does | Import face |
|---|---|---|
| `qai.visual_workspace` | qai.visual_workspace -- the interactive "Model Microscope". A self-contained, single-page HTML/CSS/JS workspace that lets you: * Watch th... | top-level |
| `qai.visual_workspace_v2` **NEW-DOC** | qai.visual_workspace_v2 — the redesigned "Model Microscope" v2. A major upgrade over v1: * Per-model-type renderers (MLP, Transformer, De... | top-level |
| `qai.workspace_html` **NEW-DOC** | qai.workspace_html — the v2 Visual Workspace frontend (single-file HTML/CSS/JS). Held in its own module so visual_workspace_v2.py stays f... | `internal` |
| `qai.webui` **NEW-DOC** | qai.webui -- interactive dashboard for browsing models/techniques/benchmarks. A self-contained HTML+JS dashboard that doesn't require any... | top-level |
| `qai.dashboard` | qai.dashboard -- a generic web UI for any qai model. Opens a localhost HTTP server (Python stdlib only -- no Flask, no FastAPI) and serve... | top-level |
| `qai.bundle/__init__` | qai.bundle -- The .qmodel bundle format (spec Section 68/76). A `.qmodel` file is a ZIP archive bundling everything needed to deploy a mo... | top-level |

## Docs, CLI & meta (8)

| Module | What it does | Import face |
|---|---|---|
| `qai.cli` | qai.cli -- Section 101's real command-line tool. `python -m qai <args>` works without needing a pip install, so this is genuinely testabl... | `internal` |
| `qai.cli_tool` | qai.cli_tool -- command-line interface for the qai framework | `internal` |
| `qai.docs` | qai.docs -- documentation utilities + tutorials. Quickstart helpers that generate example code and tutorials on demand | top-level |
| `qai.autodoc` **NEW-DOC** | qai.doc -- auto-generated markdown documentation for any qai model. Generates a human-readable markdown description of the model: - What ... | top-level |
| `qai.api_reference` **NEW-DOC** | qai.api_reference -- API reference auto-generator for the qai package. Walks the entire ``qai`` package using AST parsing (no imports of ... | top-level |
| `qai.tutorials` | qai.tutorials -- interactive Jupyter-style tutorials. Generates self-contained tutorial notebooks (.ipynb) + Python scripts that walk thr... | top-level |
| `qai.advanced` | qai.advanced -- RAG pipeline, Knowledge Distillation, and AI Agent framework. Three modules that complete the most important missing prod... | top-level |
| `qai.__main__` | `python -m qai` CLI entry point (same commands as qai.cli) | `internal` |

---

## How this page is maintained

Generated from source docstrings — regenerate after adding a module:
```bash
python3 scripts/gen_module_registry.py   # writes docs/MODULE_REGISTRY.md
```

Rules for contributors (see [CONTRIBUTING](../CONTRIBUTING.md)): every new public
module must (1) be reachable from `import qai` or be marked `internal`, (2) appear
here, and (3) get at least one usage example in the relevant Task Manual chapter.
