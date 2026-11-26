# QAI Guides — The Task Manual

*Use qai like a real-world manual: pick the task you want to do, open that chapter, follow the steps.*

> See also: [Main README](../../README.md) · [Tutorial](../TUTORIAL.md) (start-to-finish walkthrough) · [User Manual](../USER_MANUAL.md) (full API reference) · [Model Catalog](MODEL_CATALOG.md) (learn one technique at a time)

---

## "I want to..." — Task Index

| I want to... | Go to | What you'll learn |
|---|---|---|
| **Load, clean, or manipulate data** | [DATA.md](DATA.md) | CSV/JSON/Parquet/Excel/PDF loading, cleaning, dedup, feature engineering, export |
| **Train a model** | [TRAINING.md](TRAINING.md) | `qai.build` → `train` → `predict`, the Universal Trainer, pipeline templates, Auto-ML |
| **Learn one model type deeply** | [MODEL_CATALOG.md](MODEL_CATALOG.md) | All 229 techniques organized by family — what each does, when to use it, code |
| **Evaluate and explain a model** | [EVALUATION.md](EVALUATION.md) | Metrics catalog, explainability, SHAP, calibration, benchmark suite |
| **Fine-tune or align a model** | [FINETUNING_ALIGNMENT.md](FINETUNING_ALIGNMENT.md) | LoRA, QLoRA, Adapter, IA³, Prefix, Prompt + DPO, KTO, SimPO, ORPO, GRPO, RLVR |
| **Serve a model in production** | [SERVE_DEPLOY.md](SERVE_DEPLOY.md) | ProductionServer, streaming, Gateway, versioning, A/B tests, runtime signing |
| **Make a model smaller/faster** | [COMPRESS.md](COMPRESS.md) | Quantization (int8/int4/GPTQ/AWQ/GGUF), distillation v2, pruning, NAS |
| **Train across machines** | [DISTRIBUTED.md](DISTRIBUTED.md) | DDP, FSDP, federated learning with privacy (FedAvg, FedProx, DP-SGD) |
| **Build an agent or tool** | [AGENTS_MCP.md](AGENTS_MCP.md) | ReAct agents, tool use, multi-agent, MCP server exposure |
| **Give a model memory / do RAG** | [MEMORY_RAG.md](MEMORY_RAG.md) | 3-tier memory, retrieval, knowledge accumulation, RAG pipeline |
| **Secure or audit an AI system** | [SECURITY_GOVERNANCE.md](SECURITY_GOVERNANCE.md) | Poisoning detection, prompt injection guard, PII, fairness, drift, watermarking |
| **Work with images, audio, text, graphs, time series** | [MULTIMODAL.md](MULTIMODAL.md) | qai.vision, qai.audio, qai.nlp, qai.graph, qai.time_series (100+ functions) |
| **Track experiments / run MLOps** | [MLOPS.md](MLOPS.md) | Experiment tracking, model registry, drift monitoring, MLOpsPipeline |
| **See a model think (visual UI)** | [VISUAL_WORKSPACE.md](VISUAL_WORKSPACE.md) | The model microscope — architecture canvas, weight sliders, story mode |
| **Fix an error / check my setup** | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | `qai.doctor()`, health checks, preflight, OOM recovery, common errors |

---

## How the docs fit together

```
START HERE                    GO DEEP                        REFERENCE
─────────                     ───────                        ─────────
README.md  →  TUTORIAL.md  →  guides/*.md (this manual)  →  USER_MANUAL.md
              (16-section      (task-focused chapters)      (every public API,
               walkthrough)                                   full reference)

Quick answer needed?          Background / status?
────────────────────          ─────────────────────
guides/TROUBLESHOOTING.md     docs/ROADMAP.md (plans, decisions)
MODEL_CATALOG.md (per-model)  docs/FINDINGS.md (engineering notes)
                              docs/UNBUILT_FEATURES.md (spec coverage 108/108)
                              CHANGELOG.md (what shipped when)
```

**Reading paths:**

- *"I'm new to ML"* → Tutorial §1–5 → DATA.md → TRAINING.md → EVALUATION.md
- *"I know ML, new to qai"* → Tutorial → the 2–3 guides matching your task
- *"I want model X"* → MODEL_CATALOG.md → find the family → TRAINING.md
- *"I'm deploying"* → COMPRESS.md → SERVE_DEPLOY.md → MLOPS.md → SECURITY_GOVERNANCE.md
- *"I'm researching"* → AGENTS_MCP.md → MEMORY_RAG.md → ROADMAP (ideas)

---

## The one pattern to remember

Almost everything in qai follows the same four beats:

```python
import qai

model = qai.build(technique="<name>")   # 1. BUILD
model.train(X, y)                        # 2. TRAIN
result = model.predict(x)                # 3. PREDICT
model.save("model.qmodel")               # 4. PERSIST
```

Every guide chapter builds on this pattern. If you can do these four beats with
one technique, you can do them with all 229 — that's the point of the unified API.
