# QAI — Quantum AI Framework

> A unified, single-import machine learning and artificial intelligence framework.
> Build, train, predict, explain, serve, and deploy — all from `import qai`.

**Version:** 1.8.0 — *"Distributed ML & Versioning"*
**Status:** Stable · 269 Python modules · ~143,000 LOC · 108 spec sections implemented (100%)

```python
import qai

# Build, train, predict — one import, no boilerplate
model = qai.build(technique="regression")
model.train(X=[[1.0], [2.0], [3.0]], y=[2.0, 4.0, 6.0])
print(model.predict([4.0]))   # 8.0
print(model.formula())        # y = 2.000*x0 + 0.000
```

---

## Why QAI

- **Single import.** Classical ML, deep learning, RL, generative models, alignment, serving, MLOps, governance, security — all under `qai.*`.
- **Universal interface.** Every technique — from logistic regression to Mamba — exposes the same `build / train / predict / explain / save` API.
- **Graceful degradation.** The framework runs even if 90% of optional dependencies are missing. Lazy imports everywhere.
- **Backward compatible.** No breaking changes since v0.4.0. Your v0.4 scripts still work on v1.8.
- **Production-aware.** Versioning, A/B testing, cryptographic signing, drift detection, runtime bundling built in.

---

## Install

```bash
pip install -r requirements.txt
```

Or install only what you need:

```bash
qai install --vision       # CV stack (Pillow, opencv, torch vision)
qai install --nlp          # NLP stack (tokenizers, transformers)
qai install --serving      # Production serving (Flask, gunicorn)
qai install --all          # Everything
```

Check what's installed and what's missing:

```bash
qai doctor
```

---

## Quickstart

```python
import qai

# Load real data
X, y = qai.load_iris()                 # 150 samples, 4 features, 3 classes

# Build + train + predict
m = qai.build(technique="classifier")
m.train(X, y)
print(m.predict(X[0]))                 # → 0
print(m.predict_proba(X[0]))           # → [[0.98, 0.02, 0.0]]
print(m.predict_top_k(X[0], k=2))      # → [(0, 0.98), (1, 0.02)]

# Evaluate
print(m.evaluate(X, y, metrics=["accuracy"]))

# Save + load
m.save("model.qmodel")
loaded = qai.Model.load("model.qmodel")
```

For the full walkthrough see [docs/TUTORIAL.md](docs/TUTORIAL.md).

---

## Feature Map

| Area | Highlights |
|---|---|
| **Classical ML** | Regression, Classifier, Decision Tree, KNN, SVM, Naive Bayes, Random Forest, Gradient Boosting, plus 50+ sklearn-wrapped variants |
| **Deep Learning** | MLP, Transformer (RoPE, KV cache), CNN (aug + transfer learning), RNN/LSTM/GRU, Seq2Seq, GAN, VAE, Diffusion (DDPM), MoE, GNN, **Mamba/SSM** |
| **Reinforcement Learning** | TabularPolicy + GridWorld, DQN (PER + target net), PPO (clipped surrogate, actor-critic) |
| **Fine-Tuning** | LoRA, QLoRA, Adapter, IA³, Prefix, Prompt |
| **Alignment** | DPO, GRPO, RLHF, KTO, IPO, SimPO, ORPO, RLVR |
| **Inference** | Streaming, speculative, beam search, min-p, typical sampling, best-of-N, FastInferenceEngine |
| **Test-Time Scaling** | ChainOfThought, SelfConsistency, TreeOfThoughts, VerificationGuided, ReasoningChain |
| **Agents** | ReAct, ToolUse, Planner, MultiAgent, AIAgent |
| **Auto-ML** | `qai.auto_train()`, `compare_techniques()`, `suggest_technique()` |
| **Pipelines** | classification / regression / clustering / NLP / LLM / time_series / auto |
| **Data** | `qai.data` — CSV / JSON / Parquet / Excel / PDF / DOCX / EPUB, quality reports, PII detection |
| **Serving** | `qai.ProductionServer` (dynamic batching, SSE streaming), `FastInferenceEngine` |
| **Quantization** | int8, int4 (NF4), GPTQ, AWQ, GGUF, EXL2, QAT, dequantization |
| **MLOps** | `MLOpsPipeline`: train → track → register → quantize → serve → monitor |
| **Memory** | `qai.memory`: WorkingMemory, RetrievalMemory, WeightsMemory with auto-promotion |
| **Bundle Format** | `.qmodel` ZIP with weights + tokenizer + logic, HMAC-SHA256 signed |
| **MCP Server** | `qai.mcp` — 14 built-in tools, JSON-RPC over stdio/HTTP |
| **Security** | `qai.security` — poisoning detection, prompt-injection guard, PII scan, secret scan, sandbox, APIKeyVault |
| **Networking** | `qai.network` — HTTP client (retry/cache/rate-limit), WebSocket, OAuth (5 providers), CircuitBreaker, HMAC signing |
| **Federated Learning** | `qai.federation` — FedAvg, FedProx, Weighted, DP-SGD, secure aggregation |
| **Knowledge Distillation** | `qai.distillation_v2` — single/multi-teacher, feature-based, online distillation |
| **Explainability** | `feature_importance`, `permutation_importance`, `partial_dependence`, SHAP approximation |
| **Governance** | Drift detection, fairness audit, model watermarking, model cards |
| **Quantum** | Kronecker-product statevector simulator (H, X, Y, Z, S, T gates) |
| **Visual Workspace** | `qai.workspace(model)` — interactive model microscope (pure HTML/CSS/JS) |

Full feature inventory: [docs/IMPLEMENTATION_STATUS.md](docs/UNBUILT_FEATURES.md).

---

## Project Layout

```
Quantum-Frame-work/
├── README.md                  ← you are here
├── CHANGELOG.md               ← version history
├── CONTRIBUTING.md            ← contributor guide
├── LICENSE                    ← MIT
├── requirements.txt
├── .env.example               ← env-var template (copy to .env)
├── .gitignore
│
├── qai/                       ← framework source (269 modules)
│   ├── __init__.py            ← single-import surface
│   ├── __main__.py            ← `python -m qai` entry
│   ├── core/                  ← build(), Model, logic, pipeline, metrics, ...
│   ├── techniques/            ← technique registry + training enhancements
│   ├── learning/              ← layers, schedulers, early stopping
│   ├── mechanics/             ← objectives, training loops
│   ├── preprocessing/         ← scalers, encoders, transformers
│   ├── tokenizers/            ← Char / Word / BPE
│   ├── memory/                ← Working / Retrieval / Weights memory tiers
│   ├── bundle/                ← .qmodel format
│   ├── mcp/                   ← Model Context Protocol server + 14 tools
│   ├── security/              ← poisoning, PII, secrets, sandbox, APIKeyVault
│   ├── network/               ← HTTP, WebSocket, OAuth, CircuitBreaker
│   ├── runtime/               ← versioning, A/B, signing, CLI
│   ├── governance/            ← drift, fairness, watermarking
│   ├── vision/  audio/  nlp/  ← multimodal feature suites
│   ├── graph/  time_series/
│   ├── data/                  ← DataDataset, format detection, quality reports
│   ├── agents/                ← ReAct, ToolUse, Planner, MultiAgent
│   ├── creative/              ← 118 experimental research ideas
│   └── math/  system/  tracking/
│
├── docs/                      ← documentation
│   ├── USER_MANUAL.md         ← full API reference (~5,000 lines)
│   ├── TUTORIAL.md            ← 16-section walkthrough
│   ├── ROADMAP.md             ← development roadmap & decision log
│   ├── PROGRESS.md            ← implementation progress tracker
│   ├── FINDINGS.md            ← engineering findings & architecture notes
│   ├── VISUAL_DESIGN.md       ← visual workspace UI design
│   └── UNBUILT_FEATURES.md    ← implementation status (all 108 sections)
│
├── tests/                     ← pytest test suite
└── showcase/                  ← example apps (oracle, prism, evolve, nexus, atlas)
```

---

## Documentation

| Document | Purpose |
|---|---|
| [README.md](README.md) | This file — landing page, install, quickstart |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | 16-section hands-on tutorial — start here |
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | Full API reference for every public class/function |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Development roadmap, decision log, session history |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Section-by-section implementation tracker |
| [docs/FINDINGS.md](docs/FINDINGS.md) | Engineering findings, gotchas, architecture notes |
| [docs/VISUAL_DESIGN.md](docs/VISUAL_DESIGN.md) | Visual Workspace UI design spec |
| [docs/UNBUILT_FEATURES.md](docs/UNBUILT_FEATURES.md) | Implementation status by spec section (108/108 built) |
| [CHANGELOG.md](CHANGELOG.md) | Version history (latest: v1.8.0) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute — code style, tests, PRs |

---

## Examples

```python
# Auto-ML — let qai pick the best model
result = qai.auto_train(X, y, task="classification", time_budget_s=30)
print(result.summary())             # "Winner: classifier (accuracy=0.83)"
result.best_model.predict(X_new)

# Transformer from scratch
m = qai.build(technique="transformer",
    vocab_size=1000, d_model=256, n_heads=8, n_layers=6,
    norm_type="rmsnorm", attention_impl="auto")
m.train(corpus, epochs=10)

# Fine-tune with LoRA
m.finetune(X_new, y_new, method="lora", rank=8, alpha=16)

# Align with DPO
dpo = qai.DPOTrainer(model=m, beta=0.1)
dpo.train(preference_data, epochs=1)

# Quantize + serve
qai.quantize_dynamic(m)             # int8
server = qai.ProductionServer(model=m, port=8000, enable_streaming=True)
server.start()
```

More examples in [docs/TUTORIAL.md](docs/TUTORIAL.md) and the `showcase/` apps.

---

## Tests

```bash
pytest tests/ -v
```

The suite covers classical techniques, deep learning, alignment, federated learning, MCP, security, networking, bundle round-trips, and end-to-end sweeps on real HuggingFace iris data.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version:

1. Fork → branch → PR against `main`
2. Add tests for any new technique or feature
3. Keep the single-import surface stable — no breaking changes
4. Run `pytest tests/` before submitting

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

Built on top of excellent open-source work: NumPy, scikit-learn, PyTorch, HuggingFace Transformers, Flask, and many others. QAI is the unified layer on top — it does not reinvent what those libraries do well.
