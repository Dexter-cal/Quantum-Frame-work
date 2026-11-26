# QAI Framework — Tutorial

*Updated for v1.8.0*

> See also: [README](../README.md) · [User Manual](USER_MANUAL.md) · [Roadmap](ROADMAP.md) · [Implementation Status](UNBUILT_FEATURES.md) · [Changelog](../CHANGELOG.md)

Welcome to **qai** — a unified, single-import ML/AI framework. This tutorial
covers everything from basic classification to LLM fine-tuning, serving, and
MLOps pipelines.

```python
import qai
print(qai.__version__)  # 1.8.0
```

---

## Table of Contents

1. [Quickstart: 5-Minute Classification](#1-quickstart)
2. [Auto-ML: Let qai Pick the Best Model](#2-auto-ml)
3. [Data: Loading from Any Source](#3-data-loading)
4. [Training: The Universal Trainer](#4-training)
5. [Evaluation & Metrics](#5-evaluation)
6. [Model Inspection & Explainability](#6-inspection)
7. [Deep Learning: Transformers & CNNs](#7-deep-learning)
8. [Fine-Tuning: LoRA, QLoRA, Adapters](#8-fine-tuning)
9. [Alignment: DPO, KTO, SimPO, RLVR](#9-alignment)
10. [Serving: Production Server](#10-serving)
11. [Quantization & Deployment](#11-quantization)
12. [MLOps Pipeline](#12-mlops)
13. [HuggingFace Integration](#13-huggingface)
14. [Agents & Tools](#14-agents)
15. [Custom Plugins](#15-plugins)
16. [Safety & Crash Recovery](#16-safety)

---

## 1. Quickstart

```python
import qai
import numpy as np

# Load real data
X, y = qai.load_iris()  # 150 samples, 4 features, 3 classes

# Build + train
m = qai.build(technique="classifier")
m.train(X, y)

# Predict
print(m.predict(X[0]))           # → 0
print(m.predict_proba(X[0]))     # → [[0.98, 0.02, 0.0]]
print(m.predict_top_k(X[0], k=2))  # → [(0, 0.98), (1, 0.02)]

# Evaluate
print(m.evaluate(X, y, metrics=["accuracy"]))  # → {'accuracy': 0.97, ...}

# Save + load
m.save("model.qmodel")
loaded = qai.Model.load("model.qmodel")
```

---

## 2. Auto-ML

Let qai try multiple techniques and pick the best:

```python
# One call — tries RF, SVM, GB, classifier, picks winner
result = qai.auto_train(X, y, task="classification", time_budget_s=30)
print(result.summary())
# Winner: classifier (accuracy=0.83)

# Use the best model
result.best_model.predict(X_new)

# Compare specific techniques
results = qai.compare_techniques(X, y, techniques=["svm", "random_forest", "xgboost"])
for r in results:
    print(f"{r.technique:20s} accuracy={r.accuracy:.3f}")

# Get suggestions based on data size
suggestions = qai.suggest_technique(n_samples=150, n_features=4, task="classification")
# → Small data: RF, SVM, GB recommended
```

---

## 3. Data Loading

### Built-in datasets
```python
X, y = qai.load_iris()          # 150 samples, 4 features
X, y = qai.load_digits()        # 1797 samples, 64 features (8x8 images)
X, y = qai.load_wine()           # 178 samples, 13 features
X, y = qai.load_breast_cancer()  # 569 samples, 30 features
X, y = qai.load_diabetes()       # 442 samples, 10 features (regression)
```

### From files
```python
ds = qai.Dataset.from_csv("data.csv", target_column="label")
ds = qai.Dataset.from_json("data.json")
ds = qai.Dataset.from_parquet("data.parquet")
ds = qai.Dataset.from_excel("data.xlsx")
```

### From documents (PDF, Word, books → trainable text)
```python
ds = qai.Dataset.from_pdf("paper.pdf")       # PDF → text chunks
ds = qai.Dataset.from_docx("report.docx")    # Word → text chunks
ds = qai.Dataset.from_book("novel.epub")     # EPUB → text chunks
ds = qai.Dataset.from_text("notes.txt")       # TXT → text chunks
```

### Dataset operations
```python
ds = qai.Dataset.from_arrays(X, y, target_name="label")
ds.shuffle(seed=42)                   # shuffle in-place
train, test = ds.split(train=0.8, stratify_by="label")  # stratified split

for X_batch, y_batch in ds.batch(32):  # iterate batches
    model.train(X_batch, y_batch)

ds.map(lambda x: x * 2)  # transform features
print(ds.describe())      # statistics
```

### Data visualization
```python
qai.data_summary(X, y)           # text summary
qai.plot_distributions(X)         # histograms
qai.plot_class_balance(y)         # bar chart
qai.plot_correlation(X)           # heatmap
qai.plot_outliers(X)              # box plots
qai.data_dashboard(X, y, save_dir="./plots")  # all at once
```

---

## 4. Training

### The Universal Trainer
```python
trainer = qai.Trainer(
    model=m,
    train_data=X, train_targets=y,
    eval_data=X_val, eval_targets=y_val,
    optimizer="adamw", lr=3e-4,
    lr_schedule="cosine", warmup_steps=100,
    max_grad_norm=1.0, weight_decay=0.01,
    batch_size=32, grad_accum_steps=4,
    fp16=True,                        # mixed precision
    # Safety (on by default):
    auto_save=True, auto_resume=True,
    oom_recovery=True, nan_detection=True,
    interrupt_safe=True,
    # Convenience:
    validation_split=0.2,            # auto-split for validation
    early_stopping=True, patience=5,
    checkpoint_best=True,
    class_weight="balanced",
    progress_bar=True,
)
trainer.train(epochs=10)
```

### Pipeline templates (one-call workflows)
```python
result = qai.classification_pipeline(X, y)
result = qai.regression_pipeline(X, y)
result = qai.clustering_pipeline(X)
result = qai.nlp_pipeline(texts, labels)
result = qai.llm_pipeline("gpt2", train_texts=texts, method="lora")
result = qai.auto_pipeline(X, y)  # auto-detect task
```

---

## 5. Evaluation & Metrics

### Model evaluation
```python
m.evaluate(X, y, metrics=["accuracy", "mse", "mae", "r2"])
```

### Advanced metrics
```python
qai.perplexity(model, token_sequences)
qai.bleu(references, hypotheses)
qai.rouge(references, hypotheses)
qai.expected_calibration_error(y_true, y_proba)
qai.fid(real_images, fake_images)
qai.cohens_d(group1, group2)
```

### Standard benchmarks
```python
qai.benchmarks.run_gsm8k(model, n_samples=100)    # 8,500 real math problems
qai.benchmarks.run_humaneval(model, n_samples=50)   # 164 real code problems
qai.benchmarks.run_mmlu(model, n_samples=200)      # 14,042 real questions
qai.benchmarks.run_all(model)                       # run everything
```

---

## 6. Model Inspection

### Architecture summary
```python
print(m.summary())                          # full architecture
print(qai.count_parameters(m))              # param count
qai.model_to_device(m, "cuda:0")           # move to GPU
```

### Param introspection
```python
params = m.technique.get_params()          # all constructor params
m.technique.list_params()                   # with types + defaults
print(m.technique.param_help())             # human-readable
valid, errors = m.technique.validate_params(d_model=128)
```

### Explainability
```python
exp = m.explain(X[0])
# → {'prediction': 0, 'top_features': [...], 'confidence': 0.95, ...}

score = m.explainability_score()  # 0-100
translator = qai.WeightTranslator(m)
concepts = translator.translate()
# → {'x0': 'strong positive', 'x1': 'negligible', ...}
```

### Chainable output
```python
result = m.output()
result.format("json").redact("emails")
result.format("text").save("output.txt")
result.filter(["prediction"]).format("json")
```

### Health checks
```python
report = qai.health_check(m, X, y)
# → {'overall_status': 'healthy', 'checks': [...], ...}
```

---

## 7. Deep Learning

### Transformer (with modern features)
```python
m = qai.build(technique="transformer",
    vocab_size=1000, d_model=256, n_heads=8, n_layers=6,
    attention_dropout=0.1, residual_dropout=0.1,
    rope_theta=10000.0, layer_norm_eps=1e-5,
    tie_word_embeddings=True, initializer_range=0.02,
    norm_type="rmsnorm", activation="gelu",
    window_size=1024,  # sliding window attention
    attention_impl="auto",  # SDPA/FlashAttention
)
```

### CNN (with configurable architecture)
```python
m = qai.build(technique="cnn",
    in_channels=3, n_classes=10,
    conv_channels=[32, 64, 128], kernel_sizes=[3, 3, 3],
    norm_type="group_norm",     # batch_norm / group_norm / layer_norm / none
    activation="silu",           # relu / gelu / silu / mish / leaky_relu
    groups=1, dilation=1,       # grouped + dilated convolution
    se_blocks=True,              # squeeze-and-excitation
    stochastic_depth=0.1,       # drop blocks during training
)
```

### 258 techniques available
```python
# Classical ML
qai.build(technique="random_forest")
qai.build(technique="svm")
qai.build(technique="xgboost")

# Deep learning
qai.build(technique="transformer")
qai.build(technique="cnn")
qai.build(technique="rnn")

# Modern architectures
qai.build(technique="retnet")        # retention network
qai.build(technique="mamba")         # state-space model
qai.build(technique="vit")          # vision transformer
qai.build(technique="sam2")          # segment anything 2

# Reinforcement learning
qai.build(technique="dqn", double_dqn=True, use_per=True, soft_update=True)
qai.build(technique="ppo")
```

---

## 8. Fine-Tuning

### LoRA / QLoRA / Adapter / IA³
```python
m.train(X, y)  # SFT first

# LoRA (low-rank adaptation)
m.finetune(X_new, y_new, method="lora", rank=8, alpha=16)

# QLoRA (4-bit quantized LoRA)
m.finetune(X_new, y_new, method="qlora", rank=8, quantize="nf4")

# Adapter (bottleneck)
m.finetune(X_new, y_new, method="adapter", bottleneck=64)

# IA³ (per-channel scaling)
m.finetune(X_new, y_new, method="ia3")
```

---

## 9. Alignment

### DPO / KTO / SimPO / ORPO / RLVR
```python
m.train(corpus)  # SFT first

# DPO (Direct Preference Optimization)
dpo = qai.DPOTrainer(model=m, beta=0.1, lr=5e-7)
dpo.train(preference_data, epochs=1)

# KTO (doesn't need paired data — unary labels!)
kto = qai.KTOTrainer(model=m, beta=0.1)
kto.train(unary_data, epochs=1)

# SimPO (no reference model needed!)
simpo = qai.SimPOTrainer(model=m, beta=2.0, gamma=0.5)
simpo.train(preference_data, epochs=1)

# ORPO (combined SFT + alignment, no separate SFT needed)
orpo = qai.ORPOTrainer(model=m, lambda_=0.5)
orpo.train(preference_data, epochs=1)

# RLVR (verifiable rewards — for math/code)
rlvr = qai.RLVRTrainer(model=m, reward_fn=math_verifier)
rlvr.train(math_problems, epochs=10)

# Auto-select
qai.align(model=m, data=data, method="dpo")
```

---

## 10. Serving

### Production server
```python
server = qai.ProductionServer(
    model=m, port=8000,
    max_batch_size=8,        # dynamic batching
    max_concurrent=64,
    enable_streaming=True,    # SSE for generation
)
server.start()  # blocks

# Endpoints: /predict /predict_batch /generate /health /metrics
```

### Fast inference engine (vLLM-style)
```python
engine = qai.FastInferenceEngine(model=m, max_batch_size=64)
engine.start()
future = engine.generate(prompt, max_new_tokens=100)
tokens = future.result()
engine.stop()
```

---

## 11. Quantization & Deployment

### Quantization
```python
qai.quantize_dynamic(m)              # int8 (no retraining)
qai.quantize_to_int4(m, "nf4")       # int4 (NF4)
qai.dequantize(m)                     # reverse (lossy!)

# QAT (minimize quality loss before quantizing)
qat = qai.QATTrainer(model=m, quantization="int8")
qat.train(X, y, epochs=3)
qai.quantize_dynamic(m)  # now minimal quality loss

# Export
m.save("model.gguf", format="gguf")   # for llama.cpp/Ollama
m.save("model.onnx", format="onnx")   # for ONNX runtime
```

### Precision management
```python
pm = qai.PrecisionManager()
print(pm.recommend_precision())  # bf16 on Ampere+, fp16 on Volta+, fp32 on CPU

qai.set_precision(m, "bf16")
qai.compare_precision(m, X[0])  # compare fp32 vs fp16 vs bf16
```

### Deployment
```python
m.deploy(target="server")  # full precision, production server
m.deploy(target="edge")   # int8 quantized, saved to file
m.deploy(target="mobile")  # int4 quantized, smallest
```

---

## 12. MLOps Pipeline

```python
pipeline = qai.MLOpsPipeline(
    name="my-model",
    registry_dir="./registry",
    tracking_dir="./experiments",
)

# 1. Train
run = pipeline.train(model=m, train_data=X, train_targets=y, epochs=10)

# 2. Register
pipeline.register(run, description="v1")

# 3. Quantize
pipeline.quantize(method="int8")

# 4. Serve
pipeline.serve()

# 5. Monitor
report = pipeline.monitor(X_prod, y_prod)

# Experiment tracking
qai.log_experiment(name="v1", params={...}, metrics={...})
qai.compare_experiments(metric="accuracy")
```

---

## 13. HuggingFace Integration

```python
# Browse models
models = qai.browse_models("text-generation")
results = qai.search_models("bert")

# Load any HF model into qai
m = qai.from_huggingface("bert-base-uncased")
m.predict("Hello world")

# Load from zoo (curated aliases)
m = qai.load_from_zoo("gpt2")
m = qai.load_from_zoo("bert")
m = qai.load_from_zoo("whisper-tiny")

# Push your model to HF Hub
qai.to_huggingface(m, "my-username/my-model")

# Stream datasets (train on data larger than RAM)
for batch in qai.stream_dataset("imdb", split="train"):
    model.train_batch(batch["text"], batch["label"])
```

---

## 14. Agents & Tools

```python
# Define tools
@qai.Tool
def calculator(expression: str) -> str:
    '''Evaluate a math expression.'''
    return str(eval(expression))

# ReAct agent (reasoning + acting)
agent = qai.ReActAgent(model=my_llm, tools=[calculator])
result = agent.run("What is 25 * 17 + 100?")

# Multi-agent
multi = qai.MultiAgent(agents={
    "math": qai.ReActAgent(model=m, tools=[calculator]),
    "writer": qai.Agent(model=m),
})

# Test-time reasoning
answer = qai.ChainOfThought(model=m).solve("What is 15% of 240?")
answer = qai.SelfConsistency(model=m, n_samples=5).solve(problem)
answer = qai.TreeOfThoughts(model=m, n_branches=3).solve(problem)
```

---

## 15. Custom Plugins

```python
@qai.plugin
class MyModel(qai.Technique):
    name = "my_model"
    family = "custom"

    def fit(self, X, y):
        # your training code
        self._trained = True
        return self

    def forward(self, x):
        # your prediction code
        return prediction

# Use it like any built-in
m = qai.build(technique="my_model")
m.train(X, y)
m.predict(X[0])

# List plugins
print(qai.list_plugins())
```

---

## 16. Safety & Crash Recovery

### Safety is built into the Trainer (on by default)
```python
trainer = qai.Trainer(
    model=m, ...,
    auto_save=True,        # saves every N steps
    auto_resume=True,      # resumes from last checkpoint after crash
    oom_recovery=True,     # reduces batch_size if GPU OOM
    nan_detection=True,     # detects and reverts on NaN loss
    interrupt_safe=True,   # saves on Ctrl+C
)
```

### Preflight check
```python
report = qai.preflight_check(model=m, X=X, y=y, batch_size=32)
print(f"Safe to train: {report['safe']}")
print(f"Warnings: {report['warnings']}")
```

### Clean data
```python
X_clean, y_clean = qai.clean_dataset(X_dirty, strategy="mean")
```

### Hardware detection
```python
print(qai.hardware_report())
print(qai.detect_platform())  # "kaggle" / "colab" / "jupyter" / "local"
print(qai.auto_select_device(model))
```

---

## Quick Reference

| What | How |
|------|-----|
| Build a model | `qai.build(technique="classifier")` |
| Train | `m.train(X, y)` |
| Predict | `m.predict(x)` |
| Probabilities | `m.predict_proba(x)` |
| Evaluate | `m.evaluate(X, y)` |
| Save | `m.save("model.qmodel")` |
| Load | `qai.Model.load("model.qmodel")` |
| Fine-tune | `m.finetune(X, y, method="lora")` |
| Quantize | `qai.quantize_dynamic(m)` |
| Serve | `qai.ProductionServer(model=m).start()` |
| Visualize | `qai.visualize_architecture(m)` |
| Auto-ML | `qai.auto_train(X, y)` |
| Chat | `qai.chat_with_model(m)` |
| Health check | `qai.health_check(m, X, y)` |
| HuggingFace | `qai.from_huggingface("bert-base-uncased")` |
| Plugin | `@qai.plugin` |
| Doctor | `qai.doctor()` |

---

*For the full API reference, see [USER_MANUAL.md](USER_MANUAL.md) (~5,000 lines). For implementation status by spec section, see [UNBUILT_FEATURES.md](UNBUILT_FEATURES.md).*
