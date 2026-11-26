# Guide: Training — Build, Train, Predict

*You have data. Now make a model. This chapter covers every way to train in qai.*

> Back to [Guides index](README.md) · Prev: [DATA.md](DATA.md) · Model choice: [MODEL_CATALOG.md](MODEL_CATALOG.md) · Full API: [User Manual](../USER_MANUAL.md)

---

## What you'll learn

1. The four-beat pattern (build → train → predict → persist)
2. The Universal Trainer (full control: schedulers, clipping, mixed precision, safety)
3. One-call pipeline templates
4. Auto-ML (let qai pick the model)
5. Training safety (crash recovery, OOM handling, NaN detection)

---

## 1. The four beats

```python
import qai

model = qai.build(technique="classifier")
model.train(X, y)

print(model.predict(X[0]))            # → 0
print(model.predict_proba(X[0]))      # → [[0.98, 0.02, 0.0]]
print(model.predict_top_k(X[0], k=2)) # → [(0, 0.98), (1, 0.02)]

model.save("model.qmodel")
same_model = qai.Model.load("model.qmodel")
```

That's all 229 techniques. Pick a different technique name from
[MODEL_CATALOG.md](MODEL_CATALOG.md) and the beats are identical.

---

## 2. The Universal Trainer (full control)

When `model.train(X, y)` isn't enough — LR schedules, grad accumulation, mixed precision, EMA, SWA:

```python
trainer = qai.Trainer(
    model=model,
    train_data=X, train_targets=y,
    eval_data=X_val, eval_targets=y_val,
    optimizer="adamw", lr=3e-4,
    lr_schedule="cosine", warmup_steps=100,
    max_grad_norm=1.0, weight_decay=0.01,
    batch_size=32, grad_accum_steps=4,
    fp16=True,                        # mixed precision
    validation_split=0.2,             # auto-split if no eval data given
    early_stopping=True, patience=5,
    checkpoint_best=True,
    class_weight="balanced",
    progress_bar=True,
)
trainer.train(epochs=10)
```

Extras available: `ema=True` (exponential moving average weights), `swa=True`
(stochastic weight averaging), `gradient_checkpointing=True` (trade compute
for memory on big models), `auto_device=True` + `auto_precision=True`
(auto-select CUDA/CPU and bf16/fp16/fp32).

### Training-only utilities (wrap ANY torch-backed technique)

```python
qai.SAM(model)               # Sharpness-Aware Minimization
qai.SWA(model)               # Stochastic Weight Averaging
qai.EMA(model)               # Exponential Moving Average
qai.Lookahead(optimizer)     # Lookahead wrapper
qai.MixupTrainer(model)      # mixup augmentation
qai.LabelSmoother(...)       # label smoothing
qai.CosineAnnealer(...)      # cosine LR annealing
qai.OneCyclePolicy(...)      # one-cycle LR
qai.TTA(model)               # test-time augmentation
qai.PseudoLabeler(model)     # pseudo-labeling for semi-supervised
qai.SnapshotEnsemble(model)  # snapshot ensembling
```

All eleven are exposed at the top level and work on any existing technique —
see [FINDINGS.md §36](../FINDINGS.md) for why they're helpers, not new model types.

---

## 3. One-call pipeline templates

Don't want to pick anything? These run a sensible default workflow:

```python
result = qai.classification_pipeline(X, y)
result = qai.regression_pipeline(X, y)
result = qai.clustering_pipeline(X)
result = qai.nlp_pipeline(texts, labels)
result = qai.llm_pipeline("gpt2", train_texts=texts, method="lora")
result = qai.time_series_pipeline(dates, values)
result = qai.auto_pipeline(X, y)          # auto-detects the task
```

---

## 4. Auto-ML (let qai pick the best model)

```python
# Try multiple techniques, keep the winner
result = qai.auto_train(X, y, task="classification", time_budget_s=30)
print(result.summary())                 # "Winner: classifier (accuracy=0.83)"
best = result.best_model
best.predict(X_new)

# Compare specific techniques head-to-head
results = qai.compare_techniques(X, y, techniques=["svm", "random_forest", "xgboost"])
for r in results:
    print(f"{r.technique:20s} accuracy={r.accuracy:.3f}")

# Get recommendations from data shape alone
suggestions = qai.suggest_technique(n_samples=150, n_features=4, task="classification")
# → small data: RF, SVM, GB recommended
```

Tuning further: `qai.AutoTuner` runs grid/random search with cross-validation —
see [User Manual §13](../USER_MANUAL.md).

---

## 5. Train until a target metric is hit

```python
# Stop as soon as validation accuracy reaches 0.95 (or patience expires)
qai.AutoTrainTarget(model, target_metric="accuracy", target_value=0.95, patience=5)
model.train(X, y)
```

---

## 6. Training safety (all ON by default)

```python
trainer = qai.Trainer(
    model=model, ...,
    auto_save=True,        # periodic checkpoints
    auto_resume=True,      # resume from last checkpoint after a crash
    oom_recovery=True,     # halve batch size and retry on CUDA OOM
    nan_detection=True,    # detect NaN loss, roll back to last good weights
    interrupt_safe=True,   # save cleanly on Ctrl+C
)
```

Before a long run, preflight it:

```python
report = qai.preflight_check(model=model, X=X, y=y, batch_size=32)
print(f"Safe to train: {report['safe']}")
print(f"Warnings: {report['warnings']}")
```

After training, check model health:

```python
report = qai.health_check(model, X, y)
# → {'overall_status': 'healthy', 'checks': [...]}
```

---

## 7. Reinforcement learning (different loop, same framework)

```python
from qai.core.environment import GridWorld

env = GridWorld(size=5)
agent = qai.build(technique="dqn")            # or "ppo", "sarsa", "a2c", ...
agent.train_env(env, episodes=500)
```

DQN ships with experience replay, target networks, priority replay (PER),
epsilon decay, n-step returns, and distributional heads (C51, QR-DQN) —
see [MODEL_CATALOG.md](MODEL_CATALOG.md) → Reinforcement Learning.

---

## Cheat sheet

| Task | Call |
|---|---|
| Quick train | `model.train(X, y)` |
| Full control | `qai.Trainer(model=..., optimizer=..., lr_schedule=...)` |
| One-call workflow | `qai.classification_pipeline(X, y)` etc. |
| Auto-ML | `qai.auto_train(X, y, time_budget_s=30)` |
| Compare models | `qai.compare_techniques(X, y, techniques=[...])` |
| Model suggestion | `qai.suggest_technique(n_samples, n_features, task)` |
| Train to target | `qai.AutoTrainTarget(...)` |
| Preflight | `qai.preflight_check(model, X, y, batch_size)` |
| Health check | `qai.health_check(model, X, y)` |
| Hardware report | `qai.hardware_report()` |
| Resume after crash | just rerun — `auto_resume` picks up the checkpoint |
| Pick a technique | [MODEL_CATALOG.md](MODEL_CATALOG.md) |

---

## Common gotchas

- **Small data?** Don't start with a transformer. `suggest_technique()` exists for a reason — with 150 samples, random forests and SVMs win.
- **NaN loss mid-run?** `nan_detection=True` (default) rolls you back; check learning rate first.
- **CUDA OOM?** `oom_recovery=True` (default) handles it; or set `gradient_checkpointing=True` and lower `batch_size`.
- **Which technique?** That's [MODEL_CATALOG.md](MODEL_CATALOG.md).

---

*Next: [MODEL_CATALOG.md](MODEL_CATALOG.md) — every technique explained, one family at a time. Or [EVALUATION.md](EVALUATION.md) to score what you trained.*
