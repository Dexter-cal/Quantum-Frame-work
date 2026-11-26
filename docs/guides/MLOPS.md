# Guide: MLOps — Tracking, Registry, Monitoring

*From "it works in my notebook" to "I know what's running and why".*

> Back to [Guides index](README.md) · Deploy: [SERVE_DEPLOY.md](SERVE_DEPLOY.md) · Monitor drift: [SECURITY_GOVERNANCE.md](SECURITY_GOVERNANCE.md)

---

## What you'll learn

1. One-call MLOps pipeline (train → track → register → quantize → serve → monitor)
2. Experiment tracking (JSON, MLflow, W&B)
3. Model registry with versions and stages
4. Production monitoring and experiment dashboards

---

## 1. The one-call pipeline

```python
pipeline = qai.MLOpsPipeline(
    name="my-model",
    registry_dir="./registry",
    tracking_dir="./experiments",
)

run = pipeline.train(model=m, train_data=X, train_targets=y, epochs=10)  # 1. train
pipeline.register(run, description="baseline v1")                        # 2. register
pipeline.quantize(method="int8")                                         # 3. quantize
pipeline.serve()                                                          # 4. serve
report = pipeline.monitor(X_prod, y_prod)                                 # 5. monitor
```

Each stage persists its state to `registry_dir` / `tracking_dir`.

## 2. Experiment tracking

```python
# Log manually
qai.log_experiment(
    name="lr-sweep-3e-4",
    params={"lr": 3e-4, "batch_size": 32, "technique": "transformer"},
    metrics={"accuracy": 0.91, "loss": 0.32},
)

# Compare runs
qai.compare_experiments(metric="accuracy")

# Backends: JSON (default), MLflow, Weights & Biases
qai.configure_tracking(backend="mlflow", uri="http://localhost:5000")
```

## 3. Model registry

```python
from qai.core.model_registry import ModelRegistry

reg = ModelRegistry("./registry")
reg.register(m, name="fraud-detector", version="1.0.0", stage="staging")
reg.promote("fraud-detector", "1.0.0", to_stage="production")
best = reg.load(name="fraud-detector", stage="production")
print(reg.history("fraud-detector"))     # every version, stage, and timestamp
```

Runtime-level versioning (A/B tests, auto-rollback, signing) is a layer on
top of this — see [SERVE_DEPLOY.md §4](SERVE_DEPLOY.md).

## 4. Dashboards & monitoring

```python
# Experiment dashboard: leaderboard, comparison, trend, export
from qai.core.experiment_tracking import ExperimentDashboard

dash = ExperimentDashboard()
dash.log(run_id, metrics)
print(dash.leaderboard(metric="accuracy"))
print(dash.trend("accuracy"))            # over time
dash.export_csv("runs.csv")              # or export_json

# Production report
report = pipeline.monitor(X_prod, y_prod)
# accuracy trend, drift signals, latency, error rate
```

## 5. Automated scientist (research automation)

```python
from qai.core import AutomatedScientist

scientist = AutomatedScientist()
hypotheses = scientist.generate_hypotheses(X, y)
results = scientist.test(hypotheses, X, y)
print(results.conclusions())
```

---

## Cheat sheet

| Task | Call |
|---|---|
| Whole lifecycle | `qai.MLOpsPipeline(name=...)` |
| Log an experiment | `qai.log_experiment(name, params, metrics)` |
| Compare runs | `qai.compare_experiments(metric=...)` |
| Use MLflow/W&B | `qai.configure_tracking(backend=...)` |
| Register a model | `ModelRegistry.register(m, name, version)` |
| Promote to prod | `reg.promote(name, version, to_stage="production")` |
| Leaderboard | `ExperimentDashboard().leaderboard(metric)` |
| Export runs | `dash.export_csv(path)` |
| Monitor live model | `pipeline.monitor(X_prod, y_prod)` |
| A/B + rollback | see [SERVE_DEPLOY.md](SERVE_DEPLOY.md) |

---

## Common gotchas

- **Log params AND metrics.** Metrics without params are not reproducible.
- **Registry dirs are stateful.** Commit them to object storage, not git — binaries are gitignored here on purpose.
- **Monitor against training data**, not just yesterday's traffic — drift is measured as a delta.
- **Promote, don't overwrite.** Registry versions are immutable by design; a "fix" is a new version.

---

*Next: [VISUAL_WORKSPACE.md](VISUAL_WORKSPACE.md) to watch models think, or back to the [Guides index](README.md).*
