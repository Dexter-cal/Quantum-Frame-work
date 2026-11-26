# Guide: Troubleshooting — Setup, Errors, Health

*Something broke. Start here before guessing.*

> Back to [Guides index](README.md) · Related: [DATA.md](DATA.md) · [TRAINING.md](TRAINING.md) · [SERVE_DEPLOY.md](SERVE_DEPLOY.md)

---

## 1. Check your installation first

```bash
qai doctor
```

Reports: what's installed, what's missing, what's recommended, which feature
bundles would fix it. Install bundles selectively:

```bash
qai install --vision      # Pillow, opencv, torch vision
qai install --nlp         # tokenizers, transformers
qai install --serving     # Flask, gunicorn
qai install --all
```

Version check:

```bash
python -c "import qai; print(qai.__version__)"
```

If it's stale after an upgrade: `pip install -r requirements.txt --upgrade`.

## 2. Hardware & platform

```python
print(qai.hardware_report())        # CPU/RAM/GPU inventory
print(qai.get_hardware_info())
print(qai.get_optimal_device())     # cuda / mps / cpu
print(qai.detect_platform())        # kaggle / colab / jupyter / local
print(qai.auto_select_device(model))
```

## 3. Model health

```python
report = qai.health_check(model, X, y)
# → {'overall_status': 'healthy', 'checks': [dead neurons, saturation, ...]}

report = qai.preflight_check(model=model, X=X, y=y, batch_size=32)
# → {'safe': True, 'warnings': [...]}   run BEFORE long trainings
```

## 4. Training problems

| Symptom | Cause | Fix |
|---|---|---|
| NaN loss after a few steps | LR too high or bad batch | lower LR; `nan_detection=True` (default) already rolled back to last good weights |
| CUDA out of memory | batch too big for VRAM | `oom_recovery=True` (default) halves batch automatically; or `gradient_checkpointing=True` |
| Training crashed mid-run | anything | rerun — `auto_resume=True` (default) picks up the last checkpoint |
| Ctrl+C lost progress | — | `interrupt_safe=True` (default) saved on interrupt |
| Loss plateaus | LR schedule | `lr_schedule="cosine"` + `warmup_steps=100` |
| Overfitting | model memorizing | `early_stopping=True, patience=5`, `weight_decay=0.01`, mixup (`mixup_alpha=0.2`) |

All of these are `qai.Trainer` kwargs — see [TRAINING.md §2](TRAINING.md).

## 5. Import & dependency errors

| Error | Meaning | Fix |
|---|---|---|
| `ImportError: cannot import name 'X' from 'qai'` | missing optional dep or stale install | `pip install -r requirements.txt --upgrade`; `qai doctor` shows which bundle |
| `you tried to use transformer but torch isn't installed` | feature needs its dependency | `qai install --all` or the relevant bundle |
| `ModuleNotFoundError: pyarrow` | Parquet export needs it | `pip install pyarrow` |
| `Unknown technique 'foo'` | typo in technique name | the error prints the full list of valid names; also see [MODEL_CATALOG.md](MODEL_CATALOG.md) |

qai **degrades gracefully** — missing optional deps never crash the import;
you get a feature-level warning when you touch the affected area.

## 6. Data problems

| Symptom | Fix |
|---|---|
| Schema validation rejected my dict | keys must match schema columns; see `Schema.validate_row()` ([FINDINGS.md §5](../FINDINGS.md)) |
| MultinomialNB wants X >= 0 | count features must be non-negative — scale or switch to `gaussian_nb` |
| Shapes mismatch in predict | single inputs return unboxed scalars; batch inputs return arrays — check `predict` vs `predict_proba` |
| Weird clusters in KMeans | standardize features first: `qai.preprocessing.StandardScaler` |

## 7. Serving problems

| Symptom | Fix |
|---|---|
| `/health` returns 503 | model failed to load — check the server logs, verify bundle signature if `verify=True` |
| Tamper error on load | `.qmodel` was modified after signing — re-save with `sign=True` |
| A/B test shows no difference | split too small (use ≥20%) or metrics too noisy; more traffic or a better metric |
| Stale version serving | check `gw.route(name, version=...)` pinning in the Gateway |

## 8. MCP problems

| Symptom | Fix |
|---|---|
| Tool denied | `permissions` config blocks it — see [AGENTS_MCP.md](AGENTS_MCP.md) |
| Rate limit hit | `rate_limits` config; wait or raise the cap |
| Client can't connect | use stdio transport for IDE clients; HTTP transport is for testing |
| Args validation error | JSON-schema required fields missing — check the tool's schema |

## 9. The nuclear option

```python
# Reset checkpoints for a stuck trainer
import shutil; shutil.rmtree("./checkpoints", ignore_errors=True)

# Full reinstall
# pip uninstall qai && pip install -r requirements.txt && qai doctor
```

---

## Cheat sheet

| Task | Call |
|---|---|
| Diagnose install | `qai doctor` |
| Hardware report | `qai.hardware_report()` |
| Platform detect | `qai.detect_platform()` |
| Check model | `qai.health_check(m, X, y)` |
| Pre-flight training | `qai.preflight_check(model, X, y, batch_size)` |
| Clean dirty data | `qai.clean_dataset(X_dirty, strategy="mean")` |
| Install everything | `qai install --all` |

---

*Back to the [Guides index](README.md).*
