# Guide: Evaluation — Metrics, Explainability, Benchmarks

*You trained a model. Is it good? Why does it predict what it predicts? This chapter answers both.*

> Back to [Guides index](README.md) · Prev: [TRAINING.md](TRAINING.md) · Full metric list: [User Manual §11](../USER_MANUAL.md)

---

## What you'll learn

1. Score models with the built-in metric catalog
2. Explain predictions (feature importance, SHAP, weight-to-words)
3. Calibrate overconfident models
4. Run the offline benchmark suite
5. Visualize anything

---

## 1. Score a model

```python
# On any model:
m.evaluate(X, y, metrics=["accuracy", "mse", "mae", "r2"])
```

### The metric catalog

**Classification:** `classification_report`, `confusion_matrix`, `roc_auc_score`,
`log_loss`, `cohen_kappa_score`, `matthews_corrcoef`, `balanced_accuracy_score`,
`precision_recall_curve`, `roc_curve`, `det_curve`, `zero_one_loss`,
`hamming_loss`, `jaccard_score`, `multilabel_confusion_matrix`, `brier_score`,
`expected_calibration_error`

**Regression:** `mean_absolute_error`, `mean_squared_log_error`,
`mean_absolute_percentage_error`, `median_absolute_error`,
`explained_variance_score`, `r2_score`, `max_error`

**Clustering:** `silhouette_score`, `davies_bouldin_score`,
`adjusted_rand_score`, `normalized_mutual_info_score`

**Language / generative:** `qai.perplexity(model, tokens)`, `qai.bleu(refs, hyps)`,
`qai.rouge(refs, hyps)`, `qai.fid(real, fake)` (images)

**Distances & losses (qai.math):** `kl_divergence`, `js_divergence`,
`huber_loss`, `focal_loss`, `triplet_loss`, `contrastive_loss`,
`cosine_embedding_loss`, `chebyshev_distance`, `canberra_distance`,
`braycurtis_distance`, `levenshtein_distance`

Everything is importable from `qai` directly:

```python
print(qai.confusion_matrix(y_true, y_pred))
print(qai.roc_auc_score(y_true, y_proba))
print(qai.expected_calibration_error(y_true, y_proba))
```

---

## 2. Explain a prediction

```python
# Per-prediction explanation (technique-aware)
exp = m.explain(X[0])
# → {'prediction': 0, 'top_features': [...], 'confidence': 0.95, ...}

# Global feature importance
qai.feature_importance(m)             # coefficients / tree importances
qai.permutation_importance(m, X, y)   # performance drop per shuffled feature
qai.partial_dependence(m, X, feature=0)
qai.shapley_value_approximation(m, x) # marginal-contribution values
```

### Plain-English weights

```python
from qai.core.explainability import WeightToWords

translator = qai.WeightTranslator(m)
concepts = translator.translate()
# → {'x0': 'strong positive', 'x1': 'negligible', ...}
```

### Deeper interpretability toolkit

`qai.NovelInterpretability` provides decision lineage (which path did this
input take), sensitivity analysis, and counterfactuals ("change x0 by 0.3 and
the prediction flips"). `qai.AdvancedVisualizer` renders weight heatmaps and
formula diffs.

### Explainability score

```python
print(m.explainability_score())   # 0–100, how interpretable this model is
```

---

## 3. Calibration (fix overconfidence)

```python
# Is the model actually 94% right when it says 94%?
print(qai.expected_calibration_error(y_true, y_proba))
print(qai.brier_score(y_true, y_proba))

# Fix it:
m = qai.build(technique="random_forest", calibrated=True, calibration_method="isotonic")
# or post-hoc:
m = qai.build(technique="probability_calibration")  # registered as a technique too
```

---

## 4. Benchmark suite (offline, no internet needed)

```python
from qai.benchmark_suite import BenchmarkSuite

suite = BenchmarkSuite()
results = suite.run(m)      # tolerant model interface: .chat() / .predict() /
                            # .generate() / __call__ all work
print(results.summary())
```

Included benchmarks (synthetic-but-realistic subsets embedded in-module):

| Benchmark | Items | What it measures |
|---|---|---|
| MMLU | 30 questions, 6 subjects | broad knowledge |
| HumanEval | 15 Python problems, executable tests | code generation |
| GSM8K | 20 math word problems | reasoning |
| GLUE | 30 items (SST-2, MRPC, QQP) | language understanding |
| CommonsenseQA | 20 MCQs | commonsense |
| ARC | 20 science MCQs | science reasoning |
| MT-Bench | simplified judge | chat quality |

```python
qai.benchmarks.run_mmlu(model, n_samples=200)
qai.benchmarks.run_humaneval(model, n_samples=50)
qai.benchmarks.run_gsm8k(model, n_samples=100)
qai.benchmarks.run_all(model)
```

**Scale note (honest):** these are embedded subsets — MMLU ships 30 questions,
not the full 14,042. They're for relative comparison between *your* models,
not for leaderboard claims. Full-dataset download is on the
[ROADMAP](../ROADMAP.md) as a future item.

### Leaderboards

```python
from qai.benchmark_suite import Leaderboard
board = Leaderboard()
board.add(model_a, results_a)
board.add(model_b, results_b)
print(board.to_markdown())
```

---

## 5. Health & robustness checks

```python
# Model health
report = qai.health_check(m, X, y)         # dead neurons, saturation, etc.

# Stress testing
from qai.core.benchmark import StressTester
stress = StressTester(m)
print(stress.run(X).robustness_score())    # noise, edge cases, sensitivity

# Bias audit (fairness across groups)
from qai.governance import fairness_audit
fairness_audit(m, X, y, sensitive_features=["gender", "region"])
# → demographic_parity_difference, equalized_odds_difference, disparate_impact_ratio
```

Fairness in depth: [SECURITY_GOVERNANCE.md](SECURITY_GOVERNANCE.md).

---

## 6. Visualize

```python
qai.visualize_architecture(m)      # architecture diagram
qai.plot_distributions(X)          # data checks
qai.model_visualizer(m)            # weights, activations
qai.workspace(m)                   # full interactive microscope —
                                   # see VISUAL_WORKSPACE.md
```

---

## Cheat sheet

| Task | Call |
|---|---|
| Score | `m.evaluate(X, y, metrics=[...])` |
| Explain one input | `m.explain(x)` |
| Global importance | `qai.feature_importance(m)` |
| SHAP values | `qai.shapley_value_approximation(m, x)` |
| Plain English | `qai.WeightTranslator(m).translate()` |
| Calibration ECE | `qai.expected_calibration_error(y, p)` |
| Calibrate model | `calibrated=True` at build time |
| Benchmarks | `qai.benchmarks.run_all(m)` |
| Leaderboard | `Leaderboard().to_markdown()` |
| Stress test | `StressTester(m).run(X)` |
| Fairness | `qai.governance.fairness_audit(...)` |
| Interactive | `qai.workspace(m)` |

---

*Next: [FINETUNING_ALIGNMENT.md](FINETUNING_ALIGNMENT.md) to improve the model, or [SERVE_DEPLOY.md](SERVE_DEPLOY.md) to ship it.*
