# `qai.ab_testing`

> qai.ab_testing -- production A/B testing with statistical significance.

**Module summary:**
- Classes: 2 | Functions: 0 | Constants: 0
- Source: `qai/ab_testing.py`

## Table of Contents

**Classes:**
- [`ABTestResult`](#abtestresult)
- [`ABTest`](#abtest)

## Classes

<a id="abtestresult"></a>
### `ABTestResult`  
`CLASS` | `qai.ab_testing.ABTestResult`

**Signature:**
```python
class ABTestResult
```

> Result of an A/B test analysis.

Result of an A/B test analysis.

_Defined in `qai/ab_testing.py:29`._

---

<a id="abtest"></a>
### `ABTest`  
`CLASS` | `qai.ab_testing.ABTest`

**Signature:**
```python
class ABTest
```

> Production A/B testing router.

Production A/B testing router.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model_a` | `` | `` | the control model. |
| `model_b` | `` | `` | the treatment model. |
| `traffic_split` | `float` | `0.5` | fraction of traffic to send to model_a (default 0.5). |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(model_a, model_b, traffic_split: float = 0.5)` |  |
| `predict()` | `(x)` | Route x to one of the models and return the prediction. |
| `record_outcome()` | `(x, actual_label)` | Record whether the prediction for x was correct. |
| `analyze()` | `(alpha: float = 0.05)` | Analyze the collected results for statistical significance. |
| `reset()` | `()` | Clear all collected results. |

_Defined in `qai/ab_testing.py:54`._

---
