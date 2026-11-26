# `qai.active_learning`

> qai.active_learning -- interactive labeling loop for any classifier.

**Module summary:**
- Classes: 1 | Functions: 0 | Constants: 0
- Source: `qai/active_learning.py`

## Table of Contents

**Classes:**
- [`ActiveLearningLoop`](#activelearningloop)

## Classes

<a id="activelearningloop"></a>
### `ActiveLearningLoop`  
`CLASS` | `qai.active_learning.ActiveLearningLoop`

**Signature:**
```python
class ActiveLearningLoop
```

> An active-learning loop that picks samples for the human to label.

An active-learning loop that picks samples for the human to label.

**Parameters:**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `` | `` | a trained qai classifier (must support predict_proba or
predict with discrete outputs). |
| `X_unlabeled` | `` | `` | the pool of unlabeled samples (numpy array). |
| `strategy` | `str` | `'uncertainty'` | "uncertainty" | "margin" | "entropy" | "random". |
| `batch_size` | `int` | `10` | how many samples to surface per next_batch() call. |
| `initial_labeled_indices` | `Optional[set]` | `None` | optional set of indices already labeled
(they'll be skipped). |

**Methods:**

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `constructor __init__()` | `(model, X_unlabeled, strategy: str = 'uncertainty', batch_size: int = 10, initial_labeled_indices: Optional[set] = None)` |  |
| `has_more()` | `()` | Are there still unlabeled samples to label? |
| `next_batch()` | `()` | Return the indices of the next `batch_size` samples to label. |
| `commit()` | `(indices: List[int], labels: List[Any])` | Commit labeled samples: add them to the training set, retrain the model. |
| `evaluate()` | `(X_test, y_test)` | Evaluate the current model on a test set. Returns accuracy. |
| `status()` | `()` |  |

_Defined in `qai/active_learning.py:42`._

---
