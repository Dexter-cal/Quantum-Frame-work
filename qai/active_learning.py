"""
qai.active_learning -- interactive labeling loop for any classifier.

Active learning: instead of labeling a random sample of your data, let
the MODEL pick which samples it's most uncertain about. Label those,
retrain, repeat. Reaches high accuracy with far fewer labeled examples.

Strategies:
  - "uncertainty"   : pick samples where the model's max class probability
                       is lowest (model is least confident)
  - "margin"        : pick samples where the gap between the top-2 class
                       probabilities is smallest (model is torn between two)
  - "entropy"       : pick samples with the highest entropy across class
                       probabilities (model is generally confused)
  - "random"        : pick randomly (baseline for comparison)

Usage:
    import qai
    m = qai.build(technique="classifier")
    m.train(X_seed, y_seed)  # small initial labeled set

    loop = qai.ActiveLearningLoop(
        model=m,
        X_unlabeled=X_pool,
        strategy="uncertainty",
        batch_size=10,
    )

    while loop.has_more():
        batch_idx = loop.next_batch()  # indices into X_pool
        # ... human labels X_pool[batch_idx] ...
        new_y = label_these(X_pool[batch_idx])
        loop.commit(batch_idx, new_y)  # adds to training set + retrains
        print(f"accuracy now: {loop.evaluate(X_test, y_test):.3f}")
"""
from __future__ import annotations
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
import qai


class ActiveLearningLoop:
    """An active-learning loop that picks samples for the human to label.

    Args:
        model: a trained qai classifier (must support predict_proba or
               predict with discrete outputs).
        X_unlabeled: the pool of unlabeled samples (numpy array).
        strategy: "uncertainty" | "margin" | "entropy" | "random".
        batch_size: how many samples to surface per next_batch() call.
        initial_labeled_indices: optional set of indices already labeled
                                  (they'll be skipped).
    """
    STRATEGIES = ("uncertainty", "margin", "entropy", "random")

    def __init__(self, model, X_unlabeled, strategy: str = "uncertainty",
                 batch_size: int = 10,
                 initial_labeled_indices: Optional[set] = None):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"[WHAT] Unknown active-learning strategy {strategy!r}\n"
                f"[WHY] Strategy must be one of {self.STRATEGIES}\n"
                f"[FIX] Use one of the supported strategies"
            )
        self.model = model
        self.X_unlabeled = np.asarray(X_unlabeled)
        self.strategy = strategy
        self.batch_size = batch_size
        self.labeled_indices: set = set(initial_labeled_indices or [])
        # Internal training set (grows as the user commits labels)
        self.X_train: List[np.ndarray] = []
        self.y_train: List[Any] = []
        # Cache class probabilities for the unlabeled pool
        self._proba_cache: Optional[np.ndarray] = None

    def has_more(self) -> bool:
        """Are there still unlabeled samples to label?"""
        return len(self.labeled_indices) < len(self.X_unlabeled)

    def next_batch(self) -> List[int]:
        """Return the indices of the next `batch_size` samples to label.

        The indices are into the original X_unlabeled array.
        """
        if not self.has_more():
            return []
        # Get all unlabeled indices
        all_idx = set(range(len(self.X_unlabeled))) - self.labeled_indices
        all_idx_list = sorted(all_idx)
        if self.strategy == "random":
            # Random sample
            n = min(self.batch_size, len(all_idx_list))
            chosen = list(np.random.choice(all_idx_list, size=n, replace=False))
        else:
            # Compute uncertainties
            scores = self._compute_scores(all_idx_list)
            # Pick the top batch_size by uncertainty (highest score)
            n = min(self.batch_size, len(all_idx_list))
            # argsort descending, take top n
            top_indices = np.argsort(scores)[::-1][:n]
            chosen = [all_idx_list[i] for i in top_indices]
        return chosen

    def commit(self, indices: List[int], labels: List[Any]):
        """Commit labeled samples: add them to the training set, retrain the model.

        Args:
            indices: indices into X_unlabeled that were just labeled.
            labels: the labels for those indices (same length).
        """
        if len(indices) != len(labels):
            raise ValueError(
                f"[WHAT] indices and labels must have the same length\n"
                f"[WHY] got {len(indices)} indices and {len(labels)} labels\n"
                f"[FIX] pass one label per index"
            )
        for idx, label in zip(indices, labels):
            self.X_train.append(self.X_unlabeled[idx])
            self.y_train.append(label)
            self.labeled_indices.add(idx)
        # Retrain the model on the accumulated labeled set
        X = np.asarray(self.X_train)
        y = np.asarray(self.y_train)
        try:
            self.model.train(X, y, verbose=False)
        except Exception:
            try:
                self.model.partial_fit(X, y, verbose=False)
            except Exception as e:
                raise RuntimeError(f"Could not retrain model: {e}")
        # Invalidate the probability cache
        self._proba_cache = None

    def evaluate(self, X_test, y_test) -> float:
        """Evaluate the current model on a test set. Returns accuracy."""
        try:
            preds = self.model.predict(np.asarray(X_test))
            preds = np.asarray(preds).flatten()
            y_test = np.asarray(y_test).flatten()
            return float(np.mean(preds == y_test))
        except Exception:
            return 0.0

    def status(self) -> dict:
        return {
            "strategy": self.strategy,
            "batch_size": self.batch_size,
            "n_unlabeled_total": len(self.X_unlabeled),
            "n_labeled": len(self.labeled_indices),
            "n_remaining": len(self.X_unlabeled) - len(self.labeled_indices),
            "training_set_size": len(self.X_train),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_scores(self, candidate_indices: List[int]) -> np.ndarray:
        """Compute an uncertainty score for each candidate. Higher = more uncertain."""
        # Get class probabilities
        proba = self._get_proba(candidate_indices)
        if proba is None:
            # Can't get probabilities -- fall back to random scores
            return np.random.rand(len(candidate_indices))
        if self.strategy == "uncertainty":
            # 1 - max class probability (high when model is unsure)
            return 1.0 - np.max(proba, axis=1)
        elif self.strategy == "margin":
            # 1 - (top1 - top2) (high when model is torn between top 2)
            sorted_proba = np.sort(proba, axis=1)
            if sorted_proba.shape[1] >= 2:
                margin = sorted_proba[:, -1] - sorted_proba[:, -2]
            else:
                margin = np.ones(len(proba))
            return 1.0 - margin
        elif self.strategy == "entropy":
            # Standard entropy of the probability distribution
            # H = -sum(p * log(p))
            with np.errstate(divide='ignore', invalid='ignore'):
                entropy = -np.sum(proba * np.where(proba > 0, np.log(proba), 0), axis=1)
            return entropy
        else:
            return np.random.rand(len(candidate_indices))

    def _get_proba(self, candidate_indices: List[int]) -> Optional[np.ndarray]:
        """Get class probabilities for the candidate samples.

        Tries predict_proba() first; falls back to predict() + one-hot
        if necessary (returns None if neither works).
        """
        X = self.X_unlabeled[candidate_indices]
        tech = getattr(self.model, "technique", None)
        # Try predict_proba via the underlying sklearn model
        if tech is not None and hasattr(tech, "model") and hasattr(tech.model, "predict_proba"):
            try:
                return tech.model.predict_proba(X)
            except Exception:
                pass
        # Try the model itself
        if hasattr(self.model, "predict_proba"):
            try:
                return self.model.predict_proba(X)
            except Exception:
                pass
        # Fallback: return None -- caller will use random scores
        return None


__all__ = ["ActiveLearningLoop"]
