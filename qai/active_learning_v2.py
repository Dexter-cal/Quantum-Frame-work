"""
qai.active_learning_v2 -- advanced active-learning toolkit for the Quantum AI Framework.

Active learning: instead of labeling a random sample of your data, let
the MODEL pick which samples it's most uncertain about (or which would
most improve it). Label those, retrain, repeat. Reaches high accuracy
with far fewer labeled examples than random sampling.

This is the v2 module -- it builds on `qai.active_learning` (the basic
loop) and adds a strategy zoo, an orchestrating `ActiveLearner`, a
`LabelOracle` for simulation, and an `ActiveLearningExperiment` for
head-to-head strategy comparison.

Strategies (all subclass `QueryStrategy`):
  - UncertaintySampling      -- least_confident | margin | entropy
  - QueryByCommittee         -- vote_entropy | kl_divergence (bootstrap committee)
  - ExpectedModelChange      -- pick samples that would change the model the most
                                (gradient magnitude; torch models only, falls back
                                to uncertainty for sklearn models)
  - DensityWeighted          -- weight uncertainty by k-NN representativeness
  - BALD                     -- Bayesian Active Learning by Disagreement (MC-dropout)
  - CoreSet                  -- greedy k-center coverage of the feature space
  - RandomSampling           -- baseline; pick uniformly at random
  - HybridStrategy           -- weighted combination of multiple strategies

Top-level API:
  - ActiveLearner            -- orchestrates the loop (query -> teach -> evaluate)
  - LabelOracle              -- simulates a human labeler for experiments
  - ActiveLearningExperiment -- runs all strategies, returns comparison + SVG plot

Convenience:
  - create(model, strategy='uncertainty', **kwargs) -> ActiveLearner
  - compare_strategies(model_factory, X, y, strategies=None, n_iterations=20) -> dict
  - quick(model, X, y, n_query=5) -> list[int]

Usage:
    import qai
    from qai.active_learning_v2 import ActiveLearner, UncertaintySampling

    X, y = qai.load_iris()
    m = qai.build(technique='neural_network', hidden_layer_sizes=(8, 6))
    m.train(X[:10], y[:10])

    learner = ActiveLearner(model=m, strategy='uncertainty',
                            X_train=X[:10], y_train=y[:10],
                            X_pool=X[10:], y_pool=y[10:])
    for _ in range(10):
        r = learner.step(n=5)
        print(f"iter {r['iteration']}: train={r['train_size_after']}")

The model just needs `.predict()` and/or `.predict_proba()`; the toolkit
peeks into `model.technique.net` (torch) or `model.technique.model`
(sklearn) for richer probability/gradient access when available.
"""
from __future__ import annotations

import copy
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ============================================================================
# Helpers
# ============================================================================

def _to_2d(X) -> np.ndarray:
    """Coerce input to a 2-D float numpy array (n_samples, n_features).

    Empty inputs return a (0, 0) array (NOT (1, 0) -- which would be
    misinterpreted as one zero-feature sample).
    """
    arr = np.asarray(X, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _get_proba(model, X) -> Optional[np.ndarray]:
    """Try to get class probabilities from any model.

    Order of attempts:
      1. ``model.predict_proba(X)``            -- sklearn-style API
      2. ``model.technique.net(X)`` + softmax  -- torch-backed qai techniques
      3. ``model.technique.model.predict_proba``-- sklearn model nested under technique
      4. ``model.predict(X)`` + one-hot encode -- last-resort fallback

    Returns a (n_samples, n_classes) array, or ``None`` if nothing worked.
    For the degenerate single-class case (output_dim == 1) a (n, 1) array of
    ones is returned -- caller must handle the shape.
    """
    X_arr = _to_2d(X)
    n = len(X_arr)

    # 1. Try predict_proba on the model directly
    if hasattr(model, "predict_proba"):
        try:
            p = np.asarray(model.predict_proba(X_arr))
            if p.ndim == 2 and p.shape[0] == n and p.shape[1] >= 1:
                return p
        except (NotImplementedError, Exception):
            pass

    tech = getattr(model, "technique", None)
    if tech is not None:
        # 2. Try torch net + softmax (the neural_network path)
        net = getattr(tech, "net", None)
        if net is not None:
            try:
                import torch
                was_training = net.training
                net.eval()
                try:
                    with torch.no_grad():
                        x_tensor = torch.tensor(X_arr.astype(np.float32))
                        logits = net(x_tensor)
                        if logits.shape[1] == 1:
                            # Degenerate single-class output: softmax([x]) == [1.0]
                            return np.ones((n, 1))
                        probs = torch.softmax(logits, dim=1).numpy()
                        return probs
                finally:
                    if was_training:
                        net.train()
            except Exception:
                pass

        # 3. Try an sklearn-style sub-model living on the technique
        sub_model = getattr(tech, "model", None)
        if sub_model is not None and hasattr(sub_model, "predict_proba"):
            try:
                p = np.asarray(sub_model.predict_proba(X_arr))
                if p.ndim == 2 and p.shape[0] == n and p.shape[1] >= 1:
                    return p
            except Exception:
                pass

    # 4. Last resort: predict + one-hot
    if hasattr(model, "predict"):
        try:
            preds = np.asarray(model.predict(X_arr)).flatten()
            if len(preds) == n:
                classes = np.unique(preds)
                if classes.size > 0:
                    one_hot = np.zeros((n, classes.size))
                    for i, c in enumerate(classes):
                        one_hot[:, i] = (preds == c).astype(float)
                    return one_hot
        except Exception:
            pass

    return None


def _entropy(proba: np.ndarray) -> np.ndarray:
    """Per-sample Shannon entropy of a (n, k) probability matrix."""
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(proba * np.where(proba > 0, np.log(proba), 0.0), axis=1)


def _is_trained(model) -> bool:
    """Best-effort check that a model has been fit at least once."""
    tech = getattr(model, "technique", None)
    if tech is not None:
        if hasattr(tech, "is_trained"):
            try:
                return bool(tech.is_trained())
            except Exception:
                pass
        if hasattr(tech, "_trained"):
            return bool(getattr(tech, "_trained", False))
    # Fallback: predict() raises if not trained -- probe with a tiny input
    return True  # optimistically allow scoring to attempt


# ============================================================================
# Strategy base class
# ============================================================================

class QueryStrategy:
    """Abstract base class for all active-learning query strategies.

    Subclasses must implement :meth:`scores` and may override
    :meth:`select` (the default implementation does a top-n argsort of the
    scores, which is right for most strategies but not for greedy ones
    like CoreSet).
    """

    name: str = "abstract"

    def scores(self, model, X_pool) -> np.ndarray:
        """Per-sample score (higher = more valuable to label)."""
        raise NotImplementedError(f"{self.name}.scores() not implemented")

    def select(self, model, X_pool, n: int = 1,
               X_train=None, y_train=None) -> List[int]:
        """Return indices of the n most valuable samples in X_pool.

        Default implementation: compute scores, take top-n.
        Greedy strategies (CoreSet) override this.
        """
        X_arr = _to_2d(X_pool)
        n_pool = len(X_arr)
        if n_pool == 0:
            return []
        n = min(n, n_pool)
        sc = np.asarray(self.scores(model, X_arr), dtype=float)
        if sc.size != n_pool:
            # Length mismatch -- fall back to random
            sc = np.random.rand(n_pool)
        # Top-n by score (descending). Ties broken by original order.
        top = np.argsort(-sc, kind="stable")[:n]
        return [int(i) for i in top]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


# ============================================================================
# Concrete strategies
# ============================================================================

class UncertaintySampling(QueryStrategy):
    """Pick samples the model is least confident about.

    Modes:
      - ``'least_confident'`` -- 1 - max(p)              (model is unsure of the top class)
      - ``'margin'``          -- 1 - (p_top1 - p_top2)   (model is torn between two)
      - ``'entropy'``         -- Shannon entropy of p    (model is generally confused)
    """

    name = "uncertainty"
    MODES = ("least_confident", "margin", "entropy")

    def __init__(self, mode: str = "entropy"):
        if mode not in self.MODES:
            raise ValueError(
                f"[WHAT] Unknown UncertaintySampling mode {mode!r}\n"
                f"[WHY] Mode must be one of {self.MODES}\n"
                f"[FIX] Pass mode='least_confident' | 'margin' | 'entropy'"
            )
        self.mode = mode

    def scores(self, model, X_pool) -> np.ndarray:
        X_arr = _to_2d(X_pool)
        n = len(X_arr)
        if n == 0:
            return np.zeros(0)

        proba = _get_proba(model, X_arr)
        if proba is None:
            # No probabilities available -- uniform random (still better than nothing)
            return np.random.rand(n)

        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)

        # Single-class degenerate case: model can't be uncertain about anything
        if proba.shape[1] <= 1:
            return np.zeros(n)

        if self.mode == "least_confident":
            return 1.0 - np.max(proba, axis=1)
        elif self.mode == "margin":
            sorted_p = np.sort(proba, axis=1)
            margin = sorted_p[:, -1] - sorted_p[:, -2]
            return 1.0 - margin
        else:  # entropy
            return _entropy(proba)


class QueryByCommittee(QueryStrategy):
    """Train a committee of models on bootstrap samples; pick samples with
    highest disagreement.

    Modes:
      - ``'vote_entropy'``   -- entropy of the per-class vote histogram
      - ``'kl_divergence'``  -- average KL(committee member || consensus)
    """

    name = "qbc"
    MODES = ("vote_entropy", "kl_divergence")

    def __init__(self, n_members: int = 5, mode: str = "vote_entropy",
                 random_state: Optional[int] = None):
        if mode not in self.MODES:
            raise ValueError(
                f"[WHAT] Unknown QBC mode {mode!r}\n"
                f"[WHY] Mode must be one of {self.MODES}\n"
                f"[FIX] Pass mode='vote_entropy' | 'kl_divergence'"
            )
        self.n_members = n_members
        self.mode = mode
        self._rng = np.random.RandomState(random_state)
        self._committee: Optional[List[Any]] = None
        self._train_signature: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ build
    def _build_committee(self, model, X_train, y_train) -> None:
        X_train = _to_2d(X_train)
        y_train = np.asarray(y_train).flatten()
        n = len(X_train)
        if n == 0:
            self._committee = []
            return
        self._committee = []
        for _ in range(self.n_members):
            member = copy.deepcopy(model)
            idx = self._rng.randint(0, n, size=n)  # bootstrap sample (with replacement)
            X_bs = X_train[idx]
            y_bs = y_train[idx]
            # If the bootstrap sample is single-class, training a fresh deep-copy
            # will still succeed (the underlying technique handles 1 class).
            try:
                member.train(X_bs, y_bs, verbose=False)
            except Exception:
                try:
                    member.partial_fit(X_bs, y_bs, verbose=False)
                except Exception:
                    pass  # skip broken member
            self._committee.append(member)
        self._train_signature = (n, int(self._rng.randint(0, 2 ** 31 - 1)))

    def _ensure_committee(self, model, X_train=None, y_train=None) -> bool:
        if self._committee is not None:
            return True
        if X_train is not None and y_train is not None and len(X_train) > 0:
            self._build_committee(model, X_train, y_train)
            return self._committee is not None and len(self._committee) > 0
        # Try to recover training data from the model's last fit
        tech = getattr(model, "technique", None)
        if tech is not None:
            X_last = getattr(tech, "_last_X", None)
            y_last = getattr(tech, "_last_y", None)
            if X_last is not None and y_last is not None and len(X_last) > 0:
                self._build_committee(model, X_last, y_last)
                return self._committee is not None and len(self._committee) > 0
        return False

    # ------------------------------------------------------------------ scores
    def scores(self, model, X_pool) -> np.ndarray:
        X_arr = _to_2d(X_pool)
        n = len(X_arr)
        if n == 0:
            return np.zeros(0)
        if not self._ensure_committee(model):
            # No committee -> uniform
            return np.zeros(n)

        preds: List[np.ndarray] = []
        probas: List[np.ndarray] = []
        for member in self._committee:
            try:
                p = np.asarray(member.predict(X_arr)).flatten()
                preds.append(p)
            except Exception:
                pass
            pr = _get_proba(member, X_arr)
            if pr is not None:
                probas.append(np.asarray(pr, dtype=float))

        if not preds:
            return np.zeros(n)

        if self.mode == "vote_entropy":
            classes = np.unique(np.concatenate(preds))
            votes = np.zeros((n, len(classes)))
            for p in preds:
                for ci, c in enumerate(classes):
                    votes[:, ci] += (p == c).astype(float)
            votes /= len(preds)
            return _entropy(votes)
        else:  # kl_divergence
            if not probas:
                # No probas available -- fall back to vote entropy
                classes = np.unique(np.concatenate(preds))
                votes = np.zeros((n, len(classes)))
                for p in preds:
                    for ci, c in enumerate(classes):
                        votes[:, ci] += (p == c).astype(float)
                votes /= len(preds)
                return _entropy(votes)
            # Align committee probas to a common column set
            max_cols = max(p.shape[1] for p in probas) if probas else 1
            aligned = []
            for p in probas:
                if p.shape[1] < max_cols:
                    pad = np.zeros((n, max_cols - p.shape[1]))
                    p = np.hstack([p, pad])
                aligned.append(p)
            stacked = np.stack(aligned, axis=0)  # (T, n, C)
            consensus = stacked.mean(axis=0)     # (n, C)
            consensus = np.clip(consensus, 1e-12, 1.0)
            # KL(p_t || consensus) averaged over committee
            kls = np.zeros(n)
            for t in range(stacked.shape[0]):
                pt = np.clip(stacked[t], 1e-12, 1.0)
                kls += np.sum(pt * (np.log(pt) - np.log(consensus)), axis=1)
            kls /= stacked.shape[0]
            return kls

    def select(self, model, X_pool, n: int = 1,
               X_train=None, y_train=None) -> List[int]:
        # Always rebuild the committee when fresh training data is provided,
        # so the committee reflects the current training set.
        if X_train is not None and y_train is not None and len(X_train) > 0:
            self._committee = None
        self._ensure_committee(model, X_train, y_train)
        return super().select(model, X_pool, n=n, X_train=X_train, y_train=y_train)


class ExpectedModelChange(QueryStrategy):
    """Pick samples that would change the model the most.

    Approximates the expected change in model parameters by the gradient
    magnitude of the cross-entropy loss w.r.t. the model's weights, treating
    the model's own prediction as the (proxy) label. Larger gradient == more
    change if this sample were added to the training set.

    Only torch-backed techniques expose gradients; for sklearn-style models
    this falls back to entropy uncertainty sampling with a warning.
    """

    name = "expected_model_change"

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self._warned = False

    def _fallback_scores(self, model, X_arr) -> np.ndarray:
        if not self._warned:
            warnings.warn(
                "ExpectedModelChange: model has no torch gradient path; "
                "falling back to entropy uncertainty sampling."
            )
            self._warned = True
        return UncertaintySampling(mode="entropy").scores(model, X_arr)

    def scores(self, model, X_pool) -> np.ndarray:
        X_arr = _to_2d(X_pool)
        n = len(X_arr)
        if n == 0:
            return np.zeros(0)

        tech = getattr(model, "technique", None)
        net = getattr(tech, "net", None) if tech is not None else None
        if net is None:
            return self._fallback_scores(model, X_arr)

        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return self._fallback_scores(model, X_arr)

        # Detect output dim from the last Linear layer
        out_dim = None
        try:
            for m in net.modules():
                if hasattr(m, "out_features"):
                    out_dim = m.out_features
        except Exception:
            pass
        if out_dim is None or out_dim < 2:
            # Single-class output -> no gradient signal; fall back to uncertainty
            return self._fallback_scores(model, X_arr)

        was_training = net.training
        net.train()
        scores = np.zeros(n, dtype=float)
        try:
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                x_batch = torch.tensor(X_arr[start:end].astype(np.float32))
                for i in range(end - start):
                    net.zero_grad()
                    logits = net(x_batch[i:i + 1])
                    if logits.shape[1] < 2:
                        scores[start + i] = 0.0
                        continue
                    pred = logits.argmax(dim=1)
                    loss = F.cross_entropy(logits, pred)
                    loss.backward()
                    grad_sq = 0.0
                    for p in net.parameters():
                        if p.grad is not None:
                            grad_sq += float(p.grad.detach().norm() ** 2)
                    scores[start + i] = float(np.sqrt(grad_sq))
        finally:
            net.zero_grad()
            if not was_training:
                net.eval()
        return scores


class DensityWeighted(QueryStrategy):
    """Weight uncertainty by representativeness (k-NN density).

    Score = uncertainty(x) * density(x) ** beta

    where ``density(x)`` is ``1 / mean_distance_to_k_nearest_neighbors_in_pool``.
    This avoids wasting queries on outliers the model is uncertain about but
    which aren't representative of the data manifold.

    Args:
        base_mode: 'least_confident' | 'margin' | 'entropy' (uncertainty mode)
        k: number of nearest neighbors for density estimation
        beta: density exponent (0 = pure uncertainty, 1 = full density weighting)
    """

    name = "density_weighted"

    def __init__(self, base_mode: str = "entropy", k: int = 5, beta: float = 1.0):
        self.base = UncertaintySampling(mode=base_mode)
        self.k = k
        self.beta = beta

    def _knn_density(self, X_arr: np.ndarray) -> np.ndarray:
        n = len(X_arr)
        if n <= 1:
            return np.ones(n)
        k_eff = min(self.k + 1, n)  # +1 because self is the nearest neighbor
        try:
            from sklearn.neighbors import NearestNeighbors  # lazy import
            nn = NearestNeighbors(n_neighbors=k_eff).fit(X_arr)
            dists, _ = nn.kneighbors(X_arr)
            mean_d = dists[:, 1:].mean(axis=1)  # exclude self
        except Exception:
            # Pure-numpy fallback: pairwise distance matrix
            diff = X_arr[:, None, :] - X_arr[None, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            sorted_d = np.sort(dists, axis=1)
            mean_d = sorted_d[:, 1:k_eff].mean(axis=1)
        return 1.0 / (mean_d + 1e-8)

    def scores(self, model, X_pool) -> np.ndarray:
        X_arr = _to_2d(X_pool)
        n = len(X_arr)
        if n == 0:
            return np.zeros(0)
        uncertainty = np.asarray(self.base.scores(model, X_arr), dtype=float)
        density = self._knn_density(X_arr)
        return uncertainty * (density ** self.beta)


class BALD(QueryStrategy):
    """Bayesian Active Learning by Disagreement (Houlsby et al. 2011).

    Uses MC-dropout to get a distribution of predictions per sample, then
    picks samples where the committee of stochastic forward passes disagree
    the most:

        BALD(x) = H[E[p]] - E[H[p]]

    If the model has no Dropout layers (p > 0), MC-dropout produces no
    variation, so we fall back to uncertainty sampling with a warning.
    """

    name = "bald"

    def __init__(self, n_passes: int = 10, random_state: Optional[int] = None):
        self.n_passes = n_passes
        self._rng = np.random.RandomState(random_state)
        self._warned = False

    @staticmethod
    def _has_dropout(net) -> bool:
        try:
            import torch.nn as nn
            for m in net.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                    if getattr(m, "p", 0.0) > 0:
                        return True
        except Exception:
            pass
        return False

    @staticmethod
    def _output_dim(net) -> int:
        try:
            out = None
            for m in net.modules():
                if hasattr(m, "out_features"):
                    out = m.out_features
            return int(out) if out is not None else 0
        except Exception:
            return 0

    def scores(self, model, X_pool) -> np.ndarray:
        X_arr = _to_2d(X_pool)
        n = len(X_arr)
        if n == 0:
            return np.zeros(0)

        tech = getattr(model, "technique", None)
        net = getattr(tech, "net", None) if tech is not None else None

        if net is None or not self._has_dropout(net) or self._output_dim(net) < 2:
            if not self._warned:
                warnings.warn(
                    "BALD: model has no MC-dropout (or single-class output); "
                    "falling back to entropy uncertainty sampling."
                )
                self._warned = True
            return UncertaintySampling(mode="entropy").scores(model, X_arr)

        try:
            import torch
        except ImportError:
            return UncertaintySampling(mode="entropy").scores(model, X_arr)

        was_training = net.training
        net.train()  # enable dropout
        T = self.n_passes
        C = self._output_dim(net)
        all_probs = np.zeros((T, n, C), dtype=float)
        try:
            with torch.no_grad():
                x_tensor = torch.tensor(X_arr.astype(np.float32))
                for t in range(T):
                    logits = net(x_tensor)
                    if logits.shape[1] < 2:
                        all_probs[t, :, 0] = 1.0
                    else:
                        all_probs[t] = torch.softmax(logits, dim=1).numpy()
        finally:
            if not was_training:
                net.eval()

        # BALD = H[E[p]] - E[H[p]]
        mean_probs = all_probs.mean(axis=0)            # (n, C)
        total_entropy = _entropy(mean_probs)           # (n,)
        per_pass_entropy = _entropy(all_probs.reshape(T * n, C)).reshape(T, n)
        expected_entropy = per_pass_entropy.mean(axis=0)  # (n,)
        bald = total_entropy - expected_entropy
        return np.maximum(bald, 0.0)


class CoreSet(QueryStrategy):
    """Greedy k-center coverage: pick samples that maximize coverage of the
    feature space.

    The first sample is the one farthest from the (existing) training set
    centroid; each subsequent sample is the one farthest from any already
    selected sample. This is the classic CoreSet / k-center-greedy algorithm
    of Sener & Savarese (2018).
    """

    name = "coreset"

    def scores(self, model, X_pool) -> np.ndarray:
        """Per-sample "novelty" score: distance from each pool sample to the
        pool centroid. Used as a proxy when only `scores()` is called (the
        real CoreSet selection logic lives in :meth:`select`)."""
        X_arr = _to_2d(X_pool)
        n = len(X_arr)
        if n == 0:
            return np.zeros(0)
        if n == 1:
            return np.ones(1)
        centroid = X_arr.mean(axis=0, keepdims=True)
        return np.linalg.norm(X_arr - centroid, axis=1)

    def select(self, model, X_pool, n: int = 1,
               X_train=None, y_train=None) -> List[int]:
        X_arr = _to_2d(X_pool)
        n_pool = len(X_arr)
        if n_pool == 0:
            return []
        n = min(n, n_pool)
        if n == n_pool:
            return list(range(n_pool))

        # Anchor set: training data if provided, else pool centroid
        if X_train is not None and len(X_train) > 0:
            anchor = _to_2d(X_train)
        else:
            anchor = X_arr.mean(axis=0, keepdims=True)

        # Distance from each pool sample to nearest anchor sample
        diff = X_arr[:, None, :] - anchor[None, :, :]
        dist_to_anchor = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)

        selected: List[int] = []
        available = list(range(n_pool))
        # Pick the farthest sample from the anchor first
        first = int(np.argmax(dist_to_anchor))
        selected.append(first)
        available.remove(first)

        # Greedy k-center: each step pick the sample whose nearest already-selected
        # sample is farthest (maximize min-distance to selected set).
        while len(selected) < n and available:
            sel_arr = X_arr[selected]
            best_i = -1
            best_d = -1.0
            for i in available:
                d = np.linalg.norm(sel_arr - X_arr[i], axis=1).min()
                if d > best_d:
                    best_d = d
                    best_i = i
            selected.append(best_i)
            available.remove(best_i)

        return selected


class RandomSampling(QueryStrategy):
    """Baseline strategy: pick samples uniformly at random."""

    name = "random"

    def __init__(self, random_state: Optional[int] = None):
        self._rng = np.random.RandomState(random_state)

    def scores(self, model, X_pool) -> np.ndarray:
        n = len(X_pool) if hasattr(X_pool, "__len__") else 0
        return self._rng.rand(n) if n > 0 else np.zeros(0)

    def select(self, model, X_pool, n: int = 1,
               X_train=None, y_train=None) -> List[int]:
        X_arr = _to_2d(X_pool)
        n_pool = len(X_arr)
        if n_pool == 0:
            return []
        n = min(n, n_pool)
        chosen = self._rng.choice(n_pool, size=n, replace=False)
        return [int(i) for i in chosen]


class HybridStrategy(QueryStrategy):
    """Combine multiple strategies with weights.

    Each strategy's per-sample scores are min-max normalised to [0, 1]
    (so different scales are comparable), then a weighted sum is taken.

    Args:
        strategies: list of QueryStrategy instances
        weights: list of floats (same length); defaults to equal weights
    """

    name = "hybrid"

    def __init__(self, strategies: List[QueryStrategy],
                 weights: Optional[List[float]] = None):
        if not strategies:
            raise ValueError(
                "[WHAT] HybridStrategy requires at least one sub-strategy\n"
                "[WHY] strategies list is empty\n"
                "[FIX] Pass e.g. HybridStrategy([UncertaintySampling(), CoreSet()])"
            )
        self.strategies = list(strategies)
        if weights is None:
            w = 1.0 / len(self.strategies)
            self.weights = [w] * len(self.strategies)
        else:
            if len(weights) != len(self.strategies):
                raise ValueError(
                    f"[WHAT] weights and strategies length mismatch\n"
                    f"[WHY] got {len(weights)} weights, {len(self.strategies)} strategies\n"
                    f"[FIX] Pass one weight per strategy"
                )
            self.weights = list(weights)

    def scores(self, model, X_pool) -> np.ndarray:
        X_arr = _to_2d(X_pool)
        n = len(X_arr)
        if n == 0:
            return np.zeros(0)
        combined = np.zeros(n, dtype=float)
        for strat, w in zip(self.strategies, self.weights):
            sc = np.asarray(strat.scores(model, X_arr), dtype=float)
            if sc.size != n:
                sc = np.zeros(n)
            lo, hi = float(sc.min()), float(sc.max())
            if hi - lo > 1e-12:
                sc = (sc - lo) / (hi - lo)
            else:
                sc = np.zeros(n)
            combined += w * sc
        return combined


# ============================================================================
# Strategy registry / factory
# ============================================================================

_STRATEGY_REGISTRY: Dict[str, Callable[..., QueryStrategy]] = {
    "uncertainty": lambda **kw: UncertaintySampling(
        mode=kw.pop("mode", "entropy") if "mode" in kw else "entropy"),
    "uncertainty_least_confident": lambda **kw: UncertaintySampling(mode="least_confident"),
    "uncertainty_margin": lambda **kw: UncertaintySampling(mode="margin"),
    "uncertainty_entropy": lambda **kw: UncertaintySampling(mode="entropy"),
    "qbc": lambda **kw: QueryByCommittee(
        n_members=kw.pop("n_members", 5),
        mode=kw.pop("mode", "vote_entropy")),
    "committee": lambda **kw: QueryByCommittee(
        n_members=kw.pop("n_members", 5),
        mode=kw.pop("mode", "vote_entropy")),
    "expected_model_change": lambda **kw: ExpectedModelChange(),
    "emc": lambda **kw: ExpectedModelChange(),
    "density": lambda **kw: DensityWeighted(
        base_mode=kw.pop("base_mode", "entropy"),
        k=kw.pop("k", 5),
        beta=kw.pop("beta", 1.0)),
    "density_weighted": lambda **kw: DensityWeighted(),
    "bald": lambda **kw: BALD(n_passes=kw.pop("n_passes", 10)),
    "coreset": lambda **kw: CoreSet(),
    "random": lambda **kw: RandomSampling(
        random_state=kw.pop("random_state", None)),
    "hybrid": lambda **kw: HybridStrategy(
        strategies=kw.pop("strategies"),
        weights=kw.pop("weights", None)),
}


def _make_strategy(name_or_strategy, **kwargs) -> QueryStrategy:
    """Build a strategy from a name string, or pass through a QueryStrategy instance."""
    if isinstance(name_or_strategy, QueryStrategy):
        return name_or_strategy
    if isinstance(name_or_strategy, str):
        key = name_or_strategy.lower().replace("-", "_").replace(" ", "_")
        if key not in _STRATEGY_REGISTRY:
            raise ValueError(
                f"[WHAT] Unknown strategy {name_or_strategy!r}\n"
                f"[WHY] Strategy must be one of {sorted(_STRATEGY_REGISTRY.keys())}\n"
                f"[FIX] Use a registered name or pass a QueryStrategy instance"
            )
        return _STRATEGY_REGISTRY[key](**kwargs)
    raise TypeError(
        f"[WHAT] strategy must be a string or QueryStrategy instance\n"
        f"[WHY] got {type(name_or_strategy).__name__}\n"
        f"[FIX] Pass e.g. strategy='uncertainty' or strategy=UncertaintySampling()"
    )


# ============================================================================
# LabelOracle -- simulates a human labeler for experiments
# ============================================================================

class LabelOracle:
    """Simulates an oracle for testing active-learning loops.

    In real life the labels come from a human annotator; in experiments
    we hold out the true labels and the oracle reveals them on demand,
    charging one unit per label.

    Args:
        true_labels: array-like of ground-truth labels, indexed by sample id.
    """

    def __init__(self, true_labels):
        self.true_labels = np.asarray(true_labels).flatten()
        self.queried: set = set()
        self.cost = 0

    def label(self, indices) -> Dict[int, Any]:
        """Return true labels for the requested indices.

        Args:
            indices: iterable of integer sample ids.
        Returns:
            dict mapping index -> label.
        """
        out: Dict[int, Any] = {}
        for i in indices:
            i = int(i)
            if 0 <= i < len(self.true_labels):
                out[i] = self.true_labels[i]
                if i not in self.queried:
                    self.queried.add(i)
                    self.cost += 1
        return out

    def cost_sofar(self) -> int:
        """Total labels queried so far."""
        return self.cost

    def reset(self) -> None:
        self.queried.clear()
        self.cost = 0


# ============================================================================
# ActiveLearner -- orchestrates the loop
# ============================================================================

class ActiveLearner:
    """Runs the active-learning loop: query -> teach -> evaluate.

    Args:
        model: a qai Model (or any object with .train()/.predict()).
        strategy: a name string ('uncertainty', 'random', 'qbc', ...) or a
                  QueryStrategy instance.
        X_train, y_train: initial labeled set (numpy arrays).
        X_pool, y_pool: the unlabeled pool with held-out labels (the labels
                        are used internally by ``step()`` as the oracle; in a
                        real deployment the human supplies them).
        X_test, y_test: optional held-out test set used by ``step()`` /
                        ``evaluate()``.
        random_state: int seed for reproducibility (used by RandomSampling
                      and QueryByCommittee when their own seed isn't set).
    """

    def __init__(self, model, strategy: Union[str, QueryStrategy] = "uncertainty",
                 X_train=None, y_train=None,
                 X_pool=None, y_pool=None,
                 X_test=None, y_test=None,
                 random_state: Optional[int] = None):
        self.model = model
        self.strategy = _make_strategy(strategy)
        self.random_state = random_state

        self.X_train: List[Any] = (
            list(_to_2d(X_train)) if X_train is not None else []
        )
        self.y_train: List[Any] = (
            list(np.asarray(y_train).flatten()) if y_train is not None else []
        )
        self.X_pool: List[Any] = (
            list(_to_2d(X_pool)) if X_pool is not None else []
        )
        self.y_pool: List[Any] = (
            list(np.asarray(y_pool).flatten()) if y_pool is not None else []
        )
        self.X_test = X_test
        self.y_test = y_test

        self._history: List[Dict[str, Any]] = []

        # Initial training so the model is ready to query
        if self.X_train and self.y_train:
            self._retrain()

    # ------------------------------------------------------------------ retrain
    def _retrain(self) -> None:
        if not self.X_train or not self.y_train:
            return
        X = np.asarray(self.X_train)
        y = np.asarray(self.y_train)
        try:
            self.model.train(X, y, verbose=False)
        except Exception:
            try:
                self.model.partial_fit(X, y, verbose=False)
            except Exception as e:
                warnings.warn(f"ActiveLearner: could not retrain model: {e}")

    # ------------------------------------------------------------------ query
    def query(self, n: int = 1) -> List[int]:
        """Get the indices (into the current pool) of the next n samples to label."""
        if not self.X_pool:
            return []
        X_pool_arr = np.asarray(self.X_pool) if self.X_pool else np.zeros((0, 1))
        if X_pool_arr.ndim == 1:
            X_pool_arr = X_pool_arr.reshape(1, -1)
        n = min(n, len(X_pool_arr))
        X_train_arr = np.asarray(self.X_train) if self.X_train else None
        y_train_arr = np.asarray(self.y_train) if self.y_train else None
        try:
            indices = self.strategy.select(
                self.model, X_pool_arr, n=n,
                X_train=X_train_arr, y_train=y_train_arr,
            )
        except Exception as e:
            warnings.warn(f"ActiveLearner.query: strategy failed ({e}); using random")
            rng = np.random.RandomState(self.random_state)
            indices = list(rng.choice(len(X_pool_arr), size=n, replace=False))
        return [int(i) for i in indices]

    # ------------------------------------------------------------------ teach
    def teach(self, indices: Sequence[int], labels: Sequence[Any]) -> None:
        """Move samples from the pool to the training set and retrain.

        Args:
            indices: indices into the *current* pool (as returned by query()).
            labels:  labels for those indices (same length).
        """
        if len(indices) != len(labels):
            raise ValueError(
                f"[WHAT] indices and labels length mismatch\n"
                f"[WHY] got {len(indices)} indices, {len(labels)} labels\n"
                f"[FIX] pass one label per index"
            )
        # Reverse-sort by index so deletes don't shift earlier indices
        pairs = sorted(zip(indices, labels), key=lambda x: x[0], reverse=True)
        for idx, label in pairs:
            idx = int(idx)
            if not (0 <= idx < len(self.X_pool)):
                continue
            self.X_train.append(self.X_pool[idx])
            self.y_train.append(label)
            del self.X_pool[idx]
            del self.y_pool[idx]
        self._retrain()

    # ------------------------------------------------------------------ step
    def step(self, n: int = 1, X_test=None, y_test=None) -> Dict[str, Any]:
        """One full iteration: query -> teach (using internal y_pool as oracle) -> evaluate."""
        X_test = X_test if X_test is not None else self.X_test
        y_test = y_test if y_test is not None else self.y_test

        iteration = len(self._history) + 1
        pool_before = len(self.X_pool)
        train_before = len(self.X_train)

        indices = self.query(n=n)
        # Use the held-out pool labels as the oracle (simulated labeling)
        labels = [self.y_pool[i] for i in indices] if indices else []
        self.teach(indices, labels)

        pool_after = len(self.X_pool)
        train_after = len(self.X_train)

        result: Dict[str, Any] = {
            "iteration": iteration,
            "selected_indices": indices,
            "n_selected": len(indices),
            "pool_size_before": pool_before,
            "pool_size_after": pool_after,
            "train_size_before": train_before,
            "train_size_after": train_after,
            "strategy": self.strategy.name,
            "accuracy": None,
        }

        if X_test is not None and y_test is not None:
            metrics = self.evaluate(X_test, y_test)
            result.update(metrics)

        self._history.append(result)
        return result

    # ------------------------------------------------------------------ run
    def run(self, n_iterations: int = 10, n_per_iter: int = 1) -> List[Dict[str, Any]]:
        """Run the full active-learning loop for n_iterations."""
        results: List[Dict[str, Any]] = []
        for _ in range(n_iterations):
            if not self.X_pool:
                break
            r = self.step(n=n_per_iter)
            results.append(r)
        return results

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """Current model performance on a test set."""
        try:
            X_arr = _to_2d(X_test)
            preds = np.asarray(self.model.predict(X_arr)).flatten()
            y_arr = np.asarray(y_test).flatten()
            if len(preds) != len(y_arr):
                return {"accuracy": None, "error": "pred/label length mismatch"}
            accuracy = float(np.mean(preds == y_arr))
            return {
                "accuracy": accuracy,
                "n_test": len(y_arr),
                "n_train": len(self.X_train),
            }
        except Exception as e:
            return {"accuracy": None, "error": str(e)}

    # ------------------------------------------------------------------ history
    def history(self) -> List[Dict[str, Any]]:
        """Query history: one dict per step."""
        return list(self._history)

    # ------------------------------------------------------------------ stats
    def stats(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.name,
            "train_size": len(self.X_train),
            "pool_size": len(self.X_pool),
            "n_iterations": len(self._history),
            "has_test": self.X_test is not None,
            "model_technique": getattr(getattr(self.model, "technique", None),
                                        "name", "unknown"),
        }


# ============================================================================
# ActiveLearningExperiment -- runs all strategies head-to-head
# ============================================================================

class ActiveLearningExperiment:
    """Runs a full active-learning experiment comparing strategies.

    Args:
        model_factory: zero-arg callable returning a fresh, untrained model.
        strategies: dict {name: QueryStrategy} OR list of QueryStrategy
                    (names taken from each strategy's ``.name``).
        X, y: full labeled dataset (we split it into test / initial-train / pool).
        test_size: fraction of data held out for evaluation.
        initial_size: fraction of data used as the initial labeled seed
                      (same seed for every strategy, for fair comparison).
    """

    def __init__(self, model_factory: Callable[[], Any],
                 strategies: Union[Dict[str, QueryStrategy], List[QueryStrategy], None],
                 X, y,
                 test_size: float = 0.3,
                 initial_size: float = 0.05,
                 random_state: int = 42):
        self.model_factory = model_factory
        if strategies is None:
            strategies = self._default_strategies()
        if isinstance(strategies, dict):
            self.strategies: Dict[str, QueryStrategy] = dict(strategies)
        elif isinstance(strategies, (list, tuple)):
            self.strategies = {
                getattr(s, "name", f"strat_{i}"): s
                for i, s in enumerate(strategies)
            }
        else:
            raise TypeError(
                f"[WHAT] strategies must be a dict, list, or None\n"
                f"[WHY] got {type(strategies).__name__}\n"
                f"[FIX] Pass e.g. strategies={'uncertainty': UncertaintySampling()}"
            )
        self.X = np.asarray(X)
        self.y = np.asarray(y).flatten()
        self.test_size = test_size
        self.initial_size = initial_size
        self.random_state = random_state
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self._split()

    @staticmethod
    def _default_strategies() -> Dict[str, QueryStrategy]:
        return {
            "uncertainty": UncertaintySampling(mode="entropy"),
            "random": RandomSampling(random_state=42),
            "density": DensityWeighted(base_mode="entropy", k=5, beta=1.0),
            "coreset": CoreSet(),
            "qbc": QueryByCommittee(n_members=3, mode="vote_entropy",
                                    random_state=42),
        }

    def _split(self) -> None:
        n = len(self.X)
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_test = max(1, int(n * self.test_size))
        n_init = max(1, int(n * self.initial_size))
        n_init = min(n_init, n - n_test - 1)
        test_idx = perm[:n_test]
        rest_idx = perm[n_test:]
        init_idx = rest_idx[:n_init]
        pool_idx = rest_idx[n_init:]
        self.X_test = self.X[test_idx]
        self.y_test = self.y[test_idx]
        self.X_init = self.X[init_idx]
        self.y_init = self.y[init_idx]
        self.X_pool = self.X[pool_idx]
        self.y_pool = self.y[pool_idx]

    # ------------------------------------------------------------------ run
    def run(self, n_iterations: int = 20, n_per_iter: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Run all strategies, return ``{strategy_name: [step_results]}``."""
        self.results = {}
        for name, strategy in self.strategies.items():
            try:
                model = self.model_factory()
                strat = copy.deepcopy(strategy)
                learner = ActiveLearner(
                    model=model,
                    strategy=strat,
                    X_train=self.X_init.copy(),
                    y_train=self.y_init.copy(),
                    X_pool=self.X_pool.copy(),
                    y_pool=self.y_pool.copy(),
                    X_test=self.X_test,
                    y_test=self.y_test,
                )
                step_results = learner.run(n_iterations=n_iterations,
                                            n_per_iter=n_per_iter)
                # Always include the initial (iteration-0) evaluation point so
                # learning curves start before any queries.
                init_metrics = learner.evaluate(self.X_test, self.y_test)
                init_record = {
                    "iteration": 0,
                    "selected_indices": [],
                    "n_selected": 0,
                    "pool_size_before": len(self.X_pool),
                    "pool_size_after": len(self.X_pool),
                    "train_size_before": len(self.X_init),
                    "train_size_after": len(self.X_init),
                    "strategy": name,
                    **init_metrics,
                }
                self.results[name] = [init_record] + step_results
            except Exception as e:
                warnings.warn(f"ActiveLearningExperiment: strategy {name!r} failed: {e}")
                self.results[name] = []
        return self.results

    # ------------------------------------------------------------------ plot
    def plot_learning_curves(self, width: int = 640, height: int = 400) -> str:
        """SVG plot of accuracy vs # labeled samples, one line per strategy.

        Returns the SVG as a string.
        """
        margin_l, margin_r, margin_t, margin_b = 60, 20, 30, 50
        plot_w = width - margin_l - margin_r
        plot_h = height - margin_t - margin_b

        # Gather series
        series: Dict[str, List[Tuple[int, float]]] = {}
        max_labeled = 0
        for name, runs in self.results.items():
            if not runs:
                continue
            pts: List[Tuple[int, float]] = []
            for r in runs:
                acc = r.get("accuracy")
                n_lab = r.get("train_size_after",
                              r.get("train_size_before", 0))
                if acc is None:
                    continue
                pts.append((int(n_lab), float(acc)))
                max_labeled = max(max_labeled, int(n_lab))
            if pts:
                series[name] = pts

        if not series or max_labeled == 0:
            return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                    f'height="{height}"><text x="20" y="40" fill="#888">No data</text></svg>')

        # Colors
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ]

        def x_to_px(n_lab: int) -> float:
            return margin_l + (n_lab / max(max_labeled, 1)) * plot_w

        def y_to_px(acc: float) -> float:
            return margin_t + (1.0 - acc) * plot_h

        parts: List[str] = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            f'font-family="sans-serif" font-size="11">'
        )

        # Background
        parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')

        # Gridlines (5 horizontal, 5 vertical)
        for i in range(6):
            y = margin_t + (i / 5) * plot_h
            parts.append(
                f'<line x1="{margin_l}" y1="{y:.1f}" '
                f'x2="{margin_l + plot_w}" y2="{y:.1f}" '
                f'stroke="#eaeaea" stroke-width="1"/>'
            )
            label = f"{1.0 - i / 5:.1f}"
            parts.append(
                f'<text x="{margin_l - 6}" y="{y + 3:.1f}" '
                f'text-anchor="end" fill="#555">{label}</text>'
            )
        for i in range(6):
            x = margin_l + (i / 5) * plot_w
            parts.append(
                f'<line x1="{x:.1f}" y1="{margin_t}" '
                f'x2="{x:.1f}" y2="{margin_t + plot_h}" '
                f'stroke="#eaeaea" stroke-width="1"/>'
            )
            label = f"{int((i / 5) * max_labeled)}"
            parts.append(
                f'<text x="{x:.1f}" y="{margin_t + plot_h + 16}" '
                f'text-anchor="middle" fill="#555">{label}</text>'
            )

        # Axes
        parts.append(
            f'<line x1="{margin_l}" y1="{margin_t}" '
            f'x2="{margin_l}" y2="{margin_t + plot_h}" '
            f'stroke="#333" stroke-width="1.5"/>'
        )
        parts.append(
            f'<line x1="{margin_l}" y1="{margin_t + plot_h}" '
            f'x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" '
            f'stroke="#333" stroke-width="1.5"/>'
        )

        # Axis labels
        parts.append(
            f'<text x="{margin_l + plot_w / 2:.1f}" y="{height - 12}" '
            f'text-anchor="middle" fill="#333" font-weight="bold">'
            f'# labeled samples</text>'
        )
        parts.append(
            f'<text x="16" y="{margin_t + plot_h / 2:.1f}" '
            f'text-anchor="middle" fill="#333" font-weight="bold" '
            f'transform="rotate(-90 16 {margin_t + plot_h / 2:.1f})">'
            f'accuracy</text>'
        )

        # Title
        parts.append(
            f'<text x="{width / 2:.1f}" y="18" text-anchor="middle" '
            f'fill="#222" font-weight="bold" font-size="13">'
            f'Active Learning Curves</text>'
        )

        # Lines for each strategy
        legend_items: List[Tuple[str, str]] = []
        for i, (name, pts) in enumerate(series.items()):
            color = palette[i % len(palette)]
            d = " ".join(f"L{x_to_px(n):.1f} {y_to_px(a):.1f}"
                         for n, a in pts)
            d = "M" + d[1:]  # first L -> M
            parts.append(
                f'<path d="{d}" fill="none" stroke="{color}" '
                f'stroke-width="2" stroke-linejoin="round"/>'
            )
            # mark points
            for n, a in pts:
                parts.append(
                    f'<circle cx="{x_to_px(n):.1f}" cy="{y_to_px(a):.1f}" '
                    f'r="2.5" fill="{color}"/>'
                )
            legend_items.append((name, color))

        # Legend
        legend_x = margin_l + 8
        legend_y = margin_t + 8
        for i, (name, color) in enumerate(legend_items):
            ly = legend_y + i * 16
            parts.append(
                f'<rect x="{legend_x}" y="{ly}" width="12" height="12" '
                f'fill="{color}"/>'
            )
            parts.append(
                f'<text x="{legend_x + 18}" y="{ly + 10}" fill="#333">{name}</text>'
            )

        parts.append("</svg>")
        return "\n".join(parts)

    # ------------------------------------------------------------------ markdown
    def to_markdown(self) -> str:
        """Markdown report of the experiment results (includes the SVG)."""
        lines: List[str] = []
        lines.append("# Active Learning Experiment Report\n")
        lines.append("## Setup\n")
        lines.append(f"- Test size: {self.test_size:.0%} ({len(self.X_test)} samples)")
        lines.append(f"- Initial training size: {len(self.X_init)} samples")
        lines.append(f"- Pool size: {len(self.X_pool)} samples")
        lines.append(f"- Strategies compared: {len(self.strategies)}\n")

        lines.append("## Final Accuracies\n")
        lines.append("| Strategy | Final Accuracy | # Labeled | Iterations |")
        lines.append("|----------|----------------|-----------|------------|")
        for name, runs in self.results.items():
            if not runs:
                lines.append(f"| {name} | (failed) | - | - |")
                continue
            last = runs[-1]
            acc = last.get("accuracy")
            acc_str = f"{acc:.3f}" if acc is not None else "n/a"
            n_lab = last.get("train_size_after",
                             last.get("train_size_before", 0))
            n_iter = len(runs) - 1  # exclude init record
            lines.append(f"| {name} | {acc_str} | {n_lab} | {n_iter} |")
        lines.append("")

        # Find best / worst
        valid = [(name, runs[-1].get("accuracy"))
                 for name, runs in self.results.items()
                 if runs and runs[-1].get("accuracy") is not None]
        if valid:
            best = max(valid, key=lambda x: x[1])
            worst = min(valid, key=lambda x: x[1])
            lines.append("## Summary\n")
            lines.append(f"- Best strategy: **{best[0]}** ({best[1]:.3f})")
            lines.append(f"- Worst strategy: **{worst[0]}** ({worst[1]:.3f})")
            lines.append("")

        lines.append("## Learning Curves\n")
        lines.append(self.plot_learning_curves())
        lines.append("")
        return "\n".join(lines)


# ============================================================================
# Top-level convenience functions
# ============================================================================

def create(model, strategy: Union[str, QueryStrategy] = "uncertainty",
           **kwargs) -> ActiveLearner:
    """Build an ActiveLearner.

    Same as ``ActiveLearner(model, strategy=strategy, **kwargs)`` but with
    a slightly shorter name for interactive use.
    """
    return ActiveLearner(model=model, strategy=strategy, **kwargs)


def compare_strategies(model_factory: Callable[[], Any], X, y,
                       strategies: Optional[Union[Dict[str, QueryStrategy],
                                                   List[QueryStrategy]]] = None,
                       n_iterations: int = 20,
                       n_per_iter: int = 5,
                       test_size: float = 0.3,
                       initial_size: float = 0.05,
                       random_state: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Run all strategies head-to-head; return ``{name: [step_results]}``.

    Args:
        model_factory: zero-arg callable returning a fresh untrained model.
        X, y: full labeled dataset.
        strategies: dict {name: strategy} or list (defaults to a reasonable mix).
        n_iterations: how many query rounds per strategy.
        n_per_iter: how many samples to label per round.
        test_size: held-out test fraction.
        initial_size: initial labeled seed fraction.
        random_state: seed for the data split (same split for all strategies).
    """
    if strategies is None:
        # Default comparison set
        strategies = {
            "uncertainty": UncertaintySampling(mode="entropy"),
            "random": RandomSampling(random_state=random_state),
            "density": DensityWeighted(),
            "coreset": CoreSet(),
            "qbc": QueryByCommittee(n_members=3, random_state=random_state),
        }
    elif isinstance(strategies, (list, tuple)):
        strategies = {getattr(s, "name", f"strat_{i}"): s
                       for i, s in enumerate(strategies)}

    exp = ActiveLearningExperiment(
        model_factory=model_factory,
        strategies=strategies,
        X=X, y=y,
        test_size=test_size,
        initial_size=initial_size,
        random_state=random_state,
    )
    return exp.run(n_iterations=n_iterations, n_per_iter=n_per_iter)


def quick(model, X, y, n_query: int = 5) -> List[int]:
    """Quick query: pick the n_query most uncertain samples from X.

    Uses the first 10% of X (at least 1 sample) as the implicit "already
    labeled" set and queries from the remaining 90%. Returns indices into
    the original X array.
    """
    X_arr = _to_2d(X)
    n_total = len(X_arr)
    if n_total == 0:
        return []
    n_train = max(1, n_total // 10)
    X_pool = X_arr[n_train:]
    if len(X_pool) == 0:
        return []
    strat = UncertaintySampling(mode="entropy")
    pool_local = strat.select(model, X_pool, n=n_query)
    return [int(i) + n_train for i in pool_local]


__all__ = [
    # Base + strategies
    "QueryStrategy",
    "UncertaintySampling",
    "QueryByCommittee",
    "ExpectedModelChange",
    "DensityWeighted",
    "BALD",
    "CoreSet",
    "RandomSampling",
    "HybridStrategy",
    # Orchestration
    "ActiveLearner",
    "LabelOracle",
    "ActiveLearningExperiment",
    # Convenience
    "create",
    "compare_strategies",
    "quick",
]
