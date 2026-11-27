"""
qai.creative_tools — advanced creative features for the ML workflow.

Provides:
  - DataAugmentor — automatic data augmentation for any data type
  - FeatureImportanceRanker — rank features by importance (permutation, SHAP-like)
  - CompressionAdvisor — recommend best compression strategy for a model
  - AutoHyperparameterOptimizer — search hyperparameter space intelligently
  - ModelDistiller — knowledge distillation (teacher → student)
  - CurriculumScheduler — curriculum learning (easy → hard samples)

Usage:
    import qai

    # Data augmentation
    aug = qai.DataAugmentor(method="mixup", alpha=0.2)
    X_aug, y_aug = aug.augment(X, y, n_augmented=100)

    # Feature importance
    ranker = qai.FeatureImportanceRanker(model)
    rankings = ranker.rank(X, y, method="permutation")
    print(rankings)

    # Compression advisor
    advisor = qai.CompressionAdvisor(model)
    plan = advisor.recommend(target_size_mb=10)
    print(plan)

    # Knowledge distillation
    distiller = qai.ModelDistiller(teacher=big_model, student=small_model)
    distiller.distill(X, y, epochs=10)

    # Curriculum learning
    sched = qai.CurriculumScheduler(difficulty_fn=lambda x: abs(x[0]))
    batches = sched.get_curriculum_batches(X, y, n_stages=3)
"""
from __future__ import annotations
import os, time, math, random, copy
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np

try:
    import torch, torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


# =====================================================================
# 1. Data Augmentor
# =====================================================================
class DataAugmentor:
    """Automatic data augmentation for any data type.

    Methods:
      - mixup: interpolate pairs of samples (linear combination)
      - cutmix: replace a patch of one sample with another
      - noise: add Gaussian noise
      - jitter: small random shifts in feature values
      - smote: synthetic minority oversampling (for imbalanced data)
      - ensemble: combine multiple methods

    Usage:
        aug = qai.DataAugmentor(method="mixup", alpha=0.2)
        X_aug, y_aug = aug.augment(X, y, n_augmented=100)
    """
    def __init__(self, method: str = "mixup", alpha: float = 0.2,
                 noise_std: float = 0.01):
        self.method = method
        self.alpha = alpha
        self.noise_std = noise_std

    def augment(self, X, y=None, n_augmented: int = 50) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Augment data and return original + augmented.

        Args:
            X: input data (N, D)
            y: optional labels
            n_augmented: number of augmented samples to generate

        Returns:
            (X_combined, y_combined) — original + augmented
        """
        X = np.asarray(X, dtype=np.float32)
        if y is not None:
            y = np.asarray(y)

        if self.method == "mixup":
            X_aug, y_aug = self._mixup(X, y, n_augmented)
        elif self.method == "cutmix":
            X_aug, y_aug = self._cutmix(X, y, n_augmented)
        elif self.method == "noise":
            X_aug, y_aug = self._noise(X, y, n_augmented)
        elif self.method == "jitter":
            X_aug, y_aug = self._jitter(X, y, n_augmented)
        elif self.method == "smote":
            X_aug, y_aug = self._smote(X, y, n_augmented)
        elif self.method == "ensemble":
            # Combine mixup + noise + jitter
            X1, y1 = self._mixup(X, y, n_augmented // 3)
            X2, y2 = self._noise(X, y, n_augmented // 3)
            X3, y3 = self._jitter(X, y, n_augmented - 2 * (n_augmented // 3))
            X_aug = np.concatenate([X1, X2, X3], axis=0)
            y_aug = np.concatenate([y1, y2, y3], axis=0) if y is not None else None
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Combine original + augmented
        X_combined = np.concatenate([X, X_aug], axis=0)
        y_combined = np.concatenate([y, y_aug], axis=0) if y is not None else None
        return X_combined, y_combined

    def _mixup(self, X, y, n):
        """Mixup: λ*X[i] + (1-λ)*X[j]"""
        n_orig = len(X)
        idx1 = np.random.randint(0, n_orig, n)
        idx2 = np.random.randint(0, n_orig, n)
        lam = np.random.beta(self.alpha, self.alpha, size=(n, 1)).astype(np.float32)
        X_aug = lam * X[idx1] + (1 - lam) * X[idx2]
        y_aug = None
        if y is not None:
            if y.dtype.kind in ("i", "u"):  # classification
                y_aug = y[idx1]  # keep label of mixed sample (simplified)
            else:
                y_aug = lam.flatten() * y[idx1] + (1 - lam.flatten()) * y[idx2]
        return X_aug, y_aug

    def _cutmix(self, X, y, n):
        """CutMix: replace a random subset of features"""
        n_orig = len(X)
        idx1 = np.random.randint(0, n_orig, n)
        idx2 = np.random.randint(0, n_orig, n)
        X_aug = X[idx1].copy()
        n_features = X.shape[1]
        cut_size = max(1, n_features // 4)
        for i in range(n):
            start = np.random.randint(0, n_features - cut_size + 1)
            X_aug[i, start:start+cut_size] = X[idx2[i], start:start+cut_size]
        y_aug = y[idx1] if y is not None else None
        return X_aug, y_aug

    def _noise(self, X, y, n):
        """Gaussian noise augmentation"""
        idx = np.random.randint(0, len(X), n)
        noise = np.random.normal(0, self.noise_std, X[idx].shape).astype(np.float32)
        X_aug = X[idx] + noise
        y_aug = y[idx] if y is not None else None
        return X_aug, y_aug

    def _jitter(self, X, y, n):
        """Small random shifts in feature values"""
        idx = np.random.randint(0, len(X), n)
        shifts = np.random.uniform(-self.noise_std, self.noise_std, X[idx].shape).astype(np.float32)
        X_aug = X[idx] * (1 + shifts)
        y_aug = y[idx] if y is not None else None
        return X_aug, y_aug

    def _smote(self, X, y, n):
        """SMOTE: synthetic minority oversampling"""
        if y is None:
            return self._noise(X, y, n)  # fallback
        # Find minority class
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        minority_X = X[y == minority_class]
        if len(minority_X) < 2:
            return self._noise(X, y, n)  # fallback
        # Generate synthetic samples
        X_aug = []
        y_aug = []
        for _ in range(n):
            i, j = np.random.choice(len(minority_X), 2, replace=False)
            lam = np.random.uniform(0, 1)
            synthetic = minority_X[i] + lam * (minority_X[j] - minority_X[i])
            X_aug.append(synthetic)
            y_aug.append(minority_class)
        return np.array(X_aug, dtype=np.float32), np.array(y_aug)


# =====================================================================
# 2. Feature Importance Ranker
# =====================================================================
class FeatureImportanceRanker:
    """Rank features by importance using multiple methods.

    Methods:
      - permutation: permute each feature, measure accuracy drop
      - built-in: use model's built-in feature_importances_ (if available)
      - SHAP-like: approximate SHAP values via ablation
      - correlation: correlation with target

    Usage:
        ranker = qai.FeatureImportanceRanker(model)
        rankings = ranker.rank(X, y, method="permutation")
        print(f"Top features: {rankings[:3]}")
    """
    def __init__(self, model):
        self.model = model

    def rank(self, X, y=None, method: str = "permutation",
             n_samples: int = 100) -> List[Dict[str, Any]]:
        """Rank features by importance.

        Returns:
            List of {"feature": int, "name": str, "importance": float}
            sorted by importance (descending)
        """
        X = np.asarray(X, dtype=np.float32)
        if y is not None:
            y = np.asarray(y)
        n = min(len(X), n_samples)
        X_eval = X[:n]
        y_eval = y[:n] if y is not None else None

        if method == "permutation":
            scores = self._permutation_importance(X_eval, y_eval)
        elif method == "built-in":
            scores = self._builtin_importance()
        elif method == "correlation":
            scores = self._correlation_importance(X_eval, y_eval)
        elif method == "ablation":
            scores = self._ablation_importance(X_eval, y_eval)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Sort by importance (descending)
        ranked = sorted(scores, key=lambda x: -x["importance"])
        return ranked

    def _permutation_importance(self, X, y) -> List[Dict[str, Any]]:
        """Permutation importance: permute each feature, measure accuracy drop."""
        # Baseline accuracy
        baseline_preds = []
        for i in range(len(X)):
            try:
                baseline_preds.append(self.model.predict(X[i]))
            except Exception:
                baseline_preds.append(None)
        if y is not None:
            baseline_acc = sum(1 for p, t in zip(baseline_preds, y)
                               if p is not None and p == t) / max(len(y), 1)
        else:
            baseline_acc = 0.0

        results = []
        for feat in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feat])
            perm_preds = []
            for i in range(len(X_permuted)):
                try:
                    perm_preds.append(self.model.predict(X_permuted[i]))
                except Exception:
                    perm_preds.append(None)
            if y is not None:
                perm_acc = sum(1 for p, t in zip(perm_preds, y)
                               if p is not None and p == t) / max(len(y), 1)
                importance = baseline_acc - perm_acc
            else:
                importance = 0.0
            results.append({
                "feature": feat,
                "name": f"x{feat}",
                "importance": float(importance),
            })
        return results

    def _builtin_importance(self) -> List[Dict[str, Any]]:
        """Use model's built-in feature_importances_."""
        tech = self.model.technique
        if hasattr(tech, "model") and hasattr(tech.model, "feature_importances_"):
            importances = tech.model.feature_importances_
            return [{"feature": i, "name": f"x{i}", "importance": float(imp)}
                    for i, imp in enumerate(importances)]
        return []

    def _correlation_importance(self, X, y) -> List[Dict[str, Any]]:
        """Correlation with target."""
        if y is None:
            return [{"feature": i, "name": f"x{i}", "importance": 0.0}
                    for i in range(X.shape[1])]
        results = []
        for feat in range(X.shape[1]):
            corr = abs(np.corrcoef(X[:, feat], y)[0, 1]) if len(np.unique(y)) > 1 else 0
            results.append({"feature": feat, "name": f"x{feat}", "importance": float(corr)})
        return results

    def _ablation_importance(self, X, y) -> List[Dict[str, Any]]:
        """Ablation: zero out each feature, measure prediction change."""
        baseline_preds = []
        for i in range(len(X)):
            try:
                baseline_preds.append(self.model.predict(X[i]))
            except Exception:
                baseline_preds.append(None)

        results = []
        for feat in range(X.shape[1]):
            X_ablated = X.copy()
            X_ablated[:, feat] = 0  # zero out feature
            ablated_preds = []
            for i in range(len(X_ablated)):
                try:
                    ablated_preds.append(self.model.predict(X_ablated[i]))
                except Exception:
                    ablated_preds.append(None)
            # Measure prediction change
            changes = sum(1 for p1, p2 in zip(baseline_preds, ablated_preds)
                          if p1 != p2) / max(len(baseline_preds), 1)
            results.append({"feature": feat, "name": f"x{feat}", "importance": float(changes)})
        return results


# =====================================================================
# 3. Compression Advisor
# =====================================================================
class CompressionAdvisor:
    """Recommend the best compression strategy for a model.

    Analyzes the model and recommends:
      - Quantization method (int8, int4, GPTQ, AWQ)
      - Pruning strategy (structured, unstructured, magnitude)
      - Knowledge distillation (if model is large enough)
      - Expected size reduction + quality loss

    Usage:
        advisor = qai.CompressionAdvisor(model)
        plan = advisor.recommend(target_size_mb=10)
        print(plan)
    """
    def __init__(self, model):
        self.model = model
        try:
            from .model_tools import count_parameters
            self.info = count_parameters(model)
        except Exception:
            self.info = {"total": 0, "total_mb": 0}

    def recommend(self, target_size_mb: Optional[float] = None,
                  max_quality_loss: float = 0.02) -> Dict[str, Any]:
        """Recommend compression strategy.

        Args:
            target_size_mb: target model size in MB (None = aggressive)
            max_quality_loss: maximum acceptable quality loss (0-1)

        Returns:
            Dict with recommended strategy + expected outcomes
        """
        current_size = self.info["total_mb"]
        n_params = self.info["total"]

        strategies = []

        # 1. Quantization options
        if current_size > 0:
            int8_size = current_size / 4  # 32bit → 8bit = 4x reduction
            int4_size = current_size / 8  # 32bit → 4bit = 8x reduction
            strategies.append({
                "method": "int8_dynamic",
                "expected_size_mb": int8_size,
                "expected_quality_loss": 0.005,  # ~0.5%
                "requires_retraining": False,
                "recommendation": "Good default — fast, no retraining, minimal loss",
            })
            strategies.append({
                "method": "int4_nf4",
                "expected_size_mb": int4_size,
                "expected_quality_loss": 0.03,  # ~3%
                "requires_retraining": False,
                "recommendation": "For maximum compression — use QAT to minimize loss",
            })
            strategies.append({
                "method": "gptq",
                "expected_size_mb": int4_size * 1.1,
                "expected_quality_loss": 0.01,  # ~1%
                "requires_retraining": False,
                "recommendation": "Best quality at 4-bit — uses Hessian info",
            })
            strategies.append({
                "method": "awq",
                "expected_size_mb": int4_size * 1.1,
                "expected_quality_loss": 0.015,
                "requires_retraining": False,
                "recommendation": "Good for activation-sensitive models",
            })

        # 2. Pruning
        strategies.append({
            "method": "magnitude_pruning",
            "expected_size_mb": current_size * 0.7,
            "expected_quality_loss": 0.01,
            "requires_retraining": True,
            "recommendation": "Remove small weights — needs fine-tuning after",
        })

        # 3. QAT (if quality loss is too high)
        if max_quality_loss < 0.02:
            strategies.append({
                "method": "qat_then_int8",
                "expected_size_mb": int8_size,
                "expected_quality_loss": 0.002,
                "requires_retraining": True,
                "recommendation": "QAT first, then quantize — near-zero loss",
            })

        # 4. Knowledge distillation (for large models)
        if n_params > 1_000_000:
            strategies.append({
                "method": "distillation",
                "expected_size_mb": current_size * 0.25,
                "expected_quality_loss": 0.02,
                "requires_retraining": True,
                "recommendation": "Train a smaller student model from this teacher",
            })

        # Pick best strategy for target
        best = None
        if target_size_mb:
            for s in strategies:
                if s["expected_size_mb"] <= target_size_mb and s["expected_quality_loss"] <= max_quality_loss:
                    if best is None or s["expected_quality_loss"] < best["expected_quality_loss"]:
                        best = s
        if best is None:
            best = strategies[0] if strategies else None

        return {
            "current_size_mb": current_size,
            "n_parameters": n_params,
            "target_size_mb": target_size_mb,
            "max_quality_loss": max_quality_loss,
            "best_strategy": best,
            "all_strategies": strategies,
        }


# =====================================================================
# 4. Knowledge Distillation
# =====================================================================
class ModelDistiller:
    """Teacher→student distillation via the Trainer API.

    Backward-compatible delegate — the real implementation lives in
    ``qai.distillation_v2`` (canonical since v1.8.5). This class forwards
    to it, preserving the old ``(teacher, student, alpha, temperature, lr)``
    constructor and ``distill(X, y, epochs, batch_size, verbose) -> student``
    signature.
    """

    def __init__(self, teacher, student, alpha: float = 0.5,
                 temperature: float = 4.0, lr: float = 1e-3):
        self.teacher = teacher
        self.student = student
        self.alpha = alpha  # weight for hard label loss
        self.temperature = temperature  # softmax temperature for soft labels
        self.lr = lr
        from .distillation_v2 import Distiller as _Canonical, DistillationConfig as _Config
        self._config = _Config(
            temperature=temperature, alpha=alpha, lr=lr, strategy="vanilla",
        )
        self._core = _Canonical(teacher, student, self._config)
        self.last_result = None

    def distill(self, X, y=None, epochs: int = 10, batch_size: int = 32,
                verbose: bool = True):
        """Run knowledge distillation (delegates to qai.distillation_v2).

        Args:
            X: training data
            y: optional hard labels
            epochs: number of distillation epochs
        """
        self._config.epochs = epochs
        self._config.batch_size = batch_size
        self._config.verbose = verbose
        self.last_result = self._core.distill(X, y)
        if verbose:
            print(f"[Distiller] student distilled for {epochs} epochs "
                  f"(mode: {self.last_result.distillation_mode})")
            print(f"  alpha={self.alpha} (hard label weight)  "
                  f"temperature={self.temperature}")
        return self.student


# =====================================================================
# 5. Curriculum Scheduler
# =====================================================================
class CurriculumScheduler:
    """Curriculum learning: train on easy samples first, then harder ones.

    The idea: like humans, models learn better when presented with easy
    examples first, gradually increasing difficulty.

    Usage:
        sched = qai.CurriculumScheduler(difficulty_fn=lambda x: abs(x[0]))
        batches = sched.get_curriculum_batches(X, y, n_stages=3)
        for stage, (X_batch, y_batch) in enumerate(batches):
            print(f"Stage {stage}: {len(X_batch)} samples")
            model.train(X_batch, y_batch)
    """
    def __init__(self, difficulty_fn: Callable, reverse: bool = False):
        """
        Args:
            difficulty_fn: function(sample) → float (higher = harder)
            reverse: if True, start with hard samples first
        """
        self.difficulty_fn = difficulty_fn
        self.reverse = reverse

    def get_curriculum_batches(self, X, y=None, n_stages: int = 3,
                                batch_ratio: float = 1.0) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Split data into difficulty-ordered batches.

        Returns:
            List of (X_batch, y_batch) from easiest to hardest
        """
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        # Compute difficulty for each sample
        difficulties = []
        for i in range(len(X)):
            try:
                d = self.difficulty_fn(X[i])
            except Exception:
                d = 0.0
            difficulties.append((d, i))
        # Sort by difficulty
        difficulties.sort(key=lambda x: x[0], reverse=self.reverse)
        # Split into stages
        n = len(difficulties)
        stage_size = n // n_stages
        batches = []
        for stage in range(n_stages):
            start = stage * stage_size
            end = (stage + 1) * stage_size if stage < n_stages - 1 else n
            indices = [idx for _, idx in difficulties[start:end]]
            X_stage = X[indices]
            y_stage = y[indices] if y is not None else None
            batches.append((X_stage, y_stage))
        return batches


__all__ = ["DataAugmentor", "FeatureImportanceRanker", "CompressionAdvisor",
           "ModelDistiller", "CurriculumScheduler"]
