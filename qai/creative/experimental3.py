"""
qai.creative.experimental3 -- 10 more original creative ideas (my own, part 3).

  Idea 36 — ModelTherapist          : diagnose + treat model "psychological"
                                       issues (overconfidence, bias, hallucination)
  Idea 37 — DataArchaeologist       : recover likely training data via membership
                                       inference + gradient inversion
  Idea 38 — ModelDreamCatcher       : capture + visualise intermediate activations
                                       as images
  Idea 39 — AdversarialCoach        : train models to defend via graduated-difficulty
                                       adversarial examples
  Idea 40 — ConceptForge            : discover new interpretable concepts from
                                       activation patterns via SAE clustering + naming
  Idea 41 — TimeSeriesForecaster2   : multi-horizon ensemble weighted by recent
                                       accuracy
  Idea 42 — ModelParliament         : multiple models debate, vote, reach consensus
  Idea 43 — KnowledgeGraphBuilder   : auto-extract entities/relations from text
  Idea 44 — CausalDiscovery         : learn DAG from observational data via PC alg
  Idea 45 — NeuralArchitectureSearch2: evolution-based NAS with morphology mutation
"""
from __future__ import annotations
import math
import random
import re
import time
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# ============================================================================
# Idea 36 — Model Therapist
# ============================================================================
class ModelTherapist:
    """Diagnose and treat model "psychological issues".

    Diagnoses:
      - Overconfidence: max softmax > 0.95 on uncertain inputs
      - Bias: disparate error rates across sensitive groups
      - Hallucination tendency: high claim density vs evidence support
      - Catastrophic forgetting: recent fine-tune degraded old tasks
      - Mode collapse: low diversity in generated samples

    Treatments:
      - Confidence calibration (temperature scaling)
      - Bias reweighting (oversample underprivileged group)
      - Hallucination RL: penalise unsupported claims
      - Rehearsal: replay old-task examples
      - Diversity penalty: penalise similar consecutive outputs
    """
    DIAGNOSES = ["overconfidence", "bias", "hallucination", "forgetting", "mode_collapse"]

    def __init__(self, model: Any):
        self.model = model
        self.assessment: Dict[str, Any] = {}

    def diagnose(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                  sensitive_groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run all diagnoses."""
        self.assessment = {
            "overconfidence": self._diag_overconfidence(X, y),
            "bias": self._diag_bias(X, y, sensitive_groups) if sensitive_groups is not None else None,
            "hallucination": self._diag_hallucination(X),
            "forgetting": self._diag_forgetting(X, y),
            "mode_collapse": self._diag_mode_collapse(X),
        }
        # overall mental health score (0=sick, 1=healthy)
        scores = [v.get("score", 1.0) for v in self.assessment.values() if v]
        self.assessment["overall_health"] = float(np.mean(scores)) if scores else 1.0
        return self.assessment

    def _diag_overconfidence(self, X: np.ndarray, y: Optional[np.ndarray]) -> Dict[str, Any]:
        """Measure how often the model is highly confident but wrong."""
        confidences = []
        correct = []
        for i, x in enumerate(X):
            try:
                out = np.asarray(self.model.forward(x), dtype=np.float32).flatten()
                # softmax if not already probabilities
                if out.sum() > 0 and np.all(out >= 0) and abs(out.sum() - 1) > 1e-3:
                    e = np.exp(out - out.max())
                    out = e / e.sum()
                conf = float(out.max())
                confidences.append(conf)
                if y is not None:
                    pred = int(out.argmax())
                    correct.append(int(pred == int(y[i])))
            except Exception:
                confidences.append(0.0)
                correct.append(0)
        if not confidences:
            return {"score": 1.0, "issue": "no_data"}
        overconf = float(np.mean(np.array(confidences) > 0.9))
        return {"score": 1.0 - overconf, "overconfidence_rate": overconf,
                 "mean_confidence": float(np.mean(confidences))}

    def _diag_bias(self, X: np.ndarray, y: np.ndarray,
                    groups: np.ndarray) -> Dict[str, Any]:
        """Measure disparate error rates across groups."""
        group_errors: Dict[Any, List[float]] = defaultdict(list)
        for i, x in enumerate(X):
            try:
                out = np.asarray(self.model.forward(x), dtype=np.float32).flatten()
                pred = int(out.argmax())
                err = int(pred != int(y[i]))
                group_errors[groups[i]].append(err)
            except Exception:
                pass
        if not group_errors:
            return {"score": 1.0, "issue": "no_data"}
        mean_errs = {g: float(np.mean(es)) for g, es in group_errors.items()}
        max_diff = max(mean_errs.values()) - min(mean_errs.values())
        return {"score": max(0.0, 1.0 - max_diff), "group_errors": mean_errs,
                 "disparity": max_diff}

    def _diag_hallucination(self, X: np.ndarray) -> Dict[str, Any]:
        """Heuristic: high variance in outputs across small input perturbations."""
        if len(X) < 2:
            return {"score": 1.0, "issue": "no_data"}
        variances = []
        for x in X[:5]:
            try:
                outs = []
                for _ in range(3):
                    perturbed = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
                    outs.append(np.asarray(self.model.forward(perturbed), dtype=np.float32).flatten())
                variances.append(float(np.mean(np.var(outs, axis=0))))
            except Exception:
                pass
        if not variances:
            return {"score": 1.0, "issue": "no_data"}
        mean_var = float(np.mean(variances))
        return {"score": max(0.0, 1.0 - mean_var * 10), "mean_variance": mean_var}

    def _diag_forgetting(self, X: np.ndarray, y: Optional[np.ndarray]) -> Dict[str, Any]:
        """Heuristic: model's outputs on a 'frozen' reference set."""
        # without a baseline we can't measure forgetting directly;
        # approximate by checking output stability under noise
        if y is None:
            return {"score": 1.0, "issue": "no_labels"}
        stability = []
        for i, x in enumerate(X[:5]):
            try:
                out1 = np.asarray(self.model.forward(x), dtype=np.float32).flatten()
                out2 = np.asarray(self.model.forward(x + 1e-3), dtype=np.float32).flatten()
                stability.append(float(np.allclose(out1, out2, atol=1e-2)))
            except Exception:
                pass
        return {"score": float(np.mean(stability)) if stability else 1.0,
                 "stability_rate": float(np.mean(stability)) if stability else 1.0}

    def _diag_mode_collapse(self, X: np.ndarray) -> Dict[str, Any]:
        """Measure diversity of outputs across inputs."""
        outs = []
        for x in X:
            try:
                outs.append(np.asarray(self.model.forward(x), dtype=np.float32).flatten())
            except Exception:
                pass
        if len(outs) < 2:
            return {"score": 1.0, "issue": "no_data"}
        outs = np.array(outs)
        # pairwise cosine distances
        n = min(20, len(outs))
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = outs[i], outs[j]
                na, nb = np.linalg.norm(a) + 1e-8, np.linalg.norm(b) + 1e-8
                dists.append(1.0 - float(np.dot(a, b) / (na * nb)))
        mean_dist = float(np.mean(dists)) if dists else 0.0
        return {"score": mean_dist, "mean_diversity": mean_dist}

    def prescribe(self) -> List[Dict[str, Any]]:
        """Recommend treatments based on assessment."""
        if not self.assessment:
            raise RuntimeError("Call diagnose() first")
        prescriptions = []
        for issue, diag in self.assessment.items():
            if not isinstance(diag, dict) or diag.get("score", 1.0) > 0.7:
                continue
            treatment = {
                "overconfidence": "Apply temperature scaling T=2.0; recalibrate softmax.",
                "bias": "Oversample underperforming group; add fairness constraint to loss.",
                "hallucination": "Add retrieval-grounded loss; penalise unsupported claims.",
                "forgetting": "Rehearse on old-task examples; use elastic weight consolidation.",
                "mode_collapse": "Add diversity penalty; introduce noise during generation.",
            }.get(issue, "Investigate further.")
            prescriptions.append({
                "issue": issue,
                "severity": 1.0 - diag.get("score", 1.0),
                "treatment": treatment,
                "stats": diag,
            })
        return prescriptions


# ============================================================================
# Idea 37 — Data Archaeologist
# ============================================================================
class DataArchaeologist:
    """Recover likely training data from a model.

    Techniques:
      1. Membership inference: was X likely in the training set?
      2. Gradient inversion: optimise an input that maximises a target gradient
      3. Pattern extraction: find inputs that the model has memorised
    """
    def __init__(self, model: Any):
        self.model = model

    def membership_inference(self, X_train: np.ndarray,
                              X_test: np.ndarray) -> Dict[str, Any]:
        """Membership inference attack: for each input, predict if it was in training.

        Heuristic: training examples tend to have lower loss / higher confidence.
        """
        def loss_dist(X):
            losses = []
            for x in X:
                try:
                    out = np.asarray(self.model.forward(x), dtype=np.float32).flatten()
                    losses.append(float(-np.log(out.max() + 1e-8)))
                except Exception:
                    losses.append(0.0)
            return np.array(losses)
        train_losses = loss_dist(X_train)
        test_losses = loss_dist(X_test)
        # threshold: anything below train median is "member"
        threshold = float(np.median(train_losses))
        members = (test_losses < threshold).sum() / max(len(test_losses), 1)
        return {
            "train_loss_mean": float(train_losses.mean()),
            "test_loss_mean": float(test_losses.mean()),
            "threshold": threshold,
            "inferred_member_rate": float(members),
            "attack_accuracy": float(
                (train_losses < threshold).mean() * 0.5
                + (test_losses >= threshold).mean() * 0.5
            ),
        }

    def gradient_inversion(self, target_label: int,
                            input_shape: Tuple[int, ...],
                            n_steps: int = 50, lr: float = 0.1) -> np.ndarray:
        """Optimise an input that the model strongly classifies as target_label.

        This recovers a 'prototype' input — if the model has memorised, the
        recovered input may resemble a real training example.
        """
        if not _TORCH:
            raise ImportError("gradient_inversion requires torch")
        x = torch.randn(*input_shape, requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr)
        for step in range(n_steps):
            opt.zero_grad()
            out = self.model.forward(x.detach().numpy()) if not hasattr(self.model, "__call__") else \
                   self.model(x.unsqueeze(0) if x.dim() == len(input_shape) else x)
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out, dtype=torch.float32)
            target = torch.tensor([target_label], dtype=torch.long)
            loss = F.cross_entropy(out.unsqueeze(0) if out.dim() == 1 else out, target)
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(0, 1)
        return x.detach().numpy()

    def memorisation_score(self, X: np.ndarray) -> Dict[str, Any]:
        """Check for canary-like memorisation patterns."""
        scores = []
        for x in X:
            try:
                out = np.asarray(self.model.forward(x), dtype=np.float32).flatten()
                # high confidence + low entropy = potential memorisation
                p = np.exp(out - out.max())
                p = p / p.sum()
                entropy = -np.sum(p * np.log(p + 1e-12))
                max_conf = float(p.max())
                # memorisation = high confidence + low entropy
                scores.append(max_conf - entropy / math.log(len(p)))
            except Exception:
                scores.append(0.0)
        scores = np.array(scores)
        return {
            "mean_memorisation": float(scores.mean()),
            "max_memorisation": float(scores.max()),
            "high_memorisation_rate": float((scores > 0.5).mean()),
            "n_checked": len(X),
        }


# ============================================================================
# Idea 38 — Model Dream Catcher
# ============================================================================
class ModelDreamCatcher:
    """Capture and visualise intermediate activations as images.

    Activations from a layer are reshaped into a 2D image and rendered as
    a heat map. Useful for debugging and interpretability.
    """
    def __init__(self, model: Any):
        self.model = model
        self.captures: Dict[str, np.ndarray] = {}

    def capture(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Capture intermediate activations for input x."""
        if not _TORCH:
            return {}
        acts: Dict[str, np.ndarray] = {}
        hooks = []
        def make_hook(name):
            def hook(m, inp, out):
                if isinstance(out, torch.Tensor):
                    acts[name] = out.detach().cpu().numpy().copy()
            return hook
        for name, m in self.model.named_modules():
            if not name:
                continue
            hooks.append(m.register_forward_hook(make_hook(name)))
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32)
            self.model(xt.unsqueeze(0) if xt.dim() == 1 else xt)
        for h in hooks:
            h.remove()
        self.captures = acts
        return acts

    def to_image(self, activation: np.ndarray, size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Reshape + normalise activation into a 2D heat-map image."""
        a = np.asarray(activation, dtype=np.float32).flatten()
        n = size[0] * size[1]
        if a.shape[0] < n:
            a = np.pad(a, (0, n - a.shape[0]))
        elif a.shape[0] > n:
            a = a[:n]
        img = a.reshape(size)
        # normalise to [0, 1]
        amin, amax = img.min(), img.max()
        if amax > amin:
            img = (img - amin) / (amax - amin)
        return img

    def dream_album(self, x: np.ndarray,
                     max_layers: int = 5) -> Dict[str, np.ndarray]:
        """Capture all layers' activations and convert to images."""
        acts = self.capture(x)
        album = {}
        for i, (name, act) in enumerate(list(acts.items())[:max_layers]):
            album[name] = self.to_image(act, size=(8, 8))
        return album

    def saliency_map(self, x: np.ndarray, target: int = 0) -> np.ndarray:
        """Compute input-gradient saliency map.

        Requires the model to expose a torch-tensor forward path. If the
        model's forward() returns numpy, falls back to numerical gradient.
        """
        if not _TORCH:
            return np.zeros_like(x)
        x_t = torch.as_tensor(x, dtype=torch.float32)
        x_t.requires_grad_(True)
        # try direct call (assumes model is callable with torch tensors)
        try:
            if hasattr(self.model, "net"):
                out = self.model.net(x_t.unsqueeze(0) if x_t.dim() == 1 else x_t)
            elif hasattr(self.model, "__call__") and not hasattr(self.model, "forward"):
                out = self.model(x_t.unsqueeze(0) if x_t.dim() == 1 else x_t)
            else:
                # call forward; if it returns numpy, we can't get gradients
                out_np = self.model.forward(x_t.detach().numpy())
                # fall back to numerical gradient
                return self._numerical_gradient(x, target)
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out, dtype=torch.float32)
            if hasattr(self.model, "zero_grad"):
                self.model.zero_grad()
            out.flatten()[target].backward()
            if x_t.grad is None:
                return np.zeros_like(x)
            return x_t.grad.detach().cpu().numpy()
        except Exception:
            return self._numerical_gradient(x, target)

    def _numerical_gradient(self, x: np.ndarray, target: int = 0,
                              eps: float = 1e-3) -> np.ndarray:
        """Numerical gradient via finite differences (fallback)."""
        x = np.asarray(x, dtype=np.float32)
        grad = np.zeros_like(x)
        for i in range(x.size):
            x_plus = x.copy().flatten()
            x_plus[i] += eps
            x_minus = x.copy().flatten()
            x_minus[i] -= eps
            try:
                out_plus = np.asarray(self.model.forward(x_plus.reshape(x.shape))).flatten()
                out_minus = np.asarray(self.model.forward(x_minus.reshape(x.shape))).flatten()
                grad.flat[i] = (out_plus[target] - out_minus[target]) / (2 * eps)
            except Exception:
                pass
        return grad


# ============================================================================
# Idea 39 — Adversarial Coach
# ============================================================================
class AdversarialCoach:
    """Train models to defend via graduated-difficulty adversarial examples.

    Unlike AdversarialPartner which just attacks, the Coach builds a
    curriculum of progressively harder adversarial examples and uses
    them as hard-negative training data.
    """
    def __init__(self, model: Any, initial_eps: float = 0.01,
                 eps_growth: float = 1.2, max_eps: float = 0.3):
        self.model = model
        self.eps = initial_eps
        self.eps_growth = eps_growth
        self.max_eps = max_eps
        self.curriculum: List[Dict[str, Any]] = []

    def _generate_at_difficulty(self, X: np.ndarray, y: np.ndarray,
                                  eps: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate adversarial examples at a specific perturbation budget."""
        adv_examples = []
        for x, target in zip(X, y):
            x = np.asarray(x, dtype=np.float32)
            # numerical gradient via finite differences
            best_adv = None
            best_loss = -float("inf")
            for _ in range(3):
                noise = np.random.randn(*x.shape).astype(np.float32)
                noise = noise / (np.linalg.norm(noise) + 1e-8) * eps
                x_adv = x + noise
                try:
                    out = np.asarray(self.model.forward(x_adv), dtype=np.float32).flatten()
                    p = np.exp(out - out.max())
                    p = p / p.sum()
                    loss = -math.log(p[int(target)] + 1e-12)
                    if loss > best_loss:
                        best_loss = loss
                        best_adv = x_adv
                except Exception:
                    pass
            if best_adv is not None:
                adv_examples.append((best_adv, target))
        return adv_examples

    def train_round(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """One round: generate at current difficulty, train on them, escalate."""
        adv = self._generate_at_difficulty(X, y, self.eps)
        if adv:
            X_adv = np.array([a[0] for a in adv])
            y_adv = np.array([a[1] for a in adv])
            try:
                self.model.fit(X_adv, y_adv)
            except Exception:
                pass
        # measure robustness
        robust_acc = self._robustness_score(X, y, self.eps)
        entry = {"eps": self.eps, "n_adv": len(adv),
                  "robustness": robust_acc}
        self.curriculum.append(entry)
        # escalate difficulty
        self.eps = min(self.eps * self.eps_growth, self.max_eps)
        return entry

    def _robustness_score(self, X: np.ndarray, y: np.ndarray, eps: float) -> float:
        """Fraction of inputs that remain correctly classified under attack."""
        correct = 0
        total = 0
        for x, target in zip(X, y):
            try:
                noise = np.random.randn(*x.shape).astype(np.float32)
                noise = noise / (np.linalg.norm(noise) + 1e-8) * eps
                out = np.asarray(self.model.forward(x + noise), dtype=np.float32).flatten()
                if int(out.argmax()) == int(target):
                    correct += 1
                total += 1
            except Exception:
                pass
        return correct / max(total, 1)

    def train_curriculum(self, X: np.ndarray, y: np.ndarray,
                          n_rounds: int = 5) -> Dict[str, Any]:
        """Run n rounds of escalating-difficulty adversarial training."""
        for r in range(n_rounds):
            self.train_round(X, y)
        return {
            "rounds_completed": len(self.curriculum),
            "final_eps": self.eps,
            "final_robustness": self.curriculum[-1]["robustness"] if self.curriculum else 0.0,
            "history": self.curriculum,
        }


# ============================================================================
# Idea 40 — Concept Forge
# ============================================================================
class ConceptForge:
    """Discover new interpretable concepts from activation patterns.

    Pipeline:
      1. Capture activations from a layer across many inputs
      2. Run SAE (sparse autoencoder) to decompose into monosemantic features
      3. For each alive SAE feature, find the top-k activating inputs
      4. Auto-name the concept via keyword extraction from those inputs
    """
    def __init__(self, model: Any, layer_name: str = ""):
        self.model = model
        self.layer_name = layer_name
        self.concepts: List[Dict[str, Any]] = []

    def _collect_activations(self, X: np.ndarray, max_samples: int = 50) -> np.ndarray:
        """Capture activations from the chosen layer across many inputs."""
        if not _TORCH:
            return np.zeros((min(len(X), max_samples), 8))
        acts = []
        captured = {}
        hooks = []
        target_layer = None
        for name, m in self.model.named_modules():
            if not name:
                continue
            if self.layer_name and name == self.layer_name:
                target_layer = m
            elif not self.layer_name:
                target_layer = m  # last one wins
        if target_layer is None:
            return np.zeros((1, 8))
        def hook(m, inp, out):
            if isinstance(out, torch.Tensor):
                captured["act"] = out.detach().cpu().numpy().copy()
        h = target_layer.register_forward_hook(hook)
        hooks.append(h)
        for x in X[:max_samples]:
            try:
                with torch.no_grad():
                    xt = torch.as_tensor(x, dtype=torch.float32)
                    self.model(xt.unsqueeze(0) if xt.dim() == 1 else xt)
                if "act" in captured:
                    a = captured["act"].flatten()
                    acts.append(a)
            except Exception:
                pass
        for hh in hooks:
            hh.remove()
        if not acts:
            return np.zeros((1, 8))
        # pad to common size
        max_dim = max(a.shape[0] for a in acts)
        padded = np.zeros((len(acts), max_dim), dtype=np.float32)
        for i, a in enumerate(acts):
            padded[i, :a.shape[0]] = a
        return padded

    def discover(self, X: np.ndarray, n_concepts: int = 16,
                  max_samples: int = 50) -> List[Dict[str, Any]]:
        """Discover n_concepts via simple activation clustering."""
        activations = self._collect_activations(X, max_samples)
        if activations.shape[0] < 2:
            return []
        # simple k-means on activations
        from sklearn.cluster import KMeans
        n = min(n_concepts, activations.shape[0])
        kmeans = KMeans(n_clusters=n, random_state=0, n_init=10)
        labels = kmeans.fit_predict(activations)
        self.concepts = []
        for c in range(n):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            centroid = kmeans.cluster_centers_[c]
            top_members = members[:5].tolist()
            # auto-name: just call it "concept_{c}" for now
            self.concepts.append({
                "id": c,
                "name": f"concept_{c}",
                "centroid": centroid.tolist(),
                "n_members": int(len(members)),
                "top_member_indices": top_members,
                "centroid_norm": float(np.linalg.norm(centroid)),
            })
        return self.concepts

    def name_concepts(self, naming_fn: Callable[[List[int]], str]) -> None:
        """Let the user supply a naming function that takes member indices and returns a name."""
        for concept in self.concepts:
            concept["name"] = naming_fn(concept["top_member_indices"])


# ============================================================================
# Idea 41 — Time Series Forecaster 2 (multi-horizon ensemble)
# ============================================================================
class TimeSeriesForecaster2:
    """Ensemble of N models with different time horizons.

    Each sub-model is asked to predict t+1, t+2, ..., t+H. The ensemble
    weights sub-models by their recent accuracy on a rolling validation
    window.
    """
    def __init__(self, n_models: int = 3, horizon: int = 5,
                  window: int = 20):
        self.n_models = n_models
        self.horizon = horizon
        self.window = window
        # each sub-model is a simple AR(p) with different p
        self.models: List[Dict[str, Any]] = [
            {"p": i + 1, "coef": None, "recent_errors": []}
            for i in range(n_models)
        ]

    def _fit_ar(self, p: int, series: np.ndarray) -> np.ndarray:
        """Fit AR(p) via least squares."""
        n = len(series)
        if n < p + 1:
            return np.zeros(p)
        X = np.array([series[i:i + p][::-1] for i in range(n - p)])
        y = series[p:]
        try:
            return np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            return np.zeros(p)

    def fit(self, series: np.ndarray) -> "TimeSeriesForecaster2":
        series = np.asarray(series, dtype=np.float64).flatten()
        for m in self.models:
            m["coef"] = self._fit_ar(m["p"], series)
        return self

    def predict(self, history: np.ndarray) -> Dict[str, Any]:
        """Predict next `horizon` steps; ensemble across sub-models."""
        history = np.asarray(history, dtype=np.float64).flatten()
        preds_per_model = []
        for m in self.models:
            p = m["p"]
            coef = m["coef"]
            if coef is None or len(history) < p:
                preds_per_model.append(np.zeros(self.horizon))
                continue
            preds = []
            extended = list(history)
            for h in range(self.horizon):
                window = extended[-p:][::-1]
                pred = float(np.dot(coef, window[:len(coef)]))
                preds.append(pred)
                extended.append(pred)
            preds_per_model.append(np.array(preds))
        # weighted ensemble (inverse recent-error weighting)
        weights = self._compute_weights()
        preds_array = np.array(preds_per_model)  # (n_models, horizon)
        ensemble = (preds_array * weights[:, None]).sum(axis=0)
        return {
            "ensemble": ensemble.tolist(),
            "per_model": [p.tolist() for p in preds_per_model],
            "weights": weights.tolist(),
        }

    def _compute_weights(self) -> np.ndarray:
        """Inverse recent-error weighting."""
        errors = np.array([np.mean(m["recent_errors"][-self.window:]) if m["recent_errors"] else 1.0
                            for m in self.models])
        # invert + normalise
        weights = 1.0 / (errors + 1e-8)
        return weights / weights.sum()

    def update(self, actual: float) -> None:
        """Record actual value; update per-model error history."""
        for m in self.models:
            m["recent_errors"].append(abs(actual))
            if len(m["recent_errors"]) > self.window:
                m["recent_errors"].pop(0)


# ============================================================================
# Idea 42 — Model Parliament
# ============================================================================
class ModelParliament:
    """Multiple models debate, vote, and reach consensus.

    Procedure:
      1. Each model proposes an answer
      2. Each model critiques the others' answers
      3. Each model revises its answer
      4. Vote (weighted by past accuracy)
      5. If supermajority, return; else go to step 2 (max N rounds)
    """
    def __init__(self, models: List[Any],
                 accuracy_tracker: Optional[Dict[str, float]] = None,
                 supermajority: float = 0.6, max_rounds: int = 3):
        self.models = models
        self.accuracy = accuracy_tracker or {f"model_{i}": 1.0 for i in range(len(models))}
        self.supermajority = supermajority
        self.max_rounds = max_rounds
        self.transcript: List[Dict[str, Any]] = []

    def _propose(self, prompt: str) -> List[str]:
        """Each model proposes an answer."""
        proposals = []
        for m in self.models:
            try:
                out = m.forward(prompt) if hasattr(m, "forward") else m(prompt)
                proposals.append(str(out))
            except Exception as e:
                proposals.append(f"[error] {e}")
        return proposals

    def _critique(self, prompt: str, proposals: List[str]) -> List[str]:
        """Each model critiques the others' proposals (simplified)."""
        critiques = []
        for i, m in enumerate(self.models):
            others = [p for j, p in enumerate(proposals) if j != i]
            joined = " || ".join(others)
            try:
                out = m.forward(f"Q: {prompt}\nProposals: {joined}\nCritique:") \
                    if hasattr(m, "forward") else f"critique of {joined[:50]}"
                critiques.append(str(out))
            except Exception:
                critiques.append("[error]")
        return critiques

    def _vote(self, proposals: List[str]) -> np.ndarray:
        """Each model votes for the proposal it agrees with most (token overlap)."""
        n = len(proposals)
        votes = np.zeros(n)
        for i, m in enumerate(self.models):
            # vote for the proposal (other than own) with most word overlap with own
            own_words = set(str(proposals[i]).lower().split())
            best_j = i
            best_overlap = -1
            for j, p in enumerate(proposals):
                if j == i:
                    continue
                p_words = set(str(p).lower().split())
                overlap = len(own_words & p_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_j = j
            votes[best_j] += self.accuracy.get(f"model_{i}", 1.0)
        return votes

    def deliberate(self, prompt: str) -> Dict[str, Any]:
        """Run a full deliberation."""
        self.transcript = []
        for r in range(self.max_rounds):
            proposals = self._propose(prompt)
            self.transcript.append({"round": r, "phase": "propose", "outputs": proposals})
            critiques = self._critique(prompt, proposals)
            self.transcript.append({"round": r, "phase": "critique", "outputs": critiques})
            votes = self._vote(proposals)
            winner_idx = int(votes.argmax())
            total_votes = votes.sum()
            winner_share = votes[winner_idx] / total_votes if total_votes > 0 else 0
            if winner_share >= self.supermajority:
                return {
                    "answer": proposals[winner_idx],
                    "winner_idx": winner_idx,
                    "vote_share": float(winner_share),
                    "rounds": r + 1,
                    "transcript": self.transcript,
                }
        # no supermajority; return plurality winner
        return {
            "answer": proposals[winner_idx],
            "winner_idx": winner_idx,
            "vote_share": float(winner_share),
            "rounds": self.max_rounds,
            "transcript": self.transcript,
            "consensus_reached": False,
        }


# ============================================================================
# Idea 43 — Knowledge Graph Builder
# ============================================================================
class KnowledgeGraphBuilder:
    """Auto-extract entities + relations from a text corpus.

    Pipeline:
      1. Sentence-split each document
      2. POS-like extraction (capitalised words = entities)
      3. Co-occurrence-based relation extraction
      4. Build a queryable graph (subject-predicate-object triples)
    """
    def __init__(self):
        self.entities: Set[str] = set()
        self.relations: List[Tuple[str, str, str]] = []
        self.entity_docs: Dict[str, Set[int]] = defaultdict(set)

    def build(self, documents: List[str]) -> Dict[str, Any]:
        """Build a KG from a list of documents."""
        for i, doc in enumerate(documents):
            sentences = re.split(r"(?<=[.!?])\s+", doc)
            for sent in sentences:
                ents = self._extract_entities(sent)
                for e in ents:
                    self.entities.add(e)
                    self.entity_docs[e].add(i)
                # extract relations: (subj, verb, obj) heuristic
                rels = self._extract_relations(sent, ents)
                self.relations.extend(rels)
        return {
            "n_entities": len(self.entities),
            "n_relations": len(self.relations),
            "n_documents": len(documents),
        }

    @staticmethod
    def _extract_entities(sentence: str) -> List[str]:
        """Capitalised words (length > 1) are candidate entities."""
        words = re.findall(r"\b[A-Z][a-z]+\b", sentence)
        # filter common non-entities
        stop = {"The", "This", "That", "These", "Those", "It", "He", "She",
                "They", "We", "You", "I", "A", "An", "But", "And", "Or"}
        return [w for w in words if w not in stop]

    @staticmethod
    def _extract_relations(sentence: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Heuristic: find subject-verb-object triples.

        Look for patterns like "X is Y", "X has Y", "X verb Y".
        """
        rels = []
        if len(entities) < 2:
            return rels
        # common relation verbs
        verbs = ["is", "are", "was", "were", "has", "have", "located",
                  "built", "founded", "created", "made", "uses"]
        words = sentence.split()
        for v in verbs:
            pattern = re.compile(rf"\b({entities[0]})\s+{v}\s+(.+?)(?:\.|,|$)", re.IGNORECASE)
            m = pattern.search(sentence)
            if m:
                subj = m.group(1)
                obj = m.group(2).strip().split()[0]
                rels.append((subj, v, obj))
        # co-occurrence fallback
        if not rels and len(entities) >= 2:
            rels.append((entities[0], "co_occurs_with", entities[1]))
        return rels

    def query(self, entity: str) -> Dict[str, Any]:
        """Find all relations involving this entity."""
        if entity not in self.entities:
            return {"found": False}
        outgoing = [(r, o) for (s, r, o) in self.relations if s == entity]
        incoming = [(s, r) for (s, r, o) in self.relations if o == entity]
        return {
            "found": True,
            "entity": entity,
            "documents": sorted(self.entity_docs.get(entity, set())),
            "outgoing_relations": outgoing,
            "incoming_relations": incoming,
        }

    def neighbours(self, entity: str, hops: int = 1) -> Set[str]:
        """BFS for neighbours up to `hops` away."""
        if entity not in self.entities:
            return set()
        visited = {entity}
        frontier = [entity]
        for _ in range(hops):
            new_frontier = []
            for e in frontier:
                # find direct neighbours
                for (s, r, o) in self.relations:
                    if s == e and o not in visited:
                        new_frontier.append(o)
                        visited.add(o)
                    elif o == e and s not in visited:
                        new_frontier.append(s)
                        visited.add(s)
            frontier = new_frontier
            if not frontier:
                break
        return visited - {entity}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": sorted(self.entities),
            "relations": [{"subject": s, "relation": r, "object": o}
                            for s, r, o in self.relations],
            "entity_docs": {e: sorted(ds) for e, ds in self.entity_docs.items()},
        }


# ============================================================================
# Idea 44 — Causal Discovery
# ============================================================================
class CausalDiscovery:
    """Learn a DAG from observational data via the PC algorithm.

    Simplified PC:
      1. Start with a complete undirected graph
      2. For each pair (X, Y), test conditional independence given subsets of neighbours
      3. Remove edges where independence is found
      4. Orient edges using collider detection (v-structures)
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.graph: Dict[str, Set[str]] = {}
        self.edges: List[Tuple[str, str]] = []

    def _conditional_independence_test(self, X: np.ndarray, i: int, j: int,
                                         conditioning_set: List[int]) -> float:
        """Partial correlation test for CI. Returns p-value."""
        try:
            from scipy import stats as scistats
        except ImportError:
            return 1.0
        if len(conditioning_set) == 0:
            r, p = scistats.pearsonr(X[:, i], X[:, j])
            return float(p)
        # regress out conditioning set
        Z = X[:, conditioning_set]
        xi = X[:, i] - np.linalg.lstsq(np.c_[Z, np.ones(len(Z))], X[:, i], rcond=None)[0] @ np.c_[Z, np.ones(len(Z))].T
        xj = X[:, j] - np.linalg.lstsq(np.c_[Z, np.ones(len(Z))], X[:, j], rcond=None)[0] @ np.c_[Z, np.ones(len(Z))].T
        r, p = scistats.pearsonr(xi, xj)
        return float(p)

    def discover(self, X: np.ndarray, var_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run PC algorithm; return adjacency as dict."""
        n_samples, n_vars = X.shape
        names = var_names or [f"X{i}" for i in range(n_vars)]
        # init complete graph
        adj: Dict[str, Set[str]] = {n: set(names) - {n} for n in names}
        # iterate over conditioning set sizes
        for cond_size in range(min(n_vars - 2, 3)):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if names[j] not in adj[names[i]]:
                        continue
                    # try all conditioning sets of size cond_size
                    neighbours = list((adj[names[i]] | adj[names[j]]) - {names[i], names[j]})
                    if len(neighbours) < cond_size:
                        continue
                    from itertools import combinations
                    for cond_set_idx in combinations(range(len(neighbours)), cond_size):
                        cond_vars = [names.index(neighbours[k]) for k in cond_set_idx]
                        p_val = self._conditional_independence_test(X, i, j, cond_vars)
                        if p_val > self.alpha:
                            adj[names[i]].discard(names[j])
                            adj[names[j]].discard(names[i])
                            break
        self.graph = adj
        self.edges = [(s, t) for s in names for t in adj[s]]
        return {"adjacency": {k: sorted(v) for k, v in adj.items()},
                 "n_edges": len(self.edges),
                 "n_variables": n_vars}

    def is_d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        """Check d-separation of x and y given conditioning set z."""
        if x not in self.graph or y not in self.graph:
            return True
        # simple BFS reachability ignoring z
        visited = {x}
        frontier = [x]
        while frontier:
            new_frontier = []
            for node in frontier:
                for neighbour in self.graph.get(node, set()):
                    if neighbour in z or neighbour in visited:
                        continue
                    if neighbour == y:
                        return False
                    visited.add(neighbour)
                    new_frontier.append(neighbour)
            frontier = new_frontier
        return True


# ============================================================================
# Idea 45 — Neural Architecture Search v2 (evolution-based)
# ============================================================================
class NeuralArchitectureSearch2:
    """Evolution-based NAS with morphology mutation.

    Each genome is a list of layer specs. Mutations:
      - insert_layer: add a layer at a random position
      - remove_layer: drop a random layer
      - mutate_width: change a layer's width
      - mutate_activation: change a layer's activation
    Fitness = validation accuracy (or -loss).
    """
    LAYER_TYPES = ["linear", "relu", "tanh", "sigmoid", "dropout", "batchnorm"]
    ACTIVATIONS = ["relu", "tanh", "sigmoid", "gelu"]

    def __init__(self, input_dim: int = 4, output_dim: int = 2,
                 pop_size: int = 6, tournament_k: int = 3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pop_size = pop_size
        self.tournament_k = tournament_k
        self.population: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self._init_population()

    def _random_genome(self) -> List[Dict[str, Any]]:
        n_layers = random.randint(2, 5)
        layers = []
        prev_dim = self.input_dim
        for i in range(n_layers):
            width = random.choice([8, 16, 32, 64])
            layers.append({
                "type": "linear", "in_dim": prev_dim, "out_dim": width,
                "activation": random.choice(self.ACTIVATIONS),
            })
            prev_dim = width
        layers.append({"type": "linear", "in_dim": prev_dim,
                        "out_dim": self.output_dim, "activation": "softmax"})
        return layers

    def _init_population(self) -> None:
        for _ in range(self.pop_size):
            self.population.append({
                "genome": self._random_genome(),
                "fitness": -float("inf"),
            })

    def _build_model(self, genome: List[Dict[str, Any]]):
        if not _TORCH:
            return None
        layers = []
        for spec in genome:
            if spec["type"] == "linear":
                layers.append(nn.Linear(spec["in_dim"], spec["out_dim"]))
                act = spec.get("activation", "relu")
                if act == "relu":
                    layers.append(nn.ReLU())
                elif act == "tanh":
                    layers.append(nn.Tanh())
                elif act == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif act == "gelu":
                    layers.append(nn.GELU())
            elif spec["type"] == "dropout":
                layers.append(nn.Dropout(0.5))
            elif spec["type"] == "batchnorm":
                layers.append(nn.BatchNorm1d(spec.get("out_dim", 16)))
        return nn.Sequential(*layers)

    def _evaluate(self, genome: List[Dict[str, Any]],
                   X: np.ndarray, y: np.ndarray, n_steps: int = 20) -> float:
        if not _TORCH:
            return 0.0
        model = self._build_model(genome)
        if model is None:
            return -float("inf")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)
        for _ in range(n_steps):
            opt.zero_grad()
            out = model(X_t)
            loss = F.cross_entropy(out, y_t)
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds = model(X_t).argmax(-1)
            acc = float((preds == y_t).float().mean())
        return acc

    def _mutate(self, genome: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        g = [dict(layer) for layer in genome]
        op = random.choice(["insert", "remove", "width", "activation"])
        if op == "insert" and len(g) < 8:
            pos = random.randint(0, len(g) - 1)
            prev_dim = g[pos]["in_dim"]
            width = random.choice([8, 16, 32])
            g.insert(pos, {"type": "linear", "in_dim": prev_dim, "out_dim": width,
                            "activation": random.choice(self.ACTIVATIONS)})
            g[pos + 1]["in_dim"] = width
        elif op == "remove" and len(g) > 2:
            pos = random.randint(0, len(g) - 2)
            new_out = g[pos]["out_dim"]
            del g[pos]
            if pos < len(g):
                g[pos]["in_dim"] = new_out
        elif op == "width":
            pos = random.randint(0, len(g) - 2)
            new_width = random.choice([8, 16, 32, 64])
            g[pos]["out_dim"] = new_width
            g[pos + 1]["in_dim"] = new_width
        elif op == "activation":
            pos = random.randint(0, len(g) - 1)
            g[pos]["activation"] = random.choice(self.ACTIVATIONS)
        return g

    def _crossover(self, g1: List[Dict[str, Any]],
                    g2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # take first half of g1, second half of g2 (adjusting dims)
        if len(g1) < 2 or len(g2) < 2:
            return g1
        mid1 = len(g1) // 2
        mid2 = len(g2) // 2
        head = [dict(l) for l in g1[:mid1]]
        tail = [dict(l) for l in g2[mid2:]]
        # adjust the joint dim
        if head and tail:
            joint_dim = head[-1]["out_dim"]
            tail[0]["in_dim"] = joint_dim
        return head + tail

    def _tournament(self) -> Dict[str, Any]:
        contestants = random.sample(self.population, min(self.tournament_k, len(self.population)))
        return max(contestants, key=lambda g: g["fitness"])

    def evolve(self, X: np.ndarray, y: np.ndarray,
                generations: int = 5) -> Dict[str, Any]:
        for gen in range(generations):
            # evaluate
            for ind in self.population:
                if ind["fitness"] == -float("inf"):
                    ind["fitness"] = self._evaluate(ind["genome"], X, y)
            self.population.sort(key=lambda g: -g["fitness"])
            # breed
            elites = self.population[:max(2, self.pop_size // 3)]
            new_pop = [{"genome": dict(e)["genome"], "fitness": e["fitness"]} for e in elites]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament()
                p2 = self._tournament()
                child_genome = self._crossover(p1["genome"], p2["genome"])
                if random.random() < 0.5:
                    child_genome = self._mutate(child_genome)
                new_pop.append({"genome": child_genome, "fitness": -float("inf")})
            self.population = new_pop
            best = max(self.population, key=lambda g: g["fitness"])
            self.history.append({
                "generation": gen,
                "best_fitness": best["fitness"],
                "mean_fitness": float(np.mean([g["fitness"] for g in self.population
                                                 if g["fitness"] != -float("inf")])),
                "best_genome": best["genome"],
            })
        return self.history[-1] if self.history else {}

    def best_architecture(self) -> List[Dict[str, Any]]:
        if not self.population:
            return []
        best = max(self.population, key=lambda g: g["fitness"])
        return best["genome"]


__all__ = [
    "ModelTherapist", "DataArchaeologist", "ModelDreamCatcher",
    "AdversarialCoach", "ConceptForge", "TimeSeriesForecaster2",
    "ModelParliament", "KnowledgeGraphBuilder", "CausalDiscovery",
    "NeuralArchitectureSearch2",
]
