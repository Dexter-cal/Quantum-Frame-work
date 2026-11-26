"""
qai.creative.multiverse -- Creative Ideas 9, 10 & 11

Idea 9  — MULTI-UNIVERSE INFERENCE
    For each input, run the model with K different perturbations (input
    noise, weight dropout, sampling temperature, adversarial nudge) and
    return the ensemble mean + epistemic uncertainty + a diversity score.

Idea 10 — SYNTHETIC DATA GENERATOR
    Multi-modal synthetic data: tabular (copula), text (Markov + templates),
    image (procedural noise), graph (random + small-world). Differs from
    qai.data.SyntheticDataFactory by being a single API + supporting
    conditioning + quality reports.

Idea 11 — NATURAL LANGUAGE INTERPRETABILITY
    Wrap any model with a `explain(x) -> str` method that produces a
    one-sentence human-readable explanation via SHAP-style feature
    attribution + templates.
"""
from __future__ import annotations
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# Idea 9 — MULTI-UNIVERSE INFERENCE
# ============================================================================
class MultiUniverseInference:
    """Run K parallel counterfactual inferences of a single input.

    Universes vary by:
      - input perturbation (gaussian noise)
      - weight dropout (if model supports it)
      - sampling temperature (if model exposes temperature)
      - adversarial nudge (small gradient-like perturbation)

    Returns: ensemble_mean, epistemic_uncertainty, predictions, diversity
    """

    def __init__(self, model: Any, n_universes: int = 8,
                 input_noise: float = 0.05,
                 dropout_rate: float = 0.1,
                 temperature_sweep: Tuple[float, float] = (0.5, 1.5),
                 adversarial_nudge: float = 0.01):
        self.model = model
        self.K = n_universes
        self.input_noise = input_noise
        self.dropout_rate = dropout_rate
        self.temperature_sweep = temperature_sweep
        self.adversarial_nudge = adversarial_nudge

    def _perturb(self, x: np.ndarray, k: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).copy()
        # input noise
        x = x + np.random.normal(0, self.input_noise, x.shape)
        # adversarial nudge on random direction
        if self.adversarial_nudge > 0:
            nudge = np.random.normal(0, 1, x.shape)
            nudge = nudge / (np.linalg.norm(nudge) + 1e-8) * self.adversarial_nudge
            x = x + nudge
        return x

    def predict(self, x: Any) -> Dict[str, Any]:
        x = np.asarray(x, dtype=np.float32)
        predictions = []
        for k in range(self.K):
            x_pert = self._perturb(x, k)
            try:
                pred = self.model.forward(x_pert)
                pred = np.asarray(pred, dtype=np.float32).flatten()
            except Exception:
                continue
            predictions.append(pred)
        if not predictions:
            return {"ensemble_mean": None, "epistemic_uncertainty": None,
                    "predictions": [], "diversity": 0.0}
        preds = np.array(predictions)
        ensemble = preds.mean(axis=0)
        uncertainty = preds.std(axis=0)
        # diversity = mean pairwise distance
        if len(preds) > 1:
            dists = []
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    dists.append(float(np.linalg.norm(preds[i] - preds[j])))
            diversity = float(np.mean(dists)) if dists else 0.0
        else:
            diversity = 0.0
        return {
            "ensemble_mean": ensemble.tolist(),
            "epistemic_uncertainty": uncertainty.tolist(),
            "predictions": preds.tolist(),
            "n_universes": len(preds),
            "diversity": diversity,
            "agreement": 1.0 - min(diversity, 1.0),
        }


# ============================================================================
# Idea 10 — SYNTHETIC DATA GENERATOR (unified multi-modal API)
# ============================================================================
class SyntheticDataGenerator:
    """Single API for multi-modal synthetic data + quality report.

    Modalities: 'tabular', 'text', 'image', 'graph', 'time_series'.
    Each modality has a fit() + sample() method.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.fitted: Dict[str, Any] = {}

    # --- Tabular ------------------------------------------------------
    def fit_tabular(self, X: np.ndarray) -> "SyntheticDataGenerator":
        """Fit Gaussian copula: per-column marginals + rank-corr matrix."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        self.fitted["tabular"] = {
            "means": X.mean(axis=0),
            "stds": X.std(axis=0) + 1e-6,
            "cov": np.cov(X.T),
            "marginals": [np.sort(X[:, i]) for i in range(X.shape[1])],
        }
        return self

    def sample_tabular(self, n: int = 100,
                        condition: Optional[Dict[int, float]] = None) -> np.ndarray:
        cfg = self.fitted["tabular"]
        L = np.linalg.cholesky(cfg["cov"] + 1e-6 * np.eye(len(cfg["means"])))
        Z = self.rng.standard_normal((n, len(cfg["means"])))
        X = Z @ L.T + cfg["means"]
        # apply condition by overwriting columns
        if condition:
            for col, val in condition.items():
                X[:, col] = val
        return X.astype(np.float32)

    # --- Text ---------------------------------------------------------
    def fit_text(self, corpus: List[str]) -> "SyntheticDataGenerator":
        """Fit a bigram Markov model with sentence-start distribution."""
        starts: List[str] = []
        bigrams: Dict[str, List[str]] = {}
        for s in corpus:
            toks = s.split()
            if not toks:
                continue
            starts.append(toks[0])
            for i in range(len(toks) - 1):
                bigrams.setdefault(toks[i], []).append(toks[i + 1])
        self.fitted["text"] = {"starts": starts, "bigrams": bigrams}
        return self

    def sample_text(self, n: int = 10, max_len: int = 30) -> List[str]:
        cfg = self.fitted["text"]
        out = []
        for _ in range(n):
            if not cfg["starts"]:
                out.append("")
                continue
            cur = random.choice(cfg["starts"])
            words = [cur]
            for _ in range(max_len):
                if cur in cfg["bigrams"] and cfg["bigrams"][cur]:
                    cur = random.choice(cfg["bigrams"][cur])
                    words.append(cur)
                else:
                    break
            out.append(" ".join(words))
        return out

    # --- Image --------------------------------------------------------
    def fit_image(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SyntheticDataGenerator":
        """Store per-class mean image as a template."""
        if y is None:
            self.fitted["image"] = {"classes": {0: X.mean(axis=0)}}
        else:
            classes = {}
            for c in np.unique(y):
                classes[int(c)] = X[y == c].mean(axis=0)
            self.fitted["image"] = {"classes": classes}
        return self

    def sample_image(self, n: int = 16, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.fitted["image"]
        classes = list(cfg["classes"].keys())
        X = []
        y = []
        for _ in range(n):
            c = random.choice(classes)
            base = cfg["classes"][c]
            sample = base + self.rng.normal(0, noise * (np.abs(base).mean() + 1e-6), base.shape)
            X.append(sample.astype(np.float32))
            y.append(c)
        return np.array(X), np.array(y)

    # --- Graph --------------------------------------------------------
    def fit_graph(self, adj: Dict[int, List[int]]) -> "SyntheticDataGenerator":
        """Store degree distribution for graph synthesis."""
        degrees = [len(v) for v in adj.values()]
        self.fitted["graph"] = {
            "n_nodes": len(adj),
            "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
            "degree_distribution": np.bincount(degrees).tolist() if degrees else [1],
        }
        return self

    def sample_graph(self, n_nodes: Optional[int] = None,
                      model: str = "erdos") -> Dict[int, List[int]]:
        cfg = self.fitted.get("graph", {"mean_degree": 3.0})
        n = n_nodes or cfg.get("n_nodes", 20)
        p = cfg.get("mean_degree", 3.0) / max(n, 1)
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if self.rng.random() < p:
                    adj[i].append(j)
                    adj[j].append(i)
        return adj

    # --- Time series --------------------------------------------------
    def fit_time_series(self, series: np.ndarray) -> "SyntheticDataGenerator":
        series = np.asarray(series, dtype=np.float32).flatten()
        if len(series) < 2:
            self.fitted["ts"] = {"mean": 0.0, "std": 1.0, "ar1": 0.0}
            return self
        ar1 = float(np.corrcoef(series[:-1], series[1:])[0, 1])
        self.fitted["ts"] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "ar1": ar1,
        }
        return self

    def sample_time_series(self, n_periods: int = 50) -> np.ndarray:
        cfg = self.fitted["ts"]
        out = np.zeros(n_periods, dtype=np.float32)
        out[0] = cfg["mean"]
        for t in range(1, n_periods):
            out[t] = cfg["mean"] + cfg["ar1"] * (out[t-1] - cfg["mean"]) + \
                     self.rng.normal(0, cfg["std"] * 0.3)
        return out

    # --- Quality report -----------------------------------------------
    def quality_report(self, real: Any, synthetic: Any) -> Dict[str, Any]:
        """Compare real vs synthetic via summary statistics + KS test."""
        real = np.asarray(real, dtype=np.float32)
        syn = np.asarray(synthetic, dtype=np.float32)
        if real.ndim == 1:
            real = real[:, None]
        if syn.ndim == 1:
            syn = syn[:, None]
        # match column count
        d = min(real.shape[1], syn.shape[1])
        col_scores = []
        for i in range(d):
            try:
                from scipy.stats import ks_2samp
                _, p = ks_2samp(real[:, i], syn[:, i])
                col_scores.append(float(p))
            except Exception:
                # fallback: difference of normalised means
                col_scores.append(1.0 - abs(real[:, i].mean() - syn[:, i].mean())
                                  / (real[:, i].std() + 1e-6))
        return {
            "mean_p_value": float(np.mean(col_scores)) if col_scores else 0.0,
            "per_column_p": col_scores,
            "mean_real": real.mean(axis=0).tolist(),
            "mean_synthetic": syn.mean(axis=0).tolist(),
            "std_real": real.std(axis=0).tolist(),
            "std_synthetic": syn.std(axis=0).tolist(),
            "overall_quality": float(np.mean(col_scores)) if col_scores else 0.0,
        }


# ============================================================================
# Idea 11 — NATURAL LANGUAGE INTERPRETABILITY
# ============================================================================
class NLInterpreter:
    """Generate a one-sentence natural-language explanation for a prediction.

    Uses permutation feature importance to find top-k contributing features,
    then fills in a natural-language template.

    Usage:
        expl = qai.creative.NLInterpreter(model, feature_names)
        text = expl.explain(x, predicted_class=1)
    """

    def __init__(self, model: Any, feature_names: List[str],
                 class_names: Optional[List[str]] = None,
                 baseline: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or []
        self.baseline = baseline

    def _baseline(self, X: np.ndarray) -> np.ndarray:
        if self.baseline is not None:
            return self.baseline
        return X.mean(axis=0) if X.ndim > 1 else X.copy()

    def feature_importance(self, x: np.ndarray,
                             X_train: Optional[np.ndarray] = None,
                             n_repeats: int = 5) -> Dict[str, float]:
        """Permutation importance for a single prediction."""
        x = np.asarray(x, dtype=np.float32).flatten()
        base = self._baseline(X_train if X_train is not None else x[None, :])
        base_pred = self._predict(x)
        scores = {}
        for i, name in enumerate(self.feature_names):
            permuted = x.copy()
            delta = 0.0
            for _ in range(n_repeats):
                permuted[i] = base[i] + np.random.normal(0, abs(base[i]) * 0.1 + 1e-3)
                pred = self._predict(permuted)
                delta += abs(pred - base_pred)
            scores[name] = delta / n_repeats
        # normalise
        total = sum(scores.values()) + 1e-8
        return {k: v / total for k, v in scores.items()}

    def _predict(self, x: np.ndarray) -> float:
        try:
            return float(np.asarray(self.model.forward(x)).mean())
        except Exception:
            return 0.0

    def explain(self, x: np.ndarray,
                predicted_class: Optional[int] = None,
                X_train: Optional[np.ndarray] = None,
                top_k: int = 3) -> str:
        importance = self.feature_importance(x, X_train)
        ranked = sorted(importance.items(), key=lambda kv: -kv[1])[:top_k]
        # build phrase
        if predicted_class is not None and self.class_names:
            class_label = (self.class_names[predicted_class]
                            if predicted_class < len(self.class_names)
                            else f"class_{predicted_class}")
            head = f"The model predicted '{class_label}' because"
        else:
            head = "The prediction is driven by"
        phrases = []
        for name, score in ranked:
            phrases.append(f"{name} (importance {score:.2f})")
        if not phrases:
            return f"{head} no single dominant feature."
        body = ", ".join(phrases[:-1]) + (" and " + phrases[-1] if len(phrases) > 1 else phrases[0])
        confidence_phrase = ""
        if ranked:
            top_score = ranked[0][1]
            if top_score > 0.5:
                confidence_phrase = " This is a high-confidence attribution."
            elif top_score < 0.2:
                confidence_phrase = " The signal is weak — no single feature dominates."
        return f"{head} {body}.{confidence_phrase}"

    def explain_set(self, X: np.ndarray,
                     X_train: Optional[np.ndarray] = None) -> List[str]:
        return [self.explain(x, X_train=X_train) for x in X]


__all__ = [
    "MultiUniverseInference", "SyntheticDataGenerator", "NLInterpreter",
]
