"""
qai.model_card_generator -- comprehensive auto-generation of model cards.

Builds a Hugging Face-style Model Card for any model: qai Model, sklearn
estimator, or torch nn.Module. The card describes model details, intended
use, training data, evaluation results, ethical considerations, limitations,
and citation -- and can be rendered to Markdown, HTML, JSON, or YAML.

Design notes:
  * Stdlib + numpy only at import time. sklearn / torch are imported lazily
    inside the methods that actually need them, so the module loads even if
    those libraries are missing.
  * For fairness: try importing aif360; if unavailable, fall back to basic
    group statistics computed with numpy (and `qai.fairness_audit` when
    sensitive features are provided).
  * The HTML renderer produces a self-contained, professional-looking page
    (no external CSS / JS dependencies).
  * All extraction methods degrade gracefully: they never raise -- they
    return best-effort dicts/lists and leave a "note" if data was missing.

Public API:
    ModelCardData         -- structured model card data + serialize methods
    ModelCardExtractor    -- pulls architecture / metrics / fairness from a model
    ModelCardTemplate     -- pre-built templates for common model types
    ModelCardRenderer     -- markdown / HTML / YAML / HF renderer
    generate(...)         -- top-level convenience (auto-extract + render)
    quick(...)            -- quick markdown card
    from_template(...)    -- dispatch to a ModelCardTemplate method
    validate(...)         -- check required fields

Usage:
    import qai
    X, y = qai.load_iris()
    m = qai.build(technique="neural_network")
    m.train(X, y)

    from qai.model_card_generator import generate
    print(generate(m, X=X, y=y, format="markdown"))
"""
from __future__ import annotations

# Standard library imports (always available)
import html as _html
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ============================================================================
# Constants
# ============================================================================
_REQUIRED_FIELDS: Tuple[str, ...] = (
    "model_name",
    "model_type",
    "intended_use",
    "training_data",
    "eval_metrics",
    "ethical_considerations",
    "limitations",
)

_DEFAULT_LICENSE = "unknown"
_DEFAULT_VERSION = "1.0.0"
_DEFAULT_AUTHORS = ["qai Model Card Generator"]

# Tag vocabulary used to drive the Hugging Face-style YAML front matter
_TASK_TO_HF_TAG = {
    "classification": "classification",
    "regression": "regression",
    "clustering": "clustering",
    "reinforcement": "reinforcement-learning",
    "language_model": "text-generation",
    "vision": "image-classification",
    "transformer": "text-generation",
    "unknown": "qai",
}


# ============================================================================
# ModelCardData -- structured model card
# ============================================================================
@dataclass
class ModelCardData:
    """Structured data for a model card.

    Every field is optional with a sensible default; the card can be
    partially populated (e.g. when no test data is available) and the
    renderers will gracefully omit empty sections.

    Attributes:
        model_name: human-readable name (e.g. "iris_clf").
        model_version: semver-style version string.
        model_type: short label, e.g. "classifier", "regressor", "llm".
        architecture: dict with layer info (filled by extractor).
        n_params: total parameter count.
        n_layers: number of layers / depth.
        training_data: dict describing training data.
        eval_metrics: dict of evaluation metrics.
        intended_use: human-readable intended-use paragraph.
        limitations: list of limitation strings.
        ethical_considerations: list of ethical-risk strings.
        citation: BibTeX citation string.
        license: license identifier (e.g. "MIT", "Apache-2.0").
        authors: list of author names.
        date: ISO date string (YYYY-MM-DD).
        tags: list of tag strings for the HF Hub.
    """

    model_name: str = "unnamed-model"
    model_version: str = _DEFAULT_VERSION
    model_type: str = "unknown"
    architecture: Optional[Dict[str, Any]] = None
    n_params: int = 0
    n_layers: int = 0
    training_data: Optional[Dict[str, Any]] = None
    eval_metrics: Optional[Dict[str, Any]] = None
    intended_use: str = ""
    limitations: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)
    citation: str = ""
    license: str = _DEFAULT_LICENSE
    authors: List[str] = field(default_factory=lambda: list(_DEFAULT_AUTHORS))
    date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))
    tags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict (safe for json.dumps)."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "n_params": self.n_params,
            "n_layers": self.n_layers,
            "training_data": self.training_data,
            "eval_metrics": self.eval_metrics,
            "intended_use": self.intended_use,
            "limitations": list(self.limitations),
            "ethical_considerations": list(self.ethical_considerations),
            "citation": self.citation,
            "license": self.license,
            "authors": list(self.authors),
            "date": self.date,
            "tags": list(self.tags),
        }

    def to_json(self, indent: int = 2) -> str:
        """JSON serialization."""
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def to_markdown(self) -> str:
        """Render to markdown (delegates to ModelCardRenderer)."""
        return ModelCardRenderer().to_markdown(self)

    def to_html(self, theme: str = "dark") -> str:
        """Render to a self-contained HTML page."""
        return ModelCardRenderer().to_html(self, theme=theme)

    def to_yaml(self) -> str:
        """Render to YAML front-matter + markdown body."""
        return ModelCardRenderer().to_yaml(self)

    def to_hf_format(self) -> str:
        """Render to the exact Hugging Face Hub README.md format."""
        return ModelCardRenderer().to_hf_format(self)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> List[str]:
        """Return a list of missing-required-field issues (empty = OK)."""
        issues: List[str] = []
        for fname in _REQUIRED_FIELDS:
            val = getattr(self, fname, None)
            if val is None or val == "" or (isinstance(val, (list, dict)) and not val):
                issues.append(f"missing required field: '{fname}'")
        return issues


# ============================================================================
# ModelCardExtractor -- pulls metadata out of a model
# ============================================================================
class ModelCardExtractor:
    """Extract ModelCardData from a trained model.

    Supports three model flavors:
      1. qai.Model (preferred -- has `.technique` with `.name`, `.family`)
      2. sklearn estimator (has `.fit` / `.predict` and `_estimator_type`)
      3. torch.nn.Module (has `.parameters()`)

    All extract_* methods degrade gracefully: they never raise, and they
    populate a 'note' / 'warning' key on failure so the model card is
    still usable.

    Args:
        model: a trained model (qai.Model / sklearn / torch).
        X: optional evaluation inputs.
        y: optional evaluation targets.
    """

    def __init__(self, model: Any, X: Any = None, y: Any = None):
        self.model = model
        self.X = X
        self.y = y

        # Resolve the underlying technique (qai) or estimator (sklearn)
        self.technique = getattr(model, "technique", None)
        self.technique_name = (
            getattr(self.technique, "name", None)
            if self.technique is not None
            else type(model).__name__
        )
        self.family = (
            getattr(self.technique, "family", "unknown")
            if self.technique is not None
            else "unknown"
        )

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------
    def extract_all(self) -> ModelCardData:
        """Run every extractor and return a populated ModelCardData."""
        arch = self.extract_architecture()
        training_info = self.extract_training_info()
        perf = (
            self.extract_performance(self.X, self.y)
            if self.X is not None
            else {"note": "no evaluation data provided"}
        )
        fairness = self.extract_fairness(self.X, self.y)
        intended_use = self.extract_intended_use()
        limitations = self.extract_limitations()
        ethical = self.extract_ethical_considerations()

        model_type = self._detect_model_type()
        n_params = int(arch.get("total_params", 0) or 0)
        n_layers = int(arch.get("n_layers", 0) or 0)

        # Build training_data dict: what we know about the data we trained on
        training_data = self._extract_training_data_summary()
        if training_info:
            training_data["training_info"] = training_info

        # Merge performance + fairness into eval_metrics
        eval_metrics: Dict[str, Any] = dict(perf)
        eval_metrics["fairness"] = fairness

        # Tags
        tags = self._build_tags(model_type)

        # Citation
        citation = self._build_citation()

        return ModelCardData(
            model_name=self._default_model_name(),
            model_version=_DEFAULT_VERSION,
            model_type=model_type,
            architecture=arch,
            n_params=n_params,
            n_layers=n_layers,
            training_data=training_data,
            eval_metrics=eval_metrics,
            intended_use=intended_use,
            limitations=limitations,
            ethical_considerations=ethical,
            citation=citation,
            license=_DEFAULT_LICENSE,
            authors=list(_DEFAULT_AUTHORS),
            date=time.strftime("%Y-%m-%d"),
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    def extract_architecture(self) -> Dict[str, Any]:
        """Extract layer-by-layer architecture: layers, shapes, params, activations."""
        out: Dict[str, Any] = {
            "framework": "unknown",
            "type": "unknown",
            "layers": [],
            "total_params": 0,
            "n_layers": 0,
            "activations": [],
        }

        # --- qai Model wrapping a torch technique --------------------
        net = getattr(self.technique, "net", None) if self.technique else None
        if net is not None:
            try:
                return self._extract_torch_architecture(net)
            except Exception as e:
                out["warning"] = f"torch architecture extraction failed: {e}"
                return out

        # --- qai Model wrapping an sklearn estimator ----------------
        sk_model = getattr(self.technique, "model", None) if self.technique else None
        if sk_model is None and self.technique is None:
            # raw sklearn or torch model passed directly
            sk_model = self.model if self._looks_like_sklearn(self.model) else None

        if sk_model is not None:
            try:
                return self._extract_sklearn_architecture(sk_model)
            except Exception as e:
                out["warning"] = f"sklearn architecture extraction failed: {e}"
                return out

        # --- raw torch nn.Module passed directly ---------------------
        if self._looks_like_torch_module(self.model):
            try:
                return self._extract_torch_architecture(self.model)
            except Exception as e:
                out["warning"] = f"torch architecture extraction failed: {e}"
                return out

        out["warning"] = "could not determine architecture (unknown model type)"
        return out

    def _extract_torch_architecture(self, net: Any) -> Dict[str, Any]:
        import torch.nn as nn  # lazy

        layers: List[Dict[str, Any]] = []
        activations: List[str] = []
        total_params = 0
        n_linear = 0
        arch_type = "mlp"

        for name, module in net.named_modules():
            if name == "":
                continue  # root module
            entry: Dict[str, Any] = {"name": name, "type": type(module).__name__}
            if isinstance(module, nn.Linear):
                entry["in_features"] = module.in_features
                entry["out_features"] = module.out_features
                p = int(sum(p.numel() for p in module.parameters()))
                entry["params"] = p
                total_params += p
                n_linear += 1
            elif isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                entry["in_channels"] = module.in_channels
                entry["out_channels"] = module.out_channels
                entry["kernel_size"] = list(module.kernel_size)
                p = int(sum(p.numel() for p in module.parameters()))
                entry["params"] = p
                total_params += p
                arch_type = "cnn"
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                entry["input_size"] = module.input_size
                entry["hidden_size"] = module.hidden_size
                entry["num_layers"] = module.num_layers
                p = int(sum(p.numel() for p in module.parameters()))
                entry["params"] = p
                total_params += p
                arch_type = "rnn"
            elif isinstance(module, nn.MultiheadAttention):
                entry["embed_dim"] = module.embed_dim
                entry["num_heads"] = module.num_heads
                p = int(sum(p.numel() for p in module.parameters()))
                entry["params"] = p
                total_params += p
                arch_type = "transformer"
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                p = int(sum(p.numel() for p in module.parameters()))
                entry["params"] = p
                total_params += p
            elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh,
                                       nn.Sigmoid, nn.Softmax, nn.LeakyReLU)):
                entry["params"] = 0
                act_name = type(module).__name__
                if act_name not in activations:
                    activations.append(act_name)
            elif isinstance(module, (nn.Dropout,)):
                entry["params"] = 0
            else:
                entry["params"] = int(sum(p.numel() for p in module.parameters()))

            layers.append(entry)

        # If too deep (e.g. transformer with hundreds of submodules), trim
        if len(layers) > 60:
            layers = layers[:60]
            note = "...(truncated, %d total submodules)" % sum(1 for _ in net.named_modules())
            layers.append({"name": "_truncated", "type": note, "params": 0})

        # Add the embedded/total param count via top-level parameters() -- more accurate
        try:
            total_params = int(sum(p.numel() for p in net.parameters()))
        except Exception:
            pass

        return {
            "framework": "pytorch",
            "type": arch_type,
            "layers": layers,
            "total_params": total_params,
            "n_layers": n_linear or len(layers),
            "activations": activations,
        }

    def _extract_sklearn_architecture(self, sk_model: Any) -> Dict[str, Any]:
        layers: List[Dict[str, Any]] = []
        total_params = 0
        activations: List[str] = []
        arch_type = "linear"

        # Linear models: coef_ + intercept_
        if hasattr(sk_model, "coef_"):
            coef = np.asarray(sk_model.coef_)
            intercept = getattr(sk_model, "intercept_", None)
            n_coef = int(coef.size)
            n_intercept = int(np.asarray(intercept).size) if intercept is not None else 0
            total_params = n_coef + n_intercept
            layers.append({
                "name": "coef_",
                "type": "Linear",
                "shape": list(coef.shape),
                "params": n_coef,
            })
            if n_intercept:
                layers.append({
                    "name": "intercept_",
                    "type": "Bias",
                    "shape": list(np.asarray(intercept).shape),
                    "params": n_intercept,
                })
            arch_type = "linear"

        # Tree-based: count estimators / leaves
        if hasattr(sk_model, "estimators_"):
            estimators = sk_model.estimators_
            n_trees = len(estimators) if hasattr(estimators, "__len__") else 0
            total_params = 0
            leaves_per_tree = []
            for i, est in enumerate(estimators):
                if hasattr(est, "tree_") and hasattr(est.tree_, "node_count"):
                    n_leaves = int(est.tree_.n_leaves) if hasattr(est.tree_, "n_leaves") else int(est.tree_.node_count)
                    total_params += int(est.tree_.node_count)
                    if i < 5:
                        leaves_per_tree.append(n_leaves)
            avg_leaves = float(np.mean(leaves_per_tree)) if leaves_per_tree else 0.0
            layers.append({
                "name": "forest",
                "type": type(sk_model).__name__,
                "n_estimators": n_trees,
                "avg_leaves_per_tree": round(avg_leaves, 2),
                "params": total_params,
            })
            arch_type = "tree_ensemble"

        # Clustering: cluster_centers_
        if hasattr(sk_model, "cluster_centers_"):
            centers = np.asarray(sk_model.cluster_centers_)
            n_centers, n_dim = centers.shape
            total_params = int(centers.size)
            layers.append({
                "name": "cluster_centers_",
                "type": "ClusterCenters",
                "n_clusters": n_centers,
                "n_features": n_dim,
                "params": total_params,
            })
            arch_type = "clustering"

        # Feature importances (bonus)
        if hasattr(sk_model, "feature_importances_"):
            fi = np.asarray(sk_model.feature_importances_)
            layers.append({
                "name": "feature_importances_",
                "type": "FeatureImportance",
                "n_features": int(fi.size),
                "params": int(fi.size),
            })
            total_params += int(fi.size)

        # Q-table for RL
        if hasattr(sk_model, "q_table"):
            qt = np.asarray(sk_model.q_table)
            total_params = int(qt.size)
            layers.append({
                "name": "q_table",
                "type": "QTable",
                "shape": list(qt.shape),
                "params": total_params,
            })
            arch_type = "tabular_rl"

        if not layers:
            arch_type = "unknown"
            return {
                "framework": "sklearn",
                "type": arch_type,
                "layers": [],
                "total_params": 0,
                "n_layers": 0,
                "activations": [],
                "warning": "sklearn model has no inspectable coefficients / trees / centers",
            }

        return {
            "framework": "sklearn",
            "type": arch_type,
            "layers": layers,
            "total_params": total_params,
            "n_layers": len(layers),
            "activations": activations,
        }

    # ------------------------------------------------------------------
    # Training info
    # ------------------------------------------------------------------
    def extract_training_info(self) -> Dict[str, Any]:
        """Extract optimizer, lr, epochs, batch size when available."""
        info: Dict[str, Any] = {}

        tech = self.technique
        if tech is not None:
            # qai neural_network stores these as attributes
            for attr in ("optimizer", "lr", "epochs", "batch_size",
                         "loss_function", "weight_decay", "dropout",
                         "n_layers", "activation", "hidden_dim"):
                if hasattr(tech, attr):
                    val = getattr(tech, attr)
                    info[attr] = self._safe_serialize(val)
            # Resolve optimizer type name
            opt = getattr(tech, "optimizer", None)
            if opt is not None:
                info["optimizer"] = type(opt).__name__
            else:
                info.pop("optimizer", None)
            # Resolve loss function name from criterion
            if "loss_function" not in info:
                crit = getattr(tech, "criterion", None)
                if crit is not None:
                    info["loss_function"] = type(crit).__name__

        # sklearn: n_iter_ (number of iterations)
        sk_model = getattr(tech, "model", None) if tech else None
        if sk_model is None and self._looks_like_sklearn(self.model):
            sk_model = self.model
        if sk_model is not None:
            if hasattr(sk_model, "n_iter_"):
                n_iter = sk_model.n_iter_
                try:
                    info["n_iter"] = int(np.ravel(n_iter)[0])
                except Exception:
                    info["n_iter"] = n_iter
            if hasattr(sk_model, "classes_"):
                classes = sk_model.classes_
                info["classes_"] = [self._safe_serialize(c) for c in classes]

        if not info:
            info["note"] = "training info not available for this model type"
        return info

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    def extract_performance(self, X: Any, y: Any) -> Dict[str, Any]:
        """Compute accuracy / precision / recall / F1 / latency / throughput."""
        out: Dict[str, Any] = {}
        if X is None or y is None:
            out["note"] = "no evaluation data provided"
            return out

        try:
            X_arr = np.asarray(X)
            y_arr = np.asarray(y)
        except Exception as e:
            out["error"] = f"could not convert X/y to arrays: {e}"
            return out

        # Predictions
        try:
            preds = self._predict_batch(X_arr)
            preds = np.asarray(preds)
        except Exception as e:
            out["error"] = f"prediction failed: {e}"
            return out

        y_flat = y_arr.flatten()

        # If predictions are 2D logits (raw scores from a torch model), convert
        # to class labels via argmax -- but only when y looks like classification
        # labels AND the second dim of preds is > 1 (a column-vector regressor
        # should be flattened, not argmaxed).
        is_classification_target = (
            y_flat.dtype.kind in ("i", "u", "S", "U", "O")
            and len(np.unique(y_flat)) <= 50
        )
        if preds.ndim == 2:
            n_rows, n_cols = preds.shape
            if n_cols == 1:
                # Column vector -- flatten to 1D
                preds = preds[:, 0]
            elif is_classification_target and n_rows == len(y_flat):
                # Looks like logits -- take argmax along the class axis
                preds = np.argmax(preds, axis=1)
                out["prediction_backend"] = "argmax_over_logits"
            else:
                preds = preds.flatten()
                if len(preds) != len(y_flat):
                    out["prediction_warning"] = (
                        f"prediction shape {preds.shape} incompatible with y shape "
                        f"{y_flat.shape}; metrics may be unreliable"
                    )

        if is_classification_target:
            out.update(self._classification_metrics(y_flat, preds))
        else:
            out.update(self._regression_metrics(y_flat.astype(float), preds.astype(float)))

        # Latency / throughput
        try:
            latency_ms, throughput = self._measure_latency(X_arr)
            out["latency_ms"] = round(latency_ms, 4)
            out["throughput_per_s"] = round(throughput, 2)
        except Exception as e:
            out["latency_note"] = f"latency measurement failed: {e}"

        out["n_samples"] = int(len(X_arr))
        return out

    def _classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            out["accuracy"] = float(np.mean(y_true == y_pred))
        except Exception:
            pass

        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        out["n_classes"] = int(n_classes)

        # Macro precision / recall / F1 (binary-safe)
        precisions, recalls, f1s = [], [], []
        per_class: Dict[Any, Dict[str, float]] = {}
        for c in classes:
            tp = int(np.sum((y_pred == c) & (y_true == c)))
            fp = int(np.sum((y_pred == c) & (y_true != c)))
            fn = int(np.sum((y_pred != c) & (y_true == c)))
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-12)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            per_class[self._safe_serialize(c)] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "support": int(np.sum(y_true == c)),
            }

        out["precision_macro"] = round(float(np.mean(precisions)), 4) if precisions else 0.0
        out["recall_macro"] = round(float(np.mean(recalls)), 4) if recalls else 0.0
        out["f1_macro"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
        out["per_class"] = per_class

        # Confusion matrix
        try:
            cm = np.zeros((n_classes, n_classes), dtype=int)
            cls_to_idx = {c: i for i, c in enumerate(classes)}
            for t, p in zip(y_true, y_pred):
                cm[cls_to_idx[t], cls_to_idx[p]] += 1
            out["confusion_matrix"] = cm.tolist()
            out["classes"] = [self._safe_serialize(c) for c in classes]
        except Exception:
            pass

        # Optional: balanced accuracy via sklearn if available
        try:
            from sklearn.metrics import (
                balanced_accuracy_score as _bas,
                cohen_kappa_score as _cks,
            )
            out["balanced_accuracy"] = round(float(_bas(y_true, y_pred)), 4)
            out["cohen_kappa"] = round(float(_cks(y_true, y_pred)), 4)
        except Exception:
            pass
        return out

    def _regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            err = y_pred - y_true
            out["mse"] = round(float(np.mean(err ** 2)), 6)
            out["rmse"] = round(float(np.sqrt(out["mse"])), 6)
            out["mae"] = round(float(np.mean(np.abs(err))), 6)
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            out["r2"] = round(1.0 - ss_res / max(ss_tot, 1e-12), 6)
            out["max_error"] = round(float(np.max(np.abs(err))), 6)
            out["explained_variance"] = round(
                float(1.0 - np.var(err) / max(np.var(y_true), 1e-12)), 6
            )
        except Exception as e:
            out["error"] = f"regression metric computation failed: {e}"
        return out

    def _measure_latency(self, X: np.ndarray, n_warmup: int = 3, n_runs: int = 15) -> Tuple[float, float]:
        import time as _time

        X_sample = X[: min(len(X), 32)] if len(X) > 0 else X

        # Warmup
        for _ in range(n_warmup):
            try:
                self._predict_batch(X_sample)
            except Exception:
                pass

        # Timed runs
        latencies: List[float] = []
        for _ in range(n_runs):
            t0 = _time.perf_counter()
            try:
                self._predict_batch(X_sample)
            except Exception:
                pass
            t1 = _time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        if not latencies:
            return 0.0, 0.0
        avg_ms = float(np.mean(latencies))
        throughput = 1000.0 / max(avg_ms, 1e-6)  # samples per second (per batch)
        # convert to per-sample throughput
        n_per_call = max(len(X_sample), 1)
        return avg_ms / n_per_call, throughput * n_per_call

    # ------------------------------------------------------------------
    # Fairness
    # ------------------------------------------------------------------
    def extract_fairness(self, X: Any, y: Any, sensitive_features: Any = None) -> Dict[str, Any]:
        """Run a fairness audit; gracefully degrade if no sensitive features."""
        out: Dict[str, Any] = {
            "sensitive_features_provided": False,
            "note": "No sensitive features provided; full fairness audit not performed.",
        }

        if sensitive_features is None:
            # Auto-detect: if X has a column that looks categorical with low cardinality,
            # treat it as a candidate sensitive attribute. Otherwise, return basic stats.
            return out

        out["sensitive_features_provided"] = True
        out.pop("note", None)

        try:
            sf = np.asarray(sensitive_features)
            if sf.ndim > 1:
                sf = sf.ravel()
            groups = np.unique(sf)
            out["n_groups"] = int(len(groups))
            out["groups"] = [self._safe_serialize(g) for g in groups]
        except Exception as e:
            out["error"] = f"could not parse sensitive_features: {e}"
            return out

        # Need predictions + true labels for the audit
        if X is None or y is None:
            out["note"] = "X/y not provided; cannot compute group-wise metrics."
            return out

        try:
            preds = np.asarray(self._predict_batch(np.asarray(X))).flatten()
            y_flat = np.asarray(y).flatten()
        except Exception as e:
            out["error"] = f"prediction for fairness audit failed: {e}"
            return out

        # Prefer aif360 if available
        aif360_metrics = self._aif360_metrics(y_flat, preds, sf)
        if aif360_metrics is not None:
            out["backend"] = "aif360"
            out.update(aif360_metrics)
        else:
            # Fall back to basic group stats via qai.governance (numpy-only)
            try:
                from .governance import fairness_audit, demographic_parity_difference, \
                    equalized_odds_difference, disparate_impact_ratio

                # For binary positive-rate metrics, binarize: positive = preds == 1
                # If multi-class, fall back to per-group accuracy + selection rate
                if len(np.unique(y_flat)) <= 2:
                    pos_label = 1
                    pred_bin = (preds == pos_label).astype(int)
                    y_bin = (y_flat == pos_label).astype(int)
                    out["demographic_parity_difference"] = round(
                        float(demographic_parity_difference(pred_bin, sf)), 6)
                    out["equalized_odds_difference"] = round(
                        float(equalized_odds_difference(y_bin, pred_bin, sf)), 6)
                    out["disparate_impact_ratio"] = round(
                        float(disparate_impact_ratio(pred_bin, sf)), 6)
                else:
                    out["note"] = ("multi-class labels: reporting per-group accuracy "
                                    "instead of binary parity metrics.")
            except Exception as e:
                out["fairness_backend_error"] = str(e)

            # Per-group stats (always compute these -- they're the basic fallback)
            group_stats: Dict[str, Dict[str, float]] = {}
            for g in groups:
                mask = (sf == g)
                if mask.sum() == 0:
                    continue
                gpred = preds[mask]
                gtrue = y_flat[mask]
                # accuracy (classification) or MAE (regression)
                if gtrue.dtype.kind in ("i", "u"):
                    acc = float(np.mean(gpred == gtrue))
                    group_stats[self._safe_serialize(g)] = {
                        "n_samples": int(mask.sum()),
                        "accuracy": round(acc, 4),
                        "positive_prediction_rate": round(float(np.mean(gpred == 1)), 4)
                            if len(np.unique(y_flat)) <= 2 else None,
                    }
                else:
                    mae = float(np.mean(np.abs(gpred.astype(float) - gtrue.astype(float))))
                    group_stats[self._safe_serialize(g)] = {
                        "n_samples": int(mask.sum()),
                        "mae": round(mae, 6),
                    }
            out["group_stats"] = group_stats

        return out

    def _aif360_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         sf: np.ndarray) -> Optional[Dict[str, Any]]:
        """Try to compute fairness metrics using aif360. Return None if unavailable."""
        try:
            from aif360.datasets import BinaryLabelDataset  # noqa: F401
            from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
        except Exception:
            return None

        # aif360 expects binary labels -- skip if multi-class
        if len(np.unique(y_true)) > 2 or len(np.unique(y_pred)) > 2:
            return None
        try:
            import pandas as pd
            from aif360.datasets import BinaryLabelDataset

            df = pd.DataFrame({
                "label": np.asarray(y_true, dtype=int),
                "prediction": np.asarray(y_pred, dtype=int),
                "sensitive": np.asarray(sf),
            })
            bld = BinaryLabelDataset(
                favorable_label=1, unfavorable_label=0,
                df=df, label_names=["label"],
                protected_attribute_names=["sensitive"],
            )
            from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
            unprivileged = [{"sensitive": v} for v in np.unique(sf)[1:]]
            privileged = [{"sensitive": np.unique(sf)[0]}]
            metric = BinaryLabelDatasetMetric(bld,
                                              unprivileged_groups=unprivileged,
                                              privileged_groups=privileged)
            pred_bld = bld.copy()
            pred_bld.labels = np.asarray(y_pred, dtype=float).reshape(-1, 1)
            cm = ClassificationMetric(bld, pred_bld,
                                       unprivileged_groups=unprivileged,
                                       privileged_groups=privileged)
            return {
                "statistical_parity_difference": round(float(metric.statistical_parity_difference()), 6),
                "disparate_impact": round(float(metric.disparate_impact()), 6),
                "equal_opportunity_difference": round(float(cm.equal_opportunity_difference()), 6),
                "average_abs_odds_difference": round(float(cm.average_abs_odds_difference()), 6),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------
    def extract_intended_use(self) -> str:
        """Best-effort intended-use paragraph based on detected model type."""
        model_type = self._detect_model_type()
        name = self._default_model_name()

        templates = {
            "classification": (
                f"This model was trained for **multi-class classification** on tabular "
                f"feature data. It assigns each input to one of the learned classes "
                f"(technique: `{self.technique_name}`). Suitable for batch and online "
                f"prediction tasks where probabilistic or hard labels are required. "
                f"Intended for research, experimentation, and lower-stakes production use."
            ),
            "regression": (
                f"This model was trained for **regression** -- predicting a continuous "
                f"target value from tabular features (technique: `{self.technique_name}`). "
                f"Suitable for forecasting, scoring, and as a baseline for downstream tasks."
            ),
            "clustering": (
                f"This model was trained for **unsupervised clustering** -- grouping "
                f"similar inputs together (technique: `{self.technique_name}`). Use for "
                f"exploratory data analysis, segmentation, and anomaly discovery."
            ),
            "reinforcement": (
                f"This model was trained for **reinforcement learning** in a fixed "
                f"environment (technique: `{self.technique_name}`). The learned policy "
                f"selects actions to maximize cumulative reward; deploy only in the "
                f"environment it was trained on."
            ),
            "language_model": (
                f"This is a **language model** trained to predict the next token in a "
                f"sequence (technique: `{self.technique_name}`). Suitable for text "
                f"generation, completion, and as a backbone for downstream NLP tasks."
            ),
            "vision": (
                f"This model was trained for **computer vision** -- image classification "
                f"or feature extraction (technique: `{self.technique_name}`). Use on images "
                f"of the resolution and distribution it was trained on."
            ),
            "transformer": (
                f"This is a **transformer** model suitable for sequence modeling tasks "
                f"(technique: `{self.technique_name}`). Apply on tokenized inputs matching "
                f"the trained vocabulary and sequence length."
            ),
            "unknown": (
                f"This is a `{self.technique_name}` model trained with the qai framework. "
                f"Specify the intended use case, deployment context, and target users "
                f"before publishing this card."
            ),
        }
        return templates.get(model_type, templates["unknown"])

    def extract_limitations(self) -> List[str]:
        """Auto-detect limitations (small training data, missing features, etc.)."""
        limitations: List[str] = []

        # Training data size
        td = self._extract_training_data_summary()
        n_samples = td.get("n_samples", 0)
        if n_samples and n_samples < 1000:
            limitations.append(
                f"Trained on a small dataset ({n_samples} samples) -- generalization "
                f"to unseen data is uncertain and should be verified on a held-out test set."
            )
        elif n_samples and n_samples < 10000:
            limitations.append(
                f"Training set is moderately sized ({n_samples} samples); performance "
                f"may degrade on out-of-distribution inputs."
            )

        # Parameter count vs. data (overfitting heuristic)
        arch = self.extract_architecture()
        n_params = int(arch.get("total_params", 0) or 0)
        if n_params and n_samples and n_params > n_samples:
            limitations.append(
                f"Model has {n_params:,} parameters but was trained on only "
                f"{n_samples:,} samples -- high overfitting risk."
            )

        # Missing evaluation data
        if self.X is None or self.y is None:
            limitations.append(
                "No evaluation data was provided at card-generation time; reported "
                "metrics are incomplete. Re-generate with X=, y= to populate."
            )

        # Technique-specific caveats
        if self.technique_name == "neural_network":
            limitations.append(
                "Neural networks can be sensitive to input scaling -- ensure inputs "
                "are preprocessed identically at training and inference time."
            )
        elif self.technique_name in ("random_forest", "decision_tree"):
            limitations.append(
                "Tree-based models do not extrapolate beyond the range of training "
                "data -- inputs outside that range will not be predicted reliably."
            )
        elif self.technique_name == "kmeans":
            limitations.append(
                "K-means assumes spherical clusters of similar size; performance "
                "suffers on non-convex or imbalanced cluster shapes."
            )

        # Generic
        limitations.append(
            "Model performance is bounded by the quality and representativeness of "
            "the training data; it should not be the sole basis for high-stakes decisions."
        )
        limitations.append(
            "The model may not generalize to data distributions significantly "
            "different from training -- monitor for data drift post-deployment."
        )
        return limitations

    def extract_ethical_considerations(self) -> List[str]:
        """Auto-detect ethical risks (bias, privacy, dual-use)."""
        ethical: List[str] = []

        # Bias
        ethical.append(
            "The model may encode biases present in the training data. Run a fairness "
            "audit (with sensitive features) before deploying to populations the model "
            "was not evaluated on."
        )
        ethical.append(
            "Predictions should not be used as the sole basis for high-stakes decisions "
            "affecting individuals (hiring, lending, healthcare, criminal justice, etc.) "
            "without human review."
        )

        # Privacy
        ethical.append(
            "If the model was trained on personal data, ensure compliance with "
            "applicable privacy regulations (GDPR, CCPA). Consider differential "
            "privacy or federated learning for sensitive deployments."
        )

        # Dual-use
        model_type = self._detect_model_type()
        if model_type == "classification":
            ethical.append(
                "Classification models can be repurposed for surveillance or "
                "discriminatory profiling. Document downstream uses and enforce "
                "acceptable-use policies."
            )
        elif model_type in ("language_model", "transformer"):
            ethical.append(
                "Language models can generate misleading, harmful, or manipulated "
                "content at scale. Deploy with safety filters, output moderation, "
                "and provenance tracking."
            )
        elif model_type == "vision":
            ethical.append(
                "Vision models can enable unauthorized identification or surveillance. "
                "Restrict to consented use cases."
            )
        elif model_type == "reinforcement":
            ethical.append(
                "Reinforcement-learning policies can behave unpredictably outside "
                "their training environment. Validate in a sandbox before deployment."
            )

        # Small-data caveat
        td = self._extract_training_data_summary()
        n = td.get("n_samples", 0)
        if n and n < 1000:
            ethical.append(
                f"Small training set ({n} samples) increases the risk that the model "
                f"under-represents minority groups in the data."
            )

        ethical.append(
            "Consider the impact of false positives and false negatives in the "
            "specific deployment context."
        )
        return ethical

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _detect_model_type(self) -> str:
        """Classify the model into a coarse task category."""
        tech = self.technique
        tname = (self.technique_name or "").lower()
        family = self.family.lower() if self.family else ""

        # qai Model -- inspect technique
        if tech is not None:
            if tname in ("neural_network", "classifier", "svm", "knn",
                         "naive_bayes", "perceptron", "random_forest",
                         "decision_tree", "multinomial_naive_bayes"):
                # If the technique exposes a task_type attribute, prefer it
                tt = getattr(tech, "task_type", None)
                if tt == "regression":
                    return "regression"
                return "classification"
            if tname == "regression":
                return "regression"
            if tname in ("kmeans", "dbscan", "gmm", "hierarchical", "isolation_forest"):
                return "clustering"
            if tname in ("transformer", "mamba", "rnn", "lstm"):
                return "transformer"
            if tname in ("cnn",):
                return "vision"
            if "rl" in family or "reinforcement" in family or tname in ("tabular_policy",):
                return "reinforcement"
            if tname == "linear_regression":
                return "regression"
            # Heuristic on family
            if "deep" in family or "classical" in family:
                # Default for classical/deep is classification unless technique name suggests otherwise
                return "classification"

        # Raw sklearn estimator
        if self._looks_like_sklearn(self.model):
            est_type = getattr(self.model, "_estimator_type", None)
            if est_type == "regressor":
                return "regression"
            if est_type == "classifier":
                return "classification"
            if est_type == "clusterer":
                return "clustering"

        # Raw torch module -- guess from class name
        cname = type(self.model).__name__.lower()
        if "transformer" in cname or "gpt" in cname or "bert" in cname or "llama" in cname:
            return "language_model"
        if "cnn" in cname or "conv" in cname or "resnet" in cname or "vit" in cname:
            return "vision"
        if "rnn" in cname or "lstm" in cname or "gru" in cname:
            return "transformer"

        return "unknown"

    def _extract_training_data_summary(self) -> Dict[str, Any]:
        """Describe training data: shape, n_features, n_classes."""
        out: Dict[str, Any] = {}
        # Prefer training data captured on the technique
        X_train = None
        if self.technique is not None:
            X_train = getattr(self.technique, "_last_X", None)
        # Fall back to the eval data passed in (often the same)
        if X_train is None:
            X_train = self.X
        if X_train is None:
            out["note"] = "training data not available; could not summarize."
            return out
        try:
            arr = np.asarray(X_train)
            out["n_samples"] = int(arr.shape[0]) if arr.ndim >= 1 else 1
            out["n_features"] = int(arr.shape[1]) if arr.ndim >= 2 else 1
            out["shape"] = list(arr.shape)
            out["dtype"] = str(arr.dtype)
            if arr.dtype.kind in ("f", "i", "u"):
                out["feature_stats"] = {
                    "mean": round(float(np.mean(arr)), 4) if arr.size else None,
                    "std": round(float(np.std(arr)), 4) if arr.size else None,
                    "min": round(float(np.min(arr)), 4) if arr.size else None,
                    "max": round(float(np.max(arr)), 4) if arr.size else None,
                }
        except Exception as e:
            out["error"] = f"could not summarize training data: {e}"

        # Labels
        y_train = None
        if self.technique is not None:
            y_train = getattr(self.technique, "_last_y", None)
        if y_train is None:
            y_train = self.y
        if y_train is not None:
            try:
                y_arr = np.asarray(y_train)
                out["n_target_classes"] = int(len(np.unique(y_arr))) if y_arr.dtype.kind in "iuUSO" else None
                out["target_dtype"] = str(y_arr.dtype)
            except Exception:
                pass

        return out

    def _build_tags(self, model_type: str) -> List[str]:
        tags = ["qai", self.technique_name or "model"]
        hf_tag = _TASK_TO_HF_TAG.get(model_type, "qai")
        if hf_tag not in tags:
            tags.append(hf_tag)
        if self.family and self.family != "unknown" and self.family not in tags:
            tags.append(self.family)
        return tags

    def _build_citation(self) -> str:
        name = self._default_model_name()
        year = time.strftime("%Y")
        authors = ", ".join(_DEFAULT_AUTHORS)
        return (
            "@misc{" + name.replace("-", "_") + "_" + year + ",\n"
            "  title={Model Card: " + name + "},\n"
            "  author={" + authors + "},\n"
            "  year={" + year + "},\n"
            "  howpublished={qai Model Card Generator},\n"
            "  note={Auto-generated model card}\n"
            "}"
        )

    def _default_model_name(self) -> str:
        """Best-effort name for the model."""
        if self.technique is not None:
            name = getattr(self.technique, "name", None) or self.technique_name
            return f"{name}_model"
        return f"{type(self.model).__name__.lower()}_model"

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Call predict on the underlying model, handling qai / sklearn / torch."""
        m = self.model
        if hasattr(m, "predict"):
            preds = m.predict(X)
            return np.asarray(preds)
        if self.technique is not None and hasattr(self.technique, "forward"):
            preds = self.technique.forward(X)
            return np.asarray(preds)
        # raw torch module
        if self._looks_like_torch_module(m):
            import torch  # lazy
            with torch.no_grad():
                t = torch.as_tensor(np.asarray(X, dtype="float32"))
                out = m(t)
                if hasattr(out, "detach"):
                    out = out.detach().cpu().numpy()
                return np.asarray(out)
        raise RuntimeError("model has no predict()/forward() and is not a torch module")

    @staticmethod
    def _looks_like_sklearn(m: Any) -> bool:
        return (
            hasattr(m, "fit")
            and hasattr(m, "predict")
            and not hasattr(m, "technique")  # qai Model has .technique; sklearn doesn't
        )

    @staticmethod
    def _looks_like_torch_module(m: Any) -> bool:
        try:
            import torch.nn as nn  # lazy
            return isinstance(m, nn.Module)
        except Exception:
            return False

    @staticmethod
    def _safe_serialize(v: Any) -> Any:
        """Convert numpy scalars / arrays to JSON-safe Python types."""
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return [ModelCardExtractor._safe_serialize(x) for x in v]
        if isinstance(v, dict):
            return {str(k): ModelCardExtractor._safe_serialize(val) for k, val in v.items()}
        return v


# ============================================================================
# ModelCardTemplate -- pre-built templates for common model types
# ============================================================================
class ModelCardTemplate:
    """Pre-built templates for common model types.

    Each classmethod returns a fully-populated ModelCardData tuned for a
    specific task. Use these as starting points -- edit fields and re-render.
    """

    @staticmethod
    def _default_citation(name: str) -> str:
        year = time.strftime("%Y")
        return (
            "@misc{" + name.replace("-", "_") + "_" + year + ",\n"
            "  title={Model Card: " + name + "},\n"
            "  author={qai Model Card Generator},\n"
            "  year={" + year + "},\n"
            "  howpublished={qai Model Card Generator}\n"
            "}"
        )

    # ------------------------------------------------------------------
    @classmethod
    def classification_template(cls, model_name: str, accuracy: float,
                                 n_classes: int) -> ModelCardData:
        """Template for a classification model."""
        return ModelCardData(
            model_name=model_name,
            model_version=_DEFAULT_VERSION,
            model_type="classification",
            architecture={
                "framework": "template",
                "type": "classifier",
                "layers": [{"name": "head", "type": "Linear", "out_features": n_classes}],
                "total_params": 0,
                "n_layers": 1,
                "activations": [],
            },
            n_params=0,
            n_layers=1,
            training_data={
                "note": "Template-generated; populate with the real training dataset "
                        "summary via ModelCardExtractor.extract_all() for a real card.",
                "n_classes": int(n_classes),
            },
            eval_metrics={
                "accuracy": round(float(accuracy), 4),
                "n_classes": int(n_classes),
                "note": "Template metrics; replace with measured values.",
            },
            intended_use=(
                f"This model was trained for **classification** with {n_classes} target "
                f"classes. It assigns each input to one of the learned classes. Suitable "
                f"for batch and online prediction tasks where probabilistic or hard "
                f"labels are required. Achieves an accuracy of {accuracy:.2%} on the "
                f"reported evaluation set."
            ),
            limitations=[
                "Template-generated limitations: replace with model-specific caveats.",
                f"Reported accuracy ({accuracy:.2%}) reflects the evaluation set only -- "
                f"real-world performance may differ.",
                "Model performance is bounded by training-data quality and coverage.",
            ],
            ethical_considerations=[
                "Run a fairness audit (with sensitive features) before deployment.",
                "Do not use as the sole basis for high-stakes decisions about individuals.",
                "Monitor for data drift post-deployment.",
            ],
            citation=cls._default_citation(model_name),
            license=_DEFAULT_LICENSE,
            authors=list(_DEFAULT_AUTHORS),
            date=time.strftime("%Y-%m-%d"),
            tags=["qai", "classification", "classifier"],
        )

    @classmethod
    def regression_template(cls, model_name: str, mse: float, r2: float) -> ModelCardData:
        """Template for a regression model."""
        return ModelCardData(
            model_name=model_name,
            model_version=_DEFAULT_VERSION,
            model_type="regression",
            architecture={
                "framework": "template",
                "type": "regressor",
                "layers": [{"name": "head", "type": "Linear", "out_features": 1}],
                "total_params": 0,
                "n_layers": 1,
                "activations": [],
            },
            n_params=0,
            n_layers=1,
            training_data={
                "note": "Template-generated; populate with real training-data summary.",
            },
            eval_metrics={
                "mse": round(float(mse), 6),
                "rmse": round(float(np.sqrt(mse)), 6),
                "r2": round(float(r2), 6),
                "note": "Template metrics; replace with measured values.",
            },
            intended_use=(
                f"This model was trained for **regression** -- predicting a continuous "
                f"target from tabular features. Achieves an R² of {r2:.4f} (MSE={mse:.4f}) "
                f"on the reported evaluation set. Suitable for forecasting, scoring, and "
                f"as a baseline for downstream tasks."
            ),
            limitations=[
                "Template-generated limitations: replace with model-specific caveats.",
                f"Reported R² ({r2:.4f}) reflects the evaluation set only.",
                "Tree-based regressors do not extrapolate beyond training-data range.",
            ],
            ethical_considerations=[
                "Audit for biased outputs across demographic groups before deployment.",
                "Document target variable provenance and known skews.",
                "Do not use as the sole input for high-stakes individual decisions.",
            ],
            citation=cls._default_citation(model_name),
            license=_DEFAULT_LICENSE,
            authors=list(_DEFAULT_AUTHORS),
            date=time.strftime("%Y-%m-%d"),
            tags=["qai", "regression", "regressor"],
        )

    @classmethod
    def llm_template(cls, model_name: str, params_b: float,
                      training_tokens: int) -> ModelCardData:
        """Template for a large language model."""
        n_params_int = int(params_b * 1_000_000_000)
        return ModelCardData(
            model_name=model_name,
            model_version=_DEFAULT_VERSION,
            model_type="language_model",
            architecture={
                "framework": "template",
                "type": "transformer",
                "layers": [{"name": "transformer_stack", "type": "Transformer",
                            "n_params_billion": float(params_b)}],
                "total_params": n_params_int,
                "n_layers": 0,
                "activations": ["GELU"],
            },
            n_params=n_params_int,
            n_layers=0,
            training_data={
                "n_training_tokens": int(training_tokens),
                "note": "Template-generated; document actual data sources, curation, "
                        "and licensing of the pretraining corpus.",
            },
            eval_metrics={
                "n_params_billion": float(params_b),
                "n_training_tokens": int(training_tokens),
                "note": "Replace with perplexity, MMLU, HumanEval, etc.",
            },
            intended_use=(
                f"This is a **large language model** with {params_b:.2f}B parameters, "
                f"trained on approximately {training_tokens:,} tokens. Suitable for text "
                f"generation, completion, summarization, and as a backbone for downstream "
                f"NLP tasks. Intended for research and assistant-style applications."
            ),
            limitations=[
                "May produce factually incorrect or harmful outputs -- verify before "
                "deploying in user-facing contexts.",
                "Knowledge cutoff is determined by the pretraining corpus; the model "
                "has no awareness of events after training.",
                f"With {params_b:.2f}B parameters, inference cost is non-trivial -- "
                f"consider quantization or distillation for edge deployment.",
                "May reflect biases present in the pretraining corpus.",
            ],
            ethical_considerations=[
                "Language models can generate misleading, harmful, or manipulated content "
                "at scale. Deploy with safety filters, output moderation, and provenance "
                "tracking.",
                "Document training data sources and any content filtering applied.",
                "Consider red-teaming and adversarial testing before public deployment.",
                "Respect the licensing terms of all training data sources.",
            ],
            citation=cls._default_citation(model_name),
            license=_DEFAULT_LICENSE,
            authors=list(_DEFAULT_AUTHORS),
            date=time.strftime("%Y-%m-%d"),
            tags=["qai", "text-generation", "transformer", "llm"],
        )

    @classmethod
    def vision_template(cls, model_name: str, accuracy: float,
                         image_size: Union[int, Tuple[int, int]]) -> ModelCardData:
        """Template for a computer-vision model."""
        if isinstance(image_size, int):
            img_h = img_w = int(image_size)
        else:
            img_h, img_w = int(image_size[0]), int(image_size[1])
        return ModelCardData(
            model_name=model_name,
            model_version=_DEFAULT_VERSION,
            model_type="vision",
            architecture={
                "framework": "template",
                "type": "cnn",
                "input_image_size": [img_h, img_w],
                "layers": [{"name": "conv_stack", "type": "CNN",
                            "input_size": [img_h, img_w]}],
                "total_params": 0,
                "n_layers": 0,
                "activations": ["ReLU"],
            },
            n_params=0,
            n_layers=0,
            training_data={
                "image_size": [img_h, img_w],
                "note": "Template-generated; document dataset, augmentation pipeline, "
                        "and class balance.",
            },
            eval_metrics={
                "accuracy": round(float(accuracy), 4),
                "image_size": [img_h, img_w],
                "note": "Replace with top-1 / top-5 accuracy, mAP, etc.",
            },
            intended_use=(
                f"This model was trained for **image classification** on {img_h}x{img_w} "
                f"inputs. Achieves an accuracy of {accuracy:.2%} on the reported "
                f"evaluation set. Apply only on images of similar resolution and "
                f"distribution to the training data."
            ),
            limitations=[
                "Performance may degrade on images outside the trained distribution "
                "(different resolution, lighting, sensor, or class balance).",
                f"Inputs must be resized to {img_h}x{img_w} -- different sizes require retraining.",
                "Sensitive to adversarial perturbations; deploy with input validation.",
            ],
            ethical_considerations=[
                "Vision models can enable unauthorized identification or surveillance. "
                "Restrict to consented use cases.",
                "Audit for biased performance across demographic groups before deployment.",
                "Document the provenance and consent status of training images.",
            ],
            citation=cls._default_citation(model_name),
            license=_DEFAULT_LICENSE,
            authors=list(_DEFAULT_AUTHORS),
            date=time.strftime("%Y-%m-%d"),
            tags=["qai", "image-classification", "vision", "cnn"],
        )

    @classmethod
    def rl_template(cls, model_name: str, reward: float, env_name: str) -> ModelCardData:
        """Template for a reinforcement-learning policy."""
        return ModelCardData(
            model_name=model_name,
            model_version=_DEFAULT_VERSION,
            model_type="reinforcement",
            architecture={
                "framework": "template",
                "type": "policy_network",
                "environment": env_name,
                "layers": [{"name": "policy_head", "type": "Policy"}],
                "total_params": 0,
                "n_layers": 0,
                "activations": ["Tanh"],
            },
            n_params=0,
            n_layers=0,
            training_data={
                "environment": env_name,
                "note": "RL agents are trained in a simulated environment -- document "
                        "the simulator, reward shaping, and episode count.",
            },
            eval_metrics={
                "mean_episode_reward": round(float(reward), 4),
                "environment": env_name,
                "note": "Replace with success rate, episode length, etc.",
            },
            intended_use=(
                f"This is a **reinforcement-learning policy** trained in the `{env_name}` "
                f"environment to maximize cumulative reward. Mean episode reward: "
                f"{reward:.2f}. Deploy only in environments matching the training "
                f"distribution; performance is not guaranteed in novel environments."
            ),
            limitations=[
                f"Policy is specific to the `{env_name}` environment -- sim-to-real "
                f"transfer is not guaranteed.",
                "RL policies can behave unpredictably outside the training distribution.",
                "Reward hacking may have produced a policy that exploits simulator quirks "
                "rather than solving the intended task.",
            ],
            ethical_considerations=[
                "Validate in a sandbox before real-world deployment.",
                "Document the reward function -- mis-specified rewards can produce "
                "harmful behavior.",
                "For physical systems, ensure safety constraints and human oversight.",
            ],
            citation=cls._default_citation(model_name),
            license=_DEFAULT_LICENSE,
            authors=list(_DEFAULT_AUTHORS),
            date=time.strftime("%Y-%m-%d"),
            tags=["qai", "reinforcement-learning", "rl", "policy"],
        )


# ============================================================================
# ModelCardRenderer -- produces the final rendered card
# ============================================================================
class ModelCardRenderer:
    """Render a ModelCardData into the requested output format.

    Methods:
        to_markdown(card)         -> HF-style Markdown
        to_html(card, theme=...)  -> self-contained HTML page
        to_yaml(card)             -> YAML front-matter + Markdown body
        to_hf_format(card)         -> exact Hugging Face Hub README.md
    """

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    def to_markdown(self, card: ModelCardData) -> str:
        """Render to a Hugging Face-style Markdown model card."""
        lines: List[str] = []
        lines.append(f"# Model Card: {card.model_name}")
        lines.append("")
        lines.append(
            f"**Version:** `{card.model_version}` | "
            f"**Type:** `{card.model_type}` | "
            f"**Date:** {card.date} | "
            f"**License:** {card.license}"
        )
        lines.append("")
        if card.authors:
            lines.append(f"**Authors:** {', '.join(card.authors)}")
            lines.append("")
        if card.tags:
            lines.append("**Tags:** " + ", ".join(f"`{t}`" for t in card.tags))
            lines.append("")
        lines.append("---")
        lines.append("")

        # Model Details
        lines.append("## Model Details")
        lines.append("")
        lines.append(f"- **Model name:** {card.model_name}")
        lines.append(f"- **Model type:** {card.model_type}")
        lines.append(f"- **Version:** {card.model_version}")
        lines.append(f"- **Date:** {card.date}")
        lines.append(f"- **License:** {card.license}")
        lines.append(f"- **Authors:** {', '.join(card.authors) if card.authors else 'unknown'}")
        lines.append(f"- **Parameters:** {card.n_params:,}")
        lines.append(f"- **Layers:** {card.n_layers}")
        if card.architecture:
            lines.append(f"- **Architecture framework:** "
                         f"{card.architecture.get('framework', 'unknown')} "
                         f"({card.architecture.get('type', 'unknown')})")
            acts = card.architecture.get("activations", [])
            if acts:
                lines.append(f"- **Activations:** {', '.join(acts)}")
        lines.append("")

        # Intended Use
        lines.append("## Intended Use")
        lines.append("")
        lines.append(card.intended_use or "Not specified.")
        lines.append("")

        # Training Data
        lines.append("## Training Data")
        lines.append("")
        if card.training_data:
            lines.append(self._render_dict_as_bullets(card.training_data, indent=0))
        else:
            lines.append("Training data description not provided.")
        lines.append("")

        # Evaluation Results
        lines.append("## Evaluation Results")
        lines.append("")
        if card.eval_metrics:
            lines.append(self._render_dict_as_bullets(card.eval_metrics, indent=0))
        else:
            lines.append("No evaluation results available.")
        lines.append("")

        # Ethical Considerations
        lines.append("## Ethical Considerations")
        lines.append("")
        if card.ethical_considerations:
            for item in card.ethical_considerations:
                lines.append(f"- {item}")
        else:
            lines.append("Not specified.")
        lines.append("")

        # Limitations
        lines.append("## Limitations")
        lines.append("")
        if card.limitations:
            for item in card.limitations:
                lines.append(f"- {item}")
        else:
            lines.append("Not specified.")
        lines.append("")

        # Architecture details (collapsed layer list)
        if card.architecture and card.architecture.get("layers"):
            lines.append("## Architecture Details")
            lines.append("")
            layers = card.architecture["layers"]
            lines.append("| Layer | Type | Params |")
            lines.append("|-------|------|--------|")
            for layer in layers:
                ln = str(layer.get("name", ""))
                lt = str(layer.get("type", ""))
                lp = str(layer.get("params", 0))
                lines.append(f"| `{ln}` | {lt} | {lp} |")
            lines.append("")

        # Citation
        lines.append("## Citation")
        lines.append("")
        if card.citation:
            lines.append("```bibtex")
            lines.append(card.citation)
            lines.append("```")
        else:
            lines.append("If you use this model, please cite the original work.")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Auto-generated by `qai.ModelCardGenerator` on {card.date}.*")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------
    def to_html(self, card: ModelCardData, theme: str = "dark") -> str:
        """Render to a self-contained, professional HTML page."""
        is_dark = theme == "dark"
        if is_dark:
            bg = "#0d1117"
            card_bg = "#161b22"
            text = "#c9d1d9"
            muted = "#8b949e"
            border = "#30363d"
            accent = "#58a6ff"
            accent_bg = "#1f6feb"
            green = "#3fb950"
            yellow = "#d29922"
            red = "#f85149"
        else:
            bg = "#ffffff"
            card_bg = "#f6f8fa"
            text = "#24292f"
            muted = "#57606a"
            border = "#d0d7de"
            accent = "#0969da"
            accent_bg = "#ddf4ff"
            green = "#1a7f37"
            yellow = "#9a6700"
            red = "#cf222e"

        title = _html.escape(f"Model Card: {card.model_name}")
        subtitle = _html.escape(
            f"{card.model_type} | v{card.model_version} | {card.date}"
        )

        # Build detail cards
        details_html = self._html_section("Model Details", self._html_model_details(card),
                                          accent_bg, text, muted, border)
        arch_html = self._html_section("Architecture", self._html_architecture(card),
                                       card_bg, text, muted, border)
        intended_html = self._html_section("Intended Use",
                                            f"<p>{self._md_inline(_html.escape(card.intended_use or 'Not specified.'))}</p>",
                                            card_bg, text, muted, border)
        training_html = self._html_section("Training Data",
                                            self._html_dict(card.training_data or {}, muted),
                                            card_bg, text, muted, border)
        eval_html = self._html_section("Evaluation Results",
                                        self._html_eval_metrics(card.eval_metrics or {}, green, yellow, red),
                                        card_bg, text, muted, border)
        ethical_html = self._html_section("Ethical Considerations",
                                           self._html_list(card.ethical_considerations),
                                           card_bg, text, muted, border)
        limitations_html = self._html_section("Limitations",
                                               self._html_list(card.limitations),
                                               card_bg, text, muted, border)
        citation_html = self._html_section("Citation",
                                            f"<pre class='citation'>{_html.escape(card.citation or 'N/A')}</pre>",
                                            card_bg, text, muted, border)

        # Tags
        tags_html = "".join(
            f"<span class='tag'>{_html.escape(str(t))}</span>" for t in (card.tags or [])
        )

        # Big metric badges
        badges = self._html_metric_badges(card.eval_metrics or {}, green, yellow, red, accent)

        page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    background: {bg};
    color: {text};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue",
                 Arial, sans-serif;
    font-size: 15px; line-height: 1.6;
  }}
  .container {{
    max-width: 980px; margin: 0 auto; padding: 40px 24px 80px;
  }}
  .header {{
    border-bottom: 1px solid {border};
    padding-bottom: 24px; margin-bottom: 32px;
  }}
  .header h1 {{
    margin: 0 0 8px 0; font-size: 32px; font-weight: 600;
    color: {text};
  }}
  .header .subtitle {{
    color: {muted}; font-size: 14px; margin-bottom: 12px;
  }}
  .tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }}
  .tag {{
    background: {accent_bg}; color: {accent};
    padding: 3px 10px; border-radius: 12px; font-size: 12px;
    font-weight: 500;
  }}
  .badges {{
    display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0 32px 0;
  }}
  .badge {{
    background: {card_bg}; border: 1px solid {border};
    border-radius: 8px; padding: 12px 18px; min-width: 120px;
    text-align: center;
  }}
  .badge .value {{
    font-size: 22px; font-weight: 600; display: block;
  }}
  .badge .label {{
    font-size: 11px; color: {muted}; text-transform: uppercase;
    letter-spacing: 0.5px; margin-top: 4px; display: block;
  }}
  .badge.green .value {{ color: {green}; }}
  .badge.yellow .value {{ color: {yellow}; }}
  .badge.red   .value {{ color: {red}; }}
  .badge.accent .value {{ color: {accent}; }}
  .grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
  }}
  @media (max-width: 720px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .section {{
    background: {card_bg}; border: 1px solid {border};
    border-radius: 8px; padding: 20px 24px; margin-bottom: 20px;
  }}
  .section h2 {{
    margin: 0 0 14px 0; font-size: 18px; font-weight: 600;
    color: {text}; border-bottom: 1px solid {border};
    padding-bottom: 8px;
  }}
  .section ul {{ margin: 8px 0; padding-left: 22px; }}
  .section li {{ margin: 4px 0; }}
  .section p {{ margin: 8px 0; }}
  .kv {{ display: grid; grid-template-columns: 180px 1fr; gap: 6px 12px; }}
  .kv .k {{ color: {muted}; font-weight: 500; }}
  .kv .v {{ color: {text}; word-break: break-word; }}
  .sub-kv {{ padding-left: 16px; border-left: 2px solid {border}; margin: 6px 0; }}
  table {{
    width: 100%; border-collapse: collapse; margin-top: 12px;
    font-size: 13px;
  }}
  th, td {{
    text-align: left; padding: 8px 10px;
    border-bottom: 1px solid {border};
  }}
  th {{ color: {muted}; font-weight: 500; text-transform: uppercase;
       font-size: 11px; letter-spacing: 0.5px; }}
  code {{
    background: {bg}; border: 1px solid {border};
    padding: 1px 6px; border-radius: 4px; font-size: 13px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  }}
  pre.citation {{
    background: {bg}; border: 1px solid {border};
    border-radius: 6px; padding: 14px 16px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 12.5px; overflow-x: auto; white-space: pre-wrap;
    color: {text};
  }}
  .footer {{
    color: {muted}; font-size: 12px; text-align: center;
    margin-top: 40px; padding-top: 20px; border-top: 1px solid {border};
  }}
  .pill {{
    display: inline-block; padding: 1px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600;
  }}
  .pill.green {{ background: rgba(63,185,80,0.15); color: {green}; }}
  .pill.yellow {{ background: rgba(210,153,34,0.15); color: {yellow}; }}
  .pill.red {{ background: rgba(248,81,73,0.15); color: {red}; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>{title}</h1>
    <div class="subtitle">{subtitle}</div>
    {f'<div class="tags">{tags_html}</div>' if tags_html else ''}
  </div>

  {badges}

  <div class="grid">
    {details_html}
    {arch_html}
  </div>

  {intended_html}
  {training_html}
  {eval_html}

  <div class="grid">
    {ethical_html}
    {limitations_html}
  </div>

  {citation_html}

  <div class="footer">
    Auto-generated by <code>qai.ModelCardGenerator</code> on {_html.escape(card.date)}.
  </div>
</div>
</body>
</html>
"""
        return page

    # ------------------------------------------------------------------
    # YAML (front matter + markdown body)
    # ------------------------------------------------------------------
    def to_yaml(self, card: ModelCardData) -> str:
        """YAML front-matter + markdown body (Hugging Face style)."""
        front = {
            "model_name": card.model_name,
            "model_type": card.model_type,
            "model_version": card.model_version,
            "license": card.license,
            "date": card.date,
            "authors": list(card.authors),
            "tags": list(card.tags),
            "language": "en",
            "library_name": "qai",
            "n_params": int(card.n_params),
            "n_layers": int(card.n_layers),
        }
        yaml_front = _to_yaml(front, indent=0)
        body = self.to_markdown(card)
        # Strip the leading "# Model Card: ..." line from the body since
        # the YAML front-matter already encodes the name
        body_lines = body.split("\n")
        if body_lines and body_lines[0].startswith("# Model Card:"):
            body_lines = body_lines[1:]
            while body_lines and body_lines[0].strip() == "":
                body_lines = body_lines[1:]
            body = "\n".join(body_lines)
        return f"---\n{yaml_front}\n---\n\n{body}"

    # ------------------------------------------------------------------
    # Hugging Face Hub README.md format
    # ------------------------------------------------------------------
    def to_hf_format(self, card: ModelCardData) -> str:
        """Exact Hugging Face Hub README.md format: YAML front-matter + Markdown body."""
        front: Dict[str, Any] = {
            "language": "en",
            "library_name": "qai",
            "license": card.license if card.license != _DEFAULT_LICENSE else "other",
            "tags": list(card.tags),
        }
        if card.model_type == "language_model":
            front["tags"] = list(card.tags) + ["text-generation", "pytorch"]
            front["inference"] = False
        if card.model_type == "vision":
            front["tags"] = list(card.tags) + ["image-classification"]
        # Add a widget example if we have a task type
        yaml_front = _to_yaml(front, indent=0)
        body = self.to_markdown(card)
        return f"---\n{yaml_front}\n---\n\n{body}"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------
    def _html_section(self, title: str, body_html: str,
                       bg: str, text: str, muted: str, border: str) -> str:
        return (
            f"<div class='section'>"
            f"<h2>{_html.escape(title)}</h2>"
            f"{body_html}"
            f"</div>"
        )

    def _html_model_details(self, card: ModelCardData) -> str:
        rows = [
            ("Model name", card.model_name),
            ("Model type", card.model_type),
            ("Version", card.model_version),
            ("Date", card.date),
            ("License", card.license),
            ("Authors", ", ".join(card.authors) if card.authors else "unknown"),
            ("Parameters", f"{card.n_params:,}"),
            ("Layers", str(card.n_layers)),
        ]
        if card.architecture:
            rows.append(("Framework", card.architecture.get("framework", "unknown")))
            rows.append(("Architecture type", card.architecture.get("type", "unknown")))
            acts = card.architecture.get("activations", [])
            if acts:
                rows.append(("Activations", ", ".join(acts)))
        kv = "".join(
            f"<div class='k'>{_html.escape(str(k))}</div>"
            f"<div class='v'>{_html.escape(str(v))}</div>"
            for k, v in rows
        )
        return f"<div class='kv'>{kv}</div>"

    def _html_architecture(self, card: ModelCardData) -> str:
        if not card.architecture:
            return "<p><em>Architecture information not available.</em></p>"
        arch = card.architecture
        layers = arch.get("layers", [])
        if not layers:
            kv_rows = [
                ("Framework", arch.get("framework", "unknown")),
                ("Type", arch.get("type", "unknown")),
                ("Total params", f"{arch.get('total_params', 0):,}"),
                ("Layers", str(arch.get("n_layers", 0))),
            ]
            kv = "".join(
                f"<div class='k'>{_html.escape(str(k))}</div>"
                f"<div class='v'>{_html.escape(str(v))}</div>"
                for k, v in kv_rows
            )
            return f"<div class='kv'>{kv}</div>"

        rows = "".join(
            f"<tr><td><code>{_html.escape(str(l.get('name', '')))}</code></td>"
            f"<td>{_html.escape(str(l.get('type', '')))}</td>"
            f"<td>{_html.escape(str(l.get('params', 0)))}</td></tr>"
            for l in layers[:30]
        )
        return (
            f"<div class='kv'>"
            f"<div class='k'>Framework</div><div class='v'>{_html.escape(str(arch.get('framework','unknown')))}</div>"
            f"<div class='k'>Type</div><div class='v'>{_html.escape(str(arch.get('type','unknown')))}</div>"
            f"<div class='k'>Total params</div><div class='v'>{arch.get('total_params', 0):,}</div>"
            f"</div>"
            f"<table><thead><tr><th>Layer</th><th>Type</th><th>Params</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    def _html_dict(self, d: Dict[str, Any], muted: str) -> str:
        if not d:
            return "<p><em>Not specified.</em></p>"

        def _render(obj: Any, depth: int = 0) -> str:
            if isinstance(obj, dict):
                if not obj:
                    return "<em>(empty)</em>"
                items = []
                for k, v in obj.items():
                    if isinstance(v, (dict, list)) and v:
                        items.append(
                            f"<div class='k'>{_html.escape(str(k))}</div>"
                            f"<div class='v sub-kv'>{_render(v, depth + 1)}</div>"
                        )
                    else:
                        items.append(
                            f"<div class='k'>{_html.escape(str(k))}</div>"
                            f"<div class='v'>{_html.escape(self._fmt_scalar(v))}</div>"
                        )
                return f"<div class='kv' style='grid-template-columns: 160px 1fr;'>{''.join(items)}</div>"
            if isinstance(obj, list):
                if not obj:
                    return "<em>(empty list)</em>"
                items = "".join(f"<li>{_html.escape(self._fmt_scalar(x))}</li>" for x in obj)
                return f"<ul>{items}</ul>"
            return _html.escape(self._fmt_scalar(obj))

        return _render(d)

    def _html_eval_metrics(self, m: Dict[str, Any], green: str, yellow: str,
                            red: str) -> str:
        if not m:
            return "<p><em>No evaluation metrics available.</em></p>"
        # Pull out the headline metrics for pills
        headline_keys = ("accuracy", "f1_macro", "precision_macro", "recall_macro",
                         "r2", "mse", "rmse", "mae", "balanced_accuracy",
                         "cohen_kappa", "latency_ms", "throughput_per_s")
        parts: List[str] = []
        headline_html = ""
        for k in headline_keys:
            if k in m and isinstance(m[k], (int, float)):
                val = float(m[k])
                # Color: green for good, yellow for ok, red for poor
                if k in ("accuracy", "f1_macro", "precision_macro", "recall_macro",
                          "balanced_accuracy", "cohen_kappa", "r2"):
                    cls = "green" if val >= 0.85 else ("yellow" if val >= 0.6 else "red")
                elif k in ("mse", "rmse", "mae", "latency_ms"):
                    cls = "accent"  # neutral
                else:
                    cls = "accent"
                headline_html += (
                    f"<span class='pill {cls}'>{_html.escape(k)}: "
                    f"{val:.4f}</span> "
                )
        if headline_html:
            parts.append(f"<div style='margin-bottom: 12px;'>{headline_html}</div>")
        parts.append(self._html_dict(m, muted=green))
        return "".join(parts)

    def _html_list(self, items: Sequence[str]) -> str:
        if not items:
            return "<p><em>Not specified.</em></p>"
        return "<ul>" + "".join(
            f"<li>{_html.escape(str(i))}</li>" for i in items
        ) + "</ul>"

    def _html_metric_badges(self, m: Dict[str, Any], green: str, yellow: str,
                              red: str, accent: str) -> str:
        """Big metric badges at the top of the page."""
        badges: List[str] = []
        if "accuracy" in m and isinstance(m["accuracy"], (int, float)):
            val = float(m["accuracy"])
            cls = "green" if val >= 0.85 else ("yellow" if val >= 0.6 else "red")
            badges.append(self._badge("Accuracy", f"{val:.2%}", cls))
        if "f1_macro" in m and isinstance(m["f1_macro"], (int, float)):
            val = float(m["f1_macro"])
            cls = "green" if val >= 0.85 else ("yellow" if val >= 0.6 else "red")
            badges.append(self._badge("F1 (macro)", f"{val:.4f}", cls))
        if "r2" in m and isinstance(m["r2"], (int, float)):
            val = float(m["r2"])
            cls = "green" if val >= 0.7 else ("yellow" if val >= 0.4 else "red")
            badges.append(self._badge("R²", f"{val:.4f}", cls))
        if "n_params" not in m and "n_classes" in m and isinstance(m["n_classes"], (int, float)):
            badges.append(self._badge("Classes", str(int(m["n_classes"])), "accent"))
        if "latency_ms" in m and isinstance(m["latency_ms"], (int, float)):
            badges.append(self._badge("Latency", f"{float(m['latency_ms']):.3f} ms", "accent"))
        if "throughput_per_s" in m and isinstance(m["throughput_per_s"], (int, float)):
            val = float(m["throughput_per_s"])
            badges.append(self._badge("Throughput", f"{val:,.0f}/s", "accent"))
        if not badges:
            return ""
        return f"<div class='badges'>{''.join(badges)}</div>"

    def _badge(self, label: str, value: str, cls: str) -> str:
        return (
            f"<div class='badge {cls}'>"
            f"<span class='value'>{_html.escape(value)}</span>"
            f"<span class='label'>{_html.escape(label)}</span>"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _md_inline(s: str) -> str:
        """Convert a small subset of inline markdown to HTML.

        Supports `` `code` ``, `` **bold** ``, and `` *italic* ``. Other
        text is left untouched. Call this AFTER html.escape() -- it works
        on already-escaped strings (so ``*`` and backticks survive).
        """
        import re
        # `` `code` `` -- match runs between backticks (escaped text has no backticks left)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        # **bold**
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        # *italic*
        s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
        return s

    def _render_dict_as_bullets(self, d: Dict[str, Any], indent: int = 0) -> str:
        """Render a (possibly nested) dict as markdown bullets."""
        pad = "  " * indent
        lines: List[str] = []
        for k, v in d.items():
            key_str = str(k)
            if isinstance(v, dict):
                if v:
                    lines.append(f"{pad}- **{key_str}:**")
                    lines.append(self._render_dict_as_bullets(v, indent + 1))
                else:
                    lines.append(f"{pad}- **{key_str}:** {{}}")
            elif isinstance(v, list):
                if v:
                    lines.append(f"{pad}- **{key_str}:**")
                    for item in v:
                        if isinstance(item, (dict, list)):
                            lines.append(f"{pad}  - `{self._fmt_scalar(item)}`")
                        else:
                            lines.append(f"{pad}  - {self._fmt_scalar(item)}")
                else:
                    lines.append(f"{pad}- **{key_str}:** []")
            else:
                lines.append(f"{pad}- **{key_str}:** {self._fmt_scalar(v)}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_scalar(v: Any) -> str:
        """Format a scalar value for display."""
        if v is None:
            return "—"
        if isinstance(v, float):
            if abs(v) < 0.0001 and v != 0:
                return f"{v:.6e}"
            return f"{v:.6g}"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (list, tuple)):
            try:
                return ", ".join(str(x) for x in v)
            except Exception:
                return str(v)
        return str(v)


# ============================================================================
# Top-level convenience functions
# ============================================================================
def generate(model: Any, X: Any = None, y: Any = None,
              format: str = "markdown", **kwargs) -> str:
    """Auto-generate a model card for `model` and render it.

    Args:
        model: a trained qai.Model, sklearn estimator, or torch nn.Module.
        X: optional evaluation inputs.
        y: optional evaluation targets.
        format: one of "markdown", "html", "json", "yaml", "hf".
        **kwargs: optional overrides passed to ModelCardData fields
                  (e.g. model_name="my_model", license="MIT").

    Returns:
        Rendered model card string.
    """
    extractor = ModelCardExtractor(model, X=X, y=y)
    card = extractor.extract_all()

    # Apply user overrides
    for k, v in kwargs.items():
        if hasattr(card, k):
            setattr(card, k, v)

    fmt = (format or "markdown").lower()
    renderer = ModelCardRenderer()
    if fmt in ("md", "markdown"):
        return renderer.to_markdown(card)
    if fmt == "html":
        return renderer.to_html(card, theme=kwargs.get("theme", "dark"))
    if fmt == "json":
        return card.to_json()
    if fmt == "yaml":
        return renderer.to_yaml(card)
    if fmt in ("hf", "huggingface"):
        return renderer.to_hf_format(card)
    raise ValueError(
        f"[WHAT] unknown format '{format}'\n"
        f"[WHY] format must be one of: markdown, html, json, yaml, hf\n"
        f"[FIX] Use one of the supported formats."
    )


def quick(model: Any, name: Optional[str] = None) -> str:
    """Generate a quick markdown model card (no evaluation data required).

    Args:
        model: a trained model.
        name: optional model name override.

    Returns:
        Markdown string.
    """
    kwargs = {}
    if name is not None:
        kwargs["model_name"] = name
    return generate(model, format="markdown", **kwargs)


def from_template(template_name: str, **kwargs) -> ModelCardData:
    """Build a ModelCardData from a named template.

    Args:
        template_name: one of "classification", "regression", "llm",
                       "vision", "rl".
        **kwargs: template-specific arguments (e.g. accuracy, n_classes).

    Returns:
        A populated ModelCardData.

    Raises:
        ValueError: if template_name is unknown or required kwargs are missing.
    """
    templates = {
        "classification": ModelCardTemplate.classification_template,
        "classifier": ModelCardTemplate.classification_template,
        "regression": ModelCardTemplate.regression_template,
        "regressor": ModelCardTemplate.regression_template,
        "llm": ModelCardTemplate.llm_template,
        "language_model": ModelCardTemplate.llm_template,
        "vision": ModelCardTemplate.vision_template,
        "image": ModelCardTemplate.vision_template,
        "rl": ModelCardTemplate.rl_template,
        "reinforcement": ModelCardTemplate.rl_template,
    }
    if template_name not in templates:
        raise ValueError(
            f"[WHAT] unknown template '{template_name}'\n"
            f"[WHY] template must be one of: {sorted(templates.keys())}\n"
            f"[FIX] Use one of the supported template names."
        )
    return templates[template_name](**kwargs)


def validate(card: ModelCardData) -> List[str]:
    """Validate a ModelCardData; return list of missing required fields.

    Args:
        card: a ModelCardData instance.

    Returns:
        List of issue strings (empty = card is valid).
    """
    return card.validate()


# ============================================================================
# Internal helpers
# ============================================================================
def _json_default(o: Any) -> Any:
    """JSON default fallback for non-serializable types."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return list(o)
    return str(o)


def _to_yaml(data: Any, indent: int = 0) -> str:
    """Minimal YAML serializer (no external deps).

    Supports dicts, lists, strings, numbers, booleans, None. Enough for
    model-card front-matter -- not a full YAML implementation.
    """
    pad = "  " * indent
    if isinstance(data, dict):
        if not data:
            return "{}"
        lines: List[str] = []
        for k, v in data.items():
            if isinstance(v, dict):
                if v:
                    lines.append(f"{pad}{k}:")
                    lines.append(_to_yaml(v, indent + 1))
                else:
                    lines.append(f"{pad}{k}: {{}}")
            elif isinstance(v, list):
                if v:
                    lines.append(f"{pad}{k}:")
                    lines.append(_to_yaml(v, indent + 1))
                else:
                    lines.append(f"{pad}{k}: []")
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(data, list):
        if not data:
            return f"{pad}[]"
        lines = []
        for item in data:
            if isinstance(item, dict):
                if item:
                    sub = _to_yaml(item, indent + 1)
                    # First line gets the "- " prefix
                    sub_lines = sub.split("\n")
                    lines.append(f"{pad}- {sub_lines[0].strip()}")
                    for sl in sub_lines[1:]:
                        lines.append(sl)
                else:
                    lines.append(f"{pad}- {{}}")
            elif isinstance(item, list):
                lines.append(f"{pad}- {_yaml_scalar(item)}")
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return _yaml_scalar(data)


def _yaml_scalar(v: Any) -> str:
    """Format a scalar value as a YAML-safe string."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, float) else str(v)
    s = str(v)
    # Quote if string contains special chars or is empty
    needs_quote = (
        not s
        or any(c in s for c in [":", "#", "\n", '"', "'", "{", "}", "[", "]",
                                  ",", "&", "*", "?", "|", ">", "@", "`", "%"])
        or s.lower() in ("true", "false", "null", "yes", "no", "on", "off", "~")
        or s[0] in ("-", " ", "?", "[", "]", "{", "}", "&", "*", "!", "|", ">", "%", "@", "`")
    )
    if needs_quote:
        return json.dumps(s)
    return s


__all__ = [
    "ModelCardData",
    "ModelCardExtractor",
    "ModelCardTemplate",
    "ModelCardRenderer",
    "generate",
    "quick",
    "from_template",
    "validate",
]
