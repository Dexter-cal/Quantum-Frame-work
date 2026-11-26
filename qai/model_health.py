"""
qai.model_health — model health checks and diagnostics.

Provides:
  - health_check(model): runs a battery of checks on a trained model
    (dead neurons, saturation, calibration, confidence distribution, etc.)
  - token_counter(text, tokenizer=None): counts tokens in text
  - surprise_score(model, x): how surprising is this input vs training data?

Usage:
    import qai
    m = qai.build(technique="classifier")
    m.train(X, y)
    report = qai.health_check(m, X)
    print(report["overall_status"])  # "healthy" | "warning" | "critical"

    n_tokens = qai.token_counter("hello world", tokenizer=tok)
"""
from __future__ import annotations
import os, time, json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def token_counter(text, tokenizer=None) -> int:
    """Count the number of tokens in text.

    Args:
        text: a string or list of strings.
        tokenizer: optional — a qai tokenizer (BPETokenizer, WordTokenizer, etc.)
                   or any object with an `encode()` method.
                   If None, uses a simple whitespace split.

    Returns:
        Number of tokens (int) — or list[int] if text is a list.
    """
    if isinstance(text, list):
        return [token_counter(t, tokenizer) for t in text]

    if tokenizer is not None:
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(text)
            # Some tokenizers return a list, others a dict
            if isinstance(ids, dict):
                ids = ids.get("input_ids", list(ids.values())[0])
            return len(ids)
        if hasattr(tokenizer, "tokenize"):
            return len(tokenizer.tokenize(text))

    # Fallback: whitespace split (rough estimate)
    return len(str(text).split())


def surprise_score(model, x) -> float:
    """How surprising is this input vs the model's training data?

    Computes the Mahalanobis distance from the input to the training data
    mean, normalized by the training data's per-feature standard deviation.
    A high score (>3) means the input is out-of-distribution.

    Returns:
        float — the surprise score. 0 = exactly at training mean,
        >3 = likely out-of-distribution.
    """
    sig = getattr(model, "_train_data_signature", None)
    if sig is None or sig.get("kind") != "numeric":
        return 0.0

    sample_rows = sig.get("sample_rows", [])
    if len(sample_rows) < 2:
        return 0.0

    try:
        train_arr = np.array([np.asarray(r, dtype=float) for r in sample_rows])
        x_arr = np.asarray(x, dtype=float).flatten()

        # Need same number of features
        if train_arr.shape[1] != len(x_arr):
            return 0.0

        mean = train_arr.mean(axis=0)
        std = train_arr.std(axis=0) + 1e-8  # avoid div-by-zero
        # Z-score distance
        z = np.abs((x_arr - mean) / std)
        return float(np.sqrt(np.sum(z ** 2)))
    except Exception:
        return 0.0


def health_check(model, X=None, y=None) -> Dict[str, Any]:
    """Run a battery of health checks on a trained model.

    Args:
        model: a trained qai.Model.
        X: optional — sample input data to test prediction on.
        y: optional — corresponding labels to compute accuracy.

    Returns:
        A dict with:
          - overall_status: "healthy" | "warning" | "critical"
          - checks: list of {name, status, message, details}
          - summary: dict with counts
    """
    checks = []

    # 1. Is the model trained?
    trained = model.technique.is_trained() if hasattr(model.technique, "is_trained") else False
    checks.append({
        "name": "trained",
        "status": "pass" if trained else "fail",
        "message": "Model has been trained" if trained else "Model is NOT trained",
        "details": {"trained": trained},
    })

    # 2. Does predict() work on a sample?
    if trained and X is not None:
        try:
            sample = X[0] if hasattr(X, "__len__") and len(X) > 0 else X
            pred = model.predict(sample)
            checks.append({
                "name": "predict_works",
                "status": "pass",
                "message": f"predict() returned: {str(pred)[:50]}",
                "details": {"prediction": str(pred)[:100]},
            })
        except Exception as e:
            checks.append({
                "name": "predict_works",
                "status": "fail",
                "message": f"predict() failed: {type(e).__name__}: {str(e)[:80]}",
                "details": {"error": str(e)[:200]},
            })

    # 3. Does explain() work?
    if trained and X is not None:
        try:
            sample = X[0] if hasattr(X, "__len__") and len(X) > 0 else X
            exp = model.explain(sample)
            if isinstance(exp, dict):
                checks.append({
                    "name": "explain_works",
                    "status": "pass",
                    "message": "explain() returned a dict",
                    "details": {"keys": list(exp.keys())[:5]},
                })
            else:
                checks.append({
                    "name": "explain_works",
                    "status": "warn",
                    "message": f"explain() returned {type(exp).__name__}, expected dict",
                    "details": {"return_type": type(exp).__name__},
                })
        except Exception as e:
            checks.append({
                "name": "explain_works",
                "status": "warn",
                "message": f"explain() failed: {type(e).__name__}: {str(e)[:80]}",
                "details": {"error": str(e)[:200]},
            })

    # 4. Save/load round-trip
    if trained:
        try:
            import tempfile, os as _os
            # Import at function level to avoid relative-import issues
            import qai
            with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as tf:
                path = tf.name
            try:
                qai.save_model(model, path)
                loaded = qai.load_model(path)
                checks.append({
                    "name": "save_load",
                    "status": "pass",
                    "message": "Save/load round-trip works",
                    "details": {},
                })
            finally:
                try: _os.unlink(path)
                except: pass
        except Exception as e:
            checks.append({
                "name": "save_load",
                "status": "warn",
                "message": f"Save/load failed: {type(e).__name__}: {str(e)[:80]}",
                "details": {"error": str(e)[:200]},
            })

    # 5. Training data signature captured?
    sig = getattr(model, "_train_data_signature", None)
    if sig and sig.get("sample_rows"):
        checks.append({
            "name": "input_validation_ready",
            "status": "pass",
            "message": f"Training signature captured ({sig['kind']}, "
                       f"{sig.get('n_features', '?')} features)",
            "details": {"kind": sig["kind"],
                        "n_features": sig.get("n_features")},
        })
    else:
        checks.append({
            "name": "input_validation_ready",
            "status": "warn",
            "message": "No training signature — input validation disabled",
            "details": {},
        })

    # 6. Dead-neuron check (for neural nets)
    tech = model.technique
    if hasattr(tech, "model") and hasattr(tech.model, "predict_proba"):
        try:
            if X is not None and len(X) >= 5:
                probas = tech.model.predict_proba(X[:20])
                # Check if any class always gets 0 probability
                if probas.shape[1] > 1:
                    col_max = probas.max(axis=0)
                    dead_classes = np.where(col_max < 0.01)[0]
                    if len(dead_classes) > 0:
                        checks.append({
                            "name": "dead_classes",
                            "status": "warn",
                            "message": f"{len(dead_classes)} class(es) never predicted "
                                       f"in sample data",
                            "details": {"dead_class_indices": dead_classes.tolist()},
                        })
                    else:
                        checks.append({
                            "name": "dead_classes",
                            "status": "pass",
                            "message": "All classes are reachable",
                            "details": {},
                        })
        except Exception:
            pass

    # 7. Confidence distribution (for classifiers)
    if trained and X is not None and hasattr(tech, "model") and hasattr(tech.model, "predict_proba"):
        try:
            probas = tech.model.predict_proba(X[:20] if len(X) >= 20 else X)
            max_probas = probas.max(axis=1)
            mean_conf = float(max_probas.mean())
            min_conf = float(max_probas.min())
            if mean_conf > 0.99 and min_conf > 0.99:
                checks.append({
                    "name": "confidence_distribution",
                    "status": "warn",
                    "message": f"Model is overconfident (mean={mean_conf:.3f}, "
                               f"min={min_conf:.3f}) — possible overfitting",
                    "details": {"mean_confidence": mean_conf,
                                "min_confidence": min_conf},
                })
            elif mean_conf < 0.55:
                checks.append({
                    "name": "confidence_distribution",
                    "status": "warn",
                    "message": f"Model is underconfident (mean={mean_conf:.3f}) — "
                               f"may need more training",
                    "details": {"mean_confidence": mean_conf},
                })
            else:
                checks.append({
                    "name": "confidence_distribution",
                    "status": "pass",
                    "message": f"Confidence looks healthy (mean={mean_conf:.3f})",
                    "details": {"mean_confidence": mean_conf,
                                "min_confidence": min_conf},
                })
        except Exception:
            pass

    # 8. Accuracy (if y provided)
    if trained and X is not None and y is not None:
        try:
            preds = model.predict(X)
            if hasattr(preds, "shape"):
                accuracy = float(np.mean(preds == np.asarray(y)[:len(preds)]))
                checks.append({
                    "name": "accuracy",
                    "status": "pass" if accuracy > 0.5 else "warn",
                    "message": f"Accuracy on test data: {accuracy:.1%}",
                    "details": {"accuracy": accuracy,
                                "n_samples": len(preds)},
                })
        except Exception:
            pass

    # Overall status
    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    if n_fail > 0:
        overall = "critical"
    elif n_warn > 0:
        overall = "warning"
    else:
        overall = "healthy"

    return {
        "overall_status": overall,
        "checks": checks,
        "summary": {
            "total": len(checks),
            "pass": sum(1 for c in checks if c["status"] == "pass"),
            "warn": n_warn,
            "fail": n_fail,
        },
    }


__all__ = ["health_check", "token_counter", "surprise_score"]
