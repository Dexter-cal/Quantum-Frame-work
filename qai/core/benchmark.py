"""
qai.core.benchmark -- Automated Multi-Technique Comparison & Performance Benchmarking.
Evaluates accuracy, train time, and prediction latency across multiple techniques on a dataset.
"""
from __future__ import annotations
import time
from typing import List, Dict, Any
import numpy as np


def benchmark(techniques: List[str], X, y=None, k: int = 3, verbose: bool = False) -> Dict[str, Any]:
    """Sweeps multiple techniques across X, y and returns comparative metrics."""
    if not techniques:
        raise ValueError("techniques must be a non-empty list of technique names")

    X = np.asarray(X)
    if y is not None:
        y = np.asarray(y)

    from .model import build
    results = []

    for name in techniques:
        try:
            start_train = time.time()
            model = build(type=name)
            if y is not None and model.learning_technique == "supervised":
                model.train(X, y, verbose=verbose)
            else:
                model.train(X, verbose=verbose)
            train_time = time.time() - start_train

            start_pred = time.time()
            preds = model.predict(X)
            pred_time = time.time() - start_pred

            acc = model.accuracy(X, y) if y is not None and hasattr(model, "accuracy") else None

            results.append({
                "technique": name,
                "accuracy": round(float(acc), 4) if acc is not None else "N/A",
                "train_time_s": round(float(train_time), 4),
                "predict_time_s": round(float(pred_time), 4),
                "status": "success"
            })
        except Exception as err:
            results.append({
                "technique": name,
                "error": str(err),
                "status": "failed"
            })

    # Identify top performing model
    valid_results = [r for r in results if r["status"] == "success" and isinstance(r["accuracy"], (int, float))]
    best_technique = max(valid_results, key=lambda r: r["accuracy"])["technique"] if valid_results else None

    return {
        "leaderboard": results,
        "best_technique": best_technique,
        "n_samples": len(X)
    }
