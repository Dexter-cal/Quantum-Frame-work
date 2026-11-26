"""
qai.core.explain -- Model Explainability with SHAP-style Tree & Linear Feature Attribution.
Returns structured feature attribution metrics, directional impacts, and explanations.
"""
from __future__ import annotations
import numpy as np


def explain(model, x) -> dict:
    """Returns a structured explanation dict for a single prediction."""
    technique = model.technique
    # Try to convert input to a numeric array; if that fails (e.g. text
    # input to a vectorizer), keep the raw value for context.
    try:
        x_arr = np.asarray(x, dtype=float)
    except (ValueError, TypeError):
        # Non-numeric input (strings, mixed) — common for text preprocessors
        # like TfidfVectorizer / CountVectorizer / LabelEncoder.
        prediction = model.predict(x)
        return {
            "prediction": prediction,
            "method": "non-numeric input (no numeric attribution)",
            "input_summary": str(x)[:100],
            "reasoning": (
                f"This technique ('{technique.name}') operates on non-numeric "
                f"input, so per-feature numeric attribution is not applicable. "
                f"The model transformed the input and produced the prediction above."
            ),
        }
    prediction = model.predict(x)

    if hasattr(technique, "w") and technique.w is not None:
        # Coerce w to a 1-D vector — binary classifiers may have shape (1, d),
        # multiclass may have shape (n_classes, d); in that case use the row
        # corresponding to the predicted class if available, else the mean.
        w_arr = np.asarray(technique.w, dtype=float)
        if w_arr.ndim == 2:
            if w_arr.shape[0] == 1:
                w_vec = w_arr[0]
            else:
                # multiclass: pick the row of the predicted class
                try:
                    pred_idx = int(prediction) if isinstance(prediction, (int, np.integer)) else 0
                    w_vec = w_arr[pred_idx] if pred_idx < w_arr.shape[0] else w_arr.mean(axis=0)
                except (ValueError, TypeError):
                    w_vec = w_arr.mean(axis=0)
        else:
            w_vec = w_arr
        # Match length to input
        n = min(len(w_vec), len(x_arr))
        feature_contributions = {f"x{i}": round(float(w_vec[i] * x_arr[i]), 4)
                                  for i in range(n)}
        bias_contribution = round(float(technique.b), 4) if hasattr(technique, "b") and technique.b is not None else 0.0
        if feature_contributions:
            dominant_feature = max(feature_contributions, key=lambda k: abs(feature_contributions[k]))
            return {
                "prediction": prediction,
                "method": "linear contribution breakdown",
                "feature_contributions": feature_contributions,
                "bias_contribution": bias_contribution,
                "most_influential_feature": dominant_feature,
                "reasoning": f"Among actual features (excluding bias), '{dominant_feature}' contributed most "
                             f"({feature_contributions[dominant_feature]:+.4f}). "
                             f"The baseline bias contributed {bias_contribution:+.4f} separately.",
            }

    if hasattr(technique, "model") and hasattr(technique.model, "feature_importances_"):
        importances = technique.model.feature_importances_
        ranked = sorted(enumerate(importances), key=lambda t: -t[1])
        top = ranked[0]
        # Directional attribution proxy using feature value vs baseline mean
        feature_attributions = {
            f"x{i}": round(float(imp * (x_arr[i] if i < len(x_arr) else 1.0)), 4)
            for i, imp in enumerate(importances)
        }
        return {
            "prediction": prediction,
            "method": "tree feature attribution",
            "feature_attributions": feature_attributions,
            "feature_ranking": [{"feature": f"x{i}", "importance": round(float(imp), 4)} for i, imp in ranked],
            "most_influential_feature": f"x{top[0]}",
            "reasoning": f"This tree model relies most on feature x{top[0]} "
                         f"(importance {top[1]:.4f}) across splits.",
        }

    if hasattr(technique, "model") and hasattr(technique.model, "predict_proba"):
        try:
            proba = technique.model.predict_proba(x_arr.reshape(1, -1))[0]
        except Exception:
            proba = None
        if proba is not None:
            # Try to get class labels; fall back to component indices for clustering
            classes = getattr(technique.model, "classes_",
                               getattr(technique, "n_components",
                                       getattr(technique.model, "n_components",
                                               list(range(len(proba))))))
            try:
                iter(classes)
            except TypeError:
                classes = list(range(len(proba)))
            dist = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}
            winner = classes[int(np.argmax(proba))]
            return {
                "prediction": prediction,
                "method": "class probability distribution",
                "class_probabilities": dist,
                "confidence": float(np.max(proba)),
                "reasoning": f"Predicted '{winner}' with {dist[str(winner)]:.1%} confidence.",
            }

    # If the technique has its own means_ / cluster centers (clustering)
    if hasattr(technique, "model") and hasattr(technique.model, "means_"):
        means = technique.model.means_
        x_r = x_arr.reshape(1, -1) if x_arr.ndim == 1 else x_arr
        distances = np.linalg.norm(means - x_r, axis=1)
        ranked = sorted(enumerate(distances), key=lambda t: t[1])
        return {
            "prediction": prediction,
            "method": "cluster distance",
            "cluster_distances": {f"cluster_{i}": round(float(d), 4)
                                    for i, d in ranked},
            "most_influential_feature": f"cluster_{ranked[0][0]}",
            "reasoning": f"Input is closest to cluster {ranked[0][0]} "
                         f"(distance {ranked[0][1]:.4f}).",
        }

    return {
        "prediction": prediction,
        "method": "none available",
        "reasoning": f"No explanation method is implemented for technique '{technique.name}' yet.",
    }
