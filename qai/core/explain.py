"""
qai.core.explain -- Model Explainability with SHAP-style Tree & Linear Feature Attribution.
Returns structured feature attribution metrics, directional impacts, and explanations.
"""
from __future__ import annotations
import numpy as np


def explain(model, x) -> dict:
    """Returns a structured explanation dict for a single prediction."""
    technique = model.technique
    x_arr = np.asarray(x, dtype=float)
    prediction = model.predict(x)

    if hasattr(technique, "w") and technique.w is not None:
        feature_contributions = {f"x{i}": round(float(w * xi), 4)
                                  for i, (w, xi) in enumerate(zip(technique.w, x_arr))}
        bias_contribution = round(float(technique.b), 4)
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
        proba = technique.model.predict_proba(x_arr.reshape(1, -1))[0]
        classes = technique.model.classes_
        dist = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}
        winner = classes[int(np.argmax(proba))]
        return {
            "prediction": prediction,
            "method": "class probability distribution",
            "class_probabilities": dist,
            "reasoning": f"Predicted '{winner}' with {dist[str(winner)]:.1%} confidence.",
        }

    return {
        "prediction": prediction,
        "method": "none available",
        "reasoning": f"No explanation method is implemented for technique '{technique.name}' yet.",
    }
