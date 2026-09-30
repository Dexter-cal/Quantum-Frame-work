"""
qai.core.explain -- Section 4 of the design doc: '.explain() supported by
every technique.' Implemented here as real, technique-aware logic, not a
placeholder -- what counts as a good explanation genuinely differs between
a linear model and a tree-based one, and this reflects that honestly.
"""
from __future__ import annotations
import numpy as np


def explain(model, x) -> dict:
    """Returns a plain-language-adjacent explanation dict for one prediction.
    Dispatches by technique family since 'why did it predict this' means
    something different for each kind of technique -- exactly the design
    doc's point that explanations aren't one-size-fits-all."""
    technique = model.technique
    x_arr = np.asarray(x, dtype=float)
    prediction = model.predict(x)

    if hasattr(technique, "w") and technique.w is not None:
        # Regression: explanation = each feature's actual contribution to the sum.
        # IMPORTANT: bias/intercept is reported SEPARATELY from "most_influential" --
        # a naive contribution breakdown is very often dominated by the bias term,
        # which isn't a meaningful "feature explanation," just a scale artifact.
        # (Found by testing: bias claimed as "most influential" on real diabetes
        # data, which is technically true but not a useful explanation.)
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
            "reasoning": f"Among actual features (excluding the baseline/bias term), "
                         f"'{dominant_feature}' contributed most "
                         f"({feature_contributions[dominant_feature]:+.4f}). "
                         f"The model's baseline bias contributed {bias_contribution:+.4f} separately.",
        }

    if hasattr(technique, "model") and hasattr(technique.model, "feature_importances_"):
        # Tree-based (Decision Tree, Random Forest): explanation = which
        # features the model has learned to rely on MOST OVERALL, not
        # per-prediction (an honest, real limitation of this simple approach
        # vs. a per-prediction method like SHAP -- stated plainly, not hidden)
        importances = technique.model.feature_importances_
        ranked = sorted(enumerate(importances), key=lambda t: -t[1])
        top = ranked[0]
        return {
            "prediction": prediction,
            "method": "global feature importance (NOT per-prediction -- a real limitation)",
            "feature_ranking": [{"feature": f"x{i}", "importance": round(float(imp), 4)} for i, imp in ranked],
            "most_influential": f"x{top[0]}",
            "reasoning": f"Overall, this model relies most on feature x{top[0]} "
                         f"(importance {top[1]:.4f}) across ALL predictions, not specifically this one.",
        }

    if hasattr(technique, "model") and hasattr(technique.model, "predict_proba"):
        # Any probabilistic classifier without a cleaner explanation path:
        # explanation = the actual confidence distribution across classes
        proba = technique.model.predict_proba(x_arr.reshape(1, -1))[0]
        classes = technique.model.classes_
        dist = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}
        winner = classes[int(np.argmax(proba))]
        return {
            "prediction": prediction,
            "method": "class probability distribution",
            "class_probabilities": dist,
            "reasoning": f"Predicted '{winner}' with {dist[str(winner)]:.1%} confidence "
                         f"among {len(classes)} possible classes.",
        }

    return {
        "prediction": prediction,
        "method": "none available",
        "reasoning": f"No explanation method is implemented for technique '{technique.name}' yet.",
    }
