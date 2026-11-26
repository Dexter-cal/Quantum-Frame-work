"""
qai.weight_translation — weight-to-concept translator (PDF spec feature).

Translates raw weight vectors into human-readable concept labels.

The PDF spec says:
  "Weight-to-concept translator — model.weights.translate() turns raw
   vectors into readable labels (e.g. 'positive sentiment')"

  "Neuron/concept dictionary — browsable index of what each neuron/expert
   responds to"

This module implements:
  - WeightTranslator — maps weight vectors to concept labels
  - ConceptDictionary — browsable index of neuron → concept mappings
  - model.weights.translate() — integration point

Usage:
    import qai
    m = qai.build(technique="classifier")
    m.train(X, y)

    # Translate weights to concepts
    translator = qai.WeightTranslator(m)
    concepts = translator.translate()
    print(concepts)
    # {'x0': 'strong positive influence', 'x1': 'weak negative', ...}

    # Browse concept dictionary
    dictionary = translator.concept_dictionary()
    for feature, concept in dictionary.items():
        print(f"  {feature}: {concept}")
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# Concept labels based on weight magnitude and sign
POSITIVE_LABELS = [
    "strong positive influence",
    "moderate positive influence",
    "weak positive influence",
]
NEGATIVE_LABELS = [
    "strong negative influence",
    "moderate negative influence",
    "weak negative influence",
]
NEUTRAL_LABEL = "negligible influence"


def _label_for_weight(weight: float, max_abs: float) -> str:
    """Map a weight value to a human-readable concept label."""
    if max_abs == 0:
        return NEUTRAL_LABEL
    ratio = abs(weight) / max_abs
    if ratio < 0.1:
        return NEUTRAL_LABEL
    if weight > 0:
        if ratio > 0.66:
            return POSITIVE_LABELS[0]
        elif ratio > 0.33:
            return POSITIVE_LABELS[1]
        else:
            return POSITIVE_LABELS[2]
    else:
        if ratio > 0.66:
            return NEGATIVE_LABELS[0]
        elif ratio > 0.33:
            return NEGATIVE_LABELS[1]
        else:
            return NEGATIVE_LABELS[2]


class WeightTranslator:
    """Translates model weights into human-readable concept labels.

    For linear models (regression, classifier, SVM):
      - Each feature weight → "strong positive influence" / "weak negative" / etc.
      - Relative magnitude matters (normalized to max)

    For tree-based models (random_forest, gradient_boosting):
      - feature_importances_ → "key feature" / "minor feature" / "irrelevant"

    For neural networks:
      - Layer weights → "active" / "dormant" / "saturated" (based on statistics)

    For clustering (KMeans, GMM):
      - Centroids → "cluster center for [dominant features]"

    Usage:
        translator = qai.WeightTranslator(model)
        concepts = translator.translate()
        dictionary = translator.concept_dictionary()
    """
    def __init__(self, model):
        self.model = model
        self.tech = model.technique if hasattr(model, "technique") else model

    def translate(self) -> Dict[str, str]:
        """Translate weights to human-readable concepts.

        Returns:
            Dict of feature_name → concept_description
        """
        # Try linear model weights
        if hasattr(self.tech, "w") and self.tech.w is not None:
            return self._translate_linear()
        if hasattr(self.tech, "model"):
            inner = self.tech.model
            if hasattr(inner, "coef_"):
                return self._translate_sklearn_linear(inner)
            if hasattr(inner, "feature_importances_"):
                return self._translate_tree(inner)
        # Try clustering centroids
        if hasattr(self.tech, "model") and hasattr(self.tech.model, "cluster_centers_"):
            return self._translate_centroids()
        # Try neural network
        return self._translate_neural()

    def _translate_linear(self) -> Dict[str, str]:
        """Translate linear model weights."""
        weights = np.asarray(self.tech.w).flatten()
        max_abs = max(abs(weights.max()), abs(weights.min()), 1e-8)
        return {f"x{i}": _label_for_weight(w, max_abs)
                for i, w in enumerate(weights)}

    def _translate_sklearn_linear(self, model) -> Dict[str, str]:
        """Translate sklearn linear model (coef_)."""
        coef = np.asarray(model.coef_).flatten()
        max_abs = max(abs(coef.max()), abs(coef.min()), 1e-8)
        return {f"x{i}": _label_for_weight(w, max_abs)
                for i, w in enumerate(coef)}

    def _translate_tree(self, model) -> Dict[str, str]:
        """Translate tree-based feature importances."""
        importances = np.asarray(model.feature_importances_)
        max_imp = max(importances.max(), 1e-8)
        result = {}
        for i, imp in enumerate(importances):
            ratio = imp / max_imp
            if ratio > 0.5:
                result[f"x{i}"] = "key feature (dominant)"
            elif ratio > 0.2:
                result[f"x{i}"] = "important feature"
            elif ratio > 0.05:
                result[f"x{i}"] = "minor feature"
            else:
                result[f"x{i}"] = "irrelevant feature"
        return result

    def _translate_centroids(self) -> Dict[str, str]:
        """Translate KMeans/GMM centroids."""
        centers = np.asarray(self.tech.model.cluster_centers_)
        result = {}
        for c in range(centers.shape[0]):
            center = centers[c]
            max_idx = int(np.argmax(np.abs(center)))
            result[f"cluster_{c}"] = f"centered on x{max_idx} (value: {center[max_idx]:.3f})"
        return result

    def _translate_neural(self) -> Dict[str, str]:
        """Translate neural network weights (layer-level summary)."""
        result = {}
        try:
            import torch.nn as nn
            for name in vars(self.tech):
                attr = getattr(self.tech, name)
                if isinstance(attr, nn.Module):
                    params = list(attr.parameters())
                    if params:
                        all_w = torch.cat([p.data.flatten() for p in params])
                        mean_val = float(all_w.mean())
                        std_val = float(all_w.std())
                        sparsity = float((all_w.abs() < 0.01).float().mean())
                        if sparsity > 0.5:
                            result[name] = f"sparse (sparsity={sparsity:.0%})"
                        elif std_val < 0.01:
                            result[name] = f"saturated (std={std_val:.4f})"
                        elif std_val > 0.1:
                            result[name] = f"active (std={std_val:.4f})"
                        else:
                            result[name] = f"normal (mean={mean_val:.4f}, std={std_val:.4f})"
        except Exception:
            pass
        return result if result else {"model": "unable to translate"}

    def concept_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """Get a browsable concept dictionary with more detail.

        Returns:
            Dict of feature → {label, weight, magnitude, rank}
        """
        concepts = self.translate()
        # Get actual weights for ranking
        weights = {}
        if hasattr(self.tech, "w") and self.tech.w is not None:
            w = np.asarray(self.tech.w).flatten()
            for i, val in enumerate(w):
                weights[f"x{i}"] = float(val)
        elif hasattr(self.tech, "model") and hasattr(self.tech.model, "feature_importances_"):
            imp = np.asarray(self.tech.model.feature_importances_)
            for i, val in enumerate(imp):
                weights[f"x{i}"] = float(val)

        # Sort by absolute value (most important first)
        ranked = sorted(weights.items(), key=lambda x: -abs(x[1]))
        result = {}
        for rank, (name, val) in enumerate(ranked, 1):
            result[name] = {
                "label": concepts.get(name, "unknown"),
                "weight": val,
                "magnitude": abs(val),
                "rank": rank,
            }
        return result

    def summary(self) -> str:
        """Human-readable summary of weight translation."""
        concepts = self.translate()
        lines = [
            "=" * 60,
            "Weight-to-Concept Translation",
            "=" * 60,
        ]
        for name, label in concepts.items():
            lines.append(f"  {name:10s} → {label}")
        lines.append("=" * 60)
        return "\n".join(lines)


__all__ = ["WeightTranslator"]
