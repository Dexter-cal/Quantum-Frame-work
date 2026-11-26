"""
qai.model_card -- auto-generated model cards for responsible AI.

Generates a Model Card (Mitchell et al. 2019) for any qai model,
including: intended use, training data description, evaluation results,
ethical considerations, limitations, and deployment recommendations.

Usage:
    import qai
    m = qai.build(technique="random_forest")
    m.train(X_train, y_train)
    card = qai.ModelCard(m).generate(X_test=X_test, y_test=y_test,
                                      description="Fraud detection model")
    print(card)              # markdown string
    card.save("model_card.md")
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import time

try:
    from .evaluate import evaluate
except ImportError:
    evaluate = None


class ModelCardGenerator:
    """Auto-generate a Model Card for a trained qai model.

    (Named ModelCardGenerator to avoid collision with the existing
    qai.completion.ModelCard class.)

    Args:
        model: a trained qai.Model.
    """
    def __init__(self, model):
        self.model = model
        tech = getattr(model, "technique", None)
        self.technique_name = getattr(tech, "name", "unknown") if tech else "unknown"
        self.family = getattr(tech, "family", "unknown") if tech else "unknown"

    def generate(self, X_test=None, y_test=None,
                 description: str = "",
                 intended_use: str = "",
                 training_data_desc: str = "",
                 creator: str = "qai ModelCard generator",
                 save_path: Optional[str] = None) -> "ModelCardContent":
        """Generate the model card.

        Args:
            X_test, y_test: test data for evaluation metrics.
            description: what the model does.
            intended_use: what it's designed for.
            training_data_desc: description of training data.
            creator: who created the model.
            save_path: optional path to save the markdown.

        Returns:
            ModelCardContent (has .markdown + .save()).
        """
        trained = self.model.is_trained() if hasattr(self.model, "is_trained") else False
        # Evaluation
        eval_results = {}
        if X_test is not None and y_test is not None and evaluate is not None and trained:
            try:
                report = evaluate(self.model, X_test, y_test, verbose=False)
                eval_results = report.metrics
                eval_verdict = report.verdict
            except Exception:
                eval_verdict = "unknown"

        # Build the card
        lines = [
            f"# Model Card: {description or self.technique_name}",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d')}",
            f"**Creator:** {creator}",
            f"**Framework:** qai",
            "",
            "---",
            "",
            "## Model Details",
            f"- **Technique:** `{self.technique_name}`",
            f"- **Family:** `{self.family}`",
            f"- **Trained:** {trained}",
            f"- **Parameters:** {self._count_params()}",
            "",
            "## Intended Use",
            intended_use or f"This model uses the `{self.technique_name}` technique. "
            "Specify the intended use case here.",
            "",
            "## Training Data",
            training_data_desc or "Training data description not provided. "
            "Document the source, size, and characteristics of the training data.",
            "",
            "## Evaluation Results",
        ]

        if eval_results:
            for k, v in eval_results.items():
                if isinstance(v, (int, float)):
                    lines.append(f"- **{k}:** {v:.4f}")
                elif isinstance(v, (list, dict)):
                    lines.append(f"- **{k}:** {v}")
        else:
            lines.append("No evaluation results available. Provide X_test and y_test to generate metrics.")

        lines.extend([
            "",
            "## Ethical Considerations",
            "- This model should be evaluated for bias across demographic groups before deployment.",
            "- Predictions should not be used as the sole basis for high-stakes decisions.",
            "- Consider the impact of false positives and false negatives in your use case.",
            "",
            "## Limitations",
            "- The model's performance is bounded by the quality and representativeness of the training data.",
            "- The model may not generalize to data distributions significantly different from training.",
            f"- The `{self.technique_name}` technique has inherent limitations — consult the literature.",
            "",
            "## Deployment Recommendations",
            "- Monitor for data drift using `qai.DriftDetector`.",
            "- Set up A/B testing using `qai.ABTest` before full rollout.",
            "- Use `qai.evaluate()` periodically to track model health.",
            "- Consider using `qai.ResourceBudget` for resource-constrained deployments.",
            "",
            "---",
            f"*Auto-generated by qai.ModelCard on {time.strftime('%Y-%m-%d')}*",
        ])

        markdown = "\n".join(lines)
        content = ModelCardContent(markdown=markdown, save_path=save_path)
        return content

    def _count_params(self) -> int:
        """Estimate parameter count."""
        tech = getattr(self.model, "technique", None)
        if tech is None:
            return 0
        if hasattr(tech, "net") and tech.net is not None:
            try:
                return sum(p.numel() for p in tech.net.parameters())
            except Exception:
                pass
        n = 0
        if hasattr(tech, "model"):
            for attr in ("coef_", "intercept_", "feature_importances_"):
                v = getattr(tech.model, attr, None)
                if v is not None and hasattr(v, "size"):
                    try:
                        n += int(v.size)
                    except Exception:
                        pass
        return n


class ModelCardContent:
    """The generated model card content."""
    def __init__(self, markdown: str, save_path: Optional[str] = None):
        self.markdown = markdown
        if save_path:
            self.save(save_path)

    def save(self, path: str) -> str:
        """Save the model card to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.markdown)
        return path

    def __str__(self) -> str:
        return self.markdown

    def __repr__(self) -> str:
        return f"<ModelCardContent {len(self.markdown)} chars>"


__all__ = ["ModelCardGenerator", "ModelCardContent"]
