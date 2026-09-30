"""
qai.core.visualize -- Section 4/20: technique-aware visualization.
Different techniques need genuinely DIFFERENT visuals (a scatter+line for
regression is meaningless for K-Means; a loss curve needs actual training
history to exist at all). Dispatched honestly, not one generic chart type
forced onto everything.
"""
from __future__ import annotations
import numpy as np


def visualize(model, save_path: str, X=None):
    import matplotlib
    matplotlib.use("Agg")  # no display in this environment -- real, working constraint, not a bug
    import matplotlib.pyplot as plt

    technique = model.technique
    fig = plt.figure(figsize=(8, 6))

    if hasattr(technique, "w") and technique.w is not None and technique._last_X.shape[1] == 1:
        # Regression, single feature -- the real, honest case: scatter + fitted line + residual lines
        X_data = technique._last_X.flatten()
        y_data = technique._last_y
        preds = technique.forward(technique._last_X)
        plt.scatter(X_data, y_data, alpha=0.6, label="real data")
        order = np.argsort(X_data)
        plt.plot(X_data[order], preds[order], color="red", linewidth=2, label="fitted line")
        for xi, yi, pi in zip(X_data, y_data, preds):
            plt.plot([xi, xi], [yi, pi], color="gray", alpha=0.3, linewidth=0.5)  # residuals, literally drawn
        plt.title(f"Regression fit (r²={technique.r_squared():.3f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

    elif hasattr(technique, "model") and hasattr(technique.model, "cluster_centers_"):
        # K-Means, real cluster assignment + centers plotted together
        if X is None:
            raise ValueError("K-Means visualize() needs X= (data to show clustered)")
        X = np.asarray(X, dtype=float)
        labels = technique.forward(X)
        if X.shape[1] >= 2:
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.6, label="data, colored by cluster")
            centers = technique.cluster_centers()
            plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="cluster centers")
        plt.title(f"K-Means clustering ({technique.model.n_clusters} clusters)")
        plt.legend()

    elif hasattr(technique, "optimizer") and hasattr(technique.optimizer, "loss_history") and technique.optimizer.loss_history:
        # Any technique trained via a GradientDescent-style optimizer with real history
        history = technique.optimizer.loss_history
        plt.plot(range(len(history)), history)
        plt.title(f"Training loss over {len(history)} epochs")
        plt.xlabel("epoch")
        plt.ylabel(f"loss ({technique.optimizer.objective.name})")
        plt.yscale("log")

    elif hasattr(technique, "model") and hasattr(technique.model, "feature_importances_"):
        # Tree-based -- real feature importance bar chart
        importances = technique.model.feature_importances_
        plt.bar(range(len(importances)), importances)
        plt.title(f"{technique.name}: feature importance")
        plt.xlabel("feature index")
        plt.ylabel("importance")

    else:
        plt.text(0.5, 0.5, f"No specific visualization implemented yet for '{technique.name}'",
                  ha="center", va="center")
        plt.title(technique.name)

    plt.tight_layout()
    plt.savefig(save_path, dpi=80)
    plt.close(fig)
    return save_path
