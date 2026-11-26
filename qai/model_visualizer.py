"""
qai.model_visualizer — universal model architecture visualization for ALL techniques.

Generates visual diagrams (matplotlib) showing the architecture of any qai model,
regardless of technique type. Each of the 40 technique families gets a tailored
visualization:

  deep_learning       — layer-by-layer architecture diagram (CNN, NN, RNN, LSTM, GRU, …)
  classical_ml (tree) — tree ensemble / single-tree diagram
  classical_ml (boost)— boosting chain (GB, AdaBoost, XGBoost, …)
  classical_ml (linear) — feature-weight bar chart
  classical_ml (knn)  — query + neighbors scatter
  classical_ml (nb)   — Naive-Bayes factor graph
  unsupervised (cluster) — cluster scatter plot
  unsupervised (dimred)  — High-D → Reduce → Low-D projection
  preprocessing       — Input → Transform → Output
  generative          — Generator/Discriminator or VAE/Autoencoder/diffusion chain
  graph               — node-message-passing diagram
  reinforcement       — agent ↔ env loop + policy/value head
  model_based_rl      — world-model predictive loop
  planning            — search tree (MCTS) / value iteration grid
  vision              — image → backbone → head
  speech              — waveform → conv-feature → encoder → context
  multimodal          — two-tower (e.g. CLIP)
  modern_llm / modern_ssm / modern_transformer / moe / hybrid — block stack
  neuromorphic        — spiking neuron lattice
  probabilistic       — Bayesian/HMM graphical model
  recommender         — user/item embedding + scorer
  self_supervised     — contrastive anchor/positive/negative
  sequence            — encoder + state sequence
  set                 — permutation-invariant set transformer
  memory_augmented    — controller + external memory matrix
  meta_learning       — inner/outer loop (MAML)
  few_shot            — support/query episode
  metric_learning     — siamese twin towers
  filtering           — predict-update Kalman loop
  manifold            — High-D → manifold → Low-D
  decomposition       — matrix factorization diagram
  feature_selection   — feature-rank bar chart
  feedforward         — input → RBF/ELM hidden → output
  reservoir           — input → fixed reservoir → readout
  online              — single-neuron drift diagram
  time_series         — AR/MA coefficient stem plot
  supervised (other)  — generic with key parameters
  bridge / pipeline   — DAG / federated data flow

Usage:
    import qai
    m = qai.build(technique="cnn", in_channels=1, n_classes=3, conv_channels=[8,16])
    m.train(X, y)
    qai.visualize_architecture(m)                       # opens a matplotlib window
    qai.visualize_architecture(m, save="cnn_arch.png")  # save to file

    # Visualize a technique WITHOUT building a model:
    qai.visualize_technique("transformer")   # shows a generic transformer diagram

    # Visualize a pipeline:
    builder = qai.PipelineBuilder("my_pipe")
    builder.add_step("scaler", ...)
    builder.add_step("model", ..., depends_on=["scaler"])
    qai.visualize_pipeline(builder)
"""
from __future__ import annotations
import os, math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Matplotlib backend: use Agg if headless, otherwise interactive
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.font_manager as fm

# Try to use Chinese-compatible fonts
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


# Color palette for different layer/model types
COLORS = {
    "input": "#5ec8ff",
    "conv": "#00ff9d",
    "pool": "#ffb547",
    "fc": "#c084fc",
    "output": "#ff5470",
    "attention": "#ff79c6",
    "norm": "#a0aac4",
    "embedding": "#5ec8ff",
    "tree": "#00b870",
    "cluster": "#ffb547",
    "linear": "#5ec8ff",
    "ensemble": "#c084fc",
    "pipeline": "#00ff9d",
    "bridge": "#ff79c6",
    "scaler": "#a0aac4",
    "pca": "#ffb547",
    "unknown": "#6b7494",
    "generative": "#ff79c6",
    "graph": "#00ff9d",
    "rl": "#ffb547",
    "memory": "#c084fc",
    "vision": "#5ec8ff",
    "speech": "#00b870",
    "multimodal": "#ff5470",
    "probabilistic": "#a0aac4",
    "recommender": "#ffb547",
    "filtering": "#5ec8ff",
    "neuro": "#ff79c6",
}


# =====================================================================
# Public API
# =====================================================================
def _get_technique_info(model) -> Dict[str, Any]:
    """Extract technique + family info from a model."""
    tech = getattr(model, "technique", None)
    if tech is None:
        # Might be a FederatedBridge or PipelineBuilder
        if hasattr(model, "pieces"):
            return {"name": "federated", "family": "bridge", "tech_obj": None, "is_wrapper": True}
        if hasattr(model, "steps"):
            return {"name": "pipeline", "family": "pipeline", "tech_obj": None, "is_wrapper": True}
        return {"name": "unknown", "family": "unknown", "tech_obj": None, "is_wrapper": False}
    return {
        "name": getattr(tech, "name", "unknown"),
        "family": getattr(tech, "family", "unknown"),
        "tech_obj": tech,
        "is_wrapper": False,
    }


# (family, name) -> draw function  (see _DISPATCH below for resolution)
def visualize_architecture(model, save: Optional[str] = None,
                            figsize: Tuple[float, float] = (12, 8),
                            title: str = "") -> str:
    """Visualize the architecture of any qai model.

    Automatically detects the technique type and dispatches to the appropriate
    family-specific diagram. Falls back to a labeled generic card if the family
    is unknown.

    Args:
        model: any qai.Model, FederatedBridge, or PipelineBuilder.
        save: if provided, save the figure to this path.
        figsize: figure size (width, height).
        title: custom title (default: auto-generated from technique name).

    Returns:
        Path to the saved file (or empty string if not saved).
    """
    info = _get_technique_info(model)
    name = info["name"]
    family = info["family"]

    if not title:
        title = f"Architecture: {name} ({family})"

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Dispatch
    drawer = _resolve_drawer(name, family, info)
    drawer(model, ax, info)

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return save
    plt.show()
    return ""


def visualize_technique(technique_name: str, save: Optional[str] = None) -> str:
    """Visualize a technique's generic architecture WITHOUT building a model.

    Args:
        technique_name: the technique name (e.g. "cnn", "transformer", "random_forest").
        save: if provided, save to this path.

    Returns:
        Path to saved file (or empty string).
    """
    family = _guess_family(technique_name)
    info = {"name": technique_name, "family": family, "tech_obj": None, "is_wrapper": False}
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Technique: {technique_name} ({family})",
                  fontsize=14, fontweight="bold", pad=20)

    drawer = _resolve_drawer(technique_name, family, info)
    drawer(None, ax, info)

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return save
    plt.show()
    return ""


def visualize_pipeline(builder_or_pipeline, save: Optional[str] = None) -> str:
    """Visualize a PipelineBuilder or DAGPipeline as a DAG diagram."""
    if hasattr(builder_or_pipeline, "pieces"):
        return visualize_architecture(builder_or_pipeline, save=save)
    if not hasattr(builder_or_pipeline, "steps"):
        raise ValueError("Expected PipelineBuilder, DAGPipeline, or FederatedBridge")

    steps = builder_or_pipeline.steps
    order = (builder_or_pipeline._order
             if hasattr(builder_or_pipeline, "_order") else list(steps.keys()))

    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(order) * 1.5), 6),
                            constrained_layout=True)
    ax.set_xlim(-1, len(order) + 1)
    ax.set_ylim(-1, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Pipeline DAG", fontsize=14, fontweight="bold", pad=20)

    # Draw nodes
    x_positions = {}
    for i, name in enumerate(order):
        step = steps[name]
        x = i
        y = 2
        color = COLORS["pipeline"]
        _draw_box(ax, x, y, 0.9, 0.7, name,
                  type(step.model).__name__ if hasattr(step, "model") else "?",
                  color)
        x_positions[name] = (x, y)

    # Draw arrows
    for name in order:
        step = steps[name]
        for dep in getattr(step, "depends_on", []):
            if dep in x_positions:
                x1, y1 = x_positions[dep]
                x2, y2 = x_positions[name]
                _draw_arrow(ax, x1 + 0.45, y1, x2 - 0.45, y2)

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return save
    plt.show()
    return ""


# =====================================================================
# Drawing helpers
# =====================================================================
def _draw_box(ax, x, y, w, h, label, sublabel="", color="#00ff9d"):
    """Draw a labeled box."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1", facecolor=color, alpha=0.2,
                          edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y + 0.05 * h, label, ha="center", va="center",
             fontsize=8, fontweight="bold")
    if sublabel:
        ax.text(x, y - 0.18 * h, sublabel, ha="center", va="center",
                 fontsize=6, color="#6b7494")


def _draw_arrow(ax, x1, y1, x2, y2, color="#a0aac4"):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def _draw_layers(ax, layers, title=""):
    """Draw a vertical stack of layers with arrows between them.

    layers: list of (label, sublabel, color) tuples.
    """
    n = len(layers)
    y_start = 9
    y_end = 1
    y_step = (y_start - y_end) / max(n, 1)
    for i, (label, sublabel, color) in enumerate(layers):
        y = y_start - i * y_step
        _draw_box(ax, 5, y, 2.5, y_step * 0.75, label, sublabel, color)
        if i > 0:
            _draw_arrow(ax, 5, y + y_step * 0.75, 5, y + y_step * 0.40)


def _draw_loop(ax, cx, cy, rx, ry, label, color="#ffb547"):
    """Draw a circular loop with a label."""
    circ = mpatches.Ellipse((cx, cy), rx, ry, fill=False,
                             edgecolor=color, linewidth=2, linestyle="--")
    ax.add_patch(circ)
    ax.text(cx, cy + ry + 0.2, label, ha="center", va="bottom",
             fontsize=9, fontweight="bold", color=color)


# =====================================================================
# Dispatch
# =====================================================================
def _resolve_drawer(name: str, family: str, info: dict):
    """Pick the right draw function for a (name, family) pair.

    Try exact-name match first, then family, then a generic fallback.
    """
    # Bridge / pipeline wrappers
    if name == "federated" or family == "bridge":
        return _draw_federated_bridge
    if name == "pipeline" or family == "pipeline":
        return _draw_pipeline

    # Per-name (deep-learning architectures with bespoke diagrams)
    name_map = {
        "cnn": _draw_cnn,
        "transformer": _draw_transformer,
        "neural_network": _draw_neural_network,
        "rnn": _draw_rnn,
        "lstm": _draw_lstm,
        "gru": _draw_gru,
        "seq2seq": _draw_seq2seq,
        "autoencoder": _draw_autoencoder,
        "vae": _draw_vae,
        "gan": _draw_gan,
        "diffusion": _draw_diffusion,
        "moe": _draw_moe,
        "mamba": _draw_mamba,
        "gnn": _draw_gnn,

        "random_forest": _draw_tree_ensemble,
        "extra_trees": _draw_tree_ensemble,
        "decision_tree": _draw_single_tree,
        "decision_tree_regressor": _draw_single_tree,

        "gradient_boosting": _draw_boosting,
        "adaboost": _draw_boosting,
        "hist_gradient_boosting": _draw_boosting,
        "xgboost": _draw_boosting,
        "lightgbm": _draw_boosting,
        "catboost": _draw_boosting,

        "kmeans": _draw_clustering,
        "gmm": _draw_clustering,
        "dbscan": _draw_clustering,
        "hierarchical": _draw_clustering,
        "agglomerative": _draw_clustering,
        "agglomerative_clustering": _draw_clustering,
        "birch": _draw_clustering,
        "mean_shift": _draw_clustering,
        "spectral_clustering": _draw_clustering,
        "affinity_propagation": _draw_clustering,
        "optics": _draw_clustering,
        "bisecting_kmeans": _draw_clustering,
        "mini_batch_kmeans": _draw_clustering,
        "minibatch_kmeans": _draw_clustering,
        "art": _draw_clustering,
        "neural_gas": _draw_clustering,
        "som": _draw_clustering,

        "pca": _draw_dimensionality_reduction,
        "truncated_svd": _draw_dimensionality_reduction,
        "tsne": _draw_dimensionality_reduction,
        "fast_ica": _draw_dimensionality_reduction,
        "isomap": _draw_dimensionality_reduction,
        "lle": _draw_dimensionality_reduction,
        "sparse_pca": _draw_dimensionality_reduction,
        "kernel_pca": _draw_dimensionality_reduction,
        "incremental_pca": _draw_dimensionality_reduction,
        "gaussian_random_projection": _draw_dimensionality_reduction,
        "factor_analysis": _draw_dimensionality_reduction,
        "nmf": _draw_dimensionality_reduction,
        "mds": _draw_dimensionality_reduction,
        "spectral_embedding": _draw_dimensionality_reduction,
        "covariance": _draw_dimensionality_reduction,

        "knn": _draw_knn,
        "knn_regressor": _draw_knn,
        "kneighbors_regressor": _draw_knn,
        "nearest_neighbors": _draw_knn,

        "naive_bayes": _draw_naive_bayes,
        "multinomial_naive_bayes": _draw_naive_bayes,
        "gaussian_nb": _draw_naive_bayes,
        "bernoulli_naive_bayes": _draw_naive_bayes,

        "regression": _draw_linear,
        "classifier": _draw_linear,
        "svm": _draw_linear,
        "svc": _draw_linear,
        "svr": _draw_linear,
        "perceptron": _draw_linear,
        "ridge": _draw_linear,
        "lasso": _draw_linear,
        "elastic_net": _draw_linear,
        "sgd_classifier": _draw_linear,
        "sgd_regressor": _draw_linear,
        "logistic_regression": _draw_linear,
        "poisson_regressor": _draw_linear,
        "lda": _draw_linear,
        "qda": _draw_linear,
        "theil_sen": _draw_linear,
        "online_learning": _draw_linear,
        "mlp_regressor": _draw_neural_network,

        "standard_scaler": _draw_preprocessor,
        "minmax_scaler": _draw_preprocessor,
        "robust_scaler": _draw_preprocessor,
        "normalizer": _draw_preprocessor,
        "maxabs_scaler": _draw_preprocessor,
        "binarizer": _draw_preprocessor,
        "polynomial_features": _draw_preprocessor,
        "kbins_discretizer": _draw_preprocessor,
        "quantile_transformer": _draw_preprocessor,
        "power_transformer": _draw_preprocessor,
        "simple_imputer": _draw_preprocessor,
        "knn_imputer": _draw_preprocessor,
        "label_encoder": _draw_preprocessor,
        "one_hot_encoder": _draw_preprocessor,
        "ordinal_encoder": _draw_preprocessor,
        "target_encoder": _draw_preprocessor,
        "tfidf_vectorizer": _draw_preprocessor,
        "count_vectorizer": _draw_preprocessor,
        "hashing_vectorizer": _draw_preprocessor,

        "isolation_forest": _draw_anomaly,
        "one_class_svm": _draw_anomaly,
        "local_outlier_factor": _draw_anomaly,
        "kde": _draw_anomaly,
        "kernel_density": _draw_anomaly,

        "a2c": _draw_rl,
        "sac": _draw_rl,
        "ddpg": _draw_rl,
        "td3": _draw_rl,
        "sarsa": _draw_rl,
        "tabular_policy": _draw_rl,
        "dqn": _draw_rl,
        "ppo": _draw_rl,

        "world_model": _draw_model_based_rl,
        "dreamer": _draw_model_based_rl,
        "muzero": _draw_model_based_rl,

        "mcts": _draw_planning,
        "policy_iteration": _draw_planning,
        "value_iteration": _draw_planning,

        "resnet": _draw_vision,
        "unet": _draw_vision,
        "vit": _draw_vision,
        "deit": _draw_vision,
        "densenet": _draw_vision,
        "mobilenet": _draw_vision,
        "efficientnet": _draw_vision,
        "capsule": _draw_vision,
        "nerf": _draw_vision,
        "inr": _draw_vision,

        "wav2vec2": _draw_speech,
        "hubert": _draw_speech,
        "whisper": _draw_speech,

        "clip": _draw_multimodal,

        "llama_decoder": _draw_modern_block,
        "t5": _draw_modern_block,
        "rwkv": _draw_modern_block,
        "s4": _draw_modern_block,
        "mamba2": _draw_modern_block,
        "deepseek_moe": _draw_modern_block,
        "jamba": _draw_modern_block,
        "deepfm": _draw_recommender,
        "wide_deep": _draw_recommender,
        "matrix_factorization": _draw_recommender,
        "collab_filtering": _draw_recommender,
        "item2vec": _draw_recommender,

        "hmm": _draw_probabilistic,
        "bayesian_nn": _draw_probabilistic,
        "mc_dropout": _draw_probabilistic,
        "mixture_density": _draw_probabilistic,
        "topic_model": _draw_probabilistic,

        "contrastive_learning": _draw_self_supervised,
        "self_supervised": _draw_self_supervised,

        "pointer_net": _draw_sequence,
        "ctc": _draw_sequence,
        "ttt": _draw_sequence,

        "set_transformer": _draw_set,

        "dnc": _draw_memory_augmented,
        "ntm": _draw_memory_augmented,

        "maml": _draw_meta_learning,

        "prototypical": _draw_few_shot,
        "matching": _draw_few_shot,
        "relation": _draw_few_shot,

        "siamese": _draw_metric_learning,

        "kalman_filter": _draw_filtering,

        "spiking_nn": _draw_neuromorphic,

        "rbf_network": _draw_feedforward,
        "elm": _draw_feedforward,

        "echo_state": _draw_reservoir,

        "time_series_arima": _draw_time_series,

        "rfe": _draw_feature_selection,
        "select_k_best": _draw_feature_selection,
        "variance_threshold": _draw_feature_selection,

        "gaussian_process": _draw_supervised_other,
        "isotonic_regression": _draw_supervised_other,
        "dummy_regressor": _draw_supervised_other,
        "cca": _draw_supervised_other,
        "pls_canonical": _draw_supervised_other,
        "probability_calibration": _draw_supervised_other,
        "stacking_classifier": _draw_ensemble_stack,
        "voting_classifier": _draw_ensemble_vote,

        "energy_based": _draw_generative_other,
        "normalizing_flow": _draw_generative_other,
        "flow_matching": _draw_generative_other,
        "score_based": _draw_generative_other,
        "consistency_model": _draw_generative_other,
        "rbm": _draw_generative_other,
        "bernoulli_rbm": _draw_generative_other,
        "boltzmann": _draw_generative_other,
        "dbn": _draw_generative_other,

        "gcn": _draw_graph,
        "gat": _draw_graph,
        "gin": _draw_graph,
        "graphsage": _draw_graph,
        "mpnn": _draw_graph,

        "sparse_attention": _draw_attention,

        # historical / legacy / generic
        "perceptron_legacy": _draw_generic,
    }
    if name in name_map:
        return name_map[name]

    # Family-level fallbacks
    family_map = {
        "deep_learning": _draw_neural_network,
        "classical_ml": _draw_generic,
        "unsupervised": _draw_clustering,
        "preprocessing": _draw_preprocessor,
        "generative": _draw_generative_other,
        "graph": _draw_graph,
        "reinforcement": _draw_rl,
        "model_based_rl": _draw_model_based_rl,
        "planning": _draw_planning,
        "vision": _draw_vision,
        "speech": _draw_speech,
        "multimodal": _draw_multimodal,
        "modern_llm": _draw_modern_block,
        "modern_ssm": _draw_modern_block,
        "modern_transformer": _draw_modern_block,
        "moe": _draw_modern_block,
        "hybrid": _draw_modern_block,
        "sequence": _draw_sequence,
        "set": _draw_set,
        "memory_augmented": _draw_memory_augmented,
        "meta_learning": _draw_meta_learning,
        "few_shot": _draw_few_shot,
        "metric_learning": _draw_metric_learning,
        "self_supervised": _draw_self_supervised,
        "probabilistic": _draw_probabilistic,
        "recommender": _draw_recommender,
        "filtering": _draw_filtering,
        "manifold": _draw_dimensionality_reduction,
        "decomposition": _draw_dimensionality_reduction,
        "feature_selection": _draw_feature_selection,
        "feedforward": _draw_feedforward,
        "reservoir": _draw_reservoir,
        "online": _draw_linear,
        "time_series": _draw_time_series,
        "supervised": _draw_supervised_other,
        "historical": _draw_linear,
        "alignment": _draw_generic,
        "associative_memory": _draw_memory_augmented,
        "attention": _draw_attention,
        "neuromorphic": _draw_neuromorphic,
    }
    if family in family_map:
        return family_map[family]

    return _draw_generic


def _guess_family(name: str) -> str:
    """Guess the family of a technique by name."""
    dl = {"cnn", "transformer", "neural_network", "rnn", "lstm", "gru", "seq2seq",
          "gan", "vae", "diffusion", "moe", "gnn", "mamba", "autoencoder",
          "dqn", "ppo"}
    tree = {"random_forest", "decision_tree", "extra_trees",
            "gradient_boosting", "adaboost", "xgboost", "lightgbm", "catboost",
            "hist_gradient_boosting"}
    cluster = {"kmeans", "gmm", "dbscan", "hierarchical", "birch", "mean_shift",
               "spectral_clustering", "affinity_propagation", "optics",
               "bisecting_kmeans", "mini_batch_kmeans", "art", "neural_gas",
               "som", "agglomerative", "agglomerative_clustering", "minibatch_kmeans"}
    dimred = {"pca", "tsne", "truncated_svd", "fast_ica", "isomap", "lle",
              "sparse_pca", "kernel_pca", "incremental_pca",
              "gaussian_random_projection", "factor_analysis", "nmf",
              "mds", "spectral_embedding", "covariance"}
    linear = {"regression", "classifier", "svm", "perceptron", "ridge", "lasso",
              "elastic_net", "sgd_classifier", "sgd_regressor",
              "logistic_regression", "poisson_regressor", "lda", "qda",
              "theil_sen", "online_learning"}
    rl = {"a2c", "sac", "ddpg", "td3", "sarsa", "tabular_policy", "dqn", "ppo"}
    if name in dl: return "deep_learning"
    if name in tree: return "classical_ml"
    if name in cluster: return "unsupervised"
    if name in dimred: return "unsupervised"
    if name in linear: return "classical_ml"
    if name in rl: return "reinforcement"
    return "unknown"


# =====================================================================
# Family-specific drawers
# =====================================================================
def _draw_cnn(model, ax, info):
    """Draw CNN architecture."""
    tech = info.get("tech_obj")
    conv_channels = getattr(tech, "conv_channels", [8, 16]) if tech else [8, 16]
    fc_hidden = getattr(tech, "fc_hidden", 64) if tech else 64
    n_classes = getattr(tech, "n_classes", 10) if tech else 10
    in_channels = getattr(tech, "in_channels", 1) if tech else 1
    se = getattr(tech, "se_blocks", False) if tech else False
    sd = getattr(tech, "stochastic_depth", 0.0) if tech else 0.0

    layers = [("Input", f"{in_channels}ch", COLORS["input"])]
    for i, ch in enumerate(conv_channels):
        sub = f"{ch}ch"
        if se: sub += "+SE"
        layers.append((f"Conv{i+1}", sub, COLORS["conv"]))
        layers.append((f"Pool{i+1}", "", COLORS["pool"]))
    layers.append(("AdaptivePool", "", COLORS["pool"]))
    layers.append(("FC", f"{fc_hidden}", COLORS["fc"]))
    if sd > 0:
        layers.append(("StochDepth", f"p={sd}", COLORS["norm"]))
    layers.append(("Output", f"{n_classes}", COLORS["output"]))
    _draw_layers(ax, layers, title="CNN")


def _draw_transformer(model, ax, info):
    """Draw Transformer architecture."""
    tech = info.get("tech_obj")
    n_layers = getattr(tech, "n_layers", 2) if tech else 2
    d_model = getattr(tech, "d_model", 64) if tech else 64
    n_heads = getattr(tech, "n_heads", 4) if tech else 4
    use_rope = getattr(tech, "use_rope", True) if tech else True
    vocab = getattr(tech, "vocab_size", 256) if tech else 256

    layers = [("Input", f"vocab={vocab}", COLORS["input"]),
              ("Embed", f"d={d_model}", COLORS["embedding"])]
    if use_rope: layers.append(("RoPE", "", COLORS["norm"]))
    for i in range(n_layers):
        layers.append((f"Attn{i+1}", f"heads={n_heads}", COLORS["attention"]))
        layers.append((f"FFN{i+1}", f"d={d_model*4}", COLORS["fc"]))
        layers.append((f"Norm{i+1}", "", COLORS["norm"]))
    layers.append(("LM Head", f"d={d_model}", COLORS["fc"]))
    layers.append(("Output", f"vocab={vocab}", COLORS["output"]))
    _draw_layers(ax, layers, title="Transformer")


def _draw_neural_network(model, ax, info):
    """Draw generic feed-forward NN."""
    tech = info.get("tech_obj")
    hidden = getattr(tech, "hidden_dim", 16) if tech else 16
    modern = getattr(tech, "modern", False) if tech else False
    n_layers = getattr(tech, "n_layers", 2) if tech else 2

    layers = [("Input", "", COLORS["input"])]
    for i in range(n_layers if modern else 1):
        sub = f"{hidden}"
        if modern: sub += "+BN+GELU"
        layers.append((f"FC{i+1}", sub, COLORS["fc"]))
    layers.append(("Output", "", COLORS["output"]))
    _draw_layers(ax, layers, title=info["name"].upper())


def _draw_rnn(model, ax, info):
    _draw_recurrent(model, ax, info, "RNN")


def _draw_lstm(model, ax, info):
    _draw_recurrent(model, ax, info, "LSTM", gates=["forget", "input", "cell", "output"])


def _draw_gru(model, ax, info):
    _draw_recurrent(model, ax, info, "GRU", gates=["reset", "update"])


def _draw_recurrent(model, ax, info, kind, gates=None):
    """Draw an RNN/LSTM/GRU unrolled over time."""
    tech = info.get("tech_obj")
    hidden = getattr(tech, "hidden_dim", 32) if tech else 32
    n_steps = 4

    # Title
    ax.text(5, 9.4, f"{kind} (hidden={hidden})", ha="center",
             fontsize=11, fontweight="bold")

    # Unrolled cells
    for t in range(n_steps):
        x = 1.5 + t * 2.2
        # Input
        ax.text(x, 1.5, f"x_{t+1}", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
        # Cell
        cell_color = COLORS["fc"] if kind == "RNN" else COLORS["attention"]
        cell = FancyBboxPatch((x - 0.6, 4), 1.2, 1.5,
                               boxstyle="round,pad=0.1", facecolor=cell_color,
                               alpha=0.3, edgecolor=cell_color, linewidth=2)
        ax.add_patch(cell)
        ax.text(x, 4.75, f"{kind}\n$h_{t+1}$", ha="center", va="center",
                 fontsize=8, fontweight="bold")
        # Output
        ax.text(x, 8, f"y_{t+1}", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
        # Arrows
        _draw_arrow(ax, x, 1.8, x, 4)
        _draw_arrow(ax, x, 5.5, x, 7.7)
        # Horizontal connection (h_{t} -> h_{t+1})
        if t > 0:
            _draw_arrow(ax, x - 2.2 + 0.6, 4.75, x - 0.6, 4.75,
                         color=COLORS["attention"])

    # Gate legend for LSTM/GRU
    if gates:
        ax.text(5, 0.5, "gates: " + ", ".join(gates), ha="center",
                 fontsize=8, color="#6b7494")


def _draw_seq2seq(model, ax, info):
    """Draw encoder-decoder seq2seq."""
    # Encoder (left)
    ax.text(2.5, 9, "Encoder", ha="center", fontsize=11, fontweight="bold")
    for t in range(3):
        y = 7.5 - t * 1.5
        ax.text(1.5, y, f"enc_{t+1}", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
        if t > 0:
            _draw_arrow(ax, 1.5, y + 1.2, 1.5, y + 0.3, color=COLORS["attention"])
    # Context
    ax.text(5, 5, "context", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.4))
    _draw_arrow(ax, 2, 5, 4.4, 5)
    # Decoder (right)
    ax.text(7.5, 9, "Decoder", ha="center", fontsize=11, fontweight="bold")
    for t in range(3):
        y = 7.5 - t * 1.5
        ax.text(8.5, y, f"dec_{t+1}", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
        if t > 0:
            _draw_arrow(ax, 8.5, y + 0.3, 8.5, y + 1.2, color=COLORS["attention"])
        _draw_arrow(ax, 5.6, 5, 7.9, y, color="#a0aac4")


def _draw_autoencoder(model, ax, info):
    """Draw autoencoder: input → encoder → latent → decoder → output."""
    tech = info.get("tech_obj")
    latent = getattr(tech, "latent_dim", 8) if tech else 8
    hidden = getattr(tech, "hidden", 32) if tech else 32
    data_dim = getattr(tech, "data_dim", 16) if tech else 16

    layers = [
        ("Input", f"d={data_dim}", COLORS["input"]),
        ("Enc1", f"{hidden}", COLORS["fc"]),
        ("Enc2", f"{hidden//2}", COLORS["fc"]),
        ("Latent", f"{latent}", COLORS["attention"]),
        ("Dec1", f"{hidden//2}", COLORS["fc"]),
        ("Dec2", f"{hidden}", COLORS["fc"]),
        ("Output", f"d={data_dim}", COLORS["output"]),
    ]
    _draw_layers(ax, layers, title="Autoencoder")


def _draw_vae(model, ax, info):
    """Draw VAE: encoder → (μ, σ) → z → decoder."""
    tech = info.get("tech_obj")
    latent = getattr(tech, "latent_dim", 8) if tech else 8
    hidden = getattr(tech, "hidden", 32) if tech else 32
    beta = getattr(tech, "beta", 1.0) if tech else 1.0

    layers = [
        ("Input", "", COLORS["input"]),
        ("Enc", f"{hidden}", COLORS["fc"]),
        ("μ", f"d={latent}", COLORS["linear"]),
        ("σ", f"d={latent}", COLORS["linear"]),
        ("z = μ+σε", "reparam", COLORS["attention"]),
        ("Dec", f"{hidden}", COLORS["fc"]),
        ("Output", "", COLORS["output"]),
    ]
    _draw_layers(ax, layers, title=f"VAE (β={beta})")


def _draw_gan(model, ax, info):
    """Draw GAN: Generator vs Discriminator."""
    tech = info.get("tech_obj")
    data_dim = getattr(tech, "data_dim", 16) if tech else 16

    # Generator
    ax.text(2.5, 9, "Generator G", ha="center", fontsize=11, fontweight="bold")
    ax.text(2.5, 7.5, "z ~ N(0,I)", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    ax.text(2.5, 5.5, "G(z)", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["generative"], alpha=0.4))
    ax.text(2.5, 3.5, "fake x", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
    _draw_arrow(ax, 2.5, 7.2, 2.5, 5.9)
    _draw_arrow(ax, 2.5, 5.1, 2.5, 3.8)

    # Discriminator
    ax.text(7.5, 9, "Discriminator D", ha="center", fontsize=11, fontweight="bold")
    ax.text(7.5, 7.5, "real x", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    ax.text(7.5, 5.5, "D(x)", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.4))
    ax.text(7.5, 3.5, "real/fake", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["linear"], alpha=0.3))
    _draw_arrow(ax, 7.5, 7.2, 7.5, 5.9)
    _draw_arrow(ax, 7.5, 5.1, 7.5, 3.8)

    # Fake -> D
    _draw_arrow(ax, 3.2, 5.5, 6.8, 5.5, color=COLORS["generative"])


def _draw_diffusion(model, ax, info):
    """Draw diffusion: forward noise + reverse denoise."""
    tech = info.get("tech_obj")
    n_steps = 5
    # Forward (top)
    ax.text(5, 9.4, "Forward (q): add noise", ha="center",
             fontsize=10, fontweight="bold", color=COLORS["output"])
    for t in range(n_steps):
        x = 1 + t * 2
        ax.text(x, 8, f"x_{t}", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
        if t > 0:
            _draw_arrow(ax, x - 1.6, 8, x - 0.4, 8, color=COLORS["output"])
    ax.text(9.5, 8, "ε", ha="center", fontsize=10, fontweight="bold")
    _draw_arrow(ax, 8.6, 8, 9.2, 8, color=COLORS["output"])

    # Reverse (bottom)
    ax.text(5, 6, "Reverse (p): denoise", ha="center",
             fontsize=10, fontweight="bold", color=COLORS["generative"])
    for t in range(n_steps):
        x = 1 + t * 2
        ax.text(x, 4.5, f"x_{t}", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor=COLORS["generative"], alpha=0.3))
        if t < n_steps - 1:
            _draw_arrow(ax, x + 0.4, 4.5, x + 1.6, 4.5,
                         color=COLORS["generative"])

    # Net
    ax.text(5, 2, "ε-Net (U-Net)", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.4))


def _draw_moe(model, ax, info):
    """Draw Mixture of Experts: router + experts."""
    n_experts = 4
    ax.text(5, 9, "Input", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    ax.text(5, 7.5, "Router", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.4))
    _draw_arrow(ax, 5, 8.7, 5, 7.9)

    # Experts
    for i in range(n_experts):
        x = 1.5 + i * 2.2
        ax.text(x, 5, f"Expert {i+1}", ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
        _draw_arrow(ax, 5, 7.1, x, 5.4, color=COLORS["attention"])

    ax.text(5, 3, "Weighted Sum", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["ensemble"], alpha=0.4))
    for i in range(n_experts):
        x = 1.5 + i * 2.2
        _draw_arrow(ax, x, 4.6, 5, 3.4)
    ax.text(5, 1.5, "Output", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
    _draw_arrow(ax, 5, 2.6, 5, 1.8)


def _draw_mamba(model, ax, info):
    """Draw Mamba SSM block."""
    tech = info.get("tech_obj")
    d_model = getattr(tech, "d_model", 64) if tech else 64
    d_ff = getattr(tech, "d_ff", 256) if tech else 256

    layers = [
        ("Input", f"d={d_model}", COLORS["input"]),
        ("Linear", f"d={d_ff}", COLORS["fc"]),
        ("Conv1D", "", COLORS["conv"]),
        ("SiLU", "", COLORS["norm"]),
        ("SSM", "selective", COLORS["attention"]),
        ("Gate", "×SiLU", COLORS["norm"]),
        ("Linear", f"d={d_model}", COLORS["fc"]),
        ("Output", f"d={d_model}", COLORS["output"]),
    ]
    _draw_layers(ax, layers, title="Mamba block")


def _draw_gnn(model, ax, info):
    """Draw a graph neural network with message passing."""
    # Draw nodes
    nodes = [(2, 7), (4, 8), (6, 7), (3, 5.5), (5, 5.5), (4, 4)]
    edges = [(0,1), (0,3), (1,2), (1,3), (2,4), (3,4), (3,5), (4,5), (0,4)]
    for i, (x, y) in enumerate(nodes):
        ax.scatter([x], [y], s=300, color=COLORS["graph"], alpha=0.5, zorder=3)
        ax.text(x, y, f"v{i+1}", ha="center", va="center", fontsize=8, zorder=4)
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
                 color="#a0aac4", alpha=0.5, zorder=1)

    # Message-passing arrow
    ax.text(8, 6, "Aggregate\n+ Update", ha="center", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.3))
    ax.text(8, 3, "Readout", ha="center", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
    _draw_arrow(ax, 6.5, 5.5, 7.4, 6)
    _draw_arrow(ax, 8, 5.3, 8, 3.5)

    ax.text(5, 1.5, f"GNN: {len(nodes)} nodes, {len(edges)} edges",
             ha="center", fontsize=10, fontweight="bold")


def _draw_graph(model, ax, info):
    """Alias for GNN family drawer."""
    _draw_gnn(model, ax, info)


# ----------------------- Trees / ensembles -----------------------
def _draw_tree_ensemble(model, ax, info):
    """Draw a tree ensemble: input → [trees] → vote → output."""
    tech = info.get("tech_obj")
    n_trees = 0
    if tech is not None and hasattr(tech, "model") and hasattr(tech.model, "estimators_"):
        n_trees = len(tech.model.estimators_)
    elif tech is not None and hasattr(tech, "n_estimators"):
        n_trees = tech.n_estimators
    if n_trees == 0:
        n_trees = 100

    ax.text(1, 8, "Input", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    n_show = min(n_trees, 5)
    for i in range(n_show):
        y = 7 - i * 1.2
        label = f"Tree {i+1}" if i < n_show - 1 else f"... ({n_trees} trees)"
        ax.text(5, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor=COLORS["tree"], alpha=0.3))
        _draw_arrow(ax, 1.5, 8, 4, y)
        _draw_arrow(ax, 6, y, 7.5, 5)
    ax.text(8.5, 5, "Vote", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["ensemble"], alpha=0.3))
    _draw_arrow(ax, 9, 5, 9, 3)
    ax.text(9, 2.5, "Output", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))


def _draw_single_tree(model, ax, info):
    """Draw a single decision tree."""
    ax.text(5, 9, "Root", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["tree"], alpha=0.3))
    ax.text(2, 6, "Left", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor=COLORS["tree"], alpha=0.2))
    ax.text(8, 6, "Right", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor=COLORS["tree"], alpha=0.2))
    _draw_arrow(ax, 4.5, 8.7, 2.5, 6.3)
    _draw_arrow(ax, 5.5, 8.7, 7.5, 6.3)
    for x, label in [(1, "L-L"), (3, "L-R"), (7, "R-L"), (9, "R-R")]:
        ax.text(x, 3, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.2))
    _draw_arrow(ax, 1.5, 5.7, 1, 3.3)
    _draw_arrow(ax, 2.5, 5.7, 3, 3.3)
    _draw_arrow(ax, 7.5, 5.7, 7, 3.3)
    _draw_arrow(ax, 8.5, 5.7, 9, 3.3)


def _draw_boosting(model, ax, info):
    """Draw boosting: sequential weak learners."""
    tech = info.get("tech_obj")
    n_est = getattr(tech, "n_estimators", 100) if tech else 100
    ax.text(1, 5, "Input", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    n_show = min(n_est, 5)
    for i in range(n_show):
        x = 2.5 + i * 1.3
        label = f"Weak{i+1}" if i < n_show - 1 else f"... ({n_est})"
        ax.text(x, 5, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", facecolor=COLORS["tree"], alpha=0.3))
        if i == 0:
            _draw_arrow(ax, 1.5, 5, 2, 5)
        else:
            _draw_arrow(ax, x - 1.0, 5, x - 0.5, 5, color=COLORS["output"])
    _draw_arrow(ax, 2.5 + n_show * 1.3 - 0.5, 5, 8.5, 5)
    ax.text(9, 5, "Output", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))


def _draw_ensemble_stack(model, ax, info):
    """Draw stacking ensemble: base learners → meta-learner."""
    ax.text(1, 5, "Input", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    for i, name in enumerate(["Base1", "Base2", "Base3"]):
        y = 7 - i * 1.5
        ax.text(4, y, name, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor=COLORS["tree"], alpha=0.3))
        _draw_arrow(ax, 1.5, 5, 3.5, y)
    ax.text(7, 5, "Meta", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["ensemble"], alpha=0.4))
    for i in range(3):
        y = 7 - i * 1.5
        _draw_arrow(ax, 4.5, y, 6.5, 5)
    ax.text(9, 5, "Output", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
    _draw_arrow(ax, 7.5, 5, 8.5, 5)


def _draw_ensemble_vote(model, ax, info):
    """Draw voting ensemble."""
    ax.text(1, 5, "Input", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    for i, name in enumerate(["Model1", "Model2", "Model3"]):
        y = 7 - i * 1.5
        ax.text(4, y, name, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor=COLORS["tree"], alpha=0.3))
        _draw_arrow(ax, 1.5, 5, 3.5, y)
    ax.text(7, 5, "Vote", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["ensemble"], alpha=0.4))
    for i in range(3):
        y = 7 - i * 1.5
        _draw_arrow(ax, 4.5, y, 6.5, 5)
    ax.text(9, 5, "Output", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
    _draw_arrow(ax, 7.5, 5, 8.5, 5)


# ----------------------- Clustering / dimred -----------------------
def _draw_clustering(model, ax, info):
    """Draw cluster scatter."""
    tech = info.get("tech_obj")
    n_clusters = 3
    if tech is not None:
        nc = getattr(tech, "n_clusters", getattr(tech, "n_components", 3))
        if isinstance(nc, int):
            n_clusters = nc
        elif callable(nc):
            try:
                n_clusters = int(nc())
            except Exception:
                n_clusters = 3
    rng = np.random.default_rng(42)
    for i in range(n_clusters):
        cx, cy = 2 + i * 2.5, 5 + rng.uniform(-1, 1)
        pts = rng.normal(0, 0.5, (20, 2))
        ax.scatter(pts[:, 0] + cx, pts[:, 1] + cy, s=20, alpha=0.5,
                   color=plt.cm.Set1(i / max(n_clusters, 1)))
        ax.scatter([cx], [cy], marker="x", s=100, color="red", linewidths=2)
    ax.text(5, 1.5, f"{info['name']} ({n_clusters} clusters)",
             ha="center", fontsize=10, fontweight="bold")


def _draw_dimensionality_reduction(model, ax, info):
    """Draw dim-red: High-D → Reduce → Low-D + projection."""
    tech = info.get("tech_obj")
    n_comp = 2
    if tech is not None:
        n_comp = getattr(tech, "n_components", 2) if tech else 2
    ax.text(2, 7, "High-D\nInput", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    ax.text(5, 7, info["name"], ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["pca"], alpha=0.3))
    ax.text(8, 7, f"Low-D\n({n_comp} comp)", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
    _draw_arrow(ax, 3, 7, 4, 7)
    _draw_arrow(ax, 6, 7, 7, 7)
    rng = np.random.default_rng(42)
    pts = rng.normal(0, 0.5, (30, 2)) * 1.5
    ax.scatter(pts[:, 0] + 5, pts[:, 1] + 3, s=15, alpha=0.5, color=COLORS["pca"])
    ax.text(5, 1, "Projected data", ha="center", fontsize=8, color="#6b7494")


# ----------------------- Linear / KNN / NB -----------------------
def _draw_linear(model, ax, info):
    """Draw a feature-weight bar chart."""
    tech = info.get("tech_obj")
    weights = None
    if tech is not None:
        if hasattr(tech, "w") and getattr(tech, "w", None) is not None:
            arr = getattr(tech, "w")
            try:
                weights = np.asarray(arr, dtype=float).flatten()
            except Exception:
                weights = None
        if weights is None and hasattr(tech, "model"):
            m = tech.model
            if hasattr(m, "coef_") and m.coef_ is not None:
                try:
                    weights = np.asarray(m.coef_, dtype=float).flatten()
                except Exception:
                    weights = None
    if weights is None or len(weights) == 0:
        # Generic placeholder weights
        weights = np.array([0.5, -0.3, 0.8, -0.2, 0.1, -0.5, 0.3, -0.1])

    n = len(weights)
    colors_bar = [COLORS["linear"] if (w is not None and w >= 0) else COLORS["output"]
                   for w in weights]
    ax.barh(range(n), weights, color=colors_bar, alpha=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"x{i}" for i in range(n)], fontsize=8)
    ax.set_xlabel("Weight", fontsize=9)
    ax.set_title(f"{info['name']} — Feature Weights",
                  fontsize=12, fontweight="bold")
    ax.axvline(0, color="#6b7494", linewidth=0.5)


def _draw_knn(model, ax, info):
    """Draw KNN: query point + k nearest neighbors."""
    tech = info.get("tech_obj")
    k = getattr(tech, "k", getattr(tech, "n_neighbors", 5)) if tech else 5
    rng = np.random.default_rng(42)
    pts = rng.normal(0, 1, (30, 2)) * 1.5 + 5
    ax.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.3, color="#6b7494")
    qx, qy = 5, 5
    ax.scatter([qx], [qy], marker="*", s=200, color="red", zorder=5)
    dists = np.sqrt(np.sum((pts - [qx, qy])**2, axis=1))
    nn_idx = np.argsort(dists)[:k]
    ax.scatter(pts[nn_idx, 0], pts[nn_idx, 1], s=50, color=COLORS["linear"],
                zorder=4, edgecolors="red", linewidths=1)
    for idx in nn_idx:
        ax.plot([qx, pts[idx, 0]], [qy, pts[idx, 1]], "--",
                 color="red", alpha=0.3, linewidth=0.5)
    ax.set_title(f"KNN (k={k}) — query + neighbors",
                  fontsize=12, fontweight="bold")


def _draw_naive_bayes(model, ax, info):
    """Draw Naive-Bayes factor graph."""
    ax.text(5, 8, "Input Features", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    features = ["x1", "x2", "x3", "x4"]
    for i, f in enumerate(features):
        y = 6.5 - i * 1.2
        ax.text(3, y, f, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor=COLORS["linear"], alpha=0.2))
        _draw_arrow(ax, 4.5, 8, 3.5, y)
        ax.text(7, y, f"P(class|{f})", ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", facecolor=COLORS["ensemble"], alpha=0.2))
        _draw_arrow(ax, 3.5, y, 6, y)
    ax.text(7, 1.5, "Bayes: argmax P(class)∏P(xi|class)",
             ha="center", fontsize=8, color="#6b7494")


# ----------------------- Preprocessor -----------------------
def _draw_preprocessor(model, ax, info):
    """Draw preprocessor: input → transform → output."""
    name = info["name"]
    ax.text(5, 7, "Input", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    _draw_arrow(ax, 5, 6.7, 5, 5.3)
    ax.text(5, 5, name, ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["scaler"], alpha=0.3))
    _draw_arrow(ax, 5, 4.7, 5, 3.3)
    ax.text(5, 3, "Transformed", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))


# ----------------------- Anomaly detection -----------------------
def _draw_anomaly(model, ax, info):
    """Draw anomaly detection: dense cluster + outliers flagged."""
    rng = np.random.default_rng(42)
    pts = rng.normal(5, 1, (50, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.5, color=COLORS["linear"])
    # Outliers
    outliers = np.array([[1.5, 8.5], [8.5, 1.5], [9, 9]])
    ax.scatter(outliers[:, 0], outliers[:, 1], s=80, marker="X",
                color=COLORS["output"], zorder=5)
    ax.set_title(f"{info['name']} — anomaly detection",
                  fontsize=12, fontweight="bold")
    ax.text(5, 0.5, "X = outliers", ha="center", fontsize=8, color="#6b7494")


# ----------------------- Reinforcement learning -----------------------
def _draw_rl(model, ax, info):
    """Draw agent ↔ environment loop + policy/value head."""
    ax.text(2.5, 7, "Agent", ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["rl"], alpha=0.4))
    ax.text(7.5, 7, "Environment", ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.4))
    # Action
    _draw_arrow(ax, 3.5, 7.2, 6.5, 7.2, color=COLORS["generative"])
    ax.text(5, 7.6, "action", ha="center", fontsize=8, color=COLORS["generative"])
    # Reward + state
    _draw_arrow(ax, 6.5, 6.8, 3.5, 6.8, color=COLORS["output"])
    ax.text(5, 6.4, "state, reward", ha="center", fontsize=8, color=COLORS["output"])

    # Inside the agent: policy + value
    ax.text(2.5, 4.5, "Policy π(a|s)", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.3))
    ax.text(2.5, 3, "Value V(s)", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
    _draw_arrow(ax, 2.5, 6.5, 2.5, 5)
    _draw_arrow(ax, 2.5, 4, 2.5, 3.5)

    # Episode loop
    _draw_loop(ax, 5, 5, 4, 3.5, "episode loop", color=COLORS["rl"])
    ax.text(5, 1, f"{info['name']} ({info['family']})",
             ha="center", fontsize=10, fontweight="bold")


def _draw_model_based_rl(model, ax, info):
    """Draw world-model predictive loop."""
    # Agent
    ax.text(2, 7, "Agent", ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["rl"], alpha=0.4))
    # World model
    ax.text(5, 7, "World Model", ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["generative"], alpha=0.4))
    # Real env
    ax.text(8, 7, "Real Env", ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.4))

    # Predicted trajectory
    for i in range(4):
        x = 3.5 + i * 1.0
        ax.text(x, 4, f"ŝ_{i+1}", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor=COLORS["generative"], alpha=0.2))
        if i > 0:
            _draw_arrow(ax, x - 0.7, 4, x - 0.3, 4, color=COLORS["generative"])
    _draw_arrow(ax, 2.5, 6.5, 4, 4.5, color=COLORS["generative"])

    # Real trajectory (occasional)
    _draw_arrow(ax, 8, 6.5, 8, 4.5, color=COLORS["input"])
    ax.text(8, 4, "real s", ha="center", fontsize=8,
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.2))
    _draw_arrow(ax, 8, 3.5, 2.5, 6.5, color=COLORS["input"])
    ax.text(5, 1.5, "Plan in imagination; act in reality",
             ha="center", fontsize=9, color="#6b7494")


def _draw_planning(model, ax, info):
    """Draw planning: MCTS-style search tree."""
    # Root
    ax.scatter([5], [9], s=300, color=COLORS["attention"], alpha=0.5, zorder=3)
    ax.text(5, 9, "root", ha="center", va="center", fontsize=8, zorder=4)
    # Children
    children_x = [2, 4, 6, 8]
    for x in children_x:
        ax.scatter([x], [7], s=200, color=COLORS["fc"], alpha=0.5, zorder=3)
        ax.text(x, 7, "n", ha="center", va="center", fontsize=7, zorder=4)
        ax.plot([5, x], [9, 7], color="#a0aac4", alpha=0.5, zorder=1)
    # Grandchildren
    for cx in children_x:
        for dx in [-0.6, 0.6]:
            ax.scatter([cx + dx], [5], s=80, color=COLORS["linear"], alpha=0.5, zorder=3)
            ax.plot([cx, cx + dx], [7, 5], color="#a0aac4", alpha=0.3, zorder=1)
    ax.text(5, 3, "MCTS / Value Iteration",
             ha="center", fontsize=10, fontweight="bold")
    ax.text(5, 2, "expand → simulate → backprop",
             ha="center", fontsize=8, color="#6b7494")


# ----------------------- Vision -----------------------
def _draw_vision(model, ax, info):
    """Draw vision backbone: image → conv stages → head."""
    tech = info.get("tech_obj")
    n_classes = getattr(tech, "n_classes", 1000) if tech else 1000
    img_size = getattr(tech, "img_size", 224) if tech else 224

    layers = [
        ("Image", f"{img_size}×{img_size}", COLORS["input"]),
        ("Stem", "conv", COLORS["conv"]),
        ("Stage1", "+pool", COLORS["conv"]),
        ("Stage2", "down", COLORS["conv"]),
        ("Stage3", "down", COLORS["conv"]),
        ("Stage4", "down", COLORS["conv"]),
        ("GlobalAvgPool", "", COLORS["pool"]),
        ("Head", f"{n_classes}", COLORS["output"]),
    ]
    _draw_layers(ax, layers, title=info["name"].upper())


# ----------------------- Speech -----------------------
def _draw_speech(model, ax, info):
    """Draw speech model: waveform → conv-feature → encoder → context."""
    layers = [
        ("Waveform", "16kHz", COLORS["input"]),
        ("Conv1d", "feature", COLORS["conv"]),
        ("PosEnc", "", COLORS["norm"]),
        ("Encoder", "Transformer", COLORS["attention"]),
        ("Context", "hidden", COLORS["fc"]),
        ("Head", "CTC/CLS", COLORS["output"]),
    ]
    _draw_layers(ax, layers, title=info["name"].upper())


# ----------------------- Multimodal -----------------------
def _draw_multimodal(model, ax, info):
    """Draw two-tower multimodal (e.g. CLIP)."""
    # Text tower
    ax.text(2.5, 9, "Text", ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    ax.text(2.5, 7, "Tokenizer", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["norm"], alpha=0.3))
    ax.text(2.5, 5, "Encoder", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.4))
    ax.text(2.5, 3, "Text emb", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
    _draw_arrow(ax, 2.5, 8.7, 2.5, 7.3)
    _draw_arrow(ax, 2.5, 6.7, 2.5, 5.3)
    _draw_arrow(ax, 2.5, 4.7, 2.5, 3.3)

    # Image tower
    ax.text(7.5, 9, "Image", ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    ax.text(7.5, 7, "Patchify", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["norm"], alpha=0.3))
    ax.text(7.5, 5, "Encoder", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.4))
    ax.text(7.5, 3, "Image emb", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
    _draw_arrow(ax, 7.5, 8.7, 7.5, 7.3)
    _draw_arrow(ax, 7.5, 6.7, 7.5, 5.3)
    _draw_arrow(ax, 7.5, 4.7, 7.5, 3.3)

    # Contrastive loss
    ax.text(5, 2, "Contrastive\nLoss", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.4))
    _draw_arrow(ax, 3, 3, 4.5, 2.3, color=COLORS["output"])
    _draw_arrow(ax, 7, 3, 5.5, 2.3, color=COLORS["output"])


# ----------------------- Modern LLM / SSM / Transformer / MoE / Hybrid -----------------------
def _draw_modern_block(model, ax, info):
    """Draw a modern block stack (LLaMA / T5 / RWKV / S4 / Mamba2 / DeepSeek MoE / Jamba)."""
    name = info["name"]
    # Pick the right block diagram
    if name in ("llama_decoder",):
        blocks = [
            ("Input", "tokens", COLORS["input"]),
            ("Embed", "", COLORS["embedding"]),
            ("RoPE", "", COLORS["norm"]),
            ("RMSNorm", "", COLORS["norm"]),
            ("Attn", "grouped-query", COLORS["attention"]),
            ("RMSNorm", "", COLORS["norm"]),
            ("FFN", "SwiGLU", COLORS["fc"]),
            ("Output", "vocab", COLORS["output"]),
        ]
    elif name == "t5":
        blocks = [
            ("Encoder", "input", COLORS["input"]),
            ("Embed", "shared", COLORS["embedding"]),
            ("RMSNorm", "", COLORS["norm"]),
            ("Attn", "self", COLORS["attention"]),
            ("FFN", "GeLU", COLORS["fc"]),
            ("Decoder", "output", COLORS["output"]),
            ("Cross-Attn", "enc-dec", COLORS["attention"]),
            ("FFN", "GeLU", COLORS["fc"]),
        ]
    elif name in ("rwkv", "s4", "mamba2"):
        blocks = [
            ("Input", "", COLORS["input"]),
            ("Embed", "", COLORS["embedding"]),
            ("Linear", "", COLORS["fc"]),
            ("Conv1d", "", COLORS["conv"]),
            ("SSM", "state-space", COLORS["attention"]),
            ("Gate", "×SiLU", COLORS["norm"]),
            ("Linear", "", COLORS["fc"]),
            ("Output", "", COLORS["output"]),
        ]
    elif name == "deepseek_moe":
        blocks = [
            ("Input", "", COLORS["input"]),
            ("Embed", "", COLORS["embedding"]),
            ("Attn", "", COLORS["attention"]),
            ("Router", "top-k", COLORS["norm"]),
            ("Experts", "shared+ranked", COLORS["fc"]),
            ("Combine", "", COLORS["ensemble"]),
            ("Output", "", COLORS["output"]),
        ]
    elif name == "jamba":
        blocks = [
            ("Input", "", COLORS["input"]),
            ("Embed", "", COLORS["embedding"]),
            ("Attn", "", COLORS["attention"]),
            ("Mamba", "SSM", COLORS["generative"]),
            ("MLP", "", COLORS["fc"]),
            ("Output", "", COLORS["output"]),
        ]
    else:
        blocks = [
            ("Input", "", COLORS["input"]),
            ("Embed", "", COLORS["embedding"]),
            ("Block", "", COLORS["attention"]),
            ("Block", "", COLORS["attention"]),
            ("Output", "", COLORS["output"]),
        ]
    _draw_layers(ax, blocks, title=name.upper())


# ----------------------- Generative (other) -----------------------
def _draw_generative_other(model, ax, info):
    """Draw energy-based / flow / score / RBM diagram."""
    name = info["name"]
    if "flow" in name or "consistency" in name:
        layers = [
            ("Input z", "noise", COLORS["input"]),
            ("Flow 1", "invertible", COLORS["generative"]),
            ("Flow 2", "invertible", COLORS["generative"]),
            ("Flow 3", "invertible", COLORS["generative"]),
            ("Output x", "data", COLORS["output"]),
        ]
    elif "score" in name:
        layers = [
            ("Input x", "", COLORS["input"]),
            ("Score net", "∇log p", COLORS["attention"]),
            ("Langevin", "step", COLORS["generative"]),
            ("Output x~p(x)", "", COLORS["output"]),
        ]
    elif "rbm" in name or "boltzmann" in name or "dbn" in name:
        # Visible vs hidden units
        ax.text(2.5, 8, "Visible v", ha="center", fontsize=10, fontweight="bold")
        ax.text(7.5, 8, "Hidden h", ha="center", fontsize=10, fontweight="bold")
        for i in range(4):
            ax.scatter([2], [7 - i * 1.2], s=300, color=COLORS["input"], alpha=0.5, zorder=3)
            ax.text(2, 7 - i * 1.2, f"v{i+1}", ha="center", va="center", fontsize=7, zorder=4)
        for i in range(3):
            ax.scatter([8], [7.5 - i * 1.5], s=300, color=COLORS["fc"], alpha=0.5, zorder=3)
            ax.text(8, 7.5 - i * 1.5, f"h{i+1}", ha="center", va="center", fontsize=7, zorder=4)
        # Fully connected
        for vy in [7, 5.8, 4.6, 3.4]:
            for hy in [7.5, 6, 4.5]:
                ax.plot([2.2, 7.8], [vy, hy], color="#a0aac4",
                         alpha=0.2, linewidth=0.5)
        ax.text(5, 1.5, f"{name}: bipartite energy model",
                 ha="center", fontsize=10, fontweight="bold")
        return
    elif "energy" in name:
        layers = [
            ("Input x", "", COLORS["input"]),
            ("Energy E(x)", "", COLORS["generative"]),
            ("Sample", "Langevin", COLORS["attention"]),
            ("Output", "", COLORS["output"]),
        ]
    else:
        layers = [
            ("Input", "", COLORS["input"]),
            (name, "", COLORS["generative"]),
            ("Output", "", COLORS["output"]),
        ]
    _draw_layers(ax, layers, title=name.upper())


# ----------------------- Probabilistic -----------------------
def _draw_probabilistic(model, ax, info):
    """Draw probabilistic graphical model (HMM, Bayesian NN, …)."""
    name = info["name"]
    if name == "hmm":
        # Hidden chain
        for t in range(5):
            x = 1 + t * 2
            ax.scatter([x], [7], s=400, color=COLORS["attention"], alpha=0.5, zorder=3)
            ax.text(x, 7, f"z_{t}", ha="center", va="center", fontsize=8, zorder=4)
            ax.scatter([x], [4.5], s=300, color=COLORS["input"], alpha=0.5, zorder=3)
            ax.text(x, 4.5, f"x_{t}", ha="center", va="center", fontsize=8, zorder=4)
            if t > 0:
                _draw_arrow(ax, x - 1.7, 7, x - 0.3, 7, color=COLORS["attention"])
            _draw_arrow(ax, x, 6.7, x, 4.9, color="#a0aac4")
        ax.text(5, 2, "HMM: hidden chain → emissions",
                 ha="center", fontsize=10, fontweight="bold")
    elif name == "bayesian_nn":
        layers = [
            ("Input", "", COLORS["input"]),
            ("FC + σ_w", "weight ~ N", COLORS["attention"]),
            ("FC + σ_w", "weight ~ N", COLORS["attention"]),
            ("Output", "posterior", COLORS["output"]),
        ]
        _draw_layers(ax, layers, title="Bayesian NN")
    else:
        layers = [
            ("Input", "", COLORS["input"]),
            (name, "probabilistic", COLORS["attention"]),
            ("Output", "distribution", COLORS["output"]),
        ]
        _draw_layers(ax, layers, title=name.upper())


# ----------------------- Recommender -----------------------
def _draw_recommender(model, ax, info):
    """Draw recommender: user/item embeddings + scorer."""
    name = info["name"]
    # User tower
    ax.text(2, 8, "User ID", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    ax.text(2, 6, "User emb", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
    _draw_arrow(ax, 2, 7.7, 2, 6.3)
    # Item tower
    ax.text(8, 8, "Item ID", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    ax.text(8, 6, "Item emb", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
    _draw_arrow(ax, 8, 7.7, 8, 6.3)
    # Scorer
    ax.text(5, 4, "Scorer", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["ensemble"], alpha=0.4))
    _draw_arrow(ax, 2.5, 5.7, 4.5, 4.3)
    _draw_arrow(ax, 7.5, 5.7, 5.5, 4.3)
    ax.text(5, 2, "Rating / Top-K", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
    _draw_arrow(ax, 5, 3.6, 5, 2.4)
    ax.text(5, 1, name, ha="center", fontsize=10, fontweight="bold")


# ----------------------- Self-supervised -----------------------
def _draw_self_supervised(model, ax, info):
    """Draw contrastive self-supervised: anchor + positives + negatives."""
    # Anchor
    ax.text(2, 8, "Anchor", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    # Positives
    ax.text(5, 8, "Positive", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["generative"], alpha=0.3))
    # Negatives
    ax.text(8, 8, "Negatives", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))

    # Encoder
    ax.text(5, 5, "Encoder f(·)", ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.4))
    _draw_arrow(ax, 2, 7.7, 4.5, 5.3)
    _draw_arrow(ax, 5, 7.7, 5, 5.7)
    _draw_arrow(ax, 8, 7.7, 5.5, 5.3)

    # Embeddings
    ax.text(2, 3, "z_a", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
    ax.text(5, 3, "z_p", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
    ax.text(8, 3, "z_n", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
    _draw_arrow(ax, 4.5, 4.7, 2.5, 3.3)
    _draw_arrow(ax, 5, 4.7, 5, 3.3)
    _draw_arrow(ax, 5.5, 4.7, 7.5, 3.3)

    ax.text(5, 1.5, "Contrastive: pull z_a↔z_p, push z_a↔z_n",
             ha="center", fontsize=9, color="#6b7494")


# ----------------------- Sequence -----------------------
def _draw_sequence(model, ax, info):
    """Draw sequence model: encoder + state sequence + output."""
    name = info["name"]
    # Input tokens
    for t in range(4):
        x = 1.5 + t * 2
        ax.text(x, 8, f"x_{t}", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    # State sequence
    for t in range(4):
        x = 1.5 + t * 2
        ax.text(x, 5, f"h_{t}", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.3))
        _draw_arrow(ax, x, 7.7, x, 5.3)
        if t > 0:
            _draw_arrow(ax, x - 1.7, 5, x - 0.3, 5, color=COLORS["attention"])
    # Output
    for t in range(4):
        x = 1.5 + t * 2
        ax.text(x, 2, f"y_{t}", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
        _draw_arrow(ax, x, 4.7, x, 2.3)
    ax.text(5, 0.5, f"{name}: sequence-to-sequence",
             ha="center", fontsize=9, color="#6b7494")


# ----------------------- Set -----------------------
def _draw_set(model, ax, info):
    """Draw permutation-invariant set transformer."""
    # Set of inputs
    for i, x in enumerate([1.5, 3, 4.5, 6, 7.5]):
        ax.scatter([x], [8], s=200, color=COLORS["input"], alpha=0.5, zorder=3)
        ax.text(x, 8, f"x{i+1}", ha="center", va="center", fontsize=7, zorder=4)
    ax.text(5, 6.5, "permutation-invariant", ha="center",
             fontsize=8, color="#6b7494")
    # Encoder
    ax.text(5, 5, "Set Encoder", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.4))
    for x in [1.5, 3, 4.5, 6, 7.5]:
        _draw_arrow(ax, x, 7.7, 5, 5.3)
    # Pooling
    ax.text(5, 3, "Pooling (sum/mean)", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["norm"], alpha=0.3))
    _draw_arrow(ax, 5, 4.6, 5, 3.3)
    # Output
    ax.text(5, 1.5, "Output", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
    _draw_arrow(ax, 5, 2.7, 5, 1.8)


# ----------------------- Memory-augmented -----------------------
def _draw_memory_augmented(model, ax, info):
    """Draw DNC/NTM: controller + external memory matrix."""
    tech = info.get("tech_obj")
    mem_n = getattr(tech, "mem_n", 16) if tech else 16
    mem_dim = getattr(tech, "mem_dim", 8) if tech else 8

    # Controller
    ax.text(2.5, 8, "Input", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    ax.text(2.5, 6, "Controller", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.4))
    _draw_arrow(ax, 2.5, 7.7, 2.5, 6.3)

    # Memory matrix
    ax.text(7, 8, f"Memory {mem_n}×{mem_dim}", ha="center",
             fontsize=10, fontweight="bold")
    # Draw memory cells as grid
    for i in range(min(mem_n, 6)):
        for j in range(min(mem_dim, 4)):
            x = 5.5 + j * 0.6
            y = 7 - i * 0.6
            rect = Rectangle((x, y), 0.5, 0.5, facecolor=COLORS["memory"],
                              alpha=0.3, edgecolor=COLORS["memory"])
            ax.add_patch(rect)
    # Read/Write heads
    _draw_arrow(ax, 3.2, 6, 5.3, 6.5, color=COLORS["memory"])
    ax.text(4, 6.6, "read", fontsize=7, color=COLORS["memory"])
    _draw_arrow(ax, 5.3, 5.5, 3.2, 5.7, color=COLORS["output"])
    ax.text(4, 5.2, "write", fontsize=7, color=COLORS["output"])

    # Output
    ax.text(2.5, 3, "Output", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
    _draw_arrow(ax, 2.5, 5.7, 2.5, 3.3)
    ax.text(5, 1, info["name"], ha="center", fontsize=10, fontweight="bold")


# ----------------------- Meta-learning -----------------------
def _draw_meta_learning(model, ax, info):
    """Draw MAML: inner/outer loop."""
    # Outer loop
    ax.text(5, 9, "Outer loop: meta-θ", ha="center", fontsize=11, fontweight="bold",
             color=COLORS["attention"])
    # Inner loop
    ax.text(5, 7.5, "Inner loop (per task)", ha="center", fontsize=10, fontweight="bold")
    # Tasks
    for i, name in enumerate(["Task 1", "Task 2", "Task 3"]):
        x = 2 + i * 3
        ax.text(x, 5.5, name, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
        ax.text(x, 4, f"adapt θ → θ'_i", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
        _draw_arrow(ax, x, 5.2, x, 4.3)
        # Feedback to outer
        _draw_arrow(ax, x, 3.7, 5, 1.5, color=COLORS["attention"])
    # Meta-update
    ax.text(5, 1, "Meta-update: θ ← θ − β∇_θ Σ L_i(θ'_i)",
             ha="center", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.4))


# ----------------------- Few-shot -----------------------
def _draw_few_shot(model, ax, info):
    """Draw few-shot: support set + query → encoder → prediction."""
    tech = info.get("tech_obj")
    n_way = getattr(tech, "n_way", 5) if tech else 5
    n_support = getattr(tech, "n_support", 1) if tech else 1

    # Support set
    ax.text(2, 8, f"Support\n{n_way}-way {n_support}-shot",
             ha="center", fontsize=9, fontweight="bold")
    for i in range(min(n_way, 4)):
        ax.scatter([1.5], [7 - i * 0.8], s=80, color=plt.cm.Set1(i / 4), alpha=0.6)
        ax.scatter([2.5], [7 - i * 0.8], s=80, color=plt.cm.Set1(i / 4), alpha=0.6)

    # Query
    ax.text(2, 3.5, "Query", ha="center", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))

    # Encoder
    ax.text(5, 6, "Encoder", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.4))
    _draw_arrow(ax, 3, 7, 4.5, 6.3)
    _draw_arrow(ax, 2, 3.8, 4.5, 5.7)

    # Distances + prediction
    ax.text(8, 6, "Distances", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
    _draw_arrow(ax, 5.5, 6, 7.5, 6)
    ax.text(8, 3.5, "Prediction", ha="center", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
    _draw_arrow(ax, 8, 5.7, 8, 3.8)
    ax.text(5, 1, info["name"], ha="center", fontsize=10, fontweight="bold")


# ----------------------- Metric learning -----------------------
def _draw_metric_learning(model, ax, info):
    """Draw Siamese twin-tower network."""
    # Input pair
    ax.text(2.5, 8, "x_1", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    ax.text(7.5, 8, "x_2", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    # Twin encoders
    ax.text(2.5, 6, "Encoder f", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.4))
    ax.text(7.5, 6, "Encoder f", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.4))
    _draw_arrow(ax, 2.5, 7.7, 2.5, 6.3)
    _draw_arrow(ax, 7.5, 7.7, 7.5, 6.3)
    # Embeddings
    ax.text(2.5, 4, "z_1", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
    ax.text(7.5, 4, "z_2", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.3))
    _draw_arrow(ax, 2.5, 5.7, 2.5, 4.3)
    _draw_arrow(ax, 7.5, 5.7, 7.5, 4.3)
    # Distance
    ax.text(5, 2, "Distance", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.4))
    _draw_arrow(ax, 3, 3.7, 4.5, 2.3)
    _draw_arrow(ax, 7, 3.7, 5.5, 2.3)


# ----------------------- Filtering (Kalman) -----------------------
def _draw_filtering(model, ax, info):
    """Draw Kalman predict-update loop."""
    ax.text(2.5, 7, "Predict", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["attention"], alpha=0.4))
    ax.text(7.5, 7, "Update", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["fc"], alpha=0.4))
    ax.text(2.5, 5, "x = Ax\nP = APAᵀ+Q", ha="center", fontsize=9)
    ax.text(7.5, 5, "x = x + K(z-Hx)\nP = (I-KH)P", ha="center", fontsize=9)
    # Loop
    _draw_arrow(ax, 3.5, 7, 6.5, 7, color=COLORS["attention"])
    _draw_arrow(ax, 6.5, 6.5, 3.5, 6.5, color=COLORS["fc"])
    ax.text(5, 7.5, "K = Kalman gain", ha="center", fontsize=8, color="#6b7494")
    # Observation
    ax.text(9.5, 5, "z\n(measurement)", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    _draw_arrow(ax, 9, 5, 8, 5)
    ax.text(5, 2, "Kalman filter: recursive Bayesian estimator",
             ha="center", fontsize=10, fontweight="bold")


# ----------------------- Neuromorphic -----------------------
def _draw_neuromorphic(model, ax, info):
    """Draw spiking neural network: LIF neurons + spikes."""
    # LIF neuron schematic
    ax.text(5, 9, "Leaky Integrate-and-Fire neuron", ha="center",
             fontsize=10, fontweight="bold")
    # Spikes
    rng = np.random.default_rng(42)
    for layer_idx, x in enumerate([2, 5, 8]):
        for n in range(5):
            y = 7 - n * 0.8
            ax.scatter([x], [y], s=150, color=COLORS["neuro"], alpha=0.5, zorder=3)
            # Random spike trail
            spikes_x = np.linspace(x - 0.5, x + 0.5, 10)
            spikes_y = y + 0.3 * (rng.random(10) > 0.7).astype(float)
            ax.plot(spikes_x, spikes_y + y * 0, color=COLORS["output"],
                     alpha=0.5, linewidth=0.5)
        if layer_idx > 0:
            prev_x = [2, 5][layer_idx - 1]
            for n1 in range(5):
                for n2 in range(5):
                    if rng.random() > 0.7:
                        ax.plot([prev_x + 0.2, x - 0.2],
                                 [7 - n1 * 0.8, 7 - n2 * 0.8],
                                 color="#a0aac4", alpha=0.2, linewidth=0.5)
    ax.text(5, 1.5, "Spiking NN — time-encoded, energy-efficient",
             ha="center", fontsize=9, color="#6b7494")


# ----------------------- Feedforward (RBF/ELM) -----------------------
def _draw_feedforward(model, ax, info):
    """Draw feedforward: input → RBF/ELM hidden → output."""
    name = info["name"]
    tech = info.get("tech_obj")
    n_hidden = getattr(tech, "n_hidden", getattr(tech, "n_centers", 8)) if tech else 8
    layers = [
        ("Input", "", COLORS["input"]),
        (name, f"{n_hidden} units", COLORS["fc"]),
        ("Output", "", COLORS["output"]),
    ]
    _draw_layers(ax, layers, title=name.upper())


# ----------------------- Reservoir -----------------------
def _draw_reservoir(model, ax, info):
    """Draw reservoir computing: input → fixed reservoir → readout."""
    name = info["name"]
    tech = info.get("tech_obj")
    n_res = getattr(tech, "n_reservoir", 100) if tech else 100

    # Input
    ax.text(1.5, 5, "Input", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["input"], alpha=0.3))
    # Reservoir (random nodes)
    rng = np.random.default_rng(42)
    n_show = min(n_res, 12)
    pts = rng.uniform([3.5, 3], [6.5, 7], (n_show, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=80, color=COLORS["attention"], alpha=0.5)
    # Random connections
    for i in range(n_show):
        for j in range(n_show):
            if i != j and rng.random() > 0.85:
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                         color="#a0aac4", alpha=0.2, linewidth=0.3)
    # Input → reservoir
    _draw_arrow(ax, 2, 5, 3.5, 5)
    # Readout
    ax.text(8.5, 5, "Readout", ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=COLORS["output"], alpha=0.3))
    for p in pts[:5]:
        _draw_arrow(ax, p[0] + 0.1, p[1], 7.9, 5, color="#a0aac4")
    ax.text(5, 1.5, f"{name}: reservoir size {n_res}",
             ha="center", fontsize=10, fontweight="bold")


# ----------------------- Time-series -----------------------
def _draw_time_series(model, ax, info):
    """Draw AR/MA coefficient stem plot."""
    tech = info.get("tech_obj")
    p = getattr(tech, "p", 3) if tech else 3
    q = getattr(tech, "q", 2) if tech else 2

    # AR coefficients
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(2.5, 8.5, f"AR(p={p})", ha="center", fontsize=10, fontweight="bold")
    rng = np.random.default_rng(42)
    ar_vals = rng.uniform(-0.5, 0.5, p)
    for i, v in enumerate(ar_vals):
        x = 1 + i * (3 / max(p, 1))
        ax.plot([x, x], [7, 7 + abs(v) * 3], color=COLORS["input"], linewidth=2)
        ax.scatter([x], [7 + abs(v) * 3], s=50, color=COLORS["input"])
        ax.text(x, 6.7, f"φ{i+1}", ha="center", fontsize=8)

    # MA coefficients
    ax.text(7.5, 8.5, f"MA(q={q})", ha="center", fontsize=10, fontweight="bold")
    ma_vals = rng.uniform(-0.5, 0.5, q)
    for i, v in enumerate(ma_vals):
        x = 6 + i * (3 / max(q, 1))
        ax.plot([x, x], [7, 7 + abs(v) * 3], color=COLORS["output"], linewidth=2)
        ax.scatter([x], [7 + abs(v) * 3], s=50, color=COLORS["output"])
        ax.text(x, 6.7, f"θ{i+1}", ha="center", fontsize=8)

    ax.text(5, 4, "Forecast: x_t = Σφᵢx_{t-i} + Σθⱼε_{t-j}",
             ha="center", fontsize=9, color="#6b7494")
    ax.text(5, 2, f"{info['name']}", ha="center", fontsize=10, fontweight="bold")


# ----------------------- Feature selection -----------------------
def _draw_feature_selection(model, ax, info):
    """Draw feature selection: ranked bar chart."""
    name = info["name"]
    # Synthetic scores
    rng = np.random.default_rng(42)
    scores = sorted(rng.uniform(0, 1, 10), reverse=True)
    colors_bar = [COLORS["input"] if i < 5 else COLORS["unknown"]
                   for i in range(10)]
    ax.bar(range(10), scores, color=colors_bar, alpha=0.6)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"f{i}" for i in range(10)], fontsize=8)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(f"{name} — feature ranking (top-5 selected)",
                  fontsize=12, fontweight="bold")
    ax.axvline(4.5, color=COLORS["output"], linestyle="--", linewidth=1)


# ----------------------- Supervised (other) -----------------------
def _draw_supervised_other(model, ax, info):
    """Generic supervised diagram."""
    name = info["name"]
    layers = [
        ("Input", "", COLORS["input"]),
        (name, "", COLORS["fc"]),
        ("Output", "", COLORS["output"]),
    ]
    _draw_layers(ax, layers, title=name.upper())


# ----------------------- Attention -----------------------
def _draw_attention(model, ax, info):
    """Draw sparse attention pattern."""
    name = info["name"]
    # Draw attention matrix
    n = 8
    rng = np.random.default_rng(42)
    attn = rng.random((n, n))
    # Sparse mask
    mask = rng.random((n, n)) > 0.6
    attn = attn * mask
    ax.imshow(attn, cmap="viridis", alpha=0.7,
               extent=[1, 9, 1, 9], aspect="auto")
    ax.set_title(f"{name} — attention pattern",
                  fontsize=12, fontweight="bold")
    ax.set_xlabel("key", fontsize=9)
    ax.set_ylabel("query", fontsize=9)


# ----------------------- Bridge / pipeline -----------------------
def _draw_federated_bridge(model, ax, info):
    """Draw FederatedBridge data flow diagram."""
    pieces = model.pieces if model is not None and hasattr(model, "pieces") else {}
    order = model._order if model is not None and hasattr(model, "_order") else list(pieces.keys())
    if not order:
        order = ["A", "B", "C"]
    n = len(order)
    for i, name in enumerate(order):
        x = 1 + i * (8 / max(n, 1))
        y = 5
        piece = pieces.get(name) if pieces else None
        framework = getattr(piece, "framework", "model") if piece else "model"
        color = COLORS.get(framework, COLORS["bridge"])
        _draw_box(ax, x, y, 1.2, 0.8, name, framework, color)
        if i > 0:
            _draw_arrow(ax, x - 1.2, y, x - 0.6, y)
    ax.text(5, 2, f"FederatedBridge: {n} pieces",
             ha="center", fontsize=10, fontweight="bold")


def _draw_pipeline(model, ax, info):
    """Draw a DAGPipeline."""
    if model is not None and hasattr(model, "steps"):
        visualize_pipeline(model, save=None)
    else:
        _draw_generic(model, ax, info)


# ----------------------- Generic fallback -----------------------
def _draw_generic(model, ax, info):
    """Generic fallback: technique name + family + key parameters."""
    name = info["name"]
    family = info["family"]
    ax.text(5, 6, name, ha="center", va="center", fontsize=16, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["unknown"], alpha=0.2))
    ax.text(5, 4, f"family: {family}", ha="center", va="center",
             fontsize=10, color="#6b7494")
    if model is not None and hasattr(model, "technique"):
        tech = model.technique
        params = []
        for attr in ("hidden_dim", "epochs", "lr", "n_estimators", "n_clusters",
                      "n_components", "k", "alpha", "kernel", "d_model", "n_layers",
                      "latent_dim", "vocab_size", "in_dim", "out_dim", "n_actions"):
            if hasattr(tech, attr):
                params.append(f"{attr}={getattr(tech, attr)}")
        if params:
            ax.text(5, 3, "\n".join(params[:6]), ha="center", va="center",
                     fontsize=8, color="#6b7494")


__all__ = ["visualize_architecture", "visualize_technique", "visualize_pipeline"]
