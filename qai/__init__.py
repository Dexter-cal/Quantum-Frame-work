"""
qai -- unified machine learning and artificial intelligence framework.

    import qai
    model = qai.build(type="regression")
    model.train(X, y)
    model.predict(x)

Provides unified single-import access to modeling, pipelines, datasets,
cross-validation, autotuning, benchmark, persistence, quantization, evaluation metrics,
explainability, synthetic generators, hardware profiling, system utilities, math & activation functions,
serving, preprocessing, governance, and tracking.
"""
from .core import build, Model, logic, LogicError, CompatibilityError
from .core.logic import save_logic_source, load_logic_source
from .core.environment import GridWorld
from .core.dataset import Dataset
from .core.serve import serve, build_flask_app
from .core.pipeline import Pipeline
from .core.cross_validation import cross_validate, k_fold_split
from .core.ensemble import VotingEnsemble
from .core.autotune import AutoTuner, autotune
from .core.benchmark import benchmark
from .core.persistence import save_model, load_model
from .core.quantize import compress_model
from .core.metrics import (
    classification_report, confusion_matrix, mean_absolute_error, r2_score, log_loss, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score, silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, mean_squared_log_error, mean_absolute_percentage_error,
    median_absolute_error, explained_variance_score, max_error
)
from .core.explainability import feature_importance, permutation_importance
from .core.generators import make_classification, make_regression
from .core.hardware import get_hardware_info, get_optimal_device
from .techniques import register_technique, Technique
from .learning import get_learning_technique, LearningTechnique

# Advanced sub-packages & utilities
from .preprocessing import (
    StandardScaler, LabelEncoder, SimpleImputer, PolynomialFeatures, MinMaxScaler, RobustScaler,
    Normalizer, OneHotEncoder, Binarizer, VarianceThreshold, SelectKBest, MaxAbsScaler, OrdinalEncoder,
    TargetEncoder, KBinsDiscretizer, clean_dataset
)
from .governance import (
    detect_drift, weight_distance, calibration_curve, demographic_parity_difference,
    equalized_odds_difference, disparate_impact_ratio, fairness_audit
)
from .tracking import ExperimentTracker

# System and Math modules
from . import system
from . import math

# Top-level math convenience exports
from .math import (
    sigmoid, relu, softmax, gelu, swish, tanh, euclidean_distance, cosine_similarity,
    manhattan_distance, minkowski_distance, huber_loss, focal_loss, triplet_loss,
    kl_divergence, js_divergence, chebyshev_distance, canberra_distance, braycurtis_distance
)
from .system import ResourceGuard, auto_clean_memory, check_ram_threshold

__all__ = [
    "build", "Model", "register_technique", "Technique", "logic", "LogicError",
    "GridWorld", "get_learning_technique", "LearningTechnique", "CompatibilityError",
    "Dataset", "serve", "build_flask_app", "Pipeline", "cross_validate", "k_fold_split",
    "VotingEnsemble", "AutoTuner", "autotune", "benchmark", "save_model", "load_model",
    "compress_model", "classification_report", "confusion_matrix", "mean_absolute_error", "r2_score",
    "log_loss", "roc_auc_score", "cohen_kappa_score", "matthews_corrcoef", "balanced_accuracy_score",
    "silhouette_score", "davies_bouldin_score", "adjusted_rand_score", "normalized_mutual_info_score",
    "mean_squared_log_error", "mean_absolute_percentage_error", "median_absolute_error",
    "explained_variance_score", "max_error", "feature_importance", "permutation_importance",
    "make_classification", "make_regression", "get_hardware_info", "get_optimal_device",
    "save_logic_source", "load_logic_source", "StandardScaler", "LabelEncoder", "SimpleImputer",
    "PolynomialFeatures", "MinMaxScaler", "RobustScaler", "Normalizer", "OneHotEncoder", "Binarizer",
    "VarianceThreshold", "SelectKBest", "MaxAbsScaler", "OrdinalEncoder", "TargetEncoder", "KBinsDiscretizer",
    "clean_dataset", "detect_drift", "weight_distance", "calibration_curve", "demographic_parity_difference",
    "equalized_odds_difference", "disparate_impact_ratio", "fairness_audit", "ExperimentTracker",
    "system", "math", "sigmoid", "relu", "softmax", "gelu", "swish", "tanh",
    "euclidean_distance", "cosine_similarity", "manhattan_distance", "minkowski_distance",
    "huber_loss", "focal_loss", "triplet_loss", "kl_divergence", "js_divergence",
    "chebyshev_distance", "canberra_distance", "braycurtis_distance",
    "ResourceGuard", "auto_clean_memory", "check_ram_threshold"
]
__version__ = "0.1.0-prototype"
