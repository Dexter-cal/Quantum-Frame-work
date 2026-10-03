"""
qai -- unified machine learning and artificial intelligence framework.

    import qai
    model = qai.build(type="regression")
    model.train(X, y)
    model.predict(x)

Provides unified single-import access to modeling, pipelines, datasets,
cross-validation, autotuning, benchmark, persistence, quantization, evaluation metrics,
synthetic generators, hardware profiling, serving, preprocessing, governance, and tracking.
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
from .core.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score, log_loss, roc_auc_score
from .core.generators import make_classification, make_regression
from .core.hardware import get_hardware_info, get_optimal_device
from .techniques import register_technique, Technique
from .learning import get_learning_technique, LearningTechnique

# Advanced features and extensions
from .preprocessing import StandardScaler, LabelEncoder, SimpleImputer, PolynomialFeatures, MinMaxScaler, RobustScaler, Normalizer, OneHotEncoder, Binarizer, clean_dataset
from .governance import detect_drift, weight_distance, calibration_curve
from .tracking import ExperimentTracker

__all__ = [
    "build", "Model", "register_technique", "Technique", "logic", "LogicError",
    "GridWorld", "get_learning_technique", "LearningTechnique", "CompatibilityError",
    "Dataset", "serve", "build_flask_app", "Pipeline", "cross_validate", "k_fold_split",
    "VotingEnsemble", "AutoTuner", "autotune", "benchmark", "save_model", "load_model",
    "compress_model", "classification_report", "confusion_matrix", "mean_absolute_error", "r2_score",
    "log_loss", "roc_auc_score", "make_classification", "make_regression", "get_hardware_info", "get_optimal_device",
    "save_logic_source", "load_logic_source", "StandardScaler", "LabelEncoder", "SimpleImputer",
    "PolynomialFeatures", "MinMaxScaler", "RobustScaler", "Normalizer", "OneHotEncoder", "Binarizer",
    "clean_dataset", "detect_drift", "weight_distance", "calibration_curve", "ExperimentTracker"
]
__version__ = "0.1.0-prototype"
