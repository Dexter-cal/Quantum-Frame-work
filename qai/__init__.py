"""
qai -- unified machine learning and artificial intelligence framework.

    import qai
    model = qai.build(type="regression")
    model.train(X, y)
    model.predict(x)

Provides unified single-import access to modeling, pipelines, datasets,
cross-validation, autotuning, serving, preprocessing, governance, and tracking.
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
from .techniques import register_technique, Technique
from .learning import get_learning_technique, LearningTechnique

# Advanced features and extensions
from .preprocessing import StandardScaler, LabelEncoder, SimpleImputer
from .governance import detect_drift
from .tracking import ExperimentTracker

__all__ = [
    "build", "Model", "register_technique", "Technique", "logic", "LogicError",
    "GridWorld", "get_learning_technique", "LearningTechnique", "CompatibilityError",
    "Dataset", "serve", "build_flask_app", "Pipeline", "cross_validate", "k_fold_split",
    "VotingEnsemble", "AutoTuner", "autotune", "save_logic_source", "load_logic_source",
    "StandardScaler", "LabelEncoder", "SimpleImputer", "detect_drift", "ExperimentTracker"
]
__version__ = "0.1.0-prototype"
