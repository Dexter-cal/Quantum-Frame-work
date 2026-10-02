"""
qai.techniques -- The core catalogue of built-in and user-registered ML techniques.
"""
from __future__ import annotations
from typing import Dict, Type
from .base import Technique
from .regression import Regression
from .classifier import Classifier
from .decision_tree import DecisionTree
from .knn import KNearestNeighbors, KNearestNeighbors as KNN
from .svm import SVM
from .naive_bayes import NaiveBayes
from .kmeans import KMeans
from .pca import PCA
from .perceptron import Perceptron
from .random_forest import RandomForest
from .tabular_policy import TabularPolicy
from .dbscan import DBSCAN
from .neural_network import NeuralNetwork
from .gmm import GaussianMixtureModel
from .hierarchical import HierarchicalClustering
from .time_series import TimeSeriesForecaster
from .isolation_forest import IsolationForest
from .tsne import TSNE
from .multinomial_naive_bayes import MultinomialNaiveBayes
from .supervised_extensions import LDA, QDA, AdaBoost, GradientBoosting, ExtraTrees, RidgeRegression, LassoRegression, ElasticNet, KernelRidge
from .truncated_svd import TruncatedSVD

# Technique collections grouped by learning paradigm
SUPERVISED = [
    "classifier",
    "decision_tree",
    "knn",
    "svm",
    "naive_bayes",
    "perceptron",
    "random_forest",
    "regression",
    "neural_network",
    "time_series",
    "multinomial_naive_bayes",
    "lda",
    "qda",
    "adaboost",
    "gradient_boosting",
    "extra_trees",
    "ridge",
    "lasso",
    "elastic_net",
    "kernel_ridge",
]

UNSUPERVISED = [
    "kmeans",
    "pca",
    "dbscan",
    "gmm",
    "hierarchical",
    "isolation_forest",
    "tsne",
    "truncated_svd",
]

REINFORCEMENT = [
    "tabular_policy",
]

# Technique registry
_REGISTRY: Dict[str, Type[Technique]] = {
    "regression": Regression,
    "classifier": Classifier,
    "decision_tree": DecisionTree,
    "knn": KNearestNeighbors,
    "svm": SVM,
    "naive_bayes": NaiveBayes,
    "kmeans": KMeans,
    "pca": PCA,
    "perceptron": Perceptron,
    "random_forest": RandomForest,
    "tabular_policy": TabularPolicy,
    "dbscan": DBSCAN,
    "neural_network": NeuralNetwork,
    "gmm": GaussianMixtureModel,
    "hierarchical": HierarchicalClustering,
    "time_series": TimeSeriesForecaster,
    "isolation_forest": IsolationForest,
    "tsne": TSNE,
    "multinomial_naive_bayes": MultinomialNaiveBayes,
    "lda": LDA,
    "qda": QDA,
    "adaboost": AdaBoost,
    "gradient_boosting": GradientBoosting,
    "extra_trees": ExtraTrees,
    "ridge": RidgeRegression,
    "lasso": LassoRegression,
    "elastic_net": ElasticNet,
    "kernel_ridge": KernelRidge,
    "truncated_svd": TruncatedSVD,
}

def get_technique(name: str) -> Type[Technique]:
    """Retrieve a technique class by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown technique '{name}'. Available built-in techniques: {sorted(list(_REGISTRY.keys()))}"
        )
    return _REGISTRY[name]

def register_technique(name: str, cls: Type[Technique]) -> None:
    """Register a custom technique dynamically into qai."""
    if not issubclass(cls, Technique):
        raise TypeError("Custom techniques must inherit from qai.techniques.Technique")
    _REGISTRY[name] = cls

__all__ = [
    "Technique",
    "Regression",
    "Classifier",
    "DecisionTree",
    "KNN",
    "KNearestNeighbors",
    "SVM",
    "NaiveBayes",
    "KMeans",
    "PCA",
    "Perceptron",
    "RandomForest",
    "TabularPolicy",
    "DBSCAN",
    "NeuralNetwork",
    "GaussianMixtureModel",
    "HierarchicalClustering",
    "TimeSeriesForecaster",
    "IsolationForest",
    "TSNE",
    "MultinomialNaiveBayes",
    "LDA",
    "QDA",
    "AdaBoost",
    "GradientBoosting",
    "ExtraTrees",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNet",
    "KernelRidge",
    "TruncatedSVD",
    "SUPERVISED",
    "UNSUPERVISED",
    "REINFORCEMENT",
    "get_technique",
    "register_technique",
]
