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
]

UNSUPERVISED = [
    "kmeans",
    "pca",
    "dbscan",
    "gmm",
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
    "SUPERVISED",
    "UNSUPERVISED",
    "REINFORCEMENT",
    "get_technique",
    "register_technique",
]
