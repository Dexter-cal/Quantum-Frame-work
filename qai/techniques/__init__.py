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
from .more_classifiers import (
    BernoulliNaiveBayes, ComplementNaiveBayes, SGDClassifier, PassiveAggressiveClassifier,
    LinearSVC, NuSVC, RadiusNeighborsClassifier, NearestCentroid, BaggingClassifier, HistGradientBoostingClassifier
)
from .more_regressors import (
    BayesianRidge, ARDRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor,
    QuantileRegressor, DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
)
from .manifold import SpectralClustering, FastICA, Isomap, LocallyLinearEmbedding

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
    "bernoulli_naive_bayes",
    "complement_naive_bayes",
    "sgd_classifier",
    "passive_aggressive_classifier",
    "linear_svc",
    "nu_svc",
    "radius_neighbors_classifier",
    "nearest_centroid",
    "bagging_classifier",
    "hist_gradient_boosting",
    "bayesian_ridge",
    "ard_regression",
    "huber",
    "ransac",
    "theil_sen",
    "quantile_regression",
    "decision_tree_regressor",
    "random_forest_regressor",
    "adaboost_regressor",
    "gradient_boosting_regressor",
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
    "spectral_clustering",
    "fast_ica",
    "isomap",
    "lle",
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
    "bernoulli_naive_bayes": BernoulliNaiveBayes,
    "complement_naive_bayes": ComplementNaiveBayes,
    "sgd_classifier": SGDClassifier,
    "passive_aggressive_classifier": PassiveAggressiveClassifier,
    "linear_svc": LinearSVC,
    "nu_svc": NuSVC,
    "radius_neighbors_classifier": RadiusNeighborsClassifier,
    "nearest_centroid": NearestCentroid,
    "bagging_classifier": BaggingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "bayesian_ridge": BayesianRidge,
    "ard_regression": ARDRegression,
    "huber": HuberRegressor,
    "ransac": RANSACRegressor,
    "theil_sen": TheilSenRegressor,
    "quantile_regression": QuantileRegressor,
    "decision_tree_regressor": DecisionTreeRegressor,
    "random_forest_regressor": RandomForestRegressor,
    "adaboost_regressor": AdaBoostRegressor,
    "gradient_boosting_regressor": GradientBoostingRegressor,
    "spectral_clustering": SpectralClustering,
    "fast_ica": FastICA,
    "isomap": Isomap,
    "lle": LocallyLinearEmbedding,
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
    "BernoulliNaiveBayes",
    "ComplementNaiveBayes",
    "SGDClassifier",
    "PassiveAggressiveClassifier",
    "LinearSVC",
    "NuSVC",
    "RadiusNeighborsClassifier",
    "NearestCentroid",
    "BaggingClassifier",
    "HistGradientBoostingClassifier",
    "BayesianRidge",
    "ARDRegression",
    "HuberRegressor",
    "RANSACRegressor",
    "TheilSenRegressor",
    "QuantileRegressor",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "AdaBoostRegressor",
    "GradientBoostingRegressor",
    "SpectralClustering",
    "FastICA",
    "Isomap",
    "LocallyLinearEmbedding",
    "SUPERVISED",
    "UNSUPERVISED",
    "REINFORCEMENT",
    "get_technique",
    "register_technique",
]
