"""
qai.sklearn_compat -- discoverability layer for sklearn-wrapped techniques.

Many qai techniques are thin wrappers around sklearn estimators. This
module re-exports them by category so users who specifically want the
sklearn-backed techniques can find them without scanning the full 222-
entry technique registry.

Usage:
    import qai
    # Instead of searching the full registry:
    from qai.sklearn_compat import CLASSIFIERS, REGRESSORS, CLUSTERERS, PREPROCESSORS

    # Or just browse:
    print(qai.sklearn_compat.list_categories())
"""
from __future__ import annotations
from typing import Dict, List

# =====================================================================
# sklearn-wrapped classifiers (available via qai.build(technique=...))
# =====================================================================
CLASSIFIERS = {
    "classifier": "sklearn LogisticRegression",
    "decision_tree": "sklearn DecisionTreeClassifier",
    "random_forest": "sklearn RandomForestClassifier",
    "extra_trees": "sklearn ExtraTreesClassifier",
    "gradient_boosting": "sklearn GradientBoostingClassifier",
    "adaboost": "sklearn AdaBoostClassifier",
    "knn": "sklearn KNeighborsClassifier",
    "svm": "sklearn SVC",
    "naive_bayes": "sklearn GaussianNB",
    "multinomial_naive_bayes": "sklearn MultinomialNB",
    "bernoulli_naive_bayes": "sklearn BernoulliNB",
    "complement_naive_bayes": "sklearn ComplementNB",
    "perceptron": "sklearn Perceptron",
    "sgd_classifier": "sklearn SGDClassifier",
    "linear_svc": "sklearn LinearSVC",
    "nu_svc": "sklearn NuSVC",
    "ridge": "sklearn RidgeClassifier (via LDA)",
    "lda": "sklearn LinearDiscriminantAnalysis",
    "qda": "sklearn QuadraticDiscriminantAnalysis",
    "bagging_classifier": "sklearn BaggingClassifier",
    "hist_gradient_boosting": "sklearn HistGradientBoostingClassifier",
    "passive_aggressive_classifier": "sklearn PassiveAggressiveClassifier",
    "radius_neighbors_classifier": "sklearn RadiusNeighborsClassifier",
    "nearest_centroid": "sklearn NearestCentroid",
}

# =====================================================================
# sklearn-wrapped regressors
# =====================================================================
REGRESSORS = {
    "ridge": "sklearn Ridge",
    "lasso": "sklearn Lasso",
    "elastic_net": "sklearn ElasticNet",
    "kernel_ridge": "sklearn KernelRidge",
    "bayesian_ridge": "sklearn BayesianRidge",
    "ard_regression": "sklearn ARDRegression",
    "huber": "sklearn HuberRegressor",
    "ransac": "sklearn RANSACRegressor",
    "theil_sen": "sklearn TheilSenRegressor",
    "quantile_regression": "sklearn QuantileRegressor",
    "decision_tree_regressor": "sklearn DecisionTreeRegressor",
    "random_forest_regressor": "sklearn RandomForestRegressor",
    "adaboost_regressor": "sklearn AdaBoostRegressor",
    "gradient_boosting_regressor": "sklearn GradientBoostingRegressor",
    "svr": "sklearn SVR",
    "sgd_regressor": "sklearn SGDRegressor",
    "knn_regressor": "sklearn KNeighborsRegressor",
    "dummy_regressor": "sklearn DummyRegressor",
}

# =====================================================================
# sklearn-wrapped clustering / unsupervised
# =====================================================================
CLUSTERERS = {
    "kmeans": "sklearn KMeans",
    "dbscan": "sklearn DBSCAN",
    "gmm": "sklearn GaussianMixture",
    "hierarchical": "sklearn AgglomerativeClustering",
    "isolation_forest": "sklearn IsolationForest",
    "tsne": "sklearn TSNE",
    "spectral_clustering": "sklearn SpectralClustering",
    "fast_ica": "sklearn FastICA",
    "isomap": "sklearn Isomap",
    "lle": "sklearn LocallyLinearEmbedding",
    "birch": "sklearn Birch",
    "optics": "sklearn OPTICS",
    "mean_shift": "sklearn MeanShift",
    "affinity_propagation": "sklearn AffinityPropagation",
    "mds": "sklearn MDS",
}

# =====================================================================
# sklearn-wrapped preprocessing
# =====================================================================
PREPROCESSORS = {
    "standard_scaler": "sklearn StandardScaler",
    "minmax_scaler": "sklearn MinMaxScaler",
    "robust_scaler": "sklearn RobustScaler",
    "maxabs_scaler": "sklearn MaxAbsScaler",
    "normalizer": "sklearn Normalizer",
    "binarizer": "sklearn Binarizer",
    "polynomial_features": "sklearn PolynomialFeatures",
    "one_hot_encoder": "sklearn OneHotEncoder",
    "label_encoder": "sklearn LabelEncoder",
    "simple_imputer": "sklearn SimpleImputer",
    "knn_imputer": "sklearn KNNImputer",
    "kbins_discretizer": "sklearn KBinsDiscretizer",
    "select_k_best": "sklearn SelectKBest",
    "variance_threshold": "sklearn VarianceThreshold",
    "rfe": "sklearn RFE",
    "kernel_pca": "sklearn KernelPCA",
    "sparse_pca": "sklearn SparsePCA",
    "incremental_pca": "sklearn IncrementalPCA",
    "agglomerative_clustering": "sklearn AgglomerativeClustering",
    "bisecting_kmeans": "sklearn BisectingKMeans",
    "mini_batch_kmeans": "sklearn MiniBatchKMeans",
    "svc": "sklearn SVC",
    "one_class_svm": "sklearn OneClassSVM",
    "stacking_classifier": "sklearn StackingClassifier",
    "voting_classifier": "sklearn VotingClassifier",
    "nearest_neighbors": "sklearn NearestNeighbors",
    "local_outlier_factor": "sklearn LocalOutlierFactor",
    "spectral_embedding": "sklearn SpectralEmbedding",
    "bernoulli_rbm": "sklearn BernoulliRBM",
    "kernel_density": "sklearn KernelDensity",
}


def list_categories() -> Dict[str, Dict[str, str]]:
    """Return all categories + their techniques."""
    return {
        "classifiers": CLASSIFIERS,
        "regressors": REGRESSORS,
        "clusterers": CLUSTERERS,
        "preprocessors": PREPROCESSORS,
    }


def list_all() -> List[str]:
    """Return all sklearn-backed technique names as a flat list."""
    all_names = []
    for cat in (CLASSIFIERS, REGRESSORS, CLUSTERERS, PREPROCESSORS):
        all_names.extend(cat.keys())
    return sorted(set(all_names))


def is_sklearn_technique(name: str) -> bool:
    """Check if a technique name is sklearn-backed."""
    return name in list_all()


def get_sklearn_class(name: str) -> str:
    """Return the underlying sklearn class name for a qai technique."""
    for cat in (CLASSIFIERS, REGRESSORS, CLUSTERERS, PREPROCESSORS):
        if name in cat:
            return cat[name]
    return "?"


__all__ = [
    "CLASSIFIERS", "REGRESSORS", "CLUSTERERS", "PREPROCESSORS",
    "list_categories", "list_all", "is_sklearn_technique", "get_sklearn_class",
]
