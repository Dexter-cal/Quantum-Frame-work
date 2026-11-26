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
from .transformer import Transformer
from .cnn import CNN
from .rnn import RNN, LSTM, GRU, Seq2Seq
from .generative import GAN, VAE, Diffusion
from .advanced_dl import MixtureOfExperts, GNN
from .mamba import Mamba
from .missing_ml import Autoencoder, DQN, PPO, SemiSupervised, AssociationRules, SelfSupervised
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

# ============================================================================
# Frontier techniques — added in 2026 completeness audit (web-verified)
# ============================================================================
from .frontier_generative import (
    NormalizingFlow, FlowMatching, ConsistencyModel, EnergyBasedModel,
    ScoreBasedModel, RestrictedBoltzmannMachine, BoltzmannMachine,
)
from .frontier_memory import (
    ModernHopfield, NeuralTuringMachine, DifferentiableNeuralComputer,
)
from .frontier_meta import (
    SiameseNetwork, PrototypicalNetwork, MatchingNetwork, MAML, RelationNetwork,
)
from .frontier_classical import (
    SelfOrganizingMap, CapsuleNetwork, EchoStateNetwork,
    ExtremeLearningMachine, NeuralGas, ARTNetwork,
)
from .frontier_rl import (
    DDPG, TD3, SAC, A2C, ValueIteration, PolicyIteration, RLHF,
)
from .frontier_sequence import (
    S4, PointerNetwork, SetTransformer, HiddenMarkovModel, CTC,
)
from .frontier_vision import (
    VisionTransformer, DeiT, NeRF, INR, UNet,
)
from .frontier_recsys import (
    CollaborativeFiltering, MatrixFactorization, WideAndDeep, DeepFM,
    Item2Vec, GraphSAGE, GAT, GIN,
)
from .frontier_misc import (
    BayesianNeuralNetwork, MCDropout, WorldModel, Dreamer,
    MixtureDensityNetwork, GroupedQueryAttention, ARIMA, KalmanFilter,
    LDA as LDATopicModel, OnlineLearning,
)
# 2025-era frontier architectures
from ..frontier2025 import Mamba2, DeepSeekMoE, TTTModel, Jamba
# Missing classical ML (added in completeness audit part 2)
from .missing_classical import (
    GaussianProcess, NMF, FactorAnalysis, IsotonicRegression,
    BIRCH, OPTICS, MeanShift, AffinityPropagation, MDS,
    PLSCanonical, CCA, KernelDensityEstimator,
    CovarianceEstimator, ProbabilityCalibrator,
)
# Missing DL architectures
from .missing_dl import (
    ResNet, DenseNet, EfficientNet, MobileNet,
    GCN, MPNN, DBN, SpikingNN, RBFNetwork,
    MultiQueryAttention, SparseAttention, LinearAttention,
)
# Missing RL techniques
from .missing_rl import SARSA, MCTS, MuZero
# Missing sklearn estimators (wrappers)
from .missing_sklearn import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    Normalizer, Binarizer, PolynomialFeatures, OneHotEncoder,
    LabelEncoder, SimpleImputer, KNNImputer, KBinsDiscretizer,
    SelectKBest, VarianceThreshold, RFE,
    KernelPCA, SparsePCA, IncrementalPCA, GaussianRandomProjection,
    AgglomerativeClustering, BisectingKMeans, MiniBatchKMeans,
    SVR, SGDRegressor, DummyRegressor, PoissonRegressor,
    KNeighborsRegressor, MLPRegressor,
    SVC, OneClassSVM,
    StackingClassifier, VotingClassifier,
    NearestNeighbors, LocalOutlierFactor, SpectralEmbedding,
    BernoulliRBM, KernelDensity,
)
# Missing 2026 techniques (DPO, ContrastiveLearning, GRPO, text vectorizers)
from .missing_2026 import (
    DPO, ContrastiveLearning, GRPO,
    CountVectorizer, TfidfVectorizer, TargetEncoder, HashingVectorizer,
)
# Universal training enhancements that wrap any torch-based qai technique
# (SAM, SWA, EMA, MixupTrainer, LabelSmoother, CosineAnnealer, OneCyclePolicy,
# TTA, PseudoLabeler, SnapshotEnsemble, Lookahead). These are NOT new model
# types -- they are utilities you combine with the existing techniques.
from .training_enhancements import (
    SAM, SWA, EMA, Lookahead,
    MixupTrainer, LabelSmoother,
    CosineAnnealer, OneCyclePolicy,
    TTA,
    PseudoLabeler, SnapshotEnsemble,
)
# sklearn gap fillers (preprocessing, feature_selection, decomposition, etc.)
from .sklearn_gap import (
    StandardScalerTechnique, MinMaxScalerTechnique, RobustScalerTechnique,
    NormalizerTechnique, PolynomialFeaturesTechnique,
    SelectKBestTechnique, VarianceThresholdTechnique, RFETechnique,
    KernelPCATechnique, SparsePCATechnique, MiniBatchKMeansTechnique,
    AgglomerativeClusteringTechnique, GaussianNBTechnique,
    SGDRegressorTechnique, SVRTechnique, KNeighborsRegressorTechnique,
    DummyRegressorTechnique, SpectralEmbeddingTechnique,
)
# Modern industry-standard techniques (filled the gaps from 2025 audit)
from .modern import (
    XGBoost, LightGBM, CatBoost,
    LlamaDecoder, T5Model, RWKV, CLIP,
    WhisperASR, Wav2Vec2, HuBERT,
)

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
    "transformer",
    "cnn",
    "rnn",
    "lstm",
    "gru",
    "seq2seq",
    "gan",
    "vae",
    "diffusion",
    "moe",
    "gnn",
    "mamba",
    "autoencoder",
    "dqn",
    "ppo",
    "semi_supervised",
    "association_rules",
    "self_supervised",
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
    "transformer": Transformer,
    "cnn": CNN,
    "rnn": RNN,
    "lstm": LSTM,
    "gru": GRU,
    "seq2seq": Seq2Seq,
    "gan": GAN,
    "vae": VAE,
    "diffusion": Diffusion,
    "moe": MixtureOfExperts,
    "gnn": GNN,
    "mamba": Mamba,
    "autoencoder": Autoencoder,
    "dqn": DQN,
    "ppo": PPO,
    "semi_supervised": SemiSupervised,
    "association_rules": AssociationRules,
    "self_supervised": SelfSupervised,
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
    # === frontier techniques (2026 audit) ===
    "normalizing_flow": NormalizingFlow,
    "flow_matching": FlowMatching,
    "consistency_model": ConsistencyModel,
    "energy_based": EnergyBasedModel,
    "score_based": ScoreBasedModel,
    "rbm": RestrictedBoltzmannMachine,
    "boltzmann": BoltzmannMachine,
    "hopfield": ModernHopfield,
    "ntm": NeuralTuringMachine,
    "dnc": DifferentiableNeuralComputer,
    "siamese": SiameseNetwork,
    "prototypical": PrototypicalNetwork,
    "matching": MatchingNetwork,
    "maml": MAML,
    "relation": RelationNetwork,
    "som": SelfOrganizingMap,
    "capsule": CapsuleNetwork,
    "echo_state": EchoStateNetwork,
    "elm": ExtremeLearningMachine,
    "neural_gas": NeuralGas,
    "art": ARTNetwork,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
    "a2c": A2C,
    "value_iteration": ValueIteration,
    "policy_iteration": PolicyIteration,
    "rlhf": RLHF,
    "s4": S4,
    "pointer_net": PointerNetwork,
    "set_transformer": SetTransformer,
    "hmm": HiddenMarkovModel,
    "ctc": CTC,
    "vit": VisionTransformer,
    "deit": DeiT,
    "nerf": NeRF,
    "inr": INR,
    "unet": UNet,
    "collab_filtering": CollaborativeFiltering,
    "matrix_factorization": MatrixFactorization,
    "wide_deep": WideAndDeep,
    "deepfm": DeepFM,
    "item2vec": Item2Vec,
    "graphsage": GraphSAGE,
    "gat": GAT,
    "gin": GIN,
    "bayesian_nn": BayesianNeuralNetwork,
    "mc_dropout": MCDropout,
    "world_model": WorldModel,
    "dreamer": Dreamer,
    "mixture_density": MixtureDensityNetwork,
    "gqa_attention": GroupedQueryAttention,
    "time_series_arima": ARIMA,
    "kalman_filter": KalmanFilter,
    "topic_model": LDATopicModel,
    "online_learning": OnlineLearning,
    # 2025-era frontier architectures
    "mamba2": Mamba2,
    "deepseek_moe": DeepSeekMoE,
    "ttt": TTTModel,
    "jamba": Jamba,
    # Modern industry-standard techniques (2025 audit gap-fillers)
    "xgboost": XGBoost,
    "lightgbm": LightGBM,
    "catboost": CatBoost,
    "llama_decoder": LlamaDecoder,
    "t5": T5Model,
    "rwkv": RWKV,
    "clip": CLIP,
    "whisper": WhisperASR,
    "wav2vec2": Wav2Vec2,
    "hubert": HuBERT,
    # === Missing classical ML (completeness audit part 2) ===
    "gaussian_process": GaussianProcess,
    "nmf": NMF,
    "factor_analysis": FactorAnalysis,
    "isotonic_regression": IsotonicRegression,
    "birch": BIRCH,
    "optics": OPTICS,
    "mean_shift": MeanShift,
    "affinity_propagation": AffinityPropagation,
    "mds": MDS,
    "pls_canonical": PLSCanonical,
    "cca": CCA,
    "kde": KernelDensityEstimator,
    "covariance": CovarianceEstimator,
    "probability_calibration": ProbabilityCalibrator,
    # === Missing DL architectures ===
    "resnet": ResNet,
    "densenet": DenseNet,
    "efficientnet": EfficientNet,
    "mobilenet": MobileNet,
    "gcn": GCN,
    "mpnn": MPNN,
    "dbn": DBN,
    "spiking_nn": SpikingNN,
    "rbf_network": RBFNetwork,
    "mqa_attention": MultiQueryAttention,
    "sparse_attention": SparseAttention,
    "linear_attention": LinearAttention,
    # === Missing RL ===
    "sarsa": SARSA,
    "mcts": MCTS,
    "muzero": MuZero,
    # === sklearn gap fillers ===
    "standard_scaler": StandardScalerTechnique,
    "minmax_scaler": MinMaxScalerTechnique,
    "robust_scaler": RobustScalerTechnique,
    "normalizer": NormalizerTechnique,
    "polynomial_features": PolynomialFeaturesTechnique,
    "select_k_best": SelectKBestTechnique,
    "variance_threshold": VarianceThresholdTechnique,
    "rfe": RFETechnique,
    "kernel_pca": KernelPCATechnique,
    "sparse_pca": SparsePCATechnique,
    "minibatch_kmeans": MiniBatchKMeansTechnique,
    "agglomerative": AgglomerativeClusteringTechnique,
    "gaussian_nb": GaussianNBTechnique,
    "sgd_regressor": SGDRegressorTechnique,
    "svr": SVRTechnique,
    "knn_regressor": KNeighborsRegressorTechnique,
    "kneighbors_regressor": KNeighborsRegressorTechnique,  # sklearn alias
    "dummy_regressor": DummyRegressorTechnique,
    "spectral_embedding": SpectralEmbeddingTechnique,
    # === Missing sklearn wrappers (from missing_sklearn.py) ===
    "standard_scaler": StandardScaler,
    "minmax_scaler": MinMaxScaler,
    "robust_scaler": RobustScaler,
    "maxabs_scaler": MaxAbsScaler,
    "normalizer": Normalizer,
    "binarizer": Binarizer,
    "polynomial_features": PolynomialFeatures,
    "one_hot_encoder": OneHotEncoder,
    "label_encoder": LabelEncoder,
    "simple_imputer": SimpleImputer,
    "knn_imputer": KNNImputer,
    "kbins_discretizer": KBinsDiscretizer,
    "select_k_best": SelectKBest,
    "variance_threshold2": VarianceThreshold,
    "rfe2": RFE,
    "kernel_pca2": KernelPCA,
    "sparse_pca2": SparsePCA,
    "incremental_pca": IncrementalPCA,
    "gaussian_random_projection": GaussianRandomProjection,
    "agglomerative_clustering": AgglomerativeClustering,
    "bisecting_kmeans": BisectingKMeans,
    "mini_batch_kmeans": MiniBatchKMeans,
    "svc": SVC,
    "one_class_svm": OneClassSVM,
    "stacking_classifier": StackingClassifier,
    "voting_classifier": VotingClassifier,
    "nearest_neighbors": NearestNeighbors,
    "local_outlier_factor": LocalOutlierFactor,
    "spectral_embedding2": SpectralEmbedding,
    "bernoulli_rbm": BernoulliRBM,
    "kernel_density": KernelDensity,
    "poisson_regressor": PoissonRegressor,
    "mlp_regressor": MLPRegressor,
    # === 2026 techniques ===
    "dpo": DPO,
    "contrastive_learning": ContrastiveLearning,
    "grpo": GRPO,
    "count_vectorizer": CountVectorizer,
    "tfidf_vectorizer": TfidfVectorizer,
    "target_encoder": TargetEncoder,
    "hashing_vectorizer": HashingVectorizer,
}

def get_technique(name: str) -> Type[Technique]:
    """Retrieve a technique class by name."""
    if name not in _REGISTRY:
        # Try advanced architectures batch 1 (lazy-loaded)
        try:
            from ..advanced_architectures import ADVANCED_ARCHITECTURES
            if name in ADVANCED_ARCHITECTURES:
                _REGISTRY[name] = ADVANCED_ARCHITECTURES[name]
                return ADVANCED_ARCHITECTURES[name]
        except ImportError:
            pass
        # Try advanced architectures batch 2 (lazy-loaded)
        try:
            from ..advanced_architectures_2 import ADVANCED_ARCHITECTURES_2
            if name in ADVANCED_ARCHITECTURES_2:
                _REGISTRY[name] = ADVANCED_ARCHITECTURES_2[name]
                return ADVANCED_ARCHITECTURES_2[name]
        except ImportError:
            pass
        # Try modern vision models (lazy-loaded)
        try:
            from ..modern_vision import MODERN_VISION
            if name in MODERN_VISION:
                _REGISTRY[name] = MODERN_VISION[name]
                return MODERN_VISION[name]
        except ImportError:
            pass
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
    "Transformer",
    "CNN",
    "RNN",
    "LSTM",
    "GRU",
    "Seq2Seq",
    "GAN",
    "VAE",
    "Diffusion",
    "MixtureOfExperts",
    "GNN",
    "Mamba",
    "Autoencoder",
    "DQN",
    "PPO",
    "SemiSupervised",
    "AssociationRules",
    "SelfSupervised",
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
    # frontier techniques
    "NormalizingFlow", "FlowMatching", "ConsistencyModel", "EnergyBasedModel",
    "ScoreBasedModel", "RestrictedBoltzmannMachine", "BoltzmannMachine",
    "ModernHopfield", "NeuralTuringMachine", "DifferentiableNeuralComputer",
    "SiameseNetwork", "PrototypicalNetwork", "MatchingNetwork", "MAML", "RelationNetwork",
    "SelfOrganizingMap", "CapsuleNetwork", "EchoStateNetwork",
    "ExtremeLearningMachine", "NeuralGas", "ARTNetwork",
    "DDPG", "TD3", "SAC", "A2C", "ValueIteration", "PolicyIteration", "RLHF",
    "S4", "PointerNetwork", "SetTransformer", "HiddenMarkovModel", "CTC",
    "VisionTransformer", "DeiT", "NeRF", "INR", "UNet",
    "CollaborativeFiltering", "MatrixFactorization", "WideAndDeep", "DeepFM",
    "Item2Vec", "GraphSAGE", "GAT", "GIN",
    "BayesianNeuralNetwork", "MCDropout", "WorldModel", "Dreamer",
    "MixtureDensityNetwork", "GroupedQueryAttention", "ARIMA", "KalmanFilter",
    "LDATopicModel", "OnlineLearning",
    "SUPERVISED",
    "UNSUPERVISED",
    "REINFORCEMENT",
    "get_technique",
    "register_technique",
    # Universal training enhancements (NOT new model types; these wrap any
    # torch-based qai technique to add modern training improvements).
    "SAM", "SWA", "EMA", "Lookahead",
    "MixupTrainer", "LabelSmoother",
    "CosineAnnealer", "OneCyclePolicy",
    "TTA",
    "PseudoLabeler", "SnapshotEnsemble",
]
