"""
qai.configs — family-specific configuration objects for qai.build().

Instead of passing 20 kwargs to qai.build(), users can pass a config object
that bundles the architecture-specific parameters for one technique family.

    # Old way (still works):
    m = qai.build(technique="transformer", d_model=256, n_heads=8, n_layers=6)

    # New way (explicit, self-documenting):
    m = qai.build(technique="transformer",
                  config=qai.TransformerConfig(d_model=256, n_heads=8, n_layers=6))

Both are equivalent. Internally, build() unpacks the config into kwargs.

The 15 config classes cover all 218 techniques:

    Tier 1 — Deep learning + modern LLM (5 classes)
      TransformerConfig, CNNConfig, RNNConfig, ModernLLMConfig, MoEConfig

    Tier 2 — Trees + clustering + boosting (3 classes)
      TreeConfig, BoostingConfig, ClusterConfig

    Tier 3 — Specialized families (7 classes)
      GNNConfig, VisionConfig, RLConfig, GenerativeConfig,
      RecommenderConfig, SequenceConfig, ProbabilisticConfig

Plus:
  - TrainConfig — default training recipe (optimizer, LR, schedule, etc.)
  - GenerateConfig — generation-time settings (temperature, top_k, top_p, etc.)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union


# =====================================================================
# Cross-cutting configs (apply to multiple families)
# =====================================================================

@dataclass
class MoEConfig:
    """Mixture-of-Experts configuration.

    Can be attached to any architecture that supports MoE (transformer,
    modern LLM, feedforward). The host architecture dispatches each
    token to the top-k experts, then combines their outputs.
    """
    num_experts: int = 8
    top_k: int = 2
    expert_dim: Optional[int] = None  # default: same as d_model
    router_init_std: float = 0.02
    load_balancing_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    expert_bias: bool = False  # DeepSeek-V3 style expert bias

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    """Default training recipe.

    Can be passed to qai.build() as `train_config=` to set defaults that
    apply to every subsequent .train() call. Individual .train() kwargs
    override these.

    Example:
        m = qai.build(technique="transformer",
                      train_config=qai.TrainConfig(
                          optimizer="adamw", lr=3e-4, lr_schedule="cosine",
                          warmup_steps=200, weight_decay=0.01, max_grad_norm=1.0,
                      ))
        m.train(corpus, epochs=10)  # uses the train_config defaults
    """
    optimizer: str = "adamw"
    lr: float = 3e-4
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant", "step"
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    batch_size: int = 32
    grad_accum_steps: int = 1
    epochs: Optional[int] = None
    max_steps: Optional[int] = None
    seed: int = 42
    eval_strategy: str = "steps"  # "steps", "epoch", "no"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    output_dir: str = "./checkpoints"
    log_every: int = 50
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerateConfig:
    """Generation-time settings for generative models (LLM, GAN, VAE, diffusion).

    Pass to model.generate() to control sampling.

    Example:
        text = m.generate("The meaning of life is",
                          config=qai.GenerateConfig(
                              max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.95,
                          ))
    """
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 0  # 0 = disabled
    top_p: float = 1.0  # 1.0 = disabled
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1  # 1 = no beam search
    early_stopping: bool = False
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True  # KV cache

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# Tier 1 — Deep learning + modern LLM
# =====================================================================

@dataclass
class TransformerConfig:
    """Configuration for transformer-based techniques.

    Used by: transformer, modern_transformer (t5), modern_llm (llama_decoder),
    modern_ssm (rwkv), hybrid (jamba), moe (deepseek_moe).

    Example:
        m = qai.build(technique="transformer",
                      config=qai.TransformerConfig(
                          d_model=256, n_heads=8, n_layers=6,
                          position_encoding="rope", norm="rmsnorm",
                          activation="swiglu", window_size=1024,
                          moe=qai.MoEConfig(num_experts=8, top_k=2),
                      ))
    """
    # Core architecture
    vocab_size: int = 256
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 128
    d_ff: Optional[int] = None  # default: 4 * d_model
    ffn_multiplier: int = 4  # alternative to d_ff: d_ff = ffn_multiplier * d_model
    dropout: float = 0.1
    attention_dropout: float = 0.0
    embedding_dropout: float = 0.0
    residual_dropout: float = 0.0

    # Positional encoding
    position_encoding: str = "rope"  # "rope", "learned", "alibi", "none"
    rope_theta: float = 10000.0
    max_position_embeddings: Optional[int] = None  # for learned

    # Normalization / activation
    norm: str = "layernorm"  # "layernorm", "rmsnorm"
    activation: str = "gelu"  # "gelu", "relu", "swiglu", "geglu"
    norm_eps: float = 1e-5

    # Architecture variants
    use_cache: bool = True  # KV cache
    window_size: int = 0  # SWA: 0 = global attention
    tie_embeddings: bool = False  # weight tying (lm_head shares embedding)
    attention_impl: str = "auto"  # "auto", "sdpa", "flash", "math"

    # MoE (optional)
    moe: Optional[MoEConfig] = None

    # Task
    task: str = "causal_lm"  # "causal_lm", "classification", "masked_lm"
    n_classes: Optional[int] = None

    # Training
    epochs: int = 20
    lr: float = 3e-4
    batch_size: int = 32
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # Device
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Unpack MoE config
        if self.moe is not None:
            for k, v in self.moe.to_dict().items():
                d[f"moe_{k}"] = v
        return d


@dataclass
class CNNConfig:
    """Configuration for CNN-based techniques.

    Used by: cnn, resnet, densenet, mobilenet, efficientnet, convnext,
    unet (image segmentation), capsule.
    """
    # Input shape
    in_channels: int = 1
    img_size: Optional[int] = None  # for ViT-style patch embed
    n_classes: int = 10

    # Conv stack
    conv_channels: List[int] = field(default_factory=lambda: [16, 32, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2])
    strides: List[int] = field(default_factory=lambda: [1, 1, 1])

    # FC head
    fc_hidden: int = 64
    fc_dropout: float = 0.5

    # Modern enhancements
    se_blocks: bool = False  # Squeeze-and-Excitation
    stochastic_depth: float = 0.0  # 0 = disabled, 0.1 = standard
    residual: bool = False  # ResNet-style skip connections
    depthwise_separable: bool = False  # MobileNet style

    # Training
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32
    augment: bool = False

    # Device
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RNNConfig:
    """Configuration for RNN-based techniques.

    Used by: rnn, lstm, gru, seq2seq.
    """
    input_size: int = 4
    hidden_size: int = 32
    n_layers: int = 1
    n_classes: int = 2
    output_size: Optional[int] = None  # for seq2seq
    cell_type: str = "lstm"  # "rnn", "lstm", "gru"
    bidirectional: bool = False
    dropout: float = 0.0

    # Training
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32

    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModernLLMConfig:
    """Configuration for modern LLM architectures (LLaMA, T5, RWKV, DeepSeek MoE, Jamba).

    Extends TransformerConfig with modern LLM-specific defaults:
    RoPE, RMSNorm, SwiGLU, weight tying, KV cache.

    Example:
        m = qai.build(technique="llama_decoder",
                      config=qai.ModernLLMConfig(
                          vocab_size=32000, dim=4096, n_layers=32, n_heads=32,
                      ))
    """
    # LLaMA-style naming (dim instead of d_model)
    vocab_size: int = 256
    dim: int = 64  # = d_model
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: Optional[int] = None  # GQA: None = MHA, n_kv_heads < n_heads = GQA
    dim_ff: Optional[int] = None  # default: 4 * dim
    max_seq_len: int = 256

    # Modern defaults (override TransformerConfig defaults)
    position_encoding: str = "rope"
    rope_theta: float = 10000.0
    norm: str = "rmsnorm"
    activation: str = "swiglu"
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    use_cache: bool = True
    window_size: int = 0  # SWA: 0 = global

    # MoE (for deepseek_moe, jamba)
    moe: Optional[MoEConfig] = None

    # Training
    epochs: int = 5
    lr: float = 3e-4
    batch_size: int = 16
    weight_decay: float = 0.01
    warmup_steps: int = 100

    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.moe is not None:
            for k, v in self.moe.to_dict().items():
                d[f"moe_{k}"] = v
        return d


# =====================================================================
# Tier 2 — Trees + clustering + boosting
# =====================================================================

@dataclass
class TreeConfig:
    """Configuration for tree-based techniques.

    Used by: decision_tree, decision_tree_regressor, random_forest,
    random_forest_regressor, extra_trees.
    """
    n_estimators: int = 100  # only for forests
    criterion: str = "gini"  # "gini", "entropy", "log_loss", "mse", "mae"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[Union[str, int, float]] = "sqrt"
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    ccp_alpha: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoostingConfig:
    """Configuration for gradient boosting techniques.

    Used by: gradient_boosting, adaboost, hist_gradient_boosting, xgboost,
    lightgbm, catboost.
    """
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: Optional[int] = 3
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    subsample: float = 1.0
    max_features: Optional[Union[str, int, float]] = None
    loss: str = "deviance"  # "deviance", "exponential" (AdaBoost: "linear", "square", "exponential")
    criterion: str = "friedman_mse"
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0
    # XGBoost / LightGBM / CatBoost specific
    objective: Optional[str] = None  # "binary:logistic", "reg:squarederror", etc.
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterConfig:
    """Configuration for clustering techniques.

    Used by: kmeans, gmm, birch, bisecting_kmeans, mini_batch_kmeans,
    mean_shift, spectral_clustering, affinity_propagation, optics,
    agglomerative, agglomerative_clustering, dbscan.
    """
    n_clusters: int = 3  # not all use this (DBSCAN, OPTICS, MeanShift infer)
    n_components: Optional[int] = None  # GMM alias
    init: str = "kmeans"  # "kmeans", "kmeans++", "random", "random_from_data"
    n_init: int = 1  # number of restarts (sklearn 1.4+ default 1, older 10)
    max_iter: int = 300
    tol: float = 1e-4
    # GMM-specific
    covariance_type: str = "full"  # "full", "tied", "diag", "spherical"
    variational_bayes: bool = False  # GMM only
    weight_concentration_prior: float = 1e-3
    # DBSCAN / OPTICS
    eps: float = 0.5
    min_samples: int = 5
    # Agglomerative
    linkage: str = "ward"  # "ward", "complete", "average", "single"
    # Spectral
    affinity: str = "rbf"  # "rbf", "nearest_neighbors", "precomputed"
    n_neighbors: int = 10
    # General
    random_state: Optional[int] = None
    n_jobs: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# Tier 3 — Specialized families
# =====================================================================

@dataclass
class GNNConfig:
    """Configuration for graph neural networks.

    Used by: gcn, gat, gin, graphsage, mpnn, gnn.
    """
    in_dim: int = 4
    hidden: int = 16
    n_classes: int = 2
    n_layers: int = 2
    n_heads: int = 4  # GAT only
    dropout: float = 0.5
    aggregation: str = "mean"  # "mean", "max", "sum" (for GraphSAGE, GIN)
    message_passing_type: str = "gcn"  # "gcn", "gat", "gin", "sage", "mpnn"
    # Training
    epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 5e-4
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisionConfig:
    """Configuration for vision techniques (ViT, DeiT, ResNet, etc.).

    Used by: vit, deit, resnet, mobilenet, efficientnet, densenet, unet,
    capsule, convnext, nerf, inr.
    """
    # Image shape
    img_size: int = 28
    patch_size: int = 4  # ViT only
    in_ch: int = 1  # input channels
    n_classes: int = 10

    # ViT-specific
    emb_dim: int = 64  # ViT embedding dim
    depth: int = 4  # ViT number of blocks
    n_heads: int = 4  # ViT attention heads
    mlp_dim: int = 128  # ViT MLP inner dim

    # ResNet / DenseNet / MobileNet / EfficientNet
    in_channels: int = 1
    base_channels: int = 16
    n_blocks: int = 3  # ResNet
    growth_rate: int = 32  # DenseNet
    width_multiplier: float = 1.0  # MobileNet

    # UNet
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])

    # Training
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32
    augment: bool = False
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RLConfig:
    """Configuration for reinforcement learning agents.

    Used by: dqn, ppo, a2c, sac, ddpg, td3, sarsa, tabular_policy.
    """
    n_actions: int = 4
    n_states: int = 16
    obs_dim: Optional[int] = None  # for continuous obs
    action_dim: Optional[int] = None  # for continuous actions (DDPG, SAC, TD3, PPO)
    hidden: int = 64
    n_layers: int = 2

    # Q-learning family (DQN)
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 100
    replay_buffer_size: int = 10000

    # Policy gradient family (PPO, A2C)
    clip_epsilon: float = 0.2  # PPO
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95  # PPO GAE

    # SAC / TD3 specific
    alpha: float = 0.2  # SAC entropy temperature
    tau: float = 0.005  # soft target update
    policy_noise: float = 0.2  # TD3 target policy smoothing
    noise_clip: float = 0.5  # TD3

    # Training
    n_episodes: int = 1000
    max_steps: int = 1000
    batch_size: int = 64
    lr: float = 1e-3
    lr_actor: Optional[float] = None
    lr_critic: Optional[float] = None
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerativeConfig:
    """Configuration for generative models.

    Used by: gan, vae, autoencoder, diffusion, normalizing_flow,
    flow_matching, score_based, consistency_model, energy_based,
    rbm, bernoulli_rbm, boltzmann, dbn.
    """
    # Data shape
    data_dim: int = 4  # input dimension
    input_dim: Optional[int] = None  # alias
    hidden: int = 32
    latent_dim: int = 8
    n_flows: int = 4  # NormalizingFlow, FlowMatching
    n_steps: int = 100  # Diffusion timesteps
    beta_start: float = 1e-4  # Diffusion
    beta_end: float = 0.02  # Diffusion

    # VAE
    beta: float = 1.0  # β-VAE
    kl_weight: float = 1.0

    # GAN
    d_steps: int = 1  # discriminator steps per generator step
    gen_steps: int = 1

    # Training
    epochs: int = 50
    lr: float = 1e-3
    lr_disc: Optional[float] = None  # GAN
    batch_size: int = 32
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommenderConfig:
    """Configuration for recommender systems.

    Used by: matrix_factorization, collab_filtering, wide_deep, deepfm,
    item2vec.
    """
    n_users: int = 100
    n_items: int = 100
    k: int = 16  # embedding dim / latent factor count
    n_features: int = 8  # Wide & Deep, DeepFM
    hidden: int = 32
    n_layers: int = 2
    n_cross: int = 4  # DeepFM cross feature width

    # Training
    n_iter: int = 100  # matrix_factorization, collab_filtering
    epochs: int = 10  # deep models
    lr: float = 1e-3
    reg: float = 0.01  # regularization
    batch_size: int = 32
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceConfig:
    """Configuration for sequence models (SSM, Mamba2, Pointer Net, CTC, TTT,
    DNC, NTM).

    Used by: s4, mamba2, pointer_net, ctc, ttt, dnc, ntm.
    """
    # SSM / Mamba
    in_dim: int = 4
    out_dim: int = 2
    d_model: int = 16  # Mamba
    d_ff: int = 32  # Mamba inner
    state_dim: int = 16  # S4, Mamba
    n_layers: int = 1
    n_heads: int = 4  # TTT

    # DNC / NTM
    ctrl_dim: int = 16  # controller hidden
    mem_n: int = 8  # memory slots
    mem_dim: int = 4  # memory vector dim
    n_heads: Optional[int] = None  # DNC read heads

    # Pointer Net
    hidden: int = 16

    # CTC
    input_dim: int = 4
    n_classes: int = 3

    # Training
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProbabilisticConfig:
    """Configuration for probabilistic models.

    Used by: hmm, bayesian_nn, mc_dropout, mixture_density, topic_model.
    """
    # HMM
    n_states: int = 3
    n_obs: int = 4
    n_iter: int = 10

    # Bayesian NN
    in_dim: int = 4
    out_dim: int = 2
    hidden: int = 16
    n_samples: int = 3  # MC dropout samples

    # Mixture Density Network
    n_components: int = 3

    # Topic Model (LDA)
    n_topics: int = 5
    alpha: float = 0.1
    beta: float = 0.01

    # Training
    epochs: int = 30
    lr: float = 1e-3
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# Mapping: technique name → recommended config class
# =====================================================================

TECHNIQUE_CONFIG_MAP: Dict[str, type] = {
    # Tier 1
    "transformer": TransformerConfig,
    "t5": ModernLLMConfig,
    "llama_decoder": ModernLLMConfig,
    "rwkv": ModernLLMConfig,
    "deepseek_moe": ModernLLMConfig,
    "jamba": ModernLLMConfig,
    "mamba": ModernLLMConfig,
    "mamba2": ModernLLMConfig,
    "cnn": CNNConfig,
    "resnet": VisionConfig,
    "vit": VisionConfig,
    "deit": VisionConfig,
    "densenet": VisionConfig,
    "mobilenet": VisionConfig,
    "efficientnet": VisionConfig,
    "unet": VisionConfig,
    "capsule": VisionConfig,
    "nerf": VisionConfig,
    "inr": VisionConfig,
    "rnn": RNNConfig,
    "lstm": RNNConfig,
    "gru": RNNConfig,
    "seq2seq": RNNConfig,

    # Tier 2
    "decision_tree": TreeConfig,
    "decision_tree_regressor": TreeConfig,
    "random_forest": TreeConfig,
    "random_forest_regressor": TreeConfig,
    "extra_trees": TreeConfig,
    "gradient_boosting": BoostingConfig,
    "adaboost": BoostingConfig,
    "hist_gradient_boosting": BoostingConfig,
    "xgboost": BoostingConfig,
    "lightgbm": BoostingConfig,
    "catboost": BoostingConfig,
    "kmeans": ClusterConfig,
    "gmm": ClusterConfig,
    "birch": ClusterConfig,
    "bisecting_kmeans": ClusterConfig,
    "mini_batch_kmeans": ClusterConfig,
    "minibatch_kmeans": ClusterConfig,
    "mean_shift": ClusterConfig,
    "spectral_clustering": ClusterConfig,
    "affinity_propagation": ClusterConfig,
    "optics": ClusterConfig,
    "agglomerative": ClusterConfig,
    "agglomerative_clustering": ClusterConfig,
    "dbscan": ClusterConfig,

    # Tier 3
    "gcn": GNNConfig,
    "gat": GNNConfig,
    "gin": GNNConfig,
    "graphsage": GNNConfig,
    "mpnn": GNNConfig,
    "gnn": GNNConfig,
    "dqn": RLConfig,
    "ppo": RLConfig,
    "a2c": RLConfig,
    "sac": RLConfig,
    "ddpg": RLConfig,
    "td3": RLConfig,
    "sarsa": RLConfig,
    "tabular_policy": RLConfig,
    "gan": GenerativeConfig,
    "vae": GenerativeConfig,
    "autoencoder": GenerativeConfig,
    "diffusion": GenerativeConfig,
    "normalizing_flow": GenerativeConfig,
    "flow_matching": GenerativeConfig,
    "score_based": GenerativeConfig,
    "consistency_model": GenerativeConfig,
    "energy_based": GenerativeConfig,
    "rbm": GenerativeConfig,
    "bernoulli_rbm": GenerativeConfig,
    "boltzmann": GenerativeConfig,
    "dbn": GenerativeConfig,
    "matrix_factorization": RecommenderConfig,
    "collab_filtering": RecommenderConfig,
    "wide_deep": RecommenderConfig,
    "deepfm": RecommenderConfig,
    "item2vec": RecommenderConfig,
    "s4": SequenceConfig,
    "mamba2": SequenceConfig,
    "pointer_net": SequenceConfig,
    "ctc": SequenceConfig,
    "ttt": SequenceConfig,
    "dnc": SequenceConfig,
    "ntm": SequenceConfig,
    "hmm": ProbabilisticConfig,
    "bayesian_nn": ProbabilisticConfig,
    "mc_dropout": ProbabilisticConfig,
    "mixture_density": ProbabilisticConfig,
    "topic_model": ProbabilisticConfig,
}


def get_config_class(technique_name: str) -> Optional[type]:
    """Get the recommended config class for a technique, or None if the
    technique uses generic kwargs (linear models, preprocessors, etc.).
    """
    return TECHNIQUE_CONFIG_MAP.get(technique_name)


def config_to_kwargs(config) -> Dict[str, Any]:
    """Convert a config object to a kwargs dict suitable for qai.build().

    The config's fields are unpacked into top-level kwargs. Nested configs
    (like MoEConfig) are flattened with a prefix.
    """
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        d = config.to_dict()
    elif hasattr(config, "__dict__"):
        d = asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config).copy()
    else:
        d = dict(config)
    # Remove None values (don't override technique defaults with None)
    return {k: v for k, v in d.items() if v is not None}


__all__ = [
    # Cross-cutting
    "MoEConfig", "TrainConfig", "GenerateConfig",
    # Tier 1
    "TransformerConfig", "CNNConfig", "RNNConfig", "ModernLLMConfig",
    # Tier 2
    "TreeConfig", "BoostingConfig", "ClusterConfig",
    # Tier 3
    "GNNConfig", "VisionConfig", "RLConfig", "GenerativeConfig",
    "RecommenderConfig", "SequenceConfig", "ProbabilisticConfig",
    # Helpers
    "TECHNIQUE_CONFIG_MAP", "get_config_class", "config_to_kwargs",
]
