# Guide: Model Catalog — Learn One Technique at a Time

*Every buildable technique in qai, organized by family. Each family answers: what is it, when do you use it, and how do you build it.*

> Back to [Guides index](README.md) · Build/train pattern: [TRAINING.md](TRAINING.md) · Full parameter reference: [User Manual](../USER_MANUAL.md)

---

## How to use this chapter

Every technique builds with the same call:

```python
model = qai.build(technique="<name>")
```

**229 techniques** ship in the registry. Find your family below, read the
"when to use", pick a name, build it. If two techniques look similar, the
"pick between them" notes tell you the real difference.

---

## Family map

| Family | Count | Techniques |
|---|---|---|
| [Classical — Supervised](#1-classical-supervised) | ~30 | regression, classifier, decision_tree, knn, svm, ... |
| [Classical — Ensembles & Boosting](#2-ensembles--boosting) | ~10 | random_forest, gradient_boosting, xgboost, ... |
| [Classical — Unsupervised](#3-unsupervised-clustering--decomposition) | ~25 | kmeans, pca, dbscan, gmm, isolation_forest, ... |
| [Deep Learning — Core](#4-deep-learning-core) | ~12 | neural_network, transformer, cnn, rnn, lstm, gru, seq2seq |
| [Deep Learning — Generative](#5-generative-models) | ~15 | gan, vae, diffusion, normalizing_flow, score_based, ... |
| [Deep Learning — Modern Architectures](#6-modern--2026-architectures) | ~15 | mamba, mamba2, moe, deepseek_moe, rwkv, s4, ttt, jamba, ... |
| [Vision](#7-vision) | ~12 | vit, deit, unet, resnet, densenet, efficientnet, mobilenet, clip, ... |
| [Reinforcement Learning](#8-reinforcement-learning) | ~15 | dqn, ppo, sac, td3, ddpg, a2c, mcts, muzero, ... |
| [Sequence & Memory](#9-sequence--memory-models) | ~10 | hmm, ctc, pointer_net, set_transformer, ntm, dnc, hopfield, ... |
| [Graphs](#10-graph-neural-networks) | ~5 | gnn, gcn, graphsage, gat, gin |
| [Recommenders](#11-recommenders) | ~5 | collab_filtering, matrix_factorization, wide_deep, deepfm, item2vec |
| [Probabilistic](#12-probabilistic--bayesian) | ~10 | gaussian_process, bayesian_nn, mc_dropout, mixture_density, kde, ... |
| [Meta-Learning & Few-Shot](#13-meta-learning--few-shot) | ~5 | maml, prototypical, matching, relation, siamese |
| [Self-Organizing & Exotic](#14-self-organizing--exotic) | ~10 | som, capsule, echo_state, elm, neural_gas, art, spiking_nn, ... |
| [Time Series](#15-time-series) | ~3 | time_series, time_series_arima, kalman_filter |
| [Preprocessing "Models"](#16-preprocessing-as-techniques) | ~30 | standard_scaler, one_hot_encoder, select_k_best, ... |

---

## 1. Classical — Supervised

**What:** Predict a label (classification) or number (regression) from tabular features.
**When:** Your data is a table, rows are samples, columns are features. This is 80% of real-world ML.

### Classification

```python
m = qai.build(technique="classifier")        # logistic regression
m = qai.build(technique="decision_tree")     # interpretable rules
m = qai.build(technique="knn")               # nearest neighbors
m = qai.build(technique="svm")               # max-margin
m = qai.build(technique="naive_bayes")       # fast text baseline
m = qai.build(technique="multinomial_naive_bayes")  # count features (requires X >= 0)
m = qai.build(technique="bernoulli_naive_bayes")    # binary features
m = qai.build(technique="lda")               # linear discriminant analysis
m = qai.build(technique="qda")               # quadratic (nonlinear boundaries)
m = qai.build(technique="linear_svc")        # linear SVM (liblinear-style)
m = qai.build(technique="nu_svc")            # nu-parameterized SVM
m = qai.build(technique="sgd_classifier")    # online/streaming classification
m = qai.build(technique="passive_aggressive_classifier")  # online learning
m = qai.build(technique="nearest_centroid")  # simplest possible classifier
m = qai.build(technique="radius_neighbors_classifier")    # KNN with radius
m = qai.build(technique="perceptron")        # the original 1958 architecture
```

**Pick between them:**
- Need to *explain* predictions → `decision_tree` (rules) or `classifier` (coefficients)
- Text with counts → `multinomial_naive_bayes` — fast and shockingly strong
- Small data, clean boundaries → `svm`
- Streaming/infinite data → `sgd_classifier` or `passive_aggressive_classifier`

### Regression

```python
m = qai.build(technique="regression")          # linear + formula() + r_squared()
m = qai.build(technique="ridge")               # L2 regularized
m = qai.build(technique="lasso")               # L1 — zeroes out weak features
m = qai.build(technique="elastic_net")         # L1 + L2 combined
m = qai.build(technique="kernel_ridge")        # nonlinear via kernels
m = qai.build(technique="bayesian_ridge")      # uncertainty estimates
m = qai.build(technique="ard_regression")      # automatic relevance determination
m = qai.build(technique="huber")               # robust to outliers
m = qai.build(technique="ransac")              # robust fitting with outlier rejection
m = qai.build(technique="theil_sen")           # median-based robust regression
m = qai.build(technique="quantile_regression") # predict the 90th percentile, not the mean
m = qai.build(technique="decision_tree_regressor")
m = qai.build(technique="svr")                 # support vector regression
m = qai.build(technique="poisson_regressor")   # count targets
m = qai.build(technique="isotonic_regression") # monotonic, non-parametric
m = qai.build(technique="mlp_regressor")       # small neural net for regression
m = qai.build(technique="dummy_regressor")     # the baseline you must beat
```

**Pick between them:**
- Outliers in your targets → `huber`, `ransac`, or `theil_sen`
- Need prediction intervals → `bayesian_ridge` or [gaussian_process](#12-probabilistic--bayesian)
- Need a specific percentile → `quantile_regression`
- Want feature selection built-in → `lasso` or `ard_regression`

Special introspection on the classics (unique to qai's unified interface):

```python
m = qai.build(technique="regression"); m.train(X, y)
print(m.formula())        # y = 2.000*x0 + 0.000
print(m.r_squared())      # 0.94
print(m.residuals())      # array of errors

m = qai.build(technique="decision_tree"); m.train(X, y)
print(m.feature_importance())   # [0.6, 0.3, 0.1]
print(m.tree_depth())           # 4

m = qai.build(technique="svm"); m.train(X, y)
print(m.support_vectors())      # the critical training points
print(m.margin_width())         # how wide the margin is
```

---

## 2. Ensembles & Boosting

**What:** Combine many weak models into one strong one.
**When:** Tabular data, you want the strongest classical accuracy, interpretability is secondary.

```python
m = qai.build(technique="random_forest")           # bagged trees
m = qai.build(technique="random_forest_regressor")
m = qai.build(technique="gradient_boosting")       # sequential boosting
m = qai.build(technique="gradient_boosting_regressor")
m = qai.build(technique="xgboost")                 # modern boosting
m = qai.build(technique="lightgbm")                # fast, histogram-based
m = qai.build(technique="catboost")                # categorical features native
m = qai.build(technique="hist_gradient_boosting")  # sklearn-style fast boosting
m = qai.build(technique="adaboost")                # classic adaptive boosting
m = qai.build(technique="adaboost_regressor")
m = qai.build(technique="extra_trees")             # more random splits, less variance
m = qai.build(technique="bagging_classifier")      # bag any base estimator
m = qai.build(technique="stacking_classifier")     # stack models via meta-learner
m = qai.build(technique="voting_classifier")       # vote across models
```

**Modern upgrades built into the trees** (pass as kwargs, all default OFF for
backwards compatibility — see [FINDINGS.md §36](../FINDINGS.md)):

```python
m = qai.build(technique="random_forest",
    extra_trees_co=True,          # co-trained extra-trees members
    calibrated=True,              # probability calibration
    calibration_method="isotonic",
    stacking_base=True)           # usable as stacking base

m = qai.build(technique="gradient_boosting",
    early_stopping_rounds=20,
    validation_fraction=0.1,
    snapshot_ensemble=True, n_snapshots=5,
    calibrated=True)
```

**Pick between them:**
- Categorical features, no encoding wanted → `catboost`
- Very large tabular data → `lightgbm` or `hist_gradient_boosting`
- Need calibrated probabilities → pass `calibrated=True` to any forest/booster

---

## 3. Unsupervised — Clustering & Decomposition

**What:** Find structure without labels.
**When:** Exploration, customer segments, anomaly detection, dimensionality reduction.

### Clustering

```python
m = qai.build(technique="kmeans")            # centroid-based, need k
m = qai.build(technique="minibatch_kmeans")  # streaming/large-scale kmeans
m = qai.build(technique="dbscan")            # density-based, finds k itself, noise-tolerant
m = qai.build(technique="optics")            # dbscan with reachability ordering
m = qai.build(technique="birch")             # hierarchical pre-clustering for big data
m = qai.build(technique="mean_shift")        # mode-seeking, no k needed
m = qai.build(technique="affinity_propagation")  # exemplar-based
m = qai.build(technique="gmm")               # soft/probabilistic clusters
m = qai.build(technique="hierarchical")      # dendrogram
m = qai.build(technique="agglomerative")     # bottom-up hierarchical
m = qai.build(technique="bisecting_kmeans")  # divisive hierarchical
m = qai.build(technique="spectral_clustering")  # graph-structure clusters
```

**Pick between them:**
- Don't know k → `dbscan`, `mean_shift`, or `affinity_propagation`
- Want probabilities per cluster → `gmm` (`predict_proba()`, `score_samples()`)
- Huge dataset → `minibatch_kmeans` or `birch`
- Non-spherical clusters → `dbscan` or `spectral_clustering`

### Decomposition & Manifolds

```python
m = qai.build(technique="pca")               # linear compression
m = qai.build(technique="kernel_pca")        # nonlinear PCA
m = qai.build(technique="sparse_pca")        # interpretable components
m = qai.build(technique="incremental_pca")   # streaming PCA
m = qai.build(technique="truncated_svd")     # for sparse matrices (text!)
m = qai.build(technique="nmf")               # non-negative parts-based
m = qai.build(technique="factor_analysis")   # latent factors
m = qai.build(technique="ica/fast_ica")      # independent components
m = qai.build(technique="tsne")              # 2D visualization embedding
m = qai.build(technique="isomap")            # geodesic manifold
m = qai.build(technique="lle")               # locally linear embedding
m = qai.build(technique="mds")               # multidimensional scaling
m = qai.build(technique="spectral_embedding")
m = qai.build(technique="gaussian_random_projection")  # fast random projection
```

Note: PCA's `forward()` *transforms* input to lower dimensions rather than
predicting a label — a genuinely different output shape, verified against
sklearn (see [PROGRESS.md](../PROGRESS.md)).

### Anomaly detection

```python
m = qai.build(technique="isolation_forest"); m.train(X)
m.is_anomaly(x)        # boolean flag
m.anomaly_score(x)     # continuous risk score
m = qai.build(technique="local_outlier_factor")   # density-based LOF
m = qai.build(technique="one_class_svm")          # boundary-based
```

---

## 4. Deep Learning — Core

**What:** Neural networks. PyTorch-backed, real backprop, real attention.
**When:** Tabular nets, language, images, anything nonlinear at scale.

```python
m = qai.build(technique="neural_network",
    modern=True,                  # modern upgrades: BatchNorm, GELU, residuals...
    n_layers=4, dropout=0.1,
    weight_decay=0.01, mixup_alpha=0.2,
    use_sam=True, use_swa=True, cosine_schedule=True)

m = qai.build(technique="transformer",
    vocab_size=1000, d_model=256, n_heads=8, n_layers=6,
    attention_dropout=0.1, residual_dropout=0.1,
    rope_theta=10000.0, layer_norm_eps=1e-5,
    tie_word_embeddings=True,
    norm_type="rmsnorm", activation="gelu",
    window_size=1024,             # sliding-window attention (Gemma-style hybrid)
    attention_impl="auto")        # SDPA/FlashAttention when available

m = qai.build(technique="cnn",
    in_channels=3, n_classes=10,
    conv_channels=[32, 64, 128], kernel_sizes=[3, 3, 3],
    norm_type="group_norm", activation="silu",
    groups=1, dilation=1,
    se_blocks=True,               # squeeze-and-excitation
    stochastic_depth=0.1)

m = qai.build(technique="rnn")    # vanilla recurrent
m = qai.build(technique="lstm")   # gated memory cells
m = qai.build(technique="gru")    # lighter gates
m = qai.build(technique="seq2seq")  # encoder-decoder (translation, summarization)
```

All recurrent models support `bidirectional=True`, multi-layer stacking,
classification + generation tasks, teacher forcing, and hidden-state caching.

**Transformer reality check:** the transformer here is a real multi-head
attention implementation — RoPE positions, KV cache, causal masking, weight
tied LM head, AdamW + cosine LR + warmup + grad clipping, greedy/temperature/
top-k/top-p sampling. On a Python code corpus it reaches **74.17% next-char
accuracy** vs 59.43% for a 4-gram baseline (measured — see
[UNBUILT_FEATURES.md](../UNBUILT_FEATURES.md)).

---

## 5. Generative Models

**What:** Models that create new samples — images, audio features, molecules.
**When:** Synthesis, denoising, latent-space editing.

```python
m = qai.build(technique="gan")        # generator + discriminator
m = qai.build(technique="vae")        # encoder → latent → decoder
m = qai.build(technique="diffusion")  # DDPM
m = qai.build(technique="normalizing_flow")
m = qai.build(technique="flow_matching")
m = qai.build(technique="consistency_model")   # few-step diffusion
m = qai.build(technique="score_based")
m = qai.build(technique="energy_based")
m = qai.build(technique="rbm")
m = qai.build(technique="boltzmann")
m = qai.build(technique="autoencoder")         # basic/sparse/denoising/convolutional
m = qai.build(technique="mixture_density")     # predict distributions, not points
```

Built-in generative extras:
- GAN: mode-collapse detection, diversity metric, D-confidence tracking, two-timescale updates, EMA of generator, latent interpolation
- VAE: `encode()` / `decode()` API, latent arithmetic (king - man + woman ≈ queen)
- Diffusion: linear + cosine noise schedules, classifier-free guidance

---

## 6. Modern & 2026 Architectures

**What:** The state-of-the-art line-up — linear-attention transformers, mixture-of-experts, test-time-trained nets.
**When:** Long contexts, expert routing, staying current.

```python
m = qai.build(technique="mamba")        # selective SSM — O(n) scaling vs O(n²)
m = qai.build(technique="mamba2")       # 2026 revision
m = qai.build(technique="moe")          # mixture of experts (top-k routing)
m = qai.build(technique="deepseek_moe") # fine-grained expert segmentation
m = qai.build(technique="rwkv")         # RNN-transformer hybrid
m = qai.build(technique="s4")           # structured state space
m = qai.build(technique="ttt")          # test-time training
m = qai.build(technique="jamba")        # mamba + attention hybrid
m = qai.build(technique="llama_decoder")# llama-style decoder stack
m = qai.build(technique="t5")           # encoder-decoder, text-to-text
m = qai.build(technique="gqa_attention")    # grouped-query attention
m = qai.build(technique="mqa_attention")    # multi-query attention
m = qai.build(technique="sparse_attention")
m = qai.build(technique="linear_attention")
m = qai.build(technique="world_model")
m = qai.build(technique="dreamer")      # model-based RL
```

**Mamba hybrid mode:** pass `hybrid=True` (3:1 Mamba:Attention layers, the
Qwen3-Next / Kimi-Linear ratio) — see [CHANGELOG v1.x](../../CHANGELOG.md).

**MoE specifics:** top-k routing, load-balancing loss, router z-loss, expert
utilization tracking. Note that routing collapse is detectable in the visual
workspace (see [VISUAL_WORKSPACE.md](VISUAL_WORKSPACE.md)).

---

## 7. Vision

```python
m = qai.build(technique="vit")           # vision transformer
m = qai.build(technique="deit")          # distilled ViT
m = qai.build(technique="unet")          # segmentation / diffusion backbone
m = qai.build(technique="resnet")
m = qai.build(technique="densenet")
m = qai.build(technique="efficientnet")
m = qai.build(technique="mobilenet")     # edge deployment
m = qai.build(technique="clip")          # image-text contrastive
m = qai.build(technique="whisper")       # speech → text
m = qai.build(technique="wav2vec2")      # speech representations
m = qai.build(technique="hubert")        # masked speech prediction
m = qai.build(technique="nerf")          # neural radiance fields
m = qai.build(technique="inr")           # implicit neural representations
```

Plus five modern detection/segmentation models available through
`qai.modern_vision`: SAM2, Florence2, GroundingDINO, YOLOWorld, RFDETR.

CNN-specific powers: any input size (AdaptiveAvgPool2d), auto uint8
normalization, augmentation (flip/rotate/crop), `extract_features(x, layer)`
for transfer learning, `freeze_conv_base()` / `replace_classifier_head(n)`.

---

## 8. Reinforcement Learning

```python
m = qai.build(technique="dqn", double_dqn=True, use_per=True, soft_update=True)
m = qai.build(technique="ppo")
m = qai.build(technique="sac")           # soft actor-critic
m = qai.build(technique="td3")
m = qai.build(technique="ddpg")
m = qai.build(technique="a2c")
m = qai.build(technique="sarsa")
m = qai.build(technique="value_iteration")   # tabular, exact
m = qai.build(technique="policy_iteration")
m = qai.build(technique="mcts")           # Monte-Carlo tree search
m = qai.build(technique="muzero")         # learned model + planning
m = qai.build(technique="tabular_policy") # gridworld classic
m = qai.build(technique="rlhf")           # RL from human feedback
```

DQN ships with: experience replay, target networks, priority replay (PER),
epsilon decay schedules (exponential/cosine), n-step TD, C51 + QR-DQN
distributional heads.

Environment: `qai.core.environment.GridWorld` — see [TRAINING.md §7](TRAINING.md).

---

## 9. Sequence & Memory Models

```python
m = qai.build(technique="hmm")             # hidden Markov model
m = qai.build(technique="ctc")             # connectionist temporal classification
m = qai.build(technique="pointer_net")     # point at inputs (sorting, combinatorial)
m = qai.build(technique="set_transformer") # permutation-invariant sets
m = qai.build(technique="ntm")             # neural turing machine
m = qai.build(technique="dnc")             # differentiable neural computer
m = qai.build(technique="hopfield")        # associative memory
```

For *runtime* memory around any model (working/retrieval/weights tiers), that's
[qai.memory](MEMORY_RAG.md) — different thing, also built.

---

## 10. Graph Neural Networks

```python
m = qai.build(technique="gnn")          # generic message passing
m = qai.build(technique="gcn")          # graph convolutional network
m = qai.build(technique="graphsage")    # sampling-based, inductive
m = qai.build(technique="gat")          # graph attention
m = qai.build(technique="gin")          # graph isomorphism network
```

Supported tasks: node classification, graph classification, link prediction.
Edge features, graph batching, readout pooling (sum/mean/max) all supported.

---

## 11. Recommenders

```python
m = qai.build(technique="collab_filtering")
m = qai.build(technique="matrix_factorization")
m = qai.build(technique="wide_deep")
m = qai.build(technique="deepfm")
m = qai.build(technique="item2vec")
```

---

## 12. Probabilistic & Bayesian

**What:** Models that say "I don't know" with numbers.
**When:** Risk, medicine, finance, active learning.

```python
m = qai.build(technique="gaussian_process")   # uncertainty per prediction
m = qai.build(technique="bayesian_nn")        # distribution over weights
m = qai.build(technique="mc_dropout")         # cheap uncertainty via dropout at test time
m = qai.build(technique="kde")                # kernel density estimation
m = qai.build(technique="covariance")         # covariance estimation
m = qai.build(technique="probability_calibration")  # fix overconfident models
m = qai.build(technique="kalman_filter")      # state estimation over time
```

Calibration metrics: `qai.brier_score`, `qai.expected_calibration_error` —
see [EVALUATION.md](EVALUATION.md).

---

## 13. Meta-Learning & Few-Shot

```python
m = qai.build(technique="maml")            # model-agnostic meta-learning
m = qai.build(technique="prototypical")    # prototype networks
m = qai.build(technique="matching")        # matching networks
m = qai.build(technique="relation")        # relation networks
m = qai.build(technique="siamese")         # twin networks, similarity
```

---

## 14. Self-Organizing & Exotic

```python
m = qai.build(technique="som")           # self-organizing map
m = qai.build(technique="capsule")       # capsule networks
m = qai.build(technique="echo_state")    # reservoir computing
m = qai.build(technique="elm")           # extreme learning machine
m = qai.build(technique="neural_gas")    # growing neural gas
m = qai.build(technique="art")           # adaptive resonance theory
m = qai.build(technique="spiking_nn")    # spiking neural network
m = qai.build(technique="dbn")           # deep belief network
m = qai.build(technique="rbf_network")   # radial basis function network
m = qai.build(technique="online_learning")  # continuously adapting model
```

---

## 15. Time Series

```python
m = qai.build(technique="time_series")           # autoregressive forecaster
m = qai.build(technique="time_series_arima")     # classical ARIMA
m = qai.build(technique="kalman_filter")         # state-space

m.train(series)
m.forecast(steps=30)    # iterative multi-step forecasting via lag window
```

Feature engineering: `qai.data.rolling_window(series, window=7)`,
`qai.data.lag_feature(series, lag=1)`. Decomposition and stats live in
[qai.time_series](MULTIMODAL.md).

---

## 16. Preprocessing "as techniques"

Scalers, encoders, imputers, and feature selectors are registered in the same
registry so pipelines can treat them uniformly:

```python
standard_scaler, minmax_scaler, robust_scaler, maxabs_scaler, normalizer
binarizer, polynomial_features, one_hot_encoder, label_encoder
simple_imputer, knn_imputer, kbins_discretizer, target_encoder
select_k_best, variance_threshold, rfe
kernel_pca, sparse_pca, incremental_pca, gaussian_random_projection
count_vectorizer, tfidf_vectorizer, hashing_vectorizer
```

Prefer the friendly wrappers in `qai.preprocessing` for direct use — see
[DATA.md §5](DATA.md).

---

## Registering your own

```python
import qai
from qai.techniques import Technique, register_technique

@register_technique("my_model", family="custom")
class MyModel(Technique):
    name = "my_model"
    family = "custom"

    def fit(self, X, y):
        self._trained = True
        return self

    def forward(self, x):
        return prediction

m = qai.build(technique="my_model")   # now in the registry
```

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the test policy for new techniques.

---

## The full alphabetical registry (229)

<details>
<summary>Click to expand the complete list</summary>

classification family: classifier, decision_tree, knn, svm, naive_bayes,
multinomial_naive_bayes, bernoulli_naive_bayes, complement_naive_bayes,
gaussian_nb, lda, qda, sgd_classifier, passive_aggressive_classifier,
linear_svc, nu_svc, svc, one_class_svm, radius_neighbors_classifier,
nearest_centroid, perceptron, stacking_classifier, voting_classifier,
nearest_neighbors

regression family: regression, ridge, lasso, elastic_net, kernel_ridge,
bayesian_ridge, ard_regression, huber, ransac, theil_sen, quantile_regression,
decision_tree_regressor, random_forest_regressor, adaboost_regressor,
gradient_boosting_regressor, sgd_regressor, svr, poisson_regressor,
mlp_regressor, dummy_regressor, isotonic_regression, pls_canonical, cca,
knn_regressor, kneighbors_regressor

ensembles: random_forest, gradient_boosting, xgboost, lightgbm, catboost,
hist_gradient_boosting, adaboost, adaboost_regressor, extra_trees,
bagging_classifier, stacking_classifier, voting_classifier

clustering: kmeans, minibatch_kmeans, mini_batch_kmeans, dbscan, optics,
birch, mean_shift, affinity_propagation, gmm, gaussian_mixture_model,
hierarchical, agglomerative, agglomerative_clustering, bisecting_kmeans,
spectral_clustering

decomposition: pca, kernel_pca, kernel_pca2, sparse_pca, sparse_pca2,
incremental_pca, truncated_svd, nmf, factor_analysis, fast_ica, ica, tsne,
isomap, lle, mds, spectral_embedding, spectral_embedding2,
gaussian_random_projection

anomaly: isolation_forest, local_outlier_factor, one_class_svm

deep core: neural_network, transformer, cnn, rnn, lstm, gru, seq2seq

generative: gan, vae, diffusion, normalizing_flow, flow_matching,
consistency_model, energy_based, score_based, rbm, bernoulli_rbm, boltzmann,
autoencoder, mixture_density

modern: mamba, mamba2, moe, deepseek_moe, mixture_of_experts, rwkv, s4, ttt,
jamba, llama_decoder, t5, gqa_attention, mqa_attention, sparse_attention,
linear_attention, world_model, dreamer, deepseek_moe

vision: vit, deit, unet, resnet, densenet, efficientnet, mobilenet, clip,
whisper, wav2vec2, hubert, nerf, inr

reinforcement: dqn, ppo, sac, td3, ddpg, a2c, sarsa, mcts, muzero,
value_iteration, policy_iteration, tabular_policy, rlhf

sequence/memory: hmm, ctc, pointer_net, set_transformer, ntm, dnc, hopfield

graphs: gnn, gcn, graphsage, gat, gin, mpnn

recsys: collab_filtering, matrix_factorization, wide_deep, deepfm, item2vec

probabilistic: gaussian_process, bayesian_nn, mc_dropout, kde, covariance,
probability_calibration, kalman_filter

meta-learning: maml, prototypical, matching, relation, siamese

exotic: som, capsule, echo_state, elm, neural_gas, art, spiking_nn, dbn,
rbf_network, online_learning, topic_model

time series: time_series, time_series_arima, kalman_filter

preprocessing (as techniques): standard_scaler, minmax_scaler, robust_scaler,
maxabs_scaler, normalizer, binarizer, polynomial_features, one_hot_encoder,
label_encoder, simple_imputer, knn_imputer, kbins_discretizer,
select_k_best, variance_threshold, variance_threshold2, rfe, rfe2,
count_vectorizer, tfidf_vectorizer, target_encoder, hashing_vectorizer

semi/self: semi_supervised, self_supervised, association_rules,
contrastive_learning, dpo, grpo

</details>

---

*Next: train your pick with [TRAINING.md](TRAINING.md), score it with [EVALUATION.md](EVALUATION.md), shrink it with [COMPRESS.md](COMPRESS.md), ship it with [SERVE_DEPLOY.md](SERVE_DEPLOY.md).*
