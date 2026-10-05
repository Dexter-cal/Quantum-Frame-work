# QAI Framework -- User Manual & Developer Reference Guide

Welcome to the **qai** Python package! `qai` is a unified, single-import machine learning and artificial intelligence framework designed for end-to-end model development, preprocessing, governance, evaluation, and serving.

With `qai`, you import one package (`import qai`) to build classical supervised/unsupervised models, PyTorch deep neural networks, reinforcement learning policies, data pipelines, ensembles, automated hyperparameter tuners, and REST APIs.

---

## Table of Contents
1. [Quickstart & Single-Import Philosophy](#1-quickstart--single-import-philosophy)
2. [Core Model Operations (`Model` & `qai.build`)](#2-core-model-operations-model--qaibuild)
3. [Built-in Techniques Directory](#3-built-in-techniques-directory)
   - [Supervised Learning](#supervised-learning)
   - [Unsupervised Learning](#unsupervised-learning)
   - [Reinforcement Learning](#reinforcement-learning)
   - [Deep Learning (PyTorch)](#deep-learning-pytorch)
4. [Input & Output Schema Validation](#4-input--output-schema-validation)
5. [Data Preprocessing (`qai.preprocessing`)](#5-data-preprocessing-qaipreprocessing)
6. [Model Governance & Drift Detection (`qai.governance`)](#6-model-governance--drift-detection-qaigovernance)
7. [Experiment Tracking (`qai.tracking`)](#7-experiment-tracking-qaitracking)
8. [Cross-Validation & AutoML (`qai.AutoTuner`)](#8-cross-validation--automl-qaiautotuner)
9. [Pipelines & Voting Ensembles](#9-pipelines--voting-ensembles)
10. [Model Serving (`qai.serve`)](#10-model-serving-qaiserve)

---

## 1. Quickstart & Single-Import Philosophy

Everything in `qai` is accessible directly from the top-level namespace:

```python
import qai

# 1. Build and train a model
model = qai.build(type="regression")
model.train(X=[[1.0], [2.0], [3.0]], y=[2.0, 4.0, 6.0])

# 2. Predict on new input
prediction = model.predict([4.0])
print("Prediction:", prediction) # Output: 8.0

# 3. Inspect learned formula
print("Formula:", model.formula()) # Output: y = 2.000*x0 + 0.000
```

---

## 2. Core Model Operations (`Model` & `qai.build`)

### `qai.build(type, learning_technique=None, input_schema=None, output_schema=None, **params)`
Instantiates a new `Model` wrapper around the requested technique.

* **Arguments**:
  - `type` (*str*): The algorithm identifier (e.g., `'regression'`, `'decision_tree'`, `'knn'`, `'neural_network'`, `'kmeans'`, `'dbscan'`, `'gmm'`).
  - `learning_technique` (*str, optional*): Explicit paradigm selection (`'supervised'`, `'unsupervised'`, or `'reinforcement'`). Defaults to the technique's primary paradigm.
  - `input_schema` (*dict/Schema, optional*): Declared expected field names and types for training/prediction.
  - `output_schema` (*dict/Schema, optional*): Declared expected fields and types for model outputs.
  - `**params`: Algorithm-specific hyperparameters.

### Shared Model Methods
- `model.train(X, y=None, verbose=True)`: Fits the underlying technique on `X` (and optional targets `y`). Supports NumPy arrays, Pandas DataFrames, `qai.Dataset`, or dict-rows.
- `model.predict(x)`: Generates predictions for a single sample or batch of samples.
- `model.accuracy(X=None, y=None)`: Returns accuracy score.
- `model.weights`: Returns a `WeightAccessor` for inspecting and directly editing model parameters.
- `model.explain(x)`: Generates feature attribution and decision explanations.
- `model.visualize(save_path)`: Renders matplotlib diagnostic plots to disk.
- `model.export(path)`: Serializes model metadata and weights to JSON.

---

## 3. Built-in Techniques Directory

### Supervised Learning
- **`regression`**: Linear Regression ($y = w \cdot x + b$).
  - *Methods*: `formula()`, `residuals()`, `r_squared()`, `finetune(X_new, y_new)`
- **`classifier`**: Logistic Regression classification.
  - *Methods*: `class_probabilities(x)`, `confusion_matrix()`
- **`decision_tree`**: Decision Tree Classifier.
  - *Methods*: `feature_importance()`, `tree_depth()`
- **`knn`**: K-Nearest Neighbors.
  - *Methods*: `nearest_neighbors(x, n=None)`
- **`svm`**: Support Vector Machine.
  - *Methods*: `support_vectors()`, `margin_width()`
- **`naive_bayes`**: Gaussian Naive Bayes.
  - *Methods*: `class_priors()`, `likelihood_table()`
- **`perceptron`**: Single-layer Perceptron (1958 architecture).
  - *Methods*: `n_updates()`
- **`random_forest`**: Random Forest Ensemble.
  - *Methods*: `tree_count()`, `member_agreement(x)`

### Unsupervised Learning
- **`kmeans`**: K-Means Clustering.
  - *Methods*: `cluster_centers()`, `elbow_plot_data(X)`
- **`pca`**: Principal Component Analysis.
  - *Methods*: `explained_variance_ratio()`, `find_eigenvectors()`, `reconstruction_error(X)`
- **`dbscan`**: Density-Based Spatial Clustering of Applications with Noise.
  - *Methods*: `n_clusters()`, `noise_ratio()`
- **`gmm`**: Gaussian Mixture Models.
  - *Methods*: `predict_proba(x)`, `score_samples(x)`, `means()`, `covariances()`

### Reinforcement Learning
- **`tabular_policy`**: Q-Learning Tabular Reinforcement Learning Policy.
  - *Methods*: `q_values(state)`, `best_action(state)`, `cumulative_reward()`, `episode_count()`

### Deep Learning (PyTorch)
- **`neural_network`**: PyTorch Multi-Layer Perceptron (MLP) for classification and regression.
  - *Parameters*: `hidden_dim=16`, `epochs=50`, `lr=0.01`, `task_type="classification"|"regression"`
  - *Methods*: `loss_history()`, `parameter_count()`

```python
# Deep Learning Example
nn_model = qai.build(
    type="neural_network",
    hidden_dim=32,
    epochs=100,
    lr=0.01,
    task_type="classification"
)
nn_model.train(X_train, y_train)
preds = nn_model.predict(X_test)
print("Training Loss Curve:", nn_model.loss_history())
```

---

## 4. Input & Output Schema Validation

`qai` guarantees data integrity using strict schema definitions:

```python
schema = {"age": int, "income": float}

model = qai.build(
    type="regression",
    input_schema=schema,
    output_schema={"prediction": float}
)

# Accepts dict rows matching schema automatically
model.train({"age": 30, "income": 75000.0}, y=[5000.0])
result = model.predict({"age": 35, "income": 80000.0})
```

---

## 5. Data Preprocessing (`qai.preprocessing`)

- **`qai.StandardScaler`**: Z-score normalization ($(X - \mu) / \sigma$).
- **`qai.LabelEncoder`**: Converts categorical strings to integers and back.
- **`qai.SimpleImputer`**: Imputes missing values (`np.nan`) using `'mean'` or `'median'`.

```python
from qai import StandardScaler, LabelEncoder, SimpleImputer

# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Label Encoding
encoder = LabelEncoder()
y_numeric = encoder.fit_transform(["cat", "dog", "cat"])
y_original = encoder.inverse_transform(y_numeric)
```

---

## 6. Model Governance & Drift Detection (`qai.governance`)

Detect distribution drift between baseline training data and runtime production features:

```python
import qai

drift_report = qai.detect_drift(
    reference_data=baseline_features,
    current_data=production_features,
    threshold=0.1
)

if drift_report["drift_detected"]:
    print("ALERT: Feature drift detected!")
    print(drift_report["features"])
```

---

## 7. Experiment Tracking (`qai.tracking`)

Log parameters, metrics, and export experiment manifests:

```python
tracker = qai.ExperimentTracker(experiment_name="iris_classification")

# Log runs
tracker.log_run(params={"k": 3}, metrics={"accuracy": 0.95})
tracker.log_run(params={"k": 5}, metrics={"accuracy": 0.98})

# Get top run
best_run = tracker.get_best_run(metric_name="accuracy", mode="max")
print("Best Params:", best_run["params"])

# Export experiment log
tracker.export("iris_experiment_results.json")
```

---

## 8. Cross-Validation & AutoML (`qai.AutoTuner`)

Automate hyperparameter grid search using k-fold cross-validation:

```python
grid = {"k": [3, 5, 7, 9]}

# Perform automated tuning
tuner = qai.autotune("knn", param_grid=grid, X=X, y=y, k=5)

print("Best Parameters:", tuner.best_params_)
print("Best CV Score:", tuner.best_score_)

# Obtain best model pre-fitted
best_model = tuner.best_model(X, y)
```

---

## 9. Pipelines & Voting Ensembles

Chaining models sequentially or combining parallel model predictions:

```python
# Sequential Pipeline: PCA -> Classifier
pca = qai.build(type="pca", n_components=2)
clf = qai.build(type="classifier")
pipeline = pca.pipe_to(clf)
pipeline.train(X_train, y_train)

# Parallel Voting Ensemble
tree = qai.build(type="decision_tree")
knn = qai.build(type="knn")
nb = qai.build(type="naive_bayes")

ensemble = qai.VotingEnsemble([tree, knn, nb])
ensemble.train(X_train, y_train)
vote_winner = ensemble.predict(X_test[0])
print("Vote breakdown:", ensemble.vote_breakdown(X_test[0]))
```

---

## 10. Model Serving (`qai.serve`)

Deploy any trained `qai` model instantly as a production REST API:

```python
# Starts a local HTTP server exposing /predict, /health, /info
qai.serve(model, port=5000)
```

### Unsupervised Technique Addition
- **`hierarchical`**: Agglomerative Hierarchical Clustering.
  - *Methods*: `cluster_counts()`, `n_leaves()`

### Time Series Forecasting Addition
- **`time_series`**: Autoregressive Time Series Forecaster.
  - *Parameters*: `lags=3`, `horizon=1`
  - *Methods*: `forecast(steps=5)`, `residuals()`

### Anomaly Detection Addition
- **`isolation_forest`**: Unsupervised Isolation Forest for anomaly and outlier detection.
  - *Parameters*: `n_estimators=100`, `contamination=0.1`
  - *Methods*: `anomaly_score(x)`, `is_anomaly(x)`

### Multi-Stage Pipeline Updates
- **`Pipeline`**: Sequential multi-model processing.
  - *Methods*: `train(X, y=None, verbose=True)`, `predict(x)`, `pipe_to(other_model)`

### Framework Utilities & Extensions Directory
- **`qai.benchmark(techniques, X, y)`**: Evaluates accuracy and latency across multiple model types.
- **`qai.save_model(model, filepath)` / `qai.load_model(filepath)`**: Binary model persistence.
- **`qai.clean_dataset(X, y)`**: Handles NaNs, scaling, and categorical encoding in one call.
- **`tsne`**: t-Distributed Stochastic Neighbor Embedding (`embedding()`).

### Extended AI Utilities (Batch 2)
- **`multinomial_naive_bayes`**: Naive Bayes for discrete feature counts (`class_log_prior()`).
- **`qai.PolynomialFeatures`**: Generates degree-N polynomial and interaction terms.
- **`qai.classification_report(y_true, y_pred)`**: Precision, recall, and F1-score dict.
- **`qai.compress_model(model, precision)`**: Quantizes weights to `'float16'` or `'int8'`.
- **`qai.make_classification(n_samples, n_features)`**: Generates synthetic classification datasets.

### 20 Additional ML/AI Features Directory
- **Classifiers**: `lda`, `qda`, `adaboost`, `gradient_boosting`, `extra_trees`
- **Regressors**: `ridge`, `lasso`, `elastic_net`, `kernel_ridge`
- **Unsupervised**: `truncated_svd`
- **Scalers & Encoders**: `MinMaxScaler`, `RobustScaler`, `Normalizer`, `OneHotEncoder`, `Binarizer`
- **Metrics**: `mean_absolute_error`, `r2_score`, `log_loss`, `roc_auc_score`
- **Generators**: `make_regression`
- **Governance**: `weight_distance(m1, m2)`, `calibration_curve(y_true, y_prob)`

---

## 11. Logic Block & Persistence (`@qai.logic`)

`qai` provides native Python control flow decoration and source persistence via `@qai.logic`.

```python
import qai

# 1. Decorate control flow logic
@qai.logic
def evaluate_decision(x):
    model = qai.build(type="regression")
    model.train([[1.0], [2.0]], [2.0, 4.0])
    pred = model.predict([x])
    return pred * 2

# 2. Execute decorated logic
result = evaluate_decision(5.0)

# 3. Save logic source code to disk
qai.save_logic_source(evaluate_decision, "decision_logic.py")

# 4. Reload logic in a separate process
reloaded_fn = qai.load_logic_source("decision_logic.py")
```

### Additional Model Classifiers & Regressors
- **Classifiers**: `bernoulli_naive_bayes`, `complement_naive_bayes`, `sgd_classifier`, `passive_aggressive_classifier`, `linear_svc`, `nu_svc`, `radius_neighbors_classifier`, `nearest_centroid`, `bagging_classifier`, `hist_gradient_boosting`
- **Regressors**: `bayesian_ridge`, `ard_regression`, `huber`, `ransac`, `theil_sen`, `quantile_regression`, `decision_tree_regressor`, `random_forest_regressor`, `adaboost_regressor`, `gradient_boosting_regressor`

### Hardware & Environment Profiling
- **`qai.get_hardware_info()`**: Returns dict with OS, CPU logical cores, RAM capacities, and NVIDIA CUDA GPU device details.
- **`qai.get_optimal_device()`**: Returns `'cuda'` if an NVIDIA GPU is available, else `'cpu'`.

### Advanced Manifold & Clustering Techniques
- **`spectral_clustering`**: Spectral Clustering for non-convex structures.
- **`fast_ica`**: Fast Independent Component Analysis for source separation.
- **`isomap`**: Isometric Feature Mapping.
- **`lle`**: Locally Linear Embedding.

---

## 12. System Tools Package (`qai.system`)

```python
import qai

# Process RAM & Disk
print("Process Memory (MB):", qai.system.get_process_memory())
print("Disk Usage (GB):", qai.system.get_disk_usage())

# GPU Memory VRAM
print("GPU Memory:", qai.system.get_gpu_memory())

# Limit thread parallelism
qai.system.set_num_threads(4)
```

---

## 13. Comprehensive Mathematics Package (`qai.math`)

```python
import qai

# Activations
prob = qai.sigmoid(0.0)             # 0.5
softmax_probs = qai.softmax([1, 2]) # [0.2689, 0.7310]

# Distance Metrics
euc_dist = qai.euclidean_distance([1, 0], [0, 1]) # 1.4142
cos_sim = qai.cosine_similarity([1, 0], [0, 1])  # 0.0

# Linear Algebra SVD
u, s, vt = qai.math.singular_value_decomposition([[1.0, 2.0], [3.0, 4.0]])
```

### Model Explainability
```python
import qai

model = qai.build(type="random_forest")
model.train(X, y)

# Tree/Coefficient feature importances
imps = qai.feature_importance(model)

# Model-agnostic permutation importances
perm_imps = qai.permutation_importance(model, X, y)
```

### Feature Selection
```python
import qai

# Remove low-variance features
selector = qai.VarianceThreshold(threshold=0.1)
X_filtered = selector.fit_transform(X)

# Select top-K features
k_selector = qai.SelectKBest(k=5)
X_top = k_selector.fit_transform(X, y)
```

### Learning Schedulers & Early Stopping
```python
from qai.mechanics import LearningRateScheduler, EarlyStopping

scheduler = LearningRateScheduler(initial_lr=0.01, decay_factor=0.5, step_size=10)
stopper = EarlyStopping(patience=5)

for epoch in range(100):
    lr = scheduler.step()
    if stopper.update(val_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        break
```

### Advanced Loss & Divergence Functions
```python
import qai

# Loss functions
hl = qai.huber_loss(y_true, y_pred, delta=1.0)
fl = qai.focal_loss(y_true, y_pred_probs, gamma=2.0)
tl = qai.triplet_loss(anchor, positive, negative)

# Divergences
kl = qai.kl_divergence(p, q)
js = qai.js_divergence(p, q)
```

### Clustering Validation Metrics
```python
import qai

sil = qai.silhouette_score(X, labels)
db = qai.davies_bouldin_score(X, labels)
ari = qai.adjusted_rand_score(labels_true, labels_pred)
nmi = qai.normalized_mutual_info_score(labels_true, labels_pred)
```

### Comprehensive Mathematical Library (`qai.math`)
```python
import qai.math as qmath

# 1. Linear Algebra
A = [[4, 1], [1, 3]]
inv = qmath.inverse(A)
vals, vecs = qmath.eigen(A)
U, S, Vh = qmath.svd(A)

# 2. Calculus & Gradients
grad = qmath.gradient(lambda x: x[0]**2 + 3*x[1], [2.0, 1.0])
H = qmath.hessian(lambda x: x[0]**2 + 3*x[1], [2.0, 1.0])

# 3. Activations & Derivatives
sig = qmath.sigmoid([0.0, 1.0])
d_sig = qmath.sigmoid_derivative([0.0, 1.0])

# 4. Probability & Statistics
prior = qmath.class_prior([0, 0, 1, 1, 1])
ci = qmath.confidence_interval([10, 12, 11, 9, 13])

# 5. Information Theory & Distance
ent = qmath.entropy([0.5, 0.5])
dist = qmath.euclidean_distance([1, 2], [4, 6])

# 6. Signal & Convolutions
conv = qmath.convolve([1, 2, 3, 4], [1, 0, -1])
pooled = qmath.pooling([[1, 2], [3, 4]], pool_size=(2, 2), mode="max")

# 7. Helper & Inspection
print(qmath.help())
print(qmath.help("gini_impurity"))
```

### Governance Fairness Auditing
```python
import qai

# Comprehensive algorithmic bias & fairness audit
audit = qai.fairness_audit(y_true, y_pred, sensitive_features)
print(audit["demographic_parity_difference"])
print(audit["equalized_odds_difference"])
print(audit["disparate_impact_ratio"])
```

### Advanced Preprocessing & Encoders
```python
import qai

# Target Encoding
t_enc = qai.TargetEncoder()
X_encoded = t_enc.fit_transform(X_categorical, y)

# Ordinal & Discretizers
ord_enc = qai.OrdinalEncoder()
X_ord = ord_enc.fit_transform(X_categorical)

kbins = qai.KBinsDiscretizer(n_bins=5)
X_binned = kbins.fit_transform(X_continuous)
```

### Numerical Stability & Calibration Metrics
```python
import qai
import qai.math as qmath

# Stable Log-Sum-Exp & Safe Division
lse = qmath.log_sum_exp([1000.0, 1001.0, 1002.0])
div = qmath.safe_divide(10.0, 0.0)

# Confidence Calibration
bs = qai.brier_score(y_true, y_prob)
ece = qai.expected_calibration_error(y_true, y_prob, n_bins=10)

# Text & String Edit Distance
dist = qmath.levenshtein_distance("kitten", "sitting")
jacc = qmath.jaccard_similarity(["a", "b"], ["a", "c"])
```

### Advanced Framework Extensions
```python
import qai

# Distributed Multi-Node & ONNX
ctx = qai.init_distributed_context(rank=0, world_size=2)
meta = qai.export_to_onnx(model, "model.onnx")

# Bayesian Optimization & NAS
opt = qai.bayesian_optimize("classifier", X, y, {"learning_rate": [0.01, 0.1]})
nas = qai.search_architecture(X, y)

# Federated Learning & Model Watermarking
avg_w = qai.federated_averaging([w1, w2])
wm = qai.watermark_model(model, secret_key="my_secret")
valid = qai.verify_watermark(model, secret_key="my_secret")

# Quantum Circuit Simulation
qc = qai.QuantumCircuitSimulator(n_qubits=2)
exp_val = qai.quantum_expectation(qc)
```

### Complete Data Manipulation Library (`qai.data`)
```python
import qai.data as qdata

# Universal Ingestion & Document Ingestion
ds = qdata.load("dataset.parquet")
doc_ds = qdata.from_document("book.pdf", chunk_size=500)

# Summary & Cleaning
info = ds.info()
ds.remove_duplicates()
ds.handle_missing(strategy="median")
ds.filter(lambda row: row["age"] > 18)

# Formulas & Cell Edits
ds.cell(row=0, col="price").set(19.99)
ds.formula("total", "= price * qty")

# Search, Versioning & Quality
matches = ds.search("target_value")
ds.snapshot("before_clean")
ds.rollback("before_clean")

# PII Detection, Anonymization & Quality
pii = ds.detect_pii()
ds.anonymize(columns=["email"])
report = ds.quality_report()

# Multi-Format Export
ds.to_json("output.json")
ds.to_parquet("output.parquet")
ds.to_excel("output.xlsx")
```

### Advanced Time-Series & NLP Data Extensions
```python
import qai.data as qdata

ds = qdata.load("data.csv")

# Fuzzy Deduplication & Synthetic Generation
ds.fuzzy_deduplicate(threshold=0.85)
syn_ds = ds.generate_synthetic_data(n_samples=50)

# Active Learning Uncertainty Queue
queue_ds = ds.active_learning_queue(model.predict_proba, top_k=10)

# Time-Series Rolling Window & Lags
ds.rolling_window(size=5, column="sales")
ds.lag_feature(column="sales", periods=1)

# NLP Text Preprocessing
ds.remove_stopwords("text")
ds.stem("text")

# Cataloging & Incremental Sync
catalog = qdata.project_catalog()
cost = qdata.estimate_load_cost("large_file.csv")
updated_ds = qdata.incremental_sync(ds, "new_batch.csv", key_col="id")
```

### Complete Production Runtime & CLI (`qai.runtime`)
```python
import qai
import qai.runtime as qruntime

model = qai.build(type="classifier")
model.train(X, y)

# Wrap model with production runtime
rt_model = qruntime.RuntimeModelWrapper(model, name="fraud_detector")

# Semantic Versioning & Staging
rt_model.bump_version("minor") # -> "1.1.0"
rt_model.deploy(environment="staging", auto_rollback_if="accuracy < 0.90")
rt_model.promote(from_env="staging", to_env="production")

# Cryptographic HMAC Security
sig = rt_model.sign_model(private_key="my_secret_key")
is_valid = rt_model.verify_signature(private_key="my_secret_key")

# Live A/B Traffic Splitting
ab_router = qruntime.ab_test(model_a, model_b, traffic_split=0.5)

# Embedded Runtime Export
rt_model.export("model_bundle.json", include_runtime=True, runtime_scope="predict_only")

# CLI Commands
qruntime.qai_runtime_cli(["list"])
qruntime.qai_runtime_cli(["serve"])
```
