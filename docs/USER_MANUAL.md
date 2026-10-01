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
