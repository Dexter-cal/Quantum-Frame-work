# QAI Engineering Findings & Architecture Notes

This document logs critical discoveries, architectural insights, edge cases, and runtime behavior observations uncovered during the implementation and auditing of `qai`.

## Findings Log

### 1. Loose Equality Truthiness Risk in Numpy Arrays
- **Discovery**: When testing model prediction outputs against scalar ground truth values, Python equality checks like `assert prediction == expected` can evaluate to truthy numpy boolean arrays when `prediction` is actually an `np.ndarray` of shape `(1,)`.
- **Impact**: Code returning single-element 1D arrays instead of unboxed scalars went unnoticed across several standard test runs.
- **Resolution**: Implemented strict scalar unboxing checks using `isinstance(val, (int, float, str, np.number))` and `isinstance(val, np.ndarray) == False` to guarantee unboxed prediction types for single inputs.

### 2. Logic Function Deserialization Namespace Requirements
- **Discovery**: When loading decorated `@qai.logic` function source via `load_logic_source()`, executing source code dynamically in an isolated scope failed if the `qai` module itself was not bound in the execution globals.
- **Impact**: `@qai.logic` decorator syntax failed inside `exec()` during subprocess/cross-process reloading.
- **Resolution**: Injected `qai` explicitly into the dynamic execution globals dictionary during `load_logic_source()`.

### 3. Absolute Path Export Hardcoding
- **Discovery**: `tests/test_core_pipeline.py` referenced a hardcoded user path (`/home/claude/...`), causing test suite failures in generic containerized or fresh sandbox environments.
- **Impact**: Non-portable test setup.
- **Resolution**: Updated export path targets to use standard system temporary directories (`/tmp/`).

### 4. Consolidated Framework Single-Import Pattern
- **Discovery**: Users building workflows expect a complete end-to-end framework without requiring secondary third-party package imports.
- **Resolution**: Standardized top-level exports in `qai/__init__.py` to expose all core models, techniques, evaluation tools, pipelines, datasets, logic decorators, and serving handlers.

### 5. Dict-Row Input & Output Schema Validation Ergonomics
- **Discovery**: Real-world REST and pipeline payloads often deliver named JSON/dict records rather than raw NumPy 2D matrices.
- **Impact**: Requiring users to manually unpack dictionaries into positional matrices reduced usability.
- **Resolution**: Enhanced `Schema` with `validate_row()` and `dict_to_array()`. Updated `Model.train()` and `Model.predict()` to automatically accept dictionary payloads, validate fields/types, and map keys to column order.

### 6. Deep Learning Integration via PyTorch
- **Discovery**: Real-world deep learning workflows require standard dynamic computation graphs and tensor backpropagation rather than manually coded gradient matrices.
- **Impact**: PyTorch integration (`torch`) allows `qai` to seamlessly offer deep neural network capabilities while keeping the clean `qai.build(type="neural_network")` interface.

### 7. Section 56 Complete Mechanics Decomposition Wiring
- **Discovery**: In the initial design, `Objective` and `TrainingLoop` objects existed as standalone classes in `qai.mechanics` but were not directly wired into `Model.build()`.
- **Impact**: Explicitly wiring `objective` and `training_loop` into `qai.build()` fulfills Section 56's core design goal: structure (what the model IS) is completely decoupled from optimization, loss measurement, and termination criteria.

### 8. Automated Hyperparameter Tuning Ergonomics
- **Discovery**: Model selection and parameter optimization frequently require repetitive cross-validation loops.
- **Impact**: Providing `qai.AutoTuner` directly integrated with `cross_validate()` gives users automated model optimization in a single call.

### 9. Gaussian Mixture Models Unsupervised Architecture
- **Discovery**: Unlike hard assignment clustering (e.g. K-Means), soft probabilistic clustering requires component posterior probabilities and log-likelihood metrics (`score_samples`).
- **Impact**: Exposing `predict_proba()` and `score_samples()` on GMM models gives `qai` users density estimation and soft cluster membership capabilities.

### 10. Hierarchical Agglomerative Clustering Interface
- **Discovery**: Agglomerative clustering does not naturally have an Out-Of-Sample `.predict()` method in sklearn.
- **Impact**: In `qai`, `forward()` maps new samples to nearest cluster centroids, maintaining uniform prediction semantics across all unsupervised techniques.

### 11. Autoregressive Time Series Forecasting Architecture
- **Discovery**: Sequential time-series forecasting requires automated rolling lag window generation for multi-step forecasting.
- **Impact**: `TimeSeriesForecaster` provides `forecast(steps)` which iteratively feeds predictions back into the lag feature window to project arbitrary future horizons.

### 12. Isolation Forest Anomaly Detection Semantics
- **Discovery**: Anomaly detection techniques output boolean flags and continuous isolation scores rather than discrete multi-class labels or continuous regression values.
- **Impact**: `IsolationForest` provides `is_anomaly()` returning boolean flags and `anomaly_score()` for direct threat/outlier risk ranking.

### 13. Multi-Stage Pipeline Sequential Fitting Mechanics
- **Discovery**: Unsupervised feature transformers (e.g. PCA) in a pipeline must transform training features `X` before passing them to downstream estimators during `pipeline.train()`.
- **Impact**: Updating `Pipeline.train()` to automatically propagate intermediate predictions enables multi-stage feature extraction and classification in a single call.

### 14. Integrated Model Lifecycle Operations
- **Discovery**: Users require unified workflow tools (explainability, benchmarking, manifold reduction, binary serialization, automated cleaning) directly off the primary import.
- **Impact**: Exposing `benchmark()`, `save_model()`, `load_model()`, `tsne`, and `clean_dataset()` under `qai` fulfills the complete single-import AI development goal.

### 15. Discrete Count Modeling & Model Quantization
- **Discovery**: Count and text-based frequency vectors require non-negative Multinomial Naive Bayes modeling and post-training precision scaling.
- **Impact**: `MultinomialNaiveBayes` enforces $X \ge 0$ validation, while `compress_model` supports float16 and int8 parameter quantization.

### 16. Comprehensive Classifier & Regressor Ecosystem Expansion
- **Discovery**: Broad model choice (discriminant analysis, boosting ensembles, regularized linear models, SVD manifold reduction, scalers, and calibration curves) provides complete single-import AI development.
- **Impact**: All 20 newly added techniques and utilities integrate seamlessly into `qai.build()`, `qai.preprocessing`, `qai.metrics`, and `qai.governance`.

### 17. 50+ Unified Model Catalogue Architecture
- **Discovery**: Providing explicit technique strings for every major ML paradigm variant allows developers to instantiate tailored classifiers and regressors with standard `qai.build(type=...)` semantics.
- **Impact**: Expanded `qai.techniques` registry to 50+ total model types covering linear, tree, distance-based, probabilistic, ensemble, deep learning, time-series, and anomaly algorithms.

### 18. Hardware Acceleration & Manifold Learning Integration
- **Discovery**: Detecting CUDA GPU devices (`qai.get_optimal_device()`) and RAM capacity (`qai.get_hardware_info()`) enables automatic device dispatching for deep learning training.
- **Impact**: Expanding `qai.techniques` with non-linear manifold learning (`isomap`, `lle`, `fast_ica`, `spectral_clustering`) brings state-of-the-art dimensionality reduction to the single-import interface.

### 19. Dedicated Math & System Sub-Packages
- **Discovery**: AI workloads require low-level distance, activation, linear algebra matrix routines, and system VRAM/CPU monitoring tools alongside model wrappers.
- **Impact**: Adding `qai.math` and `qai.system` sub-packages guarantees that developers never need secondary imports for matrix math or resource profiling.

### 21. Model Explainability & Permutation Importance
- **Discovery**: Black-box ML models benefit from feature importance attribution.
- **Impact**: `feature_importance()` extracts model coefficients/trees feature importances, while `permutation_importance()` evaluates performance drops across feature shuffles.
