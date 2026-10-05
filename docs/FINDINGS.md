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

### 22. Feature Selection & Learning Mechanics Schedulers
- **Discovery**: Preprocessing workflows benefit from explicit low-variance filtering (`VarianceThreshold`) and correlation-based feature filtering (`SelectKBest`).
- **Impact**: Training mechanics now support decay schedulers (`LearningRateScheduler`) and early stopping convergence checks (`EarlyStopping`).

### 23. Divergence Metrics & Clustering Validation
- **Discovery**: Unsupervised clustering models require specialized evaluation metrics (`silhouette_score`, `davies_bouldin_score`, `adjusted_rand_score`, `normalized_mutual_info_score`).
- **Impact**: Added divergence functions (`kl_divergence`, `js_divergence`), specialized losses (`huber_loss`, `focal_loss`, `triplet_loss`), and distance metrics (`chebyshev_distance`, `canberra_distance`, `braycurtis_distance`).

### 24. Comprehensive Mathematical Library Architecture (`qai.math`)
- **Discovery**: Unifying core mathematical operations (linear algebra, calculus gradients, probability distributions, information theory, signal/convolutions) directly into `qai.math` eliminates indirect dependencies and accelerates custom loss/layer development.
- **Impact**: Added 70+ mathematical methods, complete with numerical derivative evaluation, Jacobian matrices, Hessian calculations, signal pooling, discrete convolutions, and documentation introspection.

### 25. Governance Fairness Auditing & Feature Encoders
- **Discovery**: Responsible AI models require fairness audit metrics (`demographic_parity_difference`, `equalized_odds_difference`, `disparate_impact_ratio`) to evaluate bias across sensitive demographic groups.
- **Impact**: Added target encoding (`TargetEncoder`), ordinal encoding (`OrdinalEncoder`), discretizers (`KBinsDiscretizer`), absolute scaling (`MaxAbsScaler`), and specialized error metrics.

### 26. Numerical Stability Safeguards & Model Calibration Metrics
- **Discovery**: Real-world ML models suffer from Log-Sum-Exp overflow and divide-by-zero crashes when calculating loss gradients.
- **Impact**: Added `log_sum_exp` and `safe_divide` to prevent silent numerical overflow/underflow, alongside calibration metrics (`brier_score`, `expected_calibration_error`) and text distance metrics (`levenshtein_distance`).

### 27. Complete Production Framework Parity
- **Discovery**: Completing distributed multi-node DDP, ONNX export, NAS, federated averaging, quantum circuit simulation, model watermarking, and bayesian optimization brings the `qai` framework to 100% feature completion.
- **Impact**: All roadmap features previously listed in `UNBUILT_FEATURES.md` have been built and verified with passing unit tests.

### 28. Comprehensive Data Manipulation & Ingestion Engine (`qai.data`)
- **Discovery**: Model training requires a robust data preparation engine supporting universal multi-format ingestion, cell-level manipulation, Excel-style dynamic formulas, version snapshotting/rollbacks, PII detection, and multi-format exports.
- **Impact**: Added complete `qai.data` sub-package with `DataDataset`, format auto-detection (`detect_format`), document text chunking (`from_document`), data quality auditing across six dimensions (`quality_report`), and Parquet/Excel exports.

### 29. Time-Series & Active Learning Data Gaps Closed
- **Discovery**: Real-world dataset engineering requires near-duplicate fuzzy matching (`fuzzy_deduplicate`), model uncertainty queueing (`active_learning_queue`), and time-series feature engineering (`rolling_window`, `lag_feature`).
- **Impact**: Added fuzzy deduplication, synthetic generation, active learning queue selection, rolling window means, lag features, text stemming, weighted sampling, and incremental file syncing.

### 30. Production Serving, Cryptographic Security & CLI Runtime (`qai.runtime`)
- **Discovery**: Production model deployments require semantic versioning (`major.minor.patch`), staging environment promotion gates, A/B traffic routing, HMAC cryptographic signatures against model supply-chain tampering, and auto-rollback triggers.
- **Impact**: Added complete `qai.runtime` sub-package with `RuntimeModelWrapper`, cryptographic signing, embedded runtime bundling, and the `qai-runtime` CLI router.
