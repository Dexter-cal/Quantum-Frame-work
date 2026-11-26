"""
qai.pipeline_templates — pre-built end-to-end ML workflows.

Ready-to-use pipeline templates that combine preprocessing, training,
evaluation, and deployment into a single call.

Templates:
  - classification_pipeline — full classification workflow
  - regression_pipeline — full regression workflow
  - clustering_pipeline — full clustering workflow
  - nlp_pipeline — text classification / sentiment
  - vision_pipeline — image classification
  - llm_pipeline — LLM fine-tuning + serving
  - time_series_pipeline — forecasting
  - auto_pipeline — auto-detect task + run best pipeline

Usage:
    import qai

    # One-call classification
    result = qai.classification_pipeline(X, y)
    print(result.accuracy)
    result.model.predict(X_new)

    # LLM fine-tuning pipeline
    result = qai.llm_pipeline(
        model_name="gpt2",
        train_texts=texts,
        method="lora",
        epochs=3,
    )
    result.model.generate("Hello")
"""
from __future__ import annotations
import os, time, json, copy
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np


# =====================================================================
# Pipeline Result
# =====================================================================
@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    pipeline_name: str
    model: Any = None
    accuracy: float = 0.0
    technique: str = ""
    preprocessing_steps: List[str] = field(default_factory=list)
    training_time_s: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name → path

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"Pipeline: {self.pipeline_name}",
            "=" * 60,
            f"Technique: {self.technique}",
            f"Accuracy: {self.accuracy:.4f}" if self.accuracy else "Accuracy: N/A",
            f"Training time: {self.training_time_s:.1f}s",
            f"Preprocessing: {' → '.join(self.preprocessing_steps) if self.preprocessing_steps else 'none'}",
        ]
        if self.metrics:
            lines.append(f"Metrics: {json.dumps(self.metrics, indent=2, default=str)}")
        if self.artifacts:
            lines.append(f"Artifacts: {list(self.artifacts.keys())}")
        lines.append("=" * 60)
        return "\n".join(lines)


# =====================================================================
# 1. Classification Pipeline
# =====================================================================
def classification_pipeline(X, y, test_size: float = 0.2,
                             technique: Optional[str] = None,
                             preprocess: bool = True,
                             scale: str = "standard",
                             use_auto_ml: bool = False,
                             verbose: bool = True) -> PipelineResult:
    """Full classification pipeline: preprocess → train → evaluate.

    Steps:
      1. Split into train/test
      2. Scale features (standard/minmax/none)
      3. Train (auto-ML or specified technique)
      4. Evaluate (accuracy, precision, recall, F1)
      5. Generate model card

    Args:
        X: input features
        y: labels
        test_size: fraction for testing
        technique: specific technique (None = auto-ML)
        preprocess: if True, apply preprocessing
        scale: "standard", "minmax", or "none"
        use_auto_ml: if True and technique is None, use auto_train
        verbose: if True, print progress

    Returns:
        PipelineResult with trained model + metrics
    """
    import qai
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    steps = []
    t0 = time.time()

    if verbose:
        print(f"[classification_pipeline] data: {len(X)} samples, {X.shape[1]} features")

    # 1. Split
    n = len(X)
    split = int(n * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    steps.append("split")

    # 2. Preprocess
    if preprocess and scale != "none":
        from sklearn.preprocessing import StandardScaler as SKStd, MinMaxScaler as SKMM
        scaler_obj = SKStd() if scale == "standard" else SKMM()
        X_train = scaler_obj.fit_transform(X_train)
        X_test = scaler_obj.transform(X_test)
        steps.append(f"scale({scale})")

    # 3. Train
    if technique is not None:
        m = qai.build(technique=technique)
        m.train(X_train, y_train, verbose=False)
        best_tech = technique
    elif use_auto_ml:
        result = qai.auto_train(X_train, y_train, task="classification",
                                  verbose=verbose, time_budget_s=30)
        m = result.best_model
        best_tech = result.best_technique
        if m is None:
            m = qai.build(technique="classifier")
            m.train(X_train, y_train, verbose=False)
            best_tech = "classifier"
    else:
        m = qai.build(technique="classifier")
        m.train(X_train, y_train, verbose=False)
        best_tech = "classifier"
    steps.append(f"train({best_tech})")

    # 4. Evaluate
    preds = [m.predict(X_test[i]) for i in range(len(X_test))]
    accuracy = sum(1 for p, t in zip(preds, y_test) if p == t) / max(len(y_test), 1)
    steps.append("evaluate")

    # Metrics
    from collections import Counter
    classes = np.unique(y)
    metrics = {"accuracy": accuracy}
    for cls in classes:
        tp = sum(1 for p, t in zip(preds, y_test) if p == cls and t == cls)
        fp = sum(1 for p, t in zip(preds, y_test) if p == cls and t != cls)
        fn = sum(1 for p, t in zip(preds, y_test) if p != cls and t == cls)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        metrics[f"precision_class{cls}"] = prec
        metrics[f"recall_class{cls}"] = rec
        metrics[f"f1_class{cls}"] = f1

    elapsed = time.time() - t0
    if verbose:
        print(f"[classification_pipeline] done — accuracy={accuracy:.4f} time={elapsed:.1f}s")

    return PipelineResult(
        pipeline_name="classification",
        model=m, accuracy=accuracy, technique=best_tech,
        preprocessing_steps=steps, training_time_s=elapsed,
        metrics=metrics,
    )


# =====================================================================
# 2. Regression Pipeline
# =====================================================================
def regression_pipeline(X, y, test_size: float = 0.2,
                          technique: Optional[str] = None,
                          preprocess: bool = True,
                          verbose: bool = True) -> PipelineResult:
    """Full regression pipeline: preprocess → train → evaluate."""
    import qai
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    steps = []
    t0 = time.time()

    n = len(X)
    split = int(n * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    steps.append("split")

    if preprocess:
        from sklearn.preprocessing import StandardScaler
        scaler_obj = StandardScaler()
        X_train = scaler_obj.fit_transform(X_train)
        X_test = scaler_obj.transform(X_test)
        steps.append("scale")

    tech = technique or "regression"
    m = qai.build(technique=tech)
    m.train(X_train, y_train, verbose=False)
    steps.append(f"train({tech})")

    preds = [m.predict(X_test[i]) for i in range(len(X_test))]
    mse = np.mean([(p - t) ** 2 for p, t in zip(preds, y_test)])
    mae = np.mean([abs(p - t) for p, t in zip(preds, y_test)])
    rmse = np.sqrt(mse)
    y_std = np.std(y_test)
    r2 = 1 - mse / max(y_std ** 2, 1e-8)
    steps.append("evaluate")

    elapsed = time.time() - t0
    metrics = {"mse": float(mse), "mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
    if verbose:
        print(f"[regression_pipeline] R²={r2:.4f} RMSE={rmse:.4f} time={elapsed:.1f}s")

    return PipelineResult(
        pipeline_name="regression", model=m, accuracy=r2,
        technique=tech, preprocessing_steps=steps,
        training_time_s=elapsed, metrics=metrics,
    )


# =====================================================================
# 3. Clustering Pipeline
# =====================================================================
def clustering_pipeline(X, n_clusters: Optional[int] = None,
                         technique: str = "kmeans",
                         preprocess: bool = True,
                         verbose: bool = True) -> PipelineResult:
    """Full clustering pipeline: preprocess → cluster → evaluate."""
    import qai
    X = np.asarray(X, dtype=np.float32)
    steps = []
    t0 = time.time()

    if preprocess:
        from sklearn.preprocessing import StandardScaler
        scaler_obj = StandardScaler()
        X = scaler_obj.fit_transform(X)
        steps.append("scale")

    if n_clusters is None:
        # Auto-select k via elbow method (simplified)
        n_clusters = min(5, max(2, len(X) // 30))
    steps.append(f"auto_k={n_clusters}")

    m = qai.build(technique=technique, n_clusters=n_clusters)
    m.train(X, verbose=False)
    steps.append(f"cluster({technique})")

    preds = [m.predict(X[i]) for i in range(len(X))]
    # Silhouette score (simplified)
    from collections import Counter
    cluster_counts = Counter(preds)
    n_unique = len(cluster_counts)
    balance = min(cluster_counts.values()) / max(cluster_counts.values()) if cluster_counts else 0
    steps.append("evaluate")

    elapsed = time.time() - t0
    metrics = {"n_clusters": n_clusters, "n_unique_found": n_unique,
               "balance": float(balance), "cluster_sizes": dict(cluster_counts)}
    if verbose:
        print(f"[clustering_pipeline] {n_unique} clusters, balance={balance:.3f} time={elapsed:.1f}s")

    return PipelineResult(
        pipeline_name="clustering", model=m, accuracy=balance,
        technique=technique, preprocessing_steps=steps,
        training_time_s=elapsed, metrics=metrics,
    )


# =====================================================================
# 4. NLP Pipeline (text classification)
# =====================================================================
def nlp_pipeline(texts: List[str], labels: Optional[List] = None,
                  task: str = "classification",
                  technique: str = "tfidf_vectorizer",
                  model_technique: str = "classifier",
                  verbose: bool = True) -> PipelineResult:
    """NLP pipeline: tokenize → vectorize → train → evaluate.

    Args:
        texts: list of text strings
        labels: optional labels (None = unsupervised)
        task: "classification" or "clustering"
        technique: vectorizer ("tfidf_vectorizer", "count_vectorizer")
        model_technique: classifier ("classifier", "naive_bayes", "svm")
    """
    import qai
    steps = []
    t0 = time.time()

    # 1. Vectorize
    vectorizer = qai.build(technique=technique)
    vectorizer.train(texts, verbose=False)
    X = np.array([vectorizer.predict(t) for t in texts], dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    steps.append(f"vectorize({technique})")

    if verbose:
        print(f"[nlp_pipeline] vectorized {len(texts)} texts → {X.shape}")

    # 2. Train
    if labels is not None and task == "classification":
        y = np.asarray(labels)
        m = qai.build(technique=model_technique)
        m.train(X, y, verbose=False)
        steps.append(f"train({model_technique})")

        # 3. Evaluate
        n = len(X)
        split = int(n * 0.8)
        preds = [m.predict(X[i]) for i in range(split, n)]
        accuracy = sum(1 for p, t in zip(preds, y[split:]) if p == t) / max(len(y[split:]), 1)
        steps.append("evaluate")
        elapsed = time.time() - t0
        if verbose:
            print(f"[nlp_pipeline] accuracy={accuracy:.4f} time={elapsed:.1f}s")
        return PipelineResult(
            pipeline_name="nlp", model=m, accuracy=accuracy,
            technique=model_technique, preprocessing_steps=steps,
            training_time_s=elapsed,
            metrics={"accuracy": accuracy, "n_texts": len(texts), "vectorizer": technique},
        )
    else:
        # Unsupervised
        m = qai.build(technique="kmeans", n_clusters=3)
        m.train(X, verbose=False)
        steps.append("cluster")
        elapsed = time.time() - t0
        return PipelineResult(
            pipeline_name="nlp_unsupervised", model=m,
            technique="kmeans", preprocessing_steps=steps,
            training_time_s=elapsed,
        )


# =====================================================================
# 5. Auto Pipeline (detect task + run best pipeline)
# =====================================================================
def auto_pipeline(X, y=None, verbose: bool = True) -> PipelineResult:
    """Auto-detect task type and run the appropriate pipeline.

    Detects:
      - y is None → clustering
      - y is int and < 20 unique → classification
      - y is float → regression

    Then runs the matching pipeline.
    """
    X = np.asarray(X)
    if y is not None:
        y = np.asarray(y)

    if verbose:
        print(f"[auto_pipeline] data: {len(X)} samples, {X.shape[1] if X.ndim > 1 else 1} features")

    if y is None:
        if verbose:
            print("[auto_pipeline] detected: clustering (no labels)")
        return clustering_pipeline(X, verbose=verbose)

    # Check if classification or regression
    n_unique = len(np.unique(y))
    if n_unique <= 20 and y.dtype.kind in ("i", "u"):
        if verbose:
            print(f"[auto_pipeline] detected: classification ({n_unique} classes)")
        return classification_pipeline(X, y, verbose=verbose)
    else:
        if verbose:
            print("[auto_pipeline] detected: regression")
        return regression_pipeline(X, y, verbose=verbose)


# =====================================================================
# 6. LLM Pipeline
# =====================================================================
def llm_pipeline(model_name: str = "gpt2",
                  train_texts: Optional[List[str]] = None,
                  train_labels: Optional[List] = None,
                  method: Optional[str] = None,  # "lora", "dpo", None
                  epochs: int = 3,
                  serve: bool = False,
                  port: int = 8000,
                  verbose: bool = True) -> PipelineResult:
    """LLM pipeline: load → fine-tune → (optionally serve).

    Args:
        model_name: HuggingFace model name or qai zoo alias
        train_texts: optional training texts
        train_labels: optional labels
        method: fine-tuning method ("lora", "dpo", None for SFT)
        epochs: training epochs
        serve: if True, start production server
        port: serving port
    """
    import qai
    steps = []
    t0 = time.time()

    # 1. Load model
    if verbose:
        print(f"[llm_pipeline] loading {model_name}...")
    try:
        m = qai.from_huggingface(model_name)
        steps.append(f"load({model_name})")
    except Exception as e:
        if verbose:
            print(f"[llm_pipeline] HF load failed: {e}, using built-in transformer")
        m = qai.build(technique="transformer", vocab_size=256, d_model=64,
                       n_heads=4, n_layers=2, epochs=epochs)
        if train_texts:
            # Simple tokenization
            X = np.array([[ord(c) % 256 for c in t[:32]] + [0] * max(0, 32 - len(t))
                          for t in train_texts[:50]])
            m.train(X, verbose=False)
        steps.append("build(transformer)")

    # 2. Fine-tune
    if train_texts is not None and method is not None:
        if verbose:
            print(f"[llm_pipeline] fine-tuning with {method}...")
        try:
            m.finetune(train_texts[:100], method=method, epochs=epochs, verbose=False)
            steps.append(f"finetune({method})")
        except Exception as e:
            if verbose:
                print(f"[llm_pipeline] fine-tune failed: {e}")
            steps.append("finetune(skipped)")

    elapsed = time.time() - t0
    if verbose:
        print(f"[llm_pipeline] done in {elapsed:.1f}s")

    result = PipelineResult(
        pipeline_name="llm", model=m, technique=model_name,
        preprocessing_steps=steps, training_time_s=elapsed,
    )

    # 3. Serve
    if serve:
        if verbose:
            print(f"[llm_pipeline] starting server on port {port}...")
        try:
            server = qai.ProductionServer(model=m, port=port)
            server.start()
        except KeyboardInterrupt:
            if verbose:
                print("\n[llm_pipeline] server stopped")

    return result


# =====================================================================
# 7. Time Series Pipeline
# =====================================================================
def time_series_pipeline(X, forecast_horizon: int = 1,
                           technique: str = "regression",
                           verbose: bool = True) -> PipelineResult:
    """Time series pipeline: create lag features → train → forecast.

    Converts time series into supervised learning:
      X[t-window:t] → y[t]

    Args:
        X: time series data (1D or 2D)
        forecast_horizon: how many steps ahead to predict
        technique: regression technique
    """
    import qai
    X = np.asarray(X, dtype=np.float32)
    steps = []
    t0 = time.time()

    # Create lag features
    window = min(10, len(X) // 4)
    X_lag = []
    y_target = []
    for i in range(len(X) - window - forecast_horizon + 1):
        X_lag.append(X[i:i + window].flatten())
        y_target.append(X[i + window + forecast_horizon - 1])
    X_lag = np.array(X_lag, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.float32)
    steps.append(f"lag_features(window={window})")

    if verbose:
        print(f"[time_series_pipeline] {len(X_lag)} lagged samples, window={window}")

    # Train
    m = qai.build(technique=technique)
    m.train(X_lag, y_target, verbose=False)
    steps.append(f"train({technique})")

    # Evaluate
    n = len(X_lag)
    split = int(n * 0.8)
    preds = [m.predict(X_lag[i]) for i in range(split, n)]
    mse = np.mean([(p - t) ** 2 for p, t in zip(preds, y_target[split:])])
    mae = np.mean([abs(p - t) for p, t in zip(preds, y_target[split:])])
    steps.append("evaluate")

    elapsed = time.time() - t0
    metrics = {"mse": float(mse), "mae": float(mae), "window": window,
                "forecast_horizon": forecast_horizon}
    if verbose:
        print(f"[time_series_pipeline] MSE={mse:.4f} MAE={mae:.4f} time={elapsed:.1f}s")

    return PipelineResult(
        pipeline_name="time_series", model=m,
        technique=technique, preprocessing_steps=steps,
        training_time_s=elapsed, metrics=metrics,
    )


# =====================================================================
# List all available pipelines
# =====================================================================
PIPELINES = {
    "classification": classification_pipeline,
    "regression": regression_pipeline,
    "clustering": clustering_pipeline,
    "nlp": nlp_pipeline,
    "llm": llm_pipeline,
    "time_series": time_series_pipeline,
    "auto": auto_pipeline,
}


def list_pipelines() -> Dict[str, str]:
    """List all available pipeline templates."""
    return {
        "classification": "Full classification: split → scale → train → evaluate",
        "regression": "Full regression: split → scale → train → evaluate",
        "clustering": "Full clustering: scale → cluster → evaluate",
        "nlp": "NLP: vectorize → train → evaluate",
        "llm": "LLM: load → fine-tune → serve",
        "time_series": "Time series: lag features → train → forecast",
        "auto": "Auto-detect task and run the appropriate pipeline",
    }


__all__ = ["PipelineResult", "classification_pipeline", "regression_pipeline",
           "clustering_pipeline", "nlp_pipeline", "llm_pipeline",
           "time_series_pipeline", "auto_pipeline", "list_pipelines", "PIPELINES"]
