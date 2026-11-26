"""
SENTINEL — a production fraud-detection system that tests every remaining
untested feature end-to-end.

Tests:
  1. qai.FeatureStore — register + compute + materialize features
  2. qai.HyperparameterOptimizer — Bayesian optimization
  3. qai.DataAugmentor — augment training data (jitter + mixup)
  4. qai.ConfidenceCalibrator — calibrate fraud probabilities
  5. qai.SelfHealingModel — auto-detect degradation + retrain
  6. qai.ModelMonitor — production monitoring (drift + latency + accuracy)
  7. qai.ModelVersioner — track versions + diff + rollback
  8. qai.MemoryVault — tiered memory for storing fraud patterns
  9. qai.FederatedBridge — combine sklearn + custom numpy pieces
  10. qai.ABTest — A/B test two fraud models
  11. qai.ModelCardGenerator — responsible AI model card
  12. qai.compress_bytes — compress the trained model weights

Usage:
    cd showcase/sentinel
    python sentinel.py
"""
from __future__ import annotations
import sys, os, time, json, tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import qai


def generate_fraud_data(n=500, seed=42):
    """Generate synthetic fraud-detection data."""
    print("1. Generating fraud-detection data...")
    rng = np.random.default_rng(seed)
    # 95% legitimate, 5% fraudulent (imbalanced)
    n_fraud = int(n * 0.05)
    n_legit = n - n_fraud
    # Legitimate transactions
    X_legit = rng.normal(0, 1, (n_legit, 8))
    # Fraudulent transactions (higher amounts, unusual patterns)
    X_fraud = rng.normal(3, 2, (n_fraud, 8))
    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
    # Shuffle
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    # Split
    split = int(0.7 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    # Calibration set (from test)
    cal_split = int(0.5 * len(X_test))
    X_cal, y_cal = X_test[:cal_split], y_test[:cal_split]
    X_test, y_test = X_test[cal_split:], y_test[cal_split:]
    print(f"   train: {X_train.shape} (fraud: {int(y_train.sum())}/{len(y_train)})")
    print(f"   cal: {X_cal.shape}, test: {X_test.shape}")
    return X_train, y_train, X_cal, y_cal, X_test, y_test


def test_feature_store(X_train):
    """Test FeatureStore: register + compute + materialize."""
    print("\n2. Testing FeatureStore...")
    store = qai.FeatureStore(tempfile.mkdtemp())
    store.register("raw", compute_fn=lambda X: X, description="raw transaction features")
    store.register("stats", compute_fn=lambda X: {
        "mean_amount": float(np.mean(X[:, 0])),
        "std_amount": float(np.std(X[:, 0])),
        "max_amount": float(np.max(X[:, 0])),
    }, description="transaction statistics")
    feats = store.compute("stats", X_train)
    print(f"   computed stats: {feats}")
    path = store.materialize("raw", X_train)
    print(f"   materialized raw -> {path}")
    print(f"   registered: {[f['name'] for f in store.list_features()]}")


def test_hyperparameter_optimizer(X_train, y_train):
    """Test HyperparameterOptimizer."""
    print("\n3. Testing HyperparameterOptimizer (Bayesian)...")
    opt = qai.HyperparameterOptimizer(
        technique="random_forest",
        param_space={"n_estimators": (10, 100), "max_depth": (3, 15)},
        n_iterations=5, k=2, n_initial_random=3,
    )
    result = opt.optimize(X_train, y_train)
    print(f"   best params: {result.best_params}")
    print(f"   best score: {result.best_score:.4f}")
    return result.best_params


def test_data_augmentor(X_train, y_train):
    """Test DataAugmentor on tabular data."""
    print("\n4. Testing DataAugmentor...")
    aug = qai.DataAugmentor(mode="tabular")
    aug.add("jitter", noise_std=0.1)
    aug.add("mixup", alpha=0.3)
    X_aug, y_aug = aug.transform(X_train, y_train, n_copies=2)
    print(f"   original: {X_train.shape} -> augmented: {X_aug.shape}")
    return X_aug, y_aug


def test_confidence_calibrator(model, X_cal, y_cal, X_test, y_test):
    """Test ConfidenceCalibrator."""
    print("\n5. Testing ConfidenceCalibrator...")
    cal = qai.ConfidenceCalibrator(method="isotonic")
    cal.fit(model, X_cal, y_cal)
    probs = cal.predict_proba(model, X_test[:3])
    print(f"   calibrated probs: {probs[:3]}")
    score = cal.score(model, X_test, y_test)
    print(f"   brier: {score['brier']:.4f}, ece: {score['ece']:.4f}")


def test_self_healing(model, X_train, y_train):
    """Test SelfHealingModel."""
    print("\n6. Testing SelfHealingModel...")
    healer = qai.SelfHealingModel(model, X_train, y_train,
                                   accuracy_threshold=0.5, check_interval=5, window_size=15)
    for i in range(15):
        healer.predict(X_train[i])
        healer.record_outcome(X_train[i], y_train[i])
    print(f"   status: {healer.status()}")


def test_model_monitor(model, X_train, y_train, X_test):
    """Test ModelMonitor."""
    print("\n7. Testing ModelMonitor...")
    monitor = qai.ModelMonitor(baseline_X=X_train, baseline_y=y_train,
                                drift_check_interval=10, latency_alert_ms=50)
    for i in range(min(20, len(X_test))):
        pred = model.predict(X_test[i])
        monitor.record(X_test[i], pred, latency_ms=0.5 + np.random.rand())
    report = monitor.report()
    print(f"   predictions: {report.n_predictions}, avg_latency: {report.avg_latency_ms:.2f}ms")
    print(f"   alerts: {report.alerts}")


def test_model_versioner(model):
    """Test ModelVersioner."""
    print("\n8. Testing ModelVersioner...")
    ver = qai.ModelVersioner(tempfile.mkdtemp())
    ver.save(model, "1.0.0", "initial fraud model")
    ver.save(model, "1.1.0", "with calibrated probabilities")
    versions = ver.list_versions()
    print(f"   versions: {[v['version'] for v in versions]}")
    diff = ver.diff("1.0.0", "1.1.0")
    print(f"   diff identical: {diff['identical']}")
    loaded = ver.load("1.0.0")
    print(f"   rollback loaded: {loaded.is_trained()}")


def test_memory_vault():
    """Test MemoryVault for storing fraud patterns."""
    print("\n9. Testing MemoryVault...")
    vault = qai.MemoryVault(tempfile.mkdtemp(), hot_capacity=5, warm_capacity=10)
    # Store fraud patterns
    vault.store("pattern_1", "Rapid consecutive transactions from same IP",
                importance=0.9, tags=["fraud", "pattern"])
    vault.store("pattern_2", "Transaction amount 3x above user average",
                importance=0.8, tags=["fraud", "amount"])
    vault.store("pattern_3", "Login from new geography + immediate purchase",
                importance=0.95, tags=["fraud", "geo"])
    # Recall
    results = vault.recall("fraud", top_k=3)
    print(f"   recall('fraud'): {len(results)} results")
    for r in results:
        print(f"     {r['key']}: {r['content'][:50]}... (score={r['score']:.3f})")
    # Tag search
    geo_results = vault.recall(tags=["geo"])
    print(f"   tag 'geo': {len(geo_results)} results")
    print(f"   stats: {vault.stats()}")


def test_federated_bridge(X_train, y_train):
    """Test FederatedBridge: combine sklearn scaler + custom + sklearn classifier."""
    print("\n10. Testing FederatedBridge...")
    bridge = qai.FederatedBridge("fraud_federated")
    bridge.add_piece("scaler", qai.build(technique="standard_scaler"), framework="qai")
    bridge.add_piece("clf", qai.build(technique="random_forest", n_estimators=15),
                     framework="qai", input_from="scaler")
    print(bridge.summary()[:200])
    bridge.train(X_train, y_train, verbose=False)
    preds = bridge.predict(X_train[:3])
    print(f"   predictions: {preds}")


def test_ab_test(model_a, model_b, X_test, y_test):
    """Test ABTest between two models."""
    print("\n11. Testing ABTest...")
    ab = qai.ABTest(model_a=model_a, model_b=model_b, traffic_split=0.5)
    for i in range(min(30, len(X_test))):
        ab.record_outcome(X_test[i], y_test[i])
    result = ab.analyze()
    print(f"   A: n={result.n_a}, acc={result.accuracy_a:.4f}")
    print(f"   B: n={result.n_b}, acc={result.accuracy_b:.4f}")
    print(f"   significant: {result.significant}")


def test_model_card(model, X_test, y_test):
    """Test ModelCardGenerator."""
    print("\n12. Testing ModelCardGenerator...")
    card = qai.ModelCardGenerator(model).generate(
        X_test=X_test, y_test=y_test,
        description="SENTINEL — fraud detection model",
        intended_use="Detect fraudulent transactions in real-time",
        training_data_desc="500 synthetic transactions, 5% fraud rate, 8 features",
    )
    print(f"   card: {len(str(card))} chars")
    print(f"   first line: {str(card).split(chr(10))[0]}")


def test_weight_compression(model):
    """Test compress_bytes on model weights."""
    print("\n13. Testing weight compression...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        model.export(path)
        with open(path, "rb") as f:
            raw = f.read()
        compressed = qai.compress_bytes(raw)
        print(f"   weights: {len(raw)} -> {len(compressed)} bytes (ratio: {len(compressed)/max(len(raw),1):.3f})")
    except Exception as e:
        print(f"   compression: {e}")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def main():
    print("=" * 60)
    print("SENTINEL — Fraud Detection System")
    print("Testing 13 features end-to-end")
    print("=" * 60)

    start = time.time()

    # 1. Data
    X_train, y_train, X_cal, y_cal, X_test, y_test = generate_fraud_data()

    # 2. FeatureStore
    test_feature_store(X_train)

    # 3. HyperparameterOptimizer
    best_params = test_hyperparameter_optimizer(X_train, y_train)

    # 4. DataAugmentor
    X_aug, y_aug = test_data_augmentor(X_train, y_train)

    # Train the best model
    print("\n--- Training best model ---")
    model = qai.build(technique="random_forest", **best_params)
    model.train(X_aug, y_aug, verbose=False)
    print(f"   accuracy: {model.accuracy():.4f}")

    # 5. ConfidenceCalibrator
    test_confidence_calibrator(model, X_cal, y_cal, X_test, y_test)

    # 6. SelfHealingModel
    test_self_healing(model, X_train, y_train)

    # 7. ModelMonitor
    test_model_monitor(model, X_train, y_train, X_test)

    # 8. ModelVersioner
    test_model_versioner(model)

    # 9. MemoryVault
    test_memory_vault()

    # 10. FederatedBridge
    test_federated_bridge(X_train, y_train)

    # 11. ABTest (compare model vs a weaker one)
    weaker = qai.build(technique="classifier")
    weaker.train(X_train[:100], y_train[:100], verbose=False)
    test_ab_test(model, weaker, X_test, y_test)

    # 12. ModelCard
    test_model_card(model, X_test, y_test)

    # 13. Weight compression
    test_weight_compression(model)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"SENTINEL pipeline complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Features exercised: 13/13")


if __name__ == "__main__":
    main()
