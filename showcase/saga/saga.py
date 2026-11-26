"""
SAGA — a production ML lifecycle showcase that tests the newest qai features.

SAGA exercises 8 features that the other 6 showcases didn't fully test:
  1. qai.FeatureStore — register, compute, materialize, cache features
  2. qai.HyperparameterOptimizer — Bayesian GP optimization
  3. qai.ModelVersioner — save versions, diff, rollback
  4. qai.ConfidenceCalibrator — Platt + isotonic calibration
  5. qai.FederatedBridge — cross-framework model (qai + custom callable)
  6. qai.SelfHealingModel — auto-detect degradation + retrain
  7. qai.ModelMonitor — production monitoring (latency + accuracy + drift)
  8. qai.DataAugmentor — tabular augmentation (jitter + mixup)

Usage:
    cd showcase/saga
    python saga.py
"""
from __future__ import annotations
import sys, os, time, tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import qai


def generate_data():
    print("1. Generating data...")
    X, y = qai.make_classification(
        n_samples=200, n_features=10, n_informative=6, n_redundant=2,
        flip_y=0.05, class_sep=1.5, random_state=42,
    )
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    split = int(0.7 * len(X))
    X_train, X_test = X[perm[:split]], X[perm[split:]]
    y_train, y_test = y[perm[:split]], y[perm[split:]]
    X_cal, y_cal = X_train[:30], y_train[:30]
    print(f"   train={X_train.shape}, test={X_test.shape}, cal={X_cal.shape}")
    return X_train, y_train, X_test, y_test, X_cal, y_cal


def test_feature_store(X_train):
    print("\n2. FeatureStore...")
    store = qai.FeatureStore(tempfile.mkdtemp())
    store.register("raw", compute_fn=lambda X: X, description="raw features")
    store.register("stats", compute_fn=lambda X: {
        "mean_0": float(np.mean(X[:, 0])), "std_0": float(np.std(X[:, 0])),
    }, dependencies=[], description="aggregate stats")
    feats = store.compute("raw", X_train)
    stats = store.compute("stats", X_train)
    assert feats is not None
    assert "mean_0" in stats
    cached = store.get("raw")
    assert cached is not None
    path = store.materialize("stats", X_train)
    assert os.path.exists(path)
    assert len(store.list_features()) == 2
    print(f"   registered 2 groups, materialized stats, {len(store.list_features())} features")
    return store


def test_hyperparam_opt(X_train, y_train):
    print("\n3. HyperparameterOptimizer (Bayesian)...")
    opt = qai.HyperparameterOptimizer(
        technique="random_forest",
        param_space={"n_estimators": (10, 80), "max_depth": (3, 15)},
        n_iterations=6, k=2, n_initial_random=3,
    )
    result = opt.optimize(X_train, y_train)
    print(f"   best params: {result.best_params}, score: {result.best_score:.4f}")
    print(f"   iterations: {result.n_iterations}, history: {len(result.history)}")
    return result


def test_model_versioner(X_train, y_train):
    print("\n4. ModelVersioner...")
    ver = qai.ModelVersioner(tempfile.mkdtemp())
    m1 = qai.build(technique="random_forest", n_estimators=10)
    m1.train(X_train, y_train, verbose=False)
    ver.save(m1, "1.0.0", "baseline")
    m2 = qai.build(technique="random_forest", n_estimators=50)
    m2.train(X_train, y_train, verbose=False)
    ver.save(m2, "2.0.0", "more trees")
    versions = ver.list_versions()
    diff = ver.diff("1.0.0", "2.0.0")
    loaded = ver.load("1.0.0")
    print(f"   versions: {[v['version'] for v in versions]}")
    print(f"   diff identical: {diff['identical']}, size_diff: {diff['size_diff_bytes']}")
    print(f"   loaded v1.0.0: {loaded.is_trained()}")
    return ver


def test_confidence_calibrator(X_train, y_train, X_cal, y_cal, X_test):
    print("\n5. ConfidenceCalibrator...")
    m = qai.build(technique="random_forest", n_estimators=20)
    m.train(X_train, y_train, verbose=False)
    for method in ("isotonic", "platt"):
        cal = qai.ConfidenceCalibrator(method=method)
        cal.fit(m, X_cal, y_cal)
        probs = cal.predict_proba(m, X_test[:5])
        score = cal.score(m, X_cal, y_cal)
        print(f"   {method}: brier={score['brier']:.4f}, ece={score['ece']:.4f}")
    return m


def test_federated_bridge(X_train, y_train, X_test):
    print("\n6. FederatedBridge (cross-framework)...")
    bridge = qai.FederatedBridge("saga_model")
    bridge.add_piece("scaler", qai.build(technique="standard_scaler"), framework="qai")
    bridge.add_piece("flatten", lambda x: np.asarray(x).reshape(len(x), -1) if x.ndim > 2 else x,
                     framework="custom", input_from="scaler")
    bridge.add_piece("clf", qai.build(technique="random_forest", n_estimators=15),
                     framework="qai", input_from="flatten")
    bridge.train(X_train, y_train, verbose=False)
    preds = bridge.predict(X_test[:5])
    print(f"   bridge predictions: {preds}")
    print(f"   pieces: {[p.name for p in bridge.pieces.values()]}")
    return bridge


def test_self_healing(X_train, y_train):
    print("\n7. SelfHealingModel...")
    m = qai.build(technique="random_forest", n_estimators=10)
    m.train(X_train, y_train, verbose=False)
    healer = qai.SelfHealingModel(m, X_train, y_train,
                                   accuracy_threshold=0.3, check_interval=5, window_size=20)
    for i in range(20):
        healer.predict(X_train[i])
        healer.record_outcome(X_train[i], y_train[i])
    s = healer.status()
    print(f"   status: accuracy={s['current_accuracy']:.3f}, healthy={s['healthy']}, retrains={s['n_retrains']}")
    events = healer.event_log()
    print(f"   events: {len(events)}")
    return healer


def test_model_monitor(X_train, y_train, X_test, y_test):
    print("\n8. ModelMonitor...")
    m = qai.build(technique="random_forest", n_estimators=10)
    m.train(X_train, y_train, verbose=False)
    monitor = qai.ModelMonitor(baseline_X=X_train, baseline_y=y_train,
                                drift_check_interval=10, latency_alert_ms=50)
    for i in range(min(30, len(X_test))):
        pred = m.predict(X_test[i])
        monitor.record(X_test[i], pred, y_test[i], latency_ms=0.5 + np.random.rand() * 2)
    report = monitor.report()
    print(f"   predictions: {report.n_predictions}")
    print(f"   avg latency: {report.avg_latency_ms:.2f}ms")
    print(f"   accuracy: {report.current_accuracy:.4f}")
    print(f"   trend: {report.accuracy_trend}")
    print(f"   alerts: {len(report.alerts)}")
    return monitor


def test_data_augmentor(X_train, y_train):
    print("\n9. DataAugmentor...")
    aug = qai.DataAugmentor(mode="tabular", seed=42)
    aug.add("jitter", noise_std=0.05)
    aug.add("mixup", alpha=0.2)
    X_aug, y_aug = aug.transform(X_train[:30], y_train[:30], n_copies=2)
    print(f"   original: {X_train[:30].shape} -> augmented: {X_aug.shape}")
    # Train on augmented data
    m = qai.build(technique="random_forest", n_estimators=10)
    m.train(X_aug, y_aug, verbose=False)
    print(f"   trained on augmented data: accuracy={m.accuracy():.4f}")
    return aug


def main():
    print("=" * 60)
    print("SAGA — Production ML Lifecycle Showcase")
    print("Testing 8 newest qai features end-to-end")
    print("=" * 60)

    start_time = time.time()

    X_train, y_train, X_test, y_test, X_cal, y_cal = generate_data()
    test_feature_store(X_train)
    test_hyperparam_opt(X_train, y_train)
    test_model_versioner(X_train, y_train)
    test_confidence_calibrator(X_train, y_train, X_cal, y_cal, X_test)
    test_federated_bridge(X_train, y_train, X_test)
    test_self_healing(X_train, y_train)
    test_model_monitor(X_train, y_train, X_test, y_test)
    test_data_augmentor(X_train, y_train)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"SAGA pipeline complete ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Features exercised: 8/8")


if __name__ == "__main__":
    main()
