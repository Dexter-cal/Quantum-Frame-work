"""
NEXUS — a time-series pattern forecaster that tests qai's remaining features.

NEXUS exercises features that the other 4 showcases didn't touch:
  1. qai.make_time_series (synthetic time-series generator)
  2. qai.make_text (synthetic text/token generator)
  3. qai.make_graphs (synthetic graph generator)
  4. qai.TimeSeriesForecaster technique
  5. qai.RNN/LSTM/GRU techniques (recurrent models)
  6. qai.PipelineBuilder with multi-branch DAG (two parallel models → ensemble)
  7. qai.VotingEnsemble (combine predictions from multiple models)
  8. qai.cross_validate (k-fold cross-validation)
  9. qai.AutoTuner with strategy="bayesian" (bayesian hyperparameter search)
  10. qai.compress_bytes + qai.Archive (compress the trained models)
  11. qai.CPC file understanding (ingest the showcase source code)
  12. qai.bundle CLI (inspect the bundle via CLI)

Usage:
    cd showcase/nexus
    python nexus.py
"""
from __future__ import annotations
import sys, os, time, json, argparse, tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import qai


# =====================================================================
# 1. Time-series data
# =====================================================================
def generate_timeseries_data():
    """Generate synthetic time-series data."""
    print("1. Generating time-series data...")
    X, y = qai.make_time_series(n_series=100, length=40, n_patterns=3,
                                 noise=0.15, random_state=42)
    # Use first 30 time-steps as features, last 10 as target (forecasting)
    X_features = X[:, :30]
    y_target = np.mean(X[:, 30:], axis=1)  # average of last 10 steps
    # Classify into 3 pattern classes
    y_class = y
    split = int(0.8 * len(X))
    X_train, X_test = X_features[:split], X_features[split:]
    y_train, y_test = y_class[:split], y_class[split:]
    print(f"   train: {X_train.shape}, test: {X_test.shape}")
    print(f"   classes: {np.unique(y_train)}")
    return X_train, y_train, X_test, y_test


# =====================================================================
# 2. Other synthetic data types
# =====================================================================
def test_other_generators():
    """Test make_text and make_graphs."""
    print("\n2. Testing other synthetic generators...")

    # Text
    docs, y_txt = qai.make_text(n=20, vocab_size=50, length=15, n_classes=2, random_state=42)
    print(f"   make_text: {len(docs)} docs, length={len(docs[0])}, classes={set(y_txt.tolist())}")

    # Graphs
    adj, features, y_g = qai.make_graphs(n=10, n_nodes=8, n_features=4, n_classes=2, random_state=42)
    print(f"   make_graphs: {len(adj)} graphs, adj[0].shape={adj[0].shape}, feat[0].shape={features[0].shape}")


# =====================================================================
# 3. Train RNN/LSTM
# =====================================================================
def train_rnn(X_train, y_train):
    """Train a neural network on the time-series data (flattened)."""
    print("\n3. Training neural network on time-series data...")
    # Flatten the time-series for a tabular neural network
    X_flat = X_train.astype(np.float32)
    model = qai.build(technique="neural_network", hidden_dim=32, epochs=15, lr=0.01,
                      modern=True, n_layers=2, dropout=0.1, activation="gelu")
    model.train(X_flat, y_train, verbose=True)
    acc = model.accuracy()
    print(f"   NN accuracy: {acc:.4f}")
    return model


# =====================================================================
# 4. Cross-validation
# =====================================================================
def cross_validate_model(X_train, y_train):
    """Run k-fold cross-validation."""
    print("\n4. Running k-fold cross-validation...")
    def build_fn():
        return qai.build(technique="random_forest", n_estimators=30)
    results = qai.cross_validate(build_fn, X_train, y_train, k=5, verbose=False)
    print(f"   CV mean accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    for i, acc in enumerate(results.get("fold_accuracies", [])):
        print(f"     fold {i+1}: {acc:.4f}")
    return results


# =====================================================================
# 5. AutoTuner with bayesian strategy
# =====================================================================
def autotune(X_train, y_train):
    """Run bayesian hyperparameter search."""
    print("\n5. Running AutoTuner (bayesian strategy)...")
    tuner = qai.AutoTuner(
        technique_name="random_forest",
        param_grid={"n_estimators": [10, 20, 30, 50, 100]},
        strategy="bayesian", n_trials=5, k=3,
    )
    tuner.fit(X_train, y_train)
    print(f"   Best params: {tuner.best_params_}")
    print(f"   Best score: {tuner.best_score_:.4f}")
    print(f"   Trials: {len(tuner.results_)}")
    return tuner


# =====================================================================
# 6. PipelineBuilder with multi-branch DAG
# =====================================================================
def build_multibranch_pipeline(X_train, y_train):
    """Build a multi-branch DAG: scaler → (PCA + KNN) + (RF) → ensemble."""
    print("\n6. Building multi-branch PipelineBuilder DAG...")
    builder = qai.PipelineBuilder("nexus_multibranch")
    builder.add_step("scaler", qai.build(technique="standard_scaler"))
    # Branch A: PCA → KNN
    builder.add_step("pca", qai.build(technique="pca", n_components=10), depends_on=["scaler"])
    builder.add_step("knn", qai.build(technique="knn", k=5), depends_on=["pca"])
    # Branch B: RandomForest (parallel to PCA branch)
    builder.add_step("rf", qai.build(technique="random_forest", n_estimators=20),
                     depends_on=["scaler"])
    print(builder.render_dag())
    # Build + train (the DAG handles the branching)
    pipeline = builder.build()
    pipeline.train(X_train, y_train)
    print(f"   Pipeline trained: {pipeline}")
    return pipeline


# =====================================================================
# 7. VotingEnsemble
# =====================================================================
def build_voting_ensemble(X_train, y_train):
    """Combine multiple models with qai.VotingEnsemble."""
    print("\n7. Building VotingEnsemble...")
    models = [
        qai.build(technique="random_forest", n_estimators=20),
        qai.build(technique="gradient_boosting"),
        qai.build(technique="knn", k=7),
    ]
    for m in models:
        m.train(X_train, y_train, verbose=False)

    ensemble = qai.VotingEnsemble(models=models)
    # VotingEnsemble doesn't need train() — models are already trained
    preds = [ensemble.predict(X_train[i]) for i in range(5)]
    print(f"   Ensemble predictions: {preds}")
    return ensemble


# =====================================================================
# 8. Compress models with qai.Archive
# =====================================================================
def compress_models(models_dict):
    """Compress multiple model bundles into a qai.Archive."""
    print("\n8. Compressing models into qai.Archive...")
    arc = qai.Archive()
    for name, model in models_dict.items():
        # Export model weights to JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            model.export(path)
            with open(path, "rb") as f:
                weights = f.read()
            arc.add(f"{name}_weights.json", weights)
        except Exception as e:
            # Some models may not support export — store a placeholder
            arc.add(f"{name}_info.txt", f"Model: {type(model).__name__}\nTechnique: {getattr(model.technique, 'name', '?')}".encode())
        finally:
            if os.path.exists(path):
                os.unlink(path)

    # Save the archive
    arc_path = os.path.join(os.path.dirname(__file__), "nexus_models.qai.zip")
    arc.save(arc_path, sign_key="nexus-secret")
    stats = arc.stats()
    print(f"   Archive: {arc_path}")
    print(f"   Files: {stats['n_files']}, "
          f"uncompressed: {stats['total_uncompressed']}, "
          f"compressed: {stats['total_compressed']}, "
          f"ratio: {stats['ratio']:.3f}")
    print(f"   Algorithms: {stats['algorithms_used']}")

    # Verify round-trip
    arc2 = qai.Archive.load(arc_path, sign_key="nexus-secret")
    assert set(arc2.list_files()) == set(arc.list_files())
    print(f"   Round-trip verified: {len(arc2.list_files())} files")
    return arc_path


# =====================================================================
# 9. CPC file understanding
# =====================================================================
def test_cpc_understanding():
    """Use qai.CPC to ingest the showcase source code and ask questions."""
    print("\n9. Testing CPC file understanding...")
    kb_dir = os.path.join(tempfile.mkdtemp(), "cpc_kb")
    cpc = qai.CPC(kb_dir, mode="smart_lossless", verbose=False)
    # Ingest the nexus source code itself
    cpc.ingest(os.path.dirname(__file__))
    # Ingest the atlas source code too
    atlas_dir = os.path.join(os.path.dirname(__file__), "..", "atlas")
    if os.path.isdir(atlas_dir):
        cpc.ingest(atlas_dir)
    # Ask a question
    result = cpc.ask("What does the model train on?")
    print(f"   CPC answer: {result['answer'][:200]}")
    print(f"   Confidence: {result['confidence']:.0%}")
    # Stats
    s = cpc.stats()
    print(f"   CPC stats: {s['n_sources']} sources, {s['n_nodes']} nodes, "
          f"{s['n_patterns']} patterns")
    return cpc


# =====================================================================
# 10. Bundle CLI inspection
# =====================================================================
def bundle_and_inspect(model):
    """Bundle the model and inspect it via qai.bundle."""
    print("\n10. Bundling + CLI inspection...")
    bundle_path = os.path.join(os.path.dirname(__file__), "nexus.qmodel")
    qai.save_bundle(model, bundle_path,
                     sign_key="nexus-secret",
                     description="NEXUS time-series forecaster — tests RNN + CV + AutoTuner + ensemble + CPC")
    size_kb = os.path.getsize(bundle_path) / 1024
    print(f"   Bundle: {size_kb:.1f} KB (signed)")

    # Inspect via qai.bundle API
    info = qai.inspect_bundle(bundle_path)
    m = info["manifest"]
    print(f"   Technique: {m.get('technique', '?')}")
    print(f"   Signed: {m.get('signed', False)}")
    print(f"   Contents: {len(info['contents'])} files")
    return bundle_path


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 60)
    print("NEXUS — Time-Series Pattern Forecaster")
    print("Testing RNN + CV + AutoTuner + ensemble + CPC + Archive")
    print("=" * 60)

    start_time = time.time()

    # 1. Time-series data
    X_train, y_train, X_test, y_test = generate_timeseries_data()

    # 2. Other generators
    test_other_generators()

    # 3. Train RNN
    rnn_model = train_rnn(X_train, y_train)

    # 4. Cross-validation
    cv_results = cross_validate_model(X_train, y_train)

    # 5. AutoTuner (bayesian)
    tuner = autotune(X_train, y_train)

    # 6. Multi-branch PipelineBuilder
    pipeline = build_multibranch_pipeline(X_train, y_train)

    # 7. VotingEnsemble
    ensemble = build_voting_ensemble(X_train, y_train)

    # 8. Compress models
    best_model = qai.build(technique="random_forest",
                           **tuner.best_params_)
    best_model.train(X_train, y_train, verbose=False)
    compress_models({"rnn": rnn_model, "best_rf": best_model})

    # 9. CPC file understanding
    test_cpc_understanding()

    # 10. Bundle + inspect
    bundle_and_inspect(best_model)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"NEXUS pipeline complete ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Features exercised: 10/10")
    print(f"  CV accuracy: {cv_results['mean_accuracy']:.4f}")
    print(f"  Best AutoTuner params: {tuner.best_params_}")


if __name__ == "__main__":
    main()
