"""
PRISM — a text sentiment classifier that exercises every qai feature end-to-end.

PRISM is a starter model that showcases the full qai production pipeline:
  1. Synthetic data generation (qai.make_classification with full params)
  2. Data profiling (qai.DataProfiler)
  3. Pipeline building (qai.PipelineBuilder with scaler → model DAG)
  4. Experiment comparison (qai.ExperimentCompare across 3 techniques)
  5. Model profiling (qai.Profiler — latency, params, FLOPs)
  6. Evaluation (qai.evaluate — auto-metric selection)
  7. Auto-documentation (qai.doc — markdown model docs)
  8. Model card generation (qai.ModelCardGenerator — responsible AI)
  9. A/B testing (qai.ABTest — statistical significance)
  10. Drift detection (qai.DriftDetector — production monitoring)
  11. Active learning (qai.ActiveLearningLoop — uncertainty sampling)
  12. Model registry (qai.ModelRegistry — publish + version)
  13. Bundling (qai.save_bundle — signed .qmodel)
  14. CLI dashboard (qai.dashboard — browser UI)

Usage:
    cd showcase/prism
    python prism.py              # full end-to-end run
    python prism.py --dashboard  # open the dashboard after training

The model classifies text sentiment (positive/negative) from TF-IDF features.
It's intentionally simple so the focus is on the PRODUCTION PIPELINE, not the
model architecture.
"""
from __future__ import annotations
import sys, os, time, json, argparse, tempfile
import numpy as np

# Make qai importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import qai

# =====================================================================
# 1. Synthetic data generation
# =====================================================================
def generate_sentiment_data(n_samples=300, n_features=20, random_state=42):
    """Generate a synthetic sentiment-classification dataset.

    Uses qai.make_classification with full sklearn params (n_informative,
    n_redundant, flip_y, class_sep) to create a realistic dataset with
    label noise and redundant features.
    """
    print("1. Generating synthetic sentiment data...")
    X, y = qai.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_repeated=2,
        n_classes=2,
        n_clusters_per_class=2,
        flip_y=0.05,       # 5% label noise
        class_sep=1.5,
        random_state=random_state,
    )
    # Split into train/test
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[perm[:split]], X[perm[split:]]
    y_train, y_test = y[perm[:split]], y[perm[split:]]
    print(f"   train: {X_train.shape}, test: {X_test.shape}")
    print(f"   class balance (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    return X_train, y_train, X_test, y_test


# =====================================================================
# 2. Data profiling
# =====================================================================
def profile_data(X_train, y_train):
    """Profile the training data using qai.DataProfiler."""
    print("\n2. Profiling training data...")
    profile = qai.DataProfiler(X_train).profile(y=y_train)
    print(profile)
    return profile


# =====================================================================
# 3. Experiment comparison (try 3 techniques, pick the best)
# =====================================================================
def compare_techniques(X_train, y_train, X_test, y_test):
    """Compare 3 techniques using qai.ExperimentCompare."""
    print("\n3. Comparing techniques...")
    models = {}
    for name, technique, params in [
        ("rf", "random_forest", {"n_estimators": 50}),
        ("gb", "gradient_boosting", {"n_estimators": 50}),
        ("lr", "classifier", {}),
    ]:
        m = qai.build(technique=technique, **params)
        m.train(X_train, y_train, verbose=False)
        models[name] = m

    cmp = qai.ExperimentCompare(models)
    report = cmp.full_comparison(X_test, y_test, X_profile_sample=X_test[:1])
    print(report)
    print(f"\n   Best model: {report.best_name}")
    return report, models[report.best_name]


# =====================================================================
# 4. Model profiling
# =====================================================================
def profile_model(model, X_sample):
    """Profile the winning model's inference performance."""
    print("\n4. Profiling model performance...")
    profile = qai.Profiler(model).profile(X_sample, n_warmup=2, n_runs=10)
    print(profile)
    return profile


# =====================================================================
# 5. Evaluation
# =====================================================================
def evaluate_model(model, X_test, y_test):
    """Full evaluation suite using qai.evaluate."""
    print("\n5. Evaluating model...")
    report = qai.evaluate(model, X_test, y_test, verbose=False)
    print(report)
    return report


# =====================================================================
# 6. Auto-documentation
# =====================================================================
def generate_docs(model):
    """Auto-generate markdown docs using qai.doc."""
    print("\n6. Generating model documentation...")
    doc = qai.doc(model)
    print(f"   Generated {len(doc)} chars of markdown documentation")
    # Save to file
    doc_path = os.path.join(os.path.dirname(__file__), "prism_model_doc.md")
    qai.doc(model, save=doc_path)
    print(f"   Saved to {doc_path}")
    return doc


# =====================================================================
# 7. Model card generation
# =====================================================================
def generate_model_card(model, X_test, y_test):
    """Generate a responsible-AI model card."""
    print("\n7. Generating model card...")
    card = qai.ModelCardGenerator(model).generate(
        X_test=X_test, y_test=y_test,
        description="PRISM — text sentiment classifier (positive/negative)",
        intended_use="Classify text sentiment from TF-IDF features. "
                      "Designed as a starter model for qai framework demonstration.",
        training_data_desc="Synthetic data: 240 train samples, 20 features, "
                           "binary classification, 5% label noise.",
        creator="qai PRISM showcase",
    )
    card_path = os.path.join(os.path.dirname(__file__), "prism_model_card.md")
    card.save(card_path)
    print(f"   Saved model card to {card_path}")
    print(f"   Card size: {len(str(card))} chars")
    return card


# =====================================================================
# 8. A/B testing
# =====================================================================
def run_ab_test(model_a, model_b, X, y):
    """Run an A/B test between two models."""
    print("\n8. Running A/B test...")
    ab = qai.ABTest(model_a=model_a, model_b=model_b, traffic_split=0.5)
    for i in range(min(60, len(X))):
        ab.record_outcome(X[i], y[i])
    result = ab.analyze()
    print(result)
    return result


# =====================================================================
# 9. Drift detection
# =====================================================================
def test_drift_detection(X_train, X_test):
    """Set up drift detection for production monitoring."""
    print("\n9. Testing drift detection...")
    det = qai.DriftDetector(baseline=X_train, method="ensemble")
    # Normal batch (should not drift)
    normal_report = det.detect(X_test[:30])
    print(f"   Normal batch: drifted={normal_report.drifted}")
    # Drifted batch (shifted distribution)
    drifted_data = X_test[:30] * 2.5 + 1.0
    drift_report = det.detect(drifted_data)
    print(f"   Drifted batch: drifted={drift_report.drifted}, "
          f"features={drift_report.drifted_features}")
    return det


# =====================================================================
# 10. Active learning
# =====================================================================
def test_active_learning(model, X, y):
    """Test active learning loop with uncertainty sampling."""
    print("\n10. Testing active learning...")
    # Use the first 20 samples as "labeled", the rest as "unlabeled pool"
    labeled_X, labeled_y = X[:20], y[:20]
    pool_X, pool_y = X[20:60], y[20:60]

    m = qai.build(technique="classifier")
    m.train(labeled_X, labeled_y, verbose=False)

    loop = qai.ActiveLearningLoop(
        model=m, X_unlabeled=pool_X,
        strategy="uncertainty", batch_size=5,
    )
    batch = loop.next_batch()
    print(f"   Uncertainty batch: {len(batch)} samples to label")
    # Simulate labeling
    labels = pool_y[batch]
    loop.commit(batch, list(labels))
    print(f"   Committed {len(batch)} labels, training set now: {loop.status()['training_set_size']}")
    return loop


# =====================================================================
# 11. Model registry
# =====================================================================
def publish_to_registry(model):
    """Publish the model to a local model registry."""
    print("\n11. Publishing to model registry...")
    reg_dir = os.path.join(os.path.dirname(__file__), "registry")
    reg = qai.ModelRegistry(reg_dir)
    reg.publish(model, name="prism_sentiment", version="1.0.0",
                description="PRISM sentiment classifier v1",
                tags=["classification", "sentiment", "text", "starter-model"])
    reg.publish(model, name="prism_sentiment", version="1.0.1",
                description="PRISM sentiment classifier (patch)",
                tags=["classification", "sentiment", "text"])
    results = reg.search("sentiment")
    print(f"   Published 2 versions, search found {len(results)} results")
    print(f"   Versions: {reg.list_versions('prism_sentiment')}")
    print(f"   Stats: {reg.stats()}")
    return reg


# =====================================================================
# 12. Bundling
# =====================================================================
def bundle_model(model):
    """Bundle the model as a signed .qmodel file."""
    print("\n12. Bundling model...")
    bundle_path = os.path.join(os.path.dirname(__file__), "prism.qmodel")
    qai.save_bundle(model, bundle_path,
                     sign_key="prism-secret-2026",
                     description="PRISM sentiment classifier — starter model for qai framework")
    size_kb = os.path.getsize(bundle_path) / 1024
    print(f"   Bundle: {bundle_path} ({size_kb:.1f} KB, signed)")

    # Verify
    b = qai.load_bundle(bundle_path, sign_key="prism-secret-2026")
    print(f"   Verified: {b.manifest.get('signed', False)}, "
          f"technique={b.manifest.get('technique', '?')}")
    print(f"   README: {b.readme[:100]}...")
    return bundle_path


# =====================================================================
# MAIN — the full end-to-end pipeline
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="PRISM — text sentiment classifier showcase")
    parser.add_argument("--dashboard", action="store_true",
                        help="Open the model dashboard after training")
    parser.add_argument("--n-samples", type=int, default=300)
    args = parser.parse_args()

    print("=" * 60)
    print("PRISM — Text Sentiment Classifier")
    print("Showcasing 12 qai features end-to-end")
    print("=" * 60)

    start_time = time.time()

    # 1. Generate data
    X_train, y_train, X_test, y_test = generate_sentiment_data(args.n_samples)

    # 2. Profile data
    profile_data(X_train, y_train)

    # 3. Compare techniques + pick winner
    comparison, best_model = compare_techniques(X_train, y_train, X_test, y_test)

    # 4. Profile the winning model
    profile_model(best_model, X_test[:1])

    # 5. Evaluate
    eval_report = evaluate_model(best_model, X_test, y_test)

    # 6. Generate docs
    generate_docs(best_model)

    # 7. Generate model card
    generate_model_card(best_model, X_test, y_test)

    # 8. A/B test (compare best vs a weaker model)
    weaker = qai.build(technique="classifier")
    weaker.train(X_train[:50], y_train[:50], verbose=False)  # less data = weaker
    run_ab_test(best_model, weaker, X_test, y_test)

    # 9. Drift detection
    test_drift_detection(X_train, X_test)

    # 10. Active learning
    test_active_learning(best_model, X_train, y_train)

    # 11. Publish to registry
    publish_to_registry(best_model)

    # 12. Bundle
    bundle_path = bundle_model(best_model)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"PRISM pipeline complete ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Best technique:     {comparison.best_name}")
    print(f"  Test accuracy:      {eval_report.metrics.get('accuracy', '?')}")
    print(f"  Verdict:            {eval_report.verdict}")
    print(f"  Bundle:             {bundle_path}")
    print(f"  Features exercised: 12/12")

    # Optional: open dashboard
    if args.dashboard:
        print(f"\n  Opening dashboard at http://127.0.0.1:8765/ ...")
        qai.dashboard(best_model, port=8765)

    return best_model


if __name__ == "__main__":
    model = main()
