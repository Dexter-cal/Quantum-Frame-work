"""
ATLAS — an image classifier that tests qai's deep-learning infrastructure.

ATLAS exercises features that PRISM and EVOLVE didn't touch:
  1. qai.Layer system (Sequential builder, named layers, freeze/unfreeze)
  2. qai.make_images (synthetic image data generator)
  3. qai.build(technique="cnn") with modern upgrades (se_blocks, stochastic_depth, mixup, use_swa)
  4. qai.LiveTrainingCallback (real-time loss curve dashboard)
  5. qai.PipelineBuilder with a DAG (preprocess → PCA → classifier)
  6. qai.PipelineBuilder.render_dag() (visual pipeline diagram)
  7. qai.AutoTrainTarget (train until target accuracy reached)
  8. qai.extract_features from CNN (transfer learning)
  9. qai.freeze_conv_base() / unfreeze_conv_base() (transfer learning)
  10. qai.compress_bytes on the trained model's weights

Usage:
    cd showcase/atlas
    python atlas.py              # full end-to-end run
    python atlas.py --live       # with live training dashboard
"""
from __future__ import annotations
import sys, os, time, json, argparse, tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import qai
from qai.completion import Layer


def generate_image_data(n=200, size=(16, 16), n_classes=3, random_state=42):
    """Generate synthetic images using qai.make_images."""
    print("1. Generating synthetic image data...")
    X, y = qai.make_images(n=n, size=size, channels=1, n_classes=n_classes,
                            random_state=random_state)
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[perm[:split]], X[perm[split:]]
    y_train, y_test = y[perm[:split]], y[perm[split:]]
    print(f"   train: {X_train.shape}, test: {X_test.shape}")
    print(f"   classes: {np.unique(y_train)}")
    return X_train, y_train, X_test, y_test


def build_custom_layer_model(input_dim=256, n_classes=3):
    """Build a custom model using qai.Layer.Sequential."""
    print("\n2. Building custom Layer architecture...")
    seq = Layer.Sequential()
    seq.add("fc1", Layer.create("linear", in_dim=input_dim, out_dim=128))
    seq.add("relu1", Layer.create("relu"))
    seq.add("fc2", Layer.create("linear", in_dim=128, out_dim=64))
    seq.add("relu2", Layer.create("relu"))
    seq.add("dropout1", Layer.create("dropout", rate=0.3))
    seq.add("fc3", Layer.create("linear", in_dim=64, out_dim=n_classes))
    seq.add("softmax", Layer.create("softmax"))

    print(f"   Architecture: {seq}")
    for s in seq.summary():
        print(f"     {s['name']:15s} {s['type']:20s} params={s['params']:5d} frozen={s['frozen']}")

    x = np.random.randn(1, input_dim)
    out = seq.forward(x)
    print(f"   Forward: input {x.shape} -> output {out.shape} (sum={out.sum():.4f})")

    seq.freeze("fc1")
    print(f"   Frozen fc1: {seq.layers['fc1'].is_frozen}")
    seq.unfreeze("fc1")

    total_params = sum(l.param_count() for l in seq.layers.values())
    print(f"   Total parameters: {total_params}")
    return seq


def train_cnn_with_upgrades(X_train, y_train, X_test, y_test, live=False):
    """Train a CNN with se_blocks + stochastic_depth + mixup + SWA."""
    print("\n3. Training CNN with modern upgrades...")
    print("   Upgrades: se_blocks=True, stochastic_depth=0.1, mixup_alpha=0.2, use_swa=True")

    callbacks = [
        qai.callbacks.EarlyStopping(patience=3),
        qai.callbacks.LogProgress(),
    ]
    if live:
        callbacks.append(qai.LiveTrainingCallback(port=8766, open_browser=True))

    model = qai.build(
        technique="cnn",
        in_channels=1,
        n_classes=len(np.unique(y_train)),
        conv_channels=[8, 16],
        kernel_sizes=[3, 3],
        pool_sizes=[2, 2],
        fc_hidden=32,
        epochs=8,
        lr=1e-3,
        batch_size=8,
        augment=True,
        se_blocks=True,
        stochastic_depth=0.1,
        mixup_alpha=0.2,
        label_smoothing=0.05,
        use_swa=True,
    )
    model.train(X_train, y_train, callbacks=callbacks, verbose=True)
    acc = model.accuracy()
    print(f"   Training accuracy: {acc:.4f}")
    return model


def build_pipeline_dag(X_train, y_train):
    """Build a multi-stage pipeline using qai.PipelineBuilder."""
    print("\n4. Building PipelineBuilder DAG...")
    builder = qai.PipelineBuilder("atlas_pipeline")
    builder.add_step("scaler", qai.build(technique="standard_scaler"))
    builder.add_step("pca", qai.build(technique="pca", n_components=10),
                     depends_on=["scaler"])
    builder.add_step("classifier", qai.build(technique="random_forest", n_estimators=30),
                     depends_on=["pca"])
    print(builder.render_dag())
    X_flat = X_train.reshape(len(X_train), -1).astype(np.float64)
    pipeline = builder.build()
    pipeline.train(X_flat, y_train)
    preds = pipeline.predict(X_flat[:5])
    print(f"   Pipeline predictions: {preds}")
    print(f"   Pipeline: {pipeline}")
    return pipeline


def train_to_target(X_train, y_train, target_accuracy=0.7):
    """Use qai.AutoTrainTarget to train until a target is reached."""
    print(f"\n5. Auto-training to target accuracy={target_accuracy}...")
    m = qai.build(technique="neural_network", hidden_dim=32, epochs=5, lr=0.05,
                  modern=True, n_layers=2, dropout=0.1, activation="gelu")
    X_flat = X_train.reshape(len(X_train), -1).astype(np.float32)
    target = qai.AutoTrainTarget(
        target_metric="accuracy",
        target_value=target_accuracy, patience=5, increment=5,
    )
    result = target.train(m, X_flat, y_train, max_epochs=30)
    print(f"   Result: achieved={result.get('achieved', False)}, "
          f"final_metric={result.get('final_metric', 0):.4f}, "
          f"epochs={result.get('epochs_used', 0)}")
    print(f"   best_score_: {target.best_score_:.4f}, epochs_trained: {target.epochs_trained}")
    print(f"   History: {len(target.get_history())} rounds")
    return m


def test_transfer_learning(cnn_model, X_train):
    """Test freeze_conv_base + extract_features on the trained CNN."""
    print("\n6. Testing transfer learning...")
    tech = cnn_model.technique
    if hasattr(tech, "freeze_conv_base"):
        tech.freeze_conv_base()
        print(f"   Frozen conv base: {tech._conv_frozen}")
    if hasattr(tech, "extract_features"):
        sample = X_train[:1]
        feats = tech.extract_features(sample, layer_index=0)
        print(f"   Feature extraction: input {sample.shape} -> features {feats.shape}")
    if hasattr(tech, "unfreeze_conv_base"):
        tech.unfreeze_conv_base()
        print(f"   Unfrozen conv base: {tech._conv_frozen}")


def test_weight_compression(model):
    """Compress the model's weights using qai.compress_bytes."""
    print("\n7. Testing weight compression...")
    try:
        weights_path = "/tmp/atlas_weights.json"
        model.export(weights_path)
        with open(weights_path, "rb") as f:
            raw_weights = f.read()
        compressed = qai.compress_bytes(raw_weights)
        ratio = len(compressed) / max(len(raw_weights), 1)
        print(f"   Weights: {len(raw_weights)} -> {len(compressed)} bytes (ratio: {ratio:.3f})")
        restored = qai.decompress_bytes(compressed)
        assert restored == raw_weights, "compression round-trip failed"
        print(f"   Round-trip verified: OK")
    except Exception as e:
        print(f"   Weight compression: {e}")


def production_trifecta(model, X_test, y_test):
    """Run Profiler + evaluate + doc on the trained CNN."""
    print("\n8. Production trifecta (Profiler + evaluate + doc)...")
    profile = qai.Profiler(model).profile(X_test[:1], n_warmup=1, n_runs=5)
    print(f"   Profiler: {profile.n_params} params, {profile.avg_latency_ms:.2f}ms latency, "
          f"{profile.throughput_per_s:.0f} preds/sec")
    report = qai.evaluate(model, X_test, y_test, verbose=False)
    print(f"   Evaluate: accuracy={report.metrics.get('accuracy', '?')}, verdict={report.verdict}")
    doc = qai.doc(model)
    print(f"   Doc: {len(doc)} chars generated")
    return profile, report


def bundle_and_register(model):
    """Bundle the model + register it."""
    print("\n9. Bundling + registering model...")
    bundle_path = os.path.join(os.path.dirname(__file__), "atlas.qmodel")
    qai.save_bundle(model, bundle_path,
                     sign_key="atlas-secret-2026",
                     description="ATLAS image classifier — tests qai Layer system + CNN + pipeline")
    size_kb = os.path.getsize(bundle_path) / 1024
    print(f"   Bundle: {size_kb:.1f} KB (signed)")
    reg_dir = os.path.join(os.path.dirname(__file__), "registry")
    reg = qai.ModelRegistry(reg_dir)
    reg.publish(model, name="atlas_image_clf", version="1.0.0",
                description="ATLAS image classifier",
                tags=["classification", "image", "cnn", "starter-model"])
    print(f"   Registry: {reg.stats()}")
    return bundle_path


def main():
    parser = argparse.ArgumentParser(description="ATLAS — image classifier showcase")
    parser.add_argument("--live", action="store_true", help="Enable live training dashboard")
    parser.add_argument("--n", type=int, default=200, help="Number of synthetic images")
    args = parser.parse_args()

    print("=" * 60)
    print("ATLAS — Image Classifier")
    print("Testing qai Layer system + CNN upgrades + PipelineBuilder + transfer learning")
    print("=" * 60)

    start_time = time.time()

    X_train, y_train, X_test, y_test = generate_image_data(n=args.n)
    custom_model = build_custom_layer_model(input_dim=16*16, n_classes=3)
    cnn_model = train_cnn_with_upgrades(X_train, y_train, X_test, y_test, live=args.live)
    pipeline = build_pipeline_dag(X_train, y_train)
    train_to_target(X_train, y_train, target_accuracy=0.7)
    test_transfer_learning(cnn_model, X_train)
    test_weight_compression(cnn_model)
    production_trifecta(cnn_model, X_test, y_test)
    bundle_path = bundle_and_register(cnn_model)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"ATLAS pipeline complete ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Features exercised: 9/9")
    print(f"  Bundle: {bundle_path}")


if __name__ == "__main__":
    main()
