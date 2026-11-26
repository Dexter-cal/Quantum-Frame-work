"""Tests for qai.techniques.cnn."""
import sys, os, unittest, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qai
from qai.techniques import CNN


def _make_synthetic_data(n=40, h=16, w=16, n_classes=3, in_channels=1, seed=0):
    """Generate synthetic image data for testing."""
    rng = np.random.default_rng(seed)
    if in_channels == 1:
        X = rng.standard_normal((n, h, w)).astype(np.float32)
    else:
        X = rng.standard_normal((n, h, w, in_channels)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n)
    return X, y


class TestCNNBuild(unittest.TestCase):
    def test_build_default(self):
        m = qai.build(technique="cnn")
        self.assertIsInstance(m.technique, CNN)
        self.assertEqual(m.technique.in_channels, 1)
        self.assertEqual(m.technique.n_classes, 10)

    def test_build_custom_arch(self):
        m = qai.build(
            technique="cnn",
            in_channels=3, n_classes=5,
            conv_channels=[32, 64, 128],
            kernel_sizes=[3, 3, 3],
            pool_sizes=[2, 2, 2],
            fc_hidden=256,
        )
        self.assertEqual(m.technique.in_channels, 3)
        self.assertEqual(m.technique.n_classes, 5)
        self.assertEqual(m.technique.conv_channels, [32, 64, 128])

    def test_architecture_summary(self):
        m = qai.build(technique="cnn", in_channels=1, n_classes=3,
                      conv_channels=[8, 16], fc_hidden=32)
        m.train(*_make_synthetic_data(), verbose=False)
        arch = m.technique.architecture_summary()
        self.assertEqual(arch["technique"], "cnn")
        self.assertEqual(arch["in_channels"], 1)
        self.assertEqual(arch["n_classes"], 3)
        self.assertEqual(arch["conv_channels"], [8, 16])
        self.assertGreater(arch["parameters"], 0)


class TestCNNFit(unittest.TestCase):
    def test_fit_grayscale(self):
        X, y = _make_synthetic_data(n=30, n_classes=3, in_channels=1)
        m = qai.build(technique="cnn", in_channels=1, n_classes=3,
                      conv_channels=[8], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=16, epochs=3, batch_size=8, augment=False)
        m.train(X, y, verbose=False)
        self.assertTrue(m.technique.is_trained())
        self.assertGreater(len(m.technique.loss_history()), 0)

    def test_fit_rgb(self):
        X, y = _make_synthetic_data(n=20, n_classes=2, in_channels=3)
        m = qai.build(technique="cnn", in_channels=3, n_classes=2,
                      conv_channels=[8], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=16, epochs=3, batch_size=4, augment=False)
        m.train(X, y, verbose=False)
        self.assertTrue(m.technique.is_trained())

    def test_fit_string_labels(self):
        X, _ = _make_synthetic_data(n=20, n_classes=3, in_channels=1)
        y_str = np.array(["cat", "dog", "bird"])[np.random.randint(0, 3, size=20)]
        m = qai.build(technique="cnn", in_channels=1, n_classes=3,
                      conv_channels=[8], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=16, epochs=3, batch_size=4, augment=False)
        m.train(X, y_str, verbose=False)
        self.assertTrue(m.technique.is_trained())
        # Predict should return string labels
        pred = m.predict(X[0])
        self.assertIn(pred, ["cat", "dog", "bird"])

    def test_fit_decreases_loss(self):
        X, y = _make_synthetic_data(n=40, n_classes=3, seed=42)
        m = qai.build(technique="cnn", in_channels=1, n_classes=3,
                      conv_channels=[8, 16], kernel_sizes=[3, 3], pool_sizes=[2, 2],
                      fc_hidden=32, epochs=15, lr=1e-2, batch_size=8, augment=False)
        m.train(X, y, verbose=False)
        losses = m.technique.loss_history()
        self.assertGreater(losses[0], losses[-1])

    def test_fit_with_augmentation(self):
        X, y = _make_synthetic_data(n=30, n_classes=3)
        m = qai.build(technique="cnn", in_channels=1, n_classes=3,
                      conv_channels=[8], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=16, epochs=3, batch_size=8, augment=True)
        m.train(X, y, verbose=False)
        self.assertTrue(m.technique.is_trained())

    def test_fit_without_y_raises(self):
        X, _ = _make_synthetic_data(n=10)
        m = qai.build(technique="cnn", in_channels=1, n_classes=3,
                      conv_channels=[8], epochs=1, augment=False)
        with self.assertRaises(ValueError):
            m.train(X, verbose=False)


class TestCNNPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = _make_synthetic_data(n=40, n_classes=3, seed=0)
        cls.model = qai.build(
            technique="cnn", in_channels=1, n_classes=3,
            conv_channels=[8, 16], kernel_sizes=[3, 3], pool_sizes=[2, 2],
            fc_hidden=32, epochs=5, batch_size=8, augment=False,
        )
        cls.model.train(cls.X, cls.y, verbose=False)

    def test_predict_single_returns_int(self):
        pred = self.model.predict(self.X[0])
        self.assertIsInstance(pred, (int, np.integer))
        self.assertGreaterEqual(pred, 0)
        self.assertLess(pred, 3)

    def test_predict_batch_returns_array(self):
        preds = self.model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_predict_proba_single(self):
        probs = self.model.predict_proba(self.X[0])
        self.assertEqual(probs.shape, (3,))
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_predict_proba_batch(self):
        probs = self.model.predict_proba(self.X[:5])
        self.assertEqual(probs.shape, (5, 3))
        for row in probs:
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_predict_2d_input(self):
        # Single 2D image
        pred = self.model.predict(self.X[0, :, :])
        self.assertIsInstance(pred, (int, np.integer))

    def test_predict_3d_input_hwc(self):
        # Single 3D image (H, W, C=1)
        img = self.X[0, :, :, None]  # (H, W, 1)
        pred = self.model.predict(img)
        self.assertIsInstance(pred, (int, np.integer))

    def test_predict_before_train_raises(self):
        m = qai.build(technique="cnn", in_channels=1, n_classes=3)
        with self.assertRaises(RuntimeError):
            m.predict(self.X[0])


class TestCNNInputFormats(unittest.TestCase):
    def test_channels_first_4d(self):
        # (N, C, H, W) — explicit channels-first
        X = np.random.randn(10, 1, 16, 16).astype(np.float32)
        y = np.random.randint(0, 2, 10)
        m = qai.build(technique="cnn", in_channels=1, n_classes=2,
                      conv_channels=[4], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=8, epochs=1, batch_size=4, augment=False)
        m.train(X, y, verbose=False)
        self.assertTrue(m.technique.is_trained())

    def test_channels_last_4d(self):
        # (N, H, W, C) — channels-last
        X = np.random.randn(10, 16, 16, 1).astype(np.float32)
        y = np.random.randint(0, 2, 10)
        m = qai.build(technique="cnn", in_channels=1, n_classes=2,
                      conv_channels=[4], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=8, epochs=1, batch_size=4, augment=False)
        m.train(X, y, verbose=False)
        self.assertTrue(m.technique.is_trained())

    def test_list_of_2d_images(self):
        X = [np.random.randn(16, 16).astype(np.float32) for _ in range(10)]
        y = np.random.randint(0, 2, 10)
        m = qai.build(technique="cnn", in_channels=1, n_classes=2,
                      conv_channels=[4], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=8, epochs=1, batch_size=4, augment=False)
        m.train(X, y, verbose=False)
        self.assertTrue(m.technique.is_trained())

    def test_uint8_normalization(self):
        # Values 0-255 should be auto-divided by 255
        X = (np.random.rand(10, 16, 16) * 255).astype(np.uint8).astype(np.float32)
        y = np.random.randint(0, 2, 10)
        m = qai.build(technique="cnn", in_channels=1, n_classes=2,
                      conv_channels=[4], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=8, epochs=1, batch_size=4, augment=False)
        m.train(X, y, verbose=False)
        self.assertTrue(m.technique.is_trained())


class TestCNNAugmentation(unittest.TestCase):
    def test_augment_flip(self):
        X, y = _make_synthetic_data(n=20, n_classes=2)
        m = qai.build(technique="cnn", in_channels=1, n_classes=2,
                      conv_channels=[8], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=16, epochs=2, batch_size=4,
                      augment=True, augment_flip_p=1.0,  # always flip
                      augment_rotate=False, augment_crop_pad=0)
        m.train(X, y, verbose=False)
        self.assertTrue(m.technique.is_trained())

    def test_augment_rotate(self):
        X, y = _make_synthetic_data(n=20, n_classes=2)
        m = qai.build(technique="cnn", in_channels=1, n_classes=2,
                      conv_channels=[8], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=16, epochs=2, batch_size=4,
                      augment=True, augment_rotate=True,
                      augment_flip_p=0.0, augment_crop_pad=0)
        m.train(X, y, verbose=False)
        self.assertTrue(m.technique.is_trained())


class TestCNNFeatureExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = _make_synthetic_data(n=20, n_classes=3, seed=0)
        cls.model = qai.build(
            technique="cnn", in_channels=1, n_classes=3,
            conv_channels=[8, 16], kernel_sizes=[3, 3], pool_sizes=[2, 2],
            fc_hidden=32, epochs=2, batch_size=4, augment=False,
        )
        cls.model.train(cls.X, cls.y, verbose=False)

    def test_extract_features_last_layer(self):
        feats = self.model.extract_features(self.X[0], layer_index=-1)
        # Last conv block has 16 channels
        self.assertEqual(feats.shape[0], 16)

    def test_extract_features_first_layer(self):
        feats = self.model.extract_features(self.X[0], layer_index=0)
        # First conv block has 8 channels
        self.assertEqual(feats.shape[0], 8)

    def test_extract_features_batch(self):
        feats = self.model.extract_features(self.X[:5], layer_index=-1)
        self.assertEqual(feats.shape[0], 5)
        self.assertEqual(feats.shape[1], 16)

    def test_extract_features_invalid_layer(self):
        with self.assertRaises(IndexError):
            self.model.extract_features(self.X[0], layer_index=99)


class TestCNNTransferLearning(unittest.TestCase):
    def test_freeze_unfreeze(self):
        X, y = _make_synthetic_data(n=20, n_classes=3)
        m = qai.build(technique="cnn", in_channels=1, n_classes=3,
                      conv_channels=[8, 16], kernel_sizes=[3, 3], pool_sizes=[2, 2],
                      fc_hidden=32, epochs=2, batch_size=4, augment=False)
        m.train(X, y, verbose=False)
        total_params_before = m.technique.parameter_count()
        m.technique.freeze_conv_base()
        trainable_after = m.technique.parameter_count()
        self.assertLess(trainable_after, total_params_before)
        m.technique.unfreeze_conv_base()
        trainable_after_unfreeze = m.technique.parameter_count()
        self.assertEqual(trainable_after_unfreeze, total_params_before)

    def test_replace_classifier_head(self):
        X, y = _make_synthetic_data(n=20, n_classes=3)
        m = qai.build(technique="cnn", in_channels=1, n_classes=3,
                      conv_channels=[8, 16], kernel_sizes=[3, 3], pool_sizes=[2, 2],
                      fc_hidden=32, epochs=2, batch_size=4, augment=False)
        m.train(X, y, verbose=False)
        m.technique.replace_classifier_head(n_classes=5)
        self.assertEqual(m.technique.n_classes, 5)

    def test_freeze_before_train_raises(self):
        m = qai.build(technique="cnn")
        with self.assertRaises(RuntimeError):
            m.technique.freeze_conv_base()


class TestCNNBundleIntegration(unittest.TestCase):
    def test_cnn_bundle_round_trip(self):
        X, y = _make_synthetic_data(n=20, n_classes=3, seed=0)
        m = qai.build(
            technique="cnn", in_channels=1, n_classes=3,
            conv_channels=[8, 16], kernel_sizes=[3, 3], pool_sizes=[2, 2],
            fc_hidden=32, epochs=2, batch_size=4, augment=False,
        )
        m.train(X, y, verbose=False)
        original_preds = m.predict(X[:5])

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, sign_key="secret")
            bundle = qai.load_bundle(path, sign_key="secret")
            loaded_preds = bundle.model.predict(X[:5])
            np.testing.assert_array_equal(original_preds, loaded_preds)
        finally:
            os.unlink(path)

    def test_cnn_bundle_inspect(self):
        X, y = _make_synthetic_data(n=10, n_classes=2)
        m = qai.build(technique="cnn", in_channels=1, n_classes=2,
                      conv_channels=[4], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=8, epochs=1, augment=False)
        m.train(X, y, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, description="cnn test")
            info = qai.inspect_bundle(path)
            self.assertEqual(info["manifest"]["technique"], "cnn")
            self.assertGreater(info["manifest"]["total_params"], 0)
        finally:
            os.unlink(path)


class TestCNNIntegrationWithQai(unittest.TestCase):
    def test_accessible_from_top_level(self):
        m = qai.build(technique="cnn")
        self.assertIsNotNone(m)

    def test_in_registry(self):
        from qai.techniques import _REGISTRY
        self.assertIn("cnn", _REGISTRY)

    def test_in_supervised_list(self):
        from qai.techniques import SUPERVISED
        self.assertIn("cnn", SUPERVISED)

    def test_accuracy(self):
        X, y = _make_synthetic_data(n=20, n_classes=3, seed=0)
        m = qai.build(technique="cnn", in_channels=1, n_classes=3,
                      conv_channels=[8], kernel_sizes=[3], pool_sizes=[2],
                      fc_hidden=16, epochs=3, batch_size=4, augment=False)
        m.train(X, y, verbose=False)
        acc = m.technique.accuracy()
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_parameter_count(self):
        m = qai.build(technique="cnn", in_channels=1, n_classes=3,
                      conv_channels=[8, 16], fc_hidden=32)
        m.train(*_make_synthetic_data(n=10), verbose=False)
        self.assertGreater(m.technique.parameter_count(), 0)


if __name__ == "__main__":
    unittest.main()
