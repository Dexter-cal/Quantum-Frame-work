"""Tests for qai.bundle — .qmodel bundle format."""
import sys, os, unittest, tempfile, zipfile, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import qai
from qai.bundle import (
    Bundle, save_bundle, load_bundle, inspect_bundle, verify_bundle,
    diff_bundles, BUNDLE_FORMAT_VERSION,
)


# Define a module-level logic function so inspect.getsource works
@qai.logic
def sample_logic_fn(x):
    """Sample logic function for bundle testing."""
    return float(x) * 2


class TestBundleSaveLoad(unittest.TestCase):
    def test_regression_bundle_round_trip(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)
        original_pred = m.predict([4.0])

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path)
            bundle = qai.load_bundle(path, verify_signature=False)
            loaded_pred = bundle.model.predict([4.0])
            self.assertAlmostEqual(original_pred, loaded_pred, places=5)
        finally:
            os.unlink(path)

    def test_classifier_bundle_round_trip(self):
        # Multi-class classifier — exercises sklearn coef_/intercept_/classes_
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        m = qai.build(technique="classifier")
        m.train(X, y, verbose=False)
        original_pred = m.predict([X[0]])

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path)
            bundle = qai.load_bundle(path, verify_signature=False)
            loaded_pred = bundle.model.predict([X[0]])
            self.assertEqual(original_pred, loaded_pred)
        finally:
            os.unlink(path)

    def test_bundle_with_logic_fn(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, logic_fns=[sample_logic_fn])
            bundle = qai.load_bundle(path, verify_signature=False)
            self.assertIn("sample_logic_fn", bundle.logic_fns)
            result = bundle.logic_fns["sample_logic_fn"](5.0)
            self.assertEqual(result, 10.0)
        finally:
            os.unlink(path)

    def test_bundle_with_tokenizer(self):
        tok = qai.CharTokenizer()
        tok.fit(["hello world"])

        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, tokenizer=tok)
            bundle = qai.load_bundle(path, verify_signature=False)
            self.assertIsNotNone(bundle.tokenizer)
            self.assertEqual(bundle.tokenizer.vocab_size, tok.vocab_size)
            self.assertEqual(bundle.tokenizer.encode("hello"), tok.encode("hello"))
        finally:
            os.unlink(path)

    def test_transformer_bundle_round_trip(self):
        tok = qai.CharTokenizer()
        tok.fit(["ab", "cd"])
        m = qai.build(technique="transformer", vocab_size=tok.vocab_size,
                      d_model=16, n_heads=2, n_layers=1, max_seq_len=8, epochs=2)
        m.train(["abcd"], tokenizer=tok, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, tokenizer=tok)
            bundle = qai.load_bundle(path, verify_signature=False)
            self.assertEqual(bundle.model.technique_name, "transformer")
            # Generate should work
            out = bundle.model.generate("a", max_new_tokens=3, tokenizer=bundle.tokenizer, temperature=0)
            self.assertIsInstance(out, str)
        finally:
            os.unlink(path)


class TestBundleSigning(unittest.TestCase):
    def test_signed_bundle_verifies(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, sign_key="secret")
            self.assertTrue(verify_bundle(path, "secret"))
        finally:
            os.unlink(path)

    def test_wrong_sign_key_fails(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, sign_key="secret")
            self.assertFalse(verify_bundle(path, "wrong_key"))
        finally:
            os.unlink(path)

    def test_unsigned_bundle_returns_false(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path)  # no sign_key
            self.assertFalse(verify_bundle(path, "any_key"))
        finally:
            os.unlink(path)

    def test_load_signed_bundle_requires_key(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, sign_key="secret")
            # Loading without sign_key should raise
            with self.assertRaises(ValueError):
                qai.load_bundle(path)  # no sign_key
            # Loading with wrong key should raise
            with self.assertRaises(ValueError):
                qai.load_bundle(path, sign_key="wrong")
            # Loading with correct key works
            bundle = qai.load_bundle(path, sign_key="secret")
            self.assertEqual(bundle.model.technique_name, "regression")
        finally:
            os.unlink(path)

    def test_tamper_detection(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, sign_key="secret")
            # Tamper: rewrite the zip with modified weights
            buf = io.BytesIO()
            with zipfile.ZipFile(path, "r") as zin:
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
                    for item in zin.infolist():
                        data = zin.read(item.filename)
                        if item.filename == "weights.npy":
                            data = bytes([data[0] ^ 0xFF]) + data[1:]
                        zout.writestr(item, data)
            with open(path, "wb") as f:
                f.write(buf.getvalue())

            self.assertFalse(verify_bundle(path, "secret"))
            with self.assertRaises(ValueError):
                qai.load_bundle(path, sign_key="secret")
        finally:
            os.unlink(path)


class TestBundleInspect(unittest.TestCase):
    def test_inspect_returns_manifest(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, description="test inspect", author="qa")
            info = inspect_bundle(path)
            self.assertEqual(info["manifest"]["technique"], "regression")
            self.assertEqual(info["manifest"]["description"], "test inspect")
            self.assertEqual(info["manifest"]["author"], "qa")
            self.assertEqual(info["manifest"]["format_version"], BUNDLE_FORMAT_VERSION)
            self.assertGreater(info["size_bytes"], 0)
            self.assertGreater(info["size_uncompressed_bytes"], 0)
            # Should have at least manifest, model, weights, README
            file_names = [c["name"] for c in info["contents"]]
            self.assertIn("manifest.json", file_names)
            self.assertIn("model.json", file_names)
            self.assertIn("weights.npy", file_names)
        finally:
            os.unlink(path)

    def test_inspect_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            inspect_bundle("/tmp/does_not_exist.qmodel")


class TestBundleDiff(unittest.TestCase):
    def test_identical_bundles_no_diff(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            p1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            p2 = f.name
        try:
            qai.save_bundle(m, p1)
            qai.save_bundle(m, p2)
            diff = diff_bundles(p1, p2)
            self.assertFalse(diff["weights_changed"])
            self.assertFalse(diff["technique_changed"])
        finally:
            os.unlink(p1); os.unlink(p2)

    def test_diff_detects_weight_changes(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        m1 = qai.build(technique="regression")
        m1.train(X, y, verbose=False)
        m2 = qai.build(technique="regression")
        m2.train(X, y, verbose=False)
        m2.technique.w[0] = 99.0

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            p1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            p2 = f.name
        try:
            qai.save_bundle(m1, p1)
            qai.save_bundle(m2, p2)
            diff = diff_bundles(p1, p2)
            self.assertTrue(diff["weights_changed"])
            # Per-weight diff should show large change in 'w'
            self.assertIn("w", diff["weight_diffs"])
            self.assertGreater(diff["weight_diffs"]["w"]["l2_distance"], 1.0)
        finally:
            os.unlink(p1); os.unlink(p2)

    def test_diff_detects_technique_change(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y_class = np.array([0, 1, 0])
        m1 = qai.build(technique="regression")
        m1.train(X, np.array([2.0, 4.0, 6.0]), verbose=False)
        m2 = qai.build(technique="classifier")
        m2.train(X, y_class, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            p1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            p2 = f.name
        try:
            qai.save_bundle(m1, p1)
            qai.save_bundle(m2, p2)
            diff = diff_bundles(p1, p2)
            self.assertTrue(diff["technique_changed"])
            self.assertEqual(diff["technique1"], "regression")
            self.assertEqual(diff["technique2"], "classifier")
        finally:
            os.unlink(p1); os.unlink(p2)


class TestBundleErrors(unittest.TestCase):
    def test_load_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            qai.load_bundle("/tmp/nonexistent.qmodel")

    def test_save_non_model_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(ValueError):
                qai.save_bundle("not a model", path)
        finally:
            os.unlink(path)

    def test_save_unsupported_tokenizer_raises(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(ValueError):
                qai.save_bundle(m, path, tokenizer="not a tokenizer")
        finally:
            os.unlink(path)


class TestBundleStructure(unittest.TestCase):
    def test_bundle_is_zip_file(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path)
            self.assertTrue(zipfile.is_zipfile(path))
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                self.assertIn("manifest.json", names)
                self.assertIn("model.json", names)
                self.assertIn("weights.npy", names)
        finally:
            os.unlink(path)

    def test_bundle_has_readme(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, description="my model")
            with zipfile.ZipFile(path) as zf:
                readme = zf.read("README.md").decode("utf-8")
                self.assertIn("regression", readme)
                self.assertIn("my model", readme)
        finally:
            os.unlink(path)


class TestBundleBundleObject(unittest.TestCase):
    def test_bundle_repr(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path, sign_key="key")
            bundle = qai.load_bundle(path, sign_key="key")
            repr_str = repr(bundle)
            self.assertIn("regression", repr_str)
            self.assertIn("signed=True", repr_str)
        finally:
            os.unlink(path)


class TestIntegrationWithQai(unittest.TestCase):
    def test_accessible_from_top_level(self):
        self.assertTrue(hasattr(qai, "save_bundle"))
        self.assertTrue(hasattr(qai, "load_bundle"))
        self.assertTrue(hasattr(qai, "inspect_bundle"))
        self.assertTrue(hasattr(qai, "verify_bundle"))
        self.assertTrue(hasattr(qai, "diff_bundles"))
        self.assertTrue(hasattr(qai, "bundle"))

    def test_bundle_with_memory_integration(self):
        """Bundle + Memory tiers work together."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        m = qai.build(technique="regression")
        m.train(X, y, verbose=False)

        # Capture weight snapshot before bundling
        wmem = qai.WeightsMemory()
        wmem.attach(m)

        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(m, path)
            bundle = qai.load_bundle(path, verify_signature=False)

            # After loading, the bundled model's weights should match the snapshot
            diff = wmem.diff(bundle.model)
            self.assertTrue(diff["overall"]["hash_match"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
