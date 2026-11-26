import unittest
import numpy as np
import tempfile
import os
import qai

class TestNewFrameworkFeatures(unittest.TestCase):
    def test_standard_scaler_happy_path_and_edge_cases(self):
        # Happy path
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        scaler = qai.StandardScaler()
        scaled = scaler.fit_transform(X)
        self.assertEqual(scaled.shape, (3, 2))
        np.testing.assert_allclose(np.mean(scaled, axis=0), [0.0, 0.0], atol=1e-6)

        # Edge case: zero variance column
        X_zero_var = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        scaler_zv = qai.StandardScaler()
        scaled_zv = scaler_zv.fit_transform(X_zero_var)
        self.assertFalse(np.isnan(scaled_zv).any())

        # Error case: transform before fit
        unfit_scaler = qai.StandardScaler()
        with self.assertRaises(RuntimeError):
            unfit_scaler.transform(X)

        # Misuse case: malformed string input
        with self.assertRaises(ValueError):
            scaler.fit([["invalid", "string"]])

    def test_label_encoder_happy_and_error(self):
        labels = ["cat", "dog", "cat", "bird"]
        encoder = qai.LabelEncoder()
        encoded = encoder.fit_transform(labels)
        self.assertEqual(list(encoded), [1, 2, 1, 0])
        decoded = encoder.inverse_transform(encoded)
        self.assertEqual(decoded, labels)

        unfit_encoder = qai.LabelEncoder()
        with self.assertRaises(RuntimeError):
            unfit_encoder.transform(labels)

    def test_simple_imputer(self):
        X_nan = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]])
        imputer = qai.SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X_nan)
        self.assertFalse(np.isnan(X_imputed).any())
        self.assertEqual(X_imputed[0, 1], 6.0)

        with self.assertRaises(ValueError):
            qai.SimpleImputer(strategy="invalid_strategy")

    def test_drift_detection(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, (1000, 2))
        cur_normal = np.random.normal(0, 1, (1000, 2))
        cur_drifted = np.random.normal(5, 1, (1000, 2))

        res_normal = qai.detect_drift(ref, cur_normal, threshold=0.1)
        self.assertFalse(res_normal["drift_detected"])

        res_drifted = qai.detect_drift(ref, cur_drifted, threshold=0.1)
        self.assertTrue(res_drifted["drift_detected"])

        with self.assertRaises(ValueError):
            qai.detect_drift(np.ones((10, 2)), np.ones((10, 3)))

    def test_experiment_tracker(self):
        tracker = qai.ExperimentTracker(experiment_name="test_exp")
        tracker.log_run({"lr": 0.01}, {"accuracy": 0.85})
        tracker.log_run({"lr": 0.001}, {"accuracy": 0.95})

        best = tracker.get_best_run("accuracy")
        self.assertEqual(best["params"]["lr"], 0.001)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tracker.export(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
