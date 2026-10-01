import unittest
import numpy as np
import qai

class TestIsolationForest(unittest.TestCase):
    def test_isolation_forest_happy_path(self):
        np.random.seed(42)
        X_normal = np.random.normal(0, 0.5, (100, 2))
        X_outlier = np.array([[10.0, 10.0], [-10.0, -10.0]])
        X_all = np.vstack([X_normal, X_outlier])

        model = qai.build(type="isolation_forest", contamination=0.02, random_state=42)
        model.train(X_all)

        self.assertTrue(model.training_status()["trained"])

        pred_normal = model.predict(X_normal[0])
        self.assertEqual(pred_normal, 1)
        self.assertFalse(model.is_anomaly(X_normal[0]))

        pred_outlier = model.predict(X_outlier[0])
        self.assertEqual(pred_outlier, -1)
        self.assertTrue(model.is_anomaly(X_outlier[0]))

        score_outlier = model.anomaly_score(X_outlier[0])
        score_normal = model.anomaly_score(X_normal[0])
        self.assertLess(score_outlier, score_normal)

    def test_isolation_forest_edge_cases(self):
        X = np.array([[1.0, 1.0], [1.1, 1.1], [1.0, 1.2]])
        model = qai.build(type="isolation_forest", contamination=0.1, random_state=42)
        model.train(X)
        pred = model.predict(X[0])
        self.assertIn(pred, [1, -1])

    def test_isolation_forest_error_cases(self):
        model = qai.build(type="isolation_forest")
        with self.assertRaises(RuntimeError):
            model.predict([1.0, 2.0])

        with self.assertRaises(RuntimeError):
            model.anomaly_score([1.0, 2.0])

    def test_security_misuse_scenario(self):
        model = qai.build(type="isolation_forest")
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        model.train(X)

        with self.assertRaises(ValueError):
            model.predict(["invalid", "garbage"])

if __name__ == "__main__":
    unittest.main()
