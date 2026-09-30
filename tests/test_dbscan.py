import unittest
import numpy as np
import qai

class TestDBSCANUnsupervised(unittest.TestCase):
    def test_dbscan_happy_path(self):
        # Happy path on 2 distinct dense clusters
        np.random.seed(42)
        c1 = np.random.normal(0, 0.2, (20, 2))
        c2 = np.random.normal(5, 0.2, (20, 2))
        X = np.vstack([c1, c2])

        model = qai.build(type="dbscan", eps=0.8, min_samples=3)
        model.train(X)

        self.assertTrue(model.training_status()["trained"])
        pred_single = model.predict(X[0])
        self.assertIsInstance(pred_single, (int, np.integer))

        pred_batch = model.predict(X[:5])
        self.assertEqual(len(pred_batch), 5)

        self.assertEqual(model.n_clusters(), 2)
        self.assertLessEqual(model.noise_ratio(), 0.1)

    def test_dbscan_edge_cases(self):
        # Empty / extremely small cluster set
        X_small = np.array([[0.0, 0.0], [0.1, 0.1]])
        model = qai.build(type="dbscan", eps=0.01, min_samples=5)
        model.train(X_small)

        self.assertEqual(model.n_clusters(), 0)
        self.assertEqual(model.noise_ratio(), 1.0)
        pred = model.predict(X_small[0])
        self.assertEqual(pred, -1)

    def test_dbscan_error_case(self):
        model = qai.build(type="dbscan")
        with self.assertRaises(RuntimeError):
            model.predict([1.0, 2.0])

    def test_dbscan_security_misuse(self):
        model = qai.build(type="dbscan")
        X = np.array([[1.0, 2.0], [2.0, 3.0]])
        model.train(X)

        # Misuse case: wrong dimension or string array input
        with self.assertRaises(ValueError):
            model.predict(["invalid", "garbage"])

if __name__ == "__main__":
    unittest.main()
