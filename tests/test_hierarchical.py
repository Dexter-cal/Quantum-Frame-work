import unittest
import numpy as np
import qai

class TestHierarchicalClustering(unittest.TestCase):
    def test_hierarchical_happy_path(self):
        np.random.seed(42)
        c1 = np.random.normal(0, 0.2, (20, 2))
        c2 = np.random.normal(5, 0.2, (20, 2))
        X = np.vstack([c1, c2])

        model = qai.build(type="hierarchical", n_clusters=2)
        model.train(X)

        self.assertTrue(model.training_status()["trained"])
        pred = model.predict(X[0])
        self.assertIsInstance(pred, (int, np.integer))

        counts = model.cluster_counts()
        self.assertEqual(len(counts), 2)
        self.assertEqual(sum(counts.values()), 40)

    def test_hierarchical_edge_cases(self):
        X = np.array([[1.0, 1.0], [1.1, 1.1], [5.0, 5.0]])
        model = qai.build(type="hierarchical", n_clusters=2)
        model.train(X)

        self.assertEqual(model.n_leaves(), 3)

    def test_hierarchical_error_cases(self):
        model = qai.build(type="hierarchical")
        with self.assertRaises(RuntimeError):
            model.predict([1.0, 2.0])

    def test_security_misuse_scenario(self):
        model = qai.build(type="hierarchical", n_clusters=2)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        model.train(X)

        with self.assertRaises(ValueError):
            model.predict(["invalid", "garbage"])

if __name__ == "__main__":
    unittest.main()
