import unittest
import numpy as np
import qai

class TestFeatureSelection(unittest.TestCase):
    def test_variance_threshold_happy_path(self):
        X = [[0, 2, 0.3], [0, 4, 0.5], [0, 6, 0.7]]
        selector = qai.VarianceThreshold(threshold=0.0)
        X_trans = selector.fit_transform(X)

        self.assertEqual(X_trans.shape[1], 2)
        self.assertEqual(selector.variances_[0], 0.0)

    def test_select_k_best_happy_path(self):
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = X[:, 0] * 3.0 + np.random.randn(50) * 0.1
        selector = qai.SelectKBest(k=2)
        X_trans = selector.fit_transform(X, y)

        self.assertEqual(X_trans.shape[1], 2)
        self.assertIn(0, selector.selected_indices_)

    def test_unfitted_transform_raises(self):
        selector = qai.VarianceThreshold()
        with self.assertRaises(RuntimeError):
            selector.transform([[1, 2]])

if __name__ == "__main__":
    unittest.main()
