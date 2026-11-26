import unittest
import numpy as np
import qai

class TestHundredFeaturesSuite(unittest.TestCase):
    def test_huber_loss_and_focal_loss(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.5, 4.0]
        hl = qai.huber_loss(y_true, y_pred, delta=1.0)
        self.assertGreater(hl, 0.0)

        fl = qai.focal_loss([1, 0, 1], [0.9, 0.1, 0.8])
        self.assertGreater(fl, 0.0)

    def test_divergences_and_distances(self):
        p = [0.4, 0.6]
        q = [0.5, 0.5]
        kl = qai.kl_divergence(p, q)
        js = qai.js_divergence(p, q)
        self.assertGreaterEqual(kl, 0.0)
        self.assertGreaterEqual(js, 0.0)

        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        self.assertEqual(qai.chebyshev_distance(x, y), 3.0)
        self.assertGreater(qai.canberra_distance(x, y), 0.0)
        self.assertGreater(qai.braycurtis_distance(x, y), 0.0)

    def test_advanced_metrics(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 0]
        self.assertIsInstance(qai.cohen_kappa_score(y_true, y_pred), float)
        self.assertIsInstance(qai.matthews_corrcoef(y_true, y_pred), float)
        self.assertIsInstance(qai.balanced_accuracy_score(y_true, y_pred), float)

        X = [[0, 0], [0, 1], [10, 10], [10, 11]]
        labels = [0, 0, 1, 1]
        self.assertGreater(qai.silhouette_score(X, labels), 0.5)
        self.assertGreater(qai.davies_bouldin_score(X, labels), 0.0)
        self.assertAlmostEqual(qai.adjusted_rand_score(labels, labels), 1.0)
        self.assertAlmostEqual(qai.normalized_mutual_info_score(labels, labels), 1.0)

if __name__ == "__main__":
    unittest.main()
