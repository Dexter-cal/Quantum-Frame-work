import unittest
import numpy as np
import qai

class TestExplainability(unittest.TestCase):
    def test_feature_importance_tree(self):
        X = [[1, 0], [2, 0], [10, 1], [11, 1]]
        y = [0, 0, 1, 1]
        model = qai.build(type="decision_tree")
        model.train(X, y)

        imps = qai.feature_importance(model)
        self.assertIn("feature_0", imps)
        self.assertIn("feature_1", imps)
        self.assertAlmostEqual(sum(imps.values()), 1.0, places=4)

    def test_permutation_importance(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = qai.build(type="classifier")
        model.train(X, y)

        perm_imps = qai.permutation_importance(model, X, y, n_repeats=3)
        self.assertIn("feature_0", perm_imps)
        self.assertIn("feature_1", perm_imps)
        self.assertIn("feature_2", perm_imps)

    def test_feature_importance_invalid_model(self):
        class DummyModel:
            pass

        with self.assertRaises(AttributeError):
            qai.feature_importance(DummyModel())

if __name__ == "__main__":
    unittest.main()
