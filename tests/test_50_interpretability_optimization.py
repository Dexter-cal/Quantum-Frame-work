import unittest
import numpy as np
import qai

class Test50InterpretabilityOptimization(unittest.TestCase):
    def test_partial_dependence_and_shapley(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([1.0, 2.0, 3.0])
        model = qai.build(type="regression")
        model.train(X, y)

        grid, pdp_vals = qai.partial_dependence(model, X, feature_index=0, grid_resolution=5)
        self.assertEqual(len(grid), 5)
        self.assertEqual(len(pdp_vals), 5)

        shap_vals = qai.shapley_value_approximation(model, X[0], X)
        self.assertIn("feature_0", shap_vals)
        self.assertIn("feature_1", shap_vals)

    def test_recursive_feature_elimination_and_select_from_model(self):
        X = np.array([[1.0, 0.0, 10.0], [2.0, 0.0, 20.0], [3.0, 0.0, 30.0], [4.0, 0.0, 40.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        tree_model = qai.build(type="decision_tree")
        rfe = qai.RecursiveFeatureElimination(estimator=tree_model, n_features_to_select=2)
        X_rfe = rfe.fit_transform(X, y)
        self.assertEqual(X_rfe.shape[1], 2)

        sfm = qai.SelectFromModel(estimator=tree_model, threshold="mean")
        X_sfm = sfm.fit_transform(X, y)
        self.assertGreater(X_sfm.shape[1], 0)

if __name__ == "__main__":
    unittest.main()
