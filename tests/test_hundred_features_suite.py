import unittest
import numpy as np
import qai
from fixtures_huggingface_iris import load as load_hf_iris

class TestHundredFeaturesSuite(unittest.TestCase):
    def test_additional_classifiers(self):
        X, y = load_hf_iris()
        classifiers = [
            "bernoulli_naive_bayes", "sgd_classifier", "passive_aggressive_classifier",
            "linear_svc", "nu_svc", "nearest_centroid", "bagging_classifier", "hist_gradient_boosting"
        ]
        for tech in classifiers:
            m = qai.build(type=tech)
            m.train(X, y, verbose=False)
            self.assertTrue(m.training_status()["trained"])
            pred = m.predict(X[0])
            self.assertIn(pred, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

    def test_additional_regressors(self):
        X, y = qai.make_regression(n_samples=50, n_features=3, random_state=42)
        regressors = [
            "bayesian_ridge", "ard_regression", "huber", "ransac", "theil_sen",
            "decision_tree_regressor", "random_forest_regressor", "adaboost_regressor", "gradient_boosting_regressor"
        ]
        for tech in regressors:
            m = qai.build(type=tech)
            m.train(X, y, verbose=False)
            self.assertTrue(m.training_status()["trained"])
            pred = m.predict(X[0])
            self.assertIsInstance(pred, (float, int, np.number))

    def test_logic_user_manual_verification(self):
        @qai.logic
        def test_fn(val):
            return val * 3

        self.assertEqual(test_fn(4), 12)

    def test_security_misuse_scenarios(self):
        # Misuse case: fitting ComplementNB on negative features
        X_neg = np.array([[-1.0, 2.0], [3.0, 4.0]])
        y_dummy = [0, 1]
        m = qai.build(type="complement_naive_bayes")
        with self.assertRaises(ValueError):
            m.train(X_neg, y_dummy)

if __name__ == "__main__":
    unittest.main()
