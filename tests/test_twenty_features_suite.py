import unittest
import numpy as np
import qai
from fixtures_huggingface_iris import load as load_hf_iris

class TestTwentyFeaturesSuite(unittest.TestCase):
    def test_supervised_classification_techniques(self):
        X, y = load_hf_iris()
        for tech in ["lda", "qda", "adaboost", "gradient_boosting", "extra_trees"]:
            m = qai.build(type=tech)
            m.train(X, y, verbose=False)
            self.assertTrue(m.training_status()["trained"])
            pred = m.predict(X[0])
            self.assertIn(pred, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
            self.assertGreater(m.accuracy(), 0.8)

    def test_supervised_regression_techniques(self):
        X, y = qai.make_regression(n_samples=50, n_features=3, random_state=42)
        for tech in ["ridge", "lasso", "elastic_net", "kernel_ridge"]:
            m = qai.build(type=tech)
            m.train(X, y, verbose=False)
            self.assertTrue(m.training_status()["trained"])
            pred = m.predict(X[0])
            self.assertIsInstance(pred, (float, int, np.number))

    def test_unsupervised_truncated_svd(self):
        X, _ = load_hf_iris()
        m = qai.build(type="truncated_svd", n_components=2)
        m.train(X)
        self.assertTrue(m.training_status()["trained"])
        res = m.predict(X[0])
        self.assertEqual(len(res), 2)
        self.assertEqual(len(m.explained_variance_ratio()), 2)

    def test_preprocessing_scalers_and_encoders(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        minmax = qai.MinMaxScaler().fit_transform(X)
        self.assertAlmostEqual(float(np.min(minmax)), 0.0)
        self.assertAlmostEqual(float(np.max(minmax)), 1.0)

        robust = qai.RobustScaler().fit_transform(X)
        self.assertEqual(robust.shape, (3, 2))

        norm = qai.Normalizer().fit_transform(X)
        self.assertEqual(norm.shape, (3, 2))

        ohe = qai.OneHotEncoder().fit_transform([[0], [1], [0]])
        self.assertEqual(ohe.shape, (3, 2))

        binarizer = qai.Binarizer(threshold=1.5).fit_transform(X)
        self.assertEqual(binarizer[0, 0], 0.0)
        self.assertEqual(binarizer[2, 0], 1.0)

    def test_metrics_and_generators(self):
        y_t = [1.0, 2.0, 3.0]
        y_p = [1.1, 1.9, 3.1]
        self.assertLess(qai.mean_absolute_error(y_t, y_p), 0.2)
        self.assertGreater(qai.r2_score(y_t, y_p), 0.9)

        X_reg, y_reg = qai.make_regression(n_samples=20, n_features=2)
        self.assertEqual(X_reg.shape, (20, 2))
        self.assertEqual(len(y_reg), 20)

        y_true_cls = [0, 1, 0, 1]
        y_prob = [0.1, 0.9, 0.2, 0.8]
        self.assertLess(qai.log_loss(y_true_cls, y_prob), 0.3)
        self.assertEqual(qai.roc_auc_score(y_true_cls, y_prob), 1.0)

    def test_governance_weight_distance_and_calibration(self):
        m1 = qai.build(type="regression")
        m2 = qai.build(type="regression")
        X = np.array([[1.0], [2.0]])
        m1.train(X, [2.0, 4.0], verbose=False)
        m2.train(X, [3.0, 6.0], verbose=False)

        dist = qai.weight_distance(m1, m2)
        self.assertGreater(dist, 0.0)

        curve = qai.calibration_curve([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], n_bins=2)
        self.assertIn("prob_true", curve)
        self.assertIn("prob_pred", curve)

    def test_security_misuse_scenarios(self):
        # Misuse: weight_distance on models without weights
        m1 = qai.build(type="kmeans")
        m2 = qai.build(type="kmeans")
        m1.train([[1.0, 2.0], [3.0, 4.0]])
        m2.train([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(AttributeError):
            qai.weight_distance(m1, m2)

        # Misuse: LDA with predict before train
        m_lda = qai.build(type="lda")
        with self.assertRaises(RuntimeError):
            m_lda.predict([1.0, 2.0])

if __name__ == "__main__":
    unittest.main()
