import unittest
import numpy as np
import qai

class TestNextFiveFeatures(unittest.TestCase):
    def test_feature1_multinomial_nb(self):
        X, y = qai.make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
        X_counts = np.abs(X * 10).astype(int)

        model = qai.build(type="multinomial_naive_bayes", alpha=1.0)
        model.train(X_counts, y)

        self.assertTrue(model.training_status()["trained"])
        pred = model.predict(X_counts[0])
        self.assertIn(pred, [0, 1])

        priors = model.class_log_prior()
        self.assertEqual(len(priors), 2)

    def test_feature2_polynomial_features(self):
        X = np.array([[2.0, 3.0], [4.0, 5.0]])
        poly = qai.PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)

        self.assertEqual(X_poly.shape, (2, 5))

    def test_feature3_classification_metrics(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 0]

        report = qai.classification_report(y_true, y_pred)
        self.assertIn("accuracy", report)

        cm = qai.confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.shape, (2, 2))

    def test_feature4_quantization(self):
        model = qai.build(type="regression")
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model.train(X, y)

        q_model = qai.compress_model(model, precision="float16")
        pred = q_model.predict([4.0])
        self.assertAlmostEqual(pred, 8.0, delta=0.5)

    def test_feature5_synthetic_generator(self):
        X, y = qai.make_classification(n_samples=100, n_features=6, n_classes=3)
        self.assertEqual(X.shape, (100, 6))
        self.assertEqual(len(np.unique(y)), 3)

    def test_security_misuse_scenarios(self):
        # Multinomial NB with negative counts -> ValueError
        X_neg = np.array([[-1.0, 2.0], [3.0, 4.0]])
        model = qai.build(type="multinomial_naive_bayes")
        with self.assertRaises(ValueError):
            model.train(X_neg, [0, 1])

        # Classification report with length mismatch -> ValueError
        with self.assertRaises(ValueError):
            qai.classification_report([0, 1], [0])

if __name__ == "__main__":
    unittest.main()
