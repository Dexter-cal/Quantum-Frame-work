import unittest
import numpy as np
import qai

class TestGMMUnsupervised(unittest.TestCase):
    def test_gmm_happy_path(self):
        np.random.seed(42)
        c1 = np.random.normal(0, 0.2, (20, 2))
        c2 = np.random.normal(5, 0.2, (20, 2))
        X = np.vstack([c1, c2])

        model = qai.build(type="gmm", n_components=2)
        model.train(X)

        self.assertTrue(model.training_status()["trained"])
        pred_single = model.predict(X[0])
        self.assertIsInstance(pred_single, (int, np.integer))

        probas = model.predict_proba(X[0])
        self.assertEqual(probas.shape, (1, 2))
        np.testing.assert_allclose(np.sum(probas), 1.0, atol=1e-5)

        scores = model.score_samples(X[:2])
        self.assertEqual(len(scores), 2)

        self.assertEqual(model.means().shape, (2, 2))

    def test_gmm_edge_cases(self):
        X = np.array([[1.0, 1.0], [1.1, 1.1]])
        model = qai.build(type="gmm", n_components=1)
        model.train(X)

        pred = model.predict(X[0])
        self.assertEqual(pred, 0)

    def test_gmm_error_cases(self):
        model = qai.build(type="gmm")
        with self.assertRaises(RuntimeError):
            model.predict([1.0, 2.0])

        with self.assertRaises(RuntimeError):
            model.predict_proba([1.0, 2.0])

    def test_security_misuse_scenario(self):
        model = qai.build(type="gmm", n_components=2)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        model.train(X)

        with self.assertRaises(ValueError):
            model.predict(["invalid", "garbage"])

if __name__ == "__main__":
    unittest.main()
