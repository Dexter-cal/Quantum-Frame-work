import unittest
import numpy as np
import qai

class TestNeuralNetworkPyTorch(unittest.TestCase):
    def test_neural_network_classification_happy_path(self):
        # Classification on synthetic 2-class data
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.where(X[:, 0] > 0, "class_A", "class_B")

        model = qai.build(type="neural_network", hidden_dim=8, epochs=20, lr=0.01, task_type="classification")
        model.train(X, y)

        self.assertTrue(model.training_status()["trained"])
        pred_single = model.predict(X[0])
        self.assertIn(pred_single, ["class_A", "class_B"])

        pred_batch = model.predict(X[:10])
        self.assertEqual(len(pred_batch), 10)

        loss_hist = model.loss_history()
        self.assertEqual(len(loss_hist), 20)
        self.assertGreater(model.parameter_count(), 0)

    def test_neural_network_regression_happy_path(self):
        # Regression task
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        model = qai.build(type="neural_network", hidden_dim=8, epochs=100, lr=0.05, task_type="regression")
        model.train(X, y)

        pred = model.predict([2.0])
        self.assertIsInstance(pred, (float, int, np.number))

    def test_neural_network_edge_and_error_cases(self):
        model = qai.build(type="neural_network")
        with self.assertRaises(RuntimeError):
            model.predict([1.0, 2.0])

    def test_neural_network_security_misuse(self):
        model = qai.build(type="neural_network")
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array(["A", "B"])
        model.train(X, y)

        with self.assertRaises(ValueError):
            model.predict(["invalid_string_instead_of_numbers"])

if __name__ == "__main__":
    unittest.main()
