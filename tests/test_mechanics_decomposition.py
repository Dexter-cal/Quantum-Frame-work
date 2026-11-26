import unittest
import numpy as np
import qai
from qai.mechanics import GradientDescent, MSE, MAE, FixedEpochs, RetrainUntil

class TestMechanicsDecomposition(unittest.TestCase):
    def test_objective_and_training_loop_build_integration_happy_path(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        # Custom objective (MAE) and custom loop (RetrainUntil)
        loop = RetrainUntil(lambda history: history[-1] < 0.05, max_epochs=2000)
        obj = MAE()

        model = qai.build(type="regression", objective=obj, training_loop=loop)
        model.train(X, y)

        self.assertTrue(model.training_status()["trained"])
        pred = model.predict([2.0])
        self.assertAlmostEqual(pred, 4.0, delta=0.5)

    def test_objective_loss_gradient_edge_cases(self):
        mse = MSE()
        mae = MAE()

        pred = np.array([1.0, 2.0])
        target = np.array([1.0, 2.0])

        # Exact match -> 0 loss and 0 gradient
        self.assertEqual(mse.compute(pred, target), 0.0)
        np.testing.assert_allclose(mse.gradient(pred, target), [0.0, 0.0])

        self.assertEqual(mae.compute(pred, target), 0.0)

    def test_training_loop_max_epochs_error_handling(self):
        # RetrainUntil that never converges under max_epochs limit
        loop = RetrainUntil(lambda history: False, max_epochs=10)
        optimizer = GradientDescent(learning_rate=0.001, training_loop=loop)

        model = qai.build(type="regression", optimizer=optimizer)
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        model.train(X, y)

        self.assertEqual(len(model.optimizer.loss_history), 10)

    def test_security_misuse_scenario(self):
        # Misuse case: passing invalid objective function type
        with self.assertRaises(AttributeError):
            model = qai.build(type="regression", objective="not_an_objective")
            X = np.array([[1.0], [2.0]])
            y = np.array([2.0, 4.0])
            model.train(X, y)

if __name__ == "__main__":
    unittest.main()
