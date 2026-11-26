import unittest
import numpy as np
import qai
from fixtures_huggingface_iris import load as load_hf_iris

class TestAutoTuner(unittest.TestCase):
    def test_autotuner_happy_path(self):
        X, y = load_hf_iris()
        grid = {"k": [3, 5, 7]}

        tuner = qai.autotune("knn", grid, X, y, k=3)
        self.assertIn("k", tuner.best_params_)
        self.assertGreater(tuner.best_score_, 0.8)

        best_m = tuner.best_model(X, y)
        self.assertTrue(best_m.training_status()["trained"])

    def test_autotuner_edge_cases(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y = np.array([0, 1, 0])
        grid = {"max_depth": [1]}

        tuner = qai.AutoTuner("decision_tree", grid, k=2)
        tuner.fit(X, y)
        self.assertEqual(len(tuner.results_), 1)

    def test_autotuner_error_case(self):
        # Empty param grid -> ValueError
        with self.assertRaises(ValueError):
            qai.AutoTuner("knn", {}, k=3)

        # best_model before fit -> RuntimeError
        tuner = qai.AutoTuner("knn", {"k": [3]}, k=3)
        with self.assertRaises(RuntimeError):
            tuner.best_model(np.ones((5, 2)), np.ones(5))

    def test_security_misuse_scenario(self):
        # Misuse case: malformed parameters or bad types
        X, y = load_hf_iris()
        tuner = qai.AutoTuner("knn", {"invalid_param_name": [1, 2]}, k=2)
        with self.assertRaises(TypeError):
            tuner.fit(X, y)

if __name__ == "__main__":
    unittest.main()
