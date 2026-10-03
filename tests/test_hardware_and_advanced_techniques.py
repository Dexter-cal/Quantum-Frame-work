import unittest
import numpy as np
import qai
from fixtures_huggingface_iris import load as load_hf_iris

class TestHardwareAndAdvancedTechniques(unittest.TestCase):
    def test_hardware_profiling(self):
        info = qai.get_hardware_info()
        self.assertIn("cpu_count_logical", info)
        self.assertIn("ram_total_gb", info)
        self.assertIn("cuda_available", info)

        device = qai.get_optimal_device()
        self.assertIn(device, ["cpu", "cuda"])

    def test_spectral_clustering(self):
        X, _ = load_hf_iris()
        m = qai.build(type="spectral_clustering", n_clusters=3)
        m.train(X)
        self.assertTrue(m.training_status()["trained"])
        pred = m.predict(X[0])
        self.assertIsInstance(pred, (int, np.integer))

    def test_fast_ica(self):
        X, _ = load_hf_iris()
        m = qai.build(type="fast_ica", n_components=2)
        m.train(X)
        res = m.predict(X[0])
        self.assertEqual(len(res), 2)

    def test_isomap_and_lle(self):
        X, _ = load_hf_iris()
        for tech in ["isomap", "lle"]:
            m = qai.build(type=tech, n_components=2, n_neighbors=5)
            m.train(X)
            res = m.predict(X[0])
            self.assertEqual(len(res), 2)

    def test_security_misuse_scenarios(self):
        m = qai.build(type="fast_ica")
        with self.assertRaises(RuntimeError):
            m.predict([1.0, 2.0])

        with self.assertRaises(ValueError):
            m.train([["invalid", "string"]])

if __name__ == "__main__":
    unittest.main()
