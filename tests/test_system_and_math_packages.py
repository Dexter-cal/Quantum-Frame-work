import unittest
import numpy as np
import qai

class TestSystemAndMathPackages(unittest.TestCase):
    def test_system_package(self):
        gpu = qai.system.get_gpu_memory()
        self.assertIn("cuda_available", gpu)

        mem = qai.system.get_process_memory()
        self.assertGreater(mem["rss_mb"], 0.0)

        disk = qai.system.get_disk_usage()
        self.assertGreater(disk["total_gb"], 0.0)

        threads = qai.system.set_num_threads(2)
        self.assertEqual(threads, 2)

        env = qai.system.get_env_summary()
        self.assertIn("python_version", env)

    def test_math_activations(self):
        x = np.array([-2.0, 0.0, 2.0])
        self.assertAlmostEqual(qai.sigmoid(0.0), 0.5)
        np.testing.assert_allclose(qai.relu(x), [0.0, 0.0, 2.0])
        np.testing.assert_allclose(np.sum(qai.softmax(x)), 1.0)
        self.assertGreater(qai.gelu(2.0), 1.9)
        self.assertGreater(qai.swish(2.0), 1.7)
        self.assertAlmostEqual(qai.tanh(0.0), 0.0)

    def test_math_distances_and_linalg(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]

        self.assertAlmostEqual(qai.euclidean_distance(a, b), np.sqrt(2.0))
        self.assertAlmostEqual(qai.cosine_similarity(a, b), 0.0)
        self.assertAlmostEqual(qai.manhattan_distance(a, b), 2.0)
        self.assertAlmostEqual(qai.minkowski_distance(a, b, p=3), 2.0 ** (1.0 / 3.0))

        # Dot & SVD
        dot_res = qai.math.dot(a, b)
        self.assertEqual(dot_res, 0.0)

        u, s, vt = qai.math.singular_value_decomposition([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(len(s), 2)

    def test_security_misuse_scenarios(self):
        # Invalid thread count
        with self.assertRaises(ValueError):
            qai.system.set_num_threads(-1)

        # Zero length vectors cosine similarity -> 0.0
        self.assertEqual(qai.cosine_similarity([0.0, 0.0], [0.0, 0.0]), 0.0)

if __name__ == "__main__":
    unittest.main()
