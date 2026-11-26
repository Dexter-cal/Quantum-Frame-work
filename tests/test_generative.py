"""Tests for qai.techniques.generative — GAN, VAE, Diffusion."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qai
from qai.techniques import GAN, VAE, Diffusion

X = np.random.RandomState(42).randn(100, 8).astype(np.float32)


class TestGAN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = qai.build(technique="gan", data_dim=8, latent_dim=16, hidden=32,
                              n_layers=2, epochs=10, batch_size=16)
        cls.model.train(X, verbose=False)

    def test_trained(self):
        self.assertTrue(self.model.technique.is_trained())

    def test_sample_shape(self):
        s = self.model.sample(5)
        self.assertEqual(s.shape, (5, 8))

    def test_g_losses(self):
        self.assertGreater(len(self.model.technique.g_losses()), 0)

    def test_d_losses(self):
        self.assertGreater(len(self.model.technique.d_losses()), 0)

    def test_diversity_metric(self):
        d = self.model.technique.diversity_metric(50)
        self.assertGreater(d, 0)

    def test_mode_collapse_detection(self):
        mc = self.model.technique.detect_mode_collapse(50)
        self.assertIn("collapsed", mc)
        self.assertIn("mean_distance", mc)

    def test_parameter_count(self):
        self.assertGreater(self.model.technique.parameter_count(), 0)

    def test_predict_before_train(self):
        m = qai.build(technique="gan", data_dim=8)
        with self.assertRaises(RuntimeError):
            m.predict(None)

    def test_d_confidence_tracking(self):
        confs = self.model.technique.d_confidences()
        self.assertEqual(len(confs), 10)


class TestVAE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = qai.build(technique="vae", data_dim=8, latent_dim=4, hidden=32,
                              n_layers=2, epochs=10, batch_size=16)
        cls.model.train(X, verbose=False)

    def test_trained(self):
        self.assertTrue(self.model.technique.is_trained())

    def test_reconstruction(self):
        r = self.model.predict(X[0])
        self.assertEqual(r.shape, (8,))

    def test_sample_shape(self):
        s = self.model.sample(5)
        self.assertEqual(s.shape, (5, 8))

    def test_encode(self):
        z = self.model.technique.encode(X[0])
        self.assertEqual(z.shape, (1, 4))

    def test_decode(self):
        z = np.random.randn(4).astype(np.float32)
        d = self.model.technique.decode(z)
        self.assertEqual(d.shape, (1, 8))

    def test_interpolate(self):
        interp = self.model.technique.interpolate(X[0], X[1], steps=5)
        self.assertEqual(interp.shape, (5, 8))

    def test_latent_arithmetic(self):
        result = self.model.technique.latent_arithmetic(X[0], X[1], X[2])
        self.assertEqual(result.shape, (8,))

    def test_losses(self):
        self.assertGreater(len(self.model.technique.losses()), 0)
        self.assertGreater(len(self.model.technique.recon_losses()), 0)
        self.assertGreater(len(self.model.technique.kl_losses()), 0)

    def test_parameter_count(self):
        self.assertGreater(self.model.technique.parameter_count(), 0)

    def test_predict_before_train(self):
        m = qai.build(technique="vae", data_dim=8)
        with self.assertRaises(RuntimeError):
            m.predict(X[0])


class TestDiffusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = qai.build(technique="diffusion", data_dim=8, hidden=32,
                              n_layers=2, t_steps=50, epochs=10, batch_size=16)
        cls.model.train(X, verbose=False)

    def test_trained(self):
        self.assertTrue(self.model.technique.is_trained())

    def test_sample_shape(self):
        s = self.model.sample(3)
        self.assertEqual(s.shape, (3, 8))

    def test_guidance(self):
        s = self.model.sample(3, guidance_scale=0.5)
        self.assertEqual(s.shape, (3, 8))

    def test_cosine_schedule(self):
        m = qai.build(technique="diffusion", data_dim=8, hidden=16, t_steps=20,
                      schedule="cosine", epochs=1)
        m.train(X[:20], verbose=False)
        self.assertTrue(m.technique.is_trained())

    def test_losses(self):
        self.assertGreater(len(self.model.technique.losses()), 0)

    def test_parameter_count(self):
        self.assertGreater(self.model.technique.parameter_count(), 0)

    def test_predict_before_train(self):
        m = qai.build(technique="diffusion", data_dim=8)
        with self.assertRaises(RuntimeError):
            m.sample(1)


class TestIntegration(unittest.TestCase):
    def test_in_registry(self):
        from qai.techniques import _REGISTRY
        for n in ("gan", "vae", "diffusion"):
            self.assertIn(n, _REGISTRY)

    def test_architecture_summary(self):
        m = qai.build(technique="gan", data_dim=8, epochs=1)
        m.train(X[:20], verbose=False)
        arch = m.technique.architecture_summary()
        self.assertEqual(arch["technique"], "gan")


if __name__ == "__main__":
    unittest.main()
