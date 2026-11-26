"""Tests for qai.techniques.missing_ml — Autoencoder, DQN, PPO, SemiSupervised, AssociationRules, SelfSupervised."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qai


X = np.random.RandomState(42).randn(50, 8).astype(np.float32)


class TestAutoencoder(unittest.TestCase):
    def test_basic(self):
        ae = qai.build(technique="autoencoder", data_dim=8, latent_dim=4, hidden=32, n_layers=1, variant="basic", epochs=5, batch_size=8)
        ae.train(X, verbose=False)
        self.assertTrue(ae.technique.is_trained())
        r = ae.predict(X[0])
        self.assertEqual(r.shape, (1, 8))

    def test_sparse(self):
        ae = qai.build(technique="autoencoder", data_dim=8, latent_dim=4, hidden=32, variant="sparse", l1_sparsity=1e-2, epochs=5, batch_size=8)
        ae.train(X, verbose=False)
        self.assertTrue(ae.technique.is_trained())

    def test_denoising(self):
        ae = qai.build(technique="autoencoder", data_dim=8, latent_dim=4, hidden=32, variant="denoising", noise_factor=0.2, epochs=5, batch_size=8)
        ae.train(X, verbose=False)
        self.assertTrue(ae.technique.is_trained())

    def test_encode_decode(self):
        ae = qai.build(technique="autoencoder", data_dim=8, latent_dim=4, hidden=32, n_layers=1, epochs=5, batch_size=8)
        ae.train(X, verbose=False)
        z = ae.technique.encode(X[0])
        self.assertEqual(z.shape, (1, 4))
        d = ae.technique.decode(z)
        self.assertEqual(d.shape, (1, 8))

    def test_sample(self):
        ae = qai.build(technique="autoencoder", data_dim=8, latent_dim=4, hidden=32, n_layers=1, epochs=5, batch_size=8)
        ae.train(X, verbose=False)
        s = ae.technique.sample(3)
        self.assertEqual(s.shape, (3, 8))


class TestDQN(unittest.TestCase):
    class _Env:
        def reset(self): return np.array([0.0, 0.0])
        def step(self, a): return np.array([0.1, 0.1]), 1.0, True, {}

    def test_train_and_predict(self):
        dqn = qai.build(technique="dqn", state_dim=2, n_actions=4, hidden=16, epochs=5, batch_size=4)
        dqn.train(environment=self._Env(), verbose=False)
        self.assertTrue(dqn.technique.is_trained())
        a = dqn.predict([0.5, 0.5])
        self.assertIsInstance(a, int)

    def test_q_values(self):
        dqn = qai.build(technique="dqn", state_dim=2, n_actions=4, hidden=16, epochs=3, batch_size=4)
        dqn.train(environment=self._Env(), verbose=False)
        q = dqn.technique.get_q_values([0.5, 0.5])
        self.assertEqual(len(q), 4)


class TestPPO(unittest.TestCase):
    class _Env:
        def reset(self): return np.array([0.0, 0.0])
        def step(self, a): return np.array([0.1, 0.1]), 0.5, True, {}

    def test_train_and_predict(self):
        ppo = qai.build(technique="ppo", state_dim=2, n_actions=4, hidden=16, max_episodes=5)
        ppo.train(environment=self._Env(), verbose=False)
        self.assertTrue(ppo.technique.is_trained())
        a = ppo.predict([0.5, 0.5])
        self.assertIsInstance(a, int)

    def test_action_probs(self):
        ppo = qai.build(technique="ppo", state_dim=2, n_actions=4, hidden=16, max_episodes=3)
        ppo.train(environment=self._Env(), verbose=False)
        probs = ppo.technique.get_action_probs([0.5, 0.5])
        self.assertAlmostEqual(sum(probs), 1.0, places=3)


class TestSemiSupervised(unittest.TestCase):
    def test_self_training(self):
        X_l = np.random.randn(20, 4).astype(np.float32)
        y_l = np.random.randint(0, 2, 20)
        X_u = np.random.randn(40, 4).astype(np.float32)
        ss = qai.build(technique="semi_supervised", confidence_threshold=0.5, max_iterations=3)
        ss.train(X_l, y=y_l, X_unlabeled=X_u, verbose=False)
        self.assertTrue(ss.technique.is_trained())
        pred = ss.predict(X_l[0])
        self.assertIsNotNone(pred)
        summary = ss.technique.training_summary()
        self.assertGreaterEqual(summary["total_count"], 20)

    def test_no_unlabeled(self):
        X_l = np.random.randn(10, 4).astype(np.float32)
        y_l = np.random.randint(0, 2, 10)
        ss = qai.build(technique="semi_supervised", max_iterations=1)
        ss.train(X_l, y=y_l, verbose=False)
        self.assertTrue(ss.technique.is_trained())


class TestAssociationRules(unittest.TestCase):
    def test_basic(self):
        transactions = [["a", "b"], ["a", "b", "c"], ["a", "c"], ["b", "c"], ["a", "b", "c"]]
        ar = qai.build(technique="association_rules", min_support=0.4, min_confidence=0.5)
        ar.train(transactions, verbose=False)
        self.assertTrue(ar.technique.is_trained())
        itemsets = ar.technique.get_frequent_itemsets()
        rules = ar.technique.get_rules()
        self.assertGreater(len(itemsets), 0)
        self.assertGreater(len(rules), 0)

    def test_predict(self):
        transactions = [["a", "b"], ["a", "b", "c"], ["a", "c"], ["b", "c"], ["a", "b", "c"]]
        ar = qai.build(technique="association_rules", min_support=0.4, min_confidence=0.5)
        ar.train(transactions, verbose=False)
        result = ar.predict(["a"])
        self.assertIsInstance(result, list)

    def test_no_frequent_items(self):
        transactions = [["x"], ["y"], ["z"]]  # all unique
        ar = qai.build(technique="association_rules", min_support=0.9, min_confidence=0.9)
        ar.train(transactions, verbose=False)
        self.assertEqual(len(ar.technique.get_frequent_itemsets()), 0)


class TestSelfSupervised(unittest.TestCase):
    def test_train_and_encode(self):
        ssl = qai.build(technique="self_supervised", data_dim=8, hidden=32, latent_dim=4, n_layers=1, mask_ratio=0.3, epochs=5, batch_size=8)
        ssl.train(X, verbose=False)
        self.assertTrue(ssl.technique.is_trained())
        z = ssl.technique.encode(X[0])
        self.assertEqual(z.shape, (1, 4))

    def test_reconstruct(self):
        ssl = qai.build(technique="self_supervised", data_dim=8, hidden=32, latent_dim=4, n_layers=1, mask_ratio=0.3, epochs=5, batch_size=8)
        ssl.train(X, verbose=False)
        r = ssl.predict(X[0])
        self.assertEqual(r.shape, (1, 8))

    def test_losses(self):
        ssl = qai.build(technique="self_supervised", data_dim=8, hidden=16, latent_dim=4, epochs=3, batch_size=8)
        ssl.train(X, verbose=False)
        self.assertGreater(len(ssl.technique.losses()), 0)


class TestIntegration(unittest.TestCase):
    def test_all_in_registry(self):
        from qai.techniques import _REGISTRY
        for name in ("autoencoder", "dqn", "ppo", "semi_supervised", "association_rules", "self_supervised"):
            self.assertIn(name, _REGISTRY)

    def test_autoencoder_variants(self):
        for v in ("basic", "sparse", "denoising", "convolutional"):
            ae = qai.build(technique="autoencoder", data_dim=8, hidden=16, latent_dim=4, n_layers=1, variant=v, epochs=1, batch_size=8)
            ae.train(X[:10], verbose=False)
            self.assertTrue(ae.technique.is_trained())


if __name__ == "__main__":
    unittest.main()
