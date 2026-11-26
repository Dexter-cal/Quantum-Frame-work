"""Tests for qai.techniques.advanced_dl — MoE + GNN."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qai
from qai.techniques import MixtureOfExperts, GNN

X = np.random.RandomState(42).randn(100, 8).astype(np.float32)
y = np.random.RandomState(42).randint(0, 3, 100)


class TestMoE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = qai.build(technique="moe", in_dim=8, d_model=8, n_experts=4,
                              top_k=2, hidden=32, n_classes=3, epochs=15, lr=1e-3, batch_size=16)
        cls.model.train(X, y, verbose=False)

    def test_trained(self):
        self.assertTrue(self.model.technique.is_trained())

    def test_predict_single(self):
        pred = self.model.predict(X[0])
        self.assertIsInstance(pred, (int, np.integer))

    def test_predict_batch(self):
        preds = self.model.predict(X[:5])
        self.assertEqual(len(preds), 5)

    def test_expert_utilization(self):
        util = self.model.technique.get_expert_utilization()
        self.assertEqual(len(util), 4)
        self.assertAlmostEqual(sum(util), 1.0, places=3)

    def test_losses(self):
        self.assertGreater(len(self.model.technique.losses()), 0)
        self.assertGreater(len(self.model.technique.aux_losses()), 0)

    def test_parameter_count(self):
        self.assertGreater(self.model.technique.parameter_count(), 0)

    def test_architecture_summary(self):
        arch = self.model.technique.architecture_summary()
        self.assertEqual(arch["technique"], "moe")
        self.assertEqual(arch["n_experts"], 4)
        self.assertEqual(arch["top_k"], 2)

    def test_predict_before_train(self):
        m = qai.build(technique="moe", in_dim=8, n_classes=3)
        with self.assertRaises(RuntimeError):
            m.predict(X[0])

    def test_top_k_exceeds_experts_raises(self):
        with self.assertRaises(AssertionError):
            qai.build(technique="moe", n_experts=2, top_k=5)

    def test_regression_task(self):
        yr = np.random.randn(100).astype(np.float32)
        m = qai.build(technique="moe", in_dim=8, d_model=8, n_experts=2, top_k=1,
                      hidden=16, task_type="regression", epochs=5, lr=1e-3, batch_size=16)
        m.train(X, y=yr, verbose=False)
        self.assertTrue(m.technique.is_trained())


class TestGNNNodeClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.N = 20
        cls.nf = np.random.randn(cls.N, 8).astype(np.float32)
        cls.adj = (np.random.rand(cls.N, cls.N) > 0.7).astype(np.float32)
        cls.adj = (cls.adj + cls.adj.T) / 2
        np.fill_diagonal(cls.adj, 1)
        cls.labels = np.random.randint(0, 3, cls.N)
        cls.model = qai.build(technique="gnn", in_dim=8, hidden=32, n_layers=2,
                              n_classes=3, task_type="node_classification", epochs=20, lr=1e-3)
        cls.model.train(cls.nf, y=cls.labels, adj_matrix=cls.adj, verbose=False)

    def test_trained(self):
        self.assertTrue(self.model.technique.is_trained())

    def test_predict_shape(self):
        preds = self.model.predict(self.nf, adj_matrix=self.adj)
        self.assertEqual(preds.shape, (self.N,))

    def test_get_embeddings(self):
        emb = self.model.technique.get_node_embeddings(self.nf, self.adj)
        self.assertEqual(emb.shape, (self.N,))

    def test_losses(self):
        self.assertGreater(len(self.model.technique.losses()), 0)

    def test_parameter_count(self):
        self.assertGreater(self.model.technique.parameter_count(), 0)

    def test_predict_before_train(self):
        m = qai.build(technique="gnn", in_dim=8, n_classes=3)
        with self.assertRaises(RuntimeError):
            m.predict(self.nf, adj_matrix=self.adj)


class TestGNNLinkPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.N = 15
        cls.nf = np.random.randn(cls.N, 8).astype(np.float32)
        cls.adj = (np.random.rand(cls.N, cls.N) > 0.7).astype(np.float32)
        cls.adj = (cls.adj + cls.adj.T) / 2; np.fill_diagonal(cls.adj, 1)
        edges = []
        for _ in range(40):
            s, d = np.random.randint(0, cls.N, 2)
            label = 1 if cls.adj[s, d] > 0 else 0
            edges.append([s, d, label])
        cls.edges = np.array(edges)
        cls.model = qai.build(technique="gnn", in_dim=8, hidden=32, n_layers=2,
                              task_type="link_prediction", epochs=20, lr=1e-3)
        cls.model.train(cls.nf, y=cls.edges, adj_matrix=cls.adj, verbose=False)

    def test_trained(self):
        self.assertTrue(self.model.technique.is_trained())

    def test_predict_edges(self):
        pairs = np.array([[0, 1], [2, 3], [4, 5]])
        probs = self.model.technique.predict_edges(pairs, self.nf, self.adj)
        self.assertEqual(len(probs), 3)
        for p in probs:
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 1)


class TestGNNGraphClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.graphs = []
        for _ in range(20):
            n = np.random.randint(5, 10)
            nf = np.random.randn(n, 8).astype(np.float32)
            am = (np.random.rand(n, n) > 0.7).astype(np.float32)
            am = (am + am.T) / 2; np.fill_diagonal(am, 1)
            cls.graphs.append((nf, am))
        cls.labels = np.random.randint(0, 2, 20)
        cls.model = qai.build(technique="gnn", in_dim=8, hidden=32, n_layers=2,
                              n_classes=2, task_type="graph_classification", epochs=20, lr=1e-3)
        cls.model.train(cls.graphs, y=cls.labels, verbose=False)

    def test_trained(self):
        self.assertTrue(self.model.technique.is_trained())

    def test_predict_single_graph(self):
        pred = self.model.predict(self.graphs[0])
        self.assertIsInstance(pred, (int, np.integer))

    def test_losses(self):
        self.assertGreater(len(self.model.technique.losses()), 0)


class TestIntegration(unittest.TestCase):
    def test_in_registry(self):
        from qai.techniques import _REGISTRY
        self.assertIn("moe", _REGISTRY)
        self.assertIn("gnn", _REGISTRY)

    def test_architecture_summary(self):
        m = qai.build(technique="moe", in_dim=8, n_classes=3, epochs=1)
        m.train(X[:20], y[:20], verbose=False)
        arch = m.technique.architecture_summary()
        self.assertIn("expert_utilization", arch)

    def test_gnn_summary(self):
        m = qai.build(technique="gnn", in_dim=8, n_classes=3)
        arch = m.technique.architecture_summary()
        self.assertEqual(arch["technique"], "gnn")


if __name__ == "__main__":
    unittest.main()
