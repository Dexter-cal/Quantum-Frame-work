"""Tests for qai.upgrades — finetune_model, FederatedCluster, MemorySpanConfig, WeightRegionLoader, WeightBlueprint, PeerConnection, create_tool_cli, publish_tool."""
import sys, os, unittest, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qai


class TestFinetuneModel(unittest.TestCase):
    def setUp(self):
        X = np.array([[1.0],[2.0],[3.0]])
        y = np.array([2.0, 4.0, 6.0])
        self.model = qai.build(technique="regression")
        self.model.train(X, y, verbose=False)

    def test_retrain_strategy(self):
        X_new = np.array([[4.0],[5.0]])
        y_new = np.array([8.0, 10.0])
        qai.finetune_model(self.model, X_new, y_new, strategy="retrain")
        pred = self.model.predict([6.0])
        self.assertIsNotNone(pred)

    def test_incremental_strategy(self):
        # Regression has finetune(), so incremental should work
        X_new = np.array([[4.0]])
        y_new = np.array([8.0])
        qai.finetune_model(self.model, X_new, y_new, strategy="incremental")
        self.assertTrue(self.model.technique.is_trained())

    def test_universal_finetune_on_classifier(self):
        X = np.array([[1.0],[2.0],[3.0],[4.0]])
        y = np.array([0, 0, 1, 1])
        model = qai.build(technique="classifier")
        model.train(X, y, verbose=False)
        # Classifier doesn't have finetune() — but universal finetune works
        X_new = np.array([[5.0],[6.0]])
        y_new = np.array([1, 1])
        qai.finetune_model(model, X_new, y_new, strategy="retrain")
        self.assertTrue(model.technique.is_trained())


class TestFederatedCluster(unittest.TestCase):
    def setUp(self):
        X = np.array([[1.0],[2.0],[3.0],[4.0]])
        y = np.array([0, 0, 1, 1])
        self.server_model = qai.build(technique="classifier")
        self.server_model.train(X, y, verbose=False)
        self.cluster = qai.FederatedCluster(self.server_model, n_rounds=3, min_clients_per_round=2)
        # Each client needs both classes for classifier training
        self.cluster.add_client("c1", data=np.array([[1.0],[3.0]]), labels=np.array([0, 1]))
        self.cluster.add_client("c2", data=np.array([[2.0],[4.0]]), labels=np.array([0, 1]))

    def test_train(self):
        result = self.cluster.train()
        self.assertIn("rounds_completed", result)
        self.assertGreater(result["rounds_completed"], 0)

    def test_list_clients(self):
        self.assertEqual(sorted(self.cluster.list_clients()), ["c1", "c2"])

    def test_remove_client(self):
        self.cluster.remove_client("c1")
        self.assertEqual(len(self.cluster.list_clients()), 1)

    def test_not_enough_clients(self):
        cluster = qai.FederatedCluster(self.server_model, n_rounds=1, min_clients_per_round=5)
        cluster.add_client("c1", data=np.array([[1.0]]), labels=np.array([0]))
        with self.assertRaises(ValueError):
            cluster.train()


class TestMemorySpanConfig(unittest.TestCase):
    def test_config(self):
        config = qai.MemorySpanConfig(memory_span=256, focus_width=64,
                                       position_scale=50000.0, stretch_factor=2.0)
        params = config.to_technique_params()
        self.assertEqual(params["max_seq_len"], 256)
        self.assertEqual(params["sliding_window"], 64)
        self.assertEqual(params["rope_theta"], 50000.0)
        self.assertEqual(params["rope_scaling"], 2.0)

    def test_configure_memory_span(self):
        X = np.array([[1.0],[2.0],[3.0]])
        y = np.array([0, 0, 1])
        model = qai.build(technique="classifier")
        model.train(X, y, verbose=False)
        result = qai.configure_memory_span(model, memory_span=256)
        self.assertTrue(result["applied"])


class TestWeightBlueprint(unittest.TestCase):
    def setUp(self):
        X = np.array([[1.0],[2.0],[3.0]])
        y = np.array([2.0, 4.0, 6.0])
        self.model = qai.build(technique="regression")
        self.model.train(X, y, verbose=False)

    def test_from_model(self):
        bp = qai.WeightBlueprint.from_model(self.model)
        self.assertGreater(len(bp.layers), 0)

    def test_save_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            bp = qai.WeightBlueprint.from_model(self.model)
            bp.save(path)
            bp2 = qai.WeightBlueprint.load(path)
            self.assertEqual(len(bp2.layers), len(bp.layers))
        finally:
            os.unlink(path)

    def test_tree(self):
        bp = qai.WeightBlueprint.from_model(self.model)
        tree = bp.tree()
        self.assertIn("regression", tree)
        self.assertIn("mean=", tree)

    def test_diff(self):
        bp1 = qai.WeightBlueprint.from_model(self.model)
        # Create a different model
        m2 = qai.build(technique="regression")
        m2.train([[1.0]], [3.0], verbose=False)
        bp2 = qai.WeightBlueprint.from_model(m2)
        diff = bp1.diff(bp2)
        self.assertIn("changes", diff)


class TestWeightRegionLoader(unittest.TestCase):
    def setUp(self):
        X = np.array([[1.0],[2.0]])
        y = np.array([0, 1])
        self.model = qai.build(technique="classifier")
        self.model.train(X, y, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            self.path = f.name
        qai.save_bundle(self.model, self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_list_regions(self):
        loader = qai.WeightRegionLoader(self.path)
        regions = loader.list_regions()
        self.assertGreater(len(regions), 0)

    def test_load_region(self):
        loader = qai.WeightRegionLoader(self.path)
        regions = loader.list_regions()
        weights = loader.load_region(regions[0])
        self.assertGreater(len(weights), 0)

    def test_unload_region(self):
        loader = qai.WeightRegionLoader(self.path)
        regions = loader.list_regions()
        loader.load_region(regions[0])
        n = loader.unload_region(regions[0])
        self.assertGreaterEqual(n, 0)

    def test_region_info(self):
        loader = qai.WeightRegionLoader(self.path)
        info = loader.get_region_info()
        self.assertIsInstance(info, dict)


class TestPeerConnection(unittest.TestCase):
    def test_init_connect_mode(self):
        peer = qai.PeerConnection(mode="connect", host="localhost", port=19999)
        self.assertFalse(peer.is_connected)

    def test_init_listen_mode(self):
        peer = qai.PeerConnection(mode="listen", port=19998)
        self.assertFalse(peer.is_connected)

    def test_connect_fails_gracefully(self):
        peer = qai.PeerConnection(mode="connect", host="localhost", port=1)
        result = peer.connect()
        self.assertFalse(result)

    def test_close(self):
        peer = qai.PeerConnection(mode="connect")
        peer.close()
        self.assertFalse(peer.is_connected)


class TestToolCLI(unittest.TestCase):
    def test_create_tool_cli(self):
        def my_tool(x, y=1):
            return {"result": int(x) + int(y)}
        code = qai.create_tool_cli("my_tool", my_tool, "Adds two numbers")
        self.assertIn("def my_tool", code)
        self.assertIn("argparse", code)

    def test_publish_tool(self):
        def my_tool(x):
            return {"result": x}
        path = qai.publish_tool("test_tool", my_tool, "Test tool")
        self.assertTrue(os.path.exists(path))
        os.unlink(path)


class TestIntegration(unittest.TestCase):
    def test_all_accessible(self):
        for name in ["finetune_model", "FederatedCluster", "MemorySpanConfig",
                      "configure_memory_span", "WeightRegionLoader", "WeightBlueprint",
                      "PeerConnection", "create_tool_cli", "publish_tool", "upgrades"]:
            self.assertTrue(hasattr(qai, name), f"Missing: {name}")

    def test_blueprint_with_bundle(self):
        """WeightBlueprint works on models saved as bundles."""
        X = np.array([[1.0],[2.0]])
        y = np.array([0, 1])
        model = qai.build(technique="classifier")
        model.train(X, y, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".qmodel", delete=False) as f:
            path = f.name
        try:
            qai.save_bundle(model, path)
            loader = qai.WeightRegionLoader(path)
            bp = qai.WeightBlueprint.from_model(model)
            self.assertGreater(len(bp.layers), 0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
