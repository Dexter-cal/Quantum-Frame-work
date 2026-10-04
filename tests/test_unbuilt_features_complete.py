import unittest
import numpy as np
import os
import qai

class TestUnbuiltFeaturesComplete(unittest.TestCase):
    def test_distributed_training(self):
        ctx = qai.init_distributed_context(rank=0, world_size=2)
        self.assertEqual(ctx["rank"], 0)
        self.assertEqual(ctx["world_size"], 2)

        model = qai.build(type="regression")
        ddp = qai.DistributedDataParallelWrapper(model, rank=0, world_size=2)
        ddp.broadcast_parameters()
        ddp.train([[1, 2], [3, 4]], [1.0, 2.0])
        self.assertTrue(hasattr(model, "predict"))

    def test_bayesian_optimization(self):
        res = qai.bayesian_optimize("classifier", [[1, 2], [3, 4]], [0, 1], {"learning_rate": [0.01, 0.1], "n_estimators": (10, 100)})
        self.assertIn("best_params", res)
        self.assertIn("best_loss", res)

    def test_onnx_export(self):
        model = qai.build(type="classifier")
        model.train([[1, 2], [3, 4]], [0, 1])
        filepath = "/tmp/test_model.onnx"
        meta = qai.export_to_onnx(model, filepath)
        self.assertTrue(os.path.exists(filepath))
        self.assertEqual(meta["format"], "ONNX-v1.12")
        os.remove(filepath)

    def test_telemetry_and_tensorboard(self):
        log = qai.log_telemetry("train_loss", 0.05, step=10)
        self.assertEqual(log["metric"], "train_loss")
        self.assertEqual(log["step"], 10)

        path = qai.sync_tensorboard("/tmp/qai_tensorboard")
        self.assertTrue(os.path.exists(path))

    def test_streaming_event_connector(self):
        connector = qai.StreamingEventConnector("kafka://localhost:9092")
        self.assertFalse(connector.connected)
        with self.assertRaises(RuntimeError):
            connector.fetch_batch(5)

        connector.connect()
        self.assertTrue(connector.connected)
        batch = connector.fetch_batch(5)
        self.assertEqual(batch.shape, (5, 4))

    def test_nas_search(self):
        nas_res = qai.search_architecture([[1, 2], [3, 4]], [0, 1])
        self.assertIn("hidden_layers", nas_res)

    def test_federated_averaging(self):
        w1 = np.array([1.0, 2.0])
        w2 = np.array([3.0, 4.0])
        avg_w = qai.federated_averaging([w1, w2])
        self.assertTrue(np.allclose(avg_w, [2.0, 3.0]))

        server = qai.FederatedServer()
        with self.assertRaises(ValueError):
            server.aggregate([])

    def test_model_watermarking(self):
        model = qai.build(type="regression")
        wm = qai.watermark_model(model, secret_key="my_secret")
        self.assertTrue(wm["watermarked"])
        self.assertTrue(qai.verify_watermark(model, secret_key="my_secret"))
        self.assertFalse(qai.verify_watermark(model, secret_key="wrong_secret"))

    def test_quantum_circuit_simulator(self):
        qc = qai.QuantumCircuitSimulator(n_qubits=2)
        qc.apply_hadamard(0)
        val = qai.quantum_expectation(qc)
        m = qc.measure()
        self.assertIsInstance(val, float)
        self.assertIn(m, [0, 1, 2, 3])

    def test_structured_sparsity_pruning(self):
        model = qai.build(type="regression")
        model.train([[1, 2], [3, 4]], [1.0, 2.0])
        prune_res = qai.prune_structured_sparsity(model, amount=0.5)
        self.assertIn("pruned_count", prune_res)

if __name__ == "__main__":
    unittest.main()
