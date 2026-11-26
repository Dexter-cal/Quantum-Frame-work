"""Tests for qai.completion — Layer, AutoTrainTarget, RewardSystem, WeightToWords, MultimodalGenerator, ModelCard, StressTester, WeightSafetyToolkit, CICDPipeline, ExperimentDashboard."""
import sys, os, unittest, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qai


class TestLayer(unittest.TestCase):
    def test_create_linear(self):
        l = qai.Layer.create("linear", in_dim=4, out_dim=2)
        out = l.forward(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(out.shape, (2,))

    def test_sequential(self):
        net = qai.Layer.Sequential()
        net.add("l1", qai.Layer.create("linear", in_dim=4, out_dim=8))
        net.add("r1", qai.Layer.create("relu"))
        net.add("l2", qai.Layer.create("linear", in_dim=8, out_dim=2))
        out = net.forward(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(out.shape, (2,))

    def test_freeze(self):
        l = qai.Layer.create("relu")
        self.assertFalse(l.is_frozen)
        l.freeze()
        self.assertTrue(l.is_frozen)

    def test_freeze_in_sequential(self):
        net = qai.Layer.Sequential()
        net.add("l1", qai.Layer.create("linear", in_dim=4, out_dim=2))
        net.add("r1", qai.Layer.create("relu"))
        net.freeze("l1")
        self.assertTrue(net.layers["l1"].is_frozen)
        self.assertFalse(net.layers["r1"].is_frozen)

    def test_summary(self):
        net = qai.Layer.Sequential()
        net.add("l1", qai.Layer.create("linear", in_dim=4, out_dim=2))
        s = net.summary()
        self.assertEqual(len(s), 1)
        self.assertEqual(s[0]["name"], "l1")

    def test_registered_layers(self):
        for name in ["linear", "relu", "sigmoid", "tanh", "softmax", "dropout", "layernorm"]:
            self.assertTrue(name in qai.Layer._registry)

    def test_unknown_layer(self):
        with self.assertRaises(ValueError):
            qai.Layer.create("nonexistent")


class TestAutoTrainTarget(unittest.TestCase):
    def test_train_to_target(self):
        X = np.array([[1.0],[2.0],[3.0],[4.0],[5.0]])
        y = np.array([0,0,0,1,1])
        model = qai.build(technique="classifier")
        trainer = qai.AutoTrainTarget(target_metric="accuracy", target_value=0.5, increment=5)
        result = trainer.train(model, X, y, max_epochs=20)
        self.assertIn("achieved", result)
        self.assertIn("epochs_used", result)

    def test_max_epochs(self):
        X = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randint(0, 2, 10)
        model = qai.build(technique="classifier")
        trainer = qai.AutoTrainTarget(target_value=1.0, increment=5)
        result = trainer.train(model, X, y, max_epochs=10)
        self.assertLessEqual(result["epochs_used"], 15)

    def test_history(self):
        X = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randint(0, 2, 10)
        model = qai.build(technique="classifier")
        trainer = qai.AutoTrainTarget(increment=5)
        trainer.train(model, X, y, max_epochs=10)
        self.assertGreater(len(trainer.get_history()), 0)


class TestRewardSystem(unittest.TestCase):
    def test_add_and_compute(self):
        rs = qai.RewardSystem()
        rs.add_reward("distance", lambda s, a: -abs(s[0] - 1.0))
        rs.add_penalty("time", lambda s, a: -0.01)
        total = rs.compute(state=[0.5], action=0)
        self.assertIsInstance(total, float)

    def test_multiple_rewards(self):
        rs = qai.RewardSystem()
        rs.add_reward("r1", lambda s, a: 1.0, weight=0.5)
        rs.add_reward("r2", lambda s, a: 2.0, weight=0.3)
        total = rs.compute(state=[0], action=0)
        self.assertAlmostEqual(total, 0.5 * 1.0 + 0.3 * 2.0)

    def test_shaping(self):
        rs = qai.RewardSystem()
        rs.add_reward("base", lambda s, a: 0.0)
        rs.set_shaping(lambda s, ns: 0.1)
        total = rs.compute(state=[0], action=0, next_state=[1])
        self.assertGreater(total, 0)

    def test_history(self):
        rs = qai.RewardSystem()
        rs.add_reward("r", lambda s, a: 1.0)
        rs.compute(state=[0], action=0)
        self.assertGreater(len(rs.get_history()), 0)


class TestWeightToWords(unittest.TestCase):
    def setUp(self):
        X = np.array([[1.0],[2.0],[3.0]])
        y = np.array([2.0,4.0,6.0])
        self.model = qai.build(technique="regression")
        self.model.train(X, y, verbose=False)

    def test_explain(self):
        w2w = qai.WeightToWords()
        text = w2w.explain(self.model, feature_names=["x"])
        self.assertIn("x", text)
        self.assertIn("weight", text.lower())

    def test_explain_with_no_names(self):
        w2w = qai.WeightToWords()
        text = w2w.explain(self.model)
        self.assertIn("feature_0", text)

    def test_compare(self):
        m2 = qai.build(technique="regression")
        m2.train([[1.0],[2.0]], [3.0, 5.0], verbose=False)
        w2w = qai.WeightToWords()
        text = w2w.compare(self.model, m2)
        self.assertIn("feature_0", text)


class TestMultimodalGenerator(unittest.TestCase):
    def test_register_and_generate(self):
        mm = qai.MultimodalGenerator()
        mm.register("text", lambda x: f"processed: {x}", input_modal="text", output_modal="text")
        result = mm.generate("hello", output_modal="text")
        self.assertIn("hello", result)

    def test_no_matching_model(self):
        mm = qai.MultimodalGenerator()
        result = mm.generate("hello", output_modal="image")
        self.assertIn("No model", result)

    def test_chain(self):
        mm = qai.MultimodalGenerator()
        mm.register("a", lambda x: f"processed({x})", input_modal="text", output_modal="text")
        result = mm.chain("start", ["text", "text", "text"])
        self.assertIn("processed", result)

    def test_list_models(self):
        mm = qai.MultimodalGenerator()
        mm.register("m1", lambda x: x, input_modal="text", output_modal="image")
        models = mm.list_models()
        self.assertEqual(len(models), 1)


class TestModelCard(unittest.TestCase):
    def setUp(self):
        self.model = qai.build(technique="regression")
        self.model.train([[1.0],[2.0]], [2.0, 4.0], verbose=False)

    def test_auto_fill(self):
        card = qai.ModelCard(model=self.model, name="test")
        self.assertIn("technique", card.details)

    def test_set_performance(self):
        card = qai.ModelCard(model=self.model)
        card.set_performance(accuracy=0.95, f1=0.93)
        self.assertEqual(card.performance["accuracy"], 0.95)

    def test_to_markdown(self):
        card = qai.ModelCard(model=self.model, name="test")
        card.set_intended_use("testing")
        card.set_limitations("only small data")
        md = card.to_markdown()
        self.assertIn("# Model Card: test", md)
        self.assertIn("testing", md)
        self.assertIn("only small data", md)

    def test_save(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            path = f.name
        try:
            card = qai.ModelCard(model=self.model, name="test")
            card.save(path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)


class TestStressTester(unittest.TestCase):
    def setUp(self):
        X = np.random.randn(30, 4).astype(np.float32)
        y = np.random.randint(0, 2, 30)
        self.model = qai.build(technique="classifier")
        self.model.train(X, y, verbose=False)
        self.X, self.y = X, y

    def test_stress_test(self):
        tester = qai.StressTester()
        report = tester.test(self.model, self.X, self.y)
        self.assertIn("robustness_score", report)
        self.assertIn("noise_robustness", report["tests"])
        self.assertIn("feature_sensitivity", report["tests"])
        self.assertIn("edge_cases", report["tests"])

    def test_robustness_score_range(self):
        tester = qai.StressTester()
        report = tester.test(self.model, self.X, self.y)
        self.assertGreaterEqual(report["robustness_score"], 0.0)


class TestWeightSafetyToolkit(unittest.TestCase):
    def setUp(self):
        X = np.array([[1.0],[2.0],[3.0]])
        y = np.array([2.0, 4.0, 6.0])
        self.model = qai.build(technique="regression")
        self.model.train(X, y, verbose=False)
        self.safety = qai.WeightSafetyToolkit()

    def test_snapshot_and_undo(self):
        original = float(self.model.technique.w[0])
        self.safety.snapshot(self.model)
        self.model.technique.w[0] = 999.0
        self.safety.undo(self.model)
        self.assertAlmostEqual(float(self.model.technique.w[0]), original)

    def test_validate_ok(self):
        result = self.safety.validate(self.model)
        self.assertTrue(result["valid"])

    def test_validate_nan(self):
        self.model.technique.w[0] = float("nan")
        result = self.safety.validate(self.model)
        self.assertFalse(result["valid"])

    def test_safe_edit_rollback(self):
        # Use classifier so accuracy comparison works
        X = np.array([[1.0],[2.0],[3.0],[4.0]])
        y = np.array([0, 0, 1, 1])
        model = qai.build(technique="classifier")
        model.train(X, y, verbose=False)
        safety = qai.WeightSafetyToolkit()
        result = safety.safe_edit(
            model,
            lambda m: setattr(m.technique.model, "coef_", np.array([[999.0]])),
            X, y, max_accuracy_drop=0.01,
        )
        self.assertFalse(result["applied"])

    def test_safe_edit_success(self):
        result = self.safety.safe_edit(
            self.model, lambda m: None  # no-op edit
        )
        self.assertTrue(result["applied"])

    def test_audit(self):
        self.safety.snapshot(self.model)
        self.assertGreater(len(self.safety.get_audit()), 0)


class TestCICDPipeline(unittest.TestCase):
    def test_all_pass(self):
        model = qai.build(technique="regression")
        model.train([[1.0],[2.0]], [2.0, 4.0], verbose=False)
        pipe = qai.CICDPipeline()
        pipe.add_stage("train", lambda m: m.train([[1.0],[2.0]], [2.0, 4.0], verbose=False) or True)
        pipe.add_stage("predict", lambda m: m.predict([1.0]) is not None)
        result = pipe.run(model)
        self.assertTrue(result["all_passed"])

    def test_fail_fast(self):
        pipe = qai.CICDPipeline(fail_fast=True)
        pipe.add_stage("fail", lambda m: False)
        pipe.add_stage("never_run", lambda m: True)
        result = pipe.run(None)
        self.assertEqual(result["n_stages_run"], 1)

    def test_history(self):
        model = qai.build(technique="regression")
        model.train([[1.0],[2.0]], [2.0, 4.0], verbose=False)
        pipe = qai.CICDPipeline()
        pipe.add_stage("predict", lambda m: True)
        pipe.run(model)
        self.assertGreater(len(pipe.get_history()), 0)


class TestExperimentDashboard(unittest.TestCase):
    def setUp(self):
        self.dash = qai.ExperimentDashboard()
        self.dash.log_experiment("exp1", technique="rf", accuracy=0.90, latency_ms=5)
        self.dash.log_experiment("exp2", technique="svm", accuracy=0.95, latency_ms=12)
        self.dash.log_experiment("exp3", technique="knn", accuracy=0.88, latency_ms=3)

    def test_leaderboard(self):
        lb = self.dash.leaderboard(metric="accuracy")
        self.assertEqual(lb[0]["name"], "exp2")  # highest accuracy

    def test_compare(self):
        cmp = self.dash.compare(["exp1", "exp2"])
        self.assertIn("exp1", cmp)
        self.assertIn("exp2", cmp)

    def test_trend(self):
        t = self.dash.trend(metric="accuracy")
        self.assertEqual(len(t), 3)

    def test_summary(self):
        s = self.dash.summary()
        self.assertEqual(s["total_experiments"], 3)

    def test_export_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.dash.export_json(path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)


class TestIntegration(unittest.TestCase):
    def test_all_accessible(self):
        for name in ["Layer", "AutoTrainTarget", "RewardSystem", "WeightToWords",
                      "MultimodalGenerator", "ModelCard", "StressTester",
                      "WeightSafetyToolkit", "CICDPipeline", "ExperimentDashboard",
                      "completion"]:
            self.assertTrue(hasattr(qai, name), f"Missing: {name}")

    def test_layer_with_model(self):
        """Layers can be used to build custom model architectures."""
        net = qai.Layer.Sequential()
        net.add("l1", qai.Layer.create("linear", in_dim=4, out_dim=8))
        net.add("relu", qai.Layer.create("relu"))
        net.add("l2", qai.Layer.create("linear", in_dim=8, out_dim=2))
        out = net.forward(np.random.randn(4))
        self.assertEqual(out.shape, (2,))

    def test_model_card_with_stress_test(self):
        """ModelCard can include StressTester results."""
        X = np.random.randn(20, 4).astype(np.float32)
        y = np.random.randint(0, 2, 20)
        model = qai.build(technique="classifier")
        model.train(X, y, verbose=False)
        tester = qai.StressTester()
        report = tester.test(model, X, y)
        card = qai.ModelCard(model=model, name="test")
        card.set_performance(robustness=report["robustness_score"])
        md = card.to_markdown()
        self.assertIn("robustness", md)


if __name__ == "__main__":
    unittest.main()
