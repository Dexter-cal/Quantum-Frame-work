"""Tests for qai.finalize — OutputControl, ExtensibilityGuardrails, LazyLoader, UniversalManipulator, LogicVariantManager, TwoTierLogic, AutomationLibrary, NumericalCorrectness, OperationalMaturity."""
import sys, os, unittest, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qai


class TestOutputControl(unittest.TestCase):
    def test_valid_output(self):
        ctrl = qai.OutputControl(max_tokens=100)
        result = ctrl.validate("hello world")
        self.assertTrue(result["valid"])

    def test_max_tokens_exceeded(self):
        ctrl = qai.OutputControl(max_tokens=2)
        result = ctrl.validate("one two three four")
        self.assertFalse(result["valid"])

    def test_banned_words(self):
        ctrl = qai.OutputControl(banned_words=["TODO"])
        result = ctrl.validate("TODO: fix this")
        self.assertFalse(result["valid"])

    def test_required_words(self):
        ctrl = qai.OutputControl(required_words=["def"])
        result = ctrl.validate("def hello(): pass")
        self.assertTrue(result["valid"])
        result2 = ctrl.validate("hello world")
        self.assertFalse(result2["valid"])

    def test_json_format(self):
        ctrl = qai.OutputControl(format="json")
        result = ctrl.validate('{"a": 1}')
        self.assertTrue(result["valid"])
        result2 = ctrl.validate("not json")
        self.assertFalse(result2["valid"])

    def test_enforce(self):
        ctrl = qai.OutputControl(required_words=["hello"])
        result = ctrl.enforce(lambda p: "hello world", "test")
        self.assertIn("hello", result)

    def test_violation_log(self):
        ctrl = qai.OutputControl(banned_words=["bad"])
        ctrl.validate("this is bad")
        self.assertGreater(len(ctrl.get_violation_log()), 0)


class TestExtensibilityGuardrails(unittest.TestCase):
    def test_valid_registration(self):
        guard = qai.ExtensibilityGuardrails()
        result = guard.check_registration("my_tech", lambda: None, type="technique")
        self.assertTrue(result["allowed"])

    def test_name_collision(self):
        guard = qai.ExtensibilityGuardrails()
        guard.check_registration("my_tech", lambda: None, type="technique")
        result = guard.check_registration("my_tech", lambda: None, type="technique")
        self.assertFalse(result["allowed"])

    def test_invalid_name(self):
        guard = qai.ExtensibilityGuardrails()
        result = guard.check_registration("_private", lambda: None)
        self.assertFalse(result["allowed"])

    def test_builtin_name(self):
        guard = qai.ExtensibilityGuardrails()
        result = guard.check_registration("eval", lambda: None)
        self.assertFalse(result["allowed"])

    def test_quota_exceeded(self):
        guard = qai.ExtensibilityGuardrails(max_techniques=1)
        guard.check_registration("t1", lambda: None, type="technique")
        result = guard.check_registration("t2", lambda: None, type="technique")
        self.assertFalse(result["allowed"])

    def test_unregister(self):
        guard = qai.ExtensibilityGuardrails()
        guard.check_registration("t1", lambda: None, type="technique")
        self.assertTrue(guard.unregister("t1"))

    def test_summary(self):
        guard = qai.ExtensibilityGuardrails()
        guard.check_registration("t1", lambda: None, type="technique")
        s = guard.summary()
        self.assertEqual(s["total_registered"], 1)


class TestLazyLoader(unittest.TestCase):
    def test_register_and_get(self):
        loader = qai.LazyLoader()
        loader.register("m1", lambda: "model_object")
        m = loader.get("m1")
        self.assertEqual(m, "model_object")

    def test_lazy_loading(self):
        load_count = [0]
        def loader_fn():
            load_count[0] += 1
            return f"model_{load_count[0]}"
        loader = qai.LazyLoader()
        loader.register("m1", loader_fn)
        loader.get("m1")
        loader.get("m1")  # should not reload
        self.assertEqual(load_count[0], 1)

    def test_release(self):
        loader = qai.LazyLoader()
        loader.register("m1", lambda: "model")
        loader.get("m1")
        self.assertTrue(loader.release("m1"))
        self.assertFalse(loader.release("m1"))

    def test_stats(self):
        loader = qai.LazyLoader()
        loader.register("m1", lambda: "model")
        loader.get("m1")
        stats = loader.get_stats()
        self.assertIn("loaded", stats)


class TestUniversalManipulator(unittest.TestCase):
    def setUp(self):
        X = np.array([[1.0],[2.0],[3.0]])
        y = np.array([2.0,4.0,6.0])
        self.model = qai.build(technique="regression")
        self.model.train(X, y, verbose=False)

    def test_scale_weights(self):
        man = qai.UniversalManipulator()
        man.load_model(self.model)
        man.scale_weights(0.5)
        result = man.apply()
        self.assertEqual(result["ops_applied"], 1)

    def test_zero_bias(self):
        man = qai.UniversalManipulator()
        man.load_model(self.model)
        man.zero_bias()
        result = man.apply()
        self.assertEqual(result["ops_applied"], 1)

    def test_dry_run(self):
        man = qai.UniversalManipulator()
        man.load_model(self.model)
        man.scale_weights(0.5)
        result = man.dry_run()
        self.assertEqual(result["ops_applied"], 1)

    def test_chained_ops(self):
        man = qai.UniversalManipulator()
        man.load_model(self.model)
        man.scale_weights(0.5).zero_bias().shift_weights(0.1)
        result = man.apply()
        self.assertEqual(result["ops_applied"], 3)


class TestLogicVariantManager(unittest.TestCase):
    def setUp(self):
        self.mgr = qai.LogicVariantManager()

    def test_save_and_activate(self):
        self.mgr.save_variant("v1", lambda x: f"v1:{x}")
        self.mgr.activate("v1")
        fn = self.mgr.get_active()
        self.assertEqual(fn("test"), "v1:test")

    def test_list_variants(self):
        self.mgr.save_variant("v1", lambda x: x)
        self.mgr.save_variant("v2", lambda x: x)
        self.assertEqual(len(self.mgr.list_variants()), 2)

    def test_score_variant(self):
        self.mgr.save_variant("v1", lambda x: x * 2)
        score = self.mgr.score_variant("v1", lambda fn: 1.0 if fn(2) == 4 else 0.0)
        self.assertEqual(score, 1.0)

    def test_auto_promote(self):
        self.mgr.save_variant("v1", lambda x: x)
        self.mgr.save_variant("v2", lambda x: x * 2)
        best = self.mgr.auto_promote(lambda fn: 1.0 if fn(2) == 4 else 0.5)
        self.assertEqual(best, "v2")


class TestTwoTierLogic(unittest.TestCase):
    def test_shallow_sufficient(self):
        logic = qai.TwoTierLogic(
            shallow_fn=lambda x: {"result": f"quick:{x}", "confidence": 0.9},
            deep_fn=lambda x: {"result": f"thorough:{x}", "confidence": 1.0},
            confidence_threshold=0.7,
        )
        result = logic.execute("test")
        self.assertEqual(result["tier_used"], "shallow")

    def test_escalate_to_deep(self):
        logic = qai.TwoTierLogic(
            shallow_fn=lambda x: {"result": "unsure", "confidence": 0.3},
            deep_fn=lambda x: {"result": "thorough", "confidence": 0.95},
            confidence_threshold=0.7,
        )
        result = logic.execute("test")
        self.assertEqual(result["tier_used"], "deep")
        self.assertTrue(result["escalated"])

    def test_force_deep(self):
        logic = qai.TwoTierLogic(
            shallow_fn=lambda x: {"result": "quick", "confidence": 1.0},
            deep_fn=lambda x: {"result": "deep", "confidence": 1.0},
        )
        result = logic.execute("test", force_tier="deep")
        self.assertEqual(result["tier_used"], "deep")

    def test_cache(self):
        logic = qai.TwoTierLogic(
            shallow_fn=lambda x: {"result": "unsure", "confidence": 0.3},
            deep_fn=lambda x: {"result": f"deep:{x}", "confidence": 1.0},
            cache_deep=True,
        )
        r1 = logic.execute("test")
        r2 = logic.execute("test")
        self.assertTrue(r2.get("cached", False))

    def test_stats(self):
        logic = qai.TwoTierLogic(shallow_fn=lambda x: x, deep_fn=lambda x: x)
        logic.execute("test")
        s = logic.get_stats()
        self.assertGreater(s["shallow_calls"], 0)


class TestAutomationLibrary(unittest.TestCase):
    def setUp(self):
        self.X = np.random.randn(30, 4).astype(np.float32)
        self.y = np.random.randint(0, 2, 30)
        self.auto = qai.AutomationLibrary()

    def test_auto_train(self):
        result = self.auto.auto_train("classifier", self.X, self.y)
        self.assertIn("model", result)
        self.assertIn("elapsed_s", result)

    def test_auto_evaluate(self):
        model = qai.build(technique="classifier")
        model.train(self.X, self.y, verbose=False)
        result = self.auto.auto_evaluate(model, self.X, self.y)
        self.assertIn("accuracy", result)

    def test_auto_compare(self):
        result = self.auto.auto_compare(["classifier", "decision_tree"], self.X, self.y)
        self.assertIn("best_technique", result)

    def test_auto_optimize(self):
        model = qai.build(technique="classifier")
        model.train(self.X, self.y, verbose=False)
        result = self.auto.auto_optimize(model, self.X, self.y,
                                          param_grid={"C": [0.1, 1.0]})
        self.assertIn("best_params", result)

    def test_workflow(self):
        self.auto.workflow("test", [lambda **kw: "step1", lambda **kw: "step2"])
        results = self.auto.run_workflow("test")
        self.assertEqual(len(results), 2)


class TestNumericalCorrectness(unittest.TestCase):
    def test_check_data_clean(self):
        X_train = np.random.randn(20, 4)
        X_test = np.random.randn(10, 4)
        checker = qai.NumericalCorrectness()
        report = checker.check_data(X_train, X_test)
        self.assertIsInstance(report["issues"], list)

    def test_check_data_leakage(self):
        X = np.random.randn(20, 4)
        checker = qai.NumericalCorrectness()
        report = checker.check_data(X, X[:5])  # overlap
        self.assertTrue(report["has_leakage"])

    def test_check_data_nan(self):
        X_train = np.array([[1.0, float("nan")], [2.0, 3.0]])
        X_test = np.array([[4.0, 5.0]])
        checker = qai.NumericalCorrectness()
        report = checker.check_data(X_train, X_test)
        self.assertIn("nan_in_X_train", report["issues"])

    def test_check_model(self):
        X = np.random.randn(20, 4).astype(np.float32)
        y = np.random.randint(0, 2, 20)
        model = qai.build(technique="classifier")
        model.train(X, y, verbose=False)
        checker = qai.NumericalCorrectness()
        report = checker.check_model(model, X, y)
        self.assertIn("stability_score", report)

    def test_reproducibility(self):
        X = np.random.randn(20, 4).astype(np.float32)
        y = np.random.randint(0, 2, 20)
        def train_fn(X, y):
            import qai
            m = qai.build(technique="classifier")
            m.train(X, y, verbose=False)
            return m
        checker = qai.NumericalCorrectness()
        result = checker.check_reproducibility(train_fn, X, y)
        self.assertIn("reproducible", result)


class TestOperationalMaturity(unittest.TestCase):
    def setUp(self):
        self.ops = qai.OperationalMaturity()

    def test_define_slo(self):
        self.ops.define_slo("latency", target_ms=100)
        self.assertEqual(len(self.ops._slos), 1)

    def test_record_metric(self):
        self.ops.record_metric("latency", 50)
        self.assertEqual(len(self.ops._metrics["latency"]), 1)

    def test_check_alerts(self):
        self.ops.define_slo("latency", target_ms=50)
        self.ops.record_metric("latency", 100)
        alerts = self.ops.check_alerts()
        self.assertGreater(len(alerts), 0)

    def test_no_alert_when_within_slo(self):
        self.ops.define_slo("latency", target_ms=200)
        self.ops.record_metric("latency", 50)
        alerts = self.ops.check_alerts()
        self.assertEqual(len(alerts), 0)

    def test_incident(self):
        inc_id = self.ops.create_incident("test incident")
        self.assertEqual(len(self.ops.get_incidents(status="open")), 1)
        self.ops.resolve_incident(inc_id)
        self.assertEqual(len(self.ops.get_incidents(status="open")), 0)

    def test_maturity_score(self):
        score = self.ops.maturity_score()
        self.assertIn("overall", score)
        self.assertIn("level", score)

    def test_summary(self):
        self.ops.define_slo("latency", 100)
        self.ops.record_metric("latency", 50)
        s = self.ops.summary()
        self.assertEqual(s["slos_defined"], 1)


class TestIntegration(unittest.TestCase):
    def test_all_accessible(self):
        for name in ["OutputControl", "ExtensibilityGuardrails", "LazyLoader",
                      "UniversalManipulator", "LogicVariantManager", "TwoTierLogic",
                      "AutomationLibrary", "NumericalCorrectness",
                      "OperationalMaturity", "finalize"]:
            self.assertTrue(hasattr(qai, name), f"Missing: {name}")

    def test_output_control_with_model(self):
        """OutputControl validates model-generated text."""
        model = qai.build(technique="regression")
        model.train([[1.0],[2.0]], [2.0, 4.0], verbose=False)
        ctrl = qai.OutputControl(max_tokens=10, required_words=[])
        text = str(model.predict([3.0]))
        result = ctrl.validate(text)
        self.assertIsInstance(result["valid"], bool)

    def test_ops_with_monitoring(self):
        """OperationalMaturity can monitor a deployed model."""
        ops = qai.OperationalMaturity()
        ops.define_slo("latency_ms", target_ms=200)  # target: 200ms max
        ops.record_metric("latency_ms", 50)  # 50ms < 200ms → within SLO → no alert
        alerts = ops.check_alerts()
        self.assertEqual(len(alerts), 0)


if __name__ == "__main__":
    unittest.main()
