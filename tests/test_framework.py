"""Tests for qai.framework — OutputRenderer, KnowledgeGapTracker, IdleTrainer, Marketplace, AdvancedVisualizer, BidirectionalChannel, RouterCore, AutomatedScientist."""
import sys, os, unittest, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qai


class TestOutputRenderer(unittest.TestCase):
    def setUp(self):
        self.r = qai.OutputRenderer()

    def test_render_text(self):
        self.assertEqual(self.r.render("hello"), "hello")

    def test_render_json(self):
        result = self.r.render({"a": 1}, format="json")
        self.assertIn("a", result)

    def test_auto_detect(self):
        result = self.r.render({"a": 1})  # should auto-detect json
        self.assertIn("a", result)

    def test_custom_renderer(self):
        self.r.register("upper", lambda x: str(x).upper(), accepts=str)
        self.assertEqual(self.r.render("hello", format="upper"), "HELLO")

    def test_capability_check(self):
        self.r.register("numbers_only", lambda x: str(x), accepts=int)
        with self.assertRaises(TypeError):
            self.r.render("text", format="numbers_only")

    def test_list_formats(self):
        self.assertIn("text", self.r.list_formats())


class TestKnowledgeGapTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = qai.KnowledgeGapTracker(failure_threshold=0.01, min_gap_size=2)
        self.X = np.random.randn(50, 4).astype(np.float32)
        self.y = np.random.randint(0, 2, 50)
        self.model = qai.build(technique="classifier")
        self.model.train(self.X, self.y, verbose=False)

    def test_identify_gaps(self):
        gaps = self.tracker.identify_gaps(self.model, self.X, self.y)
        self.assertIsInstance(gaps, list)

    def test_suggest_training_data(self):
        gaps = self.tracker.identify_gaps(self.model, self.X, self.y)
        if gaps:
            suggestions = self.tracker.suggest_training_data(gaps[0], n=5)
            self.assertEqual(len(suggestions), 5)

    def test_mark_resolved(self):
        gaps = self.tracker.identify_gaps(self.model, self.X, self.y)
        if gaps:
            self.assertTrue(self.tracker.mark_resolved(gaps[0].id))
            self.assertTrue(gaps[0].resolved)

    def test_summary(self):
        self.tracker.identify_gaps(self.model, self.X, self.y)
        s = self.tracker.summary()
        self.assertIn("total_gaps", s)


class TestIdleTrainer(unittest.TestCase):
    def test_start_stop(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model = qai.build(technique="regression")
        model.train(X, y, verbose=False)
        trainer = qai.IdleTrainer(model, X, y, max_idle_time=1.0, check_interval=0.1)
        trainer.start()
        self.assertTrue(trainer.is_running)
        time.sleep(0.5)
        trainer.stop()
        self.assertFalse(trainer.is_running)

    def test_progress(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([2.0, 4.0])
        model = qai.build(technique="regression")
        model.train(X, y, verbose=False)
        trainer = qai.IdleTrainer(model, X, y, max_idle_time=0.5, check_interval=0.1)
        trainer.start()
        time.sleep(0.3)
        trainer.stop()
        self.assertGreater(len(trainer.get_progress()), 0)


class TestMarketplace(unittest.TestCase):
    def setUp(self):
        self.mp = qai.Marketplace()
        self.entry = qai.MarketplaceEntry(id="m1", name="test-model", type="model",
                                           description="A test model", tags=["nlp", "classification"])

    def test_publish_and_get(self):
        self.mp.publish(self.entry)
        self.assertEqual(self.mp.get("m1").name, "test-model")

    def test_search(self):
        self.mp.publish(self.entry)
        results = self.mp.search("nlp")
        self.assertEqual(len(results), 1)

    def test_search_by_tags(self):
        self.mp.publish(self.entry)
        results = self.mp.search(tags=["classification"])
        self.assertEqual(len(results), 1)

    def test_rate(self):
        self.mp.publish(self.entry)
        self.mp.rate("m1", 5)
        self.assertEqual(self.mp.get("m1").rating, 5.0)

    def test_download(self):
        self.mp.publish(self.entry)
        self.mp.download("m1")
        self.assertEqual(self.mp.get("m1").downloads, 1)

    def test_subscribe(self):
        notified = []
        self.mp.subscribe("m1", lambda e: notified.append(e))
        self.mp.publish(self.entry)
        # No notification on first publish (no subscribers yet at publish time)
        # Publish again to trigger notification
        self.mp.publish(self.entry)
        self.assertGreater(len(notified), 0)

    def test_verify_checksum(self):
        entry = qai.MarketplaceEntry(id="m2", name="checked", type="model",
                                      checksum=hashlib.sha256(b"test").hexdigest())
        self.mp.publish(entry)
        self.assertTrue(self.mp.verify_checksum("m2", b"test"))
        self.assertFalse(self.mp.verify_checksum("m2", b"wrong"))


class TestAdvancedVisualizer(unittest.TestCase):
    def test_weight_heatmap(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model = qai.build(technique="regression")
        model.train(X, y, verbose=False)
        result = qai.AdvancedVisualizer.weight_heatmap(model)
        # Should return raw weights (no matplotlib in test) or None
        self.assertTrue(result is None or isinstance(result, (np.ndarray, )))

    def test_training_curve(self):
        model = qai.build(technique="regression")
        model.train([[1.0], [2.0]], [2.0, 4.0], verbose=False)
        result = qai.AdvancedVisualizer.training_curve(model)
        self.assertTrue(result is None or isinstance(result, np.ndarray))

    def test_confusion_matrix(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        result = qai.AdvancedVisualizer.confusion_matrix_heatmap(y_true, y_pred)
        self.assertIsInstance(result, np.ndarray)

    def test_formula_diff(self):
        m1 = qai.build(technique="regression"); m1.train([[1.0],[2.0]], [2.0,4.0], verbose=False)
        m2 = qai.build(technique="regression"); m2.train([[1.0],[2.0]], [2.0,4.0], verbose=False)
        diff = qai.AdvancedVisualizer.formula_diff(m1, m2)
        self.assertIn("identical", diff)


class TestBidirectionalChannel(unittest.TestCase):
    def test_exchange(self):
        def model_a(x): return f"A({x})"
        def model_b(x): return f"B({x})"
        channel = qai.BidirectionalChannel(model_a, model_b)
        result = channel.exchange("start", max_rounds=3)
        self.assertIn("final_output", result)
        self.assertIn("rounds", result)
        self.assertIn("converged", result)

    def test_convergence(self):
        # Models that converge (return same thing)
        def always_same(x): return "same"
        channel = qai.BidirectionalChannel(always_same, always_same, convergence_threshold=0.01)
        result = channel.exchange("start", max_rounds=5)
        self.assertTrue(result["converged"])

    def test_history(self):
        def a(x): return str(x) + "a"
        def b(x): return str(x) + "b"
        channel = qai.BidirectionalChannel(a, b)
        channel.exchange("x", max_rounds=2)
        self.assertGreater(len(channel.get_history()), 0)


class TestRouterCore(unittest.TestCase):
    def setUp(self):
        self.router = qai.RouterCore(max_loaded=2)
        X = np.array([[1.0],[2.0],[3.0]])
        y_cls = np.array([0,0,1])
        y_reg = np.array([2.0,4.0,6.0])
        self.clf = qai.build(technique="classifier"); self.clf.train(X, y_cls, verbose=False)
        self.reg = qai.build(technique="regression"); self.reg.train(X, y_reg, verbose=False)
        self.router.register("classifier", self.clf)
        self.router.register("regressor", self.reg)

    def test_route(self):
        result = self.router.route({"type": "classifier", "data": [2.0]})
        self.assertIn("model_name", result)
        self.assertIn("result", result)

    def test_auto_predict(self):
        result = self.router.route({"data": [2.0]})  # no type specified
        self.assertIn("model_name", result)

    def test_lru_eviction(self):
        # max_loaded=2, load 2 models, then a 3rd should evict one
        self.router.register("rf", qai.build(technique="random_forest"))
        self.router.route({"type": "classifier", "data": [2.0]})
        self.router.route({"type": "regressor", "data": [2.0]})
        self.router.route({"type": "rf", "data": [2.0]})
        stats = self.router.get_stats()
        self.assertLessEqual(len(stats["loaded"]), 2)

    def test_stats(self):
        self.router.route({"type": "classifier", "data": [2.0]})
        stats = self.router.get_stats()
        self.assertIn("usage", stats)
        self.assertGreater(stats["total_requests"], 0)


class TestAutomatedScientist(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(50, 4).astype(np.float32)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)
        self.scientist = qai.AutomatedScientist(data=self.X, target=self.y, budget=5)

    def test_run(self):
        results = self.scientist.run()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("hypothesis", r)
            self.assertIn("conclusion", r)
            self.assertIn("confidence", r)

    def test_summary(self):
        self.scientist.run()
        s = self.scientist.summary()
        self.assertIn("total_hypotheses", s)
        self.assertIn("accepted", s)

    def test_discoveries(self):
        self.scientist.run()
        self.assertGreater(len(self.scientist.get_discoveries()), 0)

    def test_budget(self):
        scientist = qai.AutomatedScientist(data=self.X, target=self.y, budget=2)
        results = scientist.run()
        self.assertLessEqual(len(results), 2)


class TestIntegration(unittest.TestCase):
    def test_all_accessible(self):
        for name in ["OutputRenderer", "KnowledgeGapTracker", "IdleTrainer",
                      "Marketplace", "AdvancedVisualizer", "BidirectionalChannel",
                      "RouterCore", "AutomatedScientist", "framework"]:
            self.assertTrue(hasattr(qai, name))

    def test_marketplace_with_bundles(self):
        """Marketplace entries can reference .qmodel bundles."""
        mp = qai.Marketplace()
        entry = qai.MarketplaceEntry(id="m1", name="model", type="model",
                                      tags=["test"])
        mp.publish(entry)
        self.assertEqual(len(mp.search("test")), 1)

    def test_router_with_multiple_models(self):
        """Router can route between different model types."""
        router = qai.RouterCore()
        X = np.array([[1.0],[2.0],[3.0]])
        y = np.array([0,0,1])
        m1 = qai.build(technique="classifier"); m1.train(X, y, verbose=False)
        m2 = qai.build(technique="decision_tree"); m2.train(X, y, verbose=False)
        router.register("clf", m1)
        router.register("dt", m2)
        r1 = router.route({"type": "clf", "data": [2.0]})
        r2 = router.route({"type": "dt", "data": [2.0]})
        self.assertNotEqual(r1["model_name"], r2["model_name"])


import hashlib

if __name__ == "__main__":
    unittest.main()
