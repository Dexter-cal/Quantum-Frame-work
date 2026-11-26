"""Tests for qai.production — Self-Play, Skills, Cascade, Contracts, Media."""
import sys, os, unittest, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import qai


class TestMultiRoleSelfPlay(unittest.TestCase):
    def setUp(self):
        def asker(context, turns, role):
            return f"Question #{context.get('difficulty', 0.5):.1f}"
        def answerer(context, turns, role):
            last = turns[-1]["output"] if turns else "unknown"
            return f"Answer to: {last}"
        self.sp = qai.MultiRoleSelfPlay(name="test", max_rounds=20)
        self.sp.add_role("asker", model=asker, goal="ask questions")
        self.sp.add_role("answerer", model=answerer, goal="answer questions")
        self.sp.set_evaluator(lambda turns: {"asker": 0.6, "answerer": 0.4})

    def test_train(self):
        result = self.sp.train(rounds=5)
        self.assertEqual(result["rounds_completed"], 5)
        self.assertIn("leaderboard", result)
        self.assertIn("asker", result["leaderboard"])

    def test_history(self):
        self.sp.train(rounds=3)
        self.assertEqual(len(self.sp.get_history()), 3)

    def test_difficulty_balancing(self):
        sp = qai.MultiRoleSelfPlay(difficulty_balancing=True)
        sp.add_role("a", model=lambda **k: "x")
        sp.add_role("b", model=lambda **k: "y")
        sp.set_evaluator(lambda t: {"a": 0.9, "b": 0.1})
        result = sp.train(rounds=5)
        # Difficulty should have increased (agents scoring high)
        self.assertGreaterEqual(result["final_difficulty"], 0.5)

    def test_role_swap(self):
        sp = qai.MultiRoleSelfPlay(swap_roles=True, max_rounds=20)
        sp.add_role("x", model=lambda **k: "x")
        sp.add_role("y", model=lambda **k: "y")
        sp.set_evaluator(lambda t: {"x": 0.5, "y": 0.5})
        sp.train(rounds=6)
        # After 5 rounds, roles should have swapped
        self.assertNotEqual(sp.role_order, ["x", "y"])

    def test_error_handling(self):
        sp = qai.MultiRoleSelfPlay(max_rounds=5)
        sp.add_role("crash", model=lambda **k: 1/0)  # always crashes
        sp.set_evaluator(lambda t: {})
        result = sp.train(rounds=2)
        self.assertEqual(result["rounds_completed"], 2)


class TestComposableSkills(unittest.TestCase):
    def setUp(self):
        self.skills = qai.ComposableSkills()
        self.skills.register(qai.Skill(
            name="split", fn=lambda text: {"tokens": text.split()},
            inputs=["text"], outputs=["tokens"]))
        self.skills.register(qai.Skill(
            name="count", fn=lambda tokens: {"count": len(tokens)},
            inputs=["tokens"], outputs=["count"], depends_on=["split"]))
        self.skills.register(qai.Skill(
            name="uppercase", fn=lambda tokens: {"upper": [t.upper() for t in tokens]},
            inputs=["tokens"], outputs=["upper"], depends_on=["split"]))

    def test_run_pipeline(self):
        result = self.skills.run_pipeline("hello world foo")
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["tokens"], ["hello", "world", "foo"])
        self.assertEqual(result["upper"], ["HELLO", "WORLD", "FOO"])

    def test_run_single_skill(self):
        result = self.skills.run_skill("count", tokens=["a", "b"])
        self.assertEqual(result["count"], 2)

    def test_list_skills(self):
        self.assertEqual(sorted(self.skills.list_skills()), ["count", "split", "uppercase"])

    def test_unregister(self):
        self.skills.unregister("count")
        self.assertNotIn("count", self.skills.list_skills())

    def test_dependency_resolution(self):
        # count depends on split — split should run first
        result = self.skills.run_pipeline("a b c")
        self.assertIn("tokens", result)  # split ran
        self.assertIn("count", result)   # count ran after split


class TestCascadeInference(unittest.TestCase):
    def setUp(self):
        self.cascade = qai.CascadeInference(escalate_if="confidence < 0.8")
        self.cascade.add_model("small", model=lambda x: f"small({x})",
                               confidence_fn=lambda x: 0.5, cost=0.1)
        self.cascade.add_model("medium", model=lambda x: f"medium({x})",
                               confidence_fn=lambda x: 0.6, cost=0.5)
        self.cascade.add_model("large", model=lambda x: f"large({x})",
                               confidence_fn=lambda x: 0.95, cost=2.0)

    def test_escalation(self):
        result = self.cascade.predict("test")
        # small (0.5) → escalate, medium (0.6) → escalate, large (0.95) → stop
        self.assertEqual(result["model_used"], "large")
        self.assertTrue(result["escalated"])
        self.assertEqual(result["path"], ["small", "medium", "large"])

    def test_no_escalation(self):
        cascade = qai.CascadeInference(escalate_if="confidence < 0.3")
        cascade.add_model("small", model=lambda x: "ok", confidence_fn=lambda x: 0.9, cost=0.1)
        result = cascade.predict("test")
        self.assertEqual(result["model_used"], "small")
        self.assertFalse(result["escalated"])

    def test_cost_tracking(self):
        self.cascade.predict("test")
        self.assertGreater(self.cascade.total_cost, 0)

    def test_escalation_log(self):
        self.cascade.predict("test")
        log = self.cascade.get_escalation_log()
        self.assertGreater(len(log), 0)

    def test_with_qai_model(self):
        import numpy as np
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0])
        m = qai.build(technique="classifier")
        m.train(X, y, verbose=False)
        cascade = qai.CascadeInference(escalate_if="confidence < 0.5")
        cascade.add_model("clf", model=m, confidence_fn=lambda x: 0.9, cost=1.0)
        result = cascade.predict([2.0])
        self.assertIn("prediction", result)


class TestBehaviorContract(unittest.TestCase):
    def setUp(self):
        self.contract = qai.BehaviorContract()
        self.contract.add_rule(qai.BehaviorRule(
            name="not_empty", description="Output must not be empty",
            check=lambda x: len(str(x)) > 0, severity="error"))
        self.contract.add_rule(qai.BehaviorRule(
            name="max_length", description="Output <= 100 chars",
            check=lambda x: len(str(x)) <= 100, severity="warning"))

    def test_passes_valid_output(self):
        result = self.contract.check("hello world")
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["violations"]), 0)

    def test_catches_empty_output(self):
        result = self.contract.check("")
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["violations"]), 1)
        self.assertEqual(result["violations"][0]["rule"], "not_empty")

    def test_warning_for_long_output(self):
        result = self.contract.check("x" * 200)
        # not_empty passes, max_length warns
        self.assertTrue(result["passed"])  # warnings don't block
        self.assertEqual(len(result["warnings"]), 1)

    def test_stats(self):
        self.contract.check("ok")
        self.contract.check("")
        self.contract.check("ok")
        stats = self.contract.get_stats()
        self.assertEqual(stats["total_checks"], 3)
        self.assertEqual(stats["total_violations"], 1)

    def test_auto_rollback(self):
        rollback_called = [False]
        def rollback():
            rollback_called[0] = True
        contract = qai.BehaviorContract(
            auto_rollback_threshold=0.1,
            rollback_fn=rollback)
        contract.add_rule(qai.BehaviorRule(
            name="always_fail", description="always fails",
            check=lambda x: False, severity="error"))
        # Run 10 checks — all will fail → 100% violation rate → rollback
        for _ in range(10):
            contract.check("test")
        self.assertTrue(rollback_called[0])

    def test_save(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.contract.save(path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_remove_rule(self):
        self.contract.remove_rule("not_empty")
        result = self.contract.check("")
        self.assertTrue(result["passed"])  # only warning rule left


class TestMediaManipulator(unittest.TestCase):
    def setUp(self):
        self.img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        self.mm = qai.MediaManipulator()

    def test_resize(self):
        result = (self.mm.load(self.img).resize(25, 25).apply())
        self.assertEqual(result.shape, (25, 25, 3))

    def test_grayscale(self):
        result = (self.mm.load(self.img).grayscale().apply())
        self.assertEqual(result.ndim, 2)  # 2D after grayscale

    def test_flip_horizontal(self):
        result = (self.mm.load(self.img).flip("horizontal").apply())
        np.testing.assert_array_equal(result, self.img[:, ::-1])

    def test_flip_vertical(self):
        result = (self.mm.load(self.img).flip("vertical").apply())
        np.testing.assert_array_equal(result, self.img[::-1, :])

    def test_rotate_90(self):
        result = (self.mm.load(self.img).rotate(90).apply())
        np.testing.assert_array_equal(result, np.rot90(self.img, k=1))

    def test_rotate_180(self):
        result = (self.mm.load(self.img).rotate(180).apply())
        np.testing.assert_array_equal(result, np.rot90(self.img, k=2))

    def test_crop(self):
        result = (self.mm.load(self.img).crop(10, 10, 20, 20).apply())
        self.assertEqual(result.shape, (20, 20, 3))

    def test_blur(self):
        result = (self.mm.load(self.img).blur(kernel_size=3).apply())
        self.assertEqual(result.shape, self.img.shape)

    def test_sharpen(self):
        result = (self.mm.load(self.img).sharpen().apply())
        self.assertEqual(result.shape, self.img.shape)

    def test_brightness(self):
        result = (self.mm.load(self.img).brightness(1.5).apply())
        self.assertEqual(result.shape, self.img.shape)

    def test_contrast(self):
        result = (self.mm.load(self.img).contrast(1.5).apply())
        self.assertEqual(result.shape, self.img.shape)

    def test_chained_ops(self):
        result = (self.mm.load(self.img)
                  .resize(30, 30)
                  .grayscale()
                  .flip("horizontal")
                  .apply())
        self.assertEqual(result.shape, (30, 30))

    def test_batch_apply(self):
        images = [np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8) for _ in range(3)]
        results = self.mm.batch_apply(images, ops=["resize:25,25", "grayscale"])
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.shape, (25, 25))

    def test_audit(self):
        self.mm.load(self.img).resize(25, 25).grayscale().apply()
        audit = self.mm.get_audit()
        self.assertGreater(len(audit), 0)

    def test_no_image_raises(self):
        with self.assertRaises(RuntimeError):
            self.mm.apply()

    def test_strip_metadata(self):
        result = (self.mm.load(self.img).strip_metadata().apply())
        np.testing.assert_array_equal(result, self.img)  # no-op for numpy


class TestIntegration(unittest.TestCase):
    def test_all_accessible_from_top_level(self):
        self.assertTrue(hasattr(qai, "MultiRoleSelfPlay"))
        self.assertTrue(hasattr(qai, "ComposableSkills"))
        self.assertTrue(hasattr(qai, "CascadeInference"))
        self.assertTrue(hasattr(qai, "BehaviorContract"))
        self.assertTrue(hasattr(qai, "MediaManipulator"))
        self.assertTrue(hasattr(qai, "production"))

    def test_skill_dataclass(self):
        s = qai.Skill(name="test", fn=lambda: "ok")
        self.assertEqual(s.execute(), "ok")

    def test_behavior_rule_dataclass(self):
        r = qai.BehaviorRule(name="test", description="test", check=lambda x: True)
        self.assertTrue(r.check("anything"))

    def test_role_dataclass(self):
        r = qai.Role(name="test", model=lambda: "ok")
        self.assertEqual(r.name, "test")

    def test_cascade_with_contract(self):
        """Cascade + BehaviorContract integration."""
        cascade = qai.CascadeInference(escalate_if="confidence < 0.5")
        cascade.add_model("m1", model=lambda x: "hello", confidence_fn=lambda x: 0.9, cost=1)
        contract = qai.BehaviorContract()
        contract.add_rule(qai.BehaviorRule(
            name="not_empty", description="non-empty",
            check=lambda x: len(str(x)) > 0, severity="error"))
        result = cascade.predict("test")
        check = contract.check(result["prediction"])
        self.assertTrue(check["passed"])


if __name__ == "__main__":
    unittest.main()
