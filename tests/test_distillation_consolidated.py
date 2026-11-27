"""Consolidated distillation tests — one feature, one code path.

Since v1.8.5, qai has exactly ONE distillation implementation
(``qai.distillation_v2``). Everything else is a backward-compatibility
delegate:

    qai.advanced.KnowledgeDistiller           → delegate
    qai.compression.Distiller (static)        → delegate
    qai.creative.experimental.KnowledgeDistiller2 → delegate
    qai.creative_tools.ModelDistiller         → delegate

The user-facing entry point is ``qai.distill(teacher, student, X, y, ...)``.

These tests run WITHOUT torch (the numpy fallback path), plus source-level
checks that the delegates contain no algorithm code of their own.

Run:
    python3 -m pytest tests/test_distillation_consolidated.py -v
"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import qai
from qai.distillation_v2 import (
    Distiller, MultiTeacherDistiller, SelfDistiller,
    DistillationConfig, DistillationResult,
    distill, multi_distill, self_distill,
    get_soft_targets, kl_divergence,
)

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
              [2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1])


def _trained(model_cls_name: str, **kw):
    m = qai.build(technique=model_cls_name, **kw)
    m.train(X, y, verbose=False)
    return m


class TestOneEntryPoint(unittest.TestCase):
    """The preferred API: qai.distill(...) with parameters."""

    def test_top_level_distill_exists(self):
        self.assertTrue(callable(qai.distill))
        self.assertTrue(callable(qai.multi_distill))
        self.assertTrue(callable(qai.self_distill))

    def test_distill_returns_real_result(self):
        teacher = _trained("random_forest")
        student = qai.build(technique="decision_tree")
        result = qai.distill(teacher, student, X, y,
                             strategy="vanilla", temperature=4.0, alpha=0.7,
                             epochs=3)
        self.assertIsInstance(result, DistillationResult)
        self.assertNotEqual(result.distillation_mode, "not-run",
                            "distill() ran but didn't record a mode")
        self.assertTrue(
            result.distillation_mode.startswith(
                ("hard-label fallback", "confidence-weighted fallback",
                 "soft-label KL (torch)", "soft-label KL + alignment")),
            f"unexpected mode: {result.distillation_mode!r}")
        # KL numbers must be real (not NaN)
        self.assertFalse(np.isnan(result.kl_before))
        self.assertFalse(np.isnan(result.kl_after))
        # Note: KL(teacher‖student) is only guaranteed to DROP on the true
        # soft-label torch path. On the hard-label fallback the untrained
        # student can sit near-uniform (tiny KL, useless) and a genuinely
        # trained student may diverge more from the teacher's soft
        # distribution while being more accurate — so we only assert the
        # numbers are real here, plus the student got trained.
        self.assertTrue(student.technique.is_trained())

    def test_config_parameters_flow(self):
        teacher = _trained("random_forest")
        student = qai.build(technique="decision_tree")
        result = qai.distill(teacher, student, X, y,
                             temperature=7.5, alpha=0.9, epochs=2)
        self.assertNotEqual(result.distillation_mode, "not-run")


class TestDelegates(unittest.TestCase):
    """Old entry points still work — and all delegate to the canonical code."""

    def test_advanced_knowledge_distiller_api(self):
        """Old signature preserved: (teacher, student, T, alpha).distill(...)"""
        teacher = _trained("random_forest")
        student = qai.build(technique="decision_tree")
        d = qai.KnowledgeDistiller(teacher=teacher, student=student,
                                   temperature=4.0, alpha=0.5)
        out = d.distill(X, y, epochs=3, lr=1e-3, batch_size=8, verbose=False)
        self.assertIs(out, student, "old API returns the trained student")
        self.assertTrue(student.technique.is_trained())
        self.assertIsInstance(d.losses(), list)
        self.assertIsInstance(d.compression_ratio(), float)
        self.assertIsNotNone(d.last_result)
        self.assertNotEqual(d.last_result.distillation_mode, "not-run")

    def test_compression_distiller_static_api(self):
        """qai.compression.Distiller.distill(...) static method delegates."""
        from qai.compression import Distiller as CompressionDistiller
        teacher = _trained("random_forest")
        student = qai.build(technique="decision_tree")
        out = CompressionDistiller.distill(teacher, student, X, y,
                                           temperature=4.0, alpha=0.5,
                                           n_steps=3)
        self.assertIsInstance(out, dict)
        self.assertIn("history", out)
        self.assertIn("final_loss", out)
        self.assertIn("mode", out)
        self.assertTrue(student.technique.is_trained())

    def test_knowledge_distiller2_delegates(self):
        from qai.creative.experimental import KnowledgeDistiller2
        teacher = _trained("random_forest")
        student = qai.build(technique="decision_tree")
        d = KnowledgeDistiller2(teacher=teacher, student=student,
                                alpha=0.5, beta=0.3, gamma=0.2)
        out = d.distill(X, y, n_steps=3)
        self.assertIsInstance(out, dict)
        self.assertIn("final_loss", out)
        self.assertIn("mode", out)
        self.assertTrue(student.technique.is_trained())

    def test_model_distiller_delegates(self):
        from qai.creative_tools import ModelDistiller
        teacher = _trained("random_forest")
        student = qai.build(technique="decision_tree")
        d = ModelDistiller(teacher=teacher, student=student,
                           alpha=0.5, temperature=4.0)
        out = d.distill(X, y, epochs=2, verbose=False)
        self.assertIs(out, student)
        self.assertTrue(student.technique.is_trained())
        self.assertNotEqual(d.last_result.distillation_mode, "not-run")

    def test_delegates_contain_no_algorithm_code(self):
        """Source check: no delegate file re-implements the KL loss."""
        forbidden = ("F.kl_div", "kl_div(")
        delegate_files = [
            "qai/advanced.py",
            "qai/compression.py",
            "qai/creative/experimental.py",
            "qai/creative_tools.py",
        ]
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        for rel in delegate_files:
            path = os.path.join(repo_root, rel)
            with open(path) as f:
                src = f.read()
            for frag in forbidden:
                self.assertNotIn(
                    frag, src,
                    f"{rel} still contains its own distillation loss "
                    f"({frag!r}) — delegates must not re-implement the "
                    f"algorithm; port it to qai/distillation_v2.py"
                )


class TestStrategiesAndVariants(unittest.TestCase):
    """strategy=, multi-teacher, and self-distillation all work."""

    def test_hint_strategy_flagged_without_torch(self):
        teacher = _trained("random_forest")
        student = qai.build(technique="decision_tree")
        result = qai.distill(teacher, student, X, y,
                             strategy="hint", align_weight=0.3, epochs=2)
        self.assertNotEqual(result.distillation_mode, "not-run")

    def test_multi_teacher(self):
        t1 = _trained("random_forest")
        t2 = _trained("decision_tree")
        student = qai.build(technique="decision_tree")
        result = qai.multi_distill([t1, t2], student, X, y, epochs=2)
        self.assertEqual(result.strategy, "multi")
        self.assertIn("probability-averaged", result.distillation_mode)
        self.assertTrue(student.technique.is_trained())

    def test_multi_teacher_weights_normalised(self):
        d = MultiTeacherDistiller(
            [_trained("random_forest"), _trained("decision_tree")],
            qai.build(technique="decision_tree"),
            teacher_weights=[2.0, 2.0],
        )
        self.assertAlmostEqual(sum(d.teacher_weights), 1.0)

    def test_self_distillation(self):
        model = _trained("random_forest")
        result = qai.self_distill(model, X, y, epochs=2)
        self.assertEqual(result.strategy, "self")
        self.assertNotEqual(result.distillation_mode, "not-run")


class TestHelpers(unittest.TestCase):
    """Soft-target extraction and KL metric behave sensibly."""

    def test_get_soft_targets_returns_proba(self):
        teacher = _trained("random_forest")
        soft = get_soft_targets(teacher, X)
        self.assertEqual(soft.ndim, 2)
        self.assertEqual(soft.shape[0], len(X))
        np.testing.assert_allclose(soft.sum(axis=1), 1.0, atol=1e-6)

    def test_kl_divergence_identical_is_zero(self):
        p = np.array([[0.7, 0.3], [0.4, 0.6]])
        self.assertAlmostEqual(kl_divergence(p, p), 0.0, places=6)

    def test_kl_divergence_direction(self):
        p = np.array([[0.98, 0.02]])  # confident
        q = np.array([[0.5, 0.5]])    # uniform
        # KL(p‖q) > 0 — confident vs uniform diverges
        self.assertGreater(kl_divergence(p, q), 0.5)


if __name__ == "__main__":
    unittest.main()
