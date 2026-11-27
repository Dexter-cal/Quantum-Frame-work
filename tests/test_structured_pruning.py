"""Tests for qai.pruning — real structured pruning.

Since v1.8.6, structured pruning is consolidated in ``qai.pruning`` with
``qai.structured_prune(...)`` as the one entry point. The three old entry
points (``qai.governance.PruningPipeline.structured_prune``,
``qai.compression.Pruner.structured_prune``, ``qai.surgery.ChannelPruner``)
are delegates.

These tests verify:
  1. Real shape change — layers actually shrink, not just zeroed weights.
  2. Downstream propagation — the next layer's input dim matches.
  3. Both cached refs and Sequential entries updated (the PyTorchMLP bug).
  4. Params count goes DOWN (not up, which was the first attempt's bug).
  5. predict() still works after pruning.
  6. Accuracy delta is reported when X_val/y_val are given.
  7. The finetune wrapper recovers some accuracy.
  8. Source-level: no delegate returns ``"not_implemented"``.

Requires torch (skipped if unavailable).

Run:
    python3 -m pytest tests/test_structured_pruning.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings; warnings.filterwarnings("ignore")

import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

import qai


# Shared fixture
def _trained_mlp(hidden=32, epochs=15):
    """Build + train a small MLP classifier."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    m = qai.build(technique="neural_network", hidden_dim=hidden, epochs=epochs)
    m.train(X, y, verbose=False)
    return m, X, y


@unittest.skipUnless(_TORCH, "requires PyTorch")
class TestStructuredPruneReal(unittest.TestCase):
    """The canonical implementation actually removes neurons."""

    def test_params_count_goes_down(self):
        """Pruning must reduce the parameter count, not increase it."""
        m, X, y = _trained_mlp()
        before = sum(p.numel() for p in m.technique.net.parameters())
        result = qai.structured_prune(m, layer_name="fc1", fraction=0.5)
        after = sum(p.numel() for p in m.technique.net.parameters())
        self.assertLess(
            after, before,
            f"params went UP after pruning: {before} → {after}. "
            "This is the 'zombie layer' bug — the cached ref and the "
            "Sequential entry must both be updated.",
        )
        self.assertLess(result["params_after"], result["params_before"])
        self.assertGreater(result["compression_ratio"], 1.0)

    def test_layer_shape_actually_shrinks(self):
        """The pruned layer's out_features must decrease."""
        m, X, y = _trained_mlp(hidden=32)
        before_shape = tuple(m.technique.net.fc1.weight.shape)
        self.assertEqual(before_shape, (32, 6))
        qai.structured_prune(m, layer_name="fc1", fraction=0.5)
        after_shape = tuple(m.technique.net.fc1.weight.shape)
        self.assertEqual(after_shape[0], 16, f"fc1 out_features didn't halve: {after_shape}")
        self.assertEqual(after_shape[1], 6, "fc1 in_features must not change")

    def test_downstream_layer_input_sliced(self):
        """The next layer's input dim must match the pruned layer's new output."""
        m, X, y = _trained_mlp(hidden=32)
        # Before: fc2.weight is (2, 32) — 32 inputs
        self.assertEqual(tuple(m.technique.net.fc2.weight.shape), (2, 32))
        qai.structured_prune(m, layer_name="fc1", fraction=0.5)
        # After: fc2.weight must be (2, 16) — 16 inputs
        self.assertEqual(
            tuple(m.technique.net.fc2.weight.shape), (2, 16),
            "downstream fc2 input dim wasn't sliced to match pruned fc1 output",
        )

    def test_cached_ref_and_sequential_both_updated(self):
        """PyTorchMLP stores fc1 both as self.fc1 AND in self.net = Sequential.
        Both must point at the new (smaller) layer after pruning."""
        m, X, y = _trained_mlp(hidden=32)
        qai.structured_prune(m, layer_name="fc1", fraction=0.5)
        fc1_attr = m.technique.net.fc1
        fc1_seq = m.technique.net.net[0]
        self.assertIs(
            fc1_attr, fc1_seq,
            "cached ref (self.fc1) and Sequential entry (self.net[0]) "
            "point at different layers after pruning",
        )
        self.assertEqual(tuple(fc1_attr.weight.shape), (16, 6))

    def test_predict_still_works_after_pruning(self):
        """The model must still produce predictions after pruning."""
        m, X, y = _trained_mlp(hidden=32)
        qai.structured_prune(m, layer_name="fc1", fraction=0.3)
        pred = m.predict(X[0])
        self.assertIn(pred, [0, 1], f"predict returned {pred!r}")

    def test_accuracy_delta_reported(self):
        """When X_val/y_val are given, accuracy before/after is reported."""
        m, X, y = _trained_mlp(hidden=32)
        result = qai.structured_prune(m, layer_name="fc1", fraction=0.4,
                                       X_val=X, y_val=y)
        self.assertIsNotNone(result["accuracy_before"])
        self.assertIsNotNone(result["accuracy_after"])
        self.assertIsNotNone(result["accuracy_delta"])
        # Pruning without finetuning usually drops accuracy a little
        self.assertGreaterEqual(result["accuracy_before"], 0.5)

    def test_prune_all_layers(self):
        """layer_name=None prunes every layer except the output head."""
        m, X, y = _trained_mlp(hidden=32)
        result = qai.structured_prune(m, fraction=0.5)
        self.assertEqual(result["n_layers_pruned"], 1,
                         "should prune fc1 only (fc2 is the output head, kept)")
        # fc2 output dim unchanged
        self.assertEqual(tuple(m.technique.net.fc2.weight.shape), (2, 16))

    def test_finetune_wrapper(self):
        """structured_prune_with_finetune recovers accuracy."""
        m, X, y = _trained_mlp(hidden=32)
        result = qai.structured_prune_with_finetune(
            m, X, y, layer_name="fc1", fraction=0.3, finetune_epochs=5,
        )
        self.assertIn("accuracy_after_finetune", result)
        self.assertGreater(result["accuracy_after_finetune"], 0.5)

    def test_criterion_l1(self):
        """L1 criterion works (ranks by abs-sum instead of L2)."""
        m, X, y = _trained_mlp(hidden=32)
        result = qai.structured_prune(m, layer_name="fc1", fraction=0.5,
                                       criterion="l1")
        self.assertNotIn("error", result)
        self.assertEqual(result["criterion"], "l1")

    def test_unknown_layer_error(self):
        """A clear error is returned for an unknown layer name."""
        m, X, y = _trained_mlp(hidden=32)
        result = qai.structured_prune(m, layer_name="nonexistent", fraction=0.5)
        self.assertIn("error", result)
        self.assertIn("not found", result["error"])

    def test_unknown_criterion_error(self):
        """A clear error is returned for an unknown criterion."""
        m, X, y = _trained_mlp(hidden=32)
        result = qai.structured_prune(m, layer_name="fc1", fraction=0.5,
                                       criterion="bogus")
        self.assertIn("error", result)


@unittest.skipUnless(_TORCH, "requires PyTorch")
class TestDelegates(unittest.TestCase):
    """Old entry points delegate to the canonical implementation."""

    def test_governance_pruning_pipeline(self):
        """qai.governance.PruningPipeline.structured_prune no longer returns
        {'status': 'not_implemented'} — it actually prunes."""
        from qai.governance.extended import PruningPipeline
        m, X, y = _trained_mlp(hidden=32)
        pp = PruningPipeline(m)
        result = pp.structured_prune(layer_name="fc1", fraction=0.5)
        self.assertNotIn("status", result,
                         "PruningPipeline.structured_prune still returns the "
                         "old 'not_implemented' stub")
        self.assertIn("layers_pruned", result)
        self.assertLess(result["params_after"], result["params_before"])

    def test_compression_pruner(self):
        """qai.compression.Pruner.structured_prune delegates and prunes."""
        from qai.compression import Pruner
        m, X, y = _trained_mlp(hidden=32)
        result = Pruner.structured_prune(m, layer_name="fc1", fraction=0.5)
        self.assertNotIn("error", result)
        self.assertLess(result["params_after"], result["params_before"])

    def test_surgery_channel_pruner_remove(self):
        """qai.surgery.ChannelPruner.remove does real structured pruning."""
        from qai.surgery import ChannelPruner
        m, X, y = _trained_mlp(hidden=32)
        result = ChannelPruner.remove(m, layer="fc1", fraction=0.5)
        self.assertNotIn("error", result)
        self.assertLess(result["params_after"], result["params_before"])

    def test_surgery_channel_pruner_prune_still_zeros(self):
        """The historical ChannelPruner.prune still zero-masks (backward compat)
        but now honestly says so in its return value.

        Uses the raw torch net (not the qai Model wrapper) and a weight-key
        name (e.g. 'fc1.weight') because _extract_weights returns parameter
        names, not layer names."""
        from qai.surgery import ChannelPruner
        m, X, y = _trained_mlp(hidden=32)
        raw_net = m.technique.net
        # _extract_weights returns keys like 'fc1.weight', not 'fc1'
        result = ChannelPruner.prune(raw_net, layer="fc1.weight", fraction=0.5)
        self.assertEqual(result["method"], "zero_mask")
        self.assertIn("note", result)
        self.assertIn("zeroed", result["note"])


class TestSourceLevelChecks(unittest.TestCase):
    """Source-level guarantees that don't need torch."""

    def test_no_not_implemented_in_structured_prune_delegates(self):
        """No delegate may return {'status': 'not_implemented'} for structured
        pruning — that was the v1.8.3 bug.

        Checks for actual return statements, not docstring mentions of the
        old behavior (which are expected and honest)."""
        import ast
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        delegate_files = [
            "qai/governance/extended.py",
            "qai/compression.py",
        ]
        for rel in delegate_files:
            path = os.path.join(repo_root, rel)
            with open(path) as f:
                tree = ast.parse(f.read())
            # Walk every function/method body; flag any return statement
            # whose value is a dict literal containing "not_implemented"
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # Only check methods named structured_prune
                if node.name != "structured_prune":
                    continue
                for child in ast.walk(node):
                    if not isinstance(child, ast.Return):
                        continue
                    val = child.value
                    if val is None:
                        continue
                    # Check for {"status": "not_implemented"} pattern
                    if isinstance(val, ast.Dict):
                        for k, v in zip(val.keys, val.values):
                            if (isinstance(k, ast.Constant) and k.value == "status"
                                    and isinstance(v, ast.Constant)
                                    and v.value == "not_implemented"):
                                self.fail(
                                    f"{rel}:{node.name}() still returns "
                                    "{{'status': 'not_implemented'}} — "
                                    "the stub hasn't been replaced"
                                )

    def test_canonical_module_exists(self):
        """qai/pruning.py exists and exports structured_prune."""
        path = os.path.join(os.path.dirname(__file__), "..", "qai", "pruning.py")
        self.assertTrue(os.path.exists(path), "qai/pruning.py doesn't exist")
        with open(path) as f:
            src = f.read()
        self.assertIn("def structured_prune(", src)
        self.assertIn("def structured_prune_with_finetune(", src)


if __name__ == "__main__":
    unittest.main()
