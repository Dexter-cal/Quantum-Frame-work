"""Security regression test for qai.bundle.

Verifies the fixes for the v1.8.3 reviewer findings:

  1. Tampering with logic.py now breaks the signature (previously the
     manifest only hashed model.json + weights, so an attacker could
     replace logic.py with arbitrary code that passed verify_bundle and
     then ran at load time).
  2. Tampered logic.py also raises at load_bundle time (defense in depth)
     even if the caller never called verify_bundle.
  3. Module-level constants referenced by a bundled logic function are
     captured at save time and re-injected at load time, so the function
     doesn't raise NameError in a fresh process.

Run:
    python3 -m pytest tests/test_bundle_security_regression.py -v
"""
import os
import sys
import json
import tempfile
import zipfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import qai


def _make_simple_bundle(path: str, sign_key: str, logic_fn=None) -> None:
    """Build a small signed bundle with optional logic function."""
    m = qai.build(technique="classifier")
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 0])
    m.train(X, y, verbose=False)
    logic = [logic_fn] if logic_fn is not None else None
    qai.save_bundle(m, path, logic_fns=logic, sign_key=sign_key)


class TestBundleLogicSigning(unittest.TestCase):
    """Item 1: logic.py must be covered by the bundle signature."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "m.qmodel")
        self.key = "secret-key-12345"

        @qai.logic
        def decision(x):
            return 1 if x[0] > 2.5 else 0
        self.logic_fn = decision
        _make_simple_bundle(self.path, self.key, decision)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_untampered_bundle_verifies(self):
        self.assertTrue(qai.verify_bundle(self.path, sign_key=self.key))

    def test_tampered_logic_breaks_signature(self):
        """Replacing logic.py with attacker code must invalidate the signature."""
        # Read the bundle, replace logic.py with evil code, write back
        tampered_path = os.path.join(self.tmpdir, "tampered.qmodel")
        evil_code = (
            "def evil():\n"
            "    import os\n"
            "    os.system('echo pwned')\n"
            "evil()\n"  # runs at exec time
        )
        with zipfile.ZipFile(self.path, "r") as zin:
            with zipfile.ZipFile(tampered_path, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in zin.namelist():
                    data = zin.read(item)
                    if item == "logic.py":
                        data = evil_code.encode("utf-8")
                    zout.writestr(item, data)

        # The signature MUST fail
        self.assertFalse(
            qai.verify_bundle(tampered_path, sign_key=self.key),
            "Tampered logic.py passed verify_bundle! This is the v1.8.3 "
            "security regression — logic.py must be covered by the signature.",
        )

    def test_tampered_logic_rejected_at_load(self):
        """load_bundle should refuse to exec tampered logic.

        Either path is acceptable:
          - signature verification fails first (the normal path — load_bundle
            calls verify_bundle and aborts before exec), OR
          - defense-in-depth hash check fires (when verify_signature=False
            is explicitly passed to bypass the signature check).
        Both prove the tampering is detected.
        """
        tampered_path = os.path.join(self.tmpdir, "tampered_load.qmodel")
        evil_code = (
            "def evil():\n"
            "    pass\n"
            "evil()\n"
        )
        with zipfile.ZipFile(self.path, "r") as zin:
            with zipfile.ZipFile(tampered_path, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in zin.namelist():
                    data = zin.read(item)
                    if item == "logic.py":
                        data = evil_code.encode("utf-8")
                    zout.writestr(item, data)

        # Path 1: normal load_bundle (signature check fires first)
        with self.assertRaises(ValueError) as ctx:
            qai.load_bundle(tampered_path, sign_key=self.key)
        msg = str(ctx.exception).lower()
        self.assertTrue(
            "tamper" in msg or "signature" in msg,
            f"Expected tamper/signature error, got: {ctx.exception}",
        )

        # Path 2: defense-in-depth — even with signature verification
        # explicitly disabled, the per-file hash check should catch tampering.
        with self.assertRaises(ValueError) as ctx:
            qai.load_bundle(
                tampered_path, sign_key=self.key, verify_signature=False
            )
        msg = str(ctx.exception)
        self.assertIn("logic.py", msg)
        self.assertIn("tamper", msg.lower())


# Module-level constant that the bundled logic function will reference.
# At bundle save time, getclosurevars() will discover it as a global dep,
# and the new constant-capture logic will serialise its value (100) into
# logic_meta.json. At load time it gets re-injected into the exec namespace.
_BUNDLED_TEST_LIMIT = 100


class TestBundleConstantsTravel(unittest.TestCase):
    """Item 3: module-level constants referenced by a logic function must
    travel with the bundle and be re-injected at load time."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "const.qmodel")
        self.key = "const-key"

        # Define a logic function that references the module-level constant.
        # The function's __globals__ is this test module's namespace, so
        # getclosurevars() will find `_BUNDLED_TEST_LIMIT` as a global dep.
        @qai.logic
        def bounded(x):
            return min(x, _BUNDLED_TEST_LIMIT)

        _make_simple_bundle(self.path, self.key, bounded)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_constant_is_captured_in_bundle(self):
        """logic_meta.json should contain the captured constant value."""
        with zipfile.ZipFile(self.path, "r") as zf:
            meta = json.loads(zf.read("logic_meta.json"))
        self.assertEqual(len(meta), 1)
        constants = meta[0].get("constants", {})
        self.assertIn(
            "_BUNDLED_TEST_LIMIT", constants,
            "Module-level constant was not captured into logic_meta.json. "
            "Item 3 regression: constants don't travel with bundled logic.",
        )
        self.assertEqual(constants["_BUNDLED_TEST_LIMIT"], 100)

    def test_constant_reinject_at_load(self):
        """The bundled function should run without NameError and use the
        captured constant value."""
        bundle = qai.load_bundle(self.path, sign_key=self.key)
        self.assertTrue(len(bundle.logic_fns) > 0, "no logic fns loaded")
        fn = next(iter(bundle.logic_fns.values()))
        # Below the limit — passes through
        self.assertEqual(fn(50), 50)
        # Above the limit — clamped to the captured constant
        self.assertEqual(fn(150), 100)


if __name__ == "__main__":
    unittest.main()
