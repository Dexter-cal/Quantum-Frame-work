"""
qai.bundle -- The .qmodel bundle format (spec Section 68/76).

A `.qmodel` file is a ZIP archive bundling everything needed to deploy a
model: weights, technique metadata, optional @qai.logic function source,
and optional tokenizer. This is the spec's answer to "logic ships bundled
with the model" — and the foundation for dynamic region loading (Section 76).

Bundle structure (ZIP archive):

    my_model.qmodel
    ├── manifest.json      # metadata: version, technique, hashes, signed_hash
    ├── model.json         # technique name, params, hyperparameters, state dict
    ├── weights.npy        # numpy weights (more efficient than JSON for big arrays)
    ├── logic.py           # optional @qai.logic function source
    ├── logic_meta.json    # optional: closure deps, logic_name, max_seconds
    ├── tokenizer.json     # optional: attached tokenizer state
    └── README.md          # auto-generated human-readable description

API:

    # Save
    qai.save_bundle(
        model,
        "my_model.qmodel",
        logic_fns=[my_decision_fn, my_generate_fn],  # optional
        tokenizer=tok,                                # optional
        sign_key="my_secret_key",                     # optional HMAC
        description="A Python code completion model", # optional
    )

    # Inspect (no loading)
    info = qai.inspect_bundle("my_model.qmodel")
    print(info["manifest"]["technique"], info["contents"])

    # Verify signature
    valid = qai.verify_bundle("my_model.qmodel", sign_key="my_secret_key")

    # Load
    bundle = qai.load_bundle("my_model.qmodel", sign_key="my_secret_key")
    print(bundle.model)              # reconstructed qai.Model
    print(bundle.tokenizer)          # attached tokenizer (or None)
    print(bundle.logic_fns)          # dict of {name: callable}
    result = bundle.model.predict([5.0])

    # Diff two bundles
    diff = qai.diff_bundles("v1.qmodel", "v2.qmodel")
    print(diff["weights_changed"], diff["logic_changed"])

Signed bundles use HMAC-SHA256 over the manifest + contents. Tampering with
any file invalidates the signature, so consumers can verify model provenance
before loading untrusted .qmodel files.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# =====================================================================
# Constants
# =====================================================================
BUNDLE_FORMAT_VERSION = "1.0"
SUPPORTED_FORMAT_VERSIONS = {"1.0"}  # versions we can load
MANIFEST_FILENAME = "manifest.json"
MODEL_FILENAME = "model.json"
WEIGHTS_FILENAME = "weights.npy"
LOGIC_FILENAME = "logic.py"
LOGIC_META_FILENAME = "logic_meta.json"
TOKENIZER_FILENAME = "tokenizer.json"
README_FILENAME = "README.md"


# =====================================================================
# Format-version migration registry
# =====================================================================
_BUNDLE_MIGRATIONS: Dict[str, Callable[[Dict], Dict]] = {}


def register_bundle_migration(from_version: str, to_version: str,
                               migrator: Callable[[Dict], Dict]):
    """Register a migration function from one bundle format version to another.

    Called by `load_bundle` when the bundle's format_version isn't in
    SUPPORTED_FORMAT_VERSIONS but a chain of migrations can reach one.
    """
    _BUNDLE_MIGRATIONS[(from_version, to_version)] = migrator


def _migrate_manifest(manifest: Dict) -> Dict:
    """Walk the migration registry to bring `manifest` up to a supported version."""
    current = manifest.get("format_version", "1.0")
    if current in SUPPORTED_FORMAT_VERSIONS:
        return manifest
    # Try a one-hop migration (extensible to multi-hop later)
    for (from_v, to_v), migrator in _BUNDLE_MIGRATIONS.items():
        if from_v == current and to_v in SUPPORTED_FORMAT_VERSIONS:
            return migrator(manifest)
    raise ValueError(
        f"[WHAT] Bundle format_version {current!r} is not supported\n"
        f"[WHY] This qai supports {sorted(SUPPORTED_FORMAT_VERSIONS)} and "
        f"no migration path is registered from {current!r}\n"
        f"[FIX] Upgrade qai, or register a migration via "
        f"qai.bundle.register_bundle_migration({current!r}, '1.0', migrator_fn)"
    )


# =====================================================================
# Bundle class — holds a loaded bundle's contents
# =====================================================================
@dataclass
class Bundle:
    """A loaded .qmodel bundle.

    Attributes:
        model: The reconstructed qai.Model (trained, ready to predict).
        tokenizer: The attached tokenizer (or None).
        logic_fns: Dict of {name: callable} for any bundled @qai.logic functions.
        manifest: The full manifest dict from the bundle.
        path: Path the bundle was loaded from.
        readme: The auto-generated README text (or None if not present).
    """
    model: Any
    tokenizer: Optional[Any] = None
    logic_fns: Dict[str, Callable] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None
    readme: Optional[str] = None

    def __repr__(self):
        return (
            f"<Bundle model={getattr(self.model, 'technique_name', '?')!r} "
            f"tokenizer={type(self.tokenizer).__name__ if self.tokenizer else 'None'} "
            f"logic_fns={list(self.logic_fns.keys())} "
            f"signed={self.manifest.get('signed', False)}>"
        )

    def print_readme(self) -> None:
        """Print the bundle's README to stdout (if present)."""
        if self.readme:
            print(self.readme)
        else:
            print("(no README in this bundle)")


# =====================================================================
# Internal helpers — weight extraction/restoration
# =====================================================================
def _extract_weights(model) -> Tuple[Dict[str, np.ndarray], str]:
    """Extract weights from a qai Model. Returns (dict, format).

    Preserves original dtype (float32 / float64 / int64) for round-trip fidelity.

    Format is one of:
        - "qai_w_b"    : technique has `w`/`b` attributes (Regression)
        - "sklearn"    : technique has sklearn `coef_`/`intercept_`
        - "pytorch_sd" : technique has `.net.state_dict()` (Transformer, NeuralNetwork)
        - "composite"  : object has `_qai_bundle_state()` method (any composite agent)
        - "pickle_fallback" : nothing else matched -- pickle the whole technique
    """
    technique = getattr(model, "technique", None)

    # Path 0 (NEW): composite-agent support.
    # Any object (qai.Model OR a custom agent) that exposes _qai_bundle_state()
    # gets full control over what to serialize. The returned dict can include
    # arbitrary picklable fields; we store them as a single pickled array.
    if hasattr(model, "_qai_bundle_state") and callable(getattr(model, "_qai_bundle_state")):
        try:
            import pickle as _pickle
            state_dict = model._qai_bundle_state()
            pickled = _pickle.dumps(state_dict)
            return {"__composite_state__": np.frombuffer(pickled, dtype=np.uint8)}, "composite"
        except Exception as e:
            # Fall through to other paths
            pass

    if technique is None:
        raise ValueError(
            "[WHAT] Cannot extract weights from this object\n"
            "[WHY] It doesn't have a `technique` attribute (not a qai Model?)\n"
            "[FIX] Pass a qai Model from qai.build(...), or implement "
            "_qai_bundle_state() on your custom agent class"
        )

    # Path 1: qai classical with w/b (Regression)
    if hasattr(technique, "w") and technique.w is not None:
        # Preserve original dtype (Regression uses float64)
        w_dtype = np.asarray(technique.w).dtype
        b_arr = np.asarray(technique.b).flatten()
        return {
            "w": np.asarray(technique.w, dtype=w_dtype),
            "b": b_arr.astype(w_dtype if np.issubdtype(b_arr.dtype, np.floating) else b_dtype),
        }, "qai_w_b"

    # Path 2: sklearn-backed with coef_/intercept_
    if hasattr(technique, "model") and hasattr(technique.model, "coef_"):
        coef_dtype = np.asarray(technique.model.coef_).dtype
        weights = {"coef_": np.asarray(technique.model.coef_, dtype=coef_dtype)}
        if hasattr(technique.model, "intercept_"):
            int_arr = np.asarray(technique.model.intercept_).flatten()
            weights["intercept_"] = int_arr.astype(coef_dtype if np.issubdtype(int_arr.dtype, np.floating) else int_arr.dtype)
        if hasattr(technique.model, "classes_"):
            weights["classes_"] = np.asarray(technique.model.classes_)
        return weights, "sklearn"

    # Path 3: PyTorch state_dict
    if hasattr(technique, "net") and technique.net is not None:
        try:
            import torch
            sd = technique.net.state_dict()
            return {k: v.detach().cpu().numpy() for k, v in sd.items()}, "pytorch_sd"
        except Exception as e:
            raise ValueError(f"Failed to extract PyTorch state_dict: {e}")

    # Path 4: sklearn with cluster_centers_ (KMeans, MeanShift, etc.)
    if hasattr(technique, "model") and hasattr(technique.model, "cluster_centers_"):
        centers = np.asarray(technique.model.cluster_centers_)
        weights = {"cluster_centers_": centers}
        if hasattr(technique.model, "labels_"):
            weights["labels_"] = np.asarray(technique.model.labels_)
        if hasattr(technique.model, "n_clusters"):
            weights["n_clusters"] = np.array([technique.model.n_clusters])
        return weights, "sklearn_centers"

    # Path 5: pickle fallback — serialize the entire technique object
    # This handles KNN (stores training data), DecisionTree (tree structure),
    # and any other technique without a standard weight format
    import pickle
    pickled = pickle.dumps(technique)
    return {"__pickled_technique__": np.frombuffer(pickled, dtype=np.uint8)}, \
           "pickle_fallback"


def _restore_weights(model, weights: Dict[str, np.ndarray], fmt: str) -> None:
    """Restore weights into a qai Model."""
    # Composite path: invoke _qai_load_bundle_state() if the model supports it
    if fmt == "composite":
        if hasattr(model, "_qai_load_bundle_state") and callable(getattr(model, "_qai_load_bundle_state")):
            import pickle as _pickle
            pickled_bytes = weights["__composite_state__"].tobytes()
            state_dict = _pickle.loads(pickled_bytes)
            model._qai_load_bundle_state(state_dict)
            return
        else:
            raise ValueError(
                "[WHAT] Bundle has composite weight format but model has no "
                "_qai_load_bundle_state() method\n"
                "[WHY] The bundle was saved from a composite agent, but the "
                "loaded model doesn't support restoring composite state\n"
                "[FIX] Implement _qai_load_bundle_state(state_dict) on your "
                "agent class"
            )

    technique = getattr(model, "technique", None)
    if technique is None:
        raise ValueError("Model has no technique attribute")

    if fmt == "qai_w_b":
        if "w" in weights:
            technique.w = weights["w"].copy()
        if "b" in weights:
            b = weights["b"].flatten()
            technique.b = float(b[0]) if b.size == 1 else b.copy()

    elif fmt == "sklearn":
        if "coef_" in weights:
            technique.model.coef_ = weights["coef_"].copy()
        if "intercept_" in weights:
            b = weights["intercept_"].flatten()
            if b.size == 1:
                technique.model.intercept_ = float(b[0])
            else:
                technique.model.intercept_ = b.copy()
        if "classes_" in weights:
            technique.model.classes_ = np.asarray(weights["classes_"])

    elif fmt == "pytorch_sd":
        if not hasattr(technique, "net") or technique.net is None:
            # Need to build the model first
            if hasattr(technique, "_build_model"):
                technique._build_model(vocab_size=technique.vocab_size)
        try:
            import torch
            with torch.no_grad():
                for k, v in weights.items():
                    if k in technique.net.state_dict():
                        technique.net.state_dict()[k].copy_(torch.from_numpy(v))
        except Exception as e:
            raise RuntimeError(f"Failed to restore PyTorch state_dict: {e}")

    elif fmt == "sklearn_centers":
        if hasattr(technique, "model"):
            if "cluster_centers_" in weights:
                technique.model.cluster_centers_ = weights["cluster_centers_"].copy()
            if "labels_" in weights:
                technique.model.labels_ = weights["labels_"].copy()
            if "n_clusters" in weights:
                technique.model.n_clusters = int(weights["n_clusters"][0])

    elif fmt == "pickle_fallback":
        # Deserialize the entire technique from pickle
        import pickle
        if "__pickled_technique__" in weights:
            pickled_bytes = weights["__pickled_technique__"].tobytes()
            restored = pickle.loads(pickled_bytes)
            # copy restored attributes back onto the technique
            for attr in dir(restored):
                if not attr.startswith("_") or attr in ("_trained", "_params"):
                    try:
                        setattr(technique, attr, getattr(restored, attr))
                    except Exception:
                        pass
            technique._trained = True

    else:
        raise ValueError(f"Unknown weight format: {fmt!r}")

    # Mark the technique as trained (skip for composite -- no technique attr)
    if fmt != "composite" and technique is not None:
        technique._trained = True


def _extract_technique_config(model) -> Dict[str, Any]:
    """Extract technique name + hyperparameters needed to reconstruct the model."""
    # Composite path: record the class path so load_bundle can import it
    if hasattr(model, "_qai_bundle_state"):
        cls = type(model)
        config = {
            "technique_name": getattr(cls, "name", cls.__name__.lower()),
            "params": {},
            "extra_params": {},
            "composite_class": f"{cls.__module__}.{cls.__name__}",
        }
        return config

    technique = getattr(model, "technique", None)
    if technique is None:
        raise ValueError("Model has no technique attribute")

    config = {
        "technique_name": technique.name,
        "params": dict(getattr(technique, "params", {})),
    }

    # Add technique-specific params needed for reconstruction
    # (we look at common attribute names that hold hyperparameters)
    extra_params = {}
    for attr in ("vocab_size", "d_model", "n_heads", "n_layers", "max_seq_len",
                 "d_ff", "dropout", "use_rope", "task_type", "n_classes",
                 "epochs", "lr", "batch_size", "weight_decay", "warmup_steps",
                 "hidden_dim", "optimizer",
                 "k", "n_clusters", "n_components", "eps", "min_samples",
                 "alpha", "l1_ratio", "kernel", "n_estimators", "max_iter",
                 "context_len", "covariance_type", "linkage", "perplexity",
                 "contamination", "lags", "horizon", "learning_rate",
                 # CNN-specific
                 "in_channels", "conv_channels", "kernel_sizes", "pool_sizes",
                 "fc_hidden", "augment", "augment_flip_p", "augment_crop_pad",
                 "augment_rotate"):
        if hasattr(technique, attr):
            v = getattr(technique, attr)
            # Skip non-serializable values
            if v is None or isinstance(v, (int, float, str, bool, list, tuple, dict)):
                extra_params[attr] = v
    config["extra_params"] = extra_params
    return config


def _hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def _hash_file_in_zip(zf: zipfile.ZipFile, name: str) -> str:
    """SHA-256 hex digest of a file inside a zip archive."""
    return _hash_bytes(zf.read(name))


def _is_jsonable_constant(v: Any) -> bool:
    """True if `v` is a simple JSON-serialisable constant we can capture
    into a bundle's logic_meta and re-inject at exec time.

    Returns True for: int, float, str, bool, None, and homogeneous lists /
    dicts / tuples of the same (recursive). Returns False for: modules,
    classes, functions, numpy arrays, custom objects, etc. — these would
    need a real closure-capture mechanism (e.g. marshal) and shouldn't be
    silently serialised.
    """
    # Fast path for atomic types
    if isinstance(v, bool) or v is None:
        return True
    if isinstance(v, (int, float, str)):
        return True
    # Recursive containers — try a round-trip through json to be sure
    if isinstance(v, (list, tuple, dict)):
        try:
            json.dumps(v, default=str)
            return True
        except (TypeError, ValueError):
            return False
    return False


# =====================================================================
# Bundle saving
# =====================================================================
def save_bundle(
    model,
    path: str,
    logic_fns: Optional[List[Callable]] = None,
    tokenizer: Optional[Any] = None,
    sign_key: Optional[str] = None,
    description: str = "",
    author: Optional[str] = None,
    include_readme: bool = True,
) -> str:
    """Save a model (with optional logic + tokenizer) to a .qmodel bundle.

    Args:
        model: A trained qai Model (qai.build(...) → model.train(...)).
        path: Output file path. Convention: ends in ".qmodel", but any path works.
        logic_fns: Optional list of @qai.logic-decorated functions to bundle.
            Their source code is saved (via inspect.getsource) so they can be
            reloaded in a separate process.
        tokenizer: Optional tokenizer (qai.CharTokenizer / WordTokenizer /
            BPETokenizer) to bundle alongside the model.
        sign_key: Optional HMAC-SHA256 signing key. If set, the manifest is
            signed and `verify_bundle()` will detect tampering.
        description: Optional human-readable description, stored in the manifest
            and the auto-generated README.
        author: Optional author name, stored in the manifest.
        include_readme: If True (default), auto-generate a README.md inside the
            bundle describing the contents.

    Returns:
        The path argument (for chaining).
    """
    if not path.endswith(".qmodel") and not path.endswith(".zip"):
        # Don't force the extension, just warn in the manifest
        pass

    # 1. Extract weights + technique config
    weights, weight_fmt = _extract_weights(model)
    tech_config = _extract_technique_config(model)

    # 2. Serialize weights to a single .npy file
    # We pack all weight tensors into a dict-of-arrays and save with np.savez
    # (single file, named arrays inside)
    weights_buf = io.BytesIO()
    np.savez(weights_buf, **weights)
    weights_bytes = weights_buf.getvalue()

    # 3. Build the model.json (technique + params + state metadata)
    model_json = {
        "technique": tech_config["technique_name"],
        "params": tech_config["params"],
        "extra_params": tech_config["extra_params"],
        "weight_format": weight_fmt,
        "weight_keys": list(weights.keys()),
        "weight_shapes": {k: list(v.shape) for k, v in weights.items()},
        "trained": True,
    }
    # Include composite_class for composite bundles so load_bundle can import it
    if "composite_class" in tech_config:
        model_json["composite_class"] = tech_config["composite_class"]

    # 4. Logic source code
    logic_files: Dict[str, bytes] = {}
    logic_meta: List[Dict[str, Any]] = []
    if logic_fns:
        import inspect
        import textwrap
        combined_source_lines = []
        for fn in logic_fns:
            # Source: prefer explicit __logic_source__ (set via @qai.logic(source=...))
            # so REPL/Jupyter-defined functions can still be bundled.
            src = getattr(fn, "__logic_source__", None)
            if src is None:
                target = fn.__wrapped__ if hasattr(fn, "__wrapped__") else fn
                try:
                    src = inspect.getsource(target)
                except (OSError, TypeError) as e:
                    raise ValueError(
                        f"[WHAT] Could not get source for logic function {fn.__name__!r}\n"
                        f"[WHY] {e}. This happens for functions defined in REPLs/Jupyter/exec.\n"
                        f"[FIX] Re-decorate with @qai.logic(source='''...your code...''') "
                        f"to ship the source explicitly."
                    ) from e
            # Dedent: leading whitespace from nesting in a class/function
            # would otherwise cause IndentationError on exec
            src = textwrap.dedent(src)

            # Capture closure deps if possible. Also capture the *values* of
            # any names that resolve to simple serialisable constants (numbers,
            # strings, lists, dicts of the same) — these get re-injected into
            # the exec namespace at load time so the bundled function can use
            # module-level constants like `LIMIT = 100` defined in its origin
            # module. Without this, exec() raises NameError in a fresh process
            # because the origin module's globals aren't available.
            closure_deps = []
            captured_constants: Dict[str, Any] = {}
            try:
                target = fn.__wrapped__ if hasattr(fn, "__wrapped__") else fn
                cv = inspect.getclosurevars(target)
                closure_deps = sorted(set(cv.nonlocals.keys()) | set(cv.globals.keys()))
                # Try to capture the current VALUE of each closure dep from
                # the origin module's globals + the function's nonlocals.
                candidate_values = {}
                candidate_values.update(cv.globals)  # name → value (resolved)
                candidate_values.update(cv.nonlocals)
                for name in closure_deps:
                    if name in candidate_values:
                        if _is_jsonable_constant(candidate_values[name]):
                            captured_constants[name] = candidate_values[name]
            except (TypeError, NameError):
                pass

            # Include `requires` from logic_metadata if set
            requires = (getattr(fn, "logic_metadata", {}) or {}).get("requires", [])
            if requires:
                closure_deps = sorted(set(closure_deps) | set(requires))

            meta = {
                "name": getattr(fn, "logic_name", fn.__name__),
                "closure_deps": closure_deps,
                "requires": requires,
                "max_seconds": getattr(fn, "logic_metadata", {}).get("max_seconds"),
                # Captured simple constants (numbers/strings/lists/dicts) so
                # they travel with the bundle and are re-injected at exec time.
                "constants": captured_constants,
            }
            logic_meta.append(meta)
            combined_source_lines.append(src)

        logic_files[LOGIC_FILENAME] = "\n\n".join(combined_source_lines).encode("utf-8")
        logic_files[LOGIC_META_FILENAME] = json.dumps(logic_meta, indent=2).encode("utf-8")

    # 5. Tokenizer state
    tokenizer_bytes: Optional[bytes] = None
    if tokenizer is not None:
        if not (hasattr(tokenizer, "save") and hasattr(tokenizer, "_get_state")):
            raise ValueError(
                "[WHAT] Tokenizer does not look like a qai tokenizer\n"
                "[WHY] It has no .save() method (expected CharTokenizer/WordTokenizer/BPETokenizer)\n"
                "[FIX] Pass a qai tokenizer, or omit tokenizer= entirely"
            )
        # Save to a temp file, read back as bytes
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            tokenizer.save(tmp_path)
            with open(tmp_path, "rb") as f:
                tokenizer_bytes = f.read()
        finally:
            os.unlink(tmp_path)

    # 6. Build manifest
    manifest: Dict[str, Any] = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "technique": tech_config["technique_name"],
        "description": description,
        "author": author,
        "contents": {
            MANIFEST_FILENAME: True,
            MODEL_FILENAME: True,
            WEIGHTS_FILENAME: True,
        },
        "weight_format": weight_fmt,
        "total_params": int(sum(v.size for v in weights.values())),
        "weight_hashes": {k: _hash_bytes(v.tobytes()) for k, v in weights.items()},
        "model_json_hash": None,  # filled below
        "weights_file_hash": None,  # filled below
        "logic_count": len(logic_fns) if logic_fns else 0,
        "has_tokenizer": tokenizer_bytes is not None,
        "signed": sign_key is not None,
        "signed_hash": None,  # filled below
    }
    if logic_files:
        manifest["contents"][LOGIC_FILENAME] = True
        manifest["contents"][LOGIC_META_FILENAME] = True
    if tokenizer_bytes is not None:
        manifest["contents"][TOKENIZER_FILENAME] = True

    # Compute model.json + weights file hashes
    model_json_bytes = json.dumps(model_json, indent=2, default=str).encode("utf-8")
    manifest["model_json_hash"] = _hash_bytes(model_json_bytes)
    manifest["weights_file_hash"] = _hash_bytes(weights_bytes)

    # Compute logic file hashes (CRITICAL: logic.py is exec'd on load, so it
    # MUST be covered by the signature. Earlier versions only hashed model.json
    # + weights + per-weight, which meant tampered logic passed verify_bundle
    # and ran attacker code at load time.)
    if LOGIC_FILENAME in logic_files:
        manifest["logic_hash"] = _hash_bytes(logic_files[LOGIC_FILENAME])
    else:
        manifest["logic_hash"] = None
    if LOGIC_META_FILENAME in logic_files:
        manifest["logic_meta_hash"] = _hash_bytes(logic_files[LOGIC_META_FILENAME])
    else:
        manifest["logic_meta_hash"] = None
    if tokenizer_bytes is not None:
        manifest["tokenizer_hash"] = _hash_bytes(tokenizer_bytes)
    else:
        manifest["tokenizer_hash"] = None

    # 7. Sign the manifest (HMAC-SHA256 over manifest + content hashes).
    # IMPORTANT: every file that gets read back at load time is in the signed
    # payload — model.json, weights, every per-weight tensor, logic.py,
    # logic_meta.json, and tokenizer.json. Adding a new bundled file?
    # Add its hash here too, or it won't be integrity-protected.
    if sign_key is not None:
        sign_payload = (
            manifest["model_json_hash"]
            + manifest["weights_file_hash"]
            + "".join(manifest["weight_hashes"].get(k, "") for k in sorted(weights.keys()))
            + (manifest.get("logic_hash") or "")
            + (manifest.get("logic_meta_hash") or "")
            + (manifest.get("tokenizer_hash") or "")
        ).encode("utf-8")
        manifest["signed_hash"] = hmac.new(
            sign_key.encode("utf-8"), sign_payload, hashlib.sha256
        ).hexdigest()

    # 8. Write the zip file
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_FILENAME, json.dumps(manifest, indent=2, default=str))
        zf.writestr(MODEL_FILENAME, model_json_bytes)
        zf.writestr(WEIGHTS_FILENAME, weights_bytes)
        for name, data in logic_files.items():
            zf.writestr(name, data)
        if tokenizer_bytes is not None:
            zf.writestr(TOKENIZER_FILENAME, tokenizer_bytes)
        if include_readme:
            zf.writestr(README_FILENAME, _generate_readme(manifest, model_json, weights, logic_meta if logic_fns else None, tokenizer))

    return path


def _generate_readme(
    manifest: Dict[str, Any],
    model_json: Dict[str, Any],
    weights: Dict[str, np.ndarray],
    logic_meta: Optional[List[Dict]],
    tokenizer: Optional[Any],
) -> str:
    """Auto-generate a human-readable README for the bundle."""
    lines = [
        f"# {manifest['technique']} model bundle",
        "",
        f"- **Format version**: {manifest['format_version']}",
        f"- **Created**: {manifest.get('created_at_iso', 'unknown')}",
        f"- **Technique**: `{manifest['technique']}`",
        f"- **Total parameters**: {manifest['total_params']:,}",
        f"- **Weight format**: `{manifest['weight_format']}`",
        f"- **Signed**: {'yes' if manifest.get('signed') else 'no'}",
        f"- **Has tokenizer**: {'yes' if manifest.get('has_tokenizer') else 'no'}",
        f"- **Logic functions**: {manifest.get('logic_count', 0)}",
        "",
    ]
    if manifest.get("description"):
        lines.append(f"## Description")
        lines.append("")
        lines.append(manifest["description"])
        lines.append("")

    if manifest.get("author"):
        lines.append(f"**Author**: {manifest['author']}")
        lines.append("")

    lines.append("## Weight tensors")
    lines.append("")
    lines.append("| Name | Shape | Dtype |")
    lines.append("|---|---|---|")
    for k, v in weights.items():
        lines.append(f"| `{k}` | {list(v.shape)} | {v.dtype} |")
    lines.append("")

    if logic_meta:
        lines.append("## Bundled logic functions")
        lines.append("")
        for m in logic_meta:
            lines.append(f"- **`{m['name']}`**")
            if m.get("closure_deps"):
                lines.append(f"  - Closure deps: {m['closure_deps']}")
        lines.append("")

    if tokenizer is not None:
        lines.append("## Tokenizer")
        lines.append("")
        lines.append(f"- Type: `{type(tokenizer).__name__}`")
        if hasattr(tokenizer, "vocab_size"):
            lines.append(f"- Vocab size: {tokenizer.vocab_size}")
        lines.append("")

    lines.append("## How to load")
    lines.append("")
    lines.append("```python")
    lines.append("import qai")
    lines.append(f'bundle = qai.load_bundle("{manifest["technique"]}.qmodel")')
    lines.append("model = bundle.model")
    lines.append("if bundle.tokenizer:")
    lines.append("    tok = bundle.tokenizer")
    lines.append("if bundle.logic_fns:")
    lines.append("    my_fn = bundle.logic_fns['function_name']")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# =====================================================================
# Bundle inspection (no loading)
# =====================================================================
def inspect_bundle(path: str) -> Dict[str, Any]:
    """Inspect a .qmodel bundle without loading the model.

    Returns a dict with:
        - manifest: the full manifest
        - contents: list of files in the bundle
        - size_bytes: total bundle size
        - size_uncompressed_bytes: sum of file sizes
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bundle not found: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        manifest_bytes = zf.read(MANIFEST_FILENAME)
        manifest = json.loads(manifest_bytes)
        contents = [{"name": info.filename,
                     "size": info.file_size,
                     "compressed_size": info.compress_size}
                    for info in zf.infolist()]
        size_uncompressed = sum(c["size"] for c in contents)

    return {
        "manifest": manifest,
        "contents": contents,
        "size_bytes": os.path.getsize(path),
        "size_uncompressed_bytes": size_uncompressed,
    }


# =====================================================================
# Bundle signature verification
# =====================================================================
def verify_bundle(path: str, sign_key: str) -> bool:
    """Verify a bundle's HMAC-SHA256 signature.

    Returns True if the signature is valid, False otherwise. Bundles saved
    without a `sign_key=` always return False (they're not signed).

    Tampering with any file in the bundle invalidates the signature —
    consumers should call this before loading untrusted .qmodel files.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bundle not found: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read(MANIFEST_FILENAME))
        if not manifest.get("signed"):
            return False
        if not manifest.get("signed_hash"):
            return False

        # Recompute the expected hash
        model_json_bytes = zf.read(MODEL_FILENAME)
        weights_bytes = zf.read(WEIGHTS_FILENAME)

        # Recompute per-weight hashes. If the weights file is corrupted/tampered,
        # np.load will fail — which itself indicates tampering, so return False.
        try:
            weights_npz = np.load(io.BytesIO(weights_bytes), allow_pickle=False)
            weight_hashes = {k: _hash_bytes(weights_npz[k].tobytes()) for k in weights_npz.files}
        except Exception:
            # Corrupted weights file = tampering detected
            return False

        # Recompute logic + tokenizer hashes if those files exist in the bundle.
        # These MUST be in the signed payload — logic.py is exec'd at load time,
        # so an unsigned logic file lets an attacker ship arbitrary code that
        # passes verify_bundle. (Older versions omitted these from the payload.)
        names = set(zf.namelist())
        logic_hash = _hash_bytes(zf.read(LOGIC_FILENAME)) if LOGIC_FILENAME in names else ""
        logic_meta_hash = (
            _hash_bytes(zf.read(LOGIC_META_FILENAME))
            if LOGIC_META_FILENAME in names else ""
        )
        tokenizer_hash = (
            _hash_bytes(zf.read(TOKENIZER_FILENAME))
            if TOKENIZER_FILENAME in names else ""
        )

        sign_payload = (
            _hash_bytes(model_json_bytes)
            + _hash_bytes(weights_bytes)
            + "".join(weight_hashes.get(k, "") for k in sorted(weight_hashes.keys()))
            + logic_hash
            + logic_meta_hash
            + tokenizer_hash
        ).encode("utf-8")

        expected = hmac.new(
            sign_key.encode("utf-8"), sign_payload, hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected, manifest["signed_hash"])


# =====================================================================
# Bundle loading
# =====================================================================
def load_bundle(
    path: str,
    sign_key: Optional[str] = None,
    verify_signature: bool = True,
    device: Optional[str] = None,
) -> Bundle:
    """Load a .qmodel bundle.

    Args:
        path: Path to a .qmodel file.
        sign_key: If the bundle is signed, pass the same key used to sign it.
            Required if `verify_signature=True` and the bundle is signed.
        verify_signature: If True (default), verify the signature before loading.
            Set to False to skip verification (NOT recommended for untrusted bundles).

    Returns:
        A Bundle containing the reconstructed model + optional tokenizer + logic_fns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bundle not found: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read(MANIFEST_FILENAME))

        # Format-version check + migration
        manifest = _migrate_manifest(manifest)

        # Verify signature if requested
        if verify_signature and manifest.get("signed"):
            if sign_key is None:
                raise ValueError(
                    "[WHAT] Bundle is signed but no sign_key provided\n"
                    "[WHY] Signed bundles require the original signing key to verify\n"
                    "[FIX] Pass sign_key='your_secret_key', or verify_signature=False "
                    "(NOT recommended for untrusted bundles)"
                )
            if not verify_bundle(path, sign_key):
                raise ValueError(
                    "[WHAT] Bundle signature verification failed\n"
                    "[WHY] The bundle may have been tampered with, or the sign_key is wrong\n"
                    "[FIX] Check that you're using the correct signing key, or reject this bundle"
                )

        # Load README (if present) -- exposed as bundle.readme
        readme_text = None
        if README_FILENAME in zf.namelist():
            readme_text = zf.read(README_FILENAME).decode("utf-8")

        # Load model.json
        model_json = json.loads(zf.read(MODEL_FILENAME))
        technique_name = model_json["technique"]

        # Load weights
        weights_bytes = zf.read(WEIGHTS_FILENAME)
        try:
            weights_npz = np.load(io.BytesIO(weights_bytes), allow_pickle=False)
            weights = {k: weights_npz[k] for k in weights_npz.files}
        except Exception as e:
            raise ValueError(
                "[WHAT] Failed to load weights from bundle\n"
                f"[WHY] {e}\n"
                "[FIX] The bundle may be corrupted or tampered with"
            ) from e

        # Verify per-weight hashes
        for k, v in weights.items():
            actual_hash = _hash_bytes(v.tobytes())
            expected_hash = manifest.get("weight_hashes", {}).get(k)
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(
                    f"[WHAT] Weight tensor '{k}' hash mismatch\n"
                    f"[WHY] The weights file may be corrupted or tampered with\n"
                    f"[FIX] Re-download or re-create the bundle"
                )

        # Reconstruct the model
        import qai
        params = {**model_json.get("params", {}), **model_json.get("extra_params", {})}
        # Remove params that the constructor doesn't accept (filter to common ones)
        # Actually, qai.build passes **params to the technique constructor, so we
        # need to be careful. Let's pass only extra_params (the actually-set attrs).
        build_params = dict(model_json.get("extra_params", {}))

        # Handle device override
        if device is not None and "device" in build_params:
            build_params["device"] = device

        # Composite-agent path: if the bundle was saved with a composite weight
        # format, the model_json will have a "composite_class" field naming the
        # class to reconstruct. The class must be importable.
        if model_json.get("weight_format") == "composite":
            composite_class_path = model_json.get("composite_class")
            if not composite_class_path:
                raise ValueError(
                    "[WHAT] Composite bundle has no composite_class field\n"
                    "[WHY] The bundle was saved from a composite agent but the "
                    "class name wasn't recorded\n"
                    "[FIX] Re-save the bundle with a _qai_bundle_state() that "
                    "includes a 'composite_class' field"
                )
            # Import the class
            module_path, class_name = composite_class_path.rsplit(".", 1)
            try:
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(
                    f"[WHAT] Cannot import composite class {composite_class_path!r}\n"
                    f"[WHY] {e}\n"
                    f"[FIX] Make sure the class is on the Python path"
                ) from e
            # Instantiate without args (composite agents typically accept no
            # constructor args OR accept the bundled params).
            try:
                model = cls(**build_params) if build_params else cls()
            except TypeError:
                model = cls()
        else:
            model = qai.build(technique=technique_name, **build_params)

        # For PyTorch-backed techniques, we need to ensure the net is built before restoring
        # (Skip for composite bundles -- they have their own restore path)
        if model_json["weight_format"] != "composite":
            technique = getattr(model, "technique", None)
            if model_json["weight_format"] == "pytorch_sd" and technique is not None:
                if hasattr(technique, "_build_model"):
                    # Different techniques have different _build_model signatures:
                    # - Transformer: _build_model(vocab_size, n_classes)
                    # - CNN: _build_model(n_classes)
                    # - NeuralNetwork: no _build_model (builds in __init__)
                    # Use inspect to call with only the args the method accepts.
                    import inspect
                    sig = inspect.signature(technique._build_model)
                    kwargs = {}
                    if "vocab_size" in sig.parameters:
                        kwargs["vocab_size"] = build_params.get("vocab_size", 256)
                    if "n_classes" in sig.parameters:
                        kwargs["n_classes"] = build_params.get("n_classes")
                    technique._build_model(**kwargs)
                    if device is not None:
                        import torch
                        technique.device = torch.device(device)
                        technique.net = technique.net.to(technique.device)
                        if hasattr(technique.net, "_rope") and technique.net._rope is not None:
                            technique.net._rope = technique.net._rope.to(technique.device)

        # Restore weights
        _restore_weights(model, weights, model_json["weight_format"])

        # Load tokenizer if present
        tokenizer = None
        if TOKENIZER_FILENAME in manifest.get("contents", {}):
            tok_bytes = zf.read(TOKENIZER_FILENAME)
            # Write to temp file and use the tokenizer's load method
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as f:
                f.write(tok_bytes)
                tmp_path = f.name
            try:
                from qai.tokenizers import BaseTokenizer
                tokenizer = BaseTokenizer.load(tmp_path)
            finally:
                os.unlink(tmp_path)

            # If the technique has a `tokenizer` attribute (Transformer does), set it
            if hasattr(technique, "tokenizer"):
                technique.tokenizer = tokenizer

        # Load logic functions if present
        logic_fns: Dict[str, Callable] = {}
        if LOGIC_FILENAME in manifest.get("contents", {}):
            logic_source_bytes = zf.read(LOGIC_FILENAME)
            logic_source = logic_source_bytes.decode("utf-8")
            logic_meta_bytes = zf.read(LOGIC_META_FILENAME) if LOGIC_META_FILENAME in zf.namelist() else None
            logic_meta_list = json.loads(logic_meta_bytes) if logic_meta_bytes else []

            # DEFENSE-IN-DEPTH: verify logic.py + logic_meta.json hashes
            # against the manifest BEFORE exec'ing them. If the bundle was
            # tampered after signing, this raises instead of running the
            # attacker's code. (Even if the caller skipped verify_bundle.)
            expected_logic_hash = manifest.get("logic_hash")
            if expected_logic_hash is not None:
                actual_logic_hash = _hash_bytes(logic_source_bytes)
                if not hmac.compare_digest(expected_logic_hash, actual_logic_hash):
                    raise ValueError(
                        "[WHAT] Bundle logic.py hash mismatch — possible tampering\n"
                        "[WHY] The manifest's logic_hash doesn't match the actual "
                        "logic.py bytes in the bundle. Someone may have modified "
                        "logic.py after the bundle was signed.\n"
                        "[FIX] Re-sign the bundle from a trusted source, or reject it"
                    )
            if logic_meta_bytes is not None:
                expected_meta_hash = manifest.get("logic_meta_hash")
                if expected_meta_hash is not None:
                    actual_meta_hash = _hash_bytes(logic_meta_bytes)
                    if not hmac.compare_digest(expected_meta_hash, actual_meta_hash):
                        raise ValueError(
                            "[WHAT] Bundle logic_meta.json hash mismatch — possible tampering\n"
                            "[WHY] The manifest's logic_meta_hash doesn't match the actual "
                            "logic_meta.json bytes.\n"
                            "[FIX] Re-sign the bundle from a trusted source, or reject it"
                        )

            # Build the exec namespace
            import sys
            exec_namespace = {
                "qai": sys.modules.get("qai") or __import__("qai"),
                "np": np,
            }
            # Provide model + tokenizer in namespace (common closure deps)
            exec_namespace["model"] = model
            if tokenizer is not None:
                exec_namespace["tokenizer"] = tokenizer

            # Re-inject captured constants from each logic_meta entry.
            # These are the simple numeric/string/list/dict values that the
            # logic function's origin module had at save time (e.g. `LIMIT = 100`
            # at module top). Without this, exec raises NameError in a fresh
            # process because the origin module's globals aren't loaded.
            for meta in logic_meta_list:
                constants = meta.get("constants") or {}
                for name, value in constants.items():
                    if name not in exec_namespace:
                        exec_namespace[name] = value

            # Verify `requires` lists (from logic_meta) are satisfied
            for meta in logic_meta_list:
                requires = meta.get("requires", [])
                missing = [r for r in requires if r not in exec_namespace]
                if missing:
                    # Try to inject from the model's technique (common case: agent)
                    for r in list(missing):
                        if hasattr(model, r):
                            exec_namespace[r] = getattr(model, r)
                            missing.remove(r)
                        elif hasattr(model, "technique") and hasattr(model.technique, r):
                            exec_namespace[r] = getattr(model.technique, r)
                            missing.remove(r)
                        # NEW: for composite agents (AgentBase subclasses), the
                        # model IS the agent -- inject `model` as `agent` so
                        # `requires=["agent"]` works automatically.
                        elif r == "agent" and hasattr(model, "run_episode"):
                            exec_namespace[r] = model
                            missing.remove(r)
                    if missing:
                        raise ValueError(
                            f"[WHAT] Logic function '{meta.get('name')}' requires "
                            f"namespace names {missing} that weren't provided\n"
                            f"[WHY] The function was decorated with "
                            f"@qai.logic(requires={missing!r}) but the bundle "
                            f"loader couldn't find those names\n"
                            f"[FIX] Pass them via the load_bundle namespace, "
                            f"or attach them to the model before loading"
                        )

            exec(logic_source, exec_namespace)

            # Extract all callable, non-dunder, non-builtin names that look like logic fns
            builtin_keys = {"qai", "np", "model", "tokenizer", "__builtins__"}
            # Add any extra injected names so we don't treat them as logic fns.
            # This includes `requires` names AND captured `constants` (they're
            # data, not callables — but be defensive in case a constant is a
            # callable that got captured by mistake).
            for meta in logic_meta_list:
                for r in meta.get("requires", []):
                    builtin_keys.add(r)
                for cname in (meta.get("constants") or {}).keys():
                    builtin_keys.add(cname)
            for name, obj in exec_namespace.items():
                if name in builtin_keys or name.startswith("_"):
                    continue
                if callable(obj) and hasattr(obj, "logic_name"):
                    logic_fns[obj.logic_name] = obj
                elif callable(obj) and not isinstance(obj, type):
                    # Could be a plain function — include it
                    logic_fns[name] = obj

    return Bundle(
        model=model,
        tokenizer=tokenizer,
        logic_fns=logic_fns,
        manifest=manifest,
        path=path,
        readme=readme_text,
    )


# =====================================================================
# Bundle diff
# =====================================================================
def diff_bundles(path1: str, path2: str) -> Dict[str, Any]:
    """Compare two .qmodel bundles.

    Returns a dict with:
        - technique_changed: bool
        - params_changed: dict of {param: (old, new)}
        - weights_changed: bool
        - weight_diffs: dict of {tensor_name: {shape_match, l2_distance, max_abs_change}}
        - logic_changed: bool
        - tokenizer_changed: bool
        - manifest_diff: dict of changed manifest fields
    """
    with zipfile.ZipFile(path1, "r") as zf1, zipfile.ZipFile(path2, "r") as zf2:
        m1 = json.loads(zf1.read(MANIFEST_FILENAME))
        m2 = json.loads(zf2.read(MANIFEST_FILENAME))
        mj1 = json.loads(zf1.read(MODEL_FILENAME))
        mj2 = json.loads(zf2.read(MODEL_FILENAME))

        # Load weights
        w1 = {k: v for k, v in np.load(io.BytesIO(zf1.read(WEIGHTS_FILENAME))).items()}
        w2 = {k: v for k, v in np.load(io.BytesIO(zf2.read(WEIGHTS_FILENAME))).items()}

    # Technique diff
    technique_changed = mj1["technique"] != mj2["technique"]

    # Params diff
    params_changed = {}
    all_params = set(mj1.get("extra_params", {}).keys()) | set(mj2.get("extra_params", {}).keys())
    for p in all_params:
        v1 = mj1.get("extra_params", {}).get(p)
        v2 = mj2.get("extra_params", {}).get(p)
        if v1 != v2:
            params_changed[p] = (v1, v2)

    # Weights diff
    weight_diffs: Dict[str, Dict[str, Any]] = {}
    all_weight_keys = set(w1.keys()) | set(w2.keys())
    weights_changed = False
    for k in all_weight_keys:
        if k not in w1:
            weight_diffs[k] = {"added": True}
            weights_changed = True
        elif k not in w2:
            weight_diffs[k] = {"removed": True}
            weights_changed = True
        else:
            v1, v2 = w1[k], w2[k]
            if v1.shape != v2.shape:
                weight_diffs[k] = {"shape_mismatch": True, "shape1": list(v1.shape), "shape2": list(v2.shape)}
                weights_changed = True
            else:
                delta = v2 - v1
                l2 = float(np.sqrt(np.sum(delta ** 2)))
                max_abs = float(np.max(np.abs(delta))) if delta.size > 0 else 0.0
                mean_abs = float(np.mean(np.abs(delta)))
                weight_diffs[k] = {
                    "l2_distance": l2,
                    "max_abs_change": max_abs,
                    "mean_abs_change": mean_abs,
                    "n_changed": int(np.sum(np.abs(delta) > 1e-8)),
                }
                if l2 > 1e-8:
                    weights_changed = True

    # Logic diff
    logic_changed = m1.get("logic_count", 0) != m2.get("logic_count", 0)

    # Tokenizer diff
    tokenizer_changed = m1.get("has_tokenizer", False) != m2.get("has_tokenizer", False)

    # Manifest diff (interesting fields only)
    manifest_diff = {}
    for key in ("format_version", "description", "author", "weight_format", "total_params"):
        if m1.get(key) != m2.get(key):
            manifest_diff[key] = {"bundle1": m1.get(key), "bundle2": m2.get(key)}

    return {
        "technique_changed": technique_changed,
        "technique1": mj1["technique"],
        "technique2": mj2["technique"],
        "params_changed": params_changed,
        "weights_changed": weights_changed,
        "weight_diffs": weight_diffs,
        "logic_changed": logic_changed,
        "tokenizer_changed": tokenizer_changed,
        "manifest_diff": manifest_diff,
    }


# =====================================================================
# Public API
# =====================================================================
__all__ = [
    "Bundle",
    "save_bundle",
    "load_bundle",
    "inspect_bundle",
    "verify_bundle",
    "diff_bundles",
    "register_bundle_migration",
    "BUNDLE_FORMAT_VERSION",
    "SUPPORTED_FORMAT_VERSIONS",
]
