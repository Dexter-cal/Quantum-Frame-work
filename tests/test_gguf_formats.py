"""v1.8.9 — ONE GGUF surface: one canonical writer + one canonical reader.

Regression tests for focus-list item 5. Before v1.8.9:

  * ``qai.formats.UniversalConverter._load(".gguf")`` raised
    NotImplementedError, claiming GGUF needs llama.cpp — while
    ``qai.conversion.import_model()`` could read GGUF all along.
  * ``qai.formats._save()`` had NO .gguf branch at all, so
    ``convert(..., ".gguf")`` fell through to a generic NotImplementedError
    even though the module docstring advertises it.
  * ``qai.formats.to_gguf()`` saved a side .safetensors file and THEN raised
    NotImplementedError pointing at llama.cpp's convert_hf_to_gguf.py.
  * The "canonical" writer itself was broken twice over:
    - ``GGUFWriter.add_block_type`` no longer exists in the gguf package, so
      every export crashed;
    - ``SafeTensorsExporter._extract_weights`` had no dict fast path and
      scanned ``vars(model)`` — a TypeError on plain weight dicts.

Now qai.formats delegates load and save to qai.conversion (one reader, one
writer), and the writer is metadata-aware. The tests use a seeded RNG and
plain weight dicts (the sandbox has no torch).
"""
import ast
import inspect
import os
import re
import sys

import numpy as np
import pytest

import qai.formats as formats
from qai.conversion import GGUFExporter, import_model


def _rng():
    return np.random.default_rng(19)


def _dict_model():
    r = _rng()
    return {
        "w1": r.normal(size=(4, 3)).astype(np.float32),
        "w2": r.normal(size=(3, 2)).astype(np.float32),
    }


def _read_all(path):
    """Full read-back via the gguf package: (kv dict, {name: ndarray})."""
    import gguf
    reader = gguf.GGUFReader(str(path))
    kv = {k: reader.get_field(k).contents()
          for k in reader.fields if reader.get_field(k) is not None}
    tensors = {t.name: t.data for t in reader.tensors}
    return kv, tensors


# ---------------------------------------------------------------------------
# The core regression: GGUF files are real, and the surface is honest
# ---------------------------------------------------------------------------

def test_to_gguf_writes_real_gguf_file(tmp_path):
    p = formats.to_gguf(_dict_model(), str(tmp_path / "m.gguf"))
    assert p.endswith("m.gguf")
    with open(p, "rb") as f:
        assert f.read(4) == b"GGUF"


def test_to_gguf_no_side_safetensors_file(tmp_path):
    # The old stub saved a side .safetensors file before raising.
    out = tmp_path / "m"
    formats.to_gguf(_dict_model(), str(out) + ".gguf")
    leftovers = sorted(x.name for x in tmp_path.iterdir())
    assert leftovers == ["m.gguf"], f"unexpected files: {leftovers}"


def test_load_returns_same_tensors(tmp_path):
    model = _dict_model()
    p = formats.to_gguf(model, str(tmp_path / "m.gguf"))
    back = formats.UniversalConverter._load(p, ".gguf")
    assert set(back) == set(model)
    for k in model:
        assert back[k].shape == model[k].shape
        assert np.allclose(back[k], model[k])


def test_import_model_detects_gguf_by_suffix(tmp_path):
    model = _dict_model()
    p = formats.to_gguf(model, str(tmp_path / "m.gguf"))
    back = import_model(p)  # no format= argument
    assert np.allclose(back["w1"], model["w1"])


def test_convert_npz_to_gguf_end_to_end(tmp_path):
    # The docstring example: convert between formats, into GGUF.
    model = _dict_model()
    src = tmp_path / "in.npz"
    np.savez(src, **model)
    out = tmp_path / "out.gguf"
    assert formats.convert(str(src), str(out)) == str(out)
    _, tensors = _read_all(out)
    assert set(tensors) == {"w1", "w2"}
    assert np.allclose(tensors["w1"], model["w1"])


def test_convert_gguf_to_npz_end_to_end(tmp_path):
    # ...and out of GGUF again (the load side used to NotImplementedError).
    model = _dict_model()
    gg = tmp_path / "m.gguf"
    formats.to_gguf(model, str(gg))
    out = formats.convert(str(gg), str(tmp_path / "back.npz"))
    assert out.endswith("back.npz")
    z = np.load(out, allow_pickle=True)
    assert set(z.files) == {"w1", "w2"}
    assert np.allclose(z["w1"], model["w1"])


def test_save_direct_gguf_branch(tmp_path):
    # UniversalConverter._save now has an actual .gguf branch.
    p = formats.UniversalConverter._save(_dict_model(), str(tmp_path / "m.gguf"), ".gguf")
    with open(p, "rb") as f:
        assert f.read(4) == b"GGUF"


# ---------------------------------------------------------------------------
# Metadata-aware writer
# ---------------------------------------------------------------------------

def test_custom_metadata_all_types_roundtrip(tmp_path):
    model = _dict_model()
    p = tmp_path / "meta.gguf"
    with pytest.warns(UserWarning, match="empty"):
        GGUFExporter().export(model, str(p), metadata={
            "run": "exp-1",
            "layers": 2,
            "learning_rate": 0.001,
            "frozen": True,
            "tags": ["a", "b"],
            "empty": [],  # GGUF cannot store empty arrays -> warned + skipped
        })
    kv, _ = _read_all(p)
    assert kv["run"] == "exp-1"
    assert kv["layers"] == 2
    assert kv["learning_rate"] == pytest.approx(0.001)
    assert kv["frozen"] is True
    assert list(kv["tags"]) == ["a", "b"]
    assert "empty" not in kv


def test_automatic_metadata_present(tmp_path):
    p = tmp_path / "auto.gguf"
    GGUFExporter().export(_dict_model(), str(p))
    kv, _ = _read_all(p)
    assert kv["qai.tensor_count"] == 2
    assert kv["qai.exported_by"] == "qai.conversion.GGUFExporter"


def test_model_name_defaults_are_sensible(tmp_path):
    class TinyNet:
        def __init__(self):
            self.dense_w = np.ones((2, 2), dtype=np.float32)

    p_obj = tmp_path / "obj.gguf"
    GGUFExporter().export(TinyNet(), str(p_obj))
    kv_obj, _ = _read_all(p_obj)
    assert kv_obj["general.name"] == "TinyNet"

    p_dict = tmp_path / "dict.gguf"
    GGUFExporter().export(_dict_model(), str(p_dict))
    kv_dict, _ = _read_all(p_dict)
    assert kv_dict["general.name"] == "qai-weights"


def test_quantization_recorded_honestly(tmp_path):
    p = formats.to_gguf(_dict_model(), str(tmp_path / "q.gguf"), quantization="Q4_K_M")
    kv, tensors = _read_all(p)
    assert kv["qai.quantization_requested"] == "Q4_K_M"
    # Weights stay float32 -- quantizing is llama.cpp's job, not this writer's.
    assert all(t.dtype == np.float32 for t in tensors.values())
    assert "llama.cpp" in kv["qai.weights_stored_as"]


def test_block_type_kwarg_backward_compatible(tmp_path):
    # The old writer called GGUFWriter.add_block_type (a method that no
    # longer exists) -- every export crashed. The kwarg survives as metadata.
    p = tmp_path / "bt.gguf"
    GGUFExporter().export(_dict_model(), str(p), block_type="qai")
    kv, _ = _read_all(p)
    assert kv["qai.block_type"] == "qai"


def test_int_weights_exported_as_float32(tmp_path):
    p = tmp_path / "ints.gguf"
    GGUFExporter().export({"iw": np.arange(6).reshape(2, 3)}, str(p))
    _, tensors = _read_all(p)
    assert tensors["iw"].dtype == np.float32
    assert np.allclose(tensors["iw"], np.arange(6).reshape(2, 3))


def test_empty_weights_rejected(tmp_path):
    with pytest.raises(ValueError, match="no extractable weights"):
        GGUFExporter().export({}, str(tmp_path / "e.gguf"))


def test_dict_models_work_for_every_exporter(tmp_path):
    # Root-bug fix: _extract_weights used to vars()-scan and crash with
    # TypeError on plain dicts -- breaking safetensors export of dict models
    # too, not just GGUF.
    safetensors = pytest.importorskip("safetensors")
    from qai.conversion import SafeTensorsExporter
    p = SafeTensorsExporter().export(_dict_model(), str(tmp_path / "s.safetensors"))
    from safetensors.numpy import load_file
    back = load_file(p)
    assert np.allclose(back["w1"], _dict_model()["w1"])


# ---------------------------------------------------------------------------
# Honest failure when the gguf package is missing
# ---------------------------------------------------------------------------

def test_missing_gguf_package_honest_error_writer(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "gguf", None)
    with pytest.raises(ImportError, match="pip install gguf"):
        GGUFExporter().export(_dict_model(), str(tmp_path / "x.gguf"))


def test_missing_gguf_package_honest_error_reader(tmp_path, monkeypatch):
    p = formats.to_gguf(_dict_model(), str(tmp_path / "m.gguf"))
    monkeypatch.setitem(sys.modules, "gguf", None)
    with pytest.raises(ImportError, match="pip install gguf"):
        import_model(p)


# ---------------------------------------------------------------------------
# One implementation, enforced at the source level
# ---------------------------------------------------------------------------

def _executable_code(module) -> str:
    """Module source with docstrings emptied and comments dropped, so checks
    see what the code DOES, not what its docs say."""
    src = inspect.getsource(module)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body[0].value.value = ""
    return ast.unparse(tree)


def test_formats_delegates_gguf_to_conversion():
    code = _executable_code(formats)
    # No llama.cpp shell-out instructions left in executable code.
    assert "convert_hf_to_gguf" not in code
    # formats.py never touches the gguf package itself.
    assert not re.search(r"\bimport gguf\b", code)
    assert "GGUFWriter" not in code
    # Both .gguf branches explicitly import the canonical implementations.
    assert "from .conversion import import_model" in code
    assert "from .conversion import GGUFExporter" in code


def test_conversion_has_exactly_one_gguf_writer():
    import qai.conversion as conv
    code = _executable_code(conv)
    assert code.count("GGUFWriter(") == 1  # only GGUFExporter constructs one
    # The dead API call that broke every export is gone.
    assert "add_block_type" not in code


def test_formats_registry_still_advertises_gguf():
    info = formats.format_info(".gguf")
    assert info["supports_quantization"] and info["stores_weights"]


# ---------------------------------------------------------------------------
# The fifth GGUF writer: qai.quantization's counterfeit, now a delegate
# ---------------------------------------------------------------------------

def test_save_gguf_writes_real_gguf(tmp_path):
    # qai.save_gguf used to write a counterfeit binary layout that carried
    # GGUF magic bytes but which NO tool could read (and it crashed with
    # NameError without torch). It now delegates to the canonical writer.
    from qai.quantization import save_gguf
    p = save_gguf(_dict_model(), str(tmp_path / "m.gguf"), quantization="q4_k_m")
    with open(p, "rb") as f:
        assert f.read(4) == b"GGUF"
    kv, tensors = _read_all(p)
    assert set(tensors) == {"w1", "w2"}
    assert kv["qai.quantization_requested"] == "q4_k_m"
    assert np.allclose(tensors["w1"], _dict_model()["w1"])


def test_save_gguf_appends_suffix(tmp_path):
    from qai.quantization import save_gguf
    p = save_gguf(_dict_model(), str(tmp_path / "noext"))
    assert p.endswith(".gguf") and os.path.exists(p)


def test_save_gguf_f16_stored_natively(tmp_path):
    from qai.quantization import save_gguf
    p = save_gguf({"w": np.full((2, 2), 0.5, np.float32)},
                  str(tmp_path / "f16.gguf"), quantization="f16")
    kv, tensors = _read_all(p)
    assert tensors["w"].dtype == np.float16
    assert kv["qai.weights_stored_as"] == "float16"


def test_save_gguf_kquant_honest_metadata(tmp_path):
    from qai.quantization import save_gguf
    p = save_gguf({"w": np.ones((2, 2), np.float32)},
                  str(tmp_path / "q4.gguf"), quantization="q4_k_m")
    kv, tensors = _read_all(p)
    assert tensors["w"].dtype == np.float32  # not fake int8 "quantization"
    assert "llama.cpp" in kv["qai.weights_stored_as"]


def test_load_gguf_reads_real_gguf(tmp_path):
    from qai.quantization import save_gguf, load_gguf
    model = _dict_model()
    p = save_gguf(model, str(tmp_path / "m.gguf"))
    back = load_gguf(p)
    assert set(back) == set(model)
    assert np.allclose(back["w2"], model["w2"])


def test_load_gguf_rejects_counterfeit_legacy_files(tmp_path):
    # A file with the OLD counterfeit layout (magic bytes + invented header)
    # is not a GGUF file; the real parser says so instead of mis-parsing.
    import struct
    p = tmp_path / "legacy.gguf"
    with open(p, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("I", 1))  # fake version
        f.write(struct.pack("I", 2))  # fake n_weights
        f.write(struct.pack("I", 4))  # fake bits
        f.write(struct.pack("I", 2))  # fake name length
        f.write(b"w1")
    with pytest.raises(ValueError):
        from qai.quantization import load_gguf
        load_gguf(str(p))


def test_save_exl2_refuses_to_fake(tmp_path):
    # save_exl2 used to route to the counterfeit GGUF writer -- a file with
    # GGUF magic bytes under an EXL2 name that exllamav2 cannot read.
    from qai.quantization import save_exl2
    with pytest.raises(ImportError, match="exllamav2"):
        save_exl2({}, str(tmp_path / "m.exl2"))


def test_model_save_gguf_returns_real_gguf(tmp_path):
    # The documented guide path: m.save("model.gguf", format="gguf").
    pytest.importorskip("sklearn")
    import qai
    m = qai.Model(technique_name="ridge")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 3)).astype(np.float32)
    y = X @ np.array([1.0, 2.0, 3.0]) + 0.1
    m.train(X, y)
    out = m.save(str(tmp_path / "model.gguf"), format="gguf")
    assert out is not None and out.endswith("model.gguf")
    kv, tensors = _read_all(out)
    assert len(tensors) >= 1
    assert kv["qai.quantization_requested"] == "q4_k_m"


def test_quantization_module_has_no_gguf_code_left():
    import qai.quantization as quant
    code = _executable_code(quant)
    assert "GGUFWriter" not in code        # no second writer
    assert "_save_gguf_fallback" not in code
    assert "b'GGUF'" not in code and 'b"GGUF"' not in code  # no magic writes
