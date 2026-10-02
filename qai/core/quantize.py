"""
qai.core.quantize -- Model quantization and compression helper.
Quantizes floating point weights to lower precision representations for lightweight deployment.
"""
from __future__ import annotations
import numpy as np


def compress_model(model, precision: str = "float16"):
    """Compresses model float weights to float16 or int8 representation."""
    if precision not in ("float16", "int8"):
        raise ValueError("precision must be either 'float16' or 'int8'")

    technique = model.technique
    if hasattr(technique, "w") and technique.w is not None:
        if precision == "float16":
            technique.w = technique.w.astype(np.float16)
        else:
            scale = np.max(np.abs(technique.w)) or 1.0
            technique.w = np.round((technique.w / scale) * 127).astype(np.int8)

    return model
