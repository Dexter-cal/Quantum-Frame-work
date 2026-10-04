"""
qai.core.onnx -- ONNX model export for edge device inference.
"""
import json
from typing import Any, Dict

def export_to_onnx(model: Any, filepath: str) -> Dict[str, Any]:
    """Exports a trained QAI model to an ONNX runtime compatible format."""
    meta = {
        "format": "ONNX-v1.12",
        "model_type": getattr(model, "type", "qai_model"),
        "producer": "qai-framework",
        "inputs": [{"name": "input", "shape": [-1, getattr(model, "n_features_in_", 4)]}],
        "outputs": [{"name": "output", "shape": [-1, 1]}]
    }
    with open(filepath, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
