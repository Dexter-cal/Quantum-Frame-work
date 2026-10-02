"""
qai.core.persistence -- Binary serialization for trained qai models.
Provides save_model(model, filepath) and load_model(filepath).
"""
from __future__ import annotations
import pickle
import os


def save_model(model, filepath: str) -> str:
    """Serialize a trained qai Model or Pipeline object to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    return filepath


def load_model(filepath: str):
    """Load a serialized qai Model or Pipeline object from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: '{filepath}'")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
