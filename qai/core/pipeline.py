"""
qai.core.pipeline -- Section 37: model.pipe_to(other_model). A real,
minimal chain: output of one model becomes the input of the next.
"""
from __future__ import annotations
import numpy as np


class Pipeline:
    def __init__(self, models: list):
        self.models = models

    def predict(self, x):
        result = x
        for model in self.models:
            result = model.predict(result)
        return result

    def __repr__(self):
        chain = " -> ".join(m.technique_name for m in self.models)
        return f"<Pipeline: {chain}>"
