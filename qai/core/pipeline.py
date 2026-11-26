"""
qai.core.pipeline -- Section 37: model.pipe_to(other_model) and multi-stage Pipeline execution.
A real, minimal chain: output of one model becomes the input of the next.
Supports sequential training and prediction across multi-stage pipelines.
"""
from __future__ import annotations
import numpy as np


class Pipeline:
    def __init__(self, models: list):
        if not models:
            raise ValueError("Pipeline requires a non-empty list of models")
        self.models = models

    def train(self, X, y=None, verbose=True):
        """Train stages sequentially: transform/predict intermediate features."""
        current_X = X
        for idx, model in enumerate(self.models):
            # Final model receives y if provided; intermediate transformers train on features only
            if idx == len(self.models) - 1 and y is not None:
                model.train(current_X, y, verbose=verbose)
            else:
                model.train(current_X, verbose=verbose)

            # Pass transformed output forward if not the last stage
            if idx < len(self.models) - 1:
                current_X = model.predict(current_X)
        return self

    def predict(self, x):
        """Pass input sequentially through all pipeline stages."""
        result = x
        for model in self.models:
            result = model.predict(result)
        return result

    def pipe_to(self, other_model):
        """Chain additional models onto the end of the existing pipeline."""
        from .model import Model
        if isinstance(other_model, Pipeline):
            return Pipeline(self.models + other_model.models)
        elif isinstance(other_model, Model):
            return Pipeline(self.models + [other_model])
        raise TypeError("pipe_to requires a Model or Pipeline instance")

    def __repr__(self):
        chain = " -> ".join(m.technique_name for m in self.models)
        return f"<Pipeline: {chain}>"
