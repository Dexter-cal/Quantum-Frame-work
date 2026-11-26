"""
qai.federated_bridge — combine model pieces from different frameworks into one.

Lets you build pieces of a model in PyTorch, sklearn, or custom numpy code,
then combine them into a single unified model that trains and predicts as one.

Example: a model where
  - Piece 1: sklearn StandardScaler (preprocessing)
  - Piece 2: PyTorch MLP (feature extraction)
  - Piece 3: sklearn LogisticRegression (classification head)

All three pieces communicate seamlessly through a unified interface.

Usage:
    import qai
    bridge = qai.FederatedBridge()

    # Add pieces built with different technologies
    bridge.add_piece("scaler", qai.build(technique="standard_scaler"), framework="qai")
    bridge.add_piece("encoder", my_torch_nn_module, framework="pytorch")
    bridge.add_piece("classifier", qai.build(technique="classifier"), framework="qai")

    # Train the whole thing as one model
    bridge.train(X_train, y_train)
    preds = bridge.predict(X_test)
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ModelPiece:
    """A single piece of a federated model."""
    name: str
    model: Any
    framework: str  # "qai", "pytorch", "sklearn", "custom"
    input_from: str = ""  # name of the piece that feeds this one (empty = root)
    params: Dict[str, Any] = field(default_factory=dict)


class FederatedBridge:
    """Combine model pieces from different frameworks into one unified model.

    Pieces communicate through a simple data-passing interface:
      - qai pieces: use model.predict() / model.train()
      - pytorch pieces: use module.forward() / module.parameters()
      - sklearn pieces: use estimator.predict() / estimator.fit()
      - custom pieces: use callable(x) -> x

    The bridge handles the framework translation automatically.
    """
    def __init__(self, name: str = "federated"):
        self.name = name
        self.pieces: Dict[str, ModelPiece] = {}
        self._order: List[str] = []

    def add_piece(self, name: str, model: Any, framework: str = "custom",
                  input_from: str = "", **params) -> "FederatedBridge":
        """Add a model piece.

        Args:
            name: unique piece name.
            model: the model/module/estimator.
            framework: "qai" | "pytorch" | "sklearn" | "custom".
            input_from: name of the piece whose output feeds this one.
                        Empty string = root (takes raw input).
            **params: extra params.
        """
        if name in self.pieces:
            raise ValueError(f"[WHAT] Piece '{name}' already exists")
        if input_from and input_from not in self.pieces:
            raise ValueError(f"[WHAT] input_from '{input_from}' not found")
        self.pieces[name] = ModelPiece(name=name, model=model, framework=framework,
                                        input_from=input_from, params=params)
        self._order.append(name)
        return self

    def forward(self, X) -> Any:
        """Run data through all pieces in order."""
        current = X
        for name in self._order:
            piece = self.pieces[name]
            current = self._forward_piece(piece, current)
        return current

    def predict(self, X) -> Any:
        """Same as forward — for API compatibility."""
        return self.forward(X)

    def train(self, X, y, verbose: bool = False, **kwargs) -> "FederatedBridge":
        """Train all pieces sequentially.

        Each piece trains on the output of the previous piece.
        The last piece uses y as the target.
        """
        current_X = X
        for i, name in enumerate(self._order):
            piece = self.pieces[name]
            is_last = (i == len(self._order) - 1)
            target = y if is_last else None
            self._train_piece(piece, current_X, target, verbose=verbose)

            # Compute this piece's output for the next piece
            current_X = self._forward_piece(piece, current_X)

        return self

    def _forward_piece(self, piece: ModelPiece, X) -> Any:
        """Forward through a single piece, handling framework differences."""
        model = piece.model
        if piece.framework == "qai":
            # qai.Model: use predict() for classifiers, transform() for preprocessors
            tech = getattr(model, "technique", None)
            if tech is not None and hasattr(tech, "transform"):
                try:
                    return tech.transform(X)
                except Exception:
                    pass
            try:
                return model.predict(X)
            except Exception:
                return X
        elif piece.framework == "pytorch":
            import torch
            model.eval()
            with torch.no_grad():
                X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
                out = model(X_t)
                return out.numpy()
        elif piece.framework == "sklearn":
            if hasattr(model, "transform"):
                return model.transform(X)
            elif hasattr(model, "predict"):
                return model.predict(X)
            return X
        else:  # custom
            if callable(model):
                return model(X)
            return X

    def _train_piece(self, piece: ModelPiece, X, y, verbose: bool = False):
        """Train a single piece, handling framework differences."""
        model = piece.model
        if piece.framework == "qai":
            if y is not None:
                try:
                    model.train(X, y, verbose=verbose)
                except TypeError:
                    model.train(X, verbose=verbose)
            else:
                # Unsupervised piece (scaler, PCA)
                try:
                    model.train(X, verbose=verbose)
                except Exception:
                    pass
        elif piece.framework == "pytorch":
            if y is not None:
                import torch
                import torch.nn as nn
                import torch.optim as optim
                if isinstance(model, nn.Module):
                    model.train()
                    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
                    y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
                    opt = optim.Adam(model.parameters(), lr=0.001)
                    crit = nn.CrossEntropyLoss()
                    for epoch in range(piece.params.get("epochs", 10)):
                        opt.zero_grad()
                        out = model(X_t)
                        loss = crit(out, y_t)
                        loss.backward()
                        opt.step()
                    if verbose:
                        print(f"  [pytorch {piece.name}] final loss: {loss.item():.4f}")
        elif piece.framework == "sklearn":
            if y is not None and hasattr(model, "fit"):
                model.fit(X, y)
            elif hasattr(model, "fit"):
                model.fit(X)
        else:  # custom
            if hasattr(model, "fit"):
                if y is not None:
                    model.fit(X, y)
                else:
                    model.fit(X)

    def summary(self) -> str:
        """Print the federated model architecture."""
        lines = [f"=== Federated Bridge: {self.name} ===", ""]
        for name in self._order:
            p = self.pieces[name]
            lines.append(f"  [{name}]")
            lines.append(f"    framework: {p.framework}")
            lines.append(f"    model: {type(p.model).__name__}")
            lines.append(f"    input_from: {p.input_from or '(root)'}")
            downstream = [n for n, pp in self.pieces.items() if pp.input_from == name]
            if downstream:
                lines.append(f"    feeds: {', '.join(downstream)}")
            lines.append("")
        return "\n".join(lines)

    def __repr__(self):
        return f"<FederatedBridge name={self.name!r} pieces={len(self.pieces)}>"


__all__ = ["FederatedBridge", "ModelPiece"]
