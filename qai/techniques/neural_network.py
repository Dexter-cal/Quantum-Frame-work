from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import Technique

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, task_type: str = "classification"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.task_type = task_type

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class NeuralNetwork(Technique):
    """Deep Learning Multi-Layer Perceptron technique powered by PyTorch."""
    name = "neural_network"
    family = "deep_learning"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, hidden_dim=16, epochs=50, lr=0.01, task_type="classification", **params):
        super().__init__(hidden_dim=hidden_dim, epochs=epochs, lr=lr, task_type=task_type, **params)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.task_type = task_type
        self.net = None
        self.classes_ = None
        self.loss_history_ = []
        self._last_X = None
        self._last_y = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if self.task_type == "classification":
            self.classes_, y_indices = np.unique(y, return_inverse=True)
            output_dim = len(self.classes_)
            y_tensor = torch.tensor(y_indices, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
        else:
            output_dim = 1 if y.ndim == 1 else y.shape[1]
            y_tensor = torch.tensor(y, dtype=torch.float32)
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.unsqueeze(1)
            criterion = nn.MSELoss()

        input_dim = X.shape[1]
        X_tensor = torch.tensor(X, dtype=torch.float32)

        self.net = PyTorchMLP(input_dim, self.hidden_dim, output_dim, self.task_type)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        self.net.train()
        self.loss_history_ = []

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.net(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            self.loss_history_.append(float(loss.item()))

        self._trained = True
        self._last_X, self._last_y = X, y
        return self

    def forward(self, x):
        if not self._trained or self.net is None:
            raise RuntimeError("Model must be trained before inference.")

        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            outputs = self.net(x_tensor)

            if self.task_type == "classification":
                preds = torch.argmax(outputs, dim=1).numpy()
                res = self.classes_[preds]
            else:
                res = outputs.squeeze(-1).numpy()

        return res[0] if single else res

    # --- unique tools ---
    def loss_history(self):
        return self.loss_history_

    def parameter_count(self):
        if self.net is None:
            return 0
        return sum(p.numel() for p in self.net.parameters())

    def accuracy(self) -> float:
        if self._last_X is None or self._last_y is None:
            return 0.0
        preds = self.forward(self._last_X)
        if self.task_type == "classification":
            return float(np.mean(preds == self._last_y))
        else:
            return float(1.0 - np.mean(np.abs(preds - self._last_y) / (np.abs(self._last_y) + 1e-8)))
