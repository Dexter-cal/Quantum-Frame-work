from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import Technique


class PyTorchMLP(nn.Module):
    """Configurable MLP. When `modern=True` it adds BatchNorm + GELU +
    Dropout + residual connections + LayerNorm + He initialisation,
    bringing it up to a 2024-era feed-forward baseline. When `modern=False`
    (the default, preserves backwards compatibility) it stays the original
    Linear -> ReLU -> Linear stack."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 task_type: str = "classification",
                 modern: bool = False,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        self.task_type = task_type
        self.modern = modern

        act_layer = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }[activation]

        if not modern:
            # Original 2-layer stack (backwards compatible)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.net = nn.Sequential(self.fc1, self.act, self.fc2)
            return

        # Modern stack: n_layers of (Linear -> BN -> act -> Dropout) with
        # residual connections + a final LayerNorm before the output head.
        layers = []
        prev = input_dim
        for _ in range(n_layers):
            block = []
            block.append(nn.Linear(prev, hidden_dim))
            block.append(nn.BatchNorm1d(hidden_dim))
            block.append(act_layer())
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            layers.extend(block)
            prev = hidden_dim
        # Residual projection (input -> hidden) so we can add x + h
        self.residual_proj = (nn.Linear(input_dim, hidden_dim, bias=False)
                              if input_dim != hidden_dim else nn.Identity())
        self.body = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        # He initialisation on every Linear
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        if not self.modern:
            return self.net(x)
        h = self.body(x)
        # Residual: project input to hidden dim if needed, then add
        h = h + self.residual_proj(x)
        h = self.norm(h)
        return self.head(h)


class NeuralNetwork(Technique):
    """Deep Learning Multi-Layer Perceptron technique powered by PyTorch.

    Builds ON TOP of the existing MLP architecture -- no new model type is
    introduced. New optional parameters activate modern training
    improvements on the same `neural_network` technique:

      modern=True          -- swap the 2-layer stack for a configurable
                              n-layer MLP with BatchNorm + GELU + Dropout
                              + residual connections + LayerNorm + He init.
      n_layers=4           -- depth of the modern stack (only used when
                              modern=True)
      dropout=0.1          -- dropout rate (modern=True)
      activation="gelu"    -- activation in the modern stack
                              ("relu" / "gelu" / "silu" / "tanh")
      weight_decay=1e-4    -- L2 regularisation strength on AdamW
      mixup_alpha=0.0      -- if > 0, apply mixup augmentation during
                              training (Zhang et al. 2018)
      label_smoothing=0.0  -- if > 0, smooth classification labels
                              (improves calibration)
      use_sam=False        -- if True, wrap the optimiser with Sharpness-
                              Aware Minimization (Foret et al. 2021)
      use_swa=False        -- if True, average weights from multiple
                              points along the trajectory (Izmailov 2018)
      cosine_schedule=False -- if True, attach a cosine-annealing LR
                              scheduler with warm restarts (SGDR)
      one_cycle=False      -- if True, attach a one-cycle super-convergence
                              schedule (Smith & Topin 2019)
    """
    name = "neural_network"
    family = "deep_learning"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, hidden_dim=16, epochs=50, lr=0.01, task_type="classification",
                 modern: bool = False,
                 n_layers: int = 2,
                 dropout: float = 0.0,
                 activation: str = "relu",
                 weight_decay: float = 0.0,
                 mixup_alpha: float = 0.0,
                 label_smoothing: float = 0.0,
                 use_sam: bool = False,
                 use_swa: bool = False,
                 cosine_schedule: bool = False,
                 one_cycle: bool = False,
                 **params):
        super().__init__(hidden_dim=hidden_dim, epochs=epochs, lr=lr, task_type=task_type, **params)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.task_type = task_type
        self.modern = modern
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation
        self.weight_decay = weight_decay
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.use_sam = use_sam
        self.use_swa = use_swa
        self.cosine_schedule = cosine_schedule
        self.one_cycle = one_cycle
        self.net = None
        self.optimizer = None
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
            if self.label_smoothing > 0:
                criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            output_dim = 1 if y.ndim == 1 else y.shape[1]
            y_tensor = torch.tensor(y, dtype=torch.float32)
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.unsqueeze(1)
            criterion = nn.MSELoss()

        input_dim = X.shape[1]
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Build the SAME MLP technique -- `modern` just adds BN/GELU/residuals.
        self.net = PyTorchMLP(
            input_dim, self.hidden_dim, output_dim, self.task_type,
            modern=self.modern,
            n_layers=self.n_layers,
            dropout=self.dropout,
            activation=self.activation,
        )
        # AdamW when weight_decay is requested, else plain Adam (back-compat).
        if self.weight_decay > 0:
            self.optimizer = optim.AdamW(self.net.parameters(),
                                         lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # Optional LR scheduler (build on top of the existing optimiser).
        if self.cosine_schedule and self.one_cycle:
            raise ValueError(
                "[WHAT] neural_network: cosine_schedule=True AND one_cycle=True\n"
                "[WHY] They are two different LR schedules and cannot both run\n"
                "[FIX] Pick one of them"
            )
        scheduler = None
        if self.cosine_schedule:
            from .training_enhancements import CosineAnnealer
            scheduler = CosineAnnealer(self.optimizer, T_max=max(self.epochs, 1))
        elif self.one_cycle:
            from .training_enhancements import OneCyclePolicy
            scheduler = OneCyclePolicy(self.optimizer, max_lr=self.lr,
                                       n_epochs=max(self.epochs, 1))

        # Optional mixup augmentation (data-space enhancement).
        mixup = None
        if self.mixup_alpha > 0:
            from .training_enhancements import MixupTrainer
            mixup = MixupTrainer(alpha=self.mixup_alpha)

        # Optional SAM wrapper (weight-space enhancement).
        sam = None
        if self.use_sam:
            from .training_enhancements import SAM
            sam = SAM(self.optimizer, self.net, rho=0.05)

        # Optional SWA (trajectory weight averaging).
        swa = None
        if self.use_swa:
            from .training_enhancements import SWA
            swa = SWA(self.net, swa_start=max(self.epochs // 4, 1),
                      swa_freq=max(self.epochs // 10, 1))

        self.net.train()
        self.loss_history_ = []

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            if mixup is not None and self.task_type == "classification":
                X_mix, (y_a, y_b, lam) = mixup.mix(X_tensor, y_tensor)
                outputs = self.net(X_mix)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = self.net(X_tensor)
                loss = criterion(outputs, y_tensor)

            if sam is not None:
                sam.first_step(loss)
                outputs = self.net(X_tensor)
                loss2 = criterion(outputs, y_tensor)
                sam.second_step(loss2)
            else:
                loss.backward()
                self.optimizer.step()

            if scheduler is not None:
                scheduler.step()
            if swa is not None:
                swa.update(epoch)

            self.loss_history_.append(float(loss.item()))

        if swa is not None:
            swa.apply()

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
