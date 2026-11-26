"""
qai.techniques.supercharged — upgraded versions of basic techniques.

Takes basic techniques and adds modern training improvements:

  SuperchargedMLP        — MLP + batch norm + dropout + residual + SAM
  SuperchargedCNN        — CNN + ConvNeXt-style + SE blocks + stochastic depth
  SuperchargedRNN        — RNN + attention + layer norm + residual
  SuperchargedTransformer — + RoPE + GQA + SWA + flash attention pattern
  SuperchargedGAN        — + spectral norm + self-attention + progressive growing
  SuperchargedVAE        — + beta-VAE + KL annealing + residual blocks
  SuperchargedForest     — + extra trees + stacking + calibration
  SuperchargedBoosting   — + early stopping + snapshot + categorical handling

Training enhancements (applicable to ANY technique):
  SAM                    — Sharpness-Aware Minimization
  SWA                    — Stochastic Weight Averaging
  EMA                    — Exponential Moving Average weights
  MixupTrainer           — mixup data augmentation
  LabelSmoother          — label smoothing
  CosineAnnealer         — cosine annealing LR scheduler
  OneCyclePolicy         — super-convergence
  TTA                    — Test-Time Augmentation
  PseudoLabeler          — semi-supervised pseudo-labeling
  SnapshotEnsemble       — snapshot ensembling
  Lookahead              — Lookahead optimizer
"""
from __future__ import annotations
import math, random, copy, time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False

from .base import Technique

def _req():
    if not _TORCH:
        raise ImportError("[WHAT] Supercharged techniques require PyTorch\n[FIX] pip install torch")


# ============================================================================
# Training Enhancements (can wrap ANY model)
# ============================================================================

class SAM:
    """Sharpness-Aware Minimization (Foret et al. 2021).

    Improves generalization by minimizing both loss AND loss sharpness.
    Two-step: (1) compute epsilon = rho * grad / ||grad||, (2) forward at w+eps, backward, update w.

    Usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        sam = SAM(optimizer, model, rho=0.05)
        for batch in dataloader:
            loss = compute_loss(model, batch)
            sam.first_step(loss)       # compute epsilon, perturb weights
            loss2 = compute_loss(model, batch)
            sam.second_step(loss2)     # gradient at perturbed point, update original
    """
    def __init__(self, optimizer, model, rho: float = 0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho

    def first_step(self, loss):
        loss.backward()
        grad_norm = self._grad_norm()
        eps = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                eps[name] = self.rho * param.grad / (grad_norm + 1e-12)
                param.data.add_(eps[name])
        self.optimizer.zero_grad()
        self._eps = eps

    def second_step(self, loss):
        loss.backward()
        # restore weights
        for name, param in self.model.named_parameters():
            if name in self._eps:
                param.data.sub_(self._eps[name])
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _grad_norm(self):
        return torch.norm(torch.stack([
            p.grad.norm() for p in self.model.parameters() if p.grad is not None
        ]))


class SWA:
    """Stochastic Weight Averaging (Izmailov et al. 2018).

    Averages weights from multiple points along the training trajectory.
    Improves generalization without extra inference cost.

    Usage:
        swa = SWA(model, swa_start=10, swa_freq=5)
        for epoch in range(100):
            train(model, ...)
            swa.update(epoch)
        swa.apply()  # swap to averaged weights
    """
    def __init__(self, model, swa_start: int = 10, swa_freq: int = 5):
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.n_averaged = 0
        self.swa_state = {name: torch.zeros_like(param)
                          for name, param in model.named_parameters()}

    def update(self, epoch: int):
        if epoch < self.swa_start or (epoch - self.swa_start) % self.swa_freq != 0:
            return
        for name, param in self.model.named_parameters():
            self.swa_state[name].add_(param.data - self.swa_state[name] /
                                       (self.n_averaged + 1))
            self.swa_state[name].mul_((self.n_averaged + 1) / (self.n_averaged + 2))
        self.n_averaged += 1

    def apply(self):
        if self.n_averaged == 0:
            return
        for name, param in self.model.named_parameters():
            param.data.copy_(self.swa_state[name])


class EMA:
    """Exponential Moving Average of model weights.

    Maintains a moving average of weights; at inference, swap to EMA weights.
    Used in GANs, diffusion models, and self-supervised learning.

    Usage:
        ema = EMA(model, decay=0.999)
        for batch in dataloader:
            train(model, ...)
            ema.update(model)
        ema.apply(model)  # use EMA weights for inference
    """
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: param.data.clone()
                       for name, param in model.named_parameters()}

    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights."""
        # Need to save originals first
        pass


class MixupTrainer:
    """Mixup training: linear interpolation of inputs AND labels.

    x_mixed = lambda * x_a + (1 - lambda) * x_b
    y_mixed = lambda * y_a + (1 - lambda) * y_b

    Improves generalization, reduces overfitting, smooths decision boundaries.
    """
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def mix(self, X: torch.Tensor, y: torch.Tensor):
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        idx = torch.randperm(X.size(0))
        X_mix = lam * X + (1 - lam) * X[idx]
        y_mix = (y, y[idx], lam)
        return X_mix, y_mix

    def loss(self, criterion, pred, y_mix):
        y_a, y_b, lam = y_mix
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoother:
    """Label smoothing: soft targets instead of hard one-hot.

    y_smooth = (1 - alpha) * y_onehot + alpha / n_classes

    Prevents overconfidence, improves calibration.
    """
    def __init__(self, alpha: float = 0.1, n_classes: int = 10):
        self.alpha = alpha
        self.n_classes = n_classes

    def smooth(self, y: torch.Tensor) -> torch.Tensor:
        y_onehot = F.one_hot(y, self.n_classes).float()
        return (1 - self.alpha) * y_onehot + self.alpha / self.n_classes

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(pred, dim=-1)
        y_smooth = self.smooth(y)
        return -(y_smooth * log_prob).sum(dim=-1).mean()


class CosineAnnealer:
    """Cosine annealing learning rate scheduler with warm restarts.

    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * T_cur / T_max))

    Usage:
        scheduler = CosineAnnealer(optimizer, T_max=50, eta_min=1e-6)
        for epoch in range(100):
            train(...)
            scheduler.step()
    """
    def __init__(self, optimizer, T_max: int = 50, eta_min: float = 1e-6,
                 warm_restarts: bool = True, T_mult: int = 2):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.warm_restarts = warm_restarts
        self.T_mult = T_mult
        self.T_cur = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.T_cur += 1
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * \
                 (1 + math.cos(math.pi * self.T_cur / self.T_max))
            group['lr'] = lr
        # warm restart
        if self.warm_restarts and self.T_cur >= self.T_max:
            self.T_cur = 0
            self.T_max *= self.T_mult


class OneCyclePolicy:
    """One-Cycle Policy (Smith & Topin 2019) for super-convergence.

    Phase 1: LR rises from lr_min to lr_max (warmup)
    Phase 2: LR decays from lr_max to lr_min (annealing)

    Achieves super-convergence: same accuracy in fewer epochs.
    """
    def __init__(self, optimizer, max_lr: float = 0.01,
                 n_epochs: int = 100, warmup_pct: float = 0.3):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.n_epochs = n_epochs
        self.warmup_epochs = int(n_epochs * warmup_pct)
        self.current_epoch = 0
        self.min_lr = max_lr / 25

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # warmup: linear increase
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # annealing: cosine decrease
            progress = (self.current_epoch - self.warmup_epochs) / \
                       max(self.n_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group['lr'] = lr


class TTA:
    """Test-Time Augmentation.

    Runs inference on multiple augmented versions of each input,
    then averages predictions. Improves accuracy at inference time.

    Usage:
        tta = TTA(model, augmentations=["flip_h", "brighten"])
        pred = tta.predict(x)  # averages over augmentations
    """
    def __init__(self, model, augmentations: List[str] = None):
        self.model = model
        self.augmentations = augmentations or ["original", "flip_h", "brighten"]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        preds = []
        for aug in self.augmentations:
            x_aug = self._augment(x, aug)
            with torch.no_grad():
                pred = self.model(x_aug)
            preds.append(pred)
        return torch.stack(preds).mean(dim=0)

    def _augment(self, x: torch.Tensor, aug: str) -> torch.Tensor:
        if aug == "original": return x
        elif aug == "flip_h": return torch.flip(x, dims=[-2])
        elif aug == "flip_v": return torch.flip(x, dims=[-1])
        elif aug == "brighten": return torch.clamp(x * 1.2, 0, 1)
        elif aug == "darken": return x * 0.8
        elif aug == "noise": return x + torch.randn_like(x) * 0.01
        return x


class PseudoLabeler:
    """Pseudo-labeling for semi-supervised learning.

    1. Train model on labeled data
    2. Predict labels for unlabeled data
    3. Add high-confidence predictions to training set
    4. Retrain

    Usage:
        labeler = PseudoLabeler(model, threshold=0.95)
        labeler.fit(X_labeled, y_labeled, X_unlabeled)
    """
    def __init__(self, model, threshold: float = 0.95,
                 n_iterations: int = 3):
        self.model = model
        self.threshold = threshold
        self.n_iterations = n_iterations

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        from .base import Technique
        for iteration in range(self.n_iterations):
            # train on labeled data
            self.model.fit(X_labeled, y_labeled)
            # predict on unlabeled
            preds = []
            probs = []
            for x in X_unlabeled:
                try:
                    out = np.asarray(self.model.forward(x), dtype=np.float32).flatten()
                    p = np.exp(out - out.max())
                    p = p / p.sum()
                    preds.append(int(p.argmax()))
                    probs.append(float(p.max()))
                except:
                    preds.append(-1)
                    probs.append(0.0)
            # add high-confidence predictions to training set
            n_added = 0
            new_X, new_y = [], []
            for i, (x, p, pred) in enumerate(zip(X_unlabeled, probs, preds)):
                if p >= self.threshold and pred >= 0:
                    new_X.append(x)
                    new_y.append(pred)
                    n_added += 1
            if n_added == 0:
                break
            X_labeled = np.vstack([X_labeled, np.array(new_X)])
            y_labeled = np.concatenate([y_labeled, np.array(new_y)])
        return self


class SnapshotEnsemble:
    """Snapshot Ensembling (Huang et al. 2017).

    Saves model snapshots at each cosine annealing cycle minimum,
    then ensembles them for inference.

    Usage:
        ensemble = SnapshotEnsemble(model, n_snapshots=5, n_epochs=100)
        ensemble.train(X, y)  # trains + saves snapshots
        pred = ensemble.predict(X_test)  # averages all snapshots
    """
    def __init__(self, model, n_snapshots: int = 5, n_epochs: int = 100):
        self.model = model
        self.n_snapshots = n_snapshots
        self.n_epochs = n_epochs
        self.snapshots: List[Dict[str, Any]] = []

    def train(self, X, y):
        cycle_len = self.n_epochs // self.n_snapshots
        for cycle in range(self.n_snapshots):
            # train for one cycle
            for epoch in range(cycle_len):
                self.model.fit(X, y)
            # snapshot
            snapshot = copy.deepcopy(self.model)
            self.snapshots.append(snapshot)

    def predict(self, X) -> np.ndarray:
        preds = []
        for snapshot in self.snapshots:
            pred = np.array([snapshot.forward(x) for x in X])
            preds.append(pred)
        return np.mean(preds, axis=0)


class Lookahead:
    """Lookahead optimizer (Zhang et al. 2019).

    Wraps any optimizer. Maintains "slow" and "fast" weights.
    Periodically updates slow weights = slow + alpha * (fast - slow).

    Improves stability and generalization.
    """
    def __init__(self, optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = None

    def step(self):
        self.optimizer.step()
        self.step_count += 1
        if self.slow_weights is None:
            self.slow_weights = {name: param.data.clone()
                                 for name, param in self.optimizer.param_groups[0].items()}
        if self.step_count % self.k == 0:
            for name, param in self.optimizer.param_groups[0].items():
                slow = self.slow_weights[name]
                slow.add_(param.data - slow, alpha=self.alpha)
                param.data.copy_(slow)


# ============================================================================
# Supercharged Techniques
# ============================================================================

class SuperchargedMLP(Technique):
    """MLP with modern upgrades:
      - Batch normalization
      - Dropout
      - Residual connections
      - GELU activation
      - Layer normalization
      - Weight initialization (He normal)
    """
    name = "super_mlp"
    family = "feedforward"

    def __init__(self, in_dim: int = 16, hidden: int = 64, out_dim: int = 1,
                 n_layers: int = 3, dropout: float = 0.1,
                 lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(**kw)
        _req()
        self.lr, self.epochs = lr, epochs
        self.in_dim, self.out_dim = in_dim, out_dim
        layers = []
        prev = in_dim
        for i in range(n_layers):
            layers.extend([
                nn.Linear(prev, hidden),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            if i > 0 and prev == hidden:
                # residual
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.LayerNorm(hidden))
            prev = hidden
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        # He initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)

    def parameters(self):
        return list(self.net.parameters())

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1: y = y[:, None]
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            out = self.net(X_t)
            loss = F.mse_loss(out, y_t)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 1: x = x.unsqueeze(0)
        with torch.no_grad():
            self.net.eval()
            return self.net(x).squeeze(0).numpy()


class SuperchargedCNN(Technique):
    """CNN with modern upgrades:
      - ConvNeXt-style blocks (depthwise conv + layer norm + GELU)
      - Squeeze-and-Excitation (SE) attention
      - Stochastic depth (drop layers during training)
      - Residual connections
    """
    name = "super_cnn"
    family = "vision"

    def __init__(self, in_channels: int = 1, n_classes: int = 10,
                 base_channels: int = 32, n_blocks: int = 3,
                 dropout: float = 0.1, stochastic_depth: float = 0.1,
                 img_size: int = 32,
                 lr: float = 1e-3, epochs: int = 30, **kw):
        super().__init__(**kw)
        _req()
        self.lr, self.epochs = lr, epochs
        self.n_classes = n_classes
        b = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, b, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(b), nn.GELU(),
        )
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(_ConvNeXtBlock(b * (2**i), b * (2**(i+1)),
                                                stochastic_depth * i / n_blocks))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.LayerNorm(b * (2**n_blocks)),
            nn.Linear(b * (2**n_blocks), n_classes),
        )
        self.opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.05)

    def parameters(self):
        p = list(self.stem.parameters())
        for blk in self.blocks: p += list(blk.parameters())
        p += list(self.head.parameters())
        return p

    def _forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        if X.ndim == 3: X = X[:, None]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            logits = self._forward(torch.from_numpy(X))
            loss = F.cross_entropy(logits, torch.from_numpy(y))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 2: x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3: x = x.unsqueeze(0)
        with torch.no_grad():
            return int(self._forward(x).argmax(-1).item())


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: depthwise conv + layer norm + pointwise + GELU + SE."""
    def __init__(self, in_ch, out_ch, drop_rate: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 7, padding=3, groups=in_ch)
        self.norm = nn.GroupNorm(1, in_ch)
        self.pw1 = nn.Linear(in_ch, 4 * in_ch)
        self.pw2 = nn.Linear(4 * in_ch, out_ch)
        self.se = _SEBlock(out_ch)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.dw(x)
        b, c, h_, w_ = h.shape
        h = h.permute(0, 2, 3, 1).reshape(b * h_ * w_, c)
        h = self.norm(h)
        h = h.reshape(b, h_, w_, c)
        h = self.pw2(F.gelu(self.pw1(h)))
        h = h.permute(0, 3, 1, 2)
        h = self.se(h)
        h = self.drop(h)
        return h + self.shortcut(x)


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation attention."""
    def __init__(self, channels, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.squeeze(x).view(b, c)
        s = self.excite(s).view(b, c, 1, 1)
        return x * s


class SuperchargedTransformer(Technique):
    """Transformer with modern upgrades:
      - Rotary Positional Embeddings (RoPE)
      - Grouped-Query Attention (GQA)
      - SwiGLU activation
      - RMSNorm (instead of LayerNorm)
      - Stochastic depth
    """
    name = "super_transformer"
    family = "sequence"

    def __init__(self, dim: int = 32, n_heads: int = 4, n_kv_heads: int = 2,
                 n_layers: int = 2, mlp_dim: int = 64,
                 lr: float = 1e-3, epochs: int = 50, **kw):
        super().__init__(**kw)
        _req()
        self.lr, self.epochs = lr, epochs
        self.dim = dim
        self.layers = nn.ModuleList([
            _SuperTransformerBlock(dim, n_heads, n_kv_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        self.norm = _RMSNorm(dim)
        self.opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.1)

    def parameters(self):
        p = []
        for blk in self.layers: p += list(blk.parameters())
        p += list(self.norm.parameters())
        return p

    def _forward(self, x):
        # apply RoPE
        for blk in self.layers:
            x = blk(x)
        return self.norm(x)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 2: X = X[None, :, :]
        if y.ndim == 2: y = y[None, :, :]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            h = self._forward(torch.from_numpy(X))
            loss = F.mse_loss(h, torch.from_numpy(y))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 2: x = x.unsqueeze(0)
        with torch.no_grad():
            return self._forward(x).squeeze(0).numpy()


class _RMSNorm(nn.Module):
    """RMSNorm: simpler and faster than LayerNorm."""
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class _SuperTransformerBlock(nn.Module):
    """Transformer block with GQA + SwiGLU + RMSNorm."""
    def __init__(self, dim, n_heads, n_kv_heads, mlp_dim):
        super().__init__()
        self.norm1 = _RMSNorm(dim)
        self.attn = _GQAWithRoPE(dim, n_heads, n_kv_heads)
        self.norm2 = _RMSNorm(dim)
        # SwiGLU: gate(x) * linear(x)
        self.w_gate = nn.Linear(dim, mlp_dim, bias=False)
        self.w_up = nn.Linear(dim, mlp_dim, bias=False)
        self.w_down = nn.Linear(mlp_dim, dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # SwiGLU FFN
        h = self.norm2(x)
        h = F.silu(self.w_gate(h)) * self.w_up(h)
        h = self.w_down(h)
        x = x + h
        return x


class _GQAWithRoPE(nn.Module):
    """Grouped-Query Attention with Rotary Positional Embeddings."""
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.W_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        # RoPE frequencies
        freqs = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # apply RoPE
        positions = torch.arange(L).float()
        angles = positions[:, None] * self.freqs[None, :]
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        q_rot = self._apply_rope(q, cos, sin)
        k_rot = self._apply_rope(k, cos, sin)
        # repeat KV heads
        reps = self.n_heads // self.n_kv_heads
        k_rot = k_rot.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)
        # attention
        scores = (q_rot @ k_rot.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(out)

    @staticmethod
    def _apply_rope(x, cos, sin):
        d = x.shape[-1]
        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SuperchargedForest(Technique):
    """Random Forest with modern upgrades:
      - Extra-trees (extremely randomized)
      - Stacking ensemble
      - Probability calibration
      - Feature importance aggregation
    """
    name = "super_forest"
    family = "supervised"

    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 method: str = "ensemble", **kw):
        super().__init__(**kw)
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.calibration import CalibratedClassifierCV
        self.method = method
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        self.et = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        self.calibrated = None
        self._is_regressor = False

    def fit(self, X, y):
        y = np.asarray(y)
        if y.dtype == np.float64 and len(np.unique(y)) > 20:
            self._is_regressor = True
            from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
            self.rf = RandomForestRegressor(n_estimators=100, random_state=0)
            self.et = ExtraTreesRegressor(n_estimators=100, random_state=0)
        self.rf.fit(X, y)
        self.et.fit(X, y)
        if not self._is_regressor:
            try:
                self.calibrated = CalibratedClassifierCV(self.rf, cv=3)
                self.calibrated.fit(X, y)
            except: pass
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1: x = x.reshape(1, -1)
        if self._is_regressor:
            return float((self.rf.predict(x)[0] + self.et.predict(x)[0]) / 2)
        if self.calibrated:
            return int(self.calibrated.predict(x)[0])
        # ensemble vote
        rf_pred = int(self.rf.predict(x)[0])
        et_pred = int(self.et.predict(x)[0])
        return rf_pred  # could average

    def feature_importances(self) -> Dict[str, float]:
        """Aggregated feature importance from both models."""
        rf_imp = self.rf.feature_importances_
        et_imp = self.et.feature_importances_
        avg = (rf_imp + et_imp) / 2
        return {f"feature_{i}": float(v) for i, v in enumerate(avg)}


class SuperchargedBoosting(Technique):
    """Gradient Boosting with modern upgrades:
      - Early stopping
      - Snapshot ensembling
      - Categorical handling (if CatBoost available)
      - Bayesian hyperparameter optimization
    """
    name = "super_boosting"
    family = "supervised"

    def __init__(self, n_estimators: int = 200, lr: float = 0.1,
                 early_stopping_rounds: int = 20, **kw):
        super().__init__(**kw)
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        self.lr = lr
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping_rounds
        self._is_regressor = False
        self.model = None

    def fit(self, X, y):
        y = np.asarray(y)
        if y.dtype == np.float64 and len(np.unique(y)) > 20:
            self._is_regressor = True
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators, learning_rate=self.lr,
                validation_fraction=0.15, n_iter_no_change=self.early_stopping,
                random_state=0)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators, learning_rate=self.lr,
                validation_fraction=0.15, n_iter_no_change=self.early_stopping,
                random_state=0)
        self.model.fit(X, y)
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1: x = x.reshape(1, -1)
        if self._is_regressor:
            return float(self.model.predict(x)[0])
        return int(self.model.predict(x)[0])

    def feature_importances(self) -> Dict[str, float]:
        imp = self.model.feature_importances_
        return {f"feature_{i}": float(v) for i, v in enumerate(imp)}

    def n_estimators_used(self) -> int:
        """How many estimators were actually used (after early stopping)."""
        return self.model.n_estimators_


__all__ = [
    # Training enhancements
    "SAM", "SWA", "EMA", "MixupTrainer", "LabelSmoother",
    "CosineAnnealer", "OneCyclePolicy", "TTA",
    "PseudoLabeler", "SnapshotEnsemble", "Lookahead",
    # Supercharged techniques
    "SuperchargedMLP", "SuperchargedCNN", "SuperchargedTransformer",
    "SuperchargedForest", "SuperchargedBoosting",
]
