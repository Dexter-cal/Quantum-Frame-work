"""
qai.techniques.training_enhancements -- universal training enhancements
that can be applied ON TOP of any existing qai technique.

These are NOT new model types. They are wrappers / mixins / utilities you
combine with the existing techniques (CNN, NeuralNetwork, Transformer,
RandomForest, GradientBoosting, ...) to add modern training improvements:

    SAM                 Sharpness-Aware Minimization (Foret et al. 2021)
    SWA                 Stochastic Weight Averaging (Izmailov et al. 2018)
    EMA                 Exponential Moving Average of weights
    MixupTrainer        mixup data augmentation (Zhang et al. 2018)
    LabelSmoother       label smoothing for better calibration
    CosineAnnealer      cosine annealing LR scheduler with warm restarts
    OneCyclePolicy      super-convergence one-cycle policy (Smith & Topin 2019)
    TTA                 Test-Time Augmentation
    PseudoLabeler       semi-supervised pseudo-labeling
    SnapshotEnsemble    snapshot ensembling (Huang et al. 2017)
    Lookahead           Lookahead optimizer (Zhang et al. 2019)

Usage pattern (build ON TOP of an existing technique, do not duplicate it):

    import qai
    from qai.techniques.training_enhancements import SAM, MixupTrainer, CosineAnnealer

    model = qai.build(technique="cnn", n_classes=10, epochs=50)
    # wrap the underlying torch optimiser with SAM, attach mixup + cosine LR
    ...

These helpers are also exposed at the top level for convenience:

    import qai
    qai.SAM, qai.SWA, qai.EMA, qai.MixupTrainer, qai.LabelSmoother,
    qai.CosineAnnealer, qai.OneCyclePolicy, qai.TTA,
    qai.PseudoLabeler, qai.SnapshotEnsemble, qai.Lookahead
"""
from __future__ import annotations
import math, copy
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


def _req_torch():
    if not _TORCH:
        raise ImportError(
            "[WHAT] This training enhancement requires PyTorch\n"
            "[WHY] It wraps torch.nn.Module / torch.optim.Optimizer internals\n"
            "[FIX] pip install torch   (or: pip install -r requirements.txt)"
        )


# ============================================================================
# Weight-space enhancements
# ============================================================================

class SAM:
    """Sharpness-Aware Minimization (Foret et al. 2021).

    Wraps any torch.optim.Optimizer. Improves generalization by minimising
    BOTH the loss AND the loss sharpness. Two-step per iteration:

      1. ``first_step(loss)``  -- backward, compute epsilon = rho * g/||g||,
         perturb weights by +epsilon.
      2. ``second_step(loss2)`` -- backward at the perturbed point, restore
         weights, then call optimizer.step() with the new gradient.

    Example (on top of an existing qai CNN):

        import qai
        from qai.techniques.training_enhancements import SAM

        model = qai.build(technique="cnn", n_classes=10)
        # ... train once so the underlying net + optimiser exist ...
        sam = SAM(model.technique.optimizer, model.technique.net, rho=0.05)
        for batch in dataloader:
            loss = criterion(model.net(x), y)
            sam.first_step(loss)
            loss2 = criterion(model.net(x), y)
            sam.second_step(loss2)
    """
    def __init__(self, optimizer, model, rho: float = 0.05):
        _req_torch()
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self._eps: Dict[str, torch.Tensor] = {}

    def first_step(self, loss):
        loss.backward()
        grad_norm = self._grad_norm()
        self._eps = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                eps = self.rho * param.grad / (grad_norm + 1e-12)
                param.data.add_(eps)
                self._eps[name] = eps
        self.optimizer.zero_grad()

    def second_step(self, loss):
        loss.backward()
        for name, param in self.model.named_parameters():
            if name in self._eps:
                param.data.sub_(self._eps[name])
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        return torch.norm(torch.stack([
            p.grad.norm() for p in self.model.parameters() if p.grad is not None
        ]))


class SWA:
    """Stochastic Weight Averaging (Izmailov et al. 2018).

    Averages weights from multiple points along the training trajectory
    (after `swa_start`, every `swa_freq` epochs). Improves generalisation
    without extra inference cost -- at the end of training call ``apply()``
    to swap the model's weights for the averaged weights.

    Example:

        swa = SWA(model.net, swa_start=10, swa_freq=5)
        for epoch in range(n_epochs):
            train_one_epoch(...)
            swa.update(epoch)
        swa.apply()  # swap in averaged weights for evaluation
    """
    def __init__(self, model, swa_start: int = 10, swa_freq: int = 5):
        _req_torch()
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

    Maintains a slow-moving shadow copy of the weights; at inference time
    you can swap to the EMA weights via ``apply()``. Used in GANs, diffusion
    models, and self-supervised learning to stabilise the final model.

        ema = EMA(model.net, decay=0.999)
        for batch in dataloader:
            train_one_step(...)
            ema.update(model.net)
        ema.apply(model.net)  # use EMA weights for inference
    """
    def __init__(self, model, decay: float = 0.999):
        _req_torch()
        self.decay = decay
        self.shadow = {name: param.data.clone()
                       for name, param in model.named_parameters()}
        self._original = {}

    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        self._original = {name: param.data.clone()
                          for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore the original (non-EMA) weights after `apply()`."""
        for name, param in model.named_parameters():
            if name in self._original:
                param.data.copy_(self._original[name])


class Lookahead:
    """Lookahead optimizer (Zhang et al. 2019).

    Wraps any torch.optim.Optimizer. Maintains "slow" and "fast" weights.
    Every ``k`` steps, slow weights are updated as
    ``slow = slow + alpha * (fast - slow)`` and the fast weights are reset
    to the slow weights. Improves stability and generalisation.

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        lookahead = Lookahead(opt, k=5, alpha=0.5)
        for batch in dataloader:
            loss.backward()
            lookahead.step()
            lookahead.zero_grad()
    """
    def __init__(self, optimizer, k: int = 5, alpha: float = 0.5):
        _req_torch()
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights: Dict[int, torch.Tensor] = {}

    def step(self):
        self.optimizer.step()
        self.step_count += 1
        if not self.slow_weights:
            self.slow_weights = {id(p): p.data.clone()
                                 for group in self.optimizer.param_groups
                                 for p in group['params']}
        if self.step_count % self.k == 0:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    slow = self.slow_weights[id(p)]
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)

    def zero_grad(self):
        self.optimizer.zero_grad()


# ============================================================================
# Data-space enhancements
# ============================================================================

class MixupTrainer:
    """Mixup training (Zhang et al. 2018).

    Linear interpolation of inputs AND labels:

        x_mixed = lambda * x_a + (1 - lambda) * x_b
        y_mixed = lambda * y_a + (1 - lambda) * y_b

    Improves generalisation, reduces overfitting, smooths decision
    boundaries. Works on top of any torch-based qai technique
    (CNN, NeuralNetwork, Transformer, ...).

        mixup = MixupTrainer(alpha=0.2)
        for x, y in dataloader:
            x_mix, (y_a, y_b, lam) = mixup.mix(x, y)
            pred = model.net(x_mix)
            loss = mixup.loss(criterion, pred, (y_a, y_b, lam))
    """
    def __init__(self, alpha: float = 0.2):
        _req_torch()
        self.alpha = alpha

    def mix(self, X: "torch.Tensor", y: "torch.Tensor"):
        lam = float(np.random.beta(self.alpha, self.alpha)) if self.alpha > 0 else 1.0
        idx = torch.randperm(X.size(0))
        X_mix = lam * X + (1 - lam) * X[idx]
        return X_mix, (y, y[idx], lam)

    def loss(self, criterion, pred, y_mix):
        y_a, y_b, lam = y_mix
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoother:
    """Label smoothing.

        y_smooth = (1 - alpha) * y_onehot + alpha / n_classes

    Prevents overconfidence and improves calibration. Can be used as a
    drop-in replacement for cross-entropy on top of any classification
    technique.

        smoother = LabelSmoother(alpha=0.1, n_classes=10)
        loss = smoother.loss(logits, y)
    """
    def __init__(self, alpha: float = 0.1, n_classes: int = 10):
        _req_torch()
        self.alpha = alpha
        self.n_classes = n_classes

    def smooth(self, y: "torch.Tensor") -> "torch.Tensor":
        y_onehot = F.one_hot(y, self.n_classes).float()
        return (1 - self.alpha) * y_onehot + self.alpha / self.n_classes

    def loss(self, pred: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        log_prob = F.log_softmax(pred, dim=-1)
        y_smooth = self.smooth(y)
        return -(y_smooth * log_prob).sum(dim=-1).mean()


# ============================================================================
# LR-schedule enhancements
# ============================================================================

class CosineAnnealer:
    """Cosine annealing LR scheduler with optional warm restarts (SGDR).

        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * T_cur / T_max))

        sched = CosineAnnealer(optimizer, T_max=50, eta_min=1e-6)
        for epoch in range(n_epochs):
            train_one_epoch(...)
            sched.step()
    """
    def __init__(self, optimizer, T_max: int = 50, eta_min: float = 1e-6,
                 warm_restarts: bool = True, T_mult: int = 2):
        _req_torch()
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
        if self.warm_restarts and self.T_cur >= self.T_max:
            self.T_cur = 0
            self.T_max *= self.T_mult


class OneCyclePolicy:
    """One-Cycle Policy (Smith & Topin 2019) for super-convergence.

    Phase 1 (warmup): LR rises linearly from lr_min to lr_max.
    Phase 2 (annealing): LR decays cosinely from lr_max back to lr_min.

    Achieves super-convergence: same accuracy in fewer epochs compared to
    a fixed LR.
    """
    def __init__(self, optimizer, max_lr: float = 0.01,
                 n_epochs: int = 100, warmup_pct: float = 0.3):
        _req_torch()
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.n_epochs = n_epochs
        self.warmup_epochs = max(int(n_epochs * warmup_pct), 1)
        self.current_epoch = 0
        self.min_lr = max_lr / 25

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / \
                       max(self.n_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group['lr'] = lr


# ============================================================================
# Inference-time enhancements
# ============================================================================

class TTA:
    """Test-Time Augmentation.

    Runs inference on multiple augmented versions of each input and
    averages the predictions. Improves accuracy at inference time at
    the cost of more compute per prediction.

        tta = TTA(model.net, augmentations=["original", "flip_h", "brighten"])
        pred = tta.predict(x)  # averaged over augmentations
    """
    def __init__(self, model, augmentations: Optional[List[str]] = None):
        _req_torch()
        self.model = model
        self.augmentations = augmentations or ["original", "flip_h", "brighten"]

    def predict(self, x: "torch.Tensor") -> "torch.Tensor":
        preds = []
        for aug in self.augmentations:
            x_aug = self._augment(x, aug)
            with torch.no_grad():
                pred = self.model(x_aug)
            preds.append(pred)
        return torch.stack(preds).mean(dim=0)

    def _augment(self, x: "torch.Tensor", aug: str) -> "torch.Tensor":
        if aug == "original":
            return x
        if aug == "flip_h":
            return torch.flip(x, dims=[-2])
        if aug == "flip_v":
            return torch.flip(x, dims=[-1])
        if aug == "brighten":
            return torch.clamp(x * 1.2, 0, 1)
        if aug == "darken":
            return x * 0.8
        if aug == "noise":
            return x + torch.randn_like(x) * 0.01
        return x


# ============================================================================
# Semi-supervised + ensembling enhancements
# ============================================================================

class PseudoLabeler:
    """Pseudo-labeling for semi-supervised learning.

    1. Train model on labelled data.
    2. Predict labels for unlabelled data.
    3. Add high-confidence predictions to the training set.
    4. Retrain. Repeat.

    Works on top of ANY qai technique that supports `fit` + `forward`
    (i.e. any classifier). Builds on the existing technique -- does NOT
    create a new model type.

        labeler = PseudoLabeler(model, threshold=0.95, n_iterations=3)
        labeler.fit(X_labeled, y_labeled, X_unlabeled)
    """
    def __init__(self, model, threshold: float = 0.95, n_iterations: int = 3):
        self.model = model
        self.threshold = threshold
        self.n_iterations = n_iterations

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        X_labeled = np.asarray(X_labeled, dtype=np.float32)
        y_labeled = np.asarray(y_labeled)
        X_unlabeled = np.asarray(X_unlabeled, dtype=np.float32)
        for _ in range(self.n_iterations):
            self.model.fit(X_labeled, y_labeled)
            preds, probs = [], []
            for x in X_unlabeled:
                try:
                    out = np.asarray(self.model.forward(x), dtype=np.float32).flatten()
                    p = np.exp(out - out.max())
                    p = p / p.sum()
                    preds.append(int(p.argmax()))
                    probs.append(float(p.max()))
                except Exception:
                    preds.append(-1)
                    probs.append(0.0)
            new_X, new_y = [], []
            for x, p, pred in zip(X_unlabeled, probs, preds):
                if p >= self.threshold and pred >= 0:
                    new_X.append(x)
                    new_y.append(pred)
            if not new_X:
                break
            X_labeled = np.vstack([X_labeled, np.array(new_X)])
            y_labeled = np.concatenate([y_labeled, np.array(new_y)])
        return self


class SnapshotEnsemble:
    """Snapshot Ensembling (Huang et al. 2017).

    Saves model snapshots at each cosine-annealing cycle minimum, then
    ensembles them for inference. Builds on top of an existing qai
    technique -- does NOT create a new model type.

        ens = SnapshotEnsemble(model, n_snapshots=5, n_epochs=100)
        ens.train(X, y)
        pred = ens.predict(X_test)  # averages all snapshots
    """
    def __init__(self, model, n_snapshots: int = 5, n_epochs: int = 100):
        self.model = model
        self.n_snapshots = n_snapshots
        self.n_epochs = n_epochs
        self.snapshots: List[Any] = []

    def train(self, X, y):
        cycle_len = max(self.n_epochs // self.n_snapshots, 1)
        for _ in range(self.n_snapshots):
            for _ in range(cycle_len):
                self.model.fit(X, y)
            self.snapshots.append(copy.deepcopy(self.model))
        return self

    def predict(self, X) -> np.ndarray:
        preds = []
        for snap in self.snapshots:
            preds.append(np.array([snap.forward(x) for x in X]))
        return np.mean(preds, axis=0)


__all__ = [
    "SAM", "SWA", "EMA", "Lookahead",
    "MixupTrainer", "LabelSmoother",
    "CosineAnnealer", "OneCyclePolicy",
    "TTA",
    "PseudoLabeler", "SnapshotEnsemble",
]
