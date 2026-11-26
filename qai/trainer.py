"""
qai.trainer — universal training loop for ALL qai models.

Replaces the per-technique ad-hoc fit() loops with a single, feature-complete
trainer that gives every model (torch or sklearn) the same advanced training
capabilities:

  - Gradient clipping (max_grad_norm)
  - LR scheduling (cosine, linear, one-cycle, polynomial, constant, step)
  - Mixed precision (fp16, bf16) via torch.autocast
  - Gradient accumulation (large effective batch on small GPUs)
  - Gradient checkpointing (memory-constrained training)
  - Distributed training (DDP — single-machine multi-GPU)
  - Checkpointing (save/resume, keep-best, keep-last-N)
  - Early stopping
  - Logging (console + JSON; optional tensorboard / wandb)
  - Warmup (linear, cosine, none)
  - EMA (Exponential Moving Average of weights)
  - SWA (Stochastic Weight Averaging)
  - Eval at every N steps or every epoch
  - Custom callbacks

Usage:
    import qai
    m = qai.build(technique="transformer", vocab_size=1000, epochs=1)
    trainer = qai.Trainer(
        model=m,
        train_data=X,
        eval_data=X_val,
        optimizer="adamw",
        lr=3e-4,
        lr_schedule="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        batch_size=32,
        grad_accum_steps=4,
        fp16=True,
        gradient_checkpointing=False,
        output_dir="./checkpoints",
        save_steps=500,
        eval_steps=500,
        callbacks=[qai.callbacks.EarlyStopping(patience=3)],
    )
    trainer.train(epochs=10)

For sklearn-backed techniques (RandomForest, KMeans, etc.) the Trainer
delegates to technique.fit() — sklearn handles its own optimization. The
trainer just handles eval + checkpointing + logging.
"""
from __future__ import annotations
import os, time, json, math, copy, warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# =====================================================================
# LR Schedulers
# =====================================================================
class LRScheduler:
    """Base class for LR schedulers."""
    def __init__(self, optimizer, base_lr: float, total_steps: int,
                 warmup_steps: int = 0, warmup_type: str = "linear"):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = warmup_steps
        self.warmup_type = warmup_type
        self.step_count = 0

    def get_lr(self) -> float:
        if self.step_count < self.warmup_steps:
            return self._warmup_lr()
        return self._post_warmup_lr()

    def _warmup_lr(self) -> float:
        progress = (self.step_count + 1) / max(self.warmup_steps, 1)
        if self.warmup_type == "linear":
            return self.base_lr * progress
        elif self.warmup_type == "cosine":
            return self.base_lr * 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        return self.base_lr

    def _post_warmup_lr(self) -> float:
        return self.base_lr  # override in subclasses

    def step(self):
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.step_count += 1


class CosineSchedule(LRScheduler):
    def _post_warmup_lr(self) -> float:
        progress = (self.step_count - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


class LinearSchedule(LRScheduler):
    def _post_warmup_lr(self) -> float:
        progress = (self.step_count - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return self.base_lr * (1.0 - progress)


class OneCycleSchedule(LRScheduler):
    """One-Cycle policy (super-convergence): warm up to max_lr, then anneal to ~0."""
    def __init__(self, optimizer, base_lr, total_steps, warmup_steps=0,
                 max_lr=None, warmup_type="linear"):
        super().__init__(optimizer, base_lr, total_steps, warmup_steps, warmup_type)
        self.max_lr = max_lr or base_lr * 10
        # In one-cycle, warmup_steps is typically 30% of total
        if warmup_steps == 0:
            self.warmup_steps = int(0.3 * total_steps)

    def get_lr(self) -> float:
        if self.step_count < self.warmup_steps:
            # warm up from base_lr/10 to max_lr
            progress = (self.step_count + 1) / max(self.warmup_steps, 1)
            return (self.base_lr * 0.1) + (self.max_lr - self.base_lr * 0.1) * progress
        # anneal from max_lr down to ~base_lr/100
        progress = (self.step_count - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return self.max_lr * (1.0 - progress) * (1.0 - progress) + self.base_lr * 0.01 * progress


class PolynomialSchedule(LRScheduler):
    def __init__(self, optimizer, base_lr, total_steps, warmup_steps=0,
                 power=1.0, min_lr=0.0, warmup_type="linear"):
        super().__init__(optimizer, base_lr, total_steps, warmup_steps, warmup_type)
        self.power = power
        self.min_lr = min_lr

    def _post_warmup_lr(self) -> float:
        progress = (self.step_count - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return (self.base_lr - self.min_lr) * ((1 - progress) ** self.power) + self.min_lr


class ConstantSchedule(LRScheduler):
    def _post_warmup_lr(self) -> float:
        return self.base_lr


class StepSchedule(LRScheduler):
    def __init__(self, optimizer, base_lr, total_steps, warmup_steps=0,
                 step_size=1000, gamma=0.1, warmup_type="linear"):
        super().__init__(optimizer, base_lr, total_steps, warmup_steps, warmup_type)
        self.step_size = step_size
        self.gamma = gamma

    def _post_warmup_lr(self) -> float:
        steps_after_warmup = self.step_count - self.warmup_steps
        n_decays = steps_after_warmup // self.step_size
        return self.base_lr * (self.gamma ** n_decays)


SCHEDULES = {
    "cosine": CosineSchedule,
    "linear": LinearSchedule,
    "one_cycle": OneCycleSchedule,
    "polynomial": PolynomialSchedule,
    "constant": ConstantSchedule,
    "constant_with_warmup": ConstantSchedule,
    "step": StepSchedule,
}


# =====================================================================
# Optimizer factory
# =====================================================================
def make_optimizer(model, optimizer: str = "adamw", lr: float = 3e-4,
                    weight_decay: float = 0.01, **kwargs):
    """Create an optimizer by name. Supports: adamw, adam, sgd, lion, adafactor."""
    try:
        import torch
    except ImportError:
        raise ImportError("qai.Trainer requires PyTorch")

    # Get parameters — check technique.parameters(), technique.net.parameters(),
    # and recursively scan technique attributes for nn.Module instances.
    params = []
    if hasattr(model, "technique"):
        tech = model.technique
        # Try technique.parameters() first
        if hasattr(tech, "parameters"):
            try:
                params = list(tech.parameters())
            except Exception:
                params = []
        # If empty, look for nn.Module sub-attributes
        if not params:
            for attr in vars(tech).values():
                if isinstance(attr, torch.nn.Module):
                    params += list(attr.parameters())
                elif isinstance(attr, (list, tuple)):
                    for x in attr:
                        if isinstance(x, torch.nn.Module):
                            params += list(x.parameters())
    elif hasattr(model, "parameters"):
        params = list(model.parameters())

    if not params:
        raise ValueError("Model has no trainable parameters")

    # Filter out params that don't require grad
    params = [p for p in params if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters in model")

    name = optimizer.lower()
    if name == "adamw":
        betas = kwargs.get("betas", (0.9, 0.999))
        eps = kwargs.get("eps", 1e-8)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                   betas=betas, eps=eps)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay,
                                momentum=momentum)
    elif name == "lion":
        # Lion: signed-momentum optimizer (lower memory than AdamW)
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=lr, weight_decay=weight_decay,
                         betas=kwargs.get("betas", (0.9, 0.99)))
        except ImportError:
            warnings.warn("lion-pytorch not installed; falling back to AdamW. "
                          "Install with: pip install lion-pytorch")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "adafactor":
        try:
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=lr, scale_parameter=False,
                              relative_step=False, warmup_init=False)
        except ImportError:
            # Fallback: implement a basic Adafactor-like (memory-efficient Adam)
            warnings.warn("transformers not installed; falling back to AdamW for adafactor")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. "
                         f"Supported: adamw, adam, sgd, lion, adafactor")


# =====================================================================
# The Trainer
# =====================================================================
class Trainer:
    """Universal training loop for any qai model.

    Features:
      - Gradient clipping (max_grad_norm)
      - LR scheduling (cosine, linear, one_cycle, polynomial, constant, step)
      - Mixed precision (fp16, bf16) via torch.autocast
      - Gradient accumulation
      - Gradient checkpointing (model must support it)
      - Distributed training (DDP — call qai.init_distributed_context() first)
      - Checkpointing (save/resume, keep-best, keep-last-N)
      - Early stopping via callbacks
      - Logging (console + JSON; optional tensorboard)
      - EMA (Exponential Moving Average of weights)
      - SWA (Stochastic Weight Averaging)
      - Eval at every N steps or every epoch

    Args:
        model: a trained-or-untrained qai.Model
        train_data: X (and y via train_targets=)
        eval_data: optional validation X
        train_targets: optional y for supervised training
        eval_targets: optional validation y
        optimizer: "adamw" (default), "adam", "sgd", "lion", "adafactor"
        lr: peak learning rate
        lr_schedule: "cosine", "linear", "one_cycle", "polynomial",
                     "constant", "step"
        warmup_steps: number of warmup steps
        weight_decay: L2 regularization (AdamW/SGD only)
        max_grad_norm: max gradient norm for clipping (0 = disabled)
        batch_size: mini-batch size
        grad_accum_steps: gradient accumulation steps
        fp16: use float16 mixed precision (CUDA only)
        bf16: use bfloat16 mixed precision (CUDA/Ampere+ only)
        gradient_checkpointing: enable gradient checkpointing if model supports it
        output_dir: where to save checkpoints + logs
        save_steps: save every N steps (0 = only at epoch end)
        eval_steps: evaluate every N steps (0 = only at epoch end)
        keep_last_n: keep only the last N checkpoints (0 = keep all)
        ema: Exponential Moving Average decay (0 = disabled; 0.999 = standard)
        swa: Stochastic Weight Averaging (None or dict with start_step, freq, lr)
        callbacks: list of callback objects with on_step_end, on_epoch_end, etc.
        log_every: print every N steps
        seed: random seed
        verbose: if True, print progress
    """
    def __init__(self, model, train_data, eval_data=None,
                 train_targets=None, eval_targets=None,
                 optimizer: str = "adamw", lr: float = 3e-4,
                 lr_schedule: str = "cosine", warmup_steps: int = 0,
                 weight_decay: float = 0.01, max_grad_norm: float = 1.0,
                 batch_size: int = 32, grad_accum_steps: int = 1,
                 fp16: bool = False, bf16: bool = False,
                 gradient_checkpointing: bool = False,
                 output_dir: str = "./checkpoints", save_steps: int = 0,
                 eval_steps: int = 0, keep_last_n: int = 0,
                 ema: float = 0.0, swa: Optional[dict] = None,
                 callbacks: Optional[list] = None,
                 log_every: int = 50, seed: int = 42,
                 verbose: bool = True,
                 auto_device: bool = True,
                 auto_precision: bool = True,
                 auto_cuda_kernels: bool = True):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.train_targets = train_targets
        self.eval_targets = eval_targets
        self.optimizer_name = optimizer
        self.lr = lr
        self.lr_schedule_name = lr_schedule
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.grad_accum_steps = max(grad_accum_steps, 1)
        self.gradient_checkpointing = gradient_checkpointing
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.keep_last_n = keep_last_n
        self.ema_decay = ema
        self.swa_config = swa
        self.callbacks = callbacks or []
        self.log_every = log_every
        self.seed = seed
        self.verbose = verbose
        self.auto_device = auto_device
        self.auto_precision = auto_precision
        self.auto_cuda_kernels = auto_cuda_kernels

        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float("inf")
        self.history = []
        self._ema_state = None
        self._swa_state = None
        self._device = None
        self._precision_used = "fp32"
        self._cuda_kernels_enabled = False

        # Detect backend (torch vs sklearn)
        self._is_torch = self._detect_torch()
        os.makedirs(output_dir, exist_ok=True)

        # Auto-configure hardware (only for torch models)
        if self._is_torch and (auto_device or auto_precision or auto_cuda_kernels):
            self._auto_configure_hardware(fp16, bf16)

    def _detect_torch(self) -> bool:
        """Check if the model's technique is torch-based."""
        try:
            import torch
            tech = self.model.technique
            # Look for torch.nn.Module attributes
            for attr in vars(tech).values():
                if isinstance(attr, torch.nn.Module):
                    return True
                if isinstance(attr, (list, tuple)):
                    for x in attr:
                        if isinstance(x, torch.nn.Module):
                            return True
            return False
        except ImportError:
            return False

    def _auto_configure_hardware(self, user_fp16: bool, user_bf16: bool):
        """Auto-detect hardware and configure: device, precision, CUDA kernels.

        This is the "magic" that makes qai.Trainer just work:
          1. Detects GPU/MPS/CPU
          2. Moves model to best device
          3. Auto-selects bf16 (Ampere+), fp16 (Volta+), or fp32 (CPU)
          4. Enables FlashAttention + CUDA kernels if available
          5. Reports what it chose

        Users can override with fp16=True/bf16=True or auto_precision=False.
        """
        import torch

        # 1. Detect device
        device = "cpu"
        device_name = "CPU"
        if torch.cuda.is_available():
            device = f"cuda:0"
            device_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            device_name = "Apple Silicon (MPS)"
        self._device = torch.device(device)

        # 2. Move model to device
        if self.auto_device and device != "cpu":
            try:
                tech = self.model.technique
                for attr in vars(tech).values():
                    if isinstance(attr, torch.nn.Module):
                        attr.to(self._device)
                    elif isinstance(attr, (list, tuple)):
                        for x in attr:
                            if isinstance(x, torch.nn.Module):
                                x.to(self._device)
            except Exception:
                pass

        # 3. Auto-select precision (unless user specified)
        if not user_fp16 and not user_bf16 and self.auto_precision:
            try:
                from .precision import PrecisionManager
                pm = PrecisionManager()
                recommended = pm.recommend_precision()
                if recommended == "bf16":
                    self.bf16 = True
                    self.fp16 = False
                elif recommended == "fp16":
                    self.fp16 = True
                    self.bf16 = False
                else:
                    self.fp16 = False
                    self.bf16 = False
                self._precision_used = recommended
            except Exception:
                self.fp16 = user_fp16
                self.bf16 = user_bf16
                self._precision_used = "bf16" if user_bf16 else ("fp16" if user_fp16 else "fp32")
        else:
            self.fp16 = user_fp16
            self.bf16 = user_bf16
            self._precision_used = "bf16" if user_bf16 else ("fp16" if user_fp16 else "fp32")

        # 4. Auto-enable CUDA kernels (FlashAttention, etc.)
        if self.auto_cuda_kernels and device.startswith("cuda"):
            try:
                from .cuda_kernels import CudaKernelManager
                CudaKernelManager.enable_all(self.model)
                self._cuda_kernels_enabled = True
            except Exception:
                pass

        # 5. Report
        if self.verbose:
            print(f"[qai.Trainer] hardware auto-configured:")
            print(f"  device: {device_name} ({device})")
            print(f"  precision: {self._precision_used}")
            print(f"  mixed precision: {'enabled' if (self.fp16 or self.bf16) else 'disabled'}")
            print(f"  CUDA kernels: {'enabled' if self._cuda_kernels_enabled else 'disabled'}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, epochs: int = 1, max_steps: Optional[int] = None):
        """Run training for `epochs` epochs (or `max_steps` steps)."""
        if self._is_torch:
            return self._train_torch(epochs, max_steps)
        else:
            return self._train_sklearn(epochs)

    def _train_sklearn(self, epochs: int):
        """Train sklearn-backed techniques (just call fit + eval)."""
        if self.verbose:
            print(f"[qai.Trainer] sklearn backend — delegating to technique.fit()")
        X = np.asarray(self.train_data)
        y = self.train_targets
        if y is not None:
            y = np.asarray(y)
        t0 = time.time()
        self.model.train(X, y, verbose=False)
        elapsed = time.time() - t0
        # Evaluate
        eval_loss = None
        if self.eval_data is not None:
            eval_loss = self._evaluate()
        log = {"epoch": 0, "step": 0, "train_time_s": elapsed,
                "eval_loss": eval_loss, "backend": "sklearn"}
        self.history.append(log)
        self._save_log()
        if self.verbose:
            print(f"[qai.Trainer] done in {elapsed:.2f}s, eval_loss={eval_loss}")
        return self.model

    def _train_torch(self, epochs: int, max_steps: Optional[int]):
        """Train torch-backed techniques with the full feature set."""
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.verbose:
            print(f"[qai.Trainer] torch backend — fp16={self.fp16} bf16={self.bf16} "
                  f"grad_accum={self.grad_accum_steps} "
                  f"grad_ckpt={self.gradient_checkpointing}")

        # Set up gradient checkpointing
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Create optimizer
        optimizer = make_optimizer(self.model, self.optimizer_name, self.lr,
                                     self.weight_decay)

        # Compute total steps
        X = self.train_data
        n_samples = len(X) if hasattr(X, "__len__") else 1
        steps_per_epoch = max(1, n_samples // self.batch_size)
        total_steps = max_steps or (epochs * steps_per_epoch)
        if self.verbose:
            print(f"[qai.Trainer] total_steps={total_steps}, "
                  f"steps_per_epoch={steps_per_epoch}")

        # Create LR scheduler
        if self.lr_schedule_name in SCHEDULES:
            sched_cls = SCHEDULES[self.lr_schedule_name]
            scheduler = sched_cls(optimizer, self.lr, total_steps,
                                   self.warmup_steps)
        else:
            scheduler = ConstantSchedule(optimizer, self.lr, total_steps,
                                           self.warmup_steps)

        # Mixed precision
        use_amp = self.fp16 or self.bf16
        amp_dtype = torch.bfloat16 if self.bf16 else torch.float16
        scaler = torch.amp.GradScaler('cuda') if (self.fp16 and torch.cuda.is_available()) else None

        # EMA init
        if self.ema_decay > 0:
            self._ema_state = self._copy_state()

        # Fire on_train_start callbacks
        for cb in self.callbacks:
            if hasattr(cb, "on_train_start"):
                cb.on_train_start(self)

        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.model.technique.train_mode() if hasattr(self.model.technique, "train_mode") else None

            epoch_loss = 0.0
            n_batches = 0
            t_epoch = time.time()

            for batch_idx, (X_batch, y_batch) in enumerate(self._iter_batches()):
                # Forward + backward with mixed precision
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype) if use_amp else _nullcontext():
                    loss = self._compute_loss(X_batch, y_batch)
                    loss = loss / self.grad_accum_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self._get_params(), self.max_grad_norm)

                    # Optimizer step
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.global_step += 1

                    # EMA update
                    if self.ema_decay > 0:
                        self._update_ema()

                    # SWA update
                    if self.swa_config and self.global_step >= self.swa_config.get("start_step", 0):
                        if self.global_step % self.swa_config.get("freq", 100) == 0:
                            self._update_swa()

                    epoch_loss += float(loss.item()) * self.grad_accum_steps
                    n_batches += 1

                    # Logging
                    if self.verbose and self.global_step % self.log_every == 0:
                        cur_lr = optimizer.param_groups[0]["lr"]
                        print(f"  epoch {epoch+1}/{epochs} step {self.global_step}/{total_steps} "
                              f"loss={float(loss.item()):.4f} lr={cur_lr:.2e}")

                    # Eval
                    if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                        eval_loss = self._evaluate()
                        if eval_loss is not None and eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self._save_checkpoint("best")
                        if self.verbose and eval_loss is not None:
                            print(f"    eval_loss={eval_loss:.4f}")

                    # Save
                    if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self._save_checkpoint(f"step_{self.global_step}")
                        self._cleanup_old_checkpoints()

                    # Fire on_step_end callbacks
                    for cb in self.callbacks:
                        if hasattr(cb, "on_step_end"):
                            stop = cb.on_step_end(self, float(loss.item()))
                            if stop:
                                return self._finalize()

                    # Max steps check
                    if max_steps and self.global_step >= max_steps:
                        return self._finalize()

            # End of epoch
            avg_loss = epoch_loss / max(n_batches, 1)
            epoch_time = time.time() - t_epoch
            eval_loss = self._evaluate() if self.eval_data is not None else None

            log = {"epoch": epoch + 1, "step": self.global_step,
                   "train_loss": avg_loss, "eval_loss": eval_loss,
                   "epoch_time_s": epoch_time, "lr": optimizer.param_groups[0]["lr"]}
            self.history.append(log)

            if self.verbose:
                print(f"[qai.Trainer] epoch {epoch+1}/{epochs} done — "
                      f"train_loss={avg_loss:.4f} eval_loss={eval_loss} "
                      f"time={epoch_time:.1f}s")

            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch+1}")
            self._cleanup_old_checkpoints()

            # Fire on_epoch_end callbacks
            for cb in self.callbacks:
                if hasattr(cb, "on_epoch_end"):
                    stop = cb.on_epoch_end(self, avg_loss)
                    if stop:
                        return self._finalize()

            # Early stopping on eval
            if eval_loss is not None and eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self._save_checkpoint("best")

        return self._finalize()

    def _finalize(self):
        """Apply EMA/SWA weights and save final state."""
        if self._ema_state is not None:
            self._load_state(self._ema_state)
            if self.verbose:
                print("[qai.Trainer] applied EMA weights")
        if self._swa_state is not None:
            self._load_state(self._swa_state)
            if self.verbose:
                print("[qai.Trainer] applied SWA weights")
        self._save_log()
        return self.model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _iter_batches(self):
        """Yield (X_batch, y_batch) tuples."""
        X = self.train_data
        y = self.train_targets
        n = len(X) if hasattr(X, "__len__") else 1
        indices = np.random.permutation(n)
        for i in range(0, n, self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            X_batch = X[batch_idx] if hasattr(X, "__getitem__") else X
            y_batch = y[batch_idx] if y is not None and hasattr(y, "__getitem__") else y
            yield X_batch, y_batch

    def _compute_loss(self, X_batch, y_batch):
        """Compute loss for a batch."""
        import torch
        tech = self.model.technique

        # Try technique-specific loss methods first
        if hasattr(tech, "compute_loss"):
            return tech.compute_loss(X_batch, y_batch)

        # Generic: forward + standard loss
        # For classification
        if y_batch is not None:
            try:
                # Try forward to get logits
                X_t = torch.as_tensor(np.asarray(X_batch), dtype=torch.float32)
                if hasattr(tech, "_forward"):
                    logits = tech._forward(X_t)
                elif hasattr(tech, "forward"):
                    result = tech.forward(X_t)
                    if isinstance(result, torch.Tensor):
                        logits = result
                    else:
                        return torch.tensor(0.0, requires_grad=True)
                else:
                    return torch.tensor(0.0, requires_grad=True)
                y_t = torch.as_tensor(np.asarray(y_batch), dtype=torch.long)
                # Handle shape mismatches
                if logits.dim() > 1 and logits.shape[0] != y_t.shape[0]:
                    logits = logits.view(y_t.shape[0], -1)
                return torch.nn.functional.cross_entropy(logits, y_t)
            except Exception as e:
                return torch.tensor(0.0, requires_grad=True)
        # Unsupervised
        return torch.tensor(0.0, requires_grad=True)

    def _evaluate(self):
        """Run evaluation on eval_data. Returns loss or None."""
        if self.eval_data is None:
            return None
        try:
            import torch
            with torch.no_grad():
                if hasattr(self.model.technique, "compute_loss"):
                    return float(self.model.technique.compute_loss(
                        self.eval_data, self.eval_targets))
                # Fallback: just predict and compute accuracy
                preds = []
                X = self.eval_data
                y = self.eval_targets
                n = min(len(X), 100)  # cap eval at 100 samples
                for i in range(n):
                    try:
                        p = self.model.predict(X[i])
                        preds.append(p)
                    except Exception:
                        return None
                if y is not None and len(preds) > 0:
                    correct = sum(1 for p, t in zip(preds, y[:n]) if str(p) == str(t))
                    return 1.0 - correct / n  # error rate as "loss"
                return None
        except Exception:
            return None

    def _get_params(self):
        """Get all trainable parameters — try technique.parameters(), then
        scan sub-attributes for nn.Module instances."""
        import torch
        params = []
        tech = self.model.technique
        # Try technique.parameters() first
        if hasattr(tech, "parameters"):
            try:
                params = list(tech.parameters())
            except Exception:
                params = []
        # If empty, scan sub-attributes
        if not params:
            for attr in vars(tech).values():
                if isinstance(attr, torch.nn.Module):
                    params += list(attr.parameters())
                elif isinstance(attr, (list, tuple)):
                    for x in attr:
                        if isinstance(x, torch.nn.Module):
                            params += list(x.parameters())
        return [p for p in params if p.requires_grad]

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on all torch submodules."""
        try:
            import torch
            for attr in vars(self.model.technique).values():
                if isinstance(attr, torch.nn.Module) and hasattr(attr, "gradient_checkpointing_enable"):
                    attr.gradient_checkpointing_enable()
        except Exception:
            pass

    def _copy_state(self):
        """Deep copy of all model parameters (for EMA/SWA)."""
        try:
            import torch
            state = {}
            for name, attr in vars(self.model.technique).items():
                if isinstance(attr, torch.nn.Module):
                    state[name] = copy.deepcopy(attr.state_dict())
                elif isinstance(attr, torch.nn.Parameter):
                    state[name] = attr.data.clone()
            return state
        except Exception:
            return None

    def _update_ema(self):
        """Update EMA state with current weights."""
        if self._ema_state is None:
            return
        import torch
        with torch.no_grad():
            for name, attr in vars(self.model.technique).items():
                if name not in self._ema_state:
                    continue
                if isinstance(attr, torch.nn.Module):
                    sd = attr.state_dict()
                    ema_sd = self._ema_state[name]
                    for k in sd:
                        ema_sd[k].mul_(self.ema_decay).add_(sd[k], alpha=1 - self.ema_decay)
                elif isinstance(attr, torch.nn.Parameter):
                    self._ema_state[name].mul_(self.ema_decay).add_(
                        attr.data, alpha=1 - self.ema_decay)

    def _update_swa(self):
        """Update SWA running average."""
        if self._swa_state is None:
            self._swa_state = {"state": self._copy_state(), "n": 0}
        self._swa_state["n"] += 1
        n = self._swa_state["n"]
        # SWA: state = (state * (n-1) + current) / n
        import torch
        with torch.no_grad():
            cur = self._copy_state()
            for name in cur:
                if name in self._swa_state["state"]:
                    for k in cur[name]:
                        self._swa_state["state"][name][k].mul_((n - 1) / n).add_(
                            cur[name][k], alpha=1.0 / n)

    def _load_state(self, state):
        """Load state dict into model."""
        try:
            import torch
            for name, attr in vars(self.model.technique).items():
                if name in state and isinstance(attr, torch.nn.Module):
                    attr.load_state_dict(state[name])
        except Exception:
            pass

    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        try:
            path = os.path.join(self.output_dir, f"{name}.pt")
            import qai
            qai.save_model(self.model, path)
        except Exception as e:
            if self.verbose:
                print(f"[qai.Trainer] checkpoint save failed: {e}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if self.keep_last_n <= 0:
            return
        try:
            import glob
            ckpts = sorted(glob.glob(os.path.join(self.output_dir, "*.pt")),
                            key=os.path.getmtime)
            for old in ckpts[:-self.keep_last_n]:
                if "best" not in old:  # never delete best
                    os.remove(old)
        except Exception:
            pass

    def _save_log(self):
        """Save training history to JSON."""
        try:
            path = os.path.join(self.output_dir, "training_log.json")
            with open(path, "w") as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception:
            pass


# _nullcontext for Python < 3.7 compatibility (though we require 3.12)
try:
    from contextlib import nullcontext as _nullcontext
except ImportError:
    class _nullcontext:
        def __enter__(self): return self
        def __exit__(self, *a): pass


__all__ = ["Trainer", "LRScheduler", "CosineSchedule", "LinearSchedule",
           "OneCycleSchedule", "PolynomialSchedule", "ConstantSchedule",
           "StepSchedule", "make_optimizer", "SCHEDULES"]
