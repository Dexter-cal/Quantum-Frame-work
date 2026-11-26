"""
qai.callbacks -- Standard training callbacks (Keras-style contract).

Each callback is an object with zero or more of these methods:
  - on_train_start(model) -> None
  - on_epoch_end(model, epoch, **kwargs) -> None  (return False to stop)
  - on_train_end(model, elapsed_s) -> None

Standard callbacks:
  - EarlyStopping(patience=5, monitor="loss", min_delta=1e-4)
  - CheckpointEvery(n_epochs=10, path="checkpoint.pkl")
  - LogProgress(every_n=1)
  - TimeLimit(seconds=300)

Usage:
    model.train(X, y, callbacks=[
        qai.callbacks.EarlyStopping(patience=3),
        qai.callbacks.LogProgress(every_n=5),
    ])
"""
from __future__ import annotations
import time, pickle, os
from typing import Any, Optional, Callable


class Callback:
    """Base callback class. Override any of the hooks."""
    def on_train_start(self, model): pass
    def on_epoch_end(self, model, epoch, **kwargs): pass
    def on_train_end(self, model, elapsed_s=None): pass


class EarlyStopping(Callback):
    """Stop training when a monitored quantity stops improving.

    Note: qai's built-in fit() doesn't expose per-epoch metrics yet
    (it's batch-oriented). This callback is most useful when used with
    a custom training loop that calls on_epoch_end with `monitor=value`.
    """
    def __init__(self, patience: int = 5, monitor: str = "loss",
                 min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode  # "min" or "max"
        self.best = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, model, epoch, **kwargs):
        current = kwargs.get(self.monitor)
        if current is None:
            return
        improved = (current < self.best - self.min_delta) if self.mode == "min" \
                   else (current > self.best + self.min_delta)
        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return False  # signal stop


class CheckpointEvery(Callback):
    """Save the model to `path` every `n_epochs` epochs."""
    def __init__(self, n_epochs: int = 10, path: str = "checkpoint.pkl"):
        self.n_epochs = n_epochs
        self.path = path
        self.n_saves = 0

    def on_epoch_end(self, model, epoch, **kwargs):
        if (epoch + 1) % self.n_epochs == 0:
            try:
                with open(self.path, "wb") as f:
                    pickle.dump(model, f)
                self.n_saves += 1
            except Exception as e:
                print(f"[CheckpointEvery] WARNING: save failed: {e}")

    def on_train_end(self, model, elapsed_s=None):
        # Always save one final checkpoint
        try:
            with open(self.path, "wb") as f:
                pickle.dump(model, f)
            self.n_saves += 1
        except Exception:
            pass


class LogProgress(Callback):
    """Print training progress every `every_n` epochs."""
    def __init__(self, every_n: int = 1, prefix: str = "[qai]"):
        self.every_n = every_n
        self.prefix = prefix
        self.start_time = None

    def on_train_start(self, model):
        self.start_time = time.time()

    def on_epoch_end(self, model, epoch, **kwargs):
        if (epoch + 1) % self.every_n == 0:
            elapsed = time.time() - (self.start_time or time.time())
            extra = " ".join(f"{k}={v:.4f}" for k, v in kwargs.items()
                              if isinstance(v, (int, float)))
            print(f"{self.prefix} epoch {epoch+1} elapsed={elapsed:.1f}s {extra}")


class TimeLimit(Callback):
    """Stop training after `seconds` wall-clock time."""
    def __init__(self, seconds: float = 300):
        self.seconds = seconds
        self.start_time = None

    def on_train_start(self, model):
        self.start_time = time.time()

    def on_epoch_end(self, model, epoch, **kwargs):
        elapsed = time.time() - (self.start_time or time.time())
        if elapsed > self.seconds:
            print(f"[TimeLimit] Stopping at epoch {epoch+1} ({elapsed:.1f}s > {self.seconds}s)")
            return False


class LambdaCallback(Callback):
    """Wrap arbitrary functions as a callback. Useful for one-off custom logic."""
    def __init__(self,
                 on_train_start: Optional[Callable] = None,
                 on_epoch_end: Optional[Callable] = None,
                 on_train_end: Optional[Callable] = None):
        self._on_train_start = on_train_start
        self._on_epoch_end = on_epoch_end
        self._on_train_end = on_train_end

    def on_train_start(self, model):
        if self._on_train_start:
            self._on_train_start(model)

    def on_epoch_end(self, model, epoch, **kwargs):
        if self._on_epoch_end:
            return self._on_epoch_end(model, epoch, **kwargs)

    def on_train_end(self, model, elapsed_s=None):
        if self._on_train_end:
            self._on_train_end(model, elapsed_s=elapsed_s)


__all__ = [
    "Callback",
    "EarlyStopping",
    "CheckpointEvery",
    "LogProgress",
    "TimeLimit",
    "LambdaCallback",
]
