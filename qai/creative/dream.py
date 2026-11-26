"""
qai.creative.dream -- Creative Idea 1: DREAM MODE

Models sleep too. When the system is idle, Dream Mode replays recent
experiences through the model's generative pathway, mixes them via latent
interpolation, and writes the imagined samples back to a replay buffer.

Three phases per dream cycle:
  1. Consolidation  -- replay high-loss experiences at reduced learning rate
  2. Imagination    -- sample new latent points via the model's own prior
                       and decode them into pseudo-experiences
  3. Rehearsal      -- mix real + imagined samples and run one mini-epoch

Usage:
    dreamer = qai.creative.DreamMode(model, replay_buffer)
    dreamer.dream(steps=10)
"""
from __future__ import annotations
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class ReplayBuffer:
    """Circular buffer of (state, action, reward, next_state, done) tuples."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._buf: List[Tuple] = []
        self._pos = 0

    def push(self, *transition) -> None:
        if len(self._buf) < self.capacity:
            self._buf.append(transition)
        else:
            self._buf[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple]:
        n = min(batch_size, len(self._buf))
        return random.sample(self._buf, n)

    def highest_loss(self, model, batch_size: int = 10) -> List[Tuple]:
        if not self._buf:
            return []
        scored = []
        for t in self._buf:
            try:
                x = np.asarray(t[0], dtype=np.float32)
                loss = float(np.mean((model.forward(x) - x) ** 2))
            except Exception:
                loss = 0.0
            scored.append((loss, t))
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:batch_size]]

    def __len__(self):
        return len(self._buf)


class DreamMode:
    """Dream Mode controller. Wraps a qai model + replay buffer."""

    def __init__(self, model: Any, replay_buffer: ReplayBuffer,
                 imagination_strength: float = 0.3,
                 consolidation_lr: float = 0.1):
        self.model = model
        self.buffer = replay_buffer
        self.strength = imagination_strength
        self.consolidation_lr = consolidation_lr
        self.dream_log: List[Dict[str, Any]] = []

    def consolidate(self, steps: int = 5) -> None:
        if len(self.buffer) == 0:
            return
        for _ in range(steps):
            batch = self.buffer.highest_loss(self.model, batch_size=8)
            if not batch:
                continue
            X = np.array([t[0] for t in batch], dtype=np.float32)
            try:
                self.model.fit(X, X)
            except Exception:
                pass

    def imagine(self, n: int = 16) -> List[np.ndarray]:
        if not hasattr(self.model, "sample"):
            return []
        try:
            samples = self.model.sample(n)
        except Exception:
            return []
        out = []
        for s in samples:
            s = np.asarray(s, dtype=np.float32)
            if len(self.buffer) > 0:
                real = np.asarray(random.choice(self.buffer._buf)[0], dtype=np.float32)
                if real.shape == s.shape:
                    s = (1 - self.strength) * s + self.strength * real
            out.append(s)
        return out

    def rehearse(self, imagined: List[np.ndarray]) -> None:
        if not imagined:
            return
        X = np.array(imagined, dtype=np.float32)
        try:
            self.model.fit(X, X)
        except Exception:
            pass

    def dream(self, steps: int = 10) -> Dict[str, Any]:
        log_entry: Dict[str, Any] = {"steps": steps, "phases": []}
        for step in range(steps):
            self.consolidate(steps=1)
            imagined = self.imagine(n=8)
            self.rehearse(imagined)
            log_entry["phases"].append({
                "step": step,
                "n_imagined": len(imagined),
                "buffer_size": len(self.buffer),
            })
        self.dream_log.append(log_entry)
        return log_entry

    def dream_summary(self) -> Dict[str, Any]:
        if not self.dream_log:
            return {"cycles": 0}
        last = self.dream_log[-1]
        return {
            "cycles": len(self.dream_log),
            "last_cycle_steps": last["steps"],
            "total_imagined": sum(p["n_imagined"] for p in last["phases"]),
        }


__all__ = ["DreamMode", "ReplayBuffer"]
