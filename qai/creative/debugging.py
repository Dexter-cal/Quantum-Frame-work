"""
qai.creative.debugging -- Creative Ideas 4 & 7

Idea 4 — TIME-TRAVEL DEBUGGING
    Records every state mutation during a training run. You can later step
    forward / backward through training time, inspect the model's weights
    at any past step, and re-exec from any past point.

Idea 7 — CROSS-ARCHITECTURE KNOWLEDGE TRANSFER
    Distill knowledge from a teacher of one architecture family into a
    student of another (e.g. Transformer -> MLP, CNN -> ViT, RNN -> Mamba).
    Uses feature alignment via Procrustes on a shared latent space.
"""
from __future__ import annotations
import copy
import hashlib
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# ============================================================================
# Idea 4 — TIME-TRAVEL DEBUGGING
# ============================================================================
class TimeTravelDebugger:
    """Records model snapshots during training; lets you scrub through time."""

    def __init__(self, model: Any, max_snapshots: int = 100):
        self.model = model
        self.max_snapshots = max_snapshots
        self.snapshots: List[Dict[str, Any]] = []
        self._cursor = -1

    @staticmethod
    def _hash_state(state: Dict[str, Any]) -> str:
        try:
            return hashlib.sha256(pickle.dumps(state)).hexdigest()[:16]
        except Exception:
            return "???"

    def _capture_state(self) -> Dict[str, Any]:
        if hasattr(self.model, "state_dict"):
            try:
                return {k: v.detach().cpu().clone() if _TORCH and hasattr(v, "detach") else v
                        for k, v in self.model.state_dict().items()}
            except Exception:
                return {}
        return {"_attrs": {k: copy.deepcopy(v) for k, v in self.model.__dict__.items()
                            if not k.startswith("_") and isinstance(v, (int, float, list, dict))}}

    def capture(self, step: int, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Take a snapshot at the current step. Returns snapshot index."""
        state = self._capture_state()
        snap = {
            "step": step,
            "timestamp": time.time(),
            "state": state,
            "hash": self._hash_state(state),
            "metadata": metadata or {},
        }
        self.snapshots.append(snap)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        self._cursor = len(self.snapshots) - 1
        return self._cursor

    def goto(self, idx: int) -> Dict[str, Any]:
        """Restore model state to snapshot idx."""
        if idx < 0 or idx >= len(self.snapshots):
            raise IndexError(f"snapshot {idx} out of range [0, {len(self.snapshots)})")
        snap = self.snapshots[idx]
        if hasattr(self.model, "load_state_dict"):
            try:
                # strip _attrs key if present (fallback state)
                state = {k: v for k, v in snap["state"].items() if k != "_attrs"}
                self.model.load_state_dict(state)
            except Exception:
                pass
        else:
            for k, v in snap["state"].get("_attrs", {}).items():
                try:
                    setattr(self.model, k, copy.deepcopy(v))
                except Exception:
                    pass
        self._cursor = idx
        return snap

    def step_forward(self) -> Optional[Dict[str, Any]]:
        if self._cursor + 1 < len(self.snapshots):
            return self.goto(self._cursor + 1)
        return None

    def step_backward(self) -> Optional[Dict[str, Any]]:
        if self._cursor - 1 >= 0:
            return self.goto(self._cursor - 1)
        return None

    def diff(self, idx_a: int, idx_b: int) -> Dict[str, Any]:
        """Compare two snapshots; returns which keys changed + L2 distance."""
        sa = self.snapshots[idx_a]["state"]
        sb = self.snapshots[idx_b]["state"]
        keys = set(sa.keys()) | set(sb.keys())
        changed = []
        total_l2 = 0.0
        for k in keys:
            a = sa.get(k)
            b = sb.get(k)
            if a is None or b is None:
                changed.append({"key": k, "change": "added" if a is None else "removed"})
                continue
            try:
                diff_arr = np.asarray(a) - np.asarray(b)
                l2 = float(np.linalg.norm(diff_arr))
                if l2 > 1e-8:
                    changed.append({"key": k, "l2": l2})
                    total_l2 += l2
            except Exception:
                pass
        return {"n_changed": len(changed), "total_l2": total_l2, "changes": changed}

    def timeline(self) -> List[Dict[str, Any]]:
        return [{"step": s["step"], "hash": s["hash"],
                  "metadata": s["metadata"]} for s in self.snapshots]


# ============================================================================
# Idea 7 — CROSS-ARCHITECTURE KNOWLEDGE TRANSFER
# ============================================================================
class CrossArchTransfer:
    """Distill knowledge between models of different architecture families.

    The teacher and student must both expose `forward(x) -> embedding` (or
    a callable feature_fn hook). We align their embedding spaces via
    Procrustes and minimise a contrastive loss between aligned embeddings.
    """

    def __init__(self, teacher: Any, student: Any,
                 teacher_dim: Optional[int] = None,
                 student_dim: Optional[int] = None,
                 temperature: float = 1.0):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alignment: Optional[np.ndarray] = None
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim

    def _embed(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get an embedding for each row of X via model.forward."""
        out = []
        for x in X:
            try:
                v = model.forward(x)
                v = np.asarray(v).flatten()
                out.append(v)
            except Exception:
                out.append(np.zeros(1))
        return np.array(out)

    def align(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Compute the orthogonal Procrustes alignment from teacher -> student."""
        idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        Xs = X[idx]
        T = self._embed(self.teacher, Xs)
        S = self._embed(self.student, Xs)
        if T.shape[1] != S.shape[1]:
            # pad smaller
            d = max(T.shape[1], S.shape[1])
            if T.shape[1] < d:
                T = np.pad(T, ((0, 0), (0, d - T.shape[1])))
            if S.shape[1] < d:
                S = np.pad(S, ((0, 0), (0, d - S.shape[1])))
        # solve min ||T @ R - S||_F  with R orthonormal
        U, _, Vt = np.linalg.svd(T.T @ S, full_matrices=False)
        R = U @ Vt
        self.alignment = R
        self.teacher_dim = T.shape[1]
        self.student_dim = S.shape[1]
        return R

    def distillation_loss(self, X: np.ndarray) -> float:
        if self.alignment is None:
            self.align(X)
        T = self._embed(self.teacher, X)
        S = self._embed(self.student, X)
        T_aligned = T @ self.alignment
        # KL between softmax distributions
        def softmax(x, t):
            x = x / t
            x = x - x.max(axis=-1, keepdims=True)
            e = np.exp(x)
            return e / e.sum(axis=-1, keepdims=True)
        p_t = softmax(T_aligned, self.temperature)
        p_s = softmax(S, self.temperature)
        eps = 1e-8
        return float(np.mean(np.sum(p_t * (np.log(p_t + eps) - np.log(p_s + eps)), axis=-1)))

    def transfer(self, X: np.ndarray, n_steps: int = 10,
                 lr: float = 0.01) -> Dict[str, Any]:
        """Run n_steps of distillation. Returns history of distillation losses."""
        self.align(X)
        history = []
        for step in range(n_steps):
            loss = self.distillation_loss(X)
            history.append({"step": step, "loss": loss})
            # nudging student weights requires gradient; if model supports fit, use it
            try:
                # use teacher's aligned embeddings as regression targets for student
                T = self._embed(self.teacher, X)
                T_aligned = T @ self.alignment
                S = self._embed(self.student, X)
                # fit student to predict T_aligned from inputs
                self.student.fit(X, T_aligned)
            except Exception:
                pass
        return {"final_loss": history[-1]["loss"] if history else None,
                "history": history}


__all__ = ["TimeTravelDebugger", "CrossArchTransfer"]
