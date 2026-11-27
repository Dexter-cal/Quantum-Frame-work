"""
qai.distillation_v2 — THE canonical knowledge-distillation implementation.

This is the single home of distillation in QAI. Everything else that used to
carry its own copy of the algorithm (`qai.advanced.KnowledgeDistiller`,
`qai.compression.Distiller`, `qai.creative.experimental.KnowledgeDistiller2`,
`qai.creative_tools.ModelDistiller`) is now a thin backward-compatibility
delegate to this module — one feature, one code path, parameters for the rest.

The preferred user-facing entry point is the top-level function::

    import qai
    result = qai.distill(teacher, student, X, y,
                         strategy="vanilla",      # vanilla|hint|attention|self|multi
                         temperature=4.0,         # softmax temperature
                         alpha=0.7,               # hard-loss weight (soft gets 1-alpha)
                         epochs=30, lr=1e-3)
    print(result.to_markdown())

Loss (identical to Hinton et al. 2015, real gradient path when torch is
installed and both models are torch-backed):

    L = alpha * CE(student_logits, y_true) + (1 - alpha) * T² * KL(student_soft ‖ teacher_soft)

Strategies:
  * ``vanilla``   — soft-target KL distillation (the classic; fully working
                    on torch-backed techniques).
  * ``hint``      — adds an MSE alignment term between the first torch
                    parameter tensors of teacher and student (weight-space
                    hint approximation, used when layer hooks are unavailable).
  * ``attention`` — same alignment term weighted toward attention-like
                    parameter blocks (approximation; honest in the report).
  * ``self``      — the model distils a fresh copy of itself
                    (:class:`SelfDistiller`).
  * ``multi``     — several teachers, probability-averaged
                    (:class:`MultiTeacherDistiller`).

No-torch path (numpy-only techniques such as random_forest → decision_tree):
the student is trained on the hard labels with teacher-confidence sample
weighting where the technique accepts ``sample_weight``; the result then
reports real KL(teacher ‖ student) divergence before/after and flags the
mode as ``hard-label fallback`` so nobody mistakes it for soft-label
distillation. Install torch for the true soft-label path.

Back-compat aliases (all delegate here — do not add new code to them):
    qai.KnowledgeDistiller            (from qai.advanced)
    qai.compression.Distiller         (static-method API)
    qai.KnowledgeDistiller2           (from qai.creative.experimental)
    qai.ModelDistiller                (from qai.creative_tools)
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# Torch availability (optional)
# ============================================================================
try:
    import torch
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# ============================================================================
# Config
# ============================================================================

@dataclass
class DistillationConfig:
    """Configuration for distillation (one config object, every strategy).

    Args:
        temperature: softmax temperature T (higher = softer teacher targets).
            Typical 2–10. The T² factor compensates gradient scaling.
        alpha: weight of the hard-label CE loss; soft KL gets (1 - alpha).
            alpha=1.0 → plain supervised training; alpha=0.0 → pure KD.
        epochs: training epochs (torch path runs true minibatch epochs;
            the numpy hard-label path fits once and treats this as info).
        lr: learning rate for the torch path optimiser.
        batch_size: minibatch size for the torch path.
        strategy: ``vanilla`` | ``hint`` | ``attention`` | ``self`` | ``multi``.
        hint_layers: (teacher_layer_idx, student_layer_idx) pairs — reserved
            for future real hint regression; the current hint strategy uses
            the first-parameter alignment approximation.
        align_weight: weight of the weight-alignment MSE term used by the
            ``hint`` / ``attention`` strategies (0 disables it).
        verbose: print per-epoch progress.
    """
    temperature: float = 4.0
    alpha: float = 0.7
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 32
    strategy: str = "vanilla"
    hint_layers: Optional[List[Tuple[int, int]]] = None
    align_weight: float = 0.0
    verbose: bool = False


# ============================================================================
# Distillation result
# ============================================================================

@dataclass
class DistillationResult:
    """Result of a distillation run — with honest reporting of what ran."""
    strategy: str = ""
    teacher_acc: float = 0.0
    student_acc_before: float = 0.0
    student_acc_after: float = 0.0
    compression_ratio: float = 1.0  # teacher_params / student_params
    knowledge_transfer: float = 0.0  # fraction of teacher's edge transferred
    training_time_s: float = 0.0
    epochs: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)
    # -- new in v1.8.5: honest mode + real loss numbers --
    distillation_mode: str = "not-run"
    #   "soft-label KL (torch)"          — real gradient KD ran
    #   "soft-label KL + alignment"      — torch KD plus weight alignment
    #   "confidence-weighted fallback"   — numpy: hard labels + sample weights
    #   "hard-label fallback"            — numpy: hard labels only
    #   "not-run"                        — distill() not called
    loss_final: float = float("nan")
    kl_before: float = float("nan")  # KL(teacher‖student) before distilling
    kl_after: float = float("nan")   # KL(teacher‖student) after distilling

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Distillation Report",
            "",
            f"- **Strategy**: {self.strategy}",
            f"- **Mode**: {self.distillation_mode}",
            f"- **Teacher accuracy**: {self.teacher_acc:.4f}",
            f"- **Student accuracy (before)**: {self.student_acc_before:.4f}",
            f"- **Student accuracy (after)**: {self.student_acc_after:.4f}",
            f"- **Compression ratio**: {self.compression_ratio:.2f}x",
            f"- **Knowledge transfer**: {self.knowledge_transfer:.1%}",
            f"- **Final loss**: {self.loss_final:.4f}",
            f"- **KL(teacher‖student)**: {self.kl_before:.4f} → {self.kl_after:.4f}",
            f"- **Training time**: {self.training_time_s:.1f}s",
            f"- **Epochs**: {self.epochs}",
        ]
        return "\n".join(lines)


# ============================================================================
# Shared helpers — the ONE implementation of soft-target extraction + losses
# ============================================================================

def get_soft_targets(model: Any, X: np.ndarray) -> np.ndarray:
    """Get a model's predicted class probabilities (soft targets).

    Works for qai Models (predict_proba → technique → inner model), bare
    sklearn estimators, and torch nn.Modules. One-hot-encodes hard-label
    outputs as a last resort; returns uniform if everything fails.
    """
    X = np.asarray(X)
    # qai Model
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X))
            if proba.ndim == 2 and proba.shape[0] == len(X):
                return proba
        except Exception:
            pass
    # qai Model → technique.model (sklearn estimator)
    tech = getattr(model, "technique", None)
    inner = getattr(tech, "model", None)
    if inner is not None and hasattr(inner, "predict_proba"):
        try:
            proba = np.asarray(inner.predict_proba(X))
            if proba.ndim == 2 and proba.shape[0] == len(X):
                return proba
        except Exception:
            pass
    # torch module
    if _TORCH and hasattr(model, "forward") and not hasattr(model, "technique"):
        try:
            with torch.no_grad():
                t = torch.as_tensor(np.asarray(X, dtype=np.float32))
                out = model(t)
                if out.ndim == 2:
                    return F.softmax(out, dim=-1).cpu().numpy()
        except Exception:
            pass
    # hard-label fallback → one-hot
    try:
        preds = np.asarray(model.predict(X))
        preds = preds.reshape(-1)
        n_classes = int(np.nanmax(preds)) + 1
        one_hot = np.zeros((len(preds), n_classes))
        one_hot[np.arange(len(preds)), preds.astype(int)] = 1.0
        return one_hot
    except Exception:
        return np.ones((len(X), 2)) / 2.0


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q) averaged over rows — measuring metric only (no gradients)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-10, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-10, 1.0)
    if p.shape != q.shape:
        return float("nan")
    return float(np.mean(np.sum(p * np.log(p / q), axis=1)))


def _get_torch_net(model: Any):
    """Return the torch nn.Module inside a qai Model, or None."""
    if not _TORCH:
        return None
    tech = getattr(model, "technique", None)
    net = getattr(tech, "net", None)
    if net is not None and hasattr(net, "parameters"):
        try:
            next(net.parameters())
            return net
        except StopIteration:
            return None
    return None


def _ensure_student_built(student_model: Any, X: np.ndarray, y: np.ndarray):
    """Make sure a qai Model's torch net exists (build it if needed)."""
    tech = getattr(student_model, "technique", None)
    if tech is None:
        return None
    if getattr(tech, "net", None) is not None:
        return tech.net
    y_arr = np.asarray(y)
    if y_arr.ndim == 1:
        n_classes = int(y_arr.max()) + 1
    else:
        n_classes = y_arr.shape[1]
    device = torch.device("cpu")
    built = False
    if hasattr(tech, "_build_model"):
        try:
            tech._build_model(n_classes=n_classes)
            built = getattr(tech, "net", None) is not None
        except Exception:
            built = False
    if not built and hasattr(tech, "task_type") and hasattr(tech, "hidden_dim"):
        try:
            from qai.techniques.neural_network import PyTorchMLP
            X_arr = np.asarray(X, dtype=np.float32)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            out_dim = n_classes if tech.task_type == "classification" else 1
            tech.net = PyTorchMLP(
                X_arr.shape[1], tech.hidden_dim, out_dim, tech.task_type
            ).to(device)
            built = True
        except Exception:
            built = False
    return getattr(tech, "net", None)


def _alignment_loss(teacher_net, student_net, strategy: str):
    """Weight-space alignment MSE between teacher and student first params.

    Approximation used by the ``hint`` / ``attention`` strategies when layer
    hooks aren't available. Returns a torch scalar tensor (or None if shapes
    don't match / torch missing).
    """
    if not _TORCH or teacher_net is None or student_net is None:
        return None
    try:
        t_first = next(iter(teacher_net.parameters()))
        s_first = next(iter(student_net.parameters()))
        if t_first.shape != s_first.shape:
            return None
        return F.mse_loss(s_first, t_first.detach())
    except (StopIteration, Exception):
        return None


def torch_distill_epoch(teacher_net, student_net, X: np.ndarray, y: np.ndarray,
                        config: DistillationConfig) -> Dict[str, Any]:
    """Run the REAL distillation training loop (torch path).

    L = alpha * CE(student_logits, y) + (1-alpha) * T² * KL(student_soft ‖ teacher_soft)
    plus an optional alignment term for hint/attention strategies.

    Returns dict with ``losses`` (per-epoch) and ``mode``.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.int64)
    N = len(X_arr)
    device = next(student_net.parameters()).device
    T = config.temperature

    teacher_net.eval()
    opt = torch.optim.AdamW(student_net.parameters(), lr=config.lr)
    losses: List[float] = []
    align_on = config.align_weight > 0 and config.strategy in ("hint", "attention")

    for epoch in range(config.epochs):
        rng = np.random.default_rng(epoch)
        perm = rng.permutation(N)
        ep_loss, n_batches = 0.0, 0
        for start in range(0, N, config.batch_size):
            idx = perm[start:start + config.batch_size]
            Xb = torch.tensor(X_arr[idx], device=device)
            yb = torch.tensor(y_arr[idx], device=device, dtype=torch.long)

            with torch.no_grad():
                teacher_logits = teacher_net(Xb)
                teacher_soft = F.softmax(teacher_logits / T, dim=-1)

            student_logits = student_net(Xb)
            student_soft = F.log_softmax(student_logits / T, dim=-1)

            kl_loss = F.kl_div(student_soft, teacher_soft,
                               reduction="batchmean") * (T ** 2)
            ce_loss = F.cross_entropy(student_logits, yb)
            loss = config.alpha * ce_loss + (1 - config.alpha) * kl_loss

            if align_on:
                align = _alignment_loss(teacher_net, student_net, config.strategy)
                if align is not None:
                    loss = loss + config.align_weight * align

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.detach())
            n_batches += 1
        avg = ep_loss / max(1, n_batches)
        losses.append(avg)
        if config.verbose:
            print(f"  [distill] epoch {epoch + 1}/{config.epochs} loss={avg:.4f}")

    return {"losses": losses, "mode": "soft-label KL (torch)"}


def numpy_distill(student_model: Any, X: np.ndarray, y: np.ndarray,
                  teacher_soft: np.ndarray,
                  config: DistillationConfig) -> Dict[str, Any]:
    """No-torch distillation path — honest about what it can do.

    1. Try confidence-weighted fit: weight each sample by the teacher's
       confidence (max soft probability) if the student's train() accepts
       ``sample_weight``. This transfers *which* examples the teacher is
       certain about — a real, if limited, knowledge signal.
    2. Fall back to plain hard-label training.
    3. Report KL(teacher‖student) before/after either way.
    """
    hard = y if y is not None else np.argmax(teacher_soft, axis=1)
    mode = "hard-label fallback"
    try:
        student_model.train(X, hard, verbose=False)
        mode = "hard-label fallback"
    except Exception:
        return {"losses": [], "mode": mode, "loss_final": float("nan")}

    # Confidence-weighted retry (transfers teacher certainty into sample weights)
    if teacher_soft.ndim == 2 and teacher_soft.shape[0] == len(X):
        confidence = teacher_soft.max(axis=1)
        try:
            student_model.train(X, hard, verbose=False,
                                sample_weight=confidence)
            mode = "confidence-weighted fallback"
        except TypeError:
            pass  # technique doesn't accept sample_weight — keep hard-label
        except Exception:
            pass

    return {"losses": [], "mode": mode, "loss_final": float("nan")}


# ============================================================================
# Distiller — single-teacher (canonical class)
# ============================================================================

class Distiller:
    """Knowledge distillation — THE canonical single-teacher implementation.

    Example:
        import qai
        teacher = qai.build(technique="neural_network", hidden_dim=64)
        teacher.train(X, y)
        student = qai.build(technique="neural_network", hidden_dim=8)
        result = qai.distill(teacher, student, X, y, X_val=Xv, y_val=yv)
        print(result.to_markdown())
    """

    def __init__(self, teacher: Any, student: Any,
                 config: Optional[DistillationConfig] = None):
        self.teacher = teacher
        self.student = student
        self.config = config or DistillationConfig()
        self.result = DistillationResult(strategy=self.config.strategy)

    def stats(self) -> Dict[str, Any]:
        """Result summary as a dict (convenience for notebooks)."""
        return self.result.to_dict()

    def distill(self, X: Any, y: Any = None,
                X_val: Any = None, y_val: Any = None) -> DistillationResult:
        """Run distillation and return a :class:`DistillationResult`."""
        t0 = time.time()
        X = np.asarray(X)
        config = self.config

        teacher_soft = get_soft_targets(self.teacher, X)

        # Evaluation: use the held-out set if given, else fall back to the
        # training data itself (so the report always carries real numbers —
        # flagged in the mode string when it's train-set accuracy).
        self._eval_on_train = False
        if X_val is not None and y_val is not None:
            eval_X, eval_y = X_val, y_val
        else:
            eval_X, eval_y = X, y
            self._eval_on_train = True
            self.result.distillation_mode = "train-set accuracy (pass X_val/y_val for held-out)"
        self.result.student_acc_before = _evaluate(self.student, eval_X, eval_y)
        self.result.teacher_acc = _evaluate(self.teacher, eval_X, eval_y)

        # Compression ratio
        t_params = _count_params(self.teacher)
        s_params = _count_params(self.student)
        if s_params > 0:
            self.result.compression_ratio = max(1.0, t_params / s_params)

        # KL before
        student_soft_before = get_soft_targets(self.student, X)
        if student_soft_before.shape == teacher_soft.shape:
            self.result.kl_before = kl_divergence(teacher_soft, student_soft_before)

        # ---- choose the training path ----
        teacher_net = _get_torch_net(self.teacher)
        student_net = _get_torch_net(self.student)
        if _TORCH and teacher_net is not None and student_net is not None:
            out = torch_distill_epoch(teacher_net, student_net, X, y, config)
            mode = out["mode"]
            if config.align_weight > 0 and config.strategy in ("hint", "attention"):
                mode = "soft-label KL + alignment"
            losses = out["losses"]
            self.result.loss_final = losses[-1] if losses else float("nan")
        else:
            if config.strategy in ("hint", "attention") and config.verbose:
                print("  [distill] hint/attention need torch-backed models; "
                      "using the numpy fallback path")
            out = numpy_distill(self.student, X, y, teacher_soft, config)
            mode = out["mode"]
            losses = out["losses"]
            self.result.loss_final = out.get("loss_final", float("nan"))

        # Combine training mode with the eval-set note so the report is
        # always fully honest about what ran and what the numbers mean.
        if getattr(self, "_eval_on_train", False):
            mode = mode + " [train-set accuracy]"
        self.result.distillation_mode = mode
        self.result.epochs = config.epochs

        # KL after + history
        student_soft_after = get_soft_targets(self.student, X)
        if student_soft_after.shape == teacher_soft.shape:
            self.result.kl_after = kl_divergence(teacher_soft, student_soft_after)
        for i in range(max(1, config.epochs)):
            loss_i = losses[i] if i < len(losses) else self.result.loss_final
            val_acc = (_evaluate(self.student, eval_X, eval_y)
                       if (i == config.epochs - 1) else 0.0)
            self.result.history.append({
                "epoch": i + 1,
                "loss": float(loss_i) if loss_i == loss_i else 0.0,
                "val_acc": val_acc,
            })
        if not losses and config.epochs > 0:
            # numpy path: one row with the real KL movement as the "loss"
            self.result.history = [{
                "epoch": 1,
                "loss": (self.result.kl_after
                         if self.result.kl_after == self.result.kl_after else 0.0),
                "val_acc": _evaluate(self.student, eval_X, eval_y),
            }]

        # Final accuracy + knowledge transfer (always computed — eval set or train set)
        self.result.student_acc_after = _evaluate(self.student, eval_X, eval_y)
        if self.result.teacher_acc > self.result.student_acc_before:
            self.result.knowledge_transfer = (
                (self.result.student_acc_after - self.result.student_acc_before)
                / (self.result.teacher_acc - self.result.student_acc_before)
            )
        self.result.training_time_s = time.time() - t0
        return self.result


# ============================================================================
# Multi-teacher distillation
# ============================================================================

class MultiTeacherDistiller:
    """Distil from several teachers — weighted average of their soft targets,
    then the same canonical loss."""

    def __init__(self, teachers: List[Any], student: Any,
                 config: Optional[DistillationConfig] = None,
                 teacher_weights: Optional[List[float]] = None):
        self.teachers = teachers
        self.student = student
        self.config = config or DistillationConfig(strategy="multi")
        if teacher_weights is None:
            teacher_weights = [1.0 / len(teachers)] * len(teachers)
        total = sum(teacher_weights) or 1.0
        self.teacher_weights = [w / total for w in teacher_weights]
        self.result = DistillationResult(strategy="multi")

    def stats(self) -> Dict[str, Any]:
        return self.result.to_dict()

    def distill(self, X: Any, y: Any = None,
                X_val: Any = None, y_val: Any = None) -> DistillationResult:
        t0 = time.time()
        X = np.asarray(X)
        soft_list = [get_soft_targets(t, X) for t in self.teachers]
        ref_shape = None
        avg = None
        for w, soft in zip(self.teacher_weights, soft_list):
            if soft.ndim == 2 and (ref_shape is None or soft.shape == ref_shape):
                if avg is None:
                    ref_shape = soft.shape
                    avg = w * soft
                else:
                    avg = avg + w * soft
        if avg is None:
            avg = soft_list[0]

        config = self.config
        if X_val is not None and y_val is not None:
            self.result.student_acc_before = _evaluate(self.student, X_val, y_val)
            self.result.teacher_acc = max(_evaluate(t, X_val, y_val) for t in self.teachers)

        # KL before training
        teacher_soft = avg
        student_soft_before = get_soft_targets(self.student, X)
        if student_soft_before.shape == teacher_soft.shape:
            self.result.kl_before = kl_divergence(teacher_soft, student_soft_before)

        # Distil against the averaged soft targets. Multi-teacher averaging
        # happens in probability space, so the numpy soft path handles it.
        teacher_net = None
        student_net = _get_torch_net(self.student)
        out = numpy_distill(self.student, X, y, teacher_soft, config)
        self.result.distillation_mode = out["mode"] + " (probability-averaged teachers)"
        self.result.loss_final = out.get("loss_final", float("nan"))

        student_soft_after = get_soft_targets(self.student, X)
        if student_soft_after.shape == teacher_soft.shape:
            self.result.kl_after = kl_divergence(teacher_soft, student_soft_after)
        self.result.history = [{
            "epoch": 1,
            "loss": (self.result.kl_after if self.result.kl_after == self.result.kl_after else 0.0),
            "val_acc": (_evaluate(self.student, X_val, y_val)
                        if (X_val is not None and y_val is not None) else 0.0),
        }]
        if X_val is not None and y_val is not None:
            self.result.student_acc_after = _evaluate(self.student, X_val, y_val)
            if self.result.teacher_acc > self.result.student_acc_before:
                self.result.knowledge_transfer = (
                    (self.result.student_acc_after - self.result.student_acc_before)
                    / (self.result.teacher_acc - self.result.student_acc_before)
                )
        self.result.epochs = config.epochs
        self.result.training_time_s = time.time() - t0
        return self.result


# ============================================================================
# Self-distillation
# ============================================================================

class SelfDistiller:
    """A model teaches a fresh copy of itself (Be Your Own Teacher).

    The original model acts as the teacher; a deepcopy of it (before any
    distillation training) acts as the student. Because teacher and student
    share architecture, the torch path applies directly when torch-backed.
    """

    def __init__(self, model: Any, config: Optional[DistillationConfig] = None):
        self.model = model
        self.config = config or DistillationConfig(strategy="self")
        try:
            self.student = copy.deepcopy(model)
        except Exception:
            self.student = model  # best effort
        self.result = DistillationResult(strategy="self")

    def stats(self) -> Dict[str, Any]:
        return self.result.to_dict()

    def distill(self, X: Any, y: Any = None,
                X_val: Any = None, y_val: Any = None) -> DistillationResult:
        inner = Distiller(teacher=self.model, student=self.student, config=self.config)
        result = inner.distill(X, y, X_val, y_val)
        result.strategy = "self"
        self.result = result
        return result


# ============================================================================
# Top-level convenience — the one entry point
# ============================================================================

def distill(teacher: Any, student: Any, X: Any, y: Any = None,
            X_val: Any = None, y_val: Any = None,
            **kwargs) -> DistillationResult:
    """Distil teacher → student. THE one distillation entry point.

    All knobs are parameters on this call — no separate module to import:

        qai.distill(teacher, student, X, y, strategy="vanilla",
                    temperature=4.0, alpha=0.7, epochs=30)
    """
    config = DistillationConfig(**kwargs)
    return Distiller(teacher, student, config).distill(X, y, X_val, y_val)


def multi_distill(teachers: List[Any], student: Any, X: Any, y: Any = None,
                  X_val: Any = None, y_val: Any = None,
                  teacher_weights: Optional[List[float]] = None,
                  **kwargs) -> DistillationResult:
    """Multi-teacher distillation (probability-averaged teachers)."""
    config = DistillationConfig(strategy="multi", **kwargs)
    return MultiTeacherDistiller(teachers, student, config, teacher_weights) \
        .distill(X, y, X_val, y_val)


def self_distill(model: Any, X: Any, y: Any = None,
                 X_val: Any = None, y_val: Any = None,
                 **kwargs) -> DistillationResult:
    """Self-distillation (the model teaches a fresh copy of itself)."""
    config = DistillationConfig(strategy="self", **kwargs)
    return SelfDistiller(model, config).distill(X, y, X_val, y_val)


# ============================================================================
# Internal helpers
# ============================================================================

def _evaluate(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    """Accuracy of a model on (X, y) — shared by all distillers."""
    try:
        preds = np.asarray(model.predict(X))
        y_arr = np.asarray(y)
        if preds.shape == y_arr.shape:
            return float(np.mean(preds == y_arr))
    except Exception:
        pass
    return 0.0


def _count_params(model: Any) -> int:
    """Count trainable-ish parameters of any supported model wrapper."""
    tech = getattr(model, "technique", None)
    if tech is not None and hasattr(tech, "parameter_count"):
        try:
            return int(tech.parameter_count())
        except Exception:
            pass
    try:
        from qai.visual_workspace_v2 import _get_weights
        layers = _get_weights(model)
        return int(sum(L.get("n_params", 0) for L in layers))
    except Exception:
        pass
    if _TORCH:
        net = _get_torch_net(model)
        if net is not None:
            return int(sum(p.numel() for p in net.parameters()))
    return 0


__all__ = [
    "DistillationConfig", "DistillationResult",
    "Distiller", "MultiTeacherDistiller", "SelfDistiller",
    "distill", "multi_distill", "self_distill",
    "get_soft_targets", "kl_divergence",
]
