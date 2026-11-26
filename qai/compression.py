"""
qai.compression -- model compression toolkit: quantize + prune + distill.

  CompressionPipeline — end-to-end compression in one call
  Quantizer           — post-training quantization (int8/int4/fp16)
  Pruner              — structured + unstructured pruning
  Distiller           — knowledge distillation from teacher to student
  Compressor          — combine all three for maximum compression
  CompressionReport   — report on compression ratio + accuracy tradeoff
"""
from __future__ import annotations
import copy
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


class Quantizer:
    """Post-training quantization for neural networks.

    Supports: int8, int4, fp16, dynamic quantization
    """

    @staticmethod
    def quantize_int8(model: Any) -> Dict[str, Any]:
        """Quantize model weights to int8 (simulated)."""
        if not _TORCH:
            raise ImportError("torch required for quantization")
        stats = {"layers": {}, "original_bytes": 0, "quantized_bytes": 0}
        if hasattr(model, "state_dict"):
            sd = model.state_dict()
            for k, v in sd.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    orig_bytes = v.numel() * 4  # float32
                    scale = v.abs().max() / 127.0 + 1e-8
                    q = (v / scale).round().clamp(-128, 127)
                    stats["layers"][k] = {
                        "shape": list(v.shape),
                        "scale": float(scale),
                        "orig_bytes": orig_bytes,
                        "quant_bytes": int(v.numel()),  # int8 = 1 byte
                    }
                    stats["original_bytes"] += orig_bytes
                    stats["quantized_bytes"] += int(v.numel())
        return {
            "method": "int8",
            "stats": stats,
            "compression_ratio": stats["original_bytes"] / max(stats["quantized_bytes"], 1),
        }

    @staticmethod
    def quantize_fp16(model: Any) -> Dict[str, Any]:
        """Quantize model to float16."""
        if not _TORCH:
            raise ImportError("torch required")
        if hasattr(model, "half"):
            model.half()
            return {"method": "fp16", "applied": True, "size_reduction": "50%"}
        return {"method": "fp16", "applied": False}

    @staticmethod
    def dynamic_quantize(model: Any) -> Any:
        """Apply torch dynamic quantization."""
        if not _TORCH:
            raise ImportError("torch required")
        try:
            quantized = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            return quantized
        except Exception as e:
            return {"error": str(e)}


class Pruner:
    """Model pruning: structured + unstructured."""

    @staticmethod
    def magnitude_prune(model: Any, sparsity: float = 0.5) -> Dict[str, Any]:
        """Unstructured magnitude pruning (zero out small weights)."""
        if not _TORCH or not hasattr(model, "parameters"):
            return {"error": "requires torch model"}
        total_params = 0
        pruned_params = 0
        all_weights = []
        for p in model.parameters():
            if p.requires_grad:
                all_weights.append(p.abs().flatten())
        all_w = torch.cat(all_weights)
        threshold = torch.quantile(all_w, sparsity).item()
        for p in model.parameters():
            if p.requires_grad:
                mask = p.abs() > threshold
                total_params += p.numel()
                pruned_params += int((~mask).sum())
                p.data.mul_(mask.float())
        return {
            "method": "magnitude",
            "sparsity": sparsity,
            "threshold": threshold,
            "total_params": total_params,
            "pruned_params": pruned_params,
            "actual_sparsity": pruned_params / max(total_params, 1),
        }

    @staticmethod
    def structured_prune(model: Any, layer_name: str,
                          fraction: float = 0.3) -> Dict[str, Any]:
        """Structured pruning: remove entire channels."""
        from .surgery import ChannelPruner
        return ChannelPruner.prune(model, layer_name, fraction)

    @staticmethod
    def lottery_ticket(model: Any, X: np.ndarray, y: np.ndarray,
                         sparsity: float = 0.5) -> Dict[str, Any]:
        """Find a lottery ticket (sparse subnetwork that still works)."""
        from .surgery import WeightProbing
        return WeightProbing.lottery_ticket_check(model, X, y, sparsity)


class Distiller:
    """Knowledge distillation: train a student to mimic a teacher."""

    @staticmethod
    def distill(teacher: Any, student: Any, X: np.ndarray, y: np.ndarray,
                 temperature: float = 4.0, alpha: float = 0.5,
                 n_steps: int = 50) -> Dict[str, Any]:
        """Run knowledge distillation."""
        if not _TORCH:
            raise ImportError("torch required for distillation")
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)
        with torch.no_grad():
            teacher_out = teacher(X_t) if isinstance(teacher, nn.Module) else \
                            torch.as_tensor(np.array([teacher.forward(x) for x in X]))
        opt = torch.optim.Adam(student.parameters(), lr=1e-3)
        history = []
        for step in range(n_steps):
            opt.zero_grad()
            s_out = student(X_t) if isinstance(student, nn.Module) else \
                     torch.as_tensor(np.array([student.forward(x) for x in X]))
            t_soft = F.softmax(teacher_out / temperature, dim=-1) if isinstance(teacher_out, torch.Tensor) else \
                       torch.as_tensor(teacher_out)
            s_log = F.log_softmax(s_out / temperature, dim=-1) if isinstance(s_out, torch.Tensor) else \
                      torch.as_tensor(s_out)
            kd_loss = F.kl_div(s_log, t_soft, reduction="batchmean") * (temperature ** 2)
            task_loss = F.cross_entropy(s_out, y_t) if isinstance(s_out, torch.Tensor) else torch.tensor(0.0)
            loss = alpha * task_loss + (1 - alpha) * kd_loss
            loss.backward()
            opt.step()
            history.append({"step": step, "loss": float(loss)})
        return {"history": history, "final_loss": history[-1]["loss"]}


class CompressionPipeline:
    """End-to-end compression: quantize + prune + distill in one call.

    Usage:
        pipe = CompressionPipeline(sparsity=0.5, quantize="int8", distill=True)
        result = pipe.compress(model, X, y, teacher_model)
        print(result["compression_ratio"], result["accuracy_loss"])
    """

    def __init__(self, sparsity: float = 0.3, quantize: str = "int8",
                 distill: bool = False, distill_steps: int = 50):
        self.sparsity = sparsity
        self.quantize = quantize
        self.distill = distill
        self.distill_steps = distill_steps

    def compress(self, model: Any, X: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 teacher: Optional[Any] = None) -> Dict[str, Any]:
        """Run the full compression pipeline."""
        results: Dict[str, Any] = {"steps": []}

        # 1. Prune
        if self.sparsity > 0:
            prune_result = Pruner.magnitude_prune(model, self.sparsity)
            results["steps"].append({"step": "prune", **prune_result})
            results["prune"] = prune_result

        # 2. Distill (if teacher provided)
        if self.distill and teacher is not None and X is not None and y is not None:
            distill_result = Distiller.distill(teacher, model, X, y,
                                                 n_steps=self.distill_steps)
            results["steps"].append({"step": "distill", "final_loss": distill_result["final_loss"]})
            results["distill"] = distill_result

        # 3. Quantize
        if self.quantize == "int8":
            quant_result = Quantizer.quantize_int8(model)
            results["steps"].append({"step": "quantize", **quant_result})
            results["quantize"] = quant_result
        elif self.quantize == "fp16":
            quant_result = Quantizer.quantize_fp16(model)
            results["steps"].append({"step": "quantize", **quant_result})

        # 4. Compute overall compression ratio
        total_ratio = 1.0
        if "prune" in results:
            total_ratio *= 1 / (1 - results["prune"].get("actual_sparsity", 0) + 0.01)
        if "quantize" in results:
            total_ratio *= results["quantize"].get("compression_ratio", 1.0)
        results["total_compression_ratio"] = total_ratio
        results["model"] = model
        return results


class CompressionReport:
    """Generate a report on compression results."""

    @staticmethod
    def generate(result: Dict[str, Any]) -> str:
        lines = ["Model Compression Report", "=" * 40]
        for step in result.get("steps", []):
            lines.append(f"\n{step['step'].upper()}:")
            for k, v in step.items():
                if k != "step":
                    lines.append(f"  {k}: {v}")
        lines.append(f"\nTotal compression ratio: {result.get('total_compression_ratio', 1.0):.2f}x")
        return "\n".join(lines)


# Need F for distillation
try:
    import torch.nn.functional as F
except ImportError:
    F = None


__all__ = [
    "Quantizer", "Pruner", "Distiller",
    "CompressionPipeline", "CompressionReport",
]
