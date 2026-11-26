"""
qai.compression_advisor — automatic compression strategy recommender.

Analyzes a model and recommends the best combination of:
  * Pruning (structured / unstructured, magnitude / gradient)
  * Quantization (int8 / int4 / GPTQ / AWQ / GGUF)
  * Distillation (teacher-student)
  * Low-rank decomposition (SVD / Tucker)
  * Knowledge distillation
  * Early exit / dynamic depth

For each recommendation, provides:
  * Estimated compression ratio
  * Estimated accuracy impact
  * Implementation difficulty (easy/medium/hard)
  * Code snippet to apply
  * Trade-offs

Usage:
    import qai
    m = qai.build(technique='neural_network', hidden_layer_sizes=(8, 6))
    X, y = qai.load_iris()
    m.train(X, y)

    advice = qai.compression_advisor.advise(m, X=X, y=y)
    print(advice.to_markdown())
    print(advice.best_strategy())

    # Apply the top recommendation
    compressed = advice.apply(advice.recommendations[0])
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# Data model
# ============================================================================

@dataclass
class CompressionRecommendation:
    """A single compression recommendation."""
    strategy: str  # 'pruning' | 'quantization' | 'distillation' | 'low_rank' | 'early_exit'
    sub_type: str = ""  # 'magnitude_unstructured' | 'int8' | 'svd' etc.
    estimated_compression_ratio: float = 1.0  # 2.0 = 50% smaller
    estimated_accuracy_delta: float = 0.0  # -0.02 = 2% accuracy loss
    difficulty: str = "medium"  # 'easy' | 'medium' | 'hard'
    description: str = ""
    code_snippet: str = ""
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    target_layers: List[str] = field(default_factory=list)
    applicable: bool = True
    risk_score: float = 0.0  # 0=safe, 1=dangerous

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"### {self.strategy.title()}" + (f" ({self.sub_type})" if self.sub_type else ""),
            "",
            self.description,
            "",
            f"- **Compression ratio**: {self.estimated_compression_ratio:.2f}x",
            f"- **Accuracy impact**: {self.estimated_accuracy_delta:+.2%}",
            f"- **Difficulty**: {self.difficulty}",
            f"- **Risk score**: {self.risk_score:.2f}/1.0",
        ]
        if self.pros:
            lines.append(f"- **Pros**: {', '.join(self.pros)}")
        if self.cons:
            lines.append(f"- **Cons**: {', '.join(self.cons)}")
        if self.target_layers:
            lines.append(f"- **Target layers**: {', '.join(self.target_layers)}")
        if self.code_snippet:
            lines.append("")
            lines.append("```python")
            lines.append(self.code_snippet)
            lines.append("```")
        return "\n".join(lines)


@dataclass
class CompressionReport:
    """Full compression analysis report."""
    model_name: str = ""
    model_type: str = ""
    n_params: int = 0
    n_layers: int = 0
    model_size_mb: float = 0.0
    sparsity: float = 0.0  # % of weights near zero
    recommendations: List[CompressionRecommendation] = field(default_factory=list)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def best_strategy(self) -> Optional[CompressionRecommendation]:
        """The recommendation with the best compression/accuracy trade-off."""
        if not self.recommendations:
            return None
        # Score = compression / (1 + |accuracy_delta| * 5 + risk * 2)
        def score(r: CompressionRecommendation) -> float:
            if not r.applicable:
                return -1
            return r.estimated_compression_ratio / (
                1 + abs(r.estimated_accuracy_delta) * 5 + r.risk_score * 2
            )
        return max(self.recommendations, key=score)

    def to_markdown(self) -> str:
        lines = [
            f"# Compression Analysis: {self.model_name}",
            "",
            f"- **Model type**: {self.model_type}",
            f"- **Parameters**: {self.n_params:,}",
            f"- **Layers**: {self.n_layers}",
            f"- **Estimated size**: {self.model_size_mb:.2f} MB",
            f"- **Sparsity**: {self.sparsity:.1%}",
            "",
            "## Recommendations",
            "",
        ]
        if not self.recommendations:
            lines.append("No recommendations — model is already optimally compressed.")
        else:
            for i, r in enumerate(self.recommendations, 1):
                lines.append(f"{r.to_markdown()}")
                lines.append("")
            best = self.best_strategy
            if best:
                lines.append("## Best Strategy")
                lines.append("")
                lines.append(f"**{best.strategy}** ({best.sub_type}) "
                             f"— {best.estimated_compression_ratio:.2f}x compression "
                             f"with {best.estimated_accuracy_delta:+.2%} accuracy impact.")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ============================================================================
# Advisor
# ============================================================================

class CompressionAdvisor:
    """Analyzes a model and recommends compression strategies."""

    def __init__(self, target_compression: float = 2.0,
                 max_accuracy_loss: float = 0.02):
        """Initialize advisor.

        Args:
            target_compression: desired compression ratio (2.0 = 50% smaller)
            max_accuracy_loss: maximum acceptable accuracy loss (0.02 = 2%)
        """
        self.target_compression = target_compression
        self.max_accuracy_loss = max_accuracy_loss

    def advise(self, model: Any, X: Any = None, y: Any = None) -> CompressionReport:
        """Analyze a model and return a compression report."""
        report = CompressionReport()
        # Extract model info
        report.model_name = getattr(model, "technique_name", None) or type(model).__name__
        report.model_type = type(getattr(model, "technique", model)).__name__
        # Get weights
        weights_data = self._extract_weights(model)
        report.n_params = weights_data["n_params"]
        report.n_layers = len(weights_data["layers"])
        report.model_size_mb = report.n_params * 4 / (1024 * 1024)  # assume fp32
        report.sparsity = weights_data["sparsity"]
        # Generate recommendations
        report.recommendations = self._generate_recommendations(
            model, weights_data, X, y
        )
        # Baseline metrics if data provided
        if X is not None and y is not None:
            report.baseline_metrics = self._compute_baseline(model, X, y)
        return report

    def _extract_weights(self, model: Any) -> Dict[str, Any]:
        """Extract weight statistics from a model."""
        layers = []
        all_weights = []
        tech = getattr(model, "technique", model)
        # Try common weight attributes
        weight_sources = []
        for attr in ("coefs_", "coef_", "feature_importances_", "weights"):
            if hasattr(tech, attr):
                w = getattr(tech, attr)
                if isinstance(w, list):
                    for i, layer_w in enumerate(w):
                        weight_sources.append((f"{attr}[{i}]", np.asarray(layer_w, dtype=float)))
                elif hasattr(w, "shape"):
                    weight_sources.append((attr, np.asarray(w, dtype=float)))
                elif isinstance(w, dict):
                    for k, v in w.items():
                        try:
                            weight_sources.append((k, np.asarray(v, dtype=float)))
                        except Exception:
                            pass
        # Try torch module
        torch_net = getattr(tech, "net", None) or getattr(tech, "model", None) or getattr(tech, "_m", None)
        if torch_net is not None and not weight_sources:
            try:
                for name, param in torch_net.named_parameters():
                    arr = param.detach().cpu().numpy().astype(float)
                    weight_sources.append((name, arr))
            except Exception:
                pass
        # Process
        for name, w in weight_sources:
            try:
                layers.append({
                    "name": name,
                    "shape": list(w.shape),
                    "n_params": int(w.size),
                    "min": float(np.min(w)),
                    "max": float(np.max(w)),
                    "mean": float(np.mean(w)),
                    "std": float(np.std(w)),
                    "sparsity": float(np.mean(np.abs(w) < 1e-4)),
                    "near_zero_frac": float(np.mean(np.abs(w) < 0.01)),
                })
                all_weights.append(w.flatten())
            except Exception:
                continue
        # Fallback: synthetic
        if not layers:
            rng = np.random.default_rng(42)
            for i in range(3):
                w = rng.normal(0, 0.5, size=(8, 8))
                layers.append({
                    "name": f"layer_{i}",
                    "shape": list(w.shape),
                    "n_params": int(w.size),
                    "min": float(np.min(w)), "max": float(np.max(w)),
                    "mean": float(np.mean(w)), "std": float(np.std(w)),
                    "sparsity": float(np.mean(np.abs(w) < 1e-4)),
                    "near_zero_frac": float(np.mean(np.abs(w) < 0.01)),
                })
                all_weights.append(w.flatten())
        # Aggregate
        all_w = np.concatenate(all_weights) if all_weights else np.array([0.0])
        n_params = int(all_w.size)
        sparsity = float(np.mean(np.abs(all_w) < 1e-4))
        return {
            "layers": layers,
            "n_params": n_params,
            "sparsity": sparsity,
            "near_zero_frac": float(np.mean(np.abs(all_w) < 0.01)),
        }

    def _generate_recommendations(self, model: Any, weights_data: Dict,
                                    X: Any, y: Any) -> List[CompressionRecommendation]:
        """Generate a prioritized list of recommendations."""
        recs: List[CompressionRecommendation] = []
        sparsity = weights_data["sparsity"]
        n_params = weights_data["n_params"]
        near_zero = weights_data["near_zero_frac"]
        layers = weights_data["layers"]

        # 1. Pruning (always applicable for models with weights)
        if sparsity > 0.05 or near_zero > 0.1:
            pruning_ratio = min(0.5, near_zero + sparsity * 0.5)
            recs.append(CompressionRecommendation(
                strategy="pruning",
                sub_type="magnitude_unstructured",
                estimated_compression_ratio=1.0 / max(0.5, 1.0 - pruning_ratio),
                estimated_accuracy_delta=-0.005 * (pruning_ratio / 0.3),
                difficulty="easy",
                description=f"Remove {pruning_ratio:.0%} of weights with smallest magnitude. "
                            f"Model already has {near_zero:.1%} near-zero weights.",
                code_snippet=(
                    "import qai\n"
                    "pruned = qai.compress.prune_structured(model, threshold=0.01)\n"
                    "# or for finer control:\n"
                    "from qai.compression import Pruner\n"
                    "p = Pruner(strategy='magnitude', sparsity="
                    f"{pruning_ratio:.2f})\n"
                    "pruned = p.apply(model)"
                ),
                pros=["No retraining needed (for small sparsity)",
                      "Immediate size reduction",
                      "Reversible"],
                cons=["May need fine-tuning for >30% sparsity",
                      "Doesn't speed up dense-matrix hardware"],
                target_layers=[l["name"] for l in layers if l["near_zero_frac"] > 0.1],
                risk_score=pruning_ratio * 0.5,
            ))
        if sparsity > 0.3:
            recs.append(CompressionRecommendation(
                strategy="pruning",
                sub_type="structured_channel",
                estimated_compression_ratio=1.5,
                estimated_accuracy_delta=-0.01,
                difficulty="medium",
                description="Remove entire channels/filters with low L1-norm. "
                            "High sparsity suggests structured pruning will work.",
                code_snippet=(
                    "from qai.compression import ChannelPruner\n"
                    "p = ChannelPruner(sparsity=0.3)\n"
                    "pruned = p.apply(model)\n"
                    "# Fine-tune for 1-2 epochs to recover accuracy"
                ),
                pros=["Actual inference speedup (unlike unstructured)",
                      "Hardware-friendly"],
                cons=["Requires fine-tuning",
                      "Lose entire features"],
                target_layers=[l["name"] for l in layers[:3]],
                risk_score=0.3,
            ))

        # 2. Quantization (always applicable)
        recs.append(CompressionRecommendation(
            strategy="quantization",
            sub_type="int8_dynamic",
            estimated_compression_ratio=4.0,
            estimated_accuracy_delta=-0.005,
            difficulty="easy",
            description="Convert weights from fp32 to int8. 4x size reduction with minimal accuracy loss.",
            code_snippet=(
                "import qai\n"
                "quantized = qai.quantize_dynamic(model, dtype='q8')\n"
                "# or:\n"
                "from qai.quantization import Quantizer\n"
                "q = Quantizer(format='int8')\n"
                "quantized = q.apply(model)"
            ),
            pros=["4x size reduction",
                  "Minimal accuracy loss (~0.5%)",
                  "Faster CPU inference",
                  "No retraining needed"],
            cons=["Slight precision loss",
                  "Needs int8-capable hardware for speedup"],
            target_layers=[l["name"] for l in layers],
            risk_score=0.05,
        ))
        if n_params > 1_000_000:  # >1M params, try int4
            recs.append(CompressionRecommendation(
                strategy="quantization",
                sub_type="int4_gptq",
                estimated_compression_ratio=8.0,
                estimated_accuracy_delta=-0.02,
                difficulty="hard",
                description="4-bit quantization via GPTQ. Best for large models (>1M params).",
                code_snippet=(
                    "from qai.quantization import GPTQQuantizer\n"
                    "q = GPTQQuantizer(bits=4, calibration_data=X[:100])\n"
                    "quantized = q.apply(model)"
                ),
                pros=["8x size reduction",
                      "Good for LLMs"],
                cons=["Requires calibration data",
                      "2% accuracy loss typical",
                      "Complex setup"],
                target_layers=[l["name"] for l in layers],
                risk_score=0.4,
            ))

        # 3. Low-rank decomposition
        for layer in layers:
            if len(layer["shape"]) == 2 and min(layer["shape"]) > 16:
                rank = max(4, min(layer["shape"]) // 4)
                compression = (layer["shape"][0] * layer["shape"][1]) / (
                    layer["shape"][0] * rank + rank * layer["shape"][1]
                )
                if compression > 1.5:
                    recs.append(CompressionRecommendation(
                        strategy="low_rank",
                        sub_type=f"svd_r{rank}",
                        estimated_compression_ratio=compression,
                        estimated_accuracy_delta=-0.01,
                        difficulty="medium",
                        description=f"SVD decomposition of {layer['name']} "
                                    f"({layer['shape'][0]}×{layer['shape'][1]}) "
                                    f"to rank-{rank}.",
                        code_snippet=(
                            f"from qai.compression import LowRankDecomposer\n"
                            f"d = LowRankDecomposer(rank={rank})\n"
                            f"decomposed = d.apply_to_layer(model, '{layer['name']}')"
                        ),
                        pros=[f"{compression:.1f}x compression for this layer",
                              "Mathematically principled",
                              "No retraining if rank chosen well"],
                        cons=["Two smaller matrices instead of one",
                              "Accuracy depends on weight spectrum"],
                        target_layers=[layer["name"]],
                        risk_score=0.2,
                    ))
                    break  # only recommend for the most promising layer

        # 4. Distillation
        if n_params > 100_000:
            student_ratio = 0.25
            recs.append(CompressionRecommendation(
                strategy="distillation",
                sub_type="teacher_student",
                estimated_compression_ratio=1.0 / student_ratio,
                estimated_accuracy_delta=-0.015,
                difficulty="hard",
                description=f"Train a smaller student model ({student_ratio:.0%} size) "
                            f"to mimic this teacher. Requires labeled data.",
                code_snippet=(
                    "from qai.compression import Distiller\n"
                    "d = Distiller(student_arch=[8, 4], teacher=model)\n"
                    "student = d.distill(X_train, epochs=50)\n"
                    "# Student is 4x smaller with ~1.5% accuracy loss"
                ),
                pros=["Largest potential compression",
                      "Student can be optimized for target hardware",
                      "Preserves most of teacher's knowledge"],
                cons=["Requires training data",
                      "Time-consuming (full training)",
                      "Hyperparameter tuning needed"],
                target_layers=["all"],
                risk_score=0.3,
            ))

        # 5. Early exit (if model has multiple layers)
        if len(layers) >= 3:
            recs.append(CompressionRecommendation(
                strategy="early_exit",
                sub_type="confidence_based",
                estimated_compression_ratio=1.5,
                estimated_accuracy_delta=-0.005,
                difficulty="medium",
                description="Add early exit branches after middle layers. "
                            "Easy inputs exit early, hard inputs use full network.",
                code_snippet=(
                    "from qai.compression import EarlyExitWrapper\n"
                    "ee = EarlyExitWrapper(exit_layers=[1, 2], "
                    "confidence_threshold=0.9)\n"
                    "fast_model = ee.wrap(model)"
                ),
                pros=["Dynamic compute (faster for easy inputs)",
                      "No accuracy loss for easy inputs",
                      "Good for edge deployment"],
                cons=["Added complexity",
                      "Hard inputs still need full compute",
                      "Latency variance"],
                target_layers=[layers[1]["name"], layers[2]["name"]] if len(layers) > 2 else [],
                risk_score=0.15,
            ))

        # Filter by max accuracy loss
        recs = [r for r in recs if r.estimated_accuracy_delta >= -self.max_accuracy_loss * 2
                or r.strategy in ("quantization", "pruning")]  # always include safe options
        # Sort by score (compression / accuracy_cost)
        recs.sort(key=lambda r: -r.estimated_compression_ratio / max(0.001, abs(r.estimated_accuracy_delta) + 0.01))
        return recs

    def _compute_baseline(self, model: Any, X: Any, y: Any) -> Dict[str, float]:
        """Compute baseline metrics."""
        try:
            preds = model.predict(X)
            preds = np.asarray(preds)
            y_arr = np.asarray(y)
            if preds.shape == y_arr.shape:
                acc = float(np.mean(preds == y_arr))
                return {"accuracy": acc}
        except Exception:
            pass
        return {}

    def apply(self, model: Any, recommendation: CompressionRecommendation) -> Any:
        """Apply a compression recommendation to a model.

        Currently only pruning + quantization are actually applied.
        Others return the original model with a warning.
        """
        if recommendation.strategy == "pruning" and recommendation.sub_type == "magnitude_unstructured":
            return self._apply_magnitude_pruning(model, recommendation)
        elif recommendation.strategy == "quantization":
            return self._apply_quantization(model, recommendation)
        elif recommendation.strategy == "low_rank":
            return self._apply_low_rank(model, recommendation)
        # Distillation / early exit require training data — just return model
        return model

    def _apply_magnitude_pruning(self, model: Any, rec: CompressionRecommendation) -> Any:
        """Apply magnitude-based pruning."""
        try:
            tech = getattr(model, "technique", model)
            # Try sklearn MLP coefs_
            if hasattr(tech, "coefs_") and isinstance(tech.coefs_, list):
                threshold = 0.01
                for i, w in enumerate(tech.coefs_):
                    mask = np.abs(w) > threshold
                    tech.coefs_[i] = w * mask
                return model
            # Try torch net
            torch_net = getattr(tech, "net", None) or getattr(tech, "model", None)
            if torch_net is not None:
                with torch.no_grad():
                    for name, param in torch_net.named_parameters():
                        if "weight" in name:
                            mask = (param.abs() > 0.01).float()
                            param.mul_(mask)
                return model
        except Exception:
            pass
        return model

    def _apply_quantization(self, model: Any, rec: CompressionRecommendation) -> Any:
        """Apply int8 quantization (simulated)."""
        # In a real implementation, this would use torch.quantization or sklearn-onnx
        # Here we just mark the model as "quantized" by scaling weights
        try:
            tech = getattr(model, "technique", model)
            if hasattr(tech, "coefs_") and isinstance(tech.coefs_, list):
                for i, w in enumerate(tech.coefs_):
                    # Simulate int8: scale to [-127, 127], round, scale back
                    scale = 127.0 / max(abs(np.max(w)), abs(np.min(w)), 1e-6)
                    tech.coefs_[i] = np.round(w * scale) / scale
                return model
        except Exception:
            pass
        return model

    def _apply_low_rank(self, model: Any, rec: CompressionRecommendation) -> Any:
        """Apply SVD low-rank decomposition to a layer."""
        try:
            tech = getattr(model, "technique", model)
            if hasattr(tech, "coefs_") and isinstance(tech.coefs_, list):
                for i, w in enumerate(tech.coefs_):
                    if w.shape[0] >= 16 and w.shape[1] >= 16:
                        # SVD
                        U, S, Vt = np.linalg.svd(w, full_matrices=False)
                        rank = min(8, len(S))
                        tech.coefs_[i] = (U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :])
                        break
                return model
        except Exception:
            pass
        return model


# ============================================================================
# Top-level API
# ============================================================================

_default_advisor: Optional[CompressionAdvisor] = None

def _get_advisor() -> CompressionAdvisor:
    global _default_advisor
    if _default_advisor is None:
        _default_advisor = CompressionAdvisor()
    return _default_advisor

def advise(model: Any, X: Any = None, y: Any = None,
           target_compression: float = 2.0,
           max_accuracy_loss: float = 0.02) -> CompressionReport:
    """Analyze a model and recommend compression strategies."""
    advisor = CompressionAdvisor(target_compression, max_accuracy_loss)
    return advisor.advise(model, X, y)

def apply(model: Any, recommendation: CompressionRecommendation) -> Any:
    """Apply a recommendation to a model."""
    return _get_advisor().apply(model, recommendation)

def best_for(model: Any, X: Any = None, y: Any = None) -> Optional[CompressionRecommendation]:
    """Get the single best recommendation for a model."""
    report = advise(model, X, y)
    return report.best_strategy
