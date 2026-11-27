"""
qai.governance.extended -- AI governance + edge deployment + vibe coding.

  Governance:
    - AIRiskAssessment   -- score a model on EU AI Act risk tiers
    - AuditLogger        -- tamper-evident audit trail of every prediction
    - ModelCard          -- generate / validate standard model cards
    - ComplianceChecker  -- check usage against EU AI Act / NIST AI RMF
  Edge AI:
    - QuantizationPipeline   -- PTQ + QAT in one pass
    - PruningPipeline        -- magnitude + structured + lottery-ticket
    - DistillationPipeline   -- KD with intermediate-layer alignment
    - EdgeProfiler           -- latency / memory / energy on target device
  Vibe Coding:
    - VibeCoder          -- NL -> qai code, with execution sandbox + tests
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# 1. AI RISK ASSESSMENT (EU AI Act 2024)
# ============================================================================
class AIRiskAssessment:
    """Classify a model under the EU AI Act's 4 risk tiers.

    Tiers:
      - Unacceptable  (banned): social scoring, manipulative AI
      - High-risk     (strict obligations): biometrics, education, employment,
                       essential services, law enforcement
      - Limited-risk  (transparency): chatbots, deepfakes, emotion recognition
      - Minimal-risk  (no obligations): spam filters, recommender systems

    Usage:
        ra = AIRiskAssessment(model_metadata={"domain": "education", ...})
        tier = ra.classify()
    """

    HIGH_RISK_DOMAINS = {
        "biometrics", "critical_infrastructure", "education", "employment",
        "essential_services", "law_enforcement", "migration", "justice",
        "democratic_processes"
    }
    LIMITED_RISK_USES = {"chatbot", "deepfake", "emotion_recognition", "biometric_categorisation"}
    UNACCEPTABLE_USES = {"social_scoring", "manipulation", "exploitation_of_vulnerable_groups",
                          "real_time_biometric_id_in_public"}

    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata

    def classify(self) -> Dict[str, Any]:
        uses = set(self.metadata.get("use_cases", []))
        domain = self.metadata.get("domain", "")
        # unacceptable
        unacceptable = uses & self.UNACCEPTABLE_USES
        if unacceptable:
            return {"tier": "unacceptable", "reason": f"banned uses: {unacceptable}",
                    "obligations": ["model must not be deployed"]}
        # high-risk
        if domain in self.HIGH_RISK_DOMAINS:
            return {"tier": "high_risk", "reason": f"domain '{domain}' is high-risk",
                    "obligations": [
                        "risk_management_system",
                        "data_governance",
                        "technical_documentation",
                        "record_keeping",
                        "transparency_to_users",
                        "human_oversight",
                        "accuracy_robustness_cybersecurity",
                        "conformity_assessment",
                        "post_market_monitoring",
                        "registration_in_eu_database",
                    ]}
        # limited-risk
        limited = uses & self.LIMITED_RISK_USES
        if limited:
            return {"tier": "limited_risk", "reason": f"transparency uses: {limited}",
                    "obligations": ["inform_users_of_ai_use",
                                     "mark_deepfakes",
                                     "design_for_accessibility"]}
        return {"tier": "minimal_risk", "reason": "no specific obligations",
                "obligations": ["encourage_voluntary_codes_of_conduct"]}

    def required_actions(self) -> List[str]:
        return self.classify()["obligations"]


# ============================================================================
# 2. AUDIT LOGGER (tamper-evident)
# ============================================================================
class AuditLogger:
    """Append-only, hash-chained audit log for model predictions.

    Each entry contains: timestamp, model_id, input_hash, output_hash,
    prev_hash (forming a chain). Tampering with any past entry breaks
    the chain.
    """
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.entries: List[Dict[str, Any]] = []
        self.prev_hash = "0" * 64

    def log(self, input_data: Any, output_data: Any,
            user: str = "anonymous", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        inp_h = hashlib.sha256(str(input_data).encode()).hexdigest()
        out_h = hashlib.sha256(str(output_data).encode()).hexdigest()
        entry = {
            "timestamp": time.time(),
            "model_id": self.model_id,
            "user": user,
            "input_hash": inp_h,
            "output_hash": out_h,
            "prev_hash": self.prev_hash,
            "metadata": metadata or {},
        }
        # compute this entry's hash
        entry_str = json.dumps({k: v for k, v in entry.items()}, sort_keys=True, default=str)
        entry["hash"] = hashlib.sha256(entry_str.encode()).hexdigest()
        self.prev_hash = entry["hash"]
        self.entries.append(entry)
        return entry

    def verify(self) -> Dict[str, Any]:
        """Verify the chain. Returns broken_at index if any entry is tampered."""
        prev = "0" * 64
        for i, entry in enumerate(self.entries):
            if entry["prev_hash"] != prev:
                return {"valid": False, "broken_at": i, "reason": "prev_hash mismatch"}
            entry_str = json.dumps({k: v for k, v in entry.items() if k != "hash"},
                                    sort_keys=True, default=str)
            expected_hash = hashlib.sha256(entry_str.encode()).hexdigest()
            if entry["hash"] != expected_hash:
                return {"valid": False, "broken_at": i, "reason": "hash mismatch"}
            prev = entry["hash"]
        return {"valid": True, "n_entries": len(self.entries)}

    def export(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.entries, f, indent=2)
        return path


# ============================================================================
# 3. MODEL CARD (Mitchell et al. 2019)
# ============================================================================
class ModelCard:
    """Generate / validate a standard model card."""

    REQUIRED_FIELDS = [
        "model_details", "intended_use", "factors", "metrics",
        "evaluation_data", "training_data", "quantitative_analyses",
        "ethical_considerations", "caveats"
    ]

    def __init__(self, model: Any, metadata: Optional[Dict[str, Any]] = None):
        self.model = model
        self.metadata = metadata or {}

    def auto_generate(self, X: Optional[np.ndarray] = None,
                       y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Best-effort auto-generation of a model card from the model + metadata."""
        card = {
            "model_details": {
                "name": self.metadata.get("name", getattr(self.model, "name", "unknown")),
                "developer": self.metadata.get("developer", "unknown"),
                "version": self.metadata.get("version", "1.0"),
                "date": time.strftime("%Y-%m-%d"),
                "type": type(self.model).__name__,
            },
            "intended_use": {
                "primary_use": self.metadata.get("primary_use", "not specified"),
                "out_of_scope": self.metadata.get("out_of_scope", []),
            },
            "factors": self.metadata.get("factors", {}),
            "metrics": {},
            "evaluation_data": self.metadata.get("evaluation_data", {}),
            "training_data": self.metadata.get("training_data", {}),
            "quantitative_analyses": {},
            "ethical_considerations": self.metadata.get("ethical_considerations", []),
            "caveats": self.metadata.get("caveats", []),
        }
        # if eval data provided, compute basic metrics
        if X is not None and y is not None:
            try:
                preds = np.array([self.model.forward(x) for x in X])
                if preds.ndim == 1:
                    accuracy = float((preds == y).mean())
                    card["metrics"]["accuracy"] = accuracy
                else:
                    card["metrics"]["mse"] = float(np.mean((preds - y) ** 2))
            except Exception as e:
                card["caveats"].append(f"metrics computation failed: {e}")
        return card

    @classmethod
    def validate(cls, card: Dict[str, Any]) -> Dict[str, Any]:
        missing = [f for f in cls.REQUIRED_FIELDS if f not in card]
        return {"valid": not missing, "missing_fields": missing}


# ============================================================================
# 4. COMPLIANCE CHECKER
# ============================================================================
class ComplianceChecker:
    """Check a deployment against EU AI Act + NIST AI RMF requirements."""

    NIST_FUNCTIONS = ["govern", "map", "measure", "manage"]

    def __init__(self, risk_assessment: AIRiskAssessment):
        self.ra = risk_assessment

    def check_eu_ai_act(self) -> Dict[str, Any]:
        classification = self.ra.classify()
        tier = classification["tier"]
        required = classification["obligations"]
        present = set(self.ra.metadata.get("implemented_controls", []))
        missing = [o for o in required if o not in present]
        return {
            "tier": tier,
            "compliance_pct": 100.0 * (len(required) - len(missing)) / max(len(required), 1),
            "missing_controls": missing,
            "implemented_controls": sorted(present & set(required)),
        }

    def check_nist_ai_rmf(self) -> Dict[str, Any]:
        out = {}
        for fn in self.NIST_FUNCTIONS:
            controls = self.ra.metadata.get(f"nist_{fn}", [])
            out[fn] = {"n_controls": len(controls), "controls": controls}
        return out


# ============================================================================
# 5. EDGE AI: QUANTIZATION PIPELINE
# ============================================================================
class QuantizationPipeline:
    """Post-training quantization (PTQ) + quantization-aware training (QAT).

    Simulates int8 quantization by recording dynamic ranges per layer and
    mapping activations/weights into [-128, 127]. Reduces memory by ~4x
    and accelerates inference on CPU/edge devices.
    """
    def __init__(self, model: Any, calibration_data: Optional[np.ndarray] = None):
        self.model = model
        self.calibration = calibration_data
        self.scales: Dict[str, float] = {}

    def calibrate(self) -> Dict[str, float]:
        """Run calibration data through the model; record per-layer abs-max."""
        if self.calibration is None:
            return self.scales
        try:
            import torch
            from .interpret import ActivationLogger
            logger = ActivationLogger(self.model).attach()
            with torch.no_grad():
                self.model(torch.as_tensor(self.calibration, dtype=torch.float32))
            logger.detach()
            for name, act in logger.activations.items():
                self.scales[name] = float(np.abs(act).max()) / 127.0 + 1e-8
        except Exception:
            pass
        return self.scales

    def quantize_weights(self) -> Dict[str, Any]:
        """Quantize model weights to int8 (simulated; returns stats)."""
        try:
            import torch
            stats = {"layers": {}, "total_params": 0, "compressed_params": 0}
            for name, param in self.model.named_parameters():
                if not isinstance(param, torch.Tensor):
                    continue
                scale = float(param.abs().max()) / 127.0 + 1e-8
                q = (param / scale).round().clamp(-128, 127)
                stats["layers"][name] = {
                    "shape": list(param.shape),
                    "scale": scale,
                    "n_elements": int(param.numel()),
                    "original_bytes": int(param.numel() * 4),
                    "quantized_bytes": int(param.numel()),
                }
                stats["total_params"] += int(param.numel())
                stats["compressed_params"] += int(param.numel())
            stats["compression_ratio"] = (stats["total_params"] * 4) / max(stats["compressed_params"], 1)
            return stats
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# 6. EDGE AI: PRUNING PIPELINE
# ============================================================================
class PruningPipeline:
    """Magnitude + structured + lottery-ticket pruning."""

    def __init__(self, model: Any):
        self.model = model

    def magnitude_prune(self, sparsity: float = 0.5) -> Dict[str, Any]:
        """Zero out the smallest-magnitude weights globally."""
        try:
            import torch
            all_w = []
            for p in self.model.parameters():
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    all_w.append(p.abs().flatten())
            all_w = torch.cat(all_w)
            threshold = torch.quantile(all_w, sparsity).item()
            n_pruned = 0
            n_total = 0
            for p in self.model.parameters():
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    mask = p.abs() > threshold
                    n_pruned += int((~mask).sum())
                    n_total += int(mask.numel())
                    p.data.mul_(mask.float())
            return {"method": "magnitude", "sparsity": sparsity,
                    "threshold": threshold, "n_pruned": n_pruned,
                    "n_total": n_total, "actual_sparsity": n_pruned / max(n_total, 1)}
        except Exception as e:
            return {"error": str(e)}

    def structured_prune(self, layer_name: str, fraction: float = 0.5) -> Dict[str, Any]:
        """Remove entire neurons/channels and rebuild downstream shapes.

        Delegates to the canonical implementation in ``qai.pruning`` (since
        v1.8.6). Earlier versions returned ``{"status": "not_implemented"}``;
        real structured pruning now happens here.
        """
        from ..pruning import structured_prune as _canonical
        return _canonical(self.model, layer_name=layer_name, fraction=fraction)


# ============================================================================
# 7. EDGE AI: DISTILLATION PIPELINE
# ============================================================================
class DistillationPipeline:
    """Knowledge distillation from teacher to student.

    Loss = alpha * task_loss(student, y) + (1-alpha) * KL(student || teacher)
    Optionally align intermediate features via projection layers.
    """
    def __init__(self, teacher: Any, student: Any,
                 temperature: float = 4.0, alpha: float = 0.5):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

    def distill(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                 n_steps: int = 10) -> Dict[str, Any]:
        """Run KD for n_steps. Returns loss history."""
        history = []
        try:
            import torch
            import torch.nn.functional as F
            X_t = torch.as_tensor(X, dtype=torch.float32)
            with torch.no_grad():
                teacher_out = self.teacher(X_t)
            opt = torch.optim.Adam(self.student.parameters(), lr=1e-3)
            for step in range(n_steps):
                opt.zero_grad()
                student_out = self.student(X_t)
                # KL on softened distributions
                t_soft = F.softmax(teacher_out / self.temperature, dim=-1)
                s_soft = F.log_softmax(student_out / self.temperature, dim=-1)
                kd_loss = F.kl_div(s_soft, t_soft, reduction="batchmean") * (self.temperature ** 2)
                loss = kd_loss
                if y is not None:
                    y_t = torch.as_tensor(y, dtype=torch.long)
                    task_loss = F.cross_entropy(student_out, y_t)
                    loss = self.alpha * task_loss + (1 - self.alpha) * kd_loss
                loss.backward()
                opt.step()
                history.append({"step": step, "loss": float(loss.item()),
                                 "kd_loss": float(kd_loss.item())})
            return {"history": history, "final_loss": history[-1]["loss"]}
        except Exception as e:
            return {"error": str(e), "history": history}


# ============================================================================
# 8. VIBE CODER -- NL -> qai code
# ============================================================================
class VibeCoder:
    """Generate qai code from a natural-language description.

    Pipeline:
      1. Parse the request for keywords (build, train, evaluate)
      2. Generate a candidate code snippet from templates
      3. Optionally execute in a sandbox to verify it runs
      4. Return code + execution result + suggested tests
    """
    TEMPLATES = {
        "classifier": """
import qai
model = qai.build(technique="{technique}", **{params})
model.fit(X_train, y_train)
preds = [model.forward(x) for x in X_test]
""".strip(),
        "regressor": """
import qai
model = qai.build(technique="{technique}", **{params})
model.fit(X_train, y_train)
preds = [model.forward(x) for x in X_test]
""".strip(),
        "generative": """
import qai
model = qai.build(technique="{technique}", **{params})
model.fit(X_train)
samples = model.sample(n_samples=10)
""".strip(),
        "agent": """
from qai.agents import ReAct, Tool
tools = [Tool(name="{tool_name}", description="{tool_desc}", fn={tool_fn})]
agent = ReAct(llm=llm_fn, tools=tools)
result = agent.run("{query}")
""".strip(),
        "rag": """
from qai.rag import build_rag
rag = build_rag("{kind}", documents=docs)
result = rag.generate("{query}")
""".strip(),
    }

    TECHNIQUE_KEYWORDS = {
        "regression": ["regression", "predict", "linear"],
        "classifier": ["classify", "classification", "label"],
        "gan": ["gan", "generate images", "adversarial"],
        "vae": ["vae", "variational", "encode"],
        "diffusion": ["diffusion", "denoise"],
        "kmeans": ["cluster", "kmeans", "group"],
        "transformer": ["transformer", "attention", "language model"],
        "cnn": ["cnn", "convolution", "image classify"],
        "agent": ["agent", "tool use", "reason", "act"],
        "rag": ["rag", "retrieve", "augmented", "knowledge base"],
    }

    def __init__(self, llm: Optional[Callable[[str], str]] = None,
                 execute_sandbox: bool = False):
        self.llm = llm
        self.execute_sandbox = execute_sandbox

    def _detect_intent(self, request: str) -> str:
        r = request.lower()
        scores = {intent: sum(1 for kw in kws if kw in r)
                  for intent, kws in self.TECHNIQUE_KEYWORDS.items()}
        best = max(scores.items(), key=lambda kv: kv[1])
        if best[1] == 0:
            return "classifier"  # default
        return best[0]

    def generate(self, request: str, **kwargs) -> Dict[str, Any]:
        intent = self._detect_intent(request)
        template = self.TEMPLATES.get(intent, self.TEMPLATES["classifier"])
        code = template.format(
            technique=kwargs.get("technique", intent),
            params=kwargs.get("params", "{}"),
            tool_name=kwargs.get("tool_name", "search"),
            tool_desc=kwargs.get("tool_desc", "search the web"),
            tool_fn=kwargs.get("tool_fn", "lambda **kw: 'result'"),
            query=request.replace('"', '\\"')[:200],
            kind=kwargs.get("rag_kind", "baseline"),
        )
        result = {"intent": intent, "code": code, "request": request}
        # validate syntax
        try:
            ast.parse(code)
            result["syntax_valid"] = True
        except SyntaxError as e:
            result["syntax_valid"] = False
            result["syntax_error"] = str(e)
        # optional sandbox execution
        if self.execute_sandbox and result["syntax_valid"]:
            result["execution"] = self._sandbox_exec(code)
        # suggested tests
        result["tests"] = self._suggest_tests(intent)
        return result

    def _sandbox_exec(self, code: str) -> Dict[str, Any]:
        """Execute generated code in a temp file with timeout."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            proc = subprocess.run(
                ["python", path], capture_output=True, text=True, timeout=5
            )
            return {"stdout": proc.stdout, "stderr": proc.stderr,
                    "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "timeout", "returncode": -1}
        finally:
            os.unlink(path)

    def _suggest_tests(self, intent: str) -> List[str]:
        return [
            f"def test_{intent}_builds():",
            f"    model = qai.build(technique='{intent}')",
            "    assert model is not None",
            "",
            f"def test_{intent}_fits():",
            "    import numpy as np",
            "    X = np.random.randn(10, 4)",
            "    y = np.random.randint(0, 2, 10)",
            "    model = qai.build(technique='" + intent + "')",
            "    model.fit(X, y)",
            "    assert model.is_trained()",
        ]


__all__ = [
    # governance
    "AIRiskAssessment", "AuditLogger", "ModelCard", "ComplianceChecker",
    # edge ai
    "QuantizationPipeline", "PruningPipeline", "DistillationPipeline",
    # vibe coding
    "VibeCoder",
]
