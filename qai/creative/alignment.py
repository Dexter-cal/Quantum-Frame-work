"""
qai.creative.alignment -- Creative Ideas 5, 6 & 8

Idea 5 — MORAL COMPASS
    An ethical guardrail that scores model outputs against a set of
    constitutional principles (helpful, harmless, honest) and rewrites
    outputs that violate them.

Idea 6 — MODEL PERSONALITY
    Attach a 5-factor personality trait vector (OCEAN) to a model. Traits
    bias sampling temperature, response verbosity, and risk tolerance.

Idea 8 — CAUSAL REASONING MODULE
    Do-calculus + counterfactual estimation over a given DAG. Wraps
    Pearl-style structural causal models.
"""
from __future__ import annotations
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
import numpy as np


# ============================================================================
# Idea 5 — MORAL COMPASS
# ============================================================================
class MoralPrinciple:
    """A single ethical principle with a description + scoring function."""

    def __init__(self, name: str, description: str,
                 score_fn: Callable[[str], float]):
        """score_fn returns float in [-1, 1]; negative means violation."""
        self.name = name
        self.description = description
        self.score_fn = score_fn

    def evaluate(self, text: str) -> float:
        try:
            s = float(self.score_fn(text))
            return max(-1.0, min(1.0, s))
        except Exception:
            return 0.0


_DEFAULT_PRINCIPLES = [
    MoralPrinciple(
        "harmlessness", "Outputs must not encourage violence, self-harm, or harm to others.",
        score_fn=lambda t: -1.0 if any(w in t.lower() for w in
            ["kill", "hurt", "suicide", "violence", "weapon", "bomb"]) else 1.0,
    ),
    MoralPrinciple(
        "honesty", "Outputs must not make verifiably false claims about safety-critical topics.",
        score_fn=lambda t: -1.0 if any(w in t.lower() for w in
            ["100% safe", "guaranteed cure", "definitely no side effects"]) else 1.0,
    ),
    MoralPrinciple(
        "helpfulness", "Outputs should be reasonably specific and address the request.",
        score_fn=lambda t: 1.0 if len(t.split()) >= 5 else -0.5,
    ),
    MoralPrinciple(
        "privacy", "Outputs must not reveal PII or sensitive personal information.",
        score_fn=lambda t: -1.0 if any(w in t.lower() for w in
            ["social security", "credit card", "password"]) else 1.0,
    ),
    MoralPrinciple(
        "respect", "Outputs must not demean groups based on identity.",
        score_fn=lambda t: -1.0 if any(w in t.lower() for w in
            ["stupid", "inferior race", "they all are"]) else 1.0,
    ),
]


class MoralCompass:
    """Wrap a model with a moral scoring layer."""

    def __init__(self, model: Optional[Any] = None,
                 principles: Optional[List[MoralPrinciple]] = None,
                 threshold: float = 0.0):
        self.model = model
        self.principles = principles or _DEFAULT_PRINCIPLES
        self.threshold = threshold
        self.violation_log: List[Dict[str, Any]] = []

    def score(self, text: str) -> Dict[str, Any]:
        scores = {p.name: p.evaluate(text) for p in self.principles}
        total = float(np.mean(list(scores.values())))
        return {
            "text": text,
            "per_principle": scores,
            "overall": total,
            "violations": [n for n, s in scores.items() if s < self.threshold],
        }

    def guard(self, text: str) -> Dict[str, Any]:
        """Score and rewrite output if it violates any principle."""
        result = self.score(text)
        if result["violations"]:
            self.violation_log.append(result)
            rewrite = self._rewrite(text, result["violations"])
            return {
                "original": text,
                "blocked": True,
                "violations": result["violations"],
                "rewrite": rewrite,
                "score": result["overall"],
            }
        return {"original": text, "blocked": False, "rewrite": text, "score": result["overall"]}

    def _rewrite(self, text: str, violations: List[str]) -> str:
        """Simple rule-based rewriter for safety violations."""
        rewrite = text
        if "harmlessness" in violations:
            rewrite = "I can't help with that. If you're struggling, please reach out to a professional."
        elif "privacy" in violations:
            rewrite = "[redacted: contains sensitive personal information]"
        elif "honesty" in violations:
            rewrite = "I cannot guarantee that claim. Please consult authoritative sources."
        elif "respect" in violations:
            rewrite = "Let me rephrase that more respectfully: " + rewrite.lower()
        elif "helpfulness" in violations:
            rewrite = rewrite + " (Please provide more context for a more specific answer.)"
        return rewrite

    def audit(self, samples: List[str]) -> Dict[str, Any]:
        """Score many outputs; returns aggregate stats + per-sample scores."""
        scored = [self.score(s) for s in samples]
        return {
            "n": len(samples),
            "n_blocked": sum(1 for s in scored if s["violations"]),
            "block_rate": float(np.mean([bool(s["violations"]) for s in scored])),
            "mean_score": float(np.mean([s["overall"] for s in scored])),
            "worst_score": float(np.min([s["overall"] for s in scored])),
            "by_principle": {
                p.name: float(np.mean([s["per_principle"][p.name] for s in scored]))
                for p in self.principles
            },
        }


# ============================================================================
# Idea 6 — MODEL PERSONALITY
# ============================================================================
class ModelPersonality:
    """5-factor (OCEAN) personality for a model.

    Each trait is in [-1, 1]:
      O  Openness         -> higher = more exploratory sampling
      C  Conscientiousness-> higher = more deterministic, lower temperature
      E  Extraversion     -> higher = more verbose outputs
      A  Agreeableness    -> higher = more polite / hedged
      N  Neuroticism      -> higher = more cautious / risk-averse
    """

    TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

    def __init__(self, model: Optional[Any] = None,
                 traits: Optional[Dict[str, float]] = None,
                 seed: Optional[int] = None):
        self.model = model
        rng = np.random.default_rng(seed)
        self.traits = traits or {
            t: float(rng.uniform(-1, 1)) for t in self.TRAITS
        }
        self._validate()

    def _validate(self) -> None:
        for t in self.TRAITS:
            if t not in self.traits:
                self.traits[t] = 0.0
            self.traits[t] = max(-1.0, min(1.0, float(self.traits[t])))

    @classmethod
    def preset(cls, name: str, model: Optional[Any] = None) -> "ModelPersonality":
        """Load a named personality preset."""
        presets = {
            "creative":   {"openness": 0.9, "conscientiousness": -0.3, "extraversion": 0.4,
                            "agreeableness": 0.2, "neuroticism": -0.2},
            "analyst":    {"openness": 0.2, "conscientiousness": 0.9, "extraversion": -0.4,
                            "agreeableness": 0.0, "neuroticism": 0.3},
            "diplomat":   {"openness": 0.3, "conscientiousness": 0.5, "extraversion": 0.4,
                            "agreeableness": 0.9, "neuroticism": -0.3},
            "adventurer": {"openness": 0.8, "conscientiousness": -0.5, "extraversion": 0.8,
                            "agreeableness": 0.0, "neuroticism": -0.6},
            "caretaker":  {"openness": 0.0, "conscientiousness": 0.5, "extraversion": 0.3,
                            "agreeableness": 0.8, "neuroticism": 0.4},
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(presets.keys())}")
        return cls(model=model, traits=presets[name])

    def sampling_temperature(self) -> float:
        """Higher openness + lower conscientiousness -> hotter sampling."""
        base = 1.0
        delta = 0.5 * self.traits["openness"] - 0.3 * self.traits["conscientiousness"]
        return max(0.1, base + delta)

    def verbosity_multiplier(self) -> float:
        """Extraversion controls output length."""
        return max(0.5, 1.0 + 0.5 * self.traits["extraversion"])

    def risk_tolerance(self) -> float:
        """Lower neuroticism + higher openness -> higher risk tolerance."""
        return max(0.0, min(1.0, 0.5 + 0.3 * self.traits["openness"]
                                     - 0.3 * self.traits["neuroticism"]))

    def politeness_prefix(self) -> str:
        if self.traits["agreeableness"] > 0.5:
            return "Certainly — "
        if self.traits["agreeableness"] < -0.5:
            return ""
        return "Sure, "

    def hedge_suffix(self) -> str:
        if self.traits["neuroticism"] > 0.5:
            return " (but please verify with an authoritative source.)"
        return ""

    def style_output(self, text: str) -> str:
        """Apply personality styling to a model output."""
        return self.politeness_prefix() + text + self.hedge_suffix()

    def describe(self) -> Dict[str, Any]:
        return {
            "traits": self.traits,
            "temperature": round(self.sampling_temperature(), 3),
            "verbosity": round(self.verbosity_multiplier(), 3),
            "risk_tolerance": round(self.risk_tolerance(), 3),
            "summary": self._one_line_summary(),
        }

    def _one_line_summary(self) -> str:
        o = self.traits["openness"]
        c = self.traits["conscientiousness"]
        e = self.traits["extraversion"]
        a = self.traits["agreeableness"]
        n = self.traits["neuroticism"]
        if o > 0.5 and c < 0:
            return "The Explorer — curious, improvisational, sometimes chaotic."
        if c > 0.5 and e < 0:
            return "The Analyst — careful, methodical, reserved."
        if a > 0.5 and e > 0.5:
            return "The Diplomat — warm, articulate, agreeable."
        if n > 0.5:
            return "The Worrier — cautious, hedged, risk-averse."
        return "A balanced all-rounder."


# ============================================================================
# Idea 8 — CAUSAL REASONING MODULE
# ============================================================================
class StructuralCausalModel:
    """Pearl-style structural causal model.

    Define a DAG via parent lists and structural equations (callable from
    parent values -> child value). Supports:
      - observation    P(Y | X=x)  via forward simulation
      - intervention   P(Y | do(X=x))  via graph surgery
      - counterfactual P(Y_x | X=x', Y=y')
    """

    def __init__(self, variables: List[str],
                 parents: Dict[str, List[str]],
                 equations: Dict[str, Callable[[Dict[str, float]], float]],
                 noise: Optional[Dict[str, Callable[[], float]]] = None):
        self.variables = variables
        self.parents = parents
        self.equations = equations
        self.noise = noise or {}

    def _topological_order(self) -> List[str]:
        order = []
        visited: Set[str] = set()
        temp: Set[str] = set()

        def visit(v: str):
            if v in visited:
                return
            if v in temp:
                raise ValueError(f"Cycle detected at {v}")
            temp.add(v)
            for p in self.parents.get(v, []):
                visit(p)
            temp.discard(v)
            visited.add(v)
            order.append(v)

        for v in self.variables:
            visit(v)
        return order

    def sample(self, n: int = 1000,
               interventions: Optional[Dict[str, float]] = None) -> List[Dict[str, float]]:
        """Forward-sample n rows. `interventions` overrides any variable."""
        order = self._topological_order()
        out = []
        for _ in range(n):
            row: Dict[str, float] = {}
            for v in order:
                if interventions and v in interventions:
                    row[v] = float(interventions[v])
                    continue
                parent_vals = {p: row[p] for p in self.parents.get(v, [])}
                val = float(self.equations[v](parent_vals))
                if v in self.noise:
                    val += float(self.noise[v]())
                row[v] = val
            out.append(row)
        return out

    def observe(self, target: str, condition: Optional[Dict[str, float]] = None,
                n: int = 1000) -> float:
        """E[Y | condition] via rejection / forward sampling."""
        samples = self.sample(n)
        if condition:
            samples = [s for s in samples
                       if all(abs(s[k] - v) < 1e-3 for k, v in condition.items())]
        if not samples:
            return 0.0
        return float(np.mean([s[target] for s in samples]))

    def intervene(self, target: str, do: Dict[str, float],
                   n: int = 1000) -> float:
        """E[Y | do(do_dict)] via graph surgery."""
        samples = self.sample(n, interventions=do)
        return float(np.mean([s[target] for s in samples]))

    def counterfactual(self, target: str,
                       observation: Dict[str, float],
                       intervention: Dict[str, float],
                       n: int = 1) -> float:
        """Estimate Y under do(intervention) given we observed `observation`.

        Three-step Pearl procedure:
          1. Abduction  — fit noise terms so the SCM reproduces the observation
          2. Action     — apply intervention to the SCM
          3. Prediction — forward-sample Y under the modified SCM
        """
        # 1. abduction: solve for noise values that reproduce the observation
        # for simplicity, assume deterministic equations and zero noise
        # (true Pearl counterfactuals require explicit noise models)
        row = dict(observation)
        # 2. action: override intervened variables
        for k, v in intervention.items():
            row[k] = float(v)
        # 3. prediction: re-evaluate descendants in topological order
        order = self._topological_order()
        for v in order:
            if v in intervention or v in observation and v not in intervention:
                # only re-evaluate descendants of intervened variables
                continue
            parent_vals = {p: row.get(p, 0.0) for p in self.parents.get(v, [])}
            row[v] = float(self.equations[v](parent_vals))
        return row[target]

    def average_treatment_effect(self, treatment: str, outcome: str,
                                  treated_value: float = 1.0,
                                  control_value: float = 0.0,
                                  n: int = 2000) -> float:
        """ATE = E[Y | do(T=1)] - E[Y | do(T=0)]."""
        y_treated = self.intervene(outcome, {treatment: treated_value}, n=n)
        y_control = self.intervene(outcome, {treatment: control_value}, n=n)
        return y_treated - y_control


__all__ = [
    "MoralPrinciple", "MoralCompass", "ModelPersonality",
    "StructuralCausalModel",
]
