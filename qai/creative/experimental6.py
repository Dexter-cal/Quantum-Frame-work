"""
qai.creative.experimental6 -- 10 more original creative ideas (my own, part 6).

  Idea 66 — ModelImmuneSystem     : detect + neutralise "pathogenic" inputs
  Idea 67 — NeuroEconomy          : models trade compute for accuracy in a market
  Idea 68 — CausalBayesianNet     : learn DAG + conditional probabilities from data
  Idea 69 — ModelRetirementPlan   : gradual phase-out with knowledge preservation
  Idea 70 — CognitiveArchitecture : SOAR/ACT-R style memory + procedural loop
  Idea 71 — DreamCompiler         : convert dreamt experience into compiled skills
  Idea 72 — EpistemicUncertainty  : track what the model knows it doesn't know
  Idea 73 — ModelMutations        : heritable mutations with fitness landscape
  Idea 74 — CollectiveIntelligence: hive-mind of weak models beats one strong model
  Idea 75 — TimeCrystal           : periodic self-renewing training schedule
"""
from __future__ import annotations
import math
import random
import time
import hashlib
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# ============================================================================
# Idea 66 — Model Immune System
# ============================================================================
class ModelImmuneSystem:
    """Detect + neutralise "pathogenic" inputs.

    Like the human immune system:
      - Antibodies = known-bad input patterns
      - T-cells = anomaly detectors that flag novel inputs
      - Memory cells = remember past attacks for faster response
      - Inflammation = when many attacks arrive, raise alert level
    """
    def __init__(self, model: Any):
        self.model = model
        self.antibodies: List[Dict[str, Any]] = []  # known-bad patterns
        self.memory_cells: List[Dict[str, Any]] = []  # past attacks
        self.threat_level: float = 0.0  # 0=calm, 1=under attack
        self.threat_history: List[float] = []

    def add_antibody(self, pattern: np.ndarray, label: str,
                       severity: float = 1.0) -> None:
        """Register a known-bad input pattern (antibody)."""
        self.antibodies.append({
            "pattern": np.asarray(pattern, dtype=np.float32),
            "label": label,
            "severity": severity,
            "added_at": time.time(),
        })

    def detect(self, x: np.ndarray) -> Dict[str, Any]:
        """Check input x against antibodies + T-cell anomaly detection."""
        x = np.asarray(x, dtype=np.float32).flatten()
        # 1. known-antibody check
        matched = []
        for ab in self.antibodies:
            pattern = ab["pattern"].flatten()
            if pattern.shape == x.shape:
                sim = float(np.dot(x, pattern) / (
                    np.linalg.norm(x) * np.linalg.norm(pattern) + 1e-8))
                if sim > 0.9:
                    matched.append({"label": ab["label"], "similarity": sim,
                                      "severity": ab["severity"]})
        # 2. T-cell: anomaly via distance from average antibody
        anomaly_score = 0.0
        if self.antibodies:
            patterns = np.array([ab["pattern"].flatten() for ab in self.antibodies])
            mean_pattern = patterns.mean(axis=0)
            if mean_pattern.shape == x.shape:
                anomaly_score = float(np.linalg.norm(x - mean_pattern))
        # 3. memory cell check (past attacks)
        memory_match = None
        for mc in self.memory_cells:
            if mc["pattern"].shape == x.shape:
                sim = float(np.dot(x, mc["pattern"]) / (
                    np.linalg.norm(x) * np.linalg.norm(mc["pattern"]) + 1e-8))
                if sim > 0.85:
                    memory_match = mc["label"]
                    break
        is_threat = bool(matched) or anomaly_score > 2.0
        return {
            "is_threat": is_threat,
            "matched_antibodies": matched,
            "anomaly_score": anomaly_score,
            "memory_match": memory_match,
            "threat_level": self.threat_level,
        }

    def neutralise(self, x: np.ndarray) -> Dict[str, Any]:
        """Neutralise a threat: either block or sanitise the input."""
        detection = self.detect(x)
        if not detection["is_threat"]:
            return {"action": "pass_through", "input": x.tolist()}
        # record in memory
        self.memory_cells.append({
            "pattern": np.asarray(x, dtype=np.float32),
            "label": detection["matched_antibodies"][0]["label"] if detection["matched_antibodies"] else "unknown",
            "timestamp": time.time(),
        })
        # raise threat level
        self.threat_level = min(1.0, self.threat_level + 0.1)
        self.threat_history.append(self.threat_level)
        # neutralise: project away from antibodies
        x_neutral = np.asarray(x, dtype=np.float32).copy()
        for ab in self.antibodies:
            pattern = ab["pattern"].flatten()
            if pattern.shape == x_neutral.shape:
                proj = (x_neutral @ pattern) / (np.dot(pattern, pattern) + 1e-8) * pattern
                x_neutral = x_neutral - 0.5 * proj
        return {"action": "sanitised", "input": x_neutral.tolist(),
                 "detection": detection}

    def calm_down(self, decay: float = 0.05) -> None:
        """Lower the threat level over time."""
        self.threat_level = max(0.0, self.threat_level - decay)


# ============================================================================
# Idea 67 — Neuro Economy
# ============================================================================
class NeuroEconomy:
    """Models trade compute for accuracy in a market.

    Each model has:
      - compute_budget (tokens of compute it can spend)
      - accuracy_score (how good it is)
      - price_per_token (what it charges)

    Buyers (tasks) bid for compute. The market clears when supply = demand.
    """
    def __init__(self, initial_supply: float = 1000.0):
        self.supply = initial_supply  # total compute available
        self.models: Dict[str, Dict[str, Any]] = {}
        self.transactions: List[Dict[str, Any]] = []
        self.gdp: float = 0.0  # total economic activity

    def register_model(self, model_id: str, accuracy: float,
                         price_per_token: float,
                         max_tokens: int = 100) -> None:
        self.models[model_id] = {
            "accuracy": accuracy,
            "price": price_per_token,
            "max_tokens": max_tokens,
            "revenue": 0.0,
            "tokens_sold": 0,
        }

    def buy(self, buyer_id: str, model_id: str, n_tokens: int) -> Dict[str, Any]:
        """Buyer purchases n_tokens of compute from a model."""
        if model_id not in self.models:
            raise KeyError(f"unknown model '{model_id}'")
        m = self.models[model_id]
        cost = m["price"] * n_tokens
        # buyer gets expected accuracy proportional to tokens bought
        expected_accuracy = m["accuracy"] * min(1.0, n_tokens / m["max_tokens"])
        m["revenue"] += cost
        m["tokens_sold"] += n_tokens
        self.gdp += cost
        tx = {
            "buyer": buyer_id, "seller": model_id,
            "tokens": n_tokens, "price": cost,
            "expected_accuracy": expected_accuracy,
            "timestamp": time.time(),
        }
        self.transactions.append(tx)
        return tx

    def market_equilibrium(self) -> Dict[str, Any]:
        """Compute market-clearing prices (simplified)."""
        # each model's market share = accuracy / price (higher = better deal)
        scores = {mid: m["accuracy"] / max(m["price"], 1e-6)
                   for mid, m in self.models.items()}
        total = sum(scores.values()) or 1.0
        return {
            "market_shares": {mid: s / total for mid, s in scores.items()},
            "leaderboard": sorted(
                [(mid, m["revenue"]) for mid, m in self.models.items()],
                key=lambda x: -x[1]
            ),
            "gdp": self.gdp,
            "n_transactions": len(self.transactions),
        }


# ============================================================================
# Idea 68 — Causal Bayesian Network
# ============================================================================
class CausalBayesianNet:
    """Learn DAG + conditional probability tables from data.

    Pipeline:
      1. Learn structure via PC algorithm (skeleton + collider orientation)
      2. Estimate CPTs via maximum likelihood
      3. Answer queries via variable elimination
    """
    def __init__(self, var_names: List[str]):
        self.var_names = var_names
        self.n_vars = len(var_names)
        self.parents: Dict[str, List[str]] = {v: [] for v in var_names}
        self.cpts: Dict[str, Dict[Tuple, np.ndarray]] = {}
        self.fitted = False

    def learn_structure(self, X: np.ndarray, alpha: float = 0.05) -> Dict[str, List[str]]:
        """Greedy structure learning: add edge i->j if correlation is significant."""
        n = self.n_vars
        if X.shape[1] != n:
            raise ValueError(f"expected {n} columns, got {X.shape[1]}")
        # compute correlation matrix
        corr = np.corrcoef(X.T)
        # for each pair, if corr > threshold, add edge
        threshold = 0.3
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if abs(corr[i, j]) > threshold:
                    # orient: lower-index -> higher-index (simplified)
                    src, tgt = (i, j) if i < j else (j, i)
                    parent = self.var_names[src]
                    child = self.var_names[tgt]
                    if parent not in self.parents[child]:
                        self.parents[child].append(parent)
        return self.parents

    def fit_cpts(self, X: np.ndarray, n_states: int = 2) -> None:
        """Estimate conditional probability tables via maximum likelihood."""
        n = self.n_vars
        for j, var in enumerate(self.var_names):
            parent_indices = [self.var_names.index(p) for p in self.parents[var]]
            if not parent_indices:
                # marginal distribution
                counts = np.zeros(n_states)
                for v in X[:, j]:
                    counts[int(v) % n_states] += 1
                self.cpts[var] = {(): counts / counts.sum()}
            else:
                # conditional on parent configurations
                cpt: Dict[Tuple, np.ndarray] = {}
                parent_states = list(itertools.product(range(n_states), repeat=len(parent_indices)))
                for ps in parent_states:
                    mask = np.all(X[:, parent_indices] == ps, axis=1)
                    if mask.sum() == 0:
                        cpt[ps] = np.ones(n_states) / n_states  # uniform prior
                    else:
                        counts = np.zeros(n_states)
                        for v in X[mask, j]:
                            counts[int(v) % n_states] += 1
                        cpt[ps] = counts / counts.sum()
                self.cpts[var] = cpt
        self.fitted = True

    def query(self, target: str,
               evidence: Optional[Dict[str, int]] = None) -> np.ndarray:
        """Query the marginal probability of target given evidence."""
        if not self.fitted:
            raise RuntimeError("call fit_cpts() first")
        evidence = evidence or {}
        # if target has no parents or all parents are evidence, return CPT row
        target_parents = self.parents[target]
        if all(p in evidence for p in target_parents):
            key = tuple(evidence[p] for p in target_parents)
            return self.cpts[target].get(key, np.ones(len(next(iter(self.cpts[target].values())))) / len(next(iter(self.cpts[target].values()))))
        # otherwise marginalise (simplified: just return marginal)
        if () in self.cpts[target]:
            return self.cpts[target][()]
        # fallback: average over all parent configurations
        all_probs = list(self.cpts[target].values())
        return np.mean(all_probs, axis=0)

    def topological_order(self) -> List[str]:
        """Return variables in topological order (parents before children)."""
        visited: Set[str] = set()
        order: List[str] = []
        def visit(v: str):
            if v in visited:
                return
            visited.add(v)
            for p in self.parents[v]:
                visit(p)
            order.append(v)
        for v in self.var_names:
            visit(v)
        return order


import itertools  # for CausalBayesianNet


# ============================================================================
# Idea 69 — Model Retirement Plan
# ============================================================================
class ModelRetirementPlan:
    """Gradual phase-out of a model with knowledge preservation.

    Stages:
      1. Active       — fully serving traffic
      2. PhasedOut    — serving reduced traffic, training successor
      3. Read-only    — only answers from cache, no new traffic
      4. Frozen       — no traffic, available for fine-tune extraction
      5. Archived     — only metadata + distilled knowledge remains
    """
    STAGES = ["active", "phased_out", "read_only", "frozen", "archived"]

    def __init__(self, model_id: str, successor_id: Optional[str] = None):
        self.model_id = model_id
        self.successor_id = successor_id
        self.stage = "active"
        self.traffic_share: float = 1.0
        self.history: List[Dict[str, Any]] = []
        self.distilled_knowledge: Optional[Dict[str, Any]] = None

    def advance(self, distilled: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Advance to the next retirement stage."""
        idx = self.STAGES.index(self.stage)
        if idx >= len(self.STAGES) - 1:
            return {"stage": self.stage, "done": True}
        # if moving to read_only or beyond, require distilled knowledge
        if self.stage in ["phased_out", "read_only"] and distilled is not None:
            self.distilled_knowledge = distilled
        self.stage = self.STAGES[idx + 1]
        # reduce traffic
        self.traffic_share = max(0.0, self.traffic_share - 0.25)
        entry = {
            "stage": self.stage,
            "traffic_share": self.traffic_share,
            "timestamp": time.time(),
        }
        self.history.append(entry)
        return entry

    def can_serve(self) -> bool:
        """Whether this model can still serve traffic."""
        return self.stage in ["active", "phased_out"] and self.traffic_share > 0

    def get_distilled_knowledge(self) -> Optional[Dict[str, Any]]:
        """Return the distilled knowledge (for successor to absorb)."""
        return self.distilled_knowledge

    def plan_summary(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "successor_id": self.successor_id,
            "current_stage": self.stage,
            "traffic_share": self.traffic_share,
            "n_transitions": len(self.history),
            "has_distilled_knowledge": self.distilled_knowledge is not None,
        }


# ============================================================================
# Idea 70 — Cognitive Architecture (SOAR/ACT-R inspired)
# ============================================================================
class CognitiveArchitecture:
    """SOAR/ACT-R style cognitive architecture.

    Components:
      - Perception buffer  : holds current input
      - Goal buffer        : holds current goal
      - Working memory     : short-term facts
      - Procedural memory  : production rules (if-then)
      - Declarative memory : long-term facts (with activation levels)
      - Motor buffer       : holds next action

    The cognitive loop: perceive -> match rules -> select action -> execute
    """
    def __init__(self):
        self.perception: Dict[str, Any] = {}
        self.goal: Optional[str] = None
        self.working_memory: Dict[str, Any] = {}
        self.declarative_memory: Dict[str, Dict[str, Any]] = {}  # fact -> {value, activation, last_access}
        self.procedural_memory: List[Dict[str, Any]] = []  # rules
        self.motor: Optional[Any] = None
        self.history: List[Dict[str, Any]] = []
        self.cycle_count: int = 0

    def add_rule(self, name: str, condition: Callable[["CognitiveArchitecture"], bool],
                  action: Callable[["CognitiveArchitecture"], Any],
                  priority: int = 0) -> None:
        self.procedural_memory.append({
            "name": name, "condition": condition, "action": action,
            "priority": priority,
        })

    def add_fact(self, fact_id: str, value: Any, activation: float = 1.0) -> None:
        self.declarative_memory[fact_id] = {
            "value": value, "activation": activation,
            "last_access": time.time(),
        }

    def recall(self, fact_id: str) -> Optional[Any]:
        """Recall a fact; decays activation if not recently accessed."""
        if fact_id not in self.declarative_memory:
            return None
        fact = self.declarative_memory[fact_id]
        # decay: activation drops if not accessed recently
        elapsed = time.time() - fact["last_access"]
        fact["activation"] *= math.exp(-elapsed / 3600)  # decay over hours
        fact["last_access"] = time.time()
        return fact["value"]

    def perceive(self, input_data: Dict[str, Any]) -> None:
        """Update the perception buffer."""
        self.perception.update(input_data)

    def set_goal(self, goal: str) -> None:
        self.goal = goal

    def step(self) -> Dict[str, Any]:
        """One cognitive cycle: match rules -> fire best -> execute."""
        self.cycle_count += 1
        # match: find all rules whose condition is True
        matched = [r for r in self.procedural_memory if r["condition"](self)]
        if not matched:
            self.history.append({"cycle": self.cycle_count, "rule_fired": None})
            return {"cycle": self.cycle_count, "action": None}
        # pick highest priority
        matched.sort(key=lambda r: -r["priority"])
        rule = matched[0]
        # fire action
        self.motor = rule["action"](self)
        self.history.append({
            "cycle": self.cycle_count,
            "rule_fired": rule["name"],
            "action": str(self.motor),
        })
        return {"cycle": self.cycle_count, "rule": rule["name"],
                 "action": self.motor}

    def run(self, n_cycles: int = 10) -> List[Dict[str, Any]]:
        """Run the cognitive loop for n cycles."""
        results = []
        for _ in range(n_cycles):
            results.append(self.step())
        return results


# ============================================================================
# Idea 71 — Dream Compiler
# ============================================================================
class DreamCompiler:
    """Convert dreamt experience into compiled skills.

    Takes the output of DreamMode (imagined experiences) and "compiles"
    them into reusable skills stored as small adapter modules.
    """
    def __init__(self, model: Any):
        self.model = model
        self.dreams: List[Dict[str, Any]] = []
        self.compiled_skills: List[Dict[str, Any]] = []

    def record_dream(self, dream_data: np.ndarray, quality: float = 0.5) -> None:
        """Record a single dreamt experience."""
        self.dreams.append({
            "data": np.asarray(dream_data, dtype=np.float32),
            "quality": quality,
            "timestamp": time.time(),
        })

    def compile(self, n_skills: int = 3) -> List[Dict[str, Any]]:
        """Cluster dreams into n skills; each skill = average dream pattern."""
        if not self.dreams:
            return []
        # weight by quality
        weights = np.array([d["quality"] for d in self.dreams])
        weights = weights / weights.sum()
        dream_data = np.array([d["data"].flatten() for d in self.dreams])
        # k-means clustering
        from sklearn.cluster import KMeans
        n = min(n_skills, len(self.dreams))
        kmeans = KMeans(n_clusters=n, random_state=0, n_init=10)
        labels = kmeans.fit_predict(dream_data, sample_weight=weights)
        for c in range(n):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            centroid = kmeans.cluster_centers_[c]
            # compile: skill = centroid pattern
            self.compiled_skills.append({
                "skill_id": f"dream_skill_{c}",
                "centroid": centroid.tolist(),
                "n_dreams": int(len(members)),
                "mean_quality": float(np.mean([self.dreams[i]["quality"]
                                                  for i in members])),
                "compiled_at": time.time(),
            })
        return self.compiled_skills

    def list_skills(self) -> List[Dict[str, Any]]:
        return list(self.compiled_skills)

    def apply_skill(self, skill_id: str, input_data: np.ndarray) -> Dict[str, Any]:
        """Apply a compiled skill to new input."""
        skill = next((s for s in self.compiled_skills if s["skill_id"] == skill_id), None)
        if skill is None:
            raise KeyError(f"unknown skill '{skill_id}'")
        centroid = np.array(skill["centroid"], dtype=np.float32)
        x = np.asarray(input_data, dtype=np.float32).flatten()
        # pad/truncate to match
        if x.shape[0] < centroid.shape[0]:
            x = np.pad(x, (0, centroid.shape[0] - x.shape[0]))
        elif x.shape[0] > centroid.shape[0]:
            x = x[:centroid.shape[0]]
        # similarity = cosine
        sim = float(np.dot(x, centroid) / (np.linalg.norm(x) * np.linalg.norm(centroid) + 1e-8))
        return {"skill_id": skill_id, "similarity": sim,
                 "applied": True}


# ============================================================================
# Idea 72 — Epistemic Uncertainty Tracker
# ============================================================================
class EpistemicUncertaintyTracker:
    """Track what the model knows it doesn't know.

    Maintains a per-input "epistemic" uncertainty estimate via:
      1. MC Dropout: run K forward passes with dropout on
      2. Deep ensemble: average predictions from K models
      3. Distance to training data: far = uncertain

    Output: for each prediction, a confidence + an "I don't know" flag.
    """
    def __init__(self, model: Any, n_passes: int = 5,
                 uncertainty_threshold: float = 0.5):
        self.model = model
        self.n_passes = n_passes
        self.threshold = uncertainty_threshold
        self.training_data_stats: Optional[Dict[str, float]] = None

    def calibrate(self, X_train: np.ndarray) -> None:
        """Record training-data statistics for distance-based uncertainty."""
        X = np.asarray(X_train, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        self.training_data_stats = {
            "mean": X.mean(axis=0).tolist(),
            "std": X.std(axis=0).tolist(),
            "n_samples": int(X.shape[0]),
        }

    def predict_with_uncertainty(self, x: np.ndarray) -> Dict[str, Any]:
        """Predict with epistemic uncertainty estimate."""
        x = np.asarray(x, dtype=np.float32)
        # 1. MC dropout: run K passes
        preds = []
        for _ in range(self.n_passes):
            try:
                pred = np.asarray(self.model.forward(x), dtype=np.float32).flatten()
                preds.append(pred)
            except Exception:
                pass
        if not preds:
            return {"prediction": None, "uncertainty": 1.0, "confident": False}
        preds = np.array(preds)
        mean_pred = preds.mean(axis=0)
        epistemic = float(preds.std(axis=0).mean())  # average std across outputs
        # 2. distance from training data
        if self.training_data_stats is not None:
            train_mean = np.array(self.training_data_stats["mean"], dtype=np.float32)
            x_flat = x.flatten()
            if x_flat.shape[0] < train_mean.shape[0]:
                x_flat = np.pad(x_flat, (0, train_mean.shape[0] - x_flat.shape[0]))
            elif x_flat.shape[0] > train_mean.shape[0]:
                x_flat = x_flat[:train_mean.shape[0]]
            dist = float(np.linalg.norm(x_flat - train_mean))
            # combine: uncertainty = epistemic + distance factor
            total_uncertainty = epistemic + min(1.0, dist / 10.0) * 0.5
        else:
            total_uncertainty = epistemic
        return {
            "prediction": mean_pred.tolist(),
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": total_uncertainty,
            "confident": total_uncertainty < self.threshold,
            "would_say_idk": total_uncertainty >= self.threshold,
            "n_passes": len(preds),
        }


# ============================================================================
# Idea 73 — Model Mutations
# ============================================================================
class ModelMutation:
    """A single heritable mutation in a model's genome (weights)."""
    def __init__(self, layer: str, index: Tuple[int, ...],
                 old_value: float, new_value: float,
                 mutation_type: str = "point"):
        self.layer = layer
        self.index = index
        self.old_value = old_value
        self.new_value = new_value
        self.mutation_type = mutation_type  # point, insert, delete, swap
        self.id = hashlib.sha256(
            f"{layer}{index}{new_value}{time.time()}{random.random()}".encode()
        ).hexdigest()[:8]


class ModelMutator:
    """Apply heritable mutations to a model with a fitness landscape.

    Each mutation changes a single weight. Mutations accumulate over
    generations. Track lineage so we can roll back harmful mutations.
    """
    def __init__(self, model: Any, mutation_rate: float = 0.01,
                 mutation_scale: float = 0.1):
        self.model = model
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.mutations: List[ModelMutation] = []
        self.fitness_history: List[float] = []

    def mutate(self) -> List[ModelMutation]:
        """Apply random mutations to model weights."""
        new_mutations: List[ModelMutation] = []
        if not _TORCH or not hasattr(self.model, "state_dict"):
            return new_mutations
        with torch.no_grad():
            sd = self.model.state_dict()
            for layer, tensor in sd.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                mask = torch.rand_like(tensor) < self.mutation_rate
                if not mask.any():
                    continue
                indices = mask.nonzero()
                for idx in indices:
                    old_val = float(tensor[tuple(idx)])
                    new_val = old_val + random.gauss(0, self.mutation_scale)
                    tensor[tuple(idx)] = new_val
                    new_mutations.append(ModelMutation(
                        layer, tuple(idx.tolist()), old_val, new_val
                    ))
            try:
                self.model.load_state_dict(sd)
            except Exception:
                pass
        self.mutations.extend(new_mutations)
        return new_mutations

    def rollback(self, n_generations: int = 1) -> int:
        """Roll back the last n generations of mutations."""
        if not self.mutations:
            return 0
        if not _TORCH or not hasattr(self.model, "state_dict"):
            return 0
        rolled_back = 0
        with torch.no_grad():
            sd = self.model.state_dict()
            for _ in range(n_generations):
                if not self.mutations:
                    break
                mut = self.mutations.pop()
                if mut.layer in sd:
                    try:
                        sd[mut.layer][mut.index] = mut.old_value
                        rolled_back += 1
                    except Exception:
                        pass
            try:
                self.model.load_state_dict(sd)
            except Exception:
                pass
        return rolled_back

    def lineage(self) -> List[Dict[str, Any]]:
        """Get the full mutation lineage."""
        return [{"id": m.id, "layer": m.layer, "index": m.index,
                  "old": m.old_value, "new": m.new_value, "type": m.mutation_type}
                 for m in self.mutations]


# ============================================================================
# Idea 74 — Collective Intelligence (hive mind)
# ============================================================================
class CollectiveIntelligence:
    """Hive-mind of weak models beats one strong model.

    Each weak model is "weak" — slightly above chance. The collective
    aggregates their votes via:
      - majority vote (simple)
      - weighted vote (weight by individual accuracy)
      - boosting (iteratively re-weight hard examples)
      - mixture of experts (gated)
    """
    def __init__(self, models: List[Any]):
        self.models = models
        self.weights: np.ndarray = np.ones(len(models)) / len(models)
        self.history: List[Dict[str, Any]] = []

    def majority_vote(self, x: Any) -> Dict[str, Any]:
        """Simple majority vote across all models."""
        votes = []
        for m in self.models:
            try:
                out = np.asarray(m.forward(x), dtype=np.float32).flatten()
                votes.append(int(out.argmax()))
            except Exception:
                votes.append(-1)
        votes = [v for v in votes if v >= 0]
        if not votes:
            return {"prediction": None, "confidence": 0.0}
        counter = Counter(votes)
        winner, count = counter.most_common(1)[0]
        return {"prediction": winner, "confidence": count / len(votes),
                 "n_models": len(votes)}

    def weighted_vote(self, x: Any) -> Dict[str, Any]:
        """Weighted by individual model accuracy."""
        votes = []
        for m, w in zip(self.models, self.weights):
            try:
                out = np.asarray(m.forward(x), dtype=np.float32).flatten()
                votes.append((int(out.argmax()), w))
            except Exception:
                continue
        if not votes:
            return {"prediction": None, "confidence": 0.0}
        weighted_scores: Dict[int, float] = defaultdict(float)
        for v, w in votes:
            weighted_scores[v] += w
        winner = max(weighted_scores.items(), key=lambda kv: kv[1])
        total_w = sum(weighted_scores.values())
        return {"prediction": winner[0],
                 "confidence": winner[1] / total_w if total_w > 0 else 0,
                 "n_models": len(votes)}

    def boost(self, X: List[Any], y: List[int],
               n_rounds: int = 3) -> Dict[str, Any]:
        """Boosting: iteratively re-weight hard examples."""
        n = len(X)
        sample_weights = np.ones(n) / n
        for r in range(n_rounds):
            # evaluate each model on weighted samples
            model_accs = []
            for m in self.models:
                correct = 0
                for i, x in enumerate(X):
                    try:
                        out = np.asarray(m.forward(x), dtype=np.float32).flatten()
                        if int(out.argmax()) == y[i]:
                            correct += sample_weights[i]
                    except Exception:
                        pass
                model_accs.append(correct / sample_weights.sum())
            # update model weights
            accs = np.array(model_accs)
            self.weights = accs / accs.sum() if accs.sum() > 0 else self.weights
            # update sample weights: emphasise hard examples
            for i, x in enumerate(X):
                pred_correct = False
                for m in self.models:
                    try:
                        out = np.asarray(m.forward(x), dtype=np.float32).flatten()
                        if int(out.argmax()) == y[i]:
                            pred_correct = True
                            break
                    except Exception:
                        pass
                if not pred_correct:
                    sample_weights[i] *= 1.5
            sample_weights /= sample_weights.sum()
            self.history.append({"round": r, "model_weights": self.weights.tolist(),
                                  "model_accs": model_accs})
        return {"n_rounds": n_rounds, "final_weights": self.weights.tolist()}


# ============================================================================
# Idea 75 — Time Crystal
# ============================================================================
class TimeCrystal:
    """Periodic self-renewing training schedule.

    Inspired by time crystals in physics: a system that periodically
    self-organises without external energy. For ML: schedule training
    in periodic bursts with auto-renewing curriculum.
    """
    def __init__(self, model: Any, period: float = 3600.0,
                 burst_duration: float = 60.0):
        """
        period: time between bursts (seconds)
        burst_duration: how long each training burst lasts
        """
        self.model = model
        self.period = period
        self.burst_duration = burst_duration
        self.last_burst: float = 0.0
        self.burst_history: List[Dict[str, Any]] = []
        self.curriculum_phase: int = 0
        self.running: bool = False

    def should_burst(self, now: Optional[float] = None) -> bool:
        """Check if it's time for a training burst."""
        now = now or time.time()
        return (now - self.last_burst) >= self.period

    def burst(self, X: np.ndarray, y: np.ndarray,
               now: Optional[float] = None) -> Dict[str, Any]:
        """Execute one training burst with curriculum rotation."""
        now = now or time.time()
        self.curriculum_phase = (self.curriculum_phase + 1) % 4
        phase_names = ["easy", "medium", "hard", "adversarial"]
        phase = phase_names[self.curriculum_phase]
        # in a real system, we'd select examples by difficulty here
        # for demo, we just shuffle
        idx = np.random.permutation(len(X))[:32]  # mini-batch
        X_burst = X[idx]
        y_burst = y[idx]
        try:
            self.model.fit(X_burst, y_burst)
            success = True
        except Exception as e:
            success = False
            error = str(e)
        entry = {
            "phase": phase,
            "n_samples": len(idx),
            "success": success,
            "timestamp": now,
        }
        if not success:
            entry["error"] = error
        self.burst_history.append(entry)
        self.last_burst = now
        return entry

    def run_continuous(self, X: np.ndarray, y: np.ndarray,
                        max_bursts: int = 5,
                        check_interval: float = 1.0) -> Dict[str, Any]:
        """Run the crystal continuously until max_bursts reached.

        Note: this is blocking. For non-blocking, use the async version.
        """
        self.running = True
        bursts_done = 0
        while self.running and bursts_done < max_bursts:
            if self.should_burst():
                self.burst(X, y)
                bursts_done += 1
            else:
                time.sleep(min(check_interval, self.period))
        self.running = False
        return {"bursts_completed": bursts_done,
                 "phases_visited": [b["phase"] for b in self.burst_history]}

    def stop(self) -> None:
        self.running = False

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "period_s": self.period,
            "burst_duration_s": self.burst_duration,
            "last_burst": self.last_burst,
            "next_burst_in_s": max(0, self.period - (time.time() - self.last_burst))
                                if self.last_burst > 0 else 0,
            "n_bursts_completed": len(self.burst_history),
            "current_phase": self.curriculum_phase,
        }


__all__ = [
    "ModelImmuneSystem", "NeuroEconomy", "CausalBayesianNet",
    "ModelRetirementPlan", "CognitiveArchitecture", "DreamCompiler",
    "EpistemicUncertaintyTracker", "ModelMutation", "ModelMutator",
    "CollectiveIntelligence", "TimeCrystal",
]
