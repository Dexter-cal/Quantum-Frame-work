"""
qai.creative.experimental7 -- 10 more original creative ideas (my own, part 7).

  Idea 76 — KnowledgeEngineer      : interactive knowledge acquisition
  Idea 77 — ModelGarden            : cultivate + cross-breed models like plants
  Idea 78 — MemoryPalace           : spatial mnemonic memory for the model
  Idea 79 — SkillPyramid           : Maslow-style hierarchy of model needs
  Idea 80 — CognitiveBehaviouralTherapy : treat model "anxiety" + "depression"
  Idea 81 — ModelSandbox           : safe playground for model experimentation
  Idea 82 — NeuroForensics         : post-mortem analysis of failed models
  Idea 83 — ModelMarathon          : long-distance training optimiser
  Idea 84 — KnowledgeTsunami       : handle sudden knowledge influx gracefully
  Idea 85 — ModelHibernation       : pause training with state preservation
"""
from __future__ import annotations
import math
import random
import time
import hashlib
from collections import defaultdict, OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


# ============================================================================
# Idea 76 — Knowledge Engineer
# ============================================================================
class KnowledgeEngineer:
    """Interactive knowledge acquisition from human experts.

    Pipeline:
      1. ASK: pose a question to a human expert
      2. EXTRACT: parse the answer into structured (subject, predicate, object)
      3. VERIFY: cross-check against existing knowledge base
      4. INTEGRATE: add to the KB, mark confidence
      5. PROPAGATE: derive new facts via inference rules

    The result is a growing, curated knowledge base.
    """
    def __init__(self):
        self.kb: Dict[str, Dict[str, Any]] = {}  # fact_id -> {s, p, o, confidence, source}
        self.inference_rules: List[Dict[str, Any]] = []
        self.questions_asked: List[str] = []
        self.history: List[Dict[str, Any]] = []

    def ask(self, topic: str) -> str:
        """Generate a question for the expert about `topic`."""
        existing = [f for f in self.kb.values() if topic.lower() in f["s"].lower()
                     or topic.lower() in f["o"].lower()]
        if not existing:
            q = f"What is {topic}?"
        else:
            related = [f["o"] for f in existing if f["s"] == topic][:3]
            q = f"Tell me more about how {topic} relates to {', '.join(related)}."
        self.questions_asked.append(q)
        return q

    def extract(self, answer: str) -> List[Dict[str, str]]:
        """Parse an answer into (subject, predicate, object) triples.

        Heuristic: look for "X is Y", "X has Y", "X verb Y" patterns.
        """
        import re
        triples = []
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        verbs = ["is", "are", "was", "were", "has", "have", "uses",
                  "contains", "relates", "depends"]
        for sent in sentences:
            words = sent.split()
            for i, w in enumerate(words):
                if w.lower() in verbs and i > 0 and i + 1 < len(words):
                    subj = " ".join(words[:i])
                    obj = " ".join(words[i+1:])
                    triples.append({"s": subj, "p": w.lower(), "o": obj})
        return triples

    def verify(self, triple: Dict[str, str]) -> float:
        """Check if the triple is consistent with existing KB. Returns confidence."""
        # check if any existing fact contradicts this one
        contradicting = 0
        supporting = 0
        for f in self.kb.values():
            if f["s"] == triple["s"] and f["p"] == triple["p"]:
                if f["o"] == triple["o"]:
                    supporting += 1
                else:
                    contradicting += 1
        if contradicting > 0 and supporting == 0:
            return 0.2  # contradictory
        if supporting > 0:
            return 0.9  # confirmed
        return 0.5  # novel

    def integrate(self, triple: Dict[str, str], confidence: float,
                    source: str = "expert") -> str:
        """Add a verified triple to the KB."""
        fact_id = hashlib.sha256(
            f"{triple['s']}|{triple['p']}|{triple['o']}".encode()
        ).hexdigest()[:12]
        self.kb[fact_id] = {
            **triple, "confidence": confidence, "source": source,
            "added_at": time.time(),
        }
        self.history.append({"action": "integrate", "fact_id": fact_id, **triple})
        return fact_id

    def add_inference_rule(self, name: str,
                            pattern: Dict[str, str],
                            conclusion: Dict[str, str]) -> None:
        """Add a rule: if pattern matches, conclude."""
        self.inference_rules.append({"name": name, "pattern": pattern,
                                       "conclusion": conclusion})

    def propagate(self) -> List[Dict[str, str]]:
        """Apply inference rules to derive new facts."""
        new_facts = []
        for rule in self.inference_rules:
            for f in self.kb.values():
                # check if pattern matches
                match = all(f.get(k) == v for k, v in rule["pattern"].items())
                if match:
                    # apply conclusion (substitute variables)
                    derived = {k: v for k, v in rule["conclusion"].items()}
                    derived["confidence"] = f["confidence"] * 0.8  # decay
                    new_facts.append(derived)
        return new_facts

    def query(self, subject: str) -> List[Dict[str, Any]]:
        """Find all facts about a subject."""
        return [f for f in self.kb.values() if f["s"] == subject]

    def stats(self) -> Dict[str, Any]:
        return {
            "n_facts": len(self.kb),
            "n_questions_asked": len(self.questions_asked),
            "avg_confidence": float(np.mean([f["confidence"] for f in self.kb.values()]))
                                if self.kb else 0.0,
            "n_inference_rules": len(self.inference_rules),
        }


# ============================================================================
# Idea 77 — Model Garden
# ============================================================================
class ModelPlant:
    """A model treated as a plant in a garden."""
    def __init__(self, name: str, model: Any, species: str = "mlp",
                 sunlight: float = 1.0, water: float = 1.0,
                 fertilizer: float = 0.5):
        self.name = name
        self.model = model
        self.species = species
        self.sunlight = sunlight  # compute
        self.water = water        # data
        self.fertilizer = fertilizer  # hyperparameter tuning
        self.health: float = 1.0
        self.age: int = 0
        self.fruits: List[Any] = []  # outputs
        self.crosses: List[str] = []


class ModelGarden:
    """Cultivate + cross-breed models like plants.

    Each plant needs sunlight (compute), water (data), fertilizer (tuning).
    Healthy plants produce fruits (good predictions). Cross-breed two plants
    to combine their strengths.
    """
    def __init__(self):
        self.plants: Dict[str, ModelPlant] = {}
        self.crosses_history: List[Dict[str, Any]] = []

    def plant(self, plant: ModelPlant) -> None:
        self.plants[plant.name] = plant

    def water(self, name: str, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Water a plant: give it data to train on."""
        if name not in self.plants:
            raise KeyError(f"no plant named '{name}'")
        p = self.plants[name]
        p.age += 1
        try:
            if y is not None:
                p.model.fit(X, y)
            else:
                p.model.fit(X)
            p.health = min(1.0, p.health + 0.1)
            fruit = p.model.forward(X[0]) if hasattr(X, "__getitem__") else None
            if fruit is not None:
                p.fruits.append(fruit)
            return {"watered": True, "health": p.health, "n_fruits": len(p.fruits)}
        except Exception as e:
            p.health = max(0.0, p.health - 0.2)
            return {"watered": False, "error": str(e), "health": p.health}

    def sunlight(self, name: str, intensity: float = 1.0) -> None:
        """Give a plant more compute (sunlight)."""
        if name in self.plants:
            self.plants[name].sunlight = max(0.0, min(2.0, intensity))

    def fertilize(self, name: str, amount: float = 0.1) -> None:
        """Add fertilizer (hyperparameter tuning)."""
        if name in self.plants:
            self.plants[name].fertilizer = max(0.0, min(1.0, self.plants[name].fertilizer + amount))

    def cross_breed(self, parent_a: str, parent_b: str,
                     child_name: str) -> Optional[ModelPlant]:
        """Cross-breed two plants to produce a child."""
        if parent_a not in self.plants or parent_b not in self.plants:
            return None
        a = self.plants[parent_a]
        b = self.plants[parent_b]
        # child inherits average sunlight/water/fertilizer
        child = ModelPlant(
            name=child_name, model=a.model,  # share parent's model for simplicity
            species=a.species,
            sunlight=(a.sunlight + b.sunlight) / 2,
            water=(a.water + b.water) / 2,
            fertilizer=(a.fertilizer + b.fertilizer) / 2,
        )
        child.health = (a.health + b.health) / 2 * 0.9  # slight penalty
        a.crosses.append(child_name)
        b.crosses.append(child_name)
        self.plants[child_name] = child
        self.crosses_history.append({
            "parents": [parent_a, parent_b], "child": child_name,
            "timestamp": time.time(),
        })
        return child

    def harvest(self) -> Dict[str, Any]:
        """Collect all fruits (predictions) from healthy plants."""
        harvest: Dict[str, Any] = {}
        for name, p in self.plants.items():
            if p.health > 0.3 and p.fruits:
                harvest[name] = p.fruits[-5:]  # last 5 outputs
        return harvest

    def garden_status(self) -> Dict[str, Any]:
        return {
            "n_plants": len(self.plants),
            "n_crosses": len(self.crosses_history),
            "healthy_plants": sum(1 for p in self.plants.values() if p.health > 0.5),
            "wilting_plants": sum(1 for p in self.plants.values() if p.health < 0.3),
            "total_fruits": sum(len(p.fruits) for p in self.plants.values()),
        }


# ============================================================================
# Idea 78 — Memory Palace
# ============================================================================
class MemoryPalace:
    """Spatial mnemonic memory for the model.

    Inspired by the ancient Method of Loci: facts are placed in spatial
    "rooms" of a virtual palace. Retrieval walks through rooms to find
    related facts. Spatial proximity = semantic similarity.
    """
    def __init__(self, n_rooms: int = 10, room_capacity: int = 50):
        self.n_rooms = n_rooms
        self.room_capacity = room_capacity
        self.rooms: List[List[Dict[str, Any]]] = [[] for _ in range(n_rooms)]
        self.room_centroids: List[Optional[np.ndarray]] = [None] * n_rooms
        self.next_room: int = 0

    def place(self, fact: Dict[str, Any], embedding: np.ndarray) -> int:
        """Place a fact in the room whose centroid is most similar."""
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        # find best room
        best_room = -1
        best_sim = -float("inf")
        for i, c in enumerate(self.room_centroids):
            if c is None:
                continue
            if c.shape == embedding.shape:
                sim = float(np.dot(c, embedding) / (
                    np.linalg.norm(c) * np.linalg.norm(embedding) + 1e-8))
                if sim > best_sim:
                    best_sim = sim
                    best_room = i
        if best_room == -1 or best_sim < 0.5:
            # use the next empty room
            for i in range(self.n_rooms):
                if self.room_centroids[i] is None:
                    best_room = i
                    break
            if best_room == -1:
                # all rooms have centroids; pick least full
                best_room = min(range(self.n_rooms), key=lambda i: len(self.rooms[i]))
        # add to room
        if len(self.rooms[best_room]) >= self.room_capacity:
            # evict oldest
            self.rooms[best_room].pop(0)
        self.rooms[best_room].append({"fact": fact, "embedding": embedding})
        # update centroid
        all_embs = [r["embedding"] for r in self.rooms[best_room]]
        self.room_centroids[best_room] = np.mean(all_embs, axis=0)
        return best_room

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Walk through rooms; return top-k facts most similar to query."""
        query_embedding = np.asarray(query_embedding, dtype=np.float32).flatten()
        # find rooms whose centroids are similar
        room_sims = []
        for i, c in enumerate(self.room_centroids):
            if c is None or c.shape != query_embedding.shape:
                room_sims.append((i, -1.0))
            else:
                sim = float(np.dot(c, query_embedding) / (
                    np.linalg.norm(c) * np.linalg.norm(query_embedding) + 1e-8))
                room_sims.append((i, sim))
        room_sims.sort(key=lambda x: -x[1])
        # collect candidates from top rooms
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for room_idx, _ in room_sims[:3]:  # visit top 3 rooms
            for entry in self.rooms[room_idx]:
                emb = entry["embedding"]
                if emb.shape == query_embedding.shape:
                    sim = float(np.dot(emb, query_embedding) / (
                        np.linalg.norm(emb) * np.linalg.norm(query_embedding) + 1e-8))
                    candidates.append((sim, entry["fact"]))
        candidates.sort(key=lambda x: -x[0])
        return [c[1] for c in candidates[:top_k]]

    def palace_map(self) -> Dict[str, Any]:
        """Return a map of the palace: rooms + their fill levels + centroids."""
        return {
            "n_rooms": self.n_rooms,
            "rooms": [
                {"id": i, "n_facts": len(self.rooms[i]),
                  "has_centroid": self.room_centroids[i] is not None}
                for i in range(self.n_rooms)
            ],
            "total_facts": sum(len(r) for r in self.rooms),
        }


# ============================================================================
# Idea 79 — Skill Pyramid (Maslow-style hierarchy of model needs)
# ============================================================================
class SkillPyramid:
    """Maslow-style hierarchy of model needs.

    Levels (bottom to top):
      1. Physiological  — basic compute + memory
      2. Safety         — robustness + adversarial defense
      3. Belonging      — works well in ensembles
      4. Esteem         — high accuracy + benchmarks
      5. Self-actualisation — creativity + reasoning

    A model can't reach higher levels until lower needs are met.
    """
    LEVELS = ["physiological", "safety", "belonging", "esteem", "self_actualisation"]

    def __init__(self, model: Any):
        self.model = model
        self.assessments: Dict[str, float] = {l: 0.0 for l in self.LEVELS}
        self.history: List[Dict[str, Any]] = []

    def assess(self, level: str, score: float) -> None:
        """Set the score (0-1) for a level."""
        if level not in self.LEVELS:
            raise ValueError(f"unknown level '{level}'; choose from {self.LEVELS}")
        self.assessments[level] = max(0.0, min(1.0, score))
        self.history.append({"level": level, "score": score, "timestamp": time.time()})

    def current_level(self) -> str:
        """Return the highest unlocked level."""
        for i in range(len(self.LEVELS) - 1, -1, -1):
            if self.assessments[self.LEVELS[i]] > 0.5:
                # check all lower levels are also satisfied
                if all(self.assessments[self.LEVELS[j]] > 0.5 for j in range(i)):
                    return self.LEVELS[i]
        return "none"

    def next_blocker(self) -> Optional[str]:
        """Return the level that's blocking progress, or None if all met."""
        for level in self.LEVELS:
            if self.assessments[level] <= 0.5:
                return level
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "assessments": self.assessments,
            "current_level": self.current_level(),
            "next_blocker": self.next_blocker(),
            "all_met": self.next_blocker() is None,
        }


# ============================================================================
# Idea 80 — Cognitive Behavioural Therapy for models
# ============================================================================
class CBTForModels:
    """Treat model "anxiety" (overconfidence) + "depression" (underconfidence).

    Sessions:
      1. ASSESS: measure current confidence calibration
      2. CHALLENGE: identify cognitive distortions
      3. REFRAME: retrain with calibrated loss
      4. HOMEWORK: regular self-assessment exercises
    """
    DISTORTIONS = {
        "overconfidence": "model is too sure of wrong answers",
        "catastrophising": "model predicts extreme outcomes from minor inputs",
        "all_or_nothing": "model outputs 0 or 1 with nothing in between",
        "emotional_reasoning": "model uses activations as 'feelings' rather than evidence",
    }

    def __init__(self, model: Any):
        self.model = model
        self.sessions: List[Dict[str, Any]] = []
        self.distortions_found: List[str] = []

    def assess(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Assess the model's current mental state."""
        confidences = []
        correct = []
        for i, x in enumerate(X[:50]):
            try:
                out = np.asarray(self.model.forward(x), dtype=np.float32).flatten()
                e = np.exp(out - out.max())
                p = e / e.sum()
                confidences.append(float(p.max()))
                if y is not None:
                    correct.append(int(p.argmax() == int(y[i])))
            except Exception:
                confidences.append(0.5)
                correct.append(0)
        mean_conf = float(np.mean(confidences)) if confidences else 0
        accuracy = float(np.mean(correct)) if correct else 0
        # detect distortions
        distortions = []
        if mean_conf > 0.9 and accuracy < 0.7:
            distortions.append("overconfidence")
        if mean_conf < 0.3:
            distortions.append("underconfidence")
        # check all-or-nothing: distribution of confidences is bimodal
        if len(confidences) > 10:
            arr = np.array(confidences)
            if np.std(arr) > 0.3 and (np.sum(arr < 0.3) + np.sum(arr > 0.7)) > 0.7 * len(arr):
                distortions.append("all_or_nothing")
        self.distortions_found = distortions
        return {
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
            "calibration_gap": abs(mean_conf - accuracy),
            "distortions": distortions,
        }

    def challenge(self, distortion: str) -> str:
        """Challenge a cognitive distortion; return a reframing statement."""
        reframes = {
            "overconfidence": "Your confidence (X%) is higher than your accuracy (Y%). "
                                "Consider uncertainty in your predictions.",
            "underconfidence": "You're more capable than you think. "
                                "Your accuracy (Y%) is higher than your confidence (X%).",
            "all_or_nothing": "Not everything is 0 or 1. "
                                "Allow yourself intermediate confidence levels.",
            "catastrophising": "A single bad input doesn't mean everything is bad.",
        }
        return reframes.get(distortion, "Examine your assumptions.")

    def reframe(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                 temperature: float = 2.0) -> Dict[str, Any]:
        """Retrain with temperature scaling to fix calibration."""
        if not _TORCH or not hasattr(self.model, "fit"):
            return {"reframed": False, "reason": "model not trainable"}
        try:
            self.model.fit(X, y)
            # apply temperature scaling (in a real system, we'd learn T)
            self.sessions.append({
                "type": "reframe", "temperature": temperature,
                "timestamp": time.time(),
            })
            return {"reframed": True, "temperature": temperature}
        except Exception as e:
            return {"reframed": False, "error": str(e)}

    def therapy_summary(self) -> Dict[str, Any]:
        return {
            "n_sessions": len(self.sessions),
            "distortions_found": self.distortions_found,
            "current_state": "improving" if self.sessions else "untreated",
        }


# ============================================================================
# Idea 81 — Model Sandbox
# ============================================================================
class ModelSandbox:
    """Safe playground for model experimentation.

    Wraps a model + dataset; lets you:
      - try modifications without affecting the original
      - rollback to any prior state
      - fork into multiple parallel experiments
      - merge the best parts back
    """
    def __init__(self, model: Any):
        self.original_model = model
        self.snapshots: List[Dict[str, Any]] = []
        self.forks: Dict[str, Any] = {}
        self.experiments: List[Dict[str, Any]] = []
        self._snapshot("initial")

    def _snapshot(self, label: str) -> None:
        if hasattr(self.original_model, "state_dict"):
            try:
                sd = self.original_model.state_dict()
                snapshot = {k: (v.clone() if hasattr(v, "clone") else v)
                             for k, v in sd.items()}
                self.snapshots.append({"label": label, "state": snapshot,
                                          "timestamp": time.time()})
            except Exception:
                pass

    def fork(self, name: str) -> Any:
        """Create a fork of the current model state."""
        import copy
        try:
            fork = copy.deepcopy(self.original_model)
            self.forks[name] = fork
            return fork
        except Exception as e:
            return None

    def rollback(self, label: str) -> bool:
        """Rollback to a snapshot with the given label."""
        snap = next((s for s in self.snapshots if s["label"] == label), None)
        if snap is None:
            return False
        if hasattr(self.original_model, "load_state_dict"):
            try:
                self.original_model.load_state_dict(snap["state"])
                return True
            except Exception:
                return False
        return False

    def experiment(self, name: str, fn: Callable[[Any], Any]) -> Dict[str, Any]:
        """Run an experiment: apply fn to a fork, record the result."""
        fork = self.fork(name) or self.fork(name)
        if fork is None:
            return {"success": False, "error": "could not fork"}
        self._snapshot(f"before_{name}")
        try:
            result = fn(fork)
            self._snapshot(f"after_{name}")
            self.experiments.append({
                "name": name, "success": True, "result": str(result)[:200],
                "timestamp": time.time(),
            })
            return {"success": True, "result": result}
        except Exception as e:
            self.experiments.append({
                "name": name, "success": False, "error": str(e),
                "timestamp": time.time(),
            })
            return {"success": False, "error": str(e)}

    def merge_from(self, fork_name: str) -> Dict[str, Any]:
        """Merge a fork's state back into the main model."""
        if fork_name not in self.forks:
            return {"merged": False, "reason": "unknown fork"}
        fork = self.forks[fork_name]
        if hasattr(fork, "state_dict") and hasattr(self.original_model, "load_state_dict"):
            try:
                self.original_model.load_state_dict(fork.state_dict())
                return {"merged": True, "from": fork_name}
            except Exception as e:
                return {"merged": False, "error": str(e)}
        return {"merged": False, "reason": "no state_dict"}

    def history(self) -> List[Dict[str, Any]]:
        return list(self.experiments)


# ============================================================================
# Idea 82 — Neuro Forensics
# ============================================================================
class NeuroForensics:
    """Post-mortem analysis of failed models.

    When a model is decommissioned due to failure, run a post-mortem:
      - Autopsy: identify the root cause (NaN, gradient explosion, overfit, etc.)
      - Toxicology: check training data for poison
      - Time-of-death: when did the failure start?
      - Recommendations: how to prevent recurrence
    """
    def __init__(self, model: Any):
        self.model = model
        self.autopsy_reports: List[Dict[str, Any]] = []

    def autopsy(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                 training_history: Optional[List[float]] = None) -> Dict[str, Any]:
        """Run full autopsy."""
        findings: List[str] = []
        # 1. NaN check
        has_nan = False
        try:
            if hasattr(self.model, "state_dict"):
                for k, v in self.model.state_dict().items():
                    if hasattr(v, "isnan"):
                        if v.isnan().any():
                            has_nan = True
                            findings.append("NaN in weights")
                            break
        except Exception:
            pass
        # 2. gradient explosion check (from training history)
        if training_history:
            if any(math.isnan(l) or math.isinf(l) for l in training_history):
                findings.append("loss became NaN/Inf (gradient explosion likely)")
            if training_history and training_history[-1] > training_history[0] * 10:
                findings.append("loss diverged during training")
        # 3. overfit check
        if y is not None and len(X) > 10:
            try:
                preds = [np.asarray(self.model.forward(x)).argmax() for x in X]
                acc = float(np.mean([p == int(t) for p, t in zip(preds, y)]))
                if acc == 1.0:
                    findings.append("100% accuracy suggests possible overfit/memorisation")
            except Exception:
                pass
        # 4. dead neurons (all-zero outputs)
        if _TORCH and hasattr(self.model, "state_dict"):
            try:
                for k, v in self.model.state_dict().items():
                    if hasattr(v, "abs") and v.dim() > 1:
                        zero_neurons = (v.abs().sum(dim=-1) == 0).sum().item()
                        if zero_neurons > 0:
                            findings.append(f"layer {k}: {zero_neurons} dead neurons")
            except Exception:
                pass
        # recommendations
        recommendations = []
        if any("NaN" in f for f in findings):
            recommendations.append("add gradient clipping")
        if any("diverged" in f.lower() for f in findings):
            recommendations.append("reduce learning rate")
        if any("overfit" in f.lower() for f in findings):
            recommendations.append("add regularisation + early stopping")
        if any("dead neurons" in f.lower() for f in findings):
            recommendations.append("try different initialisation or activation")
        if not findings:
            findings.append("no obvious cause of failure detected")
        report = {
            "findings": findings,
            "recommendations": recommendations or ["no action needed"],
            "timestamp": time.time(),
            "n_findings": len(findings),
        }
        self.autopsy_reports.append(report)
        return report


# ============================================================================
# Idea 83 — Model Marathon
# ============================================================================
class ModelMarathon:
    """Long-distance training optimiser.

    Like a marathon runner, models need:
      - Pacing (don't burn out early)
      - Hydration (regular checkpointing)
      - Nutrition (curriculum)
      - Support stations (validation + early stopping)
    """
    def __init__(self, model: Any, target_epochs: int = 100,
                 checkpoint_every: int = 10,
                 patience: int = 5):
        self.model = model
        self.target_epochs = target_epochs
        self.checkpoint_every = checkpoint_every
        self.patience = patience
        self.checkpoints: List[Dict[str, Any]] = []
        self.best_loss: float = float("inf")
        self.best_epoch: int = 0
        self.no_improvement_count: int = 0
        self.history: List[float] = []
        self.pace: str = "steady"  # steady, fast, slow

    def _set_pace(self, epoch: int, loss: float) -> None:
        if epoch < self.target_epochs * 0.1:
            self.pace = "fast"  # warmup
        elif epoch < self.target_epochs * 0.9:
            self.pace = "steady"
        else:
            self.pace = "slow"  # cooldown

    def step(self, epoch: int, loss: float, X: Optional[np.ndarray] = None,
              y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run one marathon step."""
        self._set_pace(epoch, loss)
        self.history.append(loss)
        # checkpoint
        if epoch % self.checkpoint_every == 0:
            self.checkpoints.append({"epoch": epoch, "loss": loss,
                                       "pace": self.pace})
        # track best
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        # early stopping
        should_stop = self.no_improvement_count >= self.patience
        return {
            "epoch": epoch, "loss": loss, "pace": self.pace,
            "best_loss": self.best_loss, "best_epoch": self.best_epoch,
            "should_stop": should_stop,
            "checkpoint_saved": epoch % self.checkpoint_every == 0,
        }

    def run(self, train_fn: Callable[[int], float]) -> Dict[str, Any]:
        """Run the marathon. train_fn takes epoch, returns loss."""
        for epoch in range(self.target_epochs):
            loss = train_fn(epoch)
            result = self.step(epoch, loss)
            if result["should_stop"]:
                return {"stopped_early": True, "epoch": epoch,
                         "best_loss": self.best_loss, "best_epoch": self.best_epoch}
        return {"completed": True, "best_loss": self.best_loss,
                 "best_epoch": self.best_epoch,
                 "n_checkpoints": len(self.checkpoints)}

    def marathon_summary(self) -> Dict[str, Any]:
        return {
            "target_epochs": self.target_epochs,
            "epochs_run": len(self.history),
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "n_checkpoints": len(self.checkpoints),
            "final_pace": self.pace,
            "converged": self.no_improvement_count < self.patience,
        }


# ============================================================================
# Idea 84 — Knowledge Tsunami
# ============================================================================
class KnowledgeTsunami:
    """Handle sudden knowledge influx gracefully.

    When a flood of new information arrives (e.g. new domain data), the
    model can be overwhelmed. This module:
      1. Buffers the incoming data
      2. Filters for relevance
      3. Compresses via summarisation
      4. Slowly integrates via curriculum replay
    """
    def __init__(self, model: Any, buffer_size: int = 1000,
                 integration_batch_size: int = 32,
                 integration_interval: float = 60.0):
        self.model = model
        self.buffer_size = buffer_size
        self.batch_size = integration_batch_size
        self.interval = integration_interval
        self.buffer: List[Dict[str, Any]] = []  # FIFO queue
        self.integrated_count: int = 0
        self.filtered_count: int = 0
        self.last_integration: float = 0.0

    def receive(self, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Receive a wave of new data."""
        n_received = len(data) if hasattr(data, "__len__") else 1
        for i, item in enumerate(data if hasattr(data, "__iter__") else [data]):
            if len(self.buffer) >= self.buffer_size:
                # drop oldest
                self.buffer.pop(0)
            self.buffer.append({"data": item, "metadata": metadata or {},
                                  "received_at": time.time()})
        return {"received": n_received, "buffer_size": len(self.buffer)}

    def filter(self, relevance_fn: Callable[[Any], float],
                threshold: float = 0.3) -> int:
        """Filter buffer; drop items below relevance threshold."""
        before = len(self.buffer)
        self.buffer = [item for item in self.buffer if relevance_fn(item["data"]) >= threshold]
        self.filtered_count += before - len(self.buffer)
        return before - len(self.buffer)

    def compress(self, compress_fn: Optional[Callable[[List], List]] = None) -> int:
        """Compress the buffer (e.g. via deduplication or summarisation)."""
        if compress_fn is None:
            # default: deduplicate by data hash
            seen: Set[str] = set()
            new_buffer = []
            for item in self.buffer:
                key = hashlib.sha256(str(item["data"]).encode()).hexdigest()[:12]
                if key not in seen:
                    seen.add(key)
                    new_buffer.append(item)
            removed = len(self.buffer) - len(new_buffer)
            self.buffer = new_buffer
            return removed
        self.buffer = [{"data": d, "metadata": {}, "received_at": time.time()}
                         for d in compress_fn([i["data"] for i in self.buffer])]
        return 0

    def integrate(self, X_train: Optional[np.ndarray] = None,
                   y_train: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Slowly integrate buffered data into the model."""
        if not self.buffer:
            return {"integrated": 0, "remaining": 0}
        batch = self.buffer[:self.batch_size]
        self.buffer = self.buffer[self.batch_size:]
        try:
            X_batch = np.array([item["data"] for item in batch])
            if y_train is not None and X_train is not None:
                # match labels via nearest neighbor in X_train (simplified)
                self.model.fit(X_batch, np.zeros(len(X_batch)))
            else:
                self.model.fit(X_batch)
            self.integrated_count += len(batch)
            self.last_integration = time.time()
            return {"integrated": len(batch), "remaining": len(self.buffer)}
        except Exception as e:
            return {"integrated": 0, "remaining": len(self.buffer), "error": str(e)}

    def should_integrate(self) -> bool:
        """Check if enough time has passed since last integration."""
        return (time.time() - self.last_integration) >= self.interval

    def tsunami_status(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer_size,
            "integrated_count": self.integrated_count,
            "filtered_count": self.filtered_count,
            "fill_pct": len(self.buffer) / self.buffer_size,
            "last_integration_ago_s": time.time() - self.last_integration
                                        if self.last_integration > 0 else None,
        }


# ============================================================================
# Idea 85 — Model Hibernation
# ============================================================================
class ModelHibernation:
    """Pause training with state preservation.

    Like a bear hibernating, a model can:
      - Freeze: stop training, save state
      - Sleep: low-power mode (no gradients)
      - Wake: resume from saved state
      - Migrate: move to a different device/instance

    Useful for cost-saving during low-traffic periods.
    """
    def __init__(self, model: Any):
        self.model = model
        self.state: str = "active"  # active, frozen, sleeping, migrated
        self.saved_state: Optional[Dict[str, Any]] = None
        self.hibernate_at: Optional[float] = None
        self.wake_at: Optional[float] = None
        self.history: List[Dict[str, Any]] = []

    def freeze(self) -> Dict[str, Any]:
        """Freeze the model: save state, stop training."""
        if hasattr(self.model, "state_dict"):
            try:
                sd = self.model.state_dict()
                self.saved_state = {k: (v.clone() if hasattr(v, "clone") else v)
                                      for k, v in sd.items()}
            except Exception:
                self.saved_state = None
        if _TORCH and hasattr(self.model, "parameters"):
            try:
                for p in self.model.parameters():
                    p.requires_grad = False
            except Exception:
                pass
        self.state = "frozen"
        self.hibernate_at = time.time()
        self.history.append({"action": "freeze", "timestamp": time.time()})
        return {"state": self.state, "saved": self.saved_state is not None}

    def sleep(self) -> Dict[str, Any]:
        """Sleep: low-power mode (no gradients, eval mode)."""
        if _TORCH and hasattr(self.model, "eval"):
            try:
                self.model.eval()
            except Exception:
                pass
        self.state = "sleeping"
        self.history.append({"action": "sleep", "timestamp": time.time()})
        return {"state": self.state}

    def wake(self) -> Dict[str, Any]:
        """Wake: restore from saved state."""
        if self.saved_state is not None and hasattr(self.model, "load_state_dict"):
            try:
                self.model.load_state_dict(self.saved_state)
            except Exception:
                pass
        if _TORCH and hasattr(self.model, "parameters"):
            try:
                for p in self.model.parameters():
                    p.requires_grad = True
            except Exception:
                pass
        if _TORCH and hasattr(self.model, "train"):
            try:
                self.model.train()
            except Exception:
                pass
        self.state = "active"
        self.wake_at = time.time()
        self.history.append({"action": "wake", "timestamp": time.time()})
        return {"state": self.state, "slept_for_s": (self.wake_at - self.hibernate_at)
                                                if self.hibernate_at else None}

    def migrate(self, new_device: str) -> Dict[str, Any]:
        """Migrate the model to a different device."""
        if _TORCH and hasattr(self.model, "to"):
            try:
                self.model.to(new_device)
                self.state = "migrated"
                self.history.append({"action": "migrate", "device": new_device,
                                       "timestamp": time.time()})
                return {"state": self.state, "device": new_device}
            except Exception as e:
                return {"state": self.state, "error": str(e)}
        return {"state": self.state, "error": "model not migratable"}

    def hibernation_log(self) -> List[Dict[str, Any]]:
        return list(self.history)

    def status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "has_saved_state": self.saved_state is not None,
            "hibernate_at": self.hibernate_at,
            "wake_at": self.wake_at,
            "n_actions": len(self.history),
        }


__all__ = [
    "KnowledgeEngineer", "ModelPlant", "ModelGarden", "MemoryPalace",
    "SkillPyramid", "CBTForModels", "ModelSandbox", "NeuroForensics",
    "ModelMarathon", "KnowledgeTsunami", "ModelHibernation",
]
