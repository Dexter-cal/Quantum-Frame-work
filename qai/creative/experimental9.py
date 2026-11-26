"""
qai.creative.experimental9 — 8 more original creative ideas (part 9).

  Idea 93  — ModelPrison      : quarantine misbehaving models
  Idea 94  — NeuralEcosystem  : predator-prey dynamics between models
  Idea 95  — WeightDna        : genetic code for model weights
  Idea 96  — ModelHospital    : diagnose + treat sick models
  Idea 97  — DataMarketplace  : buy/sell datasets with smart pricing
  Idea 98  — ModelPassport    : identity document for models (border control)
  Idea 99  — NeuralReligion   : models with belief systems + schisms
  Idea 100 — ModelCemetery    : dignified retirement + epitaphs for dead models
"""
from __future__ import annotations
import math, random, time, hashlib, json, os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np


class ModelPrison:
    """Quarantine misbehaving models.

    When a model produces dangerous, wrong, or anomalous outputs, it gets
    "arrested" and put in prison. In prison, it can't serve traffic.
    After rehabilitation (retraining), it can be released on parole.

    Cells:
      - Solitary: complete isolation (model frozen, no inference)
      - Minimum security: can run inference but outputs are filtered
      - Parole: released but monitored
    """
    SECURITY_LEVELS = ["solitary", "minimum", "parole", "free"]

    def __init__(self):
        self.inmates: Dict[str, Dict[str, Any]] = {}
        self.visitation_log: List[Dict[str, Any]] = []

    def arrest(self, model_id: str, model: Any, offense: str,
                severity: str = "medium") -> Dict[str, Any]:
        """Arrest a model for an offense."""
        level = "solitary" if severity == "high" else "minimum" if severity == "medium" else "minimum"
        self.inmates[model_id] = {
            "model": model, "offense": offense, "severity": severity,
            "security_level": level, "sentence_start": time.time(),
            "n_violations": 1, "parole_eligible": False, "released": False,
        }
        return {"arrested": True, "model_id": model_id, "level": level, "offense": offense}

    def visit(self, model_id: str, input_data: Any) -> Dict[str, Any]:
        """Visit an inmate (run inference with filtering)."""
        if model_id not in self.inmates:
            return {"error": "not an inmate"}
        inmate = self.inmates[model_id]
        if inmate["security_level"] == "solitary":
            return {"blocked": True, "reason": "in solitary confinement"}
        try:
            output = inmate["model"].forward(input_data)
            # minimum security: filter output
            if inmate["security_level"] == "minimum":
                if isinstance(output, (int, float)) and abs(output) > 1000:
                    return {"blocked": True, "reason": "output filtered (too extreme)"}
            self.visitation_log.append({"model_id": model_id, "timestamp": time.time()})
            return {"output": output, "filtered": inmate["security_level"] == "minimum"}
        except Exception as e:
            return {"error": str(e)}

    def rehabilitate(self, model_id: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Attempt rehabilitation via retraining."""
        if model_id not in self.inmates:
            return {"error": "not an inmate"}
        inmate = self.inmates[model_id]
        try:
            inmate["model"].fit(X, y)
            inmate["n_violations"] = max(0, inmate["n_violations"] - 1)
            if inmate["n_violations"] == 0:
                inmate["parole_eligible"] = True
            return {"rehabilitated": True, "parole_eligible": inmate["parole_eligible"]}
        except Exception as e:
            return {"rehabilitated": False, "error": str(e)}

    def release(self, model_id: str, on_parole: bool = True) -> Dict[str, Any]:
        """Release a model from prison."""
        if model_id not in self.inmates:
            return {"error": "not an inmate"}
        inmate = self.inmates[model_id]
        if not inmate["parole_eligible"] and on_parole:
            return {"released": False, "reason": "not eligible for parole"}
        inmate["released"] = True
        inmate["security_level"] = "parole" if on_parole else "free"
        return {"released": True, "on_parole": on_parole, "model_id": model_id}

    def prison_status(self) -> Dict[str, Any]:
        return {
            "n_inmates": len(self.inmates),
            "inmates": {mid: {"offense": i["offense"], "level": i["security_level"],
                              "violations": i["n_violations"], "parole_eligible": i["parole_eligible"]}
                        for mid, i in self.inmates.items()},
            "n_visitations": len(self.visitation_log),
        }


class NeuralEcosystem:
    """Predator-prey dynamics between models.

    Models compete for resources (data, compute). Strong models "eat"
    weak ones (absorb their knowledge). Creates a natural selection
    pressure that improves the population over time.

    Lotka-Volterra equations adapted for ML:
      dP/dt = α*P - β*P*Q  (prey = weak models)
      dQ/dt = δ*P*Q - γ*Q  (predator = strong models)
    """
    def __init__(self):
        self.population: Dict[str, Dict[str, Any]] = {}
        self.generations: List[Dict[str, Any]] = []
        self._resources: float = 1000.0  # total compute available

    def add_species(self, name: str, model: Any,
                     fitness: float = 0.5, is_predator: bool = False) -> None:
        self.population[name] = {
            "model": model, "fitness": fitness, "is_predator": is_predator,
            "population_size": 10, "energy": 100.0,
        }

    def tick(self) -> Dict[str, Any]:
        """One ecosystem tick (apply Lotka-Volterra dynamics)."""
        prey = [n for n, s in self.population.items() if not s["is_predator"]]
        predators = [n for n, s in self.population.items() if s["is_predator"]]
        alpha, beta, gamma, delta = 0.1, 0.02, 0.1, 0.01
        for p in prey:
            pred_pressure = sum(self.population[pr]["population_size"] for pr in predators)
            growth = alpha * self.population[p]["population_size"] - beta * self.population[p]["population_size"] * pred_pressure
            self.population[p]["population_size"] = max(0, self.population[p]["population_size"] + growth)
            self.population[p]["energy"] -= 1
        for pr in predators:
            prey_available = sum(self.population[p]["population_size"] for p in prey)
            growth = delta * self.population[pr]["population_size"] * prey_available - gamma * self.population[pr]["population_size"]
            self.population[pr]["population_size"] = max(0, self.population[pr]["population_size"] + growth)
        # extinction
        extinct = [n for n, s in self.population.items() if s["population_size"] <= 0]
        for n in extinct:
            del self.population[n]
        stats = {
            "n_species": len(self.population),
            "n_prey": len([s for s in self.population.values() if not s["is_predator"]]),
            "n_predators": len([s for s in self.population.values() if s["is_predator"]]),
            "extinct": extinct,
            "total_population": sum(s["population_size"] for s in self.population.values()),
        }
        self.generations.append(stats)
        return stats

    def evolve(self, n_ticks: int = 10) -> Dict[str, Any]:
        for _ in range(n_ticks):
            self.tick()
        return {
            "ticks": n_ticks,
            "survivors": list(self.population.keys()),
            "generations": self.generations[-5:],
        }


class WeightDna:
    """Genetic code for model weights.

    Encodes each weight as a 4-letter DNA sequence:
      A (Adenine)  = positive large weight
      T (Thymine)  = positive small weight
      C (Cytosine) = negative small weight
      G (Guanine)  = negative large weight

    This allows:
      - Genetic comparison between models (sequence alignment)
      - Mutation tracking (point mutations in weight DNA)
      - Ancestry analysis (compare DNA across model generations)
      - CRISPR-like editing (find + replace specific weight patterns)
    """
    @staticmethod
    def encode(weights: np.ndarray) -> str:
        """Encode weight array as DNA sequence."""
        arr = weights.flatten()
        max_abs = max(np.abs(arr).max(), 1e-8)
        dna = []
        for v in arr:
            ratio = v / max_abs
            if ratio > 0.3: dna.append("A")
            elif ratio > 0: dna.append("T")
            elif ratio > -0.3: dna.append("C")
            else: dna.append("G")
        return "".join(dna)

    @staticmethod
    def decode(dna: str, shape: Tuple[int, ...], scale: float = 1.0) -> np.ndarray:
        """Decode DNA sequence back to weights."""
        mapping = {"A": 0.7, "T": 0.1, "C": -0.1, "G": -0.7}
        arr = np.array([mapping.get(c, 0.0) for c in dna], dtype=np.float32) * scale
        return arr.reshape(shape)

    @staticmethod
    def model_dna(model: Any) -> Dict[str, str]:
        """Get DNA for all layers in a model."""
        from ..weights import _extract_weights
        weights = _extract_weights(model)
        return {name: WeightDna.encode(arr) for name, arr in weights.items()}

    @staticmethod
    def compare_dna(dna_a: str, dna_b: str) -> Dict[str, Any]:
        """Compare two DNA sequences (alignment score + mutations)."""
        min_len = min(len(dna_a), len(dna_b))
        matches = sum(1 for i in range(min_len) if dna_a[i] == dna_b[i])
        mutations = min_len - matches
        return {
            "similarity": matches / max(min_len, 1),
            "n_mutations": mutations,
            "length_a": len(dna_a), "length_b": len(dna_b),
            "genetic_distance": mutations / max(min_len, 1),
        }

    @staticmethod
    def crispr_edit(dna: str, find: str, replace: str) -> str:
        """CRISPR-like find-and-replace in weight DNA."""
        return dna.replace(find, replace)


class ModelHospital:
    """Diagnose + treat sick models.

    Symptoms:
      - Fever (temperature = high loss)
      - Amnesia (forgetting old tasks)
      - Paralysis (can't learn — dead gradients)
      - Obesity (too many params, overfitting)
      - Anorexia (too few params, underfitting)
      - Hallucinations (confident wrong predictions)

    Treatments:
      - Antibiotics (gradient clipping for exploding gradients)
      - Surgery (prune dead neurons)
      - Diet (regularization for overfitting)
      - Transfusion (transfer learning from healthy model)
      - Therapy (curriculum learning)
      - Vaccine (adversarial training for robustness)
    """
    SYMPTOMS = {
        "fever": "High loss — model is struggling to learn",
        "amnesia": "Catastrophic forgetting — lost old task performance",
        "paralysis": "Dead gradients — model can't learn",
        "obesity": "Overfitting — too many params for the data",
        "anorexia": "Underfitting — too few params to capture patterns",
        "hallucinations": "Confident wrong predictions",
    }

    TREATMENTS = {
        "antibiotics": "Gradient clipping — prevents exploding gradients",
        "surgery": "Prune dead neurons — remove non-functional parts",
        "diet": "Regularization (L1/L2/dropout) — prevent overfitting",
        "transfusion": "Transfer learning — inject knowledge from healthy model",
        "therapy": "Curriculum learning — retrain with progressive difficulty",
        "vaccine": "Adversarial training — improve robustness",
    }

    @staticmethod
    def diagnose(model: Any, X: np.ndarray = None,
                  y: np.ndarray = None, history: List[float] = None) -> Dict[str, Any]:
        """Diagnose a model's health."""
        symptoms: List[str] = []
        from ..weights import _extract_weights, validate_weights
        weights = _extract_weights(model)
        # check for dead neurons (paralysis)
        for name, arr in weights.items():
            if arr.ndim > 1 and (np.linalg.norm(arr, axis=-1) < 1e-8).any():
                symptoms.append("paralysis")
                break
        # check for overfitting (obesity)
        if history and len(history) > 5:
            if history[-1] < history[-3] * 0.5 and history[-1] < 0.1:
                symptoms.append("obesity")
        # check for underfitting (anorexia)
        if history and len(history) > 5:
            if history[-1] > 0.5:
                symptoms.append("anorexia")
        # check for high loss (fever)
        if history and history[-1] > 1.0:
            symptoms.append("fever")
        # check weight stats
        all_weights = np.concatenate([arr.flatten() for arr in weights.values()])
        if np.abs(all_weights).max() > 100:
            symptoms.append("hallucinations")
        if not symptoms:
            symptoms.append("healthy")
        return {
            "symptoms": symptoms,
            "n_symptoms": len(symptoms) - (1 if "healthy" in symptoms else 0),
            "is_healthy": "healthy" in symptoms,
            "recommended_treatments": [ModelHospital._recommend_treatment(s) for s in symptoms if s != "healthy"],
        }

    @staticmethod
    def _recommend_treatment(symptom: str) -> str:
        return {
            "fever": "diet", "amnesia": "therapy", "paralysis": "surgery",
            "obesity": "diet", "anorexia": "transfusion", "hallucinations": "vaccine",
        }.get(symptom, "therapy")


class DataMarketplace:
    """Buy/sell datasets with smart pricing based on value + quality.

    Each dataset has:
      - Price (based on size, quality, freshness, uniqueness)
      - Quality score (completeness, accuracy, consistency)
      - Seller reputation
      - Reviews from buyers

    Smart pricing: price = f(size, quality, freshness, demand, uniqueness)
    """
    def __init__(self):
        self.listings: Dict[str, Dict[str, Any]] = {}
        self.transactions: List[Dict[str, Any]] = []
        self.reviews: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def list_dataset(self, dataset_id: str, n_rows: int, n_cols: int,
                      quality: float = 0.8, freshness_days: int = 1,
                      seller: str = "unknown", uniqueness: float = 0.5) -> Dict[str, Any]:
        """List a dataset for sale."""
        # smart pricing
        base = n_rows * n_cols * 0.001
        quality_mult = 0.5 + quality
        freshness_mult = max(0.5, 1.0 - freshness_days / 365)
        uniqueness_mult = 0.5 + uniqueness
        price = base * quality_mult * freshness_mult * uniqueness_mult
        self.listings[dataset_id] = {
            "n_rows": n_rows, "n_cols": n_cols, "quality": quality,
            "freshness_days": freshness_days, "seller": seller,
            "uniqueness": uniqueness, "price": price,
            "n_sold": 0, "revenue": 0.0, "listed_at": time.time(),
        }
        return {"listed": True, "dataset_id": dataset_id, "price": price}

    def buy(self, dataset_id: str, buyer: str) -> Dict[str, Any]:
        """Buy a dataset."""
        if dataset_id not in self.listings:
            return {"error": "not listed"}
        listing = self.listings[dataset_id]
        price = listing["price"]
        # demand pricing: price goes up with sales
        listing["price"] *= 1.05
        listing["n_sold"] += 1
        listing["revenue"] += price
        tx = {"dataset_id": dataset_id, "buyer": buyer, "price": price, "timestamp": time.time()}
        self.transactions.append(tx)
        return {"purchased": True, "price": price, "dataset_id": dataset_id}

    def review(self, dataset_id: str, reviewer: str, rating: int,
                comment: str = "") -> None:
        """Review a dataset."""
        self.reviews[dataset_id].append({"reviewer": reviewer, "rating": rating,
                                           "comment": comment, "timestamp": time.time()})

    def market_summary(self) -> Dict[str, Any]:
        return {
            "n_datasets": len(self.listings),
            "n_transactions": len(self.transactions),
            "total_revenue": sum(l["revenue"] for l in self.listings.values()),
            "top_seller": max(self.listings.values(), key=lambda l: l["revenue"])["seller"] if self.listings else None,
            "avg_price": float(np.mean([l["price"] for l in self.listings.values()])) if self.listings else 0,
        }


class ModelPassport:
    """Identity document for models (border control).

    Contains:
      - Model identity (name, version, fingerprint)
      - Origin (creator, training data, date)
      - Vaccinations (safety checks passed)
      - Travel history (deployment environments)
      - Visas (approved for specific tasks/domains)

    Border control checks passports before allowing deployment.
    """
    def __init__(self, model_id: str, technique: str, version: str = "1.0.0"):
        self.passport = {
            "model_id": model_id, "technique": technique, "version": version,
            "fingerprint": "", "created_at": time.time(),
            "creator": "unknown", "training_data_hash": "",
            "vaccinations": [], "travel_history": [], "visas": [],
            "revoked": False,
        }

    def add_vaccination(self, vaccine: str, result: bool = True) -> None:
        """Record a safety check (vaccination)."""
        self.passport["vaccinations"].append({"name": vaccine, "passed": result,
                                                "timestamp": time.time()})

    def add_visa(self, domain: str, approved_by: str,
                  expires: Optional[float] = None) -> None:
        """Grant a visa for a specific domain/task."""
        self.passport["visas"].append({"domain": domain, "approved_by": approved_by,
                                        "expires": expires, "granted_at": time.time()})

    def record_travel(self, environment: str, status: str = "deployed") -> None:
        """Record a deployment (travel entry)."""
        self.passport["travel_history"].append({"environment": environment,
                                                   "status": status, "timestamp": time.time()})

    def border_control(self, target_domain: str) -> Dict[str, Any]:
        """Check if model can enter a domain."""
        if self.passport["revoked"]:
            return {"allowed": False, "reason": "passport revoked"}
        has_visa = any(v["domain"] == target_domain for v in self.passport["visas"])
        if not has_visa:
            return {"allowed": False, "reason": f"no visa for domain '{target_domain}'"}
        all_vaccinated = all(v["passed"] for v in self.passport["vaccinations"])
        if not all_vaccinated:
            failed = [v["name"] for v in self.passport["vaccinations"] if not v["passed"]]
            return {"allowed": False, "reason": f"failed safety checks: {failed}"}
        return {"allowed": True, "passport": self.passport}

    def revoke(self, reason: str) -> None:
        """Revoke the passport."""
        self.passport["revoked"] = True
        self.passport["revocation_reason"] = reason


class ModelCemetery:
    """Dignified retirement + epitaphs for dead models.

    Each grave records:
      - Model name + technique
      - Birth (training date) → Death (decommission date)
      - Lifespan (how long it served)
      - Best known for (peak accuracy)
      - Epitaph (auto-generated or custom)
      - Visitors (who has loaded the archived model)
    """
    def __init__(self):
        self.graves: Dict[str, Dict[str, Any]] = {}

    def bury(self, model_id: str, technique: str,
              born_at: float, died_at: float = None,
              peak_accuracy: float = 0.0, epitaph: str = "") -> Dict[str, Any]:
        """Bury a model with dignity."""
        died_at = died_at or time.time()
        lifespan = died_at - born_at
        if not epitaph:
            epitaph = self._auto_epitaph(technique, peak_accuracy, lifespan)
        grave = {
            "model_id": model_id, "technique": technique,
            "born_at": born_at, "died_at": died_at,
            "lifespan_days": lifespan / 86400,
            "peak_accuracy": peak_accuracy, "epitaph": epitaph,
            "visitors": 0, "flowers": 0,
        }
        self.graves[model_id] = grave
        return grave

    @staticmethod
    def _auto_epitaph(technique: str, accuracy: float, lifespan: float) -> str:
        days = lifespan / 86400
        if accuracy > 0.9:
            return f"Here lies {technique}. It served with {accuracy:.1%} accuracy for {days:.0f} days. A life well-lived."
        elif accuracy > 0.7:
            return f"Here lies {technique}. It tried its best at {accuracy:.1%}. We remember its contributions."
        else:
            return f"Here lies {technique}. Not all models are meant for greatness. It served for {days:.0f} days."

    def visit(self, model_id: str) -> Dict[str, Any]:
        """Visit a grave."""
        if model_id not in self.graves:
            return {"error": "grave not found"}
        self.graves[model_id]["visitors"] += 1
        return self.graves[model_id]

    def leave_flowers(self, model_id: str) -> None:
        if model_id in self.graves:
            self.graves[model_id]["flowers"] += 1

    def cemetery_walk(self) -> List[Dict[str, Any]]:
        """Walk through the cemetery and read all epitaphs."""
        return [{"model_id": g["model_id"], "epitaph": g["epitaph"],
                 "lifespan_days": g["lifespan_days"]} for g in self.graves.values()]


__all__ = [
    "ModelPrison", "NeuralEcosystem", "WeightDna",
    "ModelHospital", "DataMarketplace", "ModelPassport", "ModelCemetery",
]
