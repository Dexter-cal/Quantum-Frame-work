"""
qai.creative.experimental5 -- 10 more original creative ideas (my own, part 5).

  Idea 56 — DataTrust                  : data valuation + trust scoring market
  Idea 57 — ModelPharmacy              : drug-like model "dosing" (small model
                                          taken at intervals as cognitive aid)
  Idea 58 — ModelUnion2                : 3-way merges with conflict resolution
  Idea 59 — SkillEvolution2            : cross-species skill mating (different
                                          genome shapes can interbreed)
  Idea 60 — ModelZoo                   : auto-encapsulated ecosystem of related
                                          models that share knowledge
  Idea 61 — AdversarialEvolutionaryArmsRace : generator + attacker co-evolve
  Idea 62 — ModelGenealogy             : family tree + heredity tracking
  Idea 63 — DataValuationMarket        : Shapley-value-based data pricing
  Idea 64 — ModelEmbryology            : developmental stages of a model
  Idea 65 — NeuralSociology            : model social networks + influence graphs
"""
from __future__ import annotations
import math
import random
import time
import hashlib
import itertools
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
# Idea 56 — Data Trust
# ============================================================================
class DataTrust:
    """Data valuation + trust scoring market.

    Each data source is rated for:
      - Quality (correctness, completeness)
      - Freshness (how recently updated)
      - Coverage (how many entities/attributes it spans)
      - Trust score (historical reliability)

    Buyers pay for access; sellers earn based on the value contributed
    (measured via leave-one-out performance drop).
    """
    def __init__(self):
        self.sources: Dict[str, Dict[str, Any]] = {}
        self.transactions: List[Dict[str, Any]] = []

    def register_source(self, source_id: str, n_samples: int,
                          quality_score: float = 0.8,
                          freshness_days: int = 1,
                          coverage: float = 0.5) -> None:
        self.sources[source_id] = {
            "n_samples": n_samples,
            "quality": quality_score,
            "freshness_days": freshness_days,
            "coverage": coverage,
            "trust": 0.5,  # starts neutral
            "total_earnings": 0.0,
            "n_transactions": 0,
        }

    def value(self, source_id: str) -> Dict[str, Any]:
        """Compute a value score for a source."""
        s = self.sources[source_id]
        # composite score: weighted combination
        freshness = max(0.0, 1.0 - s["freshness_days"] / 365)
        score = (
            0.4 * s["quality"]
            + 0.2 * freshness
            + 0.2 * s["coverage"]
            + 0.2 * s["trust"]
        )
        return {
            "value_score": float(score),
            "components": {
                "quality": s["quality"], "freshness": freshness,
                "coverage": s["coverage"], "trust": s["trust"],
            },
            "suggested_price": float(score * s["n_samples"] * 0.001),
        }

    def buy(self, buyer_id: str, source_id: str) -> Dict[str, Any]:
        """Buy access to a source; price = suggested_price * (1 + demand premium)."""
        if source_id not in self.sources:
            raise KeyError(f"unknown source '{source_id}'")
        v = self.value(source_id)
        # demand premium: more transactions = higher price
        premium = 1.0 + 0.05 * self.sources[source_id]["n_transactions"]
        price = v["suggested_price"] * premium
        self.sources[source_id]["total_earnings"] += price
        self.sources[source_id]["n_transactions"] += 1
        tx = {
            "buyer": buyer_id, "source": source_id,
            "price": price, "timestamp": time.time(),
        }
        self.transactions.append(tx)
        return tx

    def update_trust(self, source_id: str, outcome: float) -> None:
        """Update a source's trust score based on a transaction outcome (0/1)."""
        if source_id not in self.sources:
            return
        # EMA update
        s = self.sources[source_id]
        s["trust"] = 0.95 * s["trust"] + 0.05 * float(outcome)

    def leaderboard(self) -> List[Dict[str, Any]]:
        items = []
        for sid, info in self.sources.items():
            v = self.value(sid)
            items.append({
                "source_id": sid,
                "value_score": v["value_score"],
                "earnings": info["total_earnings"],
                "n_transactions": info["n_transactions"],
                "trust": info["trust"],
            })
        items.sort(key=lambda x: -x["value_score"])
        return items


# ============================================================================
# Idea 57 — Model Pharmacy
# ============================================================================
class ModelPharmacy:
    """Drug-like model "dosing" — small models taken at intervals as cognitive aids.

    Treats small specialised models like pharmaceuticals:
      - Each has an active ingredient (the model itself)
      - Each has a dosage (how long to invoke per turn)
      - Each has side effects (compute cost, latency)
      - Each has interactions with other models
      - A "prescription" is a schedule of model invocations
    """
    def __init__(self):
        self.catalogue: Dict[str, Dict[str, Any]] = {}
        self.prescriptions: List[Dict[str, Any]] = []

    def register_drug(self, drug_id: str, model: Any,
                       dose_seconds: float = 0.1,
                       side_effects: Optional[List[str]] = None,
                       interactions: Optional[List[str]] = None) -> None:
        self.catalogue[drug_id] = {
            "model": model,
            "dose_seconds": dose_seconds,
            "side_effects": side_effects or [],
            "interactions": interactions or [],
        }

    def prescribe(self, patient_id: str, schedule: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prescribe a schedule of drugs.

        Each schedule entry: {drug_id, frequency_per_day, duration_days}
        """
        # validate interactions
        drug_ids = [s["drug_id"] for s in schedule]
        for s in schedule:
            drug = self.catalogue.get(s["drug_id"])
            if drug is None:
                raise KeyError(f"unknown drug '{s['drug_id']}'")
            for inter in drug["interactions"]:
                if inter in drug_ids and inter != s["drug_id"]:
                    return {
                        "prescribed": False,
                        "reason": f"interaction: {s['drug_id']} conflicts with {inter}",
                        "patient": patient_id,
                    }
        prescription = {
            "patient_id": patient_id,
            "schedule": schedule,
            "prescribed_at": time.time(),
            "total_doses": sum(s["frequency_per_day"] * s["duration_days"] for s in schedule),
        }
        self.prescriptions.append(prescription)
        return {"prescribed": True, **prescription}

    def administer(self, drug_id: str, input_data: Any) -> Dict[str, Any]:
        """Administer one dose of a drug; return the model's effect."""
        drug = self.catalogue.get(drug_id)
        if drug is None:
            raise KeyError(f"unknown drug '{drug_id}'")
        try:
            effect = drug["model"].forward(input_data)
        except Exception as e:
            effect = f"[side effect] {e}"
        return {
            "drug": drug_id,
            "effect": effect,
            "dose_seconds": drug["dose_seconds"],
            "side_effects": drug["side_effects"],
        }


# ============================================================================
# Idea 58 — Model Union 2 (3-way merges with conflict resolution)
# ============================================================================
class ModelUnion2:
    """3-way model merge with conflict resolution.

    Given a base + two fine-tuned variants, merge them. When A and B
    disagree (different signs of deltas), use the base as a tiebreaker.
    """
    def __init__(self, base: Any, model_a: Any, model_b: Any):
        self.base = base
        self.a = model_a
        self.b = model_b

    def _get_weights(self, model: Any) -> Dict[str, np.ndarray]:
        if hasattr(model, "state_dict"):
            try:
                return {k: np.asarray(v, dtype=np.float32)
                        for k, v in model.state_dict().items()}
            except Exception:
                pass
        return {}

    def _set_weights(self, model: Any, weights: Dict[str, np.ndarray]) -> None:
        if hasattr(model, "load_state_dict"):
            try:
                model.load_state_dict(weights)
                return
            except Exception:
                pass

    def merge(self, strategy: str = "conflict_aware") -> Dict[str, Any]:
        w_base = self._get_weights(self.base)
        w_a = self._get_weights(self.a)
        w_b = self._get_weights(self.b)
        merged: Dict[str, np.ndarray] = {}
        conflicts = 0
        agreements = 0
        for k in w_a:
            if k not in w_b or k not in w_base:
                merged[k] = w_a.get(k, w_b.get(k))
                continue
            if w_a[k].shape != w_b[k].shape:
                merged[k] = w_a[k]
                continue
            delta_a = w_a[k] - w_base[k]
            delta_b = w_b[k] - w_base[k]
            # detect conflicts: opposite signs of deltas
            sign_a = np.sign(delta_a)
            sign_b = np.sign(delta_b)
            conflict_mask = (sign_a * sign_b) < 0
            conflicts += int(conflict_mask.sum())
            agreements += int((~conflict_mask).sum())
            if strategy == "conflict_aware":
                # where they agree: average
                # where they conflict: keep the larger-magnitude delta
                agree_avg = (delta_a + delta_b) / 2
                mag_a = np.abs(delta_a)
                mag_b = np.abs(delta_b)
                larger = np.where(mag_a > mag_b, delta_a, delta_b)
                delta = np.where(conflict_mask, larger, agree_avg)
                merged[k] = w_base[k] + delta
            elif strategy == "average":
                merged[k] = w_base[k] + (delta_a + delta_b) / 2
            elif strategy == "max_delta":
                mag_a = np.abs(delta_a)
                mag_b = np.abs(delta_b)
                delta = np.where(mag_a > mag_b, delta_a, delta_b)
                merged[k] = w_base[k] + delta
            else:
                merged[k] = w_base[k] + (delta_a + delta_b) / 2
        self._set_weights(self.base, merged)
        return {
            "strategy": strategy,
            "n_layers_merged": len(merged),
            "n_conflicts": conflicts,
            "n_agreements": agreements,
            "conflict_rate": conflicts / max(conflicts + agreements, 1),
        }


# ============================================================================
# Idea 59 — Skill Evolution 2 (cross-species skill mating)
# ============================================================================
class CrossSpeciesSkill:
    """A skill with arbitrary genome shape (could be vector, matrix, or nested)."""
    def __init__(self, name: str, genome: Any,
                 species: str = "vector",
                 generation: int = 0):
        self.name = name
        self.genome = genome
        self.species = species
        self.generation = generation
        self.fitness: float = -math.inf
        self.id = hashlib.sha256(f"{name}{time.time()}{random.random()}".encode()).hexdigest()[:8]


class SkillEvolution2:
    """Cross-species skill evolution: skills of different genome shapes can interbreed.

    A 'vector' skill can mate with a 'matrix' skill by tiling one to match
    the other's shape, then crossover. Offspring choose a random species.
    """
    def __init__(self, fitness_fn: Callable[[Any], float],
                 pop_size: int = 8, mutation_rate: float = 0.2):
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.population: List[CrossSpeciesSkill] = []
        self.history: List[Dict[str, Any]] = []

    def _init_population(self) -> None:
        species_shapes = [
            ("vector", np.random.randn(16).astype(np.float32)),
            ("matrix", np.random.randn(4, 4).astype(np.float32)),
            ("nested", [np.random.randn(4).astype(np.float32) for _ in range(4)]),
        ]
        for i in range(self.pop_size):
            sp, g = random.choice(species_shapes)
            self.population.append(CrossSpeciesSkill(f"skill_{i}", g, sp))

    def _flatten(self, genome: Any) -> np.ndarray:
        if isinstance(genome, list):
            return np.concatenate([np.asarray(g).flatten() for g in genome])
        return np.asarray(genome).flatten()

    def _unflatten(self, flat: np.ndarray, species: str) -> Any:
        if species == "vector":
            return flat[:16].astype(np.float32)
        elif species == "matrix":
            return flat[:16].reshape(4, 4).astype(np.float32)
        else:  # nested
            return [flat[i*4:(i+1)*4].astype(np.float32) for i in range(4)]

    def _crossover(self, a: CrossSpeciesSkill, b: CrossSpeciesSkill) -> CrossSpeciesSkill:
        flat_a = self._flatten(a.genome)
        flat_b = self._flatten(b.genome)
        # tile shorter to match longer
        max_len = max(len(flat_a), len(flat_b))
        if len(flat_a) < max_len:
            flat_a = np.tile(flat_a, max_len // len(flat_a) + 1)[:max_len]
        if len(flat_b) < max_len:
            flat_b = np.tile(flat_b, max_len // len(flat_b) + 1)[:max_len]
        # uniform crossover
        mask = np.random.rand(max_len) < 0.5
        child_flat = np.where(mask, flat_a, flat_b).astype(np.float32)
        # child picks a random species
        child_species = random.choice([a.species, b.species])
        child_genome = self._unflatten(child_flat, child_species)
        return CrossSpeciesSkill(
            f"skill_g{a.generation+1}_{random.randint(0, 999)}",
            child_genome, child_species, a.generation + 1
        )

    def _mutate(self, skill: CrossSpeciesSkill) -> CrossSpeciesSkill:
        if skill.species == "vector":
            mutated = skill.genome + np.random.normal(0, 0.1, skill.genome.shape).astype(np.float32)
        elif skill.species == "matrix":
            mutated = skill.genome + np.random.normal(0, 0.1, skill.genome.shape).astype(np.float32)
        else:  # nested
            mutated = [g + np.random.normal(0, 0.1, g.shape).astype(np.float32) for g in skill.genome]
        return CrossSpeciesSkill(skill.name + "_mut", mutated, skill.species, skill.generation)

    def step(self, generation: int) -> Dict[str, Any]:
        # evaluate
        for s in self.population:
            s.fitness = self.fitness_fn(s.genome)
        self.population.sort(key=lambda s: -s.fitness)
        # elites survive
        elites = self.population[:max(2, self.pop_size // 3)]
        new_pop: List[CrossSpeciesSkill] = [
            CrossSpeciesSkill(e.name, e.genome, e.species, generation)
            for e in elites
        ]
        # cross-species mating
        while len(new_pop) < self.pop_size:
            p1, p2 = random.sample(elites, 2)
            child = self._crossover(p1, p2)
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            new_pop.append(child)
        species_counts = Counter(s.species for s in new_pop)
        stats = {
            "generation": generation,
            "best_fitness": max(s.fitness for s in self.population),
            "mean_fitness": float(np.mean([s.fitness for s in self.population])),
            "species_distribution": dict(species_counts),
        }
        self.history.append(stats)
        self.population = new_pop
        return stats

    def evolve(self, n_generations: int = 5) -> Dict[str, Any]:
        if not self.population:
            self._init_population()
        for g in range(n_generations):
            self.step(g)
        return self.history[-1] if self.history else {}

    def best(self) -> CrossSpeciesSkill:
        return max(self.population, key=lambda s: s.fitness)


# ============================================================================
# Idea 60 — Model Zoo
# ============================================================================
class EncapsulatedModel:
    """A model + its meta-info (specialty, version, dependencies)."""
    def __init__(self, model_id: str, model: Any, specialty: str,
                 version: str = "1.0"):
        self.model_id = model_id
        self.model = model
        self.specialty = specialty
        self.version = version
        self.knowledge_sharing_log: List[Dict[str, Any]] = []


class ModelZoo:
    """An ecosystem of related models that share knowledge.

    When a model learns something new, it can publish to the zoo. Other
    models can subscribe and benefit. Tracks which models contributed
    to which others' knowledge.
    """
    def __init__(self):
        self.models: Dict[str, EncapsulatedModel] = {}
        self.knowledge_graph: Dict[str, Set[str]] = defaultdict(set)
        self.publications: List[Dict[str, Any]] = []

    def add(self, encapsulated: EncapsulatedModel) -> None:
        self.models[encapsulated.model_id] = encapsulated

    def publish(self, model_id: str, knowledge: Any,
                 topic: str = "general") -> Dict[str, Any]:
        """A model publishes knowledge to the zoo."""
        if model_id not in self.models:
            raise KeyError(f"unknown model '{model_id}'")
        pub = {
            "publisher": model_id,
            "topic": topic,
            "knowledge": knowledge,
            "timestamp": time.time(),
        }
        self.publications.append(pub)
        return pub

    def subscribe(self, model_id: str, topic: str = "general") -> List[Dict[str, Any]]:
        """A model subscribes to all publications on a topic."""
        if model_id not in self.models:
            raise KeyError(f"unknown model '{model_id}'")
        relevant = [p for p in self.publications
                    if p["topic"] == topic and p["publisher"] != model_id]
        # log knowledge transfer
        for p in relevant:
            self.knowledge_graph[p["publisher"]].add(model_id)
            self.models[model_id].knowledge_sharing_log.append({
                "from": p["publisher"], "topic": topic, "timestamp": time.time(),
            })
        return relevant

    def influence_graph(self) -> Dict[str, List[str]]:
        """Return the directed influence graph (publisher -> subscribers)."""
        return {k: sorted(v) for k, v in self.knowledge_graph.items()}

    def most_influential(self) -> List[Dict[str, Any]]:
        """Rank models by how many others they've influenced."""
        items = []
        for mid, em in self.models.items():
            influenced = self.knowledge_graph.get(mid, set())
            items.append({
                "model_id": mid, "specialty": em.specialty,
                "influenced_count": len(influenced),
                "influenced": sorted(influenced),
            })
        items.sort(key=lambda x: -x["influenced_count"])
        return items

    def specialties(self) -> Dict[str, List[str]]:
        """Group models by specialty."""
        out: Dict[str, List[str]] = defaultdict(list)
        for mid, em in self.models.items():
            out[em.specialty].append(mid)
        return dict(out)


# ============================================================================
# Idea 61 — Adversarial Evolutionary Arms Race
# ============================================================================
class AdversarialArmsRace:
    """Generator + attacker co-evolve.

    The generator tries to produce robust outputs; the attacker tries to
    find adversarial perturbations that break them. Both improve over
    generations, creating a Red Queen dynamic.
    """
    def __init__(self, generator_factory: Callable[[], Any],
                 attacker_factory: Callable[[], Any],
                 generator_fitness: Callable[[Any, Any], float],
                 attacker_fitness: Callable[[Any, Any], float],
                 pop_size: int = 4):
        self.gen_factory = generator_factory
        self.atk_factory = attacker_factory
        self.gen_fit = generator_fitness
        self.atk_fit = attacker_fitness
        self.pop_size = pop_size
        self.generators: List[Any] = []
        self.attackers: List[Any] = []
        self.history: List[Dict[str, Any]] = []
        self._init_pops()

    def _init_pops(self) -> None:
        self.generators = [self.gen_factory() for _ in range(self.pop_size)]
        self.attackers = [self.atk_factory() for _ in range(self.pop_size)]

    def round(self, n_steps: int = 5) -> Dict[str, Any]:
        """One round of co-evolution."""
        # generators vs best attacker
        best_attacker = max(self.attackers, key=lambda a: max(
            self.atk_fit(a, g) for g in self.generators
        ))
        gen_scores = [self.gen_fit(g, best_attacker) for g in self.generators]
        # attackers vs best generator
        best_generator = self.generators[int(np.argmax(gen_scores))]
        atk_scores = [self.atk_fit(a, best_generator) for a in self.attackers]
        # selection: keep top half, replace bottom half with mutated copies of top
        gen_order = np.argsort(gen_scores)[::-1]
        atk_order = np.argsort(atk_scores)[::-1]
        new_gens = [self.generators[i] for i in gen_order[:self.pop_size // 2]]
        new_atks = [self.attackers[i] for i in atk_order[:self.pop_size // 2]]
        while len(new_gens) < self.pop_size:
            new_gens.append(self.gen_factory())  # "mutation" = fresh init
        while len(new_atks) < self.pop_size:
            new_atks.append(self.atk_factory())
        self.generators = new_gens
        self.attackers = new_atks
        stats = {
            "best_generator_score": float(max(gen_scores)),
            "best_attacker_score": float(max(atk_scores)),
            "mean_generator_score": float(np.mean(gen_scores)),
            "mean_attacker_score": float(np.mean(atk_scores)),
        }
        self.history.append(stats)
        return stats

    def evolve(self, n_rounds: int = 5) -> Dict[str, Any]:
        for _ in range(n_rounds):
            self.round()
        return self.history[-1] if self.history else {}

    def best_generator(self) -> Any:
        return max(self.generators,
                    key=lambda g: max(self.gen_fit(g, a) for a in self.attackers))

    def best_attacker(self) -> Any:
        return max(self.attackers,
                    key=lambda a: max(self.atk_fit(a, g) for g in self.generators))


# ============================================================================
# Idea 62 — Model Genealogy
# ============================================================================
class GenealogyNode:
    def __init__(self, model_id: str, parent_ids: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.model_id = model_id
        self.parent_ids = parent_ids or []
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.children: List[str] = []


class ModelGenealogy:
    """Family tree + heredity tracking for models.

    Tracks which models were derived from which. Can compute:
      - ancestors(model_id)  : all upstream ancestors
      - descendants(model_id): all downstream descendants
      - common_ancestors(a, b): lowest common ancestors
      - heritability: which traits (hyperparams) propagate
    """
    def __init__(self):
        self.nodes: Dict[str, GenealogyNode] = {}

    def register(self, model_id: str,
                  parent_ids: Optional[List[str]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        node = GenealogyNode(model_id, parent_ids, metadata)
        self.nodes[model_id] = node
        for pid in parent_ids or []:
            if pid in self.nodes:
                self.nodes[pid].children.append(model_id)

    def ancestors(self, model_id: str) -> Set[str]:
        """All upstream ancestors of model_id."""
        if model_id not in self.nodes:
            return set()
        visited: Set[str] = set()
        frontier = list(self.nodes[model_id].parent_ids)
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in self.nodes:
                frontier.extend(self.nodes[current].parent_ids)
        return visited

    def descendants(self, model_id: str) -> Set[str]:
        """All downstream descendants of model_id."""
        if model_id not in self.nodes:
            return set()
        visited: Set[str] = set()
        frontier = list(self.nodes[model_id].children)
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in self.nodes:
                frontier.extend(self.nodes[current].children)
        return visited

    def common_ancestors(self, a: str, b: str) -> Set[str]:
        """Common ancestors of two models."""
        return self.ancestors(a) & self.ancestors(b)

    def heritability(self, trait: str) -> Dict[str, Any]:
        """Compute heritability of a trait across the family tree.

        Returns the fraction of models that inherited the trait from any parent.
        """
        inherited = 0
        total = 0
        for node in self.nodes.values():
            if not node.parent_ids:
                continue
            total += 1
            parent_traits = []
            for pid in node.parent_ids:
                if pid in self.nodes and trait in self.nodes[pid].metadata:
                    parent_traits.append(self.nodes[pid].metadata[trait])
            if parent_traits:
                # check if child has any of the parent traits
                child_trait = node.metadata.get(trait)
                if child_trait in parent_traits:
                    inherited += 1
        return {
            "trait": trait,
            "n_models_with_parents": total,
            "n_inherited": inherited,
            "heritability": inherited / max(total, 1),
        }

    def family_tree(self, model_id: str, depth: int = 3) -> Dict[str, Any]:
        """Render a family tree starting from model_id, up to depth."""
        if model_id not in self.nodes:
            return {}
        node = self.nodes[model_id]
        result: Dict[str, Any] = {
            "model_id": model_id,
            "metadata": node.metadata,
            "parents": [],
            "children": [],
        }
        if depth > 0:
            for pid in node.parent_ids:
                result["parents"].append(self.family_tree(pid, depth - 1))
            for cid in node.children:
                result["children"].append(self.family_tree(cid, depth - 1))
        return result


# ============================================================================
# Idea 63 — Data Valuation Market (Shapley-based)
# ============================================================================
class DataValuationMarket:
    """Shapley-value-based data pricing.

    Each data source contributes to model performance; the Shapley value
    fairly attributes performance gains across sources.

    Exact Shapley is O(2^n) so we use Monte Carlo approximation.
    """
    def __init__(self, train_fn: Callable[[List[str]], float],
                 source_ids: List[str], n_samples: int = 100):
        """train_fn takes a list of source IDs, returns validation accuracy."""
        self.train_fn = train_fn
        self.source_ids = list(source_ids)
        self.n_samples = n_samples
        self.shapley_values: Dict[str, float] = {}

    def compute(self) -> Dict[str, float]:
        """Monte Carlo Shapley value approximation."""
        n = len(self.source_ids)
        if n == 0:
            return {}
        shapley = {s: 0.0 for s in self.source_ids}
        for _ in range(self.n_samples):
            # random permutation
            perm = list(self.source_ids)
            random.shuffle(perm)
            prev_value = 0.0  # empty coalition has 0 value
            current_coalition: List[str] = []
            for s in perm:
                current_coalition.append(s)
                value = self._eval_coalition(current_coalition)
                marginal = value - prev_value
                shapley[s] += marginal
                prev_value = value
        # average
        for s in shapley:
            shapley[s] /= self.n_samples
        self.shapley_values = shapley
        return shapley

    def _eval_coalition(self, coalition: List[str]) -> float:
        try:
            return float(self.train_fn(list(coalition)))
        except Exception:
            return 0.0

    def fair_payout(self, total_budget: float = 100.0) -> Dict[str, float]:
        """Distribute a total budget proportional to Shapley values."""
        if not self.shapley_values:
            self.compute()
        total = sum(max(0, v) for v in self.shapley_values.values()) or 1.0
        return {s: total_budget * max(0, v) / total
                 for s, v in self.shapley_values.items()}

    def summary(self) -> Dict[str, Any]:
        if not self.shapley_values:
            self.compute()
        sorted_sources = sorted(self.shapley_values.items(), key=lambda kv: -kv[1])
        return {
            "n_sources": len(self.source_ids),
            "n_samples": self.n_samples,
            "ranked": [{"source": s, "value": v} for s, v in sorted_sources],
            "total_value": sum(self.shapley_values.values()),
        }


# ============================================================================
# Idea 64 — Model Embryology
# ============================================================================
class ModelEmbryology:
    """Track the developmental stages of a model.

    Stages:
      1. Zygote    — initial weights just initialised
      2. Blastula  — first forward pass; activations are random
      3. Gastrula  — first gradient step; weights start to specialise
      4. Organogenesis — layers differentiate (some prune, some grow)
      5. Fetus     — converged on local optimum
      6. Birth     — ready for deployment
    """
    STAGES = ["zygote", "blastula", "gastrula", "organogenesis", "fetus", "birth"]

    def __init__(self, model: Any):
        self.model = model
        self.stage = "zygote"
        self.events: List[Dict[str, Any]] = []
        self.weight_snapshots: List[Dict[str, Any]] = []

    def _snapshot(self) -> Dict[str, Any]:
        if hasattr(self.model, "state_dict"):
            try:
                sd = self.model.state_dict()
                total_norm = 0.0
                n_layers = 0
                for k, v in sd.items():
                    arr = np.asarray(v, dtype=np.float32)
                    total_norm += float(np.linalg.norm(arr))
                    n_layers += 1
                return {"total_norm": total_norm, "n_layers": n_layers}
            except Exception:
                pass
        return {}

    def advance(self, event: str = "train_step") -> Dict[str, Any]:
        """Advance to the next stage if event triggers a transition."""
        transitions = {
            "zygote": ("forward_pass", "blastula"),
            "blastula": ("first_gradient", "gastrula"),
            "gastrula": ("differentiation", "organogenesis"),
            "organogenesis": ("convergence", "fetus"),
            "fetus": ("validation_passed", "birth"),
        }
        snap = self._snapshot()
        self.events.append({
            "from_stage": self.stage, "event": event,
            "timestamp": time.time(), **snap,
        })
        if self.stage in transitions:
            trigger, next_stage = transitions[self.stage]
            if event == trigger:
                self.stage = next_stage
        return {"stage": self.stage, "event": event, **snap}

    def is_deployment_ready(self) -> bool:
        return self.stage == "birth"

    def developmental_timeline(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def stage_description(self) -> str:
        descriptions = {
            "zygote": "Initialised but untrained. Weights are random.",
            "blastula": "First forward pass complete. Activations show no structure.",
            "gastrula": "First gradient step. Weights beginning to specialise.",
            "organogenesis": "Layers differentiating. Some pruning, some growth.",
            "fetus": "Converged on a local optimum. Stable but not yet validated.",
            "birth": "Validation passed. Ready for deployment.",
        }
        return descriptions.get(self.stage, "Unknown stage.")


# ============================================================================
# Idea 65 — Neural Sociology
# ============================================================================
class NeuralSociology:
    """Model social networks + influence graphs.

    Models form social ties (e.g. via knowledge sharing). Influence
    spreads through the network. We can measure:
      - Centrality (which models are most influential)
      - Clusters (groups of similar models)
      - Echo chambers (models that only hear from each other)
    """
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # follower -> followed
        self.messages: List[Dict[str, Any]] = []

    def add_model(self, model_id: str) -> None:
        self.nodes.add(model_id)

    def follow(self, follower: str, followed: str) -> None:
        self.nodes.add(follower)
        self.nodes.add(followed)
        self.edges[follower].add(followed)

    def post(self, author: str, content: Any) -> Dict[str, Any]:
        """A model posts content; reaches all its followers."""
        msg = {
            "author": author, "content": content,
            "timestamp": time.time(),
        }
        self.messages.append(msg)
        return msg

    def feed(self, model_id: str) -> List[Dict[str, Any]]:
        """Get a model's feed: posts from everyone it follows."""
        followed = self.edges.get(model_id, set())
        return [m for m in self.messages if m["author"] in followed]

    def centrality(self) -> Dict[str, int]:
        """In-degree centrality: how many followers each model has."""
        centrality = {n: 0 for n in self.nodes}
        for follower, followed_set in self.edges.items():
            for followed in followed_set:
                centrality[followed] = centrality.get(followed, 0) + 1
        return centrality

    def clusters(self) -> List[Set[str]]:
        """Find clusters via connected components (undirected)."""
        # build undirected adjacency
        undirected: Dict[str, Set[str]] = defaultdict(set)
        for follower, followed_set in self.edges.items():
            for followed in followed_set:
                undirected[follower].add(followed)
                undirected[followed].add(follower)
        visited: Set[str] = set()
        clusters: List[Set[str]] = []
        for start in self.nodes:
            if start in visited:
                continue
            # BFS
            cluster: Set[str] = set()
            frontier = [start]
            while frontier:
                current = frontier.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster.add(current)
                frontier.extend(undirected[current] - visited)
            clusters.append(cluster)
        return clusters

    def echo_chambers(self, threshold: float = 0.7) -> List[Set[str]]:
        """Find groups where >threshold of edges stay within the group."""
        clusters = self.clusters()
        chambers: List[Set[str]] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            internal = 0
            external = 0
            for member in cluster:
                followed = self.edges.get(member, set())
                for f in followed:
                    if f in cluster:
                        internal += 1
                    else:
                        external += 1
            total = internal + external
            if total > 0 and internal / total >= threshold:
                chambers.append(cluster)
        return chambers

    def influence_score(self, model_id: str) -> Dict[str, Any]:
        """Compute a model's influence score (followers + 2nd-order)."""
        followers = {n for n in self.nodes if model_id in self.edges.get(n, set())}
        # 2nd-order: followers of followers
        second_order: Set[str] = set()
        for f in followers:
            second_order.update({n for n in self.nodes if f in self.edges.get(n, set())})
        second_order -= followers
        second_order.discard(model_id)
        return {
            "model_id": model_id,
            "direct_followers": len(followers),
            "second_order_reach": len(second_order),
            "influence_score": len(followers) + 0.5 * len(second_order),
        }


__all__ = [
    "DataTrust", "ModelPharmacy", "ModelUnion2",
    "CrossSpeciesSkill", "SkillEvolution2",
    "EncapsulatedModel", "ModelZoo",
    "AdversarialArmsRace",
    "GenealogyNode", "ModelGenealogy",
    "DataValuationMarket",
    "ModelEmbryology",
    "NeuralSociology",
]
