"""
qai.creative.experimental4 -- 10 more original creative ideas (my own, part 4).

  Idea 46 — ModelInsurance          : predict model failure risk, issue insurance
                                       policy with premiums + payout schedule
  Idea 47 — SkillEvolution          : skills evolve over generations like organisms
  Idea 48 — NeuroEvolution          : NEAT-style weight + topology evolution
  Idea 49 — FederatedAuction        : clients bid for training tasks based on local
                                       data value; market clears
  Idea 50 — ModelConstitution       : written constitution the model must follow;
                                       judicial review of decisions
  Idea 51 — QuantumInspiredOptimizer: annealing with quantum-tunnelling-inspired
                                       jumps out of local minima
  Idea 52 — SyntheticCurriculum     : auto-generate training curriculum from a
                                       task description
  Idea 53 — ModelBiography          : auto-write the life story of a model
                                       (training history as narrative)
  Idea 54 — PredictionMarket        : models make predictions, "trade" on outcomes,
                                       prices reflect confidence
  Idea 55 — NeuralDebateChamber     : structured debate with rebuttals (richer
                                       than Parliament's propose-vote)
"""
from __future__ import annotations
import math
import random
import time
import json
import hashlib
from collections import defaultdict
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
# Idea 46 — Model Insurance
# ============================================================================
class ModelInsurance:
    """Predict model failure risk and issue an insurance policy.

    Premiums are computed from:
      - historical failure rate (claim_rate)
      - input-distribution drift since last evaluation
      - model age (older = higher premium)
      - domain risk multiplier (medical/legal = high, recommender = low)

    If the model fails during the policy period, the policyholder receives
    a payout proportional to the loss.
    """
    DOMAIN_RISK = {
        "medical": 2.5, "legal": 2.0, "finance": 1.8, "autonomous": 3.0,
        "education": 1.0, "marketing": 0.8, "recommender": 0.6,
        "general": 1.0,
    }

    def __init__(self, model_id: str, domain: str = "general",
                 base_premium: float = 100.0, coverage_limit: float = 10000.0):
        self.model_id = model_id
        self.domain = domain
        self.base_premium = base_premium
        self.coverage_limit = coverage_limit
        self.claims_history: List[Dict[str, Any]] = []
        self.policy: Optional[Dict[str, Any]] = None

    def assess_risk(self, drift_score: float = 0.0,
                     model_age_days: int = 0,
                     recent_failure_rate: float = 0.0) -> Dict[str, Any]:
        """Compute a risk score in [0, 1] and recommended premium."""
        domain_mult = self.DOMAIN_RISK.get(self.domain, 1.0)
        # weighted risk factors
        risk = (
            0.3 * min(drift_score, 1.0) +
            0.3 * min(recent_failure_rate, 1.0) +
            0.2 * min(model_age_days / 365, 1.0) +
            0.2 * (domain_mult - 0.5) / 2.5
        )
        risk = max(0.0, min(1.0, risk))
        # premium: base * risk multiplier * domain multiplier
        premium = self.base_premium * (1 + risk * 5) * domain_mult
        return {
            "risk_score": float(risk),
            "premium": float(premium),
            "domain_multiplier": domain_mult,
            "factors": {
                "drift": drift_score, "age_days": model_age_days,
                "failure_rate": recent_failure_rate, "domain": self.domain,
            },
        }

    def issue_policy(self, drift_score: float = 0.0,
                      model_age_days: int = 0,
                      recent_failure_rate: float = 0.0,
                      deductible: float = 50.0) -> Dict[str, Any]:
        """Issue a policy with the computed premium + payout terms."""
        risk = self.assess_risk(drift_score, model_age_days, recent_failure_rate)
        self.policy = {
            "policy_id": hashlib.sha256(
                f"{self.model_id}{time.time()}".encode()
            ).hexdigest()[:16],
            "model_id": self.model_id,
            "issued_at": time.time(),
            "premium": risk["premium"],
            "deductible": deductible,
            "coverage_limit": self.coverage_limit,
            "risk_score": risk["risk_score"],
            "risk_factors": risk["factors"],
            "terms": {
                "payout_on_failure": min(self.coverage_limit, 100 * risk["risk_score"]),
                "exclusions": ["intentional misuse", "war", "natural disaster"],
                "renewal_period_days": 30,
            },
        }
        return self.policy

    def file_claim(self, failure_description: str,
                    loss_amount: float) -> Dict[str, Any]:
        """File a claim against the policy."""
        if not self.policy:
            raise RuntimeError("No policy issued; call issue_policy() first")
        if loss_amount <= self.policy["deductible"]:
            return {"approved": False, "reason": "below deductible",
                     "deductible": self.policy["deductible"]}
        payout = min(self.policy["terms"]["payout_on_failure"],
                      loss_amount - self.policy["deductible"])
        claim = {
            "claim_id": hashlib.sha256(
                f"{failure_description}{time.time()}".encode()
            ).hexdigest()[:12],
            "description": failure_description,
            "loss_amount": loss_amount,
            "payout": payout,
            "approved": True,
            "timestamp": time.time(),
        }
        self.claims_history.append(claim)
        return claim

    def portfolio_summary(self) -> Dict[str, Any]:
        """Summary of all claims paid out under this policy."""
        total_paid = sum(c["payout"] for c in self.claims_history if c["approved"])
        return {
            "policy_id": self.policy["policy_id"] if self.policy else None,
            "premium_collected": self.policy["premium"] if self.policy else 0,
            "total_payouts": total_paid,
            "n_claims": len(self.claims_history),
            "net": (self.policy["premium"] if self.policy else 0) - total_paid,
        }


# ============================================================================
# Idea 47 — Skill Evolution
# ============================================================================
class EvolvableSkill:
    """A skill with a genome (params) + fitness + generation + ancestry."""
    def __init__(self, name: str, genome: np.ndarray, generation: int = 0,
                 parents: Optional[List[str]] = None):
        self.name = name
        self.genome = genome.copy()
        self.generation = generation
        self.parents = parents or []
        self.fitness: float = -math.inf
        self.id = hashlib.sha256(
            f"{name}{generation}{random.random()}".encode()
        ).hexdigest()[:8]


class SkillEvolution:
    """Skills evolve over generations via selection + mutation + crossover.

    Each skill is a numeric genome evaluated by a fitness function. Best
    skills survive, breed, and mutate. Lineage is tracked so you can see
    how skills evolved.
    """
    def __init__(self, fitness_fn: Callable[[np.ndarray], float],
                 genome_size: int = 16, pop_size: int = 8,
                 mutation_rate: float = 0.1, mutation_scale: float = 0.2,
                 elite_frac: float = 0.25):
        self.fitness_fn = fitness_fn
        self.genome_size = genome_size
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.n_elite = max(1, int(elite_frac * pop_size))
        self.population: List[EvolvableSkill] = []
        self.history: List[Dict[str, Any]] = []
        self._init_population()

    def _init_population(self) -> None:
        for i in range(self.pop_size):
            genome = np.random.randn(self.genome_size).astype(np.float32)
            self.population.append(EvolvableSkill(f"skill_{i}", genome, generation=0))

    def _evaluate(self) -> None:
        for skill in self.population:
            skill.fitness = self.fitness_fn(skill.genome)

    def _select_parent(self) -> EvolvableSkill:
        # tournament selection
        contestants = random.sample(self.population, min(3, len(self.population)))
        return max(contestants, key=lambda s: s.fitness)

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        mask = np.random.rand(*genome.shape) < self.mutation_rate
        noise = np.random.normal(0, self.mutation_scale, genome.shape).astype(np.float32)
        return genome + mask * noise

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = np.random.rand(*a.shape) < 0.5
        return np.where(mask, a, b).astype(np.float32)

    def step(self, generation: int) -> Dict[str, Any]:
        self._evaluate()
        self.population.sort(key=lambda s: -s.fitness)
        # elites survive
        elites = list(self.population[:self.n_elite])
        new_pop: List[EvolvableSkill] = [
            EvolvableSkill(e.name, e.genome, generation, [e.id])
            for e in elites
        ]
        # breed to fill rest
        while len(new_pop) < self.pop_size:
            p1 = self._select_parent()
            p2 = self._select_parent()
            child_genome = self._crossover(p1.genome, p2.genome)
            child_genome = self._mutate(child_genome)
            new_pop.append(EvolvableSkill(
                f"skill_g{generation}_{len(new_pop)}",
                child_genome, generation, [p1.id, p2.id]
            ))
        best = max(self.population, key=lambda s: s.fitness)
        stats = {
            "generation": generation,
            "best_fitness": best.fitness,
            "mean_fitness": float(np.mean([s.fitness for s in self.population])),
            "best_skill": best.name,
            "population_size": len(new_pop),
        }
        self.history.append(stats)
        self.population = new_pop
        return stats

    def evolve(self, n_generations: int = 10) -> Dict[str, Any]:
        for g in range(n_generations):
            self.step(g)
        return self.history[-1] if self.history else {}

    def best(self) -> EvolvableSkill:
        return max(self.population, key=lambda s: s.fitness)

    def lineages(self) -> Dict[str, Any]:
        """Build a lineage tree for each current skill."""
        return {
            "current_population": [
                {"name": s.name, "id": s.id, "generation": s.generation,
                  "parents": s.parents, "fitness": s.fitness}
                for s in self.population
            ],
            "history": self.history,
        }


# ============================================================================
# Idea 48 — NeuroEvolution (NEAT-style)
# ============================================================================
class _NeuroNode:
    def __init__(self, node_id: int, layer: int):
        self.id = node_id
        self.layer = layer
        self.value: float = 0.0


class _NeuroEdge:
    def __init__(self, in_node: int, out_node: int, weight: float,
                 innovation: int, enabled: bool = True):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.innovation = innovation
        self.enabled = enabled


class NeuroEvolutionGenome:
    """A NEAT-style genome: nodes + edges with innovation numbers."""
    def __init__(self, n_inputs: int, n_outputs: int):
        self.nodes: Dict[int, _NeuroNode] = {}
        self.edges: List[_NeuroEdge] = []
        self.next_node_id = 0
        self.next_innovation = 0
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # create input + output nodes
        for _ in range(n_inputs):
            self.add_node(layer=0)
        for _ in range(n_outputs):
            self.add_node(layer=2)
        # initial fully-connected edges
        input_ids = [n for n in self.nodes if self.nodes[n].layer == 0]
        output_ids = [n for n in self.nodes if self.nodes[n].layer == 2]
        for i in input_ids:
            for o in output_ids:
                self.add_edge(i, o, random.uniform(-1, 1))

    def add_node(self, layer: int) -> int:
        nid = self.next_node_id
        self.nodes[nid] = _NeuroNode(nid, layer)
        self.next_node_id += 1
        return nid

    def add_edge(self, in_node: int, out_node: int, weight: float) -> None:
        self.edges.append(_NeuroEdge(in_node, out_node, weight, self.next_innovation))
        self.next_innovation += 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the evolved network."""
        # reset all node values
        for n in self.nodes.values():
            n.value = 0.0
        # set inputs
        input_ids = sorted([n for n in self.nodes if self.nodes[n].layer == 0])
        for i, val in enumerate(x[:len(input_ids)]):
            self.nodes[input_ids[i]].value = float(val)
        # topological forward by layer
        for layer in sorted(set(n.layer for n in self.nodes.values())):
            if layer == 0:
                continue
            layer_nodes = [n for n in self.nodes if self.nodes[n].layer == layer]
            for nid in layer_nodes:
                total = 0.0
                for e in self.edges:
                    if e.enabled and e.out_node == nid:
                        total += e.weight * self.nodes[e.in_node].value
                # activation
                self.nodes[nid].value = math.tanh(total)
        # return outputs
        output_ids = sorted([n for n in self.nodes if self.nodes[n].layer == 2])
        return np.array([self.nodes[o].value for o in output_ids], dtype=np.float32)

    def mutate_add_node(self) -> None:
        """Split a random edge by adding a new node in between."""
        if not self.edges:
            return
        e = random.choice(self.edges)
        e.enabled = False
        new_id = self.add_node(layer=1)
        self.add_edge(e.in_node, new_id, 1.0)
        self.add_edge(new_id, e.out_node, e.weight)

    def mutate_add_edge(self) -> None:
        """Add a new random connection."""
        all_nodes = list(self.nodes.keys())
        if len(all_nodes) < 2:
            return
        in_node = random.choice(all_nodes)
        out_node = random.choice(all_nodes)
        if in_node == out_node:
            return
        if self.nodes[in_node].layer >= self.nodes[out_node].layer:
            return
        self.add_edge(in_node, out_node, random.uniform(-1, 1))

    def mutate_weights(self, rate: float = 0.2) -> None:
        for e in self.edges:
            if random.random() < rate:
                e.weight += random.gauss(0, 0.5)

    def mutate_toggle(self, rate: float = 0.1) -> None:
        for e in self.edges:
            if random.random() < rate:
                e.enabled = not e.enabled


class NeuroEvolution:
    """NEAT-style neuroevolution: evolve both weights AND topology.

    Mutations:
      - add_node: split an edge with a new hidden node
      - add_edge: add a new connection
      - mutate_weights: perturb existing weights
      - toggle_edge: enable/disable a random edge
    """
    def __init__(self, n_inputs: int, n_outputs: int,
                 pop_size: int = 6,
                 fitness_fn: Optional[Callable[[NeuroEvolutionGenome], float]] = None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.pop_size = pop_size
        self.fitness_fn = fitness_fn or (lambda g: 0.0)
        self.population: List[NeuroEvolutionGenome] = [
            NeuroEvolutionGenome(n_inputs, n_outputs) for _ in range(pop_size)
        ]
        self.history: List[Dict[str, Any]] = []

    def step(self, generation: int) -> Dict[str, Any]:
        # evaluate
        scored = [(self.fitness_fn(g), g) for g in self.population]
        scored.sort(key=lambda x: -x[0])
        # elites
        n_elite = max(1, self.pop_size // 3)
        new_pop = [NeuroEvolutionGenome(self.n_inputs, self.n_outputs)
                    for _ in range(self.pop_size)]
        # copy elite genomes' edges
        for i, (_, elite_g) in enumerate(scored[:n_elite]):
            new_pop[i].edges = [_NeuroEdge(e.in_node, e.out_node, e.weight,
                                            e.innovation, e.enabled)
                                 for e in elite_g.edges]
            new_pop[i].nodes = {nid: _NeuroNode(n.id, n.layer)
                                  for nid, n in elite_g.nodes.items()}
        # mutate the rest
        for i in range(n_elite, self.pop_size):
            parent = random.choice(scored[:n_elite])[1]
            new_pop[i].edges = [_NeuroEdge(e.in_node, e.out_node, e.weight,
                                            e.innovation, e.enabled)
                                 for e in parent.edges]
            new_pop[i].nodes = {nid: _NeuroNode(n.id, n.layer)
                                  for nid, n in parent.nodes.items()}
            # mutations
            new_pop[i].mutate_weights(0.3)
            if random.random() < 0.3:
                new_pop[i].mutate_add_node()
            if random.random() < 0.5:
                new_pop[i].mutate_add_edge()
            if random.random() < 0.1:
                new_pop[i].mutate_toggle(0.2)
        best = scored[0]
        self.history.append({
            "generation": generation,
            "best_fitness": best[0],
            "mean_fitness": float(np.mean([s[0] for s in scored])),
            "best_n_nodes": len(best[1].nodes),
            "best_n_edges": len(best[1].edges),
        })
        self.population = new_pop
        return self.history[-1]

    def evolve(self, n_generations: int = 5) -> Dict[str, Any]:
        for g in range(n_generations):
            self.step(g)
        return self.history[-1] if self.history else {}

    def best(self) -> NeuroEvolutionGenome:
        scored = [(self.fitness_fn(g), g) for g in self.population]
        return max(scored, key=lambda x: x[0])[1]


# ============================================================================
# Idea 49 — Federated Auction
# ============================================================================
class FederatedAuction:
    """Clients bid for training tasks; market clears.

    Each client has local data + a private valuation of how much it would
    cost to train on a given task. The server posts a task; clients bid;
    the lowest bidders win and get paid.
    """
    def __init__(self, server_budget: float = 100.0):
        self.server_budget = server_budget
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.auctions: List[Dict[str, Any]] = []

    def register_client(self, client_id: str, n_samples: int,
                          compute_capacity: float = 1.0) -> None:
        self.clients[client_id] = {
            "n_samples": n_samples,
            "capacity": compute_capacity,
            "wins": 0,
            "earnings": 0.0,
        }

    def run_auction(self, task_id: str, n_winners: int = 3,
                     min_samples: int = 100) -> Dict[str, Any]:
        """Run a sealed-bid auction; lowest bids win."""
        # each client privately computes a bid based on its cost
        bids: List[Tuple[str, float]] = []
        for cid, info in self.clients.items():
            if info["n_samples"] < min_samples:
                continue
            # cost = base + per-sample cost - capacity discount
            base = 5.0
            per_sample = 0.01 * info["n_samples"]
            capacity_disc = info["capacity"] * 2.0
            bid = base + per_sample - capacity_disc + random.gauss(0, 1)
            bids.append((cid, max(0.5, bid)))
        # sort ascending; lowest bids win
        bids.sort(key=lambda x: x[1])
        winners = bids[:n_winners]
        # pay = second-price (Vickrey) rule across the winners
        if len(winners) >= 2:
            payment = winners[-1][1]  # highest winning bid
        elif winners:
            payment = winners[0][1]
        else:
            payment = 0.0
        for cid, _ in winners:
            self.clients[cid]["wins"] += 1
            self.clients[cid]["earnings"] += payment
        result = {
            "task_id": task_id,
            "winners": [w[0] for w in winners],
            "payment_per_winner": payment,
            "total_paid": payment * len(winners),
            "n_participants": len(bids),
            "timestamp": time.time(),
        }
        self.auctions.append(result)
        return result

    def leaderboard(self) -> List[Dict[str, Any]]:
        return sorted([
            {"client_id": cid, **info}
            for cid, info in self.clients.items()
        ], key=lambda c: -c["earnings"])


# ============================================================================
# Idea 50 — Model Constitution
# ============================================================================
class Constitution:
    """A written constitution the model must follow.

    Articles: each article is (principle, enforceable_rule).
    Amendments: changes to the constitution over time.
    Judicial review: a judge (LLM or rule-based) evaluates decisions.
    """
    def __init__(self):
        self.articles: List[Dict[str, Any]] = []
        self.amendments: List[Dict[str, Any]] = []

    def add_article(self, number: int, principle: str,
                     enforceable_rule: Callable[[str], bool],
                     description: str = "") -> None:
        self.articles.append({
            "number": number,
            "principle": principle,
            "rule": enforceable_rule,
            "description": description,
            "ratified_at": time.time(),
        })

    def amend(self, article_number: int, new_principle: str,
               new_rule: Callable[[str], bool]) -> None:
        self.amendments.append({
            "article": article_number,
            "new_principle": new_principle,
            "timestamp": time.time(),
        })
        for a in self.articles:
            if a["number"] == article_number:
                a["principle"] = new_principle
                a["rule"] = new_rule

    def evaluate(self, decision: str) -> Dict[str, Any]:
        """Check a decision against all articles."""
        verdicts = []
        for a in self.articles:
            try:
                ok = bool(a["rule"](decision))
            except Exception:
                ok = False
            verdicts.append({
                "article": a["number"],
                "principle": a["principle"],
                "compliant": ok,
            })
        all_compliant = all(v["compliant"] for v in verdicts)
        return {
            "decision": decision,
            "compliant": all_compliant,
            "verdicts": verdicts,
            "n_amendments": len(self.amendments),
        }

    def text(self) -> str:
        """Render the constitution as readable text."""
        lines = ["=== MODEL CONSTITUTION ==="]
        for a in self.articles:
            lines.append(f"Article {a['number']}: {a['principle']}")
            if a["description"]:
                lines.append(f"  ({a['description']})")
        if self.amendments:
            lines.append("\n--- Amendments ---")
            for am in self.amendments:
                lines.append(f"  Amendment to Article {am['article']}: {am['new_principle']}")
        return "\n".join(lines)


class ModelConstitution:
    """Constitutional governance for a model.

    Wraps a model + a Constitution. Every decision is evaluated; if it
    violates an article, the decision is vetoed.
    """
    def __init__(self, model: Any, constitution: Constitution):
        self.model = model
        self.constitution = constitution
        self.veto_log: List[Dict[str, Any]] = []

    def decide(self, input_data: Any) -> Dict[str, Any]:
        """Make a decision; check against constitution."""
        try:
            decision = str(self.model.forward(input_data))
        except Exception as e:
            decision = f"[error] {e}"
        eval_result = self.constitution.evaluate(decision)
        if not eval_result["compliant"]:
            violated = [v["article"] for v in eval_result["verdicts"] if not v["compliant"]]
            self.veto_log.append({
                "decision": decision,
                "violated_articles": violated,
                "timestamp": time.time(),
            })
            return {
                "decision": "[VETOED by constitution]",
                "original": decision,
                "violations": violated,
                "compliant": False,
            }
        return {"decision": decision, "compliant": True, "violations": []}

    def veto_history(self) -> List[Dict[str, Any]]:
        return list(self.veto_log)


# ============================================================================
# Idea 51 — Quantum-Inspired Optimizer
# ============================================================================
class QuantumInspiredOptimizer:
    """Quantum-annealing-inspired optimisation with tunnelling.

    Simulated annealing with periodic "tunnelling" jumps: when the
    search has been stuck in a local min for many steps, take a large
    random jump to escape.

    Quantum-inspired because real quantum annealers exploit tunnelling
    through barriers; we mimic this classically via large random
    perturbations when stuck.
    """
    def __init__(self, objective: Callable[[np.ndarray], float],
                 dim: int, bounds: List[Tuple[float, float]],
                 initial_temp: float = 1.0, cooling: float = 0.99,
                 tunnel_threshold: int = 20, tunnel_scale: float = 2.0):
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.temp = initial_temp
        self.cooling = cooling
        self.tunnel_threshold = tunnel_threshold
        self.tunnel_scale = tunnel_scale
        self.history: List[Dict[str, Any]] = []

    def optimise(self, n_steps: int = 200,
                  initial: Optional[np.ndarray] = None) -> Dict[str, Any]:
        # init
        if initial is not None:
            x = np.array(initial, dtype=np.float32)
        else:
            x = np.array([random.uniform(lo, hi) for lo, hi in self.bounds], dtype=np.float32)
        fx = self.objective(x)
        best_x = x.copy()
        best_fx = fx
        stuck_counter = 0
        for step in range(n_steps):
            # check if stuck
            if stuck_counter >= self.tunnel_threshold:
                # quantum-tunnelling jump: large random perturbation
                for i in range(self.dim):
                    lo, hi = self.bounds[i]
                    span = hi - lo
                    x[i] = x[i] + random.gauss(0, span * self.tunnel_scale)
                    x[i] = max(lo, min(hi, x[i]))
                fx = self.objective(x)
                stuck_counter = 0
                self.history.append({"step": step, "event": "tunnel", "fx": fx})
            else:
                # normal SA step: small perturbation
                x_new = x.copy()
                i = random.randint(0, self.dim - 1)
                lo, hi = self.bounds[i]
                delta = random.gauss(0, (hi - lo) * 0.05)
                x_new[i] = max(lo, min(hi, x_new[i] + delta))
                fx_new = self.objective(x_new)
                # Metropolis acceptance
                delta_f = fx_new - fx
                if delta_f < 0 or random.random() < math.exp(-delta_f / max(self.temp, 1e-8)):
                    x = x_new
                    fx = fx_new
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx
            self.temp *= self.cooling
        return {"best_x": best_x.tolist(), "best_fx": float(best_fx),
                 "n_steps": n_steps, "history_len": len(self.history)}


# ============================================================================
# Idea 52 — Synthetic Curriculum
# ============================================================================
class SyntheticCurriculum:
    """Auto-generate a training curriculum from a task description.

    Given a high-level task like "classify digits", generates a sequence
    of progressively harder sub-tasks:
      1. Binary classification (2 classes)
      2. Few-class (5 classes)
      3. Full task (10 classes)
      4. With noise
      5. With class imbalance
    """
    DIFFICULTY_LEVELS = [
        ("binary", "Start with binary classification"),
        ("few_class", "Expand to a few classes"),
        ("full", "Full multi-class"),
        ("with_noise", "Add label noise"),
        ("with_imbalance", "Add class imbalance"),
        ("with_adversarial", "Add adversarial perturbations"),
    ]

    def __init__(self, task_description: str, n_classes: int = 10,
                 n_samples_per_class: int = 100, dim: int = 4):
        self.task = task_description
        self.n_classes = n_classes
        self.n_samples_per_class = n_samples_per_class
        self.dim = dim
        self.curriculum: List[Dict[str, Any]] = []

    def generate(self) -> List[Dict[str, Any]]:
        """Generate a curriculum of (X, y, difficulty) tuples."""
        rng = np.random.default_rng(0)
        # generate base "easy" data per class
        class_means = rng.standard_normal((self.n_classes, self.dim)).astype(np.float32) * 2
        for level_name, level_desc in self.DIFFICULTY_LEVELS:
            X, y = [], []
            if level_name == "binary":
                for cls in range(2):
                    for _ in range(self.n_samples_per_class):
                        X.append(class_means[cls] + rng.standard_normal(self.dim).astype(np.float32) * 0.5)
                        y.append(cls)
            elif level_name == "few_class":
                n = min(5, self.n_classes)
                for cls in range(n):
                    for _ in range(self.n_samples_per_class):
                        X.append(class_means[cls] + rng.standard_normal(self.dim).astype(np.float32) * 0.5)
                        y.append(cls)
            elif level_name == "full":
                for cls in range(self.n_classes):
                    for _ in range(self.n_samples_per_class):
                        X.append(class_means[cls] + rng.standard_normal(self.dim).astype(np.float32) * 0.7)
                        y.append(cls)
            elif level_name == "with_noise":
                for cls in range(self.n_classes):
                    for _ in range(self.n_samples_per_class):
                        X.append(class_means[cls] + rng.standard_normal(self.dim).astype(np.float32) * 0.7)
                        y.append(cls)
                # flip 10% of labels
                y = np.array(y)
                flip_idx = rng.choice(len(y), size=int(0.1 * len(y)), replace=False)
                y[flip_idx] = rng.integers(0, self.n_classes, len(flip_idx))
                y = y.tolist()
            elif level_name == "with_imbalance":
                for cls in range(self.n_classes):
                    n = self.n_samples_per_class if cls == 0 else max(10, self.n_samples_per_class // (cls + 1))
                    for _ in range(n):
                        X.append(class_means[cls] + rng.standard_normal(self.dim).astype(np.float32) * 0.7)
                        y.append(cls)
            elif level_name == "with_adversarial":
                for cls in range(self.n_classes):
                    for _ in range(self.n_samples_per_class):
                        x = class_means[cls] + rng.standard_normal(self.dim).astype(np.float32) * 0.7
                        # add adversarial-like perturbation toward a random wrong class
                        wrong = (cls + 1) % self.n_classes
                        direction = class_means[wrong] - x
                        direction = direction / (np.linalg.norm(direction) + 1e-8)
                        x = x + direction * 0.3
                        X.append(x)
                        y.append(cls)
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int64)
            # shuffle
            idx = rng.permutation(len(X))
            X, y = X[idx], y[idx]
            self.curriculum.append({
                "level": level_name,
                "description": level_desc,
                "X": X,
                "y": y,
                "n_samples": len(X),
                "n_classes": len(np.unique(y)),
            })
        return self.curriculum

    def summary(self) -> str:
        lines = [f"Synthetic Curriculum for: {self.task}"]
        for c in self.curriculum:
            lines.append(f"  Level '{c['level']}': {c['n_samples']} samples, "
                          f"{c['n_classes']} classes — {c['description']}")
        return "\n".join(lines)


# ============================================================================
# Idea 53 — Model Biography
# ============================================================================
class ModelBiography:
    """Auto-write the life story of a model.

    Reads the training history (events, losses, deployments) and
    composes a narrative biography.
    """
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.events: List[Dict[str, Any]] = []
        self.chapters: List[Dict[str, Any]] = []

    def record_event(self, event_type: str, details: Dict[str, Any]) -> None:
        self.events.append({
            "type": event_type,
            "details": details,
            "timestamp": time.time(),
        })

    def write(self) -> str:
        """Compose the biography as a narrative."""
        if not self.events:
            return f"Biography of {self.model_id}: [no events recorded]"
        chapters = []
        # Chapter 1: Birth
        birth_events = [e for e in self.events if e["type"] == "init"]
        if birth_events:
            arch = birth_events[0]["details"].get("architecture", "unknown")
            chapters.append(f"Chapter 1: Birth\n  Model {self.model_id} was created with architecture {arch}.")

        # Chapter 2: Education (training)
        train_events = [e for e in self.events if e["type"] == "train"]
        if train_events:
            n_epochs = len(train_events)
            losses = [e["details"].get("loss", 0) for e in train_events if "loss" in e["details"]]
            final_loss = losses[-1] if losses else "unknown"
            chapters.append(f"Chapter 2: Education\n  Trained for {n_epochs} epochs. "
                              f"Final loss: {final_loss}.")

        # Chapter 3: Career (deployments)
        deploy_events = [e for e in self.events if e["type"] == "deploy"]
        if deploy_events:
            n = len(deploy_events)
            last_env = deploy_events[-1]["details"].get("environment", "unknown")
            chapters.append(f"Chapter 3: Career\n  Deployed {n} times. "
                              f"Most recent environment: {last_env}.")

        # Chapter 4: Challenges (errors)
        error_events = [e for e in self.events if e["type"] == "error"]
        if error_events:
            chapters.append(f"Chapter 4: Challenges\n  Encountered {len(error_events)} "
                              f"errors during its lifetime.")

        # Chapter 5: Legacy
        eval_events = [e for e in self.events if e["type"] == "evaluate"]
        if eval_events:
            best_acc = max((e["details"].get("accuracy", 0) for e in eval_events), default=0)
            chapters.append(f"Chapter 5: Legacy\n  Best recorded accuracy: {best_acc}.")
        return "\n\n".join(chapters) + f"\n\nEnd of biography for {self.model_id}."

    def timeline(self) -> List[Dict[str, Any]]:
        """Return raw timeline of events."""
        return list(self.events)


# ============================================================================
# Idea 54 — Prediction Market
# ============================================================================
class PredictionMarket:
    """Models make predictions, "trade" on outcomes.

    Each model predicts a binary outcome (YES/NO). Prices reflect
    aggregated confidence. When the outcome is resolved, payouts go to
    the models that were correct.
    """
    def __init__(self, question: str, initial_liquidity: float = 100.0):
        self.question = question
        self.liquidity = initial_liquidity
        # shares: positive = YES, negative = NO
        self.traders: Dict[str, Dict[str, Any]] = {}
        self.price = 0.5  # probability of YES
        self.history: List[Dict[str, Any]] = []
        self.resolved: Optional[bool] = None

    def place_order(self, trader_id: str, side: str,
                     shares: float, confidence: float) -> Dict[str, Any]:
        """Place an order; market-maker mechanism adjusts price."""
        if self.resolved is not None:
            return {"accepted": False, "reason": "market resolved"}
        if side not in ("YES", "NO"):
            return {"accepted": False, "reason": "invalid side"}
        if trader_id not in self.traders:
            self.traders[trader_id] = {"YES": 0.0, "NO": 0.0, "confidence": 0.0}
        # adjust trader's position
        self.traders[trader_id][side] += shares
        self.traders[trader_id]["confidence"] = confidence
        # update market price (weighted by trader confidence)
        total_weight = sum(t["confidence"] * (t["YES"] - t["NO"])
                            for t in self.traders.values())
        total_shares = sum(abs(t["YES"]) + abs(t["NO"]) for t in self.traders.values())
        if total_shares > 0:
            self.price = 1.0 / (1.0 + math.exp(-total_weight / total_shares))
        self.history.append({
            "trader": trader_id, "side": side, "shares": shares,
            "price_after": self.price, "timestamp": time.time(),
        })
        return {"accepted": True, "current_price": self.price}

    def resolve(self, outcome: bool) -> Dict[str, Any]:
        """Resolve the market; payout to winners."""
        self.resolved = outcome
        payouts = {}
        for trader_id, pos in self.traders.items():
            yes_shares = pos["YES"]
            no_shares = pos["NO"]
            if outcome:
                payout = yes_shares - no_shares
            else:
                payout = no_shares - yes_shares
            payouts[trader_id] = float(payout)
        return {"outcome": outcome, "payouts": payouts,
                 "final_price": self.price}

    def order_book(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "current_price": self.price,
            "traders": {tid: dict(p) for tid, p in self.traders.items()},
            "resolved": self.resolved,
            "n_trades": len(self.history),
        }


# ============================================================================
# Idea 55 — Neural Debate Chamber
# ============================================================================
class NeuralDebateChamber:
    """Structured multi-model debate with rebuttals.

    Richer than ModelParliament: rounds alternate opening arguments,
    rebuttals, and closing statements. A moderator (LLM) scores each
    argument; the winner is the model with the highest cumulative score.
    """
    def __init__(self, models: List[Any],
                 moderator: Optional[Callable[[str, str], float]] = None,
                 rounds: int = 3):
        self.models = models
        self.moderator = moderator or self._default_moderator
        self.rounds = rounds
        self.transcript: List[Dict[str, Any]] = []
        self.scores: Dict[str, float] = {f"model_{i}": 0.0 for i in range(len(models))}

    @staticmethod
    def _default_moderator(argument: str, topic: str) -> float:
        """Default moderator: longer + on-topic arguments score higher."""
        words = set(argument.lower().split())
        topic_words = set(topic.lower().split())
        overlap = len(words & topic_words) / max(len(topic_words), 1)
        length_score = min(1.0, len(argument.split()) / 50.0)
        return 0.5 * length_score + 0.5 * overlap

    def debate(self, topic: str) -> Dict[str, Any]:
        self.transcript = []
        # round 1: opening arguments
        arguments: Dict[str, str] = {}
        for i, m in enumerate(self.models):
            try:
                prompt = f"Topic: {topic}\n\nPresent your opening argument."
                arg = str(m.forward(prompt) if hasattr(m, "forward") else m(prompt))
            except Exception:
                arg = "[no argument]"
            arguments[f"model_{i}"] = arg
            score = self.moderator(arg, topic)
            self.scores[f"model_{i}"] += score
            self.transcript.append({"round": 1, "phase": "opening",
                                      "model": f"model_{i}", "argument": arg,
                                      "score": score})
        # rounds 2..N-1: rebuttals
        for r in range(2, self.rounds):
            for i, m in enumerate(self.models):
                # find the strongest opponent argument to rebut
                opponents = [(k, v) for k, v in arguments.items() if k != f"model_{i}"]
                if not opponents:
                    continue
                target = max(opponents, key=lambda kv: len(kv[1]))
                try:
                    prompt = (f"Topic: {topic}\n\nOpponent's argument: {target[1][:200]}\n"
                              f"Provide a rebuttal.")
                    rebuttal = str(m.forward(prompt) if hasattr(m, "forward") else m(prompt))
                except Exception:
                    rebuttal = "[no rebuttal]"
                arguments[f"model_{i}"] = rebuttal
                score = self.moderator(rebuttal, topic)
                self.scores[f"model_{i}"] += score
                self.transcript.append({"round": r, "phase": "rebuttal",
                                          "model": f"model_{i}", "argument": rebuttal,
                                          "score": score, "rebutting": target[0]})
        # final round: closing statements
        for i, m in enumerate(self.models):
            try:
                prompt = f"Topic: {topic}\n\nGive your closing statement."
                closing = str(m.forward(prompt) if hasattr(m, "forward") else m(prompt))
            except Exception:
                closing = "[no closing]"
            score = self.moderator(closing, topic)
            self.scores[f"model_{i}"] += score
            self.transcript.append({"round": self.rounds, "phase": "closing",
                                      "model": f"model_{i}", "argument": closing,
                                      "score": score})
        winner = max(self.scores.items(), key=lambda kv: kv[1])
        return {
            "topic": topic,
            "winner": winner[0],
            "winner_score": winner[1],
            "scores": self.scores,
            "transcript": self.transcript,
        }


__all__ = [
    "ModelInsurance", "EvolvableSkill", "SkillEvolution",
    "NeuroEvolutionGenome", "NeuroEvolution",
    "FederatedAuction", "Constitution", "ModelConstitution",
    "QuantumInspiredOptimizer", "SyntheticCurriculum", "ModelBiography",
    "PredictionMarket", "NeuralDebateChamber",
]
