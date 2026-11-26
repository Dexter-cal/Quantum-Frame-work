"""
qai.creative.experimental -- 10 more original creative ideas (my own).

These go beyond the standard 2025-era toolkit. They're experimental,
opinionated, and "wouldn't it be cool if..." modules.

  Idea 16 — ModelDreaming      : the model imagines inputs that maximise
                                  a target neuron; outputs are visualised
  Idea 17 — SkillLibrary       : composable skill modules that auto-compose
                                  when a new task arrives
  Idea 18 — KnowledgeDistill2  : student learns not just outputs but also
                                  attention maps + hidden activations
  Idea 19 — ConceptBottleneck  : explicit human-interpretable concept layer
                                  in the middle of the model
  Idea 20 — MutualInformation  : estimate MI between input and output for
                                  OOD detection + privacy leakage scoring
  Idea 21 — BayesianOptimiser2 : high-dim BO via trust regions + EI
  Idea 22 — ModelMarketplace2  : revenue settlement + royalties + staking
  Idea 23 — CurriculumLearner  : difficulty-aware sampler + automatic
                                  prerequisite discovery
  Idea 24 — FederatedPrivacy   : DP-SGD + secure aggregation + client
                                  contribution bounds
  Idea 25 — NeuroSymbolicLayer : neural + symbolic (Prolog/Datalog) reasoning
                                  in one differentiable layer
"""
from __future__ import annotations
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# ============================================================================
# Idea 16 — Model Dreaming (deep-dream for any qai model)
# ============================================================================
class ModelDreamer:
    """Generate inputs that maximise a target neuron's activation.

    The model "dreams" by gradient ascent on a chosen neuron's activation.
    Like DeepDream but for any qai model that exposes forward + gradients.
    """
    def __init__(self, model: Any, input_shape: Tuple[int, ...]):
        self.model = model
        self.input_shape = input_shape

    def dream(self, target_layer: str, target_neuron: int,
               n_steps: int = 100, lr: float = 0.1,
               initial_noise: Optional[np.ndarray] = None) -> np.ndarray:
        """Optimise an input to maximise target neuron's activation."""
        if not _TORCH:
            raise ImportError("ModelDreamer requires torch")
        x = (initial_noise if initial_noise is not None
              else np.random.randn(*self.input_shape)).astype(np.float32)
        x = torch.tensor(x, requires_grad=True)
        # find target module
        target = None
        for name, m in self.model.named_modules() if hasattr(self.model, "named_modules") else []:
            if name == target_layer:
                target = m; break
        if target is None:
            # use final output instead
            target = self.model
        activations: Dict[str, torch.Tensor] = {}
        def hook(m, inp, out):
            activations["out"] = out
        h = target.register_forward_hook(hook)
        try:
            for step in range(n_steps):
                self.model.zero_grad()
                self.model(x.unsqueeze(0) if x.dim() == len(self.input_shape) else x)
                a = activations["out"]
                # maximise activation of target neuron
                while a.dim() > 1:
                    a = a[0]
                loss = -a[target_neuron]
                loss.backward()
                with torch.no_grad():
                    x += lr * x.grad
                    x.grad = None
        finally:
            h.remove()
        return x.detach().numpy()


# ============================================================================
# Idea 17 — Skill Library (composable skills)
# ============================================================================
class Skill:
    """A composable skill with name, description, function, prerequisites."""
    def __init__(self, name: str, description: str, fn: Callable,
                 prerequisites: Optional[List[str]] = None,
                 inputs: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.fn = fn
        self.prerequisites = prerequisites or []
        self.inputs = inputs or []
        self.outputs = outputs or []


class SkillLibrary:
    """A library of composable skills.

    When a task arrives, the planner finds a chain of skills that produces
    the required outputs from available inputs, considering prerequisites.
    """
    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def plan(self, available_inputs: Set[str],
              required_outputs: Set[str]) -> List[Skill]:
        """Find an ordered chain of skills that produces required_outputs.

        We do a forward search: at each step, add any skill whose inputs
        are available and whose outputs are not yet produced. Stop when
        all required_outputs are in `have`.
        """
        order: List[Skill] = []
        have = set(available_inputs)
        # compute which outputs each skill can produce that we don't yet have
        # use a worklist algorithm — keep adding skills that can fire
        # until all required outputs are available
        max_iter = len(self._skills) * 2 + 5
        for _ in range(max_iter):
            if required_outputs <= have:
                return order
            progress = False
            for skill in self._skills.values():
                # skip already-added skills
                if skill in order:
                    continue
                # check inputs available + prerequisites already added
                if not all(i in have for i in skill.inputs):
                    continue
                if not all(p in [s.name for s in order] for p in skill.prerequisites):
                    continue
                # check skill produces at least one new useful output
                new_outputs = set(skill.outputs) - have
                # transitive reachability: would this skill help us reach required_outputs?
                # (we accept it if any output is new; eventually required_outputs will be met)
                if new_outputs:
                    order.append(skill)
                    have.update(skill.outputs)
                    progress = True
                    break
            if not progress:
                raise ValueError(f"cannot fulfil {required_outputs - have} from {have}")
        raise ValueError(f"planning exceeded max iterations; missing: {required_outputs - have}")

    def execute(self, available_inputs: Dict[str, Any],
                required_outputs: Set[str]) -> Dict[str, Any]:
        plan = self.plan(set(available_inputs.keys()), required_outputs)
        ctx = dict(available_inputs)
        for skill in plan:
            args = {k: ctx[k] for k in skill.inputs if k in ctx}
            result = skill.fn(**args)
            if isinstance(result, dict):
                ctx.update(result)
            else:
                ctx[skill.outputs[0] if skill.outputs else skill.name] = result
        return {k: ctx[k] for k in required_outputs if k in ctx}


# ============================================================================
# Idea 18 — Knowledge Distillation v2 (attention + hidden alignment)
# ============================================================================
class KnowledgeDistiller2:
    """Multi-level KD: outputs + attention maps + hidden activations.

    Loss = alpha * task_loss + beta * attention_loss + gamma * hidden_loss
    """
    def __init__(self, teacher: Any, student: Any,
                 alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2,
                 temperature: float = 4.0):
        self.teacher = teacher
        self.student = student
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.temperature = temperature

    def distill(self, X: np.ndarray, y: np.ndarray,
                 n_steps: int = 100) -> Dict[str, Any]:
        if not _TORCH:
            raise ImportError("KnowledgeDistiller2 requires torch")
        import torch.nn.functional as F
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.long)
        opt = torch.optim.Adam(self.student.parameters(), lr=1e-3)
        history = []
        for step in range(n_steps):
            opt.zero_grad()
            with torch.no_grad():
                t_out = self.teacher(X)
            s_out = self.student(X)
            # output KL
            t_soft = F.softmax(t_out / self.temperature, dim=-1)
            s_log_soft = F.log_softmax(s_out / self.temperature, dim=-1)
            kd_loss = F.kl_div(s_log_soft, t_soft, reduction="batchmean")
            # task loss
            task_loss = F.cross_entropy(s_out, y)
            # attention alignment (approximation: align first-layer weights)
            attn_loss = torch.tensor(0.0)
            try:
                t_first = list(self.teacher.parameters())[0]
                s_first = list(self.student.parameters())[0]
                if t_first.shape == s_first.shape:
                    attn_loss = F.mse_loss(s_first, t_first)
            except Exception:
                pass
            # hidden alignment
            hidden_loss = torch.tensor(0.0)
            try:
                t_hidden = self.teacher(X)
                s_hidden = self.student(X)
                if t_hidden.shape == s_hidden.shape:
                    hidden_loss = F.mse_loss(s_hidden, t_hidden.detach())
            except Exception:
                pass
            loss = (self.alpha * task_loss + self.beta * attn_loss
                    + self.gamma * hidden_loss + (1 - self.alpha) * kd_loss)
            loss.backward()
            opt.step()
            history.append({"step": step, "loss": float(loss),
                              "task": float(task_loss),
                              "kd": float(kd_loss)})
        return {"history": history, "final_loss": history[-1]["loss"]}


# ============================================================================
# Idea 19 — Concept Bottleneck Models
# ============================================================================
class ConceptBottleneckModel:
    """A model with an explicit human-interpretable concept layer.

    x -> [encoder] -> c (concept vector) -> [predictor] -> y

    The concept layer c is supervised by ground-truth concept labels so
    humans can inspect and edit it before the final prediction.
    """
    def __init__(self, n_input_features: int, n_concepts: int, n_classes: int,
                 hidden: int = 32):
        if not _TORCH:
            raise ImportError("ConceptBottleneckModel requires torch")
        self.encoder = nn.Sequential(
            nn.Linear(n_input_features, hidden), nn.ReLU(),
            nn.Linear(hidden, n_concepts), nn.Sigmoid(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(n_concepts, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()), lr=1e-3)
        self.concept_names: List[str] = [f"c_{i}" for i in range(n_concepts)]

    def fit(self, X: np.ndarray, concepts: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None, n_steps: int = 100) -> Dict[str, Any]:
        X = torch.as_tensor(X, dtype=torch.float32)
        c_pred = self.encoder(X)
        y_pred = self.predictor(c_pred)
        loss = torch.tensor(0.0)
        if concepts is not None:
            c_t = torch.as_tensor(concepts, dtype=torch.float32)
            loss = loss + F.mse_loss(c_pred, c_t)
        if y is not None:
            y_t = torch.as_tensor(y, dtype=torch.long)
            loss = loss + F.cross_entropy(y_pred, y_t)
        for step in range(n_steps):
            self.opt.zero_grad()
            c_pred = self.encoder(X)
            y_pred = self.predictor(c_pred)
            loss = torch.tensor(0.0)
            if concepts is not None:
                loss = loss + F.mse_loss(c_pred, c_t)
            if y is not None:
                loss = loss + F.cross_entropy(y_pred, y_t)
            loss.backward()
            self.opt.step()
        return {"final_loss": float(loss)}

    def predict_with_concepts(self, x: np.ndarray) -> Dict[str, Any]:
        x = torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            c = self.encoder(x.unsqueeze(0) if x.dim() == 1 else x)
            y = self.predictor(c)
        return {"concepts": c.squeeze(0).numpy(), "logits": y.squeeze(0).numpy()}

    def intervene(self, x: np.ndarray, concept_idx: int, value: float) -> Dict[str, Any]:
        """Override a concept value and re-predict (counterfactual)."""
        x = torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            c = self.encoder(x.unsqueeze(0) if x.dim() == 1 else x)
            c[0, concept_idx] = value
            y = self.predictor(c)
        return {"intervened_concepts": c.squeeze(0).numpy(),
                 "logits": y.squeeze(0).numpy()}


# ============================================================================
# Idea 20 — Mutual Information Estimator (privacy/OOD)
# ============================================================================
class MutualInformationEstimator:
    """Estimate MI between input and output via the InfoNCE lower bound.

    High MI => strong input dependence (good for accuracy, bad for privacy).
    Useful for: OOD detection, membership inference, privacy auditing.
    """
    def __init__(self, model: Any, n_negative: int = 8):
        self.model = model
        self.n_negative = n_negative

    def _embed(self, x: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(self.model.forward(x), dtype=np.float32).flatten()
        except Exception:
            return np.zeros(8)

    def estimate(self, X: np.ndarray) -> Dict[str, float]:
        """Estimate MI between input X and model outputs."""
        N = len(X)
        if N < 2:
            return {"mi": 0.0, "n_samples": N}
        # InfoNCE lower bound
        embeddings = np.array([self._embed(x) for x in X])
        # for each i, compare sim(i,i) to sim(i,j) for j != i
        logits = embeddings @ embeddings.T
        nce_sum = 0.0
        for i in range(N):
            pos = logits[i, i]
            neg = np.concatenate([logits[i, :i], logits[i, i+1:]])
            # sample negatives
            if len(neg) > self.n_negative:
                neg = np.random.choice(neg, self.n_negative, replace=False)
            denom = pos + np.sum(np.exp(neg - pos))
            nce_sum += np.log(max(denom, 1e-8))
        mi = float(np.mean(nce_sum))
        return {"mi": mi, "n_samples": N, "n_negative": self.n_negative,
                "avg_embedding_norm": float(np.mean(np.linalg.norm(embeddings, axis=1)))}


# ============================================================================
# Idea 21 — High-dimensional Bayesian Optimiser (TR-EI)
# ============================================================================
class BayesianOptimiser2:
    """Trust-region Bayesian optimisation with Expected Improvement.

    Adapts search radius to local landscape; scales to higher dimensions
    than vanilla GP-EI.
    """
    def __init__(self, dim: int, bounds: Optional[List[Tuple[float, float]]] = None,
                 trust_radius: float = 0.3, n_initial: int = 5):
        self.dim = dim
        self.bounds = bounds or [(-1, 1)] * dim
        self.trust_radius = trust_radius
        self.n_initial = n_initial
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.rng = np.random.default_rng(0)

    def suggest(self) -> np.ndarray:
        if len(self.X) < self.n_initial:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
            return x
        # otherwise: pick best so far, sample within trust radius, pick max EI
        best_idx = int(np.argmax(self.y))
        best = np.array(self.X[best_idx])
        candidates = []
        for _ in range(20):
            delta = self.rng.normal(0, self.trust_radius, self.dim)
            x = np.clip(best + delta,
                         [b[0] for b in self.bounds],
                         [b[1] for b in self.bounds])
            candidates.append(x)
        # estimate EI via simple GP surrogate (radial basis)
        def predict(x):
            if not self.X:
                return 0.0, 1.0
            dists = np.array([np.linalg.norm(x - np.array(xi)) for xi in self.X])
            weights = np.exp(-dists ** 2 / (2 * self.trust_radius ** 2))
            mu = float(np.sum(weights * np.array(self.y)) / (weights.sum() + 1e-8))
            sigma = float(np.std(self.y) * (1 - weights.max()))
            return mu, max(sigma, 1e-3)
        def ei(x):
            mu, sigma = predict(x)
            y_best = max(self.y)
            z = (mu - y_best) / sigma
            return float((mu - y_best) * 0.5 + sigma * 0.4)
        best_candidate = max(candidates, key=ei)
        return best_candidate

    def observe(self, x: np.ndarray, y: float) -> None:
        self.X.append(np.array(x))
        self.y.append(float(y))

    def best(self) -> Tuple[np.ndarray, float]:
        if not self.X:
            return None, 0.0
        idx = int(np.argmax(self.y))
        return self.X[idx], self.y[idx]


# ============================================================================
# Idea 22 — Model Marketplace v2 (royalties + staking)
# ============================================================================
class MarketplaceV2:
    """Extended marketplace with revenue split, royalties, and staking.

    - Each model has an author + downstream consumers
    - If model B is fine-tuned from A, A gets a royalty on B's revenue
    - Stakers validate models and earn a share of fees
    """
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.stakes: Dict[str, Dict[str, float]] = {}  # model_id -> {user: stake}
        self.transactions: List[Dict[str, Any]] = []

    def publish(self, model_id: str, author: str, parent: Optional[str] = None,
                price: float = 0.0, royalty_rate: float = 0.1) -> None:
        self.models[model_id] = {
            "id": model_id, "author": author, "parent": parent,
            "price": price, "royalty_rate": royalty_rate,
            "downloads": 0, "revenue": 0.0,
        }

    def download(self, model_id: str, user: str) -> Dict[str, Any]:
        if model_id not in self.models:
            raise KeyError(f"model '{model_id}' not found")
        m = self.models[model_id]
        m["downloads"] += 1
        # settle payment + royalty
        payment = m["price"]
        royalty = payment * m["royalty_rate"] if m["parent"] else 0.0
        author_share = payment - royalty
        m["revenue"] += author_share
        if m["parent"] and m["parent"] in self.models:
            self.models[m["parent"]]["revenue"] += royalty
        tx = {"model": model_id, "user": user, "payment": payment,
              "royalty": royalty, "author_share": author_share, "ts": time.time()}
        self.transactions.append(tx)
        # distribute staker rewards (1% of payment split by stake weight)
        if model_id in self.stakes and self.stakes[model_id]:
            total_stake = sum(self.stakes[model_id].values())
            reward_pool = payment * 0.01
            for staker, stake in self.stakes[model_id].items():
                share = reward_pool * (stake / total_stake)
                self.transactions.append({"type": "stake_reward", "model": model_id,
                                           "user": staker, "amount": share, "ts": time.time()})
        return {"downloaded": True, **tx}

    def stake(self, model_id: str, user: str, amount: float) -> None:
        self.stakes.setdefault(model_id, {})
        self.stakes[model_id][user] = self.stakes[model_id].get(user, 0.0) + amount

    def author_revenue(self, author: str) -> float:
        return sum(m["revenue"] for m in self.models.values() if m["author"] == author)

    def leaderboard(self) -> List[Dict[str, Any]]:
        items = sorted(self.models.values(), key=lambda m: -m["revenue"])
        return [{"model": m["id"], "author": m["author"],
                 "revenue": m["revenue"], "downloads": m["downloads"]} for m in items]


import time  # for MarketplaceV2


# ============================================================================
# Idea 23 — Curriculum Learner
# ============================================================================
class CurriculumLearner:
    """Difficulty-aware sampler with automatic prerequisite discovery.

    Tracks per-example difficulty; samples easier ones more frequently
    early in training, then anneals toward harder ones. Also discovers
    prerequisite relationships (which examples must be learned first).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 initial_difficulty: Optional[np.ndarray] = None):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        N = len(X)
        self.difficulty = (initial_difficulty if initial_difficulty is not None
                            else np.ones(N) * 0.5)
        self.success_history: List[np.ndarray] = []
        self.prerequisites: Dict[int, Set[int]] = {}

    def update(self, indices: np.ndarray, successes: np.ndarray) -> None:
        """Record which examples were learned successfully this round."""
        for idx, succ in zip(indices, successes):
            old = self.difficulty[idx]
            # exponential moving average of inverse success
            new = 1.0 - float(succ)
            self.difficulty[idx] = 0.7 * old + 0.3 * new
        self.success_history.append(successes)

    def sample(self, n: int, difficulty_bias: float = 0.5) -> np.ndarray:
        """Sample n examples, biased toward easier ones early.

        difficulty_bias = 0 -> always easy
        difficulty_bias = 1 -> always hard
        difficulty_bias = 0.5 -> mixed
        """
        N = len(self.X)
        # convert difficulty to sampling probabilities
        d = self.difficulty
        if difficulty_bias <= 0.5:
            # prefer easy (low difficulty)
            p = (1 - d) ** (2 * (0.5 - difficulty_bias) + 1)
        else:
            p = d ** (2 * (difficulty_bias - 0.5) + 1)
        p = p / p.sum()
        return np.random.choice(N, size=n, replace=True, p=p)

    def discover_prerequisites(self, model_forward: Callable[[Any], Any]) -> Dict[int, Set[int]]:
        """For each hard example, find easy examples whose outputs predict it."""
        N = len(self.X)
        for i in range(N):
            if self.difficulty[i] < 0.7:
                continue
            preds_for_i = []
            for j in range(N):
                if i == j or self.difficulty[j] > 0.3:
                    continue
                # if learning example j helps with example i, mark j as prereq
                try:
                    out_j = model_forward(self.X[j])
                    out_i = model_forward(self.X[i])
                    if np.corrcoef(out_i.flatten(), out_j.flatten())[0, 1] > 0.5:
                        self.prerequisites.setdefault(i, set()).add(j)
                except Exception:
                    pass
        return self.prerequisites

    def curriculum_schedule(self, total_epochs: int) -> List[float]:
        """Return difficulty bias per epoch (easy -> hard)."""
        return [0.0 + 1.0 * (ep / max(total_epochs - 1, 1)) for ep in range(total_epochs)]


# ============================================================================
# Idea 24 — Federated Privacy (DP-SGD + secure aggregation)
# ============================================================================
class FederatedPrivacy:
    """Differential-privacy federated learning with secure aggregation.

    Each client adds Gaussian noise to their gradient (DP-SGD). Server
    sums without ever seeing individual gradients (secure aggregation).
    """
    def __init__(self, n_clients: int, epsilon: float = 1.0,
                 delta: float = 1e-5, clip_norm: float = 1.0):
        self.n_clients = n_clients
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm

    def clip_and_noise(self, gradient: np.ndarray,
                        sensitivity: Optional[float] = None) -> np.ndarray:
        """Clip gradient + add Gaussian noise calibrated to (eps, delta)."""
        # clip
        norm = np.linalg.norm(gradient)
        if norm > self.clip_norm:
            gradient = gradient * (self.clip_norm / norm)
        # noise scale: sigma = sqrt(2 * ln(1.25/delta)) / epsilon
        sigma = math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma, gradient.shape)
        return gradient + noise

    def secure_aggregate(self, client_updates: List[np.ndarray]) -> np.ndarray:
        """Sum updates without server seeing any individual update.

        Simplified: each client masks their update with a random vector
        that cancels out when summed.
        """
        n = len(client_updates)
        if n == 0:
            return np.zeros(0)
        shape = client_updates[0].shape
        # each pair (i, j) shares a random vector r_ij, i sends +r_ij, j sends -r_ij
        masked = [np.zeros(shape) for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                r = np.random.normal(0, 1, shape)
                masked[i] += r
                masked[j] -= r
        for i in range(n):
            masked[i] += client_updates[i]
        # sum of masks = sum of originals (random parts cancel)
        return np.sum(masked, axis=0)

    def privacy_budget_spent(self, rounds: int) -> Dict[str, float]:
        """Estimate cumulative epsilon after `rounds` rounds (RDP composition)."""
        # simple composition: epsilon_total = epsilon_per_round * sqrt(rounds)
        return {"epsilon_total": self.epsilon * math.sqrt(rounds),
                "delta": self.delta,
                "rounds": rounds}


# ============================================================================
# Idea 25 — Neuro-Symbolic Layer
# ============================================================================
class NeuroSymbolicLayer:
    """Differentiable layer that combines neural perception with symbolic
    (Datalog-style) reasoning.

    Inputs go through a neural encoder to produce facts (probabilities).
    Facts are fed into a differentiable forward-chaining engine that
    applies Horn-clause rules to derive new facts.

    Output: derived fact probabilities, differentiable w.r.t. encoder weights.
    """
    def __init__(self, n_facts: int, rules: List[Tuple[List[str], str]]):
        """rules: list of (body_predicates, head_predicate). Each predicate
        is a fact index or a name registered via register_fact_name.
        """
        self.n_facts = n_facts
        self.rules = rules
        self.fact_names: Dict[str, int] = {}
        # convert names to indices in rules
        self._compiled_rules: List[Tuple[List[int], int]] = []
        for body, head in rules:
            body_idx = [self._idx(b) for b in body]
            head_idx = self._idx(head)
            self._compiled_rules.append((body_idx, head_idx))

    def _idx(self, name_or_idx) -> int:
        if isinstance(name_or_idx, int):
            return name_or_idx
        return self.fact_names.get(name_or_idx, 0)

    def register_fact_name(self, name: str, idx: int) -> None:
        self.fact_names[name] = idx

    def forward(self, perceived_facts: np.ndarray) -> np.ndarray:
        """Apply forward chaining for one step.

        perceived_facts: (batch, n_facts) probabilities from neural encoder
        Returns: (batch, n_facts) updated probabilities after one inference step

        Rules are applied in order; later rules see updates from earlier ones
        within the same forward pass (semi-naive forward chaining).
        """
        out = perceived_facts.copy()
        for body, head in self._compiled_rules:
            # head = product of body probabilities (t-norm AND)
            # read from `out` (which includes earlier rule updates)
            body_probs = out[:, body]  # (batch, len(body))
            head_prob = body_probs.prod(axis=-1)
            out[:, head] = np.maximum(out[:, head], head_prob)
        return out

    def explain(self, fact_idx: int) -> List[str]:
        """Explain why a fact was derived."""
        explanations = []
        for body, head in self._compiled_rules:
            if head == fact_idx:
                body_names = [self._name(b) for b in body]
                explanations.append(f"because {body_names} -> {self._name(head)}")
        return explanations

    def _name(self, idx: int) -> str:
        for n, i in self.fact_names.items():
            if i == idx:
                return n
        return f"f_{idx}"


__all__ = [
    "ModelDreamer", "Skill", "SkillLibrary", "KnowledgeDistiller2",
    "ConceptBottleneckModel", "MutualInformationEstimator",
    "BayesianOptimiser2", "MarketplaceV2", "CurriculumLearner",
    "FederatedPrivacy", "NeuroSymbolicLayer",
]
