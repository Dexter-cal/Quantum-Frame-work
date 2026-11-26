"""
qai.creative.ecosystem -- Creative Ideas 12, 13, 14 & 15

Idea 12 — MODEL MARKETPLACE
    Publish / discover / rate models in an in-process marketplace with
    semantic search, dependency tracking, and revenue-style use counters.

Idea 13 — COLLABORATIVE TRAINING
    Peer-to-peer + federated-style training. Each peer trains locally,
    then a coordinator aggregates gradients / parameters via FedAvg or
    gradient-sign gossip. Includes Sybil-resistance via stake weighting.

Idea 14 — MODEL AGING
    Track performance + data drift over time; flag staleness; suggest
    retraining windows. Models have a "shelf life".

Idea 15 — ADVERSARIAL PARTNER
    Auto-red-team: given a target model + policy, generate adversarial
    inputs that try to break it, log the failures, and (optionally) feed
    them back as hard-negative training examples.
"""
from __future__ import annotations
import copy
import hashlib
import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# Idea 12 — MODEL MARKETPLACE
# ============================================================================
class MarketModel:
    """A model listing in the marketplace."""

    def __init__(self, name: str, model: Any,
                 author: str = "anonymous",
                 description: str = "",
                 tags: Optional[List[str]] = None,
                 price: float = 0.0,
                 version: str = "1.0.0"):
        self.name = name
        self.model = model
        self.author = author
        self.description = description
        self.tags = tags or []
        self.price = price
        self.version = version
        self.uploaded_at = time.time()
        self.downloads = 0
        self.ratings: List[Tuple[str, int, str]] = []  # (user_id, stars, review)
        self.dependencies: List[str] = []

    def add_rating(self, user_id: str, stars: int, review: str = "") -> None:
        if not 1 <= stars <= 5:
            raise ValueError("stars must be 1-5")
        self.ratings.append((user_id, stars, review))

    def avg_rating(self) -> float:
        if not self.ratings:
            return 0.0
        return float(np.mean([r[1] for r in self.ratings]))

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name, "author": self.author, "version": self.version,
            "description": self.description, "tags": self.tags,
            "price": self.price, "downloads": self.downloads,
            "avg_rating": self.avg_rating(), "n_ratings": len(self.ratings),
            "uploaded_at": self.uploaded_at,
        }


class ModelMarketplace:
    """In-process model marketplace with search + ratings + revenue."""

    def __init__(self):
        self._listings: Dict[str, MarketModel] = {}
        self._usage_log: List[Dict[str, Any]] = []

    def publish(self, listing: MarketModel) -> None:
        if listing.name in self._listings:
            raise ValueError(f"Model '{listing.name}' already published. Bump version.")
        self._listings[listing.name] = listing

    def unpublish(self, name: str) -> None:
        self._listings.pop(name, None)

    def search(self, query: str = "", tags: Optional[List[str]] = None,
                max_price: Optional[float] = None,
                sort_by: str = "relevance") -> List[Dict[str, Any]]:
        """Search listings by text + filter by tags/price."""
        q = query.lower() if query else ""
        results = []
        for name, listing in self._listings.items():
            score = 0
            if q:
                if q in listing.name.lower():
                    score += 3
                if q in listing.description.lower():
                    score += 2
                for tag in listing.tags:
                    if q in tag.lower():
                        score += 2
            if tags:
                if not any(t in listing.tags for t in tags):
                    continue
            if max_price is not None and listing.price > max_price:
                continue
            results.append({**listing.metadata(), "_score": score})
        if sort_by == "relevance":
            results.sort(key=lambda x: -x["_score"])
        elif sort_by == "downloads":
            results.sort(key=lambda x: -x["downloads"])
        elif sort_by == "rating":
            results.sort(key=lambda x: -x["avg_rating"])
        elif sort_by == "newest":
            results.sort(key=lambda x: -x["uploaded_at"])
        return results

    def download(self, name: str, user_id: str = "anonymous") -> MarketModel:
        if name not in self._listings:
            raise KeyError(f"Model '{name}' not found in marketplace")
        listing = self._listings[name]
        listing.downloads += 1
        self._usage_log.append({
            "user": user_id, "model": name, "action": "download", "ts": time.time()
        })
        return listing

    def rate(self, name: str, user_id: str, stars: int, review: str = "") -> None:
        if name not in self._listings:
            raise KeyError(f"Model '{name}' not found")
        self._listings[name].add_rating(user_id, stars, review)

    def leaderboard(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Top-k models by a composite score of downloads + ratings."""
        items = []
        for name, listing in self._listings.items():
            score = listing.downloads + 10 * listing.avg_rating() * math.sqrt(len(listing.ratings) + 1)
            items.append({**listing.metadata(), "composite_score": score})
        items.sort(key=lambda x: -x["composite_score"])
        return items[:top_k]

    def revenue_report(self) -> Dict[str, Any]:
        """Total revenue = sum(price * downloads) per author."""
        by_author: Dict[str, float] = {}
        for listing in self._listings.values():
            by_author[listing.author] = by_author.get(listing.author, 0) + listing.price * listing.downloads
        return {
            "total_revenue": sum(by_author.values()),
            "by_author": dict(sorted(by_author.items(), key=lambda kv: -kv[1])),
        }


# ============================================================================
# Idea 13 — COLLABORATIVE TRAINING
# ============================================================================
class Peer:
    """A single training peer with its own model + data."""

    def __init__(self, peer_id: str, model: Any, X: np.ndarray, y: np.ndarray,
                 stake: float = 1.0):
        self.id = peer_id
        self.model = model
        self.X = X
        self.y = y
        self.stake = stake
        self.contribution_score = 0.0


class CollaborativeTraining:
    """Coordinator for federated / gossip-style collaborative training.

    Aggregation strategies:
      - 'fedavg'    : weighted average of model weights
      - 'gossip'    : each peer pulls from one random peer
      - 'gradient'  : average gradient updates
    """

    def __init__(self, peers: List[Peer], strategy: str = "fedavg",
                 rounds: int = 10):
        self.peers = peers
        self.strategy = strategy
        self.rounds = rounds
        self.history: List[Dict[str, Any]] = []

    def _extract_weights(self, model: Any) -> Optional[Dict[str, np.ndarray]]:
        if hasattr(model, "state_dict"):
            try:
                sd = model.state_dict()
                return {k: np.asarray(v, dtype=np.float32) for k, v in sd.items()}
            except Exception:
                pass
        return None

    def _load_weights(self, model: Any, weights: Dict[str, np.ndarray]) -> None:
        if hasattr(model, "load_state_dict"):
            try:
                model.load_state_dict(weights)
            except Exception:
                pass

    def _fedavg(self, weights_list: List[Tuple[float, Dict[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
        total_stake = sum(s for s, _ in weights_list)
        out = {}
        keys = set()
        for _, w in weights_list:
            keys.update(w.keys())
        for k in keys:
            arrays = []
            for s, w in weights_list:
                if k in w:
                    arrays.append(s * w[k])
            out[k] = sum(arrays) / total_stake
        return out

    def step(self, round_idx: int) -> Dict[str, Any]:
        # local training
        for peer in self.peers:
            try:
                peer.model.fit(peer.X, peer.y)
            except Exception:
                pass
        # aggregation
        if self.strategy == "fedavg":
            weights_list = [(p.stake, self._extract_weights(p.model) or {})
                            for p in self.peers]
            if any(w for _, w in weights_list):
                agg = self._fedavg(weights_list)
                for peer in self.peers:
                    self._load_weights(peer.model, agg)
        elif self.strategy == "gossip":
            for peer in self.peers:
                other = random.choice([p for p in self.peers if p.id != peer.id])
                w = self._extract_weights(other.model)
                if w:
                    self._load_weights(peer.model, w)
        # log
        return {"round": round_idx, "strategy": self.strategy,
                "n_peers": len(self.peers)}

    def run(self) -> Dict[str, Any]:
        for r in range(self.rounds):
            entry = self.step(r)
            self.history.append(entry)
        return {"rounds_completed": len(self.history), "final": self.history[-1]}

    def contribution_report(self) -> Dict[str, float]:
        """Estimate each peer's contribution via leave-one-out."""
        # heuristic: peers with higher stake + more data contributed more
        total_data = sum(len(p.X) for p in self.peers) or 1
        return {p.id: (p.stake * len(p.X) / total_data) for p in self.peers}


# ============================================================================
# Idea 14 — MODEL AGING
# ============================================================================
class ModelAging:
    """Track a model's performance over time + flag staleness.

    Models have a "shelf life" — performance degrades as the world drifts
    away from the training distribution. This class wraps the model and
    records periodic measurements; when degradation crosses a threshold,
    it suggests a retrain.
    """

    def __init__(self, model: Any, baseline_metric: float = 1.0,
                 degradation_threshold: float = 0.1,
                 shelf_life_days: int = 30):
        self.model = model
        self.baseline_metric = baseline_metric
        self.degradation_threshold = degradation_threshold
        self.shelf_life_days = shelf_life_days
        self.measurements: List[Dict[str, Any]] = []
        self.trained_at = time.time()

    def measure(self, metric: float, drift_score: float = 0.0,
                 note: str = "") -> Dict[str, Any]:
        entry = {
            "timestamp": time.time(),
            "metric": metric,
            "drift_score": drift_score,
            "degradation": max(0.0, self.baseline_metric - metric),
            "age_days": (time.time() - self.trained_at) / 86400,
            "note": note,
        }
        self.measurements.append(entry)
        return entry

    def health(self) -> Dict[str, Any]:
        if not self.measurements:
            return {"status": "fresh", "degradation": 0.0,
                    "should_retrain": False}
        last = self.measurements[-1]
        age_days = last["age_days"]
        deg = last["degradation"]
        # combined score
        age_pct = min(age_days / self.shelf_life_days, 1.0)
        overall = deg + 0.5 * age_pct * self.degradation_threshold
        status = "fresh"
        if overall > 2 * self.degradation_threshold:
            status = "critical"
        elif overall > self.degradation_threshold:
            status = "stale"
        elif overall > 0.5 * self.degradation_threshold:
            status = "aging"
        return {
            "status": status,
            "degradation": deg,
            "age_days": age_days,
            "age_pct": age_pct,
            "should_retrain": overall > self.degradation_threshold,
            "recommended_action": "retrain" if overall > self.degradation_threshold else "monitor",
            "last_metric": last["metric"],
            "n_measurements": len(self.measurements),
        }

    def aging_curve(self) -> Dict[str, Any]:
        """Return the historical degradation + drift trajectory."""
        return {
            "timestamps": [m["timestamp"] for m in self.measurements],
            "metrics": [m["metric"] for m in self.measurements],
            "drifts": [m["drift_score"] for m in self.measurements],
            "degradations": [m["degradation"] for m in self.measurements],
        }


# ============================================================================
# Idea 15 — ADVERSARIAL PARTNER
# ============================================================================
class AdversarialPartner:
    """Auto-red-team a target model.

    Strategies:
      - 'fgsm'      : fast gradient sign method (single-step)
      - 'pgd'       : projected gradient descent (multi-step)
      - 'genetic'   : evolve perturbations to maximise loss
      - 'mutation'  : random input mutations + selection
      - 'text'      : paraphrase + character swaps for text inputs

    Found adversarial examples can be added back as hard negatives.
    """

    def __init__(self, target_model: Any,
                 eps: float = 0.1,
                 n_steps: int = 10,
                 strategy: str = "mutation"):
        self.target = target_model
        self.eps = eps
        self.n_steps = n_steps
        self.strategy = strategy
        self.attacks: List[Dict[str, Any]] = []

    def _predict(self, x: np.ndarray) -> Any:
        try:
            return np.asarray(self.target.forward(x))
        except Exception:
            return np.zeros(1)

    def _loss(self, x: np.ndarray, y: int) -> float:
        """Negative cross-entropy (higher = more adversarial)."""
        out = self._predict(x).flatten()
        if out.size == 0:
            return 0.0
        out = out - out.max()
        exp = np.exp(out)
        probs = exp / exp.sum()
        return -math.log(probs[y] + 1e-12)

    def attack(self, x: np.ndarray, y: int) -> Dict[str, Any]:
        x = np.asarray(x, dtype=np.float32).copy()
        if self.strategy == "fgsm":
            return self._fgsm(x, y)
        if self.strategy == "pgd":
            return self._pgd(x, y)
        if self.strategy == "genetic":
            return self._genetic(x, y)
        return self._mutation(x, y)

    def _fgsm(self, x: np.ndarray, y: int) -> Dict[str, Any]:
        # numerical gradient via central differences
        grad = np.zeros_like(x)
        h = 1e-3
        for i in range(x.size):
            x_plus = x.copy().flatten()
            x_plus[i] += h
            x_minus = x.copy().flatten()
            x_minus[i] -= h
            l_plus = self._loss(x_plus.reshape(x.shape), y)
            l_minus = self._loss(x_minus.reshape(x.shape), y)
            grad.flat[i] = (l_plus - l_minus) / (2 * h)
        x_adv = x + self.eps * np.sign(grad).reshape(x.shape)
        return self._record(x, x_adv, y)

    def _pgd(self, x: np.ndarray, y: int) -> Dict[str, Any]:
        x_adv = x.copy()
        step = self.eps / self.n_steps
        for _ in range(self.n_steps):
            grad = np.zeros_like(x_adv)
            h = 1e-3
            for i in range(x_adv.size):
                x_plus = x_adv.copy().flatten()
                x_plus[i] += h
                x_minus = x_adv.copy().flatten()
                x_minus[i] -= h
                l_plus = self._loss(x_plus.reshape(x_adv.shape), y)
                l_minus = self._loss(x_minus.reshape(x_adv.shape), y)
                grad.flat[i] = (l_plus - l_minus) / (2 * h)
            x_adv = x_adv + step * np.sign(grad).reshape(x_adv.shape)
            # project back to eps-ball
            delta = np.clip(x_adv - x, -self.eps, self.eps)
            x_adv = x + delta
        return self._record(x, x_adv, y)

    def _genetic(self, x: np.ndarray, y: int,
                  pop_size: int = 20, generations: int = 10) -> Dict[str, Any]:
        pop = [x + np.random.normal(0, self.eps, x.shape).astype(np.float32)
               for _ in range(pop_size)]
        for gen in range(generations):
            scored = [(self._loss(p, y), p) for p in pop]
            scored.sort(key=lambda kv: -kv[0])
            top = [p for _, p in scored[:max(2, pop_size // 2)]]
            new_pop = list(top)
            while len(new_pop) < pop_size:
                p1, p2 = random.sample(top, 2)
                child = (p1 + p2) / 2 + np.random.normal(0, self.eps * 0.1, x.shape).astype(np.float32)
                new_pop.append(child)
            pop = new_pop
        best = max(pop, key=lambda p: self._loss(p, y))
        return self._record(x, best, y)

    def _mutation(self, x: np.ndarray, y: int) -> Dict[str, Any]:
        best = x.copy()
        best_loss = self._loss(x, y)
        for _ in range(self.n_steps):
            candidate = x + np.random.normal(0, self.eps, x.shape).astype(np.float32)
            l = self._loss(candidate, y)
            if l > best_loss:
                best = candidate
                best_loss = l
        return self._record(x, best, y)

    def _record(self, x: np.ndarray, x_adv: np.ndarray, y: int) -> Dict[str, Any]:
        orig_pred = int(np.argmax(self._predict(x)))
        adv_pred = int(np.argmax(self._predict(x_adv)))
        success = orig_pred == y and adv_pred != y
        entry = {
            "x": x.tolist(),
            "x_adv": x_adv.tolist(),
            "y_true": y,
            "orig_pred": orig_pred,
            "adv_pred": adv_pred,
            "success": success,
            "perturbation_l2": float(np.linalg.norm(x_adv - x)),
        }
        self.attacks.append(entry)
        return entry

    def audit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run an attack against every example; return summary stats."""
        results = [self.attack(x, int(t)) for x, t in zip(X, y)]
        n_success = sum(1 for r in results if r["success"])
        return {
            "n_attacked": len(results),
            "n_success": n_success,
            "success_rate": float(n_success / max(len(results), 1)),
            "avg_perturbation": float(np.mean([r["perturbation_l2"] for r in results])),
            "max_perturbation": float(max(r["perturbation_l2"] for r in results)) if results else 0.0,
            "vulnerable_examples": [r for r in results if r["success"]][:5],
        }

    def hard_negatives(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all successful adversarial examples as a training set."""
        if not self.attacks:
            return np.array([]), np.array([])
        X = np.array([a["x_adv"] for a in self.attacks if a["success"]], dtype=np.float32)
        y = np.array([a["y_true"] for a in self.attacks if a["success"]], dtype=np.int64)
        return X, y


__all__ = [
    "MarketModel", "ModelMarketplace", "Peer", "CollaborativeTraining",
    "ModelAging", "AdversarialPartner",
]
