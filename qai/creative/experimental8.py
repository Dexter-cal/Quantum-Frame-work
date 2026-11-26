"""
qai.creative.experimental8 — 10 more original creative ideas (part 8).

  Idea 86 — ModelOrganism      : model as a living organism with metabolism
  Idea 87 — NeuralArchitecture : buildings/cities designed by neural nets
  Idea 88 — DataCemetary       : where old datasets go to rest + be resurrected
  Idea 89 — ModelStockMarket   : models as tradable stocks with live prices
  Idea 90 — WeightArcheology   : excavate layers of a model like geological strata
  Idea 91 — NeuralSymphony     : orchestrate multiple models like instruments
  Idea 92 — ModelWeather       : forecast model performance like weather
  Idea 93 — DataRefrigerator   : cold storage for rarely-used data + fast thaw
  Idea 94 — ModelCourt         : adversarial legal system for model validation
  Idea 95 — NeuralGarden       : grow models from seeds with pruning + grafting
"""
from __future__ import annotations
import math, random, time, hashlib
from collections import defaultdict, OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np


class ModelOrganism:
    """Model as a living organism with metabolism, growth, and death.

    - Metabolism: consumes compute (food) + data (water) to stay alive
    - Growth: gains "mass" (params) over time via training
    - Reproduction: can spawn child models
    - Death: dies if starved of compute or data for too long
    - Health: combination of accuracy + freshness + resource balance
    """
    def __init__(self, model: Any, name: str = "organism"):
        self.model = model
        self.name = name
        self.energy: float = 100.0  # starts fully energized
        self.health: float = 1.0
        self.age: int = 0
        self.mass: int = 0  # parameter count
        self.alive: bool = True
        self.children: List[str] = []
        self.last_fed: float = time.time()
        self._history: List[Dict[str, Any]] = []

    def feed(self, data: np.ndarray, compute: float = 1.0) -> Dict[str, Any]:
        """Feed the organism data + compute."""
        if not self.alive:
            return {"action": "dead", "reason": "organism is dead"}
        self.energy = min(100, self.energy + compute * 10)
        self.last_fed = time.time()
        try:
            if hasattr(self.model, "fit"):
                X = data
                y = data.sum(axis=1) if data.ndim > 1 else data
                self.model.fit(X, y)
                self.mass = sum(arr.size for arr in self._get_weights().values())
                self.age += 1
                self._history.append({"action": "feed", "energy": self.energy, "age": self.age})
                return {"action": "fed", "energy": self.energy, "mass": self.mass, "age": self.age}
        except Exception:
            pass
        return {"action": "feed_failed", "energy": self.energy}

    def metabolize(self) -> Dict[str, Any]:
        """Consume energy over time (called periodically)."""
        if not self.alive:
            return {"alive": False}
        elapsed = time.time() - self.last_fed
        self.energy = max(0, self.energy - elapsed * 0.1)
        if self.energy < 10:
            self.health = max(0, self.health - 0.1)
        if self.energy <= 0 or self.health <= 0:
            self.alive = False
            self._history.append({"action": "death", "cause": "starvation" if self.energy <= 0 else "poor_health"})
        return {"alive": self.alive, "energy": self.energy, "health": self.health, "age": self.age}

    def reproduce(self, partner: Optional["ModelOrganism"] = None) -> Optional["ModelOrganism"]:
        """Spawn a child organism."""
        if not self.alive or self.energy < 50:
            return None
        self.energy -= 30
        import copy
        child_model = copy.deepcopy(self.model)
        child = ModelOrganism(child_model, f"{self.name}_child_{len(self.children)}")
        child.energy = 50
        self.children.append(child.name)
        self._history.append({"action": "reproduce", "child": child.name})
        return child

    def _get_weights(self) -> Dict[str, np.ndarray]:
        if hasattr(self.model, "state_dict"):
            try: return {k: np.asarray(v) for k, v in self.model.state_dict().items()}
            except: pass
        if hasattr(self.model, "technique") and hasattr(self.model.technique, "state_dict"):
            try: return {k: np.asarray(v) for k, v in self.model.technique.state_dict().items()}
            except: pass
        out = {}
        for name in dir(self.model):
            if not name.startswith("_"):
                try:
                    val = getattr(self.model, name)
                    if isinstance(val, np.ndarray) and val.size > 0: out[name] = val
                except: pass
        return out

    def status(self) -> Dict[str, Any]:
        return {"name": self.name, "alive": self.alive, "energy": self.energy,
                "health": self.health, "age": self.age, "mass": self.mass,
                "children": self.children, "history": self._history[-5:]}


class ModelStockMarket:
    """Models as tradable stocks with live prices based on performance.

    Each model has a ticker, price (based on accuracy), and volume (usage).
    Traders buy/sell model shares; price adjusts based on demand.
    """
    def __init__(self):
        self.stocks: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []

    def list_model(self, ticker: str, model: Any, initial_price: float = 100.0,
                    accuracy: float = 0.5) -> None:
        self.stocks[ticker] = {
            "model": model, "price": initial_price, "accuracy": accuracy,
            "volume": 0, "holders": {}, "history": [initial_price],
            "listed_at": time.time(),
        }

    def buy(self, ticker: str, trader: str, shares: int = 1) -> Dict[str, Any]:
        if ticker not in self.stocks:
            raise KeyError(f"unknown ticker: {ticker}")
        s = self.stocks[ticker]
        cost = s["price"] * shares
        s["holders"][trader] = s["holders"].get(trader, 0) + shares
        s["volume"] += shares
        s["price"] *= 1.01  # price goes up with demand
        s["history"].append(s["price"])
        self.trades.append({"type": "buy", "ticker": ticker, "trader": trader,
                              "shares": shares, "price": cost, "timestamp": time.time()})
        return {"ticker": ticker, "shares": shares, "cost": cost, "new_price": s["price"]}

    def sell(self, ticker: str, trader: str, shares: int = 1) -> Dict[str, Any]:
        if ticker not in self.stocks:
            raise KeyError(f"unknown ticker: {ticker}")
        s = self.stocks[ticker]
        if s["holders"].get(trader, 0) < shares:
            return {"error": "insufficient shares"}
        s["holders"][trader] -= shares
        s["price"] *= 0.99  # price goes down with selling
        s["history"].append(s["price"])
        self.trades.append({"type": "sell", "ticker": ticker, "trader": trader,
                              "shares": shares, "price": s["price"] * shares, "timestamp": time.time()})
        return {"ticker": ticker, "shares": shares, "proceeds": s["price"] * shares, "new_price": s["price"]}

    def update_accuracy(self, ticker: str, accuracy: float) -> None:
        """Update model accuracy → affects price."""
        if ticker in self.stocks:
            s = self.stocks[ticker]
            s["accuracy"] = accuracy
            s["price"] = accuracy * 200  # price = accuracy * 200

    def market_summary(self) -> Dict[str, Any]:
        return {
            "n_stocks": len(self.stocks),
            "tickers": {t: {"price": s["price"], "accuracy": s["accuracy"],
                            "volume": s["volume"]} for t, s in self.stocks.items()},
            "n_trades": len(self.trades),
            "top_performer": max(self.stocks.items(), key=lambda x: x[1]["price"])[0] if self.stocks else None,
        }


class WeightArcheology:
    """Excavate layers of a model like geological strata.

    Each layer is a "stratum" with:
      - Depth (layer index)
      - Age (when it was trained)
      - Composition (weight statistics)
      - Fossils (dead neurons, important features)
    """
    @staticmethod
    def excavate(model: Any) -> Dict[str, Any]:
        from ..weights import _extract_weights
        weights = _extract_weights(model)
        strata = []
        for depth, (name, arr) in enumerate(sorted(weights.items())):
            strata.append({
                "depth": depth, "layer": name,
                "shape": list(arr.shape),
                "n_params": int(arr.size),
                "mean": float(arr.mean()), "std": float(arr.std()),
                "min": float(arr.min()), "max": float(arr.max()),
                "sparsity": float((arr == 0).mean()),
                "fossils": int((np.abs(arr) < 1e-8).sum()),  # dead weights
                "gems": int((np.abs(arr) > np.percentile(np.abs(arr), 99)).sum()),  # top 1%
            })
        return {
            "n_strata": len(strata),
            "total_params": sum(s["n_params"] for s in strata),
            "strata": strata,
            "deepest_layer": max(strata, key=lambda s: s["depth"]) if strata else None,
            "densest_layer": max(strata, key=lambda s: s["n_params"]) if strata else None,
            "sparsest_layer": max(strata, key=lambda s: s["sparsity"]) if strata else None,
        }

    @staticmethod
    def carbon_date(model: Any) -> Dict[str, Any]:
        """Estimate the "age" of each layer based on weight patterns."""
        from ..weights import _extract_weights
        weights = _extract_weights(model)
        dates = []
        for name, arr in weights.items():
            # "old" layers tend to have more converged (low variance) weights
            convergence = 1.0 / (arr.std() + 1e-8)
            dates.append({"layer": name, "estimated_age": "old" if convergence > 10 else "new",
                          "convergence_score": float(convergence)})
        return {"layers": dates, "n_layers": len(dates)}


class NeuralSymphony:
    """Orchestrate multiple models like instruments in a symphony.

    Each model is an "instrument" with a role:
      - Strings (primary model): carries the main melody
      - Woodwinds (auxiliary model): fills in details
      - Brass (ensemble model): adds power
      - Percussion (fast model): provides rhythm (quick predictions)

    The conductor decides which instruments play for each input.
    """
    INSTRUMENTS = ["strings", "woodwinds", "brass", "percussion"]

    def __init__(self):
        self.orchestra: Dict[str, Any] = {}
        self.conductor_log: List[Dict[str, Any]] = []

    def add_instrument(self, instrument: str, model: Any) -> None:
        if instrument not in self.INSTRUMENTS:
            raise ValueError(f"unknown instrument: {instrument}. Choose from {self.INSTRUMENTS}")
        self.orchestra[instrument] = model

    def perform(self, x: Any) -> Dict[str, Any]:
        """Conduct the orchestra to produce a prediction."""
        results = {}
        for instrument, model in self.orchestra.items():
            try:
                results[instrument] = float(model.forward(x))
            except Exception as e:
                results[instrument] = None
        # conductor: weighted average
        weights = {"strings": 0.4, "woodwinds": 0.3, "brass": 0.2, "percussion": 0.1}
        total_weight = sum(weights.get(k, 0) for k in results if results[k] is not None)
        if total_weight > 0:
            harmony = sum(weights.get(k, 0) * results[k] for k in results if results[k] is not None) / total_weight
        else:
            harmony = None
        self.conductor_log.append({"inputs": results, "harmony": harmony, "timestamp": time.time()})
        return {"harmony": harmony, "instruments": results}

    def performance_report(self) -> Dict[str, Any]:
        if not self.conductor_log:
            return {"n_performances": 0}
        harmonies = [l["harmony"] for l in self.conductor_log if l["harmony"] is not None]
        return {
            "n_performances": len(self.conductor_log),
            "avg_harmony": float(np.mean(harmonies)) if harmonies else None,
            "instruments_active": list(self.orchestra.keys()),
        }


class ModelWeather:
    """Forecast model performance like weather.

    Predicts:
      - "Sunny" (high accuracy, stable)
      - "Cloudy" (mediocre accuracy, may improve)
      - "Rainy" (accuracy dropping, needs retraining)
      - "Stormy" (accuracy crashing, emergency retrain)
      - "Foggy" (uncertain, needs more data)

    Based on: recent accuracy trend, data drift, model age, error rate.
    """
    CONDITIONS = {
        "sunny": "☀️ High accuracy, stable performance",
        "cloudy": "☁️ Mediocre accuracy, may improve with more data",
        "rainy": "🌧️ Accuracy dropping — consider retraining",
        "stormy": "⛈️ Accuracy crashing — emergency retrain needed!",
        "foggy": "🌫️ Uncertain — insufficient data to assess",
    }

    def __init__(self):
        self.forecasts: List[Dict[str, Any]] = []

    def forecast(self, accuracy_history: List[float],
                  drift_score: float = 0.0,
                  model_age_days: int = 0,
                  error_rate: float = 0.0) -> Dict[str, Any]:
        """Generate a weather forecast for a model."""
        if len(accuracy_history) < 3:
            condition = "foggy"
        else:
            recent = accuracy_history[-5:]
            trend = recent[-1] - recent[0]
            current = recent[-1]
            if current > 0.9 and trend >= 0:
                condition = "sunny"
            elif current > 0.7 and trend >= -0.05:
                condition = "cloudy"
            elif current > 0.5 or trend < -0.05:
                condition = "rainy"
            elif current <= 0.5 or trend < -0.1:
                condition = "stormy"
            else:
                condition = "foggy"
        # additional factors
        if drift_score > 0.3 and condition in ("sunny", "cloudy"):
            condition = "rainy"
        if model_age_days > 90 and condition == "sunny":
            condition = "cloudy"
        if error_rate > 0.1:
            condition = "stormy"
        forecast = {
            "condition": condition,
            "description": self.CONDITIONS[condition],
            "current_accuracy": accuracy_history[-1] if accuracy_history else None,
            "trend": trend if len(accuracy_history) >= 2 else None,
            "drift_score": drift_score,
            "model_age_days": model_age_days,
            "error_rate": error_rate,
            "recommendation": self._recommend(condition),
            "timestamp": time.time(),
        }
        self.forecasts.append(forecast)
        return forecast

    @staticmethod
    def _recommend(condition: str) -> str:
        return {
            "sunny": "No action needed. Model is performing well.",
            "cloudy": "Monitor closely. Consider collecting more training data.",
            "rainy": "Schedule retraining within the next week.",
            "stormy": "Immediate retraining required. Check for data drift or corruption.",
            "foggy": "Collect more evaluation data to assess model health.",
        }[condition]

    def weekly_report(self) -> Dict[str, Any]:
        if not self.forecasts:
            return {"n_forecasts": 0}
        conditions = [f["condition"] for f in self.forecasts]
        return {
            "n_forecasts": len(self.forecasts),
            "latest_condition": conditions[-1],
            "condition_distribution": dict(zip(*np.unique(conditions, return_counts=True))),
            "all_forecasts": self.forecasts[-7:],
        }


class ModelCourt:
    """Adversarial legal system for model validation.

    Roles:
      - Prosecutor: tries to find inputs that break the model
      - Defense: argues the model is robust
      - Judge: evaluates evidence and renders verdict
      - Jury: panel of evaluator models that vote

    Verdicts: "guilty" (model is unreliable) or "not guilty" (model is robust)
    """
    def __init__(self, model: Any, prosecutor: Optional[Callable] = None,
                 judge: Optional[Callable] = None):
        self.model = model
        self.prosecutor = prosecutor
        self.judge = judge
        self.cases: List[Dict[str, Any]] = []

    def prosecute(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                    n_attacks: int = 10) -> Dict[str, Any]:
        """The prosecutor tries to find failures."""
        evidence = []
        for i in range(min(n_attacks, len(X))):
            x = X[i]
            # try adversarial perturbation
            for eps in [0.01, 0.1, 0.5]:
                x_adv = x + np.random.normal(0, eps, x.shape).astype(np.float32)
                try:
                    pred = self.model.forward(x_adv)
                    orig = self.model.forward(x)
                    if pred != orig:
                        evidence.append({"type": "adversarial", "eps": eps,
                                          "original": float(orig), "perturbed": float(pred),
                                          "difference": abs(float(orig) - float(pred))})
                except Exception as e:
                    evidence.append({"type": "crash", "eps": eps, "error": str(e)})
        # defense: argue model is robust on clean inputs
        clean_correct = 0
        for i in range(min(len(X), 20)):
            try:
                pred = self.model.forward(X[i])
                if y is not None and pred == y[i]:
                    clean_correct += 1
            except: pass
        defense_evidence = {"clean_accuracy": clean_correct / max(min(len(X), 20), 1)}
        # verdict
        n_failures = len([e for e in evidence if e["type"] in ("adversarial", "crash")])
        verdict = "guilty" if n_failures > len(evidence) * 0.3 else "not_guilty"
        case = {
            "n_attacks": n_attacks, "evidence": evidence,
            "defense": defense_evidence, "verdict": verdict,
            "n_failures": n_failures, "timestamp": time.time(),
        }
        self.cases.append(case)
        return case

    def case_history(self) -> List[Dict[str, Any]]:
        return list(self.cases)

    def verdict_summary(self) -> Dict[str, Any]:
        if not self.cases:
            return {"n_cases": 0}
        verdicts = [c["verdict"] for c in self.cases]
        return {
            "n_cases": len(self.cases),
            "guilty": verdicts.count("guilty"),
            "not_guilty": verdicts.count("not_guilty"),
            "guilty_rate": verdicts.count("guilty") / len(verdicts),
            "latest_verdict": verdicts[-1],
        }


class DataRefrigerator:
    """Cold storage for rarely-used data + fast thaw when needed.

    - Freezes data to compressed format (like a refrigerator)
    - Thaws on demand (decompresses + loads into RAM)
    - Tracks "temperature" (access frequency)
    - Auto-freezes data that hasn't been accessed in N days
    - Auto-thaws data that's frequently accessed
    """
    def __init__(self, storage_dir: str = "/tmp/qai_fridge"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._items: Dict[str, Dict[str, Any]] = {}
        self._temperature: Dict[str, float] = {}  # 0=frozen, 1=hot

    def store(self, name: str, data: Any, freeze: bool = True) -> Dict[str, Any]:
        """Store data in the refrigerator."""
        import pickle, gzip
        path = os.path.join(self.storage_dir, f"{name}.frozen.gz")
        with gzip.open(path, "wb") as f:
            pickle.dump(data, f)
        size = os.path.getsize(path)
        self._items[name] = {"path": path, "size_bytes": size, "frozen": freeze,
                              "stored_at": time.time(), "n_accesses": 0, "last_access": time.time()}
        self._temperature[name] = 0.0 if freeze else 1.0
        return {"stored": True, "name": name, "size_bytes": size, "frozen": freeze}

    def thaw(self, name: str) -> Any:
        """Thaw frozen data (decompress + load)."""
        if name not in self._items:
            raise KeyError(f"item '{name}' not in refrigerator")
        import pickle, gzip
        item = self._items[name]
        with gzip.open(item["path"], "rb") as f:
            data = pickle.load(f)
        item["n_accesses"] += 1
        item["last_access"] = time.time()
        self._temperature[name] = min(1.0, self._temperature.get(name, 0) + 0.2)
        return data

    def freeze(self, name: str) -> bool:
        """Force-freeze an item."""
        if name in self._temperature:
            self._temperature[name] = 0.0
            if name in self._items:
                self._items[name]["frozen"] = True
            return True
        return False

    def auto_manage(self, threshold_days: float = 7) -> Dict[str, Any]:
        """Auto-freeze items not accessed in threshold_days."""
        now = time.time()
        frozen = []
        for name, item in self._items.items():
            days_since = (now - item["last_access"]) / 86400
            if days_since > threshold_days and self._temperature.get(name, 0) > 0:
                self._temperature[name] = 0.0
                item["frozen"] = True
                frozen.append(name)
        return {"auto_frozen": frozen, "n_items": len(self._items)}

    def inventory(self) -> Dict[str, Any]:
        return {
            "n_items": len(self._items),
            "total_size_bytes": sum(i["size_bytes"] for i in self._items.values()),
            "items": {name: {"size_bytes": i["size_bytes"], "frozen": i["frozen"],
                              "n_accesses": i["n_accesses"], "temperature": self._temperature.get(name, 0),
                              "age_days": (time.time() - i["stored_at"]) / 86400}
                       for name, i in self._items.items()},
        }


import os  # for DataRefrigerator


__all__ = [
    "ModelOrganism", "ModelStockMarket", "WeightArcheology",
    "NeuralSymphony", "ModelWeather", "ModelCourt", "DataRefrigerator",
]
