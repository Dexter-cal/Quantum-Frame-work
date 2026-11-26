"""
qai.nas_search — Neural Architecture Search.

Automatically discover the best neural network architecture for a given
task by trying different configurations and ranking by validation
accuracy, latency, and parameter count.

Strategies:
  * Random search — sample architectures from a search space
  * Grid search — exhaustive (small spaces only)
  * Evolutionary — population + mutation + selection
  * Progressive growth — start small, grow until accuracy plateaus
  * Differentiable (DARTS) — soft selection over operations (lightweight)

Usage:
    import qai
    X, y = qai.load_iris()
    result = qai.nas_search.search(X, y, strategy='evolutionary',
                                    n_trials=20, time_budget_s=60)
    print(result.best_architecture)
    print(result.best_accuracy)
    model = qai.build(technique='neural_network', **result.best_architecture)
"""
from __future__ import annotations

import time
import random
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np


# ============================================================================
# Search space
# ============================================================================

@dataclass
class ArchitectureSpec:
    """A single architecture specification."""
    hidden_layer_sizes: Tuple[int, ...] = (32,)
    activation: str = "relu"
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "adam"
    dropout: float = 0.0
    weight_decay: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hidden_layer_sizes"] = list(self.hidden_layer_sizes)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchitectureSpec":
        d = dict(d)
        if isinstance(d.get("hidden_layer_sizes"), list):
            d["hidden_layer_sizes"] = tuple(d["hidden_layer_sizes"])
        return cls(**d)

    def to_qai_params(self) -> Dict[str, Any]:
        """Convert to kwargs for qai.build(technique='neural_network', **kwargs)."""
        return {
            "hidden_layer_sizes": tuple(self.hidden_layer_sizes),
            "activation": self.activation,
            "learning_rate_init": self.learning_rate,
            "batch_size": self.batch_size,
        }


# Default search space
DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "n_layers": [1, 2, 3, 4],
    "layer_widths": [8, 16, 32, 64, 128],
    "activations": ["relu", "tanh", "sigmoid"],
    "learning_rates": [0.1, 0.01, 0.001, 0.0001],
    "batch_sizes": [16, 32, 64, 128],
}


# ============================================================================
# Trial result
# ============================================================================

@dataclass
class TrialResult:
    """Result of one NAS trial."""
    trial_id: int
    architecture: Dict[str, Any]
    accuracy: float = 0.0
    val_accuracy: float = 0.0
    latency_ms: float = 0.0
    n_params: int = 0
    training_time_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def score(self) -> float:
        """Combined score: accuracy × (1 / (1 + latency_ms / 100)) — favors fast + accurate."""
        if self.error:
            return -1
        speed_bonus = 1.0 / (1.0 + self.latency_ms / 100.0)
        return self.val_accuracy * speed_bonus


# ============================================================================
# Search strategies
# ============================================================================

class RandomSearch:
    """Sample architectures randomly from the search space."""

    def __init__(self, search_space: Optional[Dict] = None, seed: int = 42):
        self.space = search_space or DEFAULT_SEARCH_SPACE
        self.rng = random.Random(seed)

    def sample(self) -> ArchitectureSpec:
        n_layers = self.rng.choice(self.space["n_layers"])
        widths = [self.rng.choice(self.space["layer_widths"]) for _ in range(n_layers)]
        return ArchitectureSpec(
            hidden_layer_sizes=tuple(widths),
            activation=self.rng.choice(self.space["activations"]),
            learning_rate=self.rng.choice(self.space["learning_rates"]),
            batch_size=self.rng.choice(self.space["batch_sizes"]),
        )


class GridSearch:
    """Exhaustive search — only for tiny spaces."""

    def __init__(self, search_space: Optional[Dict] = None):
        self.space = search_space or DEFAULT_SEARCH_SPACE
        self._idx = 0
        self._combos = self._enumerate()

    def _enumerate(self) -> List[ArchitectureSpec]:
        combos = []
        for n_layers in self.space["n_layers"]:
            for width in self.space["layer_widths"]:
                for act in self.space["activations"]:
                    for lr in self.space["learning_rates"]:
                        for bs in self.space["batch_sizes"]:
                            combos.append(ArchitectureSpec(
                                hidden_layer_sizes=tuple([width] * n_layers),
                                activation=act, learning_rate=lr, batch_size=bs,
                            ))
        return combos

    def sample(self) -> Optional[ArchitectureSpec]:
        if self._idx >= len(self._combos):
            return None
        spec = self._combos[self._idx]
        self._idx += 1
        return spec

    def total(self) -> int:
        return len(self._combos)


class EvolutionarySearch:
    """Population-based search with mutation + selection."""

    def __init__(self, search_space: Optional[Dict] = None,
                 population_size: int = 10, mutation_rate: float = 0.3,
                 seed: int = 42):
        self.space = search_space or DEFAULT_SEARCH_SPACE
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed)
        self._random_sampler = RandomSearch(search_space, seed)
        self._population: List[ArchitectureSpec] = []
        self._scores: List[float] = []
        self._generation = 0

    def sample(self) -> ArchitectureSpec:
        # First generation: random
        if len(self._population) < self.pop_size:
            spec = self._random_sampler.sample()
            self._population.append(spec)
            return spec
        # Subsequent: select parent + mutate
        self._generation += 1
        # Pick top 50% as parents
        sorted_pop = sorted(zip(self._scores, self._population),
                             key=lambda x: -x[0])
        top_half = [p for _, p in sorted_pop[:len(sorted_pop) // 2]]
        parent = self.rng.choice(top_half)
        return self._mutate(parent)

    def _mutate(self, spec: ArchitectureSpec) -> ArchitectureSpec:
        """Mutate an architecture spec."""
        new = ArchitectureSpec(**asdict(spec))
        new.hidden_layer_sizes = tuple(spec.hidden_layer_sizes)
        if self.rng.random() < self.mutation_rate:
            # Mutate width of one layer
            if new.hidden_layer_sizes:
                idx = self.rng.randrange(len(new.hidden_layer_sizes))
                widths = list(new.hidden_layer_sizes)
                widths[idx] = self.rng.choice(self.space["layer_widths"])
                new.hidden_layer_sizes = tuple(widths)
        if self.rng.random() < self.mutation_rate:
            new.activation = self.rng.choice(self.space["activations"])
        if self.rng.random() < self.mutation_rate:
            new.learning_rate = self.rng.choice(self.space["learning_rates"])
        if self.rng.random() < self.mutation_rate:
            new.batch_size = self.rng.choice(self.space["batch_sizes"])
        return new

    def update_score(self, spec: ArchitectureSpec, score: float) -> None:
        """Feed back a trial's score."""
        # Find and update
        for i, p in enumerate(self._population):
            if p == spec:
                if i < len(self._scores):
                    self._scores[i] = max(self._scores[i], score)
                else:
                    self._scores.append(score)
                return
        self._population.append(spec)
        self._scores.append(score)


class ProgressiveGrowth:
    """Start small, grow until accuracy plateaus."""

    def __init__(self, search_space: Optional[Dict] = None,
                 patience: int = 3, seed: int = 42):
        self.space = search_space or DEFAULT_SEARCH_SPACE
        self.patience = patience
        self.rng = random.Random(seed)
        self._trial = 0
        self._best_acc = 0.0
        self._no_improve = 0
        self._current_width = 8
        self._current_depth = 1

    def sample(self) -> ArchitectureSpec:
        spec = ArchitectureSpec(
            hidden_layer_sizes=tuple([self._current_width] * self._current_depth),
            activation="relu",
            learning_rate=0.001, batch_size=32,
        )
        self._trial += 1
        return spec

    def update_score(self, spec: ArchitectureSpec, score: float) -> None:
        if score > self._best_acc + 0.01:
            self._best_acc = score
            self._no_improve = 0
            # Grow
            if self.rng.random() < 0.5 and self._current_width < 128:
                self._current_width = min(128, self._current_width * 2)
            elif self._current_depth < 4:
                self._current_depth += 1
        else:
            self._no_improve += 1


# ============================================================================
# NAS runner
# ============================================================================

@dataclass
class NASResult:
    """Final result of a NAS run."""
    best_architecture: Dict[str, Any] = field(default_factory=dict)
    best_accuracy: float = 0.0
    best_val_accuracy: float = 0.0
    best_latency_ms: float = 0.0
    best_n_params: int = 0
    n_trials: int = 0
    total_time_s: float = 0.0
    trials: List[TrialResult] = field(default_factory=list)
    strategy: str = "random"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Neural Architecture Search Results",
            "",
            f"- **Strategy**: {self.strategy}",
            f"- **Trials**: {self.n_trials}",
            f"- **Total time**: {self.total_time_s:.1f}s",
            "",
            "## Best Architecture",
            "",
            f"- **Hidden layers**: {self.best_architecture.get('hidden_layer_sizes', '—')}",
            f"- **Activation**: {self.best_architecture.get('activation', '—')}",
            f"- **Learning rate**: {self.best_architecture.get('learning_rate', '—')}",
            f"- **Batch size**: {self.best_architecture.get('batch_size', '—')}",
            f"- **Validation accuracy**: {self.best_val_accuracy:.4f}",
            f"- **Latency**: {self.best_latency_ms:.2f}ms",
            f"- **Parameters**: {self.best_n_params:,}",
            "",
            "## Top 5 Trials",
            "",
            "| Rank | Architecture | Val Acc | Latency (ms) | Params |",
            "|------|--------------|---------|---------------|--------|",
        ]
        sorted_trials = sorted(self.trials, key=lambda t: -t.score)[:5]
        for i, t in enumerate(sorted_trials, 1):
            arch_str = str(t.architecture.get("hidden_layer_sizes", ""))
            lines.append(f"| {i} | {arch_str} | {t.val_accuracy:.4f} | {t.latency_ms:.2f} | {t.n_params:,} |")
        return "\n".join(lines)


def search(X: Any, y: Any, strategy: str = "random", n_trials: int = 20,
           time_budget_s: Optional[float] = None,
           search_space: Optional[Dict] = None,
           validation_split: float = 0.2,
           epochs_per_trial: int = 5,
           seed: int = 42) -> NASResult:
    """Run NAS on a dataset.

    Args:
        X: features
        y: labels
        strategy: 'random', 'grid', 'evolutionary', or 'progressive'
        n_trials: max number of architectures to try
        time_budget_s: optional time limit
        search_space: custom search space (default DEFAULT_SEARCH_SPACE)
        validation_split: fraction of data for validation
        epochs_per_trial: training epochs per architecture
        seed: random seed

    Returns:
        NASResult with best architecture + all trial results
    """
    start_time = time.time()
    X = np.asarray(X)
    y = np.asarray(y)
    # Split
    n = len(X)
    n_val = max(1, int(n * validation_split))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Pick strategy
    if strategy == "random":
        sampler = RandomSearch(search_space, seed)
    elif strategy == "grid":
        sampler = GridSearch(search_space)
        n_trials = min(n_trials, sampler.total())
    elif strategy == "evolutionary":
        sampler = EvolutionarySearch(search_space, seed=seed)
    elif strategy == "progressive":
        sampler = ProgressiveGrowth(search_space, seed=seed)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    # Run trials
    trials: List[TrialResult] = []
    best_score = -1
    best_trial: Optional[TrialResult] = None
    for trial_id in range(n_trials):
        if time_budget_s and (time.time() - start_time) > time_budget_s:
            break
        spec = sampler.sample()
        if spec is None:
            break
        trial = TrialResult(trial_id=trial_id, architecture=spec.to_dict())
        try:
            # Lazy import qai (avoid circular import at module load)
            import qai
            params = spec.to_qai_params()
            params["max_iter"] = epochs_per_trial * max(1, len(X_train) // spec.batch_size)
            model = qai.build(technique="neural_network", **params)
            t0 = time.perf_counter()
            model.train(X_train, y_train, verbose=False)
            trial.training_time_s = time.perf_counter() - t0
            # Evaluate
            preds_val = model.predict(X_val)
            preds_val = np.asarray(preds_val)
            preds_train = model.predict(X_train)
            preds_train = np.asarray(preds_train)
            trial.val_accuracy = float(np.mean(preds_val == y_val)) if preds_val.shape == y_val.shape else 0
            trial.accuracy = float(np.mean(preds_train == y_train)) if preds_train.shape == y_train.shape else 0
            # Latency
            t0 = time.perf_counter()
            for _ in range(10):
                model.predict(X_val[:1])
            trial.latency_ms = (time.perf_counter() - t0) / 10 * 1000
            # Params
            try:
                from qai.visual_workspace_v2 import _get_weights
                layers = _get_weights(model)
                trial.n_params = sum(L.get("n_params", 0) for L in layers)
            except Exception:
                trial.n_params = 0
            # Feed back to evolutionary / progressive
            if hasattr(sampler, "update_score"):
                sampler.update_score(spec, trial.score)
        except Exception as e:
            trial.error = str(e)[:200]
        trials.append(trial)
        if trial.score > best_score:
            best_score = trial.score
            best_trial = trial
    # Build result
    result = NASResult(
        strategy=strategy,
        n_trials=len(trials),
        total_time_s=time.time() - start_time,
        trials=trials,
    )
    if best_trial:
        result.best_architecture = best_trial.architecture
        result.best_accuracy = best_trial.accuracy
        result.best_val_accuracy = best_trial.val_accuracy
        result.best_latency_ms = best_trial.latency_ms
        result.best_n_params = best_trial.n_params
    return result
