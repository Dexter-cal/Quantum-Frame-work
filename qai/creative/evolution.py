"""
qai.creative.evolution -- Creative Idea 2: EVOLUTIONARY BREEDING

Treat models as organisms. Maintain a population, evaluate fitness each
generation, crossover the fittest, mutate the offspring, and prune the weak.

Operators:
  - tournament selection
  - arithmetic / uniform crossover of weight vectors
  - gaussian / structural mutation
  - elitism (top-k survive untouched)

Wraps qai models that expose `state_dict()` / `load_state_dict()` so it
works with any technique, not just neural networks.

Usage:
    pop = qai.creative.EvolutionaryBreeder(model_factory, fitness_fn)
    pop.evolve(generations=20)
    best = pop.best()
"""
from __future__ import annotations
import copy
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class Organism:
    """Wrap a model + its fitness + lineage."""

    def __init__(self, model: Any, fitness: float = -math.inf,
                 generation: int = 0, parents: Optional[List[int]] = None):
        self.model = model
        self.fitness = fitness
        self.generation = generation
        self.parents = parents or []
        self.id = random.randint(0, 1 << 60)


def _flatten_state(model: Any) -> Tuple[List[np.ndarray], List[str]]:
    """Return flat list of weight arrays and their attribute paths."""
    arrays = []
    paths = []
    if hasattr(model, "state_dict"):
        try:
            sd = model.state_dict()
            for k, v in sd.items():
                arrays.append(np.asarray(v, dtype=np.float32))
                paths.append(k)
            return arrays, paths
        except Exception:
            pass
    # fallback: walk attributes
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            val = getattr(model, name)
            if isinstance(val, np.ndarray):
                arrays.append(val.astype(np.float32).copy())
                paths.append(name)
        except Exception:
            pass
    return arrays, paths


def _set_state(model: Any, arrays: List[np.ndarray], paths: List[str]) -> None:
    if hasattr(model, "load_state_dict"):
        try:
            sd = {p: arr for p, arr in zip(paths, arrays)}
            model.load_state_dict(sd)
            return
        except Exception:
            pass
    for p, arr in zip(paths, arrays):
        try:
            setattr(model, p, arr)
        except Exception:
            pass


class EvolutionaryBreeder:
    """Population-based model evolution.

    Parameters
    ----------
    model_factory : callable -> Any
        No-args function returning a fresh model instance.
    fitness_fn : callable(model) -> float
        Returns higher=better fitness for a candidate model.
    pop_size : int
        Population size (kept constant each generation).
    elite_frac : float
        Top fraction preserved untouched.
    mutation_rate : float
        Probability each weight is perturbed.
    mutation_scale : float
        Std of gaussian perturbation.
    """

    def __init__(self, model_factory: Callable[[], Any],
                 fitness_fn: Callable[[Any], float],
                 pop_size: int = 8,
                 elite_frac: float = 0.2,
                 mutation_rate: float = 0.1,
                 mutation_scale: float = 0.05,
                 crossover: str = "uniform"):
        self.factory = model_factory
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.n_elite = max(1, int(elite_frac * pop_size))
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.crossover = crossover
        self.population: List[Organism] = []
        self.history: List[Dict[str, Any]] = []
        self._init_population()

    def _init_population(self) -> None:
        for _ in range(self.pop_size):
            self.population.append(Organism(self.factory()))

    def evaluate(self) -> None:
        for org in self.population:
            org.fitness = self.fitness_fn(org.model)

    def _tournament(self, k: int = 3) -> Organism:
        candidates = random.sample(self.population, min(k, len(self.population)))
        return max(candidates, key=lambda o: o.fitness)

    def _crossover(self, parent_a: Organism, parent_b: Organism) -> List[np.ndarray]:
        a_arrays, a_paths = _flatten_state(parent_a.model)
        b_arrays, _ = _flatten_state(parent_b.model)
        if len(a_arrays) != len(b_arrays):
            return a_arrays
        child_arrays = []
        for arr_a, arr_b in zip(a_arrays, b_arrays):
            if arr_a.shape != arr_b.shape:
                child_arrays.append(arr_a.copy())
                continue
            if self.crossover == "uniform":
                mask = np.random.rand(*arr_a.shape) < 0.5
                child = np.where(mask, arr_a, arr_b)
            elif self.crossover == "arithmetic":
                alpha = np.random.uniform(0, 1)
                child = alpha * arr_a + (1 - alpha) * arr_b
            else:
                child = arr_a.copy()
            child_arrays.append(child)
        return child_arrays

    def _mutate(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        out = []
        for arr in arrays:
            mask = np.random.rand(*arr.shape) < self.mutation_rate
            noise = np.random.normal(0, self.mutation_scale, arr.shape)
            arr = arr + mask * noise
            out.append(arr.astype(np.float32))
        return out

    def step(self, generation: int) -> Dict[str, Any]:
        self.evaluate()
        self.population.sort(key=lambda o: -o.fitness)
        elites = self.population[:self.n_elite]
        new_pop: List[Organism] = [Organism(o.model, o.fitness,
                                              generation, [o.id])
                                     for o in elites]
        while len(new_pop) < self.pop_size:
            pa = self._tournament()
            pb = self._tournament()
            child_arrays = self._crossover(pa, pb)
            child_arrays = self._mutate(child_arrays)
            child_model = self.factory()
            _, paths = _flatten_state(child_model)
            _set_state(child_model, child_arrays, paths)
            new_pop.append(Organism(child_model, -math.inf,
                                     generation, [pa.id, pb.id]))
        self.population = new_pop
        best = max(self.population, key=lambda o: o.fitness)
        stats = {
            "generation": generation,
            "best_fitness": best.fitness,
            "mean_fitness": float(np.mean([o.fitness for o in self.population])),
            "worst_fitness": min(o.fitness for o in self.population),
            "best_id": best.id,
        }
        self.history.append(stats)
        return stats

    def evolve(self, generations: int = 10,
               on_gen: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        for g in range(generations):
            stats = self.step(g)
            if on_gen:
                on_gen(stats)
        return self.history[-1] if self.history else {}

    def best(self) -> Organism:
        return max(self.population, key=lambda o: o.fitness)


__all__ = ["EvolutionaryBreeder", "Organism"]
