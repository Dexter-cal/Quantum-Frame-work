"""
qai.environments -- Standard environment interface + registry.

The spec has `qai.GridWorld` as the only built-in environment, but every
RL agent author re-invents the reset()/step()/observe()/available_actions()
contract. This module provides:

  - Environment: the ABC every environment implements
  - register_environment(name, cls): register a custom environment
  - get_environment(name): retrieve a registered environment class
  - list_environments(): list all registered environments
  - make_environment(name, **kwargs): instantiate a registered environment

Built-in environments:
  - "grid_world":  the classic qai.GridWorld (1-D, trap+goal)
  - "bandit":      a k-armed bandit
  - "puzzle_world": 4-puzzle multi-task benchmark (maze/sequence/resource/avoid)

Usage:
    import qai
    env = qai.make_environment("grid_world")
    state = env.reset()
    while not env.done():
        action = env.available_actions()[0]
        outcome = env.step(action)
        ...

Custom environment:
    class MyEnv(qai.Environment):
        def reset(self): ...
        def step(self, action): ...
        def observe(self): ...
        def available_actions(self): ...
        def done(self): ...

    qai.register_environment("my_env", MyEnv)
    env = qai.make_environment("my_env")
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# =====================================================================
# Environment ABC
# =====================================================================
class Environment(ABC):
    """Abstract base class for all qai environments.

    Every environment must implement:
      - reset() -> state             : start a new episode
      - step(action) -> outcome      : take an action, return outcome dict
      - observe() -> state           : current state (without taking an action)
      - available_actions() -> list  : actions legal in the current state
      - done() -> bool               : whether the episode is over

    The outcome dict from step() should have:
      - reward: float
      - next_state: same type as reset() returns
      - done: bool
      - info: dict (optional, for debugging)
    """
    @abstractmethod
    def reset(self) -> Any:
        """Start a new episode. Returns the initial state."""
        ...

    @abstractmethod
    def step(self, action: Any) -> Dict[str, Any]:
        """Take an action in the current state.

        Returns:
            dict with keys:
              - reward: float
              - next_state: same type as reset() returns
              - done: bool (True if episode is over)
              - info: dict (optional)
        """
        ...

    @abstractmethod
    def observe(self) -> Any:
        """Return the current state without taking an action."""
        ...

    @abstractmethod
    def available_actions(self) -> List[Any]:
        """Return the list of actions legal in the current state."""
        ...

    @abstractmethod
    def done(self) -> bool:
        """Return True if the episode is over."""
        ...

    # Convenience alias matching the older GridWorld API
    def run(self, action: Any) -> Dict[str, Any]:
        """Alias for step() -- matches the legacy GridWorld.run() API."""
        return self.step(action)

    @property
    def ACTIONS(self) -> List[Any]:
        """Legacy attribute alias -- some old code reads env.ACTIONS.
        New code should use available_actions() which can be state-dependent."""
        return self.available_actions()

    def __repr__(self):
        return f"<{type(self).__name__} done={self.done()}>"


# =====================================================================
# Environment registry
# =====================================================================
_REGISTRY: Dict[str, type] = {}


def register_environment(name: str, cls: type) -> None:
    """Register an environment class under `name`.

    Args:
        name: the string users will pass to make_environment().
        cls: a class that subclasses qai.Environment.
    """
    if not issubclass(cls, Environment):
        raise TypeError(
            f"[WHAT] Cannot register {cls.__name__} as an environment\n"
            f"[WHY] It does not subclass qai.Environment\n"
            f"[FIX] Make {cls.__name__} inherit from qai.Environment"
        )
    _REGISTRY[name] = cls


def get_environment(name: str) -> type:
    """Retrieve a registered environment class by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"[WHAT] Unknown environment '{name}'\n"
            f"[WHY] It's not in the environment registry\n"
            f"[FIX] Use one of {sorted(_REGISTRY.keys())} or call "
            f"qai.register_environment('{name}', YourEnvClass) first"
        )
    return _REGISTRY[name]


def list_environments() -> List[str]:
    """List all registered environment names."""
    return sorted(_REGISTRY.keys())


def make_environment(name: str, **kwargs) -> Environment:
    """Instantiate a registered environment.

    Args:
        name: environment name (from list_environments()).
        **kwargs: passed to the environment's __init__.

    Returns:
        An Environment instance.
    """
    cls = get_environment(name)
    return cls(**kwargs)


# =====================================================================
# Built-in environment 1: GridWorldAdapter (wraps the legacy qai.GridWorld)
# =====================================================================
class _GridWorldAdapter(Environment):
    """Adapter that wraps the legacy qai.GridWorld (1-D trap+goal world)
    in the standard Environment interface.

    Don't instantiate directly -- use qai.make_environment("grid_world").
    """
    def __init__(self, start: int = 2):
        # Import here to avoid circular imports
        from .core.environment import GridWorld
        self._env = GridWorld(start=start)
        self._done = False

    def reset(self):
        self._done = False
        return self._env.reset()

    def step(self, action):
        outcome = self._env.run(action)
        self._done = outcome.get("done", False)
        return outcome

    def observe(self):
        return self._env.observe()

    def available_actions(self):
        return list(self._env.ACTIONS)

    def done(self):
        return self._done


# =====================================================================
# Built-in environment 2: BanditEnv (k-armed bandit)
# =====================================================================
class BanditEnv(Environment):
    """A k-armed bandit environment.

    Each arm has a fixed mean reward drawn from N(0, 1) at reset() time.
    Each step() pulls the chosen arm and returns the noisy reward
    (mean + N(0, sigma)).

    The episode never ends (done() always returns False) -- the agent
    just keeps pulling arms. Useful for testing exploration strategies.

    Args:
        k: number of arms (default 10).
        sigma: reward noise std dev (default 1.0).
        seed: random seed for reproducibility.
    """
    def __init__(self, k: int = 10, sigma: float = 1.0, seed: Optional[int] = None):
        self.k = k
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self._means: Optional[np.ndarray] = None
        self._step_count = 0

    def reset(self):
        self._means = self.rng.normal(0, 1, size=self.k)
        self._step_count = 0
        return {"step": 0, "k": self.k}

    def step(self, action: int):
        if self._means is None:
            raise RuntimeError("Call reset() before step()")
        if not (0 <= action < self.k):
            raise ValueError(f"action {action} out of range [0, {self.k})")
        reward = float(self._means[action] + self.rng.normal(0, self.sigma))
        self._step_count += 1
        return {
            "reward": reward,
            "next_state": {"step": self._step_count, "k": self.k},
            "done": False,
            "info": {"optimal_arm": int(np.argmax(self._means)),
                     "optimal_mean": float(self._means.max())},
        }

    def observe(self):
        return {"step": self._step_count, "k": self.k}

    def available_actions(self):
        return list(range(self.k))

    def done(self):
        return False


# =====================================================================
# Built-in environment 3: PuzzleWorld (4-puzzle multi-task benchmark)
# =====================================================================
# The four puzzle classes are inlined directly into qai so that
# `qai.make_environment("puzzle_world")` works on any clean install,
# without needing the showcase/ path on sys.path.

class _MazePuzzle:
    """A 4x4 grid maze. START=5, GOAL=3, TRAP=8. Walls block movement."""
    PUZZLE_TYPE = "maze"
    START = 5
    GOAL = 3
    TRAP = 8
    SIZE = 16
    WALLS = {(1, 2), (2, 1), (5, 6), (6, 5),
             (9, 10), (10, 9), (4, 8), (8, 4)}
    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self):
        self.position = self.START
        self.steps = 0
        self.max_steps = 30

    def reset(self):
        self.position = self.START
        self.steps = 0
        return (self.PUZZLE_TYPE, self.position)

    def observe(self):
        return (self.PUZZLE_TYPE, self.position)

    def run(self, action):
        self.steps += 1
        pos = self.position
        if action == "up" and pos >= 4: new_pos = pos - 4
        elif action == "down" and pos < 12: new_pos = pos + 4
        elif action == "left" and pos % 4 > 0: new_pos = pos - 1
        elif action == "right" and pos % 4 < 3: new_pos = pos + 1
        else: new_pos = pos
        if (pos, new_pos) in self.WALLS:
            new_pos = pos
        self.position = new_pos
        if self.position == self.GOAL:
            return {"reward": 10.0, "next_state": self.observe(), "done": True}
        if self.position == self.TRAP:
            return {"reward": -10.0, "next_state": self.observe(), "done": True}
        if self.steps >= self.max_steps:
            return {"reward": -0.5, "next_state": self.observe(), "done": True}
        return {"reward": -0.1, "next_state": self.observe(), "done": False}


class _SequencePuzzle:
    """Guess the next token in a periodic sequence."""
    PUZZLE_TYPE = "sequence"
    ACTIONS = [0, 1, 2, 3, 4]
    PATTERNS = [
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0],
        [0, 0, 1, 1, 2, 2, 3, 3],
        [0, 2, 4, 1, 3],
    ]

    def __init__(self):
        self.rng = np.random.default_rng()
        self.pattern = self.PATTERNS[0]
        self.idx = 0
        self.steps = 0
        self.max_steps = 20
        self.history = []

    def reset(self):
        self.pattern = self.PATTERNS[int(self.rng.integers(0, len(self.PATTERNS)))]
        self.idx = int(self.rng.integers(0, len(self.pattern)))
        self.steps = 0
        self.history = [self.pattern[self.idx % len(self.pattern)],
                        self.pattern[(self.idx + 1) % len(self.pattern)]]
        self.idx = (self.idx + 2) % len(self.pattern)
        return (self.PUZZLE_TYPE, self.history[-1], self.history[-2])

    def observe(self):
        return (self.PUZZLE_TYPE, self.history[-1], self.history[-2])

    def run(self, action):
        self.steps += 1
        correct = self.pattern[self.idx]
        self.history.append(action)
        self.idx = (self.idx + 1) % len(self.pattern)
        reward = 1.0 if action == correct else -0.5
        done = self.steps >= self.max_steps
        self.history[-1] = correct  # keep state consistent with actual pattern
        return {"reward": reward, "next_state": self.observe(), "done": done}


class _ResourcePuzzle:
    """Budgeted harvesting: 3 patches, 10 turns, harvest or wait."""
    PUZZLE_TYPE = "resource"
    ACTIONS = ["harvest_0", "harvest_1", "harvest_2", "wait"]
    MAX_TURNS = 10
    MAX_YIELD = 4.0

    def __init__(self):
        self.patches = [self.MAX_YIELD, self.MAX_YIELD, self.MAX_YIELD]
        self.turn = 0
        self.total_harvested = 0.0

    def reset(self):
        self.patches = [self.MAX_YIELD, self.MAX_YIELD, self.MAX_YIELD]
        self.turn = 0
        self.total_harvested = 0.0
        return (self.PUZZLE_TYPE, int(self.patches[0]), int(self.patches[1]),
                int(self.patches[2]), self.turn)

    def observe(self):
        return (self.PUZZLE_TYPE, int(self.patches[0]), int(self.patches[1]),
                int(self.patches[2]), self.turn)

    def run(self, action):
        self.turn += 1
        if action == "harvest_0":
            y = self.patches[0]; self.patches[0] = max(0.0, self.patches[0] - 1.0); reward = y - 0.1
        elif action == "harvest_1":
            y = self.patches[1]; self.patches[1] = max(0.0, self.patches[1] - 1.0); reward = y - 0.1
        elif action == "harvest_2":
            y = self.patches[2]; self.patches[2] = max(0.0, self.patches[2] - 1.0); reward = y - 0.1
        elif action == "wait":
            self.patches = [min(self.MAX_YIELD, p + 1.0) for p in self.patches]
            reward = -0.1
        else:
            reward = -1.0
        self.total_harvested += max(0.0, reward)
        done = self.turn >= self.MAX_TURNS
        return {"reward": reward, "next_state": self.observe(), "done": done}


class _AvoidPuzzle:
    """Dodge falling obstacles on a 1-D track."""
    PUZZLE_TYPE = "avoid"
    ACTIONS = ["left", "right", "stay"]
    WIDTH = 5
    START = 2
    MAX_STEPS = 20

    def __init__(self):
        self.rng = np.random.default_rng()
        self.pos = self.START
        self.steps = 0
        self.next_obstacle = int(self.rng.integers(0, self.WIDTH))

    def reset(self):
        self.pos = self.START
        self.steps = 0
        self.next_obstacle = int(self.rng.integers(0, self.WIDTH))
        return (self.PUZZLE_TYPE, self.pos, self.next_obstacle)

    def observe(self):
        return (self.PUZZLE_TYPE, self.pos, self.next_obstacle)

    def run(self, action):
        self.steps += 1
        if action == "left" and self.pos > 0: self.pos -= 1
        elif action == "right" and self.pos < self.WIDTH - 1: self.pos += 1
        collision = (self.pos == self.next_obstacle)
        if collision:
            return {"reward": -5.0, "next_state": self.observe(), "done": True}
        if self.steps >= self.MAX_STEPS:
            return {"reward": 1.0, "next_state": self.observe(), "done": True}
        self.next_obstacle = int(self.rng.integers(0, self.WIDTH))
        return {"reward": 1.0, "next_state": self.observe(), "done": False}


class PuzzleWorld(Environment):
    """A multi-puzzle environment that randomly selects a puzzle type on
    each episode. The agent's state is prefixed with the puzzle type, so
    a single Q-table can hold all four puzzle types' strategies.

    This is the EVOLVE agent's training and evaluation environment. It's
    a 4-puzzle multi-task benchmark (maze/sequence/resource/avoid) that
    exercises Q-learning's ability to learn generalizable strategies.

    Promoted into qai proper so `qai.make_environment("puzzle_world")`
    works on any clean install.
    """
    PUZZLE_CLASSES = [_MazePuzzle, _SequencePuzzle, _ResourcePuzzle, _AvoidPuzzle]
    PUZZLE_NAMES = ["maze", "sequence", "resource", "avoid"]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.current_puzzle = None
        self._done = False

    def reset(self, puzzle_idx: Optional[int] = None):
        self._done = False
        if puzzle_idx is None:
            puzzle_idx = int(self.rng.integers(0, len(self.PUZZLE_CLASSES)))
        self.current_puzzle = self.PUZZLE_CLASSES[puzzle_idx]()
        return self.current_puzzle.reset()

    def step(self, action):
        outcome = self.current_puzzle.run(action)
        self._done = outcome.get("done", False)
        return outcome

    def observe(self):
        return self.current_puzzle.observe()

    def available_actions(self):
        if self.current_puzzle is None:
            raise RuntimeError("Call reset() before available_actions()")
        return list(self.current_puzzle.ACTIONS)

    def done(self):
        return self._done

    def current_puzzle_name(self):
        if self.current_puzzle is None: return "none"
        return self.current_puzzle.PUZZLE_TYPE

    # Back-compat: EVOLVE's launcher calls env.run(action) directly
    def run(self, action):
        return self.step(action)

    @property
    def ACTIONS(self):
        return self.available_actions()


# =====================================================================
# Register the built-in environments
# =====================================================================
register_environment("grid_world", _GridWorldAdapter)
register_environment("bandit", BanditEnv)
register_environment("puzzle_world", PuzzleWorld)


# =====================================================================
# Public API
# =====================================================================
__all__ = [
    "Environment",
    "register_environment",
    "get_environment",
    "list_environments",
    "make_environment",
    "BanditEnv",
    "PuzzleWorld",
]
