"""
PuzzleWorld -- a multi-puzzle environment for the EVOLVE agent.

Unlike GridWorld (one puzzle), PuzzleWorld exposes FOUR distinct puzzle
types, each with its own state encoding and action set. The agent must
learn generalizable strategies across all of them.

Puzzle types:

  1. MAZE     -- a small grid maze with a goal and a trap. Actions:
                 {up, down, left, right}. Reward: +10 at goal,
                 -10 at trap, -0.1 per step.

  2. SEQUENCE -- a "guess the next token" puzzle. The agent sees the
                 last 2 tokens of a sequence and must predict the next.
                 Actions: {0, 1, 2, 3, 4}. Reward: +1 if correct,
                 -0.5 if wrong.

  3. RESOURCE -- a "budgeted harvesting" puzzle. The agent has 10 turns
                 to harvest resources from 3 patches, each patch degrades
                 as it's harvested. Actions: {harvest_0, harvest_1,
                 harvest_2, wait}. Reward: per-harvest yield minus a
                 small per-step cost.

  4. AVOID    -- a "dodge the obstacles" puzzle. The agent moves left/right
                 on a 1-D track while obstacles fall from above. Actions:
                 {left, right, stay}. Reward: +1 per step survived,
                 -5 on collision.

State encoding: each puzzle exposes a TUPLE state (hashable, suitable
for tabular Q-learning). State tuples are prefixed with the puzzle type
so the agent's Q-table can hold all four puzzle types simultaneously
without key collision.

This is a genuinely multi-task environment -- the same agent brain has
to learn four different skills and recognise which puzzle it's currently
in from the state alone. That's a much harder, much more interesting
problem than GridWorld's single-task navigation.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Tuple


# =====================================================================
# Puzzle 1: Maze
# =====================================================================
class MazePuzzle:
    """A 4x4 grid maze.

    Layout (positions 0..15, row-major):
        0  1  2  3
        4  5  6  7      START=5, GOAL=3, TRAP=8
        8  9 10 11
       12 13 14 15

    Walls: (1,2), (5,6), (9,10), (4,8) blocked.
    The agent must navigate around walls to reach the goal while
    avoiding the trap.
    """
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

    def run(self, action: str) -> Dict[str, Any]:
        self.steps += 1
        pos = self.position
        if action == "up" and pos >= 4:
            new_pos = pos - 4
        elif action == "down" and pos < 12:
            new_pos = pos + 4
        elif action == "left" and pos % 4 > 0:
            new_pos = pos - 1
        elif action == "right" and pos % 4 < 3:
            new_pos = pos + 1
        else:
            new_pos = pos  # hit boundary, stay put
        # Wall check
        if (pos, new_pos) in self.WALLS:
            new_pos = pos  # wall blocks
        self.position = new_pos

        if self.position == self.GOAL:
            return {"reward": 10.0, "next_state": self.observe(), "done": True}
        if self.position == self.TRAP:
            return {"reward": -10.0, "next_state": self.observe(), "done": True}
        if self.steps >= self.max_steps:
            return {"reward": -0.5, "next_state": self.observe(), "done": True}
        return {"reward": -0.1, "next_state": self.observe(), "done": False}


# =====================================================================
# Puzzle 2: Sequence
# =====================================================================
class SequencePuzzle:
    """Guess the next token in a periodic sequence.

    The sequence is drawn from one of these patterns (chosen randomly
    on reset):
        0, 1, 2, 3, 4, 0, 1, 2, ...      (ascending, period 5)
        4, 3, 2, 1, 0, 4, 3, 2, ...      (descending, period 5)
        0, 0, 1, 1, 2, 2, 3, 3, ...      (doubled, period 6)
        0, 2, 4, 1, 3, 0, 2, 4, ...      (skip-2 mod 5, period 5)

    The state is (puzzle_type, last_token, second_last_token). The agent
    must learn to predict the next token from this 2-token history.
    """
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
        self.history: List[int] = []

    def reset(self):
        self.pattern = self.PATTERNS[self.rng.integers(0, len(self.PATTERNS))]
        self.idx = self.rng.integers(0, len(self.pattern))
        self.steps = 0
        self.history = [self.pattern[self.idx % len(self.pattern)],
                        self.pattern[(self.idx + 1) % len(self.pattern)]]
        self.idx = (self.idx + 2) % len(self.pattern)
        # State: (type, last, second_last)
        return (self.PUZZLE_TYPE, self.history[-1], self.history[-2])

    def observe(self):
        return (self.PUZZLE_TYPE, self.history[-1], self.history[-2])

    def run(self, action: int) -> Dict[str, Any]:
        self.steps += 1
        correct = self.pattern[self.idx]
        self.history.append(action)
        self.idx = (self.idx + 1) % len(self.pattern)
        reward = 1.0 if action == correct else -0.5
        done = self.steps >= self.max_steps
        # Update history to the CORRECT token (not the agent's guess),
        # so the next state is consistent with the actual pattern.
        self.history[-1] = correct
        return {"reward": reward, "next_state": self.observe(), "done": done}


# =====================================================================
# Puzzle 3: Resource
# =====================================================================
class ResourcePuzzle:
    """Budgeted harvesting puzzle.

    Three resource patches, each starts at yield=4.0 and degrades by 1.0
    per harvest. The agent has 10 turns total. Each harvest gives the
    current yield of that patch (then degrades it). 'wait' recovers all
    patches by +1.0 (max 4.0) but yields nothing this turn.
    """
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
        # State: (type, p0, p1, p2, turn)  -- quantised to int for hashability
        return (self.PUZZLE_TYPE, int(self.patches[0]), int(self.patches[1]),
                int(self.patches[2]), self.turn)

    def observe(self):
        return (self.PUZZLE_TYPE, int(self.patches[0]), int(self.patches[1]),
                int(self.patches[2]), self.turn)

    def run(self, action: str) -> Dict[str, Any]:
        self.turn += 1
        if action == "harvest_0":
            yield_ = self.patches[0]
            self.patches[0] = max(0.0, self.patches[0] - 1.0)
            reward = yield_ - 0.1
        elif action == "harvest_1":
            yield_ = self.patches[1]
            self.patches[1] = max(0.0, self.patches[1] - 1.0)
            reward = yield_ - 0.1
        elif action == "harvest_2":
            yield_ = self.patches[2]
            self.patches[2] = max(0.0, self.patches[2] - 1.0)
            reward = yield_ - 0.1
        elif action == "wait":
            self.patches = [min(self.MAX_YIELD, p + 1.0) for p in self.patches]
            reward = -0.1
        else:
            reward = -1.0
        self.total_harvested += max(0.0, reward)
        done = self.turn >= self.MAX_TURNS
        return {"reward": reward, "next_state": self.observe(), "done": done}


# =====================================================================
# Puzzle 4: Avoid
# =====================================================================
class AvoidPuzzle:
    """Dodge falling obstacles on a 1-D track.

    The agent is at position 0..4 (start=2). Each turn, an obstacle
    falls at a random position. If it hits the agent's current position,
    collision. The agent can move left/right/stay. Reward: +1 per step
    survived, -5 on collision (which ends the episode).
    """
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

    def run(self, action: str) -> Dict[str, Any]:
        self.steps += 1
        if action == "left" and self.pos > 0:
            self.pos -= 1
        elif action == "right" and self.pos < self.WIDTH - 1:
            self.pos += 1
        # 'stay' or invalid: position unchanged

        # Obstacle falls
        collision = (self.pos == self.next_obstacle)
        if collision:
            return {"reward": -5.0, "next_state": self.observe(), "done": True}
        if self.steps >= self.MAX_STEPS:
            return {"reward": 1.0, "next_state": self.observe(), "done": True}
        # survived this step
        self.next_obstacle = int(self.rng.integers(0, self.WIDTH))
        return {"reward": 1.0, "next_state": self.observe(), "done": False}


# =====================================================================
# PuzzleWorld -- the multi-puzzle environment dispatcher
# =====================================================================
class PuzzleWorld:
    """A multi-puzzle environment that randomly selects a puzzle type on
    each episode. The agent's state is prefixed with the puzzle type, so
    a single Q-table can hold all four puzzle types' strategies.

    This is the EVOLVE agent's training and evaluation environment.
    """

    PUZZLE_CLASSES = [MazePuzzle, SequencePuzzle, ResourcePuzzle, AvoidPuzzle]
    PUZZLE_NAMES = ["maze", "sequence", "resource", "avoid"]

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.current_puzzle: Any = None

    def reset(self, puzzle_idx: int = None) -> Tuple:
        """Reset for a new episode. If puzzle_idx is None, pick randomly."""
        if puzzle_idx is None:
            puzzle_idx = int(self.rng.integers(0, len(self.PUZZLE_CLASSES)))
        self.current_puzzle = self.PUZZLE_CLASSES[puzzle_idx]()
        return self.current_puzzle.reset()

    @property
    def ACTIONS(self) -> List[str]:
        if self.current_puzzle is None:
            raise RuntimeError("Call reset() before accessing ACTIONS")
        return self.current_puzzle.ACTIONS

    def observe(self):
        return self.current_puzzle.observe()

    def run(self, action) -> Dict[str, Any]:
        return self.current_puzzle.run(action)

    def current_puzzle_name(self) -> str:
        if self.current_puzzle is None:
            return "none"
        return self.current_puzzle.PUZZLE_TYPE


# =====================================================================
# Self-test
# =====================================================================
if __name__ == "__main__":
    env = PuzzleWorld(seed=0)
    print("=== PuzzleWorld self-test ===")
    for name, idx in [("maze", 0), ("sequence", 1), ("resource", 2), ("avoid", 3)]:
        s = env.reset(puzzle_idx=idx)
        print(f"\n[{name}] initial state: {s}")
        print(f"  actions: {env.ACTIONS}")
        rng = np.random.default_rng(0)
        total_r = 0.0
        steps = 0
        for _ in range(15):
            a = rng.choice(env.ACTIONS)
            out = env.run(a)
            total_r += out["reward"]
            steps += 1
            if out["done"]:
                break
        print(f"  random agent: {steps} steps, total reward = {total_r:.2f}")
    print("\nAll puzzles operational.")
