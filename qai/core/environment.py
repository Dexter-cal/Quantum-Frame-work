"""
qai.core.environment -- a real, minimal Environment (Section 34/75) to
actually test Reinforcement learning against, not a mock. A 1D grid:
positions 0-4, goal at 4 (+10 reward), trap at 2 (-10 reward, episode ends).
Simple enough to verify Q-learning genuinely converges to the correct
policy, and small enough to check the LEARNED table by hand.
"""
from __future__ import annotations


class GridWorld:
    """observe() -> state, run(action) -> outcome, matching the real
    Environment interface the design doc calls for.

    Layout: TRAP(0) - 1 - START(2) - 3 - GOAL(4)
    The agent starts BETWEEN the trap and the goal, so 'right' (toward
    the goal) and 'left' (toward the trap) are a genuine, learnable choice
    -- not a forced path through the trap, which was a real bug in the
    first version of this environment (trap sat directly between start
    and goal on the only path, making it unavoidable no matter what the
    agent learned)."""

    GOAL = 4
    TRAP = 0
    ACTIONS = ["left", "right"]

    def __init__(self, start=2):
        self.start = start
        self.position = start

    def reset(self):
        self.position = self.start
        return self.position

    def observe(self):
        return self.position

    def run(self, action: str) -> dict:
        if action == "right":
            self.position = min(self.position + 1, self.GOAL)
        elif action == "left":
            self.position = max(self.position - 1, 0)

        if self.position == self.GOAL:
            return {"reward": 10.0, "next_state": self.position, "done": True}
        if self.position == self.TRAP:
            return {"reward": -10.0, "next_state": self.position, "done": True}
        return {"reward": -0.1, "next_state": self.position, "done": False}  # small cost per step, encourages efficiency
