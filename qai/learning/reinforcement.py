"""
qai.learning.reinforcement -- a REAL, working Reinforcement learning
implementation (tabular Q-learning), not a stub. Buildable with pure numpy,
no torch/tensorflow needed -- proves RL genuinely fits the same Model
interface, using the exact (state, decision, outcome) signature from
Section 105's signature table.
"""
from __future__ import annotations
import numpy as np
from .base import LearningTechnique


class Reinforcement(LearningTechnique):
    name = "reinforcement"
    on_step_signature = ("state", "decision", "outcome")

    def __init__(self, gamma=0.95, epsilon=0.2, learning_rate=0.1, **params):
        super().__init__(gamma=gamma, epsilon=epsilon, learning_rate=learning_rate, **params)
        self.gamma = gamma           # future_importance
        self.epsilon = epsilon        # exploration_rate
        self.lr = learning_rate
        self.q_table = {}
        self.cumulative_reward = 0.0
        self.episode_count = 0

    def requires(self) -> str:
        return "an Environment (not a labeled dataset)"

    def _get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, actions: list):
        """epsilon-greedy: explore randomly with probability epsilon,
        otherwise exploit the best known action -- the real mechanism
        the exploration_rate parameter controls."""
        if np.random.random() < self.epsilon:
            return np.random.choice(actions)
        q_values = [self._get_q(state, a) for a in actions]
        return actions[int(np.argmax(q_values))]

    def update(self, state, decision, outcome, actions: list):
        """The actual Q-learning update rule -- Bellman equation, exactly
        as documented in Section 57's formula table:
        Q(s,a) = r + gamma * max Q(s',a')"""
        reward = outcome["reward"]
        next_state = outcome["next_state"]
        done = outcome.get("done", False)

        old_q = self._get_q(state, decision)
        if done:
            target = reward
        else:
            next_max_q = max(self._get_q(next_state, a) for a in actions)
            target = reward + self.gamma * next_max_q

        new_q = old_q + self.lr * (target - old_q)
        self.q_table[(state, decision)] = new_q
        self.cumulative_reward += reward
        return new_q

    def reward(self, amount: float):
        self.cumulative_reward += amount

    def penalize(self, amount: float):
        self.cumulative_reward -= amount
