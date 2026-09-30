"""
qai.techniques.tabular_policy -- the bridge between the Technique axis and
the LearningTechnique axis for Reinforcement learning specifically.

Why this needs its own Type: Regression/Classifier/etc. all fit(X, y) on a
static dataset. Reinforcement learning genuinely doesn't -- it learns from
an Environment, step by step (Section 66's real, honest distinction).
TabularPolicy is a real Technique whose fit() takes an Environment instead
of (X, y), making RL a first-class citizen of the SAME Model interface
rather than a separate, disconnected system.
"""
from __future__ import annotations
from .base import Technique
from ..learning.reinforcement import Reinforcement


class TabularPolicy(Technique):
    name = "tabular_policy"
    family = "reinforcement"
    compatible_learning_techniques = ("reinforcement",)  # genuinely incompatible with "supervised" etc.

    def __init__(self, **params):
        super().__init__(**params)
        self.agent = Reinforcement(**params)

    def fit(self, environment, y=None, episodes=500, max_steps=50):
        """Deliberately different signature from every other Type's fit()
        -- 'environment' instead of 'X' -- because that's what's actually
        true, not something to paper over for interface consistency."""
        for _ in range(episodes):
            state = environment.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                decision = self.agent.choose_action(state, environment.ACTIONS)
                outcome = environment.run(decision)
                self.agent.update(state, decision, outcome, environment.ACTIONS)
                state = outcome["next_state"]
                done = outcome["done"]
                steps += 1
            self.agent.episode_count += 1
        self._trained = True
        return self

    def forward(self, state, actions=None):
        if actions is None:
            raise ValueError("TabularPolicy.forward(state, actions=...) requires the action list -- "
                              "unlike other Types, RL policies need to know what's choosable")
        # exploit only, no exploration, for actual deployed predictions
        q_values = [self.agent._get_q(state, a) for a in actions]
        return actions[q_values.index(max(q_values))]

    # --- unique tools, matching the RL-specific properties (Section 105) --
    @property
    def reward(self):
        return self.agent.cumulative_reward

    def penalize(self, amount):
        self.agent.penalize(amount)

    def q_table_summary(self):
        return dict(self.agent.q_table)
