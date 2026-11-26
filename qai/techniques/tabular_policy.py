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
    """Q-learning on a discrete state-action space.

    Args (via qai.build(technique="tabular_policy", **params)):
        gamma: discount factor for future rewards (default 0.95).
        epsilon: exploration rate (default 0.2).
        learning_rate: Q-learning step size (default 0.1).
        actions: optional default action list (or dict keyed by
            state-prefix) used when forward(state) is called without
            explicit actions=. Solves the "TabularPolicy.predict(state)
            requires actions= at every call site" problem for multi-task
            agents.

    Example:
        # Single-action-space agent:
        m = qai.build(technique="tabular_policy", actions=["left", "right"])
        m.predict(state)  # works without passing actions every time

        # Multi-task agent (one Q-table, multiple action spaces):
        m = qai.build(technique="tabular_policy",
                      actions={("maze",): ["up","down","left","right"],
                                ("sequence",): [0,1,2,3,4]})
        m.predict(("maze", 5))  # uses the ("maze",)-prefixed action list
    """
    name = "tabular_policy"
    family = "reinforcement"
    compatible_learning_techniques = ("reinforcement",)  # genuinely incompatible with "supervised" etc.

    def __init__(self, gamma: float = 0.95, epsilon: float = 0.2,
                 learning_rate: float = 0.1, actions=None, **params):
        super().__init__(gamma=gamma, epsilon=epsilon,
                         learning_rate=learning_rate, **params)
        self.agent = Reinforcement(gamma=gamma, epsilon=epsilon,
                                    learning_rate=learning_rate)
        # Optional default action list (or dict keyed by state-prefix tuple).
        # When set, forward(state) doesn't require actions= at every call site.
        self._default_actions = actions

    def fit(self, environment, y=None, episodes=500, max_steps=50):
        """Deliberately different signature from every other Type's fit()
        -- 'environment' instead of 'X' -- because that's what's actually
        true, not something to paper over for interface consistency."""
        for _ in range(episodes):
            state = environment.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                # Pick the action list: env.ACTIONS, or self._default_actions
                actions = self._resolve_actions(state, getattr(environment, "ACTIONS", None))
                decision = self.agent.choose_action(state, actions)
                outcome = environment.run(decision)
                self.agent.update(state, decision, outcome, actions)
                state = outcome["next_state"]
                done = outcome["done"]
                steps += 1
            self.agent.episode_count += 1
        self._trained = True
        return self

    def _resolve_actions(self, state, hint=None):
        """Pick the action list for `state`. Priority:
          1. explicit hint (env.ACTIONS)
          2. dict-keyed default matching state prefix
          3. list default
          4. raise (no actions known)
        """
        if hint is not None:
            return list(hint)
        if self._default_actions is None:
            raise ValueError(
                "[WHAT] TabularPolicy.forward(state) requires the action list\n"
                "[WHY] No `actions=` was passed and the technique was built "
                "without a default action list\n"
                "[FIX] Either pass actions= explicitly, or rebuild with "
                "qai.build(technique='tabular_policy', actions=[...])"
            )
        # Dict-keyed default: match by state prefix
        if isinstance(self._default_actions, dict):
            if isinstance(state, tuple) and len(state) > 0:
                # Try progressively shorter prefixes
                for prefix_len in range(len(state), 0, -1):
                    prefix = state[:prefix_len] if prefix_len > 1 else (state[0],)
                    if prefix in self._default_actions:
                        return list(self._default_actions[prefix])
                    # Single-element prefix as scalar
                    if (state[0],) in self._default_actions:
                        return list(self._default_actions[(state[0],)])
            raise ValueError(
                f"[WHAT] No action list found for state {state!r}\n"
                f"[WHY] The default-actions dict has no matching prefix\n"
                f"[FIX] Add an entry for the state's prefix to the actions dict"
            )
        # Plain list default
        return list(self._default_actions)

    def forward(self, state, actions=None):
        if actions is None:
            actions = self._resolve_actions(state)
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
