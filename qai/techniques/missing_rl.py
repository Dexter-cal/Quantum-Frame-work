"""
qai.techniques.missing_rl -- RL techniques we missed.

  - SARSA     : State-Action-Reward-State-Action (on-policy TD control)
  - MCTS      : Monte Carlo Tree Search (AlphaGo's planning algorithm)
  - MuZero    : model-based RL with a learned dynamics model
"""
from __future__ import annotations
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .base import Technique


# ============================================================================
# 1. SARSA — on-policy temporal-difference control
# ============================================================================
class SARSA(Technique):
    """SARSA — on-policy TD control.

    Updates Q(s,a) toward r + gamma * Q(s', a') where a' is the action
    actually taken in s' (not the greedy action like Q-learning).
    """
    name = "sarsa"
    family = "reinforcement"

    def __init__(self, n_states: int = 16, n_actions: int = 4,
                 alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.1, **kw):
        super().__init__(n_states=n_states, n_actions=n_actions,
                         alpha=alpha, gamma=gamma, epsilon=epsilon, **kw)
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(self.Q[state].argmax())

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int) -> None:
        td_target = r + self.gamma * self.Q[s_next, a_next]
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def fit(self, env, y=None):
        """Train via SARSA. env must have reset() -> state, step(action) -> (next_state, reward, done, _)."""
        for episode in range(100):
            s = env.reset()
            a = self.select_action(s)
            for t in range(200):
                s_next, r, done, _ = env.step(a)
                a_next = self.select_action(s_next)
                self.update(s, a, r, s_next, a_next)
                s, a = s_next, a_next
                if done:
                    break
        self._trained = True
        return self

    def forward(self, s):
        return int(self.Q[s].argmax())


# ============================================================================
# 2. MCTS — Monte Carlo Tree Search
# ============================================================================
class _MCTSNode:
    """A node in the MCTS search tree."""
    def __init__(self, state: Any, parent: Optional["_MCTSNode"] = None,
                 action: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[_MCTSNode] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.untried_actions: List[int] = []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c: float = 1.41) -> "_MCTSNode":
        """Select child with highest UCB1 score."""
        scores = []
        for child in self.children:
            if child.visits == 0:
                scores.append(float("inf"))
            else:
                exploit = child.value / child.visits
                explore = c * math.sqrt(math.log(self.visits) / child.visits)
                scores.append(exploit + explore)
        return self.children[int(np.argmax(scores))]

    def expand(self, action: int, state: Any) -> "_MCTSNode":
        child = _MCTSNode(state, parent=self, action=action)
        self.children.append(child)
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return child


class MCTS(Technique):
    """MCTS — Monte Carlo Tree Search.

    Builds a search tree via 4 phases:
      1. Selection: traverse via UCB1 until reaching an unexpanded node
      2. Expansion: add a new child
      3. Simulation: random rollout from new node
      4. Backpropagation: update visit counts + values up the tree

    Used in AlphaGo, AlphaZero, MuZero.
    """
    name = "mcts"
    family = "planning"

    def __init__(self, n_actions: int = 4, n_simulations: int = 100,
                 c: float = 1.41, max_depth: int = 10, **kw):
        super().__init__(n_actions=n_actions, n_simulations=n_simulations,
                         c=c, max_depth=max_depth, **kw)
        self.n_actions = n_actions
        self.n_simulations = n_simulations
        self.c = c
        self.max_depth = max_depth

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c)
        return node

    def _expand(self, node: _MCTSNode, env) -> _MCTSNode:
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop(random.randint(0, len(node.untried_actions) - 1))
        # apply action to get new state
        state = node.state
        return node.expand(action, state)

    def _simulate(self, node: _MCTSNode, env, get_valid_actions, take_action) -> float:
        """Random rollout from node; return final reward."""
        state = node.state
        for _ in range(self.max_depth):
            actions = get_valid_actions(state)
            if not actions:
                break
            action = random.choice(actions)
            state, reward, done = take_action(state, action)
            if done:
                return reward
        return 0.0  # no terminal reached

    def _backprop(self, node: _MCTSNode, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def search(self, root_state: Any, env=None,
                get_valid_actions: Callable[[Any], List[int]] = None,
                take_action: Callable[[Any, int], Tuple[Any, float, bool]] = None,
                is_terminal: Optional[Callable[[Any], bool]] = None) -> int:
        """Run MCTS from root_state; return best action."""
        if get_valid_actions is None or take_action is None:
            raise ValueError("must provide get_valid_actions and take_action callables")
        root = _MCTSNode(root_state)
        root.untried_actions = list(get_valid_actions(root_state))
        for _ in range(self.n_simulations):
            # 1. Selection
            node = self._select(root)
            # 2. Expansion
            if node.untried_actions and not (is_terminal and is_terminal(node.state)):
                node = self._expand(node, env)
                node.untried_actions = list(get_valid_actions(node.state))
            # 3. Simulation
            reward = self._simulate(node, env, get_valid_actions, take_action)
            # 4. Backpropagation
            self._backprop(node, reward)
        # pick most-visited child
        if not root.children:
            # fallback: random action
            return random.choice(get_valid_actions(root_state))
        best = max(root.children, key=lambda c: c.visits)
        return best.action

    def fit(self, env, y=None):
        """MCTS doesn't 'train' in the traditional sense; just mark as trained."""
        self._trained = True
        return self

    def forward(self, x):
        """x should be a tuple (state, get_valid_actions, take_action)."""
        if isinstance(x, tuple) and len(x) == 3:
            state, gva, ta = x
            return self.search(state, get_valid_actions=gva, take_action=ta)
        return 0


from typing import Callable  # for MCTS


# ============================================================================
# 3. MuZero — model-based RL with learned dynamics (simplified)
# ============================================================================
class MuZero(Technique):
    """MuZero (simplified) — learns a dynamics model + plans via MCTS.

    MuZero learns three functions:
      - representation: h(observation) -> hidden state
      - dynamics: g(hidden, action) -> next hidden + reward
      - prediction: f(hidden) -> policy + value

    Then plans via MCTS in the learned latent space.

    This is a simplified implementation for educational use.
    """
    name = "muzero"
    family = "model_based_rl"

    def __init__(self, obs_dim: int = 4, hidden_dim: int = 16,
                 n_actions: int = 4,
                 n_simulations: int = 50, lr: float = 1e-3, **kw):
        super().__init__(obs_dim=obs_dim, hidden_dim=hidden_dim,
                         n_actions=n_actions, n_simulations=n_simulations,
                         lr=lr, **kw)
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("MuZero requires PyTorch")
        # NOTE: do NOT store `torch` / `nn` as instance attributes — they are
        # modules and cannot be pickled (breaks qai.save_model). Import them
        # locally in any method that needs them instead.
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.n_simulations = n_simulations
        # representation function
        self.repr_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # dynamics function: (hidden, action_onehot) -> (next_hidden, reward)
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_dim + n_actions, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reward_net = nn.Linear(hidden_dim, 1)
        # prediction function: hidden -> (policy_logits, value)
        self.policy_net = nn.Linear(hidden_dim, n_actions)
        self.value_net = nn.Linear(hidden_dim, 1)
        self.opt = torch.optim.Adam([
            *self.repr_net.parameters(),
            *self.dynamics_net.parameters(),
            *self.reward_net.parameters(),
            *self.policy_net.parameters(),
            *self.value_net.parameters(),
        ], lr=lr)

    def represent(self, obs):
        """h(observation) -> hidden state."""
        return self.repr_net(obs)

    def predict(self, hidden):
        """f(hidden) -> (policy_logits, value)."""
        policy = self.policy_net(hidden)
        value = self.value_net(hidden)
        return policy, value

    def dynamics(self, hidden, action):
        """g(hidden, action) -> (next_hidden, reward)."""
        action_onehot = self.torch.zeros(hidden.shape[0], self.n_actions)
        action_onehot[:, action] = 1.0
        inp = self.torch.cat([hidden, action_onehot], dim=-1)
        next_hidden = self.dynamics_net(inp)
        reward = self.reward_net(next_hidden)
        return next_hidden, reward

    def plan(self, obs, n_simulations: Optional[int] = None) -> int:
        """Plan via simplified MCTS in latent space."""
        n_simulations = n_simulations or self.n_simulations
        obs_t = self.torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with self.torch.no_grad():
            root = self.represent(obs_t)
            policy, value = self.predict(root)
            # simplified: just return argmax of policy
            return int(policy.argmax(-1).item())

    def fit(self, trajectories: List[List[Tuple]], y=None):
        """Train on trajectories. Each trajectory is a list of (obs, action, reward, next_obs)."""
        for trajectory in trajectories:
            for obs, action, reward, next_obs in trajectory:
                obs_t = self.torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                next_obs_t = self.torch.as_tensor(np.asarray(next_obs, dtype=np.float32)).unsqueeze(0)
                # 1. represent both observations
                h = self.represent(obs_t)
                h_next_pred, r_pred = self.dynamics(h, int(action))
                h_next_actual = self.represent(next_obs_t)
                # 2. predict policy + value
                policy, value = self.predict(h)
                # 3. losses
                # dynamics loss: predicted next hidden should match actual
                dynamics_loss = self.nn.functional.mse_loss(h_next_pred, h_next_actual.detach())
                reward_loss = self.nn.functional.mse_loss(r_pred.squeeze(), self.torch.tensor(float(reward)))
                # value loss (simplified: target = reward)
                value_loss = self.nn.functional.mse_loss(value.squeeze(), self.torch.tensor(float(reward)))
                loss = dynamics_loss + reward_loss + value_loss
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        return self.plan(x)


__all__ = ["SARSA", "MCTS", "MuZero"]
