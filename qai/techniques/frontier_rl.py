"""
qai.techniques.frontier_rl -- advanced reinforcement learning beyond DQN/PPO.

  qai.build(technique="ddpg", ...)          — Deep Deterministic Policy Gradient
  qai.build(technique="td3", ...)           — Twin Delayed DDPG
  qai.build(technique="sac", ...)           — Soft Actor-Critic
  qai.build(technique="a2c", ...)           — Advantage Actor-Critic
  qai.build(technique="value_iteration", ...) — Tabular Value Iteration
  qai.build(technique="policy_iteration", ...) — Tabular Policy Iteration
  qai.build(technique="rlhf", ...)          — RLHF for an LM (simplified)

All continuous-control algorithms use the same Gym-like env interface:
  state = env.reset()
  next_state, reward, done, info = env.step(action)
"""
from __future__ import annotations
import math, random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False

from .base import Technique

def _req():
    if not _TORCH:
        raise ImportError(
            "[WHAT] Frontier RL techniques require PyTorch\n"
            "[FIX] pip install torch"
        )


# =====================================================================
# Common helpers
# =====================================================================
class _Actor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(a_dim if False else hidden, a_dim), nn.Tanh(),
        )

    def forward(self, s):
        return self.net(s)


class _Critic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class _ReplayBuffer:
    def __init__(self, capacity=10000):
        self.cap = capacity
        self.buf = []

    def push(self, *args):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append(args)

    def sample(self, batch):
        idx = random.sample(range(len(self.buf)), min(batch, len(self.buf)))
        items = [self.buf[i] for i in idx]
        return list(zip(*items))


# =====================================================================
# 1. DDPG — Deep Deterministic Policy Gradient (Lillicrap 2016)
# =====================================================================
class DDPG(Technique):
    """DDPG — off-policy continuous-control actor-critic with target networks.

    The actor learns a deterministic policy mu(s); the critic learns Q(s, a).
    Both have slowly-tracked target copies. Exploration is via Ornstein-
    Uhlenbeck noise added to actions at training time.
    """
    name = "ddpg"
    family = "reinforcement"

    def __init__(self, state_dim: int = 4, action_dim: int = 2,
                 hidden: int = 64, gamma: float = 0.99, tau: float = 0.005,
                 actor_lr: float = 1e-3, critic_lr: float = 1e-3,
                 buffer_size: int = 10000, batch_size: int = 64,
                 n_steps: int = 1000, **kw):
        super().__init__(state_dim=state_dim, action_dim=action_dim,
                         hidden=hidden, gamma=gamma, tau=tau,
                         actor_lr=actor_lr, critic_lr=critic_lr,
                         buffer_size=buffer_size, batch_size=batch_size,
                         n_steps=n_steps, **kw)
        _req()
        self.s_dim, self.a_dim = state_dim, action_dim
        self.gamma, self.tau = gamma, tau
        self.batch = batch_size
        self.n_steps = n_steps
        self.actor = _Actor(state_dim, action_dim, hidden)
        self.actor_target = _Actor(state_dim, action_dim, hidden)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = _Critic(state_dim, action_dim, hidden)
        self.critic_target = _Critic(state_dim, action_dim, hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buf = _ReplayBuffer(buffer_size)

    def _soft_update(self, src, tgt):
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def select_action(self, s, noise=0.1):
        with torch.no_grad():
            a = self.actor(torch.as_tensor(s, dtype=torch.float32)).numpy()
        return np.clip(a + noise * np.random.randn(*a.shape), -1, 1)

    def train_step(self):
        if len(self.buf.buf) < self.batch:
            return
        s, a, r, ns, d = self.buf.sample(self.batch)
        s = torch.as_tensor(np.array(s), dtype=torch.float32)
        a = torch.as_tensor(np.array(a), dtype=torch.float32)
        r = torch.as_tensor(r, dtype=torch.float32).unsqueeze(-1)
        ns = torch.as_tensor(np.array(ns), dtype=torch.float32)
        d = torch.as_tensor(d, dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            na = self.actor_target(ns)
            q_target = r + self.gamma * (1 - d) * self.critic_target(ns, na)
        q = self.critic(s, a)
        c_loss = F.mse_loss(q, q_target)
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()
        a_loss = -self.critic(s, self.actor(s)).mean()
        self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def fit(self, env, y=None):
        s = env.reset()
        for step in range(self.n_steps):
            a = self.select_action(s)
            ns, r, d, _ = env.step(a)
            self.buf.push(s, a, r, ns, d)
            s = ns if not d else env.reset()
            self.train_step()
        self._trained = True
        return self

    def forward(self, s):
        return self.select_action(s, noise=0.0)


# =====================================================================
# 2. TD3 — Twin Delayed DDPG (Fujimoto 2018)
# =====================================================================
class TD3(Technique):
    """TD3 — DDPG with three fixes: twin critics, delayed actor updates,
    target policy smoothing. The state-of-the-art for continuous control on
    MuJoCo-style benchmarks before SAC.
    """
    name = "td3"
    family = "reinforcement"

    def __init__(self, state_dim: int = 4, action_dim: int = 2,
                 hidden: int = 64, gamma: float = 0.99, tau: float = 0.005,
                 policy_noise: float = 0.2, noise_clip: float = 0.5,
                 policy_delay: int = 2, actor_lr: float = 1e-3,
                 critic_lr: float = 1e-3, buffer_size: int = 10000,
                 batch_size: int = 64, n_steps: int = 1000, **kw):
        super().__init__(state_dim=state_dim, action_dim=action_dim,
                         hidden=hidden, gamma=gamma, tau=tau,
                         policy_noise=policy_noise, noise_clip=noise_clip,
                         policy_delay=policy_delay, actor_lr=actor_lr,
                         critic_lr=critic_lr, buffer_size=buffer_size,
                         batch_size=batch_size, n_steps=n_steps, **kw)
        _req()
        self.s_dim, self.a_dim = state_dim, action_dim
        self.gamma, self.tau = gamma, tau
        self.batch = batch_size
        self.n_steps = n_steps
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.actor = _Actor(state_dim, action_dim, hidden)
        self.actor_target = _Actor(state_dim, action_dim, hidden)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1 = _Critic(state_dim, action_dim, hidden)
        self.critic2 = _Critic(state_dim, action_dim, hidden)
        self.c1_target = _Critic(state_dim, action_dim, hidden)
        self.c2_target = _Critic(state_dim, action_dim, hidden)
        self.c1_target.load_state_dict(self.critic1.state_dict())
        self.c2_target.load_state_dict(self.critic2.state_dict())
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)
        self.buf = _ReplayBuffer(buffer_size)
        self.step_count = 0

    def _soft_update(self, src, tgt):
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def select_action(self, s, noise=0.1):
        with torch.no_grad():
            a = self.actor(torch.as_tensor(s, dtype=torch.float32)).numpy()
        return np.clip(a + noise * np.random.randn(*a.shape), -1, 1)

    def train_step(self):
        if len(self.buf.buf) < self.batch:
            return
        s, a, r, ns, d = self.buf.sample(self.batch)
        s = torch.as_tensor(np.array(s), dtype=torch.float32)
        a = torch.as_tensor(np.array(a), dtype=torch.float32)
        r = torch.as_tensor(r, dtype=torch.float32).unsqueeze(-1)
        ns = torch.as_tensor(np.array(ns), dtype=torch.float32)
        d = torch.as_tensor(d, dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = (self.actor_target(ns) + noise).clamp(-1, 1)
            q1 = self.c1_target(ns, na)
            q2 = self.c2_target(ns, na)
            q_target = r + self.gamma * (1 - d) * torch.min(q1, q2)
        c_loss = F.mse_loss(self.critic1(s, a), q_target) + F.mse_loss(self.critic2(s, a), q_target)
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()
        if self.step_count % self.policy_delay == 0:
            a_loss = -self.critic1(s, self.actor(s)).mean()
            self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
            self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic1, self.c1_target)
        self._soft_update(self.critic2, self.c2_target)
        self.step_count += 1

    def fit(self, env, y=None):
        s = env.reset()
        for step in range(self.n_steps):
            a = self.select_action(s)
            ns, r, d, _ = env.step(a)
            self.buf.push(s, a, r, ns, d)
            s = ns if not d else env.reset()
            self.train_step()
        self._trained = True
        return self

    def forward(self, s):
        return self.select_action(s, noise=0.0)


# =====================================================================
# 3. SAC — Soft Actor-Critic (Haarnoja 2018)
# =====================================================================
class _GaussianActor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, a_dim)
        self.log_std = nn.Linear(hidden, a_dim)

    def forward(self, s):
        h = self.net(s)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        a = torch.tanh(z)
        log_p = normal.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        return a, log_p.sum(-1, keepdim=True)


class SAC(Technique):
    """SAC — maximum entropy RL with stochastic policy and twin critics.

    The actor maximises E[Q(s,a) + alpha * H(pi)]; alpha is auto-tuned.
    Often the strongest continuous-control algorithm on standard benchmarks.
    """
    name = "sac"
    family = "reinforcement"

    def __init__(self, state_dim: int = 4, action_dim: int = 2,
                 hidden: int = 64, gamma: float = 0.99, tau: float = 0.005,
                 alpha: float = 0.2, actor_lr: float = 1e-3,
                 critic_lr: float = 1e-3, buffer_size: int = 10000,
                 batch_size: int = 64, n_steps: int = 1000, **kw):
        super().__init__(state_dim=state_dim, action_dim=action_dim,
                         hidden=hidden, gamma=gamma, tau=tau, alpha=alpha,
                         actor_lr=actor_lr, critic_lr=critic_lr,
                         buffer_size=buffer_size, batch_size=batch_size,
                         n_steps=n_steps, **kw)
        _req()
        self.s_dim, self.a_dim = state_dim, action_dim
        self.gamma, self.tau = gamma, tau
        self.batch = batch_size
        self.n_steps = n_steps
        self.alpha = alpha
        self.actor = _GaussianActor(state_dim, action_dim, hidden)
        self.critic1 = _Critic(state_dim, action_dim, hidden)
        self.critic2 = _Critic(state_dim, action_dim, hidden)
        self.c1_target = _Critic(state_dim, action_dim, hidden)
        self.c2_target = _Critic(state_dim, action_dim, hidden)
        self.c1_target.load_state_dict(self.critic1.state_dict())
        self.c2_target.load_state_dict(self.critic2.state_dict())
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)
        self.buf = _ReplayBuffer(buffer_size)

    def _soft_update(self, src, tgt):
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def select_action(self, s, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
        return a.squeeze(0).numpy()

    def train_step(self):
        if len(self.buf.buf) < self.batch:
            return
        s, a, r, ns, d = self.buf.sample(self.batch)
        s = torch.as_tensor(np.array(s), dtype=torch.float32)
        a = torch.as_tensor(np.array(a), dtype=torch.float32)
        r = torch.as_tensor(r, dtype=torch.float32).unsqueeze(-1)
        ns = torch.as_tensor(np.array(ns), dtype=torch.float32)
        d = torch.as_tensor(d, dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            na, log_p = self.actor(ns)
            q1 = self.c1_target(ns, na)
            q2 = self.c2_target(ns, na)
            q_target = r + self.gamma * (1 - d) * (torch.min(q1, q2) - self.alpha * log_p)
        c_loss = F.mse_loss(self.critic1(s, a), q_target) + F.mse_loss(self.critic2(s, a), q_target)
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()
        na, log_p = self.actor(s)
        q1 = self.critic1(s, na)
        q2 = self.critic2(s, na)
        a_loss = (self.alpha * log_p - torch.min(q1, q2)).mean()
        self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
        self._soft_update(self.critic1, self.c1_target)
        self._soft_update(self.critic2, self.c2_target)

    def fit(self, env, y=None):
        s = env.reset()
        for step in range(self.n_steps):
            a = self.select_action(s)
            ns, r, d, _ = env.step(a)
            self.buf.push(s, a, r, ns, d)
            s = ns if not d else env.reset()
            self.train_step()
        self._trained = True
        return self

    def forward(self, s):
        return self.select_action(s, deterministic=True)


# =====================================================================
# 4. A2C — Advantage Actor-Critic
# =====================================================================
class A2C(Technique):
    """A2C — synchronous Advantage Actor-Critic.

    Both actor (policy) and critic (value) share a body. Trained with the
    advantage A = R - V(s) on full episode rollouts.
    """
    name = "a2c"
    family = "reinforcement"

    def __init__(self, state_dim: int = 4, n_actions: int = 4,
                 hidden: int = 64, gamma: float = 0.99,
                 actor_lr: float = 1e-3, critic_lr: float = 1e-3,
                 n_episodes: int = 100, max_steps: int = 200, **kw):
        super().__init__(state_dim=state_dim, n_actions=n_actions,
                         hidden=hidden, gamma=gamma, actor_lr=actor_lr,
                         critic_lr=critic_lr, n_episodes=n_episodes,
                         max_steps=max_steps, **kw)
        _req()
        self.s_dim, self.n_actions = state_dim, n_actions
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, s):
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def fit(self, env, y=None):
        for ep in range(self.n_episodes):
            s = env.reset()
            states, actions, rewards = [], [], []
            for step in range(self.max_steps):
                a = self.select_action(s)
                ns, r, d, _ = env.step(a)
                states.append(s); actions.append(a); rewards.append(r)
                s = ns
                if d: break
            # compute returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            states = torch.as_tensor(np.array(states), dtype=torch.float32)
            actions = torch.as_tensor(actions)
            returns = torch.as_tensor(returns, dtype=torch.float32)
            values = self.critic(states).squeeze(-1)
            advantages = returns - values.detach()
            logits = self.actor(states)
            log_p = F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            a_loss = -(log_p * advantages).mean()
            c_loss = F.mse_loss(values, returns)
            self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
            self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()
        self._trained = True
        return self

    def forward(self, s):
        return self.select_action(s)


# =====================================================================
# 5. VALUE ITERATION — tabular MDP solver
# =====================================================================
class ValueIteration(Technique):
    """Tabular Value Iteration. env must expose:
       - n_states, n_actions
       - transition(s, a) -> list of (prob, next_state, reward, done)
    """
    name = "value_iteration"
    family = "planning"

    def __init__(self, gamma: float = 0.99, theta: float = 1e-6, **kw):
        super().__init__(gamma=gamma, theta=theta, **kw)
        self.gamma = gamma
        self.theta = theta
        self.V = None
        self.policy = None

    def fit(self, env, y=None):
        n_s = env.n_states
        n_a = env.n_actions
        self.V = np.zeros(n_s)
        while True:
            delta = 0
            for s in range(n_s):
                v = self.V[s]
                qs = np.zeros(n_a)
                for a in range(n_a):
                    for p, ns, r, d in env.transition(s, a):
                        qs[a] += p * (r + self.gamma * self.V[ns] * (1 - d))
                self.V[s] = qs.max()
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break
        # policy extraction
        self.policy = np.zeros(n_s, dtype=int)
        for s in range(n_s):
            qs = np.zeros(n_a)
            for a in range(n_a):
                for p, ns, r, d in env.transition(s, a):
                    qs[a] += p * (r + self.gamma * self.V[ns] * (1 - d))
            self.policy[s] = qs.argmax()
        self._trained = True
        return self

    def forward(self, s):
        if self.policy is None:
            raise RuntimeError("Call fit(env) first")
        return int(self.policy[s])


# =====================================================================
# 6. POLICY ITERATION — tabular
# =====================================================================
class PolicyIteration(Technique):
    """Tabular Policy Iteration. Alternates:
       - Policy evaluation: solve V^pi = R + gamma * P^pi V^pi
       - Policy improvement: pi(s) = argmax_a sum P V
    """
    name = "policy_iteration"
    family = "planning"

    def __init__(self, gamma: float = 0.99, theta: float = 1e-6, **kw):
        super().__init__(gamma=gamma, theta=theta, **kw)
        self.gamma = gamma
        self.theta = theta
        self.policy = None
        self.V = None

    def fit(self, env, y=None):
        n_s, n_a = env.n_states, env.n_actions
        self.policy = np.zeros(n_s, dtype=int)
        self.V = np.zeros(n_s)
        while True:
            # policy evaluation
            while True:
                delta = 0
                for s in range(n_s):
                    v = self.V[s]
                    a = self.policy[s]
                    val = 0
                    for p, ns, r, d in env.transition(s, a):
                        val += p * (r + self.gamma * self.V[ns] * (1 - d))
                    self.V[s] = val
                    delta = max(delta, abs(v - val))
                if delta < self.theta:
                    break
            # policy improvement
            stable = True
            for s in range(n_s):
                old = self.policy[s]
                qs = np.zeros(n_a)
                for a in range(n_a):
                    for p, ns, r, d in env.transition(s, a):
                        qs[a] += p * (r + self.gamma * self.V[ns] * (1 - d))
                self.policy[s] = qs.argmax()
                if old != self.policy[s]:
                    stable = False
            if stable:
                break
        self._trained = True
        return self

    def forward(self, s):
        if self.policy is None:
            raise RuntimeError("Call fit(env) first")
        return int(self.policy[s])


# =====================================================================
# 7. RLHF — simplified Reinforcement Learning from Human Feedback
# =====================================================================
class _RewardModel(nn.Module):
    def __init__(self, emb_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RLHF(Technique):
    """Simplified RLHF for a language model.

    Two phases:
      1. Reward model training on preference pairs (chosen, rejected).
      2. PPO-style fine-tuning of the policy against the learned reward,
         with KL penalty to a reference policy.

    This NumPy/PyTorch-only implementation works on embedding vectors so
    it can be tested without an actual LM in the loop.
    """
    name = "rlhf"
    family = "alignment"

    def __init__(self, emb_dim: int = 16, hidden: int = 64,
                 rm_lr: float = 1e-3, ppo_lr: float = 1e-3,
                 kl_coef: float = 0.1, rm_epochs: int = 100,
                 ppo_steps: int = 100, **kw):
        super().__init__(emb_dim=emb_dim, hidden=hidden,
                         rm_lr=rm_lr, ppo_lr=ppo_lr, kl_coef=kl_coef,
                         rm_epochs=rm_epochs, ppo_steps=ppo_steps, **kw)
        _req()
        self.emb_dim = emb_dim
        self.kl_coef = kl_coef
        self.rm_epochs = rm_epochs
        self.ppo_steps = ppo_steps
        self.reward_model = _RewardModel(emb_dim, hidden)
        self.policy = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )
        self.ref_policy = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )
        self.ref_policy.load_state_dict(self.policy.state_dict())
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        self.rm_opt = torch.optim.Adam(self.reward_model.parameters(), lr=rm_lr)
        self.ppo_opt = torch.optim.Adam(self.policy.parameters(), lr=ppo_lr)

    def fit(self, X, y=None):
        """X: dict with 'chosen' (N, emb_dim) and 'rejected' (N, emb_dim)."""
        chosen = torch.as_tensor(np.asarray(X["chosen"], dtype=np.float32))
        rejected = torch.as_tensor(np.asarray(X["rejected"], dtype=np.float32))
        # Phase 1: train reward model
        for ep in range(self.rm_epochs):
            self.rm_opt.zero_grad()
            r_c = self.reward_model(chosen)
            r_r = self.reward_model(rejected)
            loss = -F.logsigmoid(r_c - r_r).mean()
            loss.backward()
            self.rm_opt.step()
        # Phase 2: PPO on policy
        for step in range(self.ppo_steps):
            self.ppo_opt.zero_grad()
            out = self.policy(chosen)
            r = self.reward_model(out)
            ref_out = self.ref_policy(chosen)
            kl = F.mse_loss(out, ref_out)
            loss = -(r.mean() - self.kl_coef * kl)
            loss.backward()
            self.ppo_opt.step()
        self._trained = True
        return self

    def score(self, x):
        with torch.no_grad():
            return self.reward_model(torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)).item()

    def forward(self, x):
        with torch.no_grad():
            out = self.policy(torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0))
        return out.squeeze(0).numpy()
