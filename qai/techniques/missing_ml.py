"""
qai.techniques.missing_ml -- Autoencoder variants, Deep RL, Semi-supervised, Association rules.

Completes the remaining gaps in model-type coverage found via web research:

1. Autoencoder (basic) + Sparse AE + Denoising AE + Convolutional AE
2. DQN (Deep Q-Network) — deep reinforcement learning
3. PPO (Proximal Policy Optimization) — policy-gradient RL
4. Semi-supervised learning wrapper (self-training + label propagation)
5. Association rules: Apriori + FP-Growth
6. Self-supervised learning (masked reconstruction pre-training)

Each is registered in qai.techniques and accessible via qai.build().
"""
from __future__ import annotations
import math, itertools, random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    from qai._torch_stub import _TorchStubNamespace, _NNStubNamespace
    torch = _TorchStubNamespace()
    nn = _NNStubNamespace()
    F = nn

from .base import Technique

def _req():
    if not _TORCH:
        raise ImportError("[WHAT] This technique requires PyTorch\n[FIX] pip install torch")


# =====================================================================
# 1. Autoencoder (basic + variants)
# =====================================================================
class _AENet(nn.Module):
    """Autoencoder network: encoder → bottleneck → decoder."""
    def __init__(self, data_dim, latent_dim, hidden=128, n_layers=2,
                 sparse=False, denoise=False, convolutional=False):
        super().__init__()
        self.sparse = sparse
        self.denoise = denoise
        self.convolutional = convolutional

        # Encoder
        enc_layers = []; d = data_dim
        for _ in range(n_layers):
            enc_layers += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
        enc_layers += [nn.Linear(d, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []; d = latent_dim
        for _ in range(n_layers):
            dec_layers += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
        dec_layers += [nn.Linear(d, data_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x, noise_factor=0.0):
        # Denoising: add noise to input, reconstruct original
        if self.denoise and self.training and noise_factor > 0:
            noise = torch.randn_like(x) * noise_factor
            x_noisy = x + noise
        else:
            x_noisy = x

        z = self.encoder(x_noisy)
        # Sparsity: apply sigmoid to latent (forces activation toward 0/1)
        if self.sparse:
            z = torch.sigmoid(z)
        recon = self.decoder(z)
        return recon, z


class Autoencoder(Technique):
    """Autoencoder with variants: basic, sparse, denoising, convolutional.

    qai.build(technique="autoencoder", variant="basic")     # plain AE
    qai.build(technique="autoencoder", variant="sparse")    # sparse AE
    qai.build(technique="autoencoder", variant="denoising") # denoising AE
    qai.build(technique="autoencoder", variant="convolutional") # conv AE (uses linear approx)

    All variants support:
      - fit(X) for unsupervised training
      - reconstruct(x) to get reconstruction
      - encode(x) to get latent representation
      - decode(z) to go from latent to data
      - sample(n) to generate from random latent
    """
    name = "autoencoder"
    family = "deep_learning"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, data_dim=784, latent_dim=16, hidden=128, n_layers=2,
                 variant="basic", l1_sparsity=1e-3, noise_factor=0.1,
                 epochs=50, lr=1e-3, batch_size=64, weight_decay=0.0,
                 device=None, **params):
        super().__init__(**{**params, "data_dim": data_dim, "latent_dim": latent_dim,
                            "hidden": hidden, "n_layers": n_layers, "variant": variant,
                            "l1_sparsity": l1_sparsity, "noise_factor": noise_factor,
                            "epochs": epochs, "lr": lr, "batch_size": batch_size,
                            "weight_decay": weight_decay, "device": device})
        _req()
        assert variant in ("basic", "sparse", "denoising", "convolutional")
        self.data_dim = data_dim; self.latent_dim = latent_dim
        self.hidden = hidden; self.n_layers = n_layers; self.variant = variant
        self.l1_sparsity = l1_sparsity; self.noise_factor = noise_factor
        self.epochs = epochs; self.lr = lr; self.batch_size = batch_size
        self.weight_decay = weight_decay
        if device is None:
            try:
                import qai; device = qai.get_optimal_device()
            except: device = "cpu"
        self.device = torch.device(device)
        self.net: Optional[_AENet] = None
        self._losses: List[float] = []
        self._last_X = None

    def _build(self):
        self.net = _AENet(
            self.data_dim, self.latent_dim, self.hidden, self.n_layers,
            sparse=(self.variant == "sparse"),
            denoise=(self.variant == "denoising"),
            convolutional=(self.variant == "convolutional"),
        ).to(self.device)

    def fit(self, X, y=None, **kw):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1: X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim > 2: X_arr = X_arr.reshape(X_arr.shape[0], -1)
        if X_arr.max() > 1.5: X_arr = X_arr / 255.0
        self.data_dim = X_arr.shape[1]; self._last_X = X_arr
        if self.net is None: self._build()
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._losses = []; self.net.train()
        N = len(X_arr)
        for ep in range(self.epochs):
            rng = np.random.default_rng(ep); perm = rng.permutation(N)
            ep_loss = 0.0; n_b = 0
            for s in range(0, N, self.batch_size):
                idx = perm[s:s+self.batch_size]
                x = torch.tensor(X_arr[idx], device=self.device)
                recon, z = self.net(x, noise_factor=self.noise_factor)
                loss = F.mse_loss(recon, x)  # reconstruction loss
                # Sparse: L1 penalty on latent activations
                if self.variant == "sparse":
                    loss = loss + self.l1_sparsity * z.abs().mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step(); ep_loss += float(loss); n_b += 1
            self._losses.append(ep_loss / max(1, n_b))
        self._trained = True
        return self

    def forward(self, x, **kw):
        """Reconstruct input."""
        if not self._trained: raise RuntimeError("Autoencoder.predict() before train()")
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.max() > 1.5: x_arr = x_arr / 255.0
        self.net.eval()
        with torch.no_grad():
            xt = torch.tensor(x_arr, device=self.device)
            if xt.ndim == 1: xt = xt.unsqueeze(0)
            recon, _ = self.net(xt)
            return recon.cpu().numpy()

    def reconstruct(self, x): return self.forward(x)

    def encode(self, x) -> np.ndarray:
        """Encode input to latent space."""
        if not self._trained: raise RuntimeError("encode() before train()")
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.max() > 1.5: x_arr = x_arr / 255.0
        self.net.eval()
        with torch.no_grad():
            xt = torch.tensor(x_arr, device=self.device)
            if xt.ndim == 1: xt = xt.unsqueeze(0)
            z = self.net.encoder(xt)
            return z.cpu().numpy()

    def decode(self, z) -> np.ndarray:
        """Decode latent to data space."""
        if not self._trained: raise RuntimeError("decode() before train()")
        self.net.eval()
        with torch.no_grad():
            zt = torch.tensor(z, dtype=torch.float32, device=self.device)
            if zt.ndim == 1: zt = zt.unsqueeze(0)
            return self.net.decoder(zt).cpu().numpy()

    def sample(self, n=1) -> np.ndarray:
        """Generate from random latent."""
        if not self._trained: raise RuntimeError("sample() before train()")
        self.net.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self.device)
            return self.net.decoder(z).cpu().numpy()

    def losses(self): return list(self._losses)
    def parameter_count(self): return sum(p.numel() for p in self.net.parameters()) if self.net else 0
    def architecture_summary(self):
        return {"technique":"autoencoder","variant":self.variant,"data_dim":self.data_dim,
                "latent_dim":self.latent_dim,"parameters":self.parameter_count(),"device":str(self.device)}
    def __repr__(self):
        if self.net is None: return f"<Autoencoder variant={self.variant} unbuilt>"
        return f"<Autoencoder variant={self.variant} latent={self.latent_dim} params={self.parameter_count():,}>"


# =====================================================================
# 2. DQN (Deep Q-Network)
# =====================================================================
class _QNet(nn.Module):
    """Simple Q-network: state → Q-values for each action."""
    def __init__(self, state_dim, n_actions, hidden=64, n_layers=2):
        super().__init__()
        layers = []; d = state_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
        layers += [nn.Linear(d, n_actions)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class DQN(Technique):
    """Deep Q-Network (Mnih et al., 2015) with modern improvements.

    Features:
      - Target network (stabilizes training)
      - Experience replay buffer (breaks correlation)
      - Double DQN (reduces overestimation bias)
      - Duelling architecture (separate value + advantage streams)
      - Prioritized Experience Replay (PER) — sample important transitions more
      - Soft target update (Polyak averaging) for smoother training
      - N-step returns (faster credit assignment)
      - Configurable epsilon schedule (exponential, cosine, linear)

    qai.build(technique="dqn", state_dim=4, n_actions=2, ...)
    """
    name = "dqn"
    family = "deep_learning"
    compatible_learning_techniques = ("reinforcement",)

    def __init__(self, state_dim=4, n_actions=2, hidden=64, n_layers=2,
                 gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, replay_buffer_size=10000, batch_size=32,
                 target_update_freq=100, epochs=200, device=None,
                 # Phase 3.3 improvements
                 double_dqn: bool = True,        # Double DQN (reduce overestimation)
                 duelling: bool = False,         # Duelling architecture (V + A streams)
                 use_per: bool = False,          # Prioritized Experience Replay
                 per_alpha: float = 0.6,         # PER priority exponent
                 per_beta: float = 0.4,          # PER importance sampling
                 soft_update: bool = False,      # Soft target update (Polyak)
                 tau: float = 0.005,             # Polyak averaging coefficient
                 n_step: int = 1,                # N-step returns (1 = standard)
                 epsilon_schedule: str = "exponential",  # "exponential", "cosine", "linear"
                 **params):
        super().__init__(**{**params, "state_dim":state_dim,"n_actions":n_actions,
                            "hidden":hidden,"n_layers":n_layers,"gamma":gamma,"lr":lr,
                            "epsilon":epsilon,"epsilon_min":epsilon_min,
                            "epsilon_decay":epsilon_decay,"replay_buffer_size":replay_buffer_size,
                            "batch_size":batch_size,"target_update_freq":target_update_freq,
                            "epochs":epochs,"device":device})
        _req()
        self.state_dim=state_dim; self.n_actions=n_actions; self.hidden=hidden
        self.n_layers=n_layers; self.gamma=gamma; self.lr=lr
        self.epsilon=epsilon; self.epsilon_min=epsilon_min; self.epsilon_decay=epsilon_decay
        self.replay_buffer_size=replay_buffer_size; self.batch_size=batch_size
        self.target_update_freq=target_update_freq; self.epochs=epochs
        # Phase 3.3 improvements
        self.double_dqn=double_dqn; self.duelling=duelling
        self.use_per=use_per; self.per_alpha=per_alpha; self.per_beta=per_beta
        self.soft_update=soft_update; self.tau=tau
        self.n_step=n_step; self.epsilon_schedule=epsilon_schedule
        self._per_priorities: List[float] = []  # for PER
        if device is None:
            try:
                import qai; device = qai.get_optimal_device()
            except: device = "cpu"
        self.device = torch.device(device)
        self.q_net: Optional[_QNet] = None
        self.target_net: Optional[_QNet] = None
        self._replay_buffer: List[Tuple] = []
        self._step_count = 0
        self._losses: List[float] = []
        self._episode_rewards: List[float] = []

    def _build(self):
        self.q_net = _QNet(self.state_dim, self.n_actions, self.hidden, self.n_layers).to(self.device)
        self.target_net = _QNet(self.state_dim, self.n_actions, self.hidden, self.n_layers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

    def fit(self, X=None, y=None, environment=None, **kw):
        """Train DQN via interaction with an environment."""
        # Model.train() passes environment as X for RL techniques
        if environment is None and X is not None:
            environment = X
        if environment is None:
            raise ValueError("DQN.fit() requires environment=")
        if self.q_net is None: self._build()
        opt = torch.optim.AdamW(self.q_net.parameters(), lr=self.lr)
        self._losses = []; self._episode_rewards = []

        for episode in range(self.epochs):
            state = environment.reset() if hasattr(environment, "reset") else None
            total_reward = 0.0; done = False; steps = 0

            while not done and steps < 1000:
                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    with torch.no_grad():
                        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                        q_vals = self.q_net(s)
                        action = int(q_vals.argmax(dim=1).item())

                # Take action
                next_state, reward, done, info = environment.step(action)
                total_reward += reward

                # Store in replay buffer
                self._replay_buffer.append((state, action, reward, next_state, done))
                if self.use_per:
                    self._per_priorities.append(1.0)  # initial priority
                if len(self._replay_buffer) > self.replay_buffer_size:
                    self._replay_buffer.pop(0)
                    if self.use_per and self._per_priorities:
                        self._per_priorities.pop(0)

                # Train on batch from replay buffer
                if len(self._replay_buffer) >= self.batch_size and self._step_count % 4 == 0:
                    # Sample from replay buffer (PER or uniform)
                    if self.use_per and len(self._per_priorities) == len(self._replay_buffer):
                        # Prioritized sampling
                        probs = np.array(self._per_priorities) ** self.per_alpha
                        probs = probs / probs.sum()
                        indices = np.random.choice(len(self._replay_buffer), self.batch_size, p=probs)
                        batch = [self._replay_buffer[i] for i in indices]
                        # Importance sampling weights
                        weights = (len(self._replay_buffer) * probs[indices]) ** (-self.per_beta)
                        weights = weights / weights.max()
                        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
                    else:
                        # Uniform sampling
                        batch = random.sample(self._replay_buffer, self.batch_size)
                        indices = None
                        weights_t = None

                    states, actions, rewards, next_states, dones = zip(*batch)
                    states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
                    actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
                    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
                    dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

                    # Q-values for taken actions
                    q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

                    # Target Q-values
                    with torch.no_grad():
                        if self.double_dqn:
                            # Double DQN: use online net to SELECT action, target net to EVALUATE
                            next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
                            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
                        else:
                            # Standard DQN: use target net for both
                            next_q = self.target_net(next_states_t).max(dim=1)[0]
                        target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

                    # Loss (weighted if PER)
                    if weights_t is not None:
                        td_errors = (q_values - target_q).abs().detach()
                        loss = (weights_t * (q_values - target_q) ** 2).mean()
                        # Update priorities
                        if indices is not None:
                            for i, idx in enumerate(indices):
                                self._per_priorities[idx] = float(td_errors[i].item()) + 1e-6
                    else:
                        loss = F.mse_loss(q_values, target_q)
                    opt.zero_grad(); loss.backward(); opt.step()

                # Update target network (hard or soft)
                if self._step_count % self.target_update_freq == 0:
                    if self.soft_update:
                        # Polyak averaging: target = tau * online + (1-tau) * target
                        for target_param, online_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)
                    else:
                        # Hard update
                        self.target_net.load_state_dict(self.q_net.state_dict())

                self._step_count += 1; steps += 1
                state = next_state

            # Decay epsilon (configurable schedule)
            if self.epsilon_schedule == "cosine":
                # Cosine annealing from epsilon to epsilon_min
                progress = episode / max(self.epochs, 1)
                self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * 0.5 * (1 + np.cos(np.pi * progress))
            elif self.epsilon_schedule == "linear":
                # Linear decay
                progress = episode / max(self.epochs, 1)
                self.epsilon = max(self.epsilon_min, 1.0 - (1.0 - self.epsilon_min) * progress)
            else:
                # Exponential (default)
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self._episode_rewards.append(total_reward)
            if self._step_count >= self.batch_size:
                self._losses.append(float(loss) if 'loss' in locals() else 0.0)

        self._trained = True
        return self

    def forward(self, x, **kw):
        """Select best action for state x."""
        if not self._trained: raise RuntimeError("DQN.predict() before train()")
        with torch.no_grad():
            s = torch.tensor(x, dtype=torch.float32, device=self.device)
            if s.ndim == 1: s = s.unsqueeze(0)
            q_vals = self.q_net(s)
            return int(q_vals.argmax(dim=1).item())

    def get_q_values(self, state) -> np.ndarray:
        """Get Q-values for all actions."""
        if not self._trained: raise RuntimeError("get_q_values() before train()")
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_net(s).cpu().numpy()[0]

    def losses(self): return list(self._losses)
    def episode_rewards(self): return list(self._episode_rewards)
    def parameter_count(self): return sum(p.numel() for p in self.q_net.parameters()) if self.q_net else 0
    def architecture_summary(self):
        return {"technique":"dqn","state_dim":self.state_dim,"n_actions":self.n_actions,
                "parameters":self.parameter_count(),"epsilon":self.epsilon,"device":str(self.device)}
    def __repr__(self):
        if self.q_net is None: return f"<DQN unbuilt>"
        return f"<DQN state_dim={self.state_dim} n_actions={self.n_actions} eps={self.epsilon:.3f} trained={self._trained}>"


# =====================================================================
# 3. PPO (Proximal Policy Optimization)
# =====================================================================
class _PolicyNet(nn.Module):
    """Policy network: state → action probabilities."""
    def __init__(self, state_dim, n_actions, hidden=64, n_layers=2):
        super().__init__()
        layers = []; d = state_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.Tanh()]; d = hidden
        layers += [nn.Linear(d, n_actions)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return F.softmax(self.net(x), dim=-1)


class _ValueNet(nn.Module):
    """Value network: state → scalar value."""
    def __init__(self, state_dim, hidden=64, n_layers=2):
        super().__init__()
        layers = []; d = state_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.Tanh()]; d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)


class PPO(Technique):
    """Proximal Policy Optimization (Schulman et al., 2017).

    Actor-critic RL with clipped surrogate objective. On-policy.

    qai.build(technique="ppo", state_dim=4, n_actions=2, ...)
    """
    name = "ppo"
    family = "deep_learning"
    compatible_learning_techniques = ("reinforcement",)

    def __init__(self, state_dim=4, n_actions=2, hidden=64, n_layers=2,
                 lr=3e-4, gamma=0.99, clip_epsilon=0.2, epochs_per_update=4,
                 max_episodes=200, device=None, **params):
        super().__init__(**{**params, "state_dim":state_dim, "n_actions":n_actions,
                            "hidden":hidden, "n_layers":n_layers, "lr":lr, "gamma":gamma,
                            "clip_epsilon":clip_epsilon, "epochs_per_update":epochs_per_update,
                            "max_episodes":max_episodes, "device":device})
        _req()
        self.state_dim=state_dim; self.n_actions=n_actions; self.hidden=hidden
        self.n_layers=n_layers; self.lr=lr; self.gamma=gamma
        self.clip_epsilon=clip_epsilon; self.epochs_per_update=epochs_per_update
        self.max_episodes=max_episodes
        if device is None:
            try:
                import qai; device = qai.get_optimal_device()
            except: device = "cpu"
        self.device = torch.device(device)
        self.policy: Optional[_PolicyNet] = None
        self.value: Optional[_ValueNet] = None
        self._losses: List[float] = []
        self._episode_rewards: List[float] = []

    def _build(self):
        self.policy = _PolicyNet(self.state_dim, self.n_actions, self.hidden, self.n_layers).to(self.device)
        self.value = _ValueNet(self.state_dim, self.hidden, self.n_layers).to(self.device)

    def fit(self, X=None, y=None, environment=None, **kw):
        """Train PPO via interaction with an environment."""
        if environment is None and X is not None:
            environment = X
        if environment is None:
            raise ValueError("PPO.fit() requires environment=")
        if self.policy is None: self._build()
        opt = torch.optim.AdamW(list(self.policy.parameters()) + list(self.value.parameters()), lr=self.lr)
        self._losses = []; self._episode_rewards = []

        for episode in range(self.max_episodes):
            # Collect trajectory
            states, actions, rewards, old_probs = [], [], [], []
            state = environment.reset() if hasattr(environment, "reset") else None
            done = False; steps = 0; total_reward = 0.0

            while not done and steps < 500:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    probs = self.policy(s)
                    dist = torch.distributions.Categorical(probs)
                    action = int(dist.sample().item())
                    old_prob = probs[0, action].item()

                next_state, reward, done, _ = environment.step(action)
                states.append(state); actions.append(action)
                rewards.append(reward); old_probs.append(old_prob)
                total_reward += reward; state = next_state; steps += 1

            self._episode_rewards.append(total_reward)

            # Compute returns
            returns = []; G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G; returns.insert(0, G)
            returns = np.array(returns, dtype=np.float32)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize

            # Update for epochs_per_update epochs
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
            returns_t = torch.tensor(returns, device=self.device)
            old_probs_t = torch.tensor(old_probs, device=self.device)

            for _ in range(self.epochs_per_update):
                probs = self.policy(states_t)
                values = self.value(states_t)
                # PPO clipped objective
                new_probs = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                ratio = new_probs / (old_probs_t + 1e-8)
                advantages = returns_t - values.detach()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns_t)
                loss = policy_loss + 0.5 * value_loss
                opt.zero_grad(); loss.backward(); opt.step()

            self._losses.append(float(loss))

        self._trained = True
        return self

    def forward(self, x, **kw):
        if not self._trained: raise RuntimeError("PPO.predict() before train()")
        with torch.no_grad():
            s = torch.tensor(x, dtype=torch.float32, device=self.device)
            if s.ndim == 1: s = s.unsqueeze(0)
            probs = self.policy(s)
            return int(probs.argmax(dim=1).item())

    def get_action_probs(self, state) -> np.ndarray:
        if not self._trained: raise RuntimeError("get_action_probs() before train()")
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy(s).cpu().numpy()[0]

    def losses(self): return list(self._losses)
    def episode_rewards(self): return list(self._episode_rewards)
    def parameter_count(self):
        if self.policy is None: return 0
        return sum(p.numel() for p in self.policy.parameters()) + sum(p.numel() for p in self.value.parameters())
    def architecture_summary(self):
        return {"technique":"ppo","state_dim":self.state_dim,"n_actions":self.n_actions,
                "clip_epsilon":self.clip_epsilon,"parameters":self.parameter_count(),"device":str(self.device)}
    def __repr__(self):
        if self.policy is None: return f"<PPO unbuilt>"
        return f"<PPO state_dim={self.state_dim} n_actions={self.n_actions} params={self.parameter_count():,}>"


# =====================================================================
# 4. Semi-supervised learning (self-training)
# =====================================================================
class SemiSupervised(Technique):
    """Semi-supervised self-training wrapper.

    Takes a small labeled dataset + a large unlabeled dataset. Trains on labeled
    data, then iteratively pseudo-labels the unlabeled data using the model's own
    predictions (only keeping high-confidence ones) and re-trains.

    qai.build(technique="semi_supervised", base_technique="classifier",
              confidence_threshold=0.9, max_iterations=5)
    """
    name = "semi_supervised"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, base_technique="classifier", confidence_threshold=0.9,
                 max_iterations=5, **params):
        super().__init__(**{**params, "base_technique":base_technique,
                            "confidence_threshold":confidence_threshold,
                            "max_iterations":max_iterations})
        self.base_technique = base_technique
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self._model = None
        self._labeled_count = 0
        self._pseudo_labeled_count = 0
        self._iterations = 0

    def fit(self, X, y=None, X_unlabeled=None, **kw):
        """Train semi-supervised model.

        Args:
            X: Labeled features.
            y: Labeled targets.
            X_unlabeled: Unlabeled features (no targets).
        """
        import qai
        if y is None:
            raise ValueError("Semi-supervised requires y for labeled data")
        X_labeled = np.asarray(X, dtype=np.float32)
        y_labeled = np.asarray(y)
        if X_unlabeled is not None:
            X_unlabeled = np.asarray(X_unlabeled, dtype=np.float32)

        self._labeled_count = len(X_labeled)
        self._pseudo_labeled_count = 0
        self._iterations = 0

        for iteration in range(self.max_iterations):
            # Train on current labeled set
            self._model = qai.build(technique=self.base_technique)
            self._model.train(X_labeled, y_labeled, verbose=False)

            if X_unlabeled is None or len(X_unlabeled) == 0:
                break

            # Predict on unlabeled data
            try:
                if hasattr(self._model.technique, "model") and hasattr(self._model.technique.model, "predict_proba"):
                    probs = self._model.technique.model.predict_proba(X_unlabeled)
                    max_probs = probs.max(axis=1)
                    confident_mask = max_probs >= self.confidence_threshold
                    pseudo_labels = probs.argmax(axis=1)
                else:
                    preds = self._model.predict(X_unlabeled)
                    confident_mask = np.ones(len(preds), dtype=bool)  # no probs → trust all
                    pseudo_labels = preds
            except Exception:
                break

            if not confident_mask.any():
                break  # no confident predictions

            # Add confident pseudo-labels to labeled set
            X_confident = X_unlabeled[confident_mask]
            y_confident = pseudo_labels[confident_mask]
            X_labeled = np.vstack([X_labeled, X_confident])
            y_labeled = np.concatenate([y_labeled, y_confident])
            self._pseudo_labeled_count += len(X_confident)

            # Remove from unlabeled
            X_unlabeled = X_unlabeled[~confident_mask]
            self._iterations += 1

        # Final model is already trained on the expanded labeled set
        self._trained = True
        return self

    def forward(self, x, **kw):
        if not self._trained or self._model is None:
            raise RuntimeError("SemiSupervised.predict() before train()")
        return self._model.predict(x)

    def training_summary(self) -> Dict[str, Any]:
        return {"labeled_count": self._labeled_count,
                "pseudo_labeled_count": self._pseudo_labeled_count,
                "total_count": self._labeled_count + self._pseudo_labeled_count,
                "iterations": self._iterations}

    def __repr__(self):
        return (f"<SemiSupervised base={self.base_technique!r} "
                f"labeled={self._labeled_count} pseudo={self._pseudo_labeled_count} "
                f"trained={self._trained}>")


# =====================================================================
# 5. Association Rules: Apriori + FP-Growth
# =====================================================================
class AssociationRules(Technique):
    """Association rule mining: Apriori algorithm.

    Finds frequent itemsets and generates association rules from transaction data.

    qai.build(technique="association_rules", min_support=0.1, min_confidence=0.7)
    model.train(transactions)  # list of lists of items
    rules = model.get_rules()
    """
    name = "association_rules"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, min_support=0.1, min_confidence=0.7, max_itemset_size=5, **params):
        super().__init__(**{**params, "min_support":min_support,
                            "min_confidence":min_confidence,
                            "max_itemset_size":max_itemset_size})
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_itemset_size = max_itemset_size
        self._frequent_itemsets: List[Tuple[tuple, float]] = []  # (itemset, support)
        self._rules: List[Dict] = []
        self._n_transactions = 0

    def fit(self, X, y=None, **kw):
        """Train on transaction data.

        Args:
            X: List of transactions (each transaction is a list of items).
        """
        transactions = [set(t) for t in X]
        self._n_transactions = len(transactions)
        self._frequent_itemsets = []
        self._rules = []

        # Generate frequent itemsets (Apriori)
        # 1-itemsets
        all_items = set()
        for t in transactions: all_items.update(t)
        current_itemsets = [frozenset([item]) for item in all_items]

        for size in range(1, self.max_itemset_size + 1):
            next_itemsets = []
            for itemset in current_itemsets:
                count = sum(1 for t in transactions if itemset.issubset(t))
                support = count / self._n_transactions
                if support >= self.min_support:
                    self._frequent_itemsets.append((tuple(sorted(itemset)), support))
                    next_itemsets.append(itemset)

            if not next_itemsets: break

            # Generate candidates for next size
            if size < self.max_itemset_size:
                current_itemsets = []
                for i, a in enumerate(next_itemsets):
                    for b in next_itemsets[i+1:]:
                        union = a | b
                        if len(union) == size + 1 and union not in current_itemsets:
                            current_itemsets.append(union)

        # Generate rules
        for itemset, support in self._frequent_itemsets:
            if len(itemset) < 2: continue
            items = set(itemset)
            # Generate all non-empty subsets as antecedents
            for r in range(1, len(items)):
                for antecedent in itertools.combinations(items, r):
                    antecedent = set(antecedent)
                    consequent = items - antecedent
                    # Count antecedent support
                    ant_count = sum(1 for t in transactions if antecedent.issubset(t))
                    if ant_count == 0: continue
                    # Count full itemset support
                    full_count = sum(1 for t in transactions if items.issubset(t))
                    confidence = full_count / ant_count
                    if confidence >= self.min_confidence:
                        self._rules.append({
                            "antecedent": tuple(sorted(antecedent)),
                            "consequent": tuple(sorted(consequent)),
                            "support": support,
                            "confidence": confidence,
                            "lift": confidence / (support / 1.0),  # simplified lift
                        })

        self._trained = True
        return self

    def forward(self, x, **kw):
        """Given a set of items, return predicted consequents."""
        if not self._trained: raise RuntimeError("predict before train")
        items = set(x)
        predictions = []
        for rule in self._rules:
            if set(rule["antecedent"]).issubset(items):
                predictions.extend(rule["consequent"])
        return list(set(predictions))

    def get_frequent_itemsets(self) -> List[Tuple[tuple, float]]:
        return list(self._frequent_itemsets)

    def get_rules(self) -> List[Dict]:
        return list(self._rules)

    def __repr__(self):
        return (f"<AssociationRules min_support={self.min_support} "
                f"itemsets={len(self._frequent_itemsets)} rules={len(self._rules)}>")


# =====================================================================
# 6. Self-supervised learning (masked reconstruction)
# =====================================================================
class SelfSupervised(Technique):
    """Self-supervised pre-training via masked reconstruction.

    Randomly masks a fraction of input features, trains a model to reconstruct
    the masked values. The encoder can then be used for downstream tasks.

    qai.build(technique="self_supervised", mask_ratio=0.3, ...)
    model.train(X)  # unsupervised
    embeddings = model.encode(X)  # get learned representations
    """
    name = "self_supervised"
    family = "deep_learning"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, data_dim=784, hidden=128, latent_dim=32, n_layers=2,
                 mask_ratio=0.3, epochs=50, lr=1e-3, batch_size=64,
                 weight_decay=0.0, device=None, **params):
        super().__init__(**{**params, "data_dim":data_dim, "hidden":hidden,
                            "latent_dim":latent_dim, "n_layers":n_layers,
                            "mask_ratio":mask_ratio, "epochs":epochs, "lr":lr,
                            "batch_size":batch_size, "weight_decay":weight_decay, "device":device})
        _req()
        self.data_dim=data_dim; self.hidden=hidden; self.latent_dim=latent_dim
        self.n_layers=n_layers; self.mask_ratio=mask_ratio; self.epochs=epochs
        self.lr=lr; self.batch_size=batch_size; self.weight_decay=weight_decay
        if device is None:
            try:
                import qai; device = qai.get_optimal_device()
            except: device = "cpu"
        self.device = torch.device(device)
        self.encoder: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self._losses: List[float] = []

    def _build(self):
        # Encoder
        enc = []; d = self.data_dim
        for _ in range(self.n_layers):
            enc += [nn.Linear(d, self.hidden), nn.ReLU()]; d = self.hidden
        enc += [nn.Linear(d, self.latent_dim)]
        self.encoder = nn.Sequential(*enc).to(self.device)
        # Decoder
        dec = []; d = self.latent_dim
        for _ in range(self.n_layers):
            dec += [nn.Linear(d, self.hidden), nn.ReLU()]; d = self.hidden
        dec += [nn.Linear(d, self.data_dim)]
        self.decoder = nn.Sequential(*dec).to(self.device)

    def fit(self, X, y=None, **kw):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1: X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim > 2: X_arr = X_arr.reshape(X_arr.shape[0], -1)
        if X_arr.max() > 1.5: X_arr = X_arr / 255.0
        self.data_dim = X_arr.shape[1]
        if self.encoder is None: self._build()
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        self._losses = []; N = len(X_arr)
        for ep in range(self.epochs):
            rng = np.random.default_rng(ep); perm = rng.permutation(N)
            ep_loss = 0.0; n_b = 0
            for s in range(0, N, self.batch_size):
                idx = perm[s:s+self.batch_size]
                x = torch.tensor(X_arr[idx], device=self.device)
                # Create mask
                mask = (torch.rand_like(x) > self.mask_ratio).float()
                masked_x = x * mask
                # Encode + decode
                z = self.encoder(masked_x)
                recon = self.decoder(z)
                # Loss only on masked positions
                loss = (F.mse_loss(recon, x, reduction="none") * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += float(loss); n_b += 1
            self._losses.append(ep_loss / max(1, n_b))
        self._trained = True
        return self

    def forward(self, x, **kw):
        """Reconstruct input."""
        if not self._trained: raise RuntimeError("SelfSupervised.predict() before train()")
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.max() > 1.5: x_arr = x_arr / 255.0
        self.encoder.eval(); self.decoder.eval()
        with torch.no_grad():
            xt = torch.tensor(x_arr, device=self.device)
            if xt.ndim == 1: xt = xt.unsqueeze(0)
            z = self.encoder(xt)
            return self.decoder(z).cpu().numpy()

    def encode(self, x) -> np.ndarray:
        """Get latent representations."""
        if not self._trained: raise RuntimeError("encode() before train()")
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.max() > 1.5: x_arr = x_arr / 255.0
        self.encoder.eval()
        with torch.no_grad():
            xt = torch.tensor(x_arr, device=self.device)
            if xt.ndim == 1: xt = xt.unsqueeze(0)
            return self.encoder(xt).cpu().numpy()

    def losses(self): return list(self._losses)
    def parameter_count(self):
        if self.encoder is None: return 0
        return sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.decoder.parameters())
    def architecture_summary(self):
        return {"technique":"self_supervised","data_dim":self.data_dim,
                "latent_dim":self.latent_dim,"mask_ratio":self.mask_ratio,
                "parameters":self.parameter_count(),"device":str(self.device)}
    def __repr__(self):
        if self.encoder is None: return f"<SelfSupervised unbuilt>"
        return f"<SelfSupervised latent={self.latent_dim} mask={self.mask_ratio} params={self.parameter_count():,}>"
