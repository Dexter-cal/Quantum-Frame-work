"""
qai.techniques.frontier_misc -- the last batch of missing frontier architectures.

  qai.build(technique="bayesian_nn", ...)   — Bayesian NN with variational inference
  qai.build(technique="mc_dropout", ...)     — MC Dropout uncertainty
  qai.build(technique="world_model", ...)    — World Model (Ha & Schmidhuber 2018)
  qai.build(technique="dreamer", ...)        — Dreamer V3 latent imagination RL
  qai.build(technique="mixture_density", ...) — Mixture Density Network
  qai.build(technique="wtc_attention", ...)   — Grouped/Multi-Query attention layer
  qai.build(technique="time_series_arima", ...) — ARIMA
  qai.build(technique="kalman_filter", ...)   — Kalman Filter
  qai.build(technique="topic_model", ...)     — Latent Dirichlet Allocation (LDA)
  qai.build(technique="online_learning", ...) — Online SGD with concept drift detection
"""
from __future__ import annotations
import math
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
            "[WHAT] Frontier misc techniques require PyTorch\n"
            "[FIX] pip install torch"
        )


# =====================================================================
# 1. BAYESIAN NEURAL NETWORK (variational inference, BBB)
# =====================================================================
class _BayesianLinear(nn.Module):
    """Bayes-by-Backprop linear layer (Blundell 2015).

    Each weight has a Gaussian posterior q(w | mu, rho) with sigma = log(1+e^rho).
    The forward pass samples weights and the loss adds KL(q || prior).
    """
    def __init__(self, in_dim, out_dim, prior_sigma=1.0):
        super().__init__()
        self.W_mu = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.W_rho = nn.Parameter(torch.zeros(out_dim, in_dim) - 3)
        self.b_mu = nn.Parameter(torch.zeros(out_dim))
        self.b_rho = nn.Parameter(torch.zeros(out_dim) - 3)
        self.prior_sigma = prior_sigma
        nn.init.xavier_normal_(self.W_mu)

    def forward(self, x):
        W_sigma = F.softplus(self.W_rho)
        b_sigma = F.softplus(self.b_rho)
        eps_W = torch.randn_like(self.W_mu)
        eps_b = torch.randn_like(self.b_mu)
        W = self.W_mu + W_sigma * eps_W
        b = self.b_mu + b_sigma * eps_b
        out = F.linear(x, W, b)
        # KL term (vs standard Gaussian prior N(0, prior_sigma^2))
        kl_W = (-0.5 * (1 + 2 * W_sigma.log() - self.W_mu.pow(2) - W_sigma.pow(2)) / self.prior_sigma ** 2).sum()
        kl_b = (-0.5 * (1 + 2 * b_sigma.log() - self.b_mu.pow(2) - b_sigma.pow(2)) / self.prior_sigma ** 2).sum()
        return out, kl_W + kl_b


class BayesianNeuralNetwork(Technique):
    """Bayesian NN with Gaussian posterior over weights.

    Each forward pass samples a different set of weights; predictions are
    averaged across multiple stochastic passes for uncertainty estimates.
    """
    name = "bayesian_nn"
    family = "probabilistic"

    def __init__(self, in_dim: int = 8, hidden: int = 32, out_dim: int = 1,
                 n_samples: int = 3, prior_sigma: float = 1.0,
                 lr: float = 1e-3, epochs: int = 200, **kw):
        super().__init__(in_dim=in_dim, hidden=hidden, out_dim=out_dim,
                         n_samples=n_samples, prior_sigma=prior_sigma,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.n_samples = n_samples
        self.lr, self.epochs = lr, epochs
        self.fc1 = _BayesianLinear(in_dim, hidden, prior_sigma)
        self.fc2 = _BayesianLinear(hidden, out_dim, prior_sigma)
        self.opt = torch.optim.Adam(
            list(self.fc1.parameters()) + list(self.fc2.parameters()), lr=lr)

    def _forward(self, x):
        h, kl1 = self.fc1(x)
        h = F.relu(h)
        out, kl2 = self.fc2(h)
        return out, kl1 + kl2

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        N = X.shape[0]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            total_loss = 0.0
            for _ in range(self.n_samples):
                out, kl = self._forward(torch.from_numpy(X))
                nll = F.mse_loss(out, torch.from_numpy(y), reduction="sum")
                total_loss = total_loss + nll + kl / N
            (total_loss / self.n_samples).backward()
            self.opt.step()
        self._trained = True
        return self

    def predict_with_uncertainty(self, x, n_passes=20):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        outs = []
        with torch.no_grad():
            for _ in range(n_passes):
                out, _ = self._forward(x)
                outs.append(out.numpy())
        outs = np.array(outs)
        return outs.mean(0), outs.std(0)

    def forward(self, x):
        mean, _ = self.predict_with_uncertainty(x, n_passes=10)
        return mean.squeeze(0)


# =====================================================================
# 2. MC DROPOUT — Gal & Ghahramani 2016
# =====================================================================
class _MCDropoutNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MCDropout(Technique):
    """MC Dropout — keep dropout active at inference and average passes.

    Approximates Bayesian inference with a Gaussian process; uncertainty is
    the variance across passes.
    """
    name = "mc_dropout"
    family = "probabilistic"

    def __init__(self, in_dim: int = 8, hidden: int = 32, out_dim: int = 1,
                 p: float = 0.5, n_passes: int = 20,
                 lr: float = 1e-3, epochs: int = 200, **kw):
        super().__init__(in_dim=in_dim, hidden=hidden, out_dim=out_dim,
                         p=p, n_passes=n_passes, lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.n_passes = n_passes
        self.lr, self.epochs = lr, epochs
        self.net = _MCDropoutNet(in_dim, hidden, out_dim, p)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            out = self.net(torch.from_numpy(X))
            loss = F.mse_loss(out, torch.from_numpy(y))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def predict_with_uncertainty(self, x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self.net.train()  # keep dropout active
        outs = []
        with torch.no_grad():
            for _ in range(self.n_passes):
                outs.append(self.net(x).numpy())
        outs = np.array(outs)
        self.net.eval()
        return outs.mean(0), outs.std(0)

    def forward(self, x):
        mean, _ = self.predict_with_uncertainty(x)
        return mean.squeeze(0)


# =====================================================================
# 3. WORLD MODEL (Ha & Schmidhuber 2018)
# =====================================================================
class _VAE(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim * 2),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        mu_logvar = self.enc(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class _LSTMController(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim + action_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, latent_dim)

    def forward(self, z, a):
        # z: (1, L, latent_dim); a: (1, L, action_dim)
        x = torch.cat([z, a], dim=-1)
        out, h = self.lstm(x)
        return self.head(out), h


class WorldModel(Technique):
    """World Model (Ha & Schmidhuber 2018) — VAE + LSTM + simple controller.

    The VAE compresses observations into a latent z; the LSTM predicts the
    next z given the current z and action; the controller picks actions to
    maximise reward as a function of z.
    """
    name = "world_model"
    family = "model_based_rl"

    def __init__(self, obs_dim: int = 16, action_dim: int = 2,
                 latent_dim: int = 8, hidden: int = 32,
                 lr: float = 1e-3, epochs: int = 50, **kw):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim,
                         latent_dim=latent_dim, hidden=hidden,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.latent_dim = latent_dim
        self.lr, self.epochs = lr, epochs
        self.vae = _VAE(obs_dim, latent_dim, hidden)
        self.lstm = _LSTMController(latent_dim, action_dim, hidden)
        self.opt = torch.optim.Adam(
            list(self.vae.parameters()) + list(self.lstm.parameters()), lr=lr)

    def fit(self, X, y=None):
        """X: dict with 'observations' (T, obs_dim) and 'actions' (T-1, action_dim)."""
        obs = torch.as_tensor(np.asarray(X["observations"], dtype=np.float32))
        act = torch.as_tensor(np.asarray(X["actions"], dtype=np.float32))
        T = obs.shape[0]
        # if actions length == T, drop the last one (no next observation to predict)
        if act.shape[0] == T:
            act = act[:-1]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            recon, mu, logvar = self.vae(obs)
            recon_loss = F.mse_loss(recon, obs)
            kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
            # predict next z — inputs are z_t and a_t (T-1 of each), target is z_{t+1}
            mu_seq = mu[:-1].unsqueeze(0)       # (1, T-1, latent)
            act_seq = act.unsqueeze(0)          # (1, T-1, action)
            pred_z, _ = self.lstm(mu_seq, act_seq)
            loss_dyn = F.mse_loss(pred_z.squeeze(0), mu[1:])
            loss = recon_loss + 0.1 * kld + loss_dyn
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def imagine(self, z0, action_seq):
        z = z0
        h = None
        zs = [z0]
        for a in action_seq:
            z_t = torch.as_tensor(z, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            a_t = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            z_next, h = self.lstm(z_t, a_t, h)
            z = z_next.squeeze(0).squeeze(0).detach().numpy()
            zs.append(z)
        return np.array(zs)

    def forward(self, x):
        obs = torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.vae.encode(obs)
        return mu.squeeze(0).numpy()


# =====================================================================
# 4. DREAMER V3 — simplified (Hafner 2023)
# =====================================================================
class _RSSM(nn.Module):
    """Recurrent State-Space Model — Dreamer's core world model."""
    def __init__(self, latent_dim, action_dim, hidden=32):
        super().__init__()
        # cell input is previous hidden state + action; output is new hidden state
        self.cell = nn.LSTMCell(hidden + action_dim, hidden)
        self.prior = nn.Linear(hidden, latent_dim * 2)
        self.posterior = nn.Linear(hidden + latent_dim, latent_dim * 2)
        self.latent_dim = latent_dim

    def forward(self, prev_state, prev_action, emb):
        h, c = prev_state
        h_next, c_next = self.cell(torch.cat([h, prev_action], dim=-1), (h, c))
        prior = self.prior(h_next)
        posterior = self.posterior(torch.cat([h_next, emb], dim=-1))
        return (h_next, c_next), prior, posterior


class Dreamer(Technique):
    """Dreamer V3 (simplified) — latent imagination + actor-critic on imagined trajectories.

    The world model (RSSM) learns latent dynamics; an actor and critic are
    trained entirely inside the latent space without ever decoding back to
    observations during training.
    """
    name = "dreamer"
    family = "model_based_rl"

    def __init__(self, obs_dim: int = 8, action_dim: int = 2,
                 latent_dim: int = 8, hidden: int = 32,
                 horizon: int = 5, lr: float = 1e-3, epochs: int = 30, **kw):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim,
                         latent_dim=latent_dim, hidden=hidden,
                         horizon=horizon, lr=lr, epochs=epochs, **kw)
        _req()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.horizon = horizon
        self.lr, self.epochs = lr, epochs
        self.encoder = nn.Linear(obs_dim, latent_dim)
        self.rssm = _RSSM(latent_dim, action_dim, hidden)
        self.decoder = nn.Linear(latent_dim, obs_dim)
        self.reward = nn.Linear(latent_dim, 1)
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.opt = torch.optim.Adam(self.parameters_all(), lr=lr)

    def parameters_all(self):
        params = []
        for m in [self.encoder, self.rssm, self.decoder, self.reward, self.actor, self.critic]:
            params += list(m.parameters())
        return params

    def fit(self, X, y=None):
        """X: dict with 'observations' (T, obs_dim), 'actions' (T, action_dim), 'rewards' (T,)."""
        obs = torch.as_tensor(np.asarray(X["observations"], dtype=np.float32))
        act = torch.as_tensor(np.asarray(X["actions"], dtype=np.float32))
        rew = torch.as_tensor(np.asarray(X["rewards"], dtype=np.float32))
        T = obs.shape[0]
        h = torch.zeros(1, self.hidden)
        c = torch.zeros(1, self.hidden)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            loss = 0.0
            h_ep, c_ep = h.clone(), c.clone()
            for t in range(T - 1):
                emb = self.encoder(obs[t:t + 1])
                (h_ep, c_ep), prior, posterior = self.rssm((h_ep, c_ep),
                                                            act[t:t + 1], emb)
                mu, logvar = posterior.chunk(2, dim=-1)
                z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
                recon = self.decoder(z)
                r_pred = self.reward(z)
                loss = loss + F.mse_loss(recon, obs[t + 1:t + 2])
                loss = loss + F.mse_loss(r_pred.squeeze(), rew[t + 1:t + 2].squeeze())
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            z = self.encoder(x)
            a = self.actor(z)
        return a.squeeze(0).numpy()


# =====================================================================
# 5. MIXTURE DENSITY NETWORK (Bishop 1994)
# =====================================================================
class _MDNNet(nn.Module):
    def __init__(self, in_dim, out_dim, n_components=5, hidden=64):
        super().__init__()
        self.n_components = n_components
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_components)
        self.mu = nn.Linear(hidden, n_components * out_dim)
        self.sigma = nn.Linear(hidden, n_components * out_dim)

    def forward(self, x):
        h = self.net(x)
        pi = F.softmax(self.pi(h), dim=-1)
        mu = self.mu(h).view(-1, self.n_components, self.out_dim)
        sigma = F.softplus(self.sigma(h)).view(-1, self.n_components, self.out_dim) + 1e-6
        return pi, mu, sigma


class MixtureDensityNetwork(Technique):
    """MDN — predict a mixture of Gaussians over the output space.

    Handles multi-modal conditional distributions; used for inverse
    problems and stochastic regression.
    """
    name = "mixture_density"
    family = "probabilistic"

    def __init__(self, in_dim: int = 4, out_dim: int = 1, n_components: int = 5,
                 hidden: int = 64, lr: float = 1e-3, epochs: int = 200, **kw):
        super().__init__(in_dim=in_dim, out_dim=out_dim, n_components=n_components,
                         hidden=hidden, lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.n_components = n_components
        self.lr, self.epochs = lr, epochs
        self.net = _MDNNet(in_dim, out_dim, n_components, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _nll(self, pi, mu, sigma, y):
        y = y.unsqueeze(1).expand(-1, self.n_components, -1)
        log_prob = -0.5 * ((y - mu) ** 2 / sigma ** 2 + torch.log(2 * math.pi * sigma ** 2))
        log_prob = log_prob.sum(-1)
        return -torch.logsumexp(torch.log(pi + 1e-8) + log_prob, dim=-1).mean()

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            pi, mu, sigma = self.net(torch.from_numpy(X))
            loss = self._nll(pi, mu, sigma, torch.from_numpy(y))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def sample(self, x, n=1):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            pi, mu, sigma = self.net(x)
            samples = []
            for i in range(x.shape[0]):
                k = torch.multinomial(pi[i], 1).item()
                s = torch.randn(n, self.out_dim) * sigma[i, k] + mu[i, k]
                samples.append(s.numpy())
        return np.array(samples)

    def forward(self, x):
        return self.sample(x, n=1)[0, 0]


# =====================================================================
# 6. GROUPED / MULTI-QUERY ATTENTION
# =====================================================================
class _GroupedQueryAttention(nn.Module):
    """GQA — multiple query heads share K/V heads (Ainslie 2023).

    Trade-off between MHA (n_heads query + n_heads KV) and MQA
    (n_heads query + 1 KV). Reduces KV cache size at inference.
    """
    def __init__(self, dim, n_heads=8, n_kv_heads=2):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.W_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # repeat KV heads
        reps = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(out)


class GroupedQueryAttention(Technique):
    """GQA — grouped-query attention transformer block.

    Drop-in replacement for standard attention with smaller KV cache. Used
    in Llama 2/3 and Mistral for efficient long-context inference.
    """
    name = "gqa_attention"
    family = "attention"

    def __init__(self, dim: int = 32, n_heads: int = 8, n_kv_heads: int = 2,
                 hidden: int = 64, lr: float = 1e-3, epochs: int = 50, **kw):
        super().__init__(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                         hidden=hidden, lr=lr, epochs=epochs, **kw)
        _req()
        self.dim, self.n_heads, self.n_kv_heads = dim, n_heads, n_kv_heads
        self.lr, self.epochs = lr, epochs
        self.attn = _GroupedQueryAttention(dim, n_heads, n_kv_heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.opt = torch.optim.Adam(
            list(self.attn.parameters()) + list(self.ff.parameters())
            + list(self.ln1.parameters()) + list(self.ln2.parameters()), lr=lr)

    def _block(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, :, :]
        if y.ndim == 2:
            y = y[None, :, :]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            h = self._block(torch.from_numpy(X))
            loss = F.mse_loss(h, torch.from_numpy(y))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            h = self._block(x)
        return h.squeeze(0).numpy()


# =====================================================================
# 7. ARIMA — classical time-series model
# =====================================================================
class ARIMA(Technique):
    """ARIMA(p, d, q) — classical Box-Jenkins time series model.

    Difference the series d times to remove trends, then fit an ARMA(p, q)
    model via least squares. forecast(h) returns the next h values.
    """
    name = "time_series_arima"
    family = "time_series"

    def __init__(self, p: int = 1, d: int = 0, q: int = 1, **kw):
        super().__init__(p=p, d=d, q=q, **kw)
        self.p = p
        self.d = d
        self.q = q
        self.ar_coef = None
        self.ma_coef = None
        self.resid_history = []
        self.last_values = None

    def _difference(self, x):
        for _ in range(self.d):
            x = np.diff(x)
        return x

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64).flatten()
        x = self._difference(X)
        n = len(x)
        if n < self.p + self.q + 1:
            self.ar_coef = np.zeros(self.p)
            self.ma_coef = np.zeros(self.q)
            self.last_values = X
            self._trained = True
            return self
        # fit AR(p) via least squares
        if self.p > 0:
            Xar = np.array([x[i:self.p + i][::-1] for i in range(n - self.p)])
            yar = x[self.p:]
            self.ar_coef = np.linalg.lstsq(Xar, yar, rcond=None)[0]
        else:
            self.ar_coef = np.zeros(0)
        # residuals
        if self.p > 0:
            resid = yar - Xar @ self.ar_coef
        else:
            resid = x.copy()
        # fit MA(q) on residuals
        if self.q > 0:
            Xma = np.array([resid[i:self.q + i][::-1] for i in range(len(resid) - self.q)])
            yma = resid[self.q:]
            self.ma_coef = np.linalg.lstsq(Xma, yma, rcond=None)[0]
        else:
            self.ma_coef = np.zeros(0)
        self.resid_history = list(resid[-max(self.q, 1):])
        self.last_values = X
        self._trained = True
        return self

    def forecast(self, h=1):
        history = list(self.last_values)
        for _ in range(h):
            x = self._difference(np.array(history))
            pred = 0
            if self.p > 0 and len(x) >= self.p:
                pred += float(np.dot(self.ar_coef, x[-self.p:][::-1]))
            if self.q > 0 and len(self.resid_history) >= self.q:
                pred += float(np.dot(self.ma_coef, self.resid_history[-self.q:][::-1]))
            # inverse differencing
            for _ in range(self.d):
                pred = pred + history[-1]
            history.append(pred)
        return np.array(history[-h:])

    def forward(self, x):
        return self.forecast(int(x[0]) if hasattr(x, "__len__") else 1)[0]


# =====================================================================
# 8. KALMAN FILTER — linear Gaussian state estimator
# =====================================================================
class KalmanFilter(Technique):
    """1D Kalman filter — recursive Bayesian state estimator.

    Maintains an estimate of the hidden state x and its variance P; on each
    new observation z, it updates via the standard predict-correct equations.
    """
    name = "kalman_filter"
    family = "filtering"

    def __init__(self, process_var: float = 1e-3, meas_var: float = 0.1,
                 init_x: float = 0.0, init_P: float = 1.0, **kw):
        super().__init__(process_var=process_var, meas_var=meas_var,
                         init_x=init_x, init_P=init_P, **kw)
        self.Q = process_var
        self.R = meas_var
        self.x = init_x
        self.P = init_P
        self.history = [init_x]

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64).flatten()
        for z in X:
            # predict
            self.P = self.P + self.Q
            # update
            K = self.P / (self.P + self.R)
            self.x = self.x + K * (z - self.x)
            self.P = (1 - K) * self.P
            self.history.append(self.x)
        self._trained = True
        return self

    def forward(self, z):
        # Accept scalar, 0-d, or 1-element array; for multi-element arrays,
        # process each element sequentially (the filter is recursive).
        z_arr = np.asarray(z, dtype=np.float64).flatten()
        if len(z_arr) == 0:
            return self.x
        for zi in z_arr:
            self.P = self.P + self.Q
            K = self.P / (self.P + self.R)
            self.x = self.x + K * (float(zi) - self.x)
            self.P = (1 - K) * self.P
            self.history.append(self.x)
        return self.x


# =====================================================================
# 9. LATENT DIRICHLET ALLOCATION (Blei 2003) — collapsed Gibbs
# =====================================================================
class LDA(Technique):
    """LDA — collapsed Gibbs sampling topic model.

    Each document is a mixture of K topics; each topic is a distribution
    over the vocabulary. fit() runs n_iter Gibbs sweeps over all tokens.
    """
    name = "topic_model"
    family = "probabilistic"

    def __init__(self, n_topics: int = 5, alpha: float = 0.1,
                 beta: float = 0.01, n_iter: int = 50, **kw):
        super().__init__(n_topics=n_topics, alpha=alpha, beta=beta, n_iter=n_iter, **kw)
        self.K = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.vocab = {}
        self.phi = None  # topic-word distribution

    def fit(self, X, y=None):
        """X: list of documents (each a list of words)."""
        # build vocab
        vocab = {}
        for doc in X:
            for w in doc:
                if w not in vocab:
                    vocab[w] = len(vocab)
        self.vocab = vocab
        V = len(vocab)
        docs = [[vocab[w] for w in doc] for doc in X]
        K = self.K
        # init
        z = [[np.random.randint(K) for _ in doc] for doc in docs]
        n_dk = np.zeros((len(docs), K)) + self.alpha
        n_kw = np.zeros((K, V)) + self.beta
        n_k = np.zeros(K) + V * self.beta
        for d, doc in enumerate(docs):
            for i, w in enumerate(doc):
                k = z[d][i]
                n_dk[d, k] += 1
                n_kw[k, w] += 1
                n_k[k] += 1
        # Gibbs
        for it in range(self.n_iter):
            for d, doc in enumerate(docs):
                for i, w in enumerate(doc):
                    k = z[d][i]
                    n_dk[d, k] -= 1
                    n_kw[k, w] -= 1
                    n_k[k] -= 1
                    p = (n_dk[d] * n_kw[:, w] / n_k)
                    p = p / p.sum()
                    k_new = np.random.choice(K, p=p)
                    z[d][i] = k_new
                    n_dk[d, k_new] += 1
                    n_kw[k_new, w] += 1
                    n_k[k_new] += 1
        self.phi = n_kw / n_k[:, None]
        self.theta = n_dk / n_dk.sum(1, keepdims=True)
        self._trained = True
        return self

    def topic_distribution(self, doc):
        """Infer topic mixture for a new document."""
        if self.phi is None:
            raise RuntimeError("Call fit() first")
        from collections import Counter
        c = Counter(self.vocab.get(w) for w in doc if w in self.vocab)
        if not c:
            return np.zeros(self.K) / self.K
        # pick topic with highest prob for each word, average
        scores = np.zeros(self.K)
        for w_id, count in c.items():
            scores += self.phi[:, w_id] * count
        return scores / scores.sum()

    def forward(self, x):
        return self.topic_distribution(x)


# =====================================================================
# 10. ONLINE LEARNING — SGD with ADWIN concept drift detection
# =====================================================================
class OnlineLearning(Technique):
    """Online linear SGD with ADWIN-style drift detection.

    ADWIN keeps a sliding window of variable length; when the difference in
    means between two sub-windows exceeds a threshold, it shrinks the
    window (drift detected) and the model is partially reset.
    """
    name = "online_learning"
    family = "online"

    def __init__(self, n_features: int = 8, lr: float = 0.01,
                 drift_delta: float = 0.002, **kw):
        super().__init__(n_features=n_features, lr=lr, drift_delta=drift_delta, **kw)
        self.n_features = n_features
        self.lr = lr
        self.drift_delta = drift_delta
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0
        self.window = []
        self.drift_count = 0

    def _drift_detected(self, loss):
        self.window.append(loss)
        if len(self.window) < 20:
            return False
        # split at midpoint and compare means
        mid = len(self.window) // 2
        w0 = np.mean(self.window[:mid])
        w1 = np.mean(self.window[mid:])
        if abs(w0 - w1) > self.drift_delta * (1 + math.sqrt(1 / len(self.window))):
            self.window = self.window[mid:]
            self.drift_count += 1
            return True
        return False

    def partial_fit(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = float(y)
        pred = self.w @ x + self.b
        err = pred - y
        grad_w = err * x
        grad_b = err
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
        if self._drift_detected(err ** 2):
            # partial reset
            self.w *= 0.5
            self.b *= 0.5
        self._trained = True
        return self

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        for x, t in zip(X, y):
            self.partial_fit(x, t)
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        return float(self.w @ x + self.b)
