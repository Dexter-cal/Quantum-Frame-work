"""
qai.techniques.frontier_generative -- cutting-edge generative architectures
discovered missing during the 2026 web completeness audit.

Built here:
  qai.build(technique="normalizing_flow", ...)   — RealNVP-style invertible flow (exact likelihood)
  qai.build(technique="flow_matching", ...)       — Continuous-time flow matching (Lipman et al. 2023)
  qai.build(technique="consistency_model", ...)   — Song et al. 2023 one-step generators
  qai.build(technique="energy_based", ...)        — Energy-based model with Langevin sampling
  qai.build(technique="score_based", ...)         — Score-based generative model (NCSN/DSM)
  qai.build(technique="rbm", ...)                 — Restricted Boltzmann Machine (CD-1)
  qai.build(technique="boltzmann", ...)           — Full Boltzmann machine

These complement the existing GAN/VAE/Diffusion trio with the four other major
families of probabilistic generative models.
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
            "[WHAT] Frontier generative techniques require PyTorch\n"
            "[FIX] pip install torch"
        )


# =====================================================================
# 1. NORMALIZING FLOW (RealNVP)
# =====================================================================
class _CouplingLayer(nn.Module):
    """RealNVP affine coupling layer."""
    def __init__(self, dim, hidden=64, mask=None):
        super().__init__()
        self.dim = dim
        if mask is None:
            mask = (torch.arange(dim) % 2 == 0).float()
        self.register_buffer("mask", mask)
        hidden_layer = 64
        self.s_net = nn.Sequential(
            nn.Linear(dim, hidden_layer), nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer), nn.ReLU(),
            nn.Linear(hidden_layer, dim), nn.Tanh(),
        )
        self.t_net = nn.Sequential(
            nn.Linear(dim, hidden_layer), nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer), nn.ReLU(),
            nn.Linear(hidden_layer, dim),
        )

    def forward(self, x, reverse=False):
        m = self.mask * x
        s = self.s_net(m) * (1 - self.mask)
        t = self.t_net(m) * (1 - self.mask)
        if not reverse:
            y = m + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = s.sum(dim=-1)
        else:
            y = m + (1 - self.mask) * ((x - t) * torch.exp(-s))
            log_det = -s.sum(dim=-1)
        return y, log_det


class _FlowNet(nn.Module):
    def __init__(self, dim, n_layers=6, hidden=64):
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            mask = (torch.arange(dim) + i) % 2 == 0
            mask = mask.float()
            layers.append(_CouplingLayer(dim, hidden, mask))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        log_det_total = 0.0
        z = x
        for layer in self.layers:
            z, ld = layer(z, reverse=False)
            log_det_total = log_det_total + ld
        return z, log_det_total

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x, _ = layer(x, reverse=True)
        return x


class NormalizingFlow(Technique):
    """RealNVP normalizing flow with exact log-likelihood.

    fit(X) trains the flow on real data; sample(n) draws new samples by
    sampling from a standard normal prior and running the inverse pass.
    """
    name = "normalizing_flow"
    family = "generative"

    def __init__(self, dim: int = 2, n_layers: int = 6, hidden: int = 64,
                 lr: float = 1e-3, epochs: int = 200, **kw):
        super().__init__(dim=dim, n_layers=n_layers, hidden=hidden,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.net = _FlowNet(dim, n_layers, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X, y=None):
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X.dim() == 1:
            X = X.unsqueeze(0)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            z, log_det = self.net(X)
            log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(-1)
            ll = (log_pz + log_det).mean()
            (-ll).backward()
            self.opt.step()
        self._trained = True
        return self

    def log_likelihood(self, X):
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            z, log_det = self.net(X)
            log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(-1)
            return (log_pz + log_det).numpy()

    def sample(self, n: int):
        with torch.no_grad():
            z = torch.randn(n, self.dim)
            x = self.net.inverse(z)
        return x.numpy()

    def forward(self, x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            z, ld = self.net(x.unsqueeze(0))
        return z.squeeze(0).numpy()


# =====================================================================
# 2. FLOW MATCHING (Lipman et al. 2023)
# =====================================================================
class _FlowMatchingNet(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1))


class FlowMatching(Technique):
    """Continuous-time flow matching.

    Learns a velocity field v(x,t) that transports samples from a standard
    Gaussian prior to the data distribution along straight-line ODE paths.
    fit(X) trains the field; sample(n) integrates with Euler ODE.
    """
    name = "flow_matching"
    family = "generative"

    def __init__(self, dim: int = 2, hidden: int = 64, lr: float = 1e-3,
                 epochs: int = 300, n_steps: int = 50, **kw):
        super().__init__(dim=dim, hidden=hidden, lr=lr, epochs=epochs,
                         n_steps=n_steps, **kw)
        _req()
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.n_steps = n_steps
        self.net = _FlowMatchingNet(dim, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X, y=None):
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X.dim() == 1:
            X = X.unsqueeze(0)
        N = X.shape[0]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            t = torch.rand(N, 1)
            eps = torch.randn_like(X)
            x_t = (1 - t) * eps + t * X
            target = X - eps
            pred = self.net(x_t, t.squeeze(-1))
            loss = F.mse_loss(pred, target)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def sample(self, n: int):
        with torch.no_grad():
            x = torch.randn(n, self.dim)
            dt = 1.0 / self.n_steps
            for i in range(self.n_steps):
                t = torch.full((n,), i * dt)
                v = self.net(x, t)
                x = x + v * dt
        return x.numpy()

    def forward(self, x):
        return self.sample(1)[0]


# =====================================================================
# 3. CONSISTENCY MODEL (Song et al. 2023) — one-step generation
# =====================================================================
class _ConsistencyNet(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1))


class ConsistencyModel(Technique):
    """Consistency model — single-step generator distilled from flow.

    The consistency function f(x_t, t) is constant along PF-ODE trajectories,
    so any x_t maps back to the same x_0. Training: pick two adjacent points
    on the trajectory and enforce consistency. Sampling: one forward eval.
    """
    name = "consistency_model"
    family = "generative"

    def __init__(self, dim: int = 2, hidden: int = 128, lr: float = 1e-3,
                 epochs: int = 400, sigma_data: float = 1.0, **kw):
        super().__init__(dim=dim, hidden=hidden, lr=lr, epochs=epochs,
                         sigma_data=sigma_data, **kw)
        _req()
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.sigma_data = sigma_data
        self.net = _ConsistencyNet(dim, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X, y=None):
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X.dim() == 1:
            X = X.unsqueeze(0)
        N = X.shape[0]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            t = torch.rand(N, 1) * 80 + 0.002  # in [0.002, 80]
            t_next = torch.maximum(t - 1.0, torch.tensor(0.002))
            eps = torch.randn_like(X) * t
            x_t = X + eps
            x_t_next = X + eps * (t_next / t)
            f1 = self.net(x_t, t.squeeze(-1))
            f2 = self.net(x_t_next, t_next.squeeze(-1))
            loss = F.mse_loss(f1, f2.detach())
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def sample(self, n: int):
        with torch.no_grad():
            x = torch.randn(n, self.dim) * 80
            for _ in range(10):
                t = torch.full((n,), 80.0)
                x = self.net(x, t)
                x = x + torch.randn_like(x) * 0.1
        return x.numpy()

    def forward(self, x):
        return self.sample(1)[0]


# =====================================================================
# 4. ENERGY-BASED MODEL (Langevin sampling)
# =====================================================================
class _EBMNet(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EnergyBasedModel(Technique):
    """Energy-based model trained with contrastive divergence + Langevin sampling.

    The model assigns a scalar energy E(x); training minimises E(real) and
    maximises E(fake) where fake samples are produced by Langevin MCMC.
    """
    name = "energy_based"
    family = "generative"

    def __init__(self, dim: int = 2, hidden: int = 128, lr: float = 1e-3,
                 epochs: int = 300, langevin_steps: int = 20,
                 langevin_lr: float = 1.0, **kw):
        super().__init__(dim=dim, hidden=hidden, lr=lr, epochs=epochs,
                         langevin_steps=langevin_steps, langevin_lr=langevin_lr, **kw)
        _req()
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.langevin_steps = langevin_steps
        self.langevin_lr = langevin_lr
        self.net = _EBMNet(dim, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X, y=None):
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X.dim() == 1:
            X = X.unsqueeze(0)
        N = X.shape[0]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            real_energy = self.net(X).mean()
            x_fake = torch.randn_like(X)
            x_fake.requires_grad_(True)
            for _ in range(self.langevin_steps):
                e = self.net(x_fake).sum()
                grad = torch.autograd.grad(e, x_fake)[0]
                x_fake = (x_fake - 0.5 * self.langevin_lr * grad
                          + torch.randn_like(x_fake) * math.sqrt(self.langevin_lr)).detach()
                x_fake.requires_grad_(True)
            fake_energy = self.net(x_fake.detach()).mean()
            loss = real_energy - fake_energy + 1e-2 * (real_energy ** 2 + fake_energy ** 2)
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def sample(self, n: int):
        with torch.no_grad():
            x = torch.randn(n, self.dim)
        x.requires_grad_(True)
        for _ in range(self.langevin_steps):
            e = self.net(x).sum()
            grad = torch.autograd.grad(e, x)[0]
            x = (x - 0.5 * self.langevin_lr * grad
                 + torch.randn_like(x) * math.sqrt(self.langevin_lr)).detach()
            x.requires_grad_(True)
        return x.detach().numpy()

    def forward(self, x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            return float(self.net(x.unsqueeze(0)).item())


# =====================================================================
# 5. SCORE-BASED GENERATIVE MODEL (NCSN / DSM)
# =====================================================================
class _ScoreNet(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, sigma):
        return self.net(torch.cat([x, sigma.unsqueeze(-1)], dim=-1))


class ScoreBasedModel(Technique):
    """Score-based generative model with denoising score matching.

    Trains a noise-conditional score network s(x, sigma) to predict the
    gradient of the data log-density at multiple noise scales. Samples are
    drawn via annealed Langevin dynamics.
    """
    name = "score_based"
    family = "generative"

    def __init__(self, dim: int = 2, hidden: int = 128, lr: float = 1e-3,
                 epochs: int = 300, sigmas=(0.1, 0.3, 1.0),
                 n_langevin: int = 30, **kw):
        super().__init__(dim=dim, hidden=hidden, lr=lr, epochs=epochs,
                         sigmas=sigmas, n_langevin=n_langevin, **kw)
        _req()
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.sigmas = list(sigmas)
        self.n_langevin = n_langevin
        self.net = _ScoreNet(dim, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X, y=None):
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X.dim() == 1:
            X = X.unsqueeze(0)
        N = X.shape[0]
        sigmas = torch.tensor(self.sigmas)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            sigma = sigmas[torch.randint(0, len(sigmas), (N,))]
            eps = torch.randn_like(X) * sigma.unsqueeze(-1)
            x_noisy = X + eps
            target_score = -eps / (sigma.unsqueeze(-1) ** 2)
            pred_score = self.net(x_noisy, sigma)
            loss = ((pred_score - target_score) ** 2).sum(-1).mean()
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def sample(self, n: int):
        with torch.no_grad():
            x = torch.randn(n, self.dim) * self.sigmas[-1]
        x.requires_grad_(True)
        for sigma in reversed(self.sigmas):
            alpha = 1e-2 * sigma ** 2 / self.sigmas[-1] ** 2
            for _ in range(self.n_langevin):
                e = self.net(x, torch.full((n,), sigma)).sum()
                grad = torch.autograd.grad(e, x)[0]
                x = (x + alpha * grad
                     + torch.randn_like(x) * math.sqrt(2 * alpha)).detach()
                x.requires_grad_(True)
        return x.detach().numpy()

    def forward(self, x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            return self.net(x.unsqueeze(0), torch.tensor([1.0])).squeeze(0).numpy()


# =====================================================================
# 6. RESTRICTED BOLTZMANN MACHINE (CD-1)
# =====================================================================
class _RBMNet(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_visible))
        self.c = nn.Parameter(torch.zeros(n_hidden))

    def v_to_h(self, v):
        return torch.sigmoid(F.linear(v, self.W, self.c))

    def h_to_v(self, h):
        return torch.sigmoid(F.linear(h, self.W.t(), self.b))


class RestrictedBoltzmannMachine(Technique):
    """RBM trained with 1-step Contrastive Divergence (Hinton 2002).

    A bipartite undirected graphical model. Visible units v and hidden units h
    are conditionally independent given each other. CD-1 reconstructs visible
    activations and applies the standard weight update.
    """
    name = "rbm"
    family = "generative"

    def __init__(self, n_visible: int = 16, n_hidden: int = 8,
                 lr: float = 0.1, epochs: int = 100, **kw):
        super().__init__(n_visible=n_visible, n_hidden=n_hidden,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        self.net = _RBMNet(n_visible, n_hidden)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr)

    def fit(self, X, y=None):
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X.dim() == 1:
            X = X.unsqueeze(0)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            v0 = (X > torch.rand_like(X)).float()
            h0_prob = self.net.v_to_h(v0)
            h0_sample = (h0_prob > torch.rand_like(h0_prob)).float()
            v1_prob = self.net.h_to_v(h0_sample)
            v1_sample = (v1_prob > torch.rand_like(v1_prob)).float()
            h1_prob = self.net.v_to_h(v1_sample)
            pos = torch.matmul(h0_prob.t(), v0)
            neg = torch.matmul(h1_prob.t(), v1_sample)
            grad_W = (neg - pos) / v0.shape[0]
            grad_b = v1_sample.mean(0) - v0.mean(0)
            grad_c = h1_prob.mean(0) - h0_prob.mean(0)
            self.net.W.grad = grad_W
            self.net.b.grad = grad_b
            self.net.c.grad = grad_c
            self.opt.step()
        self._trained = True
        return self

    def sample(self, n: int, k: int = 100):
        with torch.no_grad():
            v = (torch.rand(n, self.n_visible) > 0.5).float()
            for _ in range(k):
                h = self.net.v_to_h(v)
                h = (h > torch.rand_like(h)).float()
                v = self.net.h_to_v(h)
                v = (v > torch.rand_like(v)).float()
        return v.numpy()

    def forward(self, x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            h = self.net.v_to_h(x.unsqueeze(0))
            v_recon = self.net.h_to_v(h)
        return v_recon.squeeze(0).numpy()


# =====================================================================
# 7. FULL BOLTZMANN MACHINE (with simulated annealing sampling)
# =====================================================================
class BoltzmannMachine(Technique):
    """Fully-connected Boltzmann machine (all units symmetrically coupled).

    Trained with persistent CD; samples via Gibbs sweeps. Slower than RBM
    but more expressive — every unit is connected to every other.
    """
    name = "boltzmann"
    family = "generative"

    def __init__(self, n_units: int = 8, lr: float = 0.05,
                 epochs: int = 50, **kw):
        super().__init__(n_units=n_units, lr=lr, epochs=epochs, **kw)
        _req()
        self.n_units = n_units
        self.lr = lr
        self.epochs = epochs
        self.W = torch.zeros(n_units, n_units)
        self.b = torch.zeros(n_units)

    def _gibbs_step(self, s):
        for i in range(self.n_units):
            e = (self.W[i] * s).sum() + self.b[i]
            s[i] = 1.0 if torch.sigmoid(e).item() > random.random() else 0.0
        return s

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        for ep in range(self.epochs):
            for v in X:
                s = torch.tensor(v.copy())
                s_pos = s.clone()
                for _ in range(5):
                    s = self._gibbs_step(s)
                s_neg = s.clone()
                for i in range(self.n_units):
                    for j in range(i + 1, self.n_units):
                        self.W[i, j] += self.lr * (s_pos[i] * s_pos[j] - s_neg[i] * s_neg[j])
                        self.W[j, i] = self.W[i, j]
                    self.b[i] += self.lr * (s_pos[i] - s_neg[i])
        self._trained = True
        return self

    def sample(self, n: int, k: int = 50):
        out = []
        for _ in range(n):
            s = (torch.rand(self.n_units) > 0.5).float()
            for _ in range(k):
                s = self._gibbs_step(s)
            out.append(s.numpy())
        return np.array(out)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        s = torch.tensor(x.copy())
        for _ in range(5):
            s = self._gibbs_step(s)
        return s.numpy()
