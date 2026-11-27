"""
qai.techniques.frontier_classical -- historically important neural architectures
that were missing from the 1.0 release.

  qai.build(technique="som", ...)        — Self-Organizing Map (Kohonen 1982)
  qai.build(technique="capsule", ...)    — Capsule Network (Sabour 2017)
  qai.build(technique="echo_state", ...) — Echo State Network (Jaeger 2001)
  qai.build(technique="elm", ...)        — Extreme Learning Machine (Huang 2006)
  qai.build(technique="neural_gas", ...) — Growing Neural Gas (Martinetz 1991)
  qai.build(technique="art", ...)        — Adaptive Resonance Theory (Carpenter 1987)
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
    from qai._torch_stub import _TorchStubNamespace, _NNStubNamespace
    torch = _TorchStubNamespace()
    nn = _NNStubNamespace()
    F = nn

from .base import Technique

def _req():
    if not _TORCH:
        raise ImportError(
            "[WHAT] Frontier classical techniques require PyTorch\n"
            "[FIX] pip install torch"
        )


# =====================================================================
# 1. SELF-ORGANIZING MAP (Kohonen 1982)
# =====================================================================
class SelfOrganizingMap(Technique):
    """SOM — competitive, topology-preserving unsupervised map.

    Each neuron w_i is moved toward the input x proportional to its grid
    distance from the Best Matching Unit (BMU). Neighbourhood shrinks over
    time. Useful for visualisation, clustering, and dimensionality reduction.
    """
    name = "som"
    family = "unsupervised"

    def __init__(self, grid_h: int = 8, grid_w: int = 8, dim: int = 4,
                 lr0: float = 0.5, sigma0: float = 4.0,
                 epochs: int = 100, **kw):
        super().__init__(grid_h=grid_h, grid_w=grid_w, dim=dim,
                         lr0=lr0, sigma0=sigma0, epochs=epochs, **kw)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.dim = dim
        self.lr0 = lr0
        self.sigma0 = sigma0
        self.epochs = epochs
        rng = np.random.default_rng(0)
        self.weights = rng.standard_normal((grid_h * grid_w, dim)).astype(np.float32)
        # cache neuron grid coordinates
        gy, gx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
        self.grid = np.stack([gy.flatten(), gx.flatten()], axis=1).astype(np.float32)

    def _find_bmu(self, x):
        d = ((self.weights - x) ** 2).sum(-1)
        return int(d.argmin())

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        N = X.shape[0]
        n_iter = self.epochs * N
        for it in range(n_iter):
            t = it / n_iter
            lr = self.lr0 * math.exp(-t)
            sigma = max(self.sigma0 * math.exp(-t), 0.1)
            x = X[random.randint(0, N - 1)]
            bmu = self._find_bmu(x)
            d_grid = ((self.grid - self.grid[bmu]) ** 2).sum(-1)
            influence = np.exp(-d_grid / (2 * sigma ** 2)).astype(np.float32)
            self.weights += (lr * influence[:, None]) * (x - self.weights)
        self._trained = True
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        bmus = [self._find_bmu(x) for x in X]
        return np.array(bmus)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        return self.weights[self._find_bmu(x)]


# =====================================================================
# 2. CAPSULE NETWORK (Sabour et al. 2017) — dynamic routing
# =====================================================================
class _PrimaryCaps(nn.Module):
    def __init__(self, in_dim, n_caps, cap_dim):
        super().__init__()
        self.caps = nn.ModuleList([nn.Linear(in_dim, cap_dim) for _ in range(n_caps)])
        self.n_caps = n_caps
        self.cap_dim = cap_dim

    def forward(self, x):
        outs = torch.stack([cap(x) for cap in self.caps], dim=-2)
        return outs  # (B, n_caps, cap_dim)


class _Squash(nn.Module):
    def forward(self, x):
        norm = (x ** 2).sum(-1, keepdim=True)
        return norm / (1 + norm) * x / (norm.sqrt() + 1e-8)


class _DigitCaps(nn.Module):
    def __init__(self, n_in, dim_in, n_out, dim_out, n_iter=3):
        super().__init__()
        self.n_in, self.dim_in = n_in, dim_in
        self.n_out, self.dim_out = n_out, dim_out
        self.n_iter = n_iter
        self.W = nn.Parameter(torch.randn(n_in, n_out, dim_in, dim_out) * 0.01)

    def forward(self, x):
        # x: (B, n_in, dim_in)
        B = x.shape[0]
        x = x.unsqueeze(2).expand(-1, -1, self.n_out, -1).unsqueeze(-1)
        u_hat = (self.W.unsqueeze(0) * x).sum(-2)  # (B, n_in, n_out, dim_out)
        b = torch.zeros(B, self.n_in, self.n_out)
        for _ in range(self.n_iter):
            c = F.softmax(b, dim=-1)
            s = (c.unsqueeze(-1) * u_hat).sum(1)  # (B, n_out, dim_out)
            v = _Squash()(s)
            if self.n_iter > 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(-1)
        return v  # (B, n_out, dim_out)


class CapsuleNetwork(Technique):
    """Capsule Network with dynamic routing-by-agreement.

    Capsules replace scalar neurons with vector neurons; the length of the
    output vector represents the existence probability. The routing
    algorithm iteratively routes low-level capsules to high-level ones based
    on prediction agreement.
    """
    name = "capsule"
    family = "vision"

    def __init__(self, in_dim: int = 784, n_primary: int = 8,
                 primary_dim: int = 8, n_digit: int = 10,
                 digit_dim: int = 16, n_iter: int = 3,
                 lr: float = 1e-3, epochs: int = 50, **kw):
        super().__init__(in_dim=in_dim, n_primary=n_primary,
                         primary_dim=primary_dim, n_digit=n_digit,
                         digit_dim=digit_dim, n_iter=n_iter,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.n_digit = in_dim, n_digit
        self.lr, self.epochs = lr, epochs
        self.fc = nn.Linear(in_dim, 256)
        self.primary = _PrimaryCaps(256, n_primary, primary_dim)
        self.digit = _DigitCaps(n_primary, primary_dim, n_digit, digit_dim, n_iter)
        self.opt = torch.optim.Adam(
            list(self.fc.parameters()) + list(self.primary.parameters())
            + list(self.digit.parameters()), lr=lr)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        # Flatten image input (N, C, H, W) -> (N, C*H*W) so the first Linear
        # layer (in_dim) can consume it. Bug fix: previously the model crashed
        # when given 4D image data because Linear was applied to the wrong axis.
        if X.ndim == 4:
            X = X.reshape(X.shape[0], -1)
        y = np.asarray(y).astype(np.int64)
        N = X.shape[0]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            xt = torch.from_numpy(X)
            yt = torch.from_numpy(y)
            h = F.relu(self.fc(xt))
            p = self.primary(h)
            v = self.digit(p)
            # margin loss with one-hot target
            norms = (v ** 2).sum(-1).sqrt()  # (N, n_digit)
            onehot = F.one_hot(yt, num_classes=self.n_digit).float()
            loss = (onehot * F.relu(0.9 - norms) ** 2
                    + 0.5 * (1 - onehot) * F.relu(norms - 0.1) ** 2).sum(-1).mean()
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        # Flatten image input (C, H, W) -> (C*H*W) for single-sample inference.
        if x.ndim == 3:
            x = x.reshape(-1)
        x = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            h = F.relu(self.fc(x))
            p = self.primary(h)
            v = self.digit(p)
            norms = (v ** 2).sum(-1).sqrt()
        return norms.squeeze(0).argmax().item()


# =====================================================================
# 3. ECHO STATE NETWORK (Jaeger 2001) — reservoir computing
# =====================================================================
class EchoStateNetwork(Technique):
    """ESN — large random reservoir with only the readout trained.

    The reservoir is a sparse recurrent network with spectral radius < 1
    (echo state property). Only the linear readout W_out is trained via
    ridge regression. Much faster than training an RNN end-to-end.
    """
    name = "echo_state"
    family = "reservoir"

    def __init__(self, in_dim: int = 1, n_reservoir: int = 200,
                 spectral_radius: float = 0.9, sparsity: float = 0.2,
                 ridge_alpha: float = 1e-6, **kw):
        super().__init__(in_dim=in_dim, n_reservoir=n_reservoir,
                         spectral_radius=spectral_radius, sparsity=sparsity,
                         ridge_alpha=ridge_alpha, **kw)
        self.in_dim = in_dim
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.ridge_alpha = ridge_alpha
        rng = np.random.default_rng(0)
        # reservoir weight matrix
        W = rng.standard_normal((n_reservoir, n_reservoir))
        mask = rng.random((n_reservoir, n_reservoir)) > self.sparsity
        W *= mask
        eig = max(abs(np.linalg.eigvals(W)).max(), 1e-8)
        W = W * (spectral_radius / eig)
        self.W = W.astype(np.float32)
        self.W_in = rng.standard_normal((n_reservoir, in_dim)).astype(np.float32) * 0.5
        self.W_out = None
        self.state = np.zeros(n_reservoir, dtype=np.float32)

    def _update(self, x):
        x = np.asarray(x, dtype=np.float32)
        z = self.W @ self.state + self.W_in @ x
        self.state = np.tanh(z)
        return self.state.copy()

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        if y.ndim == 1:
            y = y[:, None]
        # warm-up
        states = []
        self.state = np.zeros(self.n_reservoir, dtype=np.float32)
        for t in range(X.shape[0]):
            s = self._update(X[t])
            states.append(s)
        S = np.array(states[10:])  # discard washout
        Y = y[10:]
        # ridge regression
        self.W_out = (np.linalg.inv(S.T @ S + self.ridge_alpha * np.eye(S.shape[1])) @ S.T @ Y).T
        self._trained = True
        return self

    def forward(self, x):
        s = self._update(x)
        if self.W_out is None:
            raise RuntimeError("Call fit() first")
        return self.W_out @ s


# =====================================================================
# 4. EXTREME LEARNING MACHINE (Huang et al. 2006)
# =====================================================================
class ExtremeLearningMachine(Technique):
    """ELM — single hidden layer with random fixed weights; only output trained.

    The hidden layer is randomly initialised once and never updated; the
    output weights are solved analytically with least-squares. Equivalent to
    random feature ridge regression but framed as a feedforward network.
    """
    name = "elm"
    family = "feedforward"

    def __init__(self, in_dim: int = 8, n_hidden: int = 100,
                 activation: str = "relu", ridge_alpha: float = 1e-4, **kw):
        super().__init__(in_dim=in_dim, n_hidden=n_hidden,
                         activation=activation, ridge_alpha=ridge_alpha, **kw)
        self.in_dim = in_dim
        self.n_hidden = n_hidden
        self.activation = activation
        self.ridge_alpha = ridge_alpha
        rng = np.random.default_rng(0)
        self.W = rng.standard_normal((in_dim, n_hidden)).astype(np.float32)
        self.b = rng.standard_normal(n_hidden).astype(np.float32)
        self.beta = None  # output weights

    def _act(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.activation == "tanh":
            return np.tanh(z)
        else:
            return z

    def _hidden(self, X):
        return self._act(X @ self.W + self.b)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        H = self._hidden(X)
        self.beta = (np.linalg.inv(H.T @ H + self.ridge_alpha * np.eye(self.n_hidden)) @ H.T @ y)
        self._trained = True
        return self

    def forward(self, x):
        if self.beta is None:
            raise RuntimeError("Call fit() first")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        H = self._hidden(x)
        return (H @ self.beta).squeeze(0)


# =====================================================================
# 5. GROWING NEURAL GAS (Martinetz 1991)
# =====================================================================
class NeuralGas(Technique):
    """Growing Neural Gas — competitive Hebbian learning with growing units.

    Starts with 2 neurons; periodically inserts new neurons where the error
    is highest. Edges track the co-activation of neurons and age out. Useful
    for online topological clustering.
    """
    name = "neural_gas"
    family = "unsupervised"

    def __init__(self, dim: int = 4, max_units: int = 50,
                 eps_b: float = 0.2, eps_n: float = 0.05,
                 alpha: float = 0.5, lam: int = 100, **kw):
        super().__init__(dim=dim, max_units=max_units, eps_b=eps_b, eps_n=eps_n,
                         alpha=alpha, lam=lam, **kw)
        self.dim = dim
        self.max_units = max_units
        self.eps_b, self.eps_n = eps_b, eps_n
        self.alpha = alpha
        self.lam = lam
        rng = np.random.default_rng(0)
        self.units = [rng.standard_normal(dim).astype(np.float32) for _ in range(2)]
        self.errors = [0.0, 0.0]
        self.edges = {}  # (i,j) -> age

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        for it, x in enumerate(X):
            d = [((u - x) ** 2).sum() for u in self.units]
            order = np.argsort(d)
            s1, s2 = int(order[0]), int(order[1])
            self.units[s1] += self.eps_b * (x - self.units[s1])
            for j in range(len(self.units)):
                if j == s1:
                    continue
                if (min(s1, j), max(s1, j)) in self.edges:
                    self.units[j] += self.eps_n * (x - self.units[j])
            self.edges[(min(s1, s2), max(s1, s2))] = 0
            to_remove = []
            for k in list(self.edges.keys()):
                self.edges[k] += 1
                if self.edges[k] > 50:
                    to_remove.append(k)
            for k in to_remove:
                del self.edges[k]
            self.errors[s1] += d[0]
            for i in range(len(self.errors)):
                self.errors[i] *= self.alpha
            if it > 0 and it % self.lam == 0 and len(self.units) < self.max_units:
                q = int(np.argmax(self.errors))
                # find neighbour with edge
                nb = None
                for (a, b) in self.edges:
                    if a == q:
                        nb = b; break
                    if b == q:
                        nb = a; break
                if nb is not None:
                    new = (self.units[q] + self.units[nb]) / 2
                    self.units.append(new.astype(np.float32))
                    self.errors.append((self.errors[q] + self.errors[nb]) / 2)
                    self.errors[q] *= self.alpha
                    self.errors[nb] *= self.alpha
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        d = [((u - x) ** 2).sum() for u in self.units]
        return self.units[int(np.argmin(d))]


# =====================================================================
# 6. ADAPTIVE RESONANCE THEORY (Carpenter & Grossberg 1987)
# =====================================================================
class ARTNetwork(Technique):
    """ART-1 — binary Adaptive Resonance Theory network.

    Maintains a set of prototype templates; for each new input, it tries to
    find a template matching beyond the vigilance threshold. If none matches,
    a new template is created.
    """
    name = "art"
    family = "unsupervised"

    def __init__(self, dim: int = 8, vigilance: float = 0.7, **kw):
        super().__init__(dim=dim, vigilance=vigilance, **kw)
        self.dim = dim
        self.vigilance = vigilance
        self.templates = []  # list of binary vectors

    def fit(self, X, y=None):
        X = np.asarray(X)
        for x in X:
            x = (x > 0.5).astype(int)
            if not self.templates:
                self.templates.append(x.copy())
                continue
            sims = [(np.logical_and(x, t).sum() / (np.logical_or(x, t).sum() + 1e-8), i)
                    for i, t in enumerate(self.templates)]
            sims.sort(reverse=True)
            matched = False
            for sim, idx in sims:
                if sim >= self.vigilance:
                    self.templates[idx] = np.logical_and(self.templates[idx], x).astype(int)
                    matched = True
                    break
            if not matched:
                self.templates.append(x.copy())
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x)
        x = (x > 0.5).astype(int)
        if not self.templates:
            return -1
        sims = [(np.logical_and(x, t).sum() / (np.logical_or(x, t).sum() + 1e-8), i)
                for i, t in enumerate(self.templates)]
        sims.sort(reverse=True)
        return sims[0][1] if sims[0][0] >= self.vigilance else -1
