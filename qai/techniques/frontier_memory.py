"""
qai.techniques.frontier_memory -- memory-augmented neural architectures.

Three foundational architectures that augment a network with explicit
external memory:

  qai.build(technique="hopfield", ...)    — Modern Hopfield Network
  qai.build(technique="ntm", ...)         — Neural Turing Machine
  qai.build(technique="dnc", ...)         — Differentiable Neural Computer
  qai.build(technique="neural_gpu", ...)  — Neural GPU (Graves et al. 2016)

All three are trainable via the standard fit(X, y) interface and integrate
with qai's bundle/save system.
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
            "[WHAT] Frontier memory techniques require PyTorch\n"
            "[FIX] pip install torch"
        )


# =====================================================================
# 1. MODERN HOPFIELD NETWORK (Ramsauer et al. 2020)
# =====================================================================
class ModernHopfield(Technique):
    """Modern Hopfield Network with continuous states.

    Stores a set of patterns Y in associative memory. retrieve(x) returns the
    pattern that is closest in energy to x via the softmax-weighted update
    rule. Modern Hopfield layers have exponential storage capacity and
    converge in one step.
    """
    name = "hopfield"
    family = "associative_memory"

    def __init__(self, dim: int = 8, beta: float = 8.0, **kw):
        super().__init__(dim=dim, beta=beta, **kw)
        _req()
        self.dim = dim
        self.beta = beta
        self.patterns = None  # (P, dim) tensor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        self.patterns = torch.from_numpy(X)
        self._trained = True
        return self

    def retrieve(self, x, n_iter: int = 1):
        if self.patterns is None:
            raise RuntimeError("Call fit(X) first.")
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for _ in range(n_iter):
            scores = self.beta * (x @ self.patterns.t())
            weights = F.softmax(scores, dim=-1)
            x = weights @ self.patterns
        return x.numpy()

    def energy(self, x):
        if self.patterns is None:
            raise RuntimeError("Call fit(X) first.")
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        scores = self.beta * (x @ self.patterns.t())
        return (-torch.logsumexp(scores, dim=-1) / self.beta
                + 0.5 * (x ** 2).sum(-1)).numpy()

    def forward(self, x):
        return self.retrieve(x)[0]


# =====================================================================
# 2. NEURAL TURING MACHINE (Graves et al. 2014)
# =====================================================================
class _NTMController(nn.Module):
    def __init__(self, in_dim, ctrl_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim + ctrl_dim, ctrl_dim)
        self.fc2 = nn.Linear(ctrl_dim, ctrl_dim)
        self.head = nn.Linear(ctrl_dim, out_dim)

    def forward(self, x, h):
        z = torch.cat([x, h], dim=-1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return self.head(z), z


class _NTMReadHead(nn.Module):
    def __init__(self, ctrl_dim, mem_dim):
        super().__init__()
        self.k = nn.Linear(ctrl_dim, mem_dim)
        self.b = nn.Linear(ctrl_dim, 1)
        self.g = nn.Linear(ctrl_dim, 1)

    def forward(self, ctrl_out, memory, prev_w):
        k = torch.tanh(self.k(ctrl_out))                      # (mem_dim,)
        beta = F.softplus(self.b(ctrl_out).squeeze()) + 1     # scalar
        # content-based addressing
        mem_norm = memory / (memory.norm(dim=-1, keepdim=True) + 1e-8)
        k_norm = k / (k.norm() + 1e-8)
        scores = beta * (mem_norm @ k_norm)                   # (mem_n,)
        wc = F.softmax(scores, dim=-1)                        # (mem_n,)
        g = torch.sigmoid(self.g(ctrl_out).squeeze())
        w = g * wc + (1 - g) * prev_w                         # (mem_n,)
        w = w / (w.sum() + 1e-8)
        r = w @ memory                                        # (mem_dim,)
        return r, w


class _NTMWriteHead(nn.Module):
    def __init__(self, ctrl_dim, mem_dim):
        super().__init__()
        self.k = nn.Linear(ctrl_dim, mem_dim)
        self.b = nn.Linear(ctrl_dim, 1)
        self.g = nn.Linear(ctrl_dim, 1)
        self.e = nn.Linear(ctrl_dim, mem_dim)
        self.a = nn.Linear(ctrl_dim, mem_dim)

    def forward(self, ctrl_out, memory, prev_w):
        k = torch.tanh(self.k(ctrl_out))
        beta = F.softplus(self.b(ctrl_out).squeeze()) + 1
        mem_norm = memory / (memory.norm(dim=-1, keepdim=True) + 1e-8)
        k_norm = k / (k.norm() + 1e-8)
        scores = beta * (mem_norm @ k_norm)
        wc = F.softmax(scores, dim=-1)
        g = torch.sigmoid(self.g(ctrl_out).squeeze())
        w = g * wc + (1 - g) * prev_w
        w = w / (w.sum() + 1e-8)
        e = torch.sigmoid(self.e(ctrl_out))
        a = torch.tanh(self.a(ctrl_out))
        memory = memory * (1 - w.unsqueeze(-1) * e)
        memory = memory + w.unsqueeze(-1) * a
        return memory, w


class NeuralTuringMachine(Technique):
    """Neural Turing Machine — RNN + differentiable external memory tape.

    The controller produces an output and an addressing signal; a read head
    retrieves content from the memory matrix M, and a write head updates it
    via erase + add vectors.
    """
    name = "ntm"
    family = "memory_augmented"

    def __init__(self, in_dim: int = 1, out_dim: int = 1,
                 ctrl_dim: int = 32, mem_n: int = 16, mem_dim: int = 8,
                 lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, out_dim=out_dim, ctrl_dim=ctrl_dim,
                         mem_n=mem_n, mem_dim=mem_dim, lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.ctrl_dim, self.mem_n, self.mem_dim = ctrl_dim, mem_n, mem_dim
        self.lr, self.epochs = lr, epochs
        self.controller = _NTMController(in_dim, ctrl_dim, out_dim)
        self.read_head = _NTMReadHead(ctrl_dim, mem_dim)
        self.write_head = _NTMWriteHead(ctrl_dim, mem_dim)
        self.opt = torch.optim.Adam(
            list(self.controller.parameters())
            + list(self.read_head.parameters())
            + list(self.write_head.parameters()), lr=lr)

    def _reset_memory(self):
        self.memory = torch.zeros(self.mem_n, self.mem_dim)
        self.wr = torch.zeros(self.mem_n)
        self.wr[0] = 1.0
        self.ww = torch.zeros(self.mem_n)
        self.ww[0] = 1.0
        self.h = torch.zeros(1, self.ctrl_dim)

    def _step(self, x):
        ctrl_out, self.h = self.controller(x.unsqueeze(0), self.h)
        r, self.wr = self.read_head(self.h.squeeze(0), self.memory, self.wr)
        self.memory, self.ww = self.write_head(self.h.squeeze(0), self.memory, self.ww)
        out = ctrl_out.squeeze(0) + r[:self.out_dim]
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None] if self.in_dim == 1 else X[None, :]
        if y.ndim == 1:
            y = y[:, None] if self.out_dim == 1 else y[None, :]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            self._reset_memory()
            loss = 0.0
            for t in range(X.shape[0]):
                out = self._step(torch.from_numpy(X[t]))
                loss = loss + F.mse_loss(out, torch.from_numpy(y[t]))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        return self._step(x).detach().numpy()


# =====================================================================
# 3. DIFFERENTIABLE NEURAL COMPUTER (Graves et al. 2016)
# =====================================================================
class DifferentiableNeuralComputer(Technique):
    """DNC — NTM with dynamic memory allocation + temporal linkage matrix.

    Simplified implementation: each step, the controller can read from
    multiple heads using content-based addressing, and writes use a usage
    vector that tracks free memory slots. A linkage matrix L captures write
    order so reads can follow temporal order.
    """
    name = "dnc"
    family = "memory_augmented"

    def __init__(self, in_dim: int = 1, out_dim: int = 1,
                 ctrl_dim: int = 32, mem_n: int = 16, mem_dim: int = 8,
                 n_heads: int = 1, lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, out_dim=out_dim, ctrl_dim=ctrl_dim,
                         mem_n=mem_n, mem_dim=mem_dim, n_heads=n_heads,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.ctrl_dim, self.mem_n, self.mem_dim = ctrl_dim, mem_n, mem_dim
        self.n_heads = n_heads
        self.lr, self.epochs = lr, epochs
        self.controller = nn.LSTMCell(in_dim + n_heads * mem_dim, ctrl_dim)
        self.out_head = nn.Linear(ctrl_dim + n_heads * mem_dim, out_dim)
        self.read_heads = nn.ModuleList([
            nn.Linear(ctrl_dim, mem_dim + 3) for _ in range(n_heads)
        ])
        self.write_head = nn.Linear(ctrl_dim, 2 * mem_dim + 2)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        params = list(self.controller.parameters()) + list(self.out_head.parameters())
        for h in self.read_heads:
            params += list(h.parameters())
        params += list(self.write_head.parameters())
        return params

    def _reset(self):
        self.memory = torch.zeros(self.mem_n, self.mem_dim)
        self.usage = torch.zeros(self.mem_n)
        self.link = torch.zeros(self.mem_n, self.mem_n)
        self.reads = [torch.zeros(self.mem_dim) for _ in range(self.n_heads)]
        self.h = torch.zeros(1, self.ctrl_dim)
        self.c = torch.zeros(1, self.ctrl_dim)
        self.ww = torch.zeros(self.mem_n)

    def _step(self, x):
        inp = torch.cat([x.unsqueeze(0), torch.cat(self.reads, dim=-1).unsqueeze(0)], dim=-1)
        self.h, self.c = self.controller(inp, (self.h, self.c))
        new_reads = []
        for i, head in enumerate(self.read_heads):
            out = head(self.h.squeeze(0))
            k = torch.tanh(out[:self.mem_dim])
            beta = F.softplus(out[self.mem_dim]) + 1
            g = torch.sigmoid(out[self.mem_dim + 1])
            fwd = torch.sigmoid(out[self.mem_dim + 2])
            scores = beta * F.cosine_similarity(self.memory, k.unsqueeze(0))
            wc = F.softmax(scores, dim=-1)
            wf = self.link @ self.ww
            w = g * wc + (1 - g) * (fwd * wf + (1 - fwd) * self.ww)
            r = w @ self.memory
            new_reads.append(r)
        self.reads = new_reads
        # write
        wo = self.write_head(self.h.squeeze(0))
        k = torch.tanh(wo[:self.mem_dim])
        beta = F.softplus(wo[self.mem_dim]) + 1
        g = torch.sigmoid(wo[self.mem_dim + 1])
        e = torch.sigmoid(wo[self.mem_dim + 2:self.mem_dim + 2 + self.mem_dim])
        a = torch.tanh(wo[-self.mem_dim:])
        scores = beta * F.cosine_similarity(self.memory, k.unsqueeze(0))
        wc = F.softmax(scores, dim=-1)
        wu = self.usage * (1 - self.ww) + self.ww
        w_alloc = (1 - wu) * torch.cumprod(1 - torch.cat([torch.tensor([0.0]), wu[:-1]]), dim=0)
        self.ww = g * wc + (1 - g) * w_alloc
        self.usage = wu + self.ww - wu * self.ww
        # update memory
        self.memory = self.memory * (1 - self.ww.unsqueeze(-1) * e)
        self.memory = self.memory + self.ww.unsqueeze(-1) * a
        # update link matrix
        w_prev = self.ww.unsqueeze(0)
        w_new = self.ww.unsqueeze(1)
        self.link = (1 - w_prev - w_new) * self.link + w_new * self.ww.unsqueeze(0)
        # output
        cat = torch.cat([self.h.squeeze(0), torch.cat(self.reads, dim=-1)])
        return self.out_head(cat)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        if y.ndim == 1:
            y = y[:, None]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            self._reset()
            loss = 0.0
            for t in range(X.shape[0]):
                out = self._step(torch.from_numpy(X[t]))
                loss = loss + F.mse_loss(out, torch.from_numpy(y[t]).unsqueeze(0))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        if not self._trained:
            self._reset()
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        return self._step(x).squeeze(0).detach().numpy()
