"""
qai.techniques.frontier_sequence -- specialised sequence architectures.

  qai.build(technique="s4", ...)              — Structured State Space (S4)
  qai.build(technique="pointer_net", ...)     — Pointer Networks (Vinyals 2015)
  qai.build(technique="set_transformer", ...) — Set Transformer (Lee 2019)
  qai.build(technique="hmm", ...)             — Hidden Markov Model (Baum-Welch)
  qai.build(technique="ctc", ...)             — Connectionist Temporal Classification
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
            "[WHAT] Frontier sequence techniques require PyTorch\n"
            "[FIX] pip install torch"
        )


# =====================================================================
# 1. S4 — Structured State Space Model (Gu, Goel, Re 2022)
# =====================================================================
class _S4Layer(nn.Module):
    """Simplified S4 layer using a discretised linear state space.

    Implements x_t = A x_{t-1} + B u_t, y_t = C x_t with A parameterised as
    a diagonal matrix. State has shape (B, state_dim); B maps (B, dim) -> (B, state_dim);
    C maps (B, state_dim) -> (B, dim). D is a skip connection.
    """
    def __init__(self, dim, state_dim=16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        # diagonal A with negative real parts for stability
        A = np.log(-0.5 * np.log(np.random.uniform(0.01, 1.0, state_dim)))
        self.A_log = nn.Parameter(torch.tensor(A, dtype=torch.float32))
        self.B_proj = nn.Linear(dim, state_dim, bias=False)
        self.C_proj = nn.Linear(state_dim, dim, bias=False)
        self.D = nn.Parameter(torch.ones(dim))

    def forward(self, u):
        """u: (B, L, D)"""
        B_size, L, _ = u.shape
        A = -torch.exp(self.A_log)               # (state_dim,)
        Abar = torch.exp(A)                       # discretised A (diagonal)
        x = torch.zeros(B_size, self.state_dim)
        ys = []
        for t in range(L):
            Bu = self.B_proj(u[:, t])             # (B, state_dim)
            x = Abar * x + Bu                     # (B, state_dim)
            y = self.C_proj(x) + self.D           # (B, dim)
            ys.append(y)
        return torch.stack(ys, dim=1)             # (B, L, dim)


class S4(Technique):
    """S4 — Structured State Space sequence model.

    Bridges RNNs, CNNs, and continuous-time models. Handles very long
    sequences with linear scaling (vs quadratic for transformers).
    """
    name = "s4"
    family = "sequence"

    def __init__(self, in_dim: int = 1, out_dim: int = 1,
                 state_dim: int = 16, hidden: int = 32,
                 lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, out_dim=out_dim,
                         state_dim=state_dim, hidden=hidden,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.state_dim = state_dim
        self.lr, self.epochs = lr, epochs
        self.s4 = _S4Layer(in_dim, state_dim)
        self.head = nn.Linear(in_dim, out_dim)
        self.opt = torch.optim.Adam(
            list(self.s4.parameters()) + list(self.head.parameters()), lr=lr)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, :, :]
        if y.ndim == 2:
            y = y[None, :, :]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            h = self.s4(torch.from_numpy(X))
            out = self.head(h)
            loss = F.mse_loss(out, torch.from_numpy(y))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            h = self.s4(x)
            out = self.head(h)
        return out.squeeze(0).numpy()


# =====================================================================
# 2. POINTER NETWORK (Vinyals et al. 2015)
# =====================================================================
class _PointerEncoder(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(dim, hidden, batch_first=True)
        self.hidden = hidden

    def forward(self, x):
        out, (h, c) = self.rnn(x)
        return out, h, c


class _PointerDecoder(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(dim, hidden, batch_first=True)
        self.W = nn.Linear(hidden, hidden)

    def forward(self, decoder_input, hidden, encoder_outputs):
        out, hidden = self.rnn(decoder_input, hidden)
        scores = self.W(encoder_outputs) @ out.transpose(-1, -2)
        attn = F.softmax(scores.squeeze(-1), dim=-1)
        return attn, hidden


class PointerNetwork(Technique):
    """Pointer Network — output is a sequence of pointers into the input.

    Classic use case: solving combinatorial problems like convex hull or
    TSP where the output is a permutation of input positions.
    """
    name = "pointer_net"
    family = "sequence"

    def __init__(self, in_dim: int = 2, hidden: int = 64,
                 lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, hidden=hidden,
                         lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.hidden = in_dim, hidden
        self.lr, self.epochs = lr, epochs
        self.encoder = _PointerEncoder(in_dim, hidden)
        self.decoder = _PointerDecoder(in_dim, hidden)
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)

    def fit(self, X, y):
        """X: (N, L, in_dim), y: (N, L) int indices into L."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        for ep in range(self.epochs):
            self.opt.zero_grad()
            enc_out, h, c = self.encoder(torch.from_numpy(X))
            loss = 0.0
            for t in range(y.shape[1]):
                # decoder input = embedding of previous pointer target
                if t == 0:
                    dec_in = torch.zeros(X.shape[0], 1, self.in_dim)
                else:
                    dec_in = torch.from_numpy(X[np.arange(X.shape[0]), y[:, t-1]])[:, None, :]
                attn, (h, c) = self.decoder(dec_in, (h, c), enc_out)
                loss = loss + F.nll_loss(torch.log(attn + 1e-8), torch.from_numpy(y[:, t]))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x, n_steps=None):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            enc_out, h, c = self.encoder(x)
            L = x.shape[1]
            n_steps = n_steps or L
            out = []
            dec_in = torch.zeros(x.shape[0], 1, self.in_dim)
            for t in range(n_steps):
                attn, (h, c) = self.decoder(dec_in, (h, c), enc_out)
                idx = attn.argmax(-1).item()
                out.append(idx)
                dec_in = x[:, idx:idx+1]
        return out


# =====================================================================
# 3. SET TRANSFORMER (Lee et al. 2019) — permutation invariant
# =====================================================================
class _MAB(nn.Module):
    """Multihead Attention Block."""
    def __init__(self, dim_q, dim_k, dim_v, n_heads=4):
        super().__init__()
        assert dim_v % n_heads == 0 or n_heads == 1, f"dim_v ({dim_v}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.head_dim = max(dim_v // n_heads, 1)
        out_dim = self.head_dim * n_heads
        self.W_q = nn.Linear(dim_q, out_dim)
        self.W_k = nn.Linear(dim_k, out_dim)
        self.W_v = nn.Linear(dim_k, out_dim)
        self.W_o = nn.Linear(out_dim, dim_q)
        self.ln1 = nn.LayerNorm(dim_q)
        self.ln2 = nn.LayerNorm(dim_q)
        self.ff = nn.Sequential(nn.Linear(dim_q, dim_q), nn.ReLU(), nn.Linear(dim_q, dim_q))

    def forward(self, Q, K):
        B, Nq, _ = Q.shape
        Nk = K.shape[1]
        Qh = self.W_q(Q).view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        Kh = self.W_k(K).view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)
        Vh = self.W_v(K).view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.softmax((Qh @ Kh.transpose(-1, -2)) / (self.head_dim ** 0.5), dim=-1)
        O = (attn @ Vh).transpose(1, 2).contiguous().view(B, Nq, -1)
        H = self.ln1(Q + self.W_o(O))
        H = self.ln2(H + self.ff(H))
        return H


class _SAB(nn.Module):
    """Self Attention Block — MAB(X, X)."""
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.mab = _MAB(dim, dim, dim, n_heads)

    def forward(self, X):
        return self.mab(X, X)


class _ISAB(nn.Module):
    """Induced Set Attention Block — uses m inducing points."""
    def __init__(self, dim, m=8, n_heads=4):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, dim))
        self.mab1 = _MAB(dim, dim, dim, n_heads)
        self.mab2 = _MAB(dim, dim, dim, n_heads)

    def forward(self, X):
        H = self.mab1(self.I.expand(X.shape[0], -1, -1), X)
        return self.mab2(X, H)


class _PMA(nn.Module):
    """Pooling by Multihead Attention — pool a set to k seed vectors."""
    def __init__(self, dim, k=1, n_heads=4):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, dim))
        self.mab = _MAB(dim, dim, dim, n_heads)

    def forward(self, X):
        return self.mab(self.S.expand(X.shape[0], -1, -1), X)


class SetTransformer(Technique):
    """Set Transformer — permutation-invariant architecture for set inputs.

    Uses Induced Set Attention Blocks (ISABs) for O(nm) attention instead of
    O(n^2), and a PMA pooling layer to produce a fixed-size output. Ideal
    for point clouds, set anomaly detection, and permutation-invariant
    classification.
    """
    name = "set_transformer"
    family = "set"

    def __init__(self, in_dim: int = 4, out_dim: int = 1,
                 hidden: int = 32, n_heads: int = 4, m: int = 4,
                 lr: float = 1e-3, epochs: int = 100, **kw):
        super().__init__(in_dim=in_dim, out_dim=out_dim, hidden=hidden,
                         n_heads=n_heads, m=m, lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.lr, self.epochs = lr, epochs
        # project input up to hidden, then run ISAB + decoder all at hidden dim
        self.proj = nn.Linear(in_dim, hidden)
        self.enc = nn.Sequential(
            _ISAB(hidden, m, n_heads),
            _ISAB(hidden, m, n_heads),
        )
        self.dec = nn.Sequential(
            _PMA(hidden, 1, n_heads),
            _SAB(hidden, n_heads),
            _SAB(hidden, n_heads),
            nn.Linear(hidden, out_dim),
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        params = list(self.proj.parameters())
        params += list(self.enc.parameters())
        params += list(self.dec.parameters())
        return params

    def fit(self, X, y):
        """X: list of (n_i, in_dim) arrays; y: (N, out_dim)."""
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            loss = 0.0
            for i, x in enumerate(X):
                xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
                xt = self.proj(xt)
                h = self.enc(xt)
                o = self.dec(h).squeeze(0)
                loss = loss + F.mse_loss(o, torch.from_numpy(y[i]))
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            x = self.proj(x)
            h = self.enc(x)
            o = self.dec(h).squeeze(0)
        return o.numpy()


# =====================================================================
# 4. HIDDEN MARKOV MODEL (Baum-Welch)
# =====================================================================
class HiddenMarkovModel(Technique):
    """Discrete HMM trained with Baum-Welch (EM).

    Maintains transition matrix A, emission matrix B, and initial state
    distribution pi. fit() runs Baum-Welch for n_iter iterations. viterbi()
    returns the most likely hidden state sequence.
    """
    name = "hmm"
    family = "probabilistic"

    def __init__(self, n_states: int = 3, n_obs: int = 5,
                 n_iter: int = 10, tol: float = 1e-4, **kw):
        super().__init__(n_states=n_states, n_obs=n_obs,
                         n_iter=n_iter, tol=tol, **kw)
        self.n_states = n_states
        self.n_obs = n_obs
        self.n_iter = n_iter
        self.tol = tol
        rng = np.random.default_rng(0)
        self.A = rng.dirichlet(np.ones(n_states), size=n_states).astype(np.float64)
        self.B = rng.dirichlet(np.ones(n_obs), size=n_states).astype(np.float64)
        self.pi = rng.dirichlet(np.ones(n_states)).astype(np.float64)

    def _forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.pi * self.B[:, obs[0]]
        alpha[0] /= alpha[0].sum() + 1e-12
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = alpha[t-1] @ self.A[:, j] * self.B[j, obs[t]]
            alpha[t] /= alpha[t].sum() + 1e-12
        return alpha

    def _backward(self, obs):
        T = len(obs)
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = (self.A[i] * self.B[:, obs[t+1]] * beta[t+1]).sum()
            beta[t] /= beta[t].sum() + 1e-12
        return beta

    def fit(self, X, y=None):
        """X: list of observation sequences (each a list of int in [0, n_obs))."""
        for it in range(self.n_iter):
            pi_new = np.zeros(self.n_states)
            A_num = np.zeros((self.n_states, self.n_states))
            A_den = np.zeros(self.n_states)
            B_num = np.zeros((self.n_states, self.n_obs))
            B_den = np.zeros(self.n_states)
            for obs in X:
                obs = np.asarray(obs).astype(int)
                alpha = self._forward(obs)
                beta = self._backward(obs)
                T = len(obs)
                # gamma
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + 1e-12
                pi_new += gamma[0]
                # xi
                for t in range(T - 1):
                    xi = (alpha[t, :, None] * self.A * self.B[:, obs[t+1]][None, :]
                          * beta[t+1][None, :])
                    xi /= xi.sum() + 1e-12
                    A_num += xi
                    A_den += gamma[t]
                for t in range(T):
                    B_num[:, obs[t]] += gamma[t]
                    B_den += gamma[t]
            self.pi = pi_new / (pi_new.sum() + 1e-12)
            self.A = A_num / (A_den[:, None] + 1e-12)
            self.B = B_num / (B_den[:, None] + 1e-12)
        self._trained = True
        return self

    def viterbi(self, obs):
        obs = np.asarray(obs).astype(int)
        T = len(obs)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        delta[0] = self.pi * self.B[:, obs[0]]
        for t in range(1, T):
            for j in range(self.n_states):
                p = delta[t-1] * self.A[:, j]
                psi[t, j] = p.argmax()
                delta[t, j] = p.max() * self.B[j, obs[t]]
        path = [delta[-1].argmax()]
        for t in range(T - 1, 0, -1):
            path.append(psi[t, path[-1]])
        return list(reversed(path))

    def forward(self, x):
        return self.viterbi(x)


# =====================================================================
# 5. CTC — Connectionist Temporal Classification (Graves 2006)
# =====================================================================
class _CTCNet(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden, n_classes + 1)  # +1 for blank

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.fc(h)


class CTC(Technique):
    """CTC — train a sequence model on unaligned (input, output) pairs.

    A bidirectional LSTM produces per-timestep logits over the alphabet
    plus a blank symbol. The CTC loss marginalises over all monotonic
    alignments between input and output. decode() uses greedy best-path
    decoding with blank collapsing.
    """
    name = "ctc"
    family = "sequence"

    def __init__(self, in_dim: int = 8, n_classes: int = 5,
                 hidden: int = 32, lr: float = 1e-3,
                 epochs: int = 50, **kw):
        super().__init__(in_dim=in_dim, n_classes=n_classes,
                         hidden=hidden, lr=lr, epochs=epochs, **kw)
        _req()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.lr, self.epochs = lr, epochs
        self.net = _CTCNet(in_dim, n_classes, hidden)
        self.blank = n_classes
        self.ctc_loss = nn.CTCLoss(blank=self.blank, reduction="mean", zero_infinity=True)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X, y):
        """X: list of (T_i, in_dim); y: list of (L_i,) int target sequences."""
        for ep in range(self.epochs):
            self.opt.zero_grad()
            total_loss = 0.0
            for x_seq, y_seq in zip(X, y):
                x = torch.from_numpy(np.asarray(x_seq, dtype=np.float32)).unsqueeze(0)
                logits = self.net(x).log_softmax(-1).transpose(0, 1)  # (T, 1, C+1)
                target = torch.as_tensor(np.asarray(y_seq).astype(np.int64))
                in_len = torch.tensor([logits.shape[0]])
                tgt_len = torch.tensor([len(y_seq)])
                total_loss = total_loss + self.ctc_loss(logits, target, in_len, tgt_len)
            total_loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def decode(self, x_seq):
        x = torch.from_numpy(np.asarray(x_seq, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = self.net(x)
            path = logits.argmax(-1).squeeze(0).tolist()
        # collapse repeats and remove blanks
        out = []
        prev = -1
        for p in path:
            if p != prev and p != self.blank:
                out.append(p)
            prev = p
        return out

    def forward(self, x):
        return self.decode(x)
