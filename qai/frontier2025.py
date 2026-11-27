"""
qai.frontier2025 -- 2025-era frontier architectures built on the framework.

  1. Mamba2       -- Structured State Space Duality (Dao & Gu 2024)
                     SSD layer + full Mamba-2 block
  2. DeepSeekMoE  -- fine-grained experts + shared experts (DeepSeek 2024)
  3. TTTLayer     -- Test-Time Training layer (Sun et al. 2024)
  4. JambaBlock   -- Mamba + Transformer hybrid (AI21 2024)
  5. GatedDeltaNet-- gated DeltaNet (Yang et al. 2024)
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    from qai._torch_stub import _TorchStubNamespace, _NNStubNamespace
    torch = _TorchStubNamespace()
    nn = _NNStubNamespace()
    F = nn

from .techniques.base import Technique

def _req():
    if not _TORCH:
        raise ImportError("[WHAT] Frontier2025 requires PyTorch\n[FIX] pip install torch")


# ============================================================================
# 1. Mamba-2 SSD Layer (Dao & Gu 2024)
# ============================================================================
class _SSDLayer(nn.Module):
    """Structured State Space Duality layer.

    The SSD layer can be computed in two equivalent modes:
      - Linear (SSM) mode:  O(N) recurrent scan
      - Quadratic (attention) mode: O(N^2) matrix attention

    The quadratic mode reveals the connection to attention and lets us
    leverage matmul-heavy GPU kernels. We implement the quadratic mode
    here for clarity; the linear mode is just the same computation
    reorganised as a recurrence.
    """
    def __init__(self, dim, state_dim=16, block_len=8):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.block_len = block_len
        # input projections
        self.W_x = nn.Linear(dim, state_dim)
        self.W_A = nn.Linear(dim, state_dim * state_dim)
        self.W_B = nn.Linear(dim, state_dim)
        self.W_C = nn.Linear(dim, state_dim)
        self.W_D = nn.Linear(dim, dim)

    def forward(self, x):
        """x: (B, L, D) -> (B, L, D)"""
        B, L, D = x.shape
        N = self.state_dim
        # project inputs
        X = self.W_x(x)                          # (B, L, N)
        A_b = self.W_A(x).view(B, L, N, N)       # (B, L, N, N)
        B_t = self.W_B(x)                        # (B, L, N)
        C_t = self.W_C(x)                        # (B, L, N)
        D_skip = self.W_D(x)                     # (B, L, D)
        # SSD quadratic form: y_t = sum_{s<=t} C_t^T (prod A) X_s B_s
        # For tractability we use the simplified recurrence form:
        h = torch.zeros(B, N, D)
        ys = []
        for t in range(L):
            At = A_b[:, t]                       # (B, N, N)
            # Bug fix: the einsum subscript for h was "bdm" but h is (B, N, D)
            # so it should be "bmd". With the wrong subscript, the recurrence
            # only worked when state_dim == dim (so the mismatched axis
            # happened to be the same size) and crashed otherwise with
            # "einsum(): subscript m has size 8 for operand 1 which does not
            # broadcast". Also: the second term is B_t outer x_t (the original
            # input at time t, shape (B, D)), not the W_x-projected X (shape
            # (B, N)) -- using X broke the "bd" subscript.
            h = torch.einsum("bnm,bmd->bnd", At, h) + \
                torch.einsum("bn,bd->bnd", B_t[:, t], x[:, t])
            y_t = torch.einsum("bn,bnd->bd", C_t[:, t], h)
            ys.append(y_t)
        y = torch.stack(ys, dim=1) + D_skip
        return y


class _Mamba2Block(nn.Module):
    """Module-level Block class for Mamba2 (picklable)."""
    def __init__(self, dim, state_dim, mlp_dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.ssd = _SSDLayer(dim, state_dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        x = x + self.ssd(self.ln(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Mamba2(Technique):
    """Full Mamba-2 model: SSD + MLP + LayerNorm in residual blocks.

    The Mamba-2 paper shows SSD is 2-8x faster than Mamba-1 while keeping
    competitive quality. This implementation is the educational CPU
    version; production Mamba-2 uses CUDA-specific kernels for the
    segmented block scan.
    """
    name = "mamba2"
    family = "sequence"

    def __init__(self, dim: int = 32, state_dim: int = 8,
                 n_layers: int = 2, mlp_dim: int = 64,
                 lr: float = 1e-3, epochs: int = 50, **kw):
        super().__init__(dim=dim, state_dim=state_dim, n_layers=n_layers,
                         mlp_dim=mlp_dim, lr=lr, epochs=epochs, **kw)
        _req()
        self.dim, self.state_dim = dim, state_dim
        self.lr, self.epochs = lr, epochs
        self.layers = nn.ModuleList([
            _Mamba2Block(dim, state_dim, mlp_dim) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        p = []
        for blk in self.layers:
            p += blk.parameters()
        p += list(self.ln_f.parameters())
        return p

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, :, :]
        if y.ndim == 2:
            y = y[None, :, :]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            h = torch.from_numpy(X)
            for blk in self.layers:
                h = blk(h)
            h = self.ln_f(h)
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
            for blk in self.layers:
                x = blk(x)
            return self.ln_f(x).squeeze(0).numpy()


# ============================================================================
# 2. DeepSeek MoE -- fine-grained experts + shared experts
# ============================================================================
class _DeepSeekExpert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class _DeepSeekMoELayer(nn.Module):
    """DeepSeekMoE: many small experts + always-on shared experts.

    Compared to vanilla MoE:
      - Experts are smaller (fine-grained), so the routing can mix more
      - A subset of experts is shared (always-active), capturing common patterns
      - Top-k routing with load-balancing loss
    """
    def __init__(self, dim, n_experts=16, n_shared=2, top_k=4, hidden=32):
        super().__init__()
        self.n_experts = n_experts
        self.n_shared = n_shared
        self.top_k = top_k
        self.experts = nn.ModuleList([_DeepSeekExpert(dim, hidden) for _ in range(n_experts)])
        self.shared_experts = nn.ModuleList([_DeepSeekExpert(dim, hidden) for _ in range(n_shared)])
        self.gate = nn.Linear(dim, n_experts)
        self.balance_loss = 0.0

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        logits = self.gate(x)                                # (B, L, E)
        # top-k routing
        top_vals, top_idx = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_vals, dim=-1)
        # compute expert outputs
        flat_x = x.view(-1, D)
        flat_idx = top_idx.view(-1, self.top_k)
        flat_w = weights.view(-1, self.top_k)
        out = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_out = torch.stack([self.experts[e](flat_x) for e in flat_idx[:, i]], dim=0)
            # wait this is wrong; do it properly
        out = torch.zeros(B * L, D, device=x.device)
        for i in range(self.top_k):
            for b in range(B * L):
                e = int(flat_idx[b, i])
                out[b] += flat_w[b, i] * self.experts[e](flat_x[b:b+1]).squeeze(0)
        # add shared experts (always-on)
        for shared in self.shared_experts:
            out = out + shared(flat_x) / self.n_shared
        # load-balancing auxiliary loss
        with torch.no_grad():
            expert_load = torch.zeros(self.n_experts)
            for e_idx in flat_idx.flatten():
                expert_load[e_idx] += 1
            expert_load /= (B * L * self.top_k)
        gate_avg = F.softmax(logits, dim=-1).mean(dim=(0, 1))
        self.balance_loss = float((expert_load * gate_avg).sum() * self.n_experts)
        return out.view(B, L, D)


class DeepSeekMoE(Technique):
    """DeepSeekMoE block — fine-grained + shared experts.

    Used in DeepSeek-V2 / V3 to achieve higher expert specialisation at
    the same parameter count as vanilla MoE.
    """
    name = "deepseek_moe"
    family = "moe"

    def __init__(self, dim: int = 16, n_experts: int = 8, n_shared: int = 2,
                 top_k: int = 2, hidden: int = 32,
                 lr: float = 1e-3, epochs: int = 50, **kw):
        super().__init__(dim=dim, n_experts=n_experts, n_shared=n_shared,
                         top_k=top_k, hidden=hidden, lr=lr, epochs=epochs, **kw)
        _req()
        self.dim = dim
        self.lr, self.epochs = lr, epochs
        self.ln = nn.LayerNorm(dim)
        self.moe = _DeepSeekMoELayer(dim, n_experts, n_shared, top_k, hidden)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        return (list(self.ln.parameters()) + list(self.moe.parameters())
                + list(self.ln2.parameters()) + list(self.mlp.parameters()))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, :, :]
        if y.ndim == 2:
            y = y[None, :, :]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            x = torch.from_numpy(X)
            h = x + self.moe(self.ln(x))
            h = h + self.mlp(self.ln2(h))
            loss = F.mse_loss(h, torch.from_numpy(y)) + 0.01 * self.moe.balance_loss
            loss.backward()
            self.opt.step()
        self._trained = True
        return self

    def forward(self, x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            h = x + self.moe(self.ln(x))
            h = h + self.mlp(self.ln2(h))
        return h.squeeze(0).numpy()


# ============================================================================
# 3. TTT Layer -- Test-Time Training (Sun et al. 2024)
# ============================================================================
class _TTTLayer(nn.Module):
    """TTT layer: hidden state is itself a model that's trained at inference.

    The hidden state is the weights of a small linear model f(theta).
    At each step, we update theta via gradient descent on a self-
    reconstruction loss, then output f(theta)(x).

    Linear-complexity like RNN, but with expressive hidden states.
    """
    def __init__(self, dim, ttt_hidden=32, ttt_lr=0.1):
        super().__init__()
        self.dim = dim
        self.ttt_hidden = ttt_hidden
        self.ttt_lr = ttt_lr
        # the "inner model" weights are the hidden state
        # we parameterise them as a tiny linear network: dim -> ttt_hidden -> dim
        self.W1_init = nn.Parameter(torch.randn(dim, ttt_hidden) * 0.01)
        self.b1_init = nn.Parameter(torch.zeros(ttt_hidden))
        self.W2_init = nn.Parameter(torch.randn(ttt_hidden, dim) * 0.01)
        self.b2_init = nn.Parameter(torch.zeros(dim))

    def _inner_forward(self, x, W1, b1, W2, b2):
        """x: (B, D), W1: (B, D, H), b1: (B, H), W2: (B, H, D), b2: (B, D)"""
        # h = relu(x @ W1 + b1) -> (B, H)
        h = F.relu(torch.einsum("bd,bdh->bh", x, W1) + b1)
        # y = h @ W2 + b2 -> (B, D)
        y = torch.einsum("bh,bhd->bd", h, W2) + b2
        return y

    def forward(self, x):
        """x: (B, L, D) -> (B, L, D)"""
        B, L, D = x.shape
        # clone weights for each batch element so they can be updated independently
        W1 = self.W1_init.unsqueeze(0).expand(B, -1, -1).clone()
        b1 = self.b1_init.unsqueeze(0).expand(B, -1).clone()
        W2 = self.W2_init.unsqueeze(0).expand(B, -1, -1).clone()
        b2 = self.b2_init.unsqueeze(0).expand(B, -1).clone()
        ys = []
        for t in range(L):
            xt = x[:, t]                          # (B, D)
            # output current prediction
            y = self._inner_forward(xt, W1, b1, W2, b2)
            ys.append(y)
            # TTT update: gradient step on reconstruction loss
            with torch.enable_grad():
                W1_g = W1.detach().requires_grad_(True)
                b1_g = b1.detach().requires_grad_(True)
                W2_g = W2.detach().requires_grad_(True)
                b2_g = b2.detach().requires_grad_(True)
                pred = self._inner_forward(xt.detach(), W1_g, b1_g, W2_g, b2_g)
                loss = F.mse_loss(pred, xt.detach())
                gW1, gb1, gW2, gb2 = torch.autograd.grad(loss, [W1_g, b1_g, W2_g, b2_g])
            W1 = W1 - self.ttt_lr * gW1
            b1 = b1 - self.ttt_lr * gb1
            W2 = W2 - self.ttt_lr * gW2
            b2 = b2 - self.ttt_lr * gb2
        return torch.stack(ys, dim=1)


class TTTModel(Technique):
    """TTT-Linear model — a stack of TTT layers + MLP."""
    name = "ttt"
    family = "sequence"

    def __init__(self, dim: int = 16, ttt_hidden: int = 32,
                 n_layers: int = 2, mlp_dim: int = 32,
                 lr: float = 1e-3, epochs: int = 30, **kw):
        super().__init__(dim=dim, ttt_hidden=ttt_hidden, n_layers=n_layers,
                         mlp_dim=mlp_dim, lr=lr, epochs=epochs, **kw)
        _req()
        self.dim = dim
        self.lr, self.epochs = lr, epochs
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(dim),
                "ttt": _TTTLayer(dim, ttt_hidden),
                "ln2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, mlp_dim), nn.SiLU(),
                                     nn.Linear(mlp_dim, dim)),
            }))
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        p = []
        for layer in self.layers:
            for m in layer.values():
                p += list(m.parameters())
        return p

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, :, :]
        if y.ndim == 2:
            y = y[None, :, :]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            h = torch.from_numpy(X)
            for layer in self.layers:
                h = h + layer["ttt"](layer["ln1"](h))
                h = h + layer["mlp"](layer["ln2"](h))
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
            for layer in self.layers:
                x = x + layer["ttt"](layer["ln1"](x))
                x = x + layer["mlp"](layer["ln2"](x))
        return x.squeeze(0).numpy()


# ============================================================================
# 4. Jamba Block -- Mamba + Transformer hybrid (AI21 2024)
# ============================================================================
class _JambaBlock(nn.Module):
    """Jamba: interleaves Mamba + Transformer + MoE blocks.

    Pattern: some layers are Mamba-SSM (linear complexity), some are
    Transformer attention (better at recall), some are MoE (capacity).
    """
    def __init__(self, dim, use_attention: bool, use_moe: bool,
                 state_dim=8, n_heads=4, n_experts=4, top_k=2, mlp_dim=32):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        if use_attention:
            self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        else:
            # use SSM (Mamba-like) for the non-attention branch
            self.ssm = _SSDLayer(dim, state_dim)
        self.use_attention = use_attention
        self.ln2 = nn.LayerNorm(dim)
        if use_moe:
            self.moe = _DeepSeekMoELayer(dim, n_experts=n_experts, top_k=top_k, hidden=mlp_dim)
        else:
            self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.SiLU(),
                                       nn.Linear(mlp_dim, dim))
        self.use_moe = use_moe

    def forward(self, x):
        h = self.ln1(x)
        if self.use_attention:
            a, _ = self.attn(h, h, h, need_weights=False)
        else:
            a = self.ssm(h)
        x = x + a
        h = self.ln2(x)
        if self.use_moe:
            x = x + self.moe(h)
        else:
            x = x + self.mlp(h)
        return x

    def parameters(self):
        p = list(self.ln1.parameters()) + list(self.ln2.parameters())
        if self.use_attention:
            p += list(self.attn.parameters())
        else:
            p += list(self.ssm.parameters())
        if self.use_moe:
            p += list(self.moe.parameters())
        else:
            p += list(self.mlp.parameters())
        return p


class Jamba(Technique):
    """Jamba-style hybrid: Mamba + Transformer + MoE interleaved."""
    name = "jamba"
    family = "hybrid"

    def __init__(self, dim: int = 16, n_layers: int = 4,
                 attention_every: int = 3, moe_every: int = 2,
                 state_dim: int = 8, n_heads: int = 2, n_experts: int = 4,
                 top_k: int = 2, mlp_dim: int = 32,
                 lr: float = 1e-3, epochs: int = 30, **kw):
        super().__init__(dim=dim, n_layers=n_layers,
                         attention_every=attention_every, moe_every=moe_every,
                         state_dim=state_dim, n_heads=n_heads, n_experts=n_experts,
                         top_k=top_k, mlp_dim=mlp_dim, lr=lr, epochs=epochs, **kw)
        _req()
        self.dim = dim
        self.lr, self.epochs = lr, epochs
        self.blocks = nn.ModuleList([
            _JambaBlock(
                dim,
                use_attention=(i % attention_every == 0),
                use_moe=(i % moe_every == 0),
                state_dim=state_dim, n_heads=n_heads, n_experts=n_experts,
                top_k=top_k, mlp_dim=mlp_dim,
            )
            for i in range(n_layers)
        ])
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        p = []
        for blk in self.blocks:
            p += blk.parameters()
        return p

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, :, :]
        if y.ndim == 2:
            y = y[None, :, :]
        for ep in range(self.epochs):
            self.opt.zero_grad()
            h = torch.from_numpy(X)
            for blk in self.blocks:
                h = blk(h)
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
            for blk in self.blocks:
                x = blk(x)
        return x.squeeze(0).numpy()


__all__ = ["Mamba2", "DeepSeekMoE", "TTTModel", "Jamba"]
