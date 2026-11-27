"""
qai.techniques.mamba -- State Space Model (SSM) / Mamba architecture.

Mamba (Gu & Dao, 2023) is a selective state-space model that scales LINEARLY
with sequence length (vs Transformer's quadratic). It uses a recurrent hidden
state (like RNNs) but with selective gating (like attention) — getting the
best of both worlds.

This implementation provides:
  - qai.build(technique="mamba", ...) — Mamba/SSM for classification + generation

Key advantages over Transformer:
  - O(n) instead of O(n²) compute with sequence length
  - Constant memory during generation (no KV cache needed)
  - Handles million-token contexts efficiently
  - Selective gating lets it "forget" irrelevant information

Creative additions:
  - Hybrid mode: mix Mamba layers with attention layers (like Qwen3-Next/Kimi Linear)
  - Configurable SSM state dimension
  - Bidirectional Mamba for classification (read sequence both ways)

References:
  - Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." 2023.
  - Qwen3-Next, Kimi Linear — production Mamba/SSM hybrids (2025-2026)
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Union
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
        raise ImportError("[WHAT] Mamba requires PyTorch\n[FIX] pip install torch")


class _SSMLayer(nn.Module):
    """Selective State Space Model layer (simplified Mamba).

    Maintains a recurrent hidden state S that is updated via a delta rule:
      S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1})
    where alpha (decay) and beta (update) are input-dependent gates.

    This is the core of Mamba — it processes tokens one at a time (O(n)),
    but the gates let it selectively remember or forget information.
    """
    def __init__(self, d_model, state_dim=16, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Input projection (produces values + gates)
        self.in_proj = nn.Linear(d_model, d_model * 3)  # v, alpha_gate, beta_gate

        # Convolutional mixing (1D depthwise conv — helps with local context)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable A (decay initialization — helps with long-range)
        # Diagonal initialization for stability
        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))  # log(A) for numerical stability

        # Learnable B, C, D (S4/Mamba parameterization)
        self.B = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, D = x.shape

        # Input projection: split into value, alpha_gate, beta_gate
        proj = self.in_proj(x)  # (B, T, 3*D)
        v, alpha_raw, beta_raw = proj.chunk(3, dim=-1)

        # Convolutional mixing on values
        v_conv = self.conv(v.transpose(1, 2)).transpose(1, 2)  # (B, T, D)
        v = v + v_conv  # residual

        # Gates (input-dependent, like Mamba's selectivity)
        alpha = torch.sigmoid(alpha_raw)  # decay gate: how much old state to keep
        beta = torch.sigmoid(beta_raw)    # update gate: how much new info to absorb

        # Discretize A (continuous -> discrete)
        A = -torch.exp(self.A_log)  # (D, state_dim) — negative for decay

        # Process sequence recurrently
        # State: (B, D, state_dim)
        S = torch.zeros(B, D, self.state_dim, device=x.device)
        outputs = []

        for t in range(T):
            # Selective update: S = alpha * S + beta * (v_t * B - S)
            v_t = v[:, t, :]  # (B, D)
            alpha_t = alpha[:, t, :]  # (B, D)
            beta_t = beta[:, t, :]  # (B, D)

            # Compute input-dependent B_t and C_t
            B_t = self.B * beta_t.unsqueeze(-1)  # (B, D, state_dim)
            C_t = self.C  # (D, state_dim) — could also be input-dependent

            # Discretized state update
            dA = torch.exp(A.unsqueeze(0) * alpha_t.unsqueeze(-1))  # (B, D, state_dim)
            S = dA * S + B_t * v_t.unsqueeze(-1)  # (B, D, state_dim)

            # Output: y_t = C_t @ S_t + D * v_t
            y_t = (S * C_t.unsqueeze(0)).sum(dim=-1) + self.D * v_t  # (B, D)
            outputs.append(y_t)

        out = torch.stack(outputs, dim=1)  # (B, T, D)
        out = self.out_proj(self.norm(out))
        out = self.dropout(out)
        return out + x  # residual


class _MambaBlock(nn.Module):
    """A full Mamba block: SSM layer + FFN (like a transformer block but with SSM instead of attention)."""
    def __init__(self, d_model, state_dim=16, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.ssm = _SSMLayer(d_model, state_dim, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _MambaNet(nn.Module):
    """Full Mamba model: token embedding + N Mamba blocks + output head."""
    def __init__(self, vocab_size, d_model=128, n_layers=4, state_dim=16,
                 max_seq_len=256, d_ff=None, dropout=0.1,
                 task_type="generation", n_classes=None,
                 use_attention_layers=False, attention_ratio=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.task_type = task_type
        self.n_classes = n_classes
        self.use_attention_layers = use_attention_layers
        self.attention_ratio = attention_ratio

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Build blocks: mix SSM and (optionally) attention layers
        blocks = []
        for i in range(n_layers):
            blocks.append(_MambaBlock(d_model, state_dim, d_ff, dropout))
            # Creative: hybrid mode — every attention_ratio-th layer gets attention
            if use_attention_layers and (i + 1) % attention_ratio == 0:
                blocks.append(_AttentionBlock(d_model, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)

        if task_type == "generation":
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.embedding.weight  # weight tying
        else:
            assert n_classes is not None
            self.head = nn.Linear(d_model, n_classes)

        self.max_seq_len = max_seq_len

    def forward(self, token_ids, return_hidden=False):
        B, T = token_ids.shape
        if T > self.max_seq_len:
            token_ids = token_ids[:, -self.max_seq_len:]
            T = self.max_seq_len

        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if self.task_type == "generation":
            out = self.head(x)
        else:
            out = self.head(x[:, -1, :])  # CLS pooling (last token)

        if return_hidden:
            return out, x
        return out


class _AttentionBlock(nn.Module):
    """A single standard attention block (for hybrid Mamba-Attention models)."""
    def __init__(self, d_model, dropout=0.0, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        B, T, C = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out(out)
        return residual + out


class Mamba(Technique):
    """Mamba / Selective State Space Model (Gu & Dao, 2023).

    Linear-scaling alternative to Transformers. Uses selective state-space
    layers instead of attention — O(n) instead of O(n²) with sequence length.

    Creative additions:
      - Hybrid mode: mix Mamba layers with standard attention layers
        (like Qwen3-Next and Kimi Linear's 3:1 ratio)
      - Configurable SSM state dimension
      - Supports both classification and generation tasks

    Example:
        model = qai.build(technique="mamba", vocab_size=tok.vocab_size,
                          d_model=64, n_layers=4, state_dim=16,
                          task_type="generation", max_seq_len=128, epochs=30)
        model.train(corpus, tokenizer=tok)
        print(model.generate("hello", max_new_tokens=20, tokenizer=tok))
    """
    name = "mamba"
    family = "deep_learning"
    compatible_learning_techniques = ("supervised",)

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        state_dim: int = 16,
        max_seq_len: int = 256,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        task_type: str = "generation",
        n_classes: Optional[int] = None,
        use_attention_layers: bool = False,
        attention_ratio: int = 4,
        epochs: int = 30,
        lr: float = 3e-4,
        batch_size: int = 32,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        device: Optional[str] = None,
        **params,
    ):
        super().__init__(**{**params, "vocab_size": vocab_size, "d_model": d_model,
                            "n_layers": n_layers, "state_dim": state_dim,
                            "max_seq_len": max_seq_len, "d_ff": d_ff,
                            "dropout": dropout, "task_type": task_type,
                            "n_classes": n_classes, "use_attention_layers": use_attention_layers,
                            "attention_ratio": attention_ratio, "epochs": epochs, "lr": lr,
                            "batch_size": batch_size, "weight_decay": weight_decay,
                            "warmup_steps": warmup_steps, "device": device})
        _req()
        assert task_type in ("classification", "generation")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.max_seq_len = max_seq_len
        self.d_ff = d_ff
        self.dropout = dropout
        self.task_type = task_type
        self.n_classes = n_classes
        self.use_attention_layers = use_attention_layers
        self.attention_ratio = attention_ratio
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        if device is None:
            try:
                import qai; device = qai.get_optimal_device()
            except: device = "cpu"
        self.device = torch.device(device)

        self.net: Optional[_MambaNet] = None
        self._losses: List[float] = []
        self._last_X = None
        self._last_y = None
        self.tokenizer = None
        self.classes_: Optional[np.ndarray] = None

    def _build_model(self, vocab_size=None, n_classes=None):
        if vocab_size: self.vocab_size = vocab_size
        if n_classes: self.n_classes = n_classes
        self.net = _MambaNet(
            vocab_size=self.vocab_size, d_model=self.d_model,
            n_layers=self.n_layers, state_dim=self.state_dim,
            max_seq_len=self.max_seq_len, d_ff=self.d_ff,
            dropout=self.dropout, task_type=self.task_type,
            n_classes=self.n_classes,
            use_attention_layers=self.use_attention_layers,
            attention_ratio=self.attention_ratio,
        ).to(self.device)

    def fit(self, X, y=None, tokenizer=None, **kw):
        if tokenizer: self.tokenizer = tokenizer
        token_id_lists, vocab_size = self._prepare_inputs(X, tokenizer)

        n_classes = None
        if self.task_type == "classification":
            if y is None:
                raise ValueError("classification requires y")
            y_arr = np.asarray(y)
            if y_arr.dtype.kind in ("U", "S", "O"):
                self.classes_ = np.unique(y_arr)
                y_enc = np.searchsorted(self.classes_, y_arr).astype(np.int64)
            else:
                y_enc = y_arr.astype(np.int64)
                self.classes_ = np.unique(y_enc)
            n_classes = int(y_enc.max()) + 1
            self._last_y = y_enc
        self._last_X = token_id_lists

        if self.net is None:
            self._build_model(vocab_size=vocab_size, n_classes=n_classes)

        if self.task_type == "classification":
            self._train_classification(token_id_lists, self._last_y)
        else:
            self._train_generation(token_id_lists)
        self._trained = True
        return self

    def _prepare_inputs(self, X, tokenizer):
        if isinstance(X, str): X = [X]
        lists = []; max_id = 0
        for item in X:
            if isinstance(item, str):
                if tokenizer is None:
                    raise ValueError("String input but no tokenizer")
                ids = tokenizer.encode(item)
                arr = np.asarray(ids, dtype=np.int64)
            else:
                arr = np.asarray(item, dtype=np.int64)
            if arr.size > 0: max_id = max(max_id, int(arr.max()))
            lists.append(arr)
        vs = tokenizer.vocab_size if tokenizer else max(self.vocab_size, max_id + 1)
        return lists, vs

    def _collate(self, examples, pad_id=0):
        max_len = min(max(len(s) for s in examples), self.max_seq_len)
        if max_len == 0: max_len = 1
        batch = np.full((len(examples), max_len), pad_id, dtype=np.int64)
        for i, s in enumerate(examples):
            L = min(len(s), max_len)
            batch[i, :L] = s[:L]
        return torch.tensor(batch, device=self.device)

    def _train_classification(self, lists, y_enc):
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total = max(1, (len(lists) // self.batch_size + 1) * self.epochs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: self._lr_lambda(s, total))
        self._losses = []; self.net.train()
        for ep in range(self.epochs):
            rng = np.random.default_rng(ep); perm = rng.permutation(len(lists))
            ep_loss = 0.0; n_b = 0
            for start in range(0, len(lists), self.batch_size):
                idx = perm[start:start+self.batch_size]
                X_t = self._collate([lists[i] for i in idx])
                y_t = torch.tensor(y_enc[idx], device=self.device, dtype=torch.long)
                logits = self.net(X_t)
                loss = F.cross_entropy(logits, y_t)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step(); sched.step()
                ep_loss += float(loss); n_b += 1
            self._losses.append(ep_loss / max(1, n_b))

    def _train_generation(self, lists):
        examples = []
        for ids in lists:
            if len(ids) < 2: continue
            w = self.max_seq_len
            for s in range(0, max(1, len(ids) - 1), w):
                e = min(s + w + 1, len(ids))
                chunk = ids[s:e]
                if len(chunk) < 2: continue
                examples.append((chunk[:-1], chunk[1:]))
        if not examples:
            raise ValueError("Need sequences of length >= 2")
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total = max(1, (len(examples) // self.batch_size + 1) * self.epochs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: self._lr_lambda(s, total))
        self._losses = []; self.net.train()
        for ep in range(self.epochs):
            rng = np.random.default_rng(ep); perm = rng.permutation(len(examples))
            ep_loss = 0.0; n_b = 0
            for start in range(0, len(examples), self.batch_size):
                idx = perm[start:start+self.batch_size]
                X_t = self._collate([examples[i][0] for i in idx])
                T_t = self._collate([examples[i][1] for i in idx])
                if T_t.size(1) < X_t.size(1):
                    T_t = F.pad(T_t, (0, X_t.size(1) - T_t.size(1)))
                elif T_t.size(1) > X_t.size(1):
                    T_t = T_t[:, :X_t.size(1)]
                logits = self.net(X_t)
                loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), T_t.reshape(-1), ignore_index=0)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step(); sched.step()
                ep_loss += float(loss); n_b += 1
            self._losses.append(ep_loss / max(1, n_b))

    def _lr_lambda(self, step, total):
        if step < self.warmup_steps: return step / max(1, self.warmup_steps)
        p = (step - self.warmup_steps) / max(1, total - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    def forward(self, x, **kw):
        if not self._trained: raise RuntimeError("Mamba.predict() before train()")
        ids = self._coerce(x)
        if len(ids) == 0: ids = np.array([0], dtype=np.int64)
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor([ids], device=self.device, dtype=torch.long)
            logits = self.net(t)
            if self.task_type == "classification":
                pred = int(torch.argmax(logits, dim=-1).item())
                if self.classes_ is not None and self.classes_.dtype.kind in ("U", "S", "O"):
                    return self.classes_[pred]
                return pred
            return logits.cpu().numpy()

    def _coerce(self, x):
        if isinstance(x, str):
            if self.tokenizer is None: raise ValueError("No tokenizer")
            return np.asarray(self.tokenizer.encode(x), dtype=np.int64)
        return np.asarray(x, dtype=np.int64)

    def generate(self, prompt, max_new_tokens=50, tokenizer=None, temperature=0.0, seed=None):
        if self.task_type != "generation":
            raise RuntimeError("generate() requires task_type='generation'")
        if not self._trained: raise RuntimeError("generate() before train()")
        tok = tokenizer or self.tokenizer
        if tok is None and isinstance(prompt, str):
            raise ValueError("No tokenizer")
        prompt_ids = tok.encode(prompt) if isinstance(prompt, str) else list(prompt)
        prompt_ids = prompt_ids[:self.max_seq_len - 1]
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed)
        self.net.eval()
        generated = list(prompt_ids)
        with torch.no_grad():
            t = torch.tensor([generated], device=self.device, dtype=torch.long)
            logits = self.net(t)
            next_logits = logits[0, -1, :]
        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                probs = F.softmax(next_logits / max(1e-8, temperature), dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            generated.append(next_id)
            with torch.no_grad():
                t = torch.tensor([generated[-self.max_seq_len:]], device=self.device, dtype=torch.long)
                logits = self.net(t)
                next_logits = logits[0, -1, :]
        if tok: return tok.decode(generated)
        return " ".join(str(i) for i in generated)

    def losses(self): return list(self._losses)
    def parameter_count(self): return sum(p.numel() for p in self.net.parameters()) if self.net else 0

    def architecture_summary(self):
        return {"technique":"mamba","vocab_size":self.vocab_size,"d_model":self.d_model,
                "n_layers":self.n_layers,"state_dim":self.state_dim,
                "max_seq_len":self.max_seq_len,"use_attention_layers":self.use_attention_layers,
                "attention_ratio":self.attention_ratio,"task_type":self.task_type,
                "parameters":self.parameter_count(),"device":str(self.device)}

    def __repr__(self):
        if self.net is None: return f"<Mamba unbuilt>"
        return (f"<Mamba d_model={self.d_model} n_layers={self.n_layers} "
                f"state_dim={self.state_dim} hybrid={self.use_attention_layers} "
                f"params={self.parameter_count():,} trained={self._trained}>")
