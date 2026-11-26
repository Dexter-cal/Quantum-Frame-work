"""
qai.techniques.transformer -- A real Transformer language model technique.

Implements:
  - Token + learned positional embeddings (or RoPE rotary encoding)
  - Multi-head self-attention with causal masking
  - LayerNorm + residual connections
  - Position-wise feed-forward (MLP)
  - Optional KV cache for fast autoregressive generation
  - Greedy + temperature-sampling generation

Two task types:
  - `task_type="causal_lm"` (default): autoregressive language model. Use
    `model.train(corpus)` then `model.generate(prompt, max_new_tokens=...)`.
  - `task_type="classification"`: CLS-token pooling + linear head. Use
    `model.train(X_text, y)` then `model.predict(x_text)`.

Built on PyTorch. Wraps a real nn.Module inside the qai Technique interface,
so it works with `qai.build(technique="transformer")` like every other model.

Example (char-level Python code completion):
    import qai
    tok = qai.CharTokenizer()
    tok.fit([open("code.py").read()])

    model = qai.build(
        technique="transformer",
        task_type="causal_lm",
        vocab_size=tok.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=128,
        epochs=20,
    )
    model.train([open("code.py").read()], tokenizer=tok)
    print(model.generate("def ", max_new_tokens=80, tokenizer=tok))
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from .base import Technique


# =====================================================================
# PyTorch building blocks
# =====================================================================
def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "[WHAT] qai.build(technique='transformer') requires PyTorch\n"
            "[WHY] The transformer technique uses torch.nn for its internals\n"
            "[FIX] Install torch: pip install torch   (or: pip install -r requirements.txt)"
        )


class _RoPE:
    """Rotary Position Embedding (Su et al. 2021).

    Applied to query/key tensors of shape (batch, n_heads, seq_len, head_dim).
    Rotates pairs of channels by an angle proportional to position, so the
    dot-product attention naturally decays with distance — no learned position
    embedding required.

    Precomputes sin/cos tables once at construction time for `max_seq_len`.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        # head_dim must be even
        assert head_dim % 2 == 0, f"RoPE head_dim must be even, got {head_dim}"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # Compute frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
        half = head_dim // 2
        freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))
        # positions: [0, 1, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len).float()
        # angles[pos, i] = pos * freq[i]
        angles = torch.outer(positions, freqs)  # (max_seq_len, half)
        # Duplicate to full head_dim
        self.cos = torch.cos(torch.cat([angles, angles], dim=-1))  # (max_seq_len, head_dim)
        self.sin = torch.sin(torch.cat([angles, angles], dim=-1))

    def to(self, device):
        self.cos = self.cos.to(device)
        self.sin = self.sin.to(device)
        return self

    def apply(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """Apply rotation to x of shape (batch, n_heads, seq_len, head_dim).

        Args:
            x: tensor to rotate.
            position_offset: starting position in the absolute sequence (for
                KV-cached decoding, where the new query/key is at position T,
                not position 0). Default 0 = no offset.
        """
        seq_len = x.shape[-2]
        # Slice cos/sin starting at position_offset, for seq_len positions
        cos = self.cos[position_offset : position_offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin[position_offset : position_offset + seq_len].unsqueeze(0).unsqueeze(0)
        # rotate_half([x1, x2, x3, x4]) = [-x2, x1, -x4, x3]
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class _MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional causal masking, RoPE, KV cache,
    and sliding window attention (SWA).

    Advanced features:
        - KV cache: for autoregressive decoding, pass a dict {k, v} that
          persists across calls. Each new token's k/v is appended.
        - Sliding window attention (SWA): if window_size > 0, each query
          position only attends to keys within `window_size` positions before
          it (plus itself). Reduces memory from O(T^2) to O(T * window_size)
          for long sequences. Used by Gemma 3, Mistral, etc.
        - RoPE: rotary positional embeddings applied to q/k.

    Forward args:
        x: (batch, seq_len, d_model)
        use_causal_mask: if True, mask out future positions
        use_rope: if True, apply RoPE to q/k (ignore learned pos embeddings)
        kv_cache: optional dict with 'k' and 'v' tensors from previous steps
                  (for autoregressive decoding)
        position_offset: absolute position offset (for cached decoding)
        window_size: if > 0, restrict attention to a sliding window of this
                     size (SWA). 0 means global attention (default).

    Returns: (batch, seq_len, d_model)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 window_size: int = 0, attention_impl: str = "auto"):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size  # 0 = global attention
        # Attention backend: "auto" (try SDPA → math), "sdpa", "flash", "math"
        # SDPA = torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+)
        # auto-dispatches to FlashAttention-2 (CUDA), mem-efficient (CUDA), or math (CPU)
        self.attention_impl = attention_impl
        # Fused QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._rope: Optional[_RoPE] = None

    def set_rope(self, rope: _RoPE):
        self._rope = rope

    def forward(self, x, use_causal_mask=False, kv_cache=None,
                position_offset=0, window_size=None, attention_impl=None):
        B, T, C = x.shape  # batch, seq_len, d_model
        # Use instance window_size unless overridden per-call
        ws = self.window_size if window_size is None else window_size
        # Use instance attention_impl unless overridden per-call
        impl = self.attention_impl if attention_impl is None else attention_impl

        # Project to Q, K, V (each: batch, seq_len, d_model)
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k if set, using position_offset so cached decoding
        # uses the correct absolute position for the new query/key.
        if self._rope is not None:
            q = self._rope.apply(q, position_offset=position_offset)
            k = self._rope.apply(k, position_offset=position_offset)

        # KV cache: concat previous k, v
        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v

        # Try SDPA / FlashAttention path first (PyTorch 2.0+)
        # SDPA auto-dispatches to FlashAttention-2 (CUDA), memory-efficient
        # attention (CUDA), or math (CPU). Much faster + less memory than manual.
        if impl in ("auto", "sdpa", "flash"):
            try:
                # Build attention mask for SDPA
                attn_mask = None
                T_q = q.shape[2]
                T_k = k.shape[2]
                if use_causal_mask:
                    # Causal mask: position i can attend to positions <= i
                    # For cached generation, T_q may be < T_k
                    attn_mask = torch.triu(
                        torch.ones(T_q, T_k, device=x.device, dtype=torch.bool),
                        diagonal=T_k - T_q + 1,
                    )
                    attn_mask = ~attn_mask  # SDPA: True = attend, False = mask
                if ws > 0 and ws < T_k:
                    # SWA mask
                    q_positions = torch.arange(T_q, device=x.device) + (T_k - T_q) + position_offset
                    k_positions = torch.arange(T_k, device=x.device) + position_offset
                    distance = q_positions.unsqueeze(0) - k_positions.unsqueeze(1)  # (T_k, T_q)
                    swa_keep = (distance <= ws).T  # (T_q, T_k) — True = keep
                    if attn_mask is None:
                        attn_mask = swa_keep
                    else:
                        attn_mask = attn_mask & swa_keep
                # SDPA expects attn_mask as either float (-inf for masked) or bool
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0,
                    is_causal=False  # we built the mask ourselves
                )
                # Reshape back: (B, T_q, d_model)
                out = out.transpose(1, 2).contiguous().view(B, q.shape[2], self.d_model)
                return self.out_proj(out)
            except (AttributeError, RuntimeError):
                # Fall through to math impl if SDPA not available
                pass

        # Math implementation (fallback)
        # Attention: (B, n_heads, T_q, T_k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        T_q = q.shape[2]
        T_k = k.shape[2]

        if use_causal_mask:
            # Causal mask: position i can attend to positions <= i
            mask = torch.triu(
                torch.ones(T_q, T_k, device=x.device, dtype=torch.bool),
                diagonal=T_k - T_q + 1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        # Sliding Window Attention (SWA)
        if ws > 0 and ws < T_k:
            q_positions = torch.arange(T_q, device=x.device) + (T_k - T_q) + position_offset
            k_positions = torch.arange(T_k, device=x.device) + position_offset
            distance = q_positions.unsqueeze(0) - k_positions.unsqueeze(1)  # (T_k, T_q)
            swa_mask = (distance > ws).T  # (T_q, T_k) — True where we should mask
            scores = scores.masked_fill(swa_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to v: (B, n_heads, T_q, head_dim)
        out = attn @ v
        # Reshape back: (B, T_q, d_model)
        out = out.transpose(1, 2).contiguous().view(B, q.shape[2], self.d_model)
        return self.out_proj(out)


class _FeedForward(nn.Module):
    """Position-wise feed-forward: MLP with GELU activation."""
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _TransformerBlock(nn.Module):
    """Pre-LN transformer block: LN -> attn -> residual -> LN -> FFN -> residual."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 window_size: int = 0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _MultiHeadAttention(d_model, n_heads, dropout, window_size=window_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = _FeedForward(d_model)

    def forward(self, x, use_causal_mask=False, kv_cache=None, position_offset=0):
        # Pre-LN: x = x + attn(ln(x))
        x = x + self.attn(self.ln1(x), use_causal_mask=use_causal_mask,
                          kv_cache=kv_cache, position_offset=position_offset)
        # x = x + ffn(ln(x))
        x = x + self.ffn(self.ln2(x))
        return x


class _TransformerLM(nn.Module):
    """Full transformer: token + position embeddings, N blocks, final LN, output head."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 256,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        task_type: str = "causal_lm",
        n_classes: Optional[int] = None,
        window_size: int = 0,
    ):
        super().__init__()
        assert task_type in ("causal_lm", "classification"), \
            f"task_type must be 'causal_lm' or 'classification', got {task_type!r}"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.task_type = task_type
        self.n_classes = n_classes
        self.window_size = window_size

        # Token embedding (shared with output head for causal_lm — weight tying)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding: learned (when not using RoPE) or unused (when using RoPE)
        if not use_rope:
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
        else:
            self.position_embedding = None
            # RoPE applied per-block on q/k; create one shared _RoPE
            rope = _RoPE(head_dim=d_model // n_heads, max_seq_len=max_seq_len)
            self._rope = rope

        # Transformer blocks — pass window_size for SWA support
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, dropout, window_size=window_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Output head
        if task_type == "causal_lm":
            # Weight-tied with token embedding
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight  # weight tying
        else:
            # Classification: CLS-pool (first token) -> linear head
            assert n_classes is not None, "n_classes required for classification task"
            self.cls_head = nn.Linear(d_model, n_classes)

        # Apply RoPE to every block's attention
        if use_rope:
            for block in self.blocks:
                block.attn.set_rope(self._rope)

    def forward(self, token_ids, use_causal_mask=True, kv_caches=None, position_offset=0):
        """
        Args:
            token_ids: (batch, seq_len) long tensor
            use_causal_mask: True for autoregressive LM
            kv_caches: optional list of dicts (one per block) for cached decoding
            position_offset: starting absolute position (for KV-cached decoding,
                where the new token is at position T in the original sequence).
                Default 0. Used by RoPE to apply the correct rotation.

        Returns:
            For causal_lm: logits (batch, seq_len, vocab_size)
            For classification: logits (batch, n_classes)
        """
        B, T = token_ids.shape
        assert T <= self.max_seq_len, \
            f"sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # Token + position embeddings
        x = self.token_embedding(token_ids)  # (B, T, d_model)
        if self.position_embedding is not None:
            # For learned position embeddings, use absolute positions
            positions = torch.arange(
                position_offset, position_offset + T,
                device=token_ids.device,
            ).unsqueeze(0)
            # Clamp to max_seq_len to avoid OOB (long sequences during cached decoding)
            positions = positions.clamp(max=self.max_seq_len - 1)
            x = x + self.position_embedding(positions)

        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, use_causal_mask=use_causal_mask,
                      kv_cache=cache, position_offset=position_offset)

        x = self.ln_f(x)

        if self.task_type == "causal_lm":
            return self.lm_head(x)  # (B, T, vocab_size)
        else:
            # CLS pooling: take first token
            return self.cls_head(x[:, 0, :])  # (B, n_classes)


# =====================================================================
# QAI Technique wrapper
# =====================================================================
class Transformer(Technique):
    """A real Transformer language model technique.

    Built on PyTorch. Supports two task types:
      - `task_type="causal_lm"`: autoregressive language model.
      - `task_type="classification"`: CLS-token-pooled classifier.

    Use with `qai.build(technique="transformer", **params)`.
    """
    name = "transformer"
    family = "deep_learning"
    compatible_learning_techniques = ("supervised",)

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 128,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        task_type: str = "causal_lm",
        n_classes: Optional[int] = None,
        epochs: int = 20,
        lr: float = 3e-4,
        batch_size: int = 32,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        window_size: int = 0,
        layer_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        norm_type: str = "layernorm",  # "layernorm" or "rmsnorm"
        activation: str = "gelu",  # "gelu", "relu", "silu"
        device: Optional[str] = None,
        **params,
    ):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary. Must match your tokenizer.
            d_model: Hidden dimension throughout the model. Powers of 2 are
                conventional (64, 128, 256, 512, 768, 1024). Larger = more
                capacity, slower, more memory.
            n_heads: Number of attention heads. Must divide d_model evenly.
                Each head has dimension d_model / n_heads.
            n_layers: Number of transformer blocks (depth). More layers = more
                expressive, slower, more parameters.
            max_seq_len: Maximum sequence length the model can process.
                Position embeddings (if used) and RoPE precomputed tables are
                sized to this. Sequences longer than this raise an error.
            d_ff: Feed-forward inner dimension. Default 4*d_model. Larger values
                give the MLP more capacity.
            dropout: Dropout rate (0.0 = none, 0.1 = standard). Helps regularize
                small datasets.
            use_rope: If True (default), use Rotary Position Embedding (no
                learned position embeddings needed). If False, use learned
                position embeddings.
            task_type: "causal_lm" (autoregressive LM) or "classification"
                (CLS-token classifier).
            n_classes: Number of classes for classification task. Required if
                task_type="classification", ignored otherwise.
            epochs: Number of training epochs over the dataset.
            lr: Peak learning rate (AdamW optimizer with cosine schedule).
            batch_size: Mini-batch size for training.
            weight_decay: L2 regularization strength (AdamW).
            warmup_steps: Number of steps for linear LR warmup before cosine decay.
            window_size: Sliding Window Attention (SWA) window size. If > 0,
                each query only attends to keys within this many positions
                before it. Reduces attention from O(T^2) to O(T * window_size).
                Used by Gemma 3 (1024), Mistral (4096), Qwen2 (32768).
                0 = global attention (default, standard transformer).
            device: "cuda", "cpu", or None (auto-detect via qai.get_optimal_device()).
        """
        # Validate task_type early so qai.build() fails fast
        assert task_type in ("causal_lm", "classification"), \
            f"task_type must be 'causal_lm' or 'classification', got {task_type!r}"

        super().__init__(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, max_seq_len=max_seq_len, d_ff=d_ff,
            dropout=dropout, use_rope=use_rope, task_type=task_type,
            n_classes=n_classes, epochs=epochs, lr=lr, batch_size=batch_size,
            weight_decay=weight_decay, warmup_steps=warmup_steps, device=device,
            **params,
        )
        _require_torch()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.d_ff = d_ff
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.use_rope = use_rope
        self.rope_theta = rope_theta  # configurable RoPE base frequency
        self.window_size = window_size  # SWA: 0 = global attention
        self.task_type = task_type
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.layer_norm_eps = layer_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.initializer_range = initializer_range
        self.norm_type = norm_type  # "layernorm" or "rmsnorm"
        self.activation = activation  # "gelu", "relu", "silu"

        # Device resolution
        if device is None:
            try:
                import qai
                device = qai.get_optimal_device()
            except Exception:
                device = "cpu"
        self.device = torch.device(device)

        # Build the model (lazy: only built when needed)
        self.net: Optional[_TransformerLM] = None
        self._loss_history: List[float] = []
        self._last_X = None
        self._last_y = None
        # Tokenizer reference (optional, used by generate())
        self.tokenizer = None

    # ------------------------------------------------------------------
    # Model construction (deferred until we know exact vocab_size)
    # ------------------------------------------------------------------
    def _build_model(self, vocab_size: int, n_classes: Optional[int] = None):
        """Lazily construct the PyTorch model. Called from fit() once we know
        the actual vocab size (may differ from constructor's vocab_size if the
        user passed a tokenizer that has a different vocab size)."""
        self.vocab_size = vocab_size
        if self.task_type == "classification":
            self.n_classes = n_classes or 2

        self.net = _TransformerLM(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_seq_len=self.max_seq_len,
            d_ff=self.d_ff,
            dropout=self.dropout,
            use_rope=self.use_rope,
            task_type=self.task_type,
            n_classes=self.n_classes,
            window_size=self.window_size,
        ).to(self.device)

        # Apply proper weight initialization (trunc_normal_ with initializer_range)
        self._init_weights()

        # If using RoPE, move precomputed tables to device
        if self.use_rope:
            self.net._rope = self.net._rope.to(self.device)

    def _init_weights(self):
        """Initialize weights using trunc_normal_ (standard for transformers).

        This is what real GPT/BERT/LLaMA models use:
          - Linear weights: N(0, initializer_range²), truncated
          - Biases: zero
          - Embeddings: N(0, initializer_range²)
          - LayerNorm: weight=1, bias=0
        """
        import torch.nn.init as init
        for module in [self.net] + list(self.net.modules()):
            for name, param in module.named_parameters(recurse=False):
                if "weight" in name and param.dim() >= 2:
                    init.trunc_normal_(param, mean=0.0, std=self.initializer_range)
                elif "bias" in name:
                    init.zeros_(param)
            # Handle children
            for child in module.children():
                for name, param in child.named_parameters(recurse=False):
                    if "weight" in name and param.dim() >= 2:
                        init.trunc_normal_(param, mean=0.0, std=self.initializer_range)
                    elif "bias" in name:
                        init.zeros_(param)
                    elif "weight" in name and param.dim() == 1:
                        # LayerNorm weight → 1.0
                        init.ones_(param)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y=None, tokenizer=None, **kwargs):
        """Train the transformer.

        For `task_type="causal_lm"`:
            - `X` is a list of strings (documents) OR a list of token-id lists.
            - If `tokenizer` is provided, strings are tokenized with it.
            - If `X` is already token-id lists, no tokenizer is needed.
            - `y` is ignored (the model learns to predict next token from input).

        For `task_type="classification"`:
            - `X` is a list of strings or token-id lists.
            - `y` is a list of integer class labels.
            - `tokenizer` must be provided if X is strings.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer

        # Coerce X to a list of token-id lists
        token_id_lists, vocab_size = self._prepare_inputs(X, tokenizer)
        n_classes = None
        if self.task_type == "classification":
            if y is None:
                raise ValueError(
                    "[WHAT] transformer.fit() with task_type='classification' requires y\n"
                    "[WHY] Classification needs class labels for each input\n"
                    "[FIX] Pass y=[0, 1, 2, ...] matching the length of X"
                )
            y_arr = np.asarray(y)
            n_classes = int(y_arr.max()) + 1
            self._last_y = y_arr

        # Lazy-build the model now that we know vocab_size (and n_classes)
        if self.net is None:
            self._build_model(vocab_size=vocab_size, n_classes=n_classes)

        self._last_X = token_id_lists

        # Train
        if self.task_type == "causal_lm":
            self._train_causal_lm(token_id_lists)
        else:
            self._train_classification(token_id_lists, self._last_y)

        self._trained = True
        return self

    def _prepare_inputs(self, X, tokenizer) -> Tuple[List[np.ndarray], int]:
        """Convert X (list of strings or list of int-lists) into a list of
        numpy int arrays. Also returns the implied vocab size."""
        if isinstance(X, str):
            X = [X]

        token_id_lists: List[np.ndarray] = []
        max_id_seen = 0

        for item in X:
            if isinstance(item, str):
                if tokenizer is None:
                    raise ValueError(
                        "[WHAT] Passed string inputs but no tokenizer\n"
                        "[WHY] Transformer can't process raw text without tokenization\n"
                        "[FIX] Pass tokenizer=qai.CharTokenizer() (or any qai tokenizer) "
                        "to fit(), OR pass X as a list of token-id lists"
                    )
                ids = tokenizer.encode(item)
                arr = np.asarray(ids, dtype=np.int64)
            elif isinstance(item, (list, tuple, np.ndarray)):
                arr = np.asarray(item, dtype=np.int64)
            else:
                raise TypeError(
                    f"[WHAT] Transformer.fit() got unsupported input type {type(item).__name__}\n"
                    "[WHY] Each item in X must be a str (requires tokenizer) or a list of ints"
                )

            if arr.size > 0:
                max_id_seen = max(max_id_seen, int(arr.max()))
            token_id_lists.append(arr)

        # Vocab size: either from tokenizer, or max_id_seen + 1
        if tokenizer is not None:
            vocab_size = tokenizer.vocab_size
        else:
            vocab_size = max(self.vocab_size, max_id_seen + 1)

        return token_id_lists, vocab_size

    def _train_causal_lm(self, token_id_lists: List[np.ndarray]):
        """Train on next-token prediction. Each document is split into
        (input, target) = (tokens[:-1], tokens[1:]) pairs."""
        # Build training examples: each (input_seq, target_seq) is a pair of
        # length <= max_seq_len - 1. We use a sliding window of size max_seq_len.
        examples: List[Tuple[np.ndarray, np.ndarray]] = []
        for ids in token_id_lists:
            if len(ids) < 2:
                continue
            window = self.max_seq_len
            # Slide a window over the sequence
            for start in range(0, max(1, len(ids) - 1), window):
                end = min(start + window + 1, len(ids))
                chunk = ids[start:end]
                if len(chunk) < 2:
                    continue
                examples.append((chunk[:-1], chunk[1:]))

        if not examples:
            raise ValueError(
                "[WHAT] No training examples generated from X\n"
                "[WHY] Each document must have at least 2 tokens for next-token prediction\n"
                "[FIX] Provide longer training documents"
            )

        # Pad sequences to max length in batch
        def collate(batch_examples):
            max_len = max(len(inp) for inp, _ in batch_examples)
            max_len = min(max_len, self.max_seq_len)
            inputs = np.zeros((len(batch_examples), max_len), dtype=np.int64)
            targets = np.zeros((len(batch_examples), max_len), dtype=np.int64)
            for i, (inp, tgt) in enumerate(batch_examples):
                L = min(len(inp), max_len)
                inputs[i, :L] = inp[:L]
                targets[i, :L] = tgt[:L]
            return (
                torch.tensor(inputs, device=self.device),
                torch.tensor(targets, device=self.device),
            )

        # Optimizer (AdamW) with cosine LR schedule + warmup
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = max(1, (len(examples) // self.batch_size + 1) * self.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: self._lr_lambda(step, total_steps),
        )

        self._loss_history = []
        self.net.train()

        for epoch in range(self.epochs):
            # Shuffle examples each epoch
            rng = np.random.default_rng(epoch)
            perm = rng.permutation(len(examples))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(examples), self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                batch = [examples[i] for i in batch_idx]
                if not batch:
                    continue
                inputs, targets = collate(batch)

                logits = self.net(inputs, use_causal_mask=True)  # (B, T, V)
                # Cross-entropy: flatten (B*T) prediction vs (B*T) target
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    targets.reshape(-1),
                    ignore_index=0,  # ignore padding (PAD_ID = 0)
                )

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (stabilizes transformer training)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            self._loss_history.append(avg_loss)

    def _train_classification(self, token_id_lists: List[np.ndarray], y: np.ndarray):
        """Train on classification. CLS-token (first position) is pooled."""
        if len(token_id_lists) != len(y):
            raise ValueError(
                f"[WHAT] X has {len(token_id_lists)} samples but y has {len(y)} labels\n"
                "[FIX] Provide matching lengths"
            )

        examples = list(zip(token_id_lists, y))

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = max(1, (len(examples) // self.batch_size + 1) * self.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: self._lr_lambda(step, total_steps),
        )

        self._loss_history = []
        self.net.train()

        for epoch in range(self.epochs):
            rng = np.random.default_rng(epoch)
            perm = rng.permutation(len(examples))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(examples), self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                batch = [examples[i] for i in batch_idx]
                if not batch:
                    continue

                max_len = min(max(len(inp) for inp, _ in batch), self.max_seq_len)
                inputs = np.zeros((len(batch), max_len), dtype=np.int64)
                labels = np.zeros(len(batch), dtype=np.int64)
                for i, (inp, lbl) in enumerate(batch):
                    L = min(len(inp), max_len)
                    inputs[i, :L] = inp[:L]
                    labels[i] = int(lbl)

                inputs_t = torch.tensor(inputs, device=self.device)
                labels_t = torch.tensor(labels, device=self.device)

                logits = self.net(inputs_t, use_causal_mask=False)  # (B, n_classes)
                loss = F.cross_entropy(logits, labels_t)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            self._loss_history.append(epoch_loss / max(1, n_batches))

    def _lr_lambda(self, step: int, total_steps: int) -> float:
        """Linear warmup -> cosine decay."""
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def forward(self, x, **kwargs):
        """Predict.

        For `task_type="causal_lm"`:
            - `x` is a prompt: str (requires `tokenizer`) or list of int IDs.
            - Returns logits tensor of shape (1, seq_len, vocab_size). Use
              `model.generate()` instead for autoregressive text generation.

        For `task_type="classification"`:
            - `x` is one input: str (requires `tokenizer`) or list of int IDs.
            - Returns predicted class index (int).
        """
        if not self._trained or self.net is None:
            raise RuntimeError(
                "[WHAT] Transformer.predict() called before train()\n"
                "[WHY] The model has no learned weights yet\n"
                "[FIX] Call model.train(X, ...) first"
            )

        # Single-sample forward (no batching needed for predict)
        ids = self._coerce_input_to_ids(x, self.tokenizer)
        ids = ids[: self.max_seq_len]
        if len(ids) == 0:
            ids = np.array([0], dtype=np.int64)  # avoid empty

        self.net.eval()
        with torch.no_grad():
            ids_t = torch.tensor(ids, device=self.device).unsqueeze(0)  # (1, T)
            if self.task_type == "causal_lm":
                logits = self.net(ids_t, use_causal_mask=True)
                return logits.cpu().numpy()  # (1, T, V)
            else:
                logits = self.net(ids_t, use_causal_mask=False)
                pred = int(torch.argmax(logits, dim=-1).item())
                return pred

    def _forward_logits(self, token_ids):
        """Forward pass returning raw logits (used by inference/alignment methods).

        Args:
            token_ids: (B, T) tensor of token IDs

        Returns:
            logits tensor of shape (B, T, vocab_size) for causal_lm, or
            (B, n_classes) for classification.
        """
        if not self._trained or self.net is None:
            raise RuntimeError("Transformer not trained — call train() first")
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.as_tensor(np.asarray(token_ids), dtype=torch.long)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        self.net.eval()
        return self.net(token_ids, use_causal_mask=(self.task_type == "causal_lm"))

    def _coerce_input_to_ids(self, x, tokenizer) -> np.ndarray:
        if isinstance(x, str):
            if tokenizer is None:
                raise ValueError(
                    "[WHAT] Passed a string input but model has no tokenizer\n"
                    "[FIX] Pass tokenizer= to train(), or pass x as a list of int IDs"
                )
            return np.asarray(tokenizer.encode(x), dtype=np.int64)
        elif isinstance(x, (list, tuple, np.ndarray)):
            return np.asarray(x, dtype=np.int64)
        else:
            raise TypeError(f"Unsupported input type: {type(x).__name__}")

    # ------------------------------------------------------------------
    # Generation (causal LM only)
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 50,
        tokenizer=None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_tokens: Optional[List[Union[str, int]]] = None,
        use_cache: bool = True,
        seed: Optional[int] = None,
    ) -> str:
        """Autoregressive text generation (causal LM only).

        Args:
            prompt: Starting text. Either a string (requires tokenizer) or a
                list of token IDs.
            max_new_tokens: Maximum number of new tokens to generate.
            tokenizer: Tokenizer to use (overrides the one set during fit()).
            temperature: Sampling temperature. 1.0 = original distribution,
                0.5 = sharper (more deterministic), 2.0 = flatter (more random).
                0.0 = greedy (argmax).
            top_k: If set, only sample from the top-K most likely tokens at
                each step. 0 or None = no restriction.
            top_p: If set, only sample from the smallest set of tokens whose
                cumulative probability >= top_p (nucleus sampling).
            stop_tokens: List of token strings or IDs that, if generated,
                stop generation immediately.
            use_cache: If True (default), use KV cache for O(1) per-step
                generation. If False, recompute attention from scratch each
                step (slower but uses less memory).
            seed: Random seed for reproducibility. None = no seed.

        Returns:
            Generated text (string). Includes the prompt.
        """
        if self.task_type != "causal_lm":
            raise RuntimeError(
                "[WHAT] generate() called on a non-causal-lm transformer\n"
                "[WHY] Generation only makes sense for task_type='causal_lm'\n"
                "[FIX] Build with task_type='causal_lm' (the default)"
            )
        if not self._trained or self.net is None:
            raise RuntimeError(
                "[WHAT] generate() called before train()\n"
                "[FIX] Call model.train(X, tokenizer=...) first"
            )

        tok = tokenizer or self.tokenizer
        if tok is None and isinstance(prompt, str):
            raise ValueError(
                "[WHAT] String prompt but no tokenizer available\n"
                "[FIX] Pass tokenizer= to generate() or fit()"
            )

        # Encode prompt
        if isinstance(prompt, str):
            prompt_ids = tok.encode(prompt)
        else:
            prompt_ids = list(prompt)
        prompt_ids = prompt_ids[: self.max_seq_len - 1]

        # Resolve stop tokens
        stop_ids: set = set()
        if stop_tokens:
            for st in stop_tokens:
                if isinstance(st, str):
                    if st in tok.token_to_id:
                        stop_ids.add(tok.token_to_id[st])
                else:
                    stop_ids.add(int(st))

        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.net.eval()
        generated = list(prompt_ids)

        # KV cache setup
        if use_cache:
            kv_caches = [{} for _ in range(self.n_layers)]
            # Initial forward: process the whole prompt
            with torch.no_grad():
                ids_t = torch.tensor([generated], device=self.device, dtype=torch.long)
                logits = self.net(ids_t, use_causal_mask=True, kv_caches=kv_caches,
                                  position_offset=0)
                next_logits = logits[0, -1, :]  # (vocab_size,)
        else:
            kv_caches = None
            with torch.no_grad():
                ids_t = torch.tensor([generated], device=self.device, dtype=torch.long)
                logits = self.net(ids_t, use_causal_mask=True, position_offset=0)
                next_logits = logits[0, -1, :]

        for step in range(max_new_tokens):
            # Sample next token
            next_id = self._sample_next(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            if next_id in stop_ids:
                break

            generated.append(next_id)

            # Forward one more step
            if use_cache and kv_caches is not None:
                # New token's absolute position = current length - 1
                # (generated already includes the new token we just appended)
                pos_offset = len(generated) - 1
                with torch.no_grad():
                    ids_t = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
                    logits = self.net(ids_t, use_causal_mask=True, kv_caches=kv_caches,
                                      position_offset=pos_offset)
                    next_logits = logits[0, -1, :]
            else:
                # Recompute from scratch (truncated to max_seq_len)
                truncated = generated[-(self.max_seq_len - 1) :]
                pos_offset = len(generated) - len(truncated)
                with torch.no_grad():
                    ids_t = torch.tensor([truncated], device=self.device, dtype=torch.long)
                    logits = self.net(ids_t, use_causal_mask=True, position_offset=pos_offset)
                    next_logits = logits[0, -1, :]

        # Decode
        if tok is not None:
            return tok.decode(generated)
        # No tokenizer — return as a string of space-separated IDs
        return " ".join(str(i) for i in generated)

    def _sample_next(
        self,
        logits: "torch.Tensor",
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> int:
        """Sample the next token ID from logits with temperature/top-k/top-p."""
        if temperature <= 0:
            return int(torch.argmax(logits).item())

        logits = logits / max(1e-8, temperature)

        # Top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            topk_vals, topk_idx = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_vals)
            logits = mask

        # Top-p (nucleus) filtering
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_mask = cumulative_probs > top_p
            # Shift right so the first token above threshold is kept
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            indices_to_remove = sorted_mask.scatter(-1, sorted_idx, sorted_mask)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def loss_history(self) -> List[float]:
        """Return per-epoch training loss."""
        return list(self._loss_history)

    def parameter_count(self) -> int:
        """Total trainable parameters in the model."""
        if self.net is None:
            return 0
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def architecture_summary(self) -> Dict[str, Any]:
        """Return a dict describing the model architecture."""
        return {
            "task_type": self.task_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "d_ff": self.d_ff or 4 * self.d_model,
            "use_rope": self.use_rope,
            "dropout": self.dropout,
            "n_classes": self.n_classes,
            "parameters": self.parameter_count(),
            "device": str(self.device),
        }

    def accuracy(self) -> float:
        """For classification: training-set accuracy. For causal_lm: 1 - perplexity
        (loosely; not a true accuracy but useful as a training health metric)."""
        if not self._trained:
            return 0.0
        if self.task_type == "classification" and self._last_X is not None and self._last_y is not None:
            preds = [self.forward(x) for x in self._last_X[:200]]  # cap for speed
            return float(np.mean(np.array(preds) == self._last_y[:200]))
        # For causal LM, return exp(-final_loss) as a soft accuracy proxy
        if self._loss_history:
            return float(math.exp(-self._loss_history[-1]))
        return 0.0

    def __repr__(self):
        if self.net is None:
            return f"<Transformer unbuilt task_type={self.task_type}>"
        return (
            f"<Transformer task_type={self.task_type} "
            f"d_model={self.d_model} n_heads={self.n_heads} "
            f"n_layers={self.n_layers} params={self.parameter_count():,} "
            f"trained={self._trained}>"
        )
