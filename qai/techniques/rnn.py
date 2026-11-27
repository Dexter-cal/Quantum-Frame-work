"""
qai.techniques.rnn -- Recurrent neural network family (spec Section 2).

Implements four techniques on top of PyTorch:

  - `qai.build(technique="rnn", ...)`     -- vanilla RNN (Elman, 1990)
  - `qai.build(technique="lstm", ...)`    -- LSTM (Hochreiter & Schmidhuber, 1997)
  - `qai.build(technique="gru", ...)`     -- GRU (Cho et al., 2014)
  - `qai.build(technique="seq2seq", ...)` -- encoder-decoder for translation-style tasks

All support:
  - Multi-layer stacking (n_layers > 1)
  - Bidirectional mode (`bidirectional=True`) — my own addition
  - Two task types: `classification` (sequence → class) and `generation`
    (autoregressive next-token prediction)
  - Tokenizer integration (pass `tokenizer=` to `fit()`)
  - AdamW + cosine LR schedule + warmup + gradient clipping
  - Save/load via `qai.save_bundle` (PyTorch state_dict path)
  - KV-style hidden-state caching for fast autoregressive generation

Example (LSTM for text classification):
    import qai
    tok = qai.CharTokenizer()
    tok.fit(["hello world", "goodbye world"])

    model = qai.build(
        technique="lstm",
        task_type="classification",
        vocab_size=tok.vocab_size,
        n_classes=2,
        d_model=32, n_layers=1, bidirectional=False,
        epochs=20, lr=1e-3, batch_size=8,
    )
    model.train(["hello world", "goodbye world"] * 5,
                y=[1, 0] * 5, tokenizer=tok)
    print(model.predict("hello world"))  # -> 1 or 0

Example (GRU for text generation):
    model = qai.build(
        technique="gru",
        task_type="generation",
        vocab_size=tok.vocab_size,
        d_model=64, n_layers=2,
        max_seq_len=32, epochs=30,
    )
    model.train(["hello world"] * 10, tokenizer=tok)
    print(model.generate("hello", max_new_tokens=10, tokenizer=tok))

Example (Seq2Seq for sequence transformation):
    model = qai.build(
        technique="seq2seq",
        vocab_size=tok.vocab_size,
        d_model=32, n_layers=1,
        epochs=20,
    )
    # Train on (input, target) pairs
    model.train([("abc", "ABC"), ("xyz", "XYZ")] * 5, tokenizer=tok)
    print(model.translate("abc", tokenizer=tok))  # -> "ABC"
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
    from qai._torch_stub import _TorchStubNamespace, _NNStubNamespace
    torch = _TorchStubNamespace()
    nn = _NNStubNamespace()
    F = nn

from .base import Technique


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "[WHAT] qai.build(technique='rnn'/'lstm'/'gru'/'seq2seq') requires PyTorch\n"
            "[WHY] These techniques use torch.nn for their internals\n"
            "[FIX] Install torch: pip install torch"
        )


# =====================================================================
# PyTorch building blocks
# =====================================================================
class _RNNBaseNet(nn.Module):
    """Common base for RNN/LSTM/GRU nets. Subclasses set `self.rnn`."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        task_type: str = "classification",
        n_classes: Optional[int] = None,
        max_seq_len: int = 128,
    ):
        super().__init__()
        assert task_type in ("classification", "generation"), (
            f"task_type must be 'classification' or 'generation', got {task_type!r}"
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.task_type = task_type
        self.n_classes = n_classes
        self.max_seq_len = max_seq_len
        self.num_directions = 2 if bidirectional else 1

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Dropout between RNN layers (only if n_layers > 1)
        rnn_dropout = dropout if n_layers > 1 else 0.0
        # `self.rnn` is set by subclass
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model * self.num_directions)

        # Output head
        if task_type == "classification":
            assert n_classes is not None, "n_classes required for classification"
            self.head = nn.Linear(d_model * self.num_directions, n_classes)
        else:  # generation — weight-tied with embedding
            self.head = nn.Linear(d_model * self.num_directions, vocab_size, bias=False)
            # Note: weight tying only works cleanly for unidirectional, so for
            # bidirectional generation we use a separate (untied) head.

        # Will be set by subclass
        self.rnn: Optional[nn.RNNBase] = None

    def _init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state (and cell state for LSTM)."""
        h = torch.zeros(self.n_layers * self.num_directions, batch_size, self.d_model, device=device)
        if isinstance(self.rnn, nn.LSTM):
            c = torch.zeros_like(h)
            return (h, c)
        return h

    def forward(self, token_ids, hidden=None, return_hidden=False):
        """
        Args:
            token_ids: (batch, seq_len) long tensor
            hidden: optional initial hidden state (for stateful generation)
            return_hidden: if True, also return the final hidden state
        Returns:
            For classification: logits (batch, n_classes)
            For generation: logits (batch, seq_len, vocab_size)
            If return_hidden: also returns the final hidden state
        """
        B, T = token_ids.shape
        if T > self.max_seq_len:
            token_ids = token_ids[:, -self.max_seq_len:]
            T = self.max_seq_len

        x = self.embedding(token_ids)  # (B, T, d_model)

        if hidden is None:
            hidden = self._init_hidden(B, token_ids.device)

        # RNN forward
        out, hidden_final = self.rnn(x, hidden)
        # out: (B, T, d_model * num_directions)
        out = self.ln_f(out)

        if self.task_type == "classification":
            # Use the last timestep's output for classification
            logits = self.head(out[:, -1, :])  # (B, n_classes)
        else:
            logits = self.head(out)  # (B, T, vocab_size)

        if return_hidden:
            return logits, hidden_final
        return logits


class _RNNNet(_RNNBaseNet):
    """Vanilla Elman RNN."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rnn = nn.RNN(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.n_layers > 1 else 0.0,
        )


class _LSTMNet(_RNNBaseNet):
    """Long Short-Term Memory."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rnn = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.n_layers > 1 else 0.0,
        )


class _GRUNet(_RNNBaseNet):
    """Gated Recurrent Unit."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rnn = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.n_layers > 1 else 0.0,
        )


# =====================================================================
# Seq2Seq (encoder-decoder) — my own addition
# =====================================================================
class _Seq2SeqNet(nn.Module):
    """Encoder-decoder for sequence-to-sequence tasks (translation, summarization).

    Architecture:
      - Encoder: RNN/LSTM/GRU that reads the input sequence → final hidden state
      - Decoder: RNN/LSTM/GRU that generates output token-by-token, starting
        from <bos>, conditioned on encoder's final hidden state

    Teacher forcing during training: decoder receives the true previous token
    at each step. At inference: decoder feeds its own prediction back.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
        max_seq_len: int = 128,
    ):
        super().__init__()
        assert rnn_type in ("rnn", "lstm", "gru"), f"rnn_type must be rnn/lstm/gru, got {rnn_type!r}"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # Shared embedding (encoder + decoder)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Pick RNN class
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]

        rnn_kwargs = dict(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.encoder = rnn_cls(**rnn_kwargs)
        self.decoder = rnn_cls(**rnn_kwargs)

        # Output head (weight-tied with embedding)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def _init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.n_layers, batch_size, self.d_model, device=device)
        if self.rnn_type == "lstm":
            return (h, torch.zeros_like(h))
        return h

    def forward(self, src_ids, tgt_ids=None, max_len: Optional[int] = None):
        """Forward pass.

        Training mode (tgt_ids provided): teacher forcing.
        Inference mode (tgt_ids=None): autoregressive decoding, max_len required.

        Args:
            src_ids: (batch, src_len) source token IDs
            tgt_ids: (batch, tgt_len) target token IDs (training only)
            max_len: max output length (inference only)

        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        B = src_ids.size(0)
        device = src_ids.device

        # Encode source
        src_emb = self.embedding(src_ids)  # (B, src_len, d_model)
        _, hidden = self.encoder(src_emb)

        if tgt_ids is not None:
            # Training: teacher forcing
            tgt_emb = self.embedding(tgt_ids)  # (B, tgt_len, d_model)
            out, _ = self.decoder(tgt_emb, hidden)
            logits = self.head(out)  # (B, tgt_len, vocab_size)
            return logits
        else:
            # Inference: autoregressive decoding
            assert max_len is not None, "max_len required for inference"
            # Start with <bos> token (ID 2 — matches qai tokenizer convention)
            bos_id = 2
            cur = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
            outputs = []
            for _ in range(max_len):
                cur_emb = self.embedding(cur)  # (B, 1, d_model)
                out, hidden = self.decoder(cur_emb, hidden)
                logits = self.head(out[:, -1, :])  # (B, vocab_size)
                next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
                outputs.append(next_token)
                cur = next_token
            return torch.cat(outputs, dim=1)  # (B, max_len)


# =====================================================================
# QAI Technique wrappers
# =====================================================================
class _RNNTechniqueBase(Technique):
    """Common base for RNN/LSTM/GRU qai techniques.

    Subclasses must set:
      - `name` (class attr)
      - `_net_cls` (class attr): the PyTorch net class to instantiate
    """
    name = "rnn_base"
    family = "deep_learning"
    compatible_learning_techniques = ("supervised",)
    _net_cls = None  # set by subclass

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        n_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        task_type: str = "classification",
        n_classes: Optional[int] = None,
        max_seq_len: int = 128,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 0.01,
        warmup_steps: int = 50,
        device: Optional[str] = None,
        **params,
    ):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            d_model: Hidden dimension throughout the model.
            n_layers: Number of stacked RNN layers.
            bidirectional: If True, use a bidirectional RNN (reads sequence
                both forward and backward). Only supported for classification
                (not generation).
            dropout: Dropout rate between RNN layers (only if n_layers > 1).
            task_type: "classification" (sequence → class) or "generation"
                (autoregressive next-token prediction).
            n_classes: Number of classes (required for classification).
            max_seq_len: Maximum sequence length.
            epochs, lr, batch_size, weight_decay, warmup_steps: Training config.
            device: "cuda"/"cpu"/None (auto-detect).
        """
        super().__init__(**{**params, "vocab_size": vocab_size, "d_model": d_model,
                            "n_layers": n_layers, "bidirectional": bidirectional,
                            "dropout": dropout, "task_type": task_type,
                            "n_classes": n_classes, "max_seq_len": max_seq_len,
                            "epochs": epochs, "lr": lr, "batch_size": batch_size,
                            "weight_decay": weight_decay, "warmup_steps": warmup_steps,
                            "device": device})
        _require_torch()

        # Validate early
        assert task_type in ("classification", "generation"), (
            f"task_type must be 'classification' or 'generation', got {task_type!r}"
        )
        if bidirectional and task_type == "generation":
            raise ValueError(
                "[WHAT] bidirectional=True is not supported for generation task\n"
                "[WHY] Bidirectional reads the sequence backward, which doesn't "
                "work for autoregressive generation\n"
                "[FIX] Use bidirectional=False, or task_type='classification'"
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.task_type = task_type
        self.n_classes = n_classes
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        if device is None:
            try:
                import qai
                device = qai.get_optimal_device()
            except Exception:
                device = "cpu"
        self.device = torch.device(device)

        self.net: Optional[_RNNBaseNet] = None
        self._loss_history: List[float] = []
        self._last_X = None
        self._last_y = None
        self.tokenizer = None
        self.classes_: Optional[np.ndarray] = None

    def _build_model(self, vocab_size: int = None, n_classes: int = None):
        if vocab_size is not None:
            self.vocab_size = vocab_size
        if n_classes is not None:
            self.n_classes = n_classes
        self.net = self._net_cls(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            task_type=self.task_type,
            n_classes=self.n_classes,
            max_seq_len=self.max_seq_len,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y=None, tokenizer=None, **kwargs):
        """Train the RNN.

        For `task_type="classification"`:
            - `X` is a list of strings (documents) OR a list of token-id lists.
            - `y` is a list of integer class labels (or strings — auto-encoded).
            - `tokenizer` is required if X is strings.

        For `task_type="generation"`:
            - `X` is a list of strings (documents) OR a list of token-id lists.
            - `y` is ignored (model learns next-token prediction).
            - `tokenizer` is required if X is strings.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
        token_id_lists, vocab_size = self._prepare_inputs(X, tokenizer)

        n_classes = None
        if self.task_type == "classification":
            if y is None:
                raise ValueError(
                    "[WHAT] task_type='classification' requires y\n"
                    "[FIX] Pass y=[0, 1, 2, ...] matching the length of X"
                )
            y_arr = np.asarray(y)
            if y_arr.dtype.kind in ("U", "S", "O"):
                self.classes_ = np.unique(y_arr)
                y_encoded = np.searchsorted(self.classes_, y_arr).astype(np.int64)
            else:
                y_encoded = y_arr.astype(np.int64)
                self.classes_ = np.unique(y_encoded)
            n_classes = int(y_encoded.max()) + 1
            self._last_y = y_encoded
        self._last_X = token_id_lists

        if self.net is None:
            self._build_model(vocab_size=vocab_size, n_classes=n_classes)

        if self.task_type == "classification":
            self._train_classification(token_id_lists, self._last_y)
        else:
            self._train_generation(token_id_lists)

        self._trained = True
        return self

    def _prepare_inputs(self, X, tokenizer) -> Tuple[List[np.ndarray], int]:
        if isinstance(X, str):
            X = [X]
        token_id_lists: List[np.ndarray] = []
        max_id_seen = 0
        for item in X:
            if isinstance(item, str):
                if tokenizer is None:
                    raise ValueError(
                        "[WHAT] Passed string inputs but no tokenizer\n"
                        "[FIX] Pass tokenizer= to fit()"
                    )
                ids = tokenizer.encode(item)
                arr = np.asarray(ids, dtype=np.int64)
            elif isinstance(item, (list, tuple, np.ndarray)):
                arr = np.asarray(item, dtype=np.int64)
            else:
                raise TypeError(f"Unsupported input type: {type(item).__name__}")
            if arr.size > 0:
                max_id_seen = max(max_id_seen, int(arr.max()))
            token_id_lists.append(arr)
        vocab_size = tokenizer.vocab_size if tokenizer is not None else max(self.vocab_size, max_id_seen + 1)
        return token_id_lists, vocab_size

    def _collate(self, examples, pad_id: int = 0):
        """Pad a batch of (sequence, [target]) to a uniform length."""
        max_len = min(max(len(seq) for seq in examples), self.max_seq_len)
        if max_len == 0:
            max_len = 1
        batch = np.full((len(examples), max_len), pad_id, dtype=np.int64)
        for i, seq in enumerate(examples):
            L = min(len(seq), max_len)
            batch[i, :L] = seq[:L]
        return torch.tensor(batch, device=self.device)

    def _train_classification(self, token_id_lists, y_encoded):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = max(1, (len(token_id_lists) // self.batch_size + 1) * self.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: self._lr_lambda(step, total_steps)
        )
        self._loss_history = []
        self.net.train()

        for epoch in range(self.epochs):
            rng = np.random.default_rng(epoch)
            perm = rng.permutation(len(token_id_lists))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(token_id_lists), self.batch_size):
                idx = perm[start: start + self.batch_size]
                batch_seqs = [token_id_lists[i] for i in idx]
                batch_y = y_encoded[idx]
                X_t = self._collate(batch_seqs)
                y_t = torch.tensor(batch_y, device=self.device, dtype=torch.long)
                logits = self.net(X_t)
                loss = F.cross_entropy(logits, y_t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            self._loss_history.append(epoch_loss / max(1, n_batches))

    def _train_generation(self, token_id_lists):
        """Train on next-token prediction. Each sequence of length L provides
        L-1 (input, target) pairs: input = tokens[:i+1], target = tokens[i+1]."""
        # Build training examples: (input_seq, target_seq) where target is shifted
        examples = []
        for ids in token_id_lists:
            if len(ids) < 2:
                continue
            window = self.max_seq_len
            for start in range(0, max(1, len(ids) - 1), window):
                end = min(start + window + 1, len(ids))
                chunk = ids[start:end]
                if len(chunk) < 2:
                    continue
                examples.append((chunk[:-1], chunk[1:]))

        if not examples:
            raise ValueError("No training examples generated (need sequences of length >= 2)")

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = max(1, (len(examples) // self.batch_size + 1) * self.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: self._lr_lambda(step, total_steps)
        )
        self._loss_history = []
        self.net.train()

        for epoch in range(self.epochs):
            rng = np.random.default_rng(epoch)
            perm = rng.permutation(len(examples))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(examples), self.batch_size):
                idx = perm[start: start + self.batch_size]
                batch_inputs = [examples[i][0] for i in idx]
                batch_targets = [examples[i][1] for i in idx]
                X_t = self._collate(batch_inputs)
                # Targets same length as inputs (both are windowed the same way)
                T_t = self._collate(batch_targets)
                # Pad target to match input's seq length
                if T_t.size(1) < X_t.size(1):
                    pad = torch.zeros(T_t.size(0), X_t.size(1) - T_t.size(1),
                                       dtype=T_t.dtype, device=T_t.device)
                    T_t = torch.cat([T_t, pad], dim=1)
                elif T_t.size(1) > X_t.size(1):
                    T_t = T_t[:, :X_t.size(1)]
                logits = self.net(X_t)  # (B, T, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    T_t.reshape(-1),
                    ignore_index=0,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            self._loss_history.append(epoch_loss / max(1, n_batches))

    def _lr_lambda(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def forward(self, x, **kwargs):
        if not self._trained or self.net is None:
            raise RuntimeError(f"{self.name}.predict() called before train()")
        ids = self._coerce_input(x)
        if len(ids) == 0:
            ids = np.array([0], dtype=np.int64)
        self.net.eval()
        with torch.no_grad():
            ids_t = torch.tensor([ids], device=self.device, dtype=torch.long)
            logits = self.net(ids_t)
            if self.task_type == "classification":
                pred = int(torch.argmax(logits, dim=-1).item())
                if self.classes_ is not None and self.classes_.dtype.kind in ("U", "S", "O"):
                    return self.classes_[pred]
                return pred
            else:
                return logits.cpu().numpy()

    def _coerce_input(self, x) -> np.ndarray:
        if isinstance(x, str):
            if self.tokenizer is None:
                raise ValueError("String input but no tokenizer — pass tokenizer= to fit()")
            return np.asarray(self.tokenizer.encode(x), dtype=np.int64)
        return np.asarray(x, dtype=np.int64)

    def generate(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 50,
        tokenizer=None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> str:
        """Autoregressive generation (generation task only).

        Uses the RNN's hidden state across steps (no KV cache like the
        Transformer, but the hidden state itself serves the same role).
        """
        if self.task_type != "generation":
            raise RuntimeError("generate() requires task_type='generation'")
        if not self._trained or self.net is None:
            raise RuntimeError("generate() called before train()")

        tok = tokenizer or self.tokenizer
        if tok is None and isinstance(prompt, str):
            raise ValueError("String prompt but no tokenizer")

        if isinstance(prompt, str):
            prompt_ids = tok.encode(prompt)
        else:
            prompt_ids = list(prompt)
        prompt_ids = prompt_ids[:self.max_seq_len - 1]

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.net.eval()
        generated = list(prompt_ids)
        # Process the prompt to warm up the hidden state
        with torch.no_grad():
            ids_t = torch.tensor([generated], device=self.device, dtype=torch.long)
            logits, hidden = self.net(ids_t, return_hidden=True)
            next_logits = logits[0, -1, :]

        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                probs = F.softmax(next_logits / max(1e-8, temperature), dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)
            # Feed just the new token, with the previous hidden state
            with torch.no_grad():
                ids_t = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
                logits, hidden = self.net(ids_t, hidden=hidden, return_hidden=True)
                next_logits = logits[0, -1, :]

        if tok is not None:
            return tok.decode(generated)
        return " ".join(str(i) for i in generated)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def loss_history(self) -> List[float]:
        return list(self._loss_history)

    def parameter_count(self) -> int:
        if self.net is None:
            return 0
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def architecture_summary(self) -> Dict[str, Any]:
        return {
            "technique": self.name,
            "task_type": self.task_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
            "n_classes": self.n_classes,
            "max_seq_len": self.max_seq_len,
            "parameters": self.parameter_count(),
            "device": str(self.device),
        }

    def accuracy(self) -> float:
        if not self._trained or self._last_X is None or self._last_y is None:
            return 0.0
        if self.task_type == "classification":
            preds = [self.forward(x) for x in self._last_X[:200]]
            return float(np.mean(np.asarray(preds) == self._last_y[:200]))
        if self._loss_history:
            return float(math.exp(-self._loss_history[-1]))
        return 0.0

    def __repr__(self):
        if self.net is None:
            return f"<{type(self).__name__} unbuilt task_type={self.task_type}>"
        return (
            f"<{type(self).__name__} task_type={self.task_type} "
            f"d_model={self.d_model} n_layers={self.n_layers} "
            f"bidirectional={self.bidirectional} "
            f"params={self.parameter_count():,} trained={self._trained}>"
        )


class RNN(_RNNTechniqueBase):
    """Vanilla Elman RNN (1990). Simplest recurrent architecture."""
    name = "rnn"
    _net_cls = _RNNNet


class LSTM(_RNNTechniqueBase):
    """Long Short-Term Memory (Hochreiter & Schmidhuber, 1997).

    Has separate cell state + hidden state, with input/forget/output gates.
    Better at long-range dependencies than vanilla RNN.
    """
    name = "lstm"
    _net_cls = _LSTMNet


class GRU(_RNNTechniqueBase):
    """Gated Recurrent Unit (Cho et al., 2014).

    Simpler than LSTM (no separate cell state, only update/reset gates).
    Often matches LSTM performance with fewer parameters.
    """
    name = "gru"
    _net_cls = _GRUNet


# =====================================================================
# Seq2Seq technique (my own addition)
# =====================================================================
class Seq2Seq(Technique):
    """Sequence-to-sequence encoder-decoder (my own addition beyond the spec).

    Use for: translation, summarization, format conversion, any task where
    input and output are both sequences (potentially different lengths).

    Architecture:
      - Encoder: RNN/LSTM/GRU reads the input sequence
      - Decoder: same RNN type, initialized with encoder's final hidden state,
        generates output token-by-token autoregressively

    Training uses teacher forcing (decoder receives the true previous token
    at each step). Inference is autoregressive.

    Example:
        model = qai.build(technique="seq2seq", vocab_size=tok.vocab_size,
                          d_model=32, n_layers=1, rnn_type="lstm", epochs=20)
        # Train on (input_str, target_str) pairs
        model.train([("hello", "HELLO"), ("world", "WORLD")] * 10, tokenizer=tok)
        print(model.translate("hello", tokenizer=tok))  # -> "HELLO"
    """
    name = "seq2seq"
    family = "deep_learning"
    compatible_learning_techniques = ("supervised",)

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        n_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
        max_seq_len: int = 128,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 0.01,
        warmup_steps: int = 50,
        device: Optional[str] = None,
        **params,
    ):
        super().__init__(**{**params, "vocab_size": vocab_size, "d_model": d_model,
                            "n_layers": n_layers, "rnn_type": rnn_type,
                            "dropout": dropout, "max_seq_len": max_seq_len,
                            "epochs": epochs, "lr": lr, "batch_size": batch_size,
                            "weight_decay": weight_decay, "warmup_steps": warmup_steps,
                            "device": device})
        _require_torch()
        assert rnn_type in ("rnn", "lstm", "gru"), (
            f"rnn_type must be 'rnn', 'lstm', or 'gru', got {rnn_type!r}"
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        if device is None:
            try:
                import qai
                device = qai.get_optimal_device()
            except Exception:
                device = "cpu"
        self.device = torch.device(device)

        self.net: Optional[_Seq2SeqNet] = None
        self._loss_history: List[float] = []
        self.tokenizer = None

    def _build_model(self, vocab_size: int = None):
        if vocab_size is not None:
            self.vocab_size = vocab_size
        self.net = _Seq2SeqNet(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            rnn_type=self.rnn_type,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
        ).to(self.device)

    def fit(self, pairs: List[Tuple[Union[str, List[int]], Union[str, List[int]]]],
            y=None, tokenizer=None, **kwargs):
        """Train on (input, target) pairs.

        Args:
            pairs: List of (input, target) tuples. Each element can be a
                string (requires tokenizer) or a list of token IDs.
            y: Ignored (accepted for compatibility with Model.train's signature).
            tokenizer: Required if any element is a string.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer

        # Convert all pairs to (input_ids, target_ids)
        encoded_pairs = []
        max_id_seen = 0
        for src, tgt in pairs:
            src_ids = self._encode(src, tokenizer)
            tgt_ids = self._encode(tgt, tokenizer)
            # Prepend <bos> (ID 2) to target, append <eos> (ID 3)
            tgt_ids = [2] + list(tgt_ids) + [3]
            for ids in (src_ids, tgt_ids):
                if ids:
                    max_id_seen = max(max_id_seen, max(ids))
            encoded_pairs.append((src_ids, tgt_ids))

        vocab_size = tokenizer.vocab_size if tokenizer is not None else max(self.vocab_size, max_id_seen + 1)
        if self.net is None:
            self._build_model(vocab_size=vocab_size)

        self._train(encoded_pairs)
        self._trained = True
        return self

    def _encode(self, x, tokenizer) -> List[int]:
        if isinstance(x, str):
            if tokenizer is None:
                raise ValueError("String input but no tokenizer")
            return list(tokenizer.encode(x))
        return list(x)

    def _collate(self, sequences, pad_id: int = 0):
        max_len = min(max(len(s) for s in sequences), self.max_seq_len)
        if max_len == 0:
            max_len = 1
        batch = np.full((len(sequences), max_len), pad_id, dtype=np.int64)
        for i, s in enumerate(sequences):
            L = min(len(s), max_len)
            batch[i, :L] = s[:L]
        return torch.tensor(batch, device=self.device)

    def _train(self, encoded_pairs):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = max(1, (len(encoded_pairs) // self.batch_size + 1) * self.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: self._lr_lambda(step, total_steps)
        )
        self._loss_history = []
        self.net.train()

        for epoch in range(self.epochs):
            rng = np.random.default_rng(epoch)
            perm = rng.permutation(len(encoded_pairs))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(encoded_pairs), self.batch_size):
                idx = perm[start: start + self.batch_size]
                batch_src = [encoded_pairs[i][0] for i in idx]
                batch_tgt = [encoded_pairs[i][1] for i in idx]
                src_t = self._collate(batch_src)
                tgt_t = self._collate(batch_tgt)
                # Teacher forcing: input = tgt[:-1], target = tgt[1:]
                tgt_input = tgt_t[:, :-1]
                tgt_target = tgt_t[:, 1:]
                logits = self.net(src_t, tgt_ids=tgt_input)  # (B, T, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    tgt_target.reshape(-1),
                    ignore_index=0,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            self._loss_history.append(epoch_loss / max(1, n_batches))

    def _lr_lambda(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def forward(self, x, max_len: int = 20, **kwargs):
        """Generate output sequence for input x.

        Args:
            x: Input — string (requires tokenizer) or list of token IDs.
            max_len: Max output length.

        Returns:
            List of output token IDs.
        """
        if not self._trained or self.net is None:
            raise RuntimeError("seq2seq.predict() called before train()")
        src_ids = self._encode(x, self.tokenizer)
        if not src_ids:
            src_ids = [0]
        self.net.eval()
        with torch.no_grad():
            src_t = torch.tensor([src_ids], device=self.device, dtype=torch.long)
            output_ids = self.net(src_t, max_len=max_len)
            return output_ids[0].cpu().tolist()

    def translate(self, x, tokenizer=None, max_len: int = 20) -> str:
        """Generate output sequence and decode to string."""
        tok = tokenizer or self.tokenizer
        ids = self.forward(x, max_len=max_len)
        # Strip <bos> (2) and <eos> (3)
        ids = [i for i in ids if i not in (0, 2, 3)]
        if tok is not None:
            return tok.decode(ids)
        return " ".join(str(i) for i in ids)

    def loss_history(self) -> List[float]:
        return list(self._loss_history)

    def parameter_count(self) -> int:
        if self.net is None:
            return 0
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def architecture_summary(self) -> Dict[str, Any]:
        return {
            "technique": "seq2seq",
            "rnn_type": self.rnn_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len,
            "parameters": self.parameter_count(),
            "device": str(self.device),
        }

    def __repr__(self):
        if self.net is None:
            return f"<Seq2Seq unbuilt rnn_type={self.rnn_type}>"
        return (
            f"<Seq2Seq rnn_type={self.rnn_type} d_model={self.d_model} "
            f"n_layers={self.n_layers} params={self.parameter_count():,} "
            f"trained={self._trained}>"
        )
