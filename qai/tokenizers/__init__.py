"""
qai.tokenizers -- Text tokenizers for character-level, BPE, and word-level
tokenization.

This is the foundational module that unlocks sequence modeling (Transformers,
RNNs, LSTMs) and the spec's memory/context-window system (Section 53/95).

Three tokenizer classes, all sharing the same interface:

    tokenizer = qai.CharTokenizer()           # one char = one token
    tokenizer = qai.WordTokenizer()           # whitespace split
    tokenizer = qai.BPETokenizer(vocab_size=1000)  # byte-pair encoding

    ids  = tokenizer.encode("hello world")    # -> list[int]
    text = tokenizer.decode(ids)              # -> str

    tokenizer.fit(training_corpus)            # learn vocab (BPE/Word)
    tokenizer.save("tok.json")                # serialize
    tok2 = qai.CharTokenizer.load("tok.json") # reload

All tokenizers also support:
    - special tokens: <pad>, <unk>, <bos>, <eos>
    - batch encode / decode
    - vocabulary introspection (vocab_size, token_to_id, id_to_token)
    - padding & truncation to fixed length
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# Special token strings (kept consistent across all tokenizer types)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

# Special token IDs (reserved 0..3)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class BaseTokenizer:
    """Base class shared by Char / Word / BPE tokenizers.

    Subclasses must implement:
      - `_build_vocab(corpus)` — populate self.token_to_id / self.id_to_token
      - `_tokenize(text)` -> list of token strings (no special tokens)
      - `_detokenize(tokens)` -> str (inverse of _tokenize)

    The base class handles:
      - special-token bookkeeping
      - encode / decode with optional BOS/EOS wrapping
      - batch encode with padding & truncation
      - save / load to JSON
    """

    name: str = "base"

    def __init__(
        self,
        lowercase: bool = False,
        unk_token: str = UNK_TOKEN,
        pad_token: str = PAD_TOKEN,
        bos_token: str = BOS_TOKEN,
        eos_token: str = EOS_TOKEN,
    ):
        """
        Args:
            lowercase: If True, lowercase all input text before tokenizing.
            unk_token: Token string for out-of-vocabulary items.
            pad_token: Token string used for padding.
            bos_token: Beginning-of-sequence token.
            eos_token: End-of-sequence token.
        """
        self.lowercase = lowercase
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # API contract — subclasses must implement
    # ------------------------------------------------------------------
    def _build_vocab(self, corpus: List[str]) -> None:
        raise NotImplementedError

    def _tokenize(self, text: str) -> List[str]:
        """Split raw text into a list of token strings (no special tokens)."""
        raise NotImplementedError

    def _detokenize(self, tokens: List[str]) -> str:
        """Inverse of _tokenize: join tokens back into text."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, corpus: Union[str, List[str]]) -> "BaseTokenizer":
        """Learn the vocabulary from a corpus.

        Args:
            corpus: Either a single string or a list of strings. Each string
                is treated as one document. The tokenizer scans all documents
                and builds its vocabulary.

        Returns:
            self (for chaining).
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        # Reset vocab and seed with special tokens
        self.token_to_id = {}
        self.id_to_token = {}
        for tok in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            self._add_token(tok)
        self._build_vocab(corpus)
        self._fitted = True
        return self

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
    ) -> List[int]:
        """Encode a string to a list of token IDs.

        Args:
            text: Input string.
            add_bos: If True, prepend the BOS token ID.
            add_eos: If True, append the EOS token ID.
            max_length: If set with `padding=True`, pad/truncate to this length.
            padding: If True, pad with PAD_ID up to `max_length`.
            truncation: If True and len(ids) > max_length, truncate (keeping
                BOS at start and EOS at end if both are present).

        Returns:
            List of integer token IDs.
        """
        if not self._fitted:
            raise RuntimeError(
                f"[WHAT] {type(self).__name__}.encode() called before fit()\n"
                f"[WHY] The tokenizer has no vocabulary yet\n"
                f"[FIX] Call tokenizer.fit(corpus) first, or load a saved tokenizer"
            )
        if self.lowercase:
            text = text.lower()
        tokens = self._tokenize(text)
        ids = [self.token_to_id.get(t, self.token_to_id[self.unk_token]) for t in tokens]

        if add_bos:
            ids = [self.token_to_id[self.bos_token]] + ids
        if add_eos:
            ids = ids + [self.token_to_id[self.eos_token]]

        if max_length is not None:
            if truncation and len(ids) > max_length:
                # Preserve EOS at the end if it was added
                if add_eos:
                    ids = ids[:max_length - 1] + [self.token_to_id[self.eos_token]]
                else:
                    ids = ids[:max_length]
            if padding and len(ids) < max_length:
                ids = ids + [self.token_to_id[self.pad_token]] * (max_length - len(ids))

        return ids

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> List[List[int]]:
        """Encode a batch of strings. If `max_length` is None and `padding` is
        True, pads all sequences to the length of the longest in the batch."""
        if not isinstance(texts, (list, tuple)):
            raise TypeError("encode_batch expects a list of strings")
        all_ids = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos,
                        max_length=None, padding=False, truncation=truncation)
            for t in texts
        ]
        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in all_ids)
            all_ids = [
                ids[:max_length] + [self.token_to_id[self.pad_token]] * max(0, max_length - len(ids))
                for ids in all_ids
            ]
        return all_ids

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
    ) -> str:
        """Decode a list of token IDs back into a string.

        Args:
            ids: List of integer token IDs.
            skip_special: If True (default), drop PAD/UNK/BOS/EOS tokens from
                the output string. If False, render them as their string forms.

        Returns:
            Decoded string.
        """
        special_ids = {
            self.token_to_id.get(self.pad_token),
            self.token_to_id.get(self.unk_token),
            self.token_to_id.get(self.bos_token),
            self.token_to_id.get(self.eos_token),
        }
        tokens: List[str] = []
        for i in ids:
            tok = self.id_to_token.get(int(i), self.unk_token)
            if skip_special and i in special_ids:
                continue
            tokens.append(tok)
        return self._detokenize(tokens)

    def decode_batch(
        self,
        ids_batch: List[List[int]],
        skip_special: bool = True,
    ) -> List[str]:
        """Decode a batch of ID sequences."""
        return [self.decode(ids, skip_special=skip_special) for ids in ids_batch]

    # ------------------------------------------------------------------
    # Vocabulary introspection
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary (including special tokens)."""
        return len(self.token_to_id)

    def token_to_id_get(self, token: str) -> int:
        """Look up a token's ID. Returns UNK_ID if not in vocab."""
        return self.token_to_id.get(token, self.token_to_id[self.unk_token])

    def id_to_token_get(self, idx: int) -> str:
        """Look up an ID's token string. Returns unk_token if out of range."""
        return self.id_to_token.get(int(idx), self.unk_token)

    def get_vocab(self) -> Dict[str, int]:
        """Return a copy of the full vocab as {token: id}."""
        return dict(self.token_to_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """Save the tokenizer to a JSON file. Reload with `.load(path)`."""
        state = self._get_state()
        state["__tokenizer_type__"] = type(self).__name__
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "BaseTokenizer":
        """Load a tokenizer from a JSON file saved by `.save()`.

        Automatically returns the correct subclass (Char/Word/BPE) based on
        the `__tokenizer_type__` field in the file.
        """
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        cls_name = state.pop("__tokenizer_type__", None)
        # Find the right subclass
        if cls_name == "CharTokenizer":
            tok = CharTokenizer()
        elif cls_name == "WordTokenizer":
            tok = WordTokenizer()
        elif cls_name == "BPETokenizer":
            tok = BPETokenizer()
        else:
            tok = cls()
        tok._set_state(state)
        return tok

    def _get_state(self) -> Dict[str, Any]:
        """Subclasses override to add extra fields (e.g. BPE merges)."""
        return {
            "name": self.name,
            "lowercase": self.lowercase,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "_fitted": self._fitted,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.name = state.get("name", self.name)
        self.lowercase = state.get("lowercase", False)
        self.unk_token = state.get("unk_token", UNK_TOKEN)
        self.pad_token = state.get("pad_token", PAD_TOKEN)
        self.bos_token = state.get("bos_token", BOS_TOKEN)
        self.eos_token = state.get("eos_token", EOS_TOKEN)
        self.token_to_id = {k: int(v) for k, v in state.get("token_to_id", {}).items()}
        self.id_to_token = {int(k): v for k, v in state.get("id_to_token", {}).items()}
        self._fitted = state.get("_fitted", False)

    # ------------------------------------------------------------------
    # Helpers (used by subclasses)
    # ------------------------------------------------------------------
    def _add_token(self, token: str) -> int:
        """Add a token to the vocab if not present; return its ID."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def __repr__(self):
        fitted = "fitted" if self._fitted else "unfitted"
        return f"<{type(self).__name__} {fitted} vocab_size={self.vocab_size}>"


# =====================================================================
# Character-level tokenizer
# =====================================================================
class CharTokenizer(BaseTokenizer):
    """One character = one token. Simplest possible tokenizer.

    Useful for: small character-level language models, code prediction,
    learning the tokenizer interface, debugging.

    Pros: trivially learnable, no OOV (every char seen becomes a token).
    Cons: very long sequences for natural language (avg 5x longer than BPE).
    """

    name = "char"

    def __init__(self, lowercase: bool = False, **kwargs):
        super().__init__(lowercase=lowercase, **kwargs)

    def _build_vocab(self, corpus: List[str]) -> None:
        for doc in corpus:
            text = doc.lower() if self.lowercase else doc
            for ch in text:
                self._add_token(ch)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _detokenize(self, tokens: List[str]) -> str:
        return "".join(tokens)


# =====================================================================
# Word-level tokenizer
# =====================================================================
class WordTokenizer(BaseTokenizer):
    """Whitespace + punctuation splitter. One word (or punctuation mark) = one token.

    Useful for: simple NLP baselines, keyword extraction, search indexing.

    Pros: human-interpretable tokens, decent for short-text classification.
    Cons: large vocabularies on real corpora, every unseen word is <unk>.
    """

    name = "word"

    # Regex matches: words (incl. contractions), numbers, individual punctuation
    _pattern = re.compile(r"\w+(?:'\w+)*|[^\w\s]")

    def __init__(
        self,
        lowercase: bool = True,
        min_freq: int = 1,
        max_vocab: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            lowercase: Lowercase all text before tokenizing (default True).
            min_freq: Minimum token frequency to be included in vocab. Tokens
                appearing fewer times than this are mapped to <unk>. Default 1
                (keep all).
            max_vocab: If set, keep only the top-N most frequent tokens. The
                rest map to <unk>. Default None (keep all).
        """
        super().__init__(lowercase=lowercase, **kwargs)
        self.min_freq = min_freq
        self.max_vocab = max_vocab

    def _build_vocab(self, corpus: List[str]) -> None:
        # First pass: count all word frequencies
        word_counts: Counter = Counter()
        for doc in corpus:
            text = doc.lower() if self.lowercase else doc
            word_counts.update(self._pattern.findall(text))

        # Filter by min_freq
        kept = [(w, c) for w, c in word_counts.most_common() if c >= self.min_freq]
        # Cap vocab size if requested
        if self.max_vocab is not None:
            kept = kept[:self.max_vocab]

        for word, _ in kept:
            self._add_token(word)

    def _tokenize(self, text: str) -> List[str]:
        return self._pattern.findall(text)

    def _detokenize(self, tokens: List[str]) -> str:
        # Best-effort: join with spaces, fix spacing around punctuation
        out = " ".join(tokens)
        # Collapse "word ." -> "word."
        out = re.sub(r"\s+([.,;:!?'\"()\[\]{}])", r"\1", out)
        return out

    def _get_state(self) -> Dict[str, Any]:
        s = super()._get_state()
        s["min_freq"] = self.min_freq
        s["max_vocab"] = self.max_vocab
        return s

    def _set_state(self, state: Dict[str, Any]) -> None:
        super()._set_state(state)
        self.min_freq = state.get("min_freq", 1)
        self.max_vocab = state.get("max_vocab", None)


# =====================================================================
# Byte-Pair Encoding tokenizer
# =====================================================================
class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding tokenizer.

    Learns subword merges greedily: starts with characters, then iteratively
    merges the most-frequent adjacent pair until reaching `vocab_size` tokens
    or running out of merges.

    Useful for: real NLP work. BPE handles rare words (via subwords), keeps
    vocab size bounded, and is the standard for modern Transformer models
    (GPT-2, GPT-NeoX, Llama all use BPE variants).

    Pros: bounded vocab, no true OOV (unseen words decompose into subwords),
        good for morphologically-rich languages.
    Cons: requires a fit pass over a corpus (slower to set up than char/word).
    """

    name = "bpe"

    def __init__(
        self,
        vocab_size: int = 1000,
        lowercase: bool = False,
        end_of_word: str = "</w>",
        min_frequency: int = 2,
        **kwargs,
    ):
        """
        Args:
            vocab_size: Target vocabulary size (including the 4 special tokens).
                The tokenizer will perform at most `vocab_size - 4 - n_chars`
                merges. Default 1000.
            lowercase: Lowercase all input before processing. Default False.
            end_of_word: Marker appended to the last character of each word,
                so the tokenizer can recover word boundaries. Default "</w>".
                Set to empty string to disable.
            min_frequency: A pair must occur at least this many times to be
                considered for merging. Default 2.
        """
        super().__init__(lowercase=lowercase, **kwargs)
        self.target_vocab_size = vocab_size
        self.end_of_word = end_of_word
        self.min_frequency = min_frequency
        self.merges: List[Tuple[Tuple[str, str], str]] = []
        # Cache for BPE tokenization of words (re-computed when merges change)
        self._word_cache: Dict[str, List[str]] = {}

    def _build_vocab(self, corpus: List[str]) -> None:
        # Step 1: split corpus into words, count frequencies
        word_freq: Counter = Counter()
        for doc in corpus:
            text = doc.lower() if self.lowercase else doc
            # Use the same regex as WordTokenizer for word splitting
            for w in re.findall(r"\w+(?:'\w+)*|[^\w\s]", text):
                word_freq[w] += 1

        # Step 2: represent each word as a tuple of chars + end-of-word marker
        eow = self.end_of_word
        # Each word is a tuple of symbols, e.g. ("h", "e", "l", "l", "o", "</w>")
        word_symbols: Dict[Tuple[str, ...], int] = {}
        for word, freq in word_freq.items():
            if eow:
                symbols = tuple(list(word) + [eow])
            else:
                symbols = tuple(list(word))
            word_symbols[symbols] = word_symbols.get(symbols, 0) + freq

        # Add all single-character tokens to the vocab
        all_chars = set()
        for symbols in word_symbols:
            for s in symbols:
                all_chars.add(s)
        for ch in sorted(all_chars):
            self._add_token(ch)

        # Step 3: greedy BPE merge loop
        self.merges = []
        max_merges = max(0, self.target_vocab_size - self.vocab_size)
        for _ in range(max_merges):
            # Count adjacent pairs across all words
            pair_counts: Counter = Counter()
            for symbols, freq in word_symbols.items():
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < self.min_frequency:
                break

            # Merge the pair in every word
            new_symbol = best_pair[0] + best_pair[1]
            new_word_symbols: Dict[Tuple[str, ...], int] = {}
            for symbols, freq in word_symbols.items():
                new_symbols = self._merge_pair(symbols, best_pair, new_symbol)
                new_word_symbols[new_symbols] = new_word_symbols.get(new_symbols, 0) + freq
            word_symbols = new_word_symbols

            # Record the merge and add the new token to vocab
            self.merges.append((best_pair, new_symbol))
            self._add_token(new_symbol)

        # Invalidate the per-word tokenization cache (merges changed)
        self._word_cache = {}

    @staticmethod
    def _merge_pair(
        symbols: Tuple[str, ...],
        pair: Tuple[str, str],
        new_symbol: str,
    ) -> Tuple[str, ...]:
        """Replace all occurrences of `pair` in `symbols` with `new_symbol`."""
        result: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                result.append(new_symbol)
                i += 2
            else:
                result.append(symbols[i])
                i += 1
        return tuple(result)

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply BPE merges to a single word. Cached per-word for speed."""
        if word in self._word_cache:
            return list(self._word_cache[word])

        eow = self.end_of_word
        if eow:
            symbols = list(word) + [eow]
        else:
            symbols = list(word)

        # Apply merges in order (each merge is a (pair, new_symbol) tuple)
        for (pair, new_symbol) in self.merges:
            i = 0
            new_symbols: List[str] = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    new_symbols.append(new_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        self._word_cache[word] = tuple(symbols)
        return symbols

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for w in re.findall(r"\w+(?:'\w+)*|[^\w\s]", text):
            tokens.extend(self._tokenize_word(w))
        return tokens

    def _detokenize(self, tokens: List[str]) -> str:
        eow = self.end_of_word
        out: List[str] = []
        for t in tokens:
            if eow and t.endswith(eow):
                out.append(t[: -len(eow)])
                out.append(" ")
            else:
                # Continuation subword — append without space
                if out and out[-1] != " ":
                    out.append(t)
                else:
                    out.append(t)
        text = "".join(out)
        # Collapse double spaces from consecutive eow markers
        text = re.sub(r"  +", " ", text).strip()
        return text

    def _get_state(self) -> Dict[str, Any]:
        s = super()._get_state()
        s["target_vocab_size"] = self.target_vocab_size
        s["end_of_word"] = self.end_of_word
        s["min_frequency"] = self.min_frequency
        # Serialize merges as [[a, b, merged], ...]
        s["merges"] = [[a, b, m] for (a, b), m in self.merges]
        return s

    def _set_state(self, state: Dict[str, Any]) -> None:
        super()._set_state(state)
        self.target_vocab_size = state.get("target_vocab_size", 1000)
        self.end_of_word = state.get("end_of_word", "</w>")
        self.min_frequency = state.get("min_frequency", 2)
        self.merges = [((a, b), m) for a, b, m in state.get("merges", [])]
        self._word_cache = {}


# =====================================================================
# Module-level convenience functions
# =====================================================================
def load_tokenizer(path: str) -> BaseTokenizer:
    """Load any tokenizer saved via `tokenizer.save(path)`.

    This is the same as `BaseTokenizer.load(path)` but exposed at the
    module level for convenience: `qai.tokenizers.load_tokenizer(path)`.
    """
    return BaseTokenizer.load(path)


# Short alias
load = load_tokenizer


__all__ = [
    "BaseTokenizer",
    "CharTokenizer",
    "WordTokenizer",
    "BPETokenizer",
    "load_tokenizer",
    "load",
    "PAD_TOKEN", "UNK_TOKEN", "BOS_TOKEN", "EOS_TOKEN",
    "PAD_ID", "UNK_ID", "BOS_ID", "EOS_ID",
    "SPECIAL_TOKENS",
]
