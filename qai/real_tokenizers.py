"""
qai.real_tokenizers — production-grade tokenizer wrappers.

Wraps the industry-standard tokenizers:
  - TiktokenWrapper — GPT-4/ChatGPT's BPE tokenizer (fast, via tiktoken)
  - SentencePieceWrapper — multilingual, used by LLaMA/T5 (via sentencepiece)
  - HuggingFaceTokenizer — any HF tokenizer (via transformers/tokenizers)
  - AutoTokenizer — auto-detect best available tokenizer

These are the tokenizers used by production LLMs. They're far more
sophisticated than qai's built-in BPE/WordPiece/Char tokenizers.

Usage:
    import qai

    # GPT-4 tokenizer
    tok = qai.TiktokenWrapper(model="gpt-4")
    ids = tok.encode("Hello, world!")
    text = tok.decode(ids)

    # SentencePiece (LLaMA-style)
    tok = qai.SentencePieceWrapper(model_path="llama.model")
    ids = tok.encode("Hello, world!")

    # HuggingFace auto tokenizer
    tok = qai.HuggingFaceTokenizer.from_pretrained("bert-base-uncased")
    ids = tok.encode("Hello, world!")
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# =====================================================================
# Base class
# =====================================================================
class RealTokenizer:
    """Base class for production tokenizers."""
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError
    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]
    def batch_decode(self, batch: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch]


# =====================================================================
# Tiktoken (GPT-4/ChatGPT BPE)
# =====================================================================
class TiktokenWrapper(RealTokenizer):
    """GPT-4/ChatGPT's BPE tokenizer via tiktoken.

    Fastest BPE implementation in Python. Used by OpenAI's GPT models.

    Requires: pip install tiktoken

    Args:
        model: "gpt-4", "gpt-3.5-turbo", "gpt-4o", or encoding name
                ("cl100k_base", "p50k_base", "r50k_base")
    """
    def __init__(self, model: str = "gpt-4"):
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "[WHAT] TiktokenWrapper requires tiktoken\n"
                "[FIX] pip install tiktoken")
        # Map model names to encodings
        model_to_encoding = {
            "gpt-4": "cl100k_base",
            "gpt-4o": "o200k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-2": "r50k_base",
            "text-davinci-003": "p50k_base",
        }
        encoding_name = model_to_encoding.get(model, model)
        self._enc = tiktoken.get_encoding(encoding_name)
        self._model = model

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab


# =====================================================================
# SentencePiece (LLaMA, T5, multilingual)
# =====================================================================
class SentencePieceWrapper(RealTokenizer):
    """SentencePiece tokenizer (used by LLaMA, T5, mT5).

    Supports multilingual text, BPE or Unigram subword segmentation.
    Can train from scratch or load pre-trained models.

    Requires: pip install sentencepiece

    Args:
        model_path: path to .model file (for loading pre-trained)
        OR
        train: if True, train from scratch (see train())
    """
    def __init__(self, model_path: Optional[str] = None,
                 sp_processor=None):
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "[WHAT] SentencePieceWrapper requires sentencepiece\n"
                "[FIX] pip install sentencepiece")
        self._spm = spm
        if sp_processor is not None:
            self._sp = sp_processor
        elif model_path is not None:
            self._sp = spm.SentencePieceProcessor()
            self._sp.load(model_path)
        else:
            self._sp = None  # needs training

    @classmethod
    def train(cls, texts: List[str], vocab_size: int = 32000,
              model_type: str = "bpe", model_path: str = "/tmp/sp.model",
              **kwargs) -> "SentencePieceWrapper":
        """Train a SentencePiece model from texts."""
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("pip install sentencepiece")
        # Write texts to a temp file
        corpus_path = model_path + ".corpus"
        with open(corpus_path, "w") as f:
            f.write("\n".join(texts))
        spm.SentencePieceTrainer.train(
            input=corpus_path, model_prefix=model_path.replace(".model", ""),
            vocab_size=vocab_size, model_type=model_type, **kwargs)
        os.remove(corpus_path)
        return cls(model_path=model_path)

    def encode(self, text: str) -> List[int]:
        if self._sp is None:
            raise RuntimeError("Tokenizer not trained — call train() first")
        return self._sp.encode(text)

    def decode(self, ids: List[int]) -> str:
        if self._sp is None:
            raise RuntimeError("Tokenizer not trained")
        return self._sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        if self._sp is None:
            return 0
        return self._sp.get_piece_size()


# =====================================================================
# HuggingFace Tokenizer (any HF model)
# =====================================================================
class HuggingFaceTokenizer(RealTokenizer):
    """HuggingFace tokenizer wrapper.

    Can load ANY tokenizer from the HuggingFace Hub.

    Requires: pip install transformers

    Args:
        model_name: HuggingFace model name (e.g. "bert-base-uncased",
                    "gpt2", "meta-llama/Llama-2-7b", "t5-base")
    """
    def __init__(self, model_name: str = "bert-base-uncased"):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "[WHAT] HuggingFaceTokenizer requires transformers\n"
                "[FIX] pip install transformers")
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str) -> "HuggingFaceTokenizer":
        return cls(model_name=model_name)

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=True)

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def batch_encode(self, texts: List[str], padding=True, truncation=True,
                      max_length=512) -> Dict[str, Any]:
        """Batch encode with padding and truncation."""
        result = self._tok(texts, padding=padding, truncation=truncation,
                            max_length=max_length, return_tensors="pt")
        return {"input_ids": result["input_ids"].tolist(),
                "attention_mask": result["attention_mask"].tolist()}

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    @property
    def special_tokens(self) -> Dict[str, int]:
        return {
            "pad": self._tok.pad_token_id,
            "bos": self._tok.bos_token_id,
            "eos": self._tok.eos_token_id,
        }


# =====================================================================
# Auto-detector: pick the best available tokenizer
# =====================================================================
def AutoTokenizer(name: str = "gpt-4") -> RealTokenizer:
    """Auto-detect the best available tokenizer for a model name.

    Tries (in order):
      1. HuggingFace (if transformers installed)
      2. Tiktoken (if tiktoken installed)
      3. SentencePiece (if sentencepiece installed)
      4. qai built-in BPE (fallback)
    """
    # Try HuggingFace
    try:
        return HuggingFaceTokenizer(name)
    except (ImportError, Exception):
        pass
    # Try Tiktoken
    try:
        return TiktokenWrapper(name)
    except (ImportError, Exception):
        pass
    # Try SentencePiece
    try:
        if os.path.exists(name):
            return SentencePieceWrapper(model_path=name)
    except (ImportError, Exception):
        pass
    # Fallback: qai built-in
    from .tokenizers import BPETokenizer
    tok = BPETokenizer()
    tok.train(["hello world foo bar baz"])  # minimal vocab
    return tok


__all__ = ["RealTokenizer", "TiktokenWrapper", "SentencePieceWrapper",
           "HuggingFaceTokenizer", "AutoTokenizer"]
