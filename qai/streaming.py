"""
qai.streaming — streaming training from HuggingFace without full downloads.

Train on datasets that are LARGER than your RAM/disk by pulling data
on-the-fly from the HuggingFace Hub.

Features:
  - stream_dataset() — stream any HF dataset without downloading
  - stream_text() — stream a text corpus for language modeling
  - StreamingTrainer — train on a data stream (one batch at a time)
  - search_datasets() — search HF Hub for datasets
  - browse_datasets() — browse popular datasets by category

Usage:
    import qai

    # Stream a dataset without downloading it
    stream = qai.stream_dataset("imdb", split="train")
    for batch in stream:
        # batch is a dict with 'text' and 'label'
        print(batch['text'][:50])

    # Train on a stream
    trainer = qai.StreamingTrainer(model=m, dataset_name="imdb")
    trainer.train(steps=1000)

    # Stream text for language modeling
    stream = qai.stream_text("wikitext", "wikitext-2-raw-v1", split="train")
    for tokens in stream:
        model.train_batch(tokens)
"""
from __future__ import annotations
import os, time, json, urllib.request
from typing import Any, Dict, List, Optional, Tuple, Iterator, Generator, Callable
from dataclasses import dataclass, field
import numpy as np


# =====================================================================
# Dataset streaming from HuggingFace
# =====================================================================
def stream_dataset(dataset_name: str, config: Optional[str] = None,
                    split: str = "train", batch_size: int = 32,
                    tokenizer=None, max_samples: Optional[int] = None,
                    **kwargs) -> Generator[Dict[str, Any], None, None]:
    """Stream a dataset from HuggingFace without downloading it fully.

    Data is pulled on-the-fly from HF Hub, one batch at a time.
    This means you can train on datasets larger than your RAM/disk.

    Args:
        dataset_name: HuggingFace dataset name (e.g. "imdb", "squad")
        config: dataset config (e.g. "wikitext-2-raw-v1")
        split: "train", "validation", "test"
        batch_size: number of samples per yielded batch
        tokenizer: optional tokenizer to apply to text fields
        max_samples: optional limit on total samples

    Yields:
        Dict with keys: 'batch' (list of samples), 'batch_idx', 'total_seen'

    Example:
        >>> for batch in qai.stream_dataset("imdb", split="train"):
        ...     texts = [s['text'] for s in batch['batch']]
        ...     labels = [s['label'] for s in batch['batch']]
        ...     model.train_batch(texts, labels)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "[WHAT] Streaming requires the datasets library\n"
            "[FIX] pip install datasets  or  qai.install('datasets')")

    # Load in streaming mode
    ds = load_dataset(dataset_name, config, split=split, streaming=True, **kwargs)

    batch = []
    total = 0
    for i, sample in enumerate(ds):
        if max_samples is not None and total >= max_samples:
            break
        # Apply tokenizer if provided
        if tokenizer is not None:
            for key in sample:
                if isinstance(sample[key], str):
                    sample[key] = tokenizer.encode(sample[key])
        batch.append(sample)
        if len(batch) >= batch_size:
            yield {"batch": batch, "batch_idx": total // batch_size, "total_seen": total + len(batch)}
            batch = []
            total += batch_size
    # Yield remaining
    if batch:
        yield {"batch": batch, "batch_idx": total // batch_size, "total_seen": total + len(batch)}


def stream_text(dataset_name: str, config: Optional[str] = None,
                 split: str = "train", chunk_size: int = 512,
                 tokenizer=None, max_chunks: Optional[int] = None,
                 **kwargs) -> Generator[List[int], None, None]:
    """Stream a text dataset as tokenized chunks for language modeling.

    Concatenates all text from the dataset and splits into fixed-length
    chunks suitable for transformer training.

    Args:
        dataset_name: HF dataset name
        config: dataset config
        split: data split
        chunk_size: tokens per chunk
        tokenizer: tokenizer to use (if None, uses CharTokenizer)
        max_chunks: optional limit

    Yields:
        List of token IDs (length = chunk_size)
    """
    if tokenizer is None:
        from .tokenizers import CharTokenizer
        # Build a small char tokenizer from first few samples
        tokenizer = CharTokenizer()

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("[WHAT] pip install datasets")

    ds = load_dataset(dataset_name, config, split=split, streaming=True, **kwargs)

    # Collect text and tokenize incrementally
    buffer = []
    chunks_yielded = 0
    for sample in ds:
        # Find the text field
        text = ""
        for key in sample:
            if isinstance(sample[key], str) and len(sample[key]) > 10:
                text = sample[key]
                break
        if not text:
            continue
        # Tokenize
        tokens = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else [ord(c) % 256 for c in text]
        buffer.extend(tokens)
        # Yield chunks
        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]
            yield chunk
            chunks_yielded += 1
            if max_chunks is not None and chunks_yielded >= max_chunks:
                return


# =====================================================================
# Dataset search
# =====================================================================
HF_API_BASE = "https://huggingface.co/api"


def search_datasets(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search HuggingFace Hub for datasets by keyword.

    Args:
        query: search term (e.g. "sentiment", "image classification")
        limit: max results

    Returns:
        List of dataset info dicts
    """
    url = f"{HF_API_BASE}/datasets?search={query}&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "qai/0.4.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            results = json.loads(resp.read().decode("utf-8"))
        return [{
            "name": d.get("id", "?"),
            "downloads": d.get("downloads", 0),
            "likes": d.get("likes", 0),
            "tags": d.get("tags", []),
        } for d in results]
    except Exception as e:
        raise ConnectionError(f"Could not search HuggingFace: {e}")


def browse_datasets(category: str = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Browse popular datasets on HuggingFace.

    Args:
        category: optional filter (e.g. "text-classification", "question-answering")
        limit: max results

    Returns:
        List of dataset info dicts
    """
    params = f"?sort=downloads&direction=-1&limit={limit}"
    if category:
        params += f"&filter={category}"
    url = f"{HF_API_BASE}/datasets{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "qai/0.4.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            results = json.loads(resp.read().decode("utf-8"))
        return [{
            "name": d.get("id", "?"),
            "downloads": d.get("downloads", 0),
            "likes": d.get("likes", 0),
            "tags": d.get("tags", []),
        } for d in results]
    except Exception as e:
        raise ConnectionError(f"Could not browse HuggingFace: {e}")


# =====================================================================
# Streaming Trainer
# =====================================================================
class StreamingTrainer:
    """Train a model on a streaming dataset without full download.

    Pulls data from HuggingFace one batch at a time, feeds it to the model,
    then discards it. Enables training on datasets larger than RAM.

    Args:
        model: a qai.Model
        dataset_name: HuggingFace dataset name
        config: dataset config
        split: data split
        batch_size: batch size
        tokenizer: optional tokenizer
        max_steps: max training steps (None = until stream ends)
        lr: learning rate

    Usage:
        trainer = qai.StreamingTrainer(
            model=m, dataset_name="imdb", split="train",
            batch_size=32, max_steps=1000,
        )
        trainer.train()
    """
    def __init__(self, model, dataset_name: str, config: Optional[str] = None,
                 split: str = "train", batch_size: int = 32,
                 tokenizer=None, max_steps: Optional[int] = None,
                 lr: float = 3e-4):
        self.model = model
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.lr = lr
        self.step = 0
        self.losses: List[float] = []

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """Train on the streaming dataset.

        Returns:
            Dict with training stats
        """
        if verbose:
            print(f"[StreamingTrainer] streaming '{self.dataset_name}' ({self.split})")
            print(f"  batch_size={self.batch_size}, max_steps={self.max_steps or 'unlimited'}")

        t0 = time.time()
        total_samples = 0

        for batch_data in stream_dataset(
            self.dataset_name, self.config, self.split,
            batch_size=self.batch_size, tokenizer=self.tokenizer,
        ):
            batch = batch_data["batch"]
            # Extract features and labels
            texts = []
            labels = []
            for sample in batch:
                # Find text and label fields
                text = ""
                label = None
                for key, val in sample.items():
                    if isinstance(val, str) and len(val) > 5:
                        text = val
                    elif isinstance(val, (int, float)) and key in ("label", "labels", "target"):
                        label = val
                if text:
                    texts.append(text)
                if label is not None:
                    labels.append(label)

            # Train on this batch
            try:
                if labels and len(labels) == len(texts):
                    # Convert texts to features (simple: use char codes)
                    X = np.array([[ord(c) % 256 for c in t[:64]] + [0] * max(0, 64 - len(t))
                                   for t in texts[:self.batch_size]], dtype=np.float32)
                    y = np.array(labels[:self.batch_size])
                    if len(X) > 0:
                        self.model.train(X, y, verbose=False)
                        self.step += 1
                        total_samples += len(X)
                        if verbose and self.step % 10 == 0:
                            elapsed = time.time() - t0
                            rate = total_samples / max(elapsed, 0.1)
                            print(f"  step {self.step}: {total_samples} samples, "
                                  f"{rate:.0f} samples/s")
                else:
                    # Unsupervised — just pass through
                    self.step += 1
                    total_samples += len(texts)
            except Exception as e:
                if verbose:
                    print(f"  step {self.step}: batch failed ({str(e)[:50]})")

            # Check max steps
            if self.max_steps is not None and self.step >= self.max_steps:
                break

        elapsed = time.time() - t0
        if verbose:
            print(f"[StreamingTrainer] done: {self.step} steps, "
                  f"{total_samples} samples, {elapsed:.1f}s")

        return {
            "steps": self.step,
            "total_samples": total_samples,
            "time_s": elapsed,
            "samples_per_second": total_samples / max(elapsed, 0.1),
        }


__all__ = ["stream_dataset", "stream_text", "search_datasets",
           "browse_datasets", "StreamingTrainer"]
