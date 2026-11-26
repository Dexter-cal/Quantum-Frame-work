"""
qai.memory -- Three-tier memory system (spec Section 41/95).

Real production AI systems need three distinct kinds of "memory":

    model.memory.weights     # baked-in, learned via training
                           # slow/permanent, changes only via finetune
    model.memory.working    # current context window
                           # fast/temporary, per-session
    model.memory.retrieval  # external store (vector-DB style)
                           # fast to update, no retraining required

This module implements all three tiers plus several creative additions:

    Memory              -- top-level coordinator holding all three tiers
    WeightsMemory       -- snapshot of a trained model's weights, with diff
    WorkingMemory       -- sliding-window context buffer with forgetting policies
    RetrievalMemory     -- vector store with cosine-similarity search + save/load
    Embedding functions -- hash, random, TF-IDF
    Forgetting policies -- LRU, LFU, time-decay, importance-threshold
    Memory snapshots    -- save/load entire memory state to JSON
    Conflict detection  -- flag contradictions in retrieval memory
    Audit log           -- who added what when

Example:
    import qai
    from qai.memory import Memory, WorkingMemory, RetrievalMemory

    mem = Memory()
    mem.working.add("user said: hello")
    mem.working.add("assistant said: hi there")
    print(mem.working.recent(5))  # last 5 items

    retriever = RetrievalMemory(embed_dim=64)
    retriever.add("Paris is the capital of France", metadata={"source": "geo"})
    retriever.add("France is in Europe", metadata={"source": "geo"})
    results = retriever.search("capital of France", k=2)
    for r in results:
        print(r.score, r.text)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# =====================================================================
# Embedding functions
# =====================================================================
def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def hash_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Deterministic hash-based embedding.

    Each character contributes to a bucket via md5 hashing. Cheap, fast,
    deterministic — good for testing and small-scale retrieval.

    Args:
        text: Input string.
        dim: Embedding dimension.

    Returns:
        L2-normalized float vector of shape (dim,).
    """
    v = np.zeros(dim, dtype=np.float32)
    # Hash each n-gram (chars and pairs) into the vector
    for n in (1, 2, 3):
        for i in range(len(text) - n + 1):
            gram = text[i : i + n]
            h = hashlib.md5(gram.encode("utf-8")).digest()
            # Use first 4 bytes as an int, mod dim
            bucket = int.from_bytes(h[:4], "big") % dim
            sign = 1.0 if (h[4] & 1) else -1.0
            v[bucket] += sign
    return _normalize(v)


def random_embedding(text: str, dim: int = 128, seed: int = 0) -> np.ndarray:
    """Seeded random embedding — same text → same vector.

    Useful as a baseline / sanity check. Each unique text gets a reproducible
    random vector.
    """
    h = hashlib.md5(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big") ^ seed)
    return _normalize(rng.standard_normal(dim).astype(np.float32))


class TFIDFEmbedder:
    """TF-IDF embedding with on-the-fly vocabulary learning.

    Each call to `embed(text)` updates the vocabulary and IDF statistics, then
    returns the TF-IDF vector for that text. The vector is sparse but returned
    as a dense numpy array (good enough for small-to-medium corpora).

    For real production use, swap this for a sentence-transformer or OpenAI
    embedding — the RetrievalMemory tier accepts any callable.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.vocab: Dict[str, int] = {}  # token -> index
        self.doc_freq: Counter = Counter()  # token -> num docs containing it
        self.n_docs = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def __call__(self, text: str) -> np.ndarray:
        """Make the embedder callable, so it can be used as embed_fn directly."""
        return self.embed(text)

    def embed(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)

        # Update vocab and doc freq
        self.n_docs += 1
        unique_tokens = set(tokens)
        for tok in unique_tokens:
            if tok not in self.vocab:
                if len(self.vocab) >= self.dim:
                    # Reuse a bucket (hash collision) — keeps the vector dim-capped
                    self.vocab[tok] = (self.vocab[tok] if tok in self.vocab
                                        else hash(tok) % self.dim)
                else:
                    self.vocab[tok] = len(self.vocab)
            self.doc_freq[tok] += 1

        # Compute TF-IDF
        v = np.zeros(self.dim, dtype=np.float32)
        tf = Counter(tokens)
        for tok, count in tf.items():
            idx = self.vocab[tok]
            idf = math.log((1 + self.n_docs) / (1 + self.doc_freq[tok])) + 1
            v[idx] = count * idf
        return _normalize(v)

    def save_state(self) -> dict:
        return {
            "dim": self.dim,
            "vocab": self.vocab,
            "doc_freq": dict(self.doc_freq),
            "n_docs": self.n_docs,
        }

    def load_state(self, state: dict) -> None:
        self.dim = state["dim"]
        self.vocab = state["vocab"]
        self.doc_freq = Counter(state["doc_freq"])
        self.n_docs = state["n_docs"]


# =====================================================================
# Working memory — sliding window with forgetting
# =====================================================================
@dataclass
class WorkingItem:
    """A single item in working memory."""
    text: str
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    importance: float = 0.0  # user-set or computed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self):
        self.access_count += 1
        self.last_accessed = time.time()


class WorkingMemory:
    """Sliding-window context buffer with forgetting policies.

    Holds the most recent N items (default 50). When full, applies the
    configured forgetting policy to decide which items to evict.

    Policies (set via `policy=`):
        - "lru"         : evict least-recently-used (default)
        - "lfu"         : evict least-frequently-used
        - "time_decay"  : evict items with oldest timestamp (FIFO-like)
        - "importance"  : evict items with lowest importance score
        - "fifo"        : evict oldest added (queue order)

    Custom policies: pass a callable `eviction_fn(items, n_to_evict) -> list[idx]`
    that returns indices to evict.
    """
    POLICIES = ("lru", "lfu", "time_decay", "importance", "fifo")

    def __init__(
        self,
        capacity: int = 50,
        policy: str = "lru",
        eviction_fn: Optional[Callable] = None,
        auto_promote_threshold: int = 0,
        auto_promote_callback: Optional[Callable[[str, Dict], None]] = None,
    ):
        """
        Args:
            capacity: Max items in working memory (default 50).
            policy: Forgetting policy. One of "lru", "lfu", "time_decay",
                "importance", "fifo".
            eviction_fn: Custom eviction function. Overrides `policy` if set.
                Signature: fn(items: list[WorkingItem], n_to_evict: int) -> list[int]
            auto_promote_threshold: If > 0, items accessed this many times are
                auto-promoted via the callback. 0 = no auto-promotion.
            auto_promote_callback: Called when an item is auto-promoted.
                Signature: fn(text: str, metadata: dict) -> None
        """
        if policy not in self.POLICIES:
            raise ValueError(
                f"[WHAT] Unknown working-memory policy '{policy}'\n"
                f"[WHY] Policy must be one of {self.POLICIES}\n"
                f"[FIX] Use one of the supported policies, or pass eviction_fn= for custom"
            )
        self.capacity = capacity
        self.policy = policy
        self.eviction_fn = eviction_fn
        self.auto_promote_threshold = auto_promote_threshold
        self.auto_promote_callback = auto_promote_callback
        self._items: List[WorkingItem] = []
        self._audit_log: List[Dict[str, Any]] = []

    def add(self, text: str, importance: float = 0.0, metadata: Optional[Dict] = None) -> None:
        """Add an item to working memory.

        Args:
            text: The memory content.
            importance: User-set importance score (used by "importance" policy).
            metadata: Optional metadata dict (e.g. source, role, tags).
        """
        now = time.time()
        item = WorkingItem(
            text=text,
            timestamp=now,
            last_accessed=now,
            importance=importance,
            metadata=dict(metadata or {}),
        )
        self._items.append(item)
        self._audit_log.append({
            "action": "add", "text": text[:80], "timestamp": now,
            "importance": importance, "metadata": dict(metadata or {}),
        })

        # Evict if over capacity
        if len(self._items) > self.capacity:
            n_to_evict = len(self._items) - self.capacity
            evicted_idx = self._evict(n_to_evict)
            for i in sorted(evicted_idx, reverse=True):
                evicted = self._items.pop(i)
                self._audit_log.append({
                    "action": "evict", "text": evicted.text[:80], "timestamp": time.time(),
                    "policy": self.policy,
                })

    def _evict(self, n_to_evict: int) -> List[int]:
        """Return indices of items to evict."""
        if self.eviction_fn is not None:
            return self.eviction_fn(self._items, n_to_evict)

        if self.policy == "lru":
            # Least recently accessed
            scored = sorted(range(len(self._items)),
                            key=lambda i: self._items[i].last_accessed)
            return scored[:n_to_evict]
        elif self.policy == "lfu":
            # Least frequently accessed
            scored = sorted(range(len(self._items)),
                            key=lambda i: self._items[i].access_count)
            return scored[:n_to_evict]
        elif self.policy == "time_decay":
            # Oldest timestamp
            scored = sorted(range(len(self._items)),
                            key=lambda i: self._items[i].timestamp)
            return scored[:n_to_evict]
        elif self.policy == "importance":
            # Lowest importance
            scored = sorted(range(len(self._items)),
                            key=lambda i: self._items[i].importance)
            return scored[:n_to_evict]
        elif self.policy == "fifo":
            # First-in
            return list(range(n_to_evict))
        return list(range(n_to_evict))

    def get(self, text: str) -> Optional[WorkingItem]:
        """Find an item by exact text match. Returns None if not found."""
        for item in self._items:
            if item.text == text:
                item.touch()
                self._check_auto_promote(item)
                return item
        return None

    def recent(self, n: int = 10) -> List[str]:
        """Return the N most recently-added items as strings."""
        return [item.text for item in self._items[-n:]]

    def all_items(self) -> List[WorkingItem]:
        """Return all items (most-recent last)."""
        return list(self._items)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Naive substring search. Returns (text, score) pairs.

        Score is 1.0 for exact match, 0.5 for substring match. For real
        semantic search, use RetrievalMemory instead.
        """
        results: List[Tuple[str, float]] = []
        q_lower = query.lower()
        for item in self._items:
            if item.text == query:
                results.append((item.text, 1.0))
                item.touch()
            elif q_lower in item.text.lower():
                results.append((item.text, 0.5))
                item.touch()
        # Sort by score desc, take top k
        results.sort(key=lambda x: -x[1])
        return results[:k]

    def clear(self) -> None:
        """Clear all items."""
        n = len(self._items)
        self._items.clear()
        self._audit_log.append({
            "action": "clear", "timestamp": time.time(), "n_items": n,
        })

    def _check_auto_promote(self, item: WorkingItem) -> None:
        """If item's access count crosses the threshold, promote via callback."""
        if (self.auto_promote_threshold > 0
                and item.access_count >= self.auto_promote_threshold
                and self.auto_promote_callback is not None
                and not item.metadata.get("_promoted", False)):
            self.auto_promote_callback(item.text, item.metadata)
            item.metadata["_promoted"] = True
            self._audit_log.append({
                "action": "auto_promote", "text": item.text[:80],
                "timestamp": time.time(), "access_count": item.access_count,
            })

    def summary(self) -> Dict[str, Any]:
        """Return summary stats about the working memory."""
        return {
            "tier": "working",
            "capacity": self.capacity,
            "size": len(self._items),
            "policy": self.policy,
            "total_accesses": sum(it.access_count for it in self._items),
            "avg_importance": (sum(it.importance for it in self._items) / len(self._items)
                                if self._items else 0.0),
        }

    def audit_log(self, last_n: int = 50) -> List[Dict]:
        """Return the last N audit-log entries."""
        return list(self._audit_log[-last_n:])

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return f"<WorkingMemory size={len(self._items)}/{self.capacity} policy={self.policy!r}>"


# =====================================================================
# Retrieval memory — vector store with cosine similarity
# =====================================================================
@dataclass
class RetrievalResult:
    """A single search result from RetrievalMemory.search()."""
    text: str
    score: float
    metadata: Dict[str, Any]
    id: int


class _DefaultEmbedder:
    """Picklable default embedder for RetrievalMemory.

    Replaces a lambda (which is unpicklable) so RetrievalMemory instances
    can be safely pickled -- which is essential because Memory holds a
    RetrievalMemory and we want Memory to be picklable for bundling.
    """
    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim

    def __call__(self, text: str) -> np.ndarray:
        return hash_embedding(text, dim=self.embed_dim)

    def __eq__(self, other):
        return isinstance(other, _DefaultEmbedder) and self.embed_dim == other.embed_dim

    def __repr__(self):
        return f"_DefaultEmbedder(embed_dim={self.embed_dim})"


class RetrievalMemory:
    """Vector store for long-term retrieval (RAG-style).

    Stores text + metadata + embedding vectors. Search via cosine similarity.

    The embedding function is pluggable — default is `hash_embedding(dim=128)`
    which is cheap and deterministic. For real semantic search, pass a
    sentence-transformer or OpenAI embedding callable.
    """
    def __init__(
        self,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        embed_dim: int = 128,
        conflict_threshold: float = 0.95,
    ):
        """
        Args:
            embed_fn: Callable that maps text -> np.ndarray (L2-normalized).
                Default: hash_embedding with embed_dim.
            embed_dim: Embedding dimension (only used if embed_fn is None).
            conflict_threshold: If a new item's max similarity to existing
                items exceeds this, it's flagged as a potential duplicate
                (or contradiction if metadata differs). 0.0 = disabled.
        """
        if embed_fn is None:
            self.embed_fn = _DefaultEmbedder(embed_dim=embed_dim)
        else:
            self.embed_fn = embed_fn
        self.embed_dim = embed_dim
        self.conflict_threshold = conflict_threshold

        # Parallel lists: ids, texts, metadatas, vectors
        self._ids: List[int] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._vectors: List[np.ndarray] = []
        self._next_id = 0
        self._audit_log: List[Dict[str, Any]] = []
        self._embedder_state: Optional[dict] = None  # for TFIDFEmbedder

        # If embed_fn is a TFIDFEmbedder, capture its state on save
        if isinstance(embed_fn, TFIDFEmbedder):
            self._embedder_is_tfidf = True
        else:
            self._embedder_is_tfidf = False

    def add(self, text: str, metadata: Optional[Dict] = None) -> Tuple[int, List[Dict]]:
        """Add an item to the retrieval memory.

        Args:
            text: The text to store.
            metadata: Optional metadata (source, tags, etc.).

        Returns:
            (item_id, conflicts) where `conflicts` is a list of dicts describing
            existing items with similarity > conflict_threshold.
        """
        vec = self.embed_fn(text)
        new_id = self._next_id
        self._next_id += 1
        self._ids.append(new_id)
        self._texts.append(text)
        self._metadatas.append(dict(metadata or {}))
        self._vectors.append(vec)

        # Detect conflicts (high-similarity existing items)
        conflicts: List[Dict] = []
        if self.conflict_threshold > 0 and len(self._vectors) > 1:
            sims = self._cosine_all(vec)
            for i, sim in enumerate(sims[:-1]):  # exclude last (the new one)
                if sim >= self.conflict_threshold:
                    is_contradiction = (
                        self._metadatas[i].get("source") != self._metadatas[-1].get("source")
                        or self._metadatas[i].get("fact") != self._metadatas[-1].get("fact")
                    )
                    conflicts.append({
                        "existing_id": self._ids[i],
                        "existing_text": self._texts[i],
                        "similarity": float(sim),
                        "type": "contradiction" if is_contradiction else "duplicate",
                    })

        self._audit_log.append({
            "action": "add", "id": new_id, "text": text[:80],
            "timestamp": time.time(), "n_conflicts": len(conflicts),
        })
        return new_id, conflicts

    def _cosine_all(self, vec: np.ndarray) -> np.ndarray:
        """Cosine similarity of `vec` against all stored vectors. Assumes L2-normalized."""
        if not self._vectors:
            return np.zeros(0, dtype=np.float32)
        M = np.stack(self._vectors)  # (N, dim)
        return M @ vec  # both normalized, so dot product = cosine

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> List[RetrievalResult]:
        """Search for items similar to `query`.

        Args:
            query: Search query text.
            k: Max number of results.
            filter_fn: Optional metadata filter. Called as fn(metadata) -> bool.
                Only items where filter_fn returns True are considered.

        Returns:
            List of RetrievalResult (text, score, metadata, id) sorted by score desc.
        """
        if not self._texts:
            return []
        q_vec = self.embed_fn(query)
        sims = self._cosine_all(q_vec)

        # Build (idx, sim) pairs, optionally filtered
        pairs = []
        for i, sim in enumerate(sims):
            if filter_fn is None or filter_fn(self._metadatas[i]):
                pairs.append((i, float(sim)))

        # Sort by sim desc, take top k
        pairs.sort(key=lambda x: -x[1])
        pairs = pairs[:k]

        return [
            RetrievalResult(
                text=self._texts[i],
                score=sim,
                metadata=dict(self._metadatas[i]),
                id=self._ids[i],
            )
            for i, sim in pairs
        ]

    def get(self, item_id: int) -> Optional[RetrievalResult]:
        """Retrieve a specific item by ID."""
        if item_id not in self._ids:
            return None
        idx = self._ids.index(item_id)
        return RetrievalResult(
            text=self._texts[idx],
            score=1.0,
            metadata=dict(self._metadatas[idx]),
            id=item_id,
        )

    def remove(self, item_id: int) -> bool:
        """Remove an item by ID. Returns True if removed, False if not found."""
        if item_id not in self._ids:
            return False
        idx = self._ids.index(item_id)
        self._ids.pop(idx)
        self._texts.pop(idx)
        self._metadatas.pop(idx)
        self._vectors.pop(idx)
        self._audit_log.append({
            "action": "remove", "id": item_id, "timestamp": time.time(),
        })
        return True

    def update_metadata(self, item_id: int, metadata: Dict) -> bool:
        """Update an item's metadata (merges with existing). Returns True if found."""
        if item_id not in self._ids:
            return False
        idx = self._ids.index(item_id)
        self._metadatas[idx].update(metadata)
        return True

    def clear(self) -> None:
        """Remove all items."""
        n = len(self._ids)
        self._ids.clear()
        self._texts.clear()
        self._metadatas.clear()
        self._vectors.clear()
        self._audit_log.append({
            "action": "clear", "timestamp": time.time(), "n_items": n,
        })

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """Save to JSON. Note: vectors are stored as lists, so files can get
        large for big retrieval memories."""
        # If using TFIDFEmbedder, capture its state
        embedder_state = None
        if self._embedder_is_tfidf:
            embedder_state = self.embed_fn.save_state()

        state = {
            "embed_dim": self.embed_dim,
            "conflict_threshold": self.conflict_threshold,
            "embedder_is_tfidf": self._embedder_is_tfidf,
            "embedder_state": embedder_state,
            "ids": self._ids,
            "texts": self._texts,
            "metadatas": self._metadatas,
            "vectors": [v.tolist() for v in self._vectors],
            "next_id": self._next_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "RetrievalMemory":
        """Load from JSON saved by .save()."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        # Reconstruct embed_fn
        if state.get("embedder_is_tfidf") and state.get("embedder_state"):
            emb = TFIDFEmbedder(dim=state["embed_dim"])
            emb.load_state(state["embedder_state"])
            mem = cls(embed_fn=emb, embed_dim=state["embed_dim"],
                     conflict_threshold=state["conflict_threshold"])
        else:
            mem = cls(embed_dim=state["embed_dim"],
                     conflict_threshold=state["conflict_threshold"])

        mem._ids = list(state["ids"])
        mem._texts = list(state["texts"])
        mem._metadatas = [dict(m) for m in state["metadatas"]]
        mem._vectors = [np.asarray(v, dtype=np.float32) for v in state["vectors"]]
        mem._next_id = state["next_id"]
        return mem

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._ids)

    def summary(self) -> Dict[str, Any]:
        return {
            "tier": "retrieval",
            "size": len(self._ids),
            "embed_dim": self.embed_dim,
            "conflict_threshold": self.conflict_threshold,
            "embedder_type": "TFIDFEmbedder" if self._embedder_is_tfidf
                              else type(self.embed_fn).__name__,
        }

    def audit_log(self, last_n: int = 50) -> List[Dict]:
        return list(self._audit_log[-last_n:])

    def __repr__(self):
        return f"<RetrievalMemory size={len(self._ids)} embed_dim={self.embed_dim}>"


# =====================================================================
# Weights memory — snapshot of trained model weights
# =====================================================================
class WeightsMemory:
    """Read-only snapshot of a trained model's weights.

    Captures the current state of a model's weights at the time of `attach()`.
    Provides:
      - `snapshot()`: returns a deep copy of the weights
      - `diff(model)`: compares current weights to snapshot, returns stats
      - `restore(model)`: restores the snapshot back into the model
      - `hash()`: SHA-256 hash of the weight tensor bytes

    This tier represents "permanent" memory — slow to change (only via
    finetune), baked in during training. Useful for:
      - Comparing weights before/after finetune
      - Detecting weight drift
      - Restoring known-good weight state
      - Federated-learning weight-diff tracking
    """
    def __init__(self):
        self._snapshot: Optional[Dict[str, np.ndarray]] = None
        self._hash: Optional[str] = None
        self._attached_at: Optional[float] = None
        self._model_repr: Optional[str] = None
        self._audit_log: List[Dict] = []

    def attach(self, model) -> None:
        """Capture a snapshot of the model's current weights.

        Args:
            model: A qai Model or any object with `weights` attribute, or any
                object with a `state_dict()` method (PyTorch models).
        """
        snapshot: Dict[str, np.ndarray] = {}

        # Try qai Model.weights (WeightAccessor)
        weights = getattr(model, "weights", None)
        if weights is not None and hasattr(weights, "raw"):
            try:
                snapshot["main"] = np.asarray(weights.raw()).copy()
                snapshot["bias"] = np.asarray(weights.bias_flat()).copy()
            except Exception:
                pass

        # Try PyTorch state_dict
        if not snapshot and hasattr(model, "state_dict"):
            try:
                import torch
                sd = model.state_dict()
                for k, v in sd.items():
                    snapshot[k] = v.detach().cpu().numpy().copy()
            except Exception:
                pass

        # Try qai technique with .net (Transformer, NeuralNetwork)
        if not snapshot:
            technique = getattr(model, "technique", None)
            if technique is not None and hasattr(technique, "net") and technique.net is not None:
                try:
                    import torch
                    for k, v in technique.net.state_dict().items():
                        snapshot[k] = v.detach().cpu().numpy().copy()
                except Exception:
                    pass

        if not snapshot:
            raise ValueError(
                "[WHAT] Could not extract weights from model\n"
                "[WHY] Model has no `weights` attribute (qai WeightAccessor), no "
                "state_dict() method (PyTorch), and no `.technique.net` (qai Transformer)\n"
                "[FIX] Pass a trained qai Model or a PyTorch nn.Module"
            )

        self._snapshot = snapshot
        self._hash = self._compute_hash(snapshot)
        self._attached_at = time.time()
        self._model_repr = repr(model)[:200]
        self._audit_log.append({
            "action": "attach", "timestamp": self._attached_at,
            "n_tensors": len(snapshot),
            "total_params": int(sum(v.size for v in snapshot.values())),
            "hash": self._hash,
        })

    @staticmethod
    def _compute_hash(snapshot: Dict[str, np.ndarray]) -> str:
        h = hashlib.sha256()
        for key in sorted(snapshot.keys()):
            h.update(key.encode("utf-8"))
            h.update(snapshot[key].tobytes())
        return h.hexdigest()

    def snapshot(self) -> Dict[str, np.ndarray]:
        """Return a deep copy of the weight snapshot."""
        if self._snapshot is None:
            raise RuntimeError("WeightsMemory not attached — call .attach(model) first")
        return {k: v.copy() for k, v in self._snapshot.items()}

    def hash(self) -> str:
        """Return the SHA-256 hash of the snapshot."""
        if self._hash is None:
            raise RuntimeError("WeightsMemory not attached")
        return self._hash

    def diff(self, model) -> Dict[str, Any]:
        """Compare current model weights to the snapshot.

        Returns:
            Dict with per-tensor L2 distance, mean abs change, max abs change,
            and overall stats.
        """
        if self._snapshot is None:
            raise RuntimeError("WeightsMemory not attached")

        # Get current weights
        current: Dict[str, np.ndarray] = {}
        weights = getattr(model, "weights", None)
        if weights is not None and hasattr(weights, "raw"):
            try:
                current["main"] = np.asarray(weights.raw()).copy()
                current["bias"] = np.asarray(weights.bias_flat()).copy()
            except Exception:
                pass
        if not current and hasattr(model, "state_dict"):
            try:
                import torch
                for k, v in model.state_dict().items():
                    current[k] = v.detach().cpu().numpy().copy()
            except Exception:
                pass
        if not current:
            technique = getattr(model, "technique", None)
            if technique is not None and hasattr(technique, "net") and technique.net is not None:
                try:
                    import torch
                    for k, v in technique.net.state_dict().items():
                        current[k] = v.detach().cpu().numpy().copy()
                except Exception:
                    pass

        if not current:
            raise ValueError("Could not extract current weights from model")

        diffs: Dict[str, Dict[str, float]] = {}
        all_abs_changes: List[float] = []
        for key, snap_v in self._snapshot.items():
            if key not in current:
                diffs[key] = {"missing": True}
                continue
            cur_v = current[key]
            if snap_v.shape != cur_v.shape:
                diffs[key] = {"shape_mismatch": True}
                continue
            delta = cur_v - snap_v
            l2 = float(np.sqrt(np.sum(delta ** 2)))
            mean_abs = float(np.mean(np.abs(delta)))
            max_abs = float(np.max(np.abs(delta))) if delta.size > 0 else 0.0
            diffs[key] = {
                "l2_distance": l2,
                "mean_abs_change": mean_abs,
                "max_abs_change": max_abs,
                "n_changed": int(np.sum(np.abs(delta) > 1e-8)),
            }
            all_abs_changes.extend(np.abs(delta).flatten().tolist())

        overall = {
            "mean_abs_change": float(np.mean(all_abs_changes)) if all_abs_changes else 0.0,
            "max_abs_change": float(np.max(all_abs_changes)) if all_abs_changes else 0.0,
            "total_changed_params": sum(d.get("n_changed", 0) for d in diffs.values()
                                          if isinstance(d, dict) and "n_changed" in d),
            "hash_match": self._compute_hash(current) == self._hash,
        }
        return {"per_tensor": diffs, "overall": overall}

    def restore(self, model) -> None:
        """Restore the snapshot back into the model.

        Useful for undoing a finetune that went wrong, or for resetting to
        a known-good state. Works for:
          - PyTorch-backed qai models (Transformer, NeuralNetwork) via state_dict
          - qai classical techniques with `w`/`b` attributes (Regression, etc.)
            via direct weight write
          - Any sklearn-backed qai model with `coef_`/`intercept_`
        """
        if self._snapshot is None:
            raise RuntimeError("WeightsMemory not attached")

        # Path 1: PyTorch state_dict (Transformer, NeuralNetwork)
        target = None
        if hasattr(model, "state_dict"):
            target = model
        else:
            technique = getattr(model, "technique", None)
            if technique is not None and hasattr(technique, "net") and technique.net is not None:
                target = technique.net

        if target is not None:
            try:
                import torch
                with torch.no_grad():
                    for k, v in self._snapshot.items():
                        if k in target.state_dict():
                            target.state_dict()[k].copy_(torch.from_numpy(v))
                self._audit_log.append({
                    "action": "restore", "timestamp": time.time(), "method": "state_dict",
                })
                return
            except Exception as e:
                raise RuntimeError(f"Failed to restore weights via state_dict: {e}")

        # Path 2: qai classical model with `w`/`b` (Regression)
        technique = getattr(model, "technique", None)
        if technique is not None and hasattr(technique, "w") and technique.w is not None:
            try:
                if "main" in self._snapshot:
                    snap_w = self._snapshot["main"]
                    cur_w = np.asarray(technique.w)
                    if snap_w.shape == cur_w.shape:
                        technique.w = snap_w.copy()
                if "bias" in self._snapshot and hasattr(technique, "b"):
                    snap_b = self._snapshot["bias"]
                    if snap_b.size == 1:
                        technique.b = float(snap_b[0])
                    else:
                        technique.b = snap_b.copy()
                self._audit_log.append({
                    "action": "restore", "timestamp": time.time(), "method": "direct_w_b",
                })
                return
            except Exception as e:
                raise RuntimeError(f"Failed to restore via w/b: {e}")

        # Path 3: sklearn-backed technique with coef_/intercept_
        if technique is not None and hasattr(technique, "model") and hasattr(technique.model, "coef_"):
            try:
                if "main" in self._snapshot:
                    snap_w = self._snapshot["main"]
                    cur_w = np.asarray(technique.model.coef_)
                    if snap_w.shape == cur_w.shape:
                        technique.model.coef_ = snap_w.copy()
                if "bias" in self._snapshot:
                    snap_b = self._snapshot["bias"]
                    if snap_b.size == 1:
                        technique.model.intercept_ = float(snap_b[0])
                    else:
                        technique.model.intercept_ = snap_b.copy()
                self._audit_log.append({
                    "action": "restore", "timestamp": time.time(), "method": "sklearn_coef",
                })
                return
            except Exception as e:
                raise RuntimeError(f"Failed to restore via sklearn coef_: {e}")

        raise ValueError(
            "[WHAT] Cannot restore weights into this model\n"
            "[WHY] Model isn't PyTorch-backed and doesn't expose w/b or coef_/intercept_\n"
            "[FIX] Use a Transformer, NeuralNetwork, Regression, or sklearn-backed qai technique"
        )

    def summary(self) -> Dict[str, Any]:
        if self._snapshot is None:
            return {"tier": "weights", "attached": False}
        return {
            "tier": "weights",
            "attached": True,
            "n_tensors": len(self._snapshot),
            "total_params": int(sum(v.size for v in self._snapshot.values())),
            "hash": self._hash[:16] + "...",
            "attached_at": self._attached_at,
            "model": self._model_repr,
        }

    def audit_log(self, last_n: int = 50) -> List[Dict]:
        return list(self._audit_log[-last_n:])

    def __repr__(self):
        if self._snapshot is None:
            return "<WeightsMemory unattached>"
        return f"<WeightsMemory n_tensors={len(self._snapshot)} hash={self._hash[:8]}...>"


# =====================================================================
# Top-level Memory coordinator
# =====================================================================
class Memory:
    """Top-level coordinator for the three-tier memory system.

    Holds references to all three tiers and provides convenience methods for
    cross-tier operations (snapshots, audits, etc.).

    Usage:
        mem = Memory()
        mem.attach(model)            # capture weights snapshot
        mem.working.add("hello")
        mem.retrieval.add("fact 1")
        mem.save("/tmp/memory.json") # full state to disk

    Picklability: WorkingMemory stores an `auto_promote_callback` closure
    that pickle cannot serialize. We implement __getstate__/__setstate__ to
    skip the callback on save and rebuild it on load. This means any agent
    that holds a `qai.memory.Memory` instance can now be pickled -- which
    is essential for bundling agents in .qmodel files.
    """
    def __init__(
        self,
        working: Optional[WorkingMemory] = None,
        retrieval: Optional[RetrievalMemory] = None,
        weights: Optional[WeightsMemory] = None,
        auto_link: bool = True,
    ):
        """
        Args:
            working: WorkingMemory instance (created with defaults if None).
            retrieval: RetrievalMemory instance (created with defaults if None).
            weights: WeightsMemory instance (created if None).
            auto_link: If True, set up auto-promotion: items accessed
                >= 5 times in working memory are auto-added to retrieval.
        """
        self.working = working or WorkingMemory()
        self.retrieval = retrieval or RetrievalMemory()
        self.weights = weights or WeightsMemory()
        # Optional permanent tier (growing accumulated state across runs).
        # Not auto-instantiated -- users opt in via attach_permanent() or by
        # passing permanent=PermanentMemory(...).
        self.permanent: Optional["PermanentMemory"] = None

        if auto_link:
            self._setup_auto_promotion()

    def _setup_auto_promotion(self, threshold: int = 5) -> None:
        """Wire up working→retrieval auto-promotion."""
        def promote(text: str, metadata: Dict) -> None:
            self.retrieval.add(text, metadata=metadata)
        self.working.auto_promote_threshold = threshold
        self.working.auto_promote_callback = promote

    # --- Picklability ----------------------------------------------------
    def __getstate__(self) -> Dict[str, Any]:
        """Strip the unpicklable auto_promote_callback before pickling.
        The callback is rebuilt on load via _setup_auto_promotion()."""
        state = self.__dict__.copy()
        # WorkingMemory's callback is a closure -- drop it
        if "working" in state and state["working"] is not None:
            w = state["working"]
            w_dict = {k: v for k, v in w.__dict__.items()
                      if k != "auto_promote_callback"}
            state["working"] = w_dict  # store as dict, restore in __setstate__
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore a pickled Memory, rebuilding the auto_promote_callback."""
        # Restore WorkingMemory from its dict (callback was stripped)
        w_dict = state.pop("working", None)
        if isinstance(w_dict, dict):
            w = WorkingMemory(
                capacity=w_dict.get("capacity", 50),
                policy=w_dict.get("policy", "lru"),
            )
            # Restore all the original attributes
            for k, v in w_dict.items():
                setattr(w, k, v)
            self.working = w
        else:
            self.working = w_dict or WorkingMemory()
        # Restore the rest
        for k, v in state.items():
            setattr(self, k, v)
        # Rebuild the auto-promotion callback
        self._setup_auto_promotion()

    def attach(self, model) -> None:
        """Capture a snapshot of the model's current weights."""
        self.weights.attach(model)

    def attach_permanent(self, permanent: "PermanentMemory") -> None:
        """Attach a PermanentMemory tier (growing accumulated state)."""
        self.permanent = permanent

    def save(self, path: str) -> str:
        """Save the full memory state (working + retrieval + weights hash
        + permanent tier if attached) to JSON.

        Note: weights themselves are NOT saved by default (they're a
        snapshot reference -- only the hash + metadata is persisted).
        To persist the model's weights, use `qai.save_bundle(model, ...)`
        or `model.export(path)` separately.

        The permanent tier, if attached, IS persisted (it represents
        growing state, not a snapshot).
        """
        state = {
            "working": {
                "capacity": self.working.capacity,
                "policy": self.working.policy,
                "items": [
                    {
                        "text": it.text, "timestamp": it.timestamp,
                        "access_count": it.access_count,
                        "last_accessed": it.last_accessed,
                        "importance": it.importance,
                        "metadata": it.metadata,
                    }
                    for it in self.working.all_items()
                ],
                "audit_log": self.working.audit_log(last_n=10000),
            },
            "retrieval_path": None,  # see note below
            "weights_summary": self.weights.summary(),
            "permanent_path": None,
        }

        # Save retrieval memory to a sibling file
        if len(self.retrieval) > 0:
            retrieval_path = path.replace(".json", ".retrieval.json")
            self.retrieval.save(retrieval_path)
            state["retrieval_path"] = retrieval_path

        # Save permanent memory to a sibling file (if attached)
        if self.permanent is not None:
            permanent_path = path.replace(".json", ".permanent.json")
            self.permanent.save(permanent_path)
            state["permanent_path"] = permanent_path

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Memory":
        """Load a memory state saved by .save(). Note: weights tier will be
        empty (snapshots aren't persisted -- re-attach a model to repopulate).
        The permanent tier IS restored if a sibling .permanent.json exists.
        """
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        # Restore working memory
        w_state = state.get("working", {})
        working = WorkingMemory(
            capacity=w_state.get("capacity", 50),
            policy=w_state.get("policy", "lru"),
        )
        for item_dict in w_state.get("items", []):
            item = WorkingItem(
                text=item_dict["text"],
                timestamp=item_dict["timestamp"],
                access_count=item_dict.get("access_count", 0),
                last_accessed=item_dict.get("last_accessed", item_dict["timestamp"]),
                importance=item_dict.get("importance", 0.0),
                metadata=item_dict.get("metadata", {}),
            )
            working._items.append(item)

        # Restore retrieval memory (from sibling file if present)
        retrieval = None
        retrieval_path = state.get("retrieval_path")
        if retrieval_path and os.path.exists(retrieval_path):
            retrieval = RetrievalMemory.load(retrieval_path)

        # Restore permanent memory (from sibling file if present)
        permanent = None
        permanent_path = state.get("permanent_path")
        if permanent_path and os.path.exists(permanent_path):
            permanent = PermanentMemory.load(permanent_path)

        mem = cls(
            working=working,
            retrieval=retrieval or RetrievalMemory(),
            weights=WeightsMemory(),
            auto_link=False,  # don't override the loaded working config
        )
        if permanent is not None:
            mem.attach_permanent(permanent)
        return mem

    def summary(self) -> Dict[str, Any]:
        return {
            "working": self.working.summary(),
            "retrieval": self.retrieval.summary(),
            "weights": self.weights.summary(),
            "permanent": (self.permanent.summary() if self.permanent else None),
        }

    def full_audit(self, last_n: int = 100) -> Dict[str, List[Dict]]:
        """Return recent audit-log entries from all tiers."""
        result = {
            "working": self.working.audit_log(last_n=last_n),
            "retrieval": self.retrieval.audit_log(last_n=last_n),
            "weights": self.weights.audit_log(last_n=last_n),
        }
        if self.permanent is not None:
            result["permanent"] = self.permanent.audit_log(last_n=last_n)
        return result

    def __repr__(self):
        return (
            f"<Memory working={len(self.working)} "
            f"retrieval={len(self.retrieval)} "
            f"weights_attached={self.weights._snapshot is not None} "
            f"permanent={'yes' if self.permanent else 'no'}>"
        )


# =====================================================================
# 5. PermanentMemory — the "growing accumulated state" tier
# =====================================================================
class PermanentMemory:
    """The "permanent but growing" tier.

    Unlike WeightsMemory (which is a read-only snapshot), PermanentMemory
    holds an arbitrary picklable agent state that GROWS over time. This is
    the bridge between "frozen training snapshot" and "ephemeral per-session
    working memory".

    Use cases:
      - An RL agent's accumulated Q-table + episode log, persisting across
        dashboard sessions.
      - A chatbot's long-term conversation history (across server restarts).
      - Any "checkpoint-on-shutdown, restore-on-launch" pattern.

    Storage format: JSON sidecar file with metadata + base64-encoded pickle
    of the user-supplied state object.

    Usage:
        pm = PermanentMemory(sidecar_path="agent.brain.json")
        pm.save(agent_state)               # checkpoint
        agent_state = pm.load_or_init(default_factory=lambda: Agent())
    """
    def __init__(self, sidecar_path: str,
                 origin_id: Optional[str] = None):
        """
        Args:
            sidecar_path: Path to the JSON sidecar file.
            origin_id: Optional identifier for the bundle/origin this
                       permanent memory belongs to (used to detect
                       "user replaced the bundle" mismatches).
        """
        self.sidecar_path = sidecar_path
        self.origin_id = origin_id
        self._audit_log: List[Dict[str, Any]] = []

    def save(self, state: Any) -> str:
        """Persist `state` (any picklable object) to the sidecar file.

        Atomic write: writes to a .tmp file then renames, so a crash
        mid-save cannot corrupt the existing brain.
        """
        import pickle as _pickle, base64 as _b64
        state_bytes = _pickle.dumps(state)
        payload_b64 = _b64.b64encode(state_bytes).decode("ascii")

        # Read existing metadata (to increment counters)
        existing_meta = self._read_metadata()

        meta = {
            "format_version": "permanent-memory-1.0",
            "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "saved_at_epoch": time.time(),
            "n_saves": existing_meta.get("n_saves", 0) + 1,
            "origin_id": self.origin_id or existing_meta.get("origin_id"),
            "state_class": type(state).__name__,
            "state_size_bytes": len(state_bytes),
        }
        out = {"metadata": meta, "state_b64": payload_b64}

        # Atomic write
        tmp_path = self.sidecar_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.sidecar_path)

        self._audit_log.append({
            "action": "save", "timestamp": meta["saved_at_epoch"],
            "n_saves": meta["n_saves"], "state_size_bytes": len(state_bytes),
        })
        return self.sidecar_path

    def load(self) -> Optional[Any]:
        """Load and return the persisted state, or None if no sidecar exists."""
        if not os.path.exists(self.sidecar_path):
            return None
        import pickle as _pickle, base64 as _b64
        with open(self.sidecar_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        state_bytes = _b64.b64decode(data["state_b64"])
        state = _pickle.loads(state_bytes)
        # Origin-id mismatch check (warn-only)
        if (self.origin_id
            and data.get("metadata", {}).get("origin_id")
            and data["metadata"]["origin_id"] != self.origin_id):
            import sys
            print(f"[PermanentMemory] WARNING: origin_id mismatch "
                  f"(sidecar was for a different bundle). Loading anyway.",
                  file=sys.stderr)
        self._audit_log.append({
            "action": "load", "timestamp": time.time(),
            "n_saves": data.get("metadata", {}).get("n_saves", 0),
        })
        return state

    def load_or_init(self, default_factory) -> Any:
        """Load the persisted state, or call `default_factory()` to create a
        fresh one (and persist it as the seed).

        Args:
            default_factory: Zero-arg callable that returns a fresh state
                             object (used when no sidecar exists yet).

        Returns:
            The loaded or freshly-created state.
        """
        state = self.load()
        if state is None:
            state = default_factory()
            self.save(state)
        return state

    def reset(self) -> bool:
        """Delete the sidecar file. The next load_or_init() will start fresh."""
        if os.path.exists(self.sidecar_path):
            os.remove(self.sidecar_path)
            self._audit_log.append({
                "action": "reset", "timestamp": time.time(),
            })
            return True
        return False

    def info(self) -> Dict[str, Any]:
        """Return metadata about the permanent memory sidecar."""
        meta = self._read_metadata()
        return {
            "sidecar_path": self.sidecar_path,
            "sidecar_exists": os.path.exists(self.sidecar_path),
            "n_saves": meta.get("n_saves", 0),
            "saved_at_iso": meta.get("saved_at_iso"),
            "saved_at_epoch": meta.get("saved_at_epoch", 0),
            "seconds_since_save": (time.time() - meta.get("saved_at_epoch", 0))
                                   if meta.get("saved_at_epoch") else None,
            "origin_id": meta.get("origin_id"),
            "expected_origin_id": self.origin_id,
            "origin_matches": (meta.get("origin_id") == self.origin_id
                                if self.origin_id and meta.get("origin_id")
                                else None),
            "state_class": meta.get("state_class"),
            "state_size_bytes": meta.get("state_size_bytes", 0),
        }

    def summary(self) -> Dict[str, Any]:
        return self.info()

    def audit_log(self, last_n: int = 100) -> List[Dict]:
        return list(self._audit_log[-last_n:])

    def _read_metadata(self) -> Dict[str, Any]:
        if not os.path.exists(self.sidecar_path):
            return {}
        try:
            with open(self.sidecar_path, "r", encoding="utf-8") as f:
                return json.load(f).get("metadata", {})
        except Exception:
            return {}


# =====================================================================
# Public API
# =====================================================================
__all__ = [
    "Memory",
    "WorkingMemory",
    "WorkingItem",
    "RetrievalMemory",
    "RetrievalResult",
    "WeightsMemory",
    "PermanentMemory",
    "TFIDFEmbedder",
    "hash_embedding",
    "random_embedding",
]
