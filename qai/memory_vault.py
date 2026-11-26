"""
qai.memory_vault — a tiered persistent memory system for AI agents.

Three tiers with smart eviction, compression, and recall scoring:
  HOT  — in-memory LRU cache (fast, small)
  WARM — on-disk SQLite/compressed JSON (medium speed, medium capacity)
  COLD — compressed archive (slow, large capacity)

Each memory item has:
  - content (text or bytes)
  - importance score (0..1)
  - access count + last_accessed timestamp
  - tags for retrieval
  - tier (hot/warm/cold)

The vault automatically promotes frequently-accessed items to hotter tiers
and demotes rarely-accessed items to colder tiers.

Usage:
    import qai
    vault = qai.MemoryVault("./my_vault")

    # Store
    vault.store("event_1", "User logged in", importance=0.8, tags=["auth", "login"])
    vault.store("event_2", b"\\x89PNG...", importance=0.5, tags=["image"])

    # Recall
    results = vault.recall("login", top_k=5)
    for r in results:
        print(r["key"], r["content"][:50], r["score"])

    # Promote/demote based on access patterns
    vault.rebalance()

    # Stats
    print(vault.stats())
"""
from __future__ import annotations
import os, json, time, hashlib, heapq, pickle, zlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import OrderedDict, defaultdict
import numpy as np


@dataclass
class MemoryItem:
    """A single memory item in the vault."""
    key: str
    content: Union[str, bytes]
    importance: float
    tags: List[str] = field(default_factory=list)
    tier: str = "warm"  # hot, warm, cold
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    content_hash: str = ""
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_accessed:
            self.last_accessed = self.created_at
        if not self.content_hash:
            if isinstance(self.content, str):
                self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            else:
                self.content_hash = hashlib.sha256(self.content).hexdigest()[:16]

    @property
    def recall_score(self) -> float:
        """Score for ranking in recall: importance * recency * frequency."""
        age = time.time() - self.last_accessed
        recency = 1.0 / (1.0 + age / 3600)  # decay over hours
        frequency = 1.0 + 0.1 * self.access_count
        return self.importance * recency * frequency


class MemoryVault:
    """A tiered persistent memory system.

    Args:
        storage_path: directory for warm/cold storage.
        hot_capacity: max items in the hot (in-memory) tier.
        warm_capacity: max items in the warm (on-disk) tier.
        auto_compress: if True, compress content when moving to cold tier.
    """
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"

    def __init__(self, storage_path: str = "./memory_vault",
                 hot_capacity: int = 100,
                 warm_capacity: int = 1000,
                 auto_compress: bool = True):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "warm"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "cold"), exist_ok=True)
        self.hot_capacity = hot_capacity
        self.warm_capacity = warm_capacity
        self.auto_compress = auto_compress

        self._hot: OrderedDict[str, MemoryItem] = OrderedDict()
        self._warm_index: Dict[str, MemoryItem] = {}  # key -> metadata (content on disk)
        self._cold_index: Dict[str, MemoryItem] = {}
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)

        self._load_indexes()

    def store(self, key: str, content: Union[str, bytes],
              importance: float = 0.5, tags: Optional[List[str]] = None,
              tier: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Store a memory item.

        Args:
            key: unique key for this memory.
            content: text or bytes to store.
            importance: 0..1 score (higher = more important).
            tags: list of tags for retrieval.
            tier: force a specific tier (default: auto-select based on capacity).
            metadata: extra metadata.

        Returns:
            The key.
        """
        tags = tags or []
        metadata = metadata or {}

        # Remove old item with same key
        if key in self._hot or key in self._warm_index or key in self._cold_index:
            self.remove(key)

        item = MemoryItem(
            key=key, content=content, importance=importance,
            tags=tags, tier=tier or self.HOT, metadata=metadata,
        )

        # Determine tier
        if tier:
            item.tier = tier
        elif len(self._hot) < self.hot_capacity:
            item.tier = self.HOT
        else:
            item.tier = self.WARM

        # Store in the appropriate tier
        if item.tier == self.HOT:
            self._hot[key] = item
        elif item.tier == self.WARM:
            self._write_warm(item)
        else:
            self._write_cold(item)

        # Update tag index
        for tag in tags:
            self._tag_index[tag].add(key)

        self._save_indexes()
        return key

    def recall(self, query: str = "", tags: Optional[List[str]] = None,
               top_k: int = 10) -> List[Dict[str, Any]]:
        """Recall memories matching the query and/or tags.

        Args:
            query: text to search in content (empty = match all).
            tags: filter by tags (OR logic).
            top_k: max results.

        Returns:
            List of {key, content, score, tier, tags, importance} sorted by score.
        """
        # Gather candidate keys
        candidates: Set[str] = set()
        if tags:
            for tag in tags:
                candidates.update(self._tag_index.get(tag, set()))
        if not tags and not query:
            candidates = set(self._hot.keys()) | set(self._warm_index.keys()) | set(self._cold_index.keys())
        if not candidates:
            candidates = set(self._hot.keys()) | set(self._warm_index.keys()) | set(self._cold_index.keys())

        # Score and sort
        scored: List[Tuple[float, str]] = []
        for key in candidates:
            item = self._get_item(key)
            if item is None:
                continue
            # Content match
            if query:
                content_str = item.content if isinstance(item.content, str) else ""
                if query.lower() not in content_str.lower():
                    continue  # skip non-matching
            scored.append((item.recall_score, key))

        scored.sort(reverse=True)
        results = []
        for score, key in scored[:top_k]:
            item = self._get_item(key)
            if item:
                item.access_count += 1
                item.last_accessed = time.time()
                results.append({
                    "key": key,
                    "content": item.content[:200] if isinstance(item.content, str) else f"[{len(item.content)} bytes]",
                    "score": round(score, 4),
                    "tier": item.tier,
                    "tags": item.tags,
                    "importance": item.importance,
                    "access_count": item.access_count,
                })
        return results

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a specific memory by key."""
        item = self._get_item(key)
        if item:
            item.access_count += 1
            item.last_accessed = time.time()
            return item.content
        return None

    def remove(self, key: str) -> bool:
        """Remove a memory item."""
        removed = False
        if key in self._hot:
            del self._hot[key]
            removed = True
        if key in self._warm_index:
            item = self._warm_index[key]
            path = os.path.join(self.storage_path, "warm", f"{key}.pkl")
            if os.path.exists(path):
                os.remove(path)
            del self._warm_index[key]
            removed = True
        if key in self._cold_index:
            item = self._cold_index[key]
            path = os.path.join(self.storage_path, "cold", f"{key}.pkl")
            if os.path.exists(path):
                os.remove(path)
            del self._cold_index[key]
            removed = True
        # Clean tag index
        for tag in list(self._tag_index.keys()):
            self._tag_index[tag].discard(key)
            if not self._tag_index[tag]:
                del self._tag_index[tag]
        if removed:
            self._save_indexes()
        return removed

    def rebalance(self):
        """Promote/demote items between tiers based on access patterns.

        - Items in HOT that haven't been accessed recently are demoted to WARM.
        - Items in WARM/COLD that are frequently accessed are promoted to HOT.
        - Items in WARM that exceed capacity are demoted to COLD.
        """
        now = time.time()

        # Demote hot -> warm (items not accessed in last hour + low importance)
        to_demote = []
        for key, item in self._hot.items():
            age = now - item.last_accessed
            if age > 3600 and item.importance < 0.5 and item.access_count < 3:
                to_demote.append(key)
        for key in to_demote:
            item = self._hot.pop(key)
            item.tier = self.WARM
            self._write_warm(item)

        # Promote warm/cold -> hot (frequently accessed)
        all_lower = list(self._warm_index.items()) + list(self._cold_index.items())
        all_lower.sort(key=lambda x: x[1].recall_score, reverse=True)
        for key, item in all_lower[:5]:  # top 5 candidates
            if item.access_count > 2 and item.importance > 0.3:
                if len(self._hot) < self.hot_capacity:
                    content = self._read_content(key, item.tier)
                    if content is not None:
                        item.content = content
                        item.tier = self.HOT
                        self._hot[key] = item
                        # Remove from lower tier
                        if key in self._warm_index:
                            del self._warm_index[key]
                        if key in self._cold_index:
                            del self._cold_index[key]

        # Demote warm -> cold (exceeds capacity)
        if len(self._warm_index) > self.warm_capacity:
            sorted_warm = sorted(self._warm_index.items(), key=lambda x: x[1].recall_score)
            to_cold = sorted_warm[:len(self._warm_index) - self.warm_capacity]
            for key, item in to_cold:
                content = self._read_content(key, self.WARM)
                if content is not None:
                    item.content = content
                    item.tier = self.COLD
                    self._write_cold(item)
                    del self._warm_index[key]

        self._save_indexes()

    def stats(self) -> Dict[str, Any]:
        """Return vault statistics."""
        total_items = len(self._hot) + len(self._warm_index) + len(self._cold_index)
        avg_importance = 0.0
        if total_items > 0:
            all_items = list(self._hot.values()) + list(self._warm_index.values()) + list(self._cold_index.values())
            avg_importance = float(np.mean([i.importance for i in all_items]))
        return {
            "total_items": total_items,
            "hot_items": len(self._hot),
            "warm_items": len(self._warm_index),
            "cold_items": len(self._cold_index),
            "hot_capacity": self.hot_capacity,
            "warm_capacity": self.warm_capacity,
            "n_tags": len(self._tag_index),
            "avg_importance": round(avg_importance, 3),
            "storage_path": self.storage_path,
        }

    def __repr__(self):
        s = self.stats()
        return f"<MemoryVault hot={s['hot_items']} warm={s['warm_items']} cold={s['cold_items']} tags={s['n_tags']}>"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_item(self, key: str) -> Optional[MemoryItem]:
        """Get a MemoryItem (loads content from disk if needed)."""
        if key in self._hot:
            return self._hot[key]
        if key in self._warm_index:
            item = self._warm_index[key]
            item.content = self._read_content(key, self.WARM) or b""
            return item
        if key in self._cold_index:
            item = self._cold_index[key]
            item.content = self._read_content(key, self.COLD) or b""
            return item
        return None

    def _write_warm(self, item: MemoryItem):
        """Write item to warm tier (on-disk)."""
        path = os.path.join(self.storage_path, "warm", f"{item.key}.pkl")
        content_to_write = item.content
        if self.auto_compress and isinstance(item.content, (str, bytes)):
            raw = item.content.encode() if isinstance(item.content, str) else item.content
            content_to_write = zlib.compress(raw)
            item.compressed = True
        with open(path, "wb") as f:
            pickle.dump({"content": content_to_write, "metadata": asdict(item)}, f)
        self._warm_index[item.key] = item

    def _write_cold(self, item: MemoryItem):
        """Write item to cold tier (compressed archive)."""
        path = os.path.join(self.storage_path, "cold", f"{item.key}.pkl")
        raw = item.content.encode() if isinstance(item.content, str) else item.content
        compressed = zlib.compress(raw, level=9)
        item.compressed = True
        with open(path, "wb") as f:
            pickle.dump({"content": compressed, "metadata": asdict(item)}, f)
        self._cold_index[item.key] = item

    def _read_content(self, key: str, tier: str) -> Optional[Union[str, bytes]]:
        """Read content from disk."""
        tier_dir = "warm" if tier == self.WARM else "cold"
        path = os.path.join(self.storage_path, tier_dir, f"{key}.pkl")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            content = data["content"]
            if isinstance(content, bytes) and data["metadata"].get("compressed"):
                content = zlib.decompress(content)
                # Try to decode as string
                try:
                    content = content.decode("utf-8")
                except UnicodeDecodeError:
                    pass
            return content
        except Exception:
            return None

    def _save_indexes(self):
        """Save warm + cold indexes to disk."""
        for tier, index in [(self.WARM, self._warm_index), (self.COLD, self._cold_index)]:
            path = os.path.join(self.storage_path, tier, "_index.json")
            data = {k: {kk: vv for kk, vv in asdict(v).items() if kk != "content"}
                    for k, v in index.items()}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def _load_indexes(self):
        """Load warm + cold indexes from disk."""
        for tier in [self.WARM, self.COLD]:
            path = os.path.join(self.storage_path, tier, "_index.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                for key, info in data.items():
                    item = MemoryItem(**{k: v for k, v in info.items() if k != "content"})
                    if tier == self.WARM:
                        self._warm_index[key] = item
                    else:
                        self._cold_index[key] = item
                    for tag in item.tags:
                        self._tag_index[tag].add(key)
            except Exception:
                pass


__all__ = ["MemoryVault", "MemoryItem"]
