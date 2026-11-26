"""
qai.lineage -- model lineage / provenance tracker.

Records the family tree of model versions (which model was fine-tuned from
which, which was quantized from which, ...) and renders it visually as a
directed acyclic graph (DAG).

  LineageNode       -- a node in the lineage graph (one model version)
  LineageEdge       -- a directed edge (parent -> child)
  LineageGraph      -- the full DAG of model versions
  LineageTracker    -- singleton-like recorder that auto-tracks operations
  LineageVisualizer -- renders the lineage graph (mermaid/dot/d3/html/json)

Usage:
    import qai
    from qai.lineage import LineageTracker, LineageVisualizer

    t = LineageTracker()
    t.record_create("model_a", "base_v1", metadata={"technique": "mlp"})
    t.record_finetune("model_a_v2", "model_a", metadata={"lr": 0.001})
    t.record_quantize("model_a_int8", "model_a_v2", format="int8")

    g = t.graph()
    print(g.stats())
    print(LineageVisualizer().to_mermaid(g))

    # Convenience module-level API
    import qai.lineage as lineage
    lineage.visualize("model_a_int8", format="html")

Conventions:
  * model_id is the user-facing identifier of a model version (a string).
  * If model_id is None it is auto-generated via hashlib.md5(name+timestamp)[:12].
  * Parent links are recorded both on the node (single parent) and as edges
    (multi-parent for `merge`).
  * Operations drive color-coding in the Mermaid / DOT / HTML renderers.
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import deque, defaultdict

# numpy is part of the qai convention; import lazily-safe (always available
# in a qai install) but not strictly required by this module.
try:
    import numpy as np  # noqa: F401
except Exception:  # pragma: no cover
    np = None  # type: ignore


# ============================================================================
# Constants
# ============================================================================
VALID_OPERATIONS: Tuple[str, ...] = (
    "create", "train", "finetune", "quantize", "prune",
    "distill", "convert", "merge", "clone", "deploy",
)

# Color codes for each operation in the visualizations
OPERATION_COLORS: Dict[str, str] = {
    "create":   "#2196F3",  # blue
    "train":    "#4CAF50",  # green
    "finetune": "#009688",  # teal
    "quantize": "#FF9800",  # orange
    "prune":    "#F44336",  # red
    "distill":  "#9C27B0",  # purple
    "convert":  "#9E9E9E",  # gray
    "merge":    "#E91E63",  # pink
    "clone":    "#00BCD4",  # cyan
    "deploy":   "#FFC107",  # gold
}


# ============================================================================
# Helpers
# ============================================================================
def _now_iso() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gen_id(name: str, timestamp: Optional[float] = None) -> str:
    """Auto-generate a 12-char hex id from name + timestamp (+ pid salt).

    Used when the caller does not supply an explicit model_id.
    """
    ts = timestamp if timestamp is not None else time.time()
    raw = f"{name}|{ts}|{os.getpid()}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def _sanitize_id(s: str) -> str:
    """Sanitize a string so it can be used as a Mermaid / DOT identifier."""
    out = []
    for ch in str(s):
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    res = "".join(out)
    if not res or res[0].isdigit():
        res = "n_" + res
    return res


def _normalize_parents(parent_ids: Union[None, str, List[str], Tuple[str, ...]]) -> List[str]:
    """Normalize parent_ids argument to a list of parent model_ids."""
    if parent_ids is None:
        return []
    if isinstance(parent_ids, str):
        return [parent_ids]
    if isinstance(parent_ids, (list, tuple)):
        return [str(p) for p in parent_ids]
    return [str(parent_ids)]


# ============================================================================
# 1. LineageNode
# ============================================================================
class LineageNode:
    """A node in the lineage graph (one model version).

    Args:
        model_id: unique id of this model version. If None, auto-generated.
        name: human-readable name (defaults to model_id).
        parent_id: model_id of the primary parent (None for roots / merges).
        operation: how this version was produced.
        metadata: arbitrary dict of extra info.
        timestamp: creation time (epoch seconds). If None, uses now.
    """
    __slots__ = ("model_id", "name", "parent_id", "operation",
                 "metadata", "timestamp")

    def __init__(self,
                 model_id: Optional[str] = None,
                 name: Optional[str] = None,
                 parent_id: Optional[str] = None,
                 operation: str = "create",
                 metadata: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[float] = None):
        if operation not in VALID_OPERATIONS:
            raise ValueError(
                f"[WHAT] Invalid operation '{operation}'\n"
                f"[WHY] Valid operations: {list(VALID_OPERATIONS)}\n"
                f"[FIX] Use one of the supported operations"
            )
        ts = float(timestamp if timestamp is not None else time.time())
        self.timestamp = ts
        self.name = name or f"model_{int(ts)}"
        self.model_id = model_id or _gen_id(self.name, ts)
        self.parent_id = parent_id
        self.operation = operation
        self.metadata = dict(metadata) if metadata else {}

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "operation": self.operation,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "timestamp_iso": _now_iso(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LineageNode":
        return cls(
            model_id=d.get("model_id"),
            name=d.get("name"),
            parent_id=d.get("parent_id"),
            operation=d.get("operation", "create"),
            metadata=d.get("metadata") or {},
            timestamp=d.get("timestamp"),
        )

    def __repr__(self) -> str:
        return (f"<LineageNode id={self.model_id} name={self.name!r} "
                f"op={self.operation} parent={self.parent_id}>")


# ============================================================================
# 2. LineageEdge
# ============================================================================
class LineageEdge:
    """A directed edge in the lineage graph (parent -> child).

    Args:
        parent_id: model_id of the parent.
        child_id:  model_id of the child.
        operation: operation that produced the child.
        notes:     optional human-readable note.
    """
    __slots__ = ("parent_id", "child_id", "operation", "notes")

    def __init__(self, parent_id: str, child_id: str,
                 operation: str = "create", notes: Optional[str] = None):
        if operation not in VALID_OPERATIONS:
            raise ValueError(
                f"[WHAT] Invalid operation '{operation}'\n"
                f"[WHY] Valid operations: {list(VALID_OPERATIONS)}\n"
                f"[FIX] Use one of the supported operations"
            )
        self.parent_id = parent_id
        self.child_id = child_id
        self.operation = operation
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "operation": self.operation,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LineageEdge":
        return cls(
            parent_id=d["parent_id"],
            child_id=d["child_id"],
            operation=d.get("operation", "create"),
            notes=d.get("notes"),
        )

    def __repr__(self) -> str:
        return (f"<LineageEdge {self.parent_id} --{self.operation}--> "
                f"{self.child_id}>")


# ============================================================================
# 3. LineageGraph
# ============================================================================
class LineageGraph:
    """The full DAG of model versions.

    Stores nodes by model_id and a list of directed edges. Maintains
    adjacency maps (_children, _parents) for O(1) traversal.
    """

    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._parents: Dict[str, List[str]] = defaultdict(list)
        self._edge_keys: Set[Tuple[str, str, str]] = set()
        self._lock = threading.RLock()

    # -- mutation ------------------------------------------------------------
    def add_node(self, node: LineageNode) -> None:
        with self._lock:
            self.nodes[node.model_id] = node
            # ensure adjacency entries exist
            self._children.setdefault(node.model_id, [])
            self._parents.setdefault(node.model_id, [])

    def add_edge(self, edge: LineageEdge) -> None:
        with self._lock:
            key = (edge.parent_id, edge.child_id, edge.operation)
            if key in self._edge_keys:
                return
            self._edge_keys.add(key)
            self.edges.append(edge)
            self._children[edge.parent_id].append(edge.child_id)
            self._parents[edge.child_id].append(edge.parent_id)

    # -- queries -------------------------------------------------------------
    def children(self, node_id: str) -> List[LineageNode]:
        with self._lock:
            return [self.nodes[c] for c in self._children.get(node_id, [])
                    if c in self.nodes]

    def parents(self, node_id: str) -> List[LineageNode]:
        with self._lock:
            return [self.nodes[p] for p in self._parents.get(node_id, [])
                    if p in self.nodes]

    def ancestors(self, node_id: str) -> List[LineageNode]:
        """All ancestors of node_id (recursive, BFS order)."""
        with self._lock:
            seen: Set[str] = set()
            out: List[LineageNode] = []
            queue: deque = deque(self._parents.get(node_id, []))
            while queue:
                cur = queue.popleft()
                if cur in seen or cur not in self.nodes:
                    continue
                seen.add(cur)
                out.append(self.nodes[cur])
                queue.extend(self._parents.get(cur, []))
            return out

    def descendants(self, node_id: str) -> List[LineageNode]:
        """All descendants of node_id (recursive, BFS order)."""
        with self._lock:
            seen: Set[str] = set()
            out: List[LineageNode] = []
            queue: deque = deque(self._children.get(node_id, []))
            while queue:
                cur = queue.popleft()
                if cur in seen or cur not in self.nodes:
                    continue
                seen.add(cur)
                out.append(self.nodes[cur])
                queue.extend(self._children.get(cur, []))
            return out

    def root_nodes(self) -> List[LineageNode]:
        """Nodes with no parents."""
        with self._lock:
            return [n for nid, n in self.nodes.items()
                    if not self._parents.get(nid)]

    def leaf_nodes(self) -> List[LineageNode]:
        """Nodes with no children."""
        with self._lock:
            return [n for nid, n in self.nodes.items()
                    if not self._children.get(nid)]

    def path(self, from_id: str, to_id: str) -> List[LineageNode]:
        """Shortest path from from_id to to_id (BFS, inclusive)."""
        with self._lock:
            if from_id not in self.nodes or to_id not in self.nodes:
                return []
            if from_id == to_id:
                return [self.nodes[from_id]]
            visited: Set[str] = {from_id}
            queue: deque = deque([(from_id, [self.nodes[from_id]])])
            while queue:
                cur, path = queue.popleft()
                for nxt in self._children.get(cur, []):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    new_path = path + [self.nodes[nxt]]
                    if nxt == to_id:
                        return new_path
                    queue.append((nxt, new_path))
            return []

    def common_ancestor(self, node_a: str, node_b: str) -> Optional[LineageNode]:
        """Lowest common ancestor of two nodes (BFS order from each)."""
        with self._lock:
            if node_a not in self.nodes or node_b not in self.nodes:
                return None
            # ancestors of a in BFS order (closest first)
            anc_a = [node_a] + [n.model_id for n in self.ancestors(node_a)]
            anc_b_set = set([node_b] + [n.model_id for n in self.ancestors(node_b)])
            for nid in anc_a:
                if nid in anc_b_set and nid in self.nodes:
                    return self.nodes[nid]
            return None

    def topological_sort(self) -> List[str]:
        """Return model_ids in topological order (roots first)."""
        with self._lock:
            in_deg: Dict[str, int] = {nid: 0 for nid in self.nodes}
            for edge in self.edges:
                if edge.child_id in in_deg:
                    in_deg[edge.child_id] += 1
            queue: deque = deque([nid for nid, d in in_deg.items() if d == 0])
            out: List[str] = []
            while queue:
                cur = queue.popleft()
                out.append(cur)
                for child in self._children.get(cur, []):
                    if child in in_deg:
                        in_deg[child] -= 1
                        if in_deg[child] == 0:
                            queue.append(child)
            return out

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "nodes": [n.to_dict() for n in self.nodes.values()],
                "edges": [e.to_dict() for e in self.edges],
            }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LineageGraph":
        g = cls()
        for nd in d.get("nodes", []):
            g.add_node(LineageNode.from_dict(nd))
        for ed in d.get("edges", []):
            g.add_edge(LineageEdge.from_dict(ed))
        return g

    def save(self, path: str) -> None:
        path = os.path.expanduser(path)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "LineageGraph":
        path = os.path.expanduser(path)
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def stats(self) -> Dict[str, Any]:
        """Summary statistics about the graph."""
        with self._lock:
            n_nodes = len(self.nodes)
            n_edges = len(self.edges)
            n_roots = len(self.root_nodes())
            n_leaves = len(self.leaf_nodes())
            # max depth (longest path from any root to any leaf)
            max_depth = 0
            if n_nodes:
                memo: Dict[str, int] = {}

                def depth(nid: str) -> int:
                    if nid in memo:
                        return memo[nid]
                    kids = self._children.get(nid, [])
                    if not kids:
                        memo[nid] = 1
                    else:
                        memo[nid] = 1 + max(
                            (depth(k) for k in kids if k in self.nodes),
                            default=0,
                        )
                    return memo[nid]

                for nid in self.nodes:
                    max_depth = max(max_depth, depth(nid))
            avg_branching = (n_edges / n_nodes) if n_nodes else 0.0
            return {
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "max_depth": max_depth,
                "n_roots": n_roots,
                "n_leaves": n_leaves,
                "avg_branching_factor": round(avg_branching, 4),
            }

    def __repr__(self) -> str:
        return (f"<LineageGraph n_nodes={len(self.nodes)} "
                f"n_edges={len(self.edges)}>")


# ============================================================================
# 4. LineageTracker
# ============================================================================
class LineageTracker:
    """Singleton-like recorder that auto-tracks model operations.

    Each ``record_*`` call adds a node (and one or more edges) to the
    underlying LineageGraph and persists it to disk at ``storage_path``.

    Args:
        storage_path: JSON file path for persistence (expanded with ~).
    """
    _instance: Optional["LineageTracker"] = None
    _instance_lock = threading.Lock()

    def __init__(self, storage_path: str = "~/.qai/lineage.json"):
        self.storage_path = os.path.expanduser(storage_path)
        self._graph = LineageGraph()
        self._lock = threading.RLock()
        self._load()

    # -- singleton -----------------------------------------------------------
    @classmethod
    def default(cls) -> "LineageTracker":
        """Return (creating if needed) the global default tracker."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def set_default(cls, tracker: "LineageTracker") -> None:
        with cls._instance_lock:
            cls._instance = tracker

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        if os.path.exists(self.storage_path):
            try:
                self._graph = LineageGraph.load(self.storage_path)
            except Exception as e:
                print(f"[qai.lineage] warning: failed to load lineage "
                      f"from {self.storage_path}: {e}", file=sys.stderr)
                self._graph = LineageGraph()

    def _save(self) -> None:
        try:
            self._graph.save(self.storage_path)
        except Exception as e:
            print(f"[qai.lineage] warning: failed to save lineage "
                  f"to {self.storage_path}: {e}", file=sys.stderr)

    # -- internal recorder ---------------------------------------------------
    def _record(self,
                model_id: Optional[str],
                name: Optional[str],
                parent_ids: Union[None, str, List[str]],
                operation: str,
                metadata: Optional[Dict[str, Any]]) -> LineageNode:
        with self._lock:
            parents_list = _normalize_parents(parent_ids)
            # for single-parent ops, store parent_id on the node; for
            # multi-parent (merge) or no-parent (create), store None
            parent_id_field = parents_list[0] if len(parents_list) == 1 else None

            node = LineageNode(
                model_id=model_id,
                name=name if name is not None else model_id,
                parent_id=parent_id_field,
                operation=operation,
                metadata=metadata,
            )
            self._graph.add_node(node)
            for pid in parents_list:
                if pid is not None and pid != node.model_id:
                    self._graph.add_edge(
                        LineageEdge(pid, node.model_id, operation)
                    )
            self._save()
            return node

    # -- public record_* API -------------------------------------------------
    def record_create(self, model_id: str, name: str,
                      metadata: Optional[Dict[str, Any]] = None) -> LineageNode:
        return self._record(model_id, name, None, "create", metadata)

    def record_train(self, model_id: str, parent_id: str,
                     metadata: Optional[Dict[str, Any]] = None) -> LineageNode:
        return self._record(model_id, None, parent_id, "train", metadata)

    def record_finetune(self, model_id: str, parent_id: str,
                        metadata: Optional[Dict[str, Any]] = None) -> LineageNode:
        return self._record(model_id, None, parent_id, "finetune", metadata)

    def record_quantize(self, model_id: str, parent_id: str,
                        format: str = "int8",
                        metadata: Optional[Dict[str, Any]] = None) -> LineageNode:
        meta = dict(metadata or {})
        meta.setdefault("format", format)
        return self._record(model_id, None, parent_id, "quantize", meta)

    def record_prune(self, model_id: str, parent_id: str,
                     sparsity: float = 0.5,
                     metadata: Optional[Dict[str, Any]] = None) -> LineageNode:
        meta = dict(metadata or {})
        meta.setdefault("sparsity", float(sparsity))
        return self._record(model_id, None, parent_id, "prune", meta)

    def record_distill(self, student_id: str, teacher_id: str,
                       metadata: Optional[Dict[str, Any]] = None) -> LineageNode:
        meta = dict(metadata or {})
        meta.setdefault("teacher_id", teacher_id)
        return self._record(student_id, None, teacher_id, "distill", meta)

    def record_convert(self, model_id: str, parent_id: str,
                       target_format: str = "onnx",
                       metadata: Optional[Dict[str, Any]] = None) -> LineageNode:
        meta = dict(metadata or {})
        meta.setdefault("target_format", target_format)
        return self._record(model_id, None, parent_id, "convert", meta)

    def record_merge(self, model_id: str, parent_ids: List[str],
                      strategy: str = "weighted_avg",
                      metadata: Optional[Dict[str, Any]] = None) -> LineageNode:
        meta = dict(metadata or {})
        meta.setdefault("strategy", strategy)
        meta.setdefault("parent_ids", list(parent_ids))
        return self._record(model_id, None, parent_ids, "merge", meta)

    def record_clone(self, model_id: str, parent_id: str,
                     metadata: Optional[Dict[str, Any]] = None) -> LineageNode:
        return self._record(model_id, None, parent_id, "clone", metadata)

    def record_deploy(self, model_id: str, parent_id: str,
                      target: str = "server",
                      metadata: Optional[Dict[str, Any]] = None) -> LineageNode:
        meta = dict(metadata or {})
        meta.setdefault("target", target)
        return self._record(model_id, None, parent_id, "deploy", meta)

    # -- queries -------------------------------------------------------------
    def graph(self) -> LineageGraph:
        """Return the current lineage graph."""
        with self._lock:
            return self._graph

    def history(self, model_id: str) -> List[LineageNode]:
        """All operations on this model (node + ancestors, time-sorted)."""
        with self._lock:
            if model_id not in self._graph.nodes:
                return []
            nodes = [self._graph.nodes[model_id]] + self._graph.ancestors(model_id)
            nodes.sort(key=lambda n: n.timestamp)
            return nodes

    def diff(self, node_a_id: str, node_b_id: str) -> Dict[str, Any]:
        """Compare two model versions by their lineage metadata."""
        with self._lock:
            if (node_a_id not in self._graph.nodes
                    or node_b_id not in self._graph.nodes):
                raise KeyError(
                    f"[WHAT] Cannot diff missing node(s): "
                    f"{node_a_id!r}, {node_b_id!r}\n"
                    f"[WHY] Both nodes must exist in the lineage graph\n"
                    f"[FIX] Record both nodes first, then diff"
                )
            a = self._graph.nodes[node_a_id]
            b = self._graph.nodes[node_b_id]
            keys = set(a.metadata.keys()) | set(b.metadata.keys())
            meta_diff: Dict[str, Dict[str, Any]] = {}
            for k in keys:
                va = a.metadata.get(k)
                vb = b.metadata.get(k)
                if va != vb:
                    meta_diff[k] = {"a": va, "b": vb}
            return {
                "node_a": node_a_id,
                "node_b": node_b_id,
                "name_a": a.name,
                "name_b": b.name,
                "operation_a": a.operation,
                "operation_b": b.operation,
                "operation_changed": a.operation != b.operation,
                "timestamp_a": a.timestamp,
                "timestamp_b": b.timestamp,
                "time_diff_seconds": b.timestamp - a.timestamp,
                "metadata_diff": meta_diff,
                "common_metadata_keys": sorted(set(a.metadata) & set(b.metadata)),
                "only_in_a": sorted(set(a.metadata) - set(b.metadata)),
                "only_in_b": sorted(set(b.metadata) - set(a.metadata)),
            }

    def clear(self) -> None:
        """Clear all tracked lineage."""
        with self._lock:
            self._graph = LineageGraph()
            self._save()

    def __repr__(self) -> str:
        return (f"<LineageTracker storage={self.storage_path!r} "
                f"n_nodes={len(self._graph.nodes)}>")


# ============================================================================
# 5. LineageVisualizer
# ============================================================================
class LineageVisualizer:
    """Renders a LineageGraph as Mermaid / DOT / D3 / HTML / JSON."""

    def to_mermaid(self, graph: LineageGraph) -> str:
        """Render graph as a Mermaid flowchart with color-coded nodes."""
        lines: List[str] = ["flowchart TD"]
        # class definitions
        for op, color in OPERATION_COLORS.items():
            lines.append(
                f"    classDef {op}Style "
                f"fill:{color},stroke:#333,color:white,stroke-width:2px"
            )
        # nodes
        for nid, node in graph.nodes.items():
            safe = _sanitize_id(nid)
            label = f"{node.name} [{node.operation}]"
            # escape double quotes in label
            label = label.replace('"', "'")
            lines.append(f'    {safe}["{label}"]:::{node.operation}Style')
        # edges
        for edge in graph.edges:
            src = _sanitize_id(edge.parent_id)
            dst = _sanitize_id(edge.child_id)
            label = edge.operation
            if edge.notes:
                label = f"{label} | {edge.notes}"
            lines.append(f"    {src} -->|{label}| {dst}")
        return "\n".join(lines)

    def to_dot(self, graph: LineageGraph) -> str:
        """Render graph as Graphviz DOT."""
        lines: List[str] = [
            "digraph lineage {",
            "  rankdir=TB;",
            "  node [shape=box, style=filled, fontname=Helvetica];",
            "  edge [fontname=Helvetica];",
        ]
        for nid, node in graph.nodes.items():
            color = OPERATION_COLORS.get(node.operation, "#CCCCCC")
            label = f"{node.name} [{node.operation}]"
            safe_id = nid.replace('"', '\\"')
            safe_label = label.replace('"', '\\"')
            lines.append(
                f'  "{safe_id}" [label="{safe_label}", '
                f'fillcolor="{color}", fontcolor="white"];'
            )
        for edge in graph.edges:
            src = edge.parent_id.replace('"', '\\"')
            dst = edge.child_id.replace('"', '\\"')
            lines.append(
                f'  "{src}" -> "{dst}" [label="{edge.operation}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    def to_d3(self, graph: LineageGraph) -> Dict[str, Any]:
        """Render graph as D3.js node-link data."""
        nodes: List[Dict[str, Any]] = []
        for nid, node in graph.nodes.items():
            nodes.append({
                "id": nid,
                "name": node.name,
                "operation": node.operation,
                "parent_id": node.parent_id,
                "color": OPERATION_COLORS.get(node.operation, "#CCCCCC"),
                "metadata": node.metadata,
                "timestamp": node.timestamp,
            })
        links: List[Dict[str, Any]] = []
        for edge in graph.edges:
            links.append({
                "source": edge.parent_id,
                "target": edge.child_id,
                "operation": edge.operation,
                "notes": edge.notes,
            })
        return {"nodes": nodes, "links": links}

    def to_html(self, graph: LineageGraph,
                title: str = "Model Lineage") -> str:
        """Render a self-contained HTML page with embedded Mermaid."""
        mermaid_src = self.to_mermaid(graph)
        stats = graph.stats()
        legend_items = "\n".join(
            f'  <span class="legend-item" '
            f'style="background:{color}">{op}</span>'
            for op, color in OPERATION_COLORS.items()
            if any(n.operation == op for n in graph.nodes.values())
        )
        # node table (for a richer page)
        node_rows = "\n".join(
            f"<tr><td>{n.model_id}</td><td>{n.name}</td>"
            f'<td><span class="legend-item" '
            f'style="background:{OPERATION_COLORS.get(n.operation, "#CCC")}">'
            f"{n.operation}</span></td>"
            f"<td>{n.parent_id or '-'}</td>"
            f"<td>{_now_iso()}</td></tr>"
            for n in graph.nodes.values()
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
          Roboto, sans-serif; margin: 24px; background: #fafafa;
          color: #333; }}
  h1 {{ margin-top: 0; }}
  .stats {{ background: white; padding: 14px 18px; border-radius: 8px;
           display: inline-block; margin-bottom: 16px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stats span {{ margin-right: 18px; color: #555; }}
  .stats b {{ color: #222; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }}
  .legend-item {{ display: inline-flex; align-items: center;
                 padding: 4px 10px; border-radius: 12px;
                 color: white; font-size: 12px; font-weight: bold; }}
  #graph {{ background: white; padding: 24px; border-radius: 8px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: auto; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px;
          background: white; border-radius: 8px; overflow: hidden;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ text-align: left; padding: 8px 12px;
           border-bottom: 1px solid #eee; font-size: 13px; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px;
         font-size: 12px; }}
  .mermaid {{ text-align: center; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="stats">
  <span><b>Nodes:</b> {stats['n_nodes']}</span>
  <span><b>Edges:</b> {stats['n_edges']}</span>
  <span><b>Roots:</b> {stats['n_roots']}</span>
  <span><b>Leaves:</b> {stats['n_leaves']}</span>
  <span><b>Max depth:</b> {stats['max_depth']}</span>
  <span><b>Avg branching:</b> {stats['avg_branching_factor']}</span>
</div>
<div class="legend">{legend_items}</div>
<div id="graph">
<pre class="mermaid">
{mermaid_src}
</pre>
</div>
<h2>Nodes</h2>
<table>
<thead><tr><th>model_id</th><th>name</th><th>operation</th>
<th>parent_id</th><th>timestamp</th></tr></thead>
<tbody>
{node_rows}
</tbody>
</table>
<script>
  mermaid.initialize({{ startOnLoad: true, securityLevel: 'loose',
                       theme: 'default', flowchart: {{ useMaxWidth: true }} }});
</script>
</body>
</html>"""

    def to_json(self, graph: LineageGraph) -> str:
        """Render graph as JSON (suitable for external tools)."""
        return json.dumps(graph.to_dict(), indent=2, default=str)


# ============================================================================
# 6. Top-level convenience API
# ============================================================================
def _default_tracker() -> LineageTracker:
    return LineageTracker.default()


def track(model: Any, operation: str = "create", **metadata) -> LineageNode:
    """Auto-track a model operation.

    Args:
        model:   any object (we look for .model_id / .name attributes,
                 falling back to type(model).__name__).
        operation: one of the valid operations.
        **metadata: extra metadata + operation-specific kwargs
                    (parent_id, format, sparsity, target, ...).
    """
    t = _default_tracker()
    # try to derive model_id / name from the model object
    model_id = (metadata.pop("model_id", None)
                or getattr(model, "model_id", None)
                or getattr(model, "id", None))
    name = (metadata.pop("name", None)
            or getattr(model, "name", None)
            or type(model).__name__)
    parent_id = metadata.pop("parent_id", None)
    meta = dict(metadata)

    if operation == "create":
        return t.record_create(model_id, name, metadata=meta or None)
    if operation == "train":
        return t.record_train(model_id, parent_id, metadata=meta or None)
    if operation == "finetune":
        return t.record_finetune(model_id, parent_id, metadata=meta or None)
    if operation == "quantize":
        fmt = meta.pop("format", "int8")
        return t.record_quantize(model_id, parent_id, format=fmt,
                                 metadata=meta or None)
    if operation == "prune":
        sparsity = meta.pop("sparsity", 0.5)
        return t.record_prune(model_id, parent_id, sparsity=sparsity,
                              metadata=meta or None)
    if operation == "distill":
        teacher_id = meta.pop("teacher_id", parent_id)
        return t.record_distill(model_id, teacher_id,
                                metadata=meta or None)
    if operation == "convert":
        tf = meta.pop("target_format", "onnx")
        return t.record_convert(model_id, parent_id, target_format=tf,
                                 metadata=meta or None)
    if operation == "merge":
        parent_ids = meta.pop("parent_ids",
                              [parent_id] if parent_id else [])
        strat = meta.pop("strategy", "weighted_avg")
        return t.record_merge(model_id, parent_ids, strategy=strat,
                              metadata=meta or None)
    if operation == "clone":
        return t.record_clone(model_id, parent_id, metadata=meta or None)
    if operation == "deploy":
        target = meta.pop("target", "server")
        return t.record_deploy(model_id, parent_id, target=target,
                               metadata=meta or None)
    raise ValueError(
        f"[WHAT] Unknown operation: {operation!r}\n"
        f"[WHY] Valid operations: {list(VALID_OPERATIONS)}\n"
        f"[FIX] Use one of the supported operations"
    )


def history(model_id: str) -> List[LineageNode]:
    """Get the full history for a model_id (node + all ancestors)."""
    return _default_tracker().history(model_id)


def visualize(model_id: Optional[str] = None,
              format: str = "html") -> str:
    """Visualize lineage as the requested format.

    Args:
        model_id: if given, only show the sub-lineage rooted around this
                  model (ancestors + descendants + node itself);
                  otherwise show the full graph.
        format:   one of 'html', 'mermaid', 'dot', 'd3', 'json'.
    """
    g = _default_tracker().graph()
    if model_id is not None and model_id in g.nodes:
        # build a subgraph
        sub = LineageGraph()
        relevant = (
            [model_id]
            + [n.model_id for n in g.ancestors(model_id)]
            + [n.model_id for n in g.descendants(model_id)]
        )
        for nid in relevant:
            if nid in g.nodes:
                sub.add_node(g.nodes[nid])
        for edge in g.edges:
            if (edge.parent_id in sub.nodes
                    and edge.child_id in sub.nodes):
                sub.add_edge(edge)
        g = sub

    viz = LineageVisualizer()
    fmt = format.lower()
    if fmt == "mermaid":
        return viz.to_mermaid(g)
    if fmt == "dot":
        return viz.to_dot(g)
    if fmt == "d3":
        return json.dumps(viz.to_d3(g), indent=2, default=str)
    if fmt == "json":
        return viz.to_json(g)
    if fmt == "html":
        return viz.to_html(g)
    raise ValueError(
        f"[WHAT] Unknown format: {format!r}\n"
        f"[WHY] Valid formats: html, mermaid, dot, d3, json\n"
        f"[FIX] Use one of the supported formats"
    )


def export(path: str) -> str:
    """Export the full lineage graph as HTML to the given path."""
    html = visualize(None, format="html")
    path = os.path.expanduser(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    return path


def clear() -> None:
    """Clear all tracked lineage."""
    _default_tracker().clear()


__all__ = [
    "LineageNode",
    "LineageEdge",
    "LineageGraph",
    "LineageTracker",
    "LineageVisualizer",
    "VALID_OPERATIONS",
    "OPERATION_COLORS",
    "track",
    "history",
    "visualize",
    "export",
    "clear",
]
