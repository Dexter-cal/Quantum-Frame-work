"""
qai.versioning — git-like versioning for ML experiments.

Tracks the full provenance of an ML experiment:
  * Code version (git SHA / file hashes)
  * Data version (content hash of dataset)
  * Model version (architecture + weight hash)
  * Hyperparameters (exact config)
  * Environment (Python version, library versions)
  * Results (metrics, plots, logs)

Each "commit" is a snapshot of all of the above. You can:
  * `commit(model, X, y, message)` — record a checkpoint
  * `log()` — view commit history
  * `checkout(commit_id)` — restore a previous state
  * `diff(commit_a, commit_b)` — compare two commits
  * `branch(name)` — create a branch for experiments
  * `tag(name, commit_id)` — tag a release

Usage:
    import qai
    repo = qai.versioning.init('my_experiment')
    m = qai.build(technique='neural_network', hidden_layer_sizes=(8, 6))
    X, y = qai.load_iris()
    m.train(X, y)
    repo.commit(m, X, y, message='initial training', params={'lr': 0.01})
    print(repo.log())
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import time
import sys
import platform
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Commit:
    """A single versioning commit."""
    commit_id: str = ""
    parent_id: Optional[str] = None
    message: str = ""
    timestamp: float = 0.0
    author: str = "anonymous"
    branch: str = "main"
    tags: List[str] = field(default_factory=list)
    # Content hashes
    model_hash: str = ""
    data_hash: str = ""
    code_hash: str = ""
    env_hash: str = ""
    # Snapshot
    model_path: Optional[str] = None  # path to pickled model
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    data_shape: Tuple[int, ...] = (0,)
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["data_shape"] = list(self.data_shape)
        return d

    def to_markdown(self) -> str:
        lines = [
            f"### Commit {self.commit_id[:8]}",
            "",
            f"**{self.message}**",
            "",
            f"- **Author**: {self.author}",
            f"- **Branch**: {self.branch}",
            f"- **Time**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"- **Parent**: {self.parent_id[:8] if self.parent_id else '—'}",
            f"- **Model hash**: {self.model_hash[:16]}…",
            f"- **Data hash**: {self.data_hash[:16]}…",
            f"- **Samples**: {self.n_samples}",
            f"- **Metrics**: {', '.join(f'{k}={v:.4f}' for k, v in self.metrics.items()) or '—'}",
        ]
        if self.params:
            lines.append(f"- **Params**: {self.params}")
        if self.tags:
            lines.append(f"- **Tags**: {', '.join(self.tags)}")
        return "\n".join(lines)


# ============================================================================
# Versioning repository
# ============================================================================

class VersionRepo:
    """Git-like versioning for ML experiments."""

    def __init__(self, name: str, storage_path: Optional[str] = None):
        self.name = name
        self.storage_path = storage_path or os.path.expanduser(f"~/.qai/versions/{name}")
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "models"), exist_ok=True)
        self.commits: Dict[str, Commit] = {}  # commit_id → Commit
        self.branches: Dict[str, str] = {"main": None}  # branch → head commit_id
        self.current_branch: str = "main"
        self.tags: Dict[str, str] = {}  # tag_name → commit_id
        self._load()

    # -- Core operations -----------------------------------------------
    def commit(self, model: Any, X: Any = None, y: Any = None,
               message: str = "", params: Optional[Dict] = None,
               metrics: Optional[Dict] = None, author: str = "anonymous") -> Commit:
        """Create a new commit."""
        # Hash content
        model_hash = self._hash_model(model)
        data_hash = self._hash_data(X, y)
        code_hash = self._hash_code()
        env_hash = self._hash_env()
        # Save model
        model_path = None
        if model is not None:
            model_path = os.path.join(self.storage_path, "models",
                                       f"{model_hash[:12]}.pkl")
            try:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            except Exception:
                model_path = None
        # Build commit
        commit_id = self._make_id(model_hash + data_hash + str(time.time()))
        parent_id = self.branches.get(self.current_branch)
        commit = Commit(
            commit_id=commit_id,
            parent_id=parent_id,
            message=message,
            timestamp=time.time(),
            author=author,
            branch=self.current_branch,
            model_hash=model_hash,
            data_hash=data_hash,
            code_hash=code_hash,
            env_hash=env_hash,
            model_path=model_path,
            params=params or {},
            metrics=metrics or {},
            environment=self._capture_env(),
            data_shape=np.asarray(X).shape if X is not None else (0,),
            n_samples=len(X) if X is not None else 0,
        )
        self.commits[commit_id] = commit
        self.branches[self.current_branch] = commit_id
        self._save()
        return commit

    def checkout(self, commit_id: str) -> Optional[Any]:
        """Restore a model from a commit."""
        commit = self.commits.get(commit_id)
        if commit is None or commit.model_path is None:
            return None
        try:
            with open(commit.model_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def diff(self, commit_a: str, commit_b: str) -> Dict[str, Any]:
        """Compare two commits."""
        a = self.commits.get(commit_a, Commit())
        b = self.commits.get(commit_b, Commit())
        return {
            "model_changed": a.model_hash != b.model_hash,
            "data_changed": a.data_hash != b.data_hash,
            "code_changed": a.code_hash != b.code_hash,
            "env_changed": a.env_hash != b.env_hash,
            "params_diff": self._dict_diff(a.params, b.params),
            "metrics_diff": self._dict_diff(a.metrics, b.metrics),
            "n_samples_diff": b.n_samples - a.n_samples,
            "time_diff_s": b.timestamp - a.timestamp,
        }

    def log(self, n: int = 20, branch: Optional[str] = None) -> List[Commit]:
        """View commit history."""
        commits = list(self.commits.values())
        if branch:
            commits = [c for c in commits if c.branch == branch]
        commits.sort(key=lambda c: -c.timestamp)
        return commits[:n]

    def log_markdown(self, n: int = 20) -> str:
        """Markdown commit log."""
        commits = self.log(n)
        lines = [f"# Version Log: {self.name}", "",
                 "| Commit | Message | Branch | Time | Metrics |",
                 "|--------|---------|--------|------|---------|"]
        for c in commits:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(c.timestamp))
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in c.metrics.items()) or "—"
            lines.append(f"| {c.commit_id[:8]} | {c.message} | {c.branch} | {ts} | {metrics_str} |")
        return "\n".join(lines)

    # -- Branches & tags -----------------------------------------------
    def branch(self, name: str, from_commit: Optional[str] = None) -> bool:
        """Create a new branch."""
        if name in self.branches:
            return False
        parent = from_commit or self.branches.get(self.current_branch)
        self.branches[name] = parent
        self._save()
        return True

    def checkout_branch(self, name: str) -> bool:
        """Switch to a branch."""
        if name not in self.branches:
            return False
        self.current_branch = name
        return True

    def tag(self, name: str, commit_id: Optional[str] = None) -> bool:
        """Tag a commit."""
        target = commit_id or self.branches.get(self.current_branch)
        if target is None or target not in self.commits:
            return False
        self.tags[name] = target
        self.commits[target].tags.append(name)
        self._save()
        return True

    def list_branches(self) -> Dict[str, Optional[str]]:
        return dict(self.branches)

    def list_tags(self) -> Dict[str, str]:
        return dict(self.tags)

    # -- Stats ---------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_commits": len(self.commits),
            "n_branches": len(self.branches),
            "n_tags": len(self.tags),
            "current_branch": self.current_branch,
            "storage_mb": self._dir_size(self.storage_path) / (1024 * 1024),
        }

    # -- Persistence ---------------------------------------------------
    def _save(self) -> None:
        meta = {
            "name": self.name,
            "commits": {cid: c.to_dict() for cid, c in self.commits.items()},
            "branches": self.branches,
            "current_branch": self.current_branch,
            "tags": self.tags,
        }
        with open(os.path.join(self.storage_path, "repo.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def _load(self) -> None:
        path = os.path.join(self.storage_path, "repo.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                meta = json.load(f)
            self.name = meta.get("name", self.name)
            self.commits = {cid: Commit(**{k: v for k, v in d.items() if k in Commit.__dataclass_fields__})
                            for cid, d in meta.get("commits", {}).items()}
            self.branches = meta.get("branches", {"main": None})
            self.current_branch = meta.get("current_branch", "main")
            self.tags = meta.get("tags", {})
        except Exception:
            pass

    # -- Hashing helpers -----------------------------------------------
    @staticmethod
    def _make_id(content: str) -> str:
        return hashlib.sha1(content.encode()).hexdigest()[:16]

    def _hash_model(self, model: Any) -> str:
        try:
            from qai.visual_workspace_v2 import _get_weights
            layers = _get_weights(model)
            content = json.dumps([{k: v for k, v in L.items() if k != "weights"} for L in layers],
                                  default=str)
            return hashlib.sha1(content.encode()).hexdigest()
        except Exception:
            return hashlib.sha1(str(model).encode()).hexdigest()

    def _hash_data(self, X: Any, y: Any = None) -> str:
        if X is None:
            return "none"
        try:
            arr = np.asarray(X)
            content = f"shape={arr.shape},sum={float(np.sum(arr))},std={float(np.std(arr))}"
            if y is not None:
                y_arr = np.asarray(y)
                content += f",y_shape={y_arr.shape},y_sum={float(np.sum(y_arr))}"
            return hashlib.sha1(content.encode()).hexdigest()
        except Exception:
            return "unknown"

    def _hash_code(self) -> str:
        """Hash of currently-running script (if any)."""
        try:
            main = sys.modules.get("__main__")
            if main and hasattr(main, "__file__"):
                with open(main.__file__) as f:
                    return hashlib.sha1(f.read().encode()).hexdigest()
        except Exception:
            pass
        return "interactive"

    def _hash_env(self) -> str:
        env = self._capture_env()
        return hashlib.sha1(json.dumps(env, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _capture_env() -> Dict[str, str]:
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        }

    @staticmethod
    def _dict_diff(a: Dict, b: Dict) -> Dict[str, Any]:
        """Diff two dicts."""
        diff = {}
        for k in set(a.keys()) | set(b.keys()):
            if a.get(k) != b.get(k):
                diff[k] = {"a": a.get(k), "b": b.get(k)}
        return diff

    @staticmethod
    def _dir_size(path: str) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except Exception:
                    pass
        return total


# ============================================================================
# Top-level convenience
# ============================================================================

_repos: Dict[str, VersionRepo] = {}

def init(name: str, storage_path: Optional[str] = None) -> VersionRepo:
    """Initialize a versioning repository."""
    if name not in _repos:
        _repos[name] = VersionRepo(name, storage_path)
    return _repos[name]

def commit(repo_name: str, model: Any, X: Any = None, y: Any = None,
           message: str = "", **kwargs) -> Commit:
    """Create a commit in a repo."""
    repo = _repos.get(repo_name) or init(repo_name)
    return repo.commit(model, X, y, message=message, **kwargs)

def log(repo_name: str, n: int = 20) -> List[Commit]:
    """View commit log."""
    repo = _repos.get(repo_name) or init(repo_name)
    return repo.log(n)

def checkout(repo_name: str, commit_id: str) -> Optional[Any]:
    """Checkout a commit."""
    repo = _repos.get(repo_name) or init(repo_name)
    return repo.checkout(commit_id)

def diff(repo_name: str, commit_a: str, commit_b: str) -> Dict[str, Any]:
    """Diff two commits."""
    repo = _repos.get(repo_name) or init(repo_name)
    return repo.diff(commit_a, commit_b)
