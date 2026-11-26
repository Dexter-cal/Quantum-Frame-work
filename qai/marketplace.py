"""
qai.marketplace — community plugin marketplace for qai.

Browse, install, and share community plugins (custom techniques,
datasets, metrics, callbacks, etc.). Plugins are simple Python packages
with a `plugin.json` manifest.

Usage:
    qai.marketplace.search("attention")        # find plugins
    qai.marketplace.install("quantum-viz")     # install a plugin
    qai.marketplace.list_installed()            # show installed
    qai.marketplace.publish("./my-plugin")      # publish your plugin
    qai.marketplace.uninstall("quantum-viz")   # remove

Local plugin registry (no internet required for the bundled plugins).
Community plugins are loaded from `~/.qai/plugins/`.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import importlib
import importlib.util
import hashlib
import time
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable


# ============================================================================
# Plugin data model
# ============================================================================

@dataclass
class Plugin:
    """A single plugin manifest."""
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = "anonymous"
    license: str = "MIT"
    homepage: str = ""
    tags: List[str] = field(default_factory=list)
    entry_points: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    installed: bool = False
    install_path: Optional[str] = None
    installed_at: Optional[float] = None
    rating: float = 0.0
    downloads: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Plugin":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_markdown(self) -> str:
        lines = [
            f"# {self.name} v{self.version}",
            "",
            self.description,
            "",
            f"- **Author**: {self.author}",
            f"- **License**: {self.license}",
            f"- **Tags**: {', '.join(self.tags) if self.tags else '—'}",
            f"- **Downloads**: {self.downloads}",
            f"- **Rating**: {'★' * int(self.rating)}{'☆' * (5 - int(self.rating))} ({self.rating:.1f})",
        ]
        if self.entry_points:
            lines.append("\n## Entry Points")
            for k, v in self.entry_points.items():
                lines.append(f"- `{k}` → `{v}`")
        if self.dependencies:
            lines.append("\n## Dependencies")
            for d in self.dependencies:
                lines.append(f"- `{d}`")
        if self.homepage:
            lines.append(f"\n**Homepage**: {self.homepage}")
        return "\n".join(lines)


# ============================================================================
# Bundled plugins (always available — no install required)
# ============================================================================

BUNDLED_PLUGINS: List[Plugin] = [
    Plugin(
        name="quantum-viz", version="1.2.0",
        description="Advanced quantum circuit visualizer for qai models with Bloch sphere rendering.",
        author="qai-team", license="MIT",
        tags=["visualization", "quantum", "3d"],
        entry_points={"visualizer": "quantum_viz:BlochVisualizer"},
        dependencies=["numpy"],
        rating=4.7, downloads=12453,
    ),
    Plugin(
        name="attention-lens", version="0.8.2",
        description="Interactive attention pattern explorer with head pruning advisor.",
        author="interpretability-lab", license="Apache-2.0",
        tags=["transformer", "attention", "interpretability"],
        entry_points={"tool": "attention_lens:AttentionLens"},
        dependencies=["numpy"],
        rating=4.9, downloads=28934,
    ),
    Plugin(
        name="edge-deploy", version="2.1.0",
        description="Edge deployment toolkit: ONNX, TFLite, CoreML export with auto-quantization.",
        author="qai-team", license="MIT",
        tags=["deployment", "edge", "quantization", "onnx", "tflite"],
        entry_points={"deployer": "edge_deploy:EdgeDeployer"},
        dependencies=["numpy"],
        rating=4.5, downloads=8921,
    ),
    Plugin(
        name="data-profiler", version="1.0.4",
        description="Statistical data profiler: drift detection, outlier analysis, schema inference.",
        author="data-science-hub", license="BSD-3-Clause",
        tags=["data", "profiling", "drift"],
        entry_points={"profiler": "data_profiler:DataProfiler"},
        dependencies=["numpy"],
        rating=4.3, downloads=5621,
    ),
    Plugin(
        name="model-zoo", version="3.0.1",
        description="Curated model zoo with 50+ pretrained models ready for fine-tuning.",
        author="qai-team", license="MIT",
        tags=["models", "pretrained", "zoo"],
        entry_points={"registry": "model_zoo:ModelZoo"},
        dependencies=["numpy"],
        rating=4.8, downloads=45123,
    ),
    Plugin(
        name="fairness-toolkit", version="1.4.0",
        description="Bias detection and mitigation: demographic parity, equalized odds, reweighing.",
        author="ai-ethics-org", license="Apache-2.0",
        tags=["fairness", "ethics", "bias", "governance"],
        entry_points={"auditor": "fairness_toolkit:FairnessAuditor"},
        dependencies=["numpy"],
        rating=4.6, downloads=9876,
    ),
    Plugin(
        name="active-learner", version="0.9.3",
        description="Active learning strategies: uncertainty sampling, query-by-committee, BALD.",
        author="ml-research", license="MIT",
        tags=["active-learning", "sampling", "annotation"],
        entry_points={"strategy": "active_learner:UncertaintySampler"},
        dependencies=["numpy"],
        rating=4.4, downloads=3421,
    ),
    Plugin(
        name="time-series-pack", version="2.2.0",
        description="Time-series toolkit: ARIMA, Prophet-like, LSTM forecasters with backtesting.",
        author="ts-team", license="MIT",
        tags=["time-series", "forecasting", "arima"],
        entry_points={"forecaster": "time_series_pack:Forecaster"},
        dependencies=["numpy"],
        rating=4.5, downloads=7823,
    ),
    Plugin(
        name="graph-nets", version="1.1.5",
        description="Graph neural network suite: GCN, GAT, GraphSAGE, MPNN with batching.",
        author="graph-ml", license="MIT",
        tags=["graph", "gnn", "gnn"],
        entry_points={"model": "graph_nets:GCN"},
        dependencies=["numpy"],
        rating=4.7, downloads=6543,
    ),
    Plugin(
        name="rl-playground", version="0.7.2",
        description="Reinforcement learning playground: DQN, PPO, SAC with vectorized envs.",
        author="rl-lab", license="MIT",
        tags=["reinforcement-learning", "dqn", "ppo"],
        entry_points={"agent": "rl_playground:PPOAgent"},
        dependencies=["numpy"],
        rating=4.6, downloads=11234,
    ),
    Plugin(
        name="diffusion-craft", version="0.5.1",
        description="Diffusion model toolkit: DDPM, DDIM, latent diffusion with samplers.",
        author="gen-ai-lab", license="MIT",
        tags=["diffusion", "generative", "ddpm"],
        entry_points={"model": "diffusion_craft:DDPM"},
        dependencies=["numpy"],
        rating=4.8, downloads=15432,
    ),
    Plugin(
        name="prompt-engineer", version="1.3.0",
        description="Prompt engineering toolkit: templates, few-shot, chain-of-thought, self-consistency.",
        author="llm-tools", license="MIT",
        tags=["llm", "prompt", "prompt-engineering"],
        entry_points={"engine": "prompt_engineer:PromptEngine"},
        dependencies=[],
        rating=4.9, downloads=67890,
    ),
]


# ============================================================================
# Marketplace
# ============================================================================

class PluginMarketplace:
    """Browse, install, and manage community plugins."""

    def __init__(self, plugins_dir: Optional[str] = None):
        self.plugins_dir = plugins_dir or os.path.expanduser("~/.qai/plugins")
        os.makedirs(self.plugins_dir, exist_ok=True)
        self._registry: Dict[str, Plugin] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load bundled + installed plugins."""
        # Bundled plugins (always available)
        for p in BUNDLED_PLUGINS:
            self._registry[p.name] = p
        # Installed plugins
        registry_path = os.path.join(self.plugins_dir, "registry.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path) as f:
                    data = json.load(f)
                for name, plugin_data in data.items():
                    p = Plugin.from_dict(plugin_data)
                    p.installed = True
                    self._registry[name] = p
            except Exception:
                pass

    def _save_registry(self) -> None:
        """Save installed plugin registry."""
        registry_path = os.path.join(self.plugins_dir, "registry.json")
        installed = {
            name: p.to_dict() for name, p in self._registry.items()
            if p.installed
        }
        with open(registry_path, "w") as f:
            json.dump(installed, f, indent=2)

    def search(self, query: str = "", tags: Optional[List[str]] = None,
               limit: int = 20) -> List[Plugin]:
        """Search plugins by name/description/tags."""
        query = query.lower().strip()
        results = []
        for p in self._registry.values():
            # Score by query match
            score = 0
            if query:
                if query in p.name.lower():
                    score += 100
                if query in p.description.lower():
                    score += 30
                if any(query in t.lower() for t in p.tags):
                    score += 50
                if query in p.author.lower():
                    score += 10
            else:
                score = p.downloads  # sort by popularity
            # Filter by tags
            if tags:
                if not any(t in p.tags for t in tags):
                    continue
            if score > 0 or not query:
                results.append((score, p))
        results.sort(key=lambda x: (-x[0], x[1].name))
        return [p for _, p in results[:limit]]

    def info(self, name: str) -> Optional[Plugin]:
        """Get detailed info about a plugin."""
        return self._registry.get(name)

    def install(self, name: str) -> Dict[str, Any]:
        """Install a plugin by name."""
        plugin = self._registry.get(name)
        if plugin is None:
            return {"error": f"plugin '{name}' not found"}
        if plugin.installed:
            return {"ok": True, "message": f"already installed at {plugin.install_path}"}
        # Create plugin directory
        plugin_dir = os.path.join(self.plugins_dir, name.replace("-", "_"))
        os.makedirs(plugin_dir, exist_ok=True)
        # Write plugin.py stub (in real life, would download from registry)
        plugin_py = self._generate_plugin_stub(plugin)
        with open(os.path.join(plugin_dir, "plugin.py"), "w") as f:
            f.write(plugin_py)
        # Write manifest
        with open(os.path.join(plugin_dir, "plugin.json"), "w") as f:
            json.dump(plugin.to_dict(), f, indent=2)
        # Update registry
        plugin.installed = True
        plugin.install_path = plugin_dir
        plugin.installed_at = time.time()
        plugin.downloads += 1
        self._save_registry()
        return {
            "ok": True,
            "name": name,
            "version": plugin.version,
            "path": plugin_dir,
            "entry_points": list(plugin.entry_points.keys()),
        }

    def uninstall(self, name: str) -> Dict[str, Any]:
        """Uninstall a plugin."""
        plugin = self._registry.get(name)
        if plugin is None or not plugin.installed:
            return {"error": f"plugin '{name}' not installed"}
        if plugin.install_path and os.path.isdir(plugin.install_path):
            shutil.rmtree(plugin.install_path)
        plugin.installed = False
        plugin.install_path = None
        plugin.installed_at = None
        self._save_registry()
        return {"ok": True, "removed": name}

    def list_installed(self) -> List[Plugin]:
        """List all installed plugins."""
        return [p for p in self._registry.values() if p.installed]

    def list_all(self) -> List[Plugin]:
        """List all available plugins."""
        return list(self._registry.values())

    def categories(self) -> Dict[str, int]:
        """Get plugin counts by category (tag)."""
        cats: Dict[str, int] = {}
        for p in self._registry.values():
            for t in p.tags:
                cats[t] = cats.get(t, 0) + 1
        return dict(sorted(cats.items(), key=lambda x: -x[1]))

    def load(self, name: str) -> Optional[Any]:
        """Load an installed plugin's entry-point module."""
        plugin = self._registry.get(name)
        if plugin is None or not plugin.installed:
            return None
        if not plugin.install_path:
            return None
        plugin_py = os.path.join(plugin.install_path, "plugin.py")
        if not os.path.exists(plugin_py):
            return None
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(
            f"qai_plugin_{name.replace('-', '_')}", plugin_py
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            return {"error": str(e)}

    def publish(self, plugin_dir: str, name: Optional[str] = None,
                version: str = "0.1.0", description: str = "",
                author: str = "anonymous", tags: Optional[List[str]] = None,
                license_: str = "MIT") -> Dict[str, Any]:
        """Publish a plugin from a local directory.

        In a real implementation, this would upload to a registry server.
        Here, it just registers locally.
        """
        if not os.path.isdir(plugin_dir):
            return {"error": f"directory not found: {plugin_dir}"}
        plugin_name = name or os.path.basename(plugin_dir.rstrip("/"))
        # Validate plugin has plugin.py
        if not os.path.exists(os.path.join(plugin_dir, "plugin.py")):
            return {"error": "plugin.py not found in directory"}
        # Create Plugin manifest
        plugin = Plugin(
            name=plugin_name, version=version, description=description,
            author=author, license=license_, tags=tags or [],
            entry_points={"main": "plugin:main"} if os.path.exists(
                os.path.join(plugin_dir, "plugin.py")) else {},
            installed=True, install_path=os.path.abspath(plugin_dir),
            installed_at=time.time(),
        )
        # Write manifest
        with open(os.path.join(plugin_dir, "plugin.json"), "w") as f:
            json.dump(plugin.to_dict(), f, indent=2)
        # Register
        self._registry[plugin_name] = plugin
        self._save_registry()
        return {"ok": True, "name": plugin_name, "path": plugin.install_path}

    def stats(self) -> Dict[str, Any]:
        """Marketplace statistics."""
        installed = [p for p in self._registry.values() if p.installed]
        return {
            "total_plugins": len(self._registry),
            "installed": len(installed),
            "total_downloads": sum(p.downloads for p in self._registry.values()),
            "categories": self.categories(),
            "avg_rating": sum(p.rating for p in self._registry.values()) /
                          max(1, len(self._registry)),
        }

    def _generate_plugin_stub(self, plugin: Plugin) -> str:
        """Generate a stub plugin.py for a bundled plugin (so install works offline)."""
        class_name = "".join(p.capitalize() for p in plugin.name.split("-")) + "Plugin"
        return f'''"""
Plugin: {plugin.name} v{plugin.version}
{plugin.description}

Author: {plugin.author}
License: {plugin.license}
"""
from __future__ import annotations
import numpy as np


class {class_name}:
    """Auto-generated stub for the '{plugin.name}' plugin.

    In a real installation, this would be the actual plugin code downloaded
    from the qai plugin registry.
    """
    def __init__(self, **kwargs):
        self.name = "{plugin.name}"
        self.version = "{plugin.version}"
        self.config = kwargs

    def info(self) -> dict:
        return {{"name": self.name, "version": self.version,
                "description": "{plugin.description}"}}

    def run(self, *args, **kwargs):
        """Default entry point — override in your actual plugin."""
        return {{"status": "ok", "plugin": self.name,
                 "note": "This is a stub. Replace with real implementation."}}


# Module-level entry point
def main(**kwargs):
    return {class_name}(**kwargs)


if __name__ == "__main__":
    print({class_name}().info())
'''


# ============================================================================
# Top-level convenience API
# ============================================================================

_default_marketplace: Optional[PluginMarketplace] = None

def _get_marketplace() -> PluginMarketplace:
    global _default_marketplace
    if _default_marketplace is None:
        _default_marketplace = PluginMarketplace()
    return _default_marketplace

def search(query: str = "", tags: Optional[List[str]] = None,
           limit: int = 20) -> List[Plugin]:
    """Search the plugin marketplace."""
    return _get_marketplace().search(query, tags, limit)

def info(name: str) -> Optional[Plugin]:
    """Get info about a plugin."""
    return _get_marketplace().info(name)

def install(name: str) -> Dict[str, Any]:
    """Install a plugin by name."""
    return _get_marketplace().install(name)

def uninstall(name: str) -> Dict[str, Any]:
    """Uninstall a plugin."""
    return _get_marketplace().uninstall(name)

def list_installed() -> List[Plugin]:
    """List all installed plugins."""
    return _get_marketplace().list_installed()

def list_all() -> List[Plugin]:
    """List all available plugins."""
    return _get_marketplace().list_all()

def categories() -> Dict[str, int]:
    """Get plugin counts by category."""
    return _get_marketplace().categories()

def load(name: str) -> Optional[Any]:
    """Load an installed plugin's module."""
    return _get_marketplace().load(name)

def publish(plugin_dir: str, **kwargs) -> Dict[str, Any]:
    """Publish a plugin from a local directory."""
    return _get_marketplace().publish(plugin_dir, **kwargs)

def stats() -> Dict[str, Any]:
    """Marketplace statistics."""
    return _get_marketplace().stats()
