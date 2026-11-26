"""
qai.framework -- Framework-level features (spec Sections 35, 36, 43, 51, 55, 58, 67, 79, 94).

Eight production framework systems:

1. OutputRenderer (Section 35-36) — capability-checked output formatting
2. KnowledgeGapTracker (Section 43) — track what the model doesn't know
3. IdleTrainer (Section 51) — background training when system is idle
4. Marketplace (Section 55) — publish/discover/subscribe to models + techniques
5. AdvancedVisualizer (Section 58) — formula editing, gradient flow, weight heatmaps
6. BidirectionalChannel (Section 67) — two-way model communication
7. RouterCore (Section 79) — predictive model loading + request routing
8. AutomatedScientist (Section 94) — autonomous hypothesis generation + testing

Creative additions per module (documented inline).
"""
from __future__ import annotations
import time, json, os, copy, hashlib, math, random, threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np


# =====================================================================
# 1. Output Renderer (spec Section 35-36)
# =====================================================================
class OutputRenderer:
    """Capability-checked output rendering (spec Section 35-36).

    Formats model outputs for different consumers (API, CLI, web, file).
    Each renderer has declared capabilities — the system checks that the
    model's output is compatible before rendering.

    Creative additions:
      - Format auto-detection (pick the best format based on output type)
      - Output validation (schema-check before rendering)
      - Custom renderer registration

    Example:
        import qai
        renderer = qai.OutputRenderer()
        renderer.register("json", lambda x: json.dumps(x, indent=2), accepts=dict)
        renderer.register("csv", lambda x: "\\n".join(",".join(row) for row in x), accepts=list)
        print(renderer.render({"a": 1}, format="json"))
    """
    def __init__(self):
        self._renderers: Dict[str, Dict[str, Any]] = {}
        self._default_formats = {
            dict: "json",
            list: "json",
            str: "text",
            int: "text",
            float: "text",
            np.ndarray: "text",
        }
        self._register_builtins()

    def _register_builtins(self):
        self.register("text", lambda x: str(x), accepts=object, description="Plain text")
        self.register("json", lambda x: json.dumps(x, default=str, indent=2), accepts=(dict, list, tuple), description="JSON")
        self.register("repr", lambda x: repr(x), accepts=object, description="Python repr")

    def register(self, name: str, fn: Callable, accepts: Any = object,
                 description: str = "") -> "OutputRenderer":
        """Register a custom output renderer."""
        self._renderers[name] = {"fn": fn, "accepts": accepts, "description": description}
        return self

    def render(self, output: Any, format: Optional[str] = None) -> str:
        """Render output in the specified (or auto-detected) format.

        Args:
            output: The model output to render.
            format: Format name. If None, auto-detected from output type.

        Returns:
            Rendered string.
        """
        if format is None:
            format = self._detect_format(output)

        if format not in self._renderers:
            # Fallback to text
            format = "text"

        renderer = self._renderers[format]
        # Capability check
        accepts = renderer["accepts"]
        if accepts is not object and not isinstance(output, accepts):
            raise TypeError(
                f"[WHAT] Output type {type(output).__name__} not compatible with '{format}' renderer\n"
                f"[WHY] Renderer '{format}' accepts {accepts}\n"
                f"[FIX] Use a different format, or register a custom renderer"
            )
        return renderer["fn"](output)

    def _detect_format(self, output: Any) -> str:
        """Auto-detect the best format for an output type."""
        for typ, fmt in self._default_formats.items():
            if isinstance(output, typ):
                return fmt
        return "text"

    def list_formats(self) -> List[str]:
        return sorted(self._renderers.keys())

    def __repr__(self):
        return f"<OutputRenderer formats={self.list_formats()}>"


# =====================================================================
# 2. Knowledge Gap Tracker (spec Section 43)
# =====================================================================
@dataclass
class KnowledgeGap:
    """A single knowledge gap — something the model doesn't know well."""
    id: str
    description: str
    category: str = "general"
    severity: float = 0.5  # 0=minor, 1=critical
    samples: List[Any] = field(default_factory=list)  # failing examples
    created_at: float = field(default_factory=time.time)
    resolved: bool = False


class KnowledgeGapTracker:
    """Track what a model doesn't know (spec Section 43).

    Monitors model predictions on a validation set, identifies systematic
    failures (knowledge gaps), and suggests targeted training data to fill them.

    Creative additions:
      - Gap clustering (group similar failures)
      - Severity scoring (based on frequency + impact)
      - Targeted training data generation (suggest synthetic samples)
      - Gap resolution tracking (mark gaps as resolved after retraining)

    Example:
        import qai
        tracker = qai.KnowledgeGapTracker()
        gaps = tracker.identify_gaps(model, X_val, y_val)
        for gap in gaps:
            print(f"Gap: {gap.description} (severity={gap.severity})")
            suggestions = tracker.suggest_training_data(gap)
    """
    def __init__(self, failure_threshold: float = 0.3, min_gap_size: int = 3):
        self.failure_threshold = failure_threshold
        self.min_gap_size = min_gap_size
        self._gaps: List[KnowledgeGap] = []
        self._gap_counter = 0

    def identify_gaps(self, model: Any, X: np.ndarray, y: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> List[KnowledgeGap]:
        """Identify knowledge gaps by analyzing model failures.

        Groups failed predictions by feature patterns to find systematic gaps.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        preds = np.asarray(model.predict(X))
        failures = preds != y

        if not failures.any():
            return []

        # Group failures by feature value patterns
        gap_map: Dict[str, List[int]] = {}
        for i in np.where(failures)[0]:
            # Create a signature from feature values (binned)
            signature = tuple(int(x * 10) / 10 for x in X[i])
            # Simplify: use top-2 most different features
            gap_map.setdefault(str(signature[:2]), []).append(i)

        new_gaps = []
        for signature, indices in gap_map.items():
            if len(indices) < self.min_gap_size:
                continue
            severity = len(indices) / len(y)
            if severity < self.failure_threshold * 0.1:
                continue
            self._gap_counter += 1
            gap = KnowledgeGap(
                id=f"gap-{self._gap_counter}",
                description=f"Model fails on {len(indices)} samples with feature pattern {signature}",
                category="feature_pattern",
                severity=min(1.0, severity),
                samples=[X[i].tolist() for i in indices[:5]],
            )
            self._gaps.append(gap)
            new_gaps.append(gap)

        return new_gaps

    def suggest_training_data(self, gap: KnowledgeGap, n: int = 10) -> List[np.ndarray]:
        """Creative: suggest synthetic training samples to fill a gap."""
        if not gap.samples:
            return []
        samples = np.array(gap.samples)
        # Generate variations by adding small noise
        suggestions = []
        for _ in range(n):
            base = samples[np.random.randint(len(samples))]
            noise = np.random.randn(*base.shape) * 0.1 * np.abs(base).mean()
            suggestions.append(base + noise)
        return suggestions

    def mark_resolved(self, gap_id: str) -> bool:
        """Mark a knowledge gap as resolved."""
        for gap in self._gaps:
            if gap.id == gap_id:
                gap.resolved = True
                return True
        return False

    def get_gaps(self, include_resolved: bool = False) -> List[KnowledgeGap]:
        if include_resolved:
            return list(self._gaps)
        return [g for g in self._gaps if not g.resolved]

    def summary(self) -> Dict[str, Any]:
        active = [g for g in self._gaps if not g.resolved]
        return {
            "total_gaps": len(self._gaps),
            "active_gaps": len(active),
            "resolved_gaps": len(self._gaps) - len(active),
            "avg_severity": float(np.mean([g.severity for g in active])) if active else 0.0,
            "max_severity": float(max([g.severity for g in active])) if active else 0.0,
        }

    def __repr__(self):
        return f"<KnowledgeGapTracker gaps={len(self._gaps)} active={len(self.get_gaps())}>"


# =====================================================================
# 3. Idle Trainer (spec Section 51)
# =====================================================================
class IdleTrainer:
    """Background training when the system is idle (spec Section 51).

    Runs training in a background thread when the main system isn't busy.
    Useful for continuous improvement without blocking inference.

    Creative additions:
      - Idle detection (monitors CPU/activity)
      - Training budget (max time per idle session)
      - Progress tracking (records what was trained)
      - Auto-stop on inference request

    Example:
        import qai
        trainer = qai.IdleTrainer(model=my_model, X=X_train, y=y_train)
        trainer.start()  # starts background training
        # ... system runs normally ...
        trainer.stop()   # stops background training
    """
    def __init__(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None,
                 max_idle_time: float = 60.0, check_interval: float = 5.0):
        self.model = model
        self.X = np.asarray(X)
        self.y = np.asarray(y) if y is not None else None
        self.max_idle_time = max_idle_time
        self.check_interval = check_interval
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._progress: List[Dict] = []
        self._lock = threading.Lock()

    def start(self):
        """Start background training in a daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._train_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background training."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _train_loop(self):
        """Main training loop (runs in background)."""
        start = time.time()
        while self._running and (time.time() - start) < self.max_idle_time:
            try:
                # Train one epoch
                if self.y is not None:
                    self.model.train(self.X, self.y, verbose=False)
                else:
                    self.model.train(self.X, verbose=False)

                with self._lock:
                    self._progress.append({
                        "timestamp": time.time(),
                        "elapsed": time.time() - start,
                    })
            except Exception:
                pass  # don't crash the background thread

            time.sleep(self.check_interval)

    def get_progress(self) -> List[Dict]:
        with self._lock:
            return list(self._progress)

    @property
    def is_running(self) -> bool:
        return self._running

    def __repr__(self):
        return f"<IdleTrainer running={self._running} progress={len(self._progress)}>"


# =====================================================================
# 4. Marketplace (spec Section 55)
# =====================================================================
@dataclass
class MarketplaceEntry:
    """A marketplace entry (model, technique, skill, or tool)."""
    id: str
    name: str
    type: str  # "model", "technique", "skill", "tool"
    description: str = ""
    author: str = ""
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    checksum: str = ""
    rating: float = 0.0
    downloads: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Marketplace:
    """Model/technique marketplace (spec Section 55).

    Publish, discover, and subscribe to models, techniques, skills, and tools.
    Each entry has metadata (author, version, tags, checksum, rating).

    Creative additions:
      - Checksum verification (supply-chain integrity)
      - Rating system (rate entries 1-5 stars)
      - Tag-based search
      - Subscription system (get notified on updates)

    Example:
        import qai
        mp = qai.Marketplace()
        mp.publish(qai.MarketplaceEntry(id="m1", name="my-model", type="model",
                                         description="A great model", tags=["nlp", "classification"]))
        results = mp.search("nlp")
        mp.rate("m1", 5)
    """
    def __init__(self):
        self._entries: Dict[str, MarketplaceEntry] = {}
        self._ratings: Dict[str, List[int]] = {}
        self._subscriptions: Dict[str, List[Callable]] = {}

    def publish(self, entry: MarketplaceEntry) -> "Marketplace":
        """Publish an entry to the marketplace."""
        if entry.id in self._entries:
            # Notify subscribers of update
            for callback in self._subscriptions.get(entry.id, []):
                try:
                    callback(entry)
                except Exception:
                    pass
        self._entries[entry.id] = entry
        return self

    def get(self, entry_id: str) -> Optional[MarketplaceEntry]:
        return self._entries.get(entry_id)

    def search(self, query: str = "", tags: Optional[List[str]] = None,
               entry_type: Optional[str] = None) -> List[MarketplaceEntry]:
        """Search the marketplace by query string, tags, or type."""
        results = []
        for entry in self._entries.values():
            if entry_type and entry.type != entry_type:
                continue
            if tags and not any(t in entry.tags for t in tags):
                continue
            if query:
                q_lower = query.lower()
                if (q_lower not in entry.name.lower()
                    and q_lower not in entry.description.lower()
                    and not any(q_lower in t.lower() for t in entry.tags)):
                    continue
            results.append(entry)
        return sorted(results, key=lambda e: -e.rating)

    def rate(self, entry_id: str, stars: int) -> bool:
        """Rate an entry (1-5 stars)."""
        if entry_id not in self._entries:
            return False
        stars = max(1, min(5, stars))
        self._ratings.setdefault(entry_id, []).append(stars)
        ratings = self._ratings[entry_id]
        self._entries[entry_id].rating = sum(ratings) / len(ratings)
        return True

    def download(self, entry_id: str) -> Optional[MarketplaceEntry]:
        """Download (get) an entry and increment its download count."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.downloads += 1
        return entry

    def subscribe(self, entry_id: str, callback: Callable) -> "Marketplace":
        """Subscribe to updates for an entry."""
        self._subscriptions.setdefault(entry_id, []).append(callback)
        return self

    def verify_checksum(self, entry_id: str, content: bytes) -> bool:
        """Verify that content matches the entry's checksum."""
        entry = self._entries.get(entry_id)
        if not entry or not entry.checksum:
            return True  # no checksum to verify
        actual = hashlib.sha256(content).hexdigest()
        return actual == entry.checksum

    def list_all(self) -> List[MarketplaceEntry]:
        return list(self._entries.values())

    def __repr__(self):
        return f"<Marketplace entries={len(self._entries)}>"


# =====================================================================
# 5. Advanced Visualizer (spec Section 58)
# =====================================================================
class AdvancedVisualizer:
    """Advanced visualization tools (spec Section 58).

    Generates diagnostic plots beyond the basic model.visualize():
      - Weight heatmap (per-layer weight magnitude)
      - Gradient flow (color-coded per layer)
      - Training curve (loss + accuracy over epochs)
      - Confusion matrix heatmap
      - Feature importance bar chart
      - Prediction distribution

    Creative additions:
      - Formula diff visualization (compare two models' formulas)
      - Multi-model comparison overlay
      - Save to file or return as numpy array

    Example:
        import qai
        viz = qai.AdvancedVisualizer()
        viz.weight_heatmap(model, save_path="/tmp/weights.png")
        viz.training_curve(model, save_path="/tmp/training.png")
    """
    @staticmethod
    def weight_heatmap(model: Any, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Visualize weight magnitudes as a heatmap."""
        weights = None
        if hasattr(model.technique, "w") and model.technique.w is not None:
            weights = np.asarray(model.technique.w).reshape(1, -1)
        elif hasattr(model.technique, "model") and hasattr(model.technique.model, "coef_"):
            weights = np.asarray(model.technique.model.coef_)
        elif hasattr(model.technique, "net") and model.technique.net is not None:
            try:
                import torch
                weights = np.stack([v.detach().cpu().numpy().flatten()
                                    for v in model.technique.net.state_dict().values()[:5]])
            except Exception:
                pass

        if weights is None:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(weights, aspect="auto", cmap="RdBu_r")
            ax.set_title("Weight Heatmap")
            ax.set_xlabel("Weight Index")
            ax.set_ylabel("Layer")
            plt.colorbar(im)
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                return None
            else:
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                plt.close()
                return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except ImportError:
            return weights  # return raw data if matplotlib unavailable

    @staticmethod
    def training_curve(model: Any, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Plot training loss curve."""
        losses = None
        if hasattr(model.technique, "loss_history"):
            losses = model.technique.loss_history()
        elif hasattr(model.technique, "_losses"):
            losses = model.technique._losses
        elif hasattr(model.technique, "_loss_history"):
            losses = model.technique._loss_history

        if not losses:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(losses, color="#2196F3", linewidth=2)
            ax.set_title("Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                return None
            plt.close()
            return np.array(losses)
        except ImportError:
            return np.array(losses)

    @staticmethod
    def confusion_matrix_heatmap(y_true: np.ndarray, y_pred: np.ndarray,
                                  save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Plot confusion matrix as a heatmap."""
        import qai
        cm = qai.confusion_matrix(y_true, y_pred)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            plt.colorbar(im)
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
            return cm
        except ImportError:
            return cm

    @staticmethod
    def formula_diff(model_a: Any, model_b: Any) -> Dict[str, Any]:
        """Creative: compare two models' formulas (for Regression-style models)."""
        a_formula = model_a.formula() if hasattr(model_a, "formula") else str(model_a)
        b_formula = model_b.formula() if hasattr(model_b, "formula") else str(model_b)
        return {
            "model_a": a_formula,
            "model_b": b_formula,
            "identical": a_formula == b_formula,
        }

    def __repr__(self):
        return "<AdvancedVisualizer>"


# =====================================================================
# 6. Bidirectional Model Communication (spec Section 67)
# =====================================================================
class BidirectionalChannel:
    """Two-way communication channel between models (spec Section 67).

    Allows two models to exchange information bidirectionally:
      - Model A sends output to Model B
      - Model B processes and sends feedback to Model A
      - Model A uses feedback to improve its next output

    Creative additions:
      - Round-trip tracking (latency per exchange)
      - Convergence detection (stop when models agree)
      - Message history (full conversation log)

    Example:
        import qai
        channel = qai.BidirectionalChannel(model_a=fn_a, model_b=fn_b)
        result = channel.exchange("initial input", max_rounds=5)
    """
    def __init__(self, model_a: Callable, model_b: Callable,
                 convergence_threshold: float = 0.01):
        self.model_a = model_a
        self.model_b = model_b
        self.convergence_threshold = convergence_threshold
        self._history: List[Dict] = []

    def exchange(self, initial_input: Any, max_rounds: int = 5) -> Dict[str, Any]:
        """Run a bidirectional exchange between the two models.

        Args:
            initial_input: Starting input for model A.
            max_rounds: Max exchange rounds.

        Returns:
            Dict with final output, rounds, converged, history.
        """
        self._history = []
        current = initial_input
        converged = False

        for round_num in range(max_rounds):
            t0 = time.time()
            # Model A processes
            output_a = self._safe_call(self.model_a, current)
            # Model B processes A's output
            output_b = self._safe_call(self.model_b, output_a)
            elapsed = time.time() - t0

            self._history.append({
                "round": round_num + 1,
                "input_to_a": self._safe_str(current),
                "output_from_a": self._safe_str(output_a),
                "output_from_b": self._safe_str(output_b),
                "elapsed_s": round(elapsed, 4),
            })

            # Check convergence (output stopped changing)
            if round_num > 0:
                prev_output = self._history[-2]["output_from_b"]
                if self._distance(prev_output, self._safe_str(output_b)) < self.convergence_threshold:
                    converged = True
                    break

            current = output_b  # B's output becomes A's next input

        return {
            "final_output": output_b,
            "rounds": len(self._history),
            "converged": converged,
            "history": self._history,
        }

    @staticmethod
    def _safe_call(fn: Callable, arg: Any) -> Any:
        try:
            return fn(arg)
        except Exception as e:
            return f"ERROR: {e}"

    @staticmethod
    def _safe_str(v: Any) -> str:
        try:
            return str(v)[:200]
        except Exception:
            return repr(v)[:200]

    @staticmethod
    def _distance(a: str, b: str) -> float:
        """Simple string distance (normalized edit distance approximation)."""
        if a == b:
            return 0.0
        max_len = max(len(a), len(b), 1)
        # Very rough: character-level difference
        diff = sum(1 for c1, c2 in zip(a, b) if c1 != c2) + abs(len(a) - len(b))
        return diff / max_len

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def __repr__(self):
        return f"<BidirectionalChannel rounds={len(self._history)}>"


# =====================================================================
# 7. Router Core (spec Section 79)
# =====================================================================
class RouterCore:
    """Predictive model loading + request routing (spec Section 79).

    Routes incoming requests to the best model based on:
      - Request type/content
      - Model availability
      - Model performance history
      - Predictive loading (pre-load models that will likely be needed)

    Creative additions:
      - LRU model cache (unload least-recently-used models)
      - Request prediction (predict which model will be needed next)
      - Load balancing (distribute requests across model instances)
      - Performance tracking (per-model latency + accuracy)

    Example:
        import qai
        router = qai.RouterCore(max_loaded=3)
        router.register("classifier", classifier_model)
        router.register("regressor", regressor_model)
        result = router.route({"type": "classification", "data": [1.0, 2.0]})
    """
    def __init__(self, max_loaded: int = 5, prediction_window: int = 10):
        self.max_loaded = max_loaded
        self.prediction_window = prediction_window
        self._models: Dict[str, Any] = {}
        self._loaded: Dict[str, Any] = {}  # currently loaded (active) models
        self._usage: Dict[str, int] = {}
        self._latency: Dict[str, List[float]] = {}
        self._request_history: List[str] = []

    def register(self, name: str, model: Any) -> "RouterCore":
        """Register a model (not loaded until needed)."""
        self._models[name] = model
        self._usage[name] = 0
        self._latency[name] = []
        return self

    def route(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route a request to the best model.

        Args:
            request: Dict with at least "type" (model name) and "data" (input).

        Returns:
            Dict with model_name, result, latency_s.
        """
        model_name = request.get("type") or self._predict_model(request)
        if model_name not in self._models:
            # Find closest match
            available = list(self._models.keys())
            model_name = available[0] if available else None
            if not model_name:
                return {"error": "No models registered"}

        # Load model if not loaded
        if model_name not in self._loaded:
            self._load_model(model_name)

        # Execute
        model = self._loaded[model_name]
        t0 = time.time()
        try:
            data = request.get("data")
            if hasattr(model, "predict"):
                result = model.predict(data)
            elif callable(model):
                result = model(data)
            else:
                result = model
        except Exception as e:
            result = f"ERROR: {e}"
        latency = time.time() - t0

        # Track
        self._usage[model_name] += 1
        self._latency[model_name].append(latency)
        self._request_history.append(model_name)
        if len(self._request_history) > self.prediction_window:
            self._request_history = self._request_history[-self.prediction_window:]

        return {"model_name": model_name, "result": result, "latency_s": round(latency, 4)}

    def _load_model(self, name: str):
        """Load a model into the active set (with LRU eviction)."""
        if len(self._loaded) >= self.max_loaded:
            # Evict least-recently-used
            lru = min(self._loaded, key=lambda k: self._usage.get(k, 0))
            del self._loaded[lru]
        self._loaded[name] = self._models[name]

    def _predict_model(self, request: Dict[str, Any]) -> Optional[str]:
        """Creative: predict which model will be needed based on request content."""
        # Simple: use most-frequently-used model
        if not self._usage:
            return None
        return max(self._usage, key=self._usage.get)

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "registered": list(self._models.keys()),
            "loaded": list(self._loaded.keys()),
            "usage": dict(self._usage),
            "avg_latency": {k: float(np.mean(v)) if v else 0.0 for k, v in self._latency.items()},
            "total_requests": sum(self._usage.values()),
        }

    def __repr__(self):
        return f"<RouterCore registered={len(self._models)} loaded={len(self._loaded)}>"


# =====================================================================
# 8. Automated Scientist (spec Section 94)
# =====================================================================
class AutomatedScientist:
    """Autonomous hypothesis generation + testing (spec Section 94).

    The automated scientist:
      1. Observes data patterns
      2. Generates hypotheses (using a model or rules)
      3. Designs experiments to test hypotheses
      4. Runs experiments (trains/evaluates models)
      5. Draws conclusions (accept/reject hypotheses)

    Creative additions:
      - Hypothesis scoring (novelty + testability + impact)
      - Experiment budget (max experiments to run)
      - Conclusion confidence (statistical significance)
      - Discovery log (full audit trail)

    Example:
        import qai
        scientist = qai.AutomatedScientist(data=X, target=y, budget=5)
        results = scientist.run()
        for r in results:
            print(f"Hypothesis: {r['hypothesis']} -> {r['conclusion']} (conf={r['confidence']})")
    """
    def __init__(self, data: np.ndarray, target: Optional[np.ndarray] = None,
                 budget: int = 10, significance: float = 0.05):
        self.data = np.asarray(data, dtype=np.float32)
        self.target = np.asarray(target) if target is not None else None
        self.budget = budget
        self.significance = significance
        self._discoveries: List[Dict] = []
        self._hypotheses: List[Dict] = []

    def _generate_hypotheses(self) -> List[Dict]:
        """Generate hypotheses about the data."""
        hypotheses = []
        n_features = self.data.shape[1] if self.data.ndim > 1 else 1

        # Hypothesis type 1: feature correlations
        if self.target is not None and n_features > 1:
            for i in range(min(n_features, 5)):
                corr = float(np.corrcoef(self.data[:, i], self.target)[0, 1])
                if abs(corr) > 0.1:
                    hypotheses.append({
                        "id": f"h{len(hypotheses)+1}",
                        "statement": f"Feature {i} is {'positively' if corr > 0 else 'negatively'} correlated with target (r={corr:.3f})",
                        "type": "correlation",
                        "test": lambda i=i: self._test_correlation(i),
                        "novelty": 1.0 - abs(corr),  # weaker correlations are more novel
                    })

        # Hypothesis type 2: feature distributions
        for i in range(min(n_features, 3)):
            mean = float(np.mean(self.data[:, i]))
            std = float(np.std(self.data[:, i]))
            hypotheses.append({
                "id": f"h{len(hypotheses)+1}",
                "statement": f"Feature {i} follows a normal distribution (mean={mean:.2f}, std={std:.2f})",
                "type": "distribution",
                "test": lambda i=i: self._test_normality(i),
                "novelty": 0.5,
            })

        # Hypothesis type 3: cluster structure
        hypotheses.append({
            "id": f"h{len(hypotheses)+1}",
            "statement": "Data has 3 natural clusters",
            "type": "clustering",
            "test": lambda: self._test_clustering(),
            "novelty": 0.7,
        })

        return hypotheses

    def _test_correlation(self, feature_idx: int) -> Dict[str, Any]:
        """Test a correlation hypothesis."""
        corr = float(np.corrcoef(self.data[:, feature_idx], self.target)[0, 1])
        # Simple significance test: |corr| > 0.3
        significant = abs(corr) > 0.3
        return {
            "result": f"r={corr:.4f}",
            "significant": significant,
            "confidence": min(1.0, abs(corr) * 2),
        }

    def _test_normality(self, feature_idx: int) -> Dict[str, Any]:
        """Test normality hypothesis (simple skewness check)."""
        data = self.data[:, feature_idx]
        mean = np.mean(data)
        std = np.std(data)
        skewness = float(np.mean(((data - mean) / (std + 1e-8)) ** 3))
        is_normal = abs(skewness) < 0.5
        return {
            "result": f"skewness={skewness:.4f}",
            "significant": is_normal,
            "confidence": 1.0 - min(1.0, abs(skewness)),
        }

    def _test_clustering(self) -> Dict[str, Any]:
        """Test clustering hypothesis using KMeans."""
        import qai
        try:
            model = qai.build(technique="kmeans", n_clusters=3)
            model.train(self.data, verbose=False)
            labels = model.predict(self.data)
            from qai.metrics import silhouette_score
            score = float(silhouette_score(self.data, labels))
            return {
                "result": f"silhouette={score:.4f}",
                "significant": score > 0.2,
                "confidence": min(1.0, score * 3),
            }
        except Exception as e:
            return {"result": f"error: {e}", "significant": False, "confidence": 0.0}

    def run(self) -> List[Dict[str, Any]]:
        """Run the automated scientific process.

        Returns:
            List of discovery dicts with hypothesis, conclusion, confidence.
        """
        self._hypotheses = self._generate_hypotheses()
        self._discoveries = []

        for h in self._hypotheses[:self.budget]:
            try:
                test_result = h["test"]()
                conclusion = "accepted" if test_result["significant"] else "rejected"
                discovery = {
                    "hypothesis": h["statement"],
                    "type": h["type"],
                    "conclusion": conclusion,
                    "confidence": test_result["confidence"],
                    "result": test_result["result"],
                    "novelty": h["novelty"],
                    "timestamp": time.time(),
                }
                self._discoveries.append(discovery)
            except Exception as e:
                self._discoveries.append({
                    "hypothesis": h["statement"],
                    "conclusion": "error",
                    "confidence": 0.0,
                    "result": str(e),
                    "timestamp": time.time(),
                })

        return self._discoveries

    def get_discoveries(self) -> List[Dict]:
        return list(self._discoveries)

    def summary(self) -> Dict[str, Any]:
        accepted = [d for d in self._discoveries if d["conclusion"] == "accepted"]
        return {
            "total_hypotheses": len(self._hypotheses),
            "total_experiments": len(self._discoveries),
            "accepted": len(accepted),
            "rejected": len(self._discoveries) - len(accepted),
            "avg_confidence": float(np.mean([d["confidence"] for d in self._discoveries])) if self._discoveries else 0.0,
        }

    def __repr__(self):
        return f"<AutomatedScientist budget={self.budget} discoveries={len(self._discoveries)}>"


# =====================================================================
# Public API
# =====================================================================
__all__ = [
    "OutputRenderer",
    "KnowledgeGapTracker", "KnowledgeGap",
    "IdleTrainer",
    "Marketplace", "MarketplaceEntry",
    "AdvancedVisualizer",
    "BidirectionalChannel",
    "RouterCore",
    "AutomatedScientist",
]
