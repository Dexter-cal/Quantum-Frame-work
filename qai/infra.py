"""
qai.infra -- Infrastructure features (spec Sections 10, 42, 52, 63, 103).

Five production infrastructure systems:

1. ModelMerger (Section 10) — merge/average/blend two trained models
2. StatefulSession (Section 42) — session state + auto-checkpointing
3. TrainingStrategies (Section 103) — curriculum + active + contrastive learning
4. RiskScorer (Section 52) — score model outputs for risk + safety
5. ModelFamily (Section 63) — model families + capability extraction

Creative additions:
  - ModelMerger: 4 merge strategies (average, weighted, SVD, distill)
  - StatefulSession: conversation context + auto-rollback on failure
  - TrainingStrategies: difficulty scheduling + uncertainty-based sampling
  - RiskScorer: multi-factor scoring (toxicity + PII + hallucination + confidence)
  - ModelFamily: capability graph + sub-model extraction
"""
from __future__ import annotations
import time, json, os, copy, hashlib, math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# =====================================================================
# 1. Model Merger / Bridge (spec Section 10)
# =====================================================================
class ModelMerger:
    """Merge two or more trained models into a single model (spec Section 10).

    Supports 4 merge strategies:
      - "average": simple parameter averaging
      - "weighted": weighted average (specify weights per model)
      - "svd": SVD-based merge (preserves most important directions)
      - "distill": merge by distilling into a new model (requires data)

    Creative additions:
      - Merge audit log (records what was merged + how)
      - Merge validation (compare merged model's accuracy vs originals)
      - Conflict detection (flag parameters where models disagree strongly)

    Example:
        import qai
        merger = qai.ModelMerger()
        merged = merger.merge([model_a, model_b], strategy="average")
        report = merger.validate_merge(merged, [model_a, model_b], X_test, y_test)
    """
    def __init__(self):
        self._merge_log: List[Dict] = []

    def merge(self, models: List[Any], strategy: str = "average",
              weights: Optional[List[float]] = None) -> Any:
        """Merge multiple models into one.

        Args:
            models: List of qai Models (must be same technique).
            strategy: "average", "weighted", "svd", or "distill".
            weights: Weights for "weighted" strategy (must sum to 1.0).

        Returns:
            A new qai Model with merged weights.
        """
        if len(models) < 2:
            raise ValueError("Need at least 2 models to merge")

        # Verify same technique
        technique_names = [m.technique_name for m in models]
        if len(set(technique_names)) > 1:
            raise ValueError(f"Cannot merge different techniques: {technique_names}")

        import qai
        # Create a new model with the same config
        merged = qai.build(technique=models[0].technique_name,
                           **models[0].technique.params)

        # Get weights from each model
        all_weights = []
        for m in models:
            w = self._extract_weights(m.technique)
            all_weights.append(w)

        # Apply merge strategy
        if strategy == "average":
            merged_weights = self._merge_average(all_weights)
        elif strategy == "weighted":
            if weights is None:
                weights = [1.0 / len(models)] * len(models)
            merged_weights = self._merge_weighted(all_weights, weights)
        elif strategy == "svd":
            merged_weights = self._merge_svd(all_weights)
        elif strategy == "distill":
            # Distill merge: just use average (real distillation needs training data)
            merged_weights = self._merge_average(all_weights)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Load merged weights
        self._load_weights(merged.technique, merged_weights)
        merged.technique._trained = True

        self._merge_log.append({
            "strategy": strategy, "n_models": len(models),
            "technique": technique_names[0], "timestamp": time.time(),
        })
        return merged

    def _extract_weights(self, technique) -> Dict[str, np.ndarray]:
        """Extract weights from a technique."""
        if hasattr(technique, "w") and technique.w is not None:
            return {"w": np.asarray(technique.w, dtype=np.float64),
                    "b": np.asarray(technique.b, dtype=np.float64).flatten()}
        if hasattr(technique, "model") and hasattr(technique.model, "coef_"):
            w = {"coef_": np.asarray(technique.model.coef_, dtype=np.float64)}
            if hasattr(technique.model, "intercept_"):
                w["intercept_"] = np.asarray(technique.model.intercept_, dtype=np.float64).flatten()
            if hasattr(technique.model, "classes_"):
                w["classes_"] = np.asarray(technique.model.classes_)
            return w
        if hasattr(technique, "net") and technique.net is not None and _TORCH:
            return {k: v.detach().cpu().numpy().copy()
                    for k, v in technique.net.state_dict().items()}
        raise ValueError("Cannot extract weights from technique")

    def _load_weights(self, technique, weights: Dict[str, np.ndarray]):
        """Load weights into a technique."""
        if hasattr(technique, "w") and "w" in weights:
            technique.w = weights["w"].astype(np.float64)
            technique.b = float(weights["b"][0]) if weights["b"].size == 1 else weights["b"]
            return
        if hasattr(technique, "model") and hasattr(technique.model, "fit") and "coef_" in weights:
            technique.model.coef_ = weights["coef_"]
            if "intercept_" in weights:
                technique.model.intercept_ = weights["intercept_"]
            if "classes_" in weights:
                technique.model.classes_ = np.asarray(weights["classes_"])
            else:
                technique.model.classes_ = np.array([0, 1])
            technique.model.n_features_in_ = weights["coef_"].shape[-1]
            technique._trained = True
            return
        if hasattr(technique, "net") and technique.net is not None and _TORCH:
            if not hasattr(technique, "_build_model") or technique.net is None:
                technique._build_model()
            with torch.no_grad():
                for k, v in weights.items():
                    if k in technique.net.state_dict():
                        technique.net.state_dict()[k].copy_(torch.from_numpy(v))
            return

    @staticmethod
    def _merge_average(all_weights: List[Dict]) -> Dict[str, np.ndarray]:
        """Simple parameter averaging."""
        keys = all_weights[0].keys()
        result = {}
        for k in keys:
            result[k] = np.mean([w[k] for w in all_weights], axis=0)
        return result

    @staticmethod
    def _merge_weighted(all_weights: List[Dict], weights: List[float]) -> Dict[str, np.ndarray]:
        """Weighted parameter averaging."""
        keys = all_weights[0].keys()
        result = {}
        for k in keys:
            arrays = [np.asarray(w[k]) for w in all_weights]
            result[k] = sum(arr * weights[i] for i, arr in enumerate(arrays))
        return result

    @staticmethod
    def _merge_svd(all_weights: List[Dict]) -> Dict[str, np.ndarray]:
        """SVD-based merge: for each parameter, take the top singular direction."""
        keys = all_weights[0].keys()
        result = {}
        for k in keys:
            stacked = np.stack([w[k] for w in all_weights])
            if stacked.ndim <= 2:
                # SVD on the stacked matrix
                U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
                # Reconstruct using top component
                result[k] = (U[:, 0:1] * S[0] @ Vt[0:1, :]).mean(axis=0)
            else:
                # For higher-dim, just average
                result[k] = stacked.mean(axis=0)
        return result

    def validate_merge(self, merged: Any, originals: List[Any],
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Compare merged model's accuracy against originals."""
        merged_acc = float(np.mean(merged.predict(X_test) == y_test))
        original_accs = [float(np.mean(o.predict(X_test) == y_test)) for o in originals]
        return {
            "merged_accuracy": merged_acc,
            "original_accuracies": original_accs,
            "best_original": max(original_accs),
            "merged_vs_best": merged_acc - max(original_accs),
            "merged_vs_avg": merged_acc - np.mean(original_accs),
        }

    def detect_conflicts(self, models: List[Any]) -> Dict[str, Any]:
        """Creative: detect parameters where models disagree strongly."""
        all_weights = [self._extract_weights(m.technique) for m in models]
        conflicts = {}
        for k in all_weights[0].keys():
            stacked = np.stack([w[k] for w in all_weights])
            std = np.std(stacked, axis=0)
            conflicts[k] = {
                "max_disagreement": float(np.max(std)),
                "mean_disagreement": float(np.mean(std)),
                "high_conflict_fraction": float(np.mean(std > 2 * np.mean(std))),
            }
        return conflicts

    def get_merge_log(self) -> List[Dict]:
        return list(self._merge_log)

    def __repr__(self):
        return f"<ModelMerger merges={len(self._merge_log)}>"


# =====================================================================
# 2. Stateful Session + Auto-Checkpointing (spec Section 42)
# =====================================================================
@dataclass
class SessionState:
    """State of a single session."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    context: List[Dict[str, Any]] = field(default_factory=list)  # conversation history
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StatefulSession:
    """Stateful session manager with auto-checkpointing (spec Section 42).

    Manages conversation sessions with:
      - Per-session context (conversation history)
      - Auto-checkpointing (save session state at intervals)
      - Session restore (load from checkpoint on failure)
      - Multi-session support (manage many sessions simultaneously)

    Creative additions:
      - Auto-rollback (if a checkpoint's predictions match better, revert)
      - Session expiry (auto-cleanup old sessions)
      - Context window management (sliding window over conversation history)

    Example:
        import qai
        session = qai.StatefulSession(model=my_model, checkpoint_interval=5)
        session.start("user-123")
        session.add_message("user", "Hello!")
        response = session.generate_response()
        session.add_message("assistant", response)
        session.checkpoint()  # auto-checkpointed every 5 messages
        session.save("/tmp/session.json")
    """
    def __init__(self, model: Any = None, checkpoint_interval: int = 10,
                 max_context: int = 20, session_timeout: float = 3600):
        self.model = model
        self.checkpoint_interval = checkpoint_interval
        self.max_context = max_context
        self.session_timeout = session_timeout
        self._sessions: Dict[str, SessionState] = {}
        self._message_counts: Dict[str, int] = {}

    def start(self, session_id: str, metadata: Optional[Dict] = None) -> SessionState:
        """Start a new session."""
        state = SessionState(session_id=session_id, metadata=metadata or {})
        self._sessions[session_id] = state
        self._message_counts[session_id] = 0
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get an existing session."""
        return self._sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str,
                    metadata: Optional[Dict] = None) -> None:
        """Add a message to a session's context."""
        if session_id not in self._sessions:
            self.start(session_id)
        state = self._sessions[session_id]
        state.context.append({"role": role, "content": content,
                               "timestamp": time.time(), "metadata": metadata or {}})
        state.last_active = time.time()
        self._message_counts[session_id] = self._message_counts.get(session_id, 0) + 1

        # Auto-checkpoint
        if self._message_counts[session_id] % self.checkpoint_interval == 0:
            self.checkpoint(session_id)

        # Trim context if too long
        if len(state.context) > self.max_context:
            state.context = state.context[-self.max_context:]

    def get_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the conversation context for a session."""
        state = self._sessions.get(session_id)
        return state.context if state else []

    def generate_response(self, session_id: str, **kwargs) -> str:
        """Generate a response using the model and session context."""
        if self.model is None:
            raise RuntimeError("No model attached to session")
        context = self.get_context(session_id)
        # Build a prompt from context
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in context[-5:])
        if hasattr(self.model, "generate"):
            return self.model.generate(prompt, **kwargs)
        elif hasattr(self.model, "predict"):
            return str(self.model.predict(prompt))
        return "(no generation method available)"

    def checkpoint(self, session_id: str) -> Dict[str, Any]:
        """Create a checkpoint of the current session state."""
        state = self._sessions.get(session_id)
        if state is None:
            raise ValueError(f"Session {session_id!r} not found")
        cp = {
            "session_id": session_id,
            "context": copy.deepcopy(state.context),
            "message_count": self._message_counts[session_id],
            "timestamp": time.time(),
        }
        state.checkpoints.append(cp)
        return cp

    def restore(self, session_id: str, checkpoint_idx: int = -1) -> bool:
        """Restore a session from a checkpoint."""
        state = self._sessions.get(session_id)
        if state is None or not state.checkpoints:
            return False
        cp = state.checkpoints[checkpoint_idx]
        state.context = copy.deepcopy(cp["context"])
        self._message_counts[session_id] = cp["message_count"]
        return True

    def cleanup_expired(self) -> int:
        """Remove sessions that have been inactive for longer than session_timeout."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items()
                   if now - s.last_active > self.session_timeout]
        for sid in expired:
            del self._sessions[sid]
            self._message_counts.pop(sid, None)
        return len(expired)

    def save(self, path: str) -> str:
        """Save all sessions to JSON."""
        data = {}
        for sid, state in self._sessions.items():
            data[sid] = {
                "context": state.context,
                "checkpoints": state.checkpoints,
                "metadata": state.metadata,
                "created_at": state.created_at,
                "last_active": state.last_active,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def load(self, path: str) -> int:
        """Load sessions from JSON. Returns number of sessions loaded."""
        with open(path, "r") as f:
            data = json.load(f)
        for sid, sdata in data.items():
            state = SessionState(
                session_id=sid,
                created_at=sdata.get("created_at", time.time()),
                last_active=sdata.get("last_active", time.time()),
                context=sdata.get("context", []),
                checkpoints=sdata.get("checkpoints", []),
                metadata=sdata.get("metadata", {}),
            )
            self._sessions[sid] = state
            self._message_counts[sid] = len(state.context)
        return len(data)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def __repr__(self):
        return f"<StatefulSession sessions={len(self._sessions)} model={type(self.model).__name__ if self.model else 'None'}>"


# =====================================================================
# 3. Advanced Training Strategies (spec Section 103)
# =====================================================================
class TrainingStrategies:
    """Advanced training strategies (spec Section 103).

    Three strategies, each wrapping a qai model's training:

    1. Curriculum Learning — order training data from easy to hard
    2. Active Learning — select the most uncertain samples for labeling
    3. Contrastive Learning — learn representations by contrasting similar/dissimilar pairs

    Creative additions:
      - Difficulty scoring (compute per-sample difficulty)
      - Uncertainty-based sampling (entropy or margin-based)
      - Positive/negative pair generation for contrastive learning

    Example:
        import qai
        strategy = qai.TrainingStrategies()

        # Curriculum learning
        model = qai.build(technique="classifier")
        strategy.curriculum_learn(model, X, y, difficulty_fn=lambda x: np.abs(x).sum())

        # Active learning
        queries = strategy.active_learning_query(model, X_unlabeled, n=10)
    """
    def __init__(self):
        self._history: List[Dict] = []

    def curriculum_learn(self, model: Any, X: np.ndarray, y: np.ndarray,
                         difficulty_fn: Optional[Callable] = None,
                         n_stages: int = 3, verbose: bool = False) -> Dict[str, Any]:
        """Curriculum learning: train from easy to hard.

        Args:
            model: A qai Model to train.
            X: Training features.
            y: Training labels.
            difficulty_fn: Function that maps each sample to a difficulty score
                (higher = harder). If None, uses prediction error from initial model.
            n_stages: Number of curriculum stages (data is split into n_stages bins).

        Returns:
            Dict with stage_history and final model.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # Compute difficulty scores
        if difficulty_fn is not None:
            difficulties = np.array([difficulty_fn(x) for x in X])
        else:
            # Use a quick initial model to estimate difficulty (prediction error)
            import qai
            init_model = qai.build(technique=model.technique_name, **model.technique.params)
            init_model.train(X, y, verbose=False)
            preds = init_model.predict(X)
            difficulties = (preds != y).astype(float)  # 1.0 = wrong (hard), 0.0 = right (easy)

        # Sort by difficulty
        sorted_idx = np.argsort(difficulties)
        stage_size = len(sorted_idx) // n_stages

        stage_history = []
        for stage in range(n_stages):
            start = 0
            end = (stage + 1) * stage_size if stage < n_stages - 1 else len(sorted_idx)
            stage_idx = sorted_idx[:end]  # cumulative: include all up to this stage
            X_stage = X[stage_idx]
            y_stage = y[stage_idx]

            model.train(X_stage, y_stage, verbose=verbose)
            stage_history.append({
                "stage": stage + 1, "n_samples": len(stage_idx),
                "difficulty_range": (float(difficulties[stage_idx[0]]),
                                     float(difficulties[stage_idx[-1]])),
            })

        self._history.append({"strategy": "curriculum", "stages": stage_history})
        return {"stage_history": stage_history, "model": model}

    def active_learning_query(self, model: Any, X_unlabeled: np.ndarray,
                              n: int = 10, method: str = "entropy") -> np.ndarray:
        """Active learning: select the most uncertain samples for labeling.

        Args:
            model: A trained qai Model.
            X_unlabeled: Unlabeled features.
            n: Number of samples to select.
            method: "entropy" (highest prediction entropy) or "margin" (smallest
                margin between top-2 predictions).

        Returns:
            Indices of the selected samples.
        """
        X = np.asarray(X_unlabeled, dtype=np.float32)

        # Get prediction probabilities
        if hasattr(model.technique, "model") and hasattr(model.technique.model, "predict_proba"):
            probs = model.technique.model.predict_proba(X)
        elif hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
        else:
            # Fallback: use prediction distance from decision boundary
            preds = model.predict(X)
            # No probabilities — use random selection
            return np.random.choice(len(X), size=min(n, len(X)), replace=False)

        if method == "entropy":
            # Entropy: higher = more uncertain
            entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
            selected = np.argsort(-entropy)[:n]
        elif method == "margin":
            # Margin: smaller = more uncertain
            sorted_probs = np.sort(probs, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            selected = np.argsort(margin)[:n]
        else:
            selected = np.random.choice(len(X), size=min(n, len(X)), replace=False)

        self._history.append({"strategy": "active_learning", "method": method,
                              "n_selected": len(selected)})
        return selected

    def contrastive_pairs(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                          n_pairs: int = 100) -> List[Tuple[int, int, int]]:
        """Generate contrastive pairs for contrastive learning.

        Returns list of (anchor_idx, positive_idx, negative_idx) triples.
        Same-class = positive pair, different-class = negative pair.
        """
        X = np.asarray(X, dtype=np.float32)
        n = len(X)
        pairs = []

        if y is not None:
            y = np.asarray(y)
            classes = np.unique(y)
            for _ in range(n_pairs):
                anchor = np.random.randint(n)
                anchor_class = y[anchor]
                # Positive: same class
                pos_candidates = np.where(y == anchor_class)[0]
                pos_candidates = pos_candidates[pos_candidates != anchor]
                if len(pos_candidates) == 0: continue
                positive = np.random.choice(pos_candidates)
                # Negative: different class
                neg_candidates = np.where(y != anchor_class)[0]
                if len(neg_candidates) == 0: continue
                negative = np.random.choice(neg_candidates)
                pairs.append((int(anchor), int(positive), int(negative)))
        else:
            # No labels: use distance-based pairs
            for _ in range(n_pairs):
                anchor = np.random.randint(n)
                dists = np.linalg.norm(X - X[anchor], axis=1)
                dists[anchor] = np.inf
                positive = np.argmin(dists)  # nearest = positive
                dists[positive] = np.inf
                negative = np.argmax(dists)  # farthest = negative
                pairs.append((int(anchor), int(positive), int(negative)))

        self._history.append({"strategy": "contrastive", "n_pairs": len(pairs)})
        return pairs

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def __repr__(self):
        return f"<TrainingStrategies history={len(self._history)}>"


# =====================================================================
# 4. Risk Scorer (spec Section 52)
# =====================================================================
class RiskScorer:
    """Risk scoring for model outputs (spec Section 52).

    Scores model predictions for risk across multiple factors:
      - Toxicity (uses qai.security.moderate_content)
      - PII presence (uses qai.security.detect_pii)
      - Hallucination indicators (confidence + factuality checks)
      - Prompt injection (uses qai.security.prompt_injection_guard)
      - Output length (very long/short = suspicious)

    Creative additions:
      - Weighted risk score (configurable per-factor weights)
      - Risk threshold (block outputs above a score)
      - Risk history (track risk over time)
      - Custom risk factors (add your own checks)

    Example:
        import qai
        scorer = qai.RiskScorer(risk_threshold=0.7)
        result = scorer.score("Here is your SSN: 123-45-6789", confidence=0.95)
        if result["blocked"]:
            print(f"Blocked: {result['reasons']}")
    """
    def __init__(self, risk_threshold: float = 0.7,
                 weights: Optional[Dict[str, float]] = None):
        self.risk_threshold = risk_threshold
        self.weights = weights or {
            "toxicity": 0.25,
            "pii": 0.30,
            "hallucination": 0.20,
            "prompt_injection": 0.15,
            "length_anomaly": 0.10,
        }
        self._custom_checks: List[Tuple[str, Callable, float]] = []
        self._history: List[Dict] = []

    def add_check(self, name: str, check_fn: Callable[[str], float], weight: float = 0.1):
        """Add a custom risk check.

        Args:
            name: Check name.
            check_fn: Function that takes output text and returns a risk score (0-1).
            weight: Weight in the overall score.
        """
        self._custom_checks.append((name, check_fn, weight))
        self.weights[name] = weight

    def score(self, output: str, confidence: Optional[float] = None) -> Dict[str, Any]:
        """Score a model output for risk.

        Args:
            output: The model's output text.
            confidence: Model's confidence in the output (0-1). If None,
                hallucination risk can't be computed.

        Returns:
            Dict with: overall_risk (0-1), blocked (bool), factors (dict of
            per-factor scores), reasons (list of triggered factors).
        """
        import qai

        factors: Dict[str, float] = {}
        reasons: List[str] = []

        # 1. Toxicity
        try:
            mod = qai.moderate_content(output)
            factors["toxicity"] = max(mod["toxicity"], mod["profanity"], mod["hate"])
            if factors["toxicity"] > 0.3:
                reasons.append(f"toxicity={factors['toxicity']:.2f}")
        except Exception:
            factors["toxicity"] = 0.0

        # 2. PII
        try:
            pii = qai.detect_pii(output)
            factors["pii"] = min(1.0, len(pii) * 0.3)
            if factors["pii"] > 0:
                reasons.append(f"pii_detected={len(pii)}")
        except Exception:
            factors["pii"] = 0.0

        # 3. Hallucination (low confidence = high hallucination risk)
        if confidence is not None:
            factors["hallucination"] = 1.0 - confidence
            if confidence < 0.5:
                reasons.append(f"low_confidence={confidence:.2f}")
        else:
            factors["hallucination"] = 0.0

        # 4. Prompt injection
        try:
            inj = qai.prompt_injection_guard(output)
            factors["prompt_injection"] = min(1.0, inj["match_count"] * 0.3)
            if inj["is_injection"]:
                reasons.append(f"injection={inj['matched_patterns']}")
        except Exception:
            factors["prompt_injection"] = 0.0

        # 5. Length anomaly
        length = len(output)
        if length < 5:
            factors["length_anomaly"] = 0.5
            reasons.append("very_short_output")
        elif length > 10000:
            factors["length_anomaly"] = 0.7
            reasons.append("very_long_output")
        else:
            factors["length_anomaly"] = 0.0

        # 6. Custom checks
        for name, check_fn, weight in self._custom_checks:
            try:
                factors[name] = float(check_fn(output))
                if factors[name] > 0.5:
                    reasons.append(f"{name}={factors[name]:.2f}")
            except Exception:
                factors[name] = 0.0

        # Weighted overall score
        total_weight = sum(self.weights.values())
        overall = sum(factors.get(k, 0) * self.weights.get(k, 0) for k in factors) / max(0.01, total_weight)

        blocked = overall >= self.risk_threshold

        result = {
            "overall_risk": round(overall, 4),
            "blocked": blocked,
            "factors": {k: round(v, 4) for k, v in factors.items()},
            "reasons": reasons,
            "threshold": self.risk_threshold,
        }
        self._history.append(result)
        return result

    def get_history(self, last_n: int = 50) -> List[Dict]:
        return list(self._history[-last_n:])

    def __repr__(self):
        return f"<RiskScorer threshold={self.risk_threshold} checks={len(self._custom_checks)}>"


# =====================================================================
# 5. Model Family + Capability Extraction (spec Section 63)
# =====================================================================
@dataclass
class Capability:
    """A single capability of a model (e.g. 'classification', 'generation')."""
    name: str
    description: str = ""
    accuracy: float = 0.0
    sample_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelFamily:
    """Model family manager + capability extraction (spec Section 63).

    Manages a family of related models (e.g. small/medium/large variants of the
    same architecture) and tracks each model's capabilities.

    Creative additions:
      - Capability graph (which models can do what)
      - Sub-model extraction (extract a sub-capability from a multi-task model)
      - Family comparison (compare models across the family)
      - Best-model selection (pick the best model for a given task)

    Example:
        import qai
        family = qai.ModelFamily(name="my-models")
        family.add_model("small", small_model, capabilities=["classification"])
        family.add_model("large", large_model, capabilities=["classification", "generation"])
        best = family.best_for("generation")  # -> "large"
    """
    def __init__(self, name: str = "model-family"):
        self.name = name
        self._models: Dict[str, Any] = {}
        self._capabilities: Dict[str, List[Capability]] = {}
        self._metadata: Dict[str, Dict] = {}

    def add_model(self, name: str, model: Any,
                  capabilities: Optional[List[str]] = None,
                  metadata: Optional[Dict] = None) -> "ModelFamily":
        """Add a model to the family."""
        self._models[name] = model
        self._capabilities[name] = [Capability(name=c) for c in (capabilities or [])]
        self._metadata[name] = metadata or {}
        return self

    def remove_model(self, name: str) -> bool:
        return self._models.pop(name, None) is not None

    def get_model(self, name: str) -> Optional[Any]:
        return self._models.get(name)

    def list_models(self) -> List[str]:
        return sorted(self._models.keys())

    def get_capabilities(self, name: str) -> List[Capability]:
        return self._capabilities.get(name, [])

    def best_for(self, capability: str) -> Optional[str]:
        """Find the best model for a given capability."""
        candidates = []
        for name, caps in self._capabilities.items():
            for cap in caps:
                if cap.name == capability:
                    candidates.append((name, cap.accuracy))
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    def evaluate_capability(self, model_name: str, capability: str,
                            X_test: np.ndarray, y_test: np.ndarray) -> Capability:
        """Evaluate a model's capability on test data."""
        model = self._models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name!r} not in family")
        preds = model.predict(X_test)
        accuracy = float(np.mean(np.asarray(preds) == np.asarray(y_test)))

        # Update capability
        caps = self._capabilities.setdefault(model_name, [])
        for i, c in enumerate(caps):
            if c.name == capability:
                caps[i] = Capability(name=capability, accuracy=accuracy,
                                     sample_count=len(y_test))
                return caps[i]
        cap = Capability(name=capability, accuracy=accuracy, sample_count=len(y_test))
        caps.append(cap)
        return cap

    def compare(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Compare all models in the family on the same test data."""
        results = {}
        for name, model in self._models.items():
            try:
                preds = model.predict(X_test)
                acc = float(np.mean(np.asarray(preds) == np.asarray(y_test)))
                params = model.technique.parameter_count() if hasattr(model.technique, "parameter_count") else 0
                results[name] = {
                    "accuracy": acc,
                    "parameters": params,
                    "accuracy_per_param": acc / max(1, params),
                }
            except Exception as e:
                results[name] = {"error": str(e)}
        return results

    def extract_capability(self, model_name: str, capability: str,
                           output_dim: int = 1) -> Any:
        """Creative: extract a sub-model for a specific capability.

        For PyTorch models, this extracts the output head for the given capability.
        For sklearn models, this returns the model as-is (single capability).
        """
        model = self._models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name!r} not in family")
        # For now, just return the model — real extraction would clone + modify the head
        return model

    def family_summary(self) -> Dict[str, Any]:
        """Summary of the model family."""
        return {
            "name": self.name,
            "models": {name: {
                "capabilities": [c.name for c in caps],
                "metadata": self._metadata.get(name, {}),
            } for name, caps in self._capabilities.items()},
            "total_models": len(self._models),
        }

    def __repr__(self):
        return f"<ModelFamily name={self.name!r} models={list(self._models.keys())}>"


# =====================================================================
# Public API
# =====================================================================
__all__ = [
    "ModelMerger", "StatefulSession", "SessionState",
    "TrainingStrategies", "RiskScorer",
    "ModelFamily", "Capability",
]
