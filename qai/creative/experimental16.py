"""
qai.creative.experimental16 -- 10 more original creative ideas (part 16).

  Idea 159 -- ModelZooCurator             : models as zoo animals + enclosures + feeding
  Idea 160 -- PredictionMuseum             : curate predictions as museum exhibits
  Idea 161 -- NeuralCourtReporter          : stenographer records every weight update
  Idea 162 -- ModelArchaeologicalDig       : dig through training history as strata
  Idea 163 -- WeightAuction                : auction off weights as art lots
  Idea 164 -- PredictionCartographer       : map prediction landscape (plains/mountains/faults)
  Idea 165 -- ModelBiographer              : write a model's life story (birth -> legacy)
  Idea 166 -- HallucinationSurgicalTheater : operate on a hallucination (preop/anesthetize/postop)
  Idea 167 -- WeightOceanography           : map weights as ocean currents + gyres + upwellings
  Idea 168 -- ModelConductor               : conduct an orchestra of models (sections + score)

A "model" here is any object that exposes its parameters via a ``.weights``
attribute (dict / list / ndarray) or via ``state_dict()`` / ``technique``
(as in the rest of qai), and that can be inferred with ``.predict`` /
``.forward`` / ``__call__``.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ----------------------------------------------------------------------------
# internal helpers -- robust to whatever "model" object is handed in
# ----------------------------------------------------------------------------
def _weights_of(model: Any) -> Dict[str, np.ndarray]:
    """Extract {layer_name: ndarray} from any model-ish object.

    Tries qai.weights._extract_weights first, then ``model.weights``
    (dict / list / ndarray), then a scan of public ndarray attributes.
    """
    try:
        from ..weights import _extract_weights
        w = _extract_weights(model)
        if w:
            return {k: np.asarray(v) for k, v in w.items()}
    except Exception:
        pass
    w_attr = getattr(model, "weights", None)
    if isinstance(w_attr, dict):
        return {k: np.asarray(v) for k, v in w_attr.items()}
    if isinstance(w_attr, (list, tuple)):
        return {f"layer_{i}": np.asarray(a) for i, a in enumerate(w_attr)}
    if isinstance(w_attr, np.ndarray):
        return {"W": w_attr}
    out: Dict[str, np.ndarray] = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            v = getattr(model, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            out[name] = v
    return out


def _predict(model: Any, x: Any) -> Any:
    """Run a forward pass using whatever method the model exposes."""
    for meth in ("predict", "forward", "__call__"):
        fn = getattr(model, meth, None)
        if callable(fn):
            try:
                return fn(x)
            except Exception:
                continue
    return None


def _confidence(pred: Any) -> float:
    """A softmax-style confidence proxy: max-|logit| / sum-|logit|."""
    try:
        a = np.asarray(pred, dtype=np.float64).ravel()
        if a.size == 0:
            return 0.0
        return float(np.max(np.abs(a)) / (np.sum(np.abs(a)) + 1e-8))
    except Exception:
        return 0.0


def _forward_activations(model: Any, X: np.ndarray) -> Dict[str, np.ndarray]:
    """Run a (tanh) forward pass through all 2-D weight matrices, returning
    the per-layer activation matrix (rows = samples, cols = neurons).
    """
    weights = _weights_of(model)
    if X.ndim == 1:
        X = X[None, :]
    cur = X.astype(np.float32, copy=False)
    acts: Dict[str, np.ndarray] = {}
    for name in sorted(weights):
        W = np.asarray(weights[name], dtype=np.float32)
        if W.ndim != 2:
            continue
        if cur.shape[-1] == W.shape[0]:
            cur = np.tanh(cur @ W)
            acts[name] = cur
        elif cur.shape[-1] == W.shape[1]:
            cur = np.tanh(cur @ W.T)
            acts[name] = cur
    return acts


def _apply(model: Any, weights: Dict[str, np.ndarray]) -> Any:
    """Apply weight arrays back to a model (best-effort, in-place)."""
    try:
        from ..weights import _apply_weights
        _apply_weights(model, weights)
    except Exception:
        pass
    w_attr = getattr(model, "weights", None)
    if isinstance(w_attr, dict):
        for k, v in weights.items():
            if k in w_attr:
                w_attr[k] = v
        return model
    if isinstance(w_attr, list):
        for i, v in enumerate(weights.values()):
            if i < len(w_attr):
                w_attr[i] = v
        return model
    for name, arr in weights.items():
        try:
            if hasattr(model, name):
                setattr(model, name, arr)
        except Exception:
            pass
    return model


def _argmax_pred(pred: Any) -> int:
    """Return the argmax of |prediction| (the model's chosen class)."""
    try:
        a = np.asarray(pred, dtype=np.float64).ravel()
        if a.size == 0:
            return -1
        return int(np.argmax(np.abs(a)))
    except Exception:
        return -1


# ----------------------------------------------------------------------------
# A minimal runnable model wrapper used internally
# ----------------------------------------------------------------------------
class _MinimalModel:
    """A minimal model wrapper: holds a weights dict and runs a tanh
    forward pass. Used so internal helpers can return a runnable model
    even when the input model was an arbitrary object.
    """

    def __init__(self, weights: Dict[str, np.ndarray]):
        self.weights = {k: np.asarray(v, dtype=np.float32)
                        for k, v in weights.items()}

    def predict(self, x: Any) -> np.ndarray:
        cur = np.asarray(x, dtype=np.float32).ravel()
        for name in sorted(self.weights):
            W = self.weights[name]
            if W.ndim == 2 and cur.size == W.shape[0]:
                cur = np.tanh(cur @ W)
            elif W.ndim == 2 and cur.size == W.shape[1]:
                cur = np.tanh(cur @ W.T)
        return cur


# ============================================================================
# Idea 159 -- ModelZooCurator
# ============================================================================
class ModelZooCurator:
    """Curate a "zoo" of models. Each model is an *animal* living in an
    *enclosure* (a habitat sized to its parameter count). Animals are
    grouped by *species* (architecture type -- classified by depth/width
    ratio of their weight stack). *Feeding* trains every animal on a batch
    and records which gained weight (loss reduction); the *veterinary
    report* identifies sick animals (whose loss went up) and healthy ones
    (whose loss went down). A balanced zoo has every species thriving.
    """

    SPECIES = ("shallow_wide", "deep_narrow", "balanced", "compact")

    def __init__(self, sick_loss_delta: float = 0.0,
                 healthy_loss_delta: float = -1e-3,
                 seed: int = 0):
        self.sick_loss_delta = sick_loss_delta
        self.healthy_loss_delta = healthy_loss_delta
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _classify_species(weights: Dict[str, np.ndarray]) -> str:
        """Classify a model by its depth/width profile."""
        mats = [np.asarray(w) for w in weights.values()
                if np.asarray(w).ndim == 2]
        if not mats:
            return "compact"
        depth = len(mats)
        mean_width = float(np.mean([min(m.shape) for m in mats]))
        if depth <= 2 and mean_width >= 64:
            return "shallow_wide"
        if depth >= 4 and mean_width <= 32:
            return "deep_narrow"
        if mean_width <= 16:
            return "compact"
        return "balanced"

    def populate(self, models: List[Any]) -> Dict[str, Any]:
        """Populate the zoo: assign each model a species, an enclosure
        (sized by parameter count), and a stable identifier."""
        animals: List[Dict[str, Any]] = []
        for i, m in enumerate(models):
            w = _weights_of(m)
            species = self._classify_species(w)
            n_params = int(sum(np.asarray(arr).size for arr in w.values()))
            animals.append({"id": i, "model": m, "species": species,
                            "n_params": n_params,
                            "enclosure_size": max(1, n_params // 1000),
                            "loss_history": []})
        by_species: Dict[str, List[int]] = {s: [] for s in self.SPECIES}
        for a in animals:
            by_species.setdefault(a["species"], []).append(a["id"])
        return {"animals": animals, "n_animals": len(animals),
                "by_species": by_species,
                "founded_at": time.time()}

    def enclosures(self, zoo: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the enclosures: one per species present, with the
        list of animal ids housed in each and the enclosure's total size."""
        out: List[Dict[str, Any]] = []
        animals = {a["id"]: a for a in zoo.get("animals", [])}
        for species, ids in zoo.get("by_species", {}).items():
            if not ids:
                continue
            total = sum(animals[i]["enclosure_size"] for i in ids)
            out.append({"species": species,
                        "animals": list(ids),
                        "enclosure_size": total,
                        "mean_params": (total * 1000) // max(1, len(ids))})
        out.sort(key=lambda e: -e["enclosure_size"])
        return out

    def feed(self, zoo: Dict[str, Any],
             X: np.ndarray, y: np.ndarray,
             lr: float = 1e-2) -> Dict[str, Any]:
        """Feed every animal one batch of (X, y). Each model takes a tiny
        gradient-descent step (one-step tanh MLP update) and the change in
        mean-squared loss is recorded. Returns which animals gained
        (improved) vs lost ground."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32).ravel()
        gains: List[int] = []
        losses: List[int] = []
        for a in zoo.get("animals", []):
            m = a["model"]
            w = _weights_of(m)
            names = sorted(w)
            # baseline loss (tanh forward + MSE against y)
            before = self._loss(m, X, y, w)
            # one-step gradient update on the first 2-D layer
            if names:
                first = np.asarray(w[names[0]], dtype=np.float32)
                if first.ndim == 2 and X.shape[1] == first.shape[0]:
                    cur = np.tanh(X @ first)
                    # propagate remaining layers
                    for nm in names[1:]:
                        Wn = np.asarray(w[nm], dtype=np.float32)
                        if Wn.ndim == 2 and cur.shape[-1] == Wn.shape[0]:
                            cur = np.tanh(cur @ Wn)
                    pred = cur.ravel()[: y.size] if cur.size >= y.size \
                        else np.pad(cur.ravel(), (0, y.size - cur.size))
                    err = pred - y
                    grad = (X.T[: first.shape[0]] * err.mean()).T \
                        if X.shape[0] > 0 else np.zeros_like(first)
                    grad = grad.reshape(first.shape) \
                        if grad.shape == first.shape else \
                        (err.mean() * X.mean(axis=0,
                            keepdims=True).T @ np.ones((1, first.shape[1])))
                    grad = np.broadcast_to(grad, first.shape).copy()
                    new_first = first - lr * grad
                    patched = {k: (new_first if k == names[0] else v)
                               for k, v in w.items()}
                    _apply(m, patched)
                    after = self._loss(m, X, y, patched)
                else:
                    after = before
            else:
                after = before
            delta = float(after - before)
            a["loss_history"].append({"before": before,
                                       "after": after, "delta": delta})
            if delta <= self.healthy_loss_delta:
                gains.append(a["id"])
            elif delta >= self.sick_loss_delta:
                losses.append(a["id"])
        return {"fed_at": time.time(), "batch_size": int(X.shape[0]),
                "gained": gains, "lost_ground": losses,
                "n_healthy": len(gains), "n_sick": len(losses)}

    @staticmethod
    def _loss(model: Any, X: np.ndarray, y: np.ndarray,
              weights: Dict[str, np.ndarray]) -> float:
        """Mean-squared error of the model's (tanh) forward pass vs y."""
        names = sorted(weights)
        if not names:
            return 0.0
        cur = X
        for nm in names:
            Wn = np.asarray(weights[nm], dtype=np.float32)
            if Wn.ndim == 2 and cur.shape[-1] == Wn.shape[0]:
                cur = np.tanh(cur @ Wn)
        pred = cur.ravel()
        yv = y.ravel()
        n = min(pred.size, yv.size)
        if n == 0:
            return 0.0
        return float(np.mean((pred[:n] - yv[:n]) ** 2))

    def veterinary_report(self, zoo: Dict[str, Any]) -> Dict[str, Any]:
        """Build a veterinary report: classify each animal as healthy /
        sick / stable based on its last few feedings, and flag enclosures
        (species) whose entire population is ailing."""
        animals = zoo.get("animals", [])
        if not animals:
            return {"healthy": [], "sick": [], "stable": [],
                    "ailing_species": []}
        healthy: List[Dict[str, Any]] = []
        sick: List[Dict[str, Any]] = []
        stable: List[Dict[str, Any]] = []
        for a in animals:
            hist = a.get("loss_history", [])
            if not hist:
                stable.append({"id": a["id"], "species": a["species"],
                               "status": "unfed"})
                continue
            recent = [h["delta"] for h in hist[-3:]]
            mean_delta = float(np.mean(recent))
            entry = {"id": a["id"], "species": a["species"],
                      "last_delta": mean_delta,
                      "n_feedings": len(hist)}
            if mean_delta <= self.healthy_loss_delta:
                healthy.append(entry)
            elif mean_delta >= self.sick_loss_delta:
                sick.append(entry)
            else:
                stable.append(entry)
        # species whose every member is sick
        ailing: List[str] = []
        by_sp = {a["species"] for a in animals}
        for sp in by_sp:
            members = [a for a in animals if a["species"] == sp]
            mem_ids = {a["id"] for a in members}
            sick_ids = {e["id"] for e in sick}
            if mem_ids and mem_ids.issubset(sick_ids):
                ailing.append(sp)
        return {"healthy": healthy, "sick": sick, "stable": stable,
                "ailing_species": ailing,
                "n_healthy": len(healthy), "n_sick": len(sick),
                "n_stable": len(stable)}


# ============================================================================
# Idea 160 -- PredictionMuseum
# ============================================================================
class PredictionMuseum:
    """Curate a *museum of predictions*. Each prediction the model makes is
    an *exhibit*. Correct predictions join the *permanent collection*
    (organised by class, like an art-historical period). Wrong predictions
    become *special exhibits* -- didactic displays about the model's blind
    spots (its bias). Visitors can take a themed *tour* that picks exhibits
    matching a topic (e.g. "high-confidence errors", "edge cases"). A
    *gallery* is a room grouping exhibits by predicted class.
    """

    def __init__(self, conf_thresh: float = 0.5,
                 interesting_conf: float = 0.7,
                 seed: int = 0):
        self.conf_thresh = conf_thresh
        self.interesting_conf = interesting_conf
        self.rng = np.random.default_rng(seed)

    def curate(self, model: Any, inputs: np.ndarray,
               labels: np.ndarray) -> Dict[str, Any]:
        """Curate the museum: run the model over every input, classify each
        prediction as a permanent (correct) or special (wrong) exhibit,
        recording confidence, predicted class, and the input itself."""
        X = np.atleast_2d(np.asarray(inputs, dtype=np.float32))
        y = np.asarray(labels, dtype=np.int64).ravel()
        exhibits: List[Dict[str, Any]] = []
        for i, x in enumerate(X):
            pred = _predict(model, x)
            if pred is None:
                continue
            pa = np.asarray(pred, dtype=np.float64).ravel()
            pred_class = _argmax_pred(pa)
            conf = _confidence(pa)
            true_class = int(y[i]) if i < y.size else -1
            correct = (pred_class == true_class)
            exhibits.append({"id": i, "input_id": i,
                             "predicted_class": pred_class,
                             "true_class": true_class,
                             "confidence": conf,
                             "logits": pa.tolist(),
                             "correct": correct,
                             "kind": "permanent" if correct else "special"})
        return {"exhibits": exhibits, "n_exhibits": len(exhibits),
                "n_permanent": sum(1 for e in exhibits if e["correct"]),
                "n_special": sum(1 for e in exhibits if not e["correct"]),
                "curated_at": time.time()}

    def galleries(self, museum: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Group exhibits into galleries -- one per predicted class. Each
        gallery has a name, a count, and a mean exhibit confidence."""
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        for e in museum.get("exhibits", []):
            by_class.setdefault(e["predicted_class"], []).append(e)
        galleries: List[Dict[str, Any]] = []
        for cls, items in sorted(by_class.items()):
            confs = [it["confidence"] for it in items]
            galleries.append({"name": f"Gallery of Class {cls}",
                              "class": cls,
                              "n_exhibits": len(items),
                              "mean_confidence": float(np.mean(confs))
                                                 if confs else 0.0,
                              "n_permanent": sum(1 for it in items
                                                 if it["correct"]),
                              "n_special": sum(1 for it in items
                                               if not it["correct"])})
        return galleries

    def permanent_collection(self,
                             museum: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the permanent collection: correct predictions, sorted by
        confidence (the most confident acquisitions first)."""
        items = [e for e in museum.get("exhibits", []) if e["correct"]]
        items.sort(key=lambda e: -e["confidence"])
        return items

    def special_exhibits(self,
                         museum: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the special exhibits: wrong predictions that are most
        instructive -- the model was highly confident yet wrong."""
        items = [e for e in museum.get("exhibits", [])
                 if not e["correct"]]
        items.sort(key=lambda e: -e["confidence"])
        return items

    def tour(self, museum: Dict[str, Any],
             theme: str = "high_confidence_errors") -> List[Dict[str, Any]]:
        """Take a themed tour of the museum. Supported themes:
        ``high_confidence_errors`` (wrong but confident -- most instructive),
        ``low_confidence_correct`` (right but unsure -- fragile knowledge),
        ``edge_cases`` (confidence near 0.5), ``all_permanent``, ``all_special``.
        """
        exhibits = museum.get("exhibits", [])
        if theme == "high_confidence_errors":
            picked = [e for e in exhibits if not e["correct"]
                      and e["confidence"] >= self.interesting_conf]
            picked.sort(key=lambda e: -e["confidence"])
        elif theme == "low_confidence_correct":
            picked = [e for e in exhibits if e["correct"]
                      and e["confidence"] < self.conf_thresh]
            picked.sort(key=lambda e: e["confidence"])
        elif theme == "edge_cases":
            picked = [e for e in exhibits
                      if abs(e["confidence"] - 0.5) < 0.1]
        elif theme == "all_permanent":
            picked = [e for e in exhibits if e["correct"]]
        elif theme == "all_special":
            picked = [e for e in exhibits if not e["correct"]]
        else:
            picked = list(exhibits)
        return picked


# ============================================================================
# Idea 161 -- NeuralCourtReporter
# ============================================================================
class NeuralCourtReporter:
    """A stenographer sits in on training and records every weight change as
    a verbatim court transcript. Each *testimony* is a single gradient
    update -- which layer spoke, which weight, how loudly (the magnitude of
    the change), and at what step. ``cross_reference`` finds every
    mention of a particular weight (the witness stand); ``relevance``
    scores how often a weight testified (its importance to the trial).
    The full *transcript* can be replayed step by step.
    """

    def __init__(self, change_thresh: float = 1e-5,
                 top_witnesses: int = 20,
                 seed: int = 0):
        self.change_thresh = change_thresh
        self.top_witnesses = top_witnesses
        self.rng = np.random.default_rng(seed)

    def transcribe(self, model: Any,
                   training_steps: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Transcribe a training run. ``training_steps`` is a list of
        ``{layer_name: ndarray}`` snapshots of the model's weights at each
        step (any two consecutive snapshots define one testimony). Returns
        a transcript: a list of per-step events with weight deltas."""
        if not training_steps:
            return {"events": [], "n_steps": 0, "weight_index": {},
                    "witnesses": []}
        # build a global weight index: (layer, flat_idx) -> weight_id
        first = training_steps[0]
        weight_index: Dict[Tuple[str, int], int] = {}
        wid = 0
        for layer in sorted(first):
            W0 = np.asarray(first[layer], dtype=np.float64)
            for fi in range(W0.size):
                weight_index[(layer, fi)] = wid
                wid += 1
        events: List[Dict[str, Any]] = []
        # per-weight mention count (for relevance scoring)
        mentions: Dict[int, int] = {}
        for step_idx in range(1, len(training_steps)):
            prev = training_steps[step_idx - 1]
            cur = training_steps[step_idx]
            step_events: List[Dict[str, Any]] = []
            for layer in sorted(cur):
                if layer not in prev:
                    continue
                Wp = np.asarray(prev[layer], dtype=np.float64).ravel()
                Wc = np.asarray(cur[layer], dtype=np.float64).ravel()
                if Wp.size != Wc.size:
                    continue
                delta = Wc - Wp
                # only record significant changes
                notable = np.where(np.abs(delta) >= self.change_thresh)[0]
                for fi in notable:
                    wid_v = weight_index.get((layer, int(fi)), -1)
                    mentions[wid_v] = mentions.get(wid_v, 0) + 1
                    step_events.append({"step": step_idx,
                                        "layer": layer,
                                        "flat_index": int(fi),
                                        "weight_id": wid_v,
                                        "delta": float(delta[fi]),
                                        "new_value": float(Wc[fi]),
                                        "magnitude": float(abs(delta[fi]))})
            events.extend(step_events)
        # top witnesses (most-changed weights)
        witnesses = sorted(mentions.items(), key=lambda kv: -kv[1])
        witnesses = [{"weight_id": w, "mentions": c}
                     for w, c in witnesses[: self.top_witnesses]]
        return {"events": events, "n_steps": len(training_steps),
                "weight_index": weight_index, "witnesses": witnesses,
                "total_mentions": int(sum(mentions.values()))}

    def testimony(self, transcript: Dict[str, Any],
                  step: int) -> str:
        """Render a verbatim testimony for a single training step: every
        weight that changed (with layer, index, and signed delta)."""
        events = [e for e in transcript.get("events", []) if e["step"] == step]
        if not events:
            return (f"Step {step}: the court is silent; no weight changed "
                    f"beyond threshold.")
        lines = [f"Step {step}: the following weights testified:"]
        # cap to a reasonable transcript length
        for e in events[:25]:
            lines.append(
                f"  - layer '{e['layer']}' weight #{e['flat_index']} "
                f"(id {e['weight_id']}) moved by {e['delta']:+.6f} "
                f"to {e['new_value']:+.6f}.")
        if len(events) > 25:
            lines.append(f"  ... and {len(events) - 25} further testimonies.")
        return "\n".join(lines)

    def cross_reference(self, transcript: Dict[str, Any],
                       weight_id: int) -> List[Dict[str, Any]]:
        """Cross-reference every mention of a weight (by its id) throughout
        the trial: the full sequence of events that touched it."""
        return [e for e in transcript.get("events", [])
                if e.get("weight_id") == weight_id]

    def relevance(self, transcript: Dict[str, Any],
                  weight_id: int) -> float:
        """Score a weight's relevance: its mention count normalised by the
        total number of testimonies in the transcript (0 = silent, 1 =
        testified at every step)."""
        events = transcript.get("events", [])
        if not events:
            return 0.0
        mentions = sum(1 for e in events if e.get("weight_id") == weight_id)
        n_steps = max(1, transcript.get("n_steps", 1))
        return float(mentions / n_steps)


# ============================================================================
# Idea 162 -- ModelArchaeologicalDig
# ============================================================================
class ModelArchaeologicalDig:
    """Dig through a model's training history like an archaeological site.
    Each epoch is a *stratum* (a layer of sediment). *Artifacts* are
    significant weight changes between strata (large deltas). *Fossils*
    are dead neurons -- units that were active in an early stratum but
    silent (always-zero activation) by the end. *Pottery shards* are
    partial features -- neurons that emerged in mid-training and persisted.
    ``carbon_date`` estimates which epoch an artifact came from by its
    depth in the dig (stratum index).
    """

    def __init__(self, artifact_quantile: float = 0.95,
                 dead_thresh: float = 1e-4,
                 seed: int = 0):
        self.artifact_quantile = artifact_quantile
        self.dead_thresh = dead_thresh
        self.rng = np.random.default_rng(seed)

    def excavate(self, model: Any,
                history: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Excavate the dig site. ``history`` is a list of per-epoch weight
        snapshots (oldest first). Each stratum records the per-layer weight
        norm and the artifacts found (large deltas vs the previous stratum)."""
        strata: List[Dict[str, Any]] = []
        for i, snap in enumerate(history):
            layer_norms = {ln: float(np.linalg.norm(np.asarray(w).ravel()))
                           for ln, w in snap.items()}
            artifacts: List[Dict[str, Any]] = []
            if i > 0:
                prev = history[i - 1]
                for layer in sorted(snap):
                    if layer not in prev:
                        continue
                    Wp = np.asarray(prev[layer], dtype=np.float64).ravel()
                    Wc = np.asarray(snap[layer], dtype=np.float64).ravel()
                    if Wp.size != Wc.size:
                        continue
                    delta = np.abs(Wc - Wp)
                    if delta.size == 0:
                        continue
                    q = float(np.quantile(delta, self.artifact_quantile))
                    notable = np.where(delta >= q)[0]
                    for fi in notable:
                        artifacts.append({"stratum": i, "layer": layer,
                                          "flat_index": int(fi),
                                          "delta_magnitude": float(delta[fi]),
                                          "depth": i})
            strata.append({"index": i, "layer_norms": layer_norms,
                           "n_artifacts": len(artifacts),
                           "artifacts": artifacts})
        # find fossils: neurons active early but dead by the final snapshot
        fossils: List[Dict[str, Any]] = []
        if len(history) >= 2:
            first_snap = history[0]
            last_snap = history[-1]
            for layer in sorted(first_snap):
                if layer not in last_snap:
                    continue
                Wf = np.asarray(first_snap[layer], dtype=np.float64)
                Wl = np.asarray(last_snap[layer], dtype=np.float64)
                if Wf.ndim != 2 or Wf.shape != Wl.shape:
                    continue
                # "active" in epoch 0: large column norm; "dead" in final:
                # column norm near zero
                col_first = np.linalg.norm(Wf, axis=0)
                col_last = np.linalg.norm(Wl, axis=0)
                for n in range(Wf.shape[1]):
                    if col_first[n] > self.dead_thresh * 10 \
                            and col_last[n] < self.dead_thresh:
                        fossils.append({"layer": layer, "neuron": n,
                                        "first_norm": float(col_first[n]),
                                        "last_norm": float(col_last[n]),
                                        "died_by": len(history) - 1})
        return {"strata": strata, "n_strata": len(strata),
                "fossils": fossils,
                "n_fossils": len(fossils),
                "excavated_at": time.time()}

    def strata(self, site: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the list of strata (one per epoch), with depth and
        artifact count for each."""
        return [{"index": s["index"],
                 "depth": s["index"],
                 "n_artifacts": s["n_artifacts"],
                 "total_norm": float(sum(s["layer_norms"].values()))}
                for s in site.get("strata", [])]

    def artifacts(self, stratum: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the artifacts (significant weight changes) found in a
        single stratum."""
        return list(stratum.get("artifacts", []))

    def carbon_date(self, artifact: Dict[str, Any]) -> str:
        """Carbon-date an artifact: estimate its epoch from its stratum
        index. Older artifacts lie deeper (lower stratum index = older)."""
        depth = int(artifact.get("stratum", artifact.get("depth", 0)))
        # qualitative dating by relative depth (caller can supply site depth)
        # here we just render a stylised date label
        if depth <= 0:
            return "epoch 0 (the founding stratum)"
        return (f"epoch {depth} (~{depth * 1000} 'years before present' "
                f"by sediment depth)")


# ============================================================================
# Idea 163 -- WeightAuction
# ============================================================================
class WeightAuction:
    """Auction off weights like an art auction. Each weight is a *lot* with
    a *provenance* (which layer + position it comes from), an *appraised
    value* (estimated from the gradient magnitude needed to displace it),
    and a *sale price* (its current magnitude). The auctioneer spends a
    fixed *budget* of "magnitude points" buying the most undervalued lots
    (high appraised value, low current price) -- these are the weights the
    model would be wisest to keep (reinforce). Unpurchased lots are
    candidates for pruning.
    """

    def __init__(self, appraisal_probe: int = 8,
                 budget_unit: float = 1e-2,
                 seed: int = 0):
        self.appraisal_probe = appraisal_probe
        self.budget_unit = budget_unit
        self.rng = np.random.default_rng(seed)

    def catalog(self, model: Any) -> Dict[str, Any]:
        """Build the auction catalog: every weight becomes a lot with
        provenance, current magnitude, and an appraised value (estimated
        by perturbing the weight and measuring confidence loss)."""
        weights = _weights_of(model)
        names = sorted(weights)
        if not names:
            return {"lots": [], "n_lots": 0, "total_magnitude": 0.0}
        first = np.asarray(weights[names[0]], dtype=np.float32)
        in_dim = first.shape[0] if first.ndim == 2 else first.size
        probe = self.rng.normal(0, 1, size=(self.appraisal_probe, in_dim)
                                ).astype(np.float32)
        base_preds = [_predict(model, p) for p in probe]
        base_conf = float(np.mean([_confidence(p) for p in base_preds
                                   if p is not None])) if base_preds else 0.0
        lots: List[Dict[str, Any]] = []
        for name in names:
            W = np.asarray(weights[name], dtype=np.float32)
            flat_abs = np.abs(W).ravel()
            # find the top-k magnitude weights per layer to appraise
            # (appraising every weight is O(n); we cap to keep it fast)
            k = min(20, flat_abs.size)
            top_idx = np.argsort(-flat_abs)[:k]
            for fi in top_idx:
                if W.ndim == 2:
                    r, c = np.unravel_index(fi, W.shape)
                    provenance = f"{name}[{r},{c}]"
                else:
                    r, c = int(fi), 0
                    provenance = f"{name}[{fi}]"
                saved = W.ravel()[fi].copy()
                # appraise: zero the weight, measure confidence drop
                Wf = W.ravel()
                Wf[fi] = 0.0
                W = Wf.reshape(W.shape)
                patched = {k2: (W if k2 == name else v2)
                           for k2, v2 in weights.items()}
                tmp = _MinimalModel(patched)
                perturbed_preds = [_predict(tmp, p) for p in probe]
                pert_conf = float(np.mean([_confidence(p)
                                           for p in perturbed_preds
                                           if p is not None])) \
                    if perturbed_preds else 0.0
                # restore
                Wf[fi] = saved
                W = Wf.reshape(W.shape)
                weights[name] = W
                appraised = float(max(0.0, base_conf - pert_conf))
                lots.append({"lot_id": f"{name}_{int(fi)}",
                             "layer": name, "row": int(r), "col": int(c),
                             "flat_index": int(fi),
                             "provenance": provenance,
                             "current_price": float(abs(saved)),
                             "appraised_value": appraised,
                             "undervaluation": float(appraised
                                                     - abs(saved))})
        return {"lots": lots, "n_lots": len(lots),
                "total_magnitude": float(sum(l["current_price"]
                                             for l in lots)),
                "baseline_confidence": base_conf}

    def appraise(self, lot: Dict[str, Any]) -> float:
        """Return a lot's appraised value (its estimated importance)."""
        return float(lot.get("appraised_value", 0.0))

    def auction(self, catalog: Dict[str, Any],
                budget: float = 1.0) -> List[Dict[str, Any]]:
        """Hold the auction: spend ``budget`` (in magnitude points) buying
        the most undervalued lots (highest appraised_value - current_price).
        Each purchased lot is one the model should *keep*; the rest are
        candidates for pruning."""
        lots = list(catalog.get("lots", []))
        # sort by undervaluation (most undervalued first)
        lots.sort(key=lambda l: -l["undervaluation"])
        purchased: List[Dict[str, Any]] = []
        spent = 0.0
        for lot in lots:
            price = lot["current_price"]
            if spent + price > budget:
                continue
            spent += price
            purchased.append({**lot, "acquired_for": price,
                              "spent_so_far": spent})
        return purchased

    def provenance(self, lot: Dict[str, Any]) -> str:
        """Render a lot's provenance as a stylised auction-house label."""
        return (f"Lot {lot.get('lot_id')}: from layer "
                f"'{lot.get('layer')}' position "
                f"({lot.get('row')}, {lot.get('col')}); "
                f"current price {lot.get('current_price', 0.0):.4f}; "
                f"appraised at {lot.get('appraised_value', 0.0):.4f}.")


# ============================================================================
# Idea 164 -- PredictionCartographer
# ============================================================================
class PredictionCartographer:
    """Map the prediction landscape. For a grid of probe inputs, the model's
    predicted class defines a *region*. *Plains* are regions where the
    model's predictions are consistent and confident (stable ground);
    *mountains* are regions of high variance -- small perturbations of the
    input flip the prediction (rough terrain). *Fault lines* are the
    explicit boundaries between predicted-class regions -- the loci where
    the prediction flips from one class to another. Useful for finding
    adversarially fragile zones in input space.
    """

    def __init__(self, n_perturbations: int = 8,
                 perturbation_scale: float = 0.05,
                 variance_thresh: float = 0.3,
                 seed: int = 0):
        self.n_perturbations = n_perturbations
        self.perturbation_scale = perturbation_scale
        self.variance_thresh = variance_thresh
        self.rng = np.random.default_rng(seed)

    def survey(self, model: Any,
               sample_inputs: np.ndarray) -> Dict[str, Any]:
        """Survey the landscape: for each sample input, run the model and
        ``n_perturbations`` noisy copies; record the predicted class,
        the confidence, and the variance of predictions across the
        perturbations (a proxy for terrain roughness)."""
        X = np.atleast_2d(np.asarray(sample_inputs, dtype=np.float32))
        regions: List[Dict[str, Any]] = []
        for i, x in enumerate(X):
            base_pred = _predict(model, x)
            base_class = _argmax_pred(base_pred) if base_pred is not None else -1
            base_conf = _confidence(base_pred) if base_pred is not None else 0.0
            classes = [base_class]
            confs = [base_conf]
            for _ in range(self.n_perturbations):
                xp = x + self.rng.normal(0, self.perturbation_scale,
                                          size=x.shape).astype(np.float32)
                pp = _predict(model, xp)
                if pp is None:
                    continue
                classes.append(_argmax_pred(pp))
                confs.append(_confidence(pp))
            # variance proxy: fraction of perturbations that flipped class
            flips = sum(1 for c in classes if c != base_class)
            var_proxy = flips / max(1, len(classes))
            regions.append({"id": i, "center": x.tolist(),
                            "predicted_class": base_class,
                            "confidence": base_conf,
                            "variance": var_proxy,
                            "class_set": sorted(set(classes))})
        return {"regions": regions, "n_regions": len(regions),
                "n_perturbations": self.n_perturbations}

    def plains(self, landscape: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return plains: regions with low variance (consistent
        predictions across perturbations)."""
        return [r for r in landscape.get("regions", [])
                if r["variance"] < self.variance_thresh]

    def mountains(self, landscape: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return mountains: high-variance regions where perturbations
        often flip the prediction."""
        return [r for r in landscape.get("regions", [])
                if r["variance"] >= self.variance_thresh]

    def fault_lines(self, landscape: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return fault lines: pairs of neighbouring regions (in input
        space) whose predicted classes differ -- the boundary between two
        prediction zones."""
        regions = landscape.get("regions", [])
        if len(regions) < 2:
            return []
        centers = np.asarray([r["center"] for r in regions],
                             dtype=np.float32)
        classes = [r["predicted_class"] for r in regions]
        lines: List[Dict[str, Any]] = []
        for i in range(len(regions)):
            # find the nearest other region
            dists = np.linalg.norm(centers - centers[i], axis=1)
            dists[i] = np.inf
            nearest = int(np.argmin(dists))
            if classes[i] != classes[nearest] and classes[i] >= 0 \
                    and classes[nearest] >= 0:
                lines.append({"from": regions[i]["id"],
                              "to": regions[nearest]["id"],
                              "class_from": classes[i],
                              "class_to": classes[nearest],
                              "distance": float(dists[nearest])})
        # dedupe (from,to) and (to,from)
        seen = set()
        unique: List[Dict[str, Any]] = []
        for ln in lines:
            key = tuple(sorted((ln["from"], ln["to"])))
            if key in seen:
                continue
            seen.add(key)
            unique.append(ln)
        return unique


# ============================================================================
# Idea 165 -- ModelBiographer
# ============================================================================
class ModelBiographer:
    """Write a biography of a model. The model's *birth* is its
    initialisation (the seed that defined its starting weights); *childhood*
    is early training (the first few epochs); *adolescence* is fine-tuning
    (rapid parameter change); *career* is deployment (the current frozen
    weights); *retirement* is archival. *Turning points* are epochs where
    the loss dropped sharply (or spiked). The model's *legacy* is a
    one-paragraph summary of what it learned to do well.
    """

    def __init__(self, turning_point_drop: float = 0.1,
                 seed: int = 0):
        self.turning_point_drop = turning_point_drop
        self.rng = np.random.default_rng(seed)

    def interview(self, model: Any,
                  training_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Interview the model and its training log. ``training_log`` is a
        list of per-epoch records: ``{"epoch": int, "loss": float,
        "weights": {layer: ndarray}}``. Returns a biography structured by
        life stage."""
        weights = _weights_of(model)
        n_params = int(sum(np.asarray(a).size for a in weights.values()))
        # birth: epoch 0 (or 'init' if log empty)
        birth = {"epoch": 0,
                 "initial_loss": (training_log[0]["loss"]
                                  if training_log else None),
                 "seed_signature": float(np.mean(
                     [float(np.mean(np.asarray(a).ravel()))
                      for a in weights.values()]) if weights else 0.0)}
        # childhood: first ~20% of training (early epochs)
        n_epochs = len(training_log)
        if n_epochs:
            childhood_end = max(1, n_epochs // 5)
            childhood_losses = [r["loss"] for r in
                                training_log[: childhood_end]]
            childhood = {"epoch_range": (0, childhood_end),
                         "loss_start": childhood_losses[0]
                                       if childhood_losses else None,
                         "loss_end": childhood_losses[-1]
                                      if childhood_losses else None}
        else:
            childhood = {"epoch_range": (0, 0), "loss_start": None,
                         "loss_end": None}
        # adolescence: the epoch of largest loss drop (rapid learning)
        adolescence: Dict[str, Any] = {"epoch": -1, "loss_drop": 0.0}
        for i in range(1, n_epochs):
            drop = training_log[i - 1]["loss"] - training_log[i]["loss"]
            if drop > adolescence["loss_drop"]:
                adolescence = {"epoch": i, "loss_drop": float(drop),
                               "loss_before": training_log[i - 1]["loss"],
                               "loss_after": training_log[i]["loss"]}
        # career: current weights (deployment snapshot)
        career = {"n_params": n_params,
                  "n_layers": len(weights),
                  "layer_norms": {k: float(np.linalg.norm(np.asarray(v)
                                                            .ravel()))
                                   for k, v in weights.items()},
                  "deployed_at": time.time()}
        return {"birth": birth, "childhood": childhood,
                "adolescence": adolescence, "career": career,
                "n_epochs": n_epochs,
                "final_loss": (training_log[-1]["loss"] if training_log
                               else None)}

    def chapters(self, biography: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the chapters of the biography (one per life stage)."""
        chapters: List[Dict[str, Any]] = []
        b = biography
        chapters.append({"title": "Chapter 1: Birth",
                         "content": (f"The model was initialised with a mean "
                                     f"weight signature of "
                                     f"{b['birth']['seed_signature']:+.4f} "
                                     f"across {b['career']['n_params']} "
                                     f"parameters in "
                                     f"{b['career']['n_layers']} layers.")})
        ch = b["childhood"]
        chapters.append({"title": "Chapter 2: Childhood",
                         "content": (f"Across epochs {ch['epoch_range'][0]}-"
                                     f"{ch['epoch_range'][1]}, loss moved "
                                     f"from {ch['loss_start']} to "
                                     f"{ch['loss_end']}.")})
        ad = b["adolescence"]
        chapters.append({"title": "Chapter 3: Adolescence",
                         "content": (f"The breakthrough came at epoch "
                                     f"{ad['epoch']}, when loss dropped by "
                                     f"{ad['loss_drop']:.4f} in a single "
                                     f"step (from {ad['loss_before']} to "
                                     f"{ad['loss_after']}).")})
        cr = b["career"]
        chapters.append({"title": "Chapter 4: Career",
                         "content": (f"Deployed with {cr['n_params']} "
                                     f"parameters across {cr['n_layers']} "
                                     f"layers; final loss "
                                     f"{b.get('final_loss')}.")})
        chapters.append({"title": "Chapter 5: Retirement",
                         "content": ("The model is now archived; its weights "
                                     "are preserved for future distillation "
                                     "and study.")})
        return chapters

    def turning_points(self, biography: Dict[str, Any],
                      training_log: Optional[List[Dict[str, Any]]] = None
                      ) -> List[Dict[str, Any]]:
        """Identify turning points: epochs where the loss dropped by more
        than ``turning_point_drop`` in a single step (the 'aha' moments)."""
        log = training_log or []
        if len(log) < 2:
            # fall back to the biography's adolescence event
            ad = biography.get("adolescence", {})
            if ad.get("epoch", -1) >= 0:
                return [{"epoch": ad["epoch"],
                         "loss_drop": ad["loss_drop"],
                         "loss_before": ad.get("loss_before"),
                         "loss_after": ad.get("loss_after")}]
            return []
        events: List[Dict[str, Any]] = []
        for i in range(1, len(log)):
            drop = log[i - 1]["loss"] - log[i]["loss"]
            if drop >= self.turning_point_drop:
                events.append({"epoch": i, "loss_drop": float(drop),
                               "loss_before": float(log[i - 1]["loss"]),
                               "loss_after": float(log[i]["loss"])})
        return events

    def legacy(self, biography: Dict[str, Any]) -> str:
        """Write the model's legacy: a one-paragraph epitaph summarising
        its life arc and what it leaves behind."""
        ad = biography.get("adolescence", {})
        cr = biography.get("career", {})
        final_loss = biography.get("final_loss")
        breakthrough = (f"Its defining moment came at epoch "
                        f"{ad.get('epoch', '?')}, when loss fell by "
                        f"{ad.get('loss_drop', 0.0):.4f}.") \
            if ad.get("epoch", -1) >= 0 else \
            "Its training was uneventful, a slow steady grind."
        return (f"This model lived across {biography.get('n_epochs', 0)} "
                f"epochs, finishing with {cr.get('n_params', 0)} parameters "
                f"and a final loss of {final_loss}. {breakthrough} "
                f"It leaves behind a weight stack in "
                f"{cr.get('n_layers', 0)} layers, preserved for "
                f"distillation. May its gradients rest in peace.")


# ============================================================================
# Idea 166 -- HallucinationSurgicalTheater
# ============================================================================
class HallucinationSurgicalTheater:
    """Operate on a hallucination like a surgeon. *Pre-op* diagnoses which
    neurons are most responsible for the hallucinated (wrong) output: the
    top-k neurons by activation when the bad input is run. *Anesthesia*
    temporarily suppresses those neurons (zeros them) without touching the
    rest of the model -- a reversible intervention. *Surgery* permanently
    ablates (zeros) the diagnosed neurons. *Post-op* verifies the
    hallucination is gone: the model's prediction on the same input now
    disagrees with the original hallucination.
    """

    def __init__(self, top_k: int = 5,
                 activation_thresh: float = 0.3,
                 seed: int = 0):
        self.top_k = top_k
        self.activation_thresh = activation_thresh
        self.rng = np.random.default_rng(seed)

    def preop(self, model: Any,
              hallucination_input: Any) -> Dict[str, Any]:
        """Pre-operative diagnosis: identify the neurons most active when
        the hallucination is produced (the surgical targets)."""
        x = np.asarray(hallucination_input, dtype=np.float32).ravel()
        acts = _forward_activations(model, x[None, :])
        pred = _predict(model, x)
        pred_class = _argmax_pred(pred) if pred is not None else -1
        # collect candidate neurons from every 2-D layer
        candidates: List[Dict[str, Any]] = []
        for name in sorted(acts):
            A = np.asarray(acts[name], dtype=np.float64).ravel()
            top = np.argsort(-np.abs(A))[: self.top_k]
            for n in top:
                candidates.append({"layer": name, "neuron": int(n),
                                   "activation": float(A[n]),
                                   "abs_activation": float(abs(A[n]))})
        candidates.sort(key=lambda c: -c["abs_activation"])
        targets = candidates[: self.top_k]
        return {"input": x.tolist(),
                "hallucinated_class": pred_class,
                "targets": targets,
                "n_targets": len(targets),
                "diagnosed_at": time.time()}

    def anesthetize(self, model: Any,
                    neuron_ids: List[Dict[str, Any]]) -> Any:
        """Anesthesia: temporarily zero the diagnosed neurons (returns a
        *copy* of the model with those weights suppressed; the original
        model is untouched)."""
        weights = {k: np.array(np.asarray(v), copy=True)
                   for k, v in _weights_of(model).items()}
        for t in neuron_ids:
            layer, neuron = t.get("layer"), t.get("neuron")
            if layer is None or neuron is None or layer not in weights:
                continue
            W = np.asarray(weights[layer], dtype=np.float32)
            if W.ndim == 2 and neuron < W.shape[1]:
                W[:, neuron] = 0.0
            elif W.ndim == 1 and neuron < W.size:
                W[neuron] = 0.0
            weights[layer] = W
        return _MinimalModel(weights)

    def operate(self, model: Any,
               diagnosis: Dict[str, Any]) -> Any:
        """Surgery: permanently ablate (zero) the diagnosed neurons in the
        given model (in-place). Returns the operated model."""
        weights = _weights_of(model)
        for t in diagnosis.get("targets", []):
            layer, neuron = t.get("layer"), t.get("neuron")
            if layer is None or neuron is None or layer not in weights:
                continue
            W = np.asarray(weights[layer], dtype=np.float32)
            if W.ndim == 2 and neuron < W.shape[1]:
                W[:, neuron] = 0.0
            elif W.ndim == 1 and neuron < W.size:
                W[neuron] = 0.0
            weights[layer] = W
        return _apply(model, weights)

    def postop(self, model: Any,
              hallucination_input: Any) -> bool:
        """Post-op check: is the hallucination gone? True if the model's
        prediction on the input no longer matches the original (hallucinated)
        argmax. Caller is expected to pass the *operated* model."""
        x = np.asarray(hallucination_input, dtype=np.float32).ravel()
        pred = _predict(model, x)
        if pred is None:
            return True  # no prediction = no hallucination
        # we don't have the original hallucination class here; we check
        # whether the model is now *unsure* (low confidence) -- a proxy for
        # the hallucination being suppressed. The caller can pass a
        # Diagnosis object instead for a strict check.
        conf = _confidence(pred)
        return bool(conf < 0.5)

    def postop_strict(self, model: Any,
                     diagnosis: Dict[str, Any]) -> bool:
        """Strict post-op: returns True iff the model's current argmax
        differs from the originally hallucinated class stored in the
        diagnosis."""
        x = np.asarray(diagnosis.get("input", []), dtype=np.float32).ravel()
        if x.size == 0:
            return True
        pred = _predict(model, x)
        if pred is None:
            return True
        new_class = _argmax_pred(pred)
        return bool(new_class != diagnosis.get("hallucinated_class", -1))


# ============================================================================
# Idea 167 -- WeightOceanography
# ============================================================================
class WeightOceanography:
    """Map a model's weights as an ocean. *Warm currents* are rows or
    columns of a 2-D weight matrix whose mean is strongly positive
    (signal flowing one way); *cold currents* are strongly negative
    (signal flowing the opposite way). *Gyres* are circular patterns:
    in a 2-D layer, a row whose weights alternate sign in a periodic
    pattern (a "swirl" of positive and negative flow). *Upwelling zones*
    are layers where many weights are near zero (a "quiet" region from
    which new structure can emerge during further training). Together
    these describe how signal circulates through the model.
    """

    def __init__(self, warm_thresh: float = 0.1,
                 cold_thresh: float = -0.1,
                 upwelling_quantile: float = 0.5,
                 gyre_period: int = 3,
                 seed: int = 0):
        self.warm_thresh = warm_thresh
        self.cold_thresh = cold_thresh
        self.upwelling_quantile = upwelling_quantile
        self.gyre_period = gyre_period
        self.rng = np.random.default_rng(seed)

    def survey(self, model: Any) -> Dict[str, Any]:
        """Survey the ocean: for each 2-D weight layer, compute per-row and
        per-column mean (the currents) and the fraction of near-zero
        weights (the upwelling potential)."""
        weights = _weights_of(model)
        layers: Dict[str, Dict[str, Any]] = {}
        for name in sorted(weights):
            W = np.asarray(weights[name], dtype=np.float32)
            if W.ndim != 2:
                continue
            row_means = W.mean(axis=1)
            col_means = W.mean(axis=0)
            abs_w = np.abs(W)
            near_zero_frac = float(np.mean(abs_w < 1e-3))
            layers[name] = {"shape": list(W.shape),
                            "row_means": row_means.tolist(),
                            "col_means": col_means.tolist(),
                            "near_zero_frac": near_zero_frac,
                            "global_mean": float(W.mean())}
        return {"layers": layers, "n_layers": len(layers)}

    def currents(self, ocean_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return all currents (warm and cold) across all layers. Each
        current is a row or column with a strongly-signed mean."""
        out: List[Dict[str, Any]] = []
        for name, info in ocean_map.get("layers", {}).items():
            for i, m in enumerate(info.get("row_means", [])):
                if m >= self.warm_thresh:
                    out.append({"layer": name, "axis": "row",
                                "index": i, "direction": "warm",
                                "temperature": float(m)})
                elif m <= self.cold_thresh:
                    out.append({"layer": name, "axis": "row",
                                "index": i, "direction": "cold",
                                "temperature": float(m)})
            for j, m in enumerate(info.get("col_means", [])):
                if m >= self.warm_thresh:
                    out.append({"layer": name, "axis": "col",
                                "index": j, "direction": "warm",
                                "temperature": float(m)})
                elif m <= self.cold_thresh:
                    out.append({"layer": name, "axis": "col",
                                "index": j, "direction": "cold",
                                "temperature": float(m)})
        out.sort(key=lambda c: -abs(c["temperature"]))
        return out

    def gyres(self, ocean_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find gyres: 2-D layers whose sign pattern is approximately
        periodic along one axis (a "swirl"). Detected by autocorrelating
        the sign of row means with a lag of ``gyre_period``."""
        gyres: List[Dict[str, Any]] = []
        for name, info in ocean_map.get("layers", {}).items():
            row_means = np.asarray(info.get("row_means", []),
                                   dtype=np.float32)
            if row_means.size < self.gyre_period * 2:
                continue
            signs = np.sign(row_means)
            lagged = signs[self.gyre_period:]
            front = signs[: -self.gyre_period]
            if front.size == 0:
                continue
            agreement = float(np.mean(lagged == front))
            if agreement >= 0.7:
                gyres.append({"layer": name, "axis": "row",
                              "period": self.gyre_period,
                              "agreement": agreement})
        return gyres

    def upwelling_zones(self,
                       ocean_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return upwelling zones: layers with a high fraction of near-zero
        weights -- quiet regions from which new structure can emerge."""
        zones: List[Dict[str, Any]] = []
        for name, info in ocean_map.get("layers", {}).items():
            frac = info.get("near_zero_frac", 0.0)
            if frac >= 0.5:
                zones.append({"layer": name,
                              "near_zero_frac": float(frac),
                              "shape": info.get("shape")})
        zones.sort(key=lambda z: -z["near_zero_frac"])
        return zones


# ============================================================================
# Idea 168 -- ModelConductor
# ============================================================================
class ModelConductor:
    """Conduct an orchestra of models like a symphony conductor. Each model
    is a *section* (strings, winds, percussion, ...) assigned by its
    weight statistics. The conductor *scores* an input: decides which
    section plays which "note" (the prediction component it contributes
    best). ``conduct`` produces a unified *performance* (a weighted ensemble
    prediction). ``balance`` reports each section's relative contribution,
    and ``crescendo`` boosts a chosen section (raises its weight in the
    ensemble) for the next performance.
    """

    SECTIONS = ("strings", "winds", "percussion", "brass", "keyboard")

    def __init__(self, agreement_thresh: float = 0.2,
                 default_tempo: float = 1.0,
                 seed: int = 0):
        self.agreement_thresh = agreement_thresh
        self.default_tempo = default_tempo
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _assign_section(weights: Dict[str, np.ndarray]) -> str:
        """Assign a model to an orchestral section based on its weight
        profile (depth + mean width)."""
        mats = [np.asarray(w) for w in weights.values()
                if np.asarray(w).ndim == 2]
        if not mats:
            return "percussion"
        depth = len(mats)
        mean_width = float(np.mean([min(m.shape) for m in mats]))
        if depth <= 2 and mean_width >= 64:
            return "strings"      # broad, expressive
        if depth >= 4 and mean_width <= 32:
            return "winds"        # narrow, agile
        if mean_width <= 16:
            return "percussion"   # small, sharp
        if depth >= 6:
            return "brass"         # deep, loud
        return "keyboard"          # balanced

    def assemble(self, models: List[Any]) -> Dict[str, Any]:
        """Assemble the orchestra: assign each model a section and an
        initial dynamic (volume) of 1.0."""
        players: List[Dict[str, Any]] = []
        section_members: Dict[str, List[int]] = {s: [] for s in self.SECTIONS}
        for i, m in enumerate(models):
            w = _weights_of(m)
            section = self._assign_section(w)
            n_params = int(sum(np.asarray(a).size for a in w.values()))
            players.append({"id": i, "model": m, "section": section,
                            "n_params": n_params,
                            "dynamic": 1.0,
                            "tempo": self.default_tempo})
            section_members.setdefault(section, []).append(i)
        return {"players": players, "n_players": len(players),
                "sections": section_members,
                "assembled_at": time.time()}

    def score(self, orchestra: Dict[str, Any],
              input: Any) -> Dict[str, Any]:
        """Score an input: each player's prediction becomes a "note"; the
        conductor records who plays what (the predicted class, confidence,
        and dynamic)."""
        x = np.asarray(input, dtype=np.float32).ravel()
        notes: List[Dict[str, Any]] = []
        for p in orchestra.get("players", []):
            pred = _predict(p["model"], x)
            if pred is None:
                continue
            pa = np.asarray(pred, dtype=np.float64).ravel()
            notes.append({"player_id": p["id"], "section": p["section"],
                          "dynamic": p["dynamic"],
                          "predicted_class": _argmax_pred(pa),
                          "confidence": _confidence(pa),
                          "logits": pa.tolist()})
        return {"input": x.tolist(), "notes": notes,
                "n_notes": len(notes),
                "tempo": self.default_tempo}

    def conduct(self, orchestra: Dict[str, Any],
               input: Any) -> Dict[str, Any]:
        """Conduct the performance: produce a unified ensemble prediction
        by averaging each player's logits weighted by its dynamic. Returns
        the final prediction and the per-section contribution."""
        score = self.score(orchestra, input)
        if not score["notes"]:
            return {"performance": [], "n_players": 0, "balance": {}}
        max_len = max(len(n["logits"]) for n in score["notes"])
        mat = np.zeros((len(score["notes"]), max_len), dtype=np.float64)
        dyn = np.zeros(len(score["notes"]), dtype=np.float64)
        secs: List[str] = []
        for i, n in enumerate(score["notes"]):
            v = np.asarray(n["logits"], dtype=np.float64)
            mat[i, : v.size] = v
            dyn[i] = n["dynamic"]
            secs.append(n["section"])
        weights_dyn = dyn / (dyn.sum() + 1e-8)
        performance = (mat * weights_dyn[:, None]).sum(axis=0)
        # per-section contribution
        section_contrib: Dict[str, float] = {}
        for s, w in zip(secs, weights_dyn):
            section_contrib[s] = section_contrib.get(s, 0.0) + float(w)
        return {"performance": performance.tolist(),
                "predicted_class": _argmax_pred(performance),
                "confidence": _confidence(performance),
                "balance": section_contrib,
                "n_players": len(score["notes"])}

    def balance(self, orchestra: Dict[str, Any]) -> Dict[str, Any]:
        """Return each section's total dynamic weight (how loud it is in
        the ensemble)."""
        per_section: Dict[str, float] = {s: 0.0 for s in self.SECTIONS}
        per_count: Dict[str, int] = {s: 0 for s in self.SECTIONS}
        for p in orchestra.get("players", []):
            s = p["section"]
            per_section[s] = per_section.get(s, 0.0) + p["dynamic"]
            per_count[s] = per_count.get(s, 0) + 1
        return {"dynamics": per_section, "counts": per_count,
                "total_dynamic": float(sum(per_section.values()))}

    def crescendo(self, orchestra: Dict[str, Any],
                 section: str,
                 factor: float = 1.5) -> Dict[str, Any]:
        """Boost a section's dynamic (volume) by ``factor``. Returns a new
        orchestra with the boosted section."""
        players = []
        for p in orchestra.get("players", []):
            np_ = dict(p)
            if p["section"] == section:
                np_["dynamic"] = float(p["dynamic"] * factor)
            players.append(np_)
        section_members = {s: [] for s in self.SECTIONS}
        for p in players:
            section_members.setdefault(p["section"], []).append(p["id"])
        return {"players": players, "n_players": len(players),
                "sections": section_members,
                "boosted_section": section,
                "boost_factor": factor,
                "assembled_at": orchestra.get("assembled_at", time.time())}


__all__ = [
    "ModelZooCurator",
    "PredictionMuseum",
    "NeuralCourtReporter",
    "ModelArchaeologicalDig",
    "WeightAuction",
    "PredictionCartographer",
    "ModelBiographer",
    "HallucinationSurgicalTheater",
    "WeightOceanography",
    "ModelConductor",
]
