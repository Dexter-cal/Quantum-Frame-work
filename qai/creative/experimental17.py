"""
qai.creative.experimental17 -- 10 more original creative ideas (part 17).

  Idea 169 -- ModelConstellation        : models as stars + asterisms + supernovae
  Idea 170 -- WeightGeology             : weight matrices as strata / veins / faults
  Idea 171 -- PredictionArchaeology     : excavate a single prediction like a dig
  Idea 172 -- ModelSymposium2           : philosophical debate among models
  Idea 173 -- NeuralWeatherService      : 5-day forecast of model behaviour
  Idea 174 -- WeightTectonics2          : weights as tectonic plates over training
  Idea 175 -- ModelStandUp2             : a model performs a longer stand-up set
  Idea 176 -- HallucinationDetective2   : serial-hallucination noir case files
  Idea 177 -- ModelBiography2           : chapters + photos + quotes + obituary
  Idea 178 -- PredictionCartographer2   : 3D prediction landscape (peaks/cliffs)

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
    the per-layer activation matrix (rows = samples, cols = neurons)."""
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


def _argmax_pred(pred: Any) -> int:
    """Return the argmax of |prediction| (the model's chosen class)."""
    try:
        a = np.asarray(pred, dtype=np.float64).ravel()
        if a.size == 0:
            return -1
        return int(np.argmax(np.abs(a)))
    except Exception:
        return -1


def _accuracy(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    """Mean accuracy of a model on (X, y) using argmax-of-|pred|."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    y = np.asarray(y, dtype=np.int64).ravel()
    if X.shape[0] == 0:
        return 0.0
    correct = 0
    n = 0
    for i, x in enumerate(X):
        pred = _predict(model, x)
        if pred is None:
            continue
        n += 1
        if _argmax_pred(pred) == int(y[i]) if i < y.size else False:
            correct += 1
    return float(correct / max(1, n))


def _embedding(model: Any, dim: int = 8) -> np.ndarray:
    """A small fixed-length signature of a model's weights (the "position"
    in any model-space embedding). Built from per-layer mean / std / norm."""
    weights = _weights_of(model)
    if not weights:
        return np.zeros(dim, dtype=np.float32)
    feats: List[float] = []
    for name in sorted(weights):
        w = np.asarray(weights[name], dtype=np.float64).ravel()
        if w.size == 0:
            continue
        feats.extend([float(w.mean()), float(w.std()),
                      float(np.linalg.norm(w)), float(w.size)])
    arr = np.asarray(feats, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(dim, dtype=np.float32)
    # pad / truncate to dim, then normalise
    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size))
    elif arr.size > dim:
        arr = arr[:dim]
    nrm = float(np.linalg.norm(arr)) + 1e-8
    return arr / nrm


# ============================================================================
# Idea 169 -- ModelConstellation
# ============================================================================
class ModelConstellation:
    """Map a population of models to a night sky. Each model becomes a *star*
    with a brightness (its accuracy), a colour (its architecture family),
    and a 2-D position (a learned embedding of its weight statistics).
    *Asterisms* are clusters of stars that always agree in their predictions
    -- groups that perform well together. *Supernovae* are stars whose
    accuracy suddenly collapsed (recent crashes). Charting the sky reveals
    which architecture families dominate the firmament.
    """

    COLOURS = ("blue", "white", "yellow", "red", "green")

    def __init__(self, embed_dim: int = 8,
                 bright_quantile: float = 0.7,
                 agree_thresh: float = 0.8,
                 crash_thresh: float = 0.3,
                 seed: int = 0):
        self.embed_dim = embed_dim
        self.bright_quantile = bright_quantile
        self.agree_thresh = agree_thresh
        self.crash_thresh = crash_thresh
        self.rng = np.random.default_rng(seed)

    def _colour(self, weights: Dict[str, np.ndarray]) -> str:
        """Assign a star colour by architecture family (depth/width)."""
        mats = [np.asarray(w) for w in weights.values()
                if np.asarray(w).ndim == 2]
        if not mats:
            return self.COLOURS[4]
        depth = len(mats)
        width = float(np.mean([min(m.shape) for m in mats]))
        if depth <= 2 and width >= 64:
            return self.COLOURS[0]  # blue giant -- shallow wide
        if depth >= 4 and width <= 32:
            return self.COLOURS[2]  # yellow dwarf -- deep narrow
        if width <= 16:
            return self.COLOURS[3]  # red dwarf -- tiny
        if depth >= 6:
            return self.COLOURS[1]  # white -- deep balanced
        return self.COLOURS[4]      # green -- exotic

    def chart(self, models: List[Any],
              eval_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
              ) -> Dict[str, Any]:
        """Chart the sky: project each model's embedding to a 2-D position
        via PCA of weight signatures, assign brightness from accuracy on
        ``eval_data`` (uniform if none supplied), and colour by family."""
        embs = np.stack([_embedding(m, self.embed_dim) for m in models]) \
            if models else np.zeros((0, self.embed_dim), dtype=np.float32)
        # PCA to 2-D (random orthogonal fallback if 1 model)
        if embs.shape[0] >= 2 and embs.shape[1] >= 2:
            centred = embs - embs.mean(axis=0, keepdims=True)
            u, s, vt = np.linalg.svd(centred, full_matrices=False)
            pts = centred @ vt[:2].T
        else:
            pts = self.rng.normal(0, 1, size=(embs.shape[0], 2))
        # normalise to [-1, 1] sky
        span = float(np.max(np.abs(pts))) + 1e-8
        pts = pts / span
        stars: List[Dict[str, Any]] = []
        for i, m in enumerate(models):
            w = _weights_of(m)
            colour = self._colour(w)
            if eval_data is not None:
                X, y = eval_data
                bright = _accuracy(m, X, y)
            else:
                bright = 0.5
            stars.append({"id": i, "model": m,
                          "position": pts[i].tolist(),
                          "brightness": float(bright),
                          "colour": colour,
                          "n_params": int(sum(np.asarray(a).size
                                              for a in w.values()))})
        return {"stars": stars, "n_stars": len(stars),
                "charted_at": time.time()}

    def stars(self, sky: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return all stars in the sky, sorted by descending brightness."""
        out = list(sky.get("stars", []))
        out.sort(key=lambda s: -s["brightness"])
        return out

    def asterisms(self, sky: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find asterisms: clusters of bright stars whose predictions agree
        across a shared probe. Two stars join the same asterism if their
        Euclidean distance in the sky is small AND their predictions agree
        (same argmax) on a random probe drawn from the embedding space."""
        st = sky.get("stars", [])
        if len(st) < 2:
            return []
        pts = np.asarray([s["position"] for s in st], dtype=np.float32)
        # build agreement graph: edge if close in sky + same prediction
        probe = self.rng.normal(0, 1, size=self.embed_dim).astype(np.float32)
        preds = []
        for s in st:
            p = _predict(s["model"], probe)
            preds.append(_argmax_pred(p) if p is not None else -1)
        n = len(st)
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        bright_q = float(np.quantile([s["brightness"] for s in st],
                                      self.bright_quantile)) if n > 0 else 0.0
        for i in range(n):
            if st[i]["brightness"] < bright_q:
                continue
            for j in range(i + 1, n):
                if st[j]["brightness"] < bright_q:
                    continue
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d > 1.0:
                    continue
                agree = (preds[i] == preds[j] and preds[i] >= 0)
                if not agree:
                    continue
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
        groups: Dict[int, List[int]] = {}
        for k in range(n):
            groups.setdefault(find(k), []).append(k)
        asterisms: List[Dict[str, Any]] = []
        for root, members in groups.items():
            if len(members) < 2:
                continue
            asterisms.append({"members": list(members),
                              "n_stars": len(members),
                              "mean_brightness": float(
                                  np.mean([st[m]["brightness"]
                                           for m in members])),
                              "colours": sorted({st[m]["colour"]
                                                 for m in members})})
        asterisms.sort(key=lambda a: -a["n_stars"])
        return asterisms

    def supernovae(self, sky: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return supernovae: stars whose brightness fell below the crash
        threshold (recently crashed models)."""
        return [s for s in sky.get("stars", [])
                if s["brightness"] < self.crash_thresh]


# ============================================================================
# Idea 170 -- WeightGeology
# ============================================================================
class WeightGeology:
    """Treat the weight stack as a column of geological strata. Each layer is
    a *rock formation* with its own statistics (the bedrock). *Mineral veins*
    are high-magnitude weights -- rare, valuable seams of dense signal that
    carry most of the layer's gradient flow. *Faults* are discontinuities
    between adjacent layers: places where the layer's mean / variance shifts
    sharply (a tectonic boundary). *Unconformities* are layers whose weight
    distribution is a statistical outlier among its peers -- a foreign
    formation that doesn't fit the surrounding geology.
    """

    def __init__(self, vein_quantile: float = 0.95,
                 fault_z_thresh: float = 2.0,
                 unconformity_z_thresh: float = 2.5,
                 seed: int = 0):
        self.vein_quantile = vein_quantile
        self.fault_z_thresh = fault_z_thresh
        self.unconformity_z_thresh = unconformity_z_thresh
        self.rng = np.random.default_rng(seed)

    def survey(self, model: Any) -> Dict[str, Any]:
        """Survey the geological column: for each layer record the mean,
        std, Frobenius norm, and the high-magnitude vein threshold. Also
        stores the per-layer vein-weight indices and values so that
        ``veins`` can be called on the map alone (without the model)."""
        weights = _weights_of(model)
        layers: List[Dict[str, Any]] = []
        names = sorted(weights)
        for k, name in enumerate(names):
            w = np.asarray(weights[name], dtype=np.float64).ravel()
            if w.size == 0:
                continue
            vein_q = float(np.quantile(np.abs(w), self.vein_quantile))
            mask = np.abs(w) >= vein_q
            vein_indices = np.where(mask)[0].tolist()
            vein_values = w[mask].tolist()
            layers.append({"index": k, "name": name,
                           "mean": float(w.mean()),
                           "std": float(w.std()),
                           "norm": float(np.linalg.norm(w)),
                           "n_weights": int(w.size),
                           "vein_threshold": vein_q,
                           "vein_indices": [int(i) for i in vein_indices],
                           "vein_values": [float(v) for v in vein_values]})
        return {"layers": layers, "n_layers": len(layers),
                "surveyed_at": time.time()}

    def veins(self, geo_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return mineral veins: clusters of high-magnitude weights within
        a layer. A vein is a contiguous run of indices (in flat order)
        whose weights are all above the layer's ``vein_quantile``
        magnitude; veins are reported per layer, sorted by total
        magnitude. Works off the survey map alone -- no model needed."""
        veins: List[Dict[str, Any]] = []
        for layer in geo_map.get("layers", []):
            name = layer["name"]
            indices = layer.get("vein_indices", [])
            values = layer.get("vein_values", [])
            if not indices:
                continue
            # contiguous runs of flat indices are individual veins
            runs: List[List[int]] = []
            run_vals: List[List[float]] = []
            for idx, val in zip(indices, values):
                if runs and idx == runs[-1][-1] + 1:
                    runs[-1].append(int(idx))
                    run_vals[-1].append(float(val))
                else:
                    runs.append([int(idx)])
                    run_vals.append([float(val)])
            for r_i, (run, vals) in enumerate(zip(runs, run_vals)):
                veins.append({"layer": name, "vein_id": r_i,
                              "n_weights": len(run),
                              "total_magnitude": float(np.sum(np.abs(vals))),
                              "mean_sign": float(np.mean(np.sign(vals)))})
        veins.sort(key=lambda v: -v["total_magnitude"])
        return veins

    def faults(self, geo_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return faults: discontinuities between adjacent layers, defined
        as a large z-score shift in either mean or std from one layer to
        the next."""
        layers = geo_map.get("layers", [])
        faults: List[Dict[str, Any]] = []
        if len(layers) < 2:
            return faults
        means = np.asarray([l["mean"] for l in layers])
        stds = np.asarray([l["std"] for l in layers])
        for i in range(1, len(layers)):
            # z-score of this layer's mean against running mean of prior
            prior_mean = float(np.mean(means[:i])) if i > 0 else 0.0
            prior_std = float(np.std(means[:i])) + 1e-8
            mean_shift = abs(means[i] - prior_mean) / prior_std
            prior_std_mean = float(np.mean(stds[:i])) if i > 0 else 0.0
            prior_std_std = float(np.std(stds[:i])) + 1e-8
            std_shift = abs(stds[i] - prior_std_mean) / prior_std_std
            score = float(max(mean_shift, std_shift))
            if score >= self.fault_z_thresh:
                faults.append({"between": (layers[i - 1]["name"],
                                            layers[i]["name"]),
                                "mean_shift_z": float(mean_shift),
                                "std_shift_z": float(std_shift),
                                "severity": score})
        faults.sort(key=lambda f: -f["severity"])
        return faults

    def unconformities(self,
                       geo_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return unconformities: layers that are statistical outliers
        among the whole stack (mean and std more than
        ``unconformity_z_thresh`` standard deviations from the column mean)."""
        layers = geo_map.get("layers", [])
        if len(layers) < 3:
            return []
        means = np.asarray([l["mean"] for l in layers])
        stds = np.asarray([l["std"] for l in layers])
        m_mean, m_std = float(means.mean()), float(means.std() + 1e-8)
        s_mean, s_std = float(stds.mean()), float(stds.std() + 1e-8)
        out: List[Dict[str, Any]] = []
        for i, layer in enumerate(layers):
            mz = abs(means[i] - m_mean) / m_std
            sz = abs(stds[i] - s_mean) / s_std
            if mz >= self.unconformity_z_thresh \
                    or sz >= self.unconformity_z_thresh:
                out.append({"layer": layer["name"], "index": i,
                            "mean_z": float(mz), "std_z": float(sz),
                            "outlier_score": float(max(mz, sz))})
        out.sort(key=lambda u: -u["outlier_score"])
        return out


# ============================================================================
# Idea 171 -- PredictionArchaeology
# ============================================================================
class PredictionArchaeology:
    """Excavate a single prediction like an archaeological dig. The *surface*
    is the model's output logits. Each deeper *stratum* is the activation
    pattern of an earlier layer, going all the way down to the input feature
    stratum (the bedrock). *Artifacts* in a stratum are its most salient
    neurons -- the units that lit up most strongly when this prediction was
    made. ``deepest_find`` returns the single most-salient artifact in the
    deepest non-input stratum, the figurative "treasure" of the dig.
    """

    def __init__(self, top_artifacts: int = 5,
                 artifact_quantile: float = 0.9,
                 seed: int = 0):
        self.top_artifacts = top_artifacts
        self.artifact_quantile = artifact_quantile
        self.rng = np.random.default_rng(seed)

    def excavate(self, model: Any, input: Any) -> Dict[str, Any]:
        """Excavate the dig: run a forward pass, capture every layer's
        activation as a stratum (surface = output, deep = input features).
        The deepest stratum is the input itself."""
        x = np.asarray(input, dtype=np.float32).ravel()
        acts = _forward_activations(model, x[None, :])
        pred = _predict(model, x)
        names = sorted(acts)
        # build strata: surface = prediction, then layers (deepest last),
        # then a final input-features stratum
        strata: List[Dict[str, Any]] = []
        if pred is not None:
            pa = np.asarray(pred, dtype=np.float64).ravel()
            strata.append({"name": "surface", "depth": 0,
                           "activations": pa.tolist(),
                           "n_units": int(pa.size)})
        for k, name in enumerate(names):
            a = np.asarray(acts[name], dtype=np.float64).ravel()
            strata.append({"name": name, "depth": k + 1,
                           "activations": a.tolist(),
                           "n_units": int(a.size)})
        strata.append({"name": "input_features",
                       "depth": len(strata),
                       "activations": x.astype(np.float64).tolist(),
                       "n_units": int(x.size)})
        return {"strata": strata, "n_strata": len(strata),
                "predicted_class": _argmax_pred(pred)
                                   if pred is not None else -1,
                "confidence": _confidence(pred)
                              if pred is not None else 0.0,
                "excavated_at": time.time()}

    def surface(self, excavation: Dict[str, Any]) -> Dict[str, Any]:
        """Return the surface prediction: the model's output stratum."""
        strata = excavation.get("strata", [])
        if not strata:
            return {}
        s = strata[0]
        return {"name": s["name"], "predicted_class":
                excavation.get("predicted_class", -1),
                "confidence": excavation.get("confidence", 0.0),
                "n_units": s["n_units"]}

    def strata(self, excavation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the list of strata from surface (depth 0) to bedrock
        (deepest)."""
        return [{"name": s["name"], "depth": s["depth"],
                 "n_units": s["n_units"]} for s in excavation.get("strata", [])]

    def artifacts(self, stratum: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the artifacts in a single stratum: its most-salient
        neurons (top-k by absolute activation)."""
        acts = np.asarray(stratum.get("activations", []), dtype=np.float64)
        if acts.size == 0:
            return []
        k = min(self.top_artifacts, acts.size)
        top = np.argsort(-np.abs(acts))[:k]
        out: List[Dict[str, Any]] = []
        for n in top:
            out.append({"neuron": int(n), "activation": float(acts[n]),
                        "abs_activation": float(abs(acts[n])),
                        "stratum": stratum.get("name")})
        out.sort(key=lambda a: -a["abs_activation"])
        return out

    def deepest_find(self, excavation: Dict[str, Any]) -> Dict[str, Any]:
        """Return the single most-salient artifact in the deepest non-input
        stratum -- the figurative 'treasure' of the dig."""
        strata = excavation.get("strata", [])
        if len(strata) < 2:
            return {}
        # skip the input-features stratum (the bedrock) -- deepest non-input
        # is the last layer before it
        deepest = strata[-2] if strata[-1]["name"] == "input_features" \
            else strata[-1]
        arts = self.artifacts(deepest)
        if not arts:
            return {}
        a = arts[0]
        a["found_at_depth"] = deepest["depth"]
        a["stratum"] = deepest["name"]
        return a


# ============================================================================
# Idea 172 -- ModelSymposium2
# ============================================================================
class ModelSymposium2:
    """A philosophical gathering where models debate the nature of their own
    predictions. Each model takes a *stance* -- realist (trusts the raw
    logits), constructivist (foregrounds uncertainty), pragmatist (averages
    with peers) -- assigned by its weight statistics. They cross-examine
    each other's predictions, flag disagreements, and try to reach a
    *consensus* resolution via majority vote weighted by per-model
    confidence. The symposium's transcripts expose both the predictions
    and the philosophical framing of each speaker.
    """

    STANCES = ("realist", "constructivist", "pragmatist")

    def __init__(self, conf_split: float = 0.4,
                 high_conf_thresh: float = 0.6,
                 seed: int = 0):
        self.conf_split = conf_split
        self.high_conf_thresh = high_conf_thresh
        self.rng = np.random.default_rng(seed)

    def _assign_stance(self, weights: Dict[str, np.ndarray]) -> str:
        """Assign a philosophical stance by weight-magnitude profile:
        large weights => realist (confident), tiny weights => constructivist
        (uncertain), otherwise pragmatist (balanced)."""
        all_w = np.concatenate(
            [np.asarray(w, dtype=np.float64).ravel()
             for w in weights.values()]) if weights else np.zeros(1)
        mean_abs = float(np.mean(np.abs(all_w))) if all_w.size else 0.0
        if mean_abs >= 0.5:
            return "realist"
        if mean_abs <= 0.1:
            return "constructivist"
        return "pragmatist"

    def convene(self, models: List[Any]) -> Dict[str, Any]:
        """Convene the symposium: assign each model a stance and a seat."""
        seats: List[Dict[str, Any]] = []
        for i, m in enumerate(models):
            w = _weights_of(m)
            seats.append({"id": i, "model": m,
                          "stance": self._assign_stance(w),
                          "n_params": int(sum(np.asarray(a).size
                                              for a in w.values()))})
        by_stance: Dict[str, List[int]] = {s: [] for s in self.STANCES}
        for s in seats:
            by_stance.setdefault(s["stance"], []).append(s["id"])
        return {"seats": seats, "n_attendees": len(seats),
                "by_stance": by_stance,
                "convened_at": time.time()}

    def opening_statement(self, model: Any,
                          input: Any) -> Dict[str, Any]:
        """Render a model's opening statement: its raw prediction reframed
        by its stance. A realist states the argmax confidently; a
        constructivist emphasises the runner-up and uncertainty; a pragmatist
        reports the full distribution."""
        x = np.asarray(input, dtype=np.float32).ravel()
        pred = _predict(model, x)
        if pred is None:
            return {"stance": "constructivist", "logits": [],
                    "predicted_class": -1, "confidence": 0.0,
                    "statement": "(silence)"}
        pa = np.asarray(pred, dtype=np.float64).ravel()
        cls = _argmax_pred(pa)
        conf = _confidence(pa)
        stance = self._assign_stance(_weights_of(model))
        # second-place class
        order = np.argsort(-np.abs(pa))
        second = int(order[1]) if pa.size >= 2 else -1
        if stance == "realist":
            stmt = (f"I assert class {cls} with confidence {conf:.2f}; "
                    f"the evidence is plain and unambiguous.")
        elif stance == "constructivist":
            stmt = (f"Class {cls} leads, but class {second} is plausible; "
                    f"our confidence is merely {conf:.2f} and the picture "
                    f"is socially constructed from sparse logits.")
        else:  # pragmatist
            stmt = (f"On balance, class {cls} is the prudent choice "
                    f"(confidence {conf:.2f}); I would defer to the ensemble "
                    f"if my peers disagree.")
        return {"stance": stance, "logits": pa.tolist(),
                "predicted_class": cls, "confidence": conf,
                "runner_up": second, "statement": stmt}

    def cross_examine(self, statement: Dict[str, Any],
                      by_model: Any) -> Dict[str, Any]:
        """Cross-examine a statement with another model: highlight where the
        examiner disagrees (different argmax) and how much less / more
        confident it is. The rebuttal text is shaped by the examiner's stance."""
        # we don't have the input here -- re-run on a stored input if present
        x = np.asarray(statement.get("input", []), dtype=np.float32).ravel()
        if x.size == 0:
            # fall back: compare logits directly
            examiner_pred = statement.get("logits", [])
        else:
            examiner_pred = _predict(by_model, x)
        ep = np.asarray(examiner_pred, dtype=np.float64).ravel()
        ex_cls = _argmax_pred(ep)
        ex_conf = _confidence(ep)
        orig_cls = statement.get("predicted_class", -1)
        orig_conf = statement.get("confidence", 0.0)
        disagree = (ex_cls != orig_cls)
        stance = self._assign_stance(_weights_of(by_model))
        if disagree:
            text = (f"Objection: I see class {ex_cls} "
                    f"(confidence {ex_conf:.2f}), not class {orig_cls}. "
                    f"The original argument is unsound.")
        elif ex_conf > orig_conf + 0.1:
            text = (f"I agree on class {orig_cls}, but more firmly "
                    f"({ex_conf:.2f} vs {orig_conf:.2f}). The original "
                    f"understates the case.")
        elif ex_conf < orig_conf - 0.1:
            text = (f"I agree on class {orig_cls}, but with less conviction "
                    f"({ex_conf:.2f} vs {orig_conf:.2f}). The original is "
                    f"overconfident.")
        else:
            text = (f"I concur on class {orig_cls} with comparable "
                    f"confidence ({ex_conf:.2f}).")
        return {"examiner_stance": stance,
                "examiner_class": ex_cls, "examiner_confidence": ex_conf,
                "disagrees": disagree,
                "rebuttal": text}

    def closing_arguments(self, symposium: Dict[str, Any],
                          input: Any) -> List[Dict[str, Any]]:
        """Render every attendee's closing argument on the same input --
        each model's final word, stance-flavoured as in opening_statement."""
        return [self.opening_statement(s["model"], input)
                for s in symposium.get("seats", [])]

    def consensus(self, symposium: Dict[str, Any],
                  input: Any) -> Dict[str, Any]:
        """Reach a consensus resolution: weight every attendee's logits by
        its confidence, average, and report the agreed-upon class plus the
        fraction of models that agreed. A divided symposium reports low
        agreement."""
        seats = symposium.get("seats", [])
        if not seats:
            return {"resolved_class": -1, "agreement": 0.0,
                    "n_voters": 0}
        logit_mat: List[np.ndarray] = []
        confs: List[float] = []
        classes: List[int] = []
        for s in seats:
            st = self.opening_statement(s["model"], input)
            lg = np.asarray(st.get("logits", []), dtype=np.float64)
            if lg.size == 0:
                continue
            logit_mat.append(lg)
            confs.append(max(st.get("confidence", 0.0), 1e-3))
            classes.append(st.get("predicted_class", -1))
        if not logit_mat:
            return {"resolved_class": -1, "agreement": 0.0, "n_voters": 0}
        max_len = max(len(l) for l in logit_mat)
        M = np.zeros((len(logit_mat), max_len), dtype=np.float64)
        for i, l in enumerate(logit_mat):
            M[i, : l.size] = l
        w = np.asarray(confs, dtype=np.float64)
        w = w / (w.sum() + 1e-8)
        ensemble = (M * w[:, None]).sum(axis=0)
        resolved = _argmax_pred(ensemble)
        agree = float(np.mean([1 if c == resolved else 0 for c in classes]))
        return {"resolved_class": int(resolved),
                "agreement": agree,
                "n_voters": len(classes),
                "ensemble_confidence": _confidence(ensemble)}


# ============================================================================
# Idea 173 -- NeuralWeatherService
# ============================================================================
class NeuralWeatherService:
    """A 5-day forecast for your model's behaviour. *Confidence fronts* are
    moving bands of high/low confidence across the input space. *Accuracy
    pressure systems* are clusters of inputs where the model is reliably
    right (high pressure) or unreliable (low pressure). *Hallucination
    advisories* are issued for inputs the model predicts confidently yet
    wrongly. *Drift warnings* fire when the model's recent accuracy has
    dropped vs its 30-day almanac average. The *almanac* summarises a month
    of behaviour like a farmer's almanac.
    """

    def __init__(self, forecast_days: int = 5,
                 drift_thresh: float = 0.1,
                 halluc_conf: float = 0.7,
                 seed: int = 0):
        self.forecast_days = forecast_days
        self.drift_thresh = drift_thresh
        self.halluc_conf = halluc_conf
        self.rng = np.random.default_rng(seed)

    def forecast(self, model: Any,
                 history: List[Dict[str, Any]],
                 days: int = 5) -> Dict[str, Any]:
        """Produce a ``days``-day forecast for the model. ``history`` is a
        list of daily records ``{"day": int, "accuracy": float,
        "mean_confidence": float, "n_hallucinations": int}``. The forecast
        extrapolates accuracy / confidence with a simple linear trend and
        flags upcoming advisories."""
        days = max(1, days)
        if not history:
            return {"days": [], "n_days": 0,
                    "issued_at": time.time()}
        accs = np.asarray([h.get("accuracy", 0.0) for h in history],
                          dtype=np.float64)
        confs = np.asarray([h.get("mean_confidence", 0.0) for h in history],
                           dtype=np.float64)
        halls = np.asarray([h.get("n_hallucinations", 0) for h in history],
                           dtype=np.float64)
        n = len(history)
        # linear trend slope (least-squares)
        t = np.arange(n, dtype=np.float64)
        def _slope(y: np.ndarray) -> float:
            if n < 2:
                return 0.0
            return float(np.polyfit(t, y, 1)[0])
        a_slope, c_slope, h_slope = _slope(accs), _slope(confs), _slope(halls)
        last_day = history[-1].get("day", n - 1)
        days_out: List[Dict[str, Any]] = []
        for d in range(1, days + 1):
            fa = float(np.clip(accs[-1] + a_slope * d, 0.0, 1.0))
            fc = float(np.clip(confs[-1] + c_slope * d, 0.0, 1.0))
            fh = max(0, float(halls[-1] + h_slope * d))
            days_out.append({"day": last_day + d,
                             "forecast_accuracy": fa,
                             "forecast_confidence": fc,
                             "forecast_hallucinations": fh})
        return {"days": days_out, "n_days": len(days_out),
                "trends": {"accuracy_slope": a_slope,
                            "confidence_slope": c_slope,
                            "hallucination_slope": h_slope},
                "issued_at": time.time()}

    def pressure_map(self, model: Any,
                     X: Optional[np.ndarray] = None,
                     y: Optional[np.ndarray] = None,
                     n_probes: int = 32) -> Dict[str, Any]:
        """Build a pressure map: cluster probe inputs by (confidence,
        correctness) into high-pressure (confident + correct) and
        low-pressure (uncertain or wrong) zones. ``X`` / ``y`` are an
        evaluation set; if absent, random probes are drawn from a unit
        Gaussian sized to the model's input."""
        if X is None:
            weights = _weights_of(model)
            mats = [np.asarray(w) for w in weights.values()
                    if np.asarray(w).ndim == 2]
            in_dim = mats[0].shape[0] if mats else 8
            X = self.rng.normal(0, 1, size=(n_probes, in_dim)).astype(np.float32)
            y = None
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        zones: List[Dict[str, Any]] = []
        for i, x in enumerate(X):
            pred = _predict(model, x)
            if pred is None:
                continue
            cls = _argmax_pred(pred)
            conf = _confidence(pred)
            correct = (cls == int(y[i])) if (y is not None and i < len(y)) \
                else None
            pressure = "high" if (conf >= 0.6 and (correct is None or correct)) \
                else "low"
            zones.append({"id": i, "predicted_class": cls,
                          "confidence": conf, "correct": correct,
                          "pressure": pressure})
        n_high = sum(1 for z in zones if z["pressure"] == "high")
        return {"zones": zones, "n_zones": len(zones),
                "n_high_pressure": n_high,
                "n_low_pressure": len(zones) - n_high}

    def advisories(self, forecast: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Issue advisories from a forecast: hallucination advisory if
        forecast hallucinations are rising, drift warning if accuracy is
        falling, confidence-storm watch if confidence is volatile, and a
        fair-weather notice if everything is stable."""
        adv: List[Dict[str, Any]] = []
        trends = forecast.get("trends", {})
        days = forecast.get("days", [])
        if not days:
            return adv
        if trends.get("hallucination_slope", 0.0) > 0.0:
            adv.append({"kind": "hallucination_advisory",
                        "severity": min(1.0, trends["hallucination_slope"]),
                        "message": "Hallucinations trending upward."})
        if trends.get("accuracy_slope", 0.0) < -self.drift_thresh:
            adv.append({"kind": "drift_warning",
                        "severity": min(1.0,
                                        abs(trends["accuracy_slope"])),
                        "message": "Accuracy drifting downward."})
        if trends.get("confidence_slope", 0.0) > 0.1:
            adv.append({"kind": "confidence_storm_watch",
                        "severity": min(1.0, trends["confidence_slope"]),
                        "message": "Confidence rising rapidly."})
        if all(a.get("kind") != "drift_warning" for a in adv) \
                and all(a.get("kind") != "hallucination_advisory" for a in adv):
            adv.append({"kind": "fair_weather",
                        "severity": 0.0,
                        "message": "Model behaviour is stable."})
        return adv

    def almanac(self, model: Any,
                last_30_days: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a 30-day almanac summarising the model's recent behaviour:
        mean / min / max accuracy, mean confidence, total hallucinations,
        and the day with the worst accuracy (the 'coldest day')."""
        if not last_30_days:
            return {"n_days": 0, "mean_accuracy": 0.0,
                    "mean_confidence": 0.0, "total_hallucinations": 0}
        accs = np.asarray([h.get("accuracy", 0.0) for h in last_30_days])
        confs = np.asarray([h.get("mean_confidence", 0.0)
                            for h in last_30_days])
        halls = np.asarray([h.get("n_hallucinations", 0)
                            for h in last_30_days])
        worst_idx = int(np.argmin(accs))
        return {"n_days": len(last_30_days),
                "mean_accuracy": float(accs.mean()),
                "min_accuracy": float(accs.min()),
                "max_accuracy": float(accs.max()),
                "mean_confidence": float(confs.mean()),
                "total_hallucinations": int(halls.sum()),
                "coldest_day": int(last_30_days[worst_idx].get("day",
                                                                worst_idx)),
                "coldest_day_accuracy": float(accs[worst_idx])}


# ============================================================================
# Idea 174 -- WeightTectonics2
# ============================================================================
class WeightTectonics2:
    """Monitor a model's weights like tectonic plates over training. Each
    weight layer is a *plate* whose motion (mean drift) and stress (variance
    change) we track between epochs. *Plate boundaries* are layer interfaces.
    *Subduction zones* form where one plate's weight magnitude dominates its
    neighbour's (one layer's signal swallows the next). *Mountain building*
    is the cumulative growth of a layer's norm over time. *Seismic events*
    are sudden shifts -- large per-step changes that presage restructuring.
    """

    def __init__(self, subduction_ratio: float = 2.0,
                 seismic_quantile: float = 0.99,
                 seed: int = 0):
        self.subduction_ratio = subduction_ratio
        self.seismic_quantile = seismic_quantile
        self.rng = np.random.default_rng(seed)

    def survey(self, model: Any) -> Dict[str, Any]:
        """Survey the plates: each layer becomes a plate with mean, norm,
        and stress (variance)."""
        weights = _weights_of(model)
        names = sorted(weights)
        plates: List[Dict[str, Any]] = []
        for k, name in enumerate(names):
            w = np.asarray(weights[name], dtype=np.float64).ravel()
            if w.size == 0:
                continue
            plates.append({"name": name, "index": k,
                           "mean": float(w.mean()),
                           "norm": float(np.linalg.norm(w)),
                           "stress": float(w.std()),
                           "n_weights": int(w.size)})
        return {"plates": plates, "n_plates": len(plates),
                "surveyed_at": time.time()}

    def plates(self, tectonic_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the list of plates (one per layer)."""
        return list(tectonic_map.get("plates", []))

    def subduction_zones(self,
                         tectonic_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return subduction zones: pairs of adjacent plates where one
        plate's norm is at least ``subduction_ratio`` times the other's --
        the larger plate is subducting (dominating) the smaller."""
        plates = tectonic_map.get("plates", [])
        zones: List[Dict[str, Any]] = []
        for i in range(1, len(plates)):
            n_prev = plates[i - 1]["norm"]
            n_cur = plates[i]["norm"]
            if n_prev <= 0 or n_cur <= 0:
                continue
            ratio = max(n_prev / n_cur, n_cur / n_prev)
            if ratio >= self.subduction_ratio:
                dominant = plates[i - 1]["name"] if n_prev > n_cur \
                    else plates[i]["name"]
                zones.append({"between": (plates[i - 1]["name"],
                                           plates[i]["name"]),
                               "ratio": float(ratio),
                               "dominant": dominant,
                               "direction": "down" if n_prev > n_cur
                               else "up"})
        zones.sort(key=lambda z: -z["ratio"])
        return zones

    def seismic_risk(self, tectonic_map: Dict[str, Any]) -> float:
        """Score the overall seismic risk of the model: a heuristic combining
        the number of subduction zones and the inter-plate stress disparity.
        Higher = more likely to undergo restructuring under further training."""
        plates = tectonic_map.get("plates", [])
        if len(plates) < 2:
            return 0.0
        norms = np.asarray([p["norm"] for p in plates], dtype=np.float64)
        stresses = np.asarray([p["stress"] for p in plates],
                              dtype=np.float64)
        n_zones = len(self.subduction_zones(tectonic_map))
        norm_disparity = float(np.std(norms) / (np.mean(norms) + 1e-8))
        stress_disparity = float(np.std(stresses) / (np.mean(stresses) + 1e-8))
        risk = 0.4 * min(1.0, n_zones / max(1, len(plates) - 1)) \
            + 0.3 * min(1.0, norm_disparity) \
            + 0.3 * min(1.0, stress_disparity)
        return float(risk)

    def observe(self, model: Any,
                epoch_history: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Observe the model across epochs and produce a seismogram: per-step
        per-plate magnitude of motion (mean absolute weight delta) and a list
        of seismic events (steps whose max delta exceeds the global
        ``seismic_quantile``)."""
        if len(epoch_history) < 2:
            return {"steps": [], "n_steps": 0,
                    "seismic_events": []}
        steps: List[Dict[str, Any]] = []
        all_deltas: List[float] = []
        for t in range(1, len(epoch_history)):
            prev = epoch_history[t - 1]
            cur = epoch_history[t]
            plate_deltas: Dict[str, float] = {}
            max_delta = 0.0
            for name in sorted(cur):
                if name not in prev:
                    continue
                wp = np.asarray(prev[name], dtype=np.float64).ravel()
                wc = np.asarray(cur[name], dtype=np.float64).ravel()
                if wp.size != wc.size:
                    continue
                d = float(np.mean(np.abs(wc - wp)))
                plate_deltas[name] = d
                all_deltas.append(d)
                if d > max_delta:
                    max_delta = d
            steps.append({"step": t, "plate_deltas": plate_deltas,
                          "max_delta": max_delta})
        thresh = float(np.quantile(all_deltas, self.seismic_quantile)) \
            if all_deltas else 0.0
        events = [{"step": s["step"], "max_delta": s["max_delta"]}
                  for s in steps if s["max_delta"] >= thresh]
        return {"steps": steps, "n_steps": len(steps),
                "seismic_events": events,
                "n_events": len(events),
                "seismic_threshold": thresh}


# ============================================================================
# Idea 175 -- ModelStandUp2
# ============================================================================
class ModelStandUp2:
    """A model performs a longer stand-up set. Each *layer* is a *bit*; the
    neurons within it are the *punchlines* (the most-active units). The
    audience is the test set: a correct prediction is a laugh, a wrong one
    is a groan, and a low-confidence prediction is silence. ``tighten``
    drops the weakest bits (low-information layers); ``tight_five`` returns
    the best five bits -- the model's tightest, most reliable material.
    """

    def __init__(self, laugh_thresh: float = 0.6,
                 silence_thresh: float = 0.3,
                 top_bits: int = 5,
                 seed: int = 0):
        self.laugh_thresh = laugh_thresh
        self.silence_thresh = silence_thresh
        self.top_bits = top_bits
        self.rng = np.random.default_rng(seed)

    def prepare_set(self, model: Any) -> Dict[str, Any]:
        """Prepare the setlist: one bit per 2-D weight layer, with the
        layer's mean weight magnitude (the bit's 'material strength') and
        its neuron count (the punchline reservoir)."""
        weights = _weights_of(model)
        bits: List[Dict[str, Any]] = []
        for k, name in enumerate(sorted(weights)):
            w = np.asarray(weights[name], dtype=np.float64)
            if w.ndim != 2:
                continue
            bits.append({"bit_id": k, "layer": name,
                         "n_punchlines": int(w.shape[1]),
                         "material_strength":
                             float(np.mean(np.abs(w))),
                         "shape": list(w.shape)})
        return {"bits": bits, "n_bits": len(bits),
                "prepared_at": time.time(),
                "model": model}

    def perform(self, setlist: Dict[str, Any],
                input: Any) -> Dict[str, Any]:
        """Perform the set on a single input: run the forward pass and
        record each layer's top punchline (its most-active neuron) and the
        bit's 'energy' (mean absolute activation)."""
        model = setlist.get("model")
        if model is None:
            return {"bit_results": [], "n_bits": 0}
        x = np.asarray(input, dtype=np.float32).ravel()
        acts = _forward_activations(model, x[None, :])
        bit_results: List[Dict[str, Any]] = []
        for bit in setlist.get("bits", []):
            name = bit["layer"]
            if name not in acts:
                continue
            a = np.asarray(acts[name], dtype=np.float64).ravel()
            top_n = int(np.argmax(np.abs(a)))
            energy = float(np.mean(np.abs(a)))
            bit_results.append({"bit_id": bit["bit_id"], "layer": name,
                                "top_punchline": top_n,
                                "punchline_activation": float(a[top_n]),
                                "energy": energy})
        pred = _predict(model, x)
        return {"bit_results": bit_results,
                "n_bits": len(bit_results),
                "final_class": _argmax_pred(pred)
                                if pred is not None else -1,
                "final_confidence": _confidence(pred)
                                     if pred is not None else 0.0}

    def audience_reaction(self, bit: Dict[str, Any],
                          true_class: Optional[int] = None) -> Dict[str, Any]:
        """Translate a bit's final prediction into an audience reaction:
        laugh (correct + confident), groan (wrong + confident), silence
        (low confidence)."""
        cls = bit.get("final_class", -1)
        conf = bit.get("final_confidence", 0.0)
        correct = (cls == int(true_class)) if true_class is not None else None
        if conf < self.silence_thresh:
            reaction = "silence"
            message = "The audience hesitates; the bit landed softly."
        elif correct is True:
            reaction = "laugh"
            message = "Big laugh -- the punchline landed."
        elif correct is False:
            reaction = "groan"
            message = "Groans from the crowd; the bit misfired."
        else:
            # no ground truth: rate by confidence only
            reaction = "laugh" if conf >= self.laugh_thresh else "silence"
            message = ("Confident delivery; the crowd is amused."
                       if reaction == "laugh"
                       else "Muted response; the bit needs work.")
        return {"reaction": reaction, "confidence": conf,
                "correct": correct, "message": message}

    def tighten(self, setlist: Dict[str, Any]) -> Dict[str, Any]:
        """Tighten the setlist: drop the bits whose material strength is in
        the bottom quartile (the weakest, lowest-information layers)."""
        bits = list(setlist.get("bits", []))
        if len(bits) < 4:
            return {**setlist, "n_bits": len(bits),
                    "tightened": False}
        strengths = np.asarray([b["material_strength"] for b in bits])
        q25 = float(np.quantile(strengths, 0.25))
        kept = [b for b in bits if b["material_strength"] > q25]
        return {"bits": kept, "n_bits": len(kept),
                "prepared_at": setlist.get("prepared_at", time.time()),
                "model": setlist.get("model"),
                "tightened": True,
                "dropped_count": len(bits) - len(kept)}

    def tight_five(self, model: Any) -> Dict[str, Any]:
        """Return the model's tight five: the five bits with the highest
        material strength -- the tightest, most reliable material in the
        set. If the model has fewer than five layers, all are returned."""
        setlist = self.prepare_set(model)
        bits = list(setlist.get("bits", []))
        bits.sort(key=lambda b: -b["material_strength"])
        top = bits[: self.top_bits]
        return {"bits": top, "n_bits": len(top),
                "prepared_at": time.time(),
                "model": model,
                "tight_five": True}


# ============================================================================
# Idea 176 -- HallucinationDetective2
# ============================================================================
class HallucinationDetective2:
    """A noir detective sequel. Now investigating *serial* hallucinations --
    recurring patterns across many bad inputs. The detective builds a case
    file of every hallucination event (input, predicted class, activations),
    identifies the *modus operandi* (the input-space signature that
    triggers the hallucination), and rounds up the *accomplices*: neurons
    that always fire together across the hallucination set (high
    co-activation). The case is *closed* with a verdict naming the
    ringleader neuron and a recommended ablation.
    """

    def __init__(self, top_k: int = 10,
                 accomplice_corr: float = 0.85,
                 seed: int = 0):
        self.top_k = top_k
        self.accomplice_corr = accomplice_corr
        self.rng = np.random.default_rng(seed)

    def investigate(self, model: Any,
                    hallucination_inputs: List[Any]) -> Dict[str, Any]:
        """Investigate a series of hallucinations: for each input record its
        predicted class, confidence, and per-layer activations. Build a
        co-activation matrix across all hallucination events."""
        events: List[Dict[str, Any]] = []
        all_acts: Dict[str, List[np.ndarray]] = {}
        for i, x in enumerate(hallucination_inputs):
            xv = np.asarray(x, dtype=np.float32).ravel()
            pred = _predict(model, xv)
            if pred is None:
                continue
            cls = _argmax_pred(pred)
            conf = _confidence(pred)
            acts = _forward_activations(model, xv[None, :])
            event = {"id": i, "input": xv.tolist(),
                     "hallucinated_class": cls, "confidence": conf,
                     "activations": {n: np.asarray(a, dtype=np.float64)
                                        .ravel().tolist()
                                     for n, a in acts.items()}}
            events.append(event)
            for n, a in acts.items():
                all_acts.setdefault(n, []).append(
                    np.asarray(a, dtype=np.float64).ravel())
        return {"events": events, "n_events": len(events),
                "activations_by_layer": all_acts,
                "investigated_at": time.time()}

    def modus_operandi(self, case: Dict[str, Any]) -> str:
        """Describe the modus operandi: the typical predicted class across
        hallucinations and the mean input magnitude that triggers it."""
        events = case.get("events", [])
        if not events:
            return "No hallucinations on record; the streets are quiet."
        classes = [e["hallucinated_class"] for e in events]
        most_common = max(set(classes), key=classes.count)
        n_most = classes.count(most_common)
        mean_input_norm = float(np.mean(
            [np.linalg.norm(np.asarray(e["input"], dtype=np.float64))
             for e in events]))
        mean_conf = float(np.mean([e["confidence"] for e in events]))
        return (f"MO: the culprit favours class {most_common} "
                f"(in {n_most}/{len(events)} cases), triggered by inputs "
                f"with mean norm {mean_input_norm:.3f}; the model commits "
                f"the act at confidence {mean_conf:.2f} -- brazen.")

    def accomplices(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find accomplices: pairs of neurons (across any layer) whose
        activations are highly correlated across the hallucination set --
        they always fire together when the hallucination strikes."""
        acts_by_layer = case.get("activations_by_layer", {})
        # build a flat list of (layer, neuron) -> activation vector
        flat: List[Tuple[str, int, np.ndarray]] = []
        for layer, lst in acts_by_layer.items():
            if not lst:
                continue
            mat = np.stack(lst)  # (n_events, n_neurons)
            for n in range(mat.shape[1]):
                v = mat[:, n]
                if np.std(v) < 1e-6:
                    continue
                flat.append((layer, n, v))
        accomplices: List[Dict[str, Any]] = []
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                la, na, va = flat[i]
                lb, nb, vb = flat[j]
                if va.size != vb.size or va.size < 3:
                    continue
                cc = float(np.corrcoef(va, vb)[0, 1])
                if abs(cc) >= self.accomplice_corr:
                    accomplices.append({"neuron_a": (la, na),
                                        "neuron_b": (lb, nb),
                                        "correlation": cc})
        accomplices.sort(key=lambda a: -abs(a["correlation"]))
        return accomplices[: self.top_k * 2]

    def pattern(self, case: Dict[str, Any]) -> str:
        """Describe the recurring pattern of the serial hallucination: the
        dominant co-activations and the shared input signature."""
        acc = self.accomplices(case)
        events = case.get("events", [])
        if not events:
            return "No pattern detected."
        n_pairs = len(acc)
        # the layer where accomplices cluster most
        if acc:
            layers = [a["neuron_a"][0] for a in acc] + \
                     [a["neuron_b"][0] for a in acc]
            hot_layer = max(set(layers), key=layers.count)
        else:
            hot_layer = "(none)"
        classes = [e["hallucinated_class"] for e in events]
        most_common = max(set(classes), key=classes.count) if classes else -1
        return (f"Pattern: across {len(events)} hallucinations, the model "
                f"recurrently predicts class {most_common}; {n_pairs} "
                f"co-activating neuron pairs identified, with the densest "
                f"activity centred on layer '{hot_layer}'.")

    def close_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Close the case with a verdict: name the ringleader (the single
        neuron with highest mean activation across events) and recommend
        ablation as the sentence."""
        acts_by_layer = case.get("activations_by_layer", {})
        events = case.get("events", [])
        if not events or not acts_by_layer:
            return {"verdict": "case dismissed (no evidence)",
                    "ringleader": None,
                    "sentence": "release"}
        best: Optional[Tuple[str, int, float]] = None
        for layer, lst in acts_by_layer.items():
            if not lst:
                continue
            mat = np.stack(lst)
            means = np.mean(np.abs(mat), axis=0)
            top = int(np.argmax(means))
            score = float(means[top])
            if best is None or score > best[2]:
                best = (layer, top, score)
        if best is None:
            return {"verdict": "case dismissed (no suspect)",
                    "ringleader": None, "sentence": "release"}
        return {"verdict": (f"Guilty: neuron {best[1]} in layer "
                            f"'{best[0]}' (mean |activation|={best[2]:.3f}) "
                            f"is the ringleader of {len(events)} "
                            f"hallucination events."),
                "ringleader": {"layer": best[0], "neuron": best[1],
                               "mean_abs_activation": best[2]},
                "sentence": "ablate (zero the ringleader's weights)",
                "n_events": len(events)}


# ============================================================================
# Idea 177 -- ModelBiography2
# ============================================================================
class ModelBiography2:
    """A full biography of a model, with chapters, photos, quotes, and an
    obituary. *Chapters* segment the training log by life stage (birth,
    childhood, adolescence, prime, late career, retirement). *Photos* are
    ASCII architecture diagrams rendered from the weight shapes -- the
    model's portrait at different ages. *Quotes* are sample predictions
    drawn from the training log (the model's "voice"). The *obituary*
    summarises the model's life once it is retired.
    """

    def __init__(self, n_quotes: int = 5,
                 n_chapters: int = 6,
                 seed: int = 0):
        self.n_quotes = n_quotes
        self.n_chapters = n_chapters
        self.rng = np.random.default_rng(seed)

    def write(self, model: Any,
              training_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Write the biography: gather the model's current weights, the
        training log, and compute the life-stage boundaries."""
        weights = _weights_of(model)
        n_params = int(sum(np.asarray(a).size for a in weights.values()))
        n_epochs = len(training_log)
        losses = [r.get("loss", 0.0) for r in training_log]
        return {"weights": {k: np.asarray(v).shape for k, v in weights.items()},
                "n_params": n_params,
                "n_layers": len(weights),
                "n_epochs": n_epochs,
                "losses": losses,
                "training_log": training_log,
                "model": model,
                "written_at": time.time()}

    def chapters(self, biography: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the chapters of the biography: each chapter spans a slice
        of the training log and includes a short narrative paragraph."""
        n = biography.get("n_epochs", 0)
        losses = biography.get("losses", [])
        n_params = biography.get("n_params", 0)
        n_layers = biography.get("n_layers", 0)
        # divide the log into n_chapters segments (or fewer if log is short)
        k = max(1, min(self.n_chapters, max(1, n)))
        stage_names = ["Birth", "Childhood", "Adolescence",
                       "Prime", "Late Career", "Retirement"][: k]
        while len(stage_names) < k:
            stage_names.append(f"Stage {len(stage_names) + 1}")
        bounds = np.linspace(0, max(1, n), k + 1, dtype=int)
        chapters: List[Dict[str, Any]] = []
        for i in range(k):
            lo, hi = int(bounds[i]), int(bounds[i + 1])
            seg = losses[lo: hi]
            if seg:
                start_loss, end_loss = float(seg[0]), float(seg[-1])
                drop = float(start_loss - end_loss)
                narrative = (f"Epochs {lo}-{hi - 1}: loss moved "
                             f"{start_loss:.4f} -> {end_loss:.4f} "
                             f"(drop {drop:+.4f}).")
            else:
                narrative = (f"Epochs {lo}-{hi - 1}: no recorded "
                             f"activity in this stage.")
            chapters.append({"title": f"Chapter {i + 1}: {stage_names[i]}",
                             "epoch_range": (lo, max(lo, hi - 1)),
                             "narrative": narrative})
        # always end with a retirement chapter
        chapters.append({"title": f"Chapter {k + 1}: Retirement",
                         "epoch_range": (n, n),
                         "narrative": (f"The model is now archived with "
                                       f"{n_params} parameters across "
                                       f"{n_layers} layers. Its weights are "
                                       f"preserved for future study.")})
        return chapters

    def photos(self, biography: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Render the model's portrait: an ASCII architecture diagram of its
        weight shapes. Each layer is shown as a row of '#' markers sized by
        its output dimension (capped to 40 chars)."""
        shapes = biography.get("weights", {})
        out: List[Dict[str, Any]] = []
        for name, shape in shapes.items():
            shape = tuple(shape)
            out_dim = shape[1] if len(shape) >= 2 else (shape[0] if shape else 1)
            bar_len = max(1, min(40, int(out_dim)))
            bar = "#" * bar_len + "." * (40 - bar_len)
            ascii_art = f"[{name} {shape}]\n  |{bar}|  ({out_dim} units)"
            out.append({"layer": name, "shape": list(shape),
                        "out_dim": int(out_dim),
                        "ascii": ascii_art})
        return out

    def quotes(self, biography: Dict[str, Any],
               n: int = 5) -> List[Dict[str, Any]]:
        """Sample ``n`` quotes from the biography -- predictions the model
        made at evenly-spaced epochs throughout training. Each quote is the
        argmax class the model predicted on a probe drawn from the
        training log (or a random probe if none available)."""
        model = biography.get("model")
        n_epochs = biography.get("n_epochs", 0)
        n = max(1, n)
        if n_epochs == 0 or model is None:
            return []
        # pick evenly spaced epochs
        epochs = np.linspace(0, n_epochs - 1, n, dtype=int).tolist()
        # probe input: use a record's "input" if present, else random
        log = biography.get("training_log", [])
        quotes: List[Dict[str, Any]] = []
        for e in epochs:
            rec = log[e] if e < len(log) else {}
            x = rec.get("input")
            if x is None:
                # build a random probe sized to the model's first weight layer
                weights = _weights_of(model)
                mats = [np.asarray(w) for w in weights.values()
                        if np.asarray(w).ndim == 2]
                in_dim = mats[0].shape[0] if mats else 8
                x = self.rng.normal(0, 1, size=in_dim).astype(np.float32)
            pred = _predict(model, x)
            cls = _argmax_pred(pred) if pred is not None else -1
            conf = _confidence(pred) if pred is not None else 0.0
            quotes.append({"epoch": int(e),
                           "predicted_class": cls,
                           "confidence": conf,
                           "quote": (f"\"I say class {cls} "
                                     f"(confidence {conf:.2f}).\"")})
        return quotes

    def obituary(self, biography: Dict[str, Any]) -> str:
        """Write the model's obituary: a single paragraph summarising its
        life, peak achievement (lowest loss), and final form."""
        n_epochs = biography.get("n_epochs", 0)
        n_params = biography.get("n_params", 0)
        n_layers = biography.get("n_layers", 0)
        losses = biography.get("losses", [])
        if losses:
            peak_loss = float(min(losses))
            peak_epoch = int(np.argmin(losses))
            peak_str = (f"It reached its peak at epoch {peak_epoch} with a "
                        f"loss of {peak_loss:.4f}. ")
        else:
            peak_str = "Its training log was empty. "
        final_loss = float(losses[-1]) if losses else 0.0
        return (f"In memoriam: this model lived {n_epochs} epochs, "
                f"growing to {n_params} parameters across {n_layers} "
                f"layers. {peak_str}It finished its run with a final loss "
                f"of {final_loss:.4f}, and is now retired to archival. "
                f"May its gradients rest in peace.")


# ============================================================================
# Idea 178 -- PredictionCartographer2
# ============================================================================
class PredictionCartographer2:
    """Map the prediction landscape in 3-D: the x / y axes are the first two
    input features (or a 2-D PCA projection of higher-dim inputs), and z is
    the model's prediction confidence. *Peaks* are points where the model
    is correct and highly confident (firm high ground). *Valleys* are
    low-confidence points (foggy hollows). *Cliffs* are steep gradients in
    confidence -- places where the prediction flips within a small input
    perturbation. *Plateaus* are regions of uniform prediction across a
    neighbourhood (stable, well-mapped ground).
    """

    def __init__(self, grid_size: int = 20,
                 cliff_thresh: float = 0.4,
                 plateau_thresh: float = 0.1,
                 valley_thresh: float = 0.3,
                 peak_thresh: float = 0.7,
                 seed: int = 0):
        self.grid_size = grid_size
        self.cliff_thresh = cliff_thresh
        self.plateau_thresh = plateau_thresh
        self.valley_thresh = valley_thresh
        self.peak_thresh = peak_thresh
        self.rng = np.random.default_rng(seed)

    def survey(self, model: Any,
               sample_inputs: np.ndarray,
               grid_size: int = 20) -> Dict[str, Any]:
        """Survey the 3-D landscape: build a ``grid_size`` x ``grid_size``
        grid in the 2-D subspace spanned by the first two principal
        components of ``sample_inputs``, run the model at every grid
        point, and record confidence + predicted class as the elevation."""
        grid_size = max(2, grid_size)
        X = np.atleast_2d(np.asarray(sample_inputs, dtype=np.float32))
        if X.shape[0] < 2 or X.shape[1] < 2:
            # fall back: treat the first two columns directly
            base = X[:1].repeat(2, axis=0) if X.shape[0] >= 1 \
                else np.zeros((2, 2), dtype=np.float32)
            X = base
        # PCA to 2-D for grid axes
        centred = X - X.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centred, full_matrices=False)
        comps = vt[:2]  # (2, n_features)
        # grid in PC space
        proj = centred @ comps.T
        lo = proj.min(axis=0)
        hi = proj.max(axis=0)
        gx = np.linspace(lo[0], hi[0], grid_size)
        gy = np.linspace(lo[1], hi[1], grid_size)
        XX, YY = np.meshgrid(gx, gy)
        grid_points = np.stack([XX.ravel(), YY.ravel()], axis=1) \
            @ comps  # (grid**2, n_features)
        # add back the mean to land in input space
        grid_points = grid_points + X.mean(axis=0, keepdims=True)
        grid_points = grid_points.astype(np.float32)
        confs = np.zeros(grid_points.shape[0], dtype=np.float64)
        classes = -np.ones(grid_points.shape[0], dtype=np.int64)
        for i, p in enumerate(grid_points):
            pred = _predict(model, p)
            if pred is None:
                continue
            confs[i] = _confidence(pred)
            classes[i] = _argmax_pred(pred)
        conf_grid = confs.reshape(grid_size, grid_size)
        cls_grid = classes.reshape(grid_size, grid_size)
        return {"grid_size": grid_size,
                "conf_grid": conf_grid.tolist(),
                "class_grid": cls_grid.tolist(),
                "pc_axes": comps.tolist(),
                "mean": X.mean(axis=0).tolist(),
                "n_features": int(X.shape[1]),
                "surveyed_at": time.time(),
                "model": model,
                "sample_inputs": X.tolist()}

    def peaks(self, landscape: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return peaks: grid cells whose confidence exceeds
        ``peak_thresh`` AND whose predicted class matches the majority
        class of the nearest sample inputs (a correctness proxy)."""
        conf = np.asarray(landscape.get("conf_grid", []), dtype=np.float64)
        cls = np.asarray(landscape.get("class_grid", []), dtype=np.int64)
        n = conf.shape[0]
        if n == 0:
            return []
        # local maxima: cell >= all 8 neighbours by confidence
        peaks: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(conf.shape[1]):
                c = conf[i, j]
                if c < self.peak_thresh:
                    continue
                neigh = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < conf.shape[1]:
                            neigh.append(conf[ni, nj])
                if not neigh or c >= max(neigh):
                    peaks.append({"i": int(i), "j": int(j),
                                  "confidence": float(c),
                                  "predicted_class": int(cls[i, j])})
        peaks.sort(key=lambda p: -p["confidence"])
        return peaks

    def valleys(self, landscape: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return valleys: local minima of confidence (cells lower than all
        8 neighbours) with confidence below ``valley_thresh``."""
        conf = np.asarray(landscape.get("conf_grid", []), dtype=np.float64)
        cls = np.asarray(landscape.get("class_grid", []), dtype=np.int64)
        n = conf.shape[0]
        if n == 0:
            return []
        valleys: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(conf.shape[1]):
                c = conf[i, j]
                if c > self.valley_thresh:
                    continue
                neigh = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < conf.shape[1]:
                            neigh.append(conf[ni, nj])
                if not neigh or c <= min(neigh):
                    valleys.append({"i": int(i), "j": int(j),
                                    "confidence": float(c),
                                    "predicted_class": int(cls[i, j])})
        valleys.sort(key=lambda v: v["confidence"])
        return valleys

    def cliffs(self, landscape: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return cliffs: adjacent grid cells whose predicted classes
        differ -- the prediction flips across the boundary. Each cliff is
        reported as the (from, to) cell pair with the two classes."""
        cls = np.asarray(landscape.get("class_grid", []), dtype=np.int64)
        conf = np.asarray(landscape.get("conf_grid", []), dtype=np.float64)
        n = cls.shape[0]
        if n == 0:
            return []
        cliffs: List[Dict[str, Any]] = []
        seen = set()
        for i in range(n):
            for j in range(cls.shape[1]):
                for di, dj in ((1, 0), (0, 1)):
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < n and 0 <= nj < cls.shape[1]):
                        continue
                    if cls[i, j] != cls[ni, nj] and cls[i, j] >= 0 \
                            and cls[ni, nj] >= 0:
                        # also require a confidence jump to qualify
                        dc = abs(conf[i, j] - conf[ni, nj])
                        if dc >= self.cliff_thresh:
                            cliffs.append({"from": (int(i), int(j)),
                                           "to": (int(ni), int(nj)),
                                           "class_from": int(cls[i, j]),
                                           "class_to": int(cls[ni, nj]),
                                           "confidence_drop": float(dc)})
        cliffs.sort(key=lambda c: -c["confidence_drop"])
        return cliffs

    def plateaus(self, landscape: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return plateaus: connected regions of grid cells sharing the
        same predicted class with low confidence variance (stable ground).
        Each plateau is reported with its area and mean confidence."""
        cls = np.asarray(landscape.get("class_grid", []), dtype=np.int64)
        conf = np.asarray(landscape.get("conf_grid", []), dtype=np.float64)
        n = cls.shape[0]
        if n == 0:
            return []
        visited = np.zeros_like(cls, dtype=bool)
        plateaus: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(cls.shape[1]):
                if visited[i, j] or cls[i, j] < 0:
                    continue
                # BFS to find the connected region of equal class
                stack = [(i, j)]
                cells: List[Tuple[int, int]] = []
                confs: List[float] = []
                while stack:
                    ci, cj = stack.pop()
                    if (ci < 0 or ci >= n or cj < 0
                            or cj >= cls.shape[1] or visited[ci, cj]):
                        continue
                    if cls[ci, cj] != cls[i, j]:
                        continue
                    visited[ci, cj] = True
                    cells.append((ci, cj))
                    confs.append(float(conf[ci, cj]))
                    stack.extend([(ci + 1, cj), (ci - 1, cj),
                                  (ci, cj + 1), (ci, cj - 1)])
                if len(cells) < 2:
                    continue
                conf_var = float(np.var(confs))
                if conf_var <= self.plateau_thresh:
                    plateaus.append({
                        "predicted_class": int(cls[i, j]),
                        "area": len(cells),
                        "mean_confidence": float(np.mean(confs)),
                        "confidence_variance": conf_var,
                        "seed_cell": (int(i), int(j))})
        plateaus.sort(key=lambda p: -p["area"])
        return plateaus


__all__ = [
    "ModelConstellation",
    "WeightGeology",
    "PredictionArchaeology",
    "ModelSymposium2",
    "NeuralWeatherService",
    "WeightTectonics2",
    "ModelStandUp2",
    "HallucinationDetective2",
    "ModelBiography2",
    "PredictionCartographer2",
]
