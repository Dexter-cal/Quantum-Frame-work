"""
qai.creative.experimental15 -- 10 more original creative ideas (part 15).

  Idea 149 -- ModelAstronomy            : layers as celestial bodies + orbits + eclipses
  Idea 150 -- HallucinationDetective   : noir detective case-file for hallucinations
  Idea 151 -- NeuralGarden              : prune / water / weed model weights as plants
  Idea 152 -- ModelStandUp              : a model doing stand-up comedy (bits + punchlines)
  Idea 153 -- WeightArchitecture        : weights as a building (load-bearing, beams, foundation)
  Idea 154 -- PredictionForecast        : weather forecast for the next batch of predictions
  Idea 155 -- ModelThermostat           : auto-adjust model temperature during inference
  Idea 156 -- NeuralLibrary             : neurons as books, shelves, checkouts, bestsellers
  Idea 157 -- ModelSymposium            : models debate the nature of intelligence
  Idea 158 -- WeightCartography         : map weight landscape as terrain (peaks/valleys/ridges)

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


# ============================================================================
# Idea 149 -- ModelAstronomy
# ============================================================================
class ModelAstronomy:
    """Map each model layer to a celestial body in a solar system.

    The input layer is the *star* (the source of all signal); hidden layers
    are *planets* orbiting it; sub-layers (multi-dimensional weight blocks)
    are *moons*; layers with high sparsity are *asteroid belts* (dropout
    zones). Data flow traces *orbits* around the star; layers whose mean
    activation collapses to ~0 form *eclipses* that block signal. The
    *habitable zone* is the range of layer indices where activations are
    neither saturated nor dead -- the region where useful representations
    can form (the equivalent of liquid water).
    """

    def __init__(self, dead_thresh: float = 1e-3,
                 sat_thresh: float = 0.95,
                 sparsity_thresh: float = 0.5,
                 seed: int = 0):
        self.dead_thresh = dead_thresh
        self.sat_thresh = sat_thresh
        self.sparsity_thresh = sparsity_thresh
        self.rng = np.random.default_rng(seed)

    def observe(self, model: Any, probe: Optional[np.ndarray] = None,
                n_probe: int = 32) -> Dict[str, Any]:
        """Observe the model's solar system: classify each layer as a body.

        If no probe inputs are given, synthesise random Gaussian probes sized
        to the first layer's input dimension.
        """
        weights = _weights_of(model)
        names = sorted(weights)
        if not names:
            return {"star": None, "planets": [], "moons": [],
                    "asteroid_belts": [], "n_bodies": 0}
        # build probes
        first = np.asarray(weights[names[0]], dtype=np.float32)
        in_dim = first.shape[0] if first.ndim == 2 else first.size
        if probe is None:
            probe = self.rng.normal(0, 1, size=(n_probe, in_dim)).astype(np.float32)
        else:
            probe = np.atleast_2d(np.asarray(probe, dtype=np.float32))
        acts = _forward_activations(model, probe)
        bodies: List[Dict[str, Any]] = []
        for i, name in enumerate(names):
            W = np.asarray(weights[name], dtype=np.float32)
            A = acts.get(name)
            if A is None:
                mean_act = 0.0
                sparsity = 1.0
            else:
                mean_act = float(np.mean(np.abs(A)))
                sparsity = float(np.mean(np.abs(A) < self.dead_thresh))
            role = "planet"
            if i == 0:
                role = "star"
            elif W.ndim > 2 or (W.ndim == 2 and min(W.shape) <= 2):
                role = "moon"
            if sparsity >= self.sparsity_thresh:
                role = "asteroid_belt"
            bodies.append({"name": name, "role": role, "index": i,
                           "mean_activation": mean_act,
                           "sparsity": sparsity,
                           "magnitude": float(np.linalg.norm(W.ravel()))})
        star = next((b for b in bodies if b["role"] == "star"), None)
        planets = [b for b in bodies if b["role"] == "planet"]
        moons = [b for b in bodies if b["role"] == "moon"]
        belts = [b for b in bodies if b["role"] == "asteroid_belt"]
        return {"star": star, "planets": planets, "moons": moons,
                "asteroid_belts": belts, "bodies": bodies,
                "n_bodies": len(bodies)}

    def orbits(self, system: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trace orbits: data-flow paths from the star outward through
        successive planet layers. Each orbit is one planet's path from
        star -> itself, with the magnitude (orbital radius) = layer weight
        norm and period (steps to traverse) = layer index.
        """
        star = system.get("star")
        bodies = system.get("bodies", [])
        if star is None or not bodies:
            return []
        orbits: List[Dict[str, Any]] = []
        for b in bodies:
            if b["role"] in ("planet", "moon"):
                orbits.append({"body": b["name"], "role": b["role"],
                               "orbital_radius": b["magnitude"],
                               "period": b["index"],
                               "mean_activation": b["mean_activation"]})
        return orbits

    def eclipses(self, system: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect eclipses: layers whose mean activation collapses to near
        zero (dead neurons blocking the star's signal from reaching the
        outer planets).
        """
        eclipses: List[Dict[str, Any]] = []
        for b in system.get("bodies", []):
            if b["role"] == "star":
                continue
            if b["mean_activation"] < self.dead_thresh:
                eclipses.append({"body": b["name"], "index": b["index"],
                                 "type": "dead_layer",
                                 "severity": 1.0 - b["mean_activation"]})
            elif b["sparsity"] >= self.sparsity_thresh:
                eclipses.append({"body": b["name"], "index": b["index"],
                                 "type": "partial_eclipse",
                                 "severity": b["sparsity"]})
        return eclipses

    def habitable_zone(self, system: Dict[str, Any]) -> float:
        """Estimate the habitable zone: fraction of layers whose mean
        activation lies in the "Goldilocks" range (not dead, not saturated).
        Life (good representations) is most likely to form here.
        """
        bodies = [b for b in system.get("bodies", []) if b["role"] != "star"]
        if not bodies:
            return 0.0
        # treat each layer's mean |activation| (after tanh, so in [0,1]) as
        # its surface temperature; habitable = neither too cold nor too hot.
        # here mean_activation is already post-tanh magnitude.
        habitable = 0
        for b in bodies:
            ma = b["mean_activation"]
            if (ma > self.dead_thresh * 5
                    and ma < self.sat_thresh):
                habitable += 1
        return float(habitable / len(bodies))


# ============================================================================
# Idea 150 -- HallucinationDetective
# ============================================================================
class HallucinationDetective:
    """A noir detective investigating hallucinations case by case.

    For each hallucination, build a case file: the *crime scene* is the input
    that triggered the hallucination, *suspects* are the neurons most active
    during the wrong prediction, *witnesses* are the training samples whose
    internal representations most resemble the input (so the detective can
    ask "have you seen anything like this before?"), and *alibis* are the
    correct outputs the model produces for nearby inputs (showing the model
    *can* be right). The detective *interrogates* each suspect neuron by
    lesioning it and seeing whether the hallucination clears, and finally
    issues a *verdict* naming the neuron whose lesion most helped.
    """

    def __init__(self, conf_thresh: float = 0.7,
                 top_suspects: int = 5,
                 n_witnesses: int = 5,
                 n_alibis: int = 5,
                 seed: int = 0):
        self.conf_thresh = conf_thresh
        self.top_suspects = top_suspects
        self.n_witnesses = n_witnesses
        self.n_alibis = n_alibis
        self.rng = np.random.default_rng(seed)

    def investigate(self, model: Any, input: Any, hallucination: Any,
                    training_set: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Build the case file for a single hallucination.

        ``hallucination`` is the wrong prediction the model produced (any
        array-like). ``training_set`` (optional) is a sample of training
        inputs used to find witnesses.
        """
        x = np.asarray(input, dtype=np.float32).ravel()
        h = np.asarray(hallucination, dtype=np.float64).ravel()
        acts = _forward_activations(model, x[None, :])
        # suspects: neurons with highest |activation| across all layers
        suspects: List[Dict[str, Any]] = []
        for name in sorted(acts):
            A = np.asarray(acts[name], dtype=np.float64).ravel()
            top_idx = np.argsort(-np.abs(A))[: max(1, self.top_suspects)]
            for idx in top_idx:
                suspects.append({"layer": name, "neuron": int(idx),
                                 "activation": float(A[idx]),
                                 "confidence_at_time": _confidence(h)})
        # witnesses: training samples with closest representation to x
        witnesses: List[Dict[str, Any]] = []
        if training_set is not None and len(training_set) > 0:
            TS = np.atleast_2d(np.asarray(training_set, dtype=np.float32))
            ts_acts = _forward_activations(model, TS)
            # represent each training sample by its concatenated activations
            mats = [np.asarray(ts_acts[k], dtype=np.float32)
                    for k in sorted(ts_acts)]
            if mats:
                rep_ts = np.concatenate(mats, axis=1)
                mats_x = [np.asarray(acts[k], dtype=np.float32)
                          for k in sorted(acts)]
                rep_x = np.concatenate(mats_x, axis=1)
                # cosine similarity
                denom = (np.linalg.norm(rep_ts, axis=1)
                         * np.linalg.norm(rep_x, axis=1) + 1e-8)
                sims = (rep_ts @ rep_x.ravel()) / denom
                top_w = np.argsort(-sims)[: self.n_witnesses]
                for j in top_w:
                    witnesses.append({"id": int(j),
                                      "similarity": float(sims[j])})
        # alibis: nearby inputs (random perturbations) on which model is right
        alibis: List[Dict[str, Any]] = []
        for _ in range(self.n_alibis * 3):
            xp = x + self.rng.normal(0, 0.05, size=x.shape).astype(np.float32)
            pp = _predict(model, xp)
            if pp is None:
                continue
            pa = np.asarray(pp, dtype=np.float64).ravel()
            if np.argmax(np.abs(pa)) != np.argmax(np.abs(h)):
                alibis.append({"input": xp.tolist(),
                               "prediction": pa.tolist(),
                               "confidence": _confidence(pa)})
            if len(alibis) >= self.n_alibis:
                break
        return {"case_id": f"H-{int(time.time() * 1000) % 100000}",
                "crime_scene": x.tolist(),
                "hallucination": h.tolist(),
                "suspects": suspects, "witnesses": witnesses,
                "alibis": alibis,
                "opened_at": time.time()}

    def suspects(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the list of suspect neurons for the case (sorted by |act|)."""
        s = list(case.get("suspects", []))
        s.sort(key=lambda d: -abs(d.get("activation", 0.0)))
        return s

    def interrogate(self, suspect: Dict[str, Any]) -> Dict[str, Any]:
        """Interrogate a suspect: return its "statement" (a stylised record
        of its layer, neuron, and activation magnitude).
        """
        return {"layer": suspect.get("layer"),
                "neuron": suspect.get("neuron"),
                "statement": (f"Neuron {suspect.get('neuron')} in layer "
                              f"{suspect.get('layer')} fired at "
                              f"{suspect.get('activation', 0.0):.3f}. "
                              f"'I was just doing my job, detective.'"),
                "evidence_strength": float(abs(suspect.get("activation", 0.0)))}

    def solve(self, model: Any, case: Dict[str, Any],
              input: Optional[Any] = None) -> Dict[str, Any]:
        """Solve the case: lesion each suspect in turn (zero it), re-run
        the model on the input, and name the suspect whose removal most
        reduced confidence in the original (hallucinated) prediction.
        """
        x = np.asarray(input if input is not None
                       else case["crime_scene"], dtype=np.float32).ravel()
        h = np.asarray(case["hallucination"], dtype=np.float64).ravel()
        base_conf = _confidence(_predict(model, x) if _predict(model, x)
                                is not None else h)
        weights = _weights_of(model)
        best: Optional[Dict[str, Any]] = None
        best_drop = -1.0
        for s in case.get("suspects", []):
            layer, neuron = s.get("layer"), s.get("neuron")
            if layer is None or neuron is None or layer not in weights:
                continue
            W = np.asarray(weights[layer], dtype=np.float32)
            saved = W.copy()
            # zero the column (output neuron) if 2-D, else the element
            if W.ndim == 2 and neuron < W.shape[1]:
                W[:, neuron] = 0.0
            elif W.ndim == 1 and neuron < W.size:
                W[neuron] = 0.0
            else:
                continue
            patched = {k: (v if k != layer else W) for k, v in weights.items()}
            tmp_model = _MinimalModel(patched)
            pp = _predict(tmp_model, x)
            conf = _confidence(pp) if pp is not None else 0.0
            drop = float(base_conf - conf)
            if drop > best_drop:
                best_drop = drop
                best = {"layer": layer, "neuron": neuron,
                        "confidence_drop": drop,
                        "new_confidence": conf}
            # restore
            weights[layer] = saved
        if best is None:
            return {"verdict": "unsolved",
                    "culprit": None,
                    "reason": "no lesionable suspect found"}
        return {"verdict": "guilty" if best_drop > 0 else "innocent",
                "culprit": best,
                "confidence_drop": best_drop}


# ============================================================================
# Idea 151 -- NeuralGarden
# ============================================================================
class NeuralGarden:
    """Tend the model like a garden. Each layer is a garden bed; weights are
    plants. *Overgrown* plants (high magnitude) get pruned to keep the bed
    tidy; *dry* patches (tiny weights) get watered to encourage growth;
    *dead* neurons (always-zero output across probes) are weeded out.
    A *bloom report* summarises per-layer health: bloom (mean |weight|),
    growth (std), and weed count.
    """

    def __init__(self, prune_quantile: float = 0.95,
                 water_quantile: float = 0.1,
                 dead_thresh: float = 1e-4,
                 seed: int = 0):
        self.prune_quantile = prune_quantile
        self.water_quantile = water_quantile
        self.dead_thresh = dead_thresh
        self.rng = np.random.default_rng(seed)

    def plant(self, model: Any,
              probe: Optional[np.ndarray] = None,
              n_probe: int = 16) -> Dict[str, Any]:
        """Plant the garden: snapshot per-layer weights and per-neuron
        activation statistics (used later for weeding dead neurons)."""
        weights = _weights_of(model)
        names = sorted(weights)
        first = np.asarray(weights[names[0]], dtype=np.float32) if names else None
        beds: List[Dict[str, Any]] = []
        if first is None:
            return {"beds": [], "n_beds": 0, "weights": {}}
        in_dim = first.shape[0] if first.ndim == 2 else first.size
        if probe is None:
            probe = self.rng.normal(0, 1, size=(n_probe, in_dim)).astype(np.float32)
        probe = np.atleast_2d(np.asarray(probe, dtype=np.float32))
        acts = _forward_activations(model, probe)
        for name in names:
            W = np.asarray(weights[name], dtype=np.float32)
            A = acts.get(name)
            abs_w = np.abs(W)
            q_hi = float(np.quantile(abs_w, self.prune_quantile)) \
                if abs_w.size else 0.0
            q_lo = float(np.quantile(abs_w, self.water_quantile)) \
                if abs_w.size else 0.0
            # dead-neuron detection (only for 2-D layers: dead = output neuron)
            dead_neurons: List[int] = []
            if A is not None and W.ndim == 2:
                col_norms = np.linalg.norm(A, axis=0)
                dead_neurons = [int(i) for i, n in enumerate(col_norms)
                                if n < self.dead_thresh]
            beds.append({"layer": name,
                         "weight_shape": list(W.shape),
                         "mean_abs": float(np.mean(abs_w)) if abs_w.size else 0.0,
                         "std": float(np.std(W)) if W.size else 0.0,
                         "prune_threshold": q_hi,
                         "water_threshold": q_lo,
                         "dead_neurons": dead_neurons})
        return {"beds": beds, "n_beds": len(beds),
                "weights": {k: np.array(np.asarray(v), copy=True)
                            for k, v in weights.items()}}

    def prune(self, garden: Dict[str, Any],
              threshold: float = 0.8) -> Dict[str, Any]:
        """Prune overgrown plants: clip every weight whose magnitude is in
        the top ``threshold`` quantile down to that quantile's value
        (shrinks the heaviest weights without zeroing them)."""
        weights = garden["weights"]
        for name, W in weights.items():
            Wm = np.asarray(W, dtype=np.float32)
            abs_w = np.abs(Wm)
            if abs_w.size == 0:
                continue
            q = float(np.quantile(abs_w, threshold))
            mask = abs_w > q
            # clip magnitudes above q down to q (preserving sign)
            Wm = np.where(mask, np.sign(Wm) * q, Wm)
            weights[name] = Wm
        garden["weights"] = weights
        return garden

    def water(self, garden: Dict[str, Any],
              threshold: float = 0.1) -> Dict[str, Any]:
        """Water dry patches: boost every weight whose magnitude is below
        the ``threshold`` quantile up to that quantile's value (preserving
        sign), encouraging underused connections to grow."""
        weights = garden["weights"]
        for name, W in weights.items():
            Wm = np.asarray(W, dtype=np.float32)
            abs_w = np.abs(Wm)
            if abs_w.size == 0:
                continue
            q = float(np.quantile(abs_w, threshold))
            if q <= 0:
                continue
            mask = abs_w < q
            Wm = np.where(mask, np.sign(Wm) * q, Wm)
            weights[name] = Wm
        garden["weights"] = weights
        return garden

    def weed(self, garden: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify dead neurons across all beds (weeds to pull)."""
        weeds: List[Dict[str, Any]] = []
        for bed in garden.get("beds", []):
            for n in bed.get("dead_neurons", []):
                weeds.append({"layer": bed["layer"], "neuron": n,
                              "reason": "always-zero activation"})
        return weeds

    def bloom_report(self, garden: Dict[str, Any]) -> Dict[str, Any]:
        """Summarise the garden: per-bed bloom (mean |weight|), growth (std),
        and weed count. Overall health = mean bloom across beds."""
        beds = garden.get("beds", [])
        if not beds:
            return {"beds": [], "overall_bloom": 0.0, "n_weeds": 0}
        per_bed = []
        for b in beds:
            per_bed.append({"layer": b["layer"],
                            "bloom": b["mean_abs"],
                            "growth": b["std"],
                            "weeds": len(b["dead_neurons"])})
        overall = float(np.mean([p["bloom"] for p in per_bed]))
        n_weeds = int(sum(p["weeds"] for p in per_bed))
        return {"beds": per_bed, "overall_bloom": overall,
                "n_weeds": n_weeds}


# ============================================================================
# Idea 152 -- ModelStandUp
# ============================================================================
class ModelStandUp:
    """A model does stand-up comedy. Each layer is a *bit* in the routine;
    neurons fire as *punchlines*. The audience (the test set) laughs when
    the prediction is correct, groans when confidently wrong, and stays
    silent when the model is unconfident. *Timing* is inference latency,
    *delivery* is the peak confidence in the punchline. The comedian can
    *tighten* the routine by cutting weak bits (layers that consistently
    fail to land).
    """

    def __init__(self, conf_low: float = 0.2, conf_high: float = 0.7,
                 seed: int = 0):
        self.conf_low = conf_low
        self.conf_high = conf_high
        self.rng = np.random.default_rng(seed)

    def set(self, model: Any, test_X: np.ndarray) -> Dict[str, Any]:
        """Prepare the routine: run the model on the test set (audience),
        recording per-layer confidence (the bit's "energy") and per-sample
        correctness (the laugh).
        """
        X = np.atleast_2d(np.asarray(test_X, dtype=np.float32))
        weights = _weights_of(model)
        names = sorted(weights)
        per_layer_energy: Dict[str, List[float]] = {n: [] for n in names}
        laughs = 0
        groans = 0
        silence = 0
        for x in X:
            t0 = time.perf_counter()
            # forward through each layer, recording confidence at each
            cur = np.asarray(x, dtype=np.float32).ravel()
            for name in names:
                W = np.asarray(weights[name], dtype=np.float32)
                if W.ndim != 2:
                    continue
                if cur.size == W.shape[0]:
                    cur = np.tanh(cur @ W)
                elif cur.size == W.shape[1]:
                    cur = np.tanh(cur @ W.T)
                else:
                    continue
                per_layer_energy[name].append(_confidence(cur))
            elapsed = time.perf_counter() - t0
            conf = _confidence(cur)
            if conf >= self.conf_high:
                laughs += 1
            elif conf < self.conf_low:
                silence += 1
            else:
                groans += 1
        # per-bit energy summary
        bits: List[Dict[str, Any]] = []
        for name in names:
            energies = per_layer_energy[name]
            bits.append({"layer": name,
                         "mean_energy": float(np.mean(energies)) if energies else 0.0,
                         "variance": float(np.var(energies)) if energies else 0.0,
                         "n_jokes": len(energies)})
        return {"bits": bits, "audience_size": int(X.shape[0]),
                "laughs": laughs, "groans": groans, "silence": silence,
                "n_bits": len(names)}

    def perform(self, routine: Dict[str, Any], input: Any) -> Dict[str, Any]:
        """Perform the routine on a single input: time the per-bit delivery
        and pick the punchline (peak activation across all layers).
        """
        x = np.asarray(input, dtype=np.float32).ravel()
        t0 = time.perf_counter()
        weights_layers = {b["layer"]: b for b in routine["bits"]}
        cur = x
        punchlines: List[Tuple[str, float, float]] = []
        for name in sorted(weights_layers):
            # we no longer have the weights here; this method is purely
            # diagnostic of the routine's stored energy. We approximate
            # the punchline using the routine's mean energy.
            energy = weights_layers[name]["mean_energy"]
            punchlines.append((name, energy, time.perf_counter() - t0))
        timing_ms = (time.perf_counter() - t0) * 1000.0
        if not punchlines:
            return {"punchline_layer": None, "timing_ms": timing_ms,
                    "delivery": 0.0, "punchlines": []}
        best = max(punchlines, key=lambda p: p[1])
        return {"punchline_layer": best[0],
                "delivery": float(best[1]),
                "timing_ms": float(timing_ms),
                "punchlines": [{"layer": n, "energy": e, "t_ms": t * 1000.0}
                               for n, e, t in punchlines]}

    def crowd_reaction(self, joke: Dict[str, Any]) -> Dict[str, Any]:
        """Read the crowd: laughter (high delivery), groan (mid), silence
        (low). Returns the reaction label and intensity.
        """
        delivery = float(joke.get("delivery", 0.0))
        if delivery >= self.conf_high:
            reaction = "laughter"
            intensity = delivery
        elif delivery < self.conf_low:
            reaction = "silence"
            intensity = 1.0 - delivery
        else:
            reaction = "groan"
            intensity = 1.0 - abs(delivery - 0.5) * 2
        return {"reaction": reaction, "intensity": float(intensity)}

    def tighten(self, routine: Dict[str, Any],
               min_energy: float = 0.1) -> Dict[str, Any]:
        """Tighten the routine: cut weak bits (layers whose mean energy
        is below ``min_energy``).
        """
        kept = [b for b in routine["bits"] if b["mean_energy"] >= min_energy]
        cut = [b["layer"] for b in routine["bits"]
               if b["mean_energy"] < min_energy]
        new_routine = dict(routine)
        new_routine["bits"] = kept
        new_routine["n_bits"] = len(kept)
        new_routine["cut_bits"] = cut
        return new_routine


# ============================================================================
# Idea 153 -- WeightArchitecture
# ============================================================================
class WeightArchitecture:
    """Treat the weight matrices as a building. Each weight is a structural
    element: the *foundation* is the input layer (carries everything above),
    the *roof* is the output layer (caps the structure), *columns* are
    load-bearing weights (remove one and the building's confidence
    collapses), *beams* are the cross-layer connections that tie columns
    together. ``demolish`` simulates removing a single weight and measures
    the resulting stability loss; ``renovate`` applies a plan of removals
    and reinforcements.
    """

    def __init__(self, stability_metric: str = "confidence",
                 collapse_thresh: float = 0.05,
                 seed: int = 0):
        self.stability_metric = stability_metric
        self.collapse_thresh = collapse_thresh
        self.rng = np.random.default_rng(seed)

    def survey(self, model: Any, probe: Optional[np.ndarray] = None,
               n_probe: int = 8) -> Dict[str, Any]:
        """Survey the building: classify each layer (foundation/roof/wall)
        and record per-weight magnitudes (the structural load each carries).
        """
        weights = _weights_of(model)
        names = sorted(weights)
        if not names:
            return {"foundation": None, "roof": None,
                    "walls": [], "elements": [], "n_elements": 0}
        first = np.asarray(weights[names[0]], dtype=np.float32)
        in_dim = first.shape[0] if first.ndim == 2 else first.size
        if probe is None:
            probe = self.rng.normal(0, 1, size=(n_probe, in_dim)).astype(np.float32)
        probe = np.atleast_2d(np.asarray(probe, dtype=np.float32))
        base = _predict(model, probe[0]) if probe is not None else None
        base_conf = _confidence(base) if base is not None else 0.0
        elements: List[Dict[str, Any]] = []
        for i, name in enumerate(names):
            W = np.asarray(weights[name], dtype=np.float32)
            role = "wall"
            if i == 0:
                role = "foundation"
            elif i == len(names) - 1:
                role = "roof"
            flat = np.abs(W).ravel()
            if flat.size:
                top_idx = np.argsort(-flat)[:5]
                top = [{"index": int(j), "magnitude": float(flat[j]),
                        "row": int(np.unravel_index(j, W.shape)[0]) if W.ndim > 1 else int(j),
                        "col": int(np.unravel_index(j, W.shape)[1]) if W.ndim > 1 else 0}
                       for j in top_idx]
            else:
                top = []
            elements.append({"layer": name, "role": role,
                             "shape": list(W.shape),
                             "mean_load": float(np.mean(flat)) if flat.size else 0.0,
                             "max_load": float(np.max(flat)) if flat.size else 0.0,
                             "top_weights": top})
        foundation = next((e for e in elements if e["role"] == "foundation"), None)
        roof = next((e for e in elements if e["role"] == "roof"), None)
        walls = [e for e in elements if e["role"] == "wall"]
        return {"foundation": foundation, "roof": roof, "walls": walls,
                "elements": elements, "n_elements": len(elements),
                "baseline_stability": base_conf,
                "weights": {k: np.array(np.asarray(v), copy=True)
                            for k, v in weights.items()}}

    def load_bearing(self, blueprint: Dict[str, Any],
                     top_k: int = 10) -> List[Dict[str, Any]]:
        """Identify load-bearing weights: zero each candidate (top-magnitude
        per layer), re-run the model on a probe, and rank by stability loss.
        """
        weights = blueprint["weights"]
        # pick candidates: top-magnitude weights per layer
        candidates: List[Tuple[str, int, int]] = []
        for name, W in weights.items():
            Wm = np.asarray(W, dtype=np.float32)
            abs_w = np.abs(Wm)
            if abs_w.size == 0:
                continue
            flat_idx = int(np.argmax(abs_w))
            if Wm.ndim == 2:
                r, c = np.unravel_index(flat_idx, Wm.shape)
                candidates.append((name, int(r), int(c)))
            else:
                candidates.append((name, flat_idx, 0))
        base_conf = blueprint.get("baseline_stability", 0.5)
        # use a deterministic probe (the first SVD component of the foundation)
        first_name = sorted(weights)[0]
        first_w = np.asarray(weights[first_name], dtype=np.float32)
        if first_w.ndim == 2 and first_w.shape[0] > 0:
            u, s, _ = np.linalg.svd(first_w, full_matrices=False)
            probe = (u[:, 0] * (s[0] if s.size else 1.0)).astype(np.float32)
        else:
            probe = np.ones(first_w.shape[0] if first_w.ndim else 1,
                            dtype=np.float32)
        results: List[Dict[str, Any]] = []
        for name, r, c in candidates:
            saved = np.array(weights[name], copy=True)
            W = np.asarray(weights[name], dtype=np.float32)
            if W.ndim == 2:
                W[r, c] = 0.0
            elif W.ndim == 1 and r < W.size:
                W[r] = 0.0
            else:
                continue
            tmp = _MinimalModel({k: v for k, v in weights.items()})
            pp = _predict(tmp, probe)
            conf = _confidence(pp) if pp is not None else 0.0
            loss = float(base_conf - conf)
            results.append({"layer": name, "row": r, "col": c,
                            "stability_loss": loss,
                            "load_bearing": bool(loss > self.collapse_thresh)})
            weights[name] = saved
        results.sort(key=lambda d: -d["stability_loss"])
        return results[:top_k]

    def demolish(self, model: Any, weight: Dict[str, Any],
                 probe: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Demolish a single weight and report the resulting stability
        (confidence before vs after on the probe input)."""
        weights = _weights_of(model)
        layer = weight["layer"]
        r = int(weight["row"])
        c = int(weight["col"])
        if layer not in weights:
            return {"before": 0.0, "after": 0.0, "loss": 0.0,
                    "collapsed": False}
        W = np.asarray(weights[layer], dtype=np.float32)
        if probe is None:
            in_dim = W.shape[0] if W.ndim == 2 else W.size
            probe = np.ones(in_dim, dtype=np.float32)
        before = _confidence(_predict(model, probe))
        saved = W.copy()
        if W.ndim == 2:
            W[r, c] = 0.0
        elif W.ndim == 1:
            W[r] = 0.0
        _apply(model, {layer: W})
        after = _confidence(_predict(model, probe))
        _apply(model, {layer: saved})  # restore
        loss = float(before - after)
        return {"before": float(before), "after": float(after),
                "loss": loss, "collapsed": bool(loss > self.collapse_thresh)}

    def renovate(self, model: Any,
                 plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a renovation plan: ``plan`` should have ``remove`` (list of
        {layer, row, col}) and ``reinforce`` (dict layer->scale). Returns a
        report of applied changes.
        """
        weights = _weights_of(model)
        removed = 0
        for entry in plan.get("remove", []):
            layer = entry.get("layer")
            r = int(entry.get("row", 0))
            c = int(entry.get("col", 0))
            if layer not in weights:
                continue
            W = np.asarray(weights[layer], dtype=np.float32)
            if W.ndim == 2 and 0 <= r < W.shape[0] and 0 <= c < W.shape[1]:
                W[r, c] = 0.0
                removed += 1
            elif W.ndim == 1 and 0 <= r < W.size:
                W[r] = 0.0
                removed += 1
            weights[layer] = W
        reinforced = 0
        for layer, scale in plan.get("reinforce", {}).items():
            if layer not in weights:
                continue
            W = np.asarray(weights[layer], dtype=np.float32)
            weights[layer] = W * float(scale)
            reinforced += 1
        _apply(model, weights)
        return {"removed": removed, "reinforced": reinforced,
                "n_layers_touched": len(weights)}


# ============================================================================
# Idea 154 -- PredictionForecast
# ============================================================================
class PredictionForecast:
    """Weather forecast for predictions. Given the model's recent
    prediction history (confidence + correctness), forecast the next
    ``horizon`` batches: *sunny* (high-confidence correct), *cloudy*
    (low-confidence but plausibly correct), *stormy* (wrong). A
    *storm warning* is issued if a confidence storm is imminent; the
    overall *climate* characterises the model's long-run weather pattern.
    """

    def __init__(self, horizon: int = 10, window: int = 5,
                 sunny_conf: float = 0.7, cloudy_conf: float = 0.3,
                 storm_drop: float = 0.25, seed: int = 0):
        self.horizon = horizon
        self.window = window
        self.sunny_conf = sunny_conf
        self.cloudy_conf = cloudy_conf
        self.storm_drop = storm_drop
        self.rng = np.random.default_rng(seed)

    def forecast(self, model: Any, history: List[Dict[str, Any]],
                 horizon: int = 0) -> Dict[str, Any]:
        """Forecast the weather for the next ``horizon`` predictions.

        ``history`` is a list of ``{"confidence": float, "correct": bool}``
        records from recent batches.
        """
        h = horizon or self.horizon
        if not history:
            return {"horizon": h, "outlook": [], "summary": "no data",
                    "sunny_days": 0, "storm_warning": False}
        confs = np.asarray([rec.get("confidence", 0.5) for rec in history],
                           dtype=np.float64)
        accs = np.asarray([1.0 if rec.get("correct", False) else 0.0
                           for rec in history], dtype=np.float64)
        # fit a tiny AR(1) on confidence
        if confs.size >= 2:
            x_prev = confs[:-1]
            y_next = confs[1:]
            denom = float(np.dot(x_prev, x_prev)) + 1e-8
            a = float(np.clip(np.dot(x_prev, y_next) / denom, -0.99, 0.99))
            b = float(y_next.mean() - a * x_prev.mean())
            vol = float(np.std(y_next - (a * x_prev + b))) + 1e-4
        else:
            a, b, vol = 0.0, float(confs.mean()), 0.1
        # accuracy: simple average + drift
        acc_drift = float(np.mean(np.diff(accs))) if accs.size >= 2 else 0.0
        cur_conf = float(confs[-1])
        cur_acc = float(accs[-1]) if accs.size else 0.5
        outlook: List[Dict[str, Any]] = []
        for t in range(h):
            cur_conf = float(np.clip(a * cur_conf + b
                                     + self.rng.normal(0, vol), 0, 1))
            cur_acc = float(np.clip(cur_acc + acc_drift
                                    + self.rng.normal(0, 0.05), 0, 1))
            if cur_conf >= self.sunny_conf and cur_acc > 0.5:
                weather = "sunny"
            elif cur_conf < self.cloudy_conf:
                weather = "cloudy"
            elif cur_acc < 0.3:
                weather = "stormy"
            else:
                weather = "cloudy"
            outlook.append({"step": t, "confidence": cur_conf,
                            "accuracy": cur_acc, "weather": weather})
        return {"horizon": h, "outlook": outlook,
                "ar1": {"a": a, "b": b, "vol": vol}}

    def sunny_days(self, forecast: Dict[str, Any]) -> int:
        """Count sunny days in the forecast."""
        return sum(1 for o in forecast.get("outlook", [])
                   if o.get("weather") == "sunny")

    def storm_warning(self, forecast: Dict[str, Any]) -> bool:
        """Issue a storm warning if any forecast step is stormy OR if
        confidence drops by more than ``storm_drop`` between two steps.
        """
        outlook = forecast.get("outlook", [])
        if not outlook:
            return False
        if any(o.get("weather") == "stormy" for o in outlook):
            return True
        confs = [o["confidence"] for o in outlook]
        for i in range(1, len(confs)):
            if confs[i - 1] - confs[i] >= self.storm_drop:
                return True
        return False

    def climate(self, model: Any, test_X: np.ndarray) -> Dict[str, Any]:
        """Determine the model's long-run climate: classify it as
        tropical (high mean confidence), arctic (low), or temperate
        (moderate), with a wet/dry accuracy regime.
        """
        X = np.atleast_2d(np.asarray(test_X, dtype=np.float32))
        confs: List[float] = []
        for x in X:
            p = _predict(model, x)
            if p is None:
                continue
            confs.append(_confidence(p))
        if not confs:
            return {"mean_confidence": 0.0, "regime": "unknown",
                    "temperature": "unknown", "wetness": "unknown"}
        arr = np.asarray(confs)
        mean_c = float(arr.mean())
        std_c = float(arr.std())
        if mean_c >= self.sunny_conf:
            temp = "tropical"
        elif mean_c <= self.cloudy_conf:
            temp = "arctic"
        else:
            temp = "temperate"
        wetness = "wet" if std_c > 0.2 else "dry"
        regime = f"{temp}-{wetness}"
        return {"mean_confidence": mean_c, "std": std_c,
                "temperature": temp, "wetness": wetness, "regime": regime}


# ============================================================================
# Idea 155 -- ModelThermostat
# ============================================================================
class ModelThermostat:
    """Auto-adjust the model's "temperature" during inference. Too cold =
    overconfident (everything looks the same; needs sharpening). Too hot =
    hallucinating (chaotic outputs; needs cooling). The thermostat
    measures the current temperature (peak-to-mean ratio of logits),
    heats up (sharpen softmax via dividing logits by T < 1), cools down
    (soften via T > 1), and finds the *auto* optimal temperature that
    maximises a confidence-entropy trade-off. The *comfort zone* is the
    (low, high) temperature range in which the model is neither over- nor
    under-confident.
    """

    def __init__(self, cold_thresh: float = 0.8,
                 hot_thresh: float = 0.3,
                 t_min: float = 0.5, t_max: float = 2.0,
                 n_search: int = 12, seed: int = 0):
        self.cold_thresh = cold_thresh
        self.hot_thresh = hot_thresh
        self.t_min = t_min
        self.t_max = t_max
        self.n_search = n_search
        self.rng = np.random.default_rng(seed)

    def measure(self, model: Any, input: Any) -> Dict[str, Any]:
        """Measure the current temperature: the peak-to-mean ratio of the
        model's output logits (high ratio = cold/overconfident, low = hot).
        """
        pred = _predict(model, input)
        if pred is None:
            return {"temperature": 1.0, "condition": "normal",
                    "logits": []}
        a = np.asarray(pred, dtype=np.float64).ravel()
        if a.size == 0:
            return {"temperature": 1.0, "condition": "normal", "logits": []}
        peak = float(np.max(np.abs(a)))
        mean = float(np.mean(np.abs(a))) + 1e-8
        ratio = peak / mean  # in [1, size]; high = cold
        # normalise: temperature = 1 / ratio (cold = high ratio -> low T)
        temperature = float(np.clip(1.0 / ratio, 0.1, 5.0))
        if temperature < self.cold_thresh:
            condition = "cold (overconfident)"
        elif temperature > self.hot_thresh:
            condition = "hot (underconfident)"
        else:
            condition = "normal"
        return {"temperature": temperature, "condition": condition,
                "peak": peak, "mean": mean,
                "logits": a.tolist()}

    def heat_up(self, model: Any, amount: float = 0.1) -> Any:
        """Heat the model up: scale all weights by ``1 / (1 - amount)``
        (sharpening logits toward more confident predictions)."""
        if amount <= 0:
            return model
        weights = _weights_of(model)
        scale = 1.0 / max(0.1, 1.0 - amount)
        for name, W in weights.items():
            weights[name] = np.asarray(W, dtype=np.float32) * float(scale)
        return _apply(model, weights)

    def cool_down(self, model: Any, amount: float = 0.1) -> Any:
        """Cool the model down: scale all weights by ``(1 - amount)``
        (softening logits toward less confident predictions).
        """
        if amount <= 0:
            return model
        weights = _weights_of(model)
        scale = max(0.1, 1.0 - amount)
        for name, W in weights.items():
            weights[name] = np.asarray(W, dtype=np.float32) * float(scale)
        return _apply(model, weights)

    def auto(self, model: Any, inputs: np.ndarray) -> Dict[str, Any]:
        """Search for the optimal temperature: try several scaling factors,
        pick the one that maximises the confidence-entropy trade-off
        (high confidence + high entropy of the prediction distribution).
        """
        X = np.atleast_2d(np.asarray(inputs, dtype=np.float32))
        weights_orig = {k: np.array(np.asarray(v), copy=True)
                        for k, v in _weights_of(model).items()}

        def score(scale: float) -> float:
            weights = {k: np.asarray(v, dtype=np.float32) * float(scale)
                       for k, v in weights_orig.items()}
            _apply(model, weights)
            confs: List[float] = []
            entropies: List[float] = []
            for x in X:
                p = _predict(model, x)
                if p is None:
                    continue
                a = np.asarray(p, dtype=np.float64).ravel()
                if a.size == 0:
                    continue
                confs.append(_confidence(a))
                pos = np.abs(a) + 1e-8
                p_norm = pos / pos.sum()
                entropies.append(float(-np.sum(p_norm * np.log(p_norm + 1e-8))))
            _apply(model, weights_orig)  # restore
            if not confs:
                return 0.0
            mean_conf = float(np.mean(confs))
            mean_ent = float(np.mean(entropies))
            max_ent = float(np.log(max(2, len(a)))) if a.size else 1.0
            norm_ent = mean_ent / (max_ent + 1e-8)
            # trade-off: high confidence AND high entropy (the model is
            # confident yet uses all of its output capacity)
            return float(mean_conf * norm_ent)

        scales = np.linspace(self.t_min, self.t_max, self.n_search)
        scores = [score(s) for s in scales]
        best_idx = int(np.argmax(scores))
        best_scale = float(scales[best_idx])
        return {"optimal_scale": best_scale,
                "optimal_score": float(scores[best_idx]),
                "all_scales": scales.tolist(),
                "all_scores": scores,
                "interpretation": ("heated" if best_scale < 1.0
                                    else "cooled" if best_scale > 1.0
                                    else "kept")}

    def comfort_zone(self, model: Any,
                     probe: Optional[np.ndarray] = None,
                     n_probe: int = 8) -> Tuple[float, float]:
        """Find the comfort zone: the (low, high) scaling-factor range
        in which the model's mean confidence stays in the moderate band.
        """
        weights_orig = {k: np.array(np.asarray(v), copy=True)
                        for k, v in _weights_of(model).items()}
        first_name = sorted(weights_orig)[0] if weights_orig else None
        if first_name is None:
            return (1.0, 1.0)
        first_w = np.asarray(weights_orig[first_name], dtype=np.float32)
        in_dim = first_w.shape[0] if first_w.ndim == 2 else first_w.size
        if probe is None:
            probe = self.rng.normal(0, 1, size=(n_probe, in_dim)).astype(np.float32)
        probe = np.atleast_2d(np.asarray(probe, dtype=np.float32))
        scales = np.linspace(self.t_min, self.t_max, self.n_search)
        in_zone: List[float] = []
        for s in scales:
            weights = {k: np.asarray(v, dtype=np.float32) * float(s)
                       for k, v in weights_orig.items()}
            _apply(model, weights)
            confs = [_confidence(_predict(model, x)) for x in probe
                     if _predict(model, x) is not None]
            if confs:
                mc = float(np.mean(confs))
                if self.hot_thresh < mc < self.cold_thresh:
                    in_zone.append(float(s))
        _apply(model, weights_orig)
        if not in_zone:
            return (1.0, 1.0)
        return (float(min(in_zone)), float(max(in_zone)))


# ============================================================================
# Idea 156 -- NeuralLibrary
# ============================================================================
class NeuralLibrary:
    """Each neuron is a book in a library. Each layer is a shelf. Books are
    organised by topic (the input feature each neuron weights most heavily).
    *Checkouts* record how often a neuron fires above a threshold across
    the sample inputs; *overdue* books are neurons that haven't fired
    recently (their last-firing "age" is past the due date); *bestsellers*
    are the most-checked-out neurons. *Recommend* suggests which neurons to
    "read" for a given input (the most active on that input).
    """

    def __init__(self, fire_thresh: float = 0.3,
                 topic_top_k: int = 3,
                 seed: int = 0):
        self.fire_thresh = fire_thresh
        self.topic_top_k = topic_top_k
        self.rng = np.random.default_rng(seed)

    def catalog(self, model: Any,
                sample_inputs: np.ndarray) -> Dict[str, Any]:
        """Build the catalog: every neuron becomes a book with a topic
        (top-k input features by |weight|) and a checkout history (the
        samples on which it fired above threshold).
        """
        X = np.atleast_2d(np.asarray(sample_inputs, dtype=np.float32))
        weights = _weights_of(model)
        acts = _forward_activations(model, X)
        books: List[Dict[str, Any]] = []
        for name in sorted(weights):
            W = np.asarray(weights[name], dtype=np.float32)
            A = acts.get(name)
            if W.ndim != 2 or A is None:
                continue
            n_neurons = W.shape[1]
            for n in range(n_neurons):
                col = W[:, n]
                top_idx = np.argsort(-np.abs(col))[: self.topic_top_k]
                topic = [{"feature": int(i), "weight": float(col[i])}
                         for i in top_idx]
                fires = (np.abs(A[:, n]) >= self.fire_thresh)
                checkout_samples = [int(i) for i in np.where(fires)[0]]
                books.append({"id": f"{name}_{n}", "shelf": name,
                              "neuron": n,
                              "topic": topic,
                              "n_checkouts": len(checkout_samples),
                              "last_fired_step": (int(checkout_samples[-1])
                                                  if checkout_samples else -1),
                              "checkouts": checkout_samples})
        return {"books": books, "n_books": len(books),
                "n_shelves": len({b["shelf"] for b in books}),
                "n_samples": int(X.shape[0])}

    def checkout(self, catalog: Dict[str, Any],
                 neuron_id: str) -> Dict[str, Any]:
        """Look up a neuron's checkout record by its id ('layer_neuron')."""
        for b in catalog.get("books", []):
            if b["id"] == neuron_id:
                return {"neuron_id": b["id"], "shelf": b["shelf"],
                        "n_checkouts": b["n_checkouts"],
                        "last_fired_step": b["last_fired_step"],
                        "topic": b["topic"]}
        return {"neuron_id": neuron_id, "n_checkouts": 0,
                "error": "book not found"}

    def overdue(self, catalog: Dict[str, Any],
                days: int = 30) -> List[Dict[str, Any]]:
        """Return overdue books: neurons whose last firing was more than
        ``days`` samples ago (or never fired)."""
        n_samples = catalog.get("n_samples", 0)
        books = catalog.get("books", [])
        overdue: List[Dict[str, Any]] = []
        for b in books:
            last = b["last_fired_step"]
            if last < 0:
                age = n_samples  # never fired
            else:
                age = n_samples - last - 1
            if age >= days:
                overdue.append({"neuron_id": b["id"], "shelf": b["shelf"],
                                "age": int(age), "ever_fired": last >= 0})
        return overdue

    def bestsellers(self, catalog: Dict[str, Any],
                    n: int = 10) -> List[Dict[str, Any]]:
        """Return the top-n most-checked-out neurons (bestsellers)."""
        books = list(catalog.get("books", []))
        books.sort(key=lambda b: -b["n_checkouts"])
        return [{"neuron_id": b["id"], "shelf": b["shelf"],
                 "n_checkouts": b["n_checkouts"],
                 "topic": b["topic"]} for b in books[:n]]

    def recommend(self, catalog: Dict[str, Any],
                  model: Any, input: Any) -> List[Dict[str, Any]]:
        """Recommend which neurons to 'read' for this input: the neurons
        that fire most strongly on it. (Re-runs a forward pass.)
        """
        x = np.asarray(input, dtype=np.float32).ravel()
        acts = _forward_activations(model, x[None, :])
        recs: List[Dict[str, Any]] = []
        for name in sorted(acts):
            A = np.asarray(acts[name], dtype=np.float64).ravel()
            top = np.argsort(-np.abs(A))[:5]
            for n in top:
                recs.append({"neuron_id": f"{name}_{int(n)}",
                             "shelf": name,
                             "activation": float(A[n])})
        recs.sort(key=lambda r: -abs(r["activation"]))
        return recs[:10]


# ============================================================================
# Idea 157 -- ModelSymposium
# ============================================================================
class ModelSymposium:
    """Models gather at a philosophical symposium to debate the nature of
    intelligence. Each model is assigned a *position* based on its weight
    statistics: empiricist (large first layer -- data-driven), rationalist
    (large last layer -- theory-driven), connectionist (balanced,
    intermediate-heavy). Models *present* their prediction on an input as
    an Argument, *critique* each other's arguments (distance between
    predictions), and reach a *consensus* (the average prediction) or a
    *dissensus* (the disagreements).
    """

    POSITIONS = ("empiricist", "rationalist", "connectionist")

    def __init__(self, agreement_thresh: float = 0.2,
                 seed: int = 0):
        self.agreement_thresh = agreement_thresh
        self.rng = np.random.default_rng(seed)

    def convene(self, models: List[Any]) -> Dict[str, Any]:
        """Convene the symposium: assign each model a position based on its
        per-layer weight norms (which layer carries the most mass)."""
        attendees: List[Dict[str, Any]] = []
        for i, m in enumerate(models):
            weights = _weights_of(m)
            names = sorted(weights)
            norms = [float(np.linalg.norm(np.asarray(weights[n]).ravel()))
                     for n in names]
            total = sum(norms) + 1e-8
            if not names:
                position = "connectionist"
            else:
                first_share = norms[0] / total
                last_share = norms[-1] / total
                mid_share = sum(norms[1:-1]) / total if len(norms) > 2 else 0.0
                if first_share > 0.5:
                    position = "empiricist"
                elif last_share > 0.5:
                    position = "rationalist"
                elif mid_share > 0.4:
                    position = "connectionist"
                else:
                    position = "connectionist"
            attendees.append({"id": i, "model": m,
                              "position": position,
                              "weight_profile": {"first": first_share,
                                                "last": last_share,
                                                "mid": mid_share}})
        return {"attendees": attendees,
                "n_attendees": len(attendees),
                "positions": {p: [a for a in attendees if a["position"] == p]
                              for p in self.POSITIONS}}

    def present(self, model: Any, input: Any) -> Dict[str, Any]:
        """A model presents its argument: its prediction on the input,
        along with confidence and a stylised thesis statement."""
        x = np.asarray(input, dtype=np.float32).ravel()
        pred = _predict(model, x)
        if pred is None:
            return {"thesis": "I have no view.", "prediction": [],
                    "confidence": 0.0}
        pa = np.asarray(pred, dtype=np.float64).ravel()
        conf = _confidence(pa)
        argmax = int(np.argmax(np.abs(pa))) if pa.size else 0
        thesis = (f"I hold that the answer is {argmax} "
                  f"(confidence {conf:.2f}); the strongest signal in my "
                  f"representation is {float(np.max(np.abs(pa))):.3f}.")
        return {"thesis": thesis, "prediction": pa.tolist(),
                "confidence": conf, "argmax": argmax}

    def critique(self, argument: Dict[str, Any],
                 by_model: Any) -> Dict[str, Any]:
        """Have ``by_model`` critique the given argument: how far is its own
        prediction from the argument's prediction?"""
        pred = argument.get("prediction", [])
        if not pred:
            return {"agreement": 0.0, "disagreement": 1.0,
                    "critique": "I cannot engage with an empty thesis."}
        # we need a probe input to evaluate by_model; use the argmax's
        # logits as a stand-in: this method needs the input that produced
        # ``argument``. We approximate by calling by_model on a vector of
        # the same length as the prediction.
        x = np.asarray(pred, dtype=np.float32)
        # if the model expects a different input dim, fall back to zeros
        try:
            other = _predict(by_model, x)
        except Exception:
            other = None
        if other is None:
            return {"agreement": 0.0, "disagreement": 1.0,
                    "critique": "My position is incomparable to yours."}
        oa = np.asarray(other, dtype=np.float64).ravel()
        n = min(len(pred), len(oa))
        diff = float(np.linalg.norm(np.asarray(pred[:n]) - oa[:n]))
        norm = float(np.linalg.norm(np.asarray(pred[:n]))) + 1e-8
        rel = diff / norm
        agreement = float(max(0.0, 1.0 - rel))
        if agreement > (1.0 - self.agreement_thresh):
            critique = "I largely agree with my esteemed colleague."
        elif rel > 1.0:
            critique = "I must strongly dissent; the argument is unfounded."
        else:
            critique = "I partially concur, but reach a different conclusion."
        return {"agreement": agreement, "disagreement": 1.0 - agreement,
                "relative_distance": rel, "critique": critique}

    def consensus(self, symposium: Dict[str, Any],
                  input: Any) -> Dict[str, Any]:
        """Reach consensus: average the predictions of all attendees on the
        input. Returns the consensus prediction + agreement level.
        """
        preds: List[np.ndarray] = []
        for a in symposium.get("attendees", []):
            arg = self.present(a["model"], input)
            if arg["prediction"]:
                preds.append(np.asarray(arg["prediction"], dtype=np.float64))
        if not preds:
            return {"consensus": [], "agreement": 0.0, "n_voters": 0}
        max_len = max(p.size for p in preds)
        mat = np.zeros((len(preds), max_len), dtype=np.float64)
        for i, p in enumerate(preds):
            mat[i, :p.size] = p
        consensus = mat.mean(axis=0)
        # agreement: 1 / (1 + std across voters)
        agreement = float(1.0 / (1.0 + float(np.mean(np.std(mat, axis=0)))))
        return {"consensus": consensus.tolist(),
                "agreement": agreement,
                "n_voters": len(preds)}

    def dissensus(self, symposium: Dict[str, Any],
                  input: Any) -> List[Dict[str, Any]]:
        """Identify dissensus: attendees whose prediction differs from the
        consensus by more than the agreement threshold.
        """
        c = self.consensus(symposium, input)
        consensus_vec = np.asarray(c.get("consensus", []), dtype=np.float64)
        if consensus_vec.size == 0:
            return []
        dissenters: List[Dict[str, Any]] = []
        for a in symposium.get("attendees", []):
            arg = self.present(a["model"], input)
            pred = np.asarray(arg.get("prediction", []), dtype=np.float64)
            if pred.size == 0:
                continue
            n = min(pred.size, consensus_vec.size)
            diff = float(np.linalg.norm(pred[:n] - consensus_vec[:n]))
            norm = float(np.linalg.norm(consensus_vec[:n])) + 1e-8
            rel = diff / norm
            if rel > self.agreement_thresh:
                dissenters.append({"attendee_id": a["id"],
                                   "position": a["position"],
                                   "relative_distance": rel,
                                   "thesis": arg.get("thesis", "")})
        return dissenters


# ============================================================================
# Idea 158 -- WeightCartography
# ============================================================================
class WeightCartography:
    """Map the weight landscape like terrain. *Peaks* are saturated weights
    (high magnitude, maxima of the |W| surface). *Valleys* are local minima
    of the weight-magnitude surface (small-magnitude weights surrounded by
    larger ones -- potential local minima of the loss). *Plateaus* are flat
    regions (low-variance patches). *Ridges* are paths along which the
    weight magnitude changes slowly -- approximate decision boundaries.
    *Expedition* plans a route through weight space from one weight to
    another, climbing or skirting terrain as needed.
    """

    def __init__(self, peak_quantile: float = 0.95,
                 valley_quantile: float = 0.05,
                 plateau_window: int = 3,
                 plateau_var_thresh: float = 1e-3,
                 seed: int = 0):
        self.peak_quantile = peak_quantile
        self.valley_quantile = valley_quantile
        self.plateau_window = plateau_window
        self.plateau_var_thresh = plateau_var_thresh
        self.rng = np.random.default_rng(seed)

    def survey(self, model: Any) -> Dict[str, Any]:
        """Survey the entire weight landscape: stack all 2-D weight matrices
        into a single elevation map (|W|)."""
        weights = _weights_of(model)
        maps: Dict[str, np.ndarray] = {}
        all_elevations: List[float] = []
        for name in sorted(weights):
            W = np.asarray(weights[name], dtype=np.float64)
            elev = np.abs(W)
            maps[name] = elev
            all_elevations.extend(elev.ravel().tolist())
        if not all_elevations:
            return {"maps": {}, "global_max": 0.0, "global_min": 0.0,
                    "global_mean": 0.0}
        arr = np.asarray(all_elevations, dtype=np.float64)
        return {"maps": maps, "n_layers": len(maps),
                "global_max": float(arr.max()),
                "global_min": float(arr.min()),
                "global_mean": float(arr.mean())}

    def peaks(self, topo: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find peaks: weights above the peak quantile in their layer."""
        peaks: List[Dict[str, Any]] = []
        for name, elev in topo.get("maps", {}).items():
            if elev.size == 0:
                continue
            q = float(np.quantile(elev, self.peak_quantile))
            rows, cols = np.where(elev >= q)
            for r, c in zip(rows.tolist(), cols.tolist()):
                peaks.append({"layer": name, "row": int(r), "col": int(c),
                              "elevation": float(elev[r, c])})
        peaks.sort(key=lambda p: -p["elevation"])
        return peaks

    def valleys(self, topo: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find valleys: weights below the valley quantile in their layer."""
        valleys: List[Dict[str, Any]] = []
        for name, elev in topo.get("maps", {}).items():
            if elev.size == 0:
                continue
            q = float(np.quantile(elev, self.valley_quantile))
            rows, cols = np.where(elev <= q)
            for r, c in zip(rows.tolist(), cols.tolist()):
                valleys.append({"layer": name, "row": int(r), "col": int(c),
                                "elevation": float(elev[r, c])})
        valleys.sort(key=lambda v: v["elevation"])
        return valleys

    def ridges(self, topo: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find ridges: row-wise paths in 2-D weight maps where the
        elevation gradient along the column direction is small (flat
        extended paths -- approximations of decision boundaries).
        """
        ridges: List[Dict[str, Any]] = []
        for name, elev in topo.get("maps", {}).items():
            if elev.ndim != 2 or elev.shape[1] < 2:
                continue
            # column-direction gradient magnitude
            grad = np.abs(np.diff(elev, axis=1))
            # a ridge row = a row whose mean gradient is below the threshold
            mean_grad_per_row = grad.mean(axis=1)
            threshold = (float(np.mean(mean_grad_per_row))
                         * self.plateau_var_thresh * 100 + 1e-6)
            for r in np.where(mean_grad_per_row < threshold)[0]:
                ridges.append({"layer": name, "row": int(r),
                               "length": int(elev.shape[1]),
                               "mean_elevation": float(np.mean(elev[r, :])),
                               "mean_gradient": float(mean_grad_per_row[r])})
        return ridges

    def expedition(self, topo: Dict[str, Any],
                   start: Tuple[str, int, int],
                   end: Tuple[str, int, int]) -> Dict[str, Any]:
        """Plan an expedition from one weight to another across the
        landscape. The path is a greedy descent: from ``start``, move to
        the neighbour (within the same layer) with the lowest elevation
        until we reach ``end`` or a local minimum (valley).
        """
        start_layer, sr, sc = start
        end_layer, er, ec = end
        maps = topo.get("maps", {})
        if start_layer not in maps or end_layer not in maps:
            return {"path": [], "reached": False,
                    "reason": "start or end layer not in landscape"}
        # if layers differ, plan within start_layer first (single-layer
        # expedition) -- this is a simplification of the full cross-layer
        # problem.
        elev = maps[start_layer]
        if elev.ndim != 2:
            return {"path": [], "reached": False,
                    "reason": "non-2-D layer"}
        path: List[Tuple[int, int, float]] = [(int(sr), int(sc),
                                               float(elev[sr, sc]))]
        visited = set()
        cur = (int(sr), int(sc))
        target = (int(er), int(ec)) if start_layer == end_layer else None
        max_steps = int(elev.size)
        for _ in range(max_steps):
            if target is not None and cur == target:
                return {"path": path, "reached": True, "n_steps": len(path)}
            visited.add(cur)
            r, c = cur
            # 4-neighbours
            neighbours = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < elev.shape[0] and 0 <= nc < elev.shape[1] \
                        and (nr, nc) not in visited:
                    neighbours.append(((nr, nc), float(elev[nr, nc])))
            if not neighbours:
                break
            # greedy: if target exists, move toward it; else descend
            if target is not None:
                neighbours.sort(key=lambda n: (abs(n[0][0] - target[0])
                                                + abs(n[0][1] - target[1]),
                                                n[1]))
            else:
                neighbours.sort(key=lambda n: n[1])
            cur = neighbours[0][0]
            path.append((cur[0], cur[1], float(elev[cur[0], cur[1]])))
        reached = (target is not None and cur == target)
        return {"path": path, "reached": reached,
                "n_steps": len(path),
                "final_elevation": float(elev[cur[0], cur[1]]),
                "reason": "reached" if reached else "stuck (local minimum)"}


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


__all__ = [
    "ModelAstronomy",
    "HallucinationDetective",
    "NeuralGarden",
    "ModelStandUp",
    "WeightArchitecture",
    "PredictionForecast",
    "ModelThermostat",
    "NeuralLibrary",
    "ModelSymposium",
    "WeightCartography",
]
