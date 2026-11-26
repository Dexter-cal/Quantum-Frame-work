"""
qai.creative.experimental12 -- 10 more original creative ideas (part 12).

  Idea 119 -- ModelTournament        : single-elimination bracket of models
  Idea 120 -- NeuralConstellation    : neurons as stars; constellations & supernovas
  Idea 121 -- ModelCryptobiology     : species / genome / crossbreed / mutate
  Idea 122 -- HallucinationAutopsy   : trace + blame neurons for a wrong output
  Idea 123 -- WeightSeismograph      : seismograph of weight shifts over training
  Idea 124 -- ModelEmbassyProtocol   : cross-framework diplomatic protocol
  Idea 125 -- NeuralArchaeology      : dig through layers as archaeological strata
  Idea 126 -- ModelSommelier         : flavor profile of a model (wine sommelier)
  Idea 127 -- PredictionGenealogy    : ancestry of a single prediction
  Idea 128 -- ModelCensus            : census of neurons (employment, gini, etc.)

A "model" here is any object that exposes its parameters via a ``.weights``
attribute (dict / list / ndarray) or via ``state_dict()`` / ``technique``
(as in the rest of qai), and that can be inferred with ``.predict`` /
``.forward`` / ``__call__``.
"""
from __future__ import annotations

import math
import time
import hashlib
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


def _to_float(x: Any) -> float:
    """Coerce any model output into a single float for comparison."""
    try:
        return float(np.mean(np.asarray(x, dtype=np.float64)))
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


class _MinimalModel:
    """A minimal model wrapper: holds a weights dict and runs a tanh
    forward pass. Used as the offspring / mutant return value of
    ModelCryptobiology so callers receive a runnable model even when the
    parents were arbitrary objects.
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
# Idea 119 -- ModelTournament
# ============================================================================
class ModelTournament:
    """Models compete in a single-elimination tournament on a benchmark.
    Models are seeded by a quick warm-up score; each round pairs the
    highest surviving seed against the lowest, the loser is eliminated,
    and play continues until one champion remains. ``upsets`` records
    matches where a lower seed beat a higher seed (the Cinderellas).
    """

    def __init__(self, metric: str = "accuracy", noise: float = 0.02,
                 seed: int = 0):
        self.metric = metric
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._bracket: Optional[Dict[str, Any]] = None
        self._upsets: List[Dict[str, Any]] = []

    def _score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Compute a single benchmark score for a model."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        try:
            preds = np.asarray([_to_float(_predict(model, xi)) for xi in X])
        except Exception:
            return 0.0
        if self.metric == "accuracy":
            return float(np.mean(np.round(preds) == y.astype(float)))
        # regression: negative MSE (higher is better)
        return float(-np.mean((preds - y.astype(float)) ** 2))

    def bracket(self, models: List[Any], X: np.ndarray,
                y: np.ndarray) -> Dict[str, Any]:
        """Seed models by score and build the initial bracket."""
        X = np.asarray(X, dtype=np.float32)
        scored = [(m, self._score(m, X, y)) for m in models]
        scored.sort(key=lambda s: -s[1])
        seeded = [{"seed": i + 1, "model": m, "score": float(sc),
                   "alive": True} for i, (m, sc) in enumerate(scored)]
        self._bracket = {"rounds": [], "seeds": seeded,
                         "n_models": len(seeded), "champion": None}
        self._upsets = []
        return self._bracket

    def play_round(self, bracket: Dict[str, Any],
                   round_idx: int) -> List[Dict[str, Any]]:
        """Play one round: pair seeds 1vN, 2vN-1, ... and eliminate losers."""
        if bracket is None:
            return []
        alive = [s for s in bracket["seeds"] if s["alive"]]
        if len(alive) <= 1:
            return []
        n = len(alive)
        results: List[Dict[str, Any]] = []
        for i in range(n // 2):
            a, b = alive[i], alive[n - 1 - i]
            ja = a["score"] + float(self.rng.normal(0, self.noise))
            jb = b["score"] + float(self.rng.normal(0, self.noise))
            winner = a if ja >= jb else b
            loser = b if winner is a else a
            loser["alive"] = False
            upset = bool(winner["seed"] > loser["seed"])
            match = {"round": round_idx, "a_seed": a["seed"],
                     "b_seed": b["seed"], "winner_seed": winner["seed"],
                     "score_a": a["score"], "score_b": b["score"],
                     "upset": upset}
            if upset:
                self._upsets.append(match)
            results.append(match)
        bracket["rounds"].append(results)
        still_alive = [s for s in bracket["seeds"] if s["alive"]]
        if len(still_alive) == 1:
            bracket["champion"] = still_alive[0]
        return results

    def champion(self) -> Any:
        """Return the champion model (or None if tournament ongoing / not started)."""
        if self._bracket is None:
            return None
        ch = self._bracket.get("champion")
        return ch["model"] if ch else None

    def upsets(self) -> List[Dict[str, Any]]:
        """Return all upset matches (where a lower seed beat a higher seed)."""
        return list(self._upsets)


# ============================================================================
# Idea 120 -- NeuralConstellation
# ============================================================================
class NeuralConstellation:
    """Map each neuron to a star in a virtual sky: brightness = how often
    it activates, color = which layer, position = a learned 2-D embedding
    of its activation pattern. ``constellations`` groups stars that
    always fire together; ``supernovas`` finds neurons that spike
    suddenly (max activation far above their typical level).
    """

    def __init__(self, constellation_thresh: float = 0.7,
                 supernova_ratio: float = 5.0, seed: int = 0):
        self.constellation_thresh = constellation_thresh
        self.supernova_ratio = supernova_ratio
        self.rng = np.random.default_rng(seed)

    def chart(self, model: Any, sample_inputs: np.ndarray) -> Dict[str, Any]:
        """Build a star map: one star per neuron across all layers."""
        X = np.asarray(sample_inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        acts = _forward_activations(model, X)
        layer_names = list(acts.keys())
        stars: List[Dict[str, Any]] = []
        for layer_idx, name in enumerate(layer_names):
            act = acts[name]  # (n_samples, n_neurons)
            n_neurons = act.shape[1]
            color = layer_idx / max(1, len(layer_names) - 1) if layer_names else 0.0
            # 2-D position = random projection of each neuron's activation vector
            proj = self.rng.normal(0, 1, size=(n_neurons, 2)).astype(np.float32)
            for n in range(n_neurons):
                a = act[:, n]
                brightness = float(np.mean(np.abs(a)))
                stars.append({
                    "layer": name, "layer_idx": layer_idx, "neuron": n,
                    "brightness": brightness, "color": color,
                    "position": proj[n].tolist(),
                    "activations": a.tolist(),
                })
        return {"stars": stars, "n_samples": int(X.shape[0]),
                "n_layers": len(layer_names)}

    def constellations(self, starmap: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find groups of stars that always activate together (real constellations)."""
        stars = starmap.get("stars", [])
        if len(stars) < 2:
            return []
        # build (n_stars x n_samples) activation matrix
        max_len = max(len(s["activations"]) for s in stars)
        if max_len == 0:
            return []
        mat = np.zeros((len(stars), max_len), dtype=np.float32)
        for i, s in enumerate(stars):
            a = np.asarray(s["activations"], dtype=np.float32)
            mat[i, : a.size] = a
        # cosine similarity between neuron activation patterns
        centered = mat - mat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1) + 1e-8
        normed = centered / norms[:, None]
        corr = normed @ normed.T
        np.fill_diagonal(corr, 0.0)
        # greedy neighbor grouping
        visited: set = set()
        groups: List[Dict[str, Any]] = []
        for i in range(len(stars)):
            if i in visited:
                continue
            peers = [int(p) for p in np.where(corr[i] >= self.constellation_thresh)[0]
                     if p not in visited and p != i]
            if not peers:
                continue
            group = [i] + peers
            for p in group:
                visited.add(p)
            mean_corr = float(np.mean([corr[i, p] for p in peers])) if peers else 0.0
            groups.append({"stars": group, "size": len(group),
                           "mean_corr": mean_corr,
                           "layers": sorted({stars[g]["layer"] for g in group})})
        return groups

    def supernovas(self, starmap: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find neurons that suddenly spike (peak >> median activation)."""
        stars = starmap.get("stars", [])
        out: List[Dict[str, Any]] = []
        for s in stars:
            a = np.asarray(s.get("activations", []), dtype=np.float32).ravel()
            if a.size < 2:
                continue
            abs_a = np.abs(a)
            peak = float(abs_a.max())
            median = float(np.median(abs_a))
            if median < 1e-8:
                continue
            ratio = peak / median
            if ratio >= self.supernova_ratio:
                out.append({"layer": s["layer"], "neuron": s["neuron"],
                            "peak": peak, "median": median, "ratio": ratio})
        return out


# ============================================================================
# Idea 121 -- ModelCryptobiology
# ============================================================================
class ModelCryptobiology:
    """Discover "species" of models by clustering their architectures.
    Each species has a genome: layer dims, activation functions, optimizer
    name. Two models with identical genomes belong to the same species.
    ``crossbreed`` produces offspring that inherit alternating layers from
    each parent; ``mutate`` perturbs weights at random.
    """

    def __init__(self, mutation_rate: float = 0.1, seed: int = 0):
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)
        self._species_genomes: Dict[str, Dict[str, Any]] = {}
        self._species_counts: Dict[str, int] = {}

    def genome(self, model: Any) -> Dict[str, Any]:
        """Extract the genome of a model (layer dims, activations, optimizer)."""
        weights = _weights_of(model)
        dims = [list(np.asarray(w).shape) for w in weights.values()]
        activations: List[str] = []
        for name in weights:
            act = "tanh"
            for attr in (f"activation_{name}", "activation"):
                v = getattr(model, attr, None)
                if v is not None:
                    act = str(v)
                    break
            activations.append(act)
        optimizer = str(getattr(model, "optimizer", "unknown"))
        h = hashlib.blake2b(digest_size=8)
        h.update(str(dims).encode("utf-8"))
        h.update(str(activations).encode("utf-8"))
        h.update(optimizer.encode("utf-8"))
        n_params = 0
        for d in dims:
            p = 1
            for s in d:
                p *= int(s) if s else 1
            n_params += p
        return {"dims": dims, "activations": activations,
                "optimizer": optimizer, "genome_id": h.hexdigest(),
                "n_layers": len(dims), "n_params": n_params}

    def classify(self, model: Any) -> Dict[str, Any]:
        """Classify a model into a species based on its genome."""
        g = self.genome(model)
        gid = g["genome_id"]
        for sp_id, sp_genome in self._species_genomes.items():
            if sp_genome["genome_id"] == gid:
                self._species_counts[sp_id] = self._species_counts.get(sp_id, 0) + 1
                return {"species_id": sp_id, "genome": g,
                        "population": self._species_counts[sp_id]}
        sp_id = f"species_{len(self._species_genomes) + 1}"
        self._species_genomes[sp_id] = g
        self._species_counts[sp_id] = 1
        return {"species_id": sp_id, "genome": g, "population": 1}

    def crossbreed(self, model_a: Any, model_b: Any) -> Any:
        """Genetic crossover: offspring inherits alternating layers from each parent."""
        wa = _weights_of(model_a)
        wb = _weights_of(model_b)
        names_a = list(wa.keys())
        names_b = list(wb.keys())
        offspring: Dict[str, np.ndarray] = {}
        n = max(len(names_a), len(names_b))
        for i in range(n):
            if i % 2 == 0 and i < len(names_a):
                name = names_a[i]
                offspring[f"L{i}_{name}"] = np.array(wa[name], copy=True)
            elif i < len(names_b):
                name = names_b[i]
                offspring[f"L{i}_{name}"] = np.array(wb[name], copy=True)
            elif i < len(names_a):
                name = names_a[i]
                offspring[f"L{i}_{name}"] = np.array(wa[name], copy=True)
        return _MinimalModel(offspring)

    def mutate(self, model: Any, rate: float = 0.1) -> Any:
        """Apply random mutations: perturb weights with probability ``rate``."""
        weights = _weights_of(model)
        mutated: Dict[str, np.ndarray] = {}
        for name, W in weights.items():
            Wm = np.array(W, dtype=np.float32, copy=True)
            mask = self.rng.random(Wm.shape) < rate
            if mask.any():
                noise = self.rng.normal(0, 0.1, size=Wm.shape).astype(np.float32)
                Wm[mask] = Wm[mask] + noise[mask]
            mutated[name] = Wm
        return _MinimalModel(mutated)


# ============================================================================
# Idea 122 -- HallucinationAutopsy
# ============================================================================
class HallucinationAutopsy:
    """When a model hallucinates, perform an autopsy: trace which neurons
    fired, which weights contributed most, and pinpoint the "cause of
    death" (the truth the model failed to produce). Each Blame is a
    single neuron whose activation most contributed to the wrong output,
    computed as the gradient of the error w.r.t. that neuron's pre-activation.
    """

    def __init__(self, top_k: int = 5, blame_thresh: float = 0.01):
        self.top_k = top_k
        self.blame_thresh = blame_thresh

    def trace_activations(self, model: Any, input: Any) -> List[Dict[str, Any]]:
        """Forward-pass recording each layer's pre-activations and activations."""
        weights = _weights_of(model)
        x = np.asarray(input, dtype=np.float32).ravel()
        cur = x.copy()
        trace: List[Dict[str, Any]] = []
        for name in sorted(weights):
            Wm = np.asarray(weights[name], dtype=np.float32)
            if Wm.ndim != 2:
                continue
            if cur.size == Wm.shape[0]:
                pre = cur @ Wm
            elif cur.size == Wm.shape[1]:
                pre = cur @ Wm.T
            else:
                continue
            act = np.tanh(pre)
            trace.append({"layer": name, "input": cur.copy(),
                          "pre_act": pre.copy(), "act": act.copy()})
            cur = act
        return trace

    def autopsy(self, model: Any, input: Any, expected: Any,
                actual: Any) -> Dict[str, Any]:
        """Perform a full autopsy on a hallucination."""
        trace = self.trace_activations(model, input)
        weights_snapshot = {k: np.array(v, copy=True)
                            for k, v in _weights_of(model).items()}
        exp = np.asarray(expected, dtype=np.float32).ravel()
        act = np.asarray(actual, dtype=np.float32).ravel()
        n = min(exp.size, act.size)
        error = float(np.linalg.norm(exp[:n] - act[:n])) if n else 0.0
        report = {"input": np.asarray(input, dtype=np.float32),
                  "expected": exp, "actual": act,
                  "error": error, "trace": trace,
                  "weights_snapshot": weights_snapshot}
        blames = self.blame(report)
        report["blames"] = blames
        report["cause_of_death"] = blames[0] if blames else None
        return report

    def blame(self, autopsy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pinpoint which neurons caused the wrong output.

        Contribution of neuron i (input to the last layer) to the error is
        ``activation_i * sum_j W[i, j] * err[j]`` where err = actual - expected.
        """
        trace = autopsy.get("trace", [])
        if not trace:
            return []
        exp = np.asarray(autopsy.get("expected", []), dtype=np.float32).ravel()
        act = np.asarray(autopsy.get("actual", []), dtype=np.float32).ravel()
        n = min(exp.size, act.size)
        if n == 0:
            return []
        err = act[:n] - exp[:n]
        weights = autopsy.get("weights_snapshot", {})
        # find the last layer with 2-D weights
        last_name: Optional[str] = None
        for entry in reversed(trace):
            if entry["layer"] in weights and np.asarray(weights[entry["layer"]]).ndim == 2:
                last_name = entry["layer"]
                break
        if last_name is None:
            return []
        W = np.asarray(weights[last_name], dtype=np.float32)
        last_input = trace[-1]["input"]
        # determine the orientation of W relative to last_input and err
        if last_input.size == W.shape[0] and n <= W.shape[1]:
            contributions = last_input * (W[:, :n] @ err)
        elif last_input.size == W.shape[1] and n <= W.shape[0]:
            contributions = last_input * (W[:n, :].T @ err)
        else:
            return []
        abs_contrib = np.abs(contributions)
        top_idx = np.argsort(-abs_contrib)[: self.top_k]
        blames: List[Dict[str, Any]] = []
        for i in top_idx:
            if abs_contrib[i] < self.blame_thresh:
                continue
            blames.append({"layer": last_name, "neuron": int(i),
                           "activation": float(last_input[i]),
                           "contribution": float(contributions[i])})
        return blames


# ============================================================================
# Idea 123 -- WeightSeismograph
# ============================================================================
class WeightSeismograph:
    """Monitor weight magnitudes over training like a seismograph monitors
    earthquakes. Each call to ``record`` adds a snapshot; the cumulative
    Seismogram is a time series of per-layer weight norms. We detect:

      - tremor     : small sustained weight shifts
      - earthquake : a large single-step shift
      - tsunami    : a shift affecting many layers simultaneously
    """

    def __init__(self, quake_thresh: float = 0.1,
                 tsunami_layers: int = 3, window: int = 5):
        self.quake_thresh = quake_thresh
        self.tsunami_layers = tsunami_layers
        self.window = window
        self._snapshots: List[Dict[str, Any]] = []

    def record(self, model: Any, epoch: int) -> Dict[str, Any]:
        """Record a weight-norm snapshot at the given epoch; return the cumulative Seismogram."""
        weights = _weights_of(model)
        norms = {name: float(np.linalg.norm(np.asarray(w).ravel()))
                 for name, w in weights.items()}
        total = float(sum(norms.values()))
        snap = {"epoch": int(epoch), "layer_norms": norms,
                "total_norm": total, "t": time.time()}
        self._snapshots.append(snap)
        return {"snapshots": list(self._snapshots), "last_epoch": int(epoch),
                "last_total": total}

    def detect_quakes(self, seismogram: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect quakes: epochs where the weight norm shifted significantly."""
        snaps = seismogram.get("snapshots", []) if isinstance(seismogram, dict) else []
        if len(snaps) < 2:
            return []
        quakes: List[Dict[str, Any]] = []
        for i in range(1, len(snaps)):
            prev_total = snaps[i - 1]["total_norm"]
            curr_total = snaps[i]["total_norm"]
            shift = abs(curr_total - prev_total) / (prev_total + 1e-8)
            layer_shifts = []
            for name in snaps[i]["layer_norms"]:
                pn = snaps[i - 1]["layer_norms"].get(name, 0.0)
                cn = snaps[i]["layer_norms"].get(name, 0.0)
                if pn > 0:
                    layer_shifts.append((name, abs(cn - pn) / (pn + 1e-8)))
            n_big = sum(1 for _, s in layer_shifts if s > self.quake_thresh)
            qtype = "tremor"
            if shift > self.quake_thresh * 10:
                qtype = "earthquake"
            if n_big >= self.tsunami_layers:
                qtype = "tsunami"
            if shift > self.quake_thresh * 0.1 or qtype != "tremor":
                quakes.append({"epoch": snaps[i]["epoch"],
                               "shift": float(shift), "type": qtype,
                               "n_layers_affected": n_big})
        return quakes

    def magnitude(self, quake: Dict[str, Any]) -> float:
        """Richter-like magnitude of a quake (log10 of the relative shift)."""
        shift = float(quake.get("shift", 0.0))
        if shift <= 0:
            return 0.0
        return float(np.log10(shift * 1000.0 + 1.0))

    def forecast(self, seismogram: Dict[str, Any]) -> str:
        """Will training converge or diverge based on the seismogram?"""
        snaps = seismogram.get("snapshots", []) if isinstance(seismogram, dict) else []
        if len(snaps) < 2:
            return "indeterminate: not enough data"
        totals = np.asarray([s["total_norm"] for s in snaps], dtype=np.float64)
        mean_total = float(totals.mean())
        if mean_total < 1e-8:
            return "indeterminate: weights are zero"
        diffs = np.diff(totals)
        drift = float(np.mean(diffs[-self.window:])) if len(diffs) >= 1 else 0.0
        vol = float(np.std(diffs)) + 1e-8
        rel_vol = vol / (abs(mean_total) + 1e-8)
        if rel_vol > 0.3:
            return "divergence: high weight volatility"
        if abs(drift) < 1e-3 * (abs(mean_total) + 1e-8):
            return "convergence: weights stabilizing"
        if drift > 0:
            return "growth: weights expanding"
        return "contraction: weights shrinking"


# ============================================================================
# Idea 124 -- ModelEmbassyProtocol
# ============================================================================
class ModelEmbassyProtocol:
    """Models from different frameworks (sklearn, torch, tf) need a
    diplomatic protocol to communicate. We accredit each model as a
    Diplomat from its framework, translate payloads (arrays, scalars)
    across framework boundaries, negotiate treaties (consensus outputs)
    between pairs of diplomats, and sign treaties only when both sides
    agree within tolerance.
    """

    def __init__(self, default_framework: str = "numpy",
                 treaty_tol: float = 1.0):
        self.default_framework = default_framework.lower()
        self.treaty_tol = treaty_tol
        self._treaties: List[Dict[str, Any]] = []

    def accredit(self, model: Any, framework: Optional[str] = None) -> Dict[str, Any]:
        """Accredit a model as a Diplomat from the given framework."""
        fw = (framework or self.default_framework).lower()
        weights = _weights_of(model)
        return {"model": model, "framework": fw,
                "n_params": sum(int(np.asarray(w).size) for w in weights.values()),
                "n_layers": len(weights),
                "accredited_at": time.time()}

    def translate(self, diplomat: Dict[str, Any], payload: Any) -> Any:
        """Translate a payload (array / list / scalar) into the diplomat's framework."""
        fw = diplomat.get("framework", self.default_framework).lower()
        try:
            arr = np.asarray(payload, dtype=np.float32)
        except Exception:
            return payload
        if fw in ("numpy", "np", "sklearn", "scikit"):
            return arr
        if fw in ("torch", "pytorch", "tensorflow", "tf", "keras"):
            # return a portable nested-list proxy (no torch / tf imports needed)
            return arr.tolist()
        return arr

    def negotiate(self, diplomat_a: Dict[str, Any],
                  diplomat_b: Dict[str, Any], input: Any) -> Dict[str, Any]:
        """Two diplomats negotiate over an input: each predicts, treaty holds the consensus."""
        pa = _predict(diplomat_a["model"], input)
        pb = _predict(diplomat_b["model"], input)
        consensus: Optional[List[float]] = None
        try:
            a = np.asarray(pa, dtype=np.float32).ravel()
            b = np.asarray(pb, dtype=np.float32).ravel()
            n = min(a.size, b.size)
            if n > 0:
                consensus = ((a[:n] + b[:n]) / 2.0).tolist()
        except Exception:
            consensus = None
        treaty = {"framework_a": diplomat_a["framework"],
                  "framework_b": diplomat_b["framework"],
                  "pred_a": pa, "pred_b": pb,
                  "consensus": consensus, "signed": False}
        self._treaties.append(treaty)
        treaty["signed"] = self.sign_treaty(treaty)
        return treaty

    def sign_treaty(self, treaty: Dict[str, Any]) -> bool:
        """Sign the treaty iff both diplomats' predictions agree within tolerance."""
        pa = treaty.get("pred_a")
        pb = treaty.get("pred_b")
        try:
            a = np.asarray(pa, dtype=np.float32).ravel()
            b = np.asarray(pb, dtype=np.float32).ravel()
            n = min(a.size, b.size)
            if n == 0:
                return False
            denom = float(np.linalg.norm(a[:n])) + 1e-8
            diff = float(np.linalg.norm(a[:n] - b[:n]) / denom)
            signed = diff < self.treaty_tol
            treaty["signed"] = signed
            return signed
        except Exception:
            treaty["signed"] = False
            return False


# ============================================================================
# Idea 125 -- NeuralArchaeology
# ============================================================================
class NeuralArchaeology:
    """Dig through a model's layers as if excavating an archaeological
    site. Deeper layers (higher layer index) are older strata. We find
    "fossils" (dead weights left over from old training), "pottery
    shards" (low-rank fragments of partial features), and "civilization
    peaks" (highly-connected eras with dense weight matrices).
    """

    def __init__(self, fossil_tol: float = 1e-4, shard_rank: int = 2,
                 peak_density: float = 0.5):
        self.fossil_tol = fossil_tol
        self.shard_rank = shard_rank
        self.peak_density = peak_density

    def excavate(self, model: Any) -> Dict[str, Any]:
        """Open the site: extract all layers + their statistics."""
        weights = _weights_of(model)
        strata: List[Dict[str, Any]] = []
        for depth, name in enumerate(sorted(weights)):
            Wm = np.asarray(weights[name], dtype=np.float32)
            n_params = int(Wm.size)
            n_zero = int((np.abs(Wm) < self.fossil_tol).sum()) if Wm.ndim >= 1 else 0
            density = 1.0
            if Wm.ndim == 2 and Wm.size > 0:
                density = float((np.abs(Wm) >= self.fossil_tol).mean())
            strata.append({
                "depth": depth, "name": name, "W": Wm,
                "shape": list(Wm.shape), "ndim": Wm.ndim,
                "n_params": n_params, "n_zero": n_zero,
                "density": density,
                "mean": float(Wm.mean()) if Wm.size else 0.0,
                "std": float(Wm.std()) if Wm.size else 0.0,
            })
        return {"strata": strata, "depth": len(strata),
                "n_total_params": sum(s["n_params"] for s in strata)}

    def stratify(self, site: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return strata ordered surface (shallowest) -> bedrock (deepest)."""
        return list(site.get("strata", []))

    def fossils(self, stratum: Dict[str, Any]) -> List[Tuple[int, ...]]:
        """Find indices of dead weights (fossils) in this stratum."""
        W = stratum.get("W")
        if W is None:
            return []
        idx = np.argwhere(np.abs(W) < self.fossil_tol)
        return [tuple(int(i) for i in coord) for coord in idx[: 1000]]  # cap to keep it sane

    def civilization_peaks(self, stratum: Dict[str, Any]) -> bool:
        """Is this stratum a 'civilization peak' (highly connected / dense)?"""
        return bool(stratum.get("density", 0.0) >= self.peak_density)

    def date(self, stratum: Dict[str, Any]) -> str:
        """Estimate which 'epoch' this stratum was last meaningfully trained in.

        Deeper strata (higher depth) are older. Fossils indicate the layer
        was trained long ago and has since become dormant; high std
        indicates recent, active training.
        """
        n_zero = stratum.get("n_zero", 0)
        n_params = max(1, stratum.get("n_params", 1))
        fossil_ratio = n_zero / n_params
        std = stratum.get("std", 0.0)
        depth = stratum.get("depth", 0)
        epoch_estimate = depth * 5 + int(fossil_ratio * 20)
        if fossil_ratio > 0.5:
            era = "Primordial"
        elif fossil_ratio > 0.2:
            era = "Archaic"
        elif std > 0.1:
            era = "Modern"
        else:
            era = "Classical"
        return f"{era} era (~epoch {epoch_estimate})"


# ============================================================================
# Idea 126 -- ModelSommelier
# ============================================================================
class ModelSommelier:
    """Like a wine sommelier, describe a model's "flavor profile":
    complexity (depth = number of layers), body (parameter count),
    acidity (loss; if not provided, estimated from weight kurtosis),
    tannins (fraction of small weights, a regularization proxy), and
    terroir (a hash of the model's weight shapes). ``describe`` produces
    a flowery description; ``pair`` suggests use cases.
    """

    def __init__(self, tannin_tol: float = 0.05):
        self.tannin_tol = tannin_tol

    def taste(self, model: Any, loss: Optional[float] = None) -> Dict[str, Any]:
        """Taste the model and produce its flavor profile."""
        weights = _weights_of(model)
        n_params = sum(int(np.asarray(w).size) for w in weights.values())
        complexity = len(weights)
        if weights:
            all_w = np.concatenate(
                [np.asarray(w, dtype=np.float64).ravel() for w in weights.values()])
        else:
            all_w = np.zeros(1, dtype=np.float64)
        if loss is None:
            std = float(all_w.std()) + 1e-8
            kurt = float(((all_w - all_w.mean()) ** 4).mean() / (std ** 4))
            acidity = float(min(2.0, max(0.0, (kurt - 3.0) / 10.0)))
        else:
            acidity = float(loss)
        tannins = float(np.mean(np.abs(all_w) < self.tannin_tol))
        h = hashlib.blake2b(digest_size=8)
        for name in sorted(weights):
            h.update(name.encode("utf-8"))
            h.update(str(np.asarray(weights[name]).shape).encode("utf-8"))
        terroir = h.hexdigest()
        return {"complexity": complexity, "body": int(n_params),
                "acidity": acidity, "tannins": tannins, "terroir": terroir}

    def describe(self, tasting: Dict[str, Any]) -> str:
        """Produce a flowery sommelier-style description of the model."""
        body = tasting["body"]
        depth = tasting["complexity"]
        acid = tasting["acidity"]
        tannin = tasting["tannins"]
        terroir = tasting["terroir"]
        if body > 1_000_000:
            body_word = "Bold"
        elif body > 10_000:
            body_word = "Medium-bodied"
        else:
            body_word = "Light"
        if depth >= 5:
            depth_phrase = "with deep, layered complexity"
        elif depth >= 2:
            depth_phrase = "with straightforward structure"
        else:
            depth_phrase = "barely more than a single vine"
        if acid > 0.5:
            acid_phrase = "crisp, almost tart acidity"
        elif acid > 0.1:
            acid_phrase = "balanced acidity"
        else:
            acid_phrase = "soft, mellow finish"
        if tannin > 0.5:
            tannin_phrase = "heavy, astringent tannins"
        elif tannin > 0.2:
            tannin_phrase = "notable tannins"
        else:
            tannin_phrase = "silky, restrained tannins"
        return (f"{body_word} model {depth_phrase}, {tannin_phrase} and "
                f"{acid_phrase} (loss {acid:.3f}). Terroir: {terroir}. "
                f"Pairs well with structured data.")

    def pair(self, tasting: Dict[str, Any]) -> List[str]:
        """Suggest use cases that pair well with this model's profile."""
        body = tasting["body"]
        depth = tasting["complexity"]
        acid = tasting["acidity"]
        tannin = tasting["tannins"]
        pairs: List[str] = []
        if body > 1_000_000:
            pairs.append("large-scale tabular classification")
            pairs.append("high-throughput batch inference")
        if depth >= 5:
            pairs.append("deep feature extraction")
            pairs.append("representation learning")
        if acid < 0.1:
            pairs.append("low-latency online prediction")
        if tannin > 0.3:
            pairs.append("regularization-sensitive tasks")
        if not pairs:
            pairs.append("general-purpose regression")
        return pairs

    def vintage(self, model: Any) -> int:
        """Return the model's vintage (epoch count) if exposed, else 0."""
        for attr in ("epoch", "epochs", "n_epochs", "epoch_count"):
            v = getattr(model, attr, None)
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    continue
        return 0


# ============================================================================
# Idea 127 -- PredictionGenealogy
# ============================================================================
class PredictionGenealogy:
    """Trace the ancestry of a single prediction: which training samples
    most influenced THIS output? Each prediction's "ancestors" are the
    training samples whose learned representations are most aligned with
    the input's representation (an influence-function approximation).
    The heritage score measures how concentrated the ancestry is.
    """

    def __init__(self, n_representatives: int = 100, top_k: int = 10):
        self.n_representatives = n_representatives
        self.top_k = top_k
        self._train_reps: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None
        self._ancestors: Dict[int, Dict[str, Any]] = {}
        self._counter = 0

    def memorize(self, X_train: np.ndarray,
                 y_train: Optional[np.ndarray] = None) -> None:
        """Cache the training set (or a representative subsample)."""
        X = np.asarray(X_train, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        y = np.asarray(y_train) if y_train is not None else None
        if X.shape[0] > self.n_representatives:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], self.n_representatives, replace=False)
            X = X[idx]
            if y is not None:
                y = y[idx]
        self._train_reps = X
        self._train_labels = y

    def _represent(self, model: Any, x: np.ndarray) -> np.ndarray:
        """Project an input through the model's weights (tanh) to its representation."""
        cur = np.asarray(x, dtype=np.float32).ravel()
        weights = _weights_of(model)
        for name in sorted(weights):
            W = np.asarray(weights[name], dtype=np.float32)
            if W.ndim == 2 and cur.size == W.shape[0]:
                cur = np.tanh(cur @ W)
            elif W.ndim == 2 and cur.size == W.shape[1]:
                cur = np.tanh(cur @ W.T)
        return cur

    def trace(self, model: Any, input: Any) -> Dict[str, Any]:
        """Trace the ancestry of a prediction: rank training samples by influence."""
        x = np.asarray(input, dtype=np.float32).ravel()
        rep = self._represent(model, x)
        pred = _predict(model, x)
        influences: List[Tuple[int, float]] = []
        if self._train_reps is not None:
            for j, xj in enumerate(self._train_reps):
                rj = self._represent(model, xj)
                n = min(rep.size, rj.size)
                if n == 0:
                    influences.append((j, 0.0))
                    continue
                a, b = rep[:n], rj[:n]
                cos = float(np.dot(a, b) /
                            (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                influences.append((j, cos))
        influences.sort(key=lambda v: -abs(v[1]))
        self._counter += 1
        ancestry = {"id": self._counter, "input": x, "prediction": pred,
                    "influences": influences,
                    "n_samples": len(influences)}
        self._ancestors[self._counter] = ancestry
        return ancestry

    def influential_samples(self, ancestry: Dict[str, Any],
                           k: int = 10) -> List[int]:
        """Return indices of the k most influential training samples."""
        infls = ancestry.get("influences", [])
        top = infls[:k]
        return [int(j) for j, _ in top]

    def heritage_score(self, ancestry: Dict[str, Any]) -> float:
        """How concentrated is the prediction's ancestry?

        1.0 = entirely inherited from a few samples; 0.0 = perfectly diffuse
        (every training sample contributed equally).
        """
        infls = ancestry.get("influences", [])
        if not infls:
            return 0.0
        vals = np.asarray([abs(v) for _, v in infls], dtype=np.float64)
        total = float(vals.sum())
        if total <= 0:
            return 0.0
        k = min(self.top_k, vals.size)
        top_share = float(vals[np.argsort(-vals)[:k]].sum()) / total
        return float(top_share)

    def family_tree(self, prediction_id: int) -> Dict[str, Any]:
        """Build a family tree (root + top-k ancestors) for a prediction id."""
        anc = self._ancestors.get(int(prediction_id))
        if anc is None:
            return {"id": prediction_id, "ancestors": []}
        infls = anc["influences"][: self.top_k]
        return {"id": anc["id"], "prediction": anc.get("prediction"),
                "n_ancestors": len(anc["influences"]),
                "ancestors": [{"sample_id": j, "influence": v}
                              for j, v in infls]}


# ============================================================================
# Idea 128 -- ModelCensus
# ============================================================================
class ModelCensus:
    """Take a census of a model's neurons: population by layer, employment
    rate (active neurons), unemployment (dead neurons), income
    distribution (weight magnitudes), and demographics (activation
    functions). The Gini coefficient measures weight inequality: 0 =
    perfectly equal weights, 1 = all weight concentrated in one neuron.
    """

    def __init__(self, active_thresh: float = 1e-3,
                 dead_tol: float = 1e-4):
        self.active_thresh = active_thresh
        self.dead_tol = dead_tol

    def conduct(self, model: Any, sample_inputs: np.ndarray) -> Dict[str, Any]:
        """Conduct a full census of the model's neurons."""
        X = np.asarray(sample_inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        weights = _weights_of(model)
        acts = _forward_activations(model, X)
        layers: List[Dict[str, Any]] = []
        all_mags: List[float] = []
        total_pop = 0
        total_dead = 0
        total_active = 0
        for name in sorted(weights):
            Wm = np.asarray(weights[name], dtype=np.float32)
            if Wm.ndim != 2:
                continue
            n_neurons = Wm.shape[1]
            col_norms = np.linalg.norm(Wm, axis=0)
            dead = int((col_norms < self.dead_tol).sum())
            act = acts.get(name)
            if act is not None:
                active = int((np.abs(act).max(axis=0) >= self.active_thresh).sum())
            else:
                active = n_neurons - dead
            mags = col_norms.tolist()
            all_mags.extend(mags)
            layers.append({"name": name, "population": n_neurons,
                           "dead": dead, "active": active,
                           "unemployed": n_neurons - active,
                           "activations": "tanh",
                           "income": mags,
                           "gini": self._gini(mags)})
            total_pop += n_neurons
            total_dead += dead
            total_active += active
        return {"layers": layers, "total_population": total_pop,
                "total_active": total_active, "total_dead": total_dead,
                "all_incomes": all_mags, "n_samples": int(X.shape[0])}

    def population_pyramid(self, census: Dict[str, Any]) -> Dict[str, int]:
        """Layer -> neuron count (the 'age' axis of the pyramid)."""
        return {l["name"]: int(l["population"]) for l in census.get("layers", [])}

    def unemployment_rate(self, census: Dict[str, Any]) -> float:
        """Fraction of neurons that are dead (unemployed)."""
        total = census.get("total_population", 0)
        dead = census.get("total_dead", 0)
        return float(dead / total) if total else 0.0

    def gini_coefficient(self, census: Dict[str, Any]) -> float:
        """Weight inequality across all neurons (0 = equal, 1 = monopoly)."""
        return self._gini(census.get("all_incomes", []))

    @staticmethod
    def _gini(values: List[float]) -> float:
        v = np.sort(np.asarray(values, dtype=np.float64))
        n = v.size
        if n == 0:
            return 0.0
        total = float(v.sum())
        if total <= 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2.0 * np.sum(idx * v) - (n + 1) * total) / (n * total))


__all__ = [
    "ModelTournament",
    "NeuralConstellation",
    "ModelCryptobiology",
    "HallucinationAutopsy",
    "WeightSeismograph",
    "ModelEmbassyProtocol",
    "NeuralArchaeology",
    "ModelSommelier",
    "PredictionGenealogy",
    "ModelCensus",
]
