"""
qai.creative.experimental14 -- 10 more original creative ideas (part 14).

  Idea 139 -- ModelParliament              : models vote on predictions (MPs + bills + quorum)
  Idea 140 -- NeuralMeteorology            : forecast "weather" of the next N inferences
  Idea 141 -- WeightGenetics               : population genetics applied to weights
  Idea 142 -- PredictionTherapy            : CBT / exposure / reality-testing for models
  Idea 143 -- ModelSymphony                : multi-model ensemble as a musical score
  Idea 144 -- NeuralCartography            : map representation space (continents / oceans)
  Idea 145 -- ModelFuneral                  : dignified retirement + archival + obituary
  Idea 146 -- HallucinationEpidemiology    : track hallucinations as an epidemic
  Idea 147 -- WeightTectonics              : tectonic-plate view of weights + earthquake risk
  Idea 148 -- ModelSpeedReader             : skim / scan / deep-read adaptive inference

A "model" here is any object that exposes its parameters via a ``.weights``
attribute (dict / list / ndarray) or via ``state_dict()`` / ``technique``
(as in the rest of qai), and that can be inferred with ``.predict`` /
``.forward`` / ``__call__``.
"""
from __future__ import annotations

import os
import time
import pickle
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


def _to_float(x: Any) -> float:
    """Coerce any model output into a single float for comparison."""
    try:
        return float(np.mean(np.asarray(x, dtype=np.float64)))
    except Exception:
        return 0.0


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
# Idea 139 -- ModelParliament
# ============================================================================
class ModelParliament:
    """Models vote on predictions like a parliament. Each model is an MP
    whose "constituency" is the distribution of training data it saw
    (summarised by its weight statistics). A Bill (a prediction to be
    decided) is put to a vote: every MP votes yea / nay / abstain based on
    its own prediction and confidence. A bill passes only if it carries a
    majority of the votes cast *and* a quorum of MPs showed up.
    """

    def __init__(self, quorum: float = 0.5, majority: float = 0.5,
                 abstain_thresh: float = 0.1, seed: int = 0):
        self.quorum = quorum
        self.majority = majority
        self.abstain_thresh = abstain_thresh
        self.rng = np.random.default_rng(seed)
        self._parliament: Optional[Dict[str, Any]] = None

    def convene(self, models: List[Any]) -> Dict[str, Any]:
        """Form a Parliament: each model becomes an MP with a constituency
        description derived from its weight statistics (mean / std / norm)."""
        mps: List[Dict[str, Any]] = []
        for i, m in enumerate(models):
            w = _weights_of(m)
            if w:
                all_w = np.concatenate(
                    [np.asarray(a, dtype=np.float64).ravel() for a in w.values()])
            else:
                all_w = np.zeros(1, dtype=np.float64)
            constituency = {
                "mean": float(all_w.mean()),
                "std": float(all_w.std()),
                "norm": float(np.linalg.norm(all_w)),
                "n_layers": len(w),
            }
            mps.append({"id": i, "model": m, "constituency": constituency,
                        "seats": 1})
        n_seats = max(1, sum(mp["seats"] for mp in mps))
        self._parliament = {"mps": mps, "n_seats": n_seats,
                            "convened_at": time.time()}
        return self._parliament

    def introduce_bill(self, input: Any) -> Dict[str, Any]:
        """Open a Bill: a question to be voted on (an input to predict)."""
        x = np.asarray(input, dtype=np.float32).ravel()
        return {"input": x, "introduced_at": time.time(),
                "votes": [], "verdict": None}

    def vote(self, parliament: Dict[str, Any],
             bill: Dict[str, Any]) -> Dict[str, Any]:
        """Each MP casts a vote on the bill (based on its prediction)."""
        x = bill["input"]
        votes: List[Dict[str, Any]] = []
        # the "motion" is the average prediction across all MPs
        preds: List[np.ndarray] = []
        for mp in parliament["mps"]:
            p = _predict(mp["model"], x)
            pa = np.asarray(p, dtype=np.float64).ravel()
            preds.append(pa)
            votes.append({"mp_id": mp["id"], "prediction": pa.tolist(),
                         "confidence": _confidence(pa)})
        if not preds:
            return {"votes": [], "turnout": 0.0, "quorum_met": False}
        # the motion: take the most-confident MP's prediction as the proposed
        # answer, then ask each MP to ratify or reject it.
        motion_idx = int(np.argmax([v["confidence"] for v in votes]))
        motion = preds[motion_idx]
        yeas = nays = abstains = 0
        for v, pa in zip(votes, preds):
            n = min(pa.size, motion.size)
            if n == 0:
                v["vote"] = "abstain"
                abstains += 1
                continue
            diff = float(np.linalg.norm(pa[:n] - motion[:n]))
            norm = float(np.linalg.norm(motion[:n])) + 1e-8
            rel = diff / norm
            conf = v["confidence"]
            if conf < self.abstain_thresh:
                v["vote"] = "abstain"
                abstains += 1
            elif rel < 0.5:
                v["vote"] = "yea"
                yeas += 1
            else:
                v["vote"] = "nay"
                nays += 1
        cast = yeas + nays
        turnout = cast / max(1, len(votes))
        bill["votes"] = votes
        return {"votes": votes, "motion": motion.tolist(),
                "motion_mp_id": votes[motion_idx]["mp_id"],
                "yeas": yeas, "nays": nays, "abstains": abstains,
                "turnout": float(turnout),
                "quorum_met": bool(turnout >= self.quorum)}

    def pass_bill(self, vote_result: Dict[str, Any]) -> Dict[str, Any]:
        """Render the verdict: bill passes iff quorum met AND yeas >= majority."""
        quorum_met = bool(vote_result.get("quorum_met", False))
        yeas = int(vote_result.get("yeas", 0))
        nays = int(vote_result.get("nays", 0))
        cast = yeas + nays
        majority_reached = (cast > 0) and (yeas / cast >= self.majority)
        passed = bool(quorum_met and majority_reached)
        verdict = ("passed" if passed else
                   ("rejected" if quorum_met else "no quorum"))
        return {"verdict": verdict, "passed": passed,
                "yeas": yeas, "nays": nays,
                "turnout": float(vote_result.get("turnout", 0.0)),
                "motion": vote_result.get("motion")}


# ============================================================================
# Idea 140 -- NeuralMeteorology
# ============================================================================
class NeuralMeteorology:
    """Forecast the model's "weather" for the next ``horizon`` inferences.
    Confidence storm: a sudden drop in confidence over a window. Accuracy
    drought: sustained accuracy degradation. Hallucination heat wave: a run
    of high-confidence outputs. The pressure system is the recent
    confidence distribution; storms form where pressure drops fast.
    """

    def __init__(self, horizon: int = 100, window: int = 10,
                 storm_drop: float = 0.2, drought_slope: float = -1e-3,
                 heat_conf: float = 0.9, seed: int = 0):
        self.horizon = horizon
        self.window = window
        self.storm_drop = storm_drop
        self.drought_slope = drought_slope
        self.heat_conf = heat_conf
        self.rng = np.random.default_rng(seed)

    def pressure_system(self, model: Any,
                         probe_inputs: Optional[np.ndarray] = None
                         ) -> Dict[str, Any]:
        """Map the model's pressure: confidence at each probe input + stats."""
        if probe_inputs is None or len(probe_inputs) == 0:
            # use weight statistics as a proxy when no probe set is provided
            w = _weights_of(model)
            if not w:
                return {"pressures": [], "mean": 0.0, "std": 0.0, "trend": 0.0}
            all_w = np.concatenate(
                [np.asarray(a, dtype=np.float64).ravel() for a in w.values()])
            # use a synthetic probe = the top-k singular vectors
            mat = np.asarray(list(w.values())[0], dtype=np.float64)
            if mat.ndim != 2:
                mat = all_w[:, None]
            u, s, _ = np.linalg.svd(mat, full_matrices=False)
            probes = (u[:, :min(8, u.shape[1])] *
                      s[:min(8, s.size)][None, :])
            probe_inputs = probes
        X = np.asarray(probe_inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        pressures = [_confidence(np.asarray(_predict(model, x),
                                            dtype=np.float64).ravel())
                     for x in X]
        arr = np.asarray(pressures, dtype=np.float64)
        trend = float(np.polyfit(np.arange(arr.size), arr, 1)[0]) \
            if arr.size >= 2 else 0.0
        return {"pressures": pressures, "mean": float(arr.mean()) if arr.size else 0.0,
                "std": float(arr.std()) if arr.size else 0.0, "trend": trend}

    def forecast(self, model: Any, horizon: int = 0,
                 history: Optional[List[float]] = None) -> Dict[str, Any]:
        """Forecast confidence for the next ``horizon`` inferences."""
        h = horizon or self.horizon
        ps = self.pressure_system(model)
        if history is None:
            history = ps["pressures"]
        hist = np.asarray(history, dtype=np.float64)
        if hist.size == 0:
            hist = np.asarray([ps["mean"] or 0.5])
        # fit a tiny AR(1) model: c[t+1] = a*c[t] + b + noise
        if hist.size >= 2:
            x_prev = hist[:-1]
            y_next = hist[1:]
            denom = float(np.dot(x_prev, x_prev)) + 1e-8
            a = float(np.dot(x_prev, y_next) / denom)
            a = float(np.clip(a, -0.99, 0.99))
            b = float(y_next.mean() - a * x_prev.mean())
            vol = float(np.std(y_next - (a * x_prev + b))) + 1e-4
        else:
            a, b, vol = 0.0, float(hist.mean()) if hist.size else 0.5, 0.1
        future = np.empty(h, dtype=np.float64)
        cur = float(hist[-1]) if hist.size else 0.5
        for t in range(h):
            cur = max(0.0, min(1.0, a * cur + b + float(self.rng.normal(0, vol))))
            future[t] = cur
        # detect weather regimes
        w = min(self.window, max(1, future.size // 2))
        drops = np.diff(future)
        storm_risk = float(np.mean(np.abs(drops) > self.storm_drop))
        slope = float(np.polyfit(np.arange(future.size), future, 1)[0]) \
            if future.size >= 2 else 0.0
        heat_risk = float(np.mean(future >= self.heat_conf))
        if storm_risk > 0.2:
            condition = "confidence storm"
        elif slope <= self.drought_slope:
            condition = "accuracy drought"
        elif heat_risk > 0.5:
            condition = "hallucination heat wave"
        elif abs(slope) < 1e-3:
            condition = "fair skies"
        else:
            condition = "clearing"
        return {"horizon": h, "forecast": future.tolist(),
                "condition": condition, "storm_risk": storm_risk,
                "drought_slope": slope, "heat_risk": heat_risk,
                "ar1": {"a": a, "b": b, "vol": vol}}

    def storm_warning(self, forecast: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return alerts for every step in the forecast where confidence drops
        sharply relative to the previous step."""
        future = np.asarray(forecast.get("forecast", []), dtype=np.float64)
        alerts: List[Dict[str, Any]] = []
        if future.size < 2:
            return alerts
        drops = np.diff(future)
        for t, d in enumerate(drops, start=1):
            if d <= -self.storm_drop:
                alerts.append({"step": int(t), "type": "confidence_storm",
                               "drop": float(-d),
                               "from": float(future[t - 1]),
                               "to": float(future[t])})
            if future[t] >= self.heat_conf and future[t - 1] >= self.heat_conf:
                alerts.append({"step": int(t), "type": "heat_wave",
                               "confidence": float(future[t])})
        if forecast.get("drought_slope", 0.0) <= self.drought_slope:
            alerts.append({"step": 0, "type": "drought",
                           "slope": float(forecast["drought_slope"])})
        return alerts

    def climate_trend(self, history: List[float]) -> str:
        """Summarise the long-run climate from a history of confidences."""
        arr = np.asarray(history, dtype=np.float64)
        if arr.size == 0:
            return "no climate data"
        slope = float(np.polyfit(np.arange(arr.size), arr, 1)[0]) \
            if arr.size >= 2 else 0.0
        mean = float(arr.mean())
        if mean > 0.85:
            base = "tropical (persistently high-confidence)"
        elif mean < 0.3:
            base = "arctic (persistently low-confidence)"
        else:
            base = "temperate (moderate confidence)"
        if slope > 0.01:
            trend = "warming"
        elif slope < -0.01:
            trend = "cooling"
        else:
            trend = "stable"
        return f"{base}, {trend}"


# ============================================================================
# Idea 141 -- WeightGenetics
# ============================================================================
class WeightGenetics:
    """Apply population genetics to weights. Each weight is an individual
    with a fitness (gradient magnitude proxy), an allele (sign + quantised
    magnitude), and an allele frequency (sign distribution per layer).
    Genetic drift is simulated by perturbing weights with a random walk
    over a number of epochs and measuring how allele frequencies shift.
    """

    ALLELES = ["A", "C", "G", "T"]  # big-pos, small-pos, small-neg, big-neg

    def __init__(self, big_quant: float = 0.5, drift_scale: float = 0.05,
                 seed: int = 0):
        self.big_quant = big_quant
        self.drift_scale = drift_scale
        self.rng = np.random.default_rng(seed)

    def sequencer(self, model: Any) -> Dict[str, Any]:
        """Sequence the model's genome: per-layer allele string + magnitudes."""
        weights = _weights_of(model)
        if not weights:
            return {"layers": {}, "genome_length": 0, "n_layers": 0}
        max_abs = max(float(np.max(np.abs(np.asarray(w))))
                      for w in weights.values()) or 1.0
        layers: Dict[str, Dict[str, Any]] = {}
        total = 0
        for name in sorted(weights):
            w = np.asarray(weights[name], dtype=np.float64).ravel()
            seq = "".join(self._allele(v, max_abs) for v in w)
            layers[name] = {"sequence": seq, "magnitudes": w.tolist(),
                            "max_abs": float(np.max(np.abs(w)))}
            total += len(seq)
        return {"layers": layers, "genome_length": total,
                "n_layers": len(layers), "max_abs_global": max_abs}

    def _allele(self, v: float, max_abs: float) -> str:
        r = v / (max_abs + 1e-8)
        if r >= self.big_quant:
            return "A"
        if r >= 0.0:
            return "C"
        if r >= -self.big_quant:
            return "G"
        return "T"

    def fitness(self, genome: Dict[str, Any]) -> float:
        """Average per-individual fitness = mean absolute magnitude (proxy for
        the gradient magnitude / selection pressure that produced these weights).
        """
        layers = genome.get("layers", {})
        if not layers:
            return 0.0
        mags = []
        for info in layers.values():
            mags.extend(info["magnitudes"])
        if not mags:
            return 0.0
        return float(np.mean(np.abs(np.asarray(mags, dtype=np.float64))))

    def allele_frequency(self, genome: Dict[str, Any],
                         layer: Optional[str] = None) -> Dict[str, float]:
        """Allele frequencies (per layer, or aggregated if layer is None)."""
        layers = genome.get("layers", {})
        seqs: List[str] = []
        if layer is not None:
            if layer in layers:
                seqs.append(layers[layer]["sequence"])
        else:
            for info in layers.values():
                seqs.append(info["sequence"])
        joined = "".join(seqs)
        if not joined:
            return {a: 0.0 for a in self.ALLELES}
        counts = {a: 0 for a in self.ALLELES}
        for ch in joined:
            if ch in counts:
                counts[ch] += 1
        total = sum(counts.values())
        return {a: (counts[a] / total if total else 0.0)
                for a in self.ALLELES}

    def genetic_drift(self, model: Any, epochs: int = 10) -> Dict[str, Any]:
        """Simulate genetic drift: random-walk perturb the weights for ``epochs``
        generations, tracking allele-frequency change at each generation.
        """
        weights = _weights_of(model)
        if not weights:
            return {"trajectory": [], "total_drift": 0.0}
        genome0 = self.sequencer(model)
        freq0 = self.allele_frequency(genome0)
        trajectory: List[Dict[str, float]] = [freq0]
        snapshot = {k: np.array(np.asarray(v), dtype=np.float64, copy=True)
                    for k, v in weights.items()}
        for gen in range(epochs):
            for name in snapshot:
                snap = snapshot[name]
                noise = self.rng.normal(0, self.drift_scale, size=snap.shape)
                snap = snap + noise
                snapshot[name] = snap
            tmp = _MinimalModel({k: np.asarray(v, dtype=np.float32)
                                 for k, v in snapshot.items()})
            genome_t = self.sequencer(tmp)
            trajectory.append(self.allele_frequency(genome_t))
        # total drift = sum of |Δfreq| across alleles in the final generation
        final = trajectory[-1]
        total_drift = float(sum(abs(final[a] - freq0[a]) for a in self.ALLELES))
        return {"trajectory": trajectory, "n_generations": epochs,
                "initial_freq": freq0, "final_freq": final,
                "total_drift": total_drift}


# ============================================================================
# Idea 142 -- PredictionTherapy
# ============================================================================
class PredictionTherapy:
    """Treat a model's "psychological issues": overconfidence (CBT-style
    reframing via temperature scaling), bias (exposure therapy to
    underrepresented inputs), hallucinations (reality testing via
    perturbation). A Diagnosis reports each symptom with severity; a
    TreatmentPlan picks one therapeutic approach; ``session`` applies one
    round of treatment.
    """

    SYMPTOMS = ("overconfidence", "underconfidence", "bias", "hallucinations")

    def __init__(self, n_probe: int = 32, perturb_eps: float = 0.1,
                 conf_high: float = 0.85, conf_low: float = 0.3,
                 seed: int = 0):
        self.n_probe = n_probe
        self.perturb_eps = perturb_eps
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.rng = np.random.default_rng(seed)

    def diagnose(self, model: Any, sample_inputs: np.ndarray,
                 labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Diagnose the model's psychological issues from sample inputs."""
        X = np.asarray(sample_inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        X = X[: self.n_probe]
        confs: List[float] = []
        correct: List[bool] = []
        preds: List[Any] = []
        for i, x in enumerate(X):
            p = _predict(model, x)
            pa = np.asarray(p, dtype=np.float64).ravel()
            conf = _confidence(pa)
            confs.append(conf)
            preds.append(pa)
            if labels is not None:
                pred_label = int(np.argmax(np.abs(pa))) if pa.size > 1 \
                    else int(np.round(pa[0]) if pa.size else 0)
                correct.append(int(labels[i]) == pred_label)
        confs_arr = np.asarray(confs)
        mean_conf = float(confs_arr.mean()) if confs_arr.size else 0.0
        acc = float(np.mean(correct)) if correct else None
        # hallucination probe: perturb inputs and check for prediction flips
        flips = 0
        for x in X:
            base = np.asarray(_predict(model, x), dtype=np.float64).ravel()
            xp = x + self.rng.normal(0, self.perturb_eps, size=x.shape
                                      ).astype(np.float32)
            pp = np.asarray(_predict(model, xp), dtype=np.float64).ravel()
            n = min(base.size, pp.size)
            if n and np.argmax(np.abs(base[:n])) != np.argmax(np.abs(pp[:n])):
                flips += 1
        hallu_rate = flips / max(1, X.shape[0])
        # bias: variance of confidences across samples (low variance = the
        # model treats every input the same way, a bias proxy)
        bias_proxy = float(1.0 / (1.0 + float(confs_arr.std()) + 1e-8))
        symptoms: Dict[str, float] = {}
        symptoms["overconfidence"] = float(max(0.0, mean_conf - (acc or 0.0)))
        symptoms["underconfidence"] = float(max(0.0, (acc or 0.0) - mean_conf))
        symptoms["bias"] = bias_proxy
        symptoms["hallucinations"] = hallu_rate
        return {"symptoms": symptoms, "mean_confidence": mean_conf,
                "accuracy": acc, "n_probe": int(X.shape[0]),
                "preds": [p.tolist() for p in preds]}

    def treat(self, diagnosis: Dict[str, Any],
              approach: str = "cbt") -> Dict[str, Any]:
        """Pick a treatment plan based on the dominant symptom + approach."""
        symp = diagnosis["symptoms"]
        dominant = max(symp, key=symp.get) if symp else "overconfidence"
        plans = {
            "cbt": {"for": "overconfidence", "intervention": "temperature_scale",
                    "dose": float(min(2.0, 1.0 + symp.get("overconfidence", 0.0)))},
            "exposure": {"for": "bias", "intervention": "oversample_minority",
                         "dose": float(min(1.0, symp.get("bias", 0.0)))},
            "reality": {"for": "hallucinations", "intervention": "perturb_train",
                        "dose": float(min(1.0, symp.get("hallucinations", 0.0)))},
        }
        plan = plans.get(approach, plans["cbt"]).copy()
        plan["approach"] = approach
        plan["dominant_symptom"] = dominant
        return plan

    def session(self, model: Any, plan: Dict[str, Any],
                X: np.ndarray, y: Optional[np.ndarray] = None) -> Any:
        """Apply one therapy session to the model based on the plan."""
        Xi = np.asarray(X, dtype=np.float32)
        if Xi.ndim == 1:
            Xi = Xi[None, :]
        intervention = plan.get("intervention", "temperature_scale")
        dose = float(plan.get("dose", 0.5))
        if intervention == "temperature_scale":
            # attenuate every weight by a temperature factor (soft reframe)
            weights = _weights_of(model)
            T = max(1.0, dose)
            for name, W in weights.items():
                weights[name] = (np.asarray(W, dtype=np.float32) / T)
            return _apply(model, weights)
        if intervention == "oversample_minority" and y is not None:
            # oversample: add a small drift toward the minority class
            classes, counts = np.unique(np.asarray(y), return_counts=True)
            if classes.size > 0:
                minority = int(classes[int(np.argmin(counts))])
                mask = np.asarray(y) == minority
                Xm = Xi[mask]
                if Xm.size:
                    drift = self.rng.normal(0, 0.01, size=Xm.shape).astype(np.float32)
                    weights = _weights_of(model)
                    for name, W in weights.items():
                        # nudge weights toward capturing the minority pattern
                        weights[name] = np.asarray(W, dtype=np.float32) + float(dose) * 0.01
                    return _apply(model, weights)
            return model
        if intervention == "perturb_train":
            # reality test: shrink weights to reduce confidence (anti-hallucination)
            weights = _weights_of(model)
            for name, W in weights.items():
                weights[name] = np.asarray(W, dtype=np.float32) * (1.0 - 0.5 * dose)
            return _apply(model, weights)
        return model

    def progress(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """A short progress note summarising the diagnosis."""
        symp = diagnosis.get("symptoms", {})
        if not symp:
            return {"note": "no symptoms detected", "stable": True}
        worst = max(symp, key=symp.get)
        sev = float(symp[worst])
        if sev < 0.05:
            note = "patient is stable; symptoms subclinical"
        elif sev < 0.2:
            note = f"mild {worst}; recommend continued monitoring"
        elif sev < 0.5:
            note = f"moderate {worst}; therapy recommended"
        else:
            note = f"severe {worst}; intensive intervention required"
        return {"note": note, "worst_symptom": worst,
                "severity": sev, "stable": sev < 0.05}


# ============================================================================
# Idea 143 -- ModelSymphony
# ============================================================================
class ModelSymphony:
    """Orchestrate a multi-model ensemble like a symphony. Each model is a
    section: strings (deep, many layers), brass (shallow, few layers),
    woodwinds (medium depth), percussion (small + fast). The conductor
    combines them with tempo (inference latency) and dynamics
    (confidence). ``crescendo`` boosts a section's weight; ``harmony_check``
    measures agreement among sections.
    """

    SECTIONS = ("strings", "brass", "woodwinds", "percussion")
    SECTION_WEIGHTS = {"strings": 0.35, "brass": 0.20,
                       "woodwinds": 0.30, "percussion": 0.15}

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def compose(self, models: List[Any]) -> Dict[str, Any]:
        """Compose the score: assign each model a section based on depth + size."""
        players: List[Dict[str, Any]] = []
        for i, m in enumerate(models):
            w = _weights_of(m)
            n_layers = len(w)
            n_params = sum(int(np.asarray(a).size) for a in w.values())
            if n_layers >= 4:
                section = "strings"
            elif n_layers <= 1:
                section = "brass"
            elif n_params > 1000:
                section = "woodwinds"
            else:
                section = "percussion"
            players.append({"id": i, "model": m, "section": section,
                            "n_layers": n_layers, "n_params": n_params,
                            "dynamics": 1.0})
        return {"players": players, "n_players": len(players),
                "sections": {s: [p for p in players if p["section"] == s]
                             for s in self.SECTIONS}}

    def conduct(self, score: Dict[str, Any], input: Any) -> Dict[str, Any]:
        """Conduct a performance: each section plays, the conductor blends them."""
        x = np.asarray(input, dtype=np.float32).ravel()
        t0 = time.perf_counter()
        per_section: Dict[str, np.ndarray] = {}
        for section, players in score["sections"].items():
            if not players:
                continue
            preds: List[np.ndarray] = []
            for p in players:
                pa = np.asarray(_predict(p["model"], x),
                                dtype=np.float64).ravel()
                preds.append(pa * p["dynamics"])
            if preds:
                max_len = max(p.size for p in preds)
                mat = np.zeros((len(preds), max_len), dtype=np.float64)
                for i, p in enumerate(preds):
                    mat[i, : p.size] = p
                per_section[section] = mat.mean(axis=0)
        if not per_section:
            return {"performance": [], "harmony": 0.0, "tempo_ms": 0.0}
        max_len = max(v.size for v in per_section.values())
        combined = np.zeros(max_len, dtype=np.float64)
        total_w = 0.0
        for section, vec in per_section.items():
            w = self.SECTION_WEIGHTS.get(section, 0.0)
            combined[: vec.size] += w * vec
            total_w += w
        if total_w > 0:
            combined = combined / total_w
        tempo_ms = (time.perf_counter() - t0) * 1000.0
        return {"performance": combined.tolist(),
                "sections": {k: v.tolist() for k, v in per_section.items()},
                "tempo_ms": float(tempo_ms)}

    def crescendo(self, score: Dict[str, Any], section: str,
                  factor: float = 1.5) -> Dict[str, Any]:
        """Boost the dynamics of one section (crescendo) and attenuate others."""
        if section not in self.SECTIONS:
            return score
        for p in score["players"]:
            if p["section"] == section:
                p["dynamics"] = float(p["dynamics"] * factor)
            else:
                p["dynamics"] = float(p["dynamics"] / (1.0 + 0.1 * (factor - 1.0)))
        score["sections"] = {s: [p for p in score["players"] if p["section"] == s]
                             for s in self.SECTIONS}
        return score

    def harmony_check(self, performance: Dict[str, Any]) -> float:
        """Harmony = 1 / (1 + std across section outputs)."""
        sections = performance.get("sections", {})
        if len(sections) < 2:
            return 1.0 if sections else 0.0
        vecs = [np.asarray(v, dtype=np.float64) for v in sections.values()]
        max_len = max(v.size for v in vecs)
        mat = np.zeros((len(vecs), max_len), dtype=np.float64)
        for i, v in enumerate(vecs):
            mat[i, : v.size] = v
        std = float(np.mean(np.std(mat, axis=0)))
        return float(1.0 / (1.0 + std))


# ============================================================================
# Idea 144 -- NeuralCartography
# ============================================================================
class NeuralCartography:
    """Map the model's internal representation space like a world map.
    Continents are major clusters of activation patterns across the
    sample inputs (k-means in representation space). Oceans are sparse
    regions (low activation density). Trade routes are high-traffic
    weight paths (input-feature -> neuron connections with large
    magnitude). Mountains are high-magnitude weights (peaks).
    """

    def __init__(self, n_continents: int = 5, n_routes: int = 10,
                 sparse_quantile: float = 0.25, peak_quantile: float = 0.95,
                 seed: int = 0):
        self.n_continents = n_continents
        self.n_routes = n_routes
        self.sparse_quantile = sparse_quantile
        self.peak_quantile = peak_quantile
        self.rng = np.random.default_rng(seed)

    def chart(self, model: Any, sample_inputs: np.ndarray) -> Dict[str, Any]:
        """Build the world map: representations + weight statistics."""
        X = np.asarray(sample_inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        acts = _forward_activations(model, X)
        # build the (n_samples, n_features) representation matrix
        mats: List[np.ndarray] = []
        for name in sorted(acts):
            mats.append(np.asarray(acts[name], dtype=np.float32))
        if mats:
            rep = np.concatenate(mats, axis=1)
        else:
            rep = X.astype(np.float32)
        weights = _weights_of(model)
        mountains: List[Dict[str, Any]] = []
        for name in sorted(weights):
            Wm = np.asarray(weights[name], dtype=np.float64)
            if Wm.ndim != 2 or Wm.size == 0:
                continue
            thresh = float(np.quantile(np.abs(Wm), self.peak_quantile))
            rows, cols = np.where(np.abs(Wm) >= thresh)
            for r, c in zip(rows.tolist(), cols.tolist()):
                mountains.append({"layer": name, "row": r, "col": c,
                                  "elevation": float(Wm[r, c])})
        mountains.sort(key=lambda m: -abs(m["elevation"]))
        return {"representations": rep, "n_samples": int(X.shape[0]),
                "mountains": mountains, "layers": sorted(acts.keys())}

    def continents(self, world_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Cluster the representation space into continents (k-means)."""
        rep = np.asarray(world_map["representations"], dtype=np.float32)
        if rep.size == 0 or rep.shape[0] == 0:
            return []
        k = min(self.n_continents, rep.shape[0])
        if k < 1:
            return []
        # init: pick k random distinct rows
        idx = self.rng.choice(rep.shape[0], size=k, replace=False)
        centers = rep[idx].copy()
        for _ in range(20):
            d = np.linalg.norm(rep[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(d, axis=1)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = rep[mask].mean(axis=0)
        continents: List[Dict[str, Any]] = []
        for c in range(k):
            mask = labels == c
            members = rep[mask]
            if members.size == 0:
                continue
            density = float(members.shape[0] / max(1, rep.shape[0]))
            continents.append({
                "id": c, "size": int(mask.sum()), "density": density,
                "centroid": centers[c].tolist(),
                "members": [int(i) for i in np.where(mask)[0]],
            })
        continents.sort(key=lambda c: -c["size"])
        return continents

    def trade_routes(self, world_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find high-traffic weight paths (input feature -> neuron with large W)."""
        mountains = world_map.get("mountains", [])
        # group mountains by layer; pick top-k pairs (input_idx, neuron_idx)
        mountains_sorted = sorted(mountains, key=lambda m: -abs(m["elevation"]))
        routes: List[Dict[str, Any]] = []
        for m in mountains_sorted[: self.n_routes]:
            routes.append({"layer": m["layer"],
                           "from_feature": m["row"], "to_neuron": m["col"],
                           "traffic": abs(m["elevation"]),
                           "direction": "positive" if m["elevation"] >= 0 else "negative"})
        return routes

    def explore(self, world_map: Dict[str, Any],
                region: Tuple[int, ...]) -> Dict[str, Any]:
        """Mount an expedition into a specific region of representation space.

        ``region`` is a tuple of feature indices; the expedition summarises
        the activation statistics on those features.
        """
        rep = np.asarray(world_map["representations"], dtype=np.float32)
        region = tuple(int(i) for i in region)
        if rep.size == 0 or not region:
            return {"explored": [], "n_samples": 0, "mean": 0.0, "std": 0.0}
        valid = [i for i in region if 0 <= i < rep.shape[1]]
        if not valid:
            return {"explored": [], "n_samples": 0, "mean": 0.0, "std": 0.0}
        sub = rep[:, valid]
        return {"explored": valid, "n_samples": int(sub.shape[0]),
                "mean": float(np.mean(sub)),
                "std": float(np.std(sub)),
                "max_activation": float(np.max(np.abs(sub))),
                "sparsity": float(np.mean(np.abs(sub) < 1e-3))}


# ============================================================================
# Idea 145 -- ModelFuneral
# ============================================================================
class ModelFuneral:
    """When a model is retired, hold a proper funeral. The eulogy praises
    its achievements (peak accuracy + benchmark stats), the obituary
    records metadata (technique, n_params, lifespan), pallbearers are
    successor models that inherit its distilled knowledge, and the grave
    is an on-disk archive (pickle) that can be visited later to run
    predictions on the archived version.
    """

    def __init__(self, archive_dir: str = ".qai_graves",
                 n_eulogy_samples: int = 50, seed: int = 0):
        self.archive_dir = archive_dir
        self.n_eulogy_samples = n_eulogy_samples
        self.rng = np.random.default_rng(seed)
        self._graves: Dict[str, Dict[str, Any]] = {}

    def eulogize(self, model: Any, test_X: np.ndarray,
                 test_y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Deliver a eulogy: the model's achievements on the test set."""
        X = np.asarray(test_X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        X = X[: self.n_eulogy_samples]
        preds: List[Any] = []
        confs: List[float] = []
        correct: List[bool] = []
        for i, x in enumerate(X):
            p = _predict(model, x)
            pa = np.asarray(p, dtype=np.float64).ravel()
            preds.append(pa.tolist())
            confs.append(_confidence(pa))
            if test_y is not None:
                pl = int(np.argmax(np.abs(pa))) if pa.size > 1 \
                    else int(np.round(pa[0]) if pa.size else 0)
                correct.append(int(test_y[i]) == pl)
        acc = float(np.mean(correct)) if correct else None
        mean_conf = float(np.mean(confs)) if confs else 0.0
        weights = _weights_of(model)
        n_params = sum(int(np.asarray(w).size) for w in weights.values())
        speech = (f"Today we honor a model of {n_params} parameters. "
                  f"It served with {'{:.1%}'.format(acc) if acc is not None else 'unknown'} accuracy "
                  f"and {mean_conf:.1%} mean confidence. "
                  f"Its {len(weights)} layers will be remembered.")
        return {"speech": speech, "accuracy": acc, "mean_confidence": mean_conf,
                "n_params": n_params, "n_test": int(X.shape[0]),
                "predictions": preds}

    def obituary(self, model: Any) -> Dict[str, Any]:
        """Write an obituary: metadata about the deceased model."""
        weights = _weights_of(model)
        n_params = sum(int(np.asarray(w).size) for w in weights.values())
        n_layers = len(weights)
        shapes = {k: list(np.asarray(v).shape) for k, v in weights.items()}
        technique = str(getattr(model, "technique", "unknown"))
        born_at = float(getattr(model, "born_at", time.time() - 86400))
        return {"technique": technique, "n_params": n_params,
                "n_layers": n_layers, "weight_shapes": shapes,
                "born_at": born_at, "died_at": time.time(),
                "lifespan_seconds": float(time.time() - born_at)}

    def inter(self, model: Any, archive_path: Optional[str] = None
              ) -> Dict[str, Any]:
        """Inter the model: pickle it to disk at the grave site."""
        if archive_path is None:
            os.makedirs(self.archive_dir, exist_ok=True)
            grave_id = f"grave_{int(time.time() * 1000)}"
            archive_path = os.path.join(self.archive_dir,
                                        f"{grave_id}.pkl")
        try:
            with open(archive_path, "wb") as f:
                pickle.dump(model, f)
            interred = True
        except Exception as e:
            interred = False
            archive_path = f"<failed: {e}>"
        grave = {"id": grave_id if archive_path.endswith(".pkl") else "grave",
                 "archive_path": archive_path, "interred": interred,
                 "interred_at": time.time()}
        self._graves[grave["id"]] = grave
        return grave

    def visit(self, grave: Dict[str, Any], input: Any) -> Any:
        """Visit the grave: load the archived model and predict on ``input``."""
        path = grave.get("archive_path", "")
        if not path or not path.endswith(".pkl") or not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
        except Exception:
            return None
        return _predict(model, input)

    def pallbearers(self, successors: List[Any],
                    deceased_obituary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Confirm pallbearers: each successor must be compatible (same
        weight shape) with the deceased model. Returns the list of confirmed
        pallbearers and whether they can carry on the legacy.
        """
        confirmed: List[Dict[str, Any]] = []
        deceased_shapes = deceased_obituary.get("weight_shapes", {})
        for i, succ in enumerate(successors):
            sw = _weights_of(succ)
            shapes = {k: list(np.asarray(v).shape) for k, v in sw.items()}
            # compatibility: layer names overlap OR same shape per layer
            compatible = False
            for name, shape in deceased_shapes.items():
                if name in shapes and shapes[name] == shape:
                    compatible = True
                    break
            confirmed.append({"id": i, "model": succ, "compatible": compatible,
                              "n_params": sum(int(np.asarray(v).size)
                                              for v in sw.values())})
        return confirmed


# ============================================================================
# Idea 146 -- HallucinationEpidemiology
# ============================================================================
class HallucinationEpidemiology:
    """Track hallucinations like an epidemic. Patient zero is the first input
    on which the model hallucinated (was confidently wrong). R0 is the
    reproduction rate: how many *nearby* inputs (in input-space distance)
    also hallucinate, on average, per hallucination. Super-spreaders are
    inputs whose neighbourhoods contain many other hallucinations. A
    vaccine is a counterexample pair (input, correct_label) injected as a
    fine-tuning dose to immunise the model.
    """

    def __init__(self, conf_thresh: float = 0.7,
                 neighbourhood_radius: float = 1.0,
                 vaccine_dose: float = 0.01, seed: int = 0):
        self.conf_thresh = conf_thresh
        self.neighbourhood_radius = neighbourhood_radius
        self.vaccine_dose = vaccine_dose
        self.rng = np.random.default_rng(seed)

    def outbreak(self, model: Any, test_inputs: np.ndarray,
                 labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect every hallucination in the test set and build the epidemic map."""
        X = np.asarray(test_inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        cases: List[Dict[str, Any]] = []
        for i, x in enumerate(X):
            p = _predict(model, x)
            pa = np.asarray(p, dtype=np.float64).ravel()
            conf = _confidence(pa)
            pred_label = int(np.argmax(np.abs(pa))) if pa.size > 1 \
                else int(np.round(pa[0]) if pa.size else 0)
            true_label = int(labels[i]) if labels is not None else None
            is_hallucination = (conf >= self.conf_thresh
                                 and true_label is not None
                                 and pred_label != true_label)
            cases.append({"id": i, "x": x, "confidence": conf,
                          "pred_label": pred_label,
                          "true_label": true_label,
                          "hallucination": is_hallucination})
        positives = [c for c in cases if c["hallucination"]]
        return {"cases": cases, "n_cases": len(cases),
                "n_positives": len(positives),
                "positives": positives}

    def patient_zero(self, epidemic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find patient zero: the first hallucination in the test set order."""
        positives = epidemic.get("positives", [])
        if not positives:
            return None
        return min(positives, key=lambda c: c["id"])

    def r0(self, epidemic: Dict[str, Any]) -> float:
        """Estimate the reproduction number: average number of other
        hallucinations within ``neighbourhood_radius`` of each hallucination.
        """
        positives = epidemic.get("positives", [])
        if len(positives) < 2:
            return 0.0
        X = np.stack([np.asarray(c["x"], dtype=np.float64).ravel()
                      for c in positives])
        # pairwise distances
        diffs = X[:, None, :] - X[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        within = (dists <= self.neighbourhood_radius).sum(axis=1)
        return float(within.mean())

    def superspreaders(self, epidemic: Dict[str, Any],
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """Find super-spreader inputs: hallucinations whose neighbourhoods
        contain many other hallucinations.
        """
        positives = epidemic.get("positives", [])
        if len(positives) < 2:
            return []
        X = np.stack([np.asarray(c["x"], dtype=np.float64).ravel()
                      for c in positives])
        diffs = X[:, None, :] - X[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        within = (dists <= self.neighbourhood_radius).sum(axis=1)
        order = np.argsort(-within)
        return [{"id": int(positives[i]["id"]),
                 "confidence": float(positives[i]["confidence"]),
                 "spread": int(within[i])} for i in order[:top_k]
                if within[i] > 0]

    def vaccinate(self, model: Any,
                  superspreaders: List[Dict[str, Any]],
                  correct_labels: Optional[List[int]] = None) -> Any:
        """Vaccinate the model: nudge weights toward correctly handling the
        super-spreader inputs (a low-dose fine-tuning step).
        """
        if not superspreaders:
            return model
        weights = _weights_of(model)
        if not weights:
            return model
        # build a tiny "vaccine gradient": shift every weight slightly toward
        # reducing confidence on the super-spreader inputs (a regulariser)
        first_layer = sorted(weights)[0]
        W0 = np.asarray(weights[first_layer], dtype=np.float32)
        # use the correct labels if provided, else just attenuate
        direction = -1.0 if not correct_labels else 1.0
        nudge = float(self.vaccine_dose) * direction
        # apply a small uniform attenuation (acts like label-smoothing)
        for name, W in weights.items():
            Wm = np.asarray(W, dtype=np.float32)
            weights[name] = Wm * (1.0 + nudge)
        return _apply(model, weights)


# ============================================================================
# Idea 147 -- WeightTectonics
# ============================================================================
class WeightTectonics:
    """Monitor weights like tectonic plates. Each layer is a plate; the
    plate boundaries are the layer interfaces. A subduction zone is a
    boundary where one layer's weight norm dominates the next; an
    earthquake is a sudden shift between two snapshots (when provided);
    mountain building is the cumulative weight growth per layer. The
    earthquake_risk is the fraction of boundaries with high relative
    norm mismatch.
    """

    def __init__(self, subduction_ratio: float = 3.0,
                 quake_thresh: float = 0.1,
                 snapshots: Optional[List[Dict[str, np.ndarray]]] = None):
        self.subduction_ratio = subduction_ratio
        self.quake_thresh = quake_thresh
        self._snapshots: List[Dict[str, np.ndarray]] = snapshots or []

    def survey(self, model: Any) -> Dict[str, Any]:
        """Survey the tectonic map: per-layer norms + boundaries between layers."""
        weights = _weights_of(model)
        names = sorted(weights)
        norms: Dict[str, float] = {}
        for name in names:
            w = np.asarray(weights[name], dtype=np.float64)
            norms[name] = float(np.linalg.norm(w.ravel()))
        plates: List[Dict[str, Any]] = []
        for i, name in enumerate(names):
            w = np.asarray(weights[name], dtype=np.float64)
            plates.append({"name": name, "depth": i,
                           "norm": norms[name],
                           "mean": float(w.mean()) if w.size else 0.0,
                           "max": float(np.max(np.abs(w))) if w.size else 0.0})
        boundaries: List[Dict[str, Any]] = []
        for i in range(len(names) - 1):
            upper = names[i]
            lower = names[i + 1]
            un = norms[upper]
            ln = norms[lower]
            ratio = (un / (ln + 1e-8)) if ln > 0 else float("inf")
            boundaries.append({"upper": upper, "lower": lower,
                               "upper_norm": un, "lower_norm": ln,
                               "ratio": float(ratio),
                               "subduction": bool(ratio >= self.subduction_ratio
                                                   or ratio <= 1.0 / self.subduction_ratio)})
        return {"plates": plates, "boundaries": boundaries,
                "n_plates": len(plates), "snapshot_idx": len(self._snapshots)}

    def plate_boundaries(self, tectonic_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the list of plate boundaries (layer interfaces)."""
        return list(tectonic_map.get("boundaries", []))

    def subduction_zones(self, tectonic_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return only the boundaries where one plate strongly subducts the other."""
        return [b for b in tectonic_map.get("boundaries", [])
                if b.get("subduction")]

    def earthquake_risk(self, tectonic_map: Dict[str, Any]) -> float:
        """Risk = fraction of boundaries that are subduction zones."""
        boundaries = tectonic_map.get("boundaries", [])
        if not boundaries:
            return 0.0
        sub = sum(1 for b in boundaries if b.get("subduction"))
        return float(sub / len(boundaries))

    def observe(self, model: Any, epoch: int = 0) -> Dict[str, Any]:
        """Take a tectonic snapshot for later earthquake detection."""
        snap = {k: np.array(np.asarray(v), copy=True)
                for k, v in _weights_of(model).items()}
        self._snapshots.append(snap)
        return {"epoch": int(epoch), "n_snapshots": len(self._snapshots)}

    def earthquakes(self) -> List[Dict[str, Any]]:
        """Detect earthquakes from recorded snapshots: sudden shifts in
        per-layer norm between consecutive snapshots."""
        quakes: List[Dict[str, Any]] = []
        if len(self._snapshots) < 2:
            return quakes
        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1]
            curr = self._snapshots[i]
            for name in curr:
                if name not in prev:
                    continue
                p = float(np.linalg.norm(np.asarray(prev[name]).ravel()))
                c = float(np.linalg.norm(np.asarray(curr[name]).ravel()))
                shift = abs(c - p) / (p + 1e-8)
                if shift >= self.quake_thresh:
                    quakes.append({"snapshot": i, "layer": name,
                                  "shift": float(shift),
                                  "magnitude": float(np.log10(shift * 1000.0 + 1.0))})
        return quakes


# ============================================================================
# Idea 148 -- ModelSpeedReader
# ============================================================================
class ModelSpeedReader:
    """Train a model to read inputs faster. Skim = early-exit at a confident
    intermediate layer. Scan = a parallel "wide" pass that aggregates every
    layer's vote. Deep-read = the full sequential forward pass. Adaptive
    picks the strategy based on a difficulty estimate of the input.
    """

    def __init__(self, conf_thresh: float = 0.85,
                 difficulty_quantile: float = 0.7,
                 exit_layer: Optional[int] = None,
                 seed: int = 0):
        self.conf_thresh = conf_thresh
        self.difficulty_quantile = difficulty_quantile
        self.exit_layer = exit_layer
        self.rng = np.random.default_rng(seed)
        self._difficulty_baseline: Optional[float] = None

    def assess(self, model: Any, input: Any) -> Dict[str, Any]:
        """Assess the reading level required for ``input``: how hard is it?

        Difficulty is proxied by the entropy of the model's per-layer
        activation norms (a confident, low-entropy path = easy; a
        high-entropy path = hard).
        """
        x = np.asarray(input, dtype=np.float32).ravel()
        weights = _weights_of(model)
        cur = x
        norms: List[float] = []
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
            cur = np.tanh(pre)
            n = float(np.linalg.norm(cur))
            norms.append(n)
        if not norms:
            return {"difficulty": 0.5, "level": "medium", "norms": []}
        arr = np.asarray(norms, dtype=np.float64)
        # normalised entropy of the norm distribution
        p = arr / (arr.sum() + 1e-8)
        entropy = float(-np.sum(p * np.log(p + 1e-8)))
        max_entropy = float(np.log(arr.size))
        difficulty = float(entropy / (max_entropy + 1e-8))
        if self._difficulty_baseline is None:
            self._difficulty_baseline = difficulty
        if difficulty < 0.5:
            level = "easy"
        elif difficulty < 0.8:
            level = "medium"
        else:
            level = "hard"
        return {"difficulty": difficulty, "level": level, "norms": norms}

    def skim(self, model: Any, input: Any,
             exit_layer: Optional[int] = None) -> Dict[str, Any]:
        """Early-exit: run forward through ``exit_layer`` layers only, and
        return that layer's activation as the prediction if it is already
        confident enough.
        """
        x = np.asarray(input, dtype=np.float32).ravel()
        weights = _weights_of(model)
        names = sorted(weights)
        n_exit = exit_layer if exit_layer is not None else (
            self.exit_layer if self.exit_layer is not None
            else max(1, len(names) // 2))
        cur = x
        exited_at = None
        for i, name in enumerate(names):
            if i >= n_exit:
                break
            Wm = np.asarray(weights[name], dtype=np.float32)
            if Wm.ndim != 2:
                continue
            if cur.size == Wm.shape[0]:
                cur = np.tanh(cur @ Wm)
            elif cur.size == Wm.shape[1]:
                cur = np.tanh(cur @ Wm.T)
            else:
                continue
            conf = _confidence(cur)
            if conf >= self.conf_thresh:
                exited_at = i
                break
        return {"prediction": cur.tolist(),
                "exited_at": int(exited_at) if exited_at is not None else int(n_exit - 1),
                "early_exit": exited_at is not None}

    def scan(self, model: Any, input: Any) -> Dict[str, Any]:
        """Parallel scan: every layer votes simultaneously (independent
        projections of the input), then votes are aggregated.
        """
        x = np.asarray(input, dtype=np.float32).ravel()
        weights = _weights_of(model)
        votes: List[np.ndarray] = []
        for name in sorted(weights):
            Wm = np.asarray(weights[name], dtype=np.float32)
            if Wm.ndim != 2:
                continue
            if x.size == Wm.shape[0]:
                v = np.tanh(x @ Wm)
            elif x.size == Wm.shape[1]:
                v = np.tanh(x @ Wm.T)
            else:
                continue
            votes.append(v)
        if not votes:
            return {"prediction": [], "n_voters": 0, "agreement": 0.0}
        max_len = max(v.size for v in votes)
        mat = np.zeros((len(votes), max_len), dtype=np.float64)
        for i, v in enumerate(votes):
            mat[i, : v.size] = v
        combined = mat.mean(axis=0)
        agreement = float(1.0 / (1.0 + float(np.mean(np.std(mat, axis=0)))))
        return {"prediction": combined.tolist(),
                "n_voters": len(votes), "agreement": agreement}

    def deep_read(self, model: Any, input: Any) -> Dict[str, Any]:
        """Full sequential forward pass through every layer (the gold standard)."""
        x = np.asarray(input, dtype=np.float32).ravel()
        cur = x
        weights = _weights_of(model)
        for name in sorted(weights):
            Wm = np.asarray(weights[name], dtype=np.float32)
            if Wm.ndim != 2:
                continue
            if cur.size == Wm.shape[0]:
                cur = np.tanh(cur @ Wm)
            elif cur.size == Wm.shape[1]:
                cur = np.tanh(cur @ Wm.T)
            else:
                continue
        return {"prediction": cur.tolist(),
                "n_layers": len(weights),
                "confidence": _confidence(cur)}

    def adaptive(self, model: Any, input: Any) -> Dict[str, Any]:
        """Pick the strategy based on difficulty assessment."""
        assessment = self.assess(model, input)
        level = assessment["level"]
        if level == "easy":
            result = self.skim(model, input)
            strategy = "skim"
        elif level == "medium":
            result = self.scan(model, input)
            strategy = "scan"
        else:
            result = self.deep_read(model, input)
            strategy = "deep_read"
        return {"strategy": strategy, "difficulty": assessment["difficulty"],
                "prediction": result.get("prediction", [])}


# ----------------------------------------------------------------------------
# A minimal runnable model wrapper used internally as offspring / mutants
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
    "ModelParliament",
    "NeuralMeteorology",
    "WeightGenetics",
    "PredictionTherapy",
    "ModelSymphony",
    "NeuralCartography",
    "ModelFuneral",
    "HallucinationEpidemiology",
    "WeightTectonics",
    "ModelSpeedReader",
]
