"""
qai.creative.experimental13 -- 10 more original creative ideas (part 13).

  Idea 129 -- ModelDiplomacy              : models negotiate + veto consensus predictions
  Idea 130 -- NeuralGeology               : weight matrices as geological strata + mineral veins
  Idea 131 -- PredictionCourt             : prediction on trial (prosecution vs defense)
  Idea 132 -- WeightTides                 : weights ebb/flow like ocean tides during training
  Idea 133 -- ModelEconomy                 : models buy samples + sell predictions (GDP/inflation)
  Idea 134 -- HallucinationCartography    : topo map of confident-wrong + uncertain-correct
  Idea 135 -- ModelChoir                   : models sing as soprano/alto/tenor/bass voices
  Idea 136 -- NeuralForensics             : crime-scene investigation of a trained model
  Idea 137 -- ModelCultivation            : grow a model as a crop (plant/water/prune/harvest)
  Idea 138 -- PredictionArcheology        : dig through a prediction's reasoning strata

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


# ============================================================================
# Idea 129 -- ModelDiplomacy
# ============================================================================
class ModelDiplomacy:
    """Models negotiate with each other to reach consensus predictions.
    Each model is a "delegate" with a voting weight proportional to its
    validation accuracy. The council debates each input by collecting
    per-delegate predictions, then resolves a verdict by weighted
    majority vote. Any delegate whose own prediction diverges too far
    from the verdict may exercise a veto.
    """

    def __init__(self, veto_tol: float = 0.5, default_weight: float = 1.0):
        self.veto_tol = veto_tol
        self.default_weight = default_weight
        self._council: Optional[Dict[str, Any]] = None

    def convene(self, models: List[Any],
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Assemble the council; weight each delegate by validation accuracy."""
        delegates: List[Dict[str, Any]] = []
        if X_val is not None and y_val is not None and len(X_val) > 0:
            Xv = np.asarray(X_val, dtype=np.float32)
            if Xv.ndim == 1:
                Xv = Xv[None, :]
            yv = np.asarray(y_val).ravel().astype(float)
            for i, m in enumerate(models):
                preds = np.asarray([_to_float(_predict(m, x)) for x in Xv])
                acc = float(np.mean(np.round(preds) == yv)) if yv.size else 0.5
                delegates.append({"id": i, "model": m,
                                  "weight": max(1e-3, acc)})
        else:
            for i, m in enumerate(models):
                delegates.append({"id": i, "model": m,
                                  "weight": self.default_weight})
        total = sum(d["weight"] for d in delegates) or 1.0
        for d in delegates:
            d["weight"] = d["weight"] / total
        self._council = {"delegates": delegates, "n": len(delegates)}
        return self._council

    def debate(self, council: Dict[str, Any], input: Any) -> List[Dict[str, Any]]:
        """Each delegate emits a Position (prediction + confidence) on the input."""
        x = np.asarray(input, dtype=np.float32).ravel()
        positions: List[Dict[str, Any]] = []
        for d in council["delegates"]:
            pred = _predict(d["model"], x)
            pa = np.asarray(pred, dtype=np.float64).ravel()
            positions.append({"model": d["model"], "delegate_id": d["id"],
                              "weight": d["weight"],
                              "prediction": pa.tolist(),
                              "confidence": _confidence(pa)})
        return positions

    def resolve(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted majority vote -> Verdict (consensus prediction + dissenters)."""
        if not positions:
            return {"prediction": None, "consensus": 0.0,
                    "dissenters": [], "positions": []}
        max_len = max(len(p["prediction"]) for p in positions)
        mat = np.zeros((len(positions), max_len), dtype=np.float64)
        wts = np.zeros(len(positions), dtype=np.float64)
        for i, p in enumerate(positions):
            arr = np.asarray(p["prediction"], dtype=np.float64)
            mat[i, :arr.size] = arr
            wts[i] = p["weight"]
        consensus_pred = (mat * wts[:, None]).sum(axis=0)
        diffs = mat - consensus_pred[None, :]
        per_sample_var = (diffs ** 2 * wts[:, None]).sum(axis=0)
        consensus = float(1.0 / (1.0 + float(np.mean(per_sample_var)) + 1e-8))
        per_delegate = np.linalg.norm(diffs, axis=1)
        thresh = float(per_delegate.mean() + per_delegate.std()) if len(per_delegate) else 0.0
        dissenters = [int(positions[i]["delegate_id"])
                     for i in np.where(per_delegate > thresh)[0]]
        return {"prediction": consensus_pred.tolist(),
                "consensus": consensus,
                "dissenters": dissenters,
                "positions": positions}

    def veto(self, verdict: Dict[str, Any], model: Any) -> bool:
        """Did delegate ``model`` veto the verdict? (its prediction diverges too far)."""
        positions = verdict.get("positions", [])
        pos = next((p for p in positions if p.get("model") is model), None)
        if pos is None:
            return False
        p = np.asarray(pos["prediction"], dtype=np.float64).ravel()
        v = np.asarray(verdict.get("prediction", []), dtype=np.float64).ravel()
        n = min(p.size, v.size)
        if n == 0:
            return False
        diff = float(np.linalg.norm(p[:n] - v[:n]))
        norm = float(np.linalg.norm(v[:n])) + 1e-8
        return (diff / norm) > self.veto_tol


# ============================================================================
# Idea 130 -- NeuralGeology
# ============================================================================
class NeuralGeology:
    """Map the model's weight matrices to geological strata. Each layer
    is a rock formation whose "mineral composition" is the histogram of
    weight magnitudes. Valuable mineral veins are clusters of
    high-magnitude weights that dominate the layer's information flow.
    Deeper formations (higher depth) are older, sedimented from earlier
    training epochs.
    """

    def __init__(self, n_bins: int = 10, vein_quantile: float = 0.95):
        self.n_bins = n_bins
        self.vein_quantile = vein_quantile

    def survey(self, model: Any) -> Dict[str, Any]:
        """Build a GeologicalMap: one Stratum per 2-D weight matrix."""
        weights = _weights_of(model)
        strata: List[Dict[str, Any]] = []
        for depth, name in enumerate(sorted(weights)):
            W = np.asarray(weights[name], dtype=np.float64)
            if W.ndim != 2:
                continue
            flat = np.abs(W).ravel()
            hist, edges = np.histogram(flat, bins=self.n_bins)
            composition = (hist / max(1, hist.sum())).tolist()
            strata.append({
                "name": name, "depth": depth, "W": W,
                "shape": list(W.shape),
                "mineral_composition": composition,
                "bin_edges": edges.tolist(),
                "mean_mag": float(flat.mean()) if flat.size else 0.0,
                "max_mag": float(flat.max()) if flat.size else 0.0,
                "hardness": float(np.std(flat)) if flat.size else 0.0,
            })
        return {"strata": strata, "n_strata": len(strata),
                "max_depth": max(0, len(strata) - 1)}

    def core_sample(self, geological_map: Dict[str, Any], layer_idx: int,
                    depth: int) -> Dict[str, Any]:
        """Pull a vertical core sample (a column) of the layer at the given depth."""
        strata = geological_map["strata"]
        if layer_idx < 0 or layer_idx >= len(strata):
            return {"column": [], "n_zero": 0, "mean": 0.0}
        W = strata[layer_idx]["W"]
        if W.ndim != 2 or depth < 0 or depth >= W.shape[1]:
            return {"column": [], "n_zero": 0, "mean": 0.0}
        col = W[:, depth]
        return {"layer": strata[layer_idx]["name"], "depth_idx": depth,
                "column": col.tolist(),
                "n_zero": int((np.abs(col) < 1e-6).sum()),
                "mean": float(col.mean()),
                "max": float(np.abs(col).max())}

    def prospect(self, geological_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find veins: high-magnitude weight entries within each stratum."""
        veins: List[Dict[str, Any]] = []
        for s in geological_map["strata"]:
            W = s["W"]
            if W.ndim != 2 or W.size == 0:
                continue
            thresh = float(np.quantile(np.abs(W), self.vein_quantile))
            rows, cols = np.where(np.abs(W) >= thresh)
            for r, c in zip(rows.tolist(), cols.tolist()):
                veins.append({"layer": s["name"], "depth": s["depth"],
                              "row": r, "col": c,
                              "magnitude": float(W[r, c])})
        veins.sort(key=lambda v: -v["magnitude"])
        return veins

    def date_formation(self, vein: Dict[str, Any]) -> str:
        """Estimate the geologic age of a vein from its depth (deeper = older)."""
        depth = int(vein.get("depth", 0))
        eras = ["Cenozoic (recent)",
                "Mesozoic (mid-training)",
                "Paleozoic (early-training)",
                "Precambrian (initialization)"]
        era = eras[min(depth, len(eras) - 1)]
        return f"{era} -- depth {depth}"


# ============================================================================
# Idea 131 -- PredictionCourt
# ============================================================================
class PredictionCourt:
    """Each prediction goes on trial. The prosecution argues against it by
    finding perturbations of the input that flip the prediction
    (adversarial counterexamples). The defense argues for it by finding
    training samples whose label matches the prediction. The judge
    weighs both arguments and renders a verdict: guilty (the prediction
    is unreliable) or not guilty.
    """

    def __init__(self, n_perturbations: int = 16, epsilon: float = 0.1,
                 seed: int = 0):
        self.n_perturbations = n_perturbations
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self._train_X: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None

    def memorize(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Cache the training set so the defense can call witnesses."""
        self._train_X = np.asarray(X_train, dtype=np.float32)
        self._train_y = np.asarray(y_train).ravel()

    def arraign(self, model: Any, input: Any,
                prediction: Any = None) -> Dict[str, Any]:
        """Open a case file for a single prediction."""
        x = np.asarray(input, dtype=np.float32).ravel()
        pred = (np.asarray(prediction, dtype=np.float32).ravel()
                if prediction is not None
                else np.asarray(_predict(model, x), dtype=np.float32).ravel())
        return {"model": model, "input": x, "prediction": pred,
                "prosecution": None, "defense": None, "verdict": None}

    def prosecute(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adversarial perturbations of the input and find any that flip the prediction."""
        model = case["model"]
        x = case["input"]
        pred = case["prediction"]
        counterexamples: List[Dict[str, Any]] = []
        for _ in range(self.n_perturbations):
            noise = self.rng.normal(0, self.epsilon,
                                    size=x.shape).astype(np.float32)
            pred_pert = _predict(model, x + noise)
            pp = np.asarray(pred_pert, dtype=np.float32).ravel()
            n = min(pp.size, pred.size)
            if n == 0:
                continue
            delta = float(np.linalg.norm(pp[:n] - pred[:n]))
            if delta > self.epsilon:
                counterexamples.append({"perturbation": noise.tolist(),
                                        "perturbed_pred": pp.tolist(),
                                        "delta": delta})
        return {"counterexamples": counterexamples,
                "n_found": len(counterexamples),
                "strength": len(counterexamples) / max(1, self.n_perturbations)}

    def defend(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Find training samples whose label agrees with the prediction."""
        if self._train_X is None or self._train_y is None:
            return {"witnesses": [], "n_supporting": 0, "defense_strength": 0.0}
        pred = case["prediction"]
        pred_label = int(np.argmax(np.abs(pred))) if pred.size > 1 \
            else int(np.round(pred[0]) if pred.size else 0)
        witnesses: List[Dict[str, Any]] = []
        for i, yi in enumerate(self._train_y):
            if int(yi) == pred_label:
                witnesses.append({"sample_id": int(i), "label": int(yi)})
        strength = float(len(witnesses) / max(1, len(self._train_y)))
        return {"witnesses": witnesses, "n_supporting": len(witnesses),
                "defense_strength": strength}

    def verdict(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Render a verdict: guilty (unreliable) iff prosecution > defense."""
        prosecution = case.get("prosecution") or self.prosecute(case)
        defense = case.get("defense") or self.defend(case)
        case["prosecution"] = prosecution
        case["defense"] = defense
        pros = prosecution["strength"]
        def_strength = defense["defense_strength"]
        guilty = pros > def_strength
        case["verdict"] = {
            "guilty": guilty,
            "verdict": "guilty (unreliable)" if guilty else "not guilty (reliable)",
            "prosecution_strength": pros,
            "defense_strength": def_strength,
        }
        return case["verdict"]


# ============================================================================
# Idea 132 -- WeightTides
# ============================================================================
class WeightTides:
    """Monitor a model's weights like ocean tides. High tide = weights
    growing during training (positive shift since last snapshot). Low
    tide = weights shrinking. Spring tide = a layer with large shifts
    (high tidal range). Neap tide = a layer with small shifts. The
    tidal_range is the difference between the largest and smallest
    per-weight change across the model.
    """

    def __init__(self, min_shift: float = 1e-4, top_k: int = 50):
        self.min_shift = min_shift
        self.top_k = top_k
        self._snapshots: List[Dict[str, np.ndarray]] = []

    def observe(self, model: Any, epoch: int) -> Dict[str, Any]:
        """Take a snapshot of every weight; compute per-weight shift vs. last snapshot."""
        weights = _weights_of(model)
        flat: Dict[str, np.ndarray] = {
            name: np.array(np.asarray(w).ravel(), dtype=np.float64, copy=True)
            for name, w in weights.items()
        }
        if self._snapshots:
            prev = self._snapshots[-1]
            shifts: Dict[str, List[float]] = {}
            for name, arr in flat.items():
                if name in prev and prev[name].shape == arr.shape:
                    shifts[name] = (arr - prev[name]).tolist()
                else:
                    shifts[name] = [0.0] * arr.size
        else:
            shifts = {name: [0.0] * arr.size for name, arr in flat.items()}
        self._snapshots.append(flat)
        return {"epoch": int(epoch), "n_snapshots": len(self._snapshots),
                "shifts": shifts}

    def high_tide(self, tide_table: Dict[str, Any], layer: str) -> List[Dict[str, Any]]:
        """Return the weights in `layer` whose magnitude grew most since the previous snapshot."""
        shifts = tide_table.get("shifts", {}).get(layer, [])
        if not shifts:
            return []
        arr = np.asarray(shifts, dtype=np.float64)
        idx = np.where(arr > self.min_shift)[0]
        idx = sorted(idx, key=lambda i: -arr[i])
        return [{"index": int(i), "shift": float(arr[i])}
                for i in idx[: self.top_k]]

    def low_tide(self, tide_table: Dict[str, Any], layer: str) -> List[Dict[str, Any]]:
        """Return the weights in `layer` whose magnitude shrank most (negative shift)."""
        shifts = tide_table.get("shifts", {}).get(layer, [])
        if not shifts:
            return []
        arr = np.asarray(shifts, dtype=np.float64)
        idx = np.where(arr < -self.min_shift)[0]
        idx = sorted(idx, key=lambda i: arr[i])  # most negative first
        return [{"index": int(i), "shift": float(arr[i])}
                for i in idx[: self.top_k]]

    def tidal_range(self, tide_table: Dict[str, Any]) -> float:
        """Difference between the largest and smallest per-weight shift across all layers."""
        all_shifts: List[float] = []
        for shifts in tide_table.get("shifts", {}).values():
            all_shifts.extend(shifts)
        if not all_shifts:
            return 0.0
        arr = np.asarray(all_shifts, dtype=np.float64)
        return float(arr.max() - arr.min())


# ============================================================================
# Idea 133 -- ModelEconomy
# ============================================================================
class ModelEconomy:
    """Models participate in a market. They 'buy' training samples with
    compute (cost = n_params * input_dim) and 'sell' predictions to
    users (price = confidence-scaled compute cost). GDP is the total
    prediction value transacted. Inflation is the rate of weight
    magnitude growth (an overfitting proxy). Unemployment is the
    fraction of dead neurons (zero-norm columns).
    """

    def __init__(self, base_price: float = 1.0, dead_tol: float = 1e-4):
        self.base_price = base_price
        self.dead_tol = dead_tol
        self._counter = 0

    def open_market(self, model: Any) -> Dict[str, Any]:
        """Open a market for a model; seed its GDP + inflation history."""
        weights = _weights_of(model)
        n_params = sum(int(np.asarray(w).size) for w in weights.values())
        total_norm = float(sum(
            float(np.linalg.norm(np.asarray(w).ravel()))
            for w in weights.values()))
        self._counter += 1
        return {"id": self._counter, "model": model,
                "n_params": n_params, "n_layers": len(weights),
                "gdp": 0.0, "n_transactions": 0,
                "inflation_history": [],
                "prev_total_norm": total_norm,
                "weights": weights}

    def quote(self, market: Dict[str, Any], input: Any) -> Dict[str, Any]:
        """Price a prediction: compute_cost * (0.5 + confidence)."""
        x = np.asarray(input, dtype=np.float32).ravel()
        compute_cost = (self.base_price
                        * (1.0 + market["n_params"] / 1e4)
                        * max(1, x.size))
        pred = _predict(market["model"], x)
        pa = np.asarray(pred, dtype=np.float64).ravel()
        conf = _confidence(pa)
        price = compute_cost * (0.5 + conf)
        # update inflation (rate of weight-norm growth)
        weights = _weights_of(market["model"])
        total_norm = float(sum(
            float(np.linalg.norm(np.asarray(w).ravel()))
            for w in weights.values()))
        prev = market.get("prev_total_norm", total_norm)
        infl = float((total_norm - prev) / (abs(prev) + 1e-8))
        market["inflation_history"].append(infl)
        market["prev_total_norm"] = total_norm
        # settle transaction
        market["gdp"] += price
        market["n_transactions"] += 1
        return {"price": price, "confidence": conf,
                "prediction": pa.tolist(), "inflation": infl}

    def trade(self, market: Dict[str, Any], model_a: Any, model_b: Any,
              input: Any = None) -> Dict[str, Any]:
        """model_a sells a prediction to model_b (trade = knowledge transfer)."""
        if input is None:
            pa = pb = None
            agreement = 0.5
        else:
            x = np.asarray(input, dtype=np.float32).ravel()
            ra = _predict(model_a, x)
            rb = _predict(model_b, x)
            pa = np.asarray(ra, dtype=np.float64).ravel()
            pb = np.asarray(rb, dtype=np.float64).ravel()
            n = min(pa.size, pb.size)
            agreement = float(1.0 / (1.0 + np.linalg.norm(pa[:n] - pb[:n]))) if n else 0.5
        price = self.base_price * (1.0 + agreement * 10.0)
        market["gdp"] += price
        market["n_transactions"] += 1
        return {"price": price, "agreement": agreement,
                "pred_a": pa.tolist() if pa is not None else None,
                "pred_b": pb.tolist() if pb is not None else None}

    def gdp(self, market: Dict[str, Any]) -> float:
        """Total value of all predictions transacted in the market."""
        return float(market.get("gdp", 0.0))

    def inflation_rate(self, market: Dict[str, Any]) -> float:
        """Smoothed recent inflation (rate of weight-norm growth)."""
        history = market.get("inflation_history", [])
        if not history:
            return 0.0
        recent = history[-5:]
        return float(np.mean(recent))

    def unemployment_rate(self, market: Dict[str, Any]) -> float:
        """Fraction of dead neurons (columns with near-zero norm)."""
        weights = market.get("weights", {})
        total = 0
        dead = 0
        for w in weights.values():
            Wm = np.asarray(w, dtype=np.float64)
            if Wm.ndim != 2:
                continue
            col_norms = np.linalg.norm(Wm, axis=0)
            total += col_norms.size
            dead += int((col_norms < self.dead_tol).sum())
        return float(dead / max(1, total))


# ============================================================================
# Idea 134 -- HallucinationCartography
# ============================================================================
class HallucinationCartography:
    """Map the model's hallucination space: where in input space does it
    make confident wrong predictions? Peaks are high-confidence wrong
    answers (the hallucinations); valleys are low-confidence correct
    answers (uncertain but right). Contour lines trace inputs of equal
    confidence, like elevation lines on a topographic map.
    """

    def __init__(self, peak_thresh: float = 0.7,
                 valley_thresh: float = 0.3,
                 contour_tol: float = 0.05):
        self.peak_thresh = peak_thresh
        self.valley_thresh = valley_thresh
        self.contour_tol = contour_tol

    def survey(self, model: Any, sample_inputs: np.ndarray,
               labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Build a topographic map: for each input, record prediction, confidence, correctness."""
        X = np.asarray(sample_inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        points: List[Dict[str, Any]] = []
        for i, x in enumerate(X):
            pred = _predict(model, x)
            pa = np.asarray(pred, dtype=np.float64).ravel()
            conf = _confidence(pa)
            pred_label = int(np.argmax(np.abs(pa))) if pa.size > 1 \
                else int(np.round(pa[0]) if pa.size else 0)
            true_label = int(labels[i]) if labels is not None else None
            correct = (true_label == pred_label) if true_label is not None else None
            points.append({"input_id": int(i), "x": x.tolist(),
                           "prediction": pa.tolist(),
                           "predicted_label": pred_label,
                           "confidence": conf,
                           "true_label": true_label, "correct": correct})
        return {"points": points, "n_points": len(points),
                "has_labels": labels is not None}

    def peaks(self, topo_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Peaks: high-confidence wrong answers (the hallucinations)."""
        if not topo_map.get("has_labels"):
            return []
        return [p for p in topo_map["points"]
                if p["correct"] is False
                and p["confidence"] >= self.peak_thresh]

    def valleys(self, topo_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Valleys: low-confidence correct answers (uncertain but right)."""
        if not topo_map.get("has_labels"):
            return []
        return [p for p in topo_map["points"]
                if p["correct"] is True
                and p["confidence"] <= self.valley_thresh]

    def contour_line(self, topo_map: Dict[str, Any],
                     confidence: float) -> List[Dict[str, Any]]:
        """Trace a contour line: all inputs whose confidence is within tol of `confidence`."""
        return [p for p in topo_map["points"]
                if abs(p["confidence"] - confidence) <= self.contour_tol]


# ============================================================================
# Idea 135 -- ModelChoir
# ============================================================================
class ModelChoir:
    """Models sing together in harmony. Each model is a voice part:
    soprano (highest density -- dense small weights), alto (the
    catch-all mid voice), tenor (fewest parameters -- fastest), bass
    (largest weight norm -- most robust). The conductor harmonizes
    them by weighted-averaging their predictions; dissonance is the
    variance of their predictions.
    """

    VOICE_WEIGHTS: Dict[str, float] = {
        "bass": 1.4, "soprano": 1.2, "alto": 1.0, "tenor": 0.8,
    }

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def assemble(self, models: List[Any]) -> Dict[str, Any]:
        """Assign each model a voice part based on its weight statistics."""
        stats: List[Dict[str, Any]] = []
        for m in models:
            weights = _weights_of(m)
            n_params = sum(int(np.asarray(w).size) for w in weights.values())
            if weights:
                all_w = np.concatenate([
                    np.asarray(w, dtype=np.float64).ravel()
                    for w in weights.values()])
            else:
                all_w = np.zeros(1, dtype=np.float64)
            total_norm = float(np.linalg.norm(all_w))
            density = float((np.abs(all_w) > 1e-4).mean())
            stats.append({"model": m, "n_params": n_params,
                          "total_norm": total_norm, "density": density})
        if not stats:
            return {"voices": [], "n": 0}
        # rank models along each axis
        by_params = sorted(range(len(stats)), key=lambda i: stats[i]["n_params"])
        by_norm = sorted(range(len(stats)), key=lambda i: stats[i]["total_norm"])
        by_density = sorted(range(len(stats)), key=lambda i: stats[i]["density"])
        assign: Dict[int, str] = {}
        assign[by_params[0]] = "tenor"          # fewest params = fastest
        if by_norm[-1] not in assign:
            assign[by_norm[-1]] = "bass"        # largest norm = robust
        if by_density[-1] not in assign:
            assign[by_density[-1]] = "soprano"  # densest = high precision
        voices = []
        for i, s in enumerate(stats):
            voices.append({
                "model": s["model"], "voice": assign.get(i, "alto"),
                "id": i, "n_params": s["n_params"],
                "density": s["density"], "total_norm": s["total_norm"],
            })
        return {"voices": voices, "n": len(voices)}

    def harmonize(self, choir: Dict[str, Any], input: Any) -> Dict[str, Any]:
        """Combine the choir's predictions into a single Chord (weighted by voice)."""
        x = np.asarray(input, dtype=np.float32).ravel()
        preds: List[np.ndarray] = []
        for v in choir["voices"]:
            pa = np.asarray(_predict(v["model"], x), dtype=np.float64).ravel()
            preds.append(pa)
        if not preds:
            return {"chord": [], "harmony": 0.0, "voice_weights": []}
        max_len = max(p.size for p in preds)
        mat = np.zeros((len(preds), max_len), dtype=np.float64)
        for i, p in enumerate(preds):
            mat[i, :p.size] = p
        w = np.asarray([self.VOICE_WEIGHTS.get(v["voice"], 1.0)
                        for v in choir["voices"]], dtype=np.float64)
        w = w / (w.sum() + 1e-8)
        chord = (mat * w[:, None]).sum(axis=0)
        diffs = mat - chord[None, :]
        harmony = float(1.0 / (1.0 + float(np.mean(np.abs(diffs))) + 1e-8))
        return {"chord": chord.tolist(), "harmony": harmony,
                "voice_weights": w.tolist()}

    def soloist(self, choir: Dict[str, Any], input: Any) -> Any:
        """Which model is most confident on this input? (the soloist)."""
        x = np.asarray(input, dtype=np.float32).ravel()
        best_model = None
        best_conf = -1.0
        for v in choir["voices"]:
            pa = np.asarray(_predict(v["model"], x), dtype=np.float64).ravel()
            conf = _confidence(pa)
            if conf > best_conf:
                best_conf = conf
                best_model = v["model"]
        return best_model

    def dissonance(self, choir: Dict[str, Any], input: Any) -> float:
        """Dissonance = std of voice predictions (0 = perfect harmony)."""
        x = np.asarray(input, dtype=np.float32).ravel()
        preds: List[np.ndarray] = []
        for v in choir["voices"]:
            preds.append(np.asarray(_predict(v["model"], x),
                                     dtype=np.float64).ravel())
        if len(preds) < 2:
            return 0.0
        max_len = max(p.size for p in preds)
        mat = np.zeros((len(preds), max_len), dtype=np.float64)
        for i, p in enumerate(preds):
            mat[i, :p.size] = p
        return float(np.mean(np.std(mat, axis=0)))


# ============================================================================
# Idea 136 -- NeuralForensics
# ============================================================================
class NeuralForensics:
    """Investigate a model like a crime scene. Dust for fingerprints:
    detect whether an input was likely in the training set (the model
    is suspiciously confident on it relative to a reference set).
    Take witness statements: each input feature's gradient-based
    attribution to the output. Build a timeline of weight changes
    across recorded snapshots (investigate() takes a snapshot; timeline()
    returns all snapshots as events).
    """

    def __init__(self, confidence_threshold: float = 0.95,
                 eps: float = 1e-3):
        self.confidence_threshold = confidence_threshold
        self.eps = eps
        self._snapshots: List[Dict[str, np.ndarray]] = []

    def investigate(self, model: Any) -> Dict[str, Any]:
        """Open a case file: snapshot current weights + model metadata."""
        weights = _weights_of(model)
        snapshot = {name: np.array(np.asarray(w), copy=True)
                    for name, w in weights.items()}
        self._snapshots.append(snapshot)
        return {"model": model, "n_layers": len(weights),
                "n_params": sum(int(np.asarray(w).size) for w in weights.values()),
                "snapshot": snapshot,
                "snapshot_idx": len(self._snapshots) - 1,
                "opened_at": time.time()}

    def dust_for_prints(self, model: Any, input: Any,
                        reference_inputs: Optional[np.ndarray] = None) -> bool:
        """Was this input likely in the training set?

        Without a reference set: yes iff model confidence exceeds
        ``confidence_threshold``. With a reference set: yes iff the
        input's confidence is in the top quartile of reference confidences.
        """
        x = np.asarray(input, dtype=np.float32).ravel()
        pa = np.asarray(_predict(model, x), dtype=np.float64).ravel()
        conf = _confidence(pa)
        if reference_inputs is None:
            return conf >= self.confidence_threshold
        X = np.asarray(reference_inputs, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        ref_confs = [_confidence(np.asarray(_predict(model, r), dtype=np.float64).ravel())
                     for r in X]
        if not ref_confs:
            return conf >= self.confidence_threshold
        thresh = float(np.quantile(ref_confs, 0.75))
        return conf >= thresh

    def witness_statements(self, model: Any, input: Any) -> List[Dict[str, Any]]:
        """Gradient-based attribution of each input feature to the output."""
        x = np.asarray(input, dtype=np.float32).ravel()
        base = np.asarray(_predict(model, x), dtype=np.float64).ravel()
        attributions: List[Dict[str, Any]] = []
        for i in range(x.size):
            xp = x.copy()
            xp[i] += self.eps
            pp = np.asarray(_predict(model, xp), dtype=np.float64).ravel()
            n = min(base.size, pp.size)
            grad = float(np.sum(pp[:n] - base[:n]) / self.eps) if n else 0.0
            attributions.append({"feature": int(i), "gradient": grad,
                                 "importance": abs(grad)})
        attributions.sort(key=lambda a: -a["importance"])
        return attributions

    def timeline(self, model: Any) -> List[Dict[str, Any]]:
        """Build a timeline of weight-change events from recorded snapshots + current state."""
        current = {name: np.array(np.asarray(w), copy=True)
                    for name, w in _weights_of(model).items()}
        snaps = list(self._snapshots) + [current]
        events: List[Dict[str, Any]] = []
        for i, snap in enumerate(snaps):
            layer_norms = {name: float(np.linalg.norm(np.asarray(w).ravel()))
                           for name, w in snap.items()}
            if i == 0:
                shifts: Dict[str, float] = {name: 0.0 for name in layer_norms}
            else:
                prev = snaps[i - 1]
                shifts = {name: (layer_norms[name]
                                 - float(np.linalg.norm(np.asarray(prev[name]).ravel())))
                          if name in prev else 0.0
                          for name in layer_norms}
            events.append({"step": i, "layer_norms": layer_norms,
                           "shifts": shifts,
                           "max_shift": float(max(shifts.values())) if shifts else 0.0})
        return events


# ============================================================================
# Idea 137 -- ModelCultivation
# ============================================================================
class ModelCultivation:
    """Grow a model like a crop. Plant seeds (initialize random weights
    matching an architecture), water them with gradient steps
    (vanilla gradient descent on MSE), prune weak weights (set small
    ones to zero), and harvest the final accuracy. Growth stages:
    seed -> seedling -> sapling -> mature -> ripe.
    """

    STAGES = ["seed", "seedling", "sapling", "mature", "ripe"]

    def __init__(self, lr: float = 0.05, seed: int = 0):
        self.lr = lr
        self.rng = np.random.default_rng(seed)

    def plant(self, architecture: List[int]) -> Dict[str, Any]:
        """Initialize weights for a fully-connected architecture (list of layer sizes)."""
        weights: Dict[str, np.ndarray] = {}
        for i in range(len(architecture) - 1):
            fan_in = max(1, architecture[i])
            fan_out = architecture[i + 1]
            W = (self.rng.normal(0, np.sqrt(2.0 / fan_in),
                                  size=(fan_in, fan_out))).astype(np.float32)
            weights[f"L{i}"] = W
        return {"architecture": list(architecture),
                "weights": weights,
                "epochs_trained": 0,
                "training_loss_history": [],
                "stage": "seed"}

    def water(self, plant: Dict[str, Any], X: np.ndarray,
              y: np.ndarray, epochs: int = 10) -> Dict[str, Any]:
        """Train the plant: forward + backward + update (tanh activations, MSE loss)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        weights = plant["weights"]
        names = sorted(weights)
        for _ in range(epochs):
            # forward
            cur = X
            acts = [cur]
            for name in names:
                W = weights[name]
                if cur.shape[-1] == W.shape[0]:
                    cur = np.tanh(cur @ W)
                elif cur.shape[-1] == W.shape[1]:
                    cur = np.tanh(cur @ W.T)
                else:
                    continue
                acts.append(cur)
            pred = acts[-1]
            # pad/truncate y to match prediction width so the backward
            # pass produces a gradient matching W's output dimension
            pred_dim = pred.shape[1]
            y_dim = y.shape[1]
            if y_dim < pred_dim:
                y_padded = np.zeros((y.shape[0], pred_dim), dtype=np.float32)
                y_padded[:, :y_dim] = y
                y = y_padded
            elif y_dim > pred_dim:
                y = y[:, :pred_dim]
            loss = float(np.mean((pred - y) ** 2))
            plant["training_loss_history"].append(loss)
            delta = (pred - y) * (1 - pred ** 2)
            # backward
            for i in range(len(names) - 1, -1, -1):
                name = names[i]
                W = weights[name]
                a_in = acts[i]
                if a_in.shape[-1] == W.shape[0]:
                    grad_W = a_in.T @ delta
                    if i > 0:
                        delta = (delta @ W.T) * (1 - a_in ** 2)
                elif a_in.shape[-1] == W.shape[1]:
                    grad_W = delta.T @ a_in
                    if i > 0:
                        delta = (delta @ W) * (1 - a_in ** 2)
                else:
                    continue
                weights[name] = W - self.lr * grad_W
            plant["epochs_trained"] += 1
        plant["stage"] = self._stage_for(plant["epochs_trained"])
        return plant

    def prune(self, plant: Dict[str, Any], threshold: float = 0.1) -> Dict[str, Any]:
        """Zero out weights below threshold (regularization)."""
        weights = plant["weights"]
        pruned = 0
        for name, W in weights.items():
            mask = np.abs(W) < threshold
            pruned += int(mask.sum())
            W[mask] = 0.0
        plant.setdefault("pruning_history", []).append(
            {"threshold": threshold, "n_pruned": pruned})
        return plant

    def harvest(self, plant: Dict[str, Any], X_test: np.ndarray,
                y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the plant on test data."""
        X = np.asarray(X_test, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        y = np.asarray(y_test, dtype=np.float32).ravel()
        cur = X
        for name in sorted(plant["weights"]):
            W = plant["weights"][name]
            if cur.shape[-1] == W.shape[0]:
                cur = np.tanh(cur @ W)
            elif cur.shape[-1] == W.shape[1]:
                cur = np.tanh(cur @ W.T)
        pred = cur
        if pred.shape[1] == 1:
            preds = np.round(pred[:, 0]).astype(int)
        else:
            preds = np.argmax(pred, axis=1)
        acc = float(np.mean(preds == y.astype(int))) if y.size else 0.0
        return {"accuracy": acc, "predictions": preds.tolist(),
                "n_test": int(X.shape[0]),
                "epochs_trained": plant["epochs_trained"]}

    def growth_stage(self, plant: Dict[str, Any]) -> str:
        """Return the current growth stage of the plant."""
        return self._stage_for(plant.get("epochs_trained", 0))

    def _stage_for(self, epochs: int) -> str:
        if epochs <= 0:
            return "seed"
        if epochs < 5:
            return "seedling"
        if epochs < 20:
            return "sapling"
        if epochs < 50:
            return "mature"
        return "ripe"


# ============================================================================
# Idea 138 -- PredictionArcheology
# ============================================================================
class PredictionArcheology:
    """Dig down through a single prediction's reasoning layers like an
    archaeologist. Surface level = the model's output. Mid levels =
    intermediate-layer activation patterns. Bedrock = the input
    features that most influenced the output. Each stratum has
    artifacts (notable activations or weights).
    """

    def __init__(self, top_k: int = 5, artifact_quantile: float = 0.9):
        self.top_k = top_k
        self.artifact_quantile = artifact_quantile

    def excavate(self, model: Any, input: Any) -> Dict[str, Any]:
        """Forward-pass through the model recording each layer's pre/activations."""
        weights = _weights_of(model)
        x = np.asarray(input, dtype=np.float32).ravel()
        cur = x
        strata: List[Dict[str, Any]] = []
        for depth, name in enumerate(sorted(weights)):
            W = np.asarray(weights[name], dtype=np.float32)
            if W.ndim != 2:
                continue
            if cur.size == W.shape[0]:
                pre = cur @ W
            elif cur.size == W.shape[1]:
                pre = cur @ W.T
            else:
                continue
            act = np.tanh(pre)
            strata.append({"depth": depth, "name": name,
                           "input": cur.copy(), "W": W,
                           "pre_act": pre, "act": act})
            cur = act
        return {"strata": strata, "input": x, "prediction": cur,
                "model": model, "n_strata": len(strata)}

    def surface_level(self, excavation: Dict[str, Any]) -> Dict[str, Any]:
        """Surface stratum = the final prediction."""
        return {"prediction": np.asarray(excavation["prediction"],
                                         dtype=np.float32).tolist(),
                "depth": excavation["n_strata"]}

    def mid_level(self, excavation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mid strata = intermediate-layer activations (everything except the output layer)."""
        strata = excavation["strata"]
        if not strata:
            return []
        mid = strata[:-1]
        return [{"depth": s["depth"], "layer": s["name"],
                 "activations": s["act"].tolist()} for s in mid]

    def bedrock(self, excavation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Bedrock = the most influential input features (saliency = |x_i * sum_j |W[i,j]|)."""
        x = excavation["input"]
        strata = excavation["strata"]
        if not strata:
            return [{"feature": int(i), "value": float(v), "importance": 0.0}
                    for i, v in enumerate(x.tolist())]
        first = strata[0]
        W = first["W"]
        if W.shape[0] == x.size:
            saliency = np.abs(x) * np.abs(W).sum(axis=1)
        elif W.shape[1] == x.size:
            saliency = np.abs(x) * np.abs(W).sum(axis=0)
        else:
            saliency = np.abs(x)
        top_idx = np.argsort(-saliency)[: self.top_k]
        return [{"feature": int(i), "value": float(x[i]),
                 "importance": float(saliency[i])} for i in top_idx]

    def artifacts(self, stratum: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract notable artifacts from a stratum (high-magnitude weights + activations)."""
        arts: List[Dict[str, Any]] = []
        W = stratum.get("W")
        act = stratum.get("act")
        if W is not None and np.asarray(W).ndim == 2 and W.size > 0:
            thresh = float(np.quantile(np.abs(W), self.artifact_quantile))
            rows, cols = np.where(np.abs(W) >= thresh)
            for r, c in zip(rows.tolist()[: self.top_k],
                            cols.tolist()[: self.top_k]):
                arts.append({"type": "weight", "row": r, "col": c,
                             "value": float(W[r, c])})
        if act is not None and np.asarray(act).size > 0:
            a = np.asarray(act)
            thresh_a = float(np.quantile(np.abs(a), self.artifact_quantile))
            for i in np.argsort(-np.abs(a))[: self.top_k]:
                if abs(a[int(i)]) >= thresh_a:
                    arts.append({"type": "activation", "neuron": int(i),
                                 "value": float(a[int(i)])})
        return arts


__all__ = [
    "ModelDiplomacy",
    "NeuralGeology",
    "PredictionCourt",
    "WeightTides",
    "ModelEconomy",
    "HallucinationCartography",
    "ModelChoir",
    "NeuralForensics",
    "ModelCultivation",
    "PredictionArcheology",
]
