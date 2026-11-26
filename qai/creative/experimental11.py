"""
qai.creative.experimental11 -- 10 more original creative ideas (part 11).

  Idea 109 -- ModelAcupuncture     : diagnose + treat a model's "energy blockages"
  Idea 110 -- NeuroCryptography    : the model's weights ARE the cipher key
  Idea 111 -- ModelBonsai          : prune / wire / root-prune a model like a bonsai
  Idea 112 -- HallucinationVaccine : inoculate a model against hallucination
  Idea 113 -- QuantumEntanglement   : find entangled neuron pairs (Bell test)
  Idea 114 -- ModelConcert         : an orchestra of models with a conductor
  Idea 115 -- NeuralWeather        : forecast the weather inside a training run
  Idea 116 -- WeightGenealogy      : trace the ancestry of every weight
  Idea 117 -- ModelShrine          : venerate a retired model as relics
  Idea 118 -- CognitiveEcho        : a model replays its own past predictions

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
    if isinstance(w_attr, (list, tuple)) and not isinstance(w_attr, tuple):
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


# ============================================================================
# Idea 109 -- ModelAcupuncture
# ============================================================================
class ModelAcupuncture:
    """Diagnose a model's "energy blockages" by sampling activations across
    layers and identifying dead / saturated / silent neurons.

    A *meridian map* shows where information flow is blocked. Treatment
    applies targeted re-initialisation (the "needle") to free the flow.

    Blockage types:
      - dead      : neuron never fires (weight column ~ 0 across samples)
      - saturated : neuron always fires at the extreme (huge |w|)
      - silent    : neuron fires with near-zero variance (no information)
    """

    def __init__(self, dead_tol: float = 1e-4, sat_z: float = 4.0,
                 silent_var: float = 1e-3, n_probe: int = 32):
        self.dead_tol = dead_tol
        self.sat_z = sat_z
        self.silent_var = silent_var
        self.n_probe = n_probe
        self.last_diagnosis: Dict[str, Any] = {}

    def meridian_map(self, model: Any,
                     sample_inputs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Build the energy-flow map: layer -> {dead, saturated, silent}."""
        weights = _weights_of(model)
        acts: Dict[str, np.ndarray] = {}
        if sample_inputs is not None:
            Xin = np.asarray(sample_inputs, dtype=np.float32)
            if Xin.ndim == 1:
                Xin = Xin[None, :]
            cur = Xin
            for name, W in weights.items():
                Wm = np.asarray(W, dtype=np.float32)
                if Wm.ndim != 2:
                    continue
                if cur.shape[-1] == Wm.shape[0]:
                    cur = np.tanh(cur @ Wm)
                    acts[name] = cur
                elif cur.shape[-1] == Wm.shape[1]:
                    cur = np.tanh(cur @ Wm.T)
                    acts[name] = cur
        meridian: Dict[str, Dict[str, List[int]]] = {}
        for name, W in weights.items():
            Wm = np.asarray(W, dtype=np.float32)
            if Wm.ndim != 2 or Wm.shape[1] < 1:
                continue
            col_norm = np.linalg.norm(Wm, axis=0)
            col_var = Wm.var(axis=0)
            std = Wm.std(axis=0) + 1e-8
            z_max = np.abs((Wm - Wm.mean(axis=0)) / std).max(axis=0)
            dead = list(np.where(col_norm < self.dead_tol)[0])
            saturated = list(np.where(z_max > self.sat_z)[0])
            silent = list(np.where(col_var < self.silent_var)[0])
            if name in acts:
                act = acts[name]
                never = np.where((np.abs(act) < self.dead_tol).all(axis=0))[0]
                dead = sorted(set(dead) | set(int(i) for i in never))
            meridian[name] = {"dead": [int(i) for i in dead],
                              "saturated": [int(i) for i in saturated],
                              "silent": [int(i) for i in silent]}
        self.last_diagnosis = meridian
        return meridian

    def diagnose(self, model: Any,
                 sample_inputs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Diagnose energy blockages; return a structured report."""
        mm = self.meridian_map(model, sample_inputs)
        total_dead = sum(len(v["dead"]) for v in mm.values())
        total_sat = sum(len(v["saturated"]) for v in mm.values())
        total_sil = sum(len(v["silent"]) for v in mm.values())
        blockages: List[Dict[str, Any]] = []
        for layer, v in mm.items():
            for btype in ("dead", "saturated", "silent"):
                if v[btype]:
                    blockages.append({"layer": layer, "type": btype,
                                      "neurons": v[btype]})
        return {
            "meridian_map": mm,
            "n_blockages": len(blockages),
            "total_dead": total_dead,
            "total_saturated": total_sat,
            "total_silent": total_sil,
            "blockages": blockages,
            "qi_flow": 1.0 - min(1.0, len(blockages) / 50.0),
        }

    def treat(self, model: Any, blockage: Dict[str, Any]) -> Any:
        """Apply a targeted re-init needle to free a single blockage."""
        weights = _weights_of(model)
        layer = blockage["layer"]
        neurons = blockage.get("neurons", [])
        btype = blockage.get("type", "dead")
        if layer not in weights or not neurons:
            return model
        W = np.array(weights[layer], dtype=np.float32, copy=True)
        if W.ndim != 2:
            return model
        rng = np.random.default_rng(blockage.get("seed", 0))
        for n in neurons:
            if 0 <= n < W.shape[1]:
                if btype == "dead":
                    W[:, n] = rng.normal(0, 0.05, size=W.shape[0]).astype(np.float32)
                elif btype == "saturated":
                    W[:, n] *= 0.1
                else:  # silent
                    W[:, n] = W[:, n] + rng.normal(
                        0, 0.1, size=W.shape[0]).astype(np.float32)
        weights[layer] = W
        return _apply(model, weights)


# ============================================================================
# Idea 110 -- NeuroCryptography
# ============================================================================
class NeuroCryptography:
    """A model whose weights ARE the cipher key.

    The keystream is derived by hashing the model's weight bytes; the
    message is XORed with the keystream. Different weights -> unforgeably
    different ciphertexts. ``key_strength`` estimates the effective
    key size in bits from the entropy of the weights.
    """

    def __init__(self, salt: bytes = b"qai-neuro-crypto",
                 max_msg: int = 1 << 24):
        self.salt = salt
        self.max_msg = max_msg

    def _keystream(self, model: Any, length: int) -> bytes:
        """Derive a pseudo-random keystream from the model's weights."""
        weights = _weights_of(model)
        h = hashlib.blake2b(key=self.salt)
        for name in sorted(weights):
            h.update(name.encode("utf-8"))
            h.update(np.asarray(weights[name], dtype=np.float32).tobytes())
        seed = int.from_bytes(h.digest()[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.bytes(max(0, length))

    def encrypt(self, model: Any, message: str) -> bytes:
        """Encrypt a plaintext string into bytes keyed by the model's weights."""
        msg = message.encode("utf-8")[: self.max_msg]
        if not msg:
            return b"\x00\x00\x00\x00"
        ks = self._keystream(model, len(msg))
        ct = bytes(a ^ b for a, b in zip(msg, ks))
        return len(msg).to_bytes(4, "big") + ct

    def decrypt(self, model: Any, ciphertext: bytes) -> str:
        """Decrypt ciphertext produced by ``encrypt`` using the same model."""
        if len(ciphertext) < 4:
            return ""
        n = int.from_bytes(ciphertext[:4], "big")
        if n <= 0 or n > self.max_msg or len(ciphertext) < 4 + n:
            return ""
        ct = ciphertext[4:4 + n]
        ks = self._keystream(model, n)
        pt = bytes(a ^ b for a, b in zip(ct, ks))
        return pt.decode("utf-8", errors="replace")

    def key_strength(self, model: Any) -> float:
        """Estimate effective key strength in bits from weight entropy."""
        weights = _weights_of(model)
        if not weights:
            return 0.0
        entropies: List[float] = []
        for W in weights.values():
            a = np.asarray(W, dtype=np.float64).ravel()
            if a.size < 2:
                continue
            std = float(a.std()) + 1e-12
            # differential entropy proxy of a Gaussian
            entropies.append(0.5 * math.log2(2 * math.pi * math.e * std * std))
        avg = float(np.mean(entropies)) if entropies else 0.0
        n_params = sum(int(np.asarray(W).size) for W in weights.values())
        bits = avg * math.sqrt(max(n_params, 1)) / 4.0
        return float(min(256.0, max(0.0, bits)))


# ============================================================================
# Idea 111 -- ModelBonsai
# ============================================================================
class ModelBonsai:
    """Treat a model as a bonsai tree: prune, wire (shape), and root-prune
    over time. Each *season* of care improves the model's structure.

      - prune_branches : thin out small weights (the twigs)
      - wire           : bend the structure toward a target shape (low-rank)
      - root_prune     : cut entire low-magnitude rows (the roots)
      - season         : a full cycle of all three, logged in the lineage
    """

    def __init__(self, prune_threshold: float = 0.01,
                 wire_rank: int = 8, root_keep: float = 0.9):
        self.prune_threshold = prune_threshold
        self.wire_rank = wire_rank
        self.root_keep = root_keep
        self._lineage: List[Dict[str, Any]] = []

    def prune_branches(self, model: Any,
                       threshold: Optional[float] = None) -> Any:
        """Thin out weight values smaller than ``threshold`` (set to 0)."""
        thr = threshold if threshold is not None else self.prune_threshold
        weights = _weights_of(model)
        n_pruned = 0
        for name, W in weights.items():
            Wm = np.array(W, dtype=np.float32, copy=True)
            mask = np.abs(Wm) < thr
            n_pruned += int(mask.sum())
            Wm[mask] = 0.0
            weights[name] = Wm
        self._lineage.append({"action": "prune", "threshold": thr,
                              "n_pruned": n_pruned, "t": time.time()})
        return _apply(model, weights)

    def wire(self, model: Any,
             target_shape: Optional[Dict[str, Tuple[int, int]]] = None) -> Any:
        """Bend the model toward ``target_shape`` via low-rank projection.

        If ``target_shape`` is None, applies a low-rank approximation (the
        "wire") to each 2-D weight matrix, compressing it toward rank
        ``self.wire_rank``.
        """
        weights = _weights_of(model)
        for name, W in weights.items():
            Wm = np.asarray(W, dtype=np.float32)
            if Wm.ndim != 2:
                continue
            if target_shape and name in target_shape:
                r = min(*target_shape[name])
            else:
                r = min(self.wire_rank, *Wm.shape)
            try:
                U, S, Vt = np.linalg.svd(Wm, full_matrices=False)
                Wm = ((U[:, :r] * S[:r]) @ Vt[:r]).astype(np.float32)
            except np.linalg.LinAlgError:
                continue
            weights[name] = Wm
        self._lineage.append({"action": "wire", "rank": self.wire_rank,
                              "t": time.time()})
        return _apply(model, weights)

    def _root_prune(self, model: Any) -> Any:
        """Cut the lowest-magnitude rows of every 2-D weight matrix."""
        weights = _weights_of(model)
        n_cut = 0
        for name, W in weights.items():
            Wm = np.array(W, dtype=np.float32, copy=True)
            if Wm.ndim != 2:
                continue
            row_norms = np.linalg.norm(Wm, axis=1)
            k = max(1, int(self.root_keep * Wm.shape[0]))
            keep_idx = np.argsort(row_norms)[-k:]
            mask = np.ones(Wm.shape[0], dtype=bool)
            mask[keep_idx] = False
            Wm[mask] = 0.0
            n_cut += int(Wm.shape[0] - k)
            weights[name] = Wm
        self._lineage.append({"action": "root_prune", "n_cut": n_cut,
                              "t": time.time()})
        return _apply(model, weights)

    def season(self, model: Any) -> Any:
        """Run one full season of bonsai care: prune -> wire -> root-prune."""
        m = self.prune_branches(model)
        m = self.wire(m)
        m = self._root_prune(m)
        self._lineage.append({"action": "season", "t": time.time()})
        return m

    def lineage(self, model: Any) -> List[Dict[str, Any]]:
        """Return the history of bonsai operations applied so far."""
        return list(self._lineage)


# ============================================================================
# Idea 112 -- HallucinationVaccine
# ============================================================================
class HallucinationVaccine:
    """Like a vaccine: inject tiny controlled doses of "hallucination
    pathogens" (mislabeled / out-of-distribution samples) to build model
    immunity. The model learns to *abstain* on pathological inputs instead
    of confidently hallucinating.

    A ``Vaccine`` is a small bundle of pathogens + an adjuvant (mixing
    ratio). ``administer`` exposes the model to the pathogens at low dose.
    """

    def __init__(self, dose: float = 0.05, n_pathogens: int = 16,
                 abstain_token: Any = None):
        self.dose = dose
        self.n_pathogens = n_pathogens
        self.abstain_token = abstain_token
        self._vaccines: Dict[str, Dict[str, Any]] = {}

    def formulate(self, model: Any,
                  weak_spots: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """Formulate a vaccine from (pathological_input, wrong_label) pairs.

        Returns a vaccine dict containing the pathogens and an adjuvant.
        """
        rng = np.random.default_rng(42)
        pathogens: List[Dict[str, Any]] = []
        if weak_spots:
            per = max(1, self.n_pathogens // max(1, len(weak_spots)))
            for x, y in weak_spots[: self.n_pathogens]:
                xa = np.asarray(x, dtype=np.float32).ravel()
                for _ in range(per):
                    noise = rng.normal(0, 0.05, size=xa.shape).astype(np.float32)
                    pathogens.append({"x": (xa + noise), "label": y})
        adjuvant = {"scale": self.dose, "abstain_token": self.abstain_token}
        vaccine = {"pathogens": pathogens, "adjuvant": adjuvant,
                   "n_strains": len(pathogens), "formulated_at": time.time()}
        self._vaccines[str(id(model))] = vaccine
        return vaccine

    def administer(self, model: Any, vaccine: Dict[str, Any]) -> Any:
        """Expose the model to the vaccine at a low dose (gentle fit)."""
        pathogens = vaccine.get("pathogens", [])
        scale = float(vaccine.get("adjuvant", {}).get("scale", self.dose))
        if not pathogens:
            return model
        X = np.stack([p["x"] for p in pathogens]).astype(np.float32)
        if self.abstain_token is not None:
            y = np.array([self.abstain_token] * len(pathogens))
        else:
            y = np.array([p["label"] for p in pathogens])
        X_aug = X * (1.0 - scale) + np.random.normal(
            0, scale, X.shape).astype(np.float32)
        try:
            fn = getattr(model, "partial_fit", None) or getattr(model, "fit", None)
            if fn is not None:
                fn(X_aug, y)
        except Exception:
            pass
        return model

    def immunity_report(self, model: Any) -> Dict[str, Any]:
        """Report the model's immunity (how brittle its weight distribution is)."""
        weights = _weights_of(model)
        if not weights:
            return {"immunity": 0.0, "n_vaccines": len(self._vaccines),
                    "dose": self.dose}
        all_w = np.concatenate(
            [np.asarray(W).ravel() for W in weights.values()])
        std = float(all_w.std()) + 1e-8
        kurt = float(((all_w - all_w.mean()) ** 4).mean() / (std ** 4)) - 3.0
        immunity = float(max(0.0, min(1.0, 1.0 - kurt / 12.0)))
        return {"immunity": immunity, "kurtosis": kurt,
                "n_vaccines": len(self._vaccines), "dose": self.dose}

    def booster(self, model: Any) -> Any:
        """Re-administer the last vaccine at half dose (a booster shot)."""
        v = self._vaccines.get(str(id(model)))
        if v is None:
            return model
        booster_v = dict(v)
        booster_v["adjuvant"] = dict(v["adjuvant"])
        booster_v["adjuvant"]["scale"] = float(v["adjuvant"]["scale"]) * 0.5
        return self.administer(model, booster_v)


# ============================================================================
# Idea 113 -- QuantumEntanglement
# ============================================================================
class QuantumEntanglement:
    """Pairs of neurons are "entangled": changes to one instantaneously
    affect the other. We detect entangled pairs by finding neurons whose
    incoming weight vectors are highly correlated (they fire together
    because they listen to the same inputs in lockstep).

    ``bell_test`` checks whether a pair is *genuinely* entangled (the
    correlation survives a permutation control) vs. accidentally similar.
    """

    def __init__(self, min_correlation: float = 0.85,
                 max_pairs: int = 50, bell_trials: int = 200):
        self.min_correlation = min_correlation
        self.max_pairs = max_pairs
        self.bell_trials = bell_trials
        self._pairs: Dict[str, List[Tuple[str, int, int, float]]] = {}

    def entangle(self, model: Any) -> List[Tuple[str, int, int, float]]:
        """Find entangled neuron pairs across layers; return the top pairs."""
        weights = _weights_of(model)
        pairs: List[Tuple[str, int, int, float]] = []
        for name, W in weights.items():
            Wm = np.asarray(W, dtype=np.float32)
            if Wm.ndim != 2 or Wm.shape[1] < 2:
                continue
            cols = Wm - Wm.mean(axis=0, keepdims=True)
            norms = np.linalg.norm(cols, axis=0) + 1e-8
            cols_n = cols / norms
            corr = cols_n.T @ cols_n
            np.fill_diagonal(corr, 0.0)
            iu = np.triu_indices_from(corr, k=1)
            if iu[0].size == 0:
                continue
            order = np.argsort(-corr[iu])
            per_layer = max(1, self.max_pairs // max(1, len(weights)))
            for idx in order[:per_layer]:
                i, j = int(iu[0][idx]), int(iu[1][idx])
                c = float(corr[i, j])
                if abs(c) >= self.min_correlation:
                    pairs.append((name, i, j, c))
        pairs.sort(key=lambda p: -abs(p[3]))
        self._pairs[str(id(model))] = pairs
        return pairs

    def measure(self, model: Any, neuron_id: Tuple[str, int]) -> float:
        """Measure the "state" (L2 norm) of a single neuron."""
        name, idx = neuron_id
        weights = _weights_of(model)
        if name not in weights:
            return 0.0
        Wm = np.asarray(weights[name], dtype=np.float32)
        if Wm.ndim != 2 or not (0 <= idx < Wm.shape[1]):
            return 0.0
        return float(np.linalg.norm(Wm[:, idx]))

    def bell_test(self, model: Any, pair: Tuple[str, int, int]) -> bool:
        """Bell-test: is the pair's correlation above chance?"""
        name, i, j = pair
        weights = _weights_of(model)
        if name not in weights:
            return False
        Wm = np.asarray(weights[name], dtype=np.float32)
        if Wm.ndim != 2 or not (0 <= i < Wm.shape[1] and 0 <= j < Wm.shape[1]):
            return False
        a = Wm[:, i] - Wm[:, i].mean()
        b = Wm[:, j] - Wm[:, j].mean()
        obs = abs(float(np.dot(a, b) /
                        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))
        rng = np.random.default_rng(0)
        null = np.empty(self.bell_trials, dtype=np.float32)
        for t in range(self.bell_trials):
            perm = rng.permutation(len(b))
            bp = b[perm]
            null[t] = abs(float(np.dot(a, bp) /
                                (np.linalg.norm(a) * np.linalg.norm(bp) + 1e-8)))
        p = float((null >= obs).mean())
        return bool(obs >= self.min_correlation and p < 0.05)


# ============================================================================
# Idea 114 -- ModelConcert
# ============================================================================
class ModelConcert:
    """Multiple models play together like an orchestra. Each model is an
    instrument with a *part* (a subset of features) on which it solo-plays
    best. A *conductor* model combines all the parts into the final
    prediction.

    ``assign_parts`` analyses each model's per-feature sensitivity to give
    it the features it is best at. ``rehearse`` fits the conductor.
    ``perform`` plays a single input through the whole ensemble.
    """

    def __init__(self, conductor: Optional[Any] = None,
                 n_features: Optional[int] = None):
        self.conductor = conductor
        self.n_features = n_features
        self.parts: Dict[int, List[int]] = {}
        self._rehearsed = False

    def assign_parts(self, models: List[Any], X: np.ndarray) -> Dict[str, Any]:
        """Give each model the features on which it is most sensitive."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        n_feat = self.n_features or X.shape[1]
        scores = np.zeros((len(models), n_feat), dtype=np.float32)
        for mi, m in enumerate(models):
            base = _to_float(_predict(m, X[0]))
            for f in range(min(n_feat, X.shape[1])):
                Xp = X.copy()
                Xp[:, f] += 0.1
                try:
                    raw = [_predict(m, xi) for xi in Xp]
                    pred = np.asarray(
                        [_to_float(r) for r in raw], dtype=np.float32)
                    scores[mi, f] = float(abs(pred.mean() - base))
                except Exception:
                    scores[mi, f] = 0.0
        owners = scores.argmax(axis=0)
        parts: Dict[int, List[int]] = {mi: [] for mi in range(len(models))}
        for f, mi in enumerate(owners):
            parts[int(mi)].append(f)
        self.parts = parts
        return {"parts": {str(k): v for k, v in parts.items()},
                "sensitivity": scores.tolist()}

    def rehearse(self, ensemble: List[Any], X: np.ndarray,
                 y: np.ndarray) -> List[Any]:
        """Rehearse: fit the conductor on the ensemble's solo predictions."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if not self.parts:
            self.assign_parts(ensemble, X)
        per_model: List[np.ndarray] = []
        for mi, m in enumerate(ensemble):
            feats = self.parts.get(mi, [])
            Xs = X[:, feats] if (feats and X.ndim == 2) else X
            try:
                preds = np.asarray(
                    [_to_float(_predict(m, xi)) for xi in Xs], dtype=np.float32)
            except Exception:
                preds = np.zeros(len(Xs), dtype=np.float32)
            per_model.append(preds)
        if not per_model or self.conductor is None:
            self._rehearsed = True
            return ensemble
        feat_mat = np.stack(per_model, axis=1)
        try:
            self.conductor.fit(feat_mat, y)
        except Exception:
            pass
        self._rehearsed = True
        return ensemble

    def perform(self, models: List[Any], x: Any) -> Any:
        """Play a single input through the whole orchestra."""
        xa = np.asarray(x, dtype=np.float32).ravel()
        solos: List[Any] = []
        for mi, m in enumerate(models):
            feats = self.parts.get(mi, [])
            xi = xa[feats] if feats else xa
            solos.append(_predict(m, xi))
        if self.conductor is not None and self._rehearsed:
            try:
                solo_vals = np.asarray(
                    [_to_float(s) for s in solos], dtype=np.float32).reshape(1, -1)
                return self.conductor.predict(solo_vals)
            except Exception:
                pass
        try:
            vals = [_to_float(s) for s in solos if s is not None]
            return float(np.mean(vals)) if vals else None
        except Exception:
            return None

    def conduct(self, models: List[Any], x: Any) -> Any:
        """Conductor-only prediction (expects already-rehearsed parts)."""
        return self.perform(models, x)


# ============================================================================
# Idea 115 -- NeuralWeather
# ============================================================================
class NeuralWeather:
    """Forecast the "weather" inside a model during training: loss storms,
    gradient droughts, attention hurricanes. Predicts convergence weather
    for the next 100 steps from a history of (loss, grad_norm) tuples.

      - storm     : loss spike > mean + storm_z * std
      - drought   : gradient norm < drought_tol sustained
      - hurricane : gradient volatility > 2x baseline volatility
    """

    def __init__(self, window: int = 20, horizon: int = 100,
                 storm_z: float = 3.0, drought_tol: float = 1e-3):
        self.window = window
        self.horizon = horizon
        self.storm_z = storm_z
        self.drought_tol = drought_tol

    def pressure_map(self, gradients: List[np.ndarray]) -> np.ndarray:
        """Build a 2-D pressure map (rows=timesteps, cols=gradient dims)."""
        grads = [np.asarray(g, dtype=np.float32).ravel()
                 for g in gradients if g is not None]
        if not grads:
            return np.zeros((0, 0), dtype=np.float32)
        width = max(g.size for g in grads)
        mat = np.zeros((len(grads), width), dtype=np.float32)
        for i, g in enumerate(grads):
            mat[i, : g.size] = g
        if mat.shape[0] >= self.window:
            kernel = np.ones(self.window, dtype=np.float32) / self.window
            smoothed = np.apply_along_axis(
                lambda v: np.convolve(v, kernel, mode="same"), 0, mat)
        else:
            smoothed = mat
        return smoothed

    def forecast(self, history: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Predict the next ``horizon`` steps of weather."""
        if len(history) < 2:
            return {"horizon": self.horizon, "conditions": "unknown",
                    "forecast_loss": [], "storm_risk": 0.0}
        losses = np.asarray([h[0] for h in history], dtype=np.float32)
        grads = np.asarray([h[1] for h in history], dtype=np.float32)
        dx = np.diff(losses)
        w = min(self.window, len(dx))
        drift = float(np.mean(dx[-w:]))
        vol = float(np.std(dx[-w:])) + 1e-8
        baseline_vol = float(np.std(dx)) + 1e-8
        rng = np.random.default_rng(0)
        future = np.empty(self.horizon, dtype=np.float32)
        cur = float(losses[-1])
        for t in range(self.horizon):
            cur = max(0.0, cur + drift + rng.normal(0, vol))
            future[t] = cur
        mean_grad = float(np.mean(grads[-w:]))
        if vol > 2.0 * baseline_vol:
            conditions = "stormy"
        elif mean_grad < self.drought_tol:
            conditions = "drought"
        elif drift < -1e-3:
            conditions = "clearing"
        else:
            conditions = "overcast"
        return {"horizon": self.horizon, "conditions": conditions,
                "forecast_loss": future.tolist(),
                "drift": drift, "volatility": vol,
                "storm_risk": float(min(1.0, vol / (abs(drift) + vol + 1e-8)))}

    def storm_warning(self, history: List[Tuple[float, float]]) -> bool:
        """Return True if a loss storm is likely within the next 100 steps."""
        if len(history) < self.window:
            return False
        losses = np.asarray([h[0] for h in history], dtype=np.float32)
        recent = losses[-self.window:]
        mu = float(recent.mean())
        sigma = float(recent.std()) + 1e-8
        return bool(losses[-1] > mu + self.storm_z * sigma)


# ============================================================================
# Idea 116 -- WeightGenealogy
# ============================================================================
class WeightGenealogy:
    """Trace the ancestry of every weight: where did it come from?
    (init? pretrained? finetuned? LoRA delta?) Build a family tree of
    weights.

    The first time we see a model, every weight is tagged as a *founder*
    (origin = init). On later calls, weights that changed beyond a
    tolerance are tagged as *immigrants* (origin = finetune delta).
    """

    def __init__(self, drift_tol: float = 1e-4, lora_rank: int = 4):
        self.drift_tol = drift_tol
        self.lora_rank = lora_rank
        self._founders: Dict[str, Dict[str, np.ndarray]] = {}
        self._log: List[Dict[str, Any]] = []

    def _model_id(self, model: Any) -> str:
        return str(id(model))

    def _register(self, model: Any) -> None:
        mid = self._model_id(model)
        if mid not in self._founders:
            self._founders[mid] = {
                k: np.asarray(v, dtype=np.float64).copy()
                for k, v in _weights_of(model).items()
            }
            self._log.append({"event": "founding", "model_id": mid,
                              "n_layers": len(self._founders[mid]),
                              "t": time.time()})

    def trace(self, model: Any, weight_id: Tuple[str, int]) -> Dict[str, Any]:
        """Trace the ancestry of a single weight."""
        self._register(model)
        mid = self._model_id(model)
        founders = self._founders[mid]
        current = _weights_of(model)
        layer, idx = weight_id
        if layer not in current or layer not in founders:
            return {"origin": "unknown"}
        cur = np.asarray(current[layer], dtype=np.float64).ravel()
        fnd = np.asarray(founders[layer], dtype=np.float64).ravel()
        if idx >= len(cur) or idx >= len(fnd):
            return {"layer": layer, "index": idx, "origin": "new (added later)"}
        delta = float(cur[idx] - fnd[idx])
        if abs(delta) < self.drift_tol:
            origin = "founder (init)"
        else:
            origin = "immigrant (finetune delta)"
        return {"layer": layer, "index": idx,
                "founder_value": float(fnd[idx]),
                "current_value": float(cur[idx]),
                "delta": delta, "origin": origin}

    def family_tree(self, model: Any) -> Dict[str, Dict[str, Any]]:
        """Build a family tree of all weights (per-layer counts)."""
        self._register(model)
        mid = self._model_id(model)
        founders = self._founders[mid]
        current = _weights_of(model)
        tree: Dict[str, Dict[str, Any]] = {}
        for layer in current:
            if layer not in founders:
                cur = np.asarray(current[layer], dtype=np.float64).ravel()
                tree[layer] = {"n_founders": 0, "n_immigrants": int(cur.size),
                               "max_delta": 0.0, "mean_delta": 0.0}
                continue
            cur = np.asarray(current[layer], dtype=np.float64).ravel()
            fnd = np.asarray(founders[layer], dtype=np.float64).ravel()
            n = min(len(cur), len(fnd))
            deltas = np.abs(cur[:n] - fnd[:n])
            n_imm = int((deltas > self.drift_tol).sum())
            tree[layer] = {"n_founders": int(n - n_imm),
                           "n_immigrants": n_imm,
                           "max_delta": float(deltas.max()) if n else 0.0,
                           "mean_delta": float(deltas.mean()) if n else 0.0}
        return tree

    def founder_weights(self, model: Any) -> List[Tuple[str, int]]:
        """Return (layer, index) of weights unchanged since init."""
        self._register(model)
        mid = self._model_id(model)
        founders = self._founders[mid]
        current = _weights_of(model)
        out: List[Tuple[str, int]] = []
        for layer in current:
            if layer not in founders:
                continue
            cur = np.asarray(current[layer], dtype=np.float64).ravel()
            fnd = np.asarray(founders[layer], dtype=np.float64).ravel()
            n = min(len(cur), len(fnd))
            stable = np.where(np.abs(cur[:n] - fnd[:n]) < self.drift_tol)[0]
            out.extend([(layer, int(i)) for i in stable])
        return out

    def immigrants(self, model: Any) -> List[Tuple[str, int]]:
        """Return (layer, index) of weights added/changed by finetuning."""
        self._register(model)
        mid = self._model_id(model)
        founders = self._founders[mid]
        current = _weights_of(model)
        out: List[Tuple[str, int]] = []
        for layer in current:
            if layer not in founders:
                cur = np.asarray(current[layer]).ravel()
                out.extend([(layer, int(i)) for i in range(len(cur))])
                continue
            cur = np.asarray(current[layer], dtype=np.float64).ravel()
            fnd = np.asarray(founders[layer], dtype=np.float64).ravel()
            n = min(len(cur), len(fnd))
            moved = np.where(np.abs(cur[:n] - fnd[:n]) >= self.drift_tol)[0]
            out.extend([(layer, int(i)) for i in moved])
        return out


# ============================================================================
# Idea 117 -- ModelShrine
# ============================================================================
class ModelShrine:
    """A retired model deserves veneration. Preserve its memory as
    *relics*: distilled knowledge (per-layer stats), sample predictions,
    and signature activations (hashes that uniquely identify the model).

    ``canonize`` builds a Shrine from a model. ``visit`` uses the shrine's
    relics to predict. ``miracle_check`` tests whether the shrine still
    predicts correctly on a held-out input.
    """

    def __init__(self, n_relics: int = 8, n_signatures: int = 16):
        self.n_relics = n_relics
        self.n_signatures = n_signatures
        self.shrines: Dict[str, Dict[str, Any]] = {}

    def canonize(self, model: Any, name: str,
                 X: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Canonize a retired model: extract relics + signature activations."""
        weights = _weights_of(model)
        distilled = {
            k: {"mean": float(np.asarray(v).mean()),
                "std": float(np.asarray(v).std()),
                "shape": list(np.asarray(v).shape)}
            for k, v in weights.items()
        }
        sample_preds: List[Any] = []
        if X is not None:
            Xin = np.asarray(X, dtype=np.float32)
            if Xin.ndim == 1:
                Xin = Xin[None, :]
            for x in Xin[: self.n_relics]:
                try:
                    sample_preds.append(_predict(model, x))
                except Exception:
                    sample_preds.append(None)
        signatures: List[Dict[str, Any]] = []
        for k in list(weights.keys())[: self.n_signatures]:
            h = hashlib.blake2b(
                np.asarray(weights[k], dtype=np.float32).tobytes(),
                digest_size=8).hexdigest()
            signatures.append({"layer": k, "sig": h})
        shrine = {"name": name, "distilled": distilled,
                  "sample_predictions": sample_preds,
                  "signatures": signatures,
                  "canonized_at": time.time(),
                  "n_params": sum(int(np.asarray(v).size)
                                  for v in weights.values())}
        self.shrines[name] = shrine
        return shrine

    def visit(self, shrine: Dict[str, Any], x: Any) -> Any:
        """Use the shrine's relics to produce a prediction.

        The shrine no longer has the original weights, so it predicts via
        nearest-neighbour over its stored sample predictions (a small
        non-parametric oracle), falling back to the distilled layer means.
        """
        xa = np.asarray(x, dtype=np.float32).ravel()
        samples = shrine.get("sample_predictions", [])
        arrs: List[np.ndarray] = []
        for s in samples:
            try:
                a = np.asarray(s, dtype=np.float32).ravel()
                arrs.append(a)
            except Exception:
                pass
        if arrs:
            mat = np.stack([a / (np.linalg.norm(a) + 1e-8) for a in arrs])
            xnorm = xa / (np.linalg.norm(xa) + 1e-8)
            if xnorm.size < mat.shape[1]:
                xnorm = np.pad(xnorm, (0, mat.shape[1] - xnorm.size))
            else:
                xnorm = xnorm[: mat.shape[1]]
            sims = mat @ xnorm
            top = int(np.argmax(sims))
            return samples[top]
        distilled = shrine.get("distilled", {})
        means = [d["mean"] for d in distilled.values()]
        if means:
            return float(np.mean(means) * float(np.mean(xa)))
        return None

    def relics(self, shrine: Dict[str, Any]) -> List[str]:
        """List the relics held by a shrine."""
        return [k for k in shrine.keys()
                if k not in ("name", "canonized_at", "n_params")]

    def miracle_check(self, shrine: Dict[str, Any], x: Any,
                      expected: Optional[Any] = None) -> bool:
        """Does the shrine still predict correctly on this input?

        If ``expected`` is None, the check is "did the shrine produce a
        non-None prediction at all" (a small miracle for a retired model).
        """
        pred = self.visit(shrine, x)
        if pred is None:
            return False
        if expected is None:
            return True
        try:
            return bool(np.allclose(
                np.asarray(pred, dtype=np.float32),
                np.asarray(expected, dtype=np.float32), atol=1e-3))
        except Exception:
            return False


# ============================================================================
# Idea 118 -- CognitiveEcho
# ============================================================================
class CognitiveEcho:
    """A model that replays its own past predictions to detect drift over
    time. Like an echo chamber but for self-monitoring: every prediction
    is recorded; when a similar input recurs, the model "hears" its past
    self and compares.

    ``drift_score`` measures how much the model's behaviour has changed
    since the first recording (by comparing current weights to the
    baseline snapshot taken on first ``record``).
    """

    def __init__(self, memory_size: int = 512, similarity_top_k: int = 5):
        self.memory_size = memory_size
        self.similarity_top_k = similarity_top_k
        self._memory: List[Dict[str, Any]] = []
        self._baseline: Optional[Dict[str, Any]] = None

    def record(self, model: Any, input: Any, output: Any) -> None:
        """Record a (input, output) prediction into echo memory."""
        entry = {"input": np.asarray(input, dtype=np.float32).ravel(),
                 "output": output, "t": time.time()}
        self._memory.append(entry)
        if len(self._memory) > self.memory_size:
            self._memory = self._memory[-self.memory_size:]
        if self._baseline is None:
            # store copies, not references, so in-place weight mutations
            # are actually detected by drift_score later
            self._baseline = {
                "weights": {k: np.array(v, copy=True)
                            for k, v in _weights_of(model).items()},
                "t": time.time(),
            }

    def echo(self, model: Any, input: Any) -> List[Any]:
        """Replay past predictions for inputs similar to ``input``."""
        if not self._memory:
            return []
        x = np.asarray(input, dtype=np.float32).ravel()
        sims: List[float] = []
        for m in self._memory:
            mi = m["input"]
            n = min(len(x), len(mi))
            if n == 0:
                sims.append(0.0)
                continue
            a, b = x[:n], mi[:n]
            cos = float(np.dot(a, b) /
                        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            sims.append(cos)
        top = np.argsort(sims)[-self.similarity_top_k:][::-1]
        return [self._memory[i]["output"] for i in top if sims[i] > 0.0]

    def drift_score(self, model: Any) -> float:
        """How much has the model changed since recording began?"""
        if self._baseline is None:
            return 0.0
        base = self._baseline["weights"]
        cur = _weights_of(model)
        deltas: List[float] = []
        for layer in set(base) & set(cur):
            a = np.asarray(base[layer], dtype=np.float64).ravel()
            b = np.asarray(cur[layer], dtype=np.float64).ravel()
            n = min(len(a), len(b))
            if n == 0:
                continue
            d = float(np.linalg.norm(a[:n] - b[:n]) /
                      (np.linalg.norm(a[:n]) + 1e-8))
            deltas.append(d)
        return float(np.mean(deltas)) if deltas else 0.0

    def silent_recording(self, model: Any) -> Dict[str, Any]:
        """A snapshot of the model's internal state without a forward pass."""
        weights = _weights_of(model)
        stats = {
            k: {"mean": float(np.asarray(v).mean()),
                "std": float(np.asarray(v).std()),
                "norm": float(np.linalg.norm(np.asarray(v).ravel()))}
            for k, v in weights.items()
        }
        return {"n_recorded": len(self._memory),
                "weight_stats": stats,
                "baseline_set": self._baseline is not None}


__all__ = [
    "ModelAcupuncture",
    "NeuroCryptography",
    "ModelBonsai",
    "HallucinationVaccine",
    "QuantumEntanglement",
    "ModelConcert",
    "NeuralWeather",
    "WeightGenealogy",
    "ModelShrine",
    "CognitiveEcho",
]
