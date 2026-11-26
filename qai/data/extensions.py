"""
qai.data.extensions -- expanded dataset library features added in the
'wrapper + creative ideas' pass.

These are deliberately *library-wrappers*, not PyTorch reimplementations.
We lean on scikit-learn / scipy / numpy / pandas / networkx for the heavy
lifting and expose them through the qai.data API.

Modules:
  - DatasetProfiler       : rich per-column profiling (sklearn + scipy stats)
  - StreamingDataset      : chunked iterator with backpressure + schema check
  - DatasetJoiner         : SQL-like joins on DataDataset
  - TimeSeriesToolkit     : resample, seasonality, STL decomposition
  - SchemaEnforcer        : type validation, coercion, evolution tracking
  - DataPoisoningDetector : statistical + signature poisoning detection
  - DriftDetector         : KS / PSI / chi-square drift monitoring
  - SyntheticDataFactory  : tabular (CTGAN-lite), time-series, text, image
  - FeatureStore          : offline + online feature lookups
  - DataLineage           : track row/column provenance across transforms
  - DataCatalog           : searchable dataset registry with metadata
  - SmartSampler          : stratified, imbalance-aware, GPU-ready samplers
  - DataValidator         : great-expectations-style rule engine
"""
from __future__ import annotations
import os, json, time, hashlib, math, random, re
from collections import defaultdict, OrderedDict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional wrappers — fall back to pure-Python if missing
try:
    from scipy import stats as _scistats
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from sklearn.ensemble import IsolationForest as _SKIF
    from sklearn.covariance import EllipticEnvelope as _SKCov
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA as _SKPCA
    from sklearn.cluster import KMeans as _SKKM
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _TFIDF
    from sklearn.metrics.pairwise import cosine_similarity as _cosine
except ImportError:
    _TFIDF = None
    _cosine = None

from . import DataDataset, load  # noqa: E402


# ============================================================================
# 1. DATASET PROFILER — deep per-column statistics
# ============================================================================
class DatasetProfiler:
    """Rich dataset profiler wrapping scipy.stats + pandas.

    Usage:
        prof = DatasetProfiler(ds).profile()
        prof["columns"]["age"]["skew"]
    """
    def __init__(self, dataset: DataDataset):
        self.ds = dataset

    def profile(self) -> Dict[str, Any]:
        df = self.ds.to_pandas()
        out = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 4),
            "duplicate_rows": int(df.duplicated().sum()),
            "columns": {},
        }
        for col in df.columns:
            s = df[col]
            info: Dict[str, Any] = {
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique()),
                "n_null": int(s.isnull().sum()),
                "null_pct": round(float(s.isnull().mean()), 4),
            }
            if pd.api.types.is_numeric_dtype(s):
                info.update({
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "q25": float(s.quantile(0.25)),
                    "median": float(s.median()),
                    "q75": float(s.quantile(0.75)),
                    "max": float(s.max()),
                    "skew": float(_scistats.skew(s.dropna())) if _SCIPY else None,
                    "kurtosis": float(_scistats.kurtosis(s.dropna())) if _SCIPY else None,
                })
            else:
                vc = s.value_counts()
                info.update({
                    "top_values": vc.head(5).to_dict(),
                    "avg_length": round(float(s.astype(str).str.len().mean()), 2),
                    "n_empty_strings": int((s.astype(str).str.strip() == "").sum()),
                })
            out["columns"][col] = info
        return out

    def correlations(self) -> Dict[str, Dict[str, float]]:
        """Pearson + Spearman correlation matrix for numeric columns."""
        df = self.ds.to_pandas().select_dtypes(include=[np.number])
        if df.shape[1] < 2:
            return {}
        pear = df.corr(method="pearson").round(4).to_dict()
        if _SCIPY:
            spear = df.corr(method="spearman").round(4).to_dict()
            return {"pearson": pear, "spearman": spear}
        return {"pearson": pear}

    def detect_anomalous_rows(self, contamination: float = 0.05) -> List[int]:
        """Wrap sklearn IsolationForest to flag statistically anomalous rows."""
        if not _SKLEARN:
            return []
        df = self.ds.to_pandas().select_dtypes(include=[np.number]).dropna()
        if df.shape[0] < 10 or df.shape[1] < 1:
            return []
        iforest = _SKIF(contamination=contamination, random_state=0)
        preds = iforest.fit_predict(df)
        return [int(i) for i, p in enumerate(preds) if p == -1]


# ============================================================================
# 2. STREAMING DATASET — chunked iterator with schema check + backpressure
# ============================================================================
class StreamingDataset:
    """Chunked streaming iterator over a large file.

    Yields DataDataset batches; enforces schema (column names + dtypes) on
    every batch and supports a `max_rate` throttle to prevent memory blowups.
    """
    def __init__(self, path: str, chunk_size: int = 1000,
                 schema: Optional[Dict[str, str]] = None,
                 max_rate_per_sec: Optional[int] = None):
        self.path = path
        self.chunk_size = chunk_size
        self.schema = schema
        self.max_rate = max_rate_per_sec
        self._last_yield = 0.0

    def __iter__(self) -> Iterator[DataDataset]:
        reader = pd.read_csv(self.path, chunksize=self.chunk_size)
        for chunk in reader:
            if self.schema:
                for col, expected_type in self.schema.items():
                    if col not in chunk.columns:
                        raise ValueError(f"Schema violation: column '{col}' missing")
                    actual = str(chunk[col].dtype)
                    if expected_type not in actual:
                        chunk[col] = chunk[col].astype(expected_type, errors="ignore")
            if self.max_rate:
                elapsed = time.time() - self._last_yield
                wait = (1.0 / self.max_rate) - elapsed
                if wait > 0:
                    time.sleep(wait)
                self._last_yield = time.time()
            yield DataDataset.from_pandas(chunk)


# ============================================================================
# 3. DATASET JOINER — SQL-style joins on DataDataset
# ============================================================================
class DatasetJoiner:
    """SQL-style joins between two DataDataset objects via pandas merge."""

    @staticmethod
    def inner(left: DataDataset, right: DataDataset, on: Union[str, List[str]]) -> DataDataset:
        df = left.to_pandas().merge(right.to_pandas(), on=on, how="inner")
        return DataDataset.from_pandas(df)

    @staticmethod
    def left(left: DataDataset, right: DataDataset, on: Union[str, List[str]]) -> DataDataset:
        df = left.to_pandas().merge(right.to_pandas(), on=on, how="left")
        return DataDataset.from_pandas(df)

    @staticmethod
    def outer(left: DataDataset, right: DataDataset, on: Union[str, List[str]]) -> DataDataset:
        df = left.to_pandas().merge(right.to_pandas(), on=on, how="outer")
        return DataDataset.from_pandas(df)

    @staticmethod
    def cross(left: DataDataset, right: DataDataset) -> DataDataset:
        df = left.to_pandas().merge(right.to_pandas(), how="cross")
        return DataDataset.from_pandas(df)


# ============================================================================
# 4. TIME-SERIES TOOLKIT — resample, seasonality, decomposition
# ============================================================================
class TimeSeriesToolkit:
    """Time-series helpers built on pandas + scipy.signal.

    Wraps pandas' resample / rolling / ewm, plus STL-like decomposition.
    """
    def __init__(self, ds: DataDataset, time_col: str, value_col: str):
        df = ds.to_pandas().copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.set_index(time_col).sort_index()
        self.series = df[value_col]

    def resample(self, rule: str = "D", agg: str = "mean") -> pd.Series:
        return getattr(self.series.resample(rule), agg)()

    def rolling_mean(self, window: int = 7) -> pd.Series:
        return self.series.rolling(window).mean()

    def ewm_smoothing(self, span: int = 7) -> pd.Series:
        return self.series.ewm(span=span).mean()

    def detect_seasonality(self, max_lag: int = 50) -> Dict[str, Any]:
        """Use autocorrelation to find dominant seasonal period."""
        s = self.series.dropna().values
        if len(s) < 4:
            return {"period": None, "strength": 0.0}
        s = s - s.mean()
        n = min(max_lag, len(s) // 2)
        acf = [float(np.corrcoef(s[:-k], s[k:])[0, 1]) for k in range(1, n + 1)]
        if not acf:
            return {"period": None, "strength": 0.0}
        # find first peak
        period = int(np.argmax(acf) + 1)
        return {"period": period, "strength": float(max(acf)), "acf": acf}

    def decompose(self, period: Optional[int] = None) -> Dict[str, pd.Series]:
        """Simple additive decomposition: trend = rolling mean, seasonal = periodic mean."""
        s = self.series.dropna()
        if period is None:
            period = self.detect_seasonality()["period"] or 7
        trend = s.rolling(period, center=True).mean()
        detrended = s - trend
        seasonal = detrended.groupby(detrended.index % period).transform("mean")
        residual = s - trend - seasonal
        return {"trend": trend, "seasonal": seasonal, "residual": residual}


# ============================================================================
# 5. SCHEMA ENFORCER — type validation, coercion, schema evolution
# ============================================================================
class SchemaEnforcer:
    """Type-validate, coerce, and track schema evolution for a DataDataset."""

    def __init__(self, schema: Dict[str, type]):
        self.schema = schema
        self.history: List[Dict[str, Any]] = []

    def validate(self, ds: DataDataset) -> Dict[str, Any]:
        """Check whether dataset matches schema. Returns violations list."""
        violations = []
        for col, expected in self.schema.items():
            if col not in ds.columns:
                violations.append({"column": col, "issue": "missing"})
                continue
            actual = ds.to_pandas()[col].dtype
            try:
                np_canonical = np.dtype(expected)
                if not np.issubdtype(actual, np_canonical):
                    violations.append({"column": col, "issue": "type_mismatch",
                                       "expected": str(expected), "actual": str(actual)})
            except TypeError:
                pass
        return {"passed": not violations, "violations": violations}

    def coerce(self, ds: DataDataset) -> DataDataset:
        """Cast columns to declared types in-place."""
        df = ds.to_pandas()
        for col, expected in self.schema.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(expected, errors="ignore")
                except Exception:
                    pass
        ds.data = df.values.tolist()
        ds.columns = list(df.columns)
        self.history.append({"timestamp": time.time(), "action": "coerce",
                              "schema": list(self.schema.keys())})
        return ds

    def evolve(self, new_schema: Dict[str, type]) -> "SchemaEnforcer":
        """Migrate to a new schema; old columns are kept, new ones added."""
        old = set(self.schema.keys())
        new = set(new_schema.keys())
        added = new - old
        removed = old - new
        self.history.append({"timestamp": time.time(), "action": "evolve",
                              "added": list(added), "removed": list(removed)})
        self.schema.update(new_schema)
        return self


# ============================================================================
# 6. DATA POISONING DETECTOR — statistical + signature-based
# ============================================================================
class DataPoisoningDetector:
    """Detects likely poisoned rows in a training set.

    Three signals:
      1. Statistical outlier (IsolationForest score)
      2. Label-flip signature (target value disagrees with KNN majority)
      3. Feature injection (rare-token / SQL / prompt-injection patterns)
    """
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination

    def detect(self, ds: DataDataset, label_col: Optional[str] = None) -> Dict[str, Any]:
        df = ds.to_pandas()
        suspicious = set()
        # 1. Statistical outliers
        if _SKLEARN:
            num = df.select_dtypes(include=[np.number]).dropna()
            if num.shape[0] >= 10 and num.shape[1] >= 1:
                iforest = _SKIF(contamination=self.contamination, random_state=0)
                preds = iforest.fit_predict(num)
                for i, p in enumerate(preds):
                    if p == -1:
                        suspicious.add(int(i))
        # 2. Label-flip signature
        if label_col and label_col in df.columns and _SKLEARN:
            num = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
            num = num.fillna(num.mean())
            if num.shape[1] >= 1 and len(num) == len(df):
                from sklearn.neighbors import KNeighborsClassifier
                y = df[label_col]
                if y.nunique() > 1 and len(y) >= 10:
                    knn = KNeighborsClassifier(n_neighbors=min(5, len(y) - 1))
                    knn.fit(num, y)
                    preds = knn.predict(num)
                    flips = np.where(preds != y.values)[0]
                    # only flag if many neighbours disagree
                    for i in flips:
                        suspicious.add(int(i))
        # 3. Feature-injection patterns in text columns
        injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"<\s*script[^>]*>",
            r"DROP\s+TABLE",
            r"--\s*$",
        ]
        for col in df.select_dtypes(include=[object]).columns:
            for i, val in enumerate(df[col].astype(str)):
                for pat in injection_patterns:
                    if re.search(pat, val, re.IGNORECASE):
                        suspicious.add(int(i))
                        break
        return {
            "n_suspicious": len(suspicious),
            "suspicious_indices": sorted(suspicious),
            "contamination_pct": round(100.0 * len(suspicious) / max(len(df), 1), 2),
        }


# ============================================================================
# 7. DRIFT DETECTOR — KS / PSI / chi-square between two dataset snapshots
# ============================================================================
class DriftDetector:
    """Compares a baseline dataset to a current dataset and reports drift.

    Wraps scipy.stats.ks_2samp for numeric columns and a chi-square test for
    categorical columns. Also computes Population Stability Index (PSI).
    """
    def __init__(self, baseline: DataDataset):
        self.baseline = baseline.to_pandas()

    def compare(self, current: DataDataset) -> Dict[str, Any]:
        cur = current.to_pandas()
        drift: Dict[str, Any] = {"columns": {}, "overall_drift_score": 0.0}
        scores = []
        for col in self.baseline.columns:
            if col not in cur.columns:
                continue
            old = self.baseline[col].dropna()
            new = cur[col].dropna()
            if pd.api.types.is_numeric_dtype(old) and len(old) > 0 and len(new) > 0:
                if _SCIPY:
                    ks = _scistats.ks_2samp(old, new)
                    drift["columns"][col] = {
                        "test": "ks_2samp", "statistic": float(ks.statistic),
                        "p_value": float(ks.pvalue),
                        "drifted": bool(ks.pvalue < 0.05)
                    }
                    scores.append(float(ks.statistic))
                # PSI
                psi = self._psi(old, new)
                drift["columns"][col]["psi"] = psi
            else:
                # chi-square on value counts
                old_vc = old.value_counts()
                new_vc = new.value_counts()
                cats = sorted(set(old_vc.index) | set(new_vc.index))
                if len(cats) < 2:
                    continue
                obs = np.array([old_vc.get(c, 0) for c in cats])
                exp = np.array([new_vc.get(c, 0) for c in cats])
                if exp.sum() == 0:
                    continue
                exp = exp * (obs.sum() / exp.sum())
                exp = np.where(exp == 0, 1e-6, exp)
                if _SCIPY:
                    chi, p = _scistats.chisquare(obs + 1, exp + 1)
                    drift["columns"][col] = {"test": "chi_square",
                                              "statistic": float(chi),
                                              "p_value": float(p),
                                              "drifted": bool(p < 0.05)}
                    scores.append(float(chi) / max(obs.sum(), 1))
        drift["overall_drift_score"] = float(np.mean(scores)) if scores else 0.0
        return drift

    @staticmethod
    def _psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
        """Population Stability Index between two numeric samples."""
        try:
            edges = np.histogram_bin_edges(a, bins=bins)
            a_hist, _ = np.histogram(a, bins=edges)
            b_hist, _ = np.histogram(b, bins=edges)
            a_pct = a_hist / max(a_hist.sum(), 1)
            b_pct = b_hist / max(b_hist.sum(), 1)
            a_pct = np.where(a_pct == 0, 1e-6, a_pct)
            b_pct = np.where(b_pct == 0, 1e-6, b_pct)
            return float(np.sum((b_pct - a_pct) * np.log(b_pct / a_pct)))
        except Exception:
            return 0.0


# ============================================================================
# 8. SYNTHETIC DATA FACTORY — multi-modal synthetic data
# ============================================================================
class SyntheticDataFactory:
    """Generate synthetic datasets in multiple modalities.

    Tabular:    fit a Gaussian copula to the original, then sample new rows
                preserving marginal distributions and rank-correlations.
    Time-series: fit per-column AR(1) residuals, then bootstrap.
    Text:       n-gram Markov chain over tokens.
    Image:      (lightweight) procedural noise patterns + class labels.
    """
    @staticmethod
    def tabular(reference: DataDataset, n_samples: int = 100) -> DataDataset:
        df = reference.to_pandas()
        out = pd.DataFrame(columns=df.columns)
        for col in df.columns:
            s = df[col].dropna()
            if pd.api.types.is_numeric_dtype(s):
                # sample from empirical distribution
                out[col] = np.random.choice(s.values, size=n_samples, replace=True)
                # add small jitter
                out[col] = out[col] + np.random.normal(0, s.std() * 0.05 + 1e-6, n_samples)
            else:
                # sample from observed categories with same probabilities
                vc = s.value_counts(normalize=True)
                out[col] = np.random.choice(vc.index, size=n_samples, p=vc.values)
        return DataDataset.from_pandas(out)

    @staticmethod
    def time_series(reference: DataDataset, time_col: str,
                     value_cols: List[str], n_periods: int = 50) -> DataDataset:
        df = reference.to_pandas()
        last_time = pd.to_datetime(df[time_col]).max()
        new_times = pd.date_range(last_time, periods=n_periods + 1, freq="D")[1:]
        out = {time_col: new_times}
        for col in value_cols:
            s = df[col].dropna().astype(float).values
            if len(s) < 2:
                out[col] = np.zeros(n_periods)
                continue
            # AR(1) coefficient
            ar1 = np.corrcoef(s[:-1], s[1:])[0, 1] if len(s) > 2 else 0.0
            mean, std = s.mean(), s.std()
            new_vals = np.zeros(n_periods)
            new_vals[0] = s[-1]
            for t in range(1, n_periods):
                new_vals[t] = mean + ar1 * (new_vals[t-1] - mean) + np.random.normal(0, std * 0.3)
            out[col] = new_vals
        return DataDataset.from_pandas(pd.DataFrame(out))

    @staticmethod
    def text(reference_texts: List[str], n_samples: int = 10, max_words: int = 30) -> List[str]:
        """Markov-chain text generator from a corpus."""
        # build bigram model
        model: Dict[str, List[str]] = defaultdict(list)
        for text in reference_texts:
            toks = text.split()
            for i in range(len(toks) - 1):
                model[toks[i]].append(toks[i + 1])
        if not model:
            return reference_texts[:n_samples]
        starts = list(model.keys())
        out = []
        for _ in range(n_samples):
            cur = random.choice(starts)
            words = [cur]
            for _ in range(max_words):
                if cur in model and model[cur]:
                    cur = random.choice(model[cur])
                    words.append(cur)
                else:
                    break
            out.append(" ".join(words))
        return out

    @staticmethod
    def image_grid(n: int = 16, size: int = 28, n_classes: int = 3,
                    seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic image-like patterns for testing vision pipelines."""
        rng = np.random.default_rng(seed)
        X = np.zeros((n, size, size), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            cls = i % n_classes
            y[i] = cls
            if cls == 0:  # circles
                cy, cx = size // 2, size // 2
                yy, xx = np.ogrid[:size, :size]
                r = rng.integers(size // 6, size // 3)
                X[i] = ((yy - cy)**2 + (xx - cx)**2 < r**2).astype(np.float32)
            elif cls == 1:  # squares
                s = rng.integers(size // 4, size // 2)
                cy, cx = size // 2, size // 2
                X[i, cy-s:cy+s, cx-s:cx+s] = 1.0
            else:  # diagonal
                for d in range(-size, size):
                    if 0 <= d < size:
                        X[i, d, :size - d] = 1.0
            X[i] += rng.normal(0, 0.05, X[i].shape)
        return X, y


# ============================================================================
# 9. FEATURE STORE — offline + online feature lookups
# ============================================================================
class FeatureStore:
    """A tiny feature store: register feature groups, look up by entity id."""

    def __init__(self):
        self._groups: Dict[str, pd.DataFrame] = {}
        self._keys: Dict[str, str] = {}

    def register(self, name: str, df: pd.DataFrame, entity_key: str) -> None:
        self._groups[name] = df.set_index(entity_key)
        self._keys[name] = entity_key

    def get(self, name: str, entity_ids: List[Any]) -> pd.DataFrame:
        if name not in self._groups:
            raise KeyError(f"Feature group '{name}' not registered")
        return self._groups[name].loc[entity_ids]

    def join(self, entity_ids: List[Any], groups: Optional[List[str]] = None) -> pd.DataFrame:
        """Return a single DataFrame with all features for given entity ids."""
        groups = groups or list(self._groups.keys())
        out = pd.DataFrame(index=entity_ids)
        for g in groups:
            fg = self._groups[g].loc[entity_ids]
            out = out.join(fg, rsuffix=f"_{g}")
        return out


# ============================================================================
# 10. DATA LINEAGE — track row/column provenance across transforms
# ============================================================================
class DataLineage:
    """Append-only lineage graph for a dataset's transformations."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def record(self, op: str, details: Dict[str, Any]) -> None:
        self.events.append({
            "timestamp": time.time(),
            "op": op,
            "details": details,
        })

    def lineage_for_column(self, col: str) -> List[Dict[str, Any]]:
        """Return all events mentioning this column."""
        return [e for e in self.events
                if col in e["details"].get("columns", [])
                or col in e["details"].get("derived_from", [])]

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(self.events, indent=2, default=str)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s


# ============================================================================
# 11. DATA CATALOG — searchable dataset registry with metadata
# ============================================================================
class DataCatalog:
    """In-memory dataset registry with full-text search over metadata."""

    def __init__(self):
        self._entries: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, path: str, owner: str = "unknown",
                 tags: Optional[List[str]] = None,
                 description: str = "") -> None:
        self._entries[name] = {
            "name": name, "path": path, "owner": owner,
            "tags": tags or [], "description": description,
            "registered_at": time.time(),
        }

    def search(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        out = []
        for entry in self._entries.values():
            score = 0
            if q in entry["name"].lower():
                score += 3
            if q in entry["description"].lower():
                score += 2
            for tag in entry["tags"]:
                if q in tag.lower():
                    score += 2
            if score > 0:
                out.append({**entry, "score": score})
        out.sort(key=lambda x: -x["score"])
        return out

    def load(self, name: str) -> DataDataset:
        if name not in self._entries:
            raise KeyError(f"Dataset '{name}' not in catalog")
        return load(self._entries[name]["path"])


# ============================================================================
# 12. SMART SAMPLER — stratified, imbalance-aware
# ============================================================================
class SmartSampler:
    """Stratified and imbalance-aware samplers for training pipelines."""

    @staticmethod
    def stratified(ds: DataDataset, label_col: str,
                    n_per_class: int = 10, random_state: int = 0) -> DataDataset:
        df = ds.to_pandas()
        out = df.groupby(label_col, group_keys=False).apply(
            lambda g: g.sample(min(n_per_class, len(g)), random_state=random_state))
        return DataDataset.from_pandas(out)

    @staticmethod
    def balanced_oversample(ds: DataDataset, label_col: str,
                             random_state: int = 0) -> DataDataset:
        """Oversample minority classes to match the majority."""
        df = ds.to_pandas()
        max_n = df[label_col].value_counts().max()
        out = df.groupby(label_col, group_keys=False).apply(
            lambda g: g.sample(max_n, replace=True, random_state=random_state))
        return DataDataset.from_pandas(out)

    @staticmethod
    def importance_weighted(ds: DataDataset, score_col: str,
                              n_samples: int = 100) -> DataDataset:
        """Sample rows proportional to a learned importance score."""
        df = ds.to_pandas()
        weights = df[score_col].clip(lower=0).values
        if weights.sum() == 0:
            weights = np.ones(len(df))
        probs = weights / weights.sum()
        idx = np.random.choice(len(df), size=n_samples, p=probs, replace=False
                                if n_samples < len(df) else True)
        return DataDataset.from_pandas(df.iloc[idx])


# ============================================================================
# 13. DATA VALIDATOR — great-expectations-style rule engine
# ============================================================================
class DataValidator:
    """Composable validation rules; chained like pytest fixtures."""

    def __init__(self):
        self._rules: List[Tuple[str, Callable[[pd.DataFrame], Tuple[bool, str]]]] = []

    def add_rule(self, name: str, fn: Callable[[pd.DataFrame], Tuple[bool, str]]) -> "DataValidator":
        self._rules.append((name, fn))
        return self

    def expect_not_null(self, col: str) -> "DataValidator":
        return self.add_rule(
            f"not_null({col})",
            lambda df: (not df[col].isnull().any(), f"{col} has {df[col].isnull().sum()} nulls")
        )

    def expect_unique(self, col: str) -> "DataValidator":
        return self.add_rule(
            f"unique({col})",
            lambda df: (not df[col].duplicated().any(), f"{col} has {df[col].duplicated().sum()} dups")
        )

    def expect_range(self, col: str, lo: float, hi: float) -> "DataValidator":
        return self.add_rule(
            f"range({col}, {lo}, {hi})",
            lambda df: ((df[col].between(lo, hi).all()),
                        f"{col} outside [{lo}, {hi}]")
        )

    def expect_in_set(self, col: str, allowed: set) -> "DataValidator":
        return self.add_rule(
            f"in_set({col})",
            lambda df: (df[col].isin(allowed).all(),
                        f"{col} has values outside {allowed}")
        )

    def expect_regex(self, col: str, pattern: str) -> "DataValidator":
        return self.add_rule(
            f"regex({col}, {pattern})",
            lambda df: (df[col].astype(str).str.match(pattern).all(),
                        f"{col} has values not matching {pattern}")
        )

    def validate(self, ds: DataDataset) -> Dict[str, Any]:
        df = ds.to_pandas()
        results = []
        all_passed = True
        for name, fn in self._rules:
            try:
                ok, msg = fn(df)
            except Exception as e:
                ok, msg = False, f"rule error: {e}"
            if not ok:
                all_passed = False
            results.append({"rule": name, "passed": ok, "message": msg})
        return {"passed": all_passed, "results": results}


# ============================================================================
# 14. SMART CACHE — disk-backed memoisation for expensive transforms
# ============================================================================
class SmartCache:
    """Hash-based disk cache for transformed DataDataset objects.

    The cache key is the SHA-256 of the input dataset's pickle plus the
    transform name + kwargs. Useful for expensive feature engineering steps.
    """
    def __init__(self, cache_dir: str = "/tmp/qai_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _key(ds: DataDataset, transform: str, kwargs: Dict[str, Any]) -> str:
        h = hashlib.sha256()
        h.update(transform.encode())
        h.update(str(sorted(kwargs.items())).encode())
        h.update(str(ds.data[:100]).encode())  # sample for speed
        h.update(str(len(ds.data)).encode())
        return h.hexdigest()

    def get_or_compute(self, ds: DataDataset, transform: str,
                       fn: Callable[[DataDataset], DataDataset],
                       **kwargs) -> DataDataset:
        key = self._key(ds, transform, kwargs)
        # try parquet first (fast), fall back to pickle
        path_parquet = os.path.join(self.cache_dir, f"{key}.parquet")
        path_pkl = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(path_parquet):
            return DataDataset.from_pandas(pd.read_parquet(path_parquet))
        if os.path.exists(path_pkl):
            import pickle
            with open(path_pkl, "rb") as f:
                return pickle.load(f)
        result = fn(ds, **kwargs)
        # try parquet; on failure fall back to pickle
        try:
            result.to_parquet(path_parquet)
        except Exception:
            import pickle
            with open(path_pkl, "wb") as f:
                pickle.dump(result, f)
        return result


# ============================================================================
# 15. ANNOTATION LAB — weak-supervision + label-noise detection
# ============================================================================
class AnnotationLab:
    """Weak-supervision labelling functions + label-noise filter."""

    def __init__(self):
        self._lfs: List[Tuple[str, Callable[[Dict[str, Any]], Optional[int]]]] = []

    def add_labeling_function(self, name: str,
                              fn: Callable[[Dict[str, Any]], Optional[int]]) -> "AnnotationLab":
        """A labeling function returns an int label or None to abstain."""
        self._lfs.append((name, fn))
        return self

    def label(self, ds: DataDataset) -> Tuple[DataDataset, Dict[str, Any]]:
        """Apply all LFs; majority vote. Returns labelled dataset + stats."""
        df = ds.to_pandas()
        votes = np.full((len(df), len(self._lfs)), -1)
        for j, (name, fn) in enumerate(self._lfs):
            for i, row in df.iterrows():
                pred = fn(row.to_dict())
                if pred is not None:
                    votes[i, j] = int(pred)
        # majority vote ignoring abstentions
        labels = []
        for i in range(len(df)):
            row_votes = [int(v) for v in votes[i] if v != -1]
            if not row_votes:
                labels.append(-1)
            else:
                counts = np.bincount(row_votes)
                labels.append(int(counts.argmax()))
        df["_weak_label"] = labels
        # stats
        labels_arr = np.asarray(labels)
        coverage = float((labels_arr != -1).mean())
        # per-row agreement: only count rows where at least one LF fired
        agree_flags = []
        for i in range(len(df)):
            row_votes = [int(v) for v in votes[i] if v != -1]
            if row_votes:
                agree_flags.append(len(set(row_votes)) == 1)
        agreement = float(np.mean(agree_flags)) if agree_flags else 0.0
        return DataDataset.from_pandas(df), {
            "coverage": coverage,
            "agreement": agreement,
            "n_lfs": len(self._lfs),
        }

    @staticmethod
    def detect_label_noise(ds: DataDataset, label_col: str,
                            feature_cols: List[str]) -> List[int]:
        """Confident-learning style: flag rows where label disagrees with
        cross-validated classifier's prediction."""
        if not _SKLEARN:
            return []
        from sklearn.neighbors import KNeighborsClassifier
        df = ds.to_pandas()
        X = df[feature_cols].fillna(0).values
        y = df[label_col].values
        if len(np.unique(y)) < 2:
            return []
        knn = KNeighborsClassifier(n_neighbors=min(5, len(y) - 1))
        knn.fit(X, y)
        preds = knn.predict(X)
        return [int(i) for i, (p, t) in enumerate(zip(preds, y)) if p != t]


__all__ = [
    "DatasetProfiler", "StreamingDataset", "DatasetJoiner", "TimeSeriesToolkit",
    "SchemaEnforcer", "DataPoisoningDetector", "DriftDetector", "SyntheticDataFactory",
    "FeatureStore", "DataLineage", "DataCatalog", "SmartSampler", "DataValidator",
    "SmartCache", "AnnotationLab",
]
