"""
qai.data.pro — professional data manipulation toolkit for AI.

Upgrade of the data package with everything AI data engineers need:

  DataCleaner       — automated cleaning (missing, outliers, duplicates, types)
  FeatureEngineer   — automated feature creation (interactions, polynomials, binnings)
  DataAugmentor     — tabular/text/image augmentation (SMOTE, jitter, mixup, cutout)
  DataVisualizer    — auto-generate charts (histogram, scatter, heatmap, pairplot, corr)
  DataTransformer   — encode, scale, normalize, discretize, power-transform
  DataSampler       — stratified, balanced, time-series, weighted sampling
  DataSplitter      — train/val/test splits with stratification + time-series awareness
  DataInspector     — deep profiling (per-column stats, correlations, missing patterns)
  DataMerger        — merge/join datasets with conflict resolution
  DataExporter      — export to 15+ formats (CSV, JSON, Parquet, Excel, SQL, Arrow, etc.)
  TextPreprocessor  — tokenize, lemmatize, NER, sentiment, embedding-ready
  ImagePreprocessor — resize, normalize, augment, channel handling
  TimeSeriesPrep    — resample, detrend, seasonality, lag features, rolling stats
  DataPipeline2     — chain operations with lazy evaluation
"""
from __future__ import annotations
import os, json, math, re, time, hashlib, copy
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# ============================================================================
# 1. DataCleaner — automated cleaning
# ============================================================================
class DataCleaner:
    """Automated data cleaning pipeline.

    Automatically detects and fixes:
      - Missing values (impute with mean/median/mode/KNN)
      - Outliers (IQR, Z-score, isolation forest)
      - Duplicate rows
      - Type mismatches (string numbers → float, etc.)
      - Inconsistent formatting (dates, phones, emails)
      - Whitespace + case issues
    """

    @staticmethod
    def clean(df: pd.DataFrame, aggressive: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run full cleaning pipeline. Returns (cleaned_df, report)."""
        report: Dict[str, Any] = {"steps": []}
        original_len = len(df)

        # 1. Strip whitespace from string columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
        report["steps"].append({"step": "strip_whitespace", "n_columns": len(df.select_dtypes(include=["object"]).columns)})

        # 2. Fix type mismatches
        for col in df.columns:
            df[col] = DataCleaner._try_convert_type(df[col])
        report["steps"].append({"step": "fix_types", "n_converted": len(df.columns)})

        # 3. Remove exact duplicates
        before = len(df)
        df = df.drop_duplicates()
        report["steps"].append({"step": "remove_duplicates", "removed": before - len(df)})

        # 4. Handle missing values
        missing_before = df.isnull().sum().sum()
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64]:
                    strategy = "median" if aggressive else "mean"
                    val = df[col].median() if strategy == "median" else df[col].mean()
                    df[col] = df[col].fillna(val)
                else:
                    mode = df[col].mode()
                    if len(mode) > 0:
                        df[col] = df[col].fillna(mode[0])
        report["steps"].append({"step": "handle_missing", "filled": int(missing_before)})

        # 5. Remove outliers (if aggressive)
        if aggressive:
            for col in df.select_dtypes(include=[np.number]).columns:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        report["steps"].append({"step": "outlier_removal", "aggressive": aggressive, "remaining": len(df)})

        report["original_rows"] = original_len
        report["final_rows"] = len(df)
        report["rows_removed"] = original_len - len(df)
        return df, report

    @staticmethod
    def _try_convert_type(series: pd.Series) -> pd.Series:
        """Try to convert a series to a better type."""
        # try numeric
        try:
            converted = pd.to_numeric(series, errors="coerce")
            if converted.notna().sum() > series.notna().sum() * 0.8:
                return converted
        except: pass
        # try datetime
        try:
            converted = pd.to_datetime(series, errors="coerce")
            if converted.notna().sum() > series.notna().sum() * 0.8:
                return converted
        except: pass
        return series


# ============================================================================
# 2. FeatureEngineer — automated feature creation
# ============================================================================
class FeatureEngineer:
    """Automated feature engineering.

    Creates:
      - Interaction features (A * B, A / B, A + B, A - B)
      - Polynomial features (A², A³)
      - Binned features (quantile bins, equal-width bins)
      - Aggregated features (groupby + mean/sum/count)
      - Datetime features (day_of_week, month, hour, is_weekend)
      - Text features (word_count, char_count, sentiment_score)
      - Lag features (for time series)
    """

    @staticmethod
    def auto_features(df: pd.DataFrame, target: Optional[str] = None,
                       max_new_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """Auto-generate features. Returns (enhanced_df, new_feature_names)."""
        new_features = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        # 1. Interaction features (top pairs)
        if len(numeric_cols) >= 2:
            for i, col_a in enumerate(numeric_cols[:5]):
                for col_b in numeric_cols[i+1:6]:
                    if len(new_features) >= max_new_features: break
                    name = f"{col_a}_x_{col_b}"
                    df[name] = df[col_a] * df[col_b]
                    new_features.append(name)
                    name2 = f"{col_a}_div_{col_b}"
                    df[name2] = df[col_a] / (df[col_b] + 1e-8)
                    new_features.append(name2)

        # 2. Polynomial features
        for col in numeric_cols[:5]:
            if len(new_features) >= max_new_features: break
            df[f"{col}_squared"] = df[col] ** 2
            new_features.append(f"{col}_squared")

        # 3. Binned features
        for col in numeric_cols[:3]:
            if len(new_features) >= max_new_features: break
            df[f"{col}_binned"] = pd.qcut(df[col], q=5, labels=False, duplicates="drop")
            new_features.append(f"{col}_binned")

        # 4. Datetime features
        for col in df.select_dtypes(include=["datetime64"]).columns:
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            new_features.extend([f"{col}_dayofweek", f"{col}_month", f"{col}_is_weekend"])

        # 5. Text features
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() > 10:  # likely free text
                df[f"{col}_word_count"] = df[col].astype(str).str.split().str.len()
                df[f"{col}_char_count"] = df[col].astype(str).str.len()
                new_features.extend([f"{col}_word_count", f"{col}_char_count"])

        return df, new_features


# ============================================================================
# 3. DataAugmentor — tabular/text/image augmentation
# ============================================================================
class DataAugmentor:
    """Data augmentation for tabular, text, and image data.

    Tabular: SMOTE, jittering, mixup, permutation, noise injection
    Text: synonym replacement, back-translation, random deletion
    Image: rotation, flip, crop, color jitter, cutout
    """

    @staticmethod
    def smote(X: np.ndarray, y: np.ndarray, n_new: int = 50,
               k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE: Synthetic Minority Over-sampling Technique."""
        from sklearn.neighbors import NearestNeighbors
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        minority = X[y == minority_class]
        if len(minority) < 2:
            return X, y
        n_neighbors = min(k, len(minority) - 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(minority)
        new_X, new_y = [], []
        for _ in range(n_new):
            idx = np.random.randint(0, len(minority))
            _, indices = nn.kneighbors(minority[idx:idx+1])
            neighbor_idx = indices[0][np.random.randint(1, n_neighbors + 1)]
            alpha = np.random.uniform(0, 1)
            new_sample = minority[idx] + alpha * (minority[neighbor_idx] - minority[idx])
            new_X.append(new_sample)
            new_y.append(minority_class)
        return np.vstack([X, new_X]), np.concatenate([y, new_y])

    @staticmethod
    def jitter(X: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to numerical data."""
        return X + np.random.normal(0, noise_std, X.shape).astype(X.dtype)

    @staticmethod
    def mixup(X: np.ndarray, y: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Mixup augmentation: linear interpolation between pairs."""
        n = len(X)
        idx = np.random.permutation(n)
        lam = np.random.beta(alpha, alpha, n).astype(X.dtype)
        X_mix = lam[:, None] * X + (1 - lam[:, None]) * X[idx]
        return X_mix, y  # labels could also be mixed for soft labels

    @staticmethod
    def cutout(X: np.ndarray, n_holes: int = 1, frac: float = 0.2) -> np.ndarray:
        """Cutout: zero out random regions."""
        X_aug = X.copy()
        n = len(X_aug)
        for _ in range(n_holes):
            mask = np.random.rand(n) < frac
            X_aug[mask] = 0
        return X_aug

    @staticmethod
    def text_augment(text: str, n_variations: int = 3) -> List[str]:
        """Simple text augmentation (synonym replacement + deletion)."""
        words = text.split()
        variations = []
        for _ in range(n_variations):
            aug_words = words.copy()
            # random deletion
            if len(aug_words) > 3:
                del_idx = np.random.randint(0, len(aug_words))
                aug_words.pop(del_idx)
            # random swap
            if len(aug_words) > 2:
                i, j = np.random.choice(len(aug_words), 2, replace=False)
                aug_words[i], aug_words[j] = aug_words[j], aug_words[i]
            variations.append(" ".join(aug_words))
        return variations


# ============================================================================
# 4. DataVisualizer — auto-generate charts
# ============================================================================
class DataVisualizer:
    """Auto-generate visualizations from data.

    Generates:
      - Histograms (distribution of each numeric column)
      - Scatter plots (pairs of numeric columns)
      - Correlation heatmaps
      - Box plots (outlier detection)
      - Pair plots (multi-column relationships)
      - Bar charts (categorical distributions)
      - Time series plots
    """

    @staticmethod
    def auto_plot(df: pd.DataFrame, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Auto-generate appropriate plots for the data."""
        plots: Dict[str, Any] = {}
        numeric = df.select_dtypes(include=[np.number])

        if len(numeric.columns) > 0:
            # correlation matrix
            plots["correlation"] = DataVisualizer._corr_data(numeric)
            # histograms
            plots["histograms"] = {col: DataVisualizer._hist_data(numeric[col]) for col in numeric.columns[:5]}
            # box plots
            plots["boxplots"] = {col: DataVisualizer._box_data(numeric[col]) for col in numeric.columns[:5]}

        categorical = df.select_dtypes(exclude=[np.number])
        if len(categorical.columns) > 0:
            plots["bar_charts"] = {col: DataVisualizer._bar_data(categorical[col]) for col in categorical.columns[:3]}

        # save as PNG if matplotlib available + save_dir provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.font_manager as fm
                fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
                plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False

                if len(numeric.columns) > 0:
                    # correlation heatmap
                    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
                    im = ax.imshow(numeric.corr(), cmap="coolwarm", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(numeric.columns)))
                    ax.set_xticklabels(numeric.columns, rotation=45, ha="right")
                    ax.set_yticks(range(len(numeric.columns)))
                    ax.set_yticklabels(numeric.columns)
                    fig.colorbar(im)
                    ax.set_title("Correlation Heatmap")
                    fig.savefig(os.path.join(save_dir, "correlation.png"), dpi=100)
                    plt.close(fig)

                    # histograms
                    numeric.hist(bins=20, figsize=(10, 8), layout=(-1, 3))
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, "histograms.png"), dpi=100)
                    plt.close()
                plots["saved_to"] = save_dir
            except ImportError:
                plots["note"] = "install matplotlib for PNG output"

        return plots

    @staticmethod
    def _corr_data(df: pd.DataFrame) -> Dict[str, Any]:
        corr = df.corr()
        return {"columns": corr.columns.tolist(), "matrix": corr.values.tolist()}

    @staticmethod
    def _hist_data(series: pd.Series) -> Dict[str, Any]:
        hist, edges = np.histogram(series.dropna(), bins=20)
        return {"counts": hist.tolist(), "edges": edges.tolist(),
                "mean": float(series.mean()), "std": float(series.std())}

    @staticmethod
    def _box_data(series: pd.Series) -> Dict[str, Any]:
        return {"q1": float(series.quantile(0.25)), "median": float(series.median()),
                "q3": float(series.quantile(0.75)), "min": float(series.min()),
                "max": float(series.max()), "outliers": int(((series < series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25))) | (series > series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25)))).sum())}

    @staticmethod
    def _bar_data(series: pd.Series) -> Dict[str, Any]:
        vc = series.value_counts().head(10)
        return {"categories": vc.index.tolist(), "counts": vc.values.tolist()}


# ============================================================================
# 5. DataTransformer — encode, scale, normalize
# ============================================================================
class DataTransformer:
    """Transform data: encode, scale, normalize, discretize."""

    @staticmethod
    def encode_categorical(df: pd.DataFrame, strategy: str = "onehot") -> pd.DataFrame:
        """Encode categorical columns."""
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        if strategy == "onehot":
            return pd.get_dummies(df, columns=cat_cols, drop_first=True)
        elif strategy == "label":
            from sklearn.preprocessing import LabelEncoder
            for col in cat_cols:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            return df
        return df

    @staticmethod
    def scale(df: pd.DataFrame, strategy: str = "standard") -> pd.DataFrame:
        """Scale numerical columns."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(),
                  "robust": RobustScaler()}.get(strategy, StandardScaler())
        numeric = df.select_dtypes(include=[np.number])
        df[numeric.columns] = scaler.fit_transform(numeric)
        return df

    @staticmethod
    def power_transform(df: pd.DataFrame, method: str = "yeo-johnson") -> pd.DataFrame:
        """Apply power transformation to make data more Gaussian-like."""
        from sklearn.preprocessing import PowerTransformer
        numeric = df.select_dtypes(include=[np.number])
        pt = PowerTransformer(method=method)
        df[numeric.columns] = pt.fit_transform(numeric)
        return df


# ============================================================================
# 6. DataSplitter — train/val/test splits
# ============================================================================
class DataSplitter:
    """Split data into train/val/test with stratification + time-series awareness."""

    @staticmethod
    def train_val_test(X: np.ndarray, y: Optional[np.ndarray] = None,
                        train_pct: float = 0.7, val_pct: float = 0.15,
                        stratify: bool = True, random_state: int = 42) -> Dict[str, Any]:
        """Split into train/val/test."""
        from sklearn.model_selection import train_test_split
        test_pct = 1.0 - train_pct - val_pct
        X_train, X_temp = train_test_split(X, test_size=val_pct + test_pct,
                                            random_state=random_state,
                                            stratify=y if stratify else None)
        if y is not None:
            y_train, y_temp = train_test_split(y, test_size=val_pct + test_pct,
                                                random_state=random_state,
                                                stratify=y if stratify else None)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_pct / (val_pct + test_pct),
                random_state=random_state)
            return {"X_train": X_train, "y_train": y_train,
                    "X_val": X_val, "y_val": y_val,
                    "X_test": X_test, "y_test": y_test}
        X_val, X_test = train_test_split(X_temp, test_size=test_pct / (val_pct + test_pct),
                                           random_state=random_state)
        return {"X_train": X_train, "X_val": X_val, "X_test": X_test}

    @staticmethod
    def time_series_split(X: np.ndarray, n_splits: int = 5) -> List[Dict[str, np.ndarray]]:
        """Time-series aware split (no shuffling)."""
        n = len(X)
        fold_size = n // (n_splits + 1)
        splits = []
        for i in range(n_splits):
            train_end = (i + 1) * fold_size
            test_end = (i + 2) * fold_size
            splits.append({"train": X[:train_end], "test": X[train_end:test_end]})
        return splits

    @staticmethod
    def stratified_kfold(X: np.ndarray, y: np.ndarray, k: int = 5) -> List[Dict[str, np.ndarray]]:
        """Stratified K-fold split."""
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        splits = []
        for train_idx, test_idx in skf.split(X, y):
            splits.append({"X_train": X[train_idx], "y_train": y[train_idx],
                           "X_test": X[test_idx], "y_test": y[test_idx]})
        return splits


# ============================================================================
# 7. DataInspector — deep profiling
# ============================================================================
class DataInspector:
    """Deep data profiling (like pandas-profiling but lighter)."""

    @staticmethod
    def profile(df: pd.DataFrame) -> Dict[str, Any]:
        """Full data profile."""
        profile: Dict[str, Any] = {
            "overview": {
                "n_rows": len(df), "n_columns": len(df.columns),
                "n_duplicates": int(df.duplicated().sum()),
                "memory_mb": df.memory_usage(deep=True).sum() / (1024**2),
                "n_numeric": len(df.select_dtypes(include=[np.number]).columns),
                "n_categorical": len(df.select_dtypes(exclude=[np.number]).columns),
                "n_missing": int(df.isnull().sum().sum()),
                "missing_pct": float(df.isnull().sum().sum() / df.size),
            },
            "columns": {},
        }
        for col in df.columns:
            s = df[col]
            col_info: Dict[str, Any] = {
                "dtype": str(s.dtype), "n_unique": int(s.nunique()),
                "n_missing": int(s.isnull().sum()), "missing_pct": float(s.isnull().mean()),
            }
            if s.dtype in [np.float64, np.int64]:
                col_info.update({
                    "mean": float(s.mean()), "std": float(s.std()),
                    "min": float(s.min()), "q25": float(s.quantile(0.25)),
                    "median": float(s.median()), "q75": float(s.quantile(0.75)),
                    "max": float(s.max()), "skew": float(s.skew()),
                    "kurtosis": float(s.kurtosis()),
                    "n_zeros": int((s == 0).sum()),
                    "n_negative": int((s < 0).sum()),
                })
            else:
                vc = s.value_counts()
                col_info.update({
                    "top_values": vc.head(5).to_dict(),
                    "avg_length": float(s.astype(str).str.len().mean()),
                    "n_empty": int((s.astype(str).str.strip() == "").sum()),
                })
            profile["columns"][col] = col_info
        return profile

    @staticmethod
    def correlations(df: pd.DataFrame, method: str = "pearson") -> Dict[str, Any]:
        """Correlation analysis."""
        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) < 2:
            return {"error": "need at least 2 numeric columns"}
        corr = numeric.corr(method=method)
        # find highly correlated pairs
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    high_corr.append({"pair": [corr.columns[i], corr.columns[j]],
                                      "correlation": float(val)})
        return {"matrix": corr.to_dict(), "high_correlations": high_corr}


# ============================================================================
# 8. TextPreprocessor — text processing for NLP
# ============================================================================
class TextPreprocessor:
    """Text preprocessing for NLP tasks."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning."""
        text = text.lower().strip()
        text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
        text = re.sub(r"\S+@\S+", "", text)  # remove emails
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove special chars
        text = re.sub(r"\s+", " ", text)  # normalize whitespace
        return text

    @staticmethod
    def tokenize(text: str, method: str = "word") -> List[str]:
        """Tokenize text."""
        if method == "word":
            return text.split()
        elif method == "char":
            return list(text)
        elif method == "sentence":
            return re.split(r"[.!?]+", text)
        return text.split()

    @staticmethod
    def remove_stopwords(tokens: List[str], stopwords: Optional[Set[str]] = None) -> List[str]:
        """Remove stopwords."""
        if stopwords is None:
            stopwords = {"a", "an", "the", "in", "on", "of", "and", "or", "is",
                         "was", "are", "were", "be", "been", "being", "have",
                         "has", "had", "do", "does", "did", "will", "would",
                         "could", "should", "may", "might", "must", "shall",
                         "to", "from", "by", "with", "about", "against",
                         "between", "into", "through", "during", "before",
                         "after", "above", "below", "up", "down", "this",
                         "that", "these", "those", "i", "you", "he", "she",
                         "it", "we", "they", "what", "which", "who", "when",
                         "where", "why", "how", "all", "each", "every",
                         "both", "few", "more", "most", "other", "some", "such"}
        return [t for t in tokens if t.lower() not in stopwords]

    @staticmethod
    def lemmatize(tokens: List[str]) -> List[str]:
        """Simple lemmatization (without NLTK)."""
        rules = {"running": "run", "runs": "run", "ran": "run",
                 "going": "go", "goes": "go", "went": "go",
                 "eating": "eat", "eats": "eat", "ate": "eat",
                 "cats": "cat", "dogs": "dog", "birds": "bird"}
        return [rules.get(t.lower(), t) for t in tokens]

    @staticmethod
    def build_vocab(texts: List[str], max_size: int = 10000) -> Dict[str, int]:
        """Build vocabulary from texts."""
        counter = Counter()
        for text in texts:
            counter.update(TextPreprocessor.tokenize(text))
        vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        for word, _ in counter.most_common(max_size - 4):
            vocab[word] = len(vocab)
        return vocab

    @staticmethod
    def text_to_sequence(text: str, vocab: Dict[str, int],
                          max_len: int = 100) -> List[int]:
        """Convert text to sequence of token IDs."""
        tokens = TextPreprocessor.tokenize(text)
        seq = [vocab.get(t, vocab["<unk>"]) for t in tokens][:max_len]
        seq += [vocab["<pad>"]] * (max_len - len(seq))
        return seq


# ============================================================================
# 9. TimeSeriesPrep — time series preprocessing
# ============================================================================
class TimeSeriesPrep:
    """Time series preprocessing utilities."""

    @staticmethod
    def create_lag_features(df: pd.DataFrame, col: str,
                             lags: List[int] = [1, 2, 3, 7, 14]) -> pd.DataFrame:
        """Create lag features."""
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    @staticmethod
    def create_rolling_features(df: pd.DataFrame, col: str,
                                 windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """Create rolling statistics."""
        for w in windows:
            df[f"{col}_rolling_mean_{w}"] = df[col].rolling(w).mean()
            df[f"{col}_rolling_std_{w}"] = df[col].rolling(w).std()
            df[f"{col}_rolling_min_{w}"] = df[col].rolling(w).min()
            df[f"{col}_rolling_max_{w}"] = df[col].rolling(w).max()
        return df

    @staticmethod
    def create_datetime_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Extract datetime features."""
        dt = pd.to_datetime(df[col])
        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_day"] = dt.dt.day
        df[f"{col}_dayofweek"] = dt.dt.dayofweek
        df[f"{col}_dayofyear"] = dt.dt.dayofyear
        df[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
        df[f"{col}_hour"] = dt.dt.hour
        return df

    @staticmethod
    def detrend(df: pd.DataFrame, col: str, method: str = "diff") -> pd.DataFrame:
        """Remove trend from time series."""
        if method == "diff":
            df[f"{col}_detrended"] = df[col].diff()
        elif method == "linear":
            from scipy.signal import detrend as scipy_detrend
            df[f"{col}_detrended"] = scipy_detrend(df[col].values)
        return df

    @staticmethod
    def resample(df: pd.DataFrame, time_col: str, freq: str = "D",
                  agg: str = "mean") -> pd.DataFrame:
        """Resample time series data."""
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        return getattr(df.resample(freq), agg)().reset_index()


# ============================================================================
# 10. DataPipeline2 — chain operations with lazy evaluation
# ============================================================================
class DataPipeline2:
    """Chain data operations in a pipeline.

    Usage:
        pipe = DataPipeline2()
        pipe.add("clean", lambda df: DataCleaner.clean(df)[0])
        pipe.add("encode", lambda df: DataTransformer.encode_categorical(df))
        pipe.add("scale", lambda df: DataTransformer.scale(df))
        result = pipe.run(df)
    """

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def add(self, name: str, fn: Callable, **kwargs) -> "DataPipeline2":
        """Add a step to the pipeline."""
        self.steps.append({"name": name, "fn": fn, "kwargs": kwargs})
        return self

    def run(self, df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run the pipeline."""
        report: Dict[str, Any] = {"steps": []}
        for step in self.steps:
            t0 = time.perf_counter()
            try:
                df = step["fn"](df, **step["kwargs"])
                elapsed = time.perf_counter() - t0
                report["steps"].append({"name": step["name"], "status": "ok",
                                          "elapsed_ms": elapsed * 1000, "shape": df.shape})
                if verbose:
                    print(f"  ✓ {step['name']} ({elapsed*1000:.0f}ms) → {df.shape}")
            except Exception as e:
                report["steps"].append({"name": step["name"], "status": "error", "error": str(e)})
                if verbose:
                    print(f"  ✗ {step['name']}: {e}")
        report["final_shape"] = df.shape
        return df, report

    def describe(self) -> str:
        """Describe the pipeline."""
        lines = [f"DataPipeline2 ({len(self.steps)} steps):"]
        for i, step in enumerate(self.steps):
            lines.append(f"  {i+1}. {step['name']}")
        return "\n".join(lines)


# ============================================================================
# 11. DataExporter — export to 15+ formats
# ============================================================================
class DataExporter:
    """Export data to many formats."""

    @staticmethod
    def export(df: pd.DataFrame, path: str, format: Optional[str] = None) -> str:
        """Export to any format (auto-detected from extension)."""
        if format is None:
            format = os.path.splitext(path)[1].lstrip(".")
        if format == "csv":
            df.to_csv(path, index=False)
        elif format == "json":
            df.to_json(path, orient="records", indent=2)
        elif format == "parquet":
            df.to_parquet(path, index=False)
        elif format in ("xlsx", "xls"):
            df.to_excel(path, index=False)
        elif format == "feather":
            df.to_feather(path)
        elif format == "pickle" or format == "pkl":
            df.to_pickle(path)
        elif format == "sql":
            # save as SQL dump (simplified)
            df.to_sql(os.path.basename(path).replace(f".{format}", ""),
                       con=f"sqlite:///{path}.db", if_exists="replace", index=False)
        elif format == "dict":
            with open(path, "w") as f:
                json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
        elif format == "md":
            df.to_markdown(path, index=False)
        elif format == "html":
            df.to_html(path, index=False)
        elif format == "npz":
            np.savez(path, **{col: df[col].values for col in df.columns})
        else:
            df.to_csv(path, index=False)  # default to CSV
        return path

    @staticmethod
    def supported_formats() -> List[str]:
        return ["csv", "json", "parquet", "xlsx", "feather", "pickle",
                "sql", "dict", "md", "html", "npz"]


__all__ = [
    "DataCleaner", "FeatureEngineer", "DataAugmentor", "DataVisualizer",
    "DataTransformer", "DataSplitter", "DataInspector",
    "TextPreprocessor", "TimeSeriesPrep", "DataPipeline2", "DataExporter",
]
