"""
qai.data.viewer — interactive data viewer for any data type.

  DataViewer       — view any data as a formatted table, summary, or plot
  DataExplorer     — interactive exploration (filter, sort, search, pivot)
  DataComparer     — side-by-side comparison of two datasets
  DataStoryteller  — auto-generate narrative reports from data
"""
from __future__ import annotations
import os, json, math, time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataViewer:
    """View any data in multiple formats.

    Formats:
      - "table"    : formatted text table (first N rows)
      - "summary"  : statistical summary (like df.describe())
      - "schema"   : column types + null counts
      - "sample"   : random sample of rows
      - "head"     : first N rows
      - "tail"     : last N rows
      - "info"     : concise info (like df.info())
      - "full"     : everything (summary + schema + sample + head)
    """

    @staticmethod
    def view(data: Any, format: str = "table", n: int = 10) -> str:
        """View data in the specified format. Returns a formatted string."""
        if isinstance(data, pd.DataFrame):
            return DataViewer._view_dataframe(data, format, n)
        elif isinstance(data, np.ndarray):
            return DataViewer._view_array(data, format, n)
        elif isinstance(data, str):
            return DataViewer._view_text(data, format, n)
        elif isinstance(data, list):
            return DataViewer._view_list(data, format, n)
        elif isinstance(data, dict):
            return DataViewer._view_dict(data, format, n)
        else:
            return str(data)

    @staticmethod
    def _view_dataframe(df: pd.DataFrame, format: str, n: int) -> str:
        if format == "table" or format == "head":
            return df.head(n).to_string()
        elif format == "tail":
            return df.tail(n).to_string()
        elif format == "summary":
            return df.describe(include="all").to_string()
        elif format == "schema":
            lines = ["Schema:"]
            for col in df.columns:
                lines.append(f"  {col}: {df[col].dtype} | "
                           f"nulls={df[col].isnull().sum()} | "
                           f"unique={df[col].nunique()}")
            return "\n".join(lines)
        elif format == "sample":
            return df.sample(min(n, len(df))).to_string()
        elif format == "info":
            lines = [f"Rows: {len(df)}", f"Columns: {len(df.columns)}",
                     f"Memory: {df.memory_usage(deep=True).sum() / 1024:.1f}KB",
                     f"Duplicates: {df.duplicated().sum()}"]
            return "\n".join(lines)
        elif format == "full":
            parts = []
            parts.append("=== INFO ===")
            parts.append(DataViewer._view_dataframe(df, "info", n))
            parts.append("\n=== SCHEMA ===")
            parts.append(DataViewer._view_dataframe(df, "schema", n))
            parts.append("\n=== SUMMARY ===")
            parts.append(DataViewer._view_dataframe(df, "summary", n))
            parts.append(f"\n=== HEAD ({n} rows) ===")
            parts.append(DataViewer._view_dataframe(df, "head", n))
            return "\n".join(parts)
        return df.head(n).to_string()

    @staticmethod
    def _view_array(arr: np.ndarray, format: str, n: int) -> str:
        if format == "table" or format == "head":
            return f"Array shape: {arr.shape}\n{arr[:n]}"
        elif format == "summary":
            return f"Shape: {arr.shape}\nDtype: {arr.dtype}\nMin: {arr.min()}\nMax: {arr.max()}\nMean: {arr.mean()}\nStd: {arr.std()}"
        elif format == "info":
            return f"Shape: {arr.shape}\nDtype: {arr.dtype}\nSize: {arr.size}\nMemory: {arr.nbytes / 1024:.1f}KB"
        elif format == "full":
            return DataViewer._view_array(arr, "summary", n) + "\n\n" + DataViewer._view_array(arr, "head", n)
        return str(arr[:n])

    @staticmethod
    def _view_text(text: str, format: str, n: int) -> str:
        words = text.split()
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if format == "table" or format == "head":
            return text[:500] + ("..." if len(text) > 500 else "")
        elif format == "summary":
            return f"Chars: {len(text)}\nWords: {len(words)}\nSentences: {len(sentences)}\nUnique words: {len(set(words))}\nAvg word length: {np.mean([len(w) for w in words]):.1f}"
        elif format == "info":
            return f"Type: text\nLength: {len(text)} chars\nWords: {len(words)}"
        return text[:500]

    @staticmethod
    def _view_list(lst: list, format: str, n: int) -> str:
        if format == "table" or format == "head":
            return "\n".join(str(x) for x in lst[:n])
        elif format == "summary":
            return f"Length: {len(lst)}\nTypes: {set(type(x).__name__ for x in lst[:100])}"
        return "\n".join(str(x) for x in lst[:n])

    @staticmethod
    def _view_dict(d: dict, format: str, n: int) -> str:
        if format == "table" or format == "head":
            return "\n".join(f"{k}: {v}" for k, v in list(d.items())[:n])
        elif format == "summary":
            return f"Keys: {len(d)}\nKey types: {set(type(k).__name__ for k in d.keys())}\nValue types: {set(type(v).__name__ for v in list(d.values())[:100])}"
        return "\n".join(f"{k}: {v}" for k, v in list(d.items())[:n])


class DataExplorer:
    """Interactive data exploration without a UI.

    Operations:
      - filter(df, condition)  — keep rows matching condition
      - sort(df, by, asc)      — sort by column
      - search(df, query)      — full-text search across all cells
      - pivot(df, rows, cols, vals) — pivot table
      - group_by(df, by, agg)  — group + aggregate
      - unique(df, col)        — unique values in a column
      - top_n(df, col, n)      — top N rows by column value
    """

    @staticmethod
    def filter(df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """Filter rows using a pandas query string."""
        try:
            return df.query(condition)
        except Exception as e:
            return df[df.apply(lambda row: condition in str(row.values), axis=1)]

    @staticmethod
    def sort(df: pd.DataFrame, by: str, ascending: bool = True) -> pd.DataFrame:
        return df.sort_values(by=by, ascending=ascending)

    @staticmethod
    def search(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search across all cells for a query string."""
        mask = df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)
        return df[mask]

    @staticmethod
    def pivot(df: pd.DataFrame, rows: str, cols: str,
              vals: str, agg: str = "mean") -> pd.DataFrame:
        """Create a pivot table."""
        return df.pivot_table(index=rows, columns=cols, values=vals, aggfunc=agg)

    @staticmethod
    def group_by(df: pd.DataFrame, by: Union[str, List[str]],
                  agg: str = "mean") -> pd.DataFrame:
        """Group + aggregate."""
        return df.groupby(by).agg(agg)

    @staticmethod
    def unique(df: pd.DataFrame, col: str) -> List[Any]:
        return df[col].unique().tolist()

    @staticmethod
    def top_n(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
        return df.nlargest(n, col)

    @staticmethod
    def cross_tab(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
        """Cross-tabulation of two columns."""
        return pd.crosstab(df[col_a], df[col_b])


class DataComparer:
    """Compare two datasets side by side.

    Reports:
      - Row count difference
      - Column differences (added, removed, renamed)
      - Schema changes (type changes)
      - Value differences (statistical shifts per column)
      - Overlap (how many rows are shared)
    """

    @staticmethod
    def compare(df_a: pd.DataFrame, df_b: pd.DataFrame,
                 key: Optional[str] = None) -> Dict[str, Any]:
        """Compare two DataFrames."""
        result: Dict[str, Any] = {}

        # row counts
        result["rows_a"] = len(df_a)
        result["rows_b"] = len(df_b)
        result["row_diff"] = len(df_b) - len(df_a)

        # columns
        cols_a, cols_b = set(df_a.columns), set(df_b.columns)
        result["columns_added"] = list(cols_b - cols_a)
        result["columns_removed"] = list(cols_a - cols_b)
        result["columns_common"] = list(cols_a & cols_b)

        # schema changes
        schema_changes = []
        for col in cols_a & cols_b:
            if df_a[col].dtype != df_b[col].dtype:
                schema_changes.append({
                    "column": col,
                    "type_a": str(df_a[col].dtype),
                    "type_b": str(df_b[col].dtype),
                })
        result["schema_changes"] = schema_changes

        # value statistics
        value_stats = {}
        for col in result["columns_common"]:
            if df_a[col].dtype in [np.float64, np.int64]:
                value_stats[col] = {
                    "mean_a": float(df_a[col].mean()),
                    "mean_b": float(df_b[col].mean()),
                    "mean_shift": float(df_b[col].mean() - df_a[col].mean()),
                    "std_a": float(df_a[col].std()),
                    "std_b": float(df_b[col].std()),
                }
        result["value_stats"] = value_stats

        # overlap
        if key and key in df_a.columns and key in df_b.columns:
            keys_a = set(df_a[key])
            keys_b = set(df_b[key])
            result["overlap"] = {
                "shared": len(keys_a & keys_b),
                "only_a": len(keys_a - keys_b),
                "only_b": len(keys_b - keys_a),
            }

        return result

    @staticmethod
    def diff_report(df_a: pd.DataFrame, df_b: pd.DataFrame,
                     key: Optional[str] = None) -> str:
        """Human-readable diff report."""
        diff = DataComparer.compare(df_a, df_b, key)
        lines = ["📊 Data Comparison Report", "=" * 40]
        lines.append(f"Rows: {diff['rows_a']} → {diff['rows_b']} (diff: {diff['row_diff']:+d})")
        if diff["columns_added"]:
            lines.append(f"Columns added: {diff['columns_added']}")
        if diff["columns_removed"]:
            lines.append(f"Columns removed: {diff['columns_removed']}")
        if diff["schema_changes"]:
            lines.append("Schema changes:")
            for sc in diff["schema_changes"]:
                lines.append(f"  {sc['column']}: {sc['type_a']} → {sc['type_b']}")
        if diff["value_stats"]:
            lines.append("Value shifts:")
            for col, stats in list(diff["value_stats"].items())[:5]:
                lines.append(f"  {col}: mean {stats['mean_a']:.2f} → {stats['mean_b']:.2f} (shift: {stats['mean_shift']:+.2f})")
        if "overlap" in diff:
            ov = diff["overlap"]
            lines.append(f"Overlap: {ov['shared']} shared, {ov['only_a']} only-A, {ov['only_b']} only-B")
        return "\n".join(lines)


class DataStoryteller:
    """Auto-generate narrative reports from data.

    Produces a human-readable story that describes:
      - What the dataset contains
      - Key statistics and patterns
      - Interesting findings (correlations, outliers, trends)
      - Recommendations for ML training
    """

    @staticmethod
    def narrate(df: pd.DataFrame, name: str = "dataset") -> str:
        """Generate a narrative report from a DataFrame."""
        lines = [f"📖 Data Story: {name}", "=" * 50]

        # Overview
        lines.append(f"\nThis dataset contains {len(df)} rows and {len(df.columns)} columns.")
        numeric = df.select_dtypes(include=[np.number])
        categorical = df.select_dtypes(exclude=[np.number])
        lines.append(f"Of these, {len(numeric.columns)} are numeric and {len(categorical.columns)} are categorical.")

        # Missing data
        missing = df.isnull().sum()
        total_missing = missing.sum()
        if total_missing > 0:
            worst_col = missing.idxmax()
            lines.append(f"\n⚠️ There are {total_missing} missing values. "
                        f"The column '{worst_col}' has the most ({missing[worst_col]} missing).")
        else:
            lines.append("\n✅ No missing values — the data is complete.")

        # Duplicates
        dups = df.duplicated().sum()
        if dups > 0:
            lines.append(f"⚠️ {dups} duplicate rows found ({dups/len(df)*100:.1f}%). Consider removing them.")

        # Numeric insights
        if len(numeric.columns) > 0:
            lines.append("\n📊 Numeric column insights:")
            for col in numeric.columns[:5]:
                mean, std = float(numeric[col].mean()), float(numeric[col].std())
                lines.append(f"  • {col}: ranges from {numeric[col].min():.2f} to {numeric[col].max():.2f} "
                            f"(mean={mean:.2f}, std={std:.2f})")
                # outlier detection
                q1, q3 = numeric[col].quantile(0.25), numeric[col].quantile(0.75)
                iqr = q3 - q1
                n_outliers = ((numeric[col] < q1 - 1.5*iqr) | (numeric[col] > q3 + 1.5*iqr)).sum()
                if n_outliers > 0:
                    lines.append(f"    → {n_outliers} outliers detected (IQR method)")

            # correlations
            if len(numeric.columns) >= 2:
                corr = numeric.corr()
                # find strongest correlation
                mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                corr_vals = corr.where(mask).stack()
                if len(corr_vals) > 0:
                    strongest = corr_vals.abs().idxmax()
                    val = corr_vals[strongest]
                    lines.append(f"\n💡 Strongest correlation: '{strongest[0]}' and '{strongest[1]}' "
                                f"(r={val:.2f})")
                    if abs(val) > 0.8:
                        lines.append(f"   These columns are highly {'positively' if val > 0 else 'negatively'} correlated.")

        # Categorical insights
        if len(categorical.columns) > 0:
            lines.append("\n📝 Categorical column insights:")
            for col in categorical.columns[:3]:
                n_unique = df[col].nunique()
                top = df[col].value_counts().head(3)
                lines.append(f"  • {col}: {n_unique} unique values")
                for val, count in top.items():
                    lines.append(f"    → '{val}': {count} ({count/len(df)*100:.1f}%)")

        # ML recommendations
        lines.append("\n🤖 ML Training Recommendations:")
        if len(df) < 100:
            lines.append("  ⚠️ Small dataset — consider data augmentation or collecting more data.")
        if total_missing > len(df) * 0.1:
            lines.append("  ⚠️ High missing rate — use imputation before training.")
        if len(numeric.columns) == 0:
            lines.append("  ⚠️ No numeric features — encoding required before training.")
        if dups > len(df) * 0.05:
            lines.append("  ⚠️ High duplicate rate — deduplicate before training.")
        if len(numeric.columns) > 0 and len(df) > 100:
            lines.append("  ✅ Dataset is suitable for training most ML models.")

        return "\n".join(lines)


__all__ = ["DataViewer", "DataExplorer", "DataComparer", "DataStoryteller"]
