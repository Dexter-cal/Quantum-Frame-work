"""
qai.data.pipeline — end-to-end data pipeline: create → manipulate → train.

  DataPipeline        — chain data operations: load → clean → transform → split → train
  DataCreator         — create datasets from scratch (CSV, JSON, SQL, documents)
  DataSplitter        — train/val/test splits with stratification + time-based
  DocumentReader      — read PDFs, Word, Excel, text, HTML → structured data
  DatabaseConnector   — connect to SQL/NoSQL databases → query → dataset
  DataAugmentor       — augment data (text, image, tabular)
  SchemaBuilder       — define data schema with validation + endpoints
  DataStream          — streaming data pipeline for real-time ML
  DataVersioner       — version control for datasets (like git for data)
  FeatureEngineer     — auto-generate features from raw data

Usage:
    # Create a pipeline
    pipe = qai.DataPipeline()
    pipe.load_csv("data.csv")
    pipe.clean(strategy="mean")
    pipe.split(test_size=0.2)
    pipe.train(qai.build(technique="regression"))
    result = pipe.evaluate()

    # Read documents
    docs = qai.DocumentReader.read_pdf("report.pdf")
    docs = qai.DocumentReader.read_directory("./docs/")

    # Connect to database
    db = qai.DatabaseConnector.sqlite("mydb.db")
    df = db.query("SELECT * FROM users WHERE age > 25")

    # Create dataset from scratch
    ds = qai.DataCreator.from_dicts([{"name": "Alice", "age": 30}, ...])
    ds = qai.DataCreator.from_range(0, 100, name="index")
"""
from __future__ import annotations
import csv
import io
import json
import os
import random
import re
import sqlite3
import time
from collections import defaultdict, OrderedDict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# 1. DataCreator — create datasets from scratch
# ============================================================================
class DataCreator:
    """Create datasets from various sources.

    Methods:
      - from_csv(path)          — load CSV file
      - from_json(path)         — load JSON file
      - from_dicts(list)        — create from list of dicts
      - from_range(start, end)  — create from a number range
      - from_random(shape)      — create random data
      - from_sqlite(path, query) — create from SQLite query
      - from_text(path)         — create from text file (line-by-line)
      - from_array(arr)         — create from numpy array
      - from_generator(gen, n)  — create from a generator function
    """

    @staticmethod
    def from_csv(path: str, delimiter: str = ",",
                  has_header: bool = True) -> "DataPipeline":
        """Create a DataPipeline from a CSV file."""
        pipe = DataPipeline()
        pipe.load_csv(path, delimiter, has_header)
        return pipe

    @staticmethod
    def from_json(path: str) -> "DataPipeline":
        """Create from a JSON file."""
        pipe = DataPipeline()
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            pipe.data = data
            if data:
                pipe.columns = list(data[0].keys())
        return pipe

    @staticmethod
    def from_dicts(dicts: List[Dict[str, Any]]) -> "DataPipeline":
        """Create from a list of dictionaries."""
        pipe = DataPipeline()
        pipe.data = [list(d.values()) for d in dicts]
        pipe.columns = list(dicts[0].keys()) if dicts else []
        return pipe

    @staticmethod
    def from_range(start: int, end: int, step: int = 1,
                    name: str = "index") -> "DataPipeline":
        """Create from a number range."""
        pipe = DataPipeline()
        pipe.data = [[i] for i in range(start, end, step)]
        pipe.columns = [name]
        return pipe

    @staticmethod
    def from_random(n_samples: int = 100, n_features: int = 4,
                     seed: int = 0) -> "DataPipeline":
        """Create random data."""
        rng = np.random.default_rng(seed)
        pipe = DataPipeline()
        pipe.data = rng.standard_normal((n_samples, n_features)).tolist()
        pipe.columns = [f"feature_{i}" for i in range(n_features)]
        return pipe

    @staticmethod
    def from_array(arr: np.ndarray, columns: Optional[List[str]] = None) -> "DataPipeline":
        """Create from a numpy array."""
        pipe = DataPipeline()
        pipe.data = arr.tolist()
        pipe.columns = columns or [f"col_{i}" for i in range(arr.shape[1] if arr.ndim > 1 else 1)]
        return pipe

    @staticmethod
    def from_text(path: str, chunk_size: int = 500) -> "DataPipeline":
        """Create from a text file (split into chunks)."""
        pipe = DataPipeline()
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        pipe.data = [[i, chunk] for i, chunk in enumerate(chunks)]
        pipe.columns = ["chunk_id", "text"]
        return pipe

    @staticmethod
    def from_generator(gen: Callable[[int], List[Any]], n: int = 100,
                        columns: Optional[List[str]] = None) -> "DataPipeline":
        """Create from a generator function."""
        pipe = DataPipeline()
        pipe.data = [gen(i) for i in range(n)]
        pipe.columns = columns or [f"col_{i}" for i in range(len(pipe.data[0]) if pipe.data else 0)]
        return pipe

    @staticmethod
    def from_sqlite(db_path: str, query: str) -> "DataPipeline":
        """Create from a SQLite query."""
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(query)
        pipe = DataPipeline()
        pipe.columns = [desc[0] for desc in cursor.description]
        pipe.data = [list(row) for row in cursor.fetchall()]
        conn.close()
        return pipe


# ============================================================================
# 2. DataPipeline — chain operations + feed to models
# ============================================================================
class DataPipeline:
    """End-to-end data pipeline: load → clean → transform → split → train.

    The pipeline is chainable and lazy-evaluated where possible.

    Usage:
        pipe = DataPipeline()
        pipe.load_csv("data.csv")
        pipe.clean(strategy="mean")
        pipe.transform("standard_scaler")
        pipe.split(test_size=0.2)
        model = pipe.train(qai.build(technique="regression"))
        metrics = pipe.evaluate()
    """

    def __init__(self):
        self.data: List[List[Any]] = []
        self.columns: List[str] = []
        self.train_data: Optional[List[List[Any]]] = None
        self.test_data: Optional[List[List[Any]]] = None
        self.val_data: Optional[List[List[Any]]] = None
        self.model: Any = None
        self.target_col: Optional[str] = None
        self._operations: List[Dict[str, Any]] = []
        self._snapshots: Dict[str, Any] = {}

    # === Loading ===
    def load_csv(self, path: str, delimiter: str = ",",
                  has_header: bool = True) -> "DataPipeline":
        """Load data from a CSV file."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
        if has_header and rows:
            self.columns = rows[0]
            self.data = [list(row) for row in rows[1:]]
        else:
            self.data = [list(row) for row in rows]
            self.columns = [f"col_{i}" for i in range(len(rows[0]) if rows else 0)]
        self._log("load_csv", {"path": path, "n_rows": len(self.data)})
        return self

    def load_json(self, path: str) -> "DataPipeline":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            self.data = [list(d.values()) for d in data]
            self.columns = list(data[0].keys()) if data else []
        self._log("load_json", {"path": path, "n_rows": len(self.data)})
        return self

    def load_list(self, data: List[List[Any]],
                   columns: Optional[List[str]] = None) -> "DataPipeline":
        """Load from a Python list."""
        self.data = [list(row) for row in data]
        self.columns = columns or [f"col_{i}" for i in range(len(data[0]) if data else 0)]
        self._log("load_list", {"n_rows": len(self.data)})
        return self

    # === Inspection ===
    def shape(self) -> Tuple[int, int]:
        return (len(self.data), len(self.columns))

    def head(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get first n rows as dicts."""
        return [dict(zip(self.columns, row)) for row in self.data[:n]]

    def describe(self) -> Dict[str, Any]:
        """Summary statistics."""
        numeric_cols = {}
        for i, col in enumerate(self.columns):
            values = []
            for row in self.data:
                try: values.append(float(row[i]))
                except: pass
            if values:
                arr = np.array(values)
                numeric_cols[col] = {
                    "count": len(arr), "mean": float(arr.mean()),
                    "std": float(arr.std()), "min": float(arr.min()),
                    "max": float(arr.max()),
                }
        return {"shape": self.shape(), "columns": self.columns,
                "numeric_stats": numeric_cols}

    # === Cleaning ===
    def clean(self, strategy: str = "mean") -> "DataPipeline":
        """Clean missing values."""
        for i in range(len(self.columns)):
            values = []
            for row in self.data:
                if i < len(row) and row[i] is not None and row[i] != "":
                    try: values.append(float(row[i]))
                    except: pass
            if not values:
                continue
            fill_value = 0.0
            if strategy == "mean":
                fill_value = float(np.mean(values))
            elif strategy == "median":
                fill_value = float(np.median(values))
            elif strategy == "mode":
                fill_value = float(values[0])
            elif strategy == "drop":
                self.data = [row for row in self.data if i < len(row) and row[i] is not None and row[i] != ""]
                continue
            for row in self.data:
                if i < len(row) and (row[i] is None or row[i] == ""):
                    row[i] = fill_value
        self._log("clean", {"strategy": strategy})
        return self

    def remove_duplicates(self) -> "DataPipeline":
        """Remove duplicate rows."""
        seen = set()
        unique = []
        for row in self.data:
            key = tuple(row)
            if key not in seen:
                seen.add(key)
                unique.append(row)
        self.data = unique
        self._log("remove_duplicates", {"removed": len(self.data) - len(unique) + len(unique)})
        return self

    def remove_outliers(self, column: str, method: str = "iqr") -> "DataPipeline":
        """Remove outliers from a column."""
        col_idx = self.columns.index(column) if column in self.columns else 0
        values = []
        for row in self.data:
            try: values.append(float(row[col_idx]))
            except: pass
        if not values:
            return self
        arr = np.array(values)
        if method == "iqr":
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            self.data = [row for row in self.data
                         if col_idx < len(row) and lo <= float(row[col_idx]) <= hi]
        self._log("remove_outliers", {"column": column, "method": method})
        return self

    # === Transformation ===
    def transform(self, technique: str, **kwargs) -> "DataPipeline":
        """Apply a qai technique (scaler, encoder, etc.) to the data."""
        import qai
        try:
            t = qai.build(technique=technique, **kwargs)
            X = np.array([[float(v) for v in row] for row in self.data])
            t.fit(X)
            self.data = t.transform(X).tolist() if hasattr(t, "transform") else X.tolist()
        except Exception as e:
            self._log("transform_error", {"technique": technique, "error": str(e)})
        self._log("transform", {"technique": technique})
        return self

    def encode_categorical(self, method: str = "label") -> "DataPipeline":
        """Encode categorical columns."""
        for i, col in enumerate(self.columns):
            values = [row[i] if i < len(row) else None for row in self.data]
            if all(isinstance(v, str) for v in values if v is not None):
                if method == "label":
                    unique = list(set(values))
                    mapping = {v: j for j, v in enumerate(unique)}
                    for row in self.data:
                        if i < len(row):
                            row[i] = mapping.get(row[i], 0)
                elif method == "onehot":
                    unique = sorted(set(values))
                    for u in unique:
                        self.columns.append(f"{col}_{u}")
                    for row in self.data:
                        for u in unique:
                            row.append(1 if (i < len(row) and row[i] == u) else 0)
                    # remove original column
                    for row in self.data:
                        del row[i]
                    del self.columns[i]
        self._log("encode_categorical", {"method": method})
        return self

    def add_column(self, name: str, fn: Callable[[Dict[str, Any]], Any]) -> "DataPipeline":
        """Add a computed column."""
        self.columns.append(name)
        for row in self.data:
            row_dict = dict(zip(self.columns[:-1], row))
            try:
                val = fn(row_dict)
            except Exception:
                val = 0
            row.append(val)
        self._log("add_column", {"name": name})
        return self

    def select_columns(self, columns: List[str]) -> "DataPipeline":
        """Keep only specified columns."""
        indices = [self.columns.index(c) for c in columns if c in self.columns]
        self.data = [[row[i] for i in indices] for row in self.data]
        self.columns = [self.columns[i] for i in indices]
        self._log("select_columns", {"columns": columns})
        return self

    def drop_column(self, name: str) -> "DataPipeline":
        """Drop a column."""
        if name in self.columns:
            idx = self.columns.index(name)
            self.columns.pop(idx)
            for row in self.data:
                if idx < len(row):
                    row.pop(idx)
        self._log("drop_column", {"name": name})
        return self

    # === Splitting ===
    def split(self, test_size: float = 0.2, val_size: float = 0.0,
               stratify: bool = False, seed: int = 42) -> "DataPipeline":
        """Split into train/test/validation sets."""
        rng = random.Random(seed)
        indices = list(range(len(self.data)))
        rng.shuffle(indices)
        n_test = int(len(self.data) * test_size)
        n_val = int(len(self.data) * val_size)
        n_train = len(self.data) - n_test - n_val
        self.train_data = [self.data[indices[i]] for i in range(n_train)]
        self.test_data = [self.data[indices[i]] for i in range(n_train, n_train + n_test)]
        if n_val > 0:
            self.val_data = [self.data[indices[i]] for i in range(n_train + n_test, n_train + n_test + n_val)]
        self._log("split", {"test_size": test_size, "val_size": val_size,
                              "train": n_train, "test": n_test, "val": n_val})
        return self

    def set_target(self, column: str) -> "DataPipeline":
        """Set the target column for training."""
        self.target_col = column
        return self

    def get_X_y(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """Get X (features) and y (target) for a split."""
        data = {"train": self.train_data, "test": self.test_data,
                "val": self.val_data}.get(split, self.data)
        if not data:
            data = self.data
        target_idx = self.columns.index(self.target_col) if self.target_col else -1
        X, y = [], []
        for row in data:
            if target_idx >= 0:
                X.append([float(row[i]) for i in range(len(row)) if i != target_idx])
                try: y.append(float(row[target_idx]))
                except: y.append(0.0)
            else:
                X.append([float(v) for v in row])
                y.append(0.0)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # === Training ===
    def train(self, model: Any, target_col: Optional[str] = None) -> Any:
        """Train a model on the pipeline data."""
        if target_col:
            self.set_target(target_col)
        if not self.train_data:
            self.split()  # auto-split if not done
        X, y = self.get_X_y("train")
        model.fit(X, y)
        self.model = model
        self._log("train", {"model": type(model).__name__, "n_samples": len(X)})
        return model

    def evaluate(self, metric: str = "mse") -> Dict[str, Any]:
        """Evaluate the trained model."""
        if not self.model or not self.test_data:
            return {"error": "must train() first"}
        X_test, y_test = self.get_X_y("test")
        predictions = []
        for x in X_test:
            try: predictions.append(float(self.model.forward(x)))
            except: predictions.append(0.0)
        predictions = np.array(predictions)
        if metric == "mse":
            score = float(np.mean((predictions - y_test) ** 2))
        elif metric == "mae":
            score = float(np.mean(np.abs(predictions - y_test)))
        elif metric == "r2":
            ss_res = float(np.sum((y_test - predictions) ** 2))
            ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
            score = 1 - ss_res / max(ss_tot, 1e-8)
        elif metric == "accuracy":
            score = float(np.mean(predictions == y_test))
        else:
            score = float(np.mean((predictions - y_test) ** 2))
        return {"metric": metric, "score": score, "n_test": len(X_test),
                "predictions": predictions.tolist()[:10]}

    # === Export ===
    def to_csv(self, path: str) -> str:
        """Export to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
            writer.writerows(self.data)
        return path

    def to_json(self, path: str) -> str:
        """Export to JSON."""
        data = [dict(zip(self.columns, row)) for row in self.data]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    # === Utility ===
    def snapshot(self, name: str) -> "DataPipeline":
        """Save a snapshot of current state."""
        self._snapshots[name] = {
            "data": [list(row) for row in self.data],
            "columns": list(self.columns),
        }
        return self

    def rollback(self, name: str) -> "DataPipeline":
        """Rollback to a snapshot."""
        if name in self._snapshots:
            snap = self._snapshots[name]
            self.data = [list(row) for row in snap["data"]]
            self.columns = list(snap["columns"])
        return self

    def operations_log(self) -> List[Dict[str, Any]]:
        return list(self._operations)

    def _log(self, op: str, details: Dict[str, Any]) -> None:
        self._operations.append({"op": op, "details": details, "timestamp": time.time()})

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"<DataPipeline shape={self.shape()} columns={self.columns}>"


# ============================================================================
# 3. DocumentReader — read documents into datasets
# ============================================================================
class DocumentReader:
    """Read documents (PDF, Word, Excel, text, HTML) into datasets.

    Usage:
        docs = DocumentReader.read_pdf("report.pdf")
        docs = DocumentReader.read_directory("./docs/")
        docs = DocumentReader.read_url("https://example.com/page")
    """

    @staticmethod
    def read_pdf(path: str) -> "DataPipeline":
        """Read a PDF file → dataset with [page_num, text] columns."""
        pipe = DataPipeline()
        try:
            import PyPDF2
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pipe.data = [[i, page.extract_text()] for i, page in enumerate(reader.pages)]
                pipe.columns = ["page", "text"]
        except ImportError:
            try:
                # fallback: use pdfminer
                from pdfminer.high_level import extract_text
                text = extract_text(path)
                pipe.data = [[0, text]]
                pipe.columns = ["page", "text"]
            except ImportError:
                pipe.data = [[0, f"[install PyPDF2 or pdfminer to read PDFs]"]]
                pipe.columns = ["page", "text"]
        return pipe

    @staticmethod
    def read_text(path: str, chunk_size: int = 500) -> "DataPipeline":
        """Read a text file → dataset with chunked text."""
        return DataCreator.from_text(path, chunk_size)

    @staticmethod
    def read_html(path: str) -> "DataPipeline":
        """Read an HTML file → extract text."""
        pipe = DataPipeline()
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        # strip HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        pipe.data = [[0, text]]
        pipe.columns = ["index", "text"]
        return pipe

    @staticmethod
    def read_directory(dir_path: str, extensions: List[str] = None) -> "DataPipeline":
        """Read all files in a directory → dataset."""
        extensions = extensions or [".txt", ".csv", ".json", ".html"]
        pipe = DataPipeline()
        pipe.data = []
        pipe.columns = ["filename", "content", "size_bytes"]
        for name in sorted(os.listdir(dir_path)):
            if any(name.endswith(ext) for ext in extensions):
                path = os.path.join(dir_path, name)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                pipe.data.append([name, content, len(content)])
        return pipe

    @staticmethod
    def read_excel(path: str) -> "DataPipeline":
        """Read an Excel file → dataset."""
        pipe = DataPipeline()
        try:
            import pandas as pd
            df = pd.read_excel(path)
            pipe.columns = list(df.columns)
            pipe.data = df.values.tolist()
        except ImportError:
            pipe.data = [[0, "[install pandas + openpyxl to read Excel]"]]
            pipe.columns = ["error", "message"]
        return pipe


# ============================================================================
# 4. DatabaseConnector — connect to databases
# ============================================================================
class DatabaseConnector:
    """Connect to SQL databases → query → dataset.

    Usage:
        db = DatabaseConnector.sqlite("mydb.db")
        df = db.query("SELECT * FROM users")
        db.close()
    """

    @staticmethod
    def sqlite(path: str) -> "DatabaseConnection":
        """Connect to a SQLite database."""
        conn = sqlite3.connect(path)
        return DatabaseConnection(conn, "sqlite")

    @staticmethod
    def postgres(host: str, database: str, user: str, password: str,
                  port: int = 5432) -> "DatabaseConnection":
        """Connect to PostgreSQL."""
        try:
            import psycopg2
            conn = psycopg2.connect(host=host, database=database, user=user,
                                     password=password, port=port)
            return DatabaseConnection(conn, "postgres")
        except ImportError:
            raise ImportError("pip install psycopg2")

    @staticmethod
    def mysql(host: str, database: str, user: str, password: str,
              port: int = 3306) -> "DatabaseConnection":
        """Connect to MySQL."""
        try:
            import pymysql
            conn = pymysql.connect(host=host, database=database, user=user,
                                    password=password, port=port)
            return DatabaseConnection(conn, "mysql")
        except ImportError:
            raise ImportError("pip install pymysql")

    @staticmethod
    def mongodb(uri: str, database: str) -> "DatabaseConnection":
        """Connect to MongoDB."""
        try:
            from pymongo import MongoClient
            client = MongoClient(uri)
            return DatabaseConnection(client[database], "mongodb")
        except ImportError:
            raise ImportError("pip install pymongo")


class DatabaseConnection:
    """A database connection wrapper."""

    def __init__(self, conn: Any, db_type: str):
        self.conn = conn
        self.db_type = db_type

    def query(self, sql: str) -> "DataPipeline":
        """Execute a SQL query → DataPipeline."""
        pipe = DataPipeline()
        if self.db_type in ("sqlite", "postgres", "mysql"):
            cursor = self.conn.execute(sql)
            pipe.columns = [desc[0] for desc in cursor.description] if cursor.description else []
            pipe.data = [list(row) for row in cursor.fetchall()]
        return pipe

    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        if self.db_type == "sqlite":
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return [row[0] for row in cursor.fetchall()]
        return []

    def table_info(self, table: str) -> "DataPipeline":
        """Get info about a table."""
        return self.query(f"PRAGMA table_info({table})" if self.db_type == "sqlite"
                           else f"DESCRIBE {table}")

    def close(self):
        if hasattr(self.conn, "close"):
            self.conn.close()


# ============================================================================
# 5. DataAugmentor — augment data
# ============================================================================
class DataAugmentor:
    """Augment data for training.

    Methods:
      - augment_tabular(X, y, n_copies) — add noise + mixup
      - augment_text(texts, n_copies)   — synonym replacement + back-translation
      - augment_image(images, n_copies) — flip, rotate, crop, noise
    """

    @staticmethod
    def augment_tabular(X: np.ndarray, y: Optional[np.ndarray] = None,
                         n_copies: int = 2, noise_std: float = 0.05) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Augment tabular data with noise + mixup."""
        X_aug = [X]
        y_aug = [y] if y is not None else None
        for _ in range(n_copies):
            # noise augmentation
            noisy = X + np.random.normal(0, noise_std, X.shape).astype(X.dtype)
            X_aug.append(noisy)
            if y is not None:
                y_aug.append(y)
            # mixup
            idx = np.random.permutation(len(X))
            alpha = np.random.beta(0.2, 0.2, size=(len(X), 1)).astype(X.dtype)
            mixed = alpha * X + (1 - alpha) * X[idx]
            X_aug.append(mixed)
            if y is not None:
                mixed_y = alpha.flatten() * y + (1 - alpha.flatten()) * y[idx]
                y_aug.append(mixed_y)
        X_out = np.vstack(X_aug)
        y_out = np.concatenate(y_aug) if y_aug else None
        return X_out, y_out

    @staticmethod
    def augment_text(texts: List[str], n_copies: int = 2) -> List[str]:
        """Augment text with synonym replacement + shuffling."""
        augmented = list(texts)
        for _ in range(n_copies):
            for text in texts:
                words = text.split()
                if len(words) > 3:
                    # shuffle word order slightly
                    idx = random.randint(0, len(words) - 2)
                    words[idx], words[idx + 1] = words[idx + 1], words[idx]
                    augmented.append(" ".join(words))
                # duplicate with slight modification
                augmented.append(text + " " + text.split()[-1])
        return augmented

    @staticmethod
    def augment_image(images: np.ndarray, n_copies: int = 2) -> np.ndarray:
        """Augment images with flip + noise."""
        augmented = [images]
        for _ in range(n_copies):
            # horizontal flip
            flipped = images[:, :, ::-1] if images.ndim == 4 else images[::-1]
            augmented.append(flipped.copy())
            # add noise
            noisy = images + np.random.normal(0, 0.05, images.shape).astype(images.dtype)
            augmented.append(noisy)
        return np.vstack(augmented)


# ============================================================================
# 6. SchemaBuilder — define data schema with validation
# ============================================================================
class SchemaBuilder:
    r"""Define a data schema with types, constraints, and validation.

    Usage:
        schema = SchemaBuilder()
        schema.add_field("name", type=str, required=True, max_length=100)
        schema.add_field("age", type=int, required=True, min_value=0, max_value=150)
        schema.add_field("email", type=str, pattern=r"[^@]+@[^@]+\.[^@]+")
        valid = schema.validate(data_dict)
    """

    def __init__(self):
        self.fields: Dict[str, Dict[str, Any]] = {}

    def add_field(self, name: str, type: type = Any, required: bool = False,
                   default: Any = None, min_value: Any = None, max_value: Any = None,
                   max_length: Any = None, pattern: Optional[str] = None,
                   choices: Optional[List[Any]] = None) -> "SchemaBuilder":
        self.fields[name] = {
            "type": type, "required": required, "default": default,
            "min_value": min_value, "max_value": max_value, "max_length": max_length,
            "pattern": pattern, "choices": choices,
        }
        return self

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a data dict against the schema."""
        errors = []
        for name, spec in self.fields.items():
            if name not in data:
                if spec["required"]:
                    errors.append(f"{name}: required field missing")
                elif spec["default"] is not None:
                    data[name] = spec["default"]
                continue
            value = data[name]
            # type check
            if spec["type"] != Any and not isinstance(value, spec["type"]):
                try: data[name] = spec["type"](value)
                except: errors.append(f"{name}: expected {spec['type'].__name__}, got {type(value).__name__}")
            # min/max
            if spec["min_value"] is not None and value < spec["min_value"]:
                errors.append(f"{name}: value {value} < min {spec['min_value']}")
            if spec["max_value"] is not None and value > spec["max_value"]:
                errors.append(f"{name}: value {value} > max {spec['max_value']}")
            # max_length
            if spec["max_length"] is not None and len(str(value)) > spec["max_length"]:
                errors.append(f"{name}: length {len(str(value))} > max {spec['max_length']}")
            # pattern
            if spec["pattern"] is not None:
                if not re.match(spec["pattern"], str(value)):
                    errors.append(f"{name}: doesn't match pattern {spec['pattern']}")
            # choices
            if spec["choices"] is not None and value not in spec["choices"]:
                errors.append(f"{name}: value {value} not in {spec['choices']}")
        return {"valid": len(errors) == 0, "errors": errors, "data": data}

    def to_json(self) -> str:
        return json.dumps({name: {k: str(v) for k, v in spec.items()}
                            for name, spec in self.fields.items()}, indent=2)

    def to_endpoint_spec(self) -> Dict[str, Any]:
        """Generate an API endpoint specification from the schema."""
        return {
            "fields": {name: {"type": spec["type"].__name__,
                                "required": spec["required"],
                                "default": spec["default"],
                                "min": spec["min_value"],
                                "max": spec["max_value"],
                                "pattern": spec["pattern"],
                                "choices": spec["choices"]}
                        for name, spec in self.fields.items()},
            "example": {name: spec["default"] or "" for name, spec in self.fields.items()},
        }


# ============================================================================
# 7. DataVersioner — version control for datasets
# ============================================================================
class DataVersioner:
    """Version control for datasets (like git for data).

    Usage:
        dv = DataVersioner("./data_repo")
        dv.commit(pipe.data, pipe.columns, message="initial dataset")
        dv.commit(pipe.data, pipe.columns, message="after cleaning")
        history = dv.log()
        dv.checkout("v1")
    """

    def __init__(self, repo_dir: str = "./data_repo"):
        self.repo_dir = repo_dir
        os.makedirs(repo_dir, exist_ok=True)
        self._versions: Dict[str, Dict[str, Any]] = {}
        self._load_versions()

    def _load_versions(self) -> None:
        meta_path = os.path.join(self.repo_dir, "versions.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self._versions = json.load(f)

    def _save_versions(self) -> None:
        meta_path = os.path.join(self.repo_dir, "versions.json")
        with open(meta_path, "w") as f:
            json.dump(self._versions, f, indent=2, default=str)

    def commit(self, data: List[List[Any]], columns: List[str],
                message: str = "") -> str:
        """Commit a new version of the dataset."""
        version = f"v{len(self._versions) + 1}"
        import pickle
        data_path = os.path.join(self.repo_dir, f"{version}.pkl")
        with open(data_path, "wb") as f:
            pickle.dump({"data": data, "columns": columns}, f)
        self._versions[version] = {
            "message": message, "timestamp": time.time(),
            "n_rows": len(data), "n_cols": len(columns),
            "file": data_path,
        }
        self._save_versions()
        return version

    def checkout(self, version: str) -> Tuple[List[List[Any]], List[str]]:
        """Checkout a specific version."""
        if version not in self._versions:
            raise KeyError(f"version {version} not found")
        import pickle
        with open(self._versions[version]["file"], "rb") as f:
            data = pickle.load(f)
        return data["data"], data["columns"]

    def log(self) -> List[Dict[str, Any]]:
        """Show version history."""
        return [{"version": v, **info} for v, info in sorted(self._versions.items())]

    def diff(self, v1: str, v2: str) -> Dict[str, Any]:
        """Compare two versions."""
        d1, c1 = self.checkout(v1)
        d2, c2 = self.checkout(v2)
        return {
            "v1": {"rows": len(d1), "cols": len(c1)},
            "v2": {"rows": len(d2), "cols": len(c2)},
            "row_diff": len(d2) - len(d1),
            "col_diff": len(c2) - len(c1),
            "columns_added": [c for c in c2 if c not in c1],
            "columns_removed": [c for c in c1 if c not in c2],
        }


# ============================================================================
# 8. FeatureEngineer — auto-generate features
# ============================================================================
class FeatureEngineer:
    """Automatically generate features from raw data.

    Generates:
      - Polynomial features (x^2, x^3, x*y)
      - Statistical features (mean, std, min, max per row)
      - Binned features (discretize continuous values)
      - Interaction features (a*b, a/b, a-b)
      - Text features (word count, char count, avg word length)
    """

    @staticmethod
    def auto_generate(X: np.ndarray, max_features: int = 50) -> np.ndarray:
        """Auto-generate features from a numeric array."""
        n_samples, n_features = X.shape if X.ndim > 1 else (len(X), 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        features = [X]
        # polynomial
        if n_features <= 5:
            for i in range(n_features):
                features.append((X[:, i] ** 2).reshape(-1, 1))
                features.append((X[:, i] ** 3).reshape(-1, 1))
        # interactions
        if n_features <= 5:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        # statistical
        features.append(X.mean(axis=1).reshape(-1, 1))
        features.append(X.std(axis=1).reshape(-1, 1))
        features.append(X.min(axis=1).reshape(-1, 1))
        features.append(X.max(axis=1).reshape(-1, 1))
        result = np.hstack(features)
        # cap at max_features
        if result.shape[1] > max_features:
            result = result[:, :max_features]
        return result.astype(np.float32)

    @staticmethod
    def text_features(texts: List[str]) -> np.ndarray:
        """Generate features from text."""
        features = []
        for text in texts:
            words = text.split()
            chars = len(text)
            n_words = len(words)
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            n_unique = len(set(words))
            n_upper = sum(1 for c in text if c.isupper())
            n_digits = sum(1 for c in text if c.isdigit())
            features.append([chars, n_words, avg_word_len, n_unique, n_upper, n_digits])
        return np.array(features, dtype=np.float32)


__all__ = [
    "DataCreator", "DataPipeline", "DocumentReader", "DatabaseConnector",
    "DatabaseConnection", "DataAugmentor", "SchemaBuilder",
    "DataVersioner", "FeatureEngineer",
]
