# Guide: Data — Load, Clean, Manipulate, Export

*You want to work with data, not models yet. Start here.*

> Back to [Guides index](README.md) · Full API: [User Manual §16](../USER_MANUAL.md) · Next chapter: [TRAINING.md](TRAINING.md)

---

## What you'll learn

1. Load data from any file format (CSV, JSON, Parquet, Excel, PDF, DOCX, EPUB, TXT)
2. Clean it (missing values, duplicates, outliers, PII)
3. Manipulate it (Excel-style formulas, snapshots, transforms, feature engineering)
4. Audit quality before training
5. Export it back out

All of this lives in `qai.data` plus dataset helpers on the top-level `qai` import.

---

## 1. Load data

### Built-in datasets (zero download, real data)

```python
import qai

X, y = qai.load_iris()            # 150 samples, 4 features, 3 classes
X, y = qai.load_digits()          # 1,797 samples, 64 features (8x8 digit images)
X, y = qai.load_wine()            # 178 samples, 13 features
X, y = qai.load_breast_cancer()   # 569 samples, 30 features
X, y = qai.load_diabetes()        # 442 samples, 10 features (regression target)
```

### From files — the universal loader

```python
from qai.data import DataDataset, detect_format

# Format auto-detection: pass a path, get a dataset
ds = DataDataset.load("sales.csv")
ds = DataDataset.load("config.json")
ds = DataDataset.load("warehouse.parquet")
ds = DataDataset.load("budget.xlsx")

# Or use explicit constructors
ds = qai.Dataset.from_csv("sales.csv", target_column="revenue")
ds = qai.Dataset.from_json("events.json")
ds = qai.Dataset.from_parquet("warehouse.parquet")
ds = qai.Dataset.from_excel("budget.xlsx")
```

### From documents (PDF / Word / books → trainable text chunks)

```python
ds = qai.Dataset.from_pdf("paper.pdf")       # PDF → text chunks
ds = qai.Dataset.from_docx("report.docx")    # Word → text chunks
ds = qai.Dataset.from_book("novel.epub")     # EPUB → text chunks
ds = qai.Dataset.from_text("notes.txt")      # TXT → text chunks

# Or one call for any document:
ds = DataDataset.from_document("anything.pdf")   # auto-detects type, chunks text
```

Text chunks feed straight into tokenizers and language models — see [MEMORY_RAG.md](MEMORY_RAG.md) for the RAG path.

### From HuggingFace (streaming — train on data bigger than RAM)

```python
for batch in qai.stream_dataset("imdb", split="train"):
    X_batch, y_batch = batch["text"], batch["label"]
    ...
```

---

## 2. Clean data

```python
# Missing values
X_clean, y_clean = qai.clean_dataset(X_dirty, strategy="mean")   # mean / median / mode / drop

# Near-duplicate fuzzy matching (catches "John Smith" vs "Jon Smith")
from qai.data import fuzzy_deduplicate
ds_unique = fuzzy_deduplicate(ds, threshold=0.9)

# Outliers (IQR method)
from qai.data import detect_outliers_iqr
outlier_mask = detect_outliers_iqr(X)          # boolean mask
X_inliers = X[~outlier_mask]

# Interpolate gaps in sequential data
from qai.data import interpolate_missing
X_filled = interpolate_missing(X, method="linear")
```

---

## 3. Manipulate data (`qai.data.DataPipeline`)

`DataPipeline` chains column operations the way a spreadsheet would — but reproducible and scriptable:

```python
from qai.data_pipeline import DataPipeline

pipe = DataPipeline()
pipe.add_step("drop_columns", ["ssn", "internal_id"])
pipe.add_step("fill_missing", column="age", strategy="median")
pipe.add_step("normalize", columns=["price", "quantity"])
pipe.add_step("derive", name="total", formula="price * quantity")

clean_ds = pipe.run(ds)
```

Each step is recorded, so the pipeline is auditable and re-runnable on new data.

### Excel-style cell manipulation

```python
# Set cells, apply formulas, version snapshots
ds.set_cell(row=3, column="price", value=19.99)
ds.apply_formula(column="total", formula="price * quantity")
snapshot_id = ds.snapshot()                    # version this state
ds.rollback(snapshot_id)                       # undo changes
```

### PII detection (know what sensitive data you hold)

```python
from qai.data import quality_report  # 6-dimension audit (below)
# PII scanning also available via qai.security:
from qai.security import detect_pii, redact_pii

findings = detect_pii(text_column)         # emails, phones, SSNs, cards, IPs, JWTs...
redacted = redact_pii(text_column)         # mask them out before training/sharing
```

Full security workflow: [SECURITY_GOVERNANCE.md](SECURITY_GOVERNANCE.md).

---

## 4. Audit quality before training

```python
from qai.data import quality_report

report = quality_report(ds)   # six dimensions: completeness, uniqueness,
                              # validity, consistency, accuracy, timeliness
print(report.summary())
```

Visual checks:

```python
qai.data_summary(X, y)             # text summary
qai.plot_distributions(X)          # histograms
qai.plot_class_balance(y)          # bar chart
qai.plot_correlation(X)            # heatmap
qai.plot_outliers(X)               # box plots
qai.data_dashboard(X, y, save_dir="./plots")   # all of the above
```

---

## 5. Split, batch, transform

```python
ds = qai.Dataset.from_arrays(X, y, target_name="label")

ds.shuffle(seed=42)
train, test = ds.split(train=0.8, stratify_by="label")

for X_batch, y_batch in ds.batch(32):
    model.train(X_batch, y_batch)

ds.map(lambda row: row * 2)          # transform features
print(ds.describe())                 # statistics
```

### Feature engineering helpers

```python
from qai.data import rolling_window, lag_feature

# Time-series features
X_roll = rolling_window(series, window=7)     # 7-day rolling means
X_lag  = lag_feature(series, lag=1)           # yesterday's value as a feature
```

### Preprocessing (the sklearn-style transformers)

```python
from qai.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, ColumnTransformer

scaler = StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_test)

# Different transforms for different columns:
ct = ColumnTransformer([
    ("num", StandardScaler(), ["age", "income"]),
    ("cat", OneHotEncoder(), ["country"]),
])
X_ready = ct.fit_transform(ds)
```

---

## 6. Export back out

```python
ds.to_csv("out.csv")
ds.to_json("out.json")
ds.to_parquet("out.parquet")
ds.to_excel("out.xlsx")
```

---

## Cheat sheet

| Task | Call |
|---|---|
| Load anything | `DataDataset.load(path)` — auto-detects format |
| Load a document | `DataDataset.from_document(path)` |
| Clean missing | `qai.clean_dataset(X, strategy="mean")` |
| Fuzzy dedup | `fuzzy_deduplicate(ds, threshold=0.9)` |
| Quality audit | `quality_report(ds)` — 6 dimensions |
| PII scan | `qai.security.detect_pii(text)` |
| Excel formulas | `ds.apply_formula(column, formula)` |
| Version data | `ds.snapshot()` / `ds.rollback(id)` |
| Pipeline steps | `qai.data_pipeline.DataPipeline().add_step(...)` |
| Stratified split | `ds.split(train=0.8, stratify_by="label")` |
| Stream big data | `qai.stream_dataset("imdb")` |
| Export | `ds.to_csv / to_json / to_parquet / to_excel` |

---

## Common gotchas

- **Dict rows vs matrices.** `Model.train()` accepts both dict payloads and NumPy 2D matrices — see [FINDINGS.md §5](../FINDINGS.md). Schemas validate field names/types automatically.
- **uint8 images.** Values > 1.5 are auto-divided by 255 in the CNN path — see [MULTIMODAL.md](MULTIMODAL.md).
- **Parquet needs pyarrow.** If it's missing, `qai.doctor()` will tell you — see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

*Next: [TRAINING.md](TRAINING.md) — turn this data into a model.*
