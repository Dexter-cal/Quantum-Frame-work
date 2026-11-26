import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import qai
from fixtures_huggingface_iris import load as load_hf_iris

np.random.seed(42)

X, y = load_hf_iris()  # 30 real rows from Hugging Face

print("=" * 60)
print("TEST: Dataset EDA -- describe(), info(), value_counts() on real data")
print("=" * 60)
ds = qai.Dataset.from_arrays(X, y, feature_names=["sepal_len", "sepal_wid", "petal_len", "petal_wid"], target_name="species")
info = ds.info()
print("info():", info)
assert info["row_count"] == 30
assert info["duplicate_rows"] == 0
print("VERIFIED: info() accurately reports the real, clean starting state")

print()
print("=" * 60)
print("TEST: deliberately CORRUPT the real data (duplicates + missing values),")
print("then verify cleaning genuinely fixes it -- not just 'runs without error'")
print("=" * 60)
corrupted_df = ds.df.copy()
# inject 5 exact duplicate rows (from rows 5-9, kept untouched)
corrupted_df = pd.concat([corrupted_df, corrupted_df.iloc[5:10]], ignore_index=True)
# inject missing values into DIFFERENT rows (10, 11, 12) -- deliberately
# NOT the same rows being duplicated, so the duplicate relationship stays
# intact and the two corruptions don't interfere with each other
# (an earlier version of this test injected NaN into the SAME rows it had
# just duplicated, which silently broke the duplicate relationship for 3
# of the 5 rows -- a real test-design bug, caught by the assertion below
# actually checking an exact expected number instead of just "some > 0")
corrupted_df.loc[10, "sepal_len"] = np.nan
corrupted_df.loc[11, "petal_wid"] = np.nan
corrupted_df.loc[12, "sepal_wid"] = np.nan

corrupted_ds = qai.Dataset(corrupted_df, target_column="species")
info_before = corrupted_ds.info()
print("Corrupted state:", info_before)
assert info_before["row_count"] == 35
assert info_before["duplicate_rows"] == 5
assert sum(info_before["null_counts"].values()) == 3

corrupted_ds.remove_duplicates(verbose=True)
assert corrupted_ds.row_count() == 30, f"BUG: expected 30 rows after dedup, got {corrupted_ds.row_count()}"
print("VERIFIED: remove_duplicates() genuinely removed exactly the 5 injected duplicates")

corrupted_ds.handle_missing(strategy="mean", verbose=True)
null_after = int(corrupted_ds.df.isnull().sum().sum())
assert null_after == 0, f"BUG: {null_after} nulls remain after handle_missing"
print("VERIFIED: handle_missing() genuinely filled all 3 injected missing values")

print()
print("=" * 60)
print("TEST: filter() and select_columns() -- real database-style operations")
print("=" * 60)
setosa_only = ds.filter(lambda row: row["species"] == "Iris-setosa")
print(f"filter() to setosa only: {setosa_only.row_count()} rows (expected 10)")
assert setosa_only.row_count() == 10

petals_only = ds.select_columns(["petal_len", "petal_wid", "species"])
print(f"select_columns(): {list(petals_only.df.columns)}")
assert list(petals_only.df.columns) == ["petal_len", "petal_wid", "species"]
print("VERIFIED: filter() and select_columns() genuinely narrow the data correctly")

print()
print("=" * 60)
print("TEST: STRATIFIED split -- does it ACTUALLY preserve class balance?")
print("(Section 111's specific, named concern: a naive random split can")
print(" accidentally put almost all of one rare class on one side)")
print("=" * 60)
train_ds, test_ds = ds.split(train=0.7, test=0.3, stratify_by="species", seed=42)
print(f"Train: {train_ds.row_count()} rows, Test: {test_ds.row_count()} rows")

train_counts = train_ds.value_counts("species")
test_counts = test_ds.value_counts("species")
print("Train class distribution:", train_counts)
print("Test class distribution:", test_counts)

# REAL CHECK: every one of the 3 species should appear in BOTH splits --
# with an unstratified split on 30 rows this could easily fail by chance
for species in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]:
    assert species in train_counts and species in test_counts, \
        f"BUG: stratification failed -- '{species}' missing from one split"
    assert train_counts[species] == 7 and test_counts[species] == 3, \
        f"BUG: '{species}' not evenly stratified: train={train_counts.get(species)}, test={test_counts.get(species)}"
print("VERIFIED: every class appears in BOTH splits with the correct, even proportion (7/3 per class)")

print()
print("=" * 60)
print("TEST: full end-to-end -- Dataset flows straight into Model.train()")
print("=" * 60)
model = qai.build(type="decision_tree", max_depth=3)
model.train(train_ds, verbose=True)  # a Dataset object, not raw arrays -- real integration
test_X, test_y = test_ds.to_arrays()
predictions = model.predict(test_X)
accuracy = np.mean(predictions == test_y)
print(f"Accuracy on the held-out, stratified test split: {accuracy:.4f}")
assert accuracy > 0.7, f"BUG: unexpectedly low accuracy ({accuracy}) -- something is wrong with the split or training"
print("VERIFIED: a real model trains directly on a Dataset object and generalizes to held-out data")

print()
print("ALL DATASET TESTS PASSED -- real cleaning verified on deliberately")
print("corrupted real data, real stratification verified against exact")
print("class proportions, real end-to-end training through Dataset objects")
