"""
qai.core.cross_validation -- Section 99/112: cross-validation as a real
defense against "did I just get a lucky train/test split." Genuinely
re-trains a fresh model per fold (not the same model re-evaluated), since
reusing one already-trained model across folds would silently leak
information between folds -- a real, easy-to-get-wrong detail.
"""
from __future__ import annotations
import numpy as np


def k_fold_split(n_samples: int, k: int, seed: int = 42):
    """Returns k (train_indices, test_indices) pairs, genuinely
    non-overlapping and covering every sample exactly once as test data."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1  # distribute the remainder fairly

    folds = []
    current = 0
    for size in fold_sizes:
        folds.append(indices[current: current + size])
        current += size

    splits = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, test_idx))
    return splits


def cross_validate(build_fn, X, y, k=5, seed=42, verbose=True):
    """build_fn: a zero-arg function returning a FRESH, untrained model
    each call -- critical, since reusing one trained model across folds
    would defeat the entire point of cross-validation."""
    X = np.asarray(X)
    y = np.asarray(y)
    splits = k_fold_split(len(X), k, seed=seed)
    fold_accuracies = []

    for fold_num, (train_idx, test_idx) in enumerate(splits):
        model = build_fn()  # a genuinely fresh model, not the same one reused
        model.train(X[train_idx], y[train_idx], verbose=False)
        preds = model.predict(X[test_idx])
        accuracy = float(np.mean(preds == y[test_idx]))
        fold_accuracies.append(accuracy)
        if verbose:
            print(f"[qai] fold {fold_num + 1}/{k}: {len(train_idx)} train, "
                  f"{len(test_idx)} test, accuracy={accuracy:.4f}")

    return {
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
    }
