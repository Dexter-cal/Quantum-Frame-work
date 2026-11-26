"""
qai.data_augmentor — configurable data augmentation pipeline for tabular
and image data.

Usage:
    import qai
    import numpy as np

    # Tabular augmentation
    aug = qai.DataAugmentor(mode="tabular")
    aug.add("jitter", noise_std=0.05)
    aug.add("mixup", alpha=0.2)
    X_aug, y_aug = aug.transform(X, y, n_copies=3)  # 3x the data

    # Image augmentation
    aug2 = qai.DataAugmentor(mode="image")
    aug2.add("flip_horizontal", prob=0.5)
    aug2.add("rotate_90", prob=0.3)
    aug2.add("brightness", factor_range=(0.8, 1.2))
    X_img_aug = aug2.transform_images(X_images, n_copies=2)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np


@dataclass
class AugmentationStep:
    """A single augmentation step."""
    name: str
    params: Dict[str, Any]
    fn: Callable


class DataAugmentor:
    """Configurable data augmentation pipeline.

    Args:
        mode: "tabular" or "image".
        seed: random seed for reproducibility.
    """
    def __init__(self, mode: str = "tabular", seed: int = 42):
        if mode not in ("tabular", "image"):
            raise ValueError(f"[WHAT] Unknown augmentation mode '{mode}'\n"
                             f"[WHY] Mode must be 'tabular' or 'image'\n"
                             f"[FIX] Use one of the supported modes")
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.steps: List[AugmentationStep] = []

    def add(self, name: str, **params) -> "DataAugmentor":
        """Add an augmentation step.

        For tabular mode:
            - "jitter": add Gaussian noise (noise_std=0.05)
            - "mixup": linear interpolation between samples (alpha=0.2)
            - "smote": synthetic minority oversampling (k=5)
            - "cutout": zero out random features (n_holes=1, length=2)
            - "scale": random scaling (scale_range=(0.9, 1.1))

        For image mode:
            - "flip_horizontal": flip left-right (prob=0.5)
            - "flip_vertical": flip top-bottom (prob=0.5)
            - "rotate_90": rotate 90 degrees (prob=0.3)
            - "brightness": adjust brightness (factor_range=(0.8, 1.2))
            - "noise": add Gaussian noise (noise_std=0.01)
            - "crop": random crop (crop_size=0.9)

        Returns self (for chaining).
        """
        fn = self._get_fn(name)
        if fn is None:
            raise ValueError(f"[WHAT] Unknown augmentation '{name}' for mode '{self.mode}'\n"
                             f"[WHY] No registered augmentation with that name\n"
                             f"[FIX] Use one of the supported augmentations")
        self.steps.append(AugmentationStep(name=name, params=params, fn=fn))
        return self

    def transform(self, X, y=None, n_copies: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentations to tabular data.

        Args:
            X: input data (n_samples, n_features).
            y: labels (optional).
            n_copies: number of augmented copies to generate.

        Returns:
            (X_augmented, y_augmented) — original + augmented data.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        all_X = [X]
        all_y = [y] if y is not None else []

        for _ in range(n_copies):
            X_aug = X.copy()
            for step in self.steps:
                X_aug = step.fn(X_aug, self.rng, **step.params)
            all_X.append(X_aug)
            if y is not None:
                all_y.append(y)

        X_result = np.concatenate(all_X, axis=0)
        y_result = np.concatenate(all_y) if y is not None else None
        return X_result, y_result

    def transform_images(self, X, n_copies: int = 1) -> np.ndarray:
        """Apply augmentations to image data.

        Args:
            X: images (n, C, H, W) or (n, H, W).
            n_copies: number of augmented copies.

        Returns:
            X_augmented — original + augmented images.
        """
        X = np.asarray(X, dtype=np.float32)
        all_X = [X]
        for _ in range(n_copies):
            X_aug = X.copy()
            for step in self.steps:
                X_aug = step.fn(X_aug, self.rng, **step.params)
            all_X.append(X_aug)
        return np.concatenate(all_X, axis=0)

    def list_available(self) -> List[str]:
        """List available augmentation names for the current mode."""
        if self.mode == "tabular":
            return ["jitter", "mixup", "smote", "cutout", "scale"]
        return ["flip_horizontal", "flip_vertical", "rotate_90",
                "brightness", "noise", "crop"]

    # ------------------------------------------------------------------
    # Augmentation functions
    # ------------------------------------------------------------------
    def _get_fn(self, name: str) -> Optional[Callable]:
        fns = self._tabular_fns() if self.mode == "tabular" else self._image_fns()
        return fns.get(name)

    @staticmethod
    def _tabular_fns() -> Dict[str, Callable]:
        return {
            "jitter": lambda X, rng, noise_std=0.05, **kw: X + rng.standard_normal(X.shape).astype(X.dtype) * noise_std,
            "mixup": lambda X, rng, alpha=0.2, **kw: DataAugmentor._mixup(X, rng, alpha),
            "smote": lambda X, rng, k=5, **kw: DataAugmentor._smote(X, rng, k),
            "cutout": lambda X, rng, n_holes=1, length=2, **kw: DataAugmentor._cutout(X, rng, n_holes, length),
            "scale": lambda X, rng, scale_range=(0.9, 1.1), **kw: X * rng.uniform(scale_range[0], scale_range[1]),
        }

    @staticmethod
    def _image_fns() -> Dict[str, Callable]:
        return {
            "flip_horizontal": lambda X, rng, prob=0.5, **kw: DataAugmentor._flip_h(X, rng, prob),
            "flip_vertical": lambda X, rng, prob=0.5, **kw: DataAugmentor._flip_v(X, rng, prob),
            "rotate_90": lambda X, rng, prob=0.3, **kw: DataAugmentor._rotate90(X, rng, prob),
            "brightness": lambda X, rng, factor_range=(0.8, 1.2), **kw: np.clip(X * rng.uniform(factor_range[0], factor_range[1]), 0, 1),
            "noise": lambda X, rng, noise_std=0.01, **kw: np.clip(X + rng.standard_normal(X.shape).astype(X.dtype) * noise_std, 0, 1),
            "crop": lambda X, rng, crop_size=0.9, **kw: DataAugmentor._crop(X, rng, crop_size),
        }

    @staticmethod
    def _mixup(X, rng, alpha):
        lam = rng.beta(alpha, alpha) if alpha > 0 else 1.0
        idx = rng.permutation(len(X))
        return lam * X + (1 - lam) * X[idx]

    @staticmethod
    def _smote(X, rng, k):
        n, d = X.shape
        result = X.copy()
        for i in range(n):
            # Find k nearest neighbors
            dists = np.sqrt(np.sum((X - X[i])**2, axis=1))
            dists[i] = np.inf
            neighbors = np.argsort(dists)[:k]
            nn = neighbors[rng.integers(0, k)]
            gap = rng.uniform(0, 1)
            result[i] = X[i] + gap * (X[nn] - X[i])
        return result

    @staticmethod
    def _cutout(X, rng, n_holes, length):
        result = X.copy()
        for _ in range(n_holes):
            feat = rng.integers(0, X.shape[1])
            result[:, feat] = 0.0
        return result

    @staticmethod
    def _flip_h(X, rng, prob):
        result = X.copy()
        for i in range(len(result)):
            if rng.random() < prob:
                result[i] = np.flip(result[i], axis=-1)
        return result

    @staticmethod
    def _flip_v(X, rng, prob):
        result = X.copy()
        for i in range(len(result)):
            if rng.random() < prob:
                result[i] = np.flip(result[i], axis=-2)
        return result

    @staticmethod
    def _rotate90(X, rng, prob):
        result = X.copy()
        for i in range(len(result)):
            if rng.random() < prob:
                result[i] = np.rot90(result[i], k=1, axes=(-2, -1))
        return result

    @staticmethod
    def _crop(X, rng, crop_size):
        n = len(X)
        h, w = X.shape[-2], X.shape[-1]
        ch, cw = int(h * crop_size), int(w * crop_size)
        result = np.zeros_like(X)
        for i in range(n):
            top = rng.integers(0, h - ch + 1)
            left = rng.integers(0, w - cw + 1)
            cropped = X[i, ..., top:top+ch, left:left+cw]
            # Resize back to original size via padding
            pad_top = (h - ch) // 2
            pad_left = (w - cw) // 2
            result[i, ..., pad_top:pad_top+ch, pad_left:pad_left+cw] = cropped
        return result

    def __repr__(self):
        return f"<DataAugmentor mode={self.mode!r} steps={[s.name for s in self.steps]}>"


__all__ = ["DataAugmentor", "AugmentationStep"]
