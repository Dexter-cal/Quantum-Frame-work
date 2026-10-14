"""
qai.vision -- Computer Vision Preprocessing & Image Utilities.
"""
import numpy as np
from typing import Tuple, Any

def resize_image(image: Any, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resizes image matrix to target (H, W)."""
    img = np.array(image, dtype=float)
    H_target, W_target = size
    if img.ndim == 2:
        H_orig, W_orig = img.shape
        row_indices = np.linspace(0, H_orig - 1, H_target).astype(int)
        col_indices = np.linspace(0, W_orig - 1, W_target).astype(int)
        return img[np.ix_(row_indices, col_indices)]
    elif img.ndim == 3:
        H_orig, W_orig, C = img.shape
        row_indices = np.linspace(0, H_orig - 1, H_target).astype(int)
        col_indices = np.linspace(0, W_orig - 1, W_target).astype(int)
        return img[np.ix_(row_indices, col_indices, np.arange(C))]
    else:
        raise ValueError("Image array must be 2D or 3D.")

def normalize_image(image: Any, mean: Tuple[float, ...] = (0.485, 0.456, 0.406), std: Tuple[float, ...] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Normalizes RGB image channels (x - mean) / std."""
    img = np.array(image, dtype=float) / 255.0 if np.max(image) > 1.0 else np.array(image, dtype=float)
    mean_arr = np.array(mean).reshape(1, 1, -1) if img.ndim == 3 else np.array(mean[0])
    std_arr = np.array(std).reshape(1, 1, -1) if img.ndim == 3 else np.array(std[0])
    return (img - mean_arr) / std_arr

def center_crop(image: Any, crop_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """Crops center region of image."""
    img = np.array(image)
    H, W = img.shape[:2]
    cH, cW = crop_size
    start_H = max(0, (H - cH) // 2)
    start_W = max(0, (W - cW) // 2)
    return img[start_H:start_H+cH, start_W:start_W+cW]

def random_flip(image: Any, horizontal: bool = True, p: float = 0.5) -> np.ndarray:
    """Randomly flips image horizontally or vertically."""
    img = np.array(image)
    if np.random.rand() < p:
        return np.fliplr(img) if horizontal else np.flipud(img)
    return img

def image_to_tensor(image: Any) -> np.ndarray:
    """Converts HWC image to CHW NCHW PyTorch-compatible tensor format."""
    img = np.array(image, dtype=float)
    if img.ndim == 3:
        return np.transpose(img, (2, 0, 1))
    return img
