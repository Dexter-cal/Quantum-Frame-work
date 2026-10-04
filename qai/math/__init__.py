"""
qai.math -- Comprehensive mathematical, activation, distance, and matrix operations library for AI/ML.
"""
from __future__ import annotations
import numpy as np


# --- Activation Functions ---
def sigmoid(x):
    """Sigmoid activation function: 1 / (1 + exp(-x))."""
    x_arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x_arr, -500, 500)))


def relu(x):
    """Rectified Linear Unit activation: max(0, x)."""
    x_arr = np.asarray(x, dtype=float)
    return np.maximum(0.0, x_arr)


def softmax(x, axis=-1):
    """Softmax probability distribution function."""
    x_arr = np.asarray(x, dtype=float)
    e_x = np.exp(x_arr - np.max(x_arr, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def gelu(x):
    """Gaussian Error Linear Unit (GELU) activation."""
    x_arr = np.asarray(x, dtype=float)
    return 0.5 * x_arr * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_arr + 0.044715 * np.power(x_arr, 3))))


def swish(x, beta=1.0):
    """Swish activation function: x * sigmoid(beta * x)."""
    x_arr = np.asarray(x, dtype=float)
    return x_arr * sigmoid(beta * x_arr)


def tanh(x):
    """Hyperbolic tangent activation function."""
    x_arr = np.asarray(x, dtype=float)
    return np.tanh(x_arr)


# --- Distance Metrics ---
def euclidean_distance(a, b) -> float:
    """Computes Euclidean distance between two vectors or matrices."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a_arr - b_arr))


def cosine_similarity(a, b) -> float:
    """Computes Cosine Similarity between two vectors: (a · b) / (||a|| ||b||)."""
    a_arr, b_arr = np.ravel(np.asarray(a, dtype=float)), np.ravel(np.asarray(b, dtype=float))
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def manhattan_distance(a, b) -> float:
    """Computes Manhattan (L1) distance between two vectors."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sum(np.abs(a_arr - b_arr)))


def minkowski_distance(a, b, p=3) -> float:
    """Computes Minkowski distance of order p between two vectors."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sum(np.abs(a_arr - b_arr) ** p) ** (1.0 / p))


# --- Matrix & Linear Algebra Operations ---
def dot(a, b):
    """Matrix / vector dot product."""
    return np.dot(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def matrix_inverse(a):
    """Computes matrix inverse or Moore-Penrose pseudo-inverse."""
    a_arr = np.asarray(a, dtype=float)
    return np.linalg.pinv(a_arr)


def eigenvalues(a):
    """Computes eigenvalues and eigenvectors of a real matrix."""
    a_arr = np.asarray(a, dtype=float)
    vals, vecs = np.linalg.eig(a_arr)
    return vals, vecs


def singular_value_decomposition(a):
    """Computes Singular Value Decomposition (SVD): U, S, Vt."""
    a_arr = np.asarray(a, dtype=float)
    return np.linalg.svd(a_arr, full_matrices=False)


# --- Loss & Statistical Functions ---
def mean_squared_error(y_true, y_pred) -> float:
    """Computes Mean Squared Error (MSE)."""
    y_t, y_p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean((y_t - y_p) ** 2))


def cross_entropy_loss(y_true, y_pred_proba, eps=1e-15) -> float:
    """Computes categorical cross-entropy loss."""
    y_t, y_p = np.asarray(y_true, dtype=float), np.clip(np.asarray(y_pred_proba, dtype=float), eps, 1 - eps)
    return float(-np.sum(y_t * np.log(y_p)) / len(y_t))


# Advanced Loss Functions
def huber_loss(y_true: Any, y_pred: Any, delta: float = 1.0) -> float:
    """Computes Huber Loss for robust regression."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    err = np.abs(y_t - y_p)
    loss = np.where(err <= delta, 0.5 * (err ** 2), delta * (err - 0.5 * delta))
    return float(np.mean(loss))


def focal_loss(y_true: Any, y_pred_probs: Any, gamma: float = 2.0, alpha: float = 0.25) -> float:
    """Computes Focal Loss for imbalanced classification."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.clip(np.array(y_pred_probs, dtype=float), 1e-12, 1.0 - 1e-12)
    pt = np.where(y_t == 1, y_p, 1 - y_p)
    loss = -alpha * ((1 - pt) ** gamma) * np.log(pt)
    return float(np.mean(loss))


def triplet_loss(anchor: Any, positive: Any, negative: Any, margin: float = 1.0) -> float:
    """Computes Triplet Loss for metric embedding learning."""
    a = np.array(anchor, dtype=float)
    p = np.array(positive, dtype=float)
    n = np.array(negative, dtype=float)
    pos_dist = np.sum((a - p) ** 2, axis=-1)
    neg_dist = np.sum((a - n) ** 2, axis=-1)
    loss = np.maximum(0.0, pos_dist - neg_dist + margin)
    return float(np.mean(loss))


# Divergences & Distance Metrics
def kl_divergence(p: Any, q: Any) -> float:
    """Computes Kullback-Leibler (KL) Divergence."""
    p_arr = np.clip(np.array(p, dtype=float), 1e-12, 1.0)
    q_arr = np.clip(np.array(q, dtype=float), 1e-12, 1.0)
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def js_divergence(p: Any, q: Any) -> float:
    """Computes Jensen-Shannon (JS) Divergence."""
    p_arr = np.array(p, dtype=float)
    q_arr = np.array(q, dtype=float)
    m = 0.5 * (p_arr + q_arr)
    return float(0.5 * kl_divergence(p_arr, m) + 0.5 * kl_divergence(q_arr, m))


def chebyshev_distance(x: Any, y: Any) -> float:
    """Computes Chebyshev distance (infinity norm)."""
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    return float(np.max(np.abs(x_arr - y_arr)))


def canberra_distance(x: Any, y: Any) -> float:
    """Computes Canberra distance metric."""
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    denom = np.abs(x_arr) + np.abs(y_arr) + 1e-12
    return float(np.sum(np.abs(x_arr - y_arr) / denom))


def braycurtis_distance(x: Any, y: Any) -> float:
    """Computes Bray-Curtis distance metric."""
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    num = np.sum(np.abs(x_arr - y_arr))
    denom = np.sum(np.abs(x_arr + y_arr)) + 1e-12
    return float(num / denom)
