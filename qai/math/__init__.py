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
