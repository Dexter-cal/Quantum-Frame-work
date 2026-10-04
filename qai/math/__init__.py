"""
qai.math -- Comprehensive mathematics, linear algebra, calculus, activations, loss functions,
probability, statistics, information theory, distance metrics, and signal/convolution library.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple, Callable, Optional, Union


# =====================================================================
# 1. LINEAR ALGEBRA
# =====================================================================

def dot(a: Any, b: Any) -> float:
    """Vector dot product."""
    return float(np.dot(a, b))

def matmul(A: Any, B: Any) -> np.ndarray:
    """Matrix multiplication."""
    return np.matmul(A, B)

def transpose(A: Any) -> np.ndarray:
    """Matrix transposition."""
    return np.transpose(A)

def inverse(A: Any) -> np.ndarray:
    """Exact matrix inverse. Raises LinAlgError if singular."""
    return np.linalg.inv(A)

def pseudo_inverse(A: Any) -> np.ndarray:
    """Moore-Penrose pseudo-inverse."""
    return np.linalg.pinv(A)

def determinant(A: Any) -> float:
    """Matrix determinant."""
    return float(np.linalg.det(A))

def trace(A: Any) -> float:
    """Sum of diagonal elements."""
    return float(np.trace(A))

def rank(A: Any) -> int:
    """Matrix rank."""
    return int(np.linalg.matrix_rank(A))

def norm(x: Any, ord: str = "l2") -> float:
    """Vector/matrix norm: l1, l2, max, frobenius."""
    arr = np.array(x, dtype=float)
    if ord == "l1":
        return float(np.sum(np.abs(arr)))
    elif ord in ("l2", "frobenius"):
        return float(np.linalg.norm(arr))
    elif ord == "max":
        return float(np.max(np.abs(arr)))
    else:
        raise ValueError(f"Unknown norm type: {ord}")

def normalize(x: Any, ord: str = "l2") -> np.ndarray:
    """Normalizes vector to unit norm."""
    arr = np.array(x, dtype=float)
    n = norm(arr, ord=ord)
    return arr / (n + 1e-12)

def eigen(A: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Computes eigenvalues and eigenvectors."""
    val, vec = np.linalg.eig(A)
    return val, vec

def svd(A: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Singular Value Decomposition (U, S, Vh)."""
    return np.linalg.svd(A)

def qr_decompose(A: Any) -> Tuple[np.ndarray, np.ndarray]:
    """QR Decomposition (Q, R)."""
    return np.linalg.qr(A)

def cholesky_decompose(A: Any) -> np.ndarray:
    """Cholesky Decomposition L such that A = L L^T."""
    return np.linalg.cholesky(A)

def solve_linear_system(A: Any, b: Any) -> np.ndarray:
    """Solves A x = b."""
    return np.linalg.solve(A, b)

def identity(n: int) -> np.ndarray:
    """Returns n x n Identity matrix."""
    return np.eye(n)

def zeros(shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """Array of zeros."""
    return np.zeros(shape)

def ones(shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """Array of ones."""
    return np.ones(shape)

def reshape(A: Any, shape: Tuple[int, ...]) -> np.ndarray:
    """Reshapes array A."""
    return np.reshape(A, shape)

def concat(arrays: List[Any], axis: int = 0) -> np.ndarray:
    """Concatenates arrays along axis."""
    return np.concatenate(arrays, axis=axis)

def outer(u: Any, v: Any) -> np.ndarray:
    """Outer product of vectors u and v."""
    return np.outer(u, v)

def cross_product(u: Any, v: Any) -> np.ndarray:
    """Vector cross product."""
    return np.cross(u, v)

def trace_of_product(A: Any, B: Any) -> float:
    """Computes Trace(A B)."""
    return float(np.trace(np.matmul(A, B)))


# =====================================================================
# 2. CALCULUS / GRADIENTS
# =====================================================================

def gradient(f: Callable[[np.ndarray], float], x: Any, eps: float = 1e-5) -> np.ndarray:
    """General-purpose central finite difference numerical gradient."""
    x_arr = np.array(x, dtype=float)
    grad = np.zeros_like(x_arr)
    it = np.nditer(x_arr, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = x_arr[idx]
        x_arr[idx] = orig + eps
        f_plus = f(x_arr)
        x_arr[idx] = orig - eps
        f_minus = f(x_arr)
        x_arr[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad

def partial_derivative(f: Callable[[np.ndarray], float], x: Any, var_index: int, eps: float = 1e-5) -> float:
    """Computes partial derivative along a specific variable index."""
    grad = gradient(f, x, eps=eps)
    return float(grad[var_index])

def second_derivative(f: Callable[[float], float], x: float, eps: float = 1e-4) -> float:
    """Computes second derivative f''(x) for 1D scalar function."""
    f_plus = f(x + eps)
    f_mid = f(x)
    f_minus = f(x - eps)
    return float((f_plus - 2 * f_mid + f_minus) / (eps ** 2))

def hessian(f: Callable[[np.ndarray], float], x: Any, eps: float = 1e-4) -> np.ndarray:
    """Computes Hessian matrix of second derivatives."""
    x_arr = np.array(x, dtype=float)
    n = x_arr.size
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                x_p = x_arr.copy(); x_p[i] += eps
                x_m = x_arr.copy(); x_m[i] -= eps
                H[i, j] = (f(x_p) - 2 * f(x_arr) + f(x_m)) / (eps ** 2)
            else:
                x_pp = x_arr.copy(); x_pp[i] += eps; x_pp[j] += eps
                x_pm = x_arr.copy(); x_pm[i] += eps; x_pm[j] -= eps
                x_mp = x_arr.copy(); x_mp[i] -= eps; x_mp[j] += eps
                x_mm = x_arr.copy(); x_mm[i] -= eps; x_mm[j] -= eps
                H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps ** 2)
    return H

def jacobian(f: Callable[[np.ndarray], np.ndarray], x: Any, eps: float = 1e-5) -> np.ndarray:
    """Computes Jacobian matrix for vector-valued functions."""
    x_arr = np.array(x, dtype=float)
    f0 = f(x_arr)
    m = f0.size
    n = x_arr.size
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        x_p = x_arr.copy(); x_p[j] += eps
        x_m = x_arr.copy(); x_m[j] -= eps
        J[:, j] = (f(x_p) - f(x_m)) / (2 * eps)
    return J

def chain_rule(df_dg: float, dg_dx: float) -> float:
    """Computes composite gradient dy/dx = (dy/dg) * (dg/dx)."""
    return float(df_dg * dg_dx)

def numerical_integrate(f: Callable[[float], float], a: float, b: float, n_steps: int = 1000) -> float:
    """Simpson's / Trapezoidal rule numerical integration of 1D scalar function f over [a, b]."""
    x = np.linspace(a, b, n_steps + 1)
    y = np.array([f(val) for val in x])
    return float(np.trapz(y, x))

def mse_gradient(y_true: Any, y_pred: Any) -> np.ndarray:
    """MSE gradient wrt y_pred: 2 * (y_pred - y_true) / N."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    return 2.0 * (y_p - y_t) / y_t.size

def mae_gradient(y_true: Any, y_pred: Any) -> np.ndarray:
    """MAE gradient wrt y_pred: sign(y_pred - y_true) / N."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    return np.sign(y_p - y_t) / y_t.size

def huber_gradient(y_true: Any, y_pred: Any, delta: float = 1.0) -> np.ndarray:
    """Huber loss gradient wrt y_pred."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    diff = y_p - y_t
    abs_diff = np.abs(diff)
    grad = np.where(abs_diff <= delta, diff, delta * np.sign(diff))
    return grad / y_t.size


# =====================================================================
# 3. ACTIVATION FUNCTIONS & DERIVATIVES
# =====================================================================

def sigmoid(x: Any) -> np.ndarray:
    """Sigmoid activation 1 / (1 + e^-x)."""
    arr = np.array(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -50, 50)))

def sigmoid_derivative(x: Any) -> np.ndarray:
    """Sigmoid derivative: s * (1 - s)."""
    s = sigmoid(x)
    return s * (1.0 - s)

def tanh(x: Any) -> np.ndarray:
    """Hyperbolic tangent activation."""
    return np.tanh(np.array(x, dtype=float))

def tanh_derivative(x: Any) -> np.ndarray:
    """Tanh derivative: 1 - tanh(x)^2."""
    t = tanh(x)
    return 1.0 - (t ** 2)

def relu(x: Any) -> np.ndarray:
    """ReLU activation max(0, x)."""
    return np.maximum(0.0, np.array(x, dtype=float))

def relu_derivative(x: Any) -> np.ndarray:
    """ReLU derivative: 1 if x > 0 else 0."""
    return np.where(np.array(x, dtype=float) > 0, 1.0, 0.0)

def leaky_relu(x: Any, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation."""
    arr = np.array(x, dtype=float)
    return np.where(arr > 0, arr, alpha * arr)

def leaky_relu_derivative(x: Any, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU derivative."""
    return np.where(np.array(x, dtype=float) > 0, 1.0, alpha)

def elu(x: Any, alpha: float = 1.0) -> np.ndarray:
    """Exponential Linear Unit (ELU)."""
    arr = np.array(x, dtype=float)
    return np.where(arr > 0, arr, alpha * (np.exp(np.clip(arr, -50, 50)) - 1.0))

def elu_derivative(x: Any, alpha: float = 1.0) -> np.ndarray:
    """ELU derivative."""
    arr = np.array(x, dtype=float)
    return np.where(arr > 0, 1.0, elu(arr, alpha) + alpha)

def gelu(x: Any) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU)."""
    arr = np.array(x, dtype=float)
    return 0.5 * arr * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (arr + 0.044715 * (arr ** 3))))

def gelu_derivative(x: Any) -> np.ndarray:
    """GELU numerical derivative."""
    return gradient(gelu, x)

def swish(x: Any, beta: float = 1.0) -> np.ndarray:
    """Swish activation: x * sigmoid(beta * x)."""
    arr = np.array(x, dtype=float)
    return arr * sigmoid(beta * arr)

def swish_derivative(x: Any, beta: float = 1.0) -> np.ndarray:
    """Swish derivative."""
    arr = np.array(x, dtype=float)
    s = sigmoid(beta * arr)
    sw = swish(arr, beta)
    return beta * sw + s * (1.0 - beta * sw)

def softmax(x: Any, axis: int = -1) -> np.ndarray:
    """Softmax probability distribution."""
    arr = np.array(x, dtype=float)
    e_x = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def softmax_derivative(x: Any) -> np.ndarray:
    """Softmax Jacobian matrix."""
    s = softmax(x)
    return np.diag(s) - np.outer(s, s)

def softplus(x: Any) -> np.ndarray:
    """Softplus activation log(1 + e^x)."""
    arr = np.array(x, dtype=float)
    return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0.0)

def softplus_derivative(x: Any) -> np.ndarray:
    """Softplus derivative is Sigmoid."""
    return sigmoid(x)

def linear(x: Any) -> np.ndarray:
    """Linear pass-through activation."""
    return np.array(x, dtype=float)

def linear_derivative(x: Any) -> np.ndarray:
    """Linear derivative is 1."""
    return np.ones_like(np.array(x, dtype=float))


# =====================================================================
# 4. LOSS FUNCTIONS & GRADIENTS
# =====================================================================

def mse(y_true: Any, y_pred: Any) -> float:
    """Mean Squared Error loss."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    return float(np.mean((y_t - y_p) ** 2))

def mae(y_true: Any, y_pred: Any) -> float:
    """Mean Absolute Error loss."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_t - y_p)))

def huber_loss(y_true: Any, y_pred: Any, delta: float = 1.0) -> float:
    """Huber Loss for robust regression."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    err = np.abs(y_t - y_p)
    loss = np.where(err <= delta, 0.5 * (err ** 2), delta * (err - 0.5 * delta))
    return float(np.mean(loss))

def hinge_loss(y_true: Any, y_pred_scores: Any) -> float:
    """Hinge loss max(0, 1 - y_true * y_pred_scores) for SVM (y in {-1, +1})."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred_scores, dtype=float)
    return float(np.mean(np.maximum(0.0, 1.0 - y_t * y_p)))

def focal_loss(y_true: Any, y_pred_probs: Any, gamma: float = 2.0, alpha: float = 0.25) -> float:
    """Focal Loss for class imbalance."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.clip(np.array(y_pred_probs, dtype=float), 1e-12, 1.0 - 1e-12)
    pt = np.where(y_t == 1, y_p, 1 - y_p)
    loss = -alpha * ((1 - pt) ** gamma) * np.log(pt)
    return float(np.mean(loss))

def cross_entropy(y_true: Any, y_pred_probs: Any) -> float:
    """Categorical Cross Entropy Loss."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.clip(np.array(y_pred_probs, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(np.sum(y_t * np.log(y_p), axis=-1 if y_t.ndim > 1 else 0)))

def binary_cross_entropy(y_true: Any, y_pred_probs: Any) -> float:
    """Binary Cross Entropy Loss."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.clip(np.array(y_pred_probs, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p)))

def kl_divergence_loss(y_true: Any, y_pred_probs: Any) -> float:
    """KL Divergence loss."""
    y_t = np.clip(np.array(y_true, dtype=float), 1e-12, 1.0)
    y_p = np.clip(np.array(y_pred_probs, dtype=float), 1e-12, 1.0)
    return float(np.sum(y_t * np.log(y_t / y_p)))


# =====================================================================
# 5. PROBABILITY & STATISTICS
# =====================================================================

def mean(x: Any) -> float:
    """Arithmetic mean."""
    return float(np.mean(x))

def median(x: Any) -> float:
    """Median value."""
    return float(np.median(x))

def mode(x: Any) -> Any:
    """Mode (most frequent value)."""
    vals, counts = np.unique(x, return_counts=True)
    return vals[np.argmax(counts)]

def variance(x: Any, ddof: int = 0) -> float:
    """Sample or population variance."""
    return float(np.var(x, ddof=ddof))

def std(x: Any, ddof: int = 0) -> float:
    """Standard deviation."""
    return float(np.std(x, ddof=ddof))

def covariance(x: Any, y: Any) -> float:
    """Covariance between x and y."""
    return float(np.cov(x, y)[0, 1])

def covariance_matrix(X: Any) -> np.ndarray:
    """Covariance matrix of dataset X."""
    return np.cov(np.array(X, dtype=float), rowvar=False)

def correlation(x: Any, y: Any) -> float:
    """Pearson correlation coefficient."""
    return float(np.corrcoef(x, y)[0, 1])

def percentile(x: Any, q: float) -> float:
    """Percentile value q in [0, 100]."""
    return float(np.percentile(x, q))

def z_score(x: Any) -> np.ndarray:
    """Z-score normalization (x - mean) / std."""
    arr = np.array(x, dtype=float)
    s = np.std(arr)
    return (arr - np.mean(arr)) / (s if s > 0 else 1.0)

def skewness(x: Any) -> float:
    """Sample skewness."""
    return float(stats.skew(x))

def kurtosis(x: Any) -> float:
    """Kurtosis measure."""
    return float(stats.kurtosis(x))

def gaussian_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Gaussian Probability Density Function."""
    return float(stats.norm.pdf(x, loc=mu, scale=sigma))

def bernoulli_pmf(k: int, p: float) -> float:
    """Bernoulli Probability Mass Function."""
    return float(stats.bernoulli.pmf(k, p))

def binomial_pmf(k: int, n: int, p: float) -> float:
    """Binomial PMF."""
    return float(stats.binom.pmf(k, n, p))

def poisson_pmf(k: int, mu: float) -> float:
    """Poisson PMF."""
    return float(stats.poisson.pmf(k, mu))

def uniform_pdf(x: float, low: float = 0.0, high: float = 1.0) -> float:
    """Uniform PDF."""
    return float(stats.uniform.pdf(x, loc=low, scale=high - low))

def exponential_pdf(x: float, scale: float = 1.0) -> float:
    """Exponential PDF."""
    return float(stats.expon.pdf(x, scale=scale))

def bayes_theorem(prior_a: float, likelihood_b_given_a: float, marginal_b: float) -> float:
    """Posterior P(A|B) = P(B|A) * P(A) / P(B)."""
    if marginal_b <= 0:
        raise ValueError("Marginal probability marginal_b must be > 0")
    return float((likelihood_b_given_a * prior_a) / marginal_b)

def joint_probability(p_a: float, p_b_given_a: float) -> float:
    """Joint probability P(A, B) = P(B|A) * P(A)."""
    return float(p_a * p_b_given_a)

def marginal_probability(joint_probabilities: Any) -> float:
    """Marginal probability sum across joint probabilities."""
    return float(np.sum(joint_probabilities))

def conditional_probability(p_joint: float, p_given: float) -> float:
    """Conditional probability P(A|B) = P(A, B) / P(B)."""
    if p_given <= 0:
        raise ValueError("p_given must be > 0")
    return float(p_joint / p_given)

def expectation(values: Any, probabilities: Any) -> float:
    """Expected value E[X] = sum(x * P(x))."""
    v = np.array(values, dtype=float)
    p = np.array(probabilities, dtype=float)
    return float(np.sum(v * p))

def maximum_likelihood_estimate(data: Any, dist_type: str = "gaussian") -> Dict[str, float]:
    """MLE parameter estimation for gaussian or exponential."""
    arr = np.array(data, dtype=float)
    if dist_type == "gaussian":
        return {"mu": float(np.mean(arr)), "sigma": float(np.std(arr))}
    elif dist_type == "exponential":
        return {"scale": float(np.mean(arr))}
    else:
        raise ValueError(f"Unsupported dist_type: {dist_type}")

def class_prior(labels: Any) -> Dict[Any, float]:
    """Computes class prior probabilities P(Y=c)."""
    arr = np.array(labels)
    unique, counts = np.unique(arr, return_counts=True)
    total = len(arr)
    return {k: float(v / total) for k, v in zip(unique, counts)}

def confidence_interval(data: Any, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes confidence interval for mean."""
    arr = np.array(data, dtype=float)
    mean_val = np.mean(arr)
    sem = stats.sem(arr)
    h = sem * stats.t.ppf((1 + confidence) / 2., len(arr) - 1)
    return float(mean_val - h), float(mean_val + h)

def sample(a: Any, size: int = 1, replace: bool = True) -> np.ndarray:
    """Random sampling utility."""
    return np.random.choice(a, size=size, replace=replace)

def shuffle(a: Any) -> np.ndarray:
    """Randomly shuffles array."""
    arr = np.array(a).copy()
    np.random.shuffle(arr)
    return arr


# =====================================================================
# 6. INFORMATION THEORY
# =====================================================================

def entropy(probabilities: Any) -> float:
    """Shannon Entropy H(P) = -sum(p * log2(p))."""
    p = np.array(probabilities, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def info_cross_entropy(p: Any, q: Any) -> float:
    """Information-theoretic cross-entropy H(P, Q) = -sum(p * log2(q))."""
    p_arr = np.array(p, dtype=float)
    q_arr = np.clip(np.array(q, dtype=float), 1e-12, 1.0)
    return float(-np.sum(p_arr * np.log2(q_arr)))

def kl_divergence(p: Any, q: Any) -> float:
    """Kullback-Leibler Divergence D_KL(P || Q)."""
    p_arr = np.clip(np.array(p, dtype=float), 1e-12, 1.0)
    q_arr = np.clip(np.array(q, dtype=float), 1e-12, 1.0)
    return float(np.sum(p_arr * np.log2(p_arr / q_arr)))

def mutual_information(x: Any, y: Any) -> float:
    """Mutual information I(X; Y) between two discrete variables."""
    from sklearn.metrics import mutual_info_score
    return float(mutual_info_score(x, y))

def information_gain(y_parent: Any, y_splits: List[Any]) -> float:
    """Information Gain IG = H(Parent) - sum(weight * H(Child))."""
    parent_probs = list(class_prior(y_parent).values())
    h_parent = entropy(parent_probs)
    n_total = len(y_parent)
    weighted_child_h = 0.0
    for child in y_splits:
        if len(child) > 0:
            child_probs = list(class_prior(child).values())
            weighted_child_h += (len(child) / n_total) * entropy(child_probs)
    return float(h_parent - weighted_child_h)

def gini_impurity(labels: Any) -> float:
    """Gini Impurity 1 - sum(p_i^2) used by DecisionTree and RandomForest."""
    p = np.array(list(class_prior(labels).values()))
    return float(1.0 - np.sum(p ** 2))

def perplexity(probabilities: Any) -> float:
    """Perplexity 2^H(P)."""
    return float(2.0 ** entropy(probabilities))


# =====================================================================
# 7. DISTANCE & SIMILARITY METRICS
# =====================================================================

def euclidean_distance(x: Any, y: Any) -> float:
    """Euclidean L2 distance."""
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    return float(np.linalg.norm(x_arr - y_arr))

def manhattan_distance(x: Any, y: Any) -> float:
    """Manhattan L1 distance."""
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    return float(np.sum(np.abs(x_arr - y_arr)))

def cosine_similarity(x: Any, y: Any) -> float:
    """Cosine similarity dot(x, y) / (||x|| * ||y||)."""
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    denom = (np.linalg.norm(x_arr) * np.linalg.norm(y_arr)) + 1e-12
    return float(np.dot(x_arr, y_arr) / denom)

def mahalanobis_distance(x: Any, y: Any, cov_inv: Any) -> float:
    """Mahalanobis distance sqrt((x-y)^T Cov^-1 (x-y))."""
    delta = np.array(x, dtype=float) - np.array(y, dtype=float)
    inv = np.array(cov_inv, dtype=float)
    return float(np.sqrt(np.dot(np.dot(delta, inv), delta)))

def hamming_distance(x: Any, y: Any) -> float:
    """Hamming distance fraction of disagreeing positions."""
    x_arr = np.array(x)
    y_arr = np.array(y)
    return float(np.mean(x_arr != y_arr))


# =====================================================================
# 8. SIGNAL / CONVOLUTION MATH
# =====================================================================

def convolve(signal: Any, kernel: Any, mode: str = "same") -> np.ndarray:
    """1D or 2D discrete convolution."""
    s = np.array(signal, dtype=float)
    k = np.array(kernel, dtype=float)
    if s.ndim == 1 and k.ndim == 1:
        return np.convolve(s, k, mode=mode)
    elif s.ndim == 2 and k.ndim == 2:
        from scipy.signal import convolve2d
        return convolve2d(s, k, mode=mode)
    else:
        raise ValueError("Dimensions of signal and kernel must match (1D or 2D)")

def pooling(matrix: Any, pool_size: Tuple[int, int] = (2, 2), mode: str = "max") -> np.ndarray:
    """2D spatial pooling (max or average)."""
    m = np.array(matrix, dtype=float)
    ph, pw = pool_size
    H, W = m.shape
    out_H = H // ph
    out_W = W // pw
    out = np.zeros((out_H, out_W), dtype=float)
    for i in range(out_H):
        for j in range(out_W):
            region = m[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            out[i, j] = np.max(region) if mode == "max" else np.mean(region)
    return out

def fourier_transform(signal: Any) -> np.ndarray:
    """Fast Fourier Transform (FFT)."""
    return np.fft.fft(np.array(signal, dtype=float))

def inverse_fourier_transform(spectrum: Any) -> np.ndarray:
    """Inverse Fast Fourier Transform (IFFT)."""
    return np.fft.ifft(spectrum)


# =====================================================================
# REFERENCE & HELPER METHODS
# =====================================================================

_CATEGORIES = {
    "linear_algebra": ["dot", "matmul", "transpose", "inverse", "pseudo_inverse", "determinant", "trace", "rank", "norm", "normalize", "eigen", "svd", "qr_decompose", "cholesky_decompose", "solve_linear_system", "identity", "zeros", "ones", "reshape", "concat", "outer", "cross_product", "trace_of_product"],
    "calculus_gradients": ["gradient", "partial_derivative", "second_derivative", "hessian", "jacobian", "chain_rule", "numerical_integrate", "mse_gradient", "mae_gradient", "huber_gradient"],
    "activations": ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "gelu", "swish", "softmax", "softplus", "linear"],
    "loss_functions": ["mse", "mae", "huber_loss", "hinge_loss", "focal_loss", "cross_entropy", "binary_cross_entropy", "kl_divergence_loss"],
    "probability_statistics": ["mean", "median", "mode", "variance", "std", "covariance", "covariance_matrix", "correlation", "percentile", "z_score", "skewness", "kurtosis", "gaussian_pdf", "bernoulli_pmf", "binomial_pmf", "poisson_pmf", "uniform_pdf", "exponential_pdf", "bayes_theorem", "joint_probability", "marginal_probability", "conditional_probability", "expectation", "maximum_likelihood_estimate", "class_prior", "confidence_interval", "sample", "shuffle"],
    "information_theory": ["entropy", "info_cross_entropy", "kl_divergence", "mutual_information", "information_gain", "gini_impurity", "perplexity"],
    "distance_similarity": ["euclidean_distance", "manhattan_distance", "cosine_similarity", "mahalanobis_distance", "hamming_distance"],
    "signal_convolution": ["convolve", "pooling", "fourier_transform", "inverse_fourier_transform"]
}

def help(method_name: Optional[str] = None) -> str:
    """Returns overview of all qai.math methods or detailed signature + example if method_name is provided."""
    if method_name is None:
        lines = ["qai.math -- Available Categories and Methods:"]
        for cat, funcs in _CATEGORIES.items():
            lines.append(f"  [{cat}]: " + ", ".join(funcs))
        return "\n".join(lines)
    else:
        if method_name in globals() and callable(globals()[method_name]):
            fn = globals()[method_name]
            return f"Method: qai.math.{method_name}\nDocstring: {fn.__doc__}\nExample: qai.math.{method_name}(...)"
        return f"Method '{method_name}' not found in qai.math."

def list_by_category(category_name: str) -> List[str]:
    """Lists all method names in a given category."""
    if category_name in _CATEGORIES:
        return _CATEGORIES[category_name]
    raise ValueError(f"Unknown category '{category_name}'. Valid categories: {list(_CATEGORIES.keys())}")
