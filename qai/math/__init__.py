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


# =====================================================================
# 9. TRANSFORMER / ATTENTION MECHANICS
# =====================================================================

def scaled_dot_product_attention(Q: Any, K: Any, V: Any, mask: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled Dot-Product Attention: Softmax(Q K^T / sqrt(d_k)) V."""
    q_arr = np.array(Q, dtype=float)
    k_arr = np.array(K, dtype=float)
    v_arr = np.array(V, dtype=float)
    d_k = q_arr.shape[-1]
    scores = np.matmul(q_arr, np.swapaxes(k_arr, -1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    attn_weights = softmax(scores, axis=-1)
    output = np.matmul(attn_weights, v_arr)
    return output, attn_weights

def query_key_value_projection(x: Any, W_q: Any, W_k: Any, W_v: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projects input x into Query, Key, and Value matrices."""
    x_arr = np.array(x, dtype=float)
    Q = np.matmul(x_arr, W_q)
    K = np.matmul(x_arr, W_k)
    V = np.matmul(x_arr, W_v)
    return Q, K, V

def multi_head_attention(Q: Any, K: Any, V: Any, W_o: Any, n_heads: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-Head Attention mechanism."""
    output, weights = scaled_dot_product_attention(Q, K, V)
    projected = np.matmul(output, W_o)
    return projected, weights

def masked_attention(Q: Any, K: Any, V: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Causal masked self-attention."""
    seq_len = np.array(Q).shape[-2]
    mask = np.tril(np.ones((seq_len, seq_len)))
    return scaled_dot_product_attention(Q, K, V, mask=mask)

def cross_attention(Q_decoder: Any, K_encoder: Any, V_encoder: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-attention mechanism between encoder and decoder."""
    return scaled_dot_product_attention(Q_decoder, K_encoder, V_encoder)

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal Positional Encoding PE(pos, 2i) = sin(pos/10000^(2i/d_model))."""
    pe = np.zeros((seq_len, d_model), dtype=float)
    position = np.arange(0, seq_len, dtype=float)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=float) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def learned_positional_embedding(seq_len: int, d_model: int) -> np.ndarray:
    """Randomly initialized learned positional embedding matrix."""
    return np.random.randn(seq_len, d_model) * 0.02

def layer_norm(x: Any, gamma: Optional[Any] = None, beta: Optional[Any] = None, eps: float = 1e-5) -> np.ndarray:
    """Layer Normalization across last dimension."""
    x_arr = np.array(x, dtype=float)
    mean_val = np.mean(x_arr, axis=-1, keepdims=True)
    var_val = np.var(x_arr, axis=-1, keepdims=True)
    x_norm = (x_arr - mean_val) / np.sqrt(var_val + eps)
    if gamma is not None:
        x_norm *= gamma
    if beta is not None:
        x_norm += beta
    return x_norm

def residual_connection(x: Any, sublayer_out: Any) -> np.ndarray:
    """Residual Add connection: x + sublayer_out."""
    return np.array(x, dtype=float) + np.array(sublayer_out, dtype=float)

def feed_forward_block(x: Any, W1: Any, b1: Any, W2: Any, b2: Any) -> np.ndarray:
    """Position-wise Feed-Forward Network: Relu(x W1 + b1) W2 + b2."""
    h = relu(np.matmul(x, W1) + b1)
    return np.matmul(h, W2) + b2

def attention_dropout(attn_weights: Any, p: float = 0.1, training: bool = True) -> np.ndarray:
    """Applies dropout mask to attention weights."""
    arr = np.array(attn_weights, dtype=float)
    if not training or p <= 0:
        return arr
    mask = (np.random.rand(*arr.shape) >= p).astype(float)
    return (arr * mask) / (1.0 - p)


# =====================================================================
# 10. DIFFUSION MODEL MATH
# =====================================================================

def forward_noise_step(x_zero: Any, noise: Any, beta_t: float) -> np.ndarray:
    """Forward single-step diffusion noise addition: sqrt(1 - beta_t) * x_0 + sqrt(beta_t) * eps."""
    x0 = np.array(x_zero, dtype=float)
    eps = np.array(noise, dtype=float)
    return np.sqrt(1.0 - beta_t) * x0 + np.sqrt(beta_t) * eps

def forward_noise_closed_form(x_zero: Any, noise: Any, alpha_bar_t: float) -> np.ndarray:
    """Closed-form forward diffusion at step t: sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps."""
    x0 = np.array(x_zero, dtype=float)
    eps = np.array(noise, dtype=float)
    return np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * eps

def noise_schedule(n_steps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, mode: str = "linear") -> np.ndarray:
    """Generates linear or cosine variance schedule beta_1...beta_T."""
    if mode == "linear":
        return np.linspace(beta_start, beta_end, n_steps)
    elif mode == "cosine":
        steps = np.arange(n_steps + 1, dtype=float)
        f_t = np.cos(((steps / n_steps) + 0.008) / 1.008 * (math.pi / 2.0)) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.999)
    else:
        raise ValueError(f"Unknown noise schedule mode: {mode}")

def alpha_bar(betas: Any) -> np.ndarray:
    """Computes cumulative product alpha_bar_t = prod(1 - beta_i)."""
    b_arr = np.array(betas, dtype=float)
    return np.cumprod(1.0 - b_arr)

def reverse_denoise_step(x_t: Any, predicted_noise: Any, beta_t: float, alpha_bar_t: float, alpha_bar_prev: float) -> np.ndarray:
    """DDPM reverse step x_{t-1} estimation."""
    xt = np.array(x_t, dtype=float)
    p_eps = np.array(predicted_noise, dtype=float)
    coeff = beta_t / np.sqrt(1.0 - alpha_bar_t)
    mean = (1.0 / np.sqrt(1.0 - beta_t)) * (xt - coeff * p_eps)
    return mean

def score_function(predicted_noise: Any, sigma_t: float) -> np.ndarray:
    """Computes score function grad_x log p(x) = -predicted_noise / sigma_t."""
    p_eps = np.array(predicted_noise, dtype=float)
    return -p_eps / (sigma_t + 1e-12)

def denoising_score_matching_loss(predicted_noise: Any, target_noise: Any) -> float:
    """Denoising Score Matching MSE Loss."""
    return mse(target_noise, predicted_noise)

def ddim_sample_step(x_t: Any, predicted_noise: Any, alpha_bar_t: float, alpha_bar_prev: float, eta: float = 0.0) -> np.ndarray:
    """Deterministic DDIM sampling step."""
    xt = np.array(x_t, dtype=float)
    eps = np.array(predicted_noise, dtype=float)
    pred_x0 = (xt - np.sqrt(1.0 - alpha_bar_t) * eps) / np.sqrt(alpha_bar_t)
    dir_xt = np.sqrt(1.0 - alpha_bar_prev - eta**2) * eps
    return np.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

def variance_schedule(betas: Any) -> np.ndarray:
    """Calculates posterior variance beta_tilde_t = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)."""
    b = np.array(betas, dtype=float)
    a_bar = alpha_bar(b)
    a_bar_prev = np.append(1.0, a_bar[:-1])
    return b * (1.0 - a_bar_prev) / (1.0 - a_bar + 1e-12)

def latent_encode(x: Any, scale_factor: float = 0.18215) -> np.ndarray:
    """Encodes image tensor into scaled latent representation."""
    return np.array(x, dtype=float) * scale_factor

def latent_decode(z: Any, scale_factor: float = 0.18215) -> np.ndarray:
    """Decodes latent representation back into pixel space."""
    return np.array(z, dtype=float) / scale_factor


# =====================================================================
# 11. REINFORCEMENT LEARNING MATH
# =====================================================================

def bellman_equation(reward: float, gamma: float, next_value: float) -> float:
    """Bellman Equation for expected return: R + gamma * V(s')."""
    return float(reward + gamma * next_value)

def discounted_return(rewards: Any, gamma: float = 0.99) -> np.ndarray:
    """Computes discounted cumulative returns G_t = sum_{k=0} gamma^k R_{t+k}."""
    r_arr = np.array(rewards, dtype=float)
    returns = np.zeros_like(r_arr)
    running_add = 0.0
    for t in reversed(range(len(r_arr))):
        running_add = r_arr[t] + gamma * running_add
        returns[t] = running_add
    return returns

def td_error(reward: float, gamma: float, next_value: float, current_value: float) -> float:
    """Temporal Difference (TD) error: R + gamma * V(s') - V(s)."""
    return float(reward + gamma * next_value - current_value)

def advantage_function(td_target: float, baseline_value: float) -> float:
    """Advantage Function A(s, a) = TD_target - V(s)."""
    return float(td_target - baseline_value)

def policy_gradient(action_log_probs: Any, advantages: Any) -> float:
    """Policy Gradient Loss: -mean(log_prob * advantage)."""
    lp = np.array(action_log_probs, dtype=float)
    adv = np.array(advantages, dtype=float)
    return float(-np.mean(lp * adv))

def q_value_update(q_old: float, reward: float, gamma: float, max_next_q: float, alpha: float = 0.1) -> float:
    """Q-learning Bellman update: Q_new = Q_old + alpha * (R + gamma * max(Q') - Q_old)."""
    return float(q_old + alpha * (reward + gamma * max_next_q - q_old))


# =====================================================================
# 12. OPTIMIZER MATH
# =====================================================================

def sgd_update(param: Any, grad: Any, lr: float = 0.01) -> np.ndarray:
    """Stochastic Gradient Descent update: param - lr * grad."""
    p = np.array(param, dtype=float)
    g = np.array(grad, dtype=float)
    return p - lr * g

def momentum_update(param: Any, grad: Any, velocity: Any, lr: float = 0.01, momentum: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Momentum SGD update."""
    p = np.array(param, dtype=float)
    g = np.array(grad, dtype=float)
    v = momentum * np.array(velocity, dtype=float) + lr * g
    return p - v, v

def rmsprop_update(param: Any, grad: Any, square_avg: Any, lr: float = 0.01, alpha: float = 0.99, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """RMSprop optimizer update."""
    p = np.array(param, dtype=float)
    g = np.array(grad, dtype=float)
    s = alpha * np.array(square_avg, dtype=float) + (1.0 - alpha) * (g ** 2)
    p_new = p - (lr / (np.sqrt(s) + eps)) * g
    return p_new, s

def adam_update(param: Any, grad: Any, m: Any, v: Any, t: int, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adam optimizer update with bias correction."""
    p = np.array(param, dtype=float)
    g = np.array(grad, dtype=float)
    m_new = beta1 * np.array(m, dtype=float) + (1.0 - beta1) * g
    v_new = beta2 * np.array(v, dtype=float) + (1.0 - beta2) * (g ** 2)
    m_hat = m_new / (1.0 - beta1 ** t)
    v_hat = v_new / (1.0 - beta2 ** t)
    p_new = p - (lr / (np.sqrt(v_hat) + eps)) * m_hat
    return p_new, m_new, v_new

def learning_rate_decay(lr_initial: float, epoch: int, decay_rate: float = 0.1) -> float:
    """Exponential learning rate decay: lr = lr_0 / (1 + decay_rate * epoch)."""
    return float(lr_initial / (1.0 + decay_rate * epoch))

def gradient_clipping(grad: Any, max_norm: float = 1.0) -> np.ndarray:
    """Clips gradient tensor to maximum L2 norm."""
    g = np.array(grad, dtype=float)
    norm_g = np.linalg.norm(g)
    if norm_g > max_norm and norm_g > 0:
        return g * (max_norm / norm_g)
    return g


# =====================================================================
# 13. GENERATIVE MODELS (GAN / VAE)
# =====================================================================

def discriminator_loss(real_scores: Any, fake_scores: Any) -> float:
    """Standard GAN Discriminator Loss: -mean(log(D(x)) + log(1 - D(G(z))))."""
    r = np.clip(np.array(real_scores, dtype=float), 1e-12, 1.0 - 1e-12)
    f = np.clip(np.array(fake_scores, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(np.log(r) + np.log(1.0 - f)))

def generator_loss(fake_scores: Any) -> float:
    """Standard GAN Generator Loss: -mean(log(D(G(z))))."""
    f = np.clip(np.array(fake_scores, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(np.log(f)))

def minimax_objective(d_real: float, d_fake: float) -> float:
    """Minimax GAN Objective Value."""
    return float(math.log(max(1e-12, d_real)) + math.log(max(1e-12, 1.0 - d_fake)))

def wasserstein_distance(p: Any, q: Any) -> float:
    """Wasserstein-1 Distance (Earth Mover's Distance)."""
    return float(stats.wasserstein_distance(np.array(p, dtype=float).ravel(), np.array(q, dtype=float).ravel()))

def reconstruction_loss(x_true: Any, x_reconstructed: Any, mode: str = "mse") -> float:
    """VAE Reconstruction loss (MSE or BCE)."""
    if mode == "mse":
        return mse(x_true, x_reconstructed)
    elif mode == "bce":
        return binary_cross_entropy(x_true, x_reconstructed)
    else:
        raise ValueError(f"Unknown reconstruction mode: {mode}")

def reparameterization_trick(mu: Any, log_var: Any) -> np.ndarray:
    """Reparameterization Trick: z = mu + std * eps."""
    m = np.array(mu, dtype=float)
    lv = np.array(log_var, dtype=float)
    std_val = np.exp(0.5 * lv)
    eps = np.random.randn(*m.shape)
    return m + std_val * eps

def evidence_lower_bound(x_true: Any, x_reconstructed: Any, mu: Any, log_var: Any) -> float:
    """Evidence Lower Bound (ELBO): -Reconstruction_Loss - KL_Divergence."""
    rec_loss = reconstruction_loss(x_true, x_reconstructed)
    m = np.array(mu, dtype=float)
    lv = np.array(log_var, dtype=float)
    kl_div = -0.5 * np.sum(1.0 + lv - (m ** 2) - np.exp(lv))
    return float(-rec_loss - kl_div)


# =====================================================================
# 14. REGULARIZATION & NORMALIZATION
# =====================================================================

def l1_regularization(weights: Any, l1_lambda: float = 0.01) -> float:
    """L1 Lasso penalty: l1_lambda * sum(|w|)."""
    w = np.array(weights, dtype=float)
    return float(l1_lambda * np.sum(np.abs(w)))

def l2_regularization(weights: Any, l2_lambda: float = 0.01) -> float:
    """L2 Ridge weight decay penalty: 0.5 * l2_lambda * sum(w^2)."""
    w = np.array(weights, dtype=float)
    return float(0.5 * l2_lambda * np.sum(w ** 2))

def dropout_mask(shape: Tuple[int, ...], drop_prob: float = 0.5) -> np.ndarray:
    """Binary dropout mask with inverted scaling."""
    if drop_prob <= 0:
        return np.ones(shape, dtype=float)
    mask = (np.random.rand(*shape) >= drop_prob).astype(float)
    return mask / (1.0 - drop_prob)

def batch_norm(x: Any, gamma: Optional[Any] = None, beta: Optional[Any] = None, eps: float = 1e-5) -> np.ndarray:
    """Batch Normalization across batch dimension (axis 0)."""
    x_arr = np.array(x, dtype=float)
    mean_val = np.mean(x_arr, axis=0, keepdims=True)
    var_val = np.var(x_arr, axis=0, keepdims=True)
    x_norm = (x_arr - mean_val) / np.sqrt(var_val + eps)
    if gamma is not None:
        x_norm *= gamma
    if beta is not None:
        x_norm += beta
    return x_norm

def group_norm(x: Any, num_groups: int = 2, gamma: Optional[Any] = None, beta: Optional[Any] = None, eps: float = 1e-5) -> np.ndarray:
    """Group Normalization across channel groups."""
    x_arr = np.array(x, dtype=float)
    N, C, *spatial = x_arr.shape
    x_reshaped = x_arr.reshape(N, num_groups, C // num_groups, *spatial)
    mean_val = np.mean(x_reshaped, axis=tuple(range(2, x_reshaped.ndim)), keepdims=True)
    var_val = np.var(x_reshaped, axis=tuple(range(2, x_reshaped.ndim)), keepdims=True)
    x_norm = (x_reshaped - mean_val) / np.sqrt(var_val + eps)
    x_norm = x_norm.reshape(N, C, *spatial)
    if gamma is not None:
        x_norm *= gamma
    if beta is not None:
        x_norm += beta
    return x_norm


# =====================================================================
# 15. ENSEMBLE / BOOSTING MATH
# =====================================================================

def gradient_boosting_update(y_true: Any, y_pred_prev: Any, learning_rate: float = 0.1) -> np.ndarray:
    """Computes negative gradient pseudo-residuals for Gradient Boosting."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred_prev, dtype=float)
    residuals = yt - yp
    return residuals

def adaboost_weight_update(sample_weights: Any, alpha: float, y_true: Any, y_pred: Any) -> np.ndarray:
    """AdaBoost sample weight update."""
    w = np.array(sample_weights, dtype=float)
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    incorrect = (yt != yp).astype(float)
    new_w = w * np.exp(alpha * incorrect)
    return new_w / np.sum(new_w)

def bagging_sample_weight(n_samples: int) -> np.ndarray:
    """Bootstrap sample weights for Bagging ensemble."""
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    weights = np.bincount(indices, minlength=n_samples)
    return weights / np.sum(weights)


# =====================================================================
# 16. GRAPH NEURAL NETWORK MATH
# =====================================================================

def adjacency_matrix_ops(adj: Any, mode: str = "normalize") -> np.ndarray:
    """Graph adjacency matrix operations: self_loops, degree, or symmetric normalization D^-1/2 A_tilde D^-1/2."""
    A = np.array(adj, dtype=float)
    if mode == "self_loops":
        return A + np.eye(A.shape[0])
    elif mode == "normalize":
        A_tilde = A + np.eye(A.shape[0])
        degree = np.sum(A_tilde, axis=1)
        d_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
        d_inv_sqrt[degree == 0] = 0.0
        D_mat = np.diag(d_inv_sqrt)
        return np.matmul(np.matmul(D_mat, A_tilde), D_mat)
    else:
        raise ValueError(f"Unknown adjacency mode: {mode}")

def graph_convolution(X: Any, A_norm: Any, W: Any) -> np.ndarray:
    """GCN Layer: H = Relu(A_norm X W)."""
    x_arr = np.array(X, dtype=float)
    a_arr = np.array(A_norm, dtype=float)
    w_arr = np.array(W, dtype=float)
    return relu(np.matmul(np.matmul(a_arr, x_arr), w_arr))

def message_passing(node_features: Any, adj: Any, aggregate_fn: str = "sum") -> np.ndarray:
    """GNN Message Passing neighbor aggregation."""
    X = np.array(node_features, dtype=float)
    A = np.array(adj, dtype=float)
    if aggregate_fn == "sum":
        return np.matmul(A, X)
    elif aggregate_fn == "mean":
        deg = np.sum(A, axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        return np.matmul(A, X) / deg
    else:
        raise ValueError(f"Unknown aggregation function: {aggregate_fn}")


# =====================================================================
# UPDATED CATEGORIES REGISTRY & HELP
# =====================================================================

_CATEGORIES = {
    "linear_algebra": ["dot", "matmul", "transpose", "inverse", "pseudo_inverse", "determinant", "trace", "rank", "norm", "normalize", "eigen", "svd", "qr_decompose", "cholesky_decompose", "solve_linear_system", "identity", "zeros", "ones", "reshape", "concat", "outer", "cross_product", "trace_of_product"],
    "calculus_gradients": ["gradient", "partial_derivative", "second_derivative", "hessian", "jacobian", "chain_rule", "numerical_integrate", "mse_gradient", "mae_gradient", "huber_gradient"],
    "activations": ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "gelu", "swish", "softmax", "softplus", "linear"],
    "loss_functions": ["mse", "mae", "huber_loss", "hinge_loss", "focal_loss", "cross_entropy", "binary_cross_entropy", "kl_divergence_loss"],
    "probability_statistics": ["mean", "median", "mode", "variance", "std", "covariance", "covariance_matrix", "correlation", "percentile", "z_score", "skewness", "kurtosis", "gaussian_pdf", "bernoulli_pmf", "binomial_pmf", "poisson_pmf", "uniform_pdf", "exponential_pdf", "bayes_theorem", "joint_probability", "marginal_probability", "conditional_probability", "expectation", "maximum_likelihood_estimate", "class_prior", "confidence_interval", "sample", "shuffle"],
    "information_theory": ["entropy", "info_cross_entropy", "kl_divergence", "mutual_information", "information_gain", "gini_impurity", "perplexity"],
    "distance_similarity": ["euclidean_distance", "manhattan_distance", "cosine_similarity", "mahalanobis_distance", "hamming_distance"],
    "signal_convolution": ["convolve", "pooling", "fourier_transform", "inverse_fourier_transform"],
    "transformer_attention": ["scaled_dot_product_attention", "query_key_value_projection", "multi_head_attention", "masked_attention", "cross_attention", "sinusoidal_positional_encoding", "learned_positional_embedding", "layer_norm", "residual_connection", "feed_forward_block", "attention_dropout"],
    "diffusion_models": ["forward_noise_step", "forward_noise_closed_form", "noise_schedule", "alpha_bar", "reverse_denoise_step", "score_function", "denoising_score_matching_loss", "ddim_sample_step", "variance_schedule", "latent_encode", "latent_decode"],
    "reinforcement_learning": ["bellman_equation", "discounted_return", "td_error", "advantage_function", "policy_gradient", "q_value_update"],
    "optimizers": ["sgd_update", "momentum_update", "rmsprop_update", "adam_update", "learning_rate_decay", "gradient_clipping"],
    "generative_models": ["discriminator_loss", "generator_loss", "minimax_objective", "wasserstein_distance", "reconstruction_loss", "reparameterization_trick", "evidence_lower_bound"],
    "regularization_normalization": ["l1_regularization", "l2_regularization", "dropout_mask", "batch_norm", "group_norm"],
    "ensemble_boosting": ["gradient_boosting_update", "adaboost_weight_update", "bagging_sample_weight"],
    "graph_neural_networks": ["adjacency_matrix_ops", "graph_convolution", "message_passing"]
}


# =====================================================================
# 17. NUMERICAL STABILITY, CALIBRATION & STRING DISTANCES
# =====================================================================

def log_sum_exp(x: Any, axis: Optional[int] = None, keepdims: bool = False) -> Union[float, np.ndarray]:
    """Numerically stable Log-Sum-Exp computation: max(x) + log(sum(exp(x - max(x))))."""
    arr = np.array(x, dtype=float)
    max_val = np.max(arr, axis=axis, keepdims=True)
    res = max_val + np.log(np.sum(np.exp(arr - max_val), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        res = np.squeeze(res, axis=axis)
    elif not keepdims and axis is None:
        res = float(res.ravel()[0])
    return res

def safe_divide(numerator: Any, denominator: Any, eps: float = 1e-12) -> Union[float, np.ndarray]:
    """Division with epsilon safeguard against divide-by-zero crashes."""
    num = np.array(numerator, dtype=float)
    den = np.array(denominator, dtype=float)
    den_safe = np.where(np.abs(den) < eps, np.sign(den) * eps + (den == 0) * eps, den)
    out = num / den_safe
    return float(out) if out.ndim == 0 else out

def clip_by_value(x: Any, min_val: float = -1e9, max_val: float = 1e9) -> np.ndarray:
    """Clips values to [min_val, max_val] range."""
    return np.clip(np.array(x, dtype=float), min_val, max_val)

def smooth_l1_loss(y_true: Any, y_pred: Any, beta: float = 1.0) -> float:
    """Smooth L1 Loss (Huber variant with beta scaling)."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    diff = np.abs(yt - yp)
    loss = np.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    return float(np.mean(loss))

def brier_score(y_true: Any, y_prob: Any) -> float:
    """Computes Brier Score mean((y_true - y_prob)^2) for probability calibration."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_prob, dtype=float)
    return float(np.mean((yt - yp) ** 2))

def expected_calibration_error(y_true: Any, y_prob: Any, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    yt = np.array(y_true, dtype=int)
    yp = np.array(y_prob, dtype=float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(yt)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (yp > bin_lower) & (yp <= bin_upper) if i > 0 else (yp >= bin_lower) & (yp <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(yt[in_bin])
            avg_confidence_in_bin = np.mean(yp[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)

def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return int(previous_row[-1])

def jaccard_similarity(a: Any, b: Any) -> float:
    """Jaccard similarity coefficient |A intersect B| / |A union B|."""
    set_a = set(a)
    set_b = set(b)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection / (union + 1e-12))
