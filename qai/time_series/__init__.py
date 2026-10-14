"""
qai.time_series -- Time-Series Forecasting & Analysis Utilities.
"""
import numpy as np
from typing import Any, Tuple

def exponential_smoothing(series: Any, alpha: float = 0.3) -> np.ndarray:
    """Applies Single Exponential Smoothing."""
    s = np.array(series, dtype=float)
    result = np.zeros_like(s)
    result[0] = s[0]
    for t in range(1, len(s)):
        result[t] = alpha * s[t] + (1.0 - alpha) * result[t - 1]
    return result

def autocorrelation_acf(series: Any, max_lag: int = 10) -> np.ndarray:
    """Computes Autocorrelation Function (ACF) up to max_lag."""
    s = np.array(series, dtype=float)
    s_mean = np.mean(s)
    s_var = np.var(s)
    N = len(s)
    acf_vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            cov = np.sum((s[:N-lag] - s_mean) * (s[lag:] - s_mean)) / N
            acf_vals.append(cov / (s_var + 1e-12))
    return np.array(acf_vals)

def seasonal_decompose(series: Any, period: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decomposes time-series into Trend, Seasonal, and Residual components."""
    s = np.array(series, dtype=float)
    trend = np.convolve(s, np.ones(period) / period, mode='same')
    detrended = s - trend
    seasonal = np.tile(np.mean(detrended.reshape(-1, period), axis=0), len(s) // period + 1)[:len(s)]
    residual = s - trend - seasonal
    return trend, seasonal, residual
