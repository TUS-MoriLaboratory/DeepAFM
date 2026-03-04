# src/utils/metrics.py

import numpy as np

def mse(a, b):
    return float(np.mean((a - b)**2))

def rmse(a, b):
    return float(np.sqrt(mse(a, b)))

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def correlation_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    a = a.flatten()
    b = b.flatten()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])