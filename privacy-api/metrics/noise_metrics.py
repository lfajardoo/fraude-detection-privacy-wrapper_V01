"""
Noise behavior metrics.

Reference: T. Chai and R. R. Draxler,
"Root Mean Square Error (RMSE) or Mean Absolute Error (MAE)?",
Geoscientific Model Development, 7, 1247-1250, 2014.

Usage:
    pairs = [(original_1, noisy_1), (original_2, noisy_2), ...]
    report = noise_summary(pairs)
    # {"mae": ..., "rmse": ..., "n_samples": ..., "rmse_mae_ratio": ...}

Design notes:
  - Pure stdlib (math only); no external dependencies.
  - Both metrics are needed: MAE treats all errors equally, RMSE penalises
    large noise spikes disproportionately — relevant when a spike can flip
    the fraud classification.
  - rmse_mae_ratio indicates error distribution skew (Chai & Draxler §2).
"""

import math
from typing import List, Tuple, Dict, Any


def mae(pairs: List[Tuple[float, float]]) -> float:
    """Mean Absolute Error between original and noisy values."""
    if not pairs:
        return 0.0
    return sum(abs(orig - noisy) for orig, noisy in pairs) / len(pairs)


def rmse(pairs: List[Tuple[float, float]]) -> float:
    """Root Mean Square Error between original and noisy values."""
    if not pairs:
        return 0.0
    return math.sqrt(sum((orig - noisy) ** 2 for orig, noisy in pairs) / len(pairs))


def noise_summary(pairs: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Full noise report for a batch of (original, noisy) value pairs."""
    n = len(pairs)
    mae_val = mae(pairs)
    rmse_val = rmse(pairs)
    noises = [abs(o - nv) for o, nv in pairs]
    mean_noise = sum(noises) / n if n > 0 else 0.0
    std_val = (
        math.sqrt(sum((nv - mean_noise) ** 2 for nv in noises) / n)
        if n >= 2 else None
    )
    return {
        "mae": mae_val,
        "rmse": rmse_val,
        # ratio is only informative with >=2 samples (equals 1.0 trivially with 1)
        "rmse_mae_ratio": rmse_val / mae_val if (mae_val > 0.0 and n >= 2) else None,
        "noise_std": std_val,
        "n_samples": n,
    }
