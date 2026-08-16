"""
Privacy-utility tradeoff metrics.

Reference: Q. Geng et al., "Tight Analysis of Privacy and Utility Tradeoff
in Approximate Differential Privacy", AISTATS 2020.

Usage:
    retention = utility_retention(utility_original=0.92, utility_with_dp=0.87)
    corr = risk_noise_correlation([(risk_1, |noise_1|), (risk_2, |noise_2|), ...])

Design notes:
  - utility_retention answers: "how much classification quality survives after DP?"
    Use any scalar utility (accuracy, f1, informedness) as input.
  - risk_noise_correlation tests whether the DP mechanism introduces a systematic
    dependency between transaction risk and noise magnitude (expected ≈ 0 for
    standard Laplace/Gaussian; non-zero for adaptive/monotone mechanisms).
  - Both functions are pure stdlib; extend by adding new tradeoff functions here
    and re-exporting from __init__.py.
"""

import math
from typing import List, Tuple, Dict, Any


def utility_retention(utility_original: float, utility_with_dp: float) -> float:
    """Fraction of utility preserved after applying DP (0–1, higher is better)."""
    if utility_original == 0.0:
        return 1.0 if utility_with_dp == 0.0 else 0.0
    return utility_with_dp / utility_original


def risk_noise_correlation(risk_noise_pairs: List[Tuple[float, float]]) -> float:
    """
    Pearson correlation between fraud risk and absolute noise magnitude.

    Args:
        risk_noise_pairs: list of (fraud_probability_original, abs_noise)
                          where abs_noise = |fraud_probability_original - noisy|

    Returns:
        Pearson r in [-1, 1]. Near 0 → noise is independent of risk (standard DP).
    """
    if len(risk_noise_pairs) < 2:
        return 0.0

    risks = [r for r, _ in risk_noise_pairs]
    noises = [n for _, n in risk_noise_pairs]
    n = len(risks)

    mean_r = sum(risks) / n
    mean_n = sum(noises) / n

    cov = sum((r - mean_r) * (nv - mean_n) for r, nv in zip(risks, noises)) / n
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in risks) / n)
    std_n = math.sqrt(sum((nv - mean_n) ** 2 for nv in noises) / n)

    if std_r == 0.0 or std_n == 0.0:
        return 0.0
    return cov / (std_r * std_n)


def privacy_utility_summary(
    utility_original: float,
    utility_with_dp: float,
    risk_noise_pairs: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """Combined privacy-utility report."""
    return {
        "utility_original": utility_original,
        "utility_with_dp": utility_with_dp,
        "utility_retention": utility_retention(utility_original, utility_with_dp),
        "risk_noise_correlation": risk_noise_correlation(risk_noise_pairs),
        "n_samples": len(risk_noise_pairs),
    }
