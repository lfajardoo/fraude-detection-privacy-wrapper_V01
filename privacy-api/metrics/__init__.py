"""
Metrics package for privacy-utility evaluation.

Structure (one module per paper):
  noise_metrics          - MAE, RMSE             [Chai & Draxler, 2014]
  classification_metrics - Accuracy, FNR, F1,
                           Informedness, MCC      [Powers, 2011; Fawcett, 2006]
  privacy_metrics        - UtilityRetention,
                           RiskNoiseCorrelation   [Geng et al., 2020]
"""

from .noise_metrics import mae, rmse, noise_summary
from .classification_metrics import (
    confusion_matrix_counts,
    accuracy,
    fnr,
    f1,
    informedness,
    markedness,
    mcc,
    classification_report,
)
from .privacy_metrics import utility_retention, risk_noise_correlation, privacy_utility_summary

__all__ = [
    # noise
    "mae",
    "rmse",
    "noise_summary",
    # classification
    "confusion_matrix_counts",
    "accuracy",
    "fnr",
    "f1",
    "informedness",
    "markedness",
    "mcc",
    "classification_report",
    # privacy-utility
    "utility_retention",
    "risk_noise_correlation",
    "privacy_utility_summary",
]
