"""
Classification quality metrics.

References:
  [Powers]  D. M. W. Powers, "Evaluation: From Precision, Recall and F-Measure
            to ROC, Informedness, Markedness & Correlation", JMLT, 2011.
  [Fawcett] T. Fawcett, "An Introduction to ROC Analysis",
            Pattern Recognition Letters, 27(8), 2006.

Usage:
    report = classification_report(y_true=[0,1,1,0], y_pred=[0,1,0,0])

Design notes:
  - Pure stdlib; no sklearn dependency to keep the metrics layer portable.
  - Accuracy alone is misleading on imbalanced datasets (creditcard ~0.17 % fraud).
    Informedness and MCC are prevalence-independent [Powers §2].
  - FNR is the primary sensitivity indicator: fraud transactions missed after DP.
  - To add a new metric: add a function here and re-export from __init__.py.
"""

import math
from typing import List, Dict, Any


def confusion_matrix_counts(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Return TP, FP, TN, FN counts."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def accuracy(cm: Dict[str, int]) -> float:
    """(TP + TN) / N [Fawcett eq. 5 / Powers eq. 5]."""
    total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
    return (cm["tp"] + cm["tn"]) / total if total > 0 else 0.0


def fnr(cm: Dict[str, int]) -> float:
    """False Negative Rate = FN / (FN + TP). Frauds that escape detection after DP."""
    denom = cm["fn"] + cm["tp"]
    return cm["fn"] / denom if denom > 0 else 0.0


def f1(cm: Dict[str, int]) -> float:
    """F1 = 2·TP / (2·TP + FP + FN) [Powers eq. 6 / Fawcett §2]."""
    denom = 2 * cm["tp"] + cm["fp"] + cm["fn"]
    return 2 * cm["tp"] / denom if denom > 0 else 0.0


def informedness(cm: Dict[str, int]) -> float:
    """Bookmaker Informedness = TPR + TNR − 1 [Powers §2]."""
    tpr_denom = cm["tp"] + cm["fn"]
    tnr_denom = cm["tn"] + cm["fp"]
    tpr = cm["tp"] / tpr_denom if tpr_denom > 0 else 0.0
    tnr = cm["tn"] / tnr_denom if tnr_denom > 0 else 0.0
    return tpr + tnr - 1.0


def markedness(cm: Dict[str, int]) -> float:
    """Markedness = PPV + NPV − 1 [Powers §2]."""
    ppv_denom = cm["tp"] + cm["fp"]
    npv_denom = cm["tn"] + cm["fn"]
    ppv = cm["tp"] / ppv_denom if ppv_denom > 0 else 0.0
    npv = cm["tn"] / npv_denom if npv_denom > 0 else 0.0
    return ppv + npv - 1.0


def mcc(inf_val: float, mark_val: float) -> float:
    """Matthews Correlation Coefficient = sqrt(Informedness × Markedness) [Powers §2]."""
    product = inf_val * mark_val
    if product >= 0:
        return math.sqrt(product)
    return -math.sqrt(-product)


def classification_report(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Full classification report from ground-truth and predicted label lists."""
    cm = confusion_matrix_counts(y_true, y_pred)
    inf_val = informedness(cm)
    mark_val = markedness(cm)
    return {
        "confusion_matrix": cm,
        "accuracy": accuracy(cm),
        "fnr": fnr(cm),
        "f1": f1(cm),
        "informedness": inf_val,
        "markedness": mark_val,
        "mcc": mcc(inf_val, mark_val),
        "n_samples": len(y_true),
    }
