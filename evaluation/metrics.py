"""
Evaluation metrics for the PLM NIDS.

All functions accept numpy arrays.

Reported metrics (matching the paper's evaluation plan):
  - PR-AUC   (minority class — attacks are rare)
  - ROC-AUC
  - F1       (binary, attack=positive)
  - FPR @ TPR = 95%   (operational point for IDS)
  - Per-class precision / recall
  - Confusion matrix
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,     # continuous score (higher = more anomalous / attack)
    threshold: Optional[float] = None,
    threshold_percentile: float = 95.0,
) -> Dict[str, float]:
    """
    Compute the full metric suite given ground-truth labels and anomaly scores.

    Args:
        y_true              : binary labels (0=benign, 1=attack)
        y_score             : anomaly score per flow (perplexity or attack logit)
        threshold           : decision threshold; if None, calibrate from percentile
        threshold_percentile: use this percentile of benign scores as threshold

    Returns dict with keys:
        pr_auc, roc_auc, f1, precision, recall, fpr_at_tpr95, threshold,
        tp, fp, tn, fn, support_benign, support_attack
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Calibrate threshold if not given
    if threshold is None:
        benign_scores = y_score[y_true == 0]
        if len(benign_scores) == 0:
            threshold = float(np.percentile(y_score, threshold_percentile))
        else:
            threshold = float(np.percentile(benign_scores, threshold_percentile))

    y_pred = (y_score >= threshold).astype(int)

    # Guard: if only one class present, some metrics are ill-defined
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class in y_true — some metrics will be NaN.")
        return {"error": "single_class", "threshold": threshold}

    pr_auc   = average_precision_score(y_true, y_score)
    roc_auc  = roc_auc_score(y_true, y_score)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    prec     = precision_score(y_true, y_pred, zero_division=0)
    rec      = recall_score(y_true, y_pred, zero_division=0)
    fpr_95   = _fpr_at_tpr(y_true, y_score, tpr_target=0.95)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "pr_auc":          float(pr_auc),
        "roc_auc":         float(roc_auc),
        "f1":              float(f1),
        "precision":       float(prec),
        "recall":          float(rec),
        "fpr_at_tpr95":    float(fpr_95),
        "threshold":       float(threshold),
        "tp":              int(tp),
        "fp":              int(fp),
        "tn":              int(tn),
        "fn":              int(fn),
        "support_benign":  int(np.sum(y_true == 0)),
        "support_attack":  int(np.sum(y_true == 1)),
    }


def _fpr_at_tpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """False-positive rate at the threshold that achieves tpr_target recall."""
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    # Find first index where TPR >= target
    idx = np.searchsorted(tpr_arr, tpr_target)
    if idx >= len(fpr_arr):
        return float(fpr_arr[-1])
    return float(fpr_arr[idx])


def calibrate_threshold(
    benign_scores: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Return threshold as the given percentile of benign anomaly scores."""
    return float(np.percentile(benign_scores, percentile))


def print_report(metrics: Dict[str, float], title: str = "Evaluation") -> None:
    """Pretty-print a metrics dict."""
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    ordered = [
        ("PR-AUC  (minority)",   "pr_auc"),
        ("ROC-AUC",              "roc_auc"),
        ("F1 (attack)",          "f1"),
        ("Precision (attack)",   "precision"),
        ("Recall (attack)",      "recall"),
        ("FPR @ TPR=95%",        "fpr_at_tpr95"),
        ("Threshold",            "threshold"),
    ]
    for label, key in ordered:
        val = metrics.get(key, float("nan"))
        print(f"  {label:<26} {val:.4f}")
    print(sep)
    print(f"  TP={metrics.get('tp')}  FP={metrics.get('fp')}  "
          f"TN={metrics.get('tn')}  FN={metrics.get('fn')}")
    print(f"  Benign={metrics.get('support_benign')}  Attack={metrics.get('support_attack')}")
    print(sep)
