"""
evaluation.py -- metric helpers shared by evaluate.py and the reports.
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

SYNTHETIC_FAULTS = ["spike", "frozen", "dropout", "drift"]


def binary_metrics(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "false_positive_rate": float(fpr),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def per_fault_type_metrics(eval_df, y_pred_col="y_pred", fault_col="fault_type"):
    """eval_df must have a 'y_pred' binary column and a fault_type column.
    Computes, for each synthetic fault type independently, that fault type
    vs 'normal' rows only (excludes other fault types from that comparison)."""
    results = {}
    for ft in SYNTHETIC_FAULTS:
        sub = eval_df[eval_df[fault_col].isin([ft, "normal"])]
        y_true = (sub[fault_col] == ft).astype(int)
        y_pred = sub[y_pred_col]
        results[ft] = {
            "n_events": int((sub[fault_col] == ft).sum()),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
    return results
