from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def confusion(reference: np.ndarray, binary: np.ndarray, support: np.ndarray) -> dict[str, int]:
    mask = support.astype(bool)
    y = reference[mask].astype(bool)
    p = binary[mask].astype(bool)
    return {
        "tp": int(np.sum(y & p)),
        "fp": int(np.sum(~y & p)),
        "fn": int(np.sum(y & ~p)),
        "tn": int(np.sum(~y & ~p)),
    }


def compute(reference: np.ndarray, probability: np.ndarray, support: np.ndarray, threshold: float) -> dict[str, float | int]:
    binary = probability >= threshold
    counts = confusion(reference, binary, support)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    iou = tp / max(1, tp + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    mask = support.astype(bool)
    y = reference[mask].astype(np.uint8)
    score = probability[mask].astype(np.float64)
    auprc = float(average_precision_score(y, score)) if np.unique(y).size > 1 else float("nan")
    positive_total = int(reference.astype(bool).sum())
    result = {
        **counts,
        "iou": float(iou), "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "auprc": auprc, "coverage": float(mask.mean()),
        "positive_coverage": float((reference.astype(bool) & mask).sum() / max(1, positive_total)),
        "unresolved_positive_mass": float((reference.astype(bool) & ~mask).sum() / max(1, positive_total)),
    }
    return result


def error_map(reference: np.ndarray, binary: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = np.zeros(reference.shape, dtype=np.uint8)
    mask = support.astype(bool)
    ref = reference.astype(bool)
    pred = binary.astype(bool)
    out[mask & ref & pred] = 1
    out[mask & ~ref & pred] = 2
    out[mask & ref & ~pred] = 3
    out[~mask] = 4
    return out
