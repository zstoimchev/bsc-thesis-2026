import json
from pathlib import Path

import numpy as np


class BinaryMetricsAccumulator:
    def __init__(self) -> None:
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

    def update(self, y_true, y_pred) -> None:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        self.tn += int(((y_true == 0) & (y_pred == 0)).sum())
        self.fp += int(((y_true == 0) & (y_pred == 1)).sum())
        self.fn += int(((y_true == 1) & (y_pred == 0)).sum())
        self.tp += int(((y_true == 1) & (y_pred == 1)).sum())

    def compute(self) -> dict:
        return compute_metrics_from_confusion(
            tn=self.tn,
            fp=self.fp,
            fn=self.fn,
            tp=self.tp,
        )


def compute_metrics(y_true, y_pred) -> dict:
    """
    Full in-memory metric computation.

    Used when y_true and y_pred already fit in memory.
    Internally uses the same accumulator so metric output is identical
    to chunked evaluation.
    """
    accumulator = BinaryMetricsAccumulator()
    accumulator.update(y_true, y_pred)
    return accumulator.compute()


def compute_metrics_from_confusion(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> dict:
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    recall_0 = tn / (tn + fp) if (tn + fp) else 0.0
    recall_1 = recall
    balanced_accuracy = (recall_0 + recall_1) / 2

    precision_0 = tn / (tn + fn) if (tn + fn) else 0.0
    precision_1 = precision

    f1_0 = (
        2 * precision_0 * recall_0 / (precision_0 + recall_0)
        if (precision_0 + recall_0)
        else 0.0
    )

    f1_1 = f1

    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    support_0 = tn + fp
    support_1 = fn + tp

    weighted_f1 = (
        ((f1_0 * support_0) + (f1_1 * support_1)) / total
        if total
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),

        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),

        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),

        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],

        "support": {
            "0": int(support_0),
            "1": int(support_1),
        },

        "predicted_distribution": {
            "0": int(tn + fn),
            "1": int(fp + tp),
        },

        "labels": {
            "0": "benign",
            "1": "attack",
        },
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)