from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common.io_utils import write_json
from common.paths import resolve_path


class MetricsError(Exception):
    """Raised when metrics cannot be computed or saved."""


def _safe_metric(func, *args, default: float | None = None, **kwargs) -> float | None:
    """
    Compute a metric safely.

    Some metrics fail when only one class is present in y_true/y_pred.
    In that case, return default instead of crashing the whole run.
    """
    try:
        value = func(*args, **kwargs)

        if isinstance(value, np.generic):
            return float(value)

        return float(value)

    except Exception:
        return default


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    label_mapping: dict[int, str] | None = None,
    problem_type: str = "binary",
) -> dict[str, Any]:
    """
    Compute common classification metrics.

    Metrics include:
    - accuracy
    - balanced accuracy
    - macro precision / recall / f1
    - weighted precision / recall / f1
    - confusion matrix
    - classification report
    - ROC-AUC when possible

    y_true and y_pred are expected to be encoded integer labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    metrics: dict[str, Any] = {
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "precision_macro": _safe_metric(
            precision_score,
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": _safe_metric(
            recall_score,
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": _safe_metric(
            f1_score,
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "precision_weighted": _safe_metric(
            precision_score,
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": _safe_metric(
            recall_score,
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": _safe_metric(
            f1_score,
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "num_classes_true": int(len(set(y_true.tolist()))),
        "num_classes_pred": int(len(set(y_pred.tolist()))),
        "labels_present": [int(x) for x in labels],
        "label_mapping": label_mapping or {},
        "problem_type": problem_type,
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = [int(x) for x in labels]

    target_names = None
    if label_mapping:
        target_names = [label_mapping.get(int(label), str(label)) for label in labels]

    try:
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        metrics["classification_report"] = report
    except Exception as exc:
        metrics["classification_report_error"] = str(exc)

    roc_auc = None

    if y_proba is not None:
        try:
            y_proba_array = np.asarray(y_proba)

            if problem_type == "binary":
                # For binary classifiers, sklearn may return:
                # - shape (n_samples, 2): use positive class probability
                # - shape (n_samples,): use directly
                if y_proba_array.ndim == 2 and y_proba_array.shape[1] >= 2:
                    positive_scores = y_proba_array[:, 1]
                else:
                    positive_scores = y_proba_array.reshape(-1)

                if len(set(y_true.tolist())) == 2:
                    roc_auc = float(roc_auc_score(y_true, positive_scores))

            else:
                if y_proba_array.ndim == 2 and len(set(y_true.tolist())) > 1:
                    roc_auc = float(
                        roc_auc_score(
                            y_true,
                            y_proba_array,
                            multi_class="ovr",
                            average="macro",
                        )
                    )

        except Exception as exc:
            metrics["roc_auc_error"] = str(exc)

    metrics["roc_auc"] = roc_auc

    return metrics


def save_confusion_matrix_csv(
    path: str | Path,
    confusion_matrix_values: list[list[int]],
    labels: list[int],
    label_mapping: dict[int, str] | None = None,
) -> Path:
    """
    Save confusion matrix as CSV.

    The first column is true label. Columns are predicted labels.
    """
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    label_mapping = label_mapping or {}

    label_names = [label_mapping.get(int(label), str(label)) for label in labels]

    with p.open("w", encoding="utf-8") as f:
        f.write("true_label," + ",".join(f"pred_{name}" for name in label_names) + "\n")

        for label, row in zip(label_names, confusion_matrix_values):
            f.write(str(label) + "," + ",".join(str(int(v)) for v in row) + "\n")

    return p


def save_metrics_bundle(
    output_dir: str | Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Save metrics.json and confusion_matrix.csv into output_dir.

    Returns the metrics dictionary with file paths added.
    """
    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_json_path = out_dir / "metrics.json"

    if "confusion_matrix" in metrics and "confusion_matrix_labels" in metrics:
        cm_path = out_dir / "confusion_matrix.csv"

        label_mapping_raw = metrics.get("label_mapping", {})
        label_mapping = {int(k): str(v) for k, v in label_mapping_raw.items()}

        save_confusion_matrix_csv(
            path=cm_path,
            confusion_matrix_values=metrics["confusion_matrix"],
            labels=metrics["confusion_matrix_labels"],
            label_mapping=label_mapping,
        )

        metrics["confusion_matrix_path"] = str(cm_path)

    metrics["metrics_json_path"] = str(metrics_json_path)

    write_json(metrics_json_path, metrics)

    return metrics


def compact_metrics_for_csv(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Keep only scalar / summary fields for the global CSV file.
    """
    keep_keys = [
        "run_id",
        "model_id",
        "dataset_id",
        "split_id",
        "stage",
        "status",
        "runner",
        "problem_type",
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc",
        "num_classes_true",
        "num_classes_pred",
        "duration_seconds",
        "metrics_json_path",
        "confusion_matrix_path",
        "error",
    ]

    return {key: metrics.get(key, "") for key in keep_keys}