from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from common.data_loading import (
    LoadedDataset,
    describe_dataframe,
    find_label_column,
    get_drop_columns,
    load_dataset_from_record,
    split_features_label,
)
from common.io_utils import utc_now_iso, write_json
from common.metrics import compute_classification_metrics, save_metrics_bundle
from common.paths import resolve_path
from common.preprocessing import (
    prepare_train_test_data,
    save_preprocessing_artifacts,
)
from common.runner_args import RunnerArgs, runner_args_to_dict
from common.splitting import (
    apply_split_indices,
    default_split_paths,
    load_indices,
)


class RunnerCoreError(Exception):
    """Raised when a shared runner operation fails."""


def dataset_record_from_runner_args(args: RunnerArgs) -> dict[str, Any]:
    """
    Build a dataset-like registry record from runner arguments.

    This lets every runner reuse common/data_loading.py without directly reading
    registries.
    """
    record: dict[str, Any] = {
        "id": args.dataset_id,
        "label_col": args.label_col,
        "drop_columns": args.drop_columns,
    }

    if args.dataset_path:
        record["path"] = args.dataset_path
        record["split_type"] = "generated"
        return record

    if args.train_path and args.test_path:
        record["train_path"] = args.train_path
        record["test_path"] = args.test_path
        record["split_type"] = "predefined"
        return record

    raise RunnerCoreError(
        "Runner received neither --dataset-path nor --train-path + --test-path."
    )


def load_train_test_from_args(args: RunnerArgs) -> tuple[LoadedDataset, Any, Any]:
    """
    Load train/test data from runner arguments.

    Supported cases:

    1. Predefined split:
        --train-path ...
        --test-path ...

    2. Single dataset + saved split:
        --dataset-path ...
        --split-id ...

       The runner expects:
        results/splits/<split_id>/train_idx.npy
        results/splits/<split_id>/test_idx.npy
    """
    record = dataset_record_from_runner_args(args)
    loaded = load_dataset_from_record(record, nrows=args.nrows)

    if loaded.train_df is not None and loaded.test_df is not None:
        return loaded, loaded.train_df, loaded.test_df

    if loaded.full_df is None:
        raise RunnerCoreError("Loaded dataset has neither full_df nor train/test frames.")

    if not args.split_id:
        raise RunnerCoreError(
            "Single-file datasets require --split-id so saved indices can be loaded."
        )

    train_idx_path, test_idx_path = default_split_paths(args.split_id)
    train_indices, test_indices = load_indices(train_idx_path, test_idx_path)

    train_df, test_df = apply_split_indices(
        df=loaded.full_df,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    return loaded, train_df, test_df


def get_prediction_scores(model: Any, X_test: Any) -> np.ndarray | None:
    """
    Try to obtain probabilities/scores from a trained model.

    Used for ROC-AUC when possible.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X_test)
        except Exception:
            return None

    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_test)
            return np.asarray(scores)
        except Exception:
            return None

    return None


def save_model_artifact(model: Any, artifact_dir: str | Path, filename: str = "model.joblib") -> Path:
    """
    Save a trained sklearn-compatible model.
    """
    out_dir = resolve_path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / filename
    joblib.dump(model, model_path)

    return model_path


def train_eval_sklearn_model(
    args: RunnerArgs,
    model: Any,
    scale_numeric: bool = True,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Shared train/eval implementation for sklearn-compatible models.

    The model must provide:
    - fit(X_train, y_train)
    - predict(X_test)

    Optional:
    - predict_proba(X_test)
    - decision_function(X_test)
    """
    start = time.time()

    run_dir = resolve_path(args.run_dir)
    artifact_dir = resolve_path(args.artifact_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "runner_args.json", runner_args_to_dict(args))

    loaded, train_df, test_df = load_train_test_from_args(args)

    label_col = find_label_column(train_df, args.label_col or loaded.label_col)
    drop_columns = args.drop_columns

    X_train, y_train = split_features_label(
        train_df,
        label_col=label_col,
        drop_columns=drop_columns,
    )

    X_test, y_test = split_features_label(
        test_df,
        label_col=label_col,
        drop_columns=drop_columns,
    )

    prepared = prepare_train_test_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        problem_type=args.problem_type,
        scale_numeric=scale_numeric,
    )

    model.fit(prepared.X_train, prepared.y_train)

    y_pred = model.predict(prepared.X_test)
    y_scores = get_prediction_scores(model, prepared.X_test)

    metrics = compute_classification_metrics(
        y_true=prepared.y_test,
        y_pred=y_pred,
        y_proba=y_scores,
        label_mapping=prepared.label_mapping,
        problem_type=args.problem_type,
    )

    model_path = save_model_artifact(model, artifact_dir)
    save_preprocessing_artifacts(str(artifact_dir), prepared)

    duration = time.time() - start

    metrics.update(
        {
            "run_started_utc": utc_now_iso(),
            "duration_seconds": round(float(duration), 4),
            "status": "success",
            "stage": args.stage,
            "mode": args.mode,
            "model_id": args.model_id,
            "dataset_id": args.dataset_id,
            "split_id": args.split_id,
            "runner": "sklearn_compatible",
            "model_artifact_path": str(model_path),
            "artifact_dir": str(artifact_dir),
            "run_dir": str(run_dir),
            "label_col": label_col,
            "drop_columns": drop_columns,
            "problem_type": args.problem_type,
            "model_params": model_params or {},
            "train_summary": describe_dataframe(train_df, label_col),
            "test_summary": describe_dataframe(test_df, label_col),
        }
    )

    saved_metrics = save_metrics_bundle(run_dir, metrics)

    # Also write exactly where orchestrate.py expects the runner output.
    metrics_out = resolve_path(args.metrics_out)
    if metrics_out != run_dir / "metrics.json":
        write_json(metrics_out, saved_metrics)

    return saved_metrics


def write_audit_only_result(
    args: RunnerArgs,
    status: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Write a metrics-shaped JSON result for runners that cannot run yet.

    This is important for your perfect audit: blocked/unfinished repositories still
    produce structured output instead of silently disappearing.
    """
    run_dir = resolve_path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "run_started_utc": utc_now_iso(),
        "duration_seconds": 0.0,
        "status": status,
        "stage": args.stage,
        "mode": args.mode,
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "split_id": args.split_id,
        "problem_type": args.problem_type,
        "accuracy": None,
        "balanced_accuracy": None,
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
        "precision_weighted": None,
        "recall_weighted": None,
        "f1_weighted": None,
        "roc_auc": None,
        "num_classes_true": None,
        "num_classes_pred": None,
        "message": message,
        "details": details or {},
        "runner_args": runner_args_to_dict(args),
    }

    write_json(run_dir / "metrics.json", result)

    metrics_out = resolve_path(args.metrics_out)
    if metrics_out != run_dir / "metrics.json":
        write_json(metrics_out, result)

    return result


def write_failure_result(
    args: RunnerArgs,
    error: BaseException,
) -> dict[str, Any]:
    """
    Write a structured failure result.
    """
    return write_audit_only_result(
        args=args,
        status="failed",
        message=str(error),
        details={
            "error_type": type(error).__name__,
        },
    )