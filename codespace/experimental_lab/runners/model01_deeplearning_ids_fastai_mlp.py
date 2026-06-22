from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.data_loading import find_label_column, split_features_label
from common.io_utils import utc_now_iso, write_json
from common.metrics import compute_classification_metrics, save_metrics_bundle
from common.paths import resolve_path
from common.preprocessing import (
    align_columns,
    clean_features,
    normalize_label_value,
    to_binary_labels,
)
from common.runner_args import RunnerArgs, add_common_runner_args, parse_drop_columns, runner_args_to_dict
from common.runner_core import (
    load_train_test_from_args,
    write_audit_only_result,
    write_failure_result,
)


def parse_layers(value: str) -> list[int]:
    """
    Parse FastAI hidden-layer configuration.

    Example:
        "200,100" -> [200, 100]
    """
    value = value.strip()

    if not value:
        return [200, 100]

    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 1 / DeepLearning-IDS FastAI MLP runner."
    )
    add_common_runner_args(parser)

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of FastAI training epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for fit_one_cycle.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size.",
    )

    parser.add_argument(
        "--layers",
        default="200,100",
        help="Comma-separated hidden layer sizes.",
    )

    parser.add_argument(
        "--source-path",
        default="external/model01_DeepLearning-IDS",
        help="Optional original repository path for audit traceability.",
    )

    parsed = parser.parse_args(argv)

    return RunnerArgs(
        mode=parsed.mode,
        stage=parsed.stage,
        model_id=parsed.model_id,
        dataset_id=parsed.dataset_id,
        split_id=parsed.split_id,
        dataset_path=parsed.dataset_path,
        train_path=parsed.train_path,
        test_path=parsed.test_path,
        label_col=parsed.label_col,
        drop_columns=parse_drop_columns(parsed.drop_columns),
        problem_type=parsed.problem_type,
        artifact_dir=parsed.artifact_dir,
        run_dir=parsed.run_dir,
        metrics_out=parsed.metrics_out,
        nrows=parsed.nrows,
        seed=parsed.seed,
        extra={
            "epochs": parsed.epochs,
            "lr": parsed.lr,
            "batch_size": parsed.batch_size,
            "layers": parse_layers(parsed.layers),
            "source_path": parsed.source_path,
            "paper": "01",
            "repository": "DeepLearning-IDS",
            "variant": "fastai_mlp",
        },
    )


def prepare_fastai_dataframe(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    drop_columns: list[str],
    problem_type: str,
) -> tuple[pd.DataFrame, list[int], list[int], list[str], list[str], str, dict[int, str]]:
    """
    Prepare one combined dataframe for FastAI TabularPandas.

    FastAI expects one dataframe plus train/validation row indices.
    """
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

    X_train = clean_features(X_train)
    X_test = clean_features(X_test)
    X_train, X_test = align_columns(X_train, X_test)

    if problem_type == "binary":
        y_train_clean = to_binary_labels(y_train)
        y_test_clean = to_binary_labels(y_test)
    elif problem_type == "multiclass":
        y_train_clean = y_train.map(normalize_label_value)
        y_test_clean = y_test.map(normalize_label_value)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    target_col = "__target"

    train_ready = X_train.copy()
    test_ready = X_test.copy()

    train_ready[target_col] = y_train_clean.astype(str).values
    test_ready[target_col] = y_test_clean.astype(str).values

    combined = pd.concat([train_ready, test_ready], ignore_index=True)

    # Make target order deterministic.
    classes = sorted(combined[target_col].astype(str).unique().tolist())
    combined[target_col] = pd.Categorical(combined[target_col].astype(str), categories=classes)

    label_mapping = {idx: label for idx, label in enumerate(classes)}

    train_indices = list(range(0, len(train_ready)))
    valid_indices = list(range(len(train_ready), len(combined)))

    cont_names: list[str] = []
    cat_names: list[str] = []

    feature_columns = [col for col in combined.columns if col != target_col]

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(combined[col]):
            cont_names.append(col)
        else:
            cat_names.append(col)
            combined[col] = combined[col].astype(str).fillna("missing")

    # FastAI handles continuous missing values through FillMissing.
    for col in cont_names:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return (
        combined,
        train_indices,
        valid_indices,
        cat_names,
        cont_names,
        target_col,
        label_mapping,
    )


def run_fastai_mlp(args: RunnerArgs) -> dict[str, Any]:
    """
    Train and evaluate the Paper 1 FastAI MLP-style model.
    """
    try:
        from fastai.tabular.all import (
            Categorify,
            CategoryBlock,
            FillMissing,
            Normalize,
            TabularPandas,
            accuracy,
            tabular_learner,
        )
    except ImportError:
        return write_audit_only_result(
            args=args,
            status="needs_dependency",
            message=(
                "FastAI is not installed. Install it with `pip install fastai` "
                "to execute this Paper 1 FastAI MLP runner."
            ),
            details={
                "missing_dependency": "fastai",
                "runner": "model01_deeplearning_ids_fastai_mlp",
                "source_path": args.extra.get("source_path"),
            },
        )

    start = time.time()

    np.random.seed(args.seed)

    run_dir = resolve_path(args.run_dir)
    artifact_dir = resolve_path(args.artifact_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "runner_args.json", runner_args_to_dict(args))

    loaded, train_df, test_df = load_train_test_from_args(args)

    label_col = find_label_column(train_df, args.label_col or loaded.label_col)

    (
        combined,
        train_indices,
        valid_indices,
        cat_names,
        cont_names,
        target_col,
        label_mapping,
    ) = prepare_fastai_dataframe(
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        drop_columns=args.drop_columns,
        problem_type=args.problem_type,
    )

    splits = (train_indices, valid_indices)

    procs = [Categorify, FillMissing, Normalize]

    tabular = TabularPandas(
        combined,
        procs=procs,
        cat_names=cat_names,
        cont_names=cont_names,
        y_names=target_col,
        y_block=CategoryBlock,
        splits=splits,
    )

    dls = tabular.dataloaders(bs=int(args.extra["batch_size"]))

    learner = tabular_learner(
        dls,
        layers=list(args.extra["layers"]),
        metrics=accuracy,
    )

    learner.fit_one_cycle(
        int(args.extra["epochs"]),
        float(args.extra["lr"]),
    )

    preds, targs = learner.get_preds(dl=dls.valid)

    y_pred = preds.argmax(dim=1).cpu().numpy()
    y_true = targs.cpu().numpy()
    y_proba = preds.cpu().numpy()

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        label_mapping=label_mapping,
        problem_type=args.problem_type,
    )

    exported_path = artifact_dir / "learner.pkl"
    learner.export(exported_path)

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
            "runner": "model01_deeplearning_ids_fastai_mlp",
            "paper": "01",
            "repository": "DeepLearning-IDS",
            "variant": "fastai_mlp",
            "problem_type": args.problem_type,
            "label_col": label_col,
            "drop_columns": args.drop_columns,
            "cat_names_count": len(cat_names),
            "cont_names_count": len(cont_names),
            "cat_names_preview": cat_names[:30],
            "cont_names_preview": cont_names[:30],
            "epochs": int(args.extra["epochs"]),
            "lr": float(args.extra["lr"]),
            "batch_size": int(args.extra["batch_size"]),
            "layers": list(args.extra["layers"]),
            "model_artifact_path": str(exported_path),
            "artifact_dir": str(artifact_dir),
            "run_dir": str(run_dir),
        }
    )

    saved_metrics = save_metrics_bundle(run_dir, metrics)

    metrics_out = resolve_path(args.metrics_out)
    if metrics_out != run_dir / "metrics.json":
        write_json(metrics_out, saved_metrics)

    return saved_metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        metrics = run_fastai_mlp(args)

        status = metrics.get("status")

        if status == "success":
            print(
                f"[success] {args.model_id} on {args.dataset_id}: "
                f"accuracy={metrics.get('accuracy')}, "
                f"f1_macro={metrics.get('f1_macro')}"
            )
            return 0

        print(f"[{status}] {args.model_id}: {metrics.get('message')}")
        return 0

    except Exception as exc:
        write_failure_result(args, exc)
        print(f"[failed] {args.model_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())