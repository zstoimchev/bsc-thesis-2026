from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from common.data_loading import find_label_column, split_features_label
from common.io_utils import utc_now_iso, write_json
from common.metrics import compute_classification_metrics, save_metrics_bundle
from common.paths import resolve_path
from common.preprocessing import prepare_train_test_data, save_preprocessing_artifacts
from common.runner_args import (
    RunnerArgs,
    add_common_runner_args,
    parse_drop_columns,
    runner_args_to_dict,
)
from common.runner_core import (
    load_train_test_from_args,
    write_audit_only_result,
    write_failure_result,
)


def parse_layers(value: str) -> list[int]:
    value = value.strip()

    if not value:
        return [128, 64, 32]

    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 2 / DDoS_Traffic_Research DNN runner."
    )
    add_common_runner_args(parser)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--layers", default="128,64,32")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--source-path",
        default="external/model02_DDoS_Traffic_Research",
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
            "paper": "02",
            "repository": "DDoS_Traffic_Research",
            "variant": "dnn",
            "source_path": parsed.source_path,
            "epochs": parsed.epochs,
            "batch_size": parsed.batch_size,
            "layers": parse_layers(parsed.layers),
            "dropout": parsed.dropout,
            "lr": parsed.lr,
        },
    )


def build_dnn_model(
    input_dim: int,
    num_classes: int,
    layers: list[int],
    dropout: float,
    lr: float,
):
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for units in layers:
        model.add(tf.keras.layers.Dense(int(units), activation="relu"))
        if dropout and dropout > 0:
            model.add(tf.keras.layers.Dropout(float(dropout)))

    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def run_dnn(args: RunnerArgs) -> dict[str, Any]:
    try:
        import tensorflow as tf
    except ImportError:
        return write_audit_only_result(
            args=args,
            status="needs_dependency",
            message=(
                "TensorFlow is not installed. Install it with `pip install tensorflow` "
                "to execute this Paper 2 DNN runner."
            ),
            details={
                "missing_dependency": "tensorflow",
                "paper": "02",
                "repository": "DDoS_Traffic_Research",
                "variant": "dnn",
                "source_path": args.extra.get("source_path"),
            },
        )

    start = time.time()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    run_dir = resolve_path(args.run_dir)
    artifact_dir = resolve_path(args.artifact_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "runner_args.json", runner_args_to_dict(args))

    loaded, train_df, test_df = load_train_test_from_args(args)

    label_col = find_label_column(train_df, args.label_col or loaded.label_col)

    X_train, y_train = split_features_label(
        train_df,
        label_col=label_col,
        drop_columns=args.drop_columns,
    )

    X_test, y_test = split_features_label(
        test_df,
        label_col=label_col,
        drop_columns=args.drop_columns,
    )

    prepared = prepare_train_test_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        problem_type=args.problem_type,
        scale_numeric=True,
    )

    input_dim = int(prepared.X_train.shape[1])
    num_classes = int(len(prepared.label_mapping))

    model = build_dnn_model(
        input_dim=input_dim,
        num_classes=num_classes,
        layers=list(args.extra["layers"]),
        dropout=float(args.extra["dropout"]),
        lr=float(args.extra["lr"]),
    )

    history = model.fit(
        prepared.X_train,
        prepared.y_train,
        validation_data=(prepared.X_test, prepared.y_test),
        epochs=int(args.extra["epochs"]),
        batch_size=int(args.extra["batch_size"]),
        verbose=1,
    )

    y_proba = model.predict(
        prepared.X_test,
        batch_size=int(args.extra["batch_size"]),
    )
    y_pred = np.argmax(y_proba, axis=1)

    metrics = compute_classification_metrics(
        y_true=prepared.y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        label_mapping=prepared.label_mapping,
        problem_type=args.problem_type,
    )

    model_path = artifact_dir / "dnn.keras"
    model.save(model_path)

    save_preprocessing_artifacts(str(artifact_dir), prepared)

    history_path = artifact_dir / "training_history.json"
    write_json(
        history_path,
        {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
    )

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
            "runner": "model02_ddos_traffic_research_dnn",
            "paper": "02",
            "repository": "DDoS_Traffic_Research",
            "variant": "dnn",
            "problem_type": args.problem_type,
            "label_col": label_col,
            "drop_columns": args.drop_columns,
            "epochs": int(args.extra["epochs"]),
            "batch_size": int(args.extra["batch_size"]),
            "layers": list(args.extra["layers"]),
            "dropout": float(args.extra["dropout"]),
            "lr": float(args.extra["lr"]),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "model_artifact_path": str(model_path),
            "history_path": str(history_path),
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
        metrics = run_dnn(args)

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