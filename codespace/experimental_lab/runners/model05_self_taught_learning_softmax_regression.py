from __future__ import annotations

import argparse
import time
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

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


def parse_hidden_layers(value: str) -> list[int]:
    value = value.strip()

    if not value:
        return [128, 64]

    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 5 / Self-Taught Learning + Softmax Regression runner."
    )
    add_common_runner_args(parser)

    parser.add_argument("--autoencoder-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-layers", default="128,64")
    parser.add_argument("--encoding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--softmax-max-iter", type=int, default=1000)
    parser.add_argument("--class-weight", default="balanced")

    parser.add_argument(
        "--source-path",
        default="external/model05_self_taught_learning_softmax_regression",
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
            "paper": "05",
            "repository": "self_taught_learning_softmax_regression",
            "variant": "autoencoder_softmax_regression",
            "source_path": parsed.source_path,
            "autoencoder_epochs": parsed.autoencoder_epochs,
            "batch_size": parsed.batch_size,
            "hidden_layers": parse_hidden_layers(parsed.hidden_layers),
            "encoding_dim": parsed.encoding_dim,
            "dropout": parsed.dropout,
            "lr": parsed.lr,
            "softmax_max_iter": parsed.softmax_max_iter,
            "class_weight": parsed.class_weight,
        },
    )


def normalize_optional_none(value):
    if value is None:
        return None

    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None

    return value


def build_autoencoder(
    input_dim: int,
    hidden_layers: list[int],
    encoding_dim: int,
    dropout: float,
    lr: float,
):
    """
    Build the self-taught feature learner.

    The autoencoder learns a compressed feature representation. Then a softmax
    regression classifier is trained on the encoded features.
    """
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(input_dim,))

    x = inputs

    for units in hidden_layers:
        x = tf.keras.layers.Dense(int(units), activation="relu")(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(float(dropout))(x)

    encoded = tf.keras.layers.Dense(int(encoding_dim), activation="relu", name="encoded")(x)

    x = encoded

    for units in reversed(hidden_layers):
        x = tf.keras.layers.Dense(int(units), activation="relu")(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(float(dropout))(x)

    decoded = tf.keras.layers.Dense(input_dim, activation="linear")(x)

    autoencoder = tf.keras.Model(inputs=inputs, outputs=decoded)
    encoder = tf.keras.Model(inputs=inputs, outputs=encoded)

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss="mse",
    )

    return autoencoder, encoder


def run_self_taught_softmax(args: RunnerArgs) -> dict[str, Any]:
    try:
        import tensorflow as tf
    except ImportError:
        return write_audit_only_result(
            args=args,
            status="needs_dependency",
            message=(
                "TensorFlow is not installed. Install it with `pip install tensorflow` "
                "to execute this Paper 5 self-taught learning runner."
            ),
            details={
                "missing_dependency": "tensorflow",
                "paper": "05",
                "variant": "autoencoder_softmax_regression",
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

    X_train_array = np.asarray(prepared.X_train).astype("float32")
    X_test_array = np.asarray(prepared.X_test).astype("float32")

    input_dim = int(X_train_array.shape[1])
    encoding_dim = int(args.extra["encoding_dim"])

    autoencoder, encoder = build_autoencoder(
        input_dim=input_dim,
        hidden_layers=list(args.extra["hidden_layers"]),
        encoding_dim=encoding_dim,
        dropout=float(args.extra["dropout"]),
        lr=float(args.extra["lr"]),
    )

    history = autoencoder.fit(
        X_train_array,
        X_train_array,
        validation_data=(X_test_array, X_test_array),
        epochs=int(args.extra["autoencoder_epochs"]),
        batch_size=int(args.extra["batch_size"]),
        verbose=1,
    )

    X_train_encoded = encoder.predict(
        X_train_array,
        batch_size=int(args.extra["batch_size"]),
    )

    X_test_encoded = encoder.predict(
        X_test_array,
        batch_size=int(args.extra["batch_size"]),
    )

    softmax = LogisticRegression(
        max_iter=int(args.extra["softmax_max_iter"]),
        class_weight=normalize_optional_none(args.extra["class_weight"]),
        random_state=int(args.seed),
    )

    softmax.fit(X_train_encoded, prepared.y_train)

    y_pred = softmax.predict(X_test_encoded)

    y_proba = None
    if hasattr(softmax, "predict_proba"):
        y_proba = softmax.predict_proba(X_test_encoded)

    metrics = compute_classification_metrics(
        y_true=prepared.y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        label_mapping=prepared.label_mapping,
        problem_type=args.problem_type,
    )

    autoencoder_path = artifact_dir / "autoencoder.keras"
    encoder_path = artifact_dir / "encoder.keras"
    softmax_path = artifact_dir / "softmax_regression.joblib"

    autoencoder.save(autoencoder_path)
    encoder.save(encoder_path)
    joblib.dump(softmax, softmax_path)

    save_preprocessing_artifacts(str(artifact_dir), prepared)

    history_path = artifact_dir / "autoencoder_training_history.json"
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
            "runner": "model05_self_taught_learning_softmax_regression",
            "paper": "05",
            "repository": "self_taught_learning_softmax_regression",
            "variant": "autoencoder_softmax_regression",
            "problem_type": args.problem_type,
            "label_col": label_col,
            "drop_columns": args.drop_columns,
            "input_dim": input_dim,
            "encoding_dim": encoding_dim,
            "hidden_layers": list(args.extra["hidden_layers"]),
            "autoencoder_epochs": int(args.extra["autoencoder_epochs"]),
            "batch_size": int(args.extra["batch_size"]),
            "dropout": float(args.extra["dropout"]),
            "lr": float(args.extra["lr"]),
            "softmax_max_iter": int(args.extra["softmax_max_iter"]),
            "class_weight": args.extra["class_weight"],
            "autoencoder_path": str(autoencoder_path),
            "encoder_path": str(encoder_path),
            "softmax_path": str(softmax_path),
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
        metrics = run_self_taught_softmax(args)

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