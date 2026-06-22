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


def parse_hidden_layers(value: str) -> list[int]:
    value = value.strip()

    if not value:
        return [128, 64]

    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 15 / Deep energy-based anomaly detector runner."
    )
    add_common_runner_args(parser)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-layers", default="128,64")
    parser.add_argument("--encoding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Energy threshold percentile computed from training normal samples.",
    )

    parser.add_argument(
        "--source-path",
        default="external/model15_deep_structured_energy_based_models",
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
            "paper": "15",
            "repository": "deep_structured_energy_based_models",
            "variant": "reconstruction_energy_anomaly_detector",
            "source_path": parsed.source_path,
            "epochs": parsed.epochs,
            "batch_size": parsed.batch_size,
            "hidden_layers": parse_hidden_layers(parsed.hidden_layers),
            "encoding_dim": parsed.encoding_dim,
            "dropout": parsed.dropout,
            "lr": parsed.lr,
            "threshold_percentile": parsed.threshold_percentile,
        },
    )


def find_normal_and_attack_labels(label_mapping: dict[int, str]) -> tuple[int, int]:
    normal_candidates = {"benign", "normal"}

    normal_label = None
    for label_id, label_name in label_mapping.items():
        if str(label_name).lower() in normal_candidates:
            normal_label = int(label_id)
            break

    if normal_label is None:
        normal_label = int(sorted(label_mapping.keys())[0])

    attack_candidates = [
        int(label_id)
        for label_id in label_mapping.keys()
        if int(label_id) != normal_label
    ]

    if not attack_candidates:
        attack_label = normal_label
    else:
        attack_label = int(attack_candidates[0])

    return normal_label, attack_label


def build_autoencoder(
    input_dim: int,
    hidden_layers: list[int],
    encoding_dim: int,
    dropout: float,
    lr: float,
):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(input_dim,))

    x = inputs

    for units in hidden_layers:
        x = tf.keras.layers.Dense(int(units), activation="relu")(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(float(dropout))(x)

    x = tf.keras.layers.Dense(int(encoding_dim), activation="relu", name="energy_bottleneck")(x)

    for units in reversed(hidden_layers):
        x = tf.keras.layers.Dense(int(units), activation="relu")(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(float(dropout))(x)

    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="energy_autoencoder")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss="mse",
    )

    return model


def reconstruction_energy(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    reconstructed = model.predict(X, batch_size=batch_size)
    return np.mean(np.square(X - reconstructed), axis=1)


def make_binary_probability_matrix(
    energies: np.ndarray,
    threshold: float,
    normal_label: int,
    attack_label: int,
    num_classes: int,
) -> np.ndarray:
    scale = float(np.std(energies)) + 1e-8
    anomaly_probability = 1.0 / (1.0 + np.exp(-((energies - threshold) / scale)))

    proba = np.zeros((len(energies), num_classes), dtype="float32")

    if normal_label == attack_label:
        proba[:, normal_label] = 1.0
        return proba

    proba[:, attack_label] = anomaly_probability
    proba[:, normal_label] = 1.0 - anomaly_probability

    return proba


def run_energy_detector(args: RunnerArgs) -> dict[str, Any]:
    try:
        import tensorflow as tf
    except ImportError:
        return write_audit_only_result(
            args=args,
            status="needs_dependency",
            message=(
                "TensorFlow is not installed. Install it with `pip install tensorflow` "
                "to execute this Paper 15 energy-based anomaly detector."
            ),
            details={
                "missing_dependency": "tensorflow",
                "paper": "15",
                "repository": "deep_structured_energy_based_models",
                "variant": "reconstruction_energy_anomaly_detector",
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
    num_classes = int(len(prepared.label_mapping))

    normal_label, attack_label = find_normal_and_attack_labels(prepared.label_mapping)

    normal_mask = prepared.y_train == normal_label

    if int(normal_mask.sum()) < 2:
        normal_mask = np.ones_like(prepared.y_train, dtype=bool)

    X_train_normal = X_train_array[normal_mask]

    model = build_autoencoder(
        input_dim=input_dim,
        hidden_layers=list(args.extra["hidden_layers"]),
        encoding_dim=int(args.extra["encoding_dim"]),
        dropout=float(args.extra["dropout"]),
        lr=float(args.extra["lr"]),
    )

    history = model.fit(
        X_train_normal,
        X_train_normal,
        validation_data=(X_test_array, X_test_array),
        epochs=int(args.extra["epochs"]),
        batch_size=int(args.extra["batch_size"]),
        verbose=1,
    )

    train_energies = reconstruction_energy(
        model=model,
        X=X_train_normal,
        batch_size=int(args.extra["batch_size"]),
    )

    test_energies = reconstruction_energy(
        model=model,
        X=X_test_array,
        batch_size=int(args.extra["batch_size"]),
    )

    threshold = float(
        np.percentile(train_energies, float(args.extra["threshold_percentile"]))
    )

    y_pred = np.where(test_energies > threshold, attack_label, normal_label)

    y_proba = make_binary_probability_matrix(
        energies=test_energies,
        threshold=threshold,
        normal_label=normal_label,
        attack_label=attack_label,
        num_classes=num_classes,
    )

    metrics = compute_classification_metrics(
        y_true=prepared.y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        label_mapping=prepared.label_mapping,
        problem_type=args.problem_type,
    )

    model_path = artifact_dir / "energy_autoencoder.keras"
    threshold_path = artifact_dir / "energy_threshold.json"
    history_path = artifact_dir / "training_history.json"

    model.save(model_path)

    save_preprocessing_artifacts(str(artifact_dir), prepared)

    write_json(
        threshold_path,
        {
            "threshold": threshold,
            "threshold_percentile": float(args.extra["threshold_percentile"]),
            "normal_label": int(normal_label),
            "attack_label": int(attack_label),
            "normal_label_name": prepared.label_mapping.get(int(normal_label)),
            "attack_label_name": prepared.label_mapping.get(int(attack_label)),
            "train_normal_samples": int(len(X_train_normal)),
            "train_energy_mean": float(np.mean(train_energies)),
            "train_energy_std": float(np.std(train_energies)),
            "test_energy_mean": float(np.mean(test_energies)),
            "test_energy_std": float(np.std(test_energies)),
        },
    )

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
            "runner": "model15_deep_energy_anomaly_detector",
            "paper": "15",
            "repository": "deep_structured_energy_based_models",
            "variant": "reconstruction_energy_anomaly_detector",
            "exact_dsebm_implementation": False,
            "problem_type": args.problem_type,
            "label_col": label_col,
            "drop_columns": args.drop_columns,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "normal_label": int(normal_label),
            "attack_label": int(attack_label),
            "normal_training_samples": int(len(X_train_normal)),
            "epochs": int(args.extra["epochs"]),
            "batch_size": int(args.extra["batch_size"]),
            "hidden_layers": list(args.extra["hidden_layers"]),
            "encoding_dim": int(args.extra["encoding_dim"]),
            "dropout": float(args.extra["dropout"]),
            "lr": float(args.extra["lr"]),
            "threshold_percentile": float(args.extra["threshold_percentile"]),
            "energy_threshold": threshold,
            "model_artifact_path": str(model_path),
            "threshold_path": str(threshold_path),
            "history_path": str(history_path),
            "artifact_dir": str(artifact_dir),
            "run_dir": str(run_dir),
            "source_path": args.extra.get("source_path"),
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
        metrics = run_energy_detector(args)

        status = metrics.get("status")

        if status == "success":
            print(
                f"[success] {args.model_id} on {args.dataset_id}: "
                f"accuracy={metrics.get('accuracy')}, "
                f"f1_macro={metrics.get('f1_macro')}, "
                f"threshold={metrics.get('energy_threshold')}"
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