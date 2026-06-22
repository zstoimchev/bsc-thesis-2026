from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import KMeans

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
    write_failure_result,
)


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 10 / Cluster-Based Anomaly Detection K-Means runner."
    )
    add_common_runner_args(parser)

    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-init", type=int, default=10)

    parser.add_argument(
        "--source-path",
        default="external/model10_clustering-based-anomaly-detection",
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
            "paper": "10",
            "repository": "clustering-based-anomaly-detection",
            "variant": "kmeans_cluster_majority_mapping",
            "source_path": parsed.source_path,
            "n_clusters": parsed.n_clusters,
            "max_iter": parsed.max_iter,
            "n_init": parsed.n_init,
        },
    )


def build_cluster_to_label_mapping(
    cluster_ids: np.ndarray,
    y_train: np.ndarray,
) -> dict[int, int]:
    """
    Map each cluster to the majority training label found inside that cluster.

    This makes the unsupervised K-Means result evaluable with the same supervised
    metrics used for the rest of the thesis audit.
    """
    cluster_to_label: dict[int, int] = {}

    global_majority_label = int(Counter(y_train.tolist()).most_common(1)[0][0])

    for cluster_id in sorted(set(cluster_ids.tolist())):
        mask = cluster_ids == cluster_id
        labels_in_cluster = y_train[mask]

        if len(labels_in_cluster) == 0:
            cluster_to_label[int(cluster_id)] = global_majority_label
            continue

        majority_label = int(Counter(labels_in_cluster.tolist()).most_common(1)[0][0])
        cluster_to_label[int(cluster_id)] = majority_label

    return cluster_to_label


def predict_labels_from_clusters(
    cluster_ids: np.ndarray,
    cluster_to_label: dict[int, int],
    default_label: int,
) -> np.ndarray:
    return np.asarray(
        [
            int(cluster_to_label.get(int(cluster_id), int(default_label)))
            for cluster_id in cluster_ids
        ]
    )


def run_kmeans(args: RunnerArgs) -> dict[str, Any]:
    start = time.time()

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

    X_train_array = np.asarray(prepared.X_train)
    X_test_array = np.asarray(prepared.X_test)

    safe_n_clusters = max(
        1,
        min(
            int(args.extra["n_clusters"]),
            int(len(X_train_array)),
            int(len(set(prepared.y_train.tolist()))),
        ),
    )

    kmeans = KMeans(
        n_clusters=safe_n_clusters,
        max_iter=int(args.extra["max_iter"]),
        n_init=int(args.extra["n_init"]),
        random_state=int(args.seed),
    )

    train_clusters = kmeans.fit_predict(X_train_array)
    test_clusters = kmeans.predict(X_test_array)

    cluster_to_label = build_cluster_to_label_mapping(
        cluster_ids=train_clusters,
        y_train=prepared.y_train,
    )

    default_label = int(Counter(prepared.y_train.tolist()).most_common(1)[0][0])

    y_pred = predict_labels_from_clusters(
        cluster_ids=test_clusters,
        cluster_to_label=cluster_to_label,
        default_label=default_label,
    )

    metrics = compute_classification_metrics(
        y_true=prepared.y_test,
        y_pred=y_pred,
        y_proba=None,
        label_mapping=prepared.label_mapping,
        problem_type=args.problem_type,
    )

    model_path = artifact_dir / "kmeans.joblib"
    mapping_path = artifact_dir / "cluster_to_label_mapping.json"

    joblib.dump(kmeans, model_path)

    write_json(
        mapping_path,
        {
            "cluster_to_label": {
                str(k): int(v)
                for k, v in cluster_to_label.items()
            },
            "default_label": int(default_label),
            "label_mapping": prepared.label_mapping,
        },
    )

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
            "runner": "model10_clustering_anomaly_detection_kmeans",
            "paper": "10",
            "repository": "clustering-based-anomaly-detection",
            "variant": "kmeans_cluster_majority_mapping",
            "problem_type": args.problem_type,
            "label_col": label_col,
            "drop_columns": args.drop_columns,
            "requested_n_clusters": int(args.extra["n_clusters"]),
            "actual_n_clusters": int(safe_n_clusters),
            "max_iter": int(args.extra["max_iter"]),
            "n_init": int(args.extra["n_init"]),
            "cluster_to_label": {
                str(k): int(v)
                for k, v in cluster_to_label.items()
            },
            "default_label": int(default_label),
            "model_artifact_path": str(model_path),
            "mapping_path": str(mapping_path),
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
        metrics = run_kmeans(args)

        print(
            f"[success] {args.model_id} on {args.dataset_id}: "
            f"accuracy={metrics.get('accuracy')}, "
            f"f1_macro={metrics.get('f1_macro')}, "
            f"clusters={metrics.get('actual_n_clusters')}"
        )

        return 0

    except Exception as exc:
        write_failure_result(args, exc)
        print(f"[failed] {args.model_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())