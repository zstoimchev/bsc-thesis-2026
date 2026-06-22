from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from common.data_loading import load_dataset_from_record
from common.io_utils import append_csv_row, read_json, utc_now_iso
from common.metrics import compact_metrics_for_csv
from common.paths import (
    ARTIFACTS_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    ensure_project_dirs,
    resolve_path,
)
from common.registry import (
    RegistryRecord,
    get_dataset,
    get_model,
    get_record,
    load_datasets,
    load_models,
    load_splits,
)
from common.splitting import get_or_create_split


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "enabled"}

    return bool(value)


def safe_run_id(text: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return "".join(ch if ch in allowed else "_" for ch in text)


def timestamp_for_run() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_models(models: list[RegistryRecord]) -> None:
    print(f"Models: {len(models)}")

    for model in models:
        enabled = bool_value(model.get("enabled"), default=True)
        trainable = bool_value(model.get("trainable"), default=False)

        print(
            f"- {model['id']} | paper={model.get('paper')} | "
            f"family={model.get('family')} | enabled={enabled} | trainable={trainable}"
        )


def print_datasets(datasets: list[RegistryRecord]) -> None:
    print(f"Datasets: {len(datasets)}")

    for dataset in datasets:
        enabled = bool_value(dataset.get("enabled"), default=True)
        split_type = dataset.get("split_type", "generated")

        print(
            f"- {dataset['id']} | split_type={split_type} | "
            f"enabled={enabled} | smoke_test={dataset.get('smoke_test', False)}"
        )


def print_splits(splits: list[RegistryRecord]) -> None:
    print(f"Splits: {len(splits)}")

    for split in splits:
        enabled = bool_value(split.get("enabled"), default=True)

        print(
            f"- {split['id']} | dataset={split.get('dataset_id')} | "
            f"enabled={enabled} | method={split.get('method')}"
        )


def find_default_split_for_dataset(
    dataset_id: str,
    splits: list[RegistryRecord],
    smoke_test: bool | None = None,
) -> RegistryRecord | None:
    candidates = [
        split
        for split in splits
        if split.get("dataset_id") == dataset_id
        and bool_value(split.get("enabled"), default=True)
    ]

    if smoke_test is not None:
        filtered = [
            split
            for split in candidates
            if bool_value(split.get("smoke_test"), default=False) == smoke_test
        ]

        if filtered:
            candidates = filtered

    if not candidates:
        return None

    return candidates[0]


def prepare_split(
    dataset_record: RegistryRecord,
    split_record: RegistryRecord,
    nrows: int | None = None,
) -> None:
    dataset_id = str(dataset_record["id"])
    split_id = str(split_record["id"])

    if dataset_record.get("split_type") == "predefined":
        print(
            f"[split] Dataset {dataset_id} already has predefined train/test files; "
            f"no generated split needed."
        )
        return

    print(f"[split] Loading dataset {dataset_id} for split {split_id}...")

    loaded = load_dataset_from_record(dataset_record, nrows=nrows)

    if loaded.full_df is None:
        raise RuntimeError(f"Dataset {dataset_id} did not load as a single dataframe.")

    problem_type = str(dataset_record.get("problem_type", "binary"))

    split_result = get_or_create_split(
        split_record=split_record,
        df=loaded.full_df,
        label_col=loaded.label_col,
        problem_type=problem_type,
    )

    action = "created" if split_result.created else "loaded existing"

    print(
        f"[split] {action}: {split_id} | "
        f"train={len(split_result.train_indices)} | "
        f"test={len(split_result.test_indices)}"
    )
    print(f"[split] train_indices: {split_result.train_indices_path}")
    print(f"[split] test_indices:  {split_result.test_indices_path}")


def selected_models_from_args(args: argparse.Namespace) -> list[RegistryRecord]:
    models = load_models()

    if args.models:
        selected = []
        for model_id in args.models:
            selected.append(get_record(models, model_id))
        return selected

    if args.all_enabled:
        return [
            model
            for model in models
            if bool_value(model.get("enabled"), default=True)
        ]

    raise ValueError("Select models with --models MODEL_ID ... or use --all-enabled.")


def dataset_args_for_runner(
    dataset_record: RegistryRecord | None,
    split_record: RegistryRecord | None,
    is_trainable: bool,
) -> list[str]:
    if not is_trainable:
        return []

    if dataset_record is None:
        raise ValueError("Trainable models require a dataset record.")

    args: list[str] = []

    split_type = str(dataset_record.get("split_type", "generated"))

    if split_type == "predefined":
        train_path = dataset_record.get("train_path")
        test_path = dataset_record.get("test_path")

        if not train_path or not test_path:
            raise ValueError(
                f"Dataset {dataset_record['id']} is predefined but missing train_path/test_path."
            )

        args.extend(["--train-path", str(train_path)])
        args.extend(["--test-path", str(test_path)])
        return args

    dataset_path = dataset_record.get("path")

    if not dataset_path:
        raise ValueError(f"Dataset {dataset_record['id']} is missing path.")

    if split_record is None:
        raise ValueError(
            f"Generated dataset {dataset_record['id']} requires a split record."
        )

    args.extend(["--dataset-path", str(dataset_path)])
    args.extend(["--split-id", str(split_record["id"])])

    return args


def smoke_extra_args_for_model(model_id: str) -> list[str]:
    """
    Small arguments used only for fast smoke tests.

    These keep neural models to one epoch and reduce tree ensemble sizes.
    The mapping is intentionally conservative and only uses arguments that the
    corresponding runner files support.
    """
    if model_id in {
        "model01_deeplearning_ids_keras_tf_mlp",
        "model02_ddos_traffic_research_dnn",
    }:
        return ["--epochs", "1", "--batch-size", "4", "--layers", "8"]

    if model_id == "model14_dnn_ids_mlp":
        return ["--epochs", "1", "--batch-size", "4", "--hidden-layers", "8"]

    if model_id == "model04_intrusion_detection_nsl_kdd_bigru_mlp":
        return ["--epochs", "1", "--batch-size", "4", "--gru-units", "8", "--dense-layers", "8"]

    if model_id == "model04_intrusion_detection_nsl_kdd_gru_mlp":
        return ["--epochs", "1", "--batch-size", "4", "--gru-units", "8", "--dense-layers", "8"]

    if model_id == "model04_intrusion_detection_nsl_kdd_blstm_mlp":
        return ["--epochs", "1", "--batch-size", "4", "--lstm-units", "8", "--dense-layers", "8"]
    
    if model_id == "model04_intrusion_detection_nsl_kdd_lstm_mlp":
        return ["--epochs", "1", "--batch-size", "4", "--lstm-units", "8", "--dense-layers", "8"]

    if model_id == "model05_self_taught_learning_softmax_regression":
        return [
            "--autoencoder-epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-layers",
            "8",
            "--encoding-dim",
            "4",
        ]

    if model_id == "model06_cic_ddos2019_deeplearning_gru":
        return ["--epochs", "1", "--batch-size", "4", "--gru-units", "8", "--dense-layers", "8"]

    if model_id == "model06_cic_ddos2019_deeplearning_lstm":
        return ["--epochs", "1", "--batch-size", "4", "--lstm-units", "8", "--dense-layers", "8"]

    if model_id == "model06_cic_ddos2019_deeplearning_cnn_lstm":
        return [
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--conv-filters",
            "8",
            "--kernel-size",
            "2",
            "--pool-size",
            "2",
            "--lstm-units",
            "8",
            "--dense-layers",
            "8",
        ]

    if model_id == "model07_cnn_ids_1d_cnn":
        return [
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--conv-filters",
            "8",
            "--kernel-size",
            "2",
            "--pool-size",
            "2",
            "--dense-layers",
            "8",
        ]

    if model_id == "model07_cnn_ids_cnn_gru":
        return [
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--conv-filters",
            "8",
            "--kernel-size",
            "2",
            "--pool-size",
            "2",
            "--gru-units",
            "8",
            "--dense-layers",
            "8",
        ]

    if model_id == "model07_cnn_ids_cnn_lstm":
        return [
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--conv-filters",
            "8",
            "--kernel-size",
            "2",
            "--pool-size",
            "2",
            "--lstm-units",
            "8",
            "--dense-layers",
            "8",
        ]

    if model_id == "model08_deep_shallow_networks_stacked_autoencoder_mlp":
        return [
            "--pretrain-epochs",
            "1",
            "--classifier-epochs",
            "1",
            "--batch-size",
            "4",
            "--encoder-layers",
            "8",
            "--encoding-dim",
            "4",
            "--classifier-layers",
            "8",
        ]

    if model_id == "model09_subset_feature_elimination_decision_tree":
        return ["--anova-k", "2", "--rfe-features", "1"]

    if model_id == "model10_clustering_anomaly_detection_kmeans":
        return ["--n-clusters", "2"]

    if model_id in {
        "model02_ddos_traffic_research_xgboost",
        "model11_network_intrusion_detection_system_xgboost",
    }:
        return ["--n-estimators", "5", "--max-depth", "2", "--learning-rate", "0.1"]

    if model_id in {
        "model02_ddos_traffic_research_random_forest",
        "model11_network_intrusion_detection_system_random_forest",
    }:
        return ["--n-estimators", "5", "--max-depth", "2"]

    if model_id == "model12_ant_colony_induced_decision_tree":
        return ["--feature-k", "2"]

    if model_id == "model15_deep_energy_anomaly_detector":
        return [
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-layers",
            "8",
            "--encoding-dim",
            "4",
            "--threshold-percentile",
            "95",
        ]

    return []


def build_runner_command(
    model_record: RegistryRecord,
    dataset_record: RegistryRecord | None,
    split_record: RegistryRecord | None,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    artifact_dir: Path,
) -> list[str]:
    model_id = str(model_record["id"])
    runner = str(model_record["runner"])
    trainable = bool_value(model_record.get("trainable"), default=False)

    dataset_id = str(dataset_record["id"]) if dataset_record else "none"
    problem_type = str(args.problem_type or (dataset_record or {}).get("problem_type", "binary"))

    mode = "train-eval" if trainable else "audit"
    stage = str(model_record.get("stage", "unified" if trainable else "audit"))

    cmd = [
        sys.executable,
        "-m",
        runner,
        "--mode",
        mode,
        "--stage",
        stage,
        "--model-id",
        model_id,
        "--dataset-id",
        dataset_id,
        "--problem-type",
        problem_type,
        "--artifact-dir",
        str(artifact_dir),
        "--run-dir",
        str(run_dir),
        "--metrics-out",
        str(run_dir / "metrics.json"),
        "--seed",
        str(args.seed),
    ]

    if args.nrows is not None and trainable:
        cmd.extend(["--nrows", str(args.nrows)])

    label_col = args.label_col or (dataset_record or {}).get("label_col")
    if label_col is not None and trainable:
        cmd.extend(["--label-col", str(label_col)])

    drop_columns = list((dataset_record or {}).get("drop_columns", []) or [])
    if args.drop_columns:
        drop_columns.extend(args.drop_columns)

    if drop_columns and trainable:
        cmd.extend(["--drop-columns", ",".join(str(col) for col in drop_columns)])

    cmd.extend(dataset_args_for_runner(dataset_record, split_record, trainable))

    if args.smoke_test:
        cmd.extend(smoke_extra_args_for_model(model_id))

    return cmd


def run_one_model(
    model_record: RegistryRecord,
    dataset_record: RegistryRecord | None,
    split_record: RegistryRecord | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_id = str(model_record["id"])
    dataset_id = str(dataset_record["id"]) if dataset_record else "none"

    run_id = safe_run_id(
        f"{timestamp_for_run()}_{model_id}_{dataset_id}"
    )

    run_dir = RUNS_DIR / run_id
    artifact_dir = ARTIFACTS_DIR / run_id

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_runner_command(
        model_record=model_record,
        dataset_record=dataset_record,
        split_record=split_record,
        args=args,
        run_id=run_id,
        run_dir=run_dir,
        artifact_dir=artifact_dir,
    )

    print("\n" + "=" * 100)
    print(f"[run] {model_id}")
    print(f"[run] dataset={dataset_id}")
    print(f"[run] run_id={run_id}")
    print("[cmd] " + " ".join(cmd))

    if args.dry_run:
        return {
            "run_id": run_id,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "status": "dry_run",
            "command": " ".join(cmd),
            "run_dir": str(run_dir),
            "artifact_dir": str(artifact_dir),
        }

    start = time.time()

    process = subprocess.run(
        cmd,
        cwd=str(resolve_path(".")),
        text=True,
    )

    duration = round(time.time() - start, 4)

    metrics_path = run_dir / "metrics.json"

    if metrics_path.exists():
        try:
            metrics = read_json(metrics_path)
        except Exception:
            metrics = {}
    else:
        metrics = {}

    metrics.setdefault("run_id", run_id)
    metrics.setdefault("model_id", model_id)
    metrics.setdefault("dataset_id", dataset_id)
    metrics.setdefault("split_id", split_record.get("id") if split_record else None)
    metrics.setdefault("duration_seconds", duration)
    metrics.setdefault("run_dir", str(run_dir))
    metrics.setdefault("artifact_dir", str(artifact_dir))

    if process.returncode != 0:
        metrics["status"] = "failed"
        metrics["error"] = f"Runner returned exit code {process.returncode}"

    index_row = compact_metrics_for_csv(metrics)
    index_row["run_id"] = run_id
    index_row["model_id"] = model_id
    index_row["dataset_id"] = dataset_id
    index_row["split_id"] = split_record.get("id") if split_record else ""
    index_row["returncode"] = process.returncode
    index_row["run_dir"] = str(run_dir)
    index_row["artifact_dir"] = str(artifact_dir)
    index_row["indexed_utc"] = utc_now_iso()

    append_csv_row(RESULTS_DIR / "runs_index.csv", index_row)

    print(
        f"[done] {model_id} | status={metrics.get('status')} | "
        f"accuracy={metrics.get('accuracy')} | f1_macro={metrics.get('f1_macro')}"
    )

    if process.returncode != 0 and not args.continue_on_error:
        raise RuntimeError(f"Stopping after failed runner: {model_id}")

    return metrics


def run_models(args: argparse.Namespace) -> None:
    selected_models = selected_models_from_args(args)

    dataset_record: RegistryRecord | None = None
    split_record: RegistryRecord | None = None

    has_trainable = any(
        bool_value(model.get("trainable"), default=False)
        for model in selected_models
    )

    if has_trainable:
        if not args.dataset_id:
            raise ValueError("Trainable runs require --dataset-id.")

        dataset_record = get_dataset(args.dataset_id)

        if not bool_value(dataset_record.get("enabled"), default=True) and not args.allow_disabled_dataset:
            raise ValueError(
                f"Dataset {args.dataset_id} is disabled. "
                "Use --allow-disabled-dataset if you really want to run it."
            )

        if dataset_record.get("split_type") != "predefined":
            splits = load_splits()

            if args.split_id:
                split_record = get_record(splits, args.split_id)
            else:
                split_record = find_default_split_for_dataset(
                    dataset_id=str(dataset_record["id"]),
                    splits=splits,
                    smoke_test=True if args.smoke_test else None,
                )

            if split_record is None:
                raise ValueError(
                    f"No enabled split found for dataset {dataset_record['id']}."
                )

            prepare_split(
                dataset_record=dataset_record,
                split_record=split_record,
                nrows=args.nrows_for_split,
            )

    completed = 0
    failed = 0

    for model in selected_models:
        try:
            trainable = bool_value(model.get("trainable"), default=False)

            current_dataset = dataset_record if trainable else None
            current_split = split_record if trainable else None

            run_one_model(
                model_record=model,
                dataset_record=current_dataset,
                split_record=current_split,
                args=args,
            )

            completed += 1

        except Exception as exc:
            failed += 1
            print(f"[orchestrate failed] {model['id']}: {exc}")

            if not args.continue_on_error:
                raise

    print("\n" + "=" * 100)
    print(f"[summary] completed={completed} failed={failed}")
    print(f"[summary] runs index: {RESULTS_DIR / 'runs_index.csv'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental lab orchestrator for thesis model audit and runs."
    )

    action = parser.add_mutually_exclusive_group(required=True)

    action.add_argument("--list", action="store_true", help="List registered models, datasets, and splits.")
    action.add_argument("--prepare-split", action="store_true", help="Prepare a dataset split from registries.")
    action.add_argument("--run", action="store_true", help="Run selected models.")

    parser.add_argument("--models", nargs="+", default=None, help="Model ids to run.")
    parser.add_argument("--all-enabled", action="store_true", help="Run all enabled models.")

    parser.add_argument("--dataset-id", default=None, help="Dataset id from registries/datasets.yaml.")
    parser.add_argument("--split-id", default=None, help="Split id from registries/splits.yaml.")
    parser.add_argument("--problem-type", default=None, choices=["binary", "multiclass"])

    parser.add_argument("--label-col", default=None, help="Override label column.")
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        help="Additional columns to drop.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit passed to runners.")
    parser.add_argument(
        "--nrows-for-split",
        type=int,
        default=None,
        help="Optional row limit only for split creation/debugging.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use small hyperparameters for quick runner smoke tests.",
    )
    parser.add_argument(
        "--allow-disabled-dataset",
        action="store_true",
        help="Allow running datasets marked enabled=false.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining models if one model fails.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ensure_project_dirs()

    args = parse_args(argv)

    try:
        if args.list:
            print_models(load_models())
            print()
            print_datasets(load_datasets())
            print()
            print_splits(load_splits())
            return 0

        if args.prepare_split:
            if not args.dataset_id:
                raise ValueError("--prepare-split requires --dataset-id.")

            dataset_record = get_dataset(args.dataset_id)

            if args.split_id:
                split_record = get_record(load_splits(), args.split_id)
            else:
                split_record = find_default_split_for_dataset(
                    dataset_id=args.dataset_id,
                    splits=load_splits(),
                    smoke_test=True if args.smoke_test else None,
                )

            if split_record is None:
                raise ValueError(f"No split found for dataset {args.dataset_id}.")

            prepare_split(
                dataset_record=dataset_record,
                split_record=split_record,
                nrows=args.nrows_for_split,
            )
            return 0

        if args.run:
            run_models(args)
            return 0

        raise RuntimeError("No action selected.")

    except Exception as exc:
        print(f"[orchestrate error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())