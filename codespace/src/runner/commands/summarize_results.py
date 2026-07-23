import csv
import json
from pathlib import Path
from typing import Any

from src.common.registry import load_model_registry
from src.runner.constants import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "results" / "runs"
MANIFEST_PATH = PROJECT_ROOT / "results" / "final_runs.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "final"
OUTPUT_CSV = OUTPUT_DIR / "final_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "final_results.json"


def add_summarize_results_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "summarize-results",
        help="Summarize the canonical final experiment runs.",
    )

    parser.add_argument(
        "--split-id",
        default=None,
        help="Only include canonical evaluations performed on this split.",
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_optional_json(path: Path) -> dict:
    return load_json(path) if path.exists() else {}


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def label_count(distribution: dict, label: int) -> int | None:
    value = distribution.get(str(label))
    return int(value) if value is not None else None


def load_split_metadata(split_id: str) -> dict:
    if not split_id:
        return {}

    path = PROJECT_ROOT / "data" / "prepared" / split_id / "metadata.json"
    return load_optional_json(path)


def resolve_training_run(run_info: dict, evaluation_run: Path) -> Path:
    model_path_value = run_info.get("model_path")

    if not model_path_value:
        return evaluation_run

    training_run = Path(model_path_value).parent

    if training_run.exists():
        return training_run

    relocated_run = RUNS_DIR / training_run.name
    return relocated_run if relocated_run.exists() else evaluation_run


def get_training_distribution(training_summary: dict, metrics: dict) -> dict:
    return (
            training_summary.get("training_label_counts")
            or training_summary.get("sampled_label_counts")
            or training_summary.get("training_sample_label_counts")
            or training_summary.get("label_counts")
            or metrics.get("training_label_counts")
            or metrics.get("training_sample_label_counts")
            or {}
    )


def get_evaluation_distribution(metrics: dict, metadata: dict) -> dict:
    return (
            metrics.get("evaluation_label_counts")
            or metrics.get("support")
            or metadata.get("test_label_distribution")
            or {}
    )


def validate_manifest_run(
        experiment_id: str,
        experiment_cfg: dict,
        model_id: str,
        run_dir: Path,
        run_info: dict,
) -> None:
    if run_info.get("model_id") != model_id:
        raise ValueError(
            f"Manifest model mismatch for {experiment_id}/{model_id}: run contains {run_info.get('model_id')!r}."
        )

    expected_split = experiment_cfg.get("evaluation_split_id")
    actual_split = run_info.get("split_id")

    if expected_split and actual_split != expected_split:
        raise ValueError(
            f"Manifest split mismatch for {experiment_id}/{model_id}: expected {expected_split!r}, found {actual_split!r}."
        )

    expected_feature_set = experiment_cfg.get("feature_set_id")
    actual_feature_set = run_info.get("feature_set_id")

    if expected_feature_set and actual_feature_set != expected_feature_set:
        raise ValueError(
            f"Manifest feature-set mismatch for {experiment_id}/{model_id}: expected {expected_feature_set!r}, found {actual_feature_set!r}."
        )

    if not (run_dir / "metrics.json").exists():
        raise FileNotFoundError(f"Missing metrics.json in canonical run: {run_dir}")


def build_row(
        experiment_id: str,
        experiment_cfg: dict,
        model_id: str,
        model_cfg: dict,
        evaluation_run: Path,
) -> dict:
    evaluation_run_info = load_json(evaluation_run / "run_info.json")
    metrics = load_json(evaluation_run / "metrics.json")

    validate_manifest_run(
        experiment_id=experiment_id,
        experiment_cfg=experiment_cfg,
        model_id=model_id,
        run_dir=evaluation_run,
        run_info=evaluation_run_info,
    )

    training_run = resolve_training_run(evaluation_run_info, evaluation_run)
    training_summary = load_optional_json(training_run / "training_summary.json")
    training_run_info = load_optional_json(training_run / "run_info.json")

    trained_split = first_not_none(
        metrics.get("training_split_id"),
        training_summary.get("split_id"),
        evaluation_run_info.get("source_training_split_id"),
        training_run_info.get("split_id"),
        "",
    )
    evaluated_split = evaluation_run_info.get("split_id", "")

    training_metadata = load_split_metadata(trained_split)
    evaluation_metadata = load_split_metadata(evaluated_split)

    training_distribution = get_training_distribution(training_summary, metrics)
    evaluation_distribution = get_evaluation_distribution(metrics, evaluation_metadata)

    confusion_matrix = metrics.get("confusion_matrix") or [[None, None], [None, None]]
    feature_columns = (
            metrics.get("feature_columns")
            or training_summary.get("feature_columns")
            or evaluation_run_info.get("feature_columns")
            or []
    )

    full_training_rows = first_not_none(
        training_summary.get("full_training_rows"),
        metrics.get("full_training_rows"),
        training_metadata.get("train_rows"),
    )
    training_rows_used = first_not_none(
        training_summary.get("training_rows"),
        training_summary.get("sampled_training_rows"),
        training_summary.get("training_sample_rows"),
        metrics.get("training_rows"),
        metrics.get("training_sample_rows"),
    )
    train_row_cap = first_not_none(
        training_summary.get("train_row_cap"),
        metrics.get("train_row_cap"),
        evaluation_run_info.get("train_row_cap"),
    )

    if (
            train_row_cap is None
            and full_training_rows is not None
            and training_rows_used is not None
            and training_rows_used < full_training_rows
    ):
        train_row_cap = training_rows_used

    training_seconds = training_run_info.get("training_seconds")
    evaluation_seconds = first_not_none(
        evaluation_run_info.get("evaluation_seconds"),
        metrics.get("evaluation_seconds"),
    )
    total_seconds = (
        training_seconds + evaluation_seconds
        if training_seconds is not None and evaluation_seconds is not None
        else None
    )

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_cfg.get("name", experiment_id),
        "experiment_order": experiment_cfg.get("order"),

        "model": model_id,
        "model_name": model_cfg.get("name", evaluation_run_info.get("model_name", "")),
        "seed": evaluation_run_info.get("seed"),

        "trained_on_split": trained_split,
        "trained_on_dataset": first_not_none(
            training_metadata.get("dataset_id"),
            training_run_info.get("dataset_id"),
            "",
        ),
        "evaluated_on_split": evaluated_split,
        "evaluated_on_dataset": evaluation_run_info.get("dataset_id", ""),
        "feature_set": evaluation_run_info.get("feature_set_id", ""),
        "split_method": evaluation_metadata.get("split_method", {}).get("type", ""),

        "full_training_rows": full_training_rows,
        "training_rows_used": training_rows_used,
        "training_label_0": label_count(training_distribution, 0),
        "training_label_1": label_count(training_distribution, 1),

        "evaluation_rows": first_not_none(
            metrics.get("evaluation_rows"),
            evaluation_metadata.get("test_rows"),
        ),
        "evaluation_label_0": label_count(evaluation_distribution, 0),
        "evaluation_label_1": label_count(evaluation_distribution, 1),

        "num_features": len(feature_columns),
        "feature_columns": feature_columns,

        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),

        "tn": confusion_matrix[0][0],
        "fp": confusion_matrix[0][1],
        "fn": confusion_matrix[1][0],
        "tp": confusion_matrix[1][1],

        "train_row_cap": train_row_cap,
        "epochs": first_not_none(metrics.get("epochs"), training_summary.get("epochs")),
        "batch_size": first_not_none(metrics.get("batch_size"), training_summary.get("batch_size")),
        "learning_rate": first_not_none(metrics.get("learning_rate"), training_summary.get("learning_rate")),
        "threshold": first_not_none(metrics.get("threshold"), training_summary.get("threshold")),
        "threshold_tuned": first_not_none(
            metrics.get("threshold_tuned"),
            training_summary.get("threshold_tuned"),
            False,
        ),
        "use_class_weight": first_not_none(
            metrics.get("use_class_weight"),
            training_summary.get("use_class_weight"),
            False,
        ),

        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": total_seconds,

        "model_configuration": (
                training_summary.get("architecture")
                or training_summary.get("params")
                or metrics.get("architecture")
                or metrics.get("params")
                or {}
        ),

        "training_run": training_run.name,
        "evaluation_run": evaluation_run.name,
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    return value


def metric_text(value: Any) -> str:
    return "-" if value is None else f"{value:.4f}"


def print_summary(rows: list[dict]) -> None:
    current_experiment = None

    for row in rows:
        if row["experiment_id"] != current_experiment:
            current_experiment = row["experiment_id"]
            print()
            print(f"{row['experiment_name']} [{current_experiment}]")
            print(
                f"{'Model':<20}"
                f"{'Accuracy':>10}"
                f"{'BalAcc':>10}"
                f"{'Precision':>10}"
                f"{'Recall':>10}"
                f"{'F1':>10}"
            )
            print("-" * 70)

        print(
            f"{row['model']:<20}"
            f"{metric_text(row['accuracy']):>10}"
            f"{metric_text(row['balanced_accuracy']):>10}"
            f"{metric_text(row['precision']):>10}"
            f"{metric_text(row['recall']):>10}"
            f"{metric_text(row['f1']):>10}"
        )


def run_summarize_results(args) -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Canonical run manifest not found: {MANIFEST_PATH}")

    manifest = load_json(MANIFEST_PATH)
    experiments = manifest.get("experiments")

    if not isinstance(experiments, dict) or not experiments:
        raise ValueError(f"Manifest must contain a non-empty 'experiments' object: {MANIFEST_PATH}")

    model_registry = load_model_registry()
    rows = []

    ordered_experiments = sorted(
        experiments.items(),
        key=lambda item: (item[1].get("order", 999), item[0]),
    )

    for experiment_id, experiment_cfg in ordered_experiments:
        evaluation_split_id = experiment_cfg.get("evaluation_split_id")

        if args.split_id is not None and evaluation_split_id != args.split_id:
            continue

        runs = experiment_cfg.get("runs")

        if not isinstance(runs, dict) or not runs:
            raise ValueError(f"Experiment {experiment_id!r} has no canonical runs.")

        for model_id, run_name in runs.items():
            run_dir = RUNS_DIR / run_name

            if not run_dir.exists():
                raise FileNotFoundError(f"Canonical run directory does not exist: {run_dir}")

            model_cfg = model_registry.get(model_id, {})
            rows.append(
                build_row(
                    experiment_id=experiment_id,
                    experiment_cfg=experiment_cfg,
                    model_id=model_id,
                    model_cfg=model_cfg,
                    evaluation_run=run_dir,
                )
            )

    if not rows:
        split_message = f" for split {args.split_id}" if args.split_id else ""
        print(f"[summary] No canonical results found{split_message}.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()

        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})

    print_summary(rows)

    print()
    print(f"[summary] Rows: {len(rows)}")
    print(f"[summary] CSV:  {OUTPUT_CSV}")
    print(f"[summary] JSON: {OUTPUT_JSON}")
