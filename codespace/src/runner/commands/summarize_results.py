import csv
import json
from pathlib import Path

from src.common.registry import load_model_registry
from src.runner.constants import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "results" / "runs"
OUTPUT_CSV = PROJECT_ROOT / "results" / "latest_summary.csv"
OUTPUT_JSON = PROJECT_ROOT / "results" / "latest_summary.json"


def add_summarize_results_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "summarize-results",
        help="Summarize the latest completed evaluation for each enabled model.",
    )

    parser.add_argument(
        "--split-id",
        default=None,
        help="Only use evaluations performed on this split.",
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_split_metadata(split_id: str) -> dict:
    path = PROJECT_ROOT / "data" / "prepared" / split_id / "metadata.json"
    return load_json(path) if path.exists() else {}


def resolve_training_run(run_info: dict) -> Path:
    model_path = Path(run_info["model_path"])
    training_run = model_path.parent

    if training_run.exists():
        return training_run

    # Handles a project moved to another machine/path.
    return RUNS_DIR / training_run.name


def latest_completed_run(
        model_id: str,
        split_id: str | None = None,
) -> Path | None:
    candidates = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        run_info_path = run_dir / "run_info.json"
        metrics_path = run_dir / "metrics.json"

        if not run_info_path.exists() or not metrics_path.exists():
            continue

        run_info = load_json(run_info_path)

        if run_info.get("model_id") != model_id:
            continue

        if split_id is not None and run_info.get("split_id") != split_id:
            continue

        candidates.append(run_dir)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: (path / "metrics.json").stat().st_mtime,
    )


def label_count(distribution: dict, label: int) -> int | None:
    value = distribution.get(str(label))
    return int(value) if value is not None else None


def build_row(model_id: str, model_cfg: dict, run_dir: Path) -> dict:
    run_info = load_json(run_dir / "run_info.json")
    metrics = load_json(run_dir / "metrics.json")

    training_run = resolve_training_run(run_info)
    training_summary = load_json(training_run / "training_summary.json")

    trained_split = training_summary.get("split_id", "")
    evaluated_split = run_info.get("split_id", "")

    training_metadata = load_split_metadata(trained_split)
    evaluation_metadata = load_split_metadata(evaluated_split)

    training_distribution = (
            training_summary.get("sampled_label_counts")
            or training_summary.get("label_counts")
            or {}
    )

    test_distribution = evaluation_metadata.get(
        "test_label_distribution",
        {},
    )

    confusion_matrix = metrics.get(
        "confusion_matrix",
        [[None, None], [None, None]],
    )

    feature_columns = (
            metrics.get("feature_columns")
            or training_summary.get("feature_columns")
            or []
    )

    return {
        "model": model_id,
        "model_name": model_cfg.get("name", ""),
        "seed": run_info.get("seed"),
        "trained_on_split": trained_split,
        "trained_on_dataset": training_metadata.get("dataset_id", ""),
        "evaluated_on_split": evaluated_split,
        "evaluated_on_dataset": run_info.get("dataset_id", ""),
        "feature_set": run_info.get("feature_set_id", ""),
        "split_method": (evaluation_metadata.get("split_method", {}).get("type", "")),
        "source_rows": evaluation_metadata.get("source_rows"),
        "prepared_rows": evaluation_metadata.get("prepared_rows"),
        "split_train_rows": evaluation_metadata.get("train_rows"),
        "split_test_rows": evaluation_metadata.get("test_rows"),
        "full_training_rows": (training_summary.get("full_training_rows") or training_summary.get("training_rows")),
        "training_rows_used": (training_summary.get("sampled_training_rows") or training_summary.get("training_rows")),
        "training_label_0": label_count(training_distribution, 0),
        "training_label_1": label_count(training_distribution, 1),
        "evaluation_rows": metrics.get("evaluation_rows"),
        "evaluation_label_0": label_count(test_distribution, 0),
        "evaluation_label_1": label_count(test_distribution, 1),
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
        "epochs": training_summary.get("epochs"),
        "batch_size": training_summary.get("batch_size"),
        "learning_rate": training_summary.get("learning_rate"),
        "threshold": training_summary.get("threshold"),
        "model_configuration": (training_summary.get("architecture") or training_summary.get("params") or {}),
        "training_run": training_run.name,
        "evaluation_run": run_dir.name,
    }


def csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    return value


def print_summary(rows: list[dict]) -> None:
    print()
    print(
        f"{'Model':<20}"
        f"{'Trained on':<30}"
        f"{'Evaluated on':<30}"
        f"{'Accuracy':>10}"
        f"{'BalAcc':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )
    print("-" * 120)

    for row in rows:
        print(
            f"{row['model']:<20}"
            f"{row['trained_on_split']:<30}"
            f"{row['evaluated_on_split']:<30}"
            f"{row['accuracy']:>10.4f}"
            f"{row['balanced_accuracy']:>10.4f}"
            f"{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}"
        )


def run_summarize_results(args) -> None:
    model_registry = load_model_registry()
    rows = []

    for model_id, model_cfg in model_registry.items():
        if not model_cfg.get("ready", False):
            continue

        if not model_cfg.get("enabled", True):
            continue

        run_dir = latest_completed_run(model_id=model_id, split_id=args.split_id)

        if run_dir is None:
            split_message = (f" on split {args.split_id}" if args.split_id else "")
            print(f"[summary] No completed evaluation found for {model_id}{split_message}")
            continue

        rows.append(build_row(model_id, model_cfg, run_dir))

    if not rows:
        print("[summary] No completed evaluations found.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()

        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})

    print_summary(rows)

    print()
    print(f"[summary] CSV:  {OUTPUT_CSV}")
    print(f"[summary] JSON: {OUTPUT_JSON}")
