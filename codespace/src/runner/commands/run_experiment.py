import argparse
import importlib
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from src.common.metrics import save_json
from src.common.shared import load_split_context
from src.runner.constants import PROJECT_ROOT
from src.common.registry import load_model_registry


def get_model_modules(model_cfg: dict):
    module_base = model_cfg["module"]

    train_module = importlib.import_module(f"{module_base}.train")
    evaluate_module = importlib.import_module(f"{module_base}.evaluate")

    return train_module, evaluate_module


def resolve_registry_items(
        registry: dict,
        selected_item: str | None,
        item_name: str,
        include_not_ready: bool,
        include_disabled: bool,
        extra_filter: Callable[[dict], bool] | None = None,
) -> list[str]:
    if selected_item:
        if selected_item not in registry:
            raise ValueError(f"Unknown {item_name}: {selected_item}")

        cfg = registry[selected_item]

        if not include_not_ready and not cfg.get("ready", False):
            raise ValueError(
                f"{item_name.capitalize()} '{selected_item}' is not ready yet. "
                "Use --include-not-ready if you really want to run it."
            )

        if not include_disabled and not cfg.get("enabled", True):
            raise ValueError(
                f"{item_name.capitalize()} '{selected_item}' is disabled. "
                "Use --include-disabled if you really want to run it."
            )

        return [selected_item]

    selected = []

    for item_id, cfg in registry.items():
        if not include_not_ready and not cfg.get("ready", False):
            continue

        if not include_disabled and not cfg.get("enabled", True):
            continue

        if extra_filter and not extra_filter(cfg):
            continue

        selected.append(item_id)

    if not selected:
        raise ValueError(f"No {item_name}s selected. Check ready/enabled flags.")

    return selected


def resolve_models_to_run(
        model_registry: dict,
        selected_model: str | None,
        selected_tags: list[str] | None,
        include_not_ready: bool,
        include_disabled: bool,
) -> list[str]:
    return resolve_registry_items(
        registry=model_registry,
        selected_item=selected_model,
        item_name="model",
        include_not_ready=include_not_ready,
        include_disabled=include_disabled,
        extra_filter=(
            None
            if not selected_tags
            else lambda cfg: bool(set(cfg.get("tags", [])) & set(selected_tags or []))
        ),
    )


def load_prepared_metadata(split_id: str, split_cfg: dict) -> dict:
    metadata_path = (
            PROJECT_ROOT
            / split_cfg["output"]["output_dir"]
            / split_id
            / split_cfg["output"]["metadata_file"]
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Prepared split metadata not found: {metadata_path}. "
            "Run prepare-split first."
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_output_dir(
        split_id: str,
        model_id: str,
        seed: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = (
            PROJECT_ROOT
            / "results"
            / "runs"
            / f"{timestamp}_{split_id}_{model_id}_seed{seed}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_latest_trained_model_path(
        model_id: str,
        seed: int,
) -> Path:
    runs_root = PROJECT_ROOT / "results" / "runs"

    if not runs_root.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_root}")

    candidates = []

    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue

        model_path = run_dir / "model.joblib"
        run_info_path = run_dir / "run_info.json"

        if not model_path.exists():
            continue

        if not run_info_path.exists():
            continue

        with run_info_path.open("r", encoding="utf-8") as f:
            run_info = json.load(f)

        if run_info.get("model_id") != model_id:
            continue

        if run_info.get("seed") != seed:
            continue

        candidates.append(
            {
                "model_path": model_path,
                "run_dir": run_dir,
                "modified_time": model_path.stat().st_mtime,
            }
        )

    if not candidates:
        raise FileNotFoundError(
            f"No trained model found for model={model_id}, seed={seed}. "
            "Run train or train-evaluate first."
        )

    latest = max(candidates, key=lambda item: item["modified_time"])

    return latest["model_path"]


def resolve_model_path_for_mode(
        mode: str,
        output_dir: Path,
        model_id: str,
        seed: int,
) -> Path:
    if mode in {"train", "train-evaluate"}:
        return output_dir / "model.joblib"

    if mode == "evaluate":
        return find_latest_trained_model_path(model_id=model_id, seed=seed)

    raise ValueError(f"Unknown experiment mode: {mode}")


def run_one_experiment(
        args,
        mode: str,
        split_id: str,
        split_cfg: dict,
        split_metadata: dict,
        dataset_cfg: dict,
        model_id: str,
        model_cfg: dict,
) -> None:
    print("\n[orchestrate] Starting experiment")
    print(f"[orchestrate] Mode: {mode}")
    print(f"[orchestrate] Split: {split_id}")
    print(f"[orchestrate] Dataset: {split_cfg['dataset_id']}")
    print(f"[orchestrate] Feature set: {split_cfg['feature_set_id']}")
    print(f"[orchestrate] Model: {model_id}")
    print(f"[orchestrate] Seed: {args.seed}")

    if args.dry_run:
        print("[orchestrate] Dry run only. Nothing will be trained or evaluated.")
        return

    output_dir = prepare_output_dir(split_id=split_id, model_id=model_id, seed=args.seed)
    # model_path = output_dir / "model.joblib"
    model_path = resolve_model_path_for_mode(
        mode=mode,
        output_dir=output_dir,
        model_id=model_id,
        seed=args.seed,
    )
    metrics_path = output_dir / "metrics.json"
    run_info_path = output_dir / "run_info.json"

    run_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "pipeline": "prepared_split",
        "split_id": split_id,
        "dataset_id": split_cfg["dataset_id"],
        "dataset_name": dataset_cfg.get("name", ""),
        "feature_set_id": split_cfg["feature_set_id"],
        "model_id": model_id,
        "model_name": model_cfg.get("name", ""),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "train_row_cap": args.cap,
        "train_file": split_metadata["train_file"],
        "test_file": split_metadata["test_file"],
        "label_column": split_metadata["label_column"],
        "feature_columns": split_metadata["feature_columns"],
    }

    save_json(run_info, run_info_path)

    train_module, evaluate_module = get_model_modules(model_cfg)

    if mode in {"train", "train-evaluate"}:
        train_module.train(
            output_dir=output_dir,
            model_path=model_path,
            project_root=PROJECT_ROOT,
            seed=args.seed,
            split_id=split_id,
            split_metadata=split_metadata,
            cap=args.cap,
        )

    if mode in {"evaluate", "train-evaluate"}:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Cannot evaluate because model file does not exist: {model_path}"
            )

        metrics = evaluate_module.evaluate(
            model_path=model_path,
            project_root=PROJECT_ROOT,
            seed=args.seed,
            split_id=split_id,
            split_cfg=split_cfg,
            split_metadata=split_metadata,
        )

        save_json(metrics, metrics_path)
        print("\n[orchestrate] Evaluation summary")
        print("-" * 40)
        print(f"Accuracy:          {metrics.get('accuracy', 0):.4f}")
        print(f"Balanced accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        print(f"Precision:         {metrics.get('precision', 0):.4f}")
        print(f"Recall:            {metrics.get('recall', 0):.4f}")
        print(f"F1:                {metrics.get('f1', 0):.4f}")
        print(f"F1 macro:          {metrics.get('f1_macro', 0):.4f}")
        cm = metrics.get("confusion_matrix")
        if cm:
            print("\nConfusion matrix:")
            print("               predicted_0  predicted_1")
            print(f"actual_0       {cm[0][0]:>11}  {cm[0][1]:>11}")
            print(f"actual_1       {cm[1][0]:>11}  {cm[1][1]:>11}")
        print("-" * 40)
        print(f"[orchestrate] Saved metrics to: {metrics_path}")

    print("[orchestrate] Done")


def run_experiments(args, mode: str) -> None:
    if args.cap is not None and args.cap <= 0:
        raise ValueError("--cap must be greater than 0.")

    model_registry = load_model_registry()

    split_id = args.split_id
    split_cfg, dataset_cfg, _ = load_split_context(split_id=split_id, require_feature=False)

    split_type = split_cfg["split_method"]["type"]
    if split_type == "external_full" and mode != "evaluate":
        raise ValueError("An external_full split can only be used with the evaluate command.")

    split_metadata = load_prepared_metadata(split_id, split_cfg)

    model_ids = resolve_models_to_run(
        model_registry=model_registry,
        selected_model=args.model,
        selected_tags=args.tags,
        include_not_ready=args.include_not_ready,
        include_disabled=args.include_disabled,
    )

    print(f"[orchestrate] Selected split: {split_id}")
    print(f"[orchestrate] Selected models: {model_ids}")

    for model_id in model_ids:
        model_cfg = model_registry[model_id]

        run_one_experiment(
            args=args,
            mode=mode,
            split_id=split_id,
            split_cfg=split_cfg,
            split_metadata=split_metadata,
            dataset_cfg=dataset_cfg,
            model_id=model_id,
            model_cfg=model_cfg,
        )


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="Model ID. If omitted, all ready/enabled models are used.",
    )

    parser.add_argument(
        "--split-id",
        required=True,
        help="Prepared split ID from splits.json.",
    )

    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Run models with at least one of the selected tags.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected experiments without running them.",
    )

    parser.add_argument(
        "--include-not-ready",
        action="store_true",
        help="Allow running models/datasets marked as ready=false.",
    )

    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Allow running models/datasets marked as enabled=false.",
    )

    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help=(
            "Maximum number of training rows. "
            "Omit this argument to use the complete training split."
        ),
    )


def add_experiment_parsers(subparsers) -> None:
    train_parser = subparsers.add_parser("train", help="Train selected model(s).")
    add_experiment_args(train_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate selected model(s).")
    add_experiment_args(evaluate_parser)

    train_eval_parser = subparsers.add_parser(
        "train-evaluate",
        help="Train and evaluate selected model(s).",
    )
    add_experiment_args(train_eval_parser)
