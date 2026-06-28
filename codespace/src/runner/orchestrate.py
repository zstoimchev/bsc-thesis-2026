import argparse
import importlib

from datetime import datetime
from pathlib import Path
from collections.abc import Callable

from src.common.metrics import save_json
from src.runner.paths import PROJECT_ROOT
from src.runner.registry import load_registries
from src.runner.commands.inspect_dataset import (
    add_inspect_dataset_parser,
    run_inspect_dataset,
)

def list_registry_items(title: str, registry: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))

    for item_id, cfg in registry.items():
        name = cfg.get("name", "")
        ready = cfg.get("ready", False)
        enabled = cfg.get("enabled", True)
        description = cfg.get("description", "")

        print(f"{item_id}: {name}")
        print(f"  ready={ready}, enabled={enabled}")

        if description:
            print(f"  {description}")

    print()


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
        raise ValueError(
            f"No {item_name}s selected. Check ready/enabled flags."
        )

    return selected


def resolve_models_to_run(
    model_registry,
    selected_model,
    selected_tags,
    include_not_ready,
    include_disabled,
):
    return resolve_registry_items(
        registry=model_registry,
        selected_item=selected_model,
        item_name="model",
        include_not_ready=include_not_ready,
        include_disabled=include_disabled,
        extra_filter=(
            None
            if not selected_tags
            else lambda cfg: bool(set(cfg.get("tags", [])) & set(selected_tags))
        ),
    )


def resolve_datasets_to_run(
    dataset_registry,
    selected_dataset,
    include_not_ready,
    include_disabled,
):
    return resolve_registry_items(
        registry=dataset_registry,
        selected_item=selected_dataset,
        item_name="dataset",
        include_not_ready=include_not_ready,
        include_disabled=include_disabled,
    )


def prepare_output_dir(
    dataset_id: str,
    model_id: str,
    split: str,
    feature_set: str,
    seed: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "runs"
        / f"{timestamp}_{dataset_id}_{model_id}_{split}_{feature_set}_seed{seed}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_one_experiment(
    args,
    mode: str,
    dataset_id: str,
    model_id: str,
    dataset_cfg: dict,
    model_cfg: dict,
) -> None:
    print("\n[orchestrate] Starting experiment")
    print(f"[orchestrate] Mode: {mode}")
    print(f"[orchestrate] Dataset: {dataset_id}")
    print(f"[orchestrate] Model: {model_id}")
    print(f"[orchestrate] Split: {args.split}")
    print(f"[orchestrate] Feature set: {args.feature_set}")
    print(f"[orchestrate] Chunk size: {args.chunk_size}")

    if args.dry_run:
        print("[orchestrate] Dry run only. Nothing will be trained or evaluated.")
        return

    output_dir = prepare_output_dir(
        dataset_id=dataset_id,
        model_id=model_id,
        split=args.split,
        feature_set=args.feature_set,
        seed=args.seed,
    )

    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"
    run_info_path = output_dir / "run_info.json"

    run_info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "dataset": dataset_id,
        "dataset_name": dataset_cfg.get("name", ""),
        "model": model_id,
        "model_name": model_cfg.get("name", ""),
        "split": args.split,
        "feature_set": args.feature_set,
        "test_size": args.test_size,
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "pipeline": "chunked",
    }

    save_json(run_info, run_info_path)

    train_module, evaluate_module = get_model_modules(model_cfg)

    if mode in {"train", "train-evaluate"}:
        train_module.train(
            dataset_cfg=dataset_cfg,
            model_cfg=model_cfg,
            output_dir=output_dir,
            model_path=model_path,
            project_root=PROJECT_ROOT,
            feature_set=args.feature_set,
            chunk_size=args.chunk_size,
            split=args.split,
            test_size=args.test_size,
            seed=args.seed,
        )

    if mode in {"evaluate", "train-evaluate"}:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Cannot evaluate because model file does not exist: {model_path}"
            )

        metrics = evaluate_module.evaluate(
            dataset_cfg=dataset_cfg,
            model_cfg=model_cfg,
            output_dir=output_dir,
            model_path=model_path,
            project_root=PROJECT_ROOT,
            feature_set=args.feature_set,
            chunk_size=args.chunk_size,
            split=args.split,
            test_size=args.test_size,
            seed=args.seed,
        )

        save_json(metrics, metrics_path)
        print(f"[orchestrate] Saved metrics to: {metrics_path}")

    print("[orchestrate] Done")


def run_experiments(args, mode: str) -> None:
    dataset_registry, model_registry = load_registries()

    dataset_ids = resolve_datasets_to_run(
        dataset_registry=dataset_registry,
        selected_dataset=args.dataset,
        include_not_ready=args.include_not_ready,
        include_disabled=args.include_disabled,
    )

    model_ids = resolve_models_to_run(
        model_registry=model_registry,
        selected_model=args.model,
        selected_tags=args.tags,
        include_not_ready=args.include_not_ready,
        include_disabled=args.include_disabled,
    )

    print(f"[orchestrate] Selected datasets: {dataset_ids}")
    print(f"[orchestrate] Selected models: {model_ids}")

    for dataset_id in dataset_ids:
        dataset_cfg = dataset_registry[dataset_id]

        for model_id in model_ids:
            model_cfg = model_registry[model_id]

            run_one_experiment(
                args=args,
                mode=mode,
                dataset_id=dataset_id,
                model_id=model_id,
                dataset_cfg=dataset_cfg,
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
        "--dataset",
        required=False,
        default=None,
        help="Dataset ID. If omitted, all ready/enabled datasets are used.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Rows processed per chunk for chunk-compatible models.",
    )

    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Run models with at least one of the selected tags.",
    )

    parser.add_argument(
        "--split",
        default="random",
        choices=["random"],
    )

    parser.add_argument(
        "--feature-set",
        default="all",
        choices=["all", "common"],
    )

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible IDS experiments."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models")
    subparsers.add_parser("list-datasets")

    add_inspect_dataset_parser(subparsers)

    train_parser = subparsers.add_parser("train")
    add_experiment_args(train_parser)

    evaluate_parser = subparsers.add_parser("evaluate")
    add_experiment_args(evaluate_parser)

    train_eval_parser = subparsers.add_parser("train-evaluate")
    add_experiment_args(train_eval_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_registry, model_registry = load_registries()

    if args.command == "list-models":
        list_registry_items("Available models", model_registry)
        return

    if args.command == "list-datasets":
        list_registry_items("Available datasets", dataset_registry)
        return

    if args.command == "inspect-dataset":
        run_inspect_dataset(args)
        return
    
    if args.command in {"train", "evaluate", "train-evaluate"}:
        run_experiments(args, mode=args.command)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()