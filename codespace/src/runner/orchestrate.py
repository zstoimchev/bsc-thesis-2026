import argparse

from src.runner.commands.generate_figures import add_generate_figures_parser, run_generate_figures
from src.runner.commands.inspect_dataset import add_inspect_dataset_parser, run_inspect_dataset
from src.runner.commands.inspect_model import add_inspect_model_parser, run_inspect_model
from src.runner.commands.prepare_split import add_prepare_split_parser, run_prepare_split
from src.runner.commands.list_registry import (
    add_list_registry_parsers,
    run_list_models,
    run_list_datasets,
    run_list_splits,
    run_list_features,
)
from src.runner.commands.run_experiment import add_experiment_parsers, run_experiments
from src.runner.commands.summarize_results import add_summarize_results_parser, run_summarize_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible IDS experiments."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_list_registry_parsers(subparsers)
    add_inspect_dataset_parser(subparsers)
    add_inspect_model_parser(subparsers)
    add_prepare_split_parser(subparsers)
    add_experiment_parsers(subparsers)
    add_summarize_results_parser(subparsers)
    add_generate_figures_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-models":
        run_list_models()
        return

    if args.command == "list-datasets":
        run_list_datasets()
        return

    if args.command == "list-splits":
        run_list_splits()
        return

    if args.command == "list-features":
        run_list_features()
        return

    if args.command == "inspect-dataset":
        run_inspect_dataset(args)
        return

    if args.command == "inspect-model":
        run_inspect_model(args)
        return

    if args.command == "prepare-split":
        run_prepare_split(args)
        return

    if args.command in {"train", "evaluate", "train-evaluate"}:
        run_experiments(args, mode=args.command)
        return

    if args.command == "summarize-results":
        run_summarize_results(args)
        return

    if args.command == "generate-figures":
        run_generate_figures(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
