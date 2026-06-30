import numpy as np
import pandas as pd

from collections import Counter

from src.runner.paths import PROJECT_ROOT
from src.runner.registry import load_dataset_registry
from src.common.preprocessing import (
    clean_column_name,
    iter_prepared_xy_chunks,
)


def print_counts(title: str, values: pd.Series) -> None:
    print(title)
    counts = values.value_counts(dropna=False)

    for value, count in counts.items():
        print(f"  {value}: {count}")

    print()


def add_inspect_dataset_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "inspect-dataset",
        help="Inspect one dataset after loading and preprocessing.",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset ID from datasets.json.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Number of rows to process per chunk.",
    )

    parser.add_argument(
        "--feature-set",
        default="all",
        choices=["all", "common"],
        help="Feature set to inspect.",
    )

    parser.add_argument(
        "--include-not-ready",
        action="store_true",
        help="Allow inspecting datasets marked ready=false.",
    )

    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Allow inspecting datasets marked enabled=false.",
    )


def print_counter(title: str, counter: Counter) -> None:
    print(title)

    for value, count in counter.most_common():
        print(f"  {value}: {count}")

    print()


def run_inspect_dataset(args) -> None:
    dataset_registry = load_dataset_registry()

    if args.dataset not in dataset_registry:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataset_cfg = dataset_registry[args.dataset]

    if not args.include_not_ready and not dataset_cfg.get("ready", False):
        raise ValueError(
            f"Dataset '{args.dataset}' is marked ready=false. "
            "Use --include-not-ready if you want to inspect it anyway."
        )

    if not args.include_disabled and not dataset_cfg.get("enabled", True):
        raise ValueError(
            f"Dataset '{args.dataset}' is marked enabled=false. "
            "Use --include-disabled if you want to inspect it anyway."
        )

    print("=" * 80)
    print(f"Dataset ID: {args.dataset}")
    print(f"Name: {dataset_cfg.get('name', '')}")
    print(f"Path: {dataset_cfg.get('path')}")
    print(f"Format: {dataset_cfg.get('format')}")
    print(f"Feature set: {args.feature_set}")
    print(f"Chunk size: {args.chunk_size}")
    print("=" * 80)
    print()

    label_column = clean_column_name(dataset_cfg["label_column"])

    chunk_count = 0
    total_rows = 0
    feature_columns = None

    binary_label_counts = Counter()
    numeric_features = set()
    non_numeric_features = set()

    missing_values_total = 0
    infinite_values_total = 0

    for x, y, current_feature_columns in iter_prepared_xy_chunks(
            dataset_cfg=dataset_cfg,
            project_root=PROJECT_ROOT,
            feature_set=args.feature_set,
            chunk_size=args.chunk_size,
    ):
        chunk_count += 1
        total_rows += len(x)

        if feature_columns is None:
            feature_columns = current_feature_columns

        binary_label_counts.update(y.value_counts(dropna=False).to_dict())

        numeric_columns = set(x.select_dtypes(include=["number"]).columns)
        non_numeric_columns = set(x.columns) - numeric_columns

        numeric_features.update(numeric_columns)
        non_numeric_features.update(non_numeric_columns)

        missing_values_total += int(x.isna().sum().sum())

        numeric_x = x.select_dtypes(include=["number"])
        if not numeric_x.empty:
            infinite_values_total += int(np.isinf(numeric_x.to_numpy()).sum())

        if chunk_count == 1 or chunk_count % 7 == 0:
            print(
                f"Processed chunk {chunk_count}: "
                f"{len(x)} rows "
                f"(total: {total_rows})"
            )

    print(
        f"Processed chunk {chunk_count}: "
        f"{total_rows} rows "
        f"(total: {total_rows})"
    )

    print()
    print("=" * 80)
    print("Inspection summary")
    print("=" * 80)
    print()

    print(f"Label column: {label_column}")
    print(f"Total chunks: {chunk_count}")
    print(f"Total rows: {total_rows}")
    print()

    print_counter(
        title="Binary label distribution after normalization:",
        counter=binary_label_counts,
    )

    print(f"Selected features: {len(feature_columns or [])}")
    for feature in feature_columns or []:
        print(f"  - {feature}")
    print()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Non-numeric features: {len(non_numeric_features)}")

    if non_numeric_features:
        print("Non-numeric feature columns:")
        for column in sorted(non_numeric_features):
            print(f"  - {column}")

    print()

    print(f"Missing values after preprocessing: {missing_values_total}")
    print(f"Infinite numeric values after preprocessing: {infinite_values_total}")
    print()

    print("Inspection completed successfully.")
