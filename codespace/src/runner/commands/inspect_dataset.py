from collections import Counter

import numpy as np
import pandas as pd

from src.runner.constants import PROJECT_ROOT
from src.common.registry import (
    load_dataset_registry,
    load_feature_registry,
)
from src.common.data_loader import iterate_dataset
from src.common.label_mapping import normalize_binary_labels
from src.common.preprocessing import (
    clean_dataframe_columns,
    clean_column_name,
)

SPLIT_COLUMNS = [
    "class",
    "dataset_source",
    "capture_file",
    "label_source",
]

TIME_COLUMNS = [
    "timestamp",
    "flow_start_ts",
]


def add_inspect_dataset_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "inspect-dataset",
        help="Inspect one dataset after loading and column cleaning.",
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
    feature_registry = load_feature_registry()

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
    print(f"Chunk size: {args.chunk_size}")
    print("=" * 80)
    print()

    label_column = clean_column_name(dataset_cfg["label_column"])

    chunk_count = 0
    total_rows = 0

    columns_seen = []
    non_label_columns_seen = []

    raw_label_counts = Counter()
    binary_label_counts = Counter()

    numeric_columns = set()
    non_numeric_columns = set()

    missing_values_total = 0
    missing_values_by_column = Counter()
    infinite_values_total = 0

    available_split_columns = []
    available_time_columns = []

    # Key: (column, value, binary_label)
    grouped_label_counts = Counter()

    earliest_times = {}
    latest_times = {}

    # Key: (time_column, binary_label)
    missing_time_counts = Counter()

    for chunk in iterate_dataset(
            dataset_cfg=dataset_cfg,
            project_root=PROJECT_ROOT,
            chunk_size=args.chunk_size,
    ):
        chunk_count += 1
        chunk = clean_dataframe_columns(chunk)

        if label_column not in chunk.columns:
            raise ValueError(f"Label column not found after cleaning: {label_column}")

        if not columns_seen:
            columns_seen = list(chunk.columns)

            non_label_columns_seen = [
                column
                for column in columns_seen
                if column != label_column
            ]

            available_split_columns = [
                column
                for column in SPLIT_COLUMNS
                if column in chunk.columns and column != label_column
            ]

            available_time_columns = [
                column
                for column in TIME_COLUMNS
                if column in chunk.columns
            ]

        total_rows += len(chunk)

        raw_label_counts.update(
            chunk[label_column]
            .value_counts(dropna=False)
            .to_dict()
        )

        binary_labels = normalize_binary_labels(chunk[label_column])

        binary_label_counts.update(
            binary_labels
            .value_counts(dropna=False)
            .to_dict()
        )

        for column in available_split_columns:
            grouped = pd.DataFrame(
                {
                    "value": chunk[column]
                    .fillna("<missing>")
                    .astype(str),
                    "label": binary_labels,
                }
            ).value_counts()

            for (value, binary_label), count in grouped.items():
                grouped_label_counts[
                    (column, value, int(binary_label))
                ] += int(count)

        for column in available_time_columns:
            values = chunk[column]

            if pd.api.types.is_numeric_dtype(values):
                parsed_times = pd.to_datetime(
                    values,
                    unit="s",
                    errors="coerce",
                    utc=True,
                )
            else:
                parsed_times = pd.to_datetime(
                    values,
                    errors="coerce",
                    utc=True,
                )

            missing_mask = parsed_times.isna()

            missing_labels = (
                binary_labels[missing_mask]
                .value_counts()
                .to_dict()
            )

            for binary_label, count in missing_labels.items():
                missing_time_counts[
                    (column, int(binary_label))
                ] += int(count)

            valid_times = parsed_times.dropna()

            if not valid_times.empty:
                chunk_earliest = valid_times.min()
                chunk_latest = valid_times.max()

                if (
                        column not in earliest_times
                        or chunk_earliest < earliest_times[column]
                ):
                    earliest_times[column] = chunk_earliest

                if (
                        column not in latest_times
                        or chunk_latest > latest_times[column]
                ):
                    latest_times[column] = chunk_latest

        x = chunk.drop(columns=[label_column])

        current_numeric_columns = set(
            x.select_dtypes(include=["number"]).columns
        )

        numeric_columns.update(current_numeric_columns)
        non_numeric_columns.update(
            set(x.columns) - current_numeric_columns
        )

        chunk_missing = x.isna().sum()

        missing_values_total += int(chunk_missing.sum())

        missing_values_by_column.update(
            {
                column: int(count)
                for column, count in chunk_missing.items()
                if count > 0
            }
        )

        numeric_x = x.select_dtypes(include=["number"])

        if not numeric_x.empty:
            infinite_values_total += int(
                np.isinf(numeric_x.to_numpy()).sum()
            )

        if chunk_count == 1 or chunk_count % 7 == 0:
            print(
                f"Processed chunk {chunk_count}: "
                f"{len(chunk)} rows "
                f"(total: {total_rows})"
            )

    if chunk_count != 1 and chunk_count % 7 != 0:
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
        title="Raw label distribution:",
        counter=raw_label_counts,
    )

    print_counter(
        title="Binary label distribution after normalization:",
        counter=binary_label_counts,
    )

    print(f"Total columns: {len(columns_seen)}")
    print(f"Non-label columns: {len(non_label_columns_seen)}")
    print()

    print("Columns:")

    for column in columns_seen:
        marker = " [LABEL]" if column == label_column else ""
        print(f"  - {column}{marker}")

    print()

    print(f"Numeric non-label columns: {len(numeric_columns)}")
    print(f"Non-numeric non-label columns: {len(non_numeric_columns)}")
    print()

    if non_numeric_columns:
        print("Non-numeric non-label columns:")

        for column in sorted(non_numeric_columns):
            print(f"  - {column}")

        print()

    print(
        f"Missing values in non-label columns: "
        f"{missing_values_total}"
    )
    print(
        f"Infinite numeric values in non-label columns: "
        f"{infinite_values_total}"
    )
    print()

    if missing_values_by_column:
        print_counter(
            title="Missing values by column:",
            counter=missing_values_by_column,
        )

    for column in available_split_columns:
        print(f"Binary label distribution grouped by '{column}':")

        values = {
            value
            for counted_column, value, _ in grouped_label_counts
            if counted_column == column
        }

        values = sorted(
            values,
            key=lambda value: -(
                    grouped_label_counts[(column, value, 0)]
                    + grouped_label_counts[(column, value, 1)]
            ),
        )

        for value in values:
            benign = grouped_label_counts[(column, value, 0)]
            malicious = grouped_label_counts[(column, value, 1)]

            print(
                f"  {value}: "
                f"total={benign + malicious}, "
                f"benign={benign}, "
                f"malicious={malicious}"
            )

        print()

    for column in available_time_columns:
        print(f"Time column '{column}':")
        print(f"  Earliest: {earliest_times.get(column)}")
        print(f"  Latest:   {latest_times.get(column)}")
        print(
            "  Missing:  "
            f"benign={missing_time_counts[(column, 0)]}, "
            f"malicious={missing_time_counts[(column, 1)]}"
        )
        print()

    print("Feature-set compatibility:")

    dataset_columns = set(columns_seen)

    dropped_columns = {
        clean_column_name(column)
        for column in dataset_cfg.get("drop_columns", [])
    }

    for feature_id, feature_cfg in feature_registry.items():
        configured_features = feature_cfg.get("features")

        if isinstance(configured_features, list):
            cleaned_features = [
                clean_column_name(column)
                for column in configured_features
            ]

            missing_features = [
                column
                for column in cleaned_features
                if column not in dataset_columns
            ]

            if missing_features:
                print(
                    f"  {feature_id}: not compatible "
                    f"({len(missing_features)} missing)"
                )
                print(
                    f"    Missing: "
                    f"{', '.join(missing_features)}"
                )
            else:
                print(
                    f"  {feature_id}: compatible "
                    f"({len(cleaned_features)} features)"
                )

        elif configured_features == "all_numeric_except_dropped":
            usable_features = sorted(numeric_columns - dropped_columns)

            print(
                f"  {feature_id}: compatible "
                f"({len(usable_features)} usable numeric features)"
            )

        else:
            print(
                f"  {feature_id}: "
                f"unknown feature definition"
            )

    print()
    print("Inspection completed successfully.")
