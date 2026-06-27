import numpy as np
import pandas as pd

from src.common.data_loader import load_dataset
from src.common.preprocessing import clean_column_name, clean_dataframe_columns, prepare_xy
from src.runner.paths import PROJECT_ROOT
from src.runner.registry import load_dataset_registry


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
        help="Dataset ID from dataset_registry.json.",
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
    print("=" * 80)
    print()

    df = load_dataset(dataset_cfg, project_root=PROJECT_ROOT)

    print(f"Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print()

    cleaned_df = clean_dataframe_columns(df)
    label_column = clean_column_name(dataset_cfg["label_column"])

    if label_column not in cleaned_df.columns:
        raise ValueError(
            f"Label column not found after cleaning: {label_column}. "
            f"Available columns: {list(cleaned_df.columns)}"
        )

    print_counts(
        title=f"Raw label distribution ({label_column}):",
        values=cleaned_df[label_column],
    )

    x, y, feature_columns = prepare_xy(
        df=df,
        dataset_cfg=dataset_cfg,
        feature_set=args.feature_set,
    )

    print(f"Prepared X shape: {x.shape[0]} rows x {x.shape[1]} columns")
    print(f"Prepared y shape: {y.shape[0]} rows")
    print()

    print("Selected features:")
    for feature in feature_columns:
        print(f"  - {feature}")
    print()

    print_counts(
        title="Binary label distribution after normalization:",
        values=y,
    )

    numeric_columns = list(x.select_dtypes(include=["number"]).columns)
    non_numeric_columns = [c for c in x.columns if c not in numeric_columns]

    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Non-numeric features: {len(non_numeric_columns)}")

    if non_numeric_columns:
        print("Non-numeric feature columns:")
        for column in non_numeric_columns:
            print(f"  - {column}")

    print()

    missing_values = int(x.isna().sum().sum())
    print(f"Missing values after preprocessing: {missing_values}")

    numeric_x = x.select_dtypes(include=["number"])
    if numeric_x.empty:
        infinite_values = 0
    else:
        infinite_values = int(np.isinf(numeric_x.to_numpy()).sum())

    print(f"Infinite numeric values after preprocessing: {infinite_values}")
    print()

    print("Inspection completed successfully.")