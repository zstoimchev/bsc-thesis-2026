from datetime import datetime

import numpy as np
import pandas as pd

from src.common.data_loader import load_dataset
from src.common.label_mapping import normalize_binary_labels
from src.common.metrics import save_json
from src.common.preprocessing import clean_dataframe_columns, clean_column_name
from src.common.shared import load_split_context
from src.common.splitting import split_dataframe
from src.runner.constants import PROJECT_ROOT


def add_prepare_split_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "prepare-split",
        help="Prepare train/test files from a registered split.",
    )

    parser.add_argument(
        "--split-id",
        required=True,
        help="Split ID from registries/splits.json.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared split files.",
    )


def run_prepare_split(args) -> None:
    split_id = args.split_id

    split_cfg, dataset_cfg, feature_cfg = load_split_context(
        split_id=split_id,
        require_feature=True,
    )

    if feature_cfg is None:
        raise ValueError(f"Missing feature config for split: {split_id}")

    dataset_id = split_cfg["dataset_id"]
    feature_set_id = split_cfg["feature_set_id"]
    output_cfg = split_cfg["output"]

    output_format = output_cfg.get("format", "parquet").lower()
    if output_format != "parquet":
        raise ValueError(f"Only parquet output is supported for now. Got: {output_format}")

    output_base_dir = PROJECT_ROOT / output_cfg["output_dir"]
    prepared_dir = output_base_dir / split_id

    prepared_root = (PROJECT_ROOT / "data" / "prepared").resolve()
    prepared_dir = prepared_dir.resolve()

    if prepared_root not in prepared_dir.parents and prepared_dir != prepared_root:
        raise ValueError(
            f"Invalid output_dir for split '{split_id}': {prepared_dir}\n"
            f"Prepared split output must be under: {prepared_root}"
        )

    train_path = prepared_dir / output_cfg["train_file"]
    test_path = prepared_dir / output_cfg["test_file"]
    metadata_path = prepared_dir / output_cfg["metadata_file"]

    if prepared_dir.exists() and not args.overwrite:
        if train_path.exists() or test_path.exists() or metadata_path.exists():
            raise FileExistsError(
                f"Prepared split already exists: {prepared_dir}. "
                "Use --overwrite to regenerate it."
            )

    prepared_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare-split] split_id={split_id}")
    print(f"[prepare-split] dataset_id={dataset_id}")
    print(f"[prepare-split] feature_set_id={feature_set_id}")
    print(f"[prepare-split] output_dir={prepared_dir.relative_to(PROJECT_ROOT)}")

    df = load_dataset(dataset_cfg=dataset_cfg, project_root=PROJECT_ROOT)
    source_rows = int(len(df))

    df = clean_dataframe_columns(df)

    label_column = clean_column_name(split_cfg.get("label_column") or dataset_cfg["label_column"])

    if label_column not in df.columns:
        raise ValueError(f"Label column not found after cleaning: {label_column}")

    df[label_column] = normalize_binary_labels(df[label_column])

    final_df, final_features, dropped_columns, split_columns = select_final_columns(
        df=df,
        dataset_cfg=dataset_cfg,
        split_cfg=split_cfg,
        feature_cfg=feature_cfg,
        label_column=label_column,
    )

    train_df, test_df = split_dataframe(
        df=final_df,
        label_column=label_column,
        split_cfg=split_cfg,
    )

    # Split-only columns are needed to create the split, but must not be used for training and evaluation.
    if split_columns:
        train_df = train_df.drop(columns=split_columns, errors="ignore")
        test_df = test_df.drop(columns=split_columns, errors="ignore")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    metadata = {
        "split_id": split_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_id,
        "dataset_name": dataset_cfg.get("name", ""),
        "dataset_path": dataset_cfg.get("path"),
        "feature_set_id": feature_set_id,
        "output_base_dir": output_cfg["output_dir"],
        "output_dir": str(prepared_dir.relative_to(PROJECT_ROOT)),
        "train_file": str(train_path.relative_to(PROJECT_ROOT)),
        "test_file": str(test_path.relative_to(PROJECT_ROOT)),
        "label_column": label_column,
        "split_method": split_cfg["split_method"],
        "feature_columns": final_features,
        "split_columns": split_columns,
        "dropped_columns": dropped_columns,
        "source_rows": source_rows,
        "prepared_rows": int(len(final_df)),
        "dropped_rows_during_preparation": int(source_rows - len(final_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_label_distribution": {
            str(k): int(v)
            for k, v in train_df[label_column].value_counts().sort_index().items()
        },
        "test_label_distribution": {
            str(k): int(v)
            for k, v in test_df[label_column].value_counts().sort_index().items()
        },
    }

    save_json(metadata, metadata_path)

    print(f"[prepare-split] source rows: {source_rows}")
    print(f"[prepare-split] prepared rows: {len(final_df)}")
    print(f"[prepare-split] train rows: {len(train_df)}")
    print(f"[prepare-split] test rows: {len(test_df)}")
    print(f"[prepare-split] saved train: {train_path}")
    print(f"[prepare-split] saved test: {test_path}")
    print(f"[prepare-split] saved metadata: {metadata_path}")


def select_final_columns(
        df: pd.DataFrame,
        dataset_cfg: dict,
        split_cfg: dict,
        feature_cfg: dict,
        label_column: str,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    df = df.copy()

    dataset_drop_columns = [
        clean_column_name(c)
        for c in dataset_cfg.get("drop_columns", [])
    ]

    split_drop_columns = [
        clean_column_name(c)
        for c in split_cfg.get("drop_columns", [])
    ]

    if split_cfg.get("use_dataset_drop_columns", True):
        dropped_columns = list(dict.fromkeys(dataset_drop_columns + split_drop_columns))
    else:
        dropped_columns = list(dict.fromkeys(split_drop_columns))

    feature_definition = feature_cfg["features"]

    if isinstance(feature_definition, list):
        selected_features = [
            clean_column_name(c)
            for c in feature_definition
        ]

        missing = [c for c in selected_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing selected features: {missing}")

    elif feature_definition == "all_numeric_except_dropped":
        blocked = set(dropped_columns)
        blocked.add(label_column)

        selected_features = [
            c
            for c in df.columns
            if c not in blocked and pd.api.types.is_numeric_dtype(df[c])
        ]

    else:
        raise ValueError(f"Unsupported feature definition: {feature_definition}")

    # Some split methods require an additional column that is not a model
    # feature. For the DDoS holdout this is the "class" column.
    split_columns = []

    group_column = split_cfg["split_method"].get("group_column")

    if group_column:
        group_column = clean_column_name(group_column)

        if group_column not in df.columns:
            raise ValueError(f"Split group column not found: {group_column}")

        split_columns.append(group_column)

    final_columns = list(
        dict.fromkeys(
            selected_features
            + split_columns
            + [label_column]
        )
    )

    final_df = df[final_columns].copy()
    final_df = final_df.replace([np.inf, -np.inf], np.nan)

    for column in selected_features:
        final_df[column] = pd.to_numeric(final_df[column], errors="coerce")

    # Only model features and the label must be complete.
    # Missing unrelated metadata must not remove valid records.
    final_df = final_df.dropna(subset=selected_features + [label_column])
    final_df[label_column] = (final_df[label_column].astype(int))

    return final_df.reset_index(drop=True), selected_features, dropped_columns, split_columns
