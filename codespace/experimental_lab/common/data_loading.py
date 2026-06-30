from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import resolve_path


class DataLoadingError(Exception):
    """Raised when a dataset cannot be loaded or interpreted."""


@dataclass
class LoadedDataset:
    """
    Container returned by load_dataset_from_record().

    For predefined train/test datasets:
        train_df is not None
        test_df is not None
        full_df is None

    For single-file datasets:
        full_df is not None
        train_df is None
        test_df is None
    """

    dataset_id: str
    label_col: str
    split_type: str
    full_df: pd.DataFrame | None = None
    train_df: pd.DataFrame | None = None
    test_df: pd.DataFrame | None = None


def read_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """
    Read CSV or Parquet dataset.

    Supported:
    - .csv
    - .txt    treated as CSV
    - .parquet
    - .pq
    """
    p = resolve_path(path)

    if not p.exists():
        raise DataLoadingError(f"Dataset file does not exist: {p}")

    if not p.is_file():
        raise DataLoadingError(f"Dataset path is not a file: {p}")

    suffix = p.suffix.lower()

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(p, nrows=nrows, low_memory=False)

    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
        if nrows is not None:
            df = df.head(nrows)
        return df

    raise DataLoadingError(f"Unsupported dataset file format: {p}")


def clean_column_name(name: str) -> str:
    """
    Normalize one column name.

    This is useful for comparing columns from different CIC-style datasets.
    It does not mutate a dataframe unless clean_columns() is called.
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of dataframe with normalized column names.
    """
    out = df.copy()
    out.columns = [clean_column_name(c) for c in out.columns]
    return out


def find_label_column(df: pd.DataFrame, requested_label_col: str | None) -> str:
    """
    Find the label column.

    First checks exact match. If that fails, checks normalized column names.
    This helps with Label / label / class / result differences.
    """
    if not requested_label_col:
        candidates = [
            "label",
            "Label",
            "class",
            "Class",
            "result",
            "Result",
            "attack_label",
            "Attack_label",
            "attack_cat",
            "Attack",
        ]

        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        normalized = {clean_column_name(c): c for c in df.columns}
        for candidate in candidates:
            key = clean_column_name(candidate)
            if key in normalized:
                return normalized[key]

        raise DataLoadingError(
            "No label column specified and no common label column found. "
            f"Available columns: {list(df.columns)[:30]}"
        )

    if requested_label_col in df.columns:
        return requested_label_col

    requested_clean = clean_column_name(requested_label_col)
    normalized = {clean_column_name(c): c for c in df.columns}

    if requested_clean in normalized:
        return normalized[requested_clean]

    raise DataLoadingError(
        f"Requested label column '{requested_label_col}' not found. "
        f"Available columns: {list(df.columns)[:30]}"
    )


def load_dataset_from_record(
    dataset_record: dict[str, Any],
    nrows: int | None = None,
) -> LoadedDataset:
    """
    Load dataset using a registry record.

    Supported registry styles:

    Single-file dataset:
        id: cic_superset
        path: data/raw/cic_superset.parquet
        label_col: label
        split_type: generated

    Predefined train/test dataset:
        id: cic_ddos2019_train_test
        train_path: ...
        test_path: ...
        label_col: Label
        split_type: predefined
    """
    dataset_id = str(dataset_record["id"])
    split_type = str(dataset_record.get("split_type", "generated"))
    requested_label_col = dataset_record.get("label_col")

    if "path" in dataset_record:
        df = read_table(dataset_record["path"], nrows=nrows)
        label_col = find_label_column(df, requested_label_col)

        return LoadedDataset(
            dataset_id=dataset_id,
            label_col=label_col,
            split_type=split_type,
            full_df=df,
        )

    if "train_path" in dataset_record and "test_path" in dataset_record:
        train_df = read_table(dataset_record["train_path"], nrows=nrows)
        test_df = read_table(dataset_record["test_path"], nrows=nrows)

        train_label_col = find_label_column(train_df, requested_label_col)
        test_label_col = find_label_column(test_df, requested_label_col)

        if clean_column_name(train_label_col) != clean_column_name(test_label_col):
            raise DataLoadingError(
                "Train and test label columns do not match: "
                f"{train_label_col} vs {test_label_col}"
            )

        return LoadedDataset(
            dataset_id=dataset_id,
            label_col=train_label_col,
            split_type=split_type,
            train_df=train_df,
            test_df=test_df,
        )

    raise DataLoadingError(
        f"Dataset record '{dataset_id}' must contain either 'path' or "
        "'train_path' + 'test_path'."
    )


def get_drop_columns(dataset_record: dict[str, Any]) -> list[str]:
    """
    Return columns that should be removed from X before training/evaluation.

    Registry can define:
        drop_columns:
          - src_ip
          - dst_ip
          - flow_id
    """
    raw = dataset_record.get("drop_columns", [])

    if raw is None:
        return []

    if not isinstance(raw, list):
        raise DataLoadingError(
            f"drop_columns for dataset '{dataset_record.get('id')}' must be a list."
        )

    return [str(c) for c in raw]


def split_features_label(
    df: pd.DataFrame,
    label_col: str,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into X and y.

    Removes:
    - label column
    - any columns listed in drop_columns
    """
    if label_col not in df.columns:
        raise DataLoadingError(f"Label column '{label_col}' not found in dataframe.")

    drop_columns = drop_columns or []

    y = df[label_col].copy()

    columns_to_drop = {label_col}
    existing_columns = set(df.columns)

    for col in drop_columns:
        if col in existing_columns:
            columns_to_drop.add(col)

    X = df.drop(columns=list(columns_to_drop), errors="ignore").copy()

    return X, y


def describe_dataframe(df: pd.DataFrame, label_col: str | None = None) -> dict[str, Any]:
    """
    Small dataset summary for audit output.
    """
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names_preview": list(df.columns[:20]),
    }

    if label_col and label_col in df.columns:
        counts = df[label_col].value_counts(dropna=False).head(30)
        summary["label_col"] = label_col
        summary["label_counts_top30"] = {
            str(k): int(v) for k, v in counts.to_dict().items()
        }

    return summary