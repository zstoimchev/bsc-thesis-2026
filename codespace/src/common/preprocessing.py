import re

import numpy as np
import pandas as pd

from src.common.feature_sets import resolve_feature_columns
from src.common.label_mapping import normalize_binary_labels


def clean_column_name(column: str) -> str:
    column = column.strip().lower()
    column = column.replace("/", "_")
    column = column.replace(" ", "_")
    column = re.sub(r"[^a-zA-Z0-9_]", "", column)
    column = re.sub(r"_+", "_", column)
    return column


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(col) for col in df.columns]
    return df


def prepare_xy(
    df: pd.DataFrame,
    dataset_cfg: dict,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = clean_dataframe_columns(df)

    label_column = clean_column_name(dataset_cfg["label_column"])

    if label_column not in df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    strict_features = dataset_cfg.get("strict_features", True)

    feature_columns = resolve_feature_columns(
        available_columns=list(df.columns),
        feature_set_id=feature_set,
        label_column=label_column,
        strict=strict_features,
    )

    if not feature_columns:
        raise ValueError("No usable feature columns found.")

    x = df[feature_columns].copy()
    y = normalize_binary_labels(df[label_column])

    x = x.replace([np.inf, -np.inf], np.nan)

    for column in x.columns:
        if pd.api.types.is_numeric_dtype(x[column]):
            x[column] = x[column].fillna(0)
        else:
            x[column] = x[column].fillna("unknown")

    return x, y, feature_columns