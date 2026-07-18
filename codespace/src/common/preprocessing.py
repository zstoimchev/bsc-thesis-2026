import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_column_name(column: str) -> str:
    column = column.strip().lower()
    column = column.replace("/", "_")
    column = column.replace(" ", "_")
    column = re.sub(r"[^a-zA-Z0-9_]", "", column)
    column = re.sub(r"_+", "_", column)

    aliases = {
        "classlabel": "class",
        "labelclass": "class",
        "class_label": "class",
        "label_class": "class",

        "total_backward_packets": "total_bwd_packets",
        "fwd_packets_length_total": "total_fwd_bytes",
        "bwd_packets_length_total": "total_bwd_bytes",
        "total_length_of_fwd_packets": "total_fwd_bytes",
        "total_length_of_bwd_packets": "total_bwd_bytes",
    }

    return aliases.get(column, column)


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(col) for col in df.columns]
    return df


def split_xy(
        df: pd.DataFrame,
        label_column: str,
        feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    df = clean_dataframe_columns(df)

    label_column = clean_column_name(label_column)
    feature_columns = [clean_column_name(col) for col in feature_columns]

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Prepared split is missing feature columns: {missing}")

    if label_column not in df.columns:
        raise ValueError(f"Prepared split is missing label column: {label_column}")

    x = df[feature_columns].copy()
    y = df[label_column].astype(int)

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(0)

    return x, y


def cap_training_dataframe(
        df: pd.DataFrame,
        label_column: str,
        cap: int | None,
        seed: int,
) -> pd.DataFrame:
    if cap is None or cap >= len(df):
        return df.reset_index(drop=True)

    if cap <= 0:
        raise ValueError("--cap must be greater than 0.")

    stratify = None

    if df[label_column].nunique() > 1:
        stratify = df[label_column]

    capped_df, _ = train_test_split(
        df,
        train_size=cap,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    return capped_df.reset_index(drop=True)
