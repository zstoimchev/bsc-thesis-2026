import re
import gc

import numpy as np
import pandas as pd

from collections.abc import Iterator
from pathlib import Path
from typing import cast

from src.common.feature_sets import resolve_feature_columns
from src.common.label_mapping import normalize_binary_labels
from src.common.data_loader import iterate_dataset


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


def prepare_xy(
        df: pd.DataFrame,
        dataset_cfg: dict,
        feature_set: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = clean_dataframe_columns(df)

    label_column = clean_column_name(dataset_cfg["label_column"])

    if label_column not in df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    drop_columns = {
        clean_column_name(col)
        for col in dataset_cfg.get("drop_columns", [])
    }

    target_like_columns = {
        "label",
        "class",
        "target",
        "attack",
        "attack_label",
        "result",
        "label_source",
    }

    for column in target_like_columns:
        if column != label_column:
            drop_columns.add(column)

    feature_columns = resolve_feature_columns(
        available_columns=list(df.columns),
        feature_set_id=feature_set,
        label_column=label_column,
        drop_columns=list(drop_columns)
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


def _make_split_mask(
        start_index: int,
        size: int,
        test_size: float,
        seed: int,
) -> bool:
    indices = np.arange(start_index, start_index + size, dtype=np.int64)

    hashed = (indices * 1103515245 + seed * 12345 + 12345) & 0x7FFFFFFF
    values = hashed / float(0x7FFFFFFF)

    return values < test_size


def iter_prepared_xy_chunks(
        dataset_cfg: dict,
        project_root: Path,
        feature_set: str,
        chunk_size: int = 250_000,
        split_part: str | None = None,  # "train", "test", None
        test_size: float = 0.2,
        seed: int = 42,
) -> Iterator[tuple[pd.DataFrame, pd.Series, list[str]]]:
    row_offset = 0

    for chunk in iterate_dataset(
            dataset_cfg=dataset_cfg,
            project_root=project_root,
            chunk_size=chunk_size,
    ):
        original_chunk_size = len(chunk)

        x, y, feature_columns = prepare_xy(
            df=chunk,
            dataset_cfg=dataset_cfg,
            feature_set=feature_set,
        )

        if split_part is not None:
            test_mask = _make_split_mask(
                start_index=row_offset,
                size=len(y),
                test_size=test_size,
                seed=seed,
            )

            if split_part == "train":
                keep_mask = ~test_mask
            elif split_part == "test":
                keep_mask = test_mask
            else:
                raise ValueError(f"Unknown split_part: {split_part}")

            x = cast(pd.DataFrame, x.loc[keep_mask].reset_index(drop=True))
            y = cast(pd.Series, y.loc[keep_mask].reset_index(drop=True))

        row_offset += original_chunk_size

        if len(y) > 0:
            yield x, y, feature_columns

        del chunk
        del x
        del y
        gc.collect()
