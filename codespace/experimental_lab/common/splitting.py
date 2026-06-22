from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common.paths import SPLITS_DIR, resolve_path
from common.preprocessing import to_binary_labels


class SplittingError(Exception):
    """Raised when train/test split creation or loading fails."""


@dataclass
class SplitResult:
    """
    Result returned by get_or_create_split().
    """

    split_id: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_indices_path: Path
    test_indices_path: Path
    created: bool


def split_dir_for_id(split_id: str) -> Path:
    """
    Return folder where split index files are stored.
    """
    return SPLITS_DIR / split_id


def default_split_paths(split_id: str) -> tuple[Path, Path]:
    """
    Default train/test index paths for a split id.
    """
    split_dir = split_dir_for_id(split_id)
    return split_dir / "train_idx.npy", split_dir / "test_idx.npy"


def load_indices(train_indices_path: str | Path, test_indices_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load saved train/test indices.
    """
    train_path = resolve_path(train_indices_path)
    test_path = resolve_path(test_indices_path)

    if not train_path.exists():
        raise SplittingError(f"Missing train indices file: {train_path}")

    if not test_path.exists():
        raise SplittingError(f"Missing test indices file: {test_path}")

    return np.load(train_path), np.load(test_path)


def save_indices(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    train_indices_path: str | Path,
    test_indices_path: str | Path,
) -> None:
    """
    Save train/test indices as .npy files.
    """
    train_path = resolve_path(train_indices_path)
    test_path = resolve_path(test_indices_path)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(train_path, train_indices)
    np.save(test_path, test_indices)


def make_stratify_target(y: pd.Series, problem_type: str) -> pd.Series:
    """
    Create stratification target.

    For binary experiments, labels are converted to benign/attack first.
    For multiclass experiments, original labels are used.
    """
    if problem_type == "binary":
        return to_binary_labels(y)

    if problem_type == "multiclass":
        return y.astype(str)

    raise SplittingError(
        f"Unsupported problem_type '{problem_type}'. Use 'binary' or 'multiclass'."
    )


def can_stratify(y: pd.Series) -> bool:
    """
    Check whether stratification is safe.

    Stratified split fails if any class has fewer than 2 samples.
    """
    counts = y.value_counts(dropna=False)

    if len(counts) < 2:
        return False

    return bool((counts >= 2).all())


def create_split_indices(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    seed: int = 42,
    method: str = "stratified",
    problem_type: str = "binary",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create train/test split indices for a single-file dataset.

    method:
        stratified -> stratified if possible, otherwise fallback to random
        random     -> random split without stratification
    """
    if label_col not in df.columns:
        raise SplittingError(f"Label column '{label_col}' not found in dataframe.")

    if len(df) < 2:
        raise SplittingError("Cannot split dataframe with fewer than 2 rows.")

    if not (0.0 < float(test_size) < 1.0):
        raise SplittingError(f"test_size must be between 0 and 1, got {test_size}")

    all_indices = np.arange(len(df))
    y = df[label_col]

    stratify = None

    if method == "stratified":
        stratify_target = make_stratify_target(y, problem_type=problem_type)

        if can_stratify(stratify_target):
            stratify = stratify_target
        else:
            print(
                "[WARN] Stratified split requested, but at least one class has fewer "
                "than 2 samples. Falling back to random split."
            )

    elif method == "random":
        stratify = None

    else:
        raise SplittingError(
            f"Unsupported split method '{method}'. Use 'stratified' or 'random'."
        )

    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=float(test_size),
        random_state=int(seed),
        shuffle=True,
        stratify=stratify,
    )

    return np.asarray(train_indices), np.asarray(test_indices)


def get_or_create_split(
    split_record: dict[str, Any],
    df: pd.DataFrame,
    label_col: str,
    problem_type: str = "binary",
) -> SplitResult:
    """
    Load an existing split if index files exist, otherwise create and save it.

    Expected split registry record:

        id: cic_superset_v1
        dataset: cic_superset
        method: stratified
        test_size: 0.2
        seed: 42
        train_indices: results/splits/cic_superset_v1/train_idx.npy
        test_indices: results/splits/cic_superset_v1/test_idx.npy

    train_indices/test_indices are optional. If missing, default paths are used.
    """
    split_id = str(split_record["id"])

    default_train_path, default_test_path = default_split_paths(split_id)

    train_indices_path = resolve_path(split_record.get("train_indices", default_train_path))
    test_indices_path = resolve_path(split_record.get("test_indices", default_test_path))

    if train_indices_path.exists() and test_indices_path.exists():
        train_indices, test_indices = load_indices(train_indices_path, test_indices_path)

        return SplitResult(
            split_id=split_id,
            train_indices=train_indices,
            test_indices=test_indices,
            train_indices_path=train_indices_path,
            test_indices_path=test_indices_path,
            created=False,
        )

    method = str(split_record.get("method", "stratified"))
    test_size = float(split_record.get("test_size", 0.2))
    seed = int(split_record.get("seed", 42))

    train_indices, test_indices = create_split_indices(
        df=df,
        label_col=label_col,
        test_size=test_size,
        seed=seed,
        method=method,
        problem_type=problem_type,
    )

    save_indices(
        train_indices=train_indices,
        test_indices=test_indices,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
    )

    return SplitResult(
        split_id=split_id,
        train_indices=train_indices,
        test_indices=test_indices,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
        created=True,
    )


def apply_split_indices(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply saved train/test indices to a dataframe.
    """
    max_index = len(df) - 1

    if len(train_indices) == 0:
        raise SplittingError("Train indices are empty.")

    if len(test_indices) == 0:
        raise SplittingError("Test indices are empty.")

    if int(np.max(train_indices)) > max_index:
        raise SplittingError(
            f"Train split index out of bounds. Max index={int(np.max(train_indices))}, "
            f"dataframe max index={max_index}"
        )

    if int(np.max(test_indices)) > max_index:
        raise SplittingError(
            f"Test split index out of bounds. Max index={int(np.max(test_indices))}, "
            f"dataframe max index={max_index}"
        )

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    return train_df, test_df


def describe_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
) -> dict[str, Any]:
    """
    Return a small summary of a train/test split.
    """
    out: dict[str, Any] = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_col": label_col,
    }

    if label_col in train_df.columns:
        out["train_label_counts"] = {
            str(k): int(v)
            for k, v in train_df[label_col].value_counts(dropna=False).to_dict().items()
        }

    if label_col in test_df.columns:
        out["test_label_counts"] = {
            str(k): int(v)
            for k, v in test_df[label_col].value_counts(dropna=False).to_dict().items()
        }

    return out