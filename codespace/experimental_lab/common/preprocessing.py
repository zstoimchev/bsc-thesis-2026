from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from common.paths import resolve_path


class PreprocessingError(Exception):
    """Raised when preprocessing cannot be completed."""


BENIGN_VALUES = {
    "benign",
    "normal",
    "0",
    "false",
    "legitimate",
    "background",
}


@dataclass
class PreparedData:
    """
    Output container used by runners.

    X_train / X_test:
        transformed feature matrices

    y_train / y_test:
        encoded labels

    label_encoder:
        sklearn LabelEncoder fitted on y_train/y_test

    preprocessor:
        fitted sklearn ColumnTransformer

    label_mapping:
        encoded integer -> original label text
    """

    X_train: Any
    X_test: Any
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    preprocessor: ColumnTransformer
    label_mapping: dict[int, str]
    problem_type: str


def normalize_label_value(value: Any) -> str:
    """
    Normalize one label value into a lowercase string.
    """
    if pd.isna(value):
        return "missing"

    text = str(value).strip()

    # CIC labels sometimes include spaces or inconsistent casing.
    text = text.replace("-", "_").replace(" ", "_")

    return text.lower()


def to_binary_labels(y: pd.Series) -> pd.Series:
    """
    Convert labels into binary labels:
        benign -> benign
        everything else -> attack

    This is important for fair DDoS / IDS binary experiments.
    """
    normalized = y.map(normalize_label_value)

    return normalized.map(lambda v: "benign" if v in BENIGN_VALUES else "attack")


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Clean feature dataframe before preprocessing.

    - Normalizes column names lightly by stripping whitespace.
    - Replaces inf/-inf with NaN.
    - Drops columns that are entirely NaN.
    - Drops duplicate columns.
    """
    if not isinstance(X, pd.DataFrame):
        raise PreprocessingError(f"Expected pandas DataFrame, got {type(X).__name__}")

    out = X.copy()

    out.columns = [str(c).strip() for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]

    out = out.replace([np.inf, -np.inf], np.nan)

    all_nan_cols = [col for col in out.columns if out[col].isna().all()]
    if all_nan_cols:
        out = out.drop(columns=all_nan_cols)

    if out.shape[1] == 0:
        raise PreprocessingError("No usable feature columns remain after cleaning.")

    return out


def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align train/test columns.

    Missing test columns are created as NaN.
    Extra test columns are dropped.
    Column order follows X_train.
    """
    train = X_train.copy()
    test = X_test.copy()

    for col in train.columns:
        if col not in test.columns:
            test[col] = np.nan

    test = test[train.columns]

    return train, test


def infer_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Split dataframe columns into numeric and categorical lists.
    """
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_preprocessor(
    X_train: pd.DataFrame,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer.

    Categorical columns use OrdinalEncoder instead of OneHotEncoder on purpose:
    - lower memory usage;
    - safer for large CIC datasets;
    - works with tree models, boosting, SVM, and neural baselines.

    Unknown categories in test data are encoded as -1.
    """
    numeric_cols, categorical_cols = infer_feature_columns(X_train)

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise PreprocessingError("No numeric or categorical columns found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def encode_labels(
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, dict[int, str]]:
    """
    Encode labels.

    problem_type:
        binary      -> benign vs attack
        multiclass  -> preserve original labels
    """
    if problem_type not in {"binary", "multiclass"}:
        raise PreprocessingError(
            f"Unsupported problem_type '{problem_type}'. Use 'binary' or 'multiclass'."
        )

    if problem_type == "binary":
        y_train_clean = to_binary_labels(y_train)
        y_test_clean = to_binary_labels(y_test)
    else:
        y_train_clean = y_train.map(normalize_label_value)
        y_test_clean = y_test.map(normalize_label_value)

    encoder = LabelEncoder()

    combined = pd.concat([y_train_clean, y_test_clean], axis=0)
    encoder.fit(combined)

    y_train_encoded = encoder.transform(y_train_clean)
    y_test_encoded = encoder.transform(y_test_clean)

    label_mapping = {
        int(index): str(label)
        for index, label in enumerate(encoder.classes_)
    }

    return y_train_encoded, y_test_encoded, encoder, label_mapping


def prepare_train_test_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str = "binary",
    scale_numeric: bool = True,
) -> PreparedData:
    """
    Full preprocessing pipeline for most runners.

    This function:
    - cleans train/test features;
    - aligns test columns to train columns;
    - fits preprocessing on train only;
    - transforms train and test;
    - encodes labels.
    """
    X_train_clean = clean_features(X_train)
    X_test_clean = clean_features(X_test)

    X_train_clean, X_test_clean = align_columns(X_train_clean, X_test_clean)

    preprocessor = build_preprocessor(X_train_clean, scale_numeric=scale_numeric)

    X_train_out = preprocessor.fit_transform(X_train_clean)
    X_test_out = preprocessor.transform(X_test_clean)

    y_train_out, y_test_out, label_encoder, label_mapping = encode_labels(
        y_train=y_train,
        y_test=y_test,
        problem_type=problem_type,
    )

    return PreparedData(
        X_train=X_train_out,
        X_test=X_test_out,
        y_train=y_train_out,
        y_test=y_test_out,
        label_encoder=label_encoder,
        preprocessor=preprocessor,
        label_mapping=label_mapping,
        problem_type=problem_type,
    )


def save_preprocessing_artifacts(
    artifact_dir: str,
    prepared: PreparedData,
) -> None:
    """
    Save preprocessor and label encoder into artifact directory.
    """
    path = resolve_path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(prepared.preprocessor, path / "preprocessor.joblib")
    joblib.dump(prepared.label_encoder, path / "label_encoder.joblib")


def load_preprocessing_artifacts(
    artifact_dir: str,
) -> tuple[ColumnTransformer, LabelEncoder]:
    """
    Load preprocessor and label encoder from artifact directory.
    """
    path = resolve_path(artifact_dir)

    preprocessor_path = path / "preprocessor.joblib"
    label_encoder_path = path / "label_encoder.joblib"

    if not preprocessor_path.exists():
        raise PreprocessingError(f"Missing preprocessor artifact: {preprocessor_path}")

    if not label_encoder_path.exists():
        raise PreprocessingError(f"Missing label encoder artifact: {label_encoder_path}")

    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)

    return preprocessor, label_encoder