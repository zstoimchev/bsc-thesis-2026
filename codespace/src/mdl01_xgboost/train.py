import json
from pathlib import Path

import joblib
import pandas as pd

from src.common.data_loader import load_dataset
from src.common.preprocessing import split_xy, cap_training_dataframe
from src.mdl01_xgboost.model import build_xgboost_classifier


def _check_numeric_features(x: pd.DataFrame) -> None:
    non_numeric = [
        column
        for column in x.columns
        if not pd.api.types.is_numeric_dtype(x[column])
    ]

    if non_numeric:
        raise ValueError("XGBoost expects numeric features only. Non-numeric columns found: {non_numeric}")


def train(
        output_dir: Path,
        model_path: Path,
        project_root: Path,
        seed: int,
        split_id: str,
        split_metadata: dict,
        cap: int | None = None,
) -> None:
    print("[mdl01_xgboost] Training XGBoost")
    print(f"[mdl01_xgboost] split_id={split_id}")
    print("[mdl01_xgboost] loading train split")

    train_df = load_dataset(
        dataset_cfg={
            "path": split_metadata["train_file"],
            "format": "parquet",
        },
        project_root=project_root,
    )

    full_training_rows = len(train_df)
    train_df = cap_training_dataframe(
        df=train_df,
        label_column=split_metadata["label_column"],
        cap=cap,
        seed=seed,
    )

    x_train, y_train = split_xy(
        df=train_df,
        label_column=split_metadata["label_column"],
        feature_columns=split_metadata["feature_columns"],
    )

    _check_numeric_features(x_train)

    x_train = x_train.astype("float32")

    print(f"[mdl01_xgboost] available training rows={full_training_rows}")
    print(f"[mdl01_xgboost] used train shape={x_train.shape}")
    print(f"[mdl01_xgboost] used label counts={y_train.value_counts().sort_index().to_dict()}")

    model = build_xgboost_classifier(seed=seed)
    model.fit(x_train, y_train)

    artifact = {
        "model": model,
        "model_type": "xgboost_classifier",
        "feature_columns": list(x_train.columns),
        "split_id": split_id,
        "seed": seed,
        "train_row_cap": cap,
        "full_training_rows": int(full_training_rows),
        "training_rows": int(len(y_train)),
        "training_label_counts": {
            str(k): int(v)
            for k, v in (
                y_train.value_counts()
                .sort_index()
                .items()
            )
        },
        "params": model.get_params(),
    }

    joblib.dump(artifact, model_path)

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in artifact.items() if k != "model"},
            f,
            indent=2,
        )

    print(f"[mdl01_xgboost] saved model to: {model_path}")
