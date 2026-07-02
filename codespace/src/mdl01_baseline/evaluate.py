from pathlib import Path

import joblib

from src.common.data_loader import load_dataset
from src.common.metrics import compute_metrics
from src.common.preprocessing import split_xy


def evaluate(
    model_path: Path,
    project_root: Path,
    seed: int,
    split_id: str,
    split_cfg: dict,
    split_metadata: dict,
) -> dict:
    print("[mdl01_baseline] Evaluating majority-class baseline")

    artifact = joblib.load(model_path)
    model = artifact["model"]

    test_df = load_dataset(
        dataset_cfg={
            "path": split_metadata["test_file"],
            "format": "parquet",
        },
        project_root=project_root,
    )

    x_test, y_test = split_xy(
        df=test_df,
        label_column=split_metadata["label_column"],
        feature_columns=split_metadata["feature_columns"],
    )

    expected_features = artifact["feature_columns"]
    x_test = x_test[expected_features]

    y_pred = model.predict(x_test)

    metrics = compute_metrics(y_test, y_pred)

    metrics["model_type"] = "majority_class_baseline"
    metrics["majority_class"] = artifact["majority_class"]
    metrics["training_label_counts"] = artifact["label_counts"]
    metrics["training_rows"] = artifact["training_rows"]
    metrics["evaluation_rows"] = int(len(y_test))
    metrics["feature_columns"] = expected_features
    metrics["split_id"] = split_id
    metrics["seed"] = seed

    return metrics