from pathlib import Path

import joblib
import pandas as pd

from src.common.data_loader import load_dataset
from src.common.metrics import compute_metrics
from src.common.preprocessing import split_xy


def _check_numeric_features(x: pd.DataFrame) -> None:
    non_numeric = [
        column
        for column in x.columns
        if not pd.api.types.is_numeric_dtype(x[column])
    ]

    if non_numeric:
        raise ValueError(
            "MLP expects numeric features only. "
            f"Non-numeric columns found: {non_numeric}"
        )


def evaluate(
        model_path: Path,
        project_root: Path,
        seed: int,
        split_id: str,
        split_cfg: dict,
        split_metadata: dict,
) -> dict:
    print("[mdl03_mlp] Evaluating MLP/DNN")
    print(f"[mdl03_mlp] split_id={split_id}")
    print("[mdl03_mlp] loading test split")

    artifact = joblib.load(model_path)

    model = artifact["model"]
    expected_features = artifact["feature_columns"]

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

    missing_features = [
        feature
        for feature in expected_features
        if feature not in x_test.columns
    ]

    if missing_features:
        raise ValueError(
            "Evaluation split is missing features expected by model: "
            f"{missing_features}"
        )

    _check_numeric_features(x_test)

    x_test = x_test[expected_features].astype("float32")

    print(f"[mdl03_mlp] test shape={x_test.shape}")
    print(f"[mdl03_mlp] test label counts={y_test.value_counts().sort_index().to_dict()}")

    y_pred = model.predict(x_test)

    metrics = compute_metrics(y_test, y_pred)

    metrics["model_type"] = "mlp_classifier"
    metrics["split_id"] = split_id
    metrics["seed"] = seed
    metrics["training_sample_rows"] = artifact["sampled_training_rows"]
    metrics["training_sample_label_counts"] = artifact["sampled_label_counts"]
    metrics["evaluation_rows"] = int(len(y_test))
    metrics["feature_columns"] = expected_features
    metrics["hidden_layer_sizes"] = artifact["hidden_layer_sizes"]
    metrics["max_iter"] = artifact["max_iter"]
    metrics["actual_iterations"] = artifact["actual_iterations"]
    metrics["training_loss"] = artifact["loss"]

    return metrics
