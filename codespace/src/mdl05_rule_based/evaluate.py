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
    print("[mdl05_rule_based] Evaluating rule-based detector")
    print(f"[mdl05_rule_based] split_id={split_id}")

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

    missing_features = [feature for feature in expected_features if feature not in x_test.columns]

    if missing_features:
        raise ValueError("Evaluation split is missing features expected by the rule-based detector: {missing_features}")

    x_test = x_test[expected_features]

    print(f"[mdl05_rule_based] test shape={x_test.shape}")
    print(f"[mdl05_rule_based] test label counts={y_test.value_counts().sort_index().to_dict()}")

    y_pred = model.predict(x_test)
    metrics = compute_metrics(y_test, y_pred)

    metrics["model_type"] = "rule_based_detector"
    metrics["split_id"] = split_id
    metrics["seed"] = seed
    metrics["training_rows"] = artifact["training_rows"]
    metrics["training_label_counts"] = artifact["training_label_counts"]
    metrics["train_row_cap"] = artifact["train_row_cap"]
    metrics["full_training_rows"] = artifact["full_training_rows"]
    metrics["benign_calibration_rows"] = artifact["benign_calibration_rows"]
    metrics["evaluation_rows"] = int(len(y_test))
    metrics["feature_columns"] = expected_features
    metrics["params"] = artifact["params"]

    return metrics
