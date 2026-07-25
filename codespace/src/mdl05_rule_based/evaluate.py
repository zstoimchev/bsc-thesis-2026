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
    print("[mdl05_rule_based] Evaluating fixed rule-based detector")
    print(f"[mdl05_rule_based] split_id={split_id}")

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

    print(f"[mdl05_rule_based] test shape={x_test.shape}")
    print(f"[mdl05_rule_based] test label counts={y_test.value_counts().sort_index().to_dict()}")

    y_pred, rule_hits = model.predict_with_details(x_test)
    metrics = compute_metrics(y_test, y_pred)

    used_features = model.used_features(x_test)

    print(f"[mdl05_rule_based] rule features used={len(used_features)}")
    print(f"[mdl05_rule_based] rule hits={rule_hits}")

    metrics["model_type"] = "fixed_rule_based_detector"
    metrics["training_split_id"] = artifact["split_id"]
    metrics["training_rows"] = 0
    metrics["training_label_counts"] = {}
    metrics["train_row_cap"] = None
    metrics["full_training_rows"] = 0
    metrics["evaluation_rows"] = int(len(y_test))
    metrics["evaluation_label_counts"] = {
        str(label): int(count)
        for label, count in (
            y_test.value_counts().sort_index().items()
        )
    }
    metrics["feature_columns"] = list(x_test.columns)
    metrics["rule_features_used"] = used_features
    metrics["rule_hits"] = rule_hits
    metrics["split_id"] = split_id
    metrics["seed"] = seed
    metrics["params"] = artifact["params"]

    return metrics