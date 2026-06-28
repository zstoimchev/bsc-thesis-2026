from pathlib import Path

import joblib
import pandas as pd

from src.common.metrics import BinaryMetricsAccumulator
from src.common.preprocessing import iter_prepared_xy_chunks


LOG_EVERY_N_CHUNKS = 5


def _check_numeric_features(x: pd.DataFrame) -> None:
    non_numeric = [
        column for column in x.columns
        if not pd.api.types.is_numeric_dtype(x[column])
    ]

    if non_numeric:
        raise ValueError(
            "XGBoost currently expects numeric features only. "
            f"Non-numeric columns found: {non_numeric}"
        )


def evaluate(
    dataset_cfg: dict,
    model_path: Path,
    project_root: Path,
    feature_set: str,
    chunk_size: int,
    split: str,
    test_size: float,
    seed: int,
) -> dict:
    if split != "random":
        raise NotImplementedError(
            f"Split strategy '{split}' is not implemented for mdl02_xgboost."
        )

    artifact = joblib.load(model_path)

    model = artifact["model"]
    expected_features = artifact["feature_columns"]

    metrics = BinaryMetricsAccumulator()

    chunk_count = 0
    total_rows = 0

    print("[mdl02_xgboost] Evaluating XGBoost")
    print("[mdl02_xgboost] Using split part: test")

    for x_chunk, y_chunk, feature_columns in iter_prepared_xy_chunks(
        dataset_cfg=dataset_cfg,
        project_root=project_root,
        feature_set=feature_set,
        chunk_size=chunk_size,
        split_part="test",
        test_size=test_size,
        seed=seed,
    ):
        missing_features = [
            feature for feature in expected_features
            if feature not in x_chunk.columns
        ]

        if missing_features:
            raise ValueError(
                "Evaluation dataset is missing features expected by model: "
                f"{missing_features}"
            )

        _check_numeric_features(x_chunk)

        x_chunk = x_chunk[expected_features].astype("float32")

        y_pred = model.predict(x_chunk)

        metrics.update(y_chunk, y_pred)

        chunk_count += 1
        total_rows += len(y_chunk)

        if chunk_count == 1 or chunk_count % LOG_EVERY_N_CHUNKS == 0:
            print(
                f"[mdl02_xgboost] evaluated chunk {chunk_count}: "
                f"total rows so far: {total_rows}"
            )

    result = metrics.compute()

    result["model_type"] = "xgboost_classifier"
    result["training_sample_rows"] = artifact["sampled_training_rows"]
    result["training_sample_label_counts"] = artifact["sampled_label_counts"]
    result["scanned_training_rows"] = artifact["scanned_training_rows"]
    result["evaluation_rows"] = total_rows
    result["chunk_count"] = chunk_count
    result["feature_columns"] = expected_features
    result["feature_set"] = feature_set
    result["split"] = split
    result["split_part"] = "test"
    result["test_size"] = test_size
    result["seed"] = seed

    return result