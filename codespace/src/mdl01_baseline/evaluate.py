from pathlib import Path

import joblib

from src.common.metrics import BinaryMetricsAccumulator
from src.common.preprocessing import iter_prepared_xy_chunks


def evaluate(
        dataset_cfg: dict,
        model_cfg: dict,
        output_dir: Path,
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
            f"Split strategy '{split}' is not implemented for mdl01_baseline."
        )

    artifact = joblib.load(model_path)

    model = artifact["model"]
    expected_features = artifact["feature_columns"]

    metrics = BinaryMetricsAccumulator()

    chunk_count = 0
    total_rows = 0

    print("[mdl01_baseline] Evaluating majority-class baseline")
    print("[mdl01_baseline] Using split part: test")

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
            feature
            for feature in expected_features
            if feature not in x_chunk.columns
        ]

        if missing_features:
            raise ValueError(
                "Evaluation dataset is missing features expected by model: "
                f"{missing_features}"
            )

        x_chunk = x_chunk[expected_features]

        y_pred = model.predict(x_chunk)
        metrics.update(y_chunk, y_pred)

        chunk_count += 1
        total_rows += len(y_chunk)

        print(
            f"[mdl01_baseline] evaluated chunk {chunk_count}: "
            f"{len(y_chunk)} rows "
            f"(total: {total_rows})"
        )

    result = metrics.compute()

    result["model_type"] = "majority_class_baseline"
    result["majority_class"] = artifact["majority_class"]
    result["training_label_counts"] = artifact["label_counts"]
    result["training_rows"] = artifact["total_rows"]
    result["evaluation_rows"] = total_rows
    result["chunk_count"] = chunk_count
    result["feature_columns"] = expected_features
    result["feature_set"] = feature_set
    result["split"] = split
    result["split_part"] = "test"
    result["test_size"] = test_size
    result["seed"] = seed

    return result
