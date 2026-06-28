import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.preprocessing import iter_prepared_xy_chunks
from src.mdl02_xgboost.model import build_xgboost_classifier

MAX_TRAIN_ROWS = 500_000
LOG_EVERY_N_CHUNKS = 7


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


def train(
        dataset_cfg: dict,
        output_dir: Path,
        model_path: Path,
        project_root: Path,
        feature_set: str,
        chunk_size: int,
        split: str,
        test_size: float,
        seed: int,
) -> None:
    if split != "random":
        raise NotImplementedError(
            f"Split strategy '{split}' is not implemented for mdl02_xgboost."
        )

    max_per_class = MAX_TRAIN_ROWS // 2

    collected_x = []
    collected_y = []

    sampled_counts = {
        0: 0,
        1: 0,
    }

    scanned_rows = 0
    chunk_count = 0
    feature_columns = None

    print("[mdl02_xgboost] Training XGBoost")
    print("[mdl02_xgboost] Using split part: train")
    print(
        f"[mdl02_xgboost] Sampling up to {max_per_class} benign "
        f"and {max_per_class} attack rows"
    )

    for x_chunk, y_chunk, current_feature_columns in iter_prepared_xy_chunks(
            dataset_cfg=dataset_cfg,
            project_root=project_root,
            feature_set=feature_set,
            chunk_size=chunk_size,
            split_part="train",
            test_size=test_size,
            seed=seed,
    ):
        chunk_count += 1
        scanned_rows += len(y_chunk)

        _check_numeric_features(x_chunk)

        if feature_columns is None:
            feature_columns = list(current_feature_columns)
        elif feature_columns != list(current_feature_columns):
            raise ValueError(
                "Feature columns changed between chunks. "
                "This means preprocessing is inconsistent."
            )

        for label in [0, 1]:
            remaining = max_per_class - sampled_counts[label]

            if remaining <= 0:
                continue

            label_indices = y_chunk[y_chunk == label].index[:remaining]

            if len(label_indices) > 0:
                collected_x.append(x_chunk.loc[label_indices])
                collected_y.append(y_chunk.loc[label_indices])
                sampled_counts[label] += len(label_indices)

        if chunk_count == 1 or chunk_count % LOG_EVERY_N_CHUNKS == 0:
            print(
                f"[mdl02_xgboost] scanned chunk {chunk_count}: "
                f"scanned rows={scanned_rows}, "
                f"sampled={sampled_counts}"
            )

        if sampled_counts[0] >= max_per_class and sampled_counts[1] >= max_per_class:
            print("[mdl02_xgboost] Reached training sample limit")
            break

    if not collected_x or not collected_y:
        raise ValueError("No training data collected for XGBoost.")

    x_train = pd.concat(collected_x, ignore_index=True)
    y_train = pd.concat(collected_y, ignore_index=True).astype(int)

    x_train = x_train.astype("float32")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(y_train))

    x_train = x_train.iloc[permutation].reset_index(drop=True)
    y_train = y_train.iloc[permutation].reset_index(drop=True)

    print(f"[mdl02_xgboost] Final training shape: {x_train.shape}")
    print(f"[mdl02_xgboost] Final label counts: {y_train.value_counts().to_dict()}")

    model = build_xgboost_classifier(seed=seed)
    model.fit(x_train, y_train)

    artifact = {
        "model": model,
        "model_type": "xgboost_classifier",
        "feature_columns": feature_columns,
        "feature_set": feature_set,
        "split": split,
        "split_part": "train",
        "test_size": test_size,
        "seed": seed,
        "max_train_rows": MAX_TRAIN_ROWS,
        "sampled_label_counts": {
            "0": int((y_train == 0).sum()),
            "1": int((y_train == 1).sum()),
        },
        "sampled_training_rows": int(len(y_train)),
        "scanned_training_rows": int(scanned_rows),
        "chunk_count": int(chunk_count),
        "params": model.get_params(),
    }

    joblib.dump(artifact, model_path)

    summary = {
        key: value
        for key, value in artifact.items()
        if key != "model"
    }

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[mdl02_xgboost] saved model to: {model_path}")
