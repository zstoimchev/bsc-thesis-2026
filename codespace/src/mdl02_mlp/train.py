import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.data_loader import load_dataset
from src.common.preprocessing import split_xy
from src.mdl02_mlp.model import build_mlp_classifier

MAX_TRAIN_ROWS = 200_000


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


def _sample_training_data(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    max_per_class = MAX_TRAIN_ROWS // 2
    rng = np.random.default_rng(seed)

    sampled_indices = []

    for label in [0, 1]:
        label_indices = y_train[y_train == label].index.to_numpy()
        sample_size = min(len(label_indices), max_per_class)

        if sample_size == 0:
            continue

        selected = rng.choice(
            label_indices,
            size=sample_size,
            replace=False,
        )

        sampled_indices.extend(selected.tolist())

    if not sampled_indices:
        raise ValueError("No rows sampled for MLP training.")

    sampled_indices = rng.permutation(sampled_indices)

    x_sample = x_train.loc[sampled_indices].reset_index(drop=True)
    y_sample = y_train.loc[sampled_indices].reset_index(drop=True)

    return x_sample, y_sample


def train(
        output_dir: Path,
        model_path: Path,
        project_root: Path,
        seed: int,
        split_id: str,
        split_cfg: dict,
        split_metadata: dict,
) -> None:
    print("[mdl02_mlp] Training MLP/DNN")
    print(f"[mdl02_mlp] split_id={split_id}")
    print("[mdl02_mlp] loading train split")

    train_df = load_dataset(
        dataset_cfg={
            "path": split_metadata["train_file"],
            "format": "parquet",
        },
        project_root=project_root,
    )

    x_train, y_train = split_xy(
        df=train_df,
        label_column=split_metadata["label_column"],
        feature_columns=split_metadata["feature_columns"],
    )

    _check_numeric_features(x_train)

    print(f"[mdl02_mlp] full train shape={x_train.shape}")
    print(f"[mdl02_mlp] full label counts={y_train.value_counts().sort_index().to_dict()}")

    x_train, y_train = _sample_training_data(
        x_train=x_train,
        y_train=y_train,
        seed=seed,
    )

    x_train = x_train.astype("float32")

    print(f"[mdl02_mlp] sampled train shape={x_train.shape}")
    print(f"[mdl02_mlp] sampled label counts={y_train.value_counts().sort_index().to_dict()}")

    model = build_mlp_classifier(seed=seed)
    model.fit(x_train, y_train)

    mlp = model.named_steps["mlp"]

    artifact = {
        "model": model,
        "model_type": "mlp_classifier",
        "feature_columns": list(x_train.columns),
        "split_id": split_id,
        "seed": seed,
        "max_train_rows": MAX_TRAIN_ROWS,
        "full_training_rows": int(len(train_df)),
        "sampled_training_rows": int(len(y_train)),
        "sampled_label_counts": {
            str(k): int(v)
            for k, v in y_train.value_counts().sort_index().items()
        },
        "hidden_layer_sizes": mlp.hidden_layer_sizes,
        "max_iter": mlp.max_iter,
        "actual_iterations": int(mlp.n_iter_),
        "loss": float(mlp.loss_),
        "params": model.get_params(),
    }

    joblib.dump(artifact, model_path)

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in artifact.items() if k != "model"},
            f,
            indent=2,
            default=str,
        )

    print(f"[mdl02_mlp] actual iterations={mlp.n_iter_}")
    print(f"[mdl02_mlp] final loss={mlp.loss_:.6f}")
    print(f"[mdl02_mlp] saved model to: {model_path}")
