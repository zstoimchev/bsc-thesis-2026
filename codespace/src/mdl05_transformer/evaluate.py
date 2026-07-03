from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.common.data_loader import load_dataset
from src.common.metrics import compute_metrics
from src.common.preprocessing import split_xy
from src.mdl05_transformer.model import build_tabular_transformer


BATCH_SIZE = 4096


def _check_numeric_features(x: pd.DataFrame) -> None:
    non_numeric = [
        column
        for column in x.columns
        if not pd.api.types.is_numeric_dtype(x[column])
    ]

    if non_numeric:
        raise ValueError(
            "Tabular Transformer expects numeric features only. "
            f"Non-numeric columns found: {non_numeric}"
        )


def _standardize_eval_data(
    x_test: pd.DataFrame,
    feature_mean: list[float],
    feature_std: list[float],
) -> np.ndarray:
    x_np = x_test.to_numpy(dtype=np.float32)

    mean = np.array(feature_mean, dtype=np.float32)
    std = np.array(feature_std, dtype=np.float32)

    std[std == 0] = 1.0

    x_np = (x_np - mean) / std

    return x_np.astype(np.float32)


def evaluate(
    model_path: Path,
    project_root: Path,
    seed: int,
    split_id: str,
    split_cfg: dict,
    split_metadata: dict,
) -> dict:
    print("[mdl05_transformer] Evaluating PyTorch Tabular Transformer")
    print(f"[mdl05_transformer] split_id={split_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mdl05_transformer] device={device}")

    artifact = torch.load(model_path, map_location=device)

    expected_features = artifact["feature_columns"]

    print("[mdl05_transformer] loading test split")

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

    x_test = x_test[expected_features]

    print(f"[mdl05_transformer] test shape={x_test.shape}")
    print(
        "[mdl05_transformer] test label counts="
        f"{y_test.value_counts().sort_index().to_dict()}"
    )

    x_np = _standardize_eval_data(
        x_test=x_test,
        feature_mean=artifact["feature_mean"],
        feature_std=artifact["feature_std"],
    )

    y_np = y_test.to_numpy(dtype=np.int64)

    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    dataset = TensorDataset(x_tensor)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = build_tabular_transformer(
        num_features=artifact["num_features"],
        d_model=artifact["architecture"]["d_model"],
        num_heads=artifact["architecture"]["num_heads"],
        num_layers=artifact["architecture"]["num_layers"],
        dropout=artifact["architecture"]["dropout"],
    ).to(device)

    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    predictions = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            predictions.append(preds.cpu().numpy())

    y_pred = np.concatenate(predictions)

    metrics = compute_metrics(y_np, y_pred)

    metrics["model_type"] = "tabular_transformer"
    metrics["split_id"] = split_id
    metrics["seed"] = seed
    metrics["training_sample_rows"] = artifact["sampled_training_rows"]
    metrics["training_sample_label_counts"] = artifact["sampled_label_counts"]
    metrics["evaluation_rows"] = int(len(y_test))
    metrics["feature_columns"] = expected_features
    metrics["epochs"] = artifact["epochs"]
    metrics["batch_size"] = artifact["batch_size"]
    metrics["learning_rate"] = artifact["learning_rate"]
    metrics["architecture"] = artifact["architecture"]
    metrics["training_history"] = artifact["history"]

    return metrics