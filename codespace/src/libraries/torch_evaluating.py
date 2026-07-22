import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.common.data_loader import load_dataset
from src.common.metrics import compute_metrics
from src.common.preprocessing import split_xy
from src.libraries.torch_common import (
    TorchBinaryTrainingConfig,
    DEFAULT_TORCH_BINARY_CONFIG,
    get_device,
    check_numeric_features
)


def standardize_eval_data(
        x_test: pd.DataFrame,
        feature_mean: list[float],
        feature_std: list[float],
) -> np.ndarray:
    x_np = x_test.to_numpy(dtype=np.float32)

    mean = np.array(feature_mean, dtype=np.float32)
    std = np.array(feature_std, dtype=np.float32)

    std[std == 0] = 1.0

    x_np = (x_np - mean) / std
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)

    return x_np.astype(np.float32)


def ensure_dataframe(x: pd.Series | pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(x)


def evaluate_pytorch_binary_classifier(
        model_path,
        project_root,
        seed: int,
        split_id: str,
        split_metadata: dict,
        model_id: str,
        model_name: str,
        model_type: str,
        build_model_from_artifact_fn,
        config: TorchBinaryTrainingConfig = DEFAULT_TORCH_BINARY_CONFIG,
) -> dict:
    print(f"{model_id} Evaluating {model_name}")
    print(f"{model_id} split_id={split_id}")

    device = get_device()
    print(f"{model_id} device={device}")

    artifact = torch.load(model_path, map_location=device)

    if artifact.get("model_type") != model_type:
        raise ValueError(
            f"Loaded artifact model_type={artifact.get('model_type')} "
            f"but expected model_type={model_type}"
        )

    expected_features = artifact["feature_columns"]

    print(f"{model_id} loading test split")

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

    check_numeric_features(model_name, x_test)

    x_test = x_test[expected_features]

    print(f"{model_id} test shape={x_test.shape}")
    print(f"{model_id} test label counts={y_test.value_counts().sort_index().to_dict()}")

    x_np = standardize_eval_data(
        x_test=ensure_dataframe(x_test),
        feature_mean=artifact["feature_mean"],
        feature_std=artifact["feature_std"],
    )

    y_np = y_test.to_numpy(dtype=np.int64)

    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    dataset = TensorDataset(x_tensor)

    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
    )

    model = build_model_from_artifact_fn(artifact).to(device)

    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    probabilities = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            logits = model(batch_x)
            probs = torch.sigmoid(logits)

            probabilities.append(probs.cpu().numpy())

    y_prob = np.concatenate(probabilities)

    selected_threshold = float(artifact.get("threshold", config.threshold))
    default_threshold = float(artifact.get("default_threshold", config.threshold))
    selected_predictions = (y_prob >= selected_threshold).astype(np.int64)
    default_predictions = (y_prob >= default_threshold).astype(np.int64)
    metrics = compute_metrics(y_np, selected_predictions)
    default_threshold_test_metrics = compute_metrics(y_np, default_predictions)

    print(
        f"{model_id} default test threshold="
        f"{default_threshold:.2f} "
        f"f1={default_threshold_test_metrics['f1']:.4f} "
        f"balanced_accuracy="
        f"{default_threshold_test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"{model_id} selected test threshold="
        f"{selected_threshold:.2f} "
        f"f1={metrics['f1']:.4f} "
        f"balanced_accuracy="
        f"{metrics['balanced_accuracy']:.4f}"
    )

    metrics["model_type"] = artifact.get("model_type", model_type)
    metrics["training_split_id"] = artifact.get("split_id")
    metrics["evaluation_split_id"] = split_id
    metrics["training_seed"] = artifact.get("seed")
    metrics["train_row_cap"] = artifact.get("train_row_cap", artifact.get("max_train_rows"))
    metrics["full_training_rows"] = artifact.get("full_training_rows")
    metrics["training_rows"] = artifact.get(
        "training_rows",
        artifact.get(
            "sampled_training_rows",
            artifact.get("sampled_rows"),
        ),
    )
    metrics["fitting_rows"] = artifact.get("fitting_rows")
    metrics["validation_rows"] = artifact.get("validation_rows")
    metrics["training_label_counts"] = (
            artifact.get("training_label_counts")
            or artifact.get("sampled_label_counts")
            or artifact.get("label_counts")
            or {}
    )
    metrics["evaluation_rows"] = int(len(y_test))
    metrics["evaluation_label_counts"] = {
        str(k): int(v)
        for k, v in y_test.value_counts().sort_index().items()
    }
    metrics["feature_columns"] = expected_features
    metrics["epochs"] = artifact.get("epochs")
    metrics["batch_size"] = artifact.get("batch_size")
    metrics["evaluation_batch_size"] = config.eval_batch_size
    metrics["learning_rate"] = artifact.get("learning_rate")
    metrics["weight_decay"] = artifact.get("weight_decay")
    metrics["validation_fraction"] = artifact.get("validation_fraction")
    metrics["default_threshold"] = default_threshold
    metrics["default_threshold_validation_metrics"] = artifact.get("default_threshold_validation_metrics")
    metrics["default_threshold_test_metrics"] = default_threshold_test_metrics
    metrics["threshold"] = selected_threshold
    metrics["threshold_tuned"] = artifact.get("threshold_tuned", False)
    metrics["threshold_selection_metric"] = artifact.get("threshold_selection_metric")
    metrics["threshold_validation_metrics"] = artifact.get("threshold_validation_metrics")
    metrics["use_class_weight"] = artifact.get("use_class_weight", False)
    metrics["positive_class_weight"] = artifact.get("positive_class_weight", 1.0)
    metrics["architecture"] = artifact.get("architecture", {})
    metrics["history"] = artifact.get("history", [])

    return metrics
