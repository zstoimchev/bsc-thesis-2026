import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

from src.common.data_loader import load_dataset
from src.common.metrics import compute_metrics
from src.common.preprocessing import split_xy, cap_training_dataframe
from src.libraries.torch_common import TorchBinaryTrainingConfig, DEFAULT_TORCH_BINARY_CONFIG, get_device, \
    check_numeric_features


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize_training_data(x_train: pd.DataFrame) -> tuple[np.ndarray, list[float], list[float]]:
    x_np = x_train.to_numpy(dtype=np.float32)

    mean = x_np.mean(axis=0)
    std = x_np.std(axis=0)

    std[std == 0] = 1.0

    x_np = (x_np - mean) / std

    return x_np.astype(np.float32), mean.tolist(), std.tolist()


def select_validation_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, (probabilities >= best_threshold).astype(np.int64))

    for value in range(5, 96):
        threshold = value / 100

        predictions = (probabilities >= threshold).astype(np.int64)
        metrics = compute_metrics(y_true, predictions)

        current_score = (metrics["f1"], metrics["balanced_accuracy"])
        best_score = (best_metrics["f1"], best_metrics["balanced_accuracy"])

        if current_score > best_score:
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def train_pytorch_binary_classifier(
        output_dir,
        model_path,
        project_root,
        seed: int,
        split_id: str,
        split_metadata: dict,
        model_id: str,
        model_name: str,
        model_type: str,
        build_model_fn,
        architecture: dict,
        config: TorchBinaryTrainingConfig = DEFAULT_TORCH_BINARY_CONFIG,
        cap: int | None = None,
        tune_threshold: bool = False,
        use_class_weight: bool = False,
) -> None:
    print(f"{model_id} Training {model_name}")
    print(f"{model_id} split_id={split_id}")

    set_seed(seed)
    device = get_device()

    print(f"{model_id} device={device}")
    print(f"{model_id} loading train split")

    train_df = load_dataset(
        dataset_cfg={
            "path": split_metadata["train_file"],
            "format": "parquet",
        },
        project_root=project_root,
    )

    full_training_rows = len(train_df)

    train_df = cap_training_dataframe(
        df=train_df,
        label_column=split_metadata["label_column"],
        cap=cap,
        seed=seed,
    )

    x_train, y_train = split_xy(
        df=train_df,
        label_column=split_metadata["label_column"],
        feature_columns=split_metadata["feature_columns"],
    )

    check_numeric_features(model_name, x_train)

    print(f"{model_id} available training rows={full_training_rows}")
    print(f"{model_id} used train shape={x_train.shape}")
    print(f"{model_id} used label counts={y_train.value_counts().sort_index().to_dict()}")

    feature_columns = list(x_train.columns)

    x_np, feature_mean, feature_std = standardize_training_data(x_train)
    y_np = y_train.to_numpy(dtype=np.float32)

    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)

    if len(dataset) < 2:
        raise ValueError("PyTorch training requires at least 2 rows.")

    val_size = max(1, int(len(dataset) * config.validation_fraction))
    val_size = min(val_size, len(dataset) - 1, )
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    model = build_model_fn(len(feature_columns)).to(device)

    positive_weight = 1.0

    if use_class_weight:
        fitting_indices = np.asarray(train_dataset.indices, dtype=np.int64)
        fitting_labels = y_np[fitting_indices].astype(np.int64)
        negative_count = int(np.count_nonzero(fitting_labels == 0))
        positive_count = int(np.count_nonzero(fitting_labels == 1))

        if negative_count == 0 or positive_count == 0:
            raise ValueError("Class weighting requires both benign and malicious records in the fitting partition.")

        positive_weight = negative_count / positive_count

        print(f"{model_id} fitting label counts={{0: {negative_count}, 1: {positive_count}}}")
        print(f"{model_id} positive class weight={positive_weight:.4f}")

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weight,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_examples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            batch_size = batch_x.size(0)
            train_loss_sum += loss.item() * batch_size
            train_examples += batch_size

        train_loss = train_loss_sum / train_examples

        model.eval()

        val_loss_sum = 0.0
        val_examples = 0
        correct = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                probs = torch.sigmoid(logits)
                preds = (probs >= config.threshold).float()

                batch_size = batch_x.size(0)
                val_loss_sum += loss.item() * batch_size
                val_examples += batch_size
                correct += (preds == batch_y).sum().item()

        val_loss = val_loss_sum / val_examples
        val_accuracy = correct / val_examples

        epoch_info = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
        }

        history.append(epoch_info)

        print(
            f"{model_id} epoch={epoch:02d}/{config.epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

    selected_threshold = config.threshold
    default_threshold_metrics: dict | None = None
    threshold_validation_metrics: dict | None = None

    if tune_threshold:
        validation_probabilities = []
        validation_labels = []

        model.eval()

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)

                logits = model(batch_x)
                probabilities = torch.sigmoid(logits)

                validation_probabilities.append(probabilities.cpu().numpy())
                validation_labels.append(batch_y.cpu().numpy().astype(np.int64))

        validation_probabilities_np = np.concatenate(validation_probabilities)
        validation_labels_np = np.concatenate(validation_labels)

        default_threshold_metrics = compute_metrics(
            validation_labels_np,
            (
                    validation_probabilities_np >= config.threshold
            ).astype(np.int64),
        )

        selected_threshold, threshold_validation_metrics = (
            select_validation_threshold(
                y_true=validation_labels_np,
                probabilities=validation_probabilities_np,
            )
        )

        print(
            f"{model_id} default validation threshold="
            f"{config.threshold:.2f} "
            f"validation_f1="
            f"{default_threshold_metrics['f1']:.4f} "
            f"validation_balanced_accuracy="
            f"{default_threshold_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"{model_id} selected validation threshold="
            f"{selected_threshold:.2f} "
            f"validation_f1="
            f"{threshold_validation_metrics['f1']:.4f} "
            f"validation_balanced_accuracy="
            f"{threshold_validation_metrics['balanced_accuracy']:.4f}"
        )

    artifact = {
        "model_state_dict": model.state_dict(),
        "model_type": model_type,
        "num_features": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "split_id": split_id,
        "seed": seed,
        "train_row_cap": cap,
        "full_training_rows": int(full_training_rows),
        "training_rows": int(len(y_train)),
        "fitting_rows": int(train_size),
        "validation_rows": int(val_size),
        "training_label_counts": {
            str(k): int(v)
            for k, v in y_train.value_counts().sort_index().items()
        },
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "use_class_weight": bool(use_class_weight),
        "positive_class_weight": float(positive_weight),
        "validation_fraction": config.validation_fraction,
        "default_threshold": float(config.threshold),
        "default_threshold_validation_metrics": default_threshold_metrics,
        "threshold": float(selected_threshold),
        "threshold_tuned": bool(tune_threshold),
        "threshold_selection_metric": "f1" if tune_threshold else None,
        "threshold_validation_metrics": threshold_validation_metrics,

        "architecture": architecture,
        "history": history,
    }

    torch.save(artifact, model_path)

    summary_path = output_dir / "training_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in artifact.items() if k != "model_state_dict"},
            f,
            indent=2,
        )

    print(f"{model_id} saved model to: {model_path}")
