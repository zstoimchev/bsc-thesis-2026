import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

from src.common.data_loader import load_dataset
from src.common.preprocessing import split_xy, cap_training_dataframe
from src.libraries.torch_common import TorchBinaryTrainingConfig, DEFAULT_TORCH_BINARY_CONFIG, get_device, \
    check_numeric_features


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize_training_data(
        x_train: pd.DataFrame
) -> tuple[np.ndarray, list[float], list[float]]:
    x_np = x_train.to_numpy(dtype=np.float32)

    mean = x_np.mean(axis=0)
    std = x_np.std(axis=0)

    std[std == 0] = 1.0

    x_np = (x_np - mean) / std

    return x_np.astype(np.float32), mean.tolist(), std.tolist()


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
    val_size = min(val_size, len(dataset) - 1,)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = build_model_fn(len(feature_columns)).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

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
        "validation_fraction": config.validation_fraction,
        "threshold": config.threshold,
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
