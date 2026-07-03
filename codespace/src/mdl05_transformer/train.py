import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.common.data_loader import load_dataset
from src.common.preprocessing import split_xy
from src.mdl05_transformer.model import build_tabular_transformer


MAX_TRAIN_ROWS = 200_000
BATCH_SIZE = 2048
EPOCHS = 10
LEARNING_RATE = 0.001


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        raise ValueError("No rows sampled for Tabular Transformer training.")

    sampled_indices = rng.permutation(sampled_indices)

    x_sample = x_train.loc[sampled_indices].reset_index(drop=True)
    y_sample = y_train.loc[sampled_indices].reset_index(drop=True)

    return x_sample, y_sample


def _standardize_training_data(
    x_train: pd.DataFrame,
) -> tuple[np.ndarray, list[float], list[float]]:
    x_np = x_train.to_numpy(dtype=np.float32)

    mean = x_np.mean(axis=0)
    std = x_np.std(axis=0)

    std[std == 0] = 1.0

    x_np = (x_np - mean) / std

    return x_np.astype(np.float32), mean.tolist(), std.tolist()


def train(
    output_dir: Path,
    model_path: Path,
    project_root: Path,
    seed: int,
    split_id: str,
    split_cfg: dict,
    split_metadata: dict,
) -> None:
    print("[mdl05_transformer] Training PyTorch Tabular Transformer")
    print(f"[mdl05_transformer] split_id={split_id}")

    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mdl05_transformer] device={device}")

    print("[mdl05_transformer] loading train split")

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

    print(f"[mdl05_transformer] full train shape={x_train.shape}")
    print(
        "[mdl05_transformer] full label counts="
        f"{y_train.value_counts().sort_index().to_dict()}"
    )

    x_train, y_train = _sample_training_data(
        x_train=x_train,
        y_train=y_train,
        seed=seed,
    )

    print(f"[mdl05_transformer] sampled train shape={x_train.shape}")
    print(
        "[mdl05_transformer] sampled label counts="
        f"{y_train.value_counts().sort_index().to_dict()}"
    )

    feature_columns = list(x_train.columns)

    x_np, feature_mean, feature_std = _standardize_training_data(x_train)
    y_np = y_train.to_numpy(dtype=np.float32)

    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)

    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = build_tabular_transformer(
        num_features=len(feature_columns),
        d_model=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    history = []

    for epoch in range(1, EPOCHS + 1):
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
                preds = (probs >= 0.5).float()

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
            f"[mdl05_transformer] epoch={epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

    artifact = {
        "model_state_dict": model.state_dict(),
        "model_type": "tabular_transformer",
        "num_features": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "split_id": split_id,
        "seed": seed,
        "max_train_rows": MAX_TRAIN_ROWS,
        "full_training_rows": int(len(train_df)),
        "sampled_training_rows": int(len(y_train)),
        "sampled_label_counts": {
            str(k): int(v)
            for k, v in y_train.value_counts().sort_index().items()
        },
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "architecture": {
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
        },
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

    print(f"[mdl05_transformer] saved model to: {model_path}")