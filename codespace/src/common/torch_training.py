from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


@dataclass(frozen=True)
class TorchBinaryTrainingConfig:
    max_train_rows: int = 200_000
    batch_size: int = 4096
    eval_batch_size: int = 4096
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_fraction: float = 0.1
    threshold: float = 0.5


DEFAULT_TORCH_BINARY_CONFIG = TorchBinaryTrainingConfig()


def config_to_dict(config: TorchBinaryTrainingConfig) -> dict:
    return asdict(config)


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_numeric_features(
        x: pd.DataFrame,
        model_name: str,
) -> None:
    non_numeric = [
        column
        for column in x.columns
        if not pd.api.types.is_numeric_dtype(x[column])
    ]

    if non_numeric:
        raise ValueError(
            f"{model_name} expects numeric features only. "
            f"Non-numeric columns found: {non_numeric}"
        )


def sample_balanced_binary(
        x: pd.DataFrame,
        y: pd.Series,
        max_rows: int,
        seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    max_per_class = max_rows // 2
    rng = np.random.default_rng(seed)

    sampled_indices = []

    for label in [0, 1]:
        label_indices = y[y == label].index.to_numpy()
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
        raise ValueError("No rows sampled for binary training.")

    sampled_indices = rng.permutation(sampled_indices)

    x_sample = x.loc[sampled_indices].reset_index(drop=True)
    y_sample = y.loc[sampled_indices].reset_index(drop=True)

    return x_sample, y_sample


def fit_standardizer(
        x: pd.DataFrame,
) -> tuple[np.ndarray, list[float], list[float]]:
    x_np = x.to_numpy(dtype=np.float32)

    mean = x_np.mean(axis=0)
    std = x_np.std(axis=0)

    std[std == 0] = 1.0

    x_np = (x_np - mean) / std
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)

    return x_np.astype(np.float32), mean.tolist(), std.tolist()


def apply_standardizer(
        x: pd.DataFrame,
        feature_mean: list[float],
        feature_std: list[float],
) -> np.ndarray:
    x_np = x.to_numpy(dtype=np.float32)

    mean = np.array(feature_mean, dtype=np.float32)
    std = np.array(feature_std, dtype=np.float32)

    std[std == 0] = 1.0

    x_np = (x_np - mean) / std
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)

    return x_np.astype(np.float32)


def make_binary_dataloaders(
        x_np: np.ndarray,
        y_np: np.ndarray,
        config: TorchBinaryTrainingConfig,
        seed: int,
) -> tuple[DataLoader, DataLoader, int, int]:
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)

    val_size = int(len(dataset) * config.validation_fraction)
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

    return train_loader, val_loader, train_size, val_size


def train_binary_torch_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TorchBinaryTrainingConfig,
        device: torch.device,
        model_name: str,
) -> list[dict]:
    criterion = nn.BCEWithLogitsLoss()

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
            f"[{model_name}] epoch={epoch:02d}/{config.epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

    return history


def predict_binary_torch_model(
        model: nn.Module,
        x_np: np.ndarray,
        config: TorchBinaryTrainingConfig,
        device: torch.device,
) -> np.ndarray:
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=config.eval_batch_size,
        shuffle=False,
    )

    predictions = []

    model.eval()

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            preds = (probs >= config.threshold).long()

            predictions.append(preds.cpu().numpy())

    return np.concatenate(predictions)
