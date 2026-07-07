from dataclasses import dataclass

import pandas as pd
import torch


@dataclass(frozen=True)
class TorchBinaryTrainingConfig:
    max_train_rows: int = 500_000
    batch_size: int = 4096
    eval_batch_size: int = 4096
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_fraction: float = 0.1
    threshold: float = 0.5


DEFAULT_TORCH_BINARY_CONFIG = TorchBinaryTrainingConfig()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_numeric_features(model_name: str, x: pd.DataFrame) -> None:
    non_numeric = [
        column
        for column in x.columns
        if not pd.api.types.is_numeric_dtype(x[column])
    ]

    if non_numeric:
        raise ValueError(f"{model_name} expects numeric features only. Non-numeric columns found: {non_numeric}")
