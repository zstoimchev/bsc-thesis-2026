import json
from pathlib import Path

import torch

from src.common.data_loader import load_prepared_xy
from src.common.torch_training import (
    DEFAULT_TORCH_BINARY_CONFIG,
    config_to_dict,
    ensure_numeric_features,
    fit_standardizer,
    get_torch_device,
    make_binary_dataloaders,
    sample_balanced_binary,
    set_torch_seed,
    train_binary_torch_model,
)
from src.mdl04_transformer.model import build_tabular_transformer

MODEL_NAME = "mdl04_transformer"
MODEL_TYPE = "tabular_transformer"

D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1


def train(
        output_dir: Path,
        model_path: Path,
        project_root: Path,
        seed: int,
        split_id: str,
        split_cfg: dict,
        split_metadata: dict,
) -> None:
    print(f"[{MODEL_NAME}] Training PyTorch Tabular Transformer")
    print(f"[{MODEL_NAME}] split_id={split_id}")

    config = DEFAULT_TORCH_BINARY_CONFIG

    set_torch_seed(seed)

    device = get_torch_device()
    print(f"[{MODEL_NAME}] device={device}")

    print(f"[{MODEL_NAME}] loading train split")

    x_train, y_train = load_prepared_xy(
        split_metadata=split_metadata,
        project_root=project_root,
        part="train",
    )

    ensure_numeric_features(
        x=x_train,
        model_name="Tabular Transformer",
    )

    full_training_rows = int(len(y_train))

    print(f"[{MODEL_NAME}] full train shape={x_train.shape}")
    print(f"[{MODEL_NAME}] full label counts={y_train.value_counts().sort_index().to_dict()}")

    x_train, y_train = sample_balanced_binary(
        x=x_train,
        y=y_train,
        max_rows=config.max_train_rows,
        seed=seed,
    )

    print(f"[{MODEL_NAME}] sampled train shape={x_train.shape}")
    print(f"[{MODEL_NAME}] sampled label counts={y_train.value_counts().sort_index().to_dict()}")

    feature_columns = list(x_train.columns)

    x_np, feature_mean, feature_std = fit_standardizer(x_train)
    y_np = y_train.to_numpy(dtype="float32")

    train_loader, val_loader, train_size, val_size = make_binary_dataloaders(
        x_np=x_np,
        y_np=y_np,
        config=config,
        seed=seed,
    )

    model = build_tabular_transformer(
        num_features=len(feature_columns),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    history = train_binary_torch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        model_name=MODEL_NAME,
    )

    artifact = {
        "model_state_dict": model.state_dict(),
        "model_type": MODEL_TYPE,
        "num_features": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "split_id": split_id,
        "seed": seed,
        "full_training_rows": full_training_rows,
        "sampled_training_rows": int(len(y_train)),
        "sampled_label_counts": {
            str(k): int(v)
            for k, v in y_train.value_counts().sort_index().items()
        },
        "train_size_after_validation_split": int(train_size),
        "val_size_after_validation_split": int(val_size),
        "training_config": config_to_dict(config),
        "architecture": {
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
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

    print(f"[{MODEL_NAME}] saved model to: {model_path}")
