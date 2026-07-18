from pathlib import Path

from src.libraries.torch_training import train_pytorch_binary_classifier
from src.mdl04_transformer.model import build_tabular_transformer

MODEL_ID = "[mdl04_transformer]"
MODEL_NAME = "PyTorch Tabular Transformer"
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
        split_metadata: dict,
        cap: int | None = None,
) -> None:
    train_pytorch_binary_classifier(
        output_dir=output_dir,
        model_path=model_path,
        project_root=project_root,
        seed=seed,
        split_id=split_id,
        split_metadata=split_metadata,
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        model_type=MODEL_TYPE,
        build_model_fn=lambda num_features: build_tabular_transformer(
            num_features=num_features,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ),
        architecture={
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "activation": "gelu",
            "pooling": "mean",
            "dropout": DROPOUT,
        },
        cap=cap,
    )
