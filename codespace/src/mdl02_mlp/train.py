from pathlib import Path

from src.common.torch_training import train_pytorch_binary_classifier
from src.mdl02_mlp.model import build_mlp_classifier

MODEL_ID = "[mdl02_mlp]"
MODEL_NAME = "PyTorch MLP/DNN"
MODEL_TYPE = "pytorch_mlp_classifier"

HIDDEN_LAYER_SIZES = (64, 32)
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
        build_model_fn=lambda num_features: build_mlp_classifier(
            num_features=num_features,
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            dropout=DROPOUT,
        ),
        architecture={
            "hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
            "activation": "relu",
            "dropout": DROPOUT,
        },
    )
