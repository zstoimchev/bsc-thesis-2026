from pathlib import Path

from src.libraries.torch_training import train_pytorch_binary_classifier
from src.mdl03_gru.model import build_gru_classifier

MODEL_ID = "[mdl03_gru]"
MODEL_NAME = "PyTorch GRU"
MODEL_TYPE = "pytorch_gru_classifier"

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.1
BIDIRECTIONAL = False


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
        build_model_fn=lambda num_features: build_gru_classifier(
            num_features=num_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            bidirectional=BIDIRECTIONAL,
        ),
        architecture={
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "bidirectional": BIDIRECTIONAL,
            "input_size": 1,
            "sequence_representation": "features_as_timesteps",
            "pooling": "last_hidden_state",
            "classifier": "layer_norm_dropout_linear",
        },
        cap=cap,
        tune_threshold=False,
        use_class_weight=False,
    )
