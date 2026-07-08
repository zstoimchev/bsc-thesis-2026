from pathlib import Path

from src.libraries.torch_evaluating import evaluate_pytorch_binary_classifier
from src.mdl02_mlp.model import build_mlp_classifier

MODEL_ID = "[mdl02_mlp]"
MODEL_NAME = "PyTorch MLP/DNN"
MODEL_TYPE = "pytorch_mlp_classifier"


def evaluate(
        model_path: Path,
        project_root: Path,
        seed: int,
        split_id: str,
        split_cfg: dict,
        split_metadata: dict,
) -> dict:
    return evaluate_pytorch_binary_classifier(
        model_path=model_path,
        project_root=project_root,
        seed=seed,
        split_id=split_id,
        split_metadata=split_metadata,
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        model_type=MODEL_TYPE,
        build_model_from_artifact_fn=lambda artifact: build_mlp_classifier(
            num_features=artifact["num_features"],
            hidden_layer_sizes=tuple(
                artifact["architecture"]["hidden_layer_sizes"]
            ),
            dropout=artifact["architecture"]["dropout"],
        ),
    )
