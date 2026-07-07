from pathlib import Path

from src.libraries.torch_evaluating import evaluate_pytorch_binary_classifier
from src.mdl04_transformer.model import build_tabular_transformer

MODEL_ID = "[mdl04_transformer]"
MODEL_NAME = "PyTorch Tabular Transformer"
MODEL_TYPE = "tabular_transformer"


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
        build_model_from_artifact_fn=lambda artifact: build_tabular_transformer(
            num_features=artifact["num_features"],
            d_model=artifact["architecture"]["d_model"],
            num_heads=artifact["architecture"]["num_heads"],
            num_layers=artifact["architecture"]["num_layers"],
            dropout=artifact["architecture"]["dropout"],
        ),
    )
