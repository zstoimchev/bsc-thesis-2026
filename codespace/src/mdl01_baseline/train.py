import json
from pathlib import Path

import joblib

from src.common.data_loader import load_dataset
from src.common.preprocessing import split_xy
from src.mdl01_baseline.model import MajorityClassBaseline


def train(
    output_dir: Path,
    model_path: Path,
    project_root: Path,
    seed: int,
    split_id: str,
    split_cfg: dict,
    split_metadata: dict,
) -> None:
    print("[mdl01_baseline] Training majority-class baseline")

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

    model = MajorityClassBaseline()
    model.partial_fit(x_train, y_train)
    model.finalize()

    artifact = {
        "model": model,
        "model_type": "majority_class_baseline",
        "majority_class": model.majority_class,
        "label_counts": dict(model.label_counts),
        "feature_columns": model.feature_columns,
        "split_id": split_id,
        "seed": seed,
        "training_rows": int(len(y_train)),
    }

    joblib.dump(artifact, model_path)

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in artifact.items() if k != "model"},
            f,
            indent=2,
        )

    print(f"[mdl01_baseline] training rows={len(y_train)}")
    print(f"[mdl01_baseline] majority_class={model.majority_class}")
    print(f"[mdl01_baseline] saved model to: {model_path}")