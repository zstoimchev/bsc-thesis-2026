import json
from pathlib import Path

import joblib

from src.common.data_loader import load_dataset
from src.common.preprocessing import split_xy, cap_training_dataframe
from src.mdl00_baseline.model import MajorityClassBaseline


def train(
        output_dir: Path,
        model_path: Path,
        project_root: Path,
        seed: int,
        split_id: str,
        split_metadata: dict,
        cap: int | None = None,
) -> None:
    print("[mdl00_baseline] Training majority-class baseline")

    train_df = load_dataset(
        dataset_cfg={
            "path": split_metadata["train_file"],
            "format": "parquet",
        },
        project_root=project_root,
    )

    full_training_rows = len(train_df)

    train_df = cap_training_dataframe(
        df=train_df,
        label_column=split_metadata["label_column"],
        cap=cap,
        seed=seed,
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
        "train_row_cap": cap,
        "full_training_rows": int(full_training_rows),
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

    print(f"[mdl00_baseline] available training rows={full_training_rows}")
    print(f"[mdl00_baseline] used training rows={len(y_train)}")
    print(f"[mdl00_baseline] majority_class={model.majority_class}")
    print(f"[mdl00_baseline] saved model to: {model_path}")
