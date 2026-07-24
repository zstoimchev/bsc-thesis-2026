import json
from pathlib import Path

import joblib

from src.common.data_loader import load_dataset
from src.common.preprocessing import cap_training_dataframe, split_xy
from src.mdl05_rule_based.model import RuleBasedDetector


def train(
        output_dir: Path,
        model_path: Path,
        project_root: Path,
        seed: int,
        split_id: str,
        split_metadata: dict,
        cap: int | None = None,
) -> None:
    print("[mdl05_rule_based] Calibrating rule-based detector")
    print(f"[mdl05_rule_based] split_id={split_id}")

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

    model = RuleBasedDetector()
    model.fit(x_train, y_train)

    label_counts = {
        str(label): int(count)
        for label, count in y_train.value_counts().sort_index().items()
    }

    artifact = {
        "model": model,
        "model_type": "rule_based_detector",
        "feature_columns": model.feature_columns,
        "split_id": split_id,
        "seed": seed,
        "train_row_cap": cap,
        "full_training_rows": int(full_training_rows),
        "training_rows": int(len(y_train)),
        "training_label_counts": label_counts,
        "benign_calibration_rows": int((y_train == 0).sum()),
        "params": {
            "required_votes": 2,
            "packet_rate_quantile": 0.995,
            "byte_rate_quantile": 0.995,
            "syn_count_quantile": 0.99,
            "backward_packets_quantile": 0.10,
            "short_duration_quantile": 0.10,
            "forward_packets_quantile": 0.99,
            "thresholds": model.thresholds,
        },
    }

    joblib.dump(artifact, model_path)

    summary_path = output_dir / "training_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                key: value
                for key, value in artifact.items()
                if key != "model"
            },
            file,
            indent=2,
        )

    print(f"[mdl05_rule_based] available training rows={full_training_rows}")
    print(f"[mdl05_rule_based] used training rows={len(y_train)}")
    print(f"[mdl05_rule_based] benign calibration rows={artifact['benign_calibration_rows']}")
    print(f"[mdl05_rule_based] thresholds={model.thresholds}")
    print(f"[mdl05_rule_based] saved model to: {model_path}")
