import json
from pathlib import Path

import joblib

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
    print(
        "[mdl05_rule_based] Initializing fixed "
        "rule-based detector"
    )
    print(f"[mdl05_rule_based] split_id={split_id}")

    feature_columns = split_metadata["feature_columns"]

    missing = [
        feature
        for feature in RuleBasedDetector.CORE_FEATURES
        if feature not in feature_columns
    ]

    if missing:
        raise ValueError(f"The selected split is missing core rule features: {missing}")

    model = RuleBasedDetector()

    artifact = {
        "model": model,
        "model_type": "fixed_rule_based_detector",
        "split_id": split_id,
        "seed": seed,
        "train_row_cap": None,
        "full_training_rows": 0,
        "training_rows": 0,
        "training_label_counts": {},
        "source_feature_columns": list(feature_columns),
        "params": {
            "method": "fixed_hardcoded_rules",
            "training_data_used": False,
            "thresholds": RuleBasedDetector.THRESHOLDS,
        },
    }

    joblib.dump(artifact, model_path)

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                key: value
                for key, value in artifact.items()
                if key != "model"
            },
            file,
            indent=2,
        )

    print("[mdl05_rule_based] training data used=0")
    print(f"[mdl05_rule_based] source features={len(feature_columns)}")
    print(f"[mdl05_rule_based] fixed thresholds={RuleBasedDetector.THRESHOLDS}")
    print(f"[mdl05_rule_based] saved detector to: {model_path}")