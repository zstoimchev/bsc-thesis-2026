import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.common.data_loader import load_dataset
from src.common.metrics import compute_metrics
from src.common.preprocessing import (
    cap_training_dataframe,
    split_xy,
)
from src.mdl05_rule_based.model import RuleBasedDetector


PROFILES = [
    {
        "name": "very_conservative",
        "anomaly_quantile": 0.995,
        "required_votes": 1,
    },
    {
        "name": "conservative",
        "anomaly_quantile": 0.99,
        "required_votes": 1,
    },
    {
        "name": "moderate",
        "anomaly_quantile": 0.975,
        "required_votes": 1,
    },
    {
        "name": "sensitive",
        "anomaly_quantile": 0.95,
        "required_votes": 1,
    },
    {
        "name": "very_sensitive",
        "anomaly_quantile": 0.90,
        "required_votes": 1,
    },
    {
        "name": "moderate_two_votes",
        "anomaly_quantile": 0.95,
        "required_votes": 2,
    },
    {
        "name": "sensitive_two_votes",
        "anomaly_quantile": 0.90,
        "required_votes": 2,
    },
    {
        "name": "very_sensitive_two_votes",
        "anomaly_quantile": 0.85,
        "required_votes": 2,
    },
]


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
        "[mdl05_rule_based] Calibrating rule-based detector"
    )
    print(
        f"[mdl05_rule_based] split_id={split_id}"
    )

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

    development_cap = min(
        500_000,
        len(train_df),
    )

    development_df = cap_training_dataframe(
        df=train_df,
        label_column=split_metadata["label_column"],
        cap=development_cap,
        seed=seed,
    )

    x_development, y_development = split_xy(
        df=development_df,
        label_column=split_metadata["label_column"],
        feature_columns=split_metadata["feature_columns"],
    )

    (
        x_calibration,
        x_validation,
        y_calibration,
        y_validation,
    ) = train_test_split(
        x_development,
        y_development,
        test_size=0.20,
        random_state=seed,
        stratify=y_development,
    )

    validation_results = []

    print()
    print("[mdl05_rule_based] Validation profiles")
    print("----------------------------------------")

    for profile in PROFILES:
        candidate = RuleBasedDetector(
            anomaly_quantile=profile[
                "anomaly_quantile"
            ],
            required_votes=profile[
                "required_votes"
            ],
        )

        candidate.fit(
            x_calibration,
            y_calibration,
        )

        y_pred = candidate.predict(x_validation)

        metrics = compute_metrics(
            y_validation,
            y_pred,
        )

        result = {
            "name": profile["name"],
            "anomaly_quantile": profile[
                "anomaly_quantile"
            ],
            "required_votes": profile[
                "required_votes"
            ],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics[
                "balanced_accuracy"
            ],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "confusion_matrix": metrics[
                "confusion_matrix"
            ],
        }

        validation_results.append(result)

        print(
            f"{profile['name']:<28} "
            f"q={profile['anomaly_quantile']:<5} "
            f"votes={profile['required_votes']} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"balanced={metrics['balanced_accuracy']:.4f}"
        )

    selected = max(
        validation_results,
        key=lambda result: (
            result["f1"],
            result["balanced_accuracy"],
            result["precision"],
        ),
    )

    print("----------------------------------------")
    print(
        "[mdl05_rule_based] selected profile="
        f"{selected['name']}"
    )
    print(
        "[mdl05_rule_based] selected validation F1="
        f"{selected['f1']:.4f}"
    )

    model = RuleBasedDetector(
        anomaly_quantile=selected[
            "anomaly_quantile"
        ],
        required_votes=selected[
            "required_votes"
        ],
    )

    model.fit(
        x_train,
        y_train,
    )

    label_counts = {
        str(label): int(count)
        for label, count in (
            y_train
            .value_counts()
            .sort_index()
            .items()
        )
    }

    artifact = {
        "model": model,
        "model_type": "rule_based_detector",
        "feature_columns": model.feature_columns,
        "split_id": split_id,
        "seed": seed,
        "train_row_cap": cap,
        "full_training_rows": int(
            full_training_rows
        ),
        "training_rows": int(len(y_train)),
        "training_label_counts": label_counts,
        "benign_calibration_rows": int(
            (y_train == 0).sum()
        ),
        "params": {
            "selected_profile": selected["name"],
            "selection_metric": "validation_f1",
            "anomaly_quantile": selected[
                "anomaly_quantile"
            ],
            "required_votes": selected[
                "required_votes"
            ],
            "development_rows": int(
                len(y_development)
            ),
            "validation_rows": int(
                len(y_validation)
            ),
            "validation_results": validation_results,
            "thresholds": model.thresholds,
        },
    }

    joblib.dump(
        artifact,
        model_path,
    )

    summary_path = (
        output_dir
        / "training_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                key: value
                for key, value in artifact.items()
                if key != "model"
            },
            file,
            indent=2,
        )

    print()
    print(
        "[mdl05_rule_based] available training rows="
        f"{full_training_rows}"
    )
    print(
        "[mdl05_rule_based] used training rows="
        f"{len(y_train)}"
    )
    print(
        "[mdl05_rule_based] benign calibration rows="
        f"{artifact['benign_calibration_rows']}"
    )
    print(
        "[mdl05_rule_based] final thresholds="
        f"{model.thresholds}"
    )
    print(
        "[mdl05_rule_based] saved model to: "
        f"{model_path}"
    )