import json
from pathlib import Path

import joblib

from src.common.preprocessing import iter_prepared_xy_chunks
from src.mdl01_baseline.model import MajorityClassBaseline


def train(
        dataset_cfg: dict,
        model_cfg: dict,
        output_dir: Path,
        model_path: Path,
        project_root: Path,
        feature_set: str,
        chunk_size: int,
        split: str,
        test_size: float,
        seed: int,
) -> None:
    if split != "random":
        raise NotImplementedError(
            f"Split strategy '{split}' is not implemented for mdl01_baseline."
        )

    model = MajorityClassBaseline()

    chunk_count = 0
    total_rows = 0

    print("[mdl01_baseline] Training majority-class baseline")
    print("[mdl01_baseline] Using split part: train")

    for x_chunk, y_chunk, feature_columns in iter_prepared_xy_chunks(
            dataset_cfg=dataset_cfg,
            project_root=project_root,
            feature_set=feature_set,
            chunk_size=chunk_size,
            split_part="train",
            test_size=test_size,
            seed=seed,
    ):
        chunk_count += 1
        total_rows += len(y_chunk)

        model.partial_fit(x_chunk, y_chunk)

        print(
            f"[mdl01_baseline] trained chunk {chunk_count}: "
            f"{len(y_chunk)} rows "
            f"(total: {total_rows})"
        )

    model.finalize()

    artifact = {
        "model": model,
        "model_type": "majority_class_baseline",
        "majority_class": model.majority_class,
        "label_counts": dict(model.label_counts),
        "feature_columns": model.feature_columns,
        "chunk_count": chunk_count,
        "total_rows": total_rows,
        "feature_set": feature_set,
        "split": split,
        "split_part": "train",
        "test_size": test_size,
        "seed": seed,
    }

    joblib.dump(artifact, model_path)

    training_summary = {
        "model_type": "majority_class_baseline",
        "majority_class": model.majority_class,
        "label_counts": dict(model.label_counts),
        "feature_columns": model.feature_columns,
        "chunk_count": chunk_count,
        "total_rows": total_rows,
        "feature_set": feature_set,
        "split": split,
        "split_part": "train",
        "test_size": test_size,
        "seed": seed,
    }

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    print(f"[mdl01_baseline] label_counts={dict(model.label_counts)}")
    print(f"[mdl01_baseline] majority_class={model.majority_class}")
    print(f"[mdl01_baseline] saved model to: {model_path}")
