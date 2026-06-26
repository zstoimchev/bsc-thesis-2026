from pathlib import Path

import pandas as pd


def load_dataset(dataset_cfg: dict, project_root: Path) -> pd.DataFrame:
    dataset_path = project_root / dataset_cfg["path"]
    dataset_format = dataset_cfg.get("format", "").lower()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_format == "csv":
        return pd.read_csv(dataset_path)

    if dataset_format == "parquet":
        return pd.read_parquet(dataset_path)

    raise ValueError(f"Unsupported dataset format: {dataset_format}")