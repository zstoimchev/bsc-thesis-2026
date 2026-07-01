from collections.abc import Iterator
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


def iterate_dataset(
        dataset_cfg: dict,
        project_root: Path,
        chunk_size: int = 250_000,
) -> Iterator[pd.DataFrame]:
    dataset_path = project_root / dataset_cfg["path"]
    dataset_format = dataset_cfg.get("format", "").lower()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_format == "csv":
        yield from pd.read_csv(dataset_path, chunksize=chunk_size)
        return

    if dataset_format == "parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(dataset_path)

        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()

        return

    raise ValueError(f"Unsupported dataset format: {dataset_format}")
