from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from common.paths import REGISTRIES_DIR, resolve_path


RegistryRecord = dict[str, Any]


class RegistryError(Exception):
    """Raised when a registry file is missing, malformed, or inconsistent."""


def _read_registry_file(path: str | Path) -> Any:
    """
    Read a registry file.

    Supported formats:
    - .yaml / .yml
    - .json
    """
    p = resolve_path(path)

    if not p.exists():
        raise RegistryError(f"Registry file does not exist: {p}")

    if not p.is_file():
        raise RegistryError(f"Registry path is not a file: {p}")

    suffix = p.suffix.lower()

    try:
        if suffix in {".yaml", ".yml"}:
            with p.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        if suffix == ".json":
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)

    except Exception as exc:
        raise RegistryError(f"Failed to parse registry file {p}: {exc}") from exc

    raise RegistryError(f"Unsupported registry format for file: {p}")


def _validate_records(records: Any, source: Path) -> list[RegistryRecord]:
    """
    Validate that registry content is a list of dictionaries with unique ids.
    """
    if records is None:
        raise RegistryError(f"Registry is empty: {source}")

    if not isinstance(records, list):
        raise RegistryError(
            f"Registry content must be a list of records in {source}, "
            f"got {type(records).__name__}"
        )

    seen_ids: set[str] = set()
    validated: list[RegistryRecord] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RegistryError(
                f"Registry record #{index} in {source} must be a dictionary, "
                f"got {type(record).__name__}"
            )

        record_id = record.get("id")

        if not record_id or not isinstance(record_id, str):
            raise RegistryError(
                f"Registry record #{index} in {source} is missing string field 'id'"
            )

        if record_id in seen_ids:
            raise RegistryError(f"Duplicate registry id '{record_id}' in {source}")

        seen_ids.add(record_id)
        validated.append(record)

    return validated


def load_registry(path: str | Path, root_key: str | None = None) -> list[RegistryRecord]:
    """
    Load a registry file.

    Two styles are supported.

    Style A:
        models:
          - id: model_a
            ...

    Style B:
        - id: model_a
          ...

    If root_key is provided, the function expects Style A.
    If root_key is omitted, the function accepts either Style A with one top-level key
    or Style B.
    """
    p = resolve_path(path)
    raw = _read_registry_file(p)

    if root_key is not None:
        if not isinstance(raw, dict):
            raise RegistryError(
                f"Registry {p} must be a dictionary with top-level key '{root_key}'"
            )

        if root_key not in raw:
            raise RegistryError(f"Registry {p} is missing top-level key '{root_key}'")

        records = raw[root_key]
        return _validate_records(records, p)

    if isinstance(raw, list):
        return _validate_records(raw, p)

    if isinstance(raw, dict):
        if len(raw) != 1:
            raise RegistryError(
                f"Registry {p} has multiple top-level keys. "
                f"Pass root_key explicitly."
            )

        only_key = next(iter(raw.keys()))
        return _validate_records(raw[only_key], p)

    raise RegistryError(
        f"Registry {p} must be either a list or dictionary, got {type(raw).__name__}"
    )


def get_record(records: list[RegistryRecord], record_id: str) -> RegistryRecord:
    """
    Find one record by id.
    """
    for record in records:
        if record.get("id") == record_id:
            return record

    available = ", ".join(sorted(str(r.get("id")) for r in records))
    raise RegistryError(f"Unknown registry id '{record_id}'. Available: {available}")


def load_models() -> list[RegistryRecord]:
    return load_registry(REGISTRIES_DIR / "models.yaml", root_key="models")


def load_datasets() -> list[RegistryRecord]:
    return load_registry(REGISTRIES_DIR / "datasets.yaml", root_key="datasets")


def load_splits() -> list[RegistryRecord]:
    split_file = REGISTRIES_DIR / "splits.yaml"

    if not split_file.exists():
        return []

    return load_registry(split_file, root_key="splits")


def get_model(model_id: str) -> RegistryRecord:
    return get_record(load_models(), model_id)


def get_dataset(dataset_id: str) -> RegistryRecord:
    return get_record(load_datasets(), dataset_id)


def get_split(split_id: str) -> RegistryRecord:
    return get_record(load_splits(), split_id)


def list_ids(records: list[RegistryRecord]) -> list[str]:
    return sorted(str(record["id"]) for record in records)


def print_registry_summary() -> None:
    """
    Small helper used by orchestrate.py inspect.
    """
    models = load_models()
    datasets = load_datasets()
    splits = load_splits()

    print(f"Models:   {len(models)}")
    for model_id in list_ids(models):
        print(f"  - {model_id}")

    print(f"\nDatasets: {len(datasets)}")
    for dataset_id in list_ids(datasets):
        print(f"  - {dataset_id}")

    print(f"\nSplits:   {len(splits)}")
    for split_id in list_ids(splits):
        print(f"  - {split_id}")