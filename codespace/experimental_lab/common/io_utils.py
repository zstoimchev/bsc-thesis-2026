from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.paths import resolve_path


def utc_now_iso() -> str:
    """
    Return current UTC time in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_json_safe(value: Any) -> Any:
    """
    Convert common non-JSON-safe Python / NumPy values into JSON-safe values.

    This keeps metrics.json writing stable even when values come from pandas,
    numpy, sklearn, etc.
    """
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

    except Exception:
        pass

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]

    return value


def read_json(path: str | Path) -> dict[str, Any]:
    """
    Read a JSON file as dictionary.
    """
    p = resolve_path(path)

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {p}, got {type(data).__name__}")

    return data


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    """
    Write dictionary as pretty JSON.
    """
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    safe_data = make_json_safe(data)

    with p.open("w", encoding="utf-8") as f:
        json.dump(safe_data, f, indent=2, sort_keys=True)

    return p


def write_text(path: str | Path, text: str) -> Path:
    """
    Write plain text file.
    """
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def read_text(path: str | Path) -> str:
    """
    Read plain text file.
    """
    p = resolve_path(path)
    return p.read_text(encoding="utf-8")


def append_csv_row(path: str | Path, row: dict[str, Any]) -> Path:
    """
    Append one dictionary row to CSV.

    If the file does not exist, the header is created automatically.
    If the file exists, its original column order is reused.
    """
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    safe_row = make_json_safe(row)

    file_exists = p.exists()

    if file_exists:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = list(safe_row.keys())
    else:
        header = list(safe_row.keys())

    # Keep old columns, append new columns at the end if needed.
    for key in safe_row.keys():
        if key not in header:
            header.append(key)

    rows_to_rewrite: list[dict[str, Any]] = []

    if file_exists:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows_to_rewrite = list(reader)

    rows_to_rewrite.append({key: safe_row.get(key, "") for key in header})

    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows_to_rewrite)

    return p


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested dictionaries for CSV summaries.

    Example:
        {"a": {"b": 1}} -> {"a.b": 1}
    """
    flat: dict[str, Any] = {}

    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key))
        else:
            flat[new_key] = value

    return flat