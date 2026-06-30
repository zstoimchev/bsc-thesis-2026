import json
from pathlib import Path

from src.runner.paths import DATASET_REGISTRY_PATH, MODEL_REGISTRY_PATH, FEATURE_REGISTRY_PATH, SPLIT_REGISTRY_PATH


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_registries() -> tuple[dict, dict]:
    dataset_registry = load_json(DATASET_REGISTRY_PATH)
    model_registry = load_json(MODEL_REGISTRY_PATH)
    return dataset_registry, model_registry


def load_dataset_registry() -> dict:
    return load_json(DATASET_REGISTRY_PATH)


def load_model_registry() -> dict:
    return load_json(MODEL_REGISTRY_PATH)


def load_feature_registry() -> dict:
    return load_json(FEATURE_REGISTRY_PATH)


def load_split_registry() -> dict:
    return load_json(SPLIT_REGISTRY_PATH)
