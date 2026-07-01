from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = Path(__file__).resolve().parent

REGISTRIES_DIR = RUNNER_DIR / "registries"

DATASET_REGISTRY_PATH = REGISTRIES_DIR / "datasets.json"
MODEL_REGISTRY_PATH = REGISTRIES_DIR / "models.json"
FEATURE_REGISTRY_PATH = REGISTRIES_DIR / "features.json"
SPLIT_REGISTRY_PATH = REGISTRIES_DIR / "splits.json"
