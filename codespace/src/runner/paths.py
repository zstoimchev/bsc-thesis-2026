from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = Path(__file__).resolve().parent

DATASET_REGISTRY_PATH = RUNNER_DIR / "dataset_registry.json"
MODEL_REGISTRY_PATH = RUNNER_DIR / "model_registry.json"