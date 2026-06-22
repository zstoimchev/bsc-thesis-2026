from __future__ import annotations

from pathlib import Path


# experimental_lab/
ROOT_DIR = Path(__file__).resolve().parents[1]

# Main project folders
DATA_DIR = ROOT_DIR / "data"
EXTERNAL_DIR = ROOT_DIR / "external"
RUNNERS_DIR = ROOT_DIR / "runners"
REGISTRIES_DIR = ROOT_DIR / "registries"
RESULTS_DIR = ROOT_DIR / "results"

# Results subfolders
RUNS_DIR = RESULTS_DIR / "runs"
LOGS_DIR = RESULTS_DIR / "logs"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
SPLITS_DIR = RESULTS_DIR / "splits"


def resolve_path(path: str | Path) -> Path:
    """
    Resolve a path relative to experimental_lab root.

    Absolute paths are returned unchanged.
    Relative paths are interpreted as relative to ROOT_DIR.
    """
    p = Path(path)

    if p.is_absolute():
        return p

    return (ROOT_DIR / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist and return it.
    """
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_file(path: str | Path) -> Path:
    """
    Resolve a path and fail clearly if it is not an existing file.
    """
    p = resolve_path(path)

    if not p.exists():
        raise FileNotFoundError(f"Required file does not exist: {p}")

    if not p.is_file():
        raise FileNotFoundError(f"Expected a file but got directory: {p}")

    return p


def require_dir(path: str | Path) -> Path:
    """
    Resolve a path and fail clearly if it is not an existing directory.
    """
    p = resolve_path(path)

    if not p.exists():
        raise FileNotFoundError(f"Required directory does not exist: {p}")

    if not p.is_dir():
        raise FileNotFoundError(f"Expected a directory but got file: {p}")

    return p


def ensure_project_dirs() -> None:
    """
    Create the expected experimental_lab folder structure.
    Safe to call every time orchestrate.py starts.
    """
    for directory in [
        DATA_DIR,
        EXTERNAL_DIR,
        RUNNERS_DIR,
        REGISTRIES_DIR,
        RESULTS_DIR,
        RUNS_DIR,
        LOGS_DIR,
        ARTIFACTS_DIR,
        SPLITS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)