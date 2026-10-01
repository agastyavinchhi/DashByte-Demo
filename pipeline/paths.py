"""Hand-off folder locations. Folders are the only contract between stages.

The ``*_dir`` helpers are pure: they return paths and never touch the disk.
Only ``ensure_data_dirs`` writes, and it only creates directories.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.config import PROJECT_ROOT, Config

RAW = "raw"
FEATURES = "features"
MODELS = "models"
PREDICTIONS = "predictions"
QUALITY = "quality"

DATA_DIR_NAMES = (RAW, FEATURES, MODELS, PREDICTIONS, QUALITY)


def raw_dir(cfg: Config) -> Path:
    return cfg.data_dir / RAW


def features_dir(cfg: Config) -> Path:
    return cfg.data_dir / FEATURES


def models_dir(cfg: Config) -> Path:
    return cfg.data_dir / MODELS


def predictions_dir(cfg: Config) -> Path:
    return cfg.data_dir / PREDICTIONS


def quality_dir(cfg: Config) -> Path:
    return cfg.data_dir / QUALITY


def display_path(path: Path) -> str:
    """Short form for logs: relative to the project folder when inside it."""
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_data_dirs(cfg: Config) -> dict[str, tuple[Path, bool]]:
    """Create every data dir if missing. Safe to run repeatedly.

    Returns ``{name: (path, created)}`` where ``created`` is True only if
    this call made the directory.
    """
    result: dict[str, tuple[Path, bool]] = {}
    for name in DATA_DIR_NAMES:
        path = cfg.data_dir / name
        existed = path.is_dir()
        path.mkdir(parents=True, exist_ok=True)
        result[name] = (path, not existed)
    return result
