"""Utility helpers for logging, directories, and persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the project.

    Parameters
    ----------
    level:
        Logging level. Defaults to ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directories() -> None:
    """Ensure project output directories exist."""
    for directory in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save a dictionary as a prettified JSON file.

    Parameters
    ----------
    data:
        JSON-serializable dictionary.
    output_path:
        Path to output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
