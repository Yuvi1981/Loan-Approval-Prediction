"""
Configuration loader and validator for the loan-telemetry project.
"""

from pathlib import Path
from typing import Any

import yaml

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and return configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_path(path_key: str, config: dict[str, Any] = None) -> Path:
    """Resolve a configured relative path to an absolute Path object.

    Args:
        path_key: Key inside `paths` section of config.yaml (e.g. 'raw_data').
        config: Optional pre-loaded configuration dictionary.

    Returns:
        Path: Absolute resolved path.
    """
    if config is None:
        config = load_config()

    rel_path = config.get("paths", {}).get(path_key)
    if rel_path is None:
        raise KeyError(f"Path key '{path_key}' not found in configuration paths.")

    return PROJECT_ROOT / rel_path


# Global instance of configuration
CONFIG = load_config()
