"""Config loader that merges a model-specific YAML on top of base.yaml."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key == "defaults":
            continue
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(model_name: str | None = None, config_dir: str | Path = "configs") -> dict[str, Any]:
    """
    Load base.yaml and optionally merge a model-specific YAML on top.

    Parameters
    ----------
    model_name : str | None
        One of 'efficientnetb0', 'mobilenetv2', 'resnet50', or None for base only.
    config_dir : str | Path
        Directory containing YAML files.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    config_dir = Path(config_dir)
    base_path = config_dir / "base.yaml"

    with base_path.open() as f:
        cfg = yaml.safe_load(f)

    if model_name is not None:
        model_path = config_dir / f"{model_name}.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"No config found for model '{model_name}' at {model_path}")
        with model_path.open() as f:
            model_cfg = yaml.safe_load(f)
        cfg = _deep_merge(cfg, model_cfg)

    return cfg
