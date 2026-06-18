from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def hydra_config_path(group: str) -> str:
    """Resolve Hydra configs as files, not as a Python config module."""
    candidates = []
    if config_root := os.environ.get("SLASSL_CONFIG_ROOT"):
        candidates.append(Path(config_root).expanduser() / group)
    candidates.extend(
        (
            Path(__file__).resolve().parents[2] / "configs" / group,
            Path.cwd() / "configs" / group,
        )
    )
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve())
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not locate Hydra config group '{group}'. Searched: {searched}. "
        "Set SLASSL_CONFIG_ROOT to the repository's configs directory."
    )


class Config(dict):
    """Small recursive mapping with attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({key: _to_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_config(item) for item in value]
    return value


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a mapping")
    current[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = deepcopy(raw)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        _set_nested(raw, key, yaml.safe_load(raw_value))
    return _to_config(raw)


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a nested config value; may be repeated",
    )
    return parser
