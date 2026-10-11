"""Configuration helpers for experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    seed: int
    seeds: list[int]
    n_splits: int
    data: dict[str, Any]
    fairness: dict[str, Any]
    metrics: dict[str, Any]
    methods: list[str]
    outputs: dict[str, str]
    optimization: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str | Path) -> ExperimentConfig:
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    return ExperimentConfig(**raw)

