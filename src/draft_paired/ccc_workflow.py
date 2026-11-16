"""Standalone communication workflow for two trained lineages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .analysis.ccc import run_ccc_analysis_from_config
from .utils import load_yaml_config


def run_draft_ccc_workflow(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        loaded = load_yaml_config(path)
        loaded.setdefault("base_dir", str(Path(".").resolve()))
        report = run_ccc_analysis_from_config(loaded)
        report["config_path"] = str(path)
        return report
    return run_ccc_analysis_from_config(config)
