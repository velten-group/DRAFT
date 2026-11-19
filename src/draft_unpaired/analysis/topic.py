"""Regulatory-topic analysis for unpaired TF-TG networks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from draft_paired.analysis.topic import (
    run_topic_analysis_from_config as _run_topic_analysis_from_config,
)

from ..graph_builder import ordered_tree_table
from ..lineage import load_tree_table
from ..utils import load_yaml_config, resolve_config_path


TOPIC_DEFAULTS: dict[str, Any] = {
    "selection_mode": "all",
    "top_n_per_cluster": 10_000,
    "score_threshold": 0.99,
    "n_topics": 10,
    "random_state": 42,
    "lda_max_iter": 200,
    "lda_learning_method": "batch",
    "save_fig": True,
    "heatmap_cmap": "OrRd",
    "do_enrich": False,
    "enrich_db": "GO_Biological_Process_2021",
    "enrich_organism": "Mouse",
    "enrich_background": "network_targets",
}


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _default_cluster_labels(config: dict[str, Any]) -> list[str] | None:
    data_cfg = _section(config, "data")
    tree_value = data_cfg.get("tree")
    if not tree_value:
        return None
    tree_df = load_tree_table(resolve_config_path(config, tree_value))
    return ordered_tree_table(tree_df)["node_id"].astype(str).tolist()


def _patched_config(config: dict[str, Any]) -> dict[str, Any]:
    patched = copy.deepcopy(config)
    section = dict(_section(patched, "topic"))
    for key, value in TOPIC_DEFAULTS.items():
        if int(config.get("config_version", 1)) >= 2 and key in {"enrich_db", "enrich_organism"}:
            value = None
        section.setdefault(key, value)
    if section.get("celltypes") is None:
        labels = _default_cluster_labels(patched)
        if labels:
            section["celltypes"] = labels
    patched["topic"] = section
    return patched


def run_topic_analysis_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return _run_topic_analysis_from_config(_patched_config(config))


def run_topic_analysis(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        loaded = load_yaml_config(config)
        report = run_topic_analysis_from_config(loaded)
        report["config_path"] = str(Path(config).resolve())
        return report
    return run_topic_analysis_from_config(config)


__all__ = ["run_topic_analysis", "run_topic_analysis_from_config"]
