"""Cluster unpaired TF-TG edges by their patterns across stages."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from draft_paired.analysis.edge_cluster import (
    run_edge_cluster_from_config as _run_edge_cluster_from_config,
)

from ..graph_builder import ordered_tree_table
from ..lineage import load_tree_table
from ..utils import load_yaml_config, resolve_config_path


EDGE_CLUSTER_DEFAULTS: dict[str, Any] = {
    "selection_mode": "all",
    "top_n_per_cluster": 10_000,
    "score_threshold": 0.90,
    "n_clusters": 12,
    "random_state": 5465,
    "save_fig": False,
    "top_tf_per_cluster": 20,
    "top_tg_per_cluster": 20,
    "heatmap_cmap": "OrRd",
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
    section = dict(_section(patched, "edge_cluster"))
    for key, value in EDGE_CLUSTER_DEFAULTS.items():
        section.setdefault(key, value)
    if section.get("celltypes") is None:
        labels = _default_cluster_labels(patched)
        if labels:
            section["celltypes"] = labels
    patched["edge_cluster"] = section
    return patched


def run_edge_cluster_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return _run_edge_cluster_from_config(_patched_config(config))


def run_edge_cluster(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        loaded = load_yaml_config(config)
        report = run_edge_cluster_from_config(loaded)
        report["config_path"] = str(Path(config).resolve())
        return report
    return run_edge_cluster_from_config(config)


__all__ = ["run_edge_cluster", "run_edge_cluster_from_config"]
