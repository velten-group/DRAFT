"""Downstream analysis modules.

Lazy imports keep optional analysis dependencies separate from training,
embedding export and network export.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "build_fbgn_to_name": (".single_tf", "build_fbgn_to_name"),
    "build_germ_labels": (".repro", "build_germ_labels"),
    "build_soma_labels": (".repro", "build_soma_labels"),
    "centrality_from_prob": (".single_tf", "centrality_from_prob"),
    "export_lineage_networks": (".network", "export_lineage_networks"),
    "f1_at_threshold_over_time": (".repro", "f1_at_threshold_over_time"),
    "infer_ec_specificity_topmix": (".edge_cluster", "infer_ec_specificity_topmix"),
    "infer_topic_celltype_pref": (".topic", "infer_topic_celltype_pref"),
    "plot_smoothness_heatmap": (".repro", "plot_smoothness_heatmap"),
    "run_ccc_analysis": (".ccc", "run_ccc_analysis"),
    "run_continuous_activity": (".continuous_activity", "run_continuous_activity"),
    "run_edge_cluster": (".edge_cluster", "run_edge_cluster"),
    "run_single_tf_analysis": (".single_tf", "run_single_tf_analysis"),
    "run_topic_analysis": (".topic", "run_topic_analysis"),
    "save_embeddings_by_label": (".repro", "save_embeddings_by_label"),
    "summarize_smoothness": (".repro", "summarize_smoothness"),
    "tf_tg_prob_frobenius_dist_matrix": (".repro", "tf_tg_prob_frobenius_dist_matrix"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
