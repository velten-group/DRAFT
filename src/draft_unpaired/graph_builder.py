"""Graph construction for draft_unpaired."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .preprocess import CommonInputs, prepare_common_inputs


def zscore_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True) + eps
    return (values - mean) / std


def pad_to(values: np.ndarray, target_cols: int) -> np.ndarray:
    if values.shape[1] >= target_cols:
        return values
    pad = np.zeros((values.shape[0], target_cols - values.shape[1]), dtype=values.dtype)
    return np.concatenate([values, pad], axis=1)


def _split_parent_tokens(raw_value: str) -> list[str]:
    return [
        item.strip()
        for item in str(raw_value).replace(";", "|").replace(",", "|").split("|")
        if item.strip()
    ]


def ordered_tree_table(tree_df: pd.DataFrame) -> pd.DataFrame:
    base = tree_df.copy()
    base["node_id"] = base["node_id"].astype(str).str.strip()
    base["parent_id"] = base["parent_id"].astype(str).str.strip()
    base["stage_id"] = pd.to_numeric(base["stage_id"], errors="coerce").fillna(0).astype(int)
    if "branch" not in base.columns:
        base["branch"] = ""
    base["branch"] = base["branch"].astype(str).str.strip()

    names = base["node_id"].astype(str).tolist()
    row_by_name = {str(row.node_id): row for row in base.itertuples(index=False)}
    parents_by_name: dict[str, list[str]] = {name: [] for name in names}
    children_by_name: dict[str, list[str]] = {name: [] for name in names}

    for row in base.itertuples(index=False):
        node_name = str(row.node_id)
        parent_names = [item for item in _split_parent_tokens(str(row.parent_id)) if item in row_by_name]
        parents_by_name[node_name] = parent_names
        for parent_name in parent_names:
            children_by_name.setdefault(parent_name, []).append(node_name)

    roots = [name for name in names if len(parents_by_name.get(name, [])) == 0]
    if not roots and names:
        roots = [names[0]]

    subtree_sizes: dict[str, int] = {}

    def _subtree_size(node_name: str) -> int:
        cached = subtree_sizes.get(node_name)
        if cached is not None:
            return cached
        total = 1
        for child_name in children_by_name.get(node_name, []):
            total += _subtree_size(child_name)
        subtree_sizes[node_name] = total
        return total

    def _node_sort_key(node_name: str) -> tuple[int, int, str]:
        row = row_by_name[node_name]
        return (int(getattr(row, "stage_id", 0)), -_subtree_size(node_name), str(node_name))

    def _child_sort_key(node_name: str) -> tuple[int, int, str]:
        row = row_by_name[node_name]
        return (-_subtree_size(node_name), int(getattr(row, "stage_id", 0)), str(node_name))

    ordered_names: list[str] = []
    visited: set[str] = set()

    def _dfs(node_name: str) -> None:
        if node_name in visited:
            return
        visited.add(node_name)
        ordered_names.append(node_name)
        for child_name in sorted(children_by_name.get(node_name, []), key=_child_sort_key):
            _dfs(child_name)

    for root_name in sorted(roots, key=_node_sort_key):
        _dfs(root_name)

    for node_name in names:
        if node_name not in visited:
            _dfs(node_name)

    return base.set_index("node_id").loc[ordered_names].reset_index()


def _tree_parents(tree_df: pd.DataFrame) -> tuple[list[int], list[list[int]], list[str]]:
    ordered = ordered_tree_table(tree_df)
    names = ordered["node_id"].astype(str).tolist()
    index_by_name = {name: idx for idx, name in enumerate(names)}
    parents: list[list[int]] = [[] for _ in names]
    for idx, raw_value in enumerate(ordered["parent_id"].astype(str).tolist()):
        parents[idx] = [index_by_name[item] for item in _split_parent_tokens(raw_value) if item in index_by_name]
    topo = list(range(len(names)))
    return topo, parents, names


def _node_graphs(common: CommonInputs, ordered_tree: pd.DataFrame) -> tuple[list[HeteroData], int]:
    tf_list = common.tf_list
    tg_list = common.tg_list
    expr_df = common.expr_df
    anno = common.anno
    prior_df = common.prior_df
    node_names = ordered_tree["node_id"].astype(str).tolist()
    max_cells = max(int(len(subanno)) for _, subanno in anno.groupby("cell_type"))
    tf_index = {str(name): idx for idx, name in enumerate(tf_list)}
    tg_index = {str(name): idx for idx, name in enumerate(tg_list)}

    graphs: list[HeteroData] = []
    for node_id in node_names:
        subanno = anno[anno["cell_type"].astype(str) == str(node_id)].copy()
        cells = subanno["cell_id"].astype(str).tolist()
        edges_frame = prior_df[prior_df["node_id"].astype(str) == str(node_id)]
        explicit_tf_scope = [name for name in common.node_tf_scope.get(str(node_id), []) if name in tf_index]
        explicit_tg_scope = [name for name in common.node_tg_scope.get(str(node_id), []) if name in tg_index]
        node_tf_names = (
            set(explicit_tf_scope)
            if explicit_tf_scope
            else {
                str(value)
                for value in edges_frame["TF"].astype(str).tolist()
                if str(value) in tf_index
            }
        )
        node_tg_names = (
            set(explicit_tg_scope)
            if explicit_tg_scope
            else {
                str(value)
                for value in edges_frame["TG"].astype(str).tolist()
                if str(value) in tg_index
            }
        )

        # Features outside the node-local gene scope are zeroed.
        # Without explicit scope tables, genes are selected from that node's prior.
        tf_expr = np.zeros((len(tf_list), len(cells)), dtype=np.float32)
        tg_expr = np.zeros((len(tg_list), len(cells)), dtype=np.float32)
        if node_tf_names:
            tf_rows = [tf_index[name] for name in tf_list if name in node_tf_names]
            tf_values = expr_df.loc[[tf_list[idx] for idx in tf_rows], cells].fillna(0.0).to_numpy(dtype=np.float32)
            tf_expr[tf_rows, :] = tf_values
        if node_tg_names:
            tg_rows = [tg_index[name] for name in tg_list if name in node_tg_names]
            tg_values = expr_df.loc[[tg_list[idx] for idx in tg_rows], cells].fillna(0.0).to_numpy(dtype=np.float32)
            tg_expr[tg_rows, :] = tg_values
        tf_expr = pad_to(zscore_rows(np.log1p(tf_expr)), max_cells)
        tg_expr = pad_to(zscore_rows(np.log1p(tg_expr)), max_cells)

        tf_idx = np.array([tf_index[value] for value in edges_frame["TF"].astype(str).tolist()], dtype=np.int64)
        tg_idx = np.array([tg_index[value] for value in edges_frame["TG"].astype(str).tolist()], dtype=np.int64)
        edge_index = (
            np.stack([tf_idx, tg_idx], axis=0)
            if len(tf_idx) > 0
            else np.zeros((2, 0), dtype=np.int64)
        )

        graph = HeteroData()
        graph["TF"].x = torch.tensor(tf_expr, dtype=torch.float32, device=common.device)
        graph["TG"].x = torch.tensor(tg_expr, dtype=torch.float32, device=common.device)
        graph["TF", "regulate", "TG"].edge_index = torch.tensor(edge_index, dtype=torch.long, device=common.device)
        graph["TG", "regulated_by", "TF"].edge_index = torch.tensor(edge_index[::-1].copy(), dtype=torch.long, device=common.device)
        graphs.append(graph)

    return graphs, max_cells


def build_lineage_graph_artifacts(
    config: dict[str, Any], *, device: torch.device | None = None,
) -> dict[str, Any]:
    from draft_paired.graph_input import has_graph_input, load_graph_artifacts

    if has_graph_input(config):
        return load_graph_artifacts(config, "unpaired")
    common = prepare_common_inputs(config, device=device)
    ordered_tree = ordered_tree_table(common.tree_df)
    graphs, max_cells = _node_graphs(common, ordered_tree)
    topo_order, parents_list, time_names = _tree_parents(ordered_tree)
    summary_lines = list(common.summary_lines) + [
        f"Graphs                  : {len(graphs)}",
        f"Max cells per node      : {max_cells}",
        (
            "Node feature scope      : explicit node TF/TG scope tables"
            if common.node_tf_scope or common.node_tg_scope
            else "Node feature scope      : prior-derived TF/TG rows kept, others zeroed"
        ),
        f"Ordered nodes           : {', '.join(time_names)}",
    ]
    return {
        "mode": "tree",
        "common": common,
        "ordered_tree": ordered_tree,
        "graphs_lineage": graphs,
        "topo_order": topo_order,
        "parents_list": parents_list,
        "time_names": time_names,
        "maps": common.maps,
        "in_channels": int(max_cells),
        "num_nodes": len(graphs),
        "device": str(common.device),
        "summary_lines": summary_lines,
    }


__all__ = ["build_lineage_graph_artifacts", "ordered_tree_table", "pad_to", "zscore_rows"]
