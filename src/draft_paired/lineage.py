"""Lineage-tree parsing and graph ordering."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .input_schema import read_table


def tail_num(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    return int(match.group(1)) if match else 1 << 30


def parse_stage_branch(name: str) -> tuple[int, str]:
    text = str(name)
    match_num = re.search(r"(\d+)", text)
    stage = int(match_num.group(1)) if match_num else 10**9
    match_branch = re.search(r"(?:[_\-])([A-Za-z]+)$", text)
    branch = match_branch.group(1).lower() if match_branch else ""
    return stage, branch


def select_lineage_with_branching(
    graphs_dict: dict[str, Any],
    prefix_keys: tuple[str, ...] = ("somatic", "soma"),
) -> tuple[list[Any], list[int], list[list[int]], list[str]]:
    items = [
        (cell_type, graphs_dict[cell_type])
        for cell_type in graphs_dict
        if any(cell_type.lower().startswith(prefix) for prefix in prefix_keys)
        or any(prefix in cell_type.lower() for prefix in prefix_keys)
    ]
    if not items:
        return [], [], [], []
    items.sort(key=lambda pair: parse_stage_branch(pair[0]))
    names = [cell_type for cell_type, _ in items]
    graph_list = [graph for _, graph in items]
    stages = [parse_stage_branch(name)[0] for name in names]
    idx_by_stage: dict[int, list[int]] = {}
    for idx, stage in enumerate(stages):
        idx_by_stage.setdefault(stage, []).append(idx)
    topo = list(range(len(graph_list)))
    parents = [[] for _ in graph_list]
    sorted_stages = sorted(idx_by_stage.keys())
    for stage_idx, stage in enumerate(sorted_stages):
        if stage_idx == 0:
            continue
        prev_indices = idx_by_stage[sorted_stages[stage_idx - 1]]
        for idx in idx_by_stage[stage]:
            parents[idx] = list(prev_indices)
    return graph_list, topo, parents, names


def _pick_tree_col(columns: list[str], candidates: list[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def load_tree_table(path: str | Path, *, strict: bool = False, sort: bool = True) -> pd.DataFrame:
    tree_path = Path(path)
    frame = read_table(tree_path).fillna("")
    columns = frame.columns.tolist()

    node_col = _pick_tree_col(columns, ["node_id", "node", "cell_type", "name", "id"])
    parent_col = _pick_tree_col(columns, ["parent_id", "parent", "parents", "parent_node"])
    stage_col = _pick_tree_col(columns, ["stage_id", "stage", "time", "time_idx", "timepoint"])
    branch_col = _pick_tree_col(columns, ["branch", "branch_id"])

    if node_col is None:
        raise ValueError(f"{tree_path} must contain one of node_id/node/cell_type/name/id.")
    if strict and (parent_col is None or stage_col is None):
        raise ValueError(f"{tree_path} requires node_id, parent_id, stage_id columns.")

    output = pd.DataFrame()
    output["node_id"] = frame[node_col].astype(str).str.strip()
    output["parent_id"] = (
        frame[parent_col].astype(str).str.strip() if parent_col is not None else ""
    )

    if stage_col is not None:
        stage_values = pd.to_numeric(frame[stage_col], errors="coerce")
        if strict and (stage_values.isna().any() or (stage_values % 1 != 0).any()):
            raise ValueError(f"{tree_path} stage_id values must be finite integers.")
        output["stage_id"] = [
            int(value) if pd.notna(value) else idx for idx, value in enumerate(stage_values)
        ]
    else:
        output["stage_id"] = list(range(len(frame)))

    if branch_col is not None:
        output["branch"] = frame[branch_col].astype(str).str.strip()
    else:
        output["branch"] = ""

    if strict:
        if output.empty or output.node_id.eq("").any() or output.node_id.duplicated().any():
            raise ValueError(f"{tree_path} requires distinct nonempty node_id values.")
        stages = dict(zip(output.node_id, output.stage_id))
        for node, raw in zip(output.node_id, output.parent_id):
            for parent in re.split(r"[|;,]+", raw):
                parent = parent.strip()
                if not parent:
                    continue
                if parent not in stages:
                    raise ValueError(f"{tree_path}: unknown parent {parent!r} for {node!r}.")
                if stages[parent] >= stages[node]:
                    raise ValueError(f"{tree_path}: parent {parent!r} must precede {node!r} in stage_id.")
    output = output[output["node_id"].astype(str).str.len() > 0].copy()
    output = output.drop_duplicates(subset=["node_id"], keep="first")
    if sort:
        output = output.sort_values(["stage_id", "node_id"], kind="mergesort")
    output = output.reset_index(drop=True)
    return output


def select_lineage_from_tree(
    graphs_dict: dict[str, Any],
    tree_path: str | Path,
    *, strict: bool = False,
) -> tuple[list[Any], list[int], list[list[int]], list[str], pd.DataFrame]:
    tree = load_tree_table(tree_path, strict=strict)
    node_names = tree["node_id"].astype(str).tolist()
    missing = [name for name in node_names if name not in graphs_dict]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Tree nodes are missing from annotation-derived graphs: {preview}")

    graph_list = [graphs_dict[name] for name in node_names]
    topo = list(range(len(node_names)))
    idx_by_name = {name: idx for idx, name in enumerate(node_names)}
    parents: list[list[int]] = [[] for _ in node_names]

    for idx, parent_value in enumerate(tree["parent_id"].astype(str).tolist()):
        raw = parent_value.strip()
        if raw == "":
            continue
        parent_names = [item.strip() for item in re.split(r"[|;,]+", raw) if item.strip()]
        missing_parents = [name for name in parent_names if name not in idx_by_name]
        if missing_parents:
            preview = ", ".join(missing_parents[:10])
            raise ValueError(f"Tree parent nodes are missing from tree order: {preview}")
        parents[idx] = [idx_by_name[name] for name in parent_names]

    return graph_list, topo, parents, node_names, tree
