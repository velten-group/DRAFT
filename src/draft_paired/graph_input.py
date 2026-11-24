"""Load preprocessed PyG lineage bundles at the preprocessing/training boundary."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .utils import resolve_config_path


def has_graph_input(config: dict[str, Any]) -> bool:
    return bool(config.get("data", {}).get("graphs"))


def load_graph_artifacts(config: dict[str, Any], family: str) -> dict[str, Any]:
    path = resolve_config_path(config, config["data"]["graphs"])
    payload = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    required = {
        "graphs",
        "time_names",
        "topo_order",
        "parents_list",
        "in_channels",
        "maps",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"{path} must contain {sorted(required)}.")
    graphs = payload["graphs"]
    names = list(payload["time_names"])
    parents = [list(items) for items in payload["parents_list"]]
    width = int(payload["in_channels"])
    if "T_sel" in payload and int(payload["T_sel"]) != len(graphs):
        raise ValueError("T_sel must match the graph count.")
    if not graphs or len(names) != len(graphs) or len(parents) != len(graphs):
        raise ValueError(
            "Graphs, time_names and parents_list must have the same nonzero length."
        )
    if not all(isinstance(name, str) and name for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Stage names must be distinct nonempty strings.")
    if list(payload["topo_order"]) != list(range(len(graphs))):
        raise ValueError(
            "Store graphs in topological order with topo_order equal to their list indices."
        )
    for idx, items in enumerate(parents):
        if len(set(items)) != len(items) or any(
            not isinstance(p, int) or p < 0 or p >= idx for p in items
        ):
            raise ValueError(
                f"Parents of graph {idx} must be distinct earlier graph indices."
            )
    node_types = {"TF", "RE", "TG"} if family == "paired" else {"TF", "TG"}
    relations = (
        {
            ("TF", "combine", "RE"),
            ("RE", "combined_by", "TF"),
            ("RE", "regulate", "TG"),
            ("TG", "regulated_by", "RE"),
        }
        if family == "paired"
        else {("TF", "regulate", "TG"), ("TG", "regulated_by", "TF")}
    )
    maps = payload["maps"]
    if not {kind.lower() for kind in node_types}.issubset(maps):
        raise ValueError("The graph bundle lacks node ID mappings.")
    if any(not len(maps[kind.lower()]) for kind in node_types):
        raise ValueError("Each node role must have at least one ID.")
    if any(
        len(set(maps[kind.lower()])) != len(maps[kind.lower()]) for kind in node_types
    ):
        raise ValueError("Node IDs must be unique within each role.")
    total_edges = 0
    for idx, graph in enumerate(graphs):
        if not isinstance(graph, HeteroData) or set(graph.node_types) != node_types:
            raise ValueError(f"Graph {idx} does not match the {family} node types.")
        if set(graph.edge_types) != relations:
            raise ValueError(f"Graph {idx} does not match the {family} relation types.")
        graph.validate(raise_on_error=True)
        for kind in node_types:
            x = graph[kind].x
            if (
                x.ndim != 2
                or width <= 0
                or x.shape != (len(maps[kind.lower()]), width)
                or not x.is_floating_point()
            ):
                raise ValueError(
                    f"Graph {idx}, {kind}: invalid feature shape or dtype."
                )
            for start in range(0, x.shape[0], 2048):
                if not torch.isfinite(x[start : start + 2048]).all():
                    raise ValueError(f"Graph {idx}, {kind}: nonfinite features.")
        total_edges += sum(int(graph[edge].edge_index.shape[1]) for edge in relations)
    if total_edges == 0:
        raise ValueError("Training needs at least one prior edge.")
    requested = config.get("training", {}).get("device", "auto")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "auto"
        else torch.device(requested)
    )
    graphs = [graph.to(device) for graph in graphs]
    ordered_tree = pd.DataFrame(
        {
            "node_id": names,
            "parent_id": ["|".join(names[p] for p in items) for items in parents],
            "stage_id": range(len(names)),
        }
    )
    children = [0] * len(graphs)
    for items in parents:
        for parent in items:
            children[parent] += 1
    return {
        "mode": "tree",
        "graphs_lineage": graphs,
        "time_names": names,
        "topo_order": list(range(len(graphs))),
        "parents_list": parents,
        "in_channels": width,
        "maps": maps,
        "device": device,
        "ordered_tree": ordered_tree,
        "source_graph_path": str(path),
        "model_kind": "multi_parent",
        "source_model_kind": payload.get("model_kind"),
        "topology_kind": "branching"
        if any(n > 1 for n in children) or any(len(p) > 1 for p in parents)
        else "linear",
        "num_nodes": {kind: len(maps[kind.lower()]) for kind in node_types}
        if family == "paired"
        else len(graphs),
        "summary_lines": [
            f"Graph bundle: {path}",
            f"Stages: {', '.join(names)}",
            f"Input width: {width}",
            f"Device: {device}",
        ],
    }
