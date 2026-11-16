"""Explicit preprocessing and a shared entry point for graph-based training."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
from uuid import uuid4

from .utils import load_yaml_config, resolve_config_path


def graph_input_mode(path: str | Path) -> str:
    """Identify a prepared input from its node roles, checking its mode metadata."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or not payload.get("graphs"):
        raise ValueError("Expected prepared stage graphs, not a model checkpoint.")
    nodes = set(payload["graphs"][0].node_types)
    if nodes == {"TF", "RE", "TG"}:
        mode = "paired"
    elif nodes == {"TF", "TG"}:
        mode = "unpaired"
    else:
        raise ValueError(f"Unsupported graph node types: {sorted(nodes)}")
    if payload.get("input_mode", mode) != mode:
        raise ValueError("Graph input_mode does not match its node types.")
    return mode


def _graph_payload(artifacts: dict[str, Any], config, family):
    mode = artifacts["mode"]
    if mode == "tree":
        graphs = artifacts["graphs_lineage"]
        names = artifacts["time_names"]
        parents = artifacts["parents_list"]
        order = artifacts["topo_order"]
        width = artifacts["in_channels"]
    elif mode == "germ":
        graphs = artifacts["graphs_germline"]
        names = artifacts["germline_types"]
        parents = [[] if p < 0 else [p] for p in artifacts["parent"]]
        order = artifacts["topo_order"]
        width = artifacts["in_channels"]
    elif mode == "soma":
        graphs = artifacts["graphs_somatic"]
        names = artifacts["somatic_names"]
        parents = artifacts["parents_somatic"]
        order = artifacts["topo_somatic"]
        width = artifacts["in_channels_somatic"]
    else:
        raise ValueError(f"Unsupported graph-construction mode: {mode}")
    return {
        "graphs": [graph.cpu() for graph in graphs],
        "time_names": list(names),
        "topo_order": list(order),
        "parents_list": [list(items) for items in parents],
        "in_channels": int(width),
        "T_sel": len(graphs),
        "maps": artifacts["maps"],
        "lineage": config.get("lineage", "lineage1"),
        "input_mode": family,
        "model_kind": artifacts.get("model_kind", "multi_parent"),
    }


def preprocess_lineage(
    config_path: str | Path,
    family: str,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build CPU stage graphs and write a config that trains without raw inputs."""
    import torch
    import yaml

    from .graph_input import load_graph_artifacts

    if family not in {"paired", "unpaired"}:
        raise ValueError(f"Unknown input mode: {family}")
    config_path = Path(config_path).expanduser().resolve()
    config = load_yaml_config(config_path)
    if config.get("data", {}).get("graphs"):
        raise ValueError("This config already has prepared graphs; use draft train.")
    out = Path(output_dir).expanduser().resolve()
    graph_path, train_path = out / "graphs.pt", out / "train.yaml"
    if train_path == config_path:
        raise ValueError("Preprocessing must not overwrite its input config.")
    for target in (graph_path, train_path):
        if target.exists() and not force:
            raise FileExistsError(
                f"{target} already exists; choose another --output or use --force."
            )

    settings = config.get("prior_preparation", {})
    if settings.get("enabled", False) and settings.get("method") == "auto_motif":
        from .prior_preparation import prepare_prior

        prepared = prepare_prior(config, out / "prior", mode=family, force=force)
        config = load_yaml_config(prepared["config_path"])
    if family == "unpaired":
        from draft_unpaired.graph_builder import build_lineage_graph_artifacts

        if config.get("prior_preparation", {}).get("enabled", False):
            from draft_unpaired.prior import prepare_stage_prior

            prepared = prepare_stage_prior(config, out / "prior", force=force)
            config = load_yaml_config(prepared["config_path"])
    else:
        from .graph_builder import build_lineage_graph_artifacts

    artifacts = build_lineage_graph_artifacts(config, device=torch.device("cpu"))
    payload = _graph_payload(artifacts, config, family)
    train_config = copy.deepcopy(config)
    train_config["base_dir"] = str(
        Path(config.get("base_dir", ".")).expanduser().resolve()
    )
    train_config["input_mode"] = family
    train_config.setdefault("data", {})["graphs"] = str(graph_path)
    # Training reads the saved graphs.
    train_config["prior_preparation"] = {"enabled": False}
    out.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = out / f".{token}.pt"
    temporary_config = out / f".{token}.yaml"
    try:
        torch.save(payload, temporary)
        validation = {"data": {"graphs": str(temporary)}, "training": {"device": "cpu"}}
        validated = load_graph_artifacts(validation, family)
        del validated
        temporary_config.write_text(
            yaml.safe_dump(train_config, sort_keys=False), encoding="utf-8"
        )
        temporary.replace(graph_path)
        temporary_config.replace(train_path)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_config.unlink(missing_ok=True)
    return {
        "input_mode": family,
        "graph_path": str(graph_path),
        "config_path": str(train_path),
        "stages": len(payload["graphs"]),
        "status": "preprocessing_completed",
    }


def train_graph_model(config_path: str | Path) -> dict[str, Any]:
    """Select the paired or unpaired trainer for the prepared graphs."""
    config = load_yaml_config(config_path)
    graphs = config.get("data", {}).get("graphs")
    if not graphs:
        raise ValueError(
            "draft train requires data.graphs; run paired or unpaired preprocess first."
        )
    if config.get("prior_preparation", {}).get("enabled", False):
        raise ValueError(
            "Prior preparation belongs in preprocess; use its generated train.yaml."
        )
    mode = graph_input_mode(resolve_config_path(config, graphs))
    if mode == "paired":
        from .train import train_lineage_model
    else:
        from draft_unpaired.train import train_lineage_model
    return train_lineage_model(config_path)
