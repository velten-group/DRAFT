"""Inference helpers for unpaired TF-TG lineage checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .checkpoint import find_latest_checkpoint, torch_load_compat
from .models.gnn import build_model
from .utils import resolve_config_path


def _select_runtime_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _artifact_filename(config: dict[str, Any]) -> str:
    artifacts_cfg = _section(config, "artifacts")
    explicit = str(artifacts_cfg.get("filename", "")).strip()
    if explicit:
        return explicit
    lineage = str(config.get("lineage", "unpaired")).strip().lower() or "unpaired"
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in lineage)
    return f"{safe}_graphs.pt"


def _artifact_path(config: dict[str, Any]) -> Path:
    if config.get("data", {}).get("graphs"):
        return resolve_config_path(config, config["data"]["graphs"])
    reproduce_cfg = _section(config, "reproduce")
    explicit = reproduce_cfg.get("artifact_path")
    if explicit:
        return resolve_config_path(config, explicit)
    artifacts_cfg = _section(config, "artifacts")
    artifact_dir = resolve_config_path(
        config,
        artifacts_cfg.get("dir", f"result/{config.get('lineage', 'unpaired')}/train/artifacts"),
    )
    return artifact_dir / _artifact_filename(config)


def _checkpoint_path(config: dict[str, Any]) -> Path:
    reproduce_cfg = _section(config, "reproduce")
    explicit = reproduce_cfg.get("checkpoint_path")
    if explicit:
        return resolve_config_path(config, explicit)
    checkpoints_cfg = _section(config, "checkpoints")
    checkpoint_dir = resolve_config_path(
        config,
        checkpoints_cfg.get("dir", f"result/{config.get('lineage', 'unpaired')}/model"),
    )
    pattern = str(reproduce_cfg.get("checkpoint_pattern", "*.pt"))
    return find_latest_checkpoint(checkpoint_dir, pattern=pattern)


def reload_tree_from_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    artifact_path = _artifact_path(config)
    checkpoint_path = _checkpoint_path(config)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Lineage artifact was not found: {artifact_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Lineage checkpoint was not found: {checkpoint_path}")

    device = _select_runtime_device()
    artifact = torch_load_compat(artifact_path, map_location="cpu")
    graphs = [graph.to(device) for graph in artifact["graphs"]]
    topo_order = list(artifact["topo_order"])
    parents_list = [list(item) for item in artifact["parents_list"]]
    time_names = list(artifact["time_names"])
    in_channels = int(artifact["in_channels"])

    checkpoint = torch_load_compat(checkpoint_path, map_location=device)
    hp = checkpoint.get("hp")
    if hp is None:
        raise RuntimeError("The unpaired checkpoint does not contain 'hp'; model reconstruction failed.")

    model = build_model(
        in_channels=in_channels,
        hidden_channels=int(hp["hidden_channels"]),
        out_channels=int(hp["out_channels"]),
        heads=int(hp["gat_heads"]),
        dropout_rate=float(hp["dropout_rate"]),
    ).to(device)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise RuntimeError("The unpaired checkpoint does not contain 'state_dict'.")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        out_tf, out_tg = model(graphs, topo_order, parents_list)

    meta = checkpoint.get("meta", {})
    extra = checkpoint.get("extra", {})
    return {
        "lineage": str(meta.get("lineage_name", config.get("lineage", "unpaired"))),
        "device": str(device),
        "artifact_path": str(artifact_path),
        "checkpoint_path": str(checkpoint_path),
        "graphs": graphs,
        "graphs_lineage": graphs,
        "topo_order": topo_order,
        "parents_list": parents_list,
        "time_names": time_names,
        "in_channels": in_channels,
        "maps": artifact.get("maps", meta.get("maps", {})),
        "hp": hp,
        "meta": meta,
        "extra": extra,
        "checkpoint": checkpoint,
        "model": model,
        "out_tf": out_tf,
        "out_tg": out_tg,
    }


def run_inference(config: dict[str, Any]) -> dict[str, Any]:
    return reload_tree_from_checkpoint(config)


__all__ = ["reload_tree_from_checkpoint", "run_inference"]
