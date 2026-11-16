"""Embedding export and smoothness summaries for unpaired trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from draft_paired.analysis.repro import (
    build_generic_labels,
    plot_smoothness_heatmap,
    summarize_smoothness,
)

from .checkpoint import save_checkpoint_metadata
from .infer import reload_tree_from_checkpoint
from .models.gnn import cosine_abs_prob
from .utils import ensure_directory, load_yaml_config, resolve_config_path


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _reproduce_output_dir(config: dict[str, Any]) -> Path:
    reproduce_cfg = _section(config, "reproduce")
    lineage = str(config.get("lineage", "unpaired"))
    return ensure_directory(
        resolve_config_path(config, reproduce_cfg.get("output_dir", f"result/{lineage}/reproduce"))
    )


def _save_embeddings_enabled(config: dict[str, Any]) -> bool:
    return bool(_section(config, "reproduce").get("save_embeddings", True))


def _cluster_ids(config: dict[str, Any], time_count: int) -> list[str]:
    reproduce_cfg = _section(config, "reproduce")
    provided = reproduce_cfg.get("cluster_ids")
    if isinstance(provided, list) and len(provided) == time_count:
        return [str(value) for value in provided]
    return [f"t{idx}" for idx in range(time_count)]


def _smoothness_settings(config: dict[str, Any]) -> dict[str, Any]:
    smooth_cfg = _section(_section(config, "reproduce"), "smoothness")
    return {
        "save_heatmap": bool(smooth_cfg.get("save_heatmap", True)),
    }


def _save_embeddings_by_label(
    out_tf: Any,
    out_tg: Any,
    labels: list[str],
    output_dir: Path,
) -> list[dict[str, str]]:
    saved_paths: list[dict[str, str]] = []
    for time_idx, label in enumerate(labels):
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(label))
        tf_path = output_dir / f"embedding_{safe}_TF.txt"
        tg_path = output_dir / f"embedding_{safe}_TG.txt"
        np.savetxt(tf_path, out_tf[:, time_idx, :].detach().cpu().numpy(), fmt="%.6f", delimiter="\t")
        np.savetxt(tg_path, out_tg[:, time_idx, :].detach().cpu().numpy(), fmt="%.6f", delimiter="\t")
        saved_paths.append({"label": str(label), "tf": str(tf_path), "tg": str(tg_path)})
    return saved_paths


def _probability_distance_matrix(out_tf: Any, out_tg: Any) -> np.ndarray:
    sims, _ = cosine_abs_prob(out_tf, out_tg)
    prob_mats = [item.detach().cpu().numpy().astype(np.float32, copy=False) for item in sims]
    time_count = len(prob_mats)
    dist = np.zeros((time_count, time_count), dtype=np.float64)
    for i in range(time_count):
        for j in range(i + 1, time_count):
            value = float(np.linalg.norm(prob_mats[i] - prob_mats[j]))
            dist[i, j] = value
            dist[j, i] = value
    return dist


def _save_matrix(path: str | Path, matrix: np.ndarray) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, matrix, fmt="%.10f", delimiter="\t")
    return str(output_path)


def reproduce_lineage_model(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    lineage_name = str(config.get("lineage", "unpaired"))
    output_dir = _reproduce_output_dir(config)
    smoothness_cfg = _smoothness_settings(config)

    inference = reload_tree_from_checkpoint(config)
    out_tf = inference["out_tf"]
    out_tg = inference["out_tg"]
    time_count = int(out_tf.size(1))

    labels = build_generic_labels(
        time_count,
        time_names=inference.get("time_names"),
        cluster_ids=_cluster_ids(config, time_count),
    )
    saved_embeddings = (
        _save_embeddings_by_label(out_tf, out_tg, labels, output_dir)
        if _save_embeddings_enabled(config)
        else []
    )

    dist_mat = _probability_distance_matrix(out_tf, out_tg)
    smoothness = summarize_smoothness(dist_mat, inference.get("parents_list"), labels)
    heatmap_path = None
    if smoothness_cfg["save_heatmap"]:
        heatmap_path = output_dir / "smoothness_tf_tg_heatmap.png"
    saved_heatmap = plot_smoothness_heatmap(
        smoothness["dist_mat_norm"],
        labels,
        float(smoothness["overall_smoothness"]),
        f"TF-TG Smoothness Heatmap - {lineage_name}",
        output_path=heatmap_path,
        cbar_label="Frobenius distance",
    )
    matrix_path = _save_matrix(output_dir / "tf_tg_prob_frobenius_dist_matrix.tsv", dist_mat)

    report = {
        "lineage": lineage_name,
        "config_path": str(Path(config_path).resolve()),
        "output_dir": str(output_dir),
        "artifact_path": inference["artifact_path"],
        "checkpoint_path": inference["checkpoint_path"],
        "seed": inference.get("meta", {}).get("seed"),
        "labels": labels,
        "saved_embeddings": saved_embeddings,
        "smoothness": {
            "avg_adjacent": float(smoothness["avg_adjacent"]),
            "max_pairwise": float(smoothness["max_pairwise"]),
            "overall_smoothness": float(smoothness["overall_smoothness"]),
            "branches": smoothness.get("branches", []),
            "aggregation": smoothness.get("aggregation", "linear"),
            "matrix_path": matrix_path,
            "heatmap_path": saved_heatmap,
        },
        "status": "reproduced",
    }
    save_checkpoint_metadata(output_dir / "reproduce_report.json", report)
    return report


__all__ = ["reproduce_lineage_model"]
