"""Network export for unpaired TF-TG trajectories."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from draft_paired.analysis.network import (
    build_edge_dataframe,
    discover_network_time_indices,
    load_and_select_all,
    load_exported_tftg_df,
    lowered_cutoff_warning,
    min_edge_fraction,
    select_edges_from_network_df,
    stage_score_cutoff,
)
from draft_paired.input_schema import read_gene_table

from ..checkpoint import save_checkpoint_metadata
from ..infer import run_inference
from ..io import InputPaths, strip_copy_suffix
from ..preprocess import CommonInputs, prepare_common_inputs
from ..utils import (
    emit_progress,
    ensure_directory,
    load_yaml_config,
    resolve_config_path,
)


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _network_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config, "network")
    lineage = str(config.get("lineage", "unpaired"))
    return {
        "output_dir": str(section.get("output_dir", f"result/{lineage}/network")),
        "active_frac_tf_tg": float(section.get("active_frac_tf_tg", 0.30)),
        "top_k_global": int(section.get("top_k_global", 50_000)),
        "min_score_keep": float(section.get("min_score_keep", 0.85)),
        "min_edge_fraction": min_edge_fraction(section),
        "m_min": int(section.get("m_min", 0)),
        "m_max": int(section.get("m_max", 5_000)),
        "gamma": float(section.get("gamma", 10.0)),
        "eps": float(section.get("eps", 1e-12)),
        "save_report": bool(section.get("save_report", True)),
    }


def _dense_network_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(_section(config, "network"), "dense")
    lineage = str(config.get("lineage", "unpaired"))
    return {
        "enabled": bool(section.get("enabled", False)),
        "output_dir": str(section.get("output_dir", f"result/{lineage}/network_dense")),
        "n_interp": int(section.get("n_interp", 4)),
        "sigma_time": float(section.get("sigma_time", 0.7)),
        "active_thr": float(section.get("active_thr", 0.30)),
        "min_score_keep": float(section.get("min_score_keep", 0.85)),
        "min_edge_fraction": min_edge_fraction(section),
        "top_k_global": int(section.get("top_k_global", 50_000)),
        "rescue_gamma": float(section.get("rescue_gamma", 10.0)),
        "rescue_m_min": int(section.get("rescue_m_min", 0)),
        "rescue_m_max": int(section.get("rescue_m_max", 5_000)),
        "rescue_eps": float(section.get("rescue_eps", 1e-8)),
        "use_one_based": bool(section.get("use_one_based", False)),
        "keep_id_cols": bool(section.get("keep_id_cols", True)),
        "add_prefix": bool(section.get("add_prefix", False)),
    }


def _read_allowed_tfs(path: Path | None, fallback: list[str]) -> set[str]:
    if path is None or not path.exists():
        return {strip_copy_suffix(value) for value in fallback}
    frame = read_gene_table(path)
    values = frame.gene_id.tolist() + frame.gene_name.tolist()
    cleaned = {strip_copy_suffix(value) for value in values if str(value).strip()}
    return cleaned or {strip_copy_suffix(value) for value in fallback}


def _normalize_rows(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return values / values.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def _prob_matrix(out_tf: torch.Tensor, out_tg: torch.Tensor, time_idx: int) -> torch.Tensor:
    tf_vec = _normalize_rows(out_tf[:, time_idx, :])
    tg_vec = _normalize_rows(out_tg[:, time_idx, :])
    return torch.abs(tf_vec @ tg_vec.T)


def _strict_filename(time_idx: int) -> str:
    return f"t{time_idx:02d}_TF_TG.tsv"


def _dense_filename(label: str) -> str:
    return f"TF_TG_dense_t{label}.tsv"


def _dense_name(value: str, node_type: str, add_prefix: bool) -> str:
    return f"{node_type}:{value}" if add_prefix else str(value)


def _make_dense_labels(time_count: int, n_interp: int) -> tuple[torch.Tensor, list[str], list[int], list[bool]]:
    tgt_times: list[float] = []
    labels: list[str] = []
    left_indices: list[int] = []
    is_integer: list[bool] = []
    for time_idx in range(time_count - 1):
        tgt_times.append(float(time_idx))
        labels.append(f"{time_idx + 1:.2f}".rstrip("0").rstrip("."))
        left_indices.append(time_idx)
        is_integer.append(True)
        for step in range(1, n_interp + 1):
            frac = step / (n_interp + 1)
            tgt_times.append(float(time_idx + frac))
            labels.append(f"{time_idx + 1 + frac:.2f}".rstrip("0").rstrip("."))
            left_indices.append(time_idx)
            is_integer.append(False)
    tgt_times.append(float(time_count - 1))
    labels.append(f"{time_count:.2f}".rstrip("0").rstrip("."))
    left_indices.append(time_count - 1)
    is_integer.append(True)
    return torch.tensor(tgt_times, dtype=torch.float32), labels, left_indices, is_integer


def _l2_normalize_rows(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return values / values.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def resample_embeddings_gaussian(
    embeddings: torch.Tensor,
    src_times: torch.Tensor,
    tgt_times: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    device = embeddings.device
    dtype = embeddings.dtype
    src_times = src_times.to(device=device, dtype=dtype)
    tgt_times = tgt_times.to(device=device, dtype=dtype)
    dt = tgt_times.view(tgt_times.numel(), 1) - src_times.view(1, src_times.numel())
    kernel = torch.exp(-0.5 * (dt / sigma) ** 2)
    kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)
    flat = embeddings.permute(1, 0, 2).reshape(src_times.numel(), -1)
    resampled = kernel @ flat
    return resampled.view(tgt_times.numel(), embeddings.size(0), embeddings.size(2)).permute(1, 0, 2).contiguous()


def _raw_fraction_table(
    common: CommonInputs,
    node_names: list[str],
    genes: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=genes, columns=node_names, dtype=float)
    anno = common.anno.copy()
    anno["cell_type"] = anno["cell_type"].astype(str)
    anno["cell_id"] = anno["cell_id"].astype(str)
    expr_df = common.expr_df
    for node_name in node_names:
        cells = anno.loc[anno["cell_type"] == str(node_name), "cell_id"].tolist()
        cells = [cell for cell in cells if cell in expr_df.columns]
        if not cells:
            continue
        values = expr_df.loc[genes, cells].fillna(0.0).to_numpy(dtype=np.float32)
        frame[node_name] = (values > 0).mean(axis=1)
    return frame


def _tf_specificity(tf_frac_df: pd.DataFrame, time_key: Any, eps: float) -> pd.Series:
    current = tf_frac_df[time_key]
    others = [column for column in tf_frac_df.columns if column != time_key]
    other_max = tf_frac_df[others].max(axis=1) if others else pd.Series(0.0, index=tf_frac_df.index)
    return current / (current + other_max + eps)


def _tf_frac_table(
    common: CommonInputs,
    time_names: list[str],
    tf_names: list[str],
    whitelist_idx: list[int],
) -> pd.DataFrame:
    frame = _raw_fraction_table(common, time_names, tf_names)
    if whitelist_idx:
        frame = frame.iloc[whitelist_idx]
    return frame


def _edge_frame(
    edges: list[tuple[int, int, float]],
    time_idx: int,
    tf_names: list[str],
    tg_names: list[str],
) -> pd.DataFrame:
    if not edges:
        return pd.DataFrame(
            columns=[
                "time_idx",
                "TF_idx",
                "TG_idx",
                "TF_id",
                "TF_name",
                "TG_id",
                "TG_name",
                "score_cos",
                "prob",
            ]
        )
    tf_idx = np.array([item[0] for item in edges], dtype=int)
    tg_idx = np.array([item[1] for item in edges], dtype=int)
    score = np.array([item[2] for item in edges], dtype=float)
    return pd.DataFrame(
        {
            "time_idx": np.full(len(edges), time_idx, dtype=int),
            "TF_idx": tf_idx,
            "TG_idx": tg_idx,
            "TF_id": [tf_names[idx] for idx in tf_idx.tolist()],
            "TF_name": [tf_names[idx] for idx in tf_idx.tolist()],
            "TG_id": [tg_names[idx] for idx in tg_idx.tolist()],
            "TG_name": [tg_names[idx] for idx in tg_idx.tolist()],
            "score_cos": score,
            "prob": score,
        }
    )


def _strict_threshold_warning(
    summary_df: pd.DataFrame,
    threshold: float,
) -> str | None:
    if summary_df.empty or "score_max_active" not in summary_df.columns:
        return None
    max_scores = pd.to_numeric(summary_df["score_max_active"], errors="coerce").dropna().to_numpy(dtype=float)
    if max_scores.size == 0:
        return None
    below = int((max_scores < float(threshold)).sum())
    total = int(max_scores.size)
    if below <= total // 2:
        return None
    sorted_scores = np.sort(max_scores)
    majority_cutoff = float(sorted_scores[total // 2])
    all_timepoints_cutoff = float(sorted_scores[0])
    return (
        f"Configured min_score_keep={threshold:.3f} exceeds score_max_active for {below}/{total} "
        "timepoints; threshold may be too high. "
        f"Use <= {majority_cutoff:.3f} to retain strict edges for a majority of timepoints, "
        f"or <= {all_timepoints_cutoff:.3f} to retain at least one strict edge for all timepoints."
    )


def _export_strict_networks(
    output_dir: Path,
    common: CommonInputs,
    graphs: list[Any],
    out_tf: torch.Tensor,
    out_tg: torch.Tensor,
    tf_names: list[str],
    tg_names: list[str],
    time_names: list[str],
    allowed_tfs: set[str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    whitelist_idx = [idx for idx, name in enumerate(tf_names) if strip_copy_suffix(name) in allowed_tfs]
    if not whitelist_idx:
        whitelist_idx = list(range(len(tf_names)))
    tf_frac_df = _tf_frac_table(common, time_names, tf_names, whitelist_idx)
    tg_frac_df = _raw_fraction_table(common, time_names, tg_names)
    tf_frac_path = output_dir / "tf_activity_fraction.tsv"
    tg_frac_path = output_dir / "tg_activity_fraction.tsv"
    tf_frac_df.to_csv(tf_frac_path, sep="\t")
    tg_frac_df.to_csv(tg_frac_path, sep="\t")

    summary_rows: list[dict[str, Any]] = []
    saved_paths: list[str] = []
    whitelist_mask = np.zeros(len(tf_names), dtype=bool)
    whitelist_mask[whitelist_idx] = True

    for time_idx, graph in enumerate(graphs):
        time_name = str(time_names[time_idx]) if time_idx < len(time_names) else f"t{time_idx:02d}"
        tf_frac = tf_frac_df.get(time_name, pd.Series(0.0, index=tf_frac_df.index)).reindex(tf_names, fill_value=0.0).to_numpy(dtype=float)
        tg_frac = tg_frac_df.get(time_name, pd.Series(0.0, index=tg_frac_df.index)).reindex(tg_names, fill_value=0.0).to_numpy(dtype=float)
        tf_active_raw = np.where((tf_frac >= settings["active_frac_tf_tg"]) & whitelist_mask)[0]
        tg_active_raw = np.where(tg_frac >= settings["active_frac_tf_tg"])[0]
        tf_active = tf_active_raw.copy()
        tg_active = tg_active_raw.copy()
        fallback_tf_all = False
        fallback_tg_all = False
        if tf_active.size == 0:
            tf_active = np.asarray(whitelist_idx, dtype=int)
            fallback_tf_all = True
        if tg_active.size == 0:
            tg_active = np.arange(len(tg_names), dtype=int)
            fallback_tg_all = True

        prob = _prob_matrix(out_tf, out_tg, time_idx).detach().cpu().numpy()
        sub = prob[np.ix_(tf_active, tg_active)]
        cutoff = stage_score_cutoff(sub, settings["min_score_keep"], settings["min_edge_fraction"])
        eligible = np.argwhere(sub >= cutoff)
        flat_full = prob.reshape(-1)
        flat_active = sub.reshape(-1)
        score_max_full = float(flat_full.max()) if flat_full.size > 0 else float("nan")
        score_mean_full = float(flat_full.mean()) if flat_full.size > 0 else float("nan")
        score_max_active = float(flat_active.max()) if flat_active.size > 0 else float("nan")
        score_mean_active = float(flat_active.mean()) if flat_active.size > 0 else float("nan")
        score_p95_active = float(np.quantile(flat_active, 0.95)) if flat_active.size > 0 else float("nan")
        score_p99_active = float(np.quantile(flat_active, 0.99)) if flat_active.size > 0 else float("nan")
        score_p999_active = float(np.quantile(flat_active, 0.999)) if flat_active.size > 0 else float("nan")

        selected_map: dict[tuple[int, int], float] = {}
        if eligible.size > 0:
            scored = [
                (int(tf_active[i]), int(tg_active[j]), float(sub[i, j]))
                for i, j in eligible.tolist()
            ]
            scored.sort(key=lambda item: -item[2])
            for tf_idx, tg_idx, score in scored[: settings["top_k_global"]]:
                selected_map[(tf_idx, tg_idx)] = score

        specificity = _tf_specificity(tf_frac_df, time_name, settings["eps"])
        rescued_tf = 0
        for tf_idx in tf_active.tolist():
            tf_name = tf_names[int(tf_idx)]
            spec = float(specificity.get(tf_name, 0.0))
            keep_count = int(
                round(
                    settings["m_min"]
                    + (settings["m_max"] - settings["m_min"]) * (spec ** settings["gamma"])
                )
            )
            if keep_count <= 0:
                continue
            row = prob[int(tf_idx), tg_active]
            valid = np.where(row >= cutoff)[0]
            if valid.size == 0:
                continue
            keep_now = min(keep_count, int(valid.size))
            if keep_now <= 0:
                continue
            if keep_now == int(valid.size):
                ordered = valid[np.argsort(-row[valid], kind="mergesort")]
            else:
                top_local = valid[np.argpartition(-row[valid], keep_now - 1)[:keep_now]]
                ordered = top_local[np.argsort(-row[top_local], kind="mergesort")]
            for tg_local in ordered.tolist():
                tg_idx = int(tg_active[tg_local])
                key = (int(tf_idx), tg_idx)
                score_value = float(row[tg_local])
                prev = selected_map.get(key)
                if prev is None or score_value > prev:
                    selected_map[key] = score_value
            rescued_tf += 1

        selected = [(tf_idx, tg_idx, score) for (tf_idx, tg_idx), score in selected_map.items()]
        selected.sort(key=lambda item: -item[2])
        frame = _edge_frame(selected, time_idx, tf_names, tg_names)
        output_path = output_dir / _strict_filename(time_idx)
        frame.to_csv(output_path, sep="\t", index=False)
        saved_paths.append(str(output_path))

        summary_rows.append(
            {
                "time_idx": int(time_idx),
                "time_name": time_name,
                "n_TF_total": int(len(tf_names)),
                "n_TG_total": int(len(tg_names)),
                "n_TF_whitelist": int(len(whitelist_idx)),
                "n_TF_active_raw": int(tf_active_raw.size),
                "n_TG_active_raw": int(tg_active_raw.size),
                "n_TF_active": int(tf_active.size),
                "n_TG_active": int(tg_active.size),
                "fallback_tf_all": bool(fallback_tf_all),
                "fallback_tg_all": bool(fallback_tg_all),
                "n_active_pairs": int(flat_active.size),
                "n_eligible_edges": int(eligible.shape[0]),
                "eligible_fraction_active": (
                    float(eligible.shape[0]) / float(flat_active.size)
                    if flat_active.size > 0
                    else 0.0
                ),
                "score_max_full": score_max_full,
                "score_mean_full": score_mean_full,
                "score_max_active": score_max_active,
                "score_mean_active": score_mean_active,
                "score_p95_active": score_p95_active,
                "score_p99_active": score_p99_active,
                "score_p999_active": score_p999_active,
                "n_selected_edges": int(len(selected)),
                "n_rescued_tf": int(rescued_tf),
                "score_cutoff": cutoff,
            }
        )

    summary_path = output_dir / "strict_active_counts.tsv"
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, sep="\t", index=False)
    total_selected_edges = int(summary_df["n_selected_edges"].sum()) if not summary_df.empty else 0
    total_eligible_edges = int(summary_df["n_eligible_edges"].sum()) if not summary_df.empty else 0
    warnings: list[str] = []
    threshold_warning = (
        lowered_cutoff_warning(
            summary_df["score_cutoff"].tolist() if not summary_df.empty else [],
            settings["min_score_keep"],
            settings["min_edge_fraction"],
            "network",
        )
        if settings["min_edge_fraction"] > 0
        else _strict_threshold_warning(summary_df, float(settings["min_score_keep"]))
    )
    if threshold_warning:
        warnings.append(threshold_warning)
        emit_progress("WARN", "network-export", threshold_warning)
    report = {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "tf_activity_fraction_path": str(tf_frac_path),
        "tg_activity_fraction_path": str(tg_frac_path),
        "network_paths": saved_paths,
        "all_timepoints_empty": bool(total_selected_edges == 0),
        "total_selected_edges": total_selected_edges,
        "total_eligible_edges": total_eligible_edges,
        "warnings": warnings,
        "settings": settings,
    }
    save_checkpoint_metadata(output_dir / "network_report.json", report)
    return report


def _dense_active_masks(
    common: CommonInputs,
    time_names: list[str],
    tf_names: list[str],
    tg_names: list[str],
    active_thr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time_count = len(time_names)
    num_tf = len(tf_names)
    num_tg = len(tg_names)
    tf_frac_df = _raw_fraction_table(common, time_names, tf_names)
    tg_frac_df = _raw_fraction_table(common, time_names, tg_names)
    tf_active = torch.zeros(time_count, num_tf, dtype=torch.bool)
    tg_active = torch.zeros(time_count, num_tg, dtype=torch.bool)
    tf_frac = torch.zeros(time_count, num_tf, dtype=torch.float32)
    for time_idx, time_name in enumerate(time_names):
        tf_frac_now = torch.tensor(
            tf_frac_df.get(time_name, pd.Series(0.0, index=tf_frac_df.index)).reindex(tf_names, fill_value=0.0).to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        tg_frac_now = torch.tensor(
            tg_frac_df.get(time_name, pd.Series(0.0, index=tg_frac_df.index)).reindex(tg_names, fill_value=0.0).to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        tf_frac[time_idx] = tf_frac_now
        tf_idx = torch.where(tf_frac_now >= active_thr)[0]
        tg_idx = torch.where(tg_frac_now >= active_thr)[0]
        if tf_idx.numel() == 0:
            tf_idx = torch.arange(num_tf, dtype=torch.long)
        if tg_idx.numel() == 0:
            tg_idx = torch.arange(num_tg, dtype=torch.long)
        tf_active[time_idx, tf_idx] = True
        tg_active[time_idx, tg_idx] = True
    return tf_active, tg_active, tf_frac


def _dense_tf_specificity(tf_frac_int: torch.Tensor, eps: float) -> torch.Tensor:
    time_count = int(tf_frac_int.size(0))
    output = torch.zeros_like(tf_frac_int)
    for time_idx in range(time_count):
        current = tf_frac_int[time_idx]
        others = [idx for idx in range(time_count) if idx != time_idx]
        other_max = tf_frac_int[others].max(dim=0).values if others else torch.zeros_like(current)
        output[time_idx] = current / (current + other_max + eps)
    return output


def _export_dense_networks(
    output_dir: Path | None,
    common: CommonInputs,
    graphs: list[Any],
    out_tf: torch.Tensor,
    out_tg: torch.Tensor,
    tf_names: list[str],
    tg_names: list[str],
    time_names: list[str],
    allowed_tfs: set[str],
    settings: dict[str, Any],
    frame_consumer: Callable[[str, pd.DataFrame], None] | None = None,
) -> dict[str, Any]:
    whitelist = torch.tensor(
        [strip_copy_suffix(name) in allowed_tfs for name in tf_names],
        dtype=torch.bool,
    )
    tf_active_int, tg_active_int, tf_frac_int = _dense_active_masks(
        common,
        time_names,
        tf_names,
        tg_names,
        settings["active_thr"],
    )
    tf_spec_int = _dense_tf_specificity(tf_frac_int, settings["rescue_eps"])

    tf_embed_int = _l2_normalize_rows(out_tf.detach().cpu().float())
    tg_embed_int = _l2_normalize_rows(out_tg.detach().cpu().float())
    time_count = int(tf_embed_int.size(1))
    src_times = torch.arange(time_count, dtype=torch.float32)
    tgt_times, dense_labels, left_indices, is_integer = _make_dense_labels(time_count, settings["n_interp"])
    tf_embed_dense = resample_embeddings_gaussian(tf_embed_int, src_times, tgt_times, settings["sigma_time"])
    tg_embed_dense = resample_embeddings_gaussian(tg_embed_int, src_times, tgt_times, settings["sigma_time"])
    for dense_idx, flag in enumerate(is_integer):
        if flag:
            time_idx = left_indices[dense_idx]
            tf_embed_dense[:, dense_idx, :] = tf_embed_int[:, time_idx, :]
            tg_embed_dense[:, dense_idx, :] = tg_embed_int[:, time_idx, :]
    tf_embed_dense = _l2_normalize_rows(tf_embed_dense)
    tg_embed_dense = _l2_normalize_rows(tg_embed_dense)

    summary_rows: list[dict[str, Any]] = []
    saved_paths: list[str] = []
    cutoffs: list[float] = []
    for dense_idx, label in enumerate(dense_labels):
        left_idx = left_indices[dense_idx]
        tf_mask = tf_active_int[left_idx].clone() & whitelist
        tg_mask = tg_active_int[left_idx].clone()
        tf_idx_active = np.where(tf_mask.numpy())[0]
        tg_idx_active = np.where(tg_mask.numpy())[0]
        if len(tf_idx_active) == 0:
            tf_idx_active = np.where(whitelist.numpy())[0]
        if len(tg_idx_active) == 0:
            tg_idx_active = np.arange(len(tg_names), dtype=int)
        tf_mask = torch.zeros_like(tf_mask)
        tg_mask = torch.zeros_like(tg_mask)
        if len(tf_idx_active) > 0:
            tf_mask[torch.tensor(tf_idx_active, dtype=torch.long)] = True
        if len(tg_idx_active) > 0:
            tg_mask[torch.tensor(tg_idx_active, dtype=torch.long)] = True

        score_matrix = (tf_embed_dense[:, dense_idx, :][tf_mask] @ tg_embed_dense[:, dense_idx, :][tg_mask].T).abs().numpy()
        cutoff = stage_score_cutoff(score_matrix, settings["min_score_keep"], settings["min_edge_fraction"])
        cutoffs.append(cutoff)
        keep_mask = score_matrix >= cutoff
        tf_local_map = {int(global_idx): local_idx for local_idx, global_idx in enumerate(tf_idx_active.tolist())}
        selected: list[tuple[int, int, float]] = []
        for tf_local, tf_global in enumerate(tf_idx_active.tolist()):
            valid_tg_local = np.flatnonzero(keep_mask[tf_local])
            for tg_local in valid_tg_local.tolist():
                selected.append((int(tf_global), int(tg_idx_active[tg_local]), float(score_matrix[tf_local, tg_local])))
        selected.sort(key=lambda item: -item[2])
        selected = selected[: settings["top_k_global"]]

        selected_set = {(tf_idx, tg_idx) for tf_idx, tg_idx, _score in selected}
        selected_counts: dict[int, int] = {}
        for tf_idx, _tg_idx, _score in selected:
            selected_counts[tf_idx] = selected_counts.get(tf_idx, 0) + 1

        rescued_edges = 0
        spec_t = tf_spec_int[left_idx].numpy()
        for tf_global in tf_idx_active.tolist():
            specificity = float(spec_t[int(tf_global)])
            keep_count = int(
                round(
                    settings["rescue_m_min"]
                    + (settings["rescue_m_max"] - settings["rescue_m_min"])
                    * (specificity ** settings["rescue_gamma"])
                )
            )
            if keep_count <= 0:
                continue
            current_count = selected_counts.get(int(tf_global), 0)
            if current_count >= keep_count:
                continue
            tf_local = tf_local_map[int(tf_global)]
            row_scores = score_matrix[tf_local]
            valid_tg_local = np.flatnonzero(row_scores >= cutoff)
            if valid_tg_local.size == 0:
                continue
            ordered_tg_local = valid_tg_local[np.argsort(-row_scores[valid_tg_local], kind="mergesort")]
            for tg_local in ordered_tg_local.tolist():
                if current_count >= keep_count:
                    break
                tg_global = int(tg_idx_active[tg_local])
                pair = (int(tf_global), tg_global)
                if pair in selected_set:
                    continue
                selected.append((int(tf_global), tg_global, float(row_scores[tg_local])))
                selected_set.add(pair)
                current_count += 1
                rescued_edges += 1
            selected_counts[int(tf_global)] = current_count

        selected.sort(key=lambda item: -item[2])
        payload: dict[str, Any] = {
            "time_label": [label] * len(selected),
            "left_int_idx": np.full(len(selected), left_idx, dtype=int),
            "is_integer": np.full(len(selected), bool(is_integer[dense_idx]), dtype=bool),
            "TF": [_dense_name(tf_names[item[0]], "TF", settings["add_prefix"]) for item in selected],
            "TG": [_dense_name(tg_names[item[1]], "TG", settings["add_prefix"]) for item in selected],
            "score_cos": np.array([item[2] for item in selected], dtype=float),
            "prob": np.array([item[2] for item in selected], dtype=float),
        }
        if settings["keep_id_cols"]:
            tf_idx = np.array([item[0] for item in selected], dtype=int)
            tg_idx = np.array([item[1] for item in selected], dtype=int)
            if settings["use_one_based"]:
                payload["TF_idx"] = tf_idx + 1
                payload["TG_idx"] = tg_idx + 1
            else:
                payload["TF_idx"] = tf_idx
                payload["TG_idx"] = tg_idx
        frame = pd.DataFrame(payload)
        if frame_consumer is not None:
            frame_consumer(label, frame)
        if output_dir is not None:
            output_path = output_dir / _dense_filename(label)
            frame.to_csv(output_path, sep="\t", index=False)
            saved_paths.append(str(output_path))
        summary_rows.append(
            {
                "dense_idx": int(dense_idx),
                "label": label,
                "left_int_idx": int(left_idx),
                "is_integer": bool(is_integer[dense_idx]),
                "n_TF_active": int(len(tf_idx_active)),
                "n_TG_active": int(len(tg_idx_active)),
                "n_selected_edges": int(len(selected)),
                "n_rescued_edges": int(rescued_edges),
                "score_cutoff": cutoffs[-1],
            }
        )

    warning = lowered_cutoff_warning(
        cutoffs, settings["min_score_keep"], settings["min_edge_fraction"], "network.dense"
    )
    if warning:
        emit_progress("WARN", "dense-network", warning)
    if output_dir is None:
        return {"settings": settings, "point_count": len(summary_rows)}
    summary_path = output_dir / "dense_active_counts.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    report = {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "dense_paths": saved_paths,
        "warnings": [warning] if warning else [],
        "settings": settings,
    }
    save_checkpoint_metadata(output_dir / "dense_network_report.json", report)
    return report


def _network_context(config: dict[str, Any]) -> dict[str, Any]:
    paths = InputPaths.from_config(config)
    strict_settings = _network_settings(config)
    dense_settings = _dense_network_settings(config)
    strict_settings["output_dir"] = str(resolve_config_path(config, strict_settings["output_dir"]))
    dense_settings["output_dir"] = str(resolve_config_path(config, dense_settings["output_dir"]))

    common = prepare_common_inputs(config)
    inference = run_inference(config)
    graphs = inference.get("graphs") or inference.get("graphs_lineage")
    if not graphs:
        raise ValueError("Inference did not return graph sequence for network export.")

    tf_names = [str(value) for value in common.tf_list]
    tg_names = [str(value) for value in common.tg_list]
    allowed_tfs = _read_allowed_tfs(paths.tf_filter_file, tf_names)

    return {
        "paths": paths,
        "strict_settings": strict_settings,
        "dense_settings": dense_settings,
        "common": common,
        "inference": inference,
        "graphs": graphs,
        "tf_names": tf_names,
        "tg_names": tg_names,
        "time_names": list(inference.get("time_names") or []),
        "allowed_tfs": allowed_tfs,
    }


def export_lineage_networks_from_config(config: dict[str, Any], *, dense: bool = False) -> dict[str, Any]:
    context = _network_context(config)
    strict_dir = ensure_directory(context["strict_settings"]["output_dir"])
    strict_report = _export_strict_networks(
        output_dir=strict_dir,
        common=context["common"],
        graphs=context["graphs"],
        out_tf=context["inference"]["out_tf"],
        out_tg=context["inference"]["out_tg"],
        tf_names=context["tf_names"],
        tg_names=context["tg_names"],
        time_names=context["time_names"],
        allowed_tfs=context["allowed_tfs"],
        settings=context["strict_settings"],
    )

    dense_report: dict[str, Any] | None = None
    if dense:
        context["dense_settings"]["enabled"] = True
        dense_dir = ensure_directory(context["dense_settings"]["output_dir"])
        dense_report = _export_dense_networks(
            output_dir=dense_dir,
            common=context["common"],
            graphs=context["graphs"],
            out_tf=context["inference"]["out_tf"],
            out_tg=context["inference"]["out_tg"],
            tf_names=context["tf_names"],
            tg_names=context["tg_names"],
            time_names=context["time_names"],
            allowed_tfs=context["allowed_tfs"],
            settings=context["dense_settings"],
        )

    return {
        "lineage": str(config.get("lineage", "unpaired")),
        "strict": strict_report,
        "dense": dense_report,
        "checkpoint_path": context["inference"]["checkpoint_path"],
    }


def export_lineage_networks(config_path: str | Path, *, dense: bool = False) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    report = export_lineage_networks_from_config(config, dense=dense)
    report["config_path"] = str(Path(config_path).resolve())
    return report


__all__ = [
    "build_edge_dataframe",
    "discover_network_time_indices",
    "export_lineage_networks",
    "export_lineage_networks_from_config",
    "load_and_select_all",
    "load_exported_tftg_df",
    "select_edges_from_network_df",
]
