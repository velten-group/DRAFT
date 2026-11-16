"""TF activity and target-gene regulon analyses."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..checkpoint import save_checkpoint_metadata
from ..infer import run_inference
from ..preprocess import prepare_common_inputs
from ..utils import ensure_directory, load_yaml_config, resolve_config_path
from .gene_keys import match_symbol
from .network import (
    _build_re_coordinate_arrays,
    _load_tg_meta,
    _network_settings,
    _select_composite_re_rows,
    _select_valid_cols,
)
from .repro import build_generic_labels, build_germ_labels, build_soma_labels, safe_label


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _single_tf_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config, "single_tf")
    lineage = str(config.get("lineage", "")).lower()
    config_v1_germ = int(config.get("config_version", 1)) < 2 and lineage in {"germ", "germline"}
    default_max_expr_frac = 0.7 if config_v1_germ else 0.8
    default_time_for_top: Any = 3 if config_v1_germ else 0
    default_topn_tg: Any = 25 if config_v1_germ else 50
    tf_symbols = section.get("tf_symbols")
    if tf_symbols is None:
        tf_symbol = section.get("tf_symbol")
        tf_symbols = [tf_symbol] if tf_symbol else []
    elif not isinstance(tf_symbols, list):
        tf_symbols = [tf_symbols]

    return {
        "enabled": bool(section.get("enabled", False)),
        "output_dir": str(section.get("output_dir", f"result/{lineage}/single_tf")),
        "tf_symbols": [str(value) for value in tf_symbols if str(value).strip()],
        "centrality_mode": str(section.get("centrality_mode", "topk_sum")),
        "topk": int(section.get("topk", 500)),
        "threshold": float(section.get("threshold", 0.5)),
        "min_expr_frac": float(section.get("min_expr_frac", 0.1)),
        "max_expr_frac": float(section.get("max_expr_frac", default_max_expr_frac)),
        "time_for_top": section.get("time_for_top", default_time_for_top),
        "topn_tg": section.get("topn_tg", default_topn_tg),
        "user_thresh": float(section.get("user_thresh", 0.20)),
        "network_seed": int(section.get("network_seed", 42)),
        "save_fig": bool(section.get("save_fig", True)),
        "save_tables": bool(section.get("save_tables", True)),
        "fs_title": int(section.get("fs_title", 18)),
        "fs_other": int(section.get("fs_other", 12)),
        "fs_gene": int(section.get("fs_gene", 15)),
        "fs_tf_center": int(section.get("fs_tf_center", 20)),
        "activity_dpi": int(section.get("activity_dpi", 300)),
        "network_dpi": int(section.get("network_dpi", 300)),
        "consensus": _section(section, "consensus"),
    }


def _pick_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def build_fbgn_to_name(common: dict[str, Any]) -> dict[str, str]:
    fb2name: dict[str, str] = {}

    tg_df = common.get("tg_df")
    if isinstance(tg_df, pd.DataFrame):
        fb_col = _pick_col(tg_df, ["fbgn_id", "fbgn", "fbgnid", "gene_id", "ensembl_id", "id"])
        name_col = _pick_col(tg_df, ["gene_name", "symbol", "gene", "name"])
        if fb_col and name_col:
            fb2name.update(dict(zip(tg_df[fb_col].astype(str), tg_df[name_col].astype(str))))

    tf_df = common.get("tf_df")
    if isinstance(tf_df, pd.DataFrame):
        fb_col = _pick_col(tf_df, ["fbgn_id", "fbgn", "fbgnid", "gene_id", "ensembl_id", "id"])
        name_col = _pick_col(tf_df, ["gene_name", "symbol", "gene", "name"])
        if fb_col and name_col:
            fb2name.update(dict(zip(tf_df[fb_col].astype(str), tf_df[name_col].astype(str))))

    sym2fb = common.get("sym2fb")
    if isinstance(sym2fb, dict):
        for symbol, fbgn in sym2fb.items():
            fbgn = str(fbgn)
            if fbgn not in fb2name:
                fb2name[fbgn] = str(symbol)

    return fb2name


def centrality_from_prob(
    prob: np.ndarray,
    mode: str = "topk_sum",
    topk: int = 500,
    thresh: float = 0.5,
) -> float:
    values = np.asarray(prob, dtype=float)
    if values.size == 0:
        return 0.0
    if mode == "topk_sum":
        keep = int(min(topk, values.size))
        if keep <= 0:
            return 0.0
        idx = np.argpartition(values, -keep)[-keep:]
        return float(values[idx].sum())
    if mode == "topk_mean":
        keep = int(min(topk, values.size))
        if keep <= 0:
            return 0.0
        idx = np.argpartition(values, -keep)[-keep:]
        return float(values[idx].mean())
    if mode == "threshold_count":
        return float(np.sum(values >= thresh))
    if mode == "threshold_sum":
        return float(values[values >= thresh].sum())
    raise ValueError(f"Unsupported centrality mode: {mode}")


def _l2norm_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def resolve_tf_id_and_index(
    tf_symbol: str,
    tf_ids: list[str],
    tf_display_names: list[str],
    common: dict[str, Any],
) -> tuple[str, int]:
    # Exact spelling first; case is ignored only where that names one gene
    # (Drosophila H and h are different TFs).
    tf_id = None
    sym2fb = common.get("sym2fb")
    if isinstance(sym2fb, dict):
        keys = list(sym2fb)
        hit = match_symbol(tf_symbol, keys)
        if hit is not None:
            tf_id = str(sym2fb[keys[hit]])

    tf_df = common.get("tf_df")
    if tf_id is None and isinstance(tf_df, pd.DataFrame):
        gene_col = _pick_col(tf_df, ["gene_name", "symbol"])
        fbgn_col = _pick_col(tf_df, ["fbgn_id", "fbgn", "fbgnid", "gene_id", "ensembl_id", "id"])
        if gene_col and fbgn_col:
            hit = match_symbol(tf_symbol, tf_df[gene_col].astype(str))
            if hit is not None:
                tf_id = str(tf_df.iloc[hit][fbgn_col])

    if tf_id is None:
        hit = match_symbol(tf_symbol, tf_ids)
        if hit is not None:
            tf_id = str(tf_ids[hit])

    if tf_id is None:
        hit = match_symbol(tf_symbol, tf_display_names)
        if hit is not None:
            tf_id = str(tf_ids[hit])

    if tf_id is None or tf_id not in tf_ids:
        raise RuntimeError(f"Could not resolve TF '{tf_symbol}' inside the TF graph namespace.")

    return tf_id, tf_ids.index(tf_id)


def resolve_time_index(
    time_spec: Any,
    time_labels: list[str],
    time_count: int,
    tf_symbol: str | None = None,
) -> int:
    if isinstance(time_spec, dict):
        if tf_symbol is not None:
            if tf_symbol in time_spec:
                return resolve_time_index(time_spec[tf_symbol], time_labels, time_count, tf_symbol=tf_symbol)
            tf_key = tf_symbol.lower()
            if tf_key in time_spec:
                return resolve_time_index(time_spec[tf_key], time_labels, time_count, tf_symbol=tf_symbol)
        return resolve_time_index(next(iter(time_spec.values())), time_labels, time_count, tf_symbol=tf_symbol)

    if isinstance(time_spec, (int, np.integer)):
        time_idx = int(time_spec)
        if time_idx < 0 or time_idx >= time_count:
            raise ValueError(f"time={time_idx} is outside the valid range [0, {time_count - 1}].")
        return time_idx

    if isinstance(time_spec, str):
        if time_spec in time_labels:
            return time_labels.index(time_spec)
        match = re.search(r"(\d+)$", time_spec)
        if match:
            time_idx = int(match.group(1))
            if 0 <= time_idx < time_count:
                return time_idx
        raise ValueError(f"Could not resolve time label '{time_spec}'.")

    raise TypeError("time_for_top must be int, str, or a tf-specific mapping.")


def resolve_topn(topn_spec: Any, tf_symbol: str) -> int:
    if isinstance(topn_spec, dict):
        if tf_symbol in topn_spec:
            return int(topn_spec[tf_symbol])
        tf_key = tf_symbol.lower()
        if tf_key in topn_spec:
            return int(topn_spec[tf_key])
        return int(next(iter(topn_spec.values())))
    return int(topn_spec)


def _time_labels(lineage: str, inference: dict[str, Any], time_count: int) -> list[str]:
    time_names = inference.get("time_names")
    if isinstance(time_names, list) and len(time_names) == time_count:
        return build_generic_labels(time_count, time_names=time_names)
    if lineage in {"germ", "germline"}:
        germline_types = inference.get("germline_types")
        return build_germ_labels(time_count, germline_types=germline_types)
    somatic_names = inference.get("somatic_names")
    if isinstance(somatic_names, list) and len(somatic_names) == time_count:
        return build_soma_labels(time_count, somatic_names=somatic_names)
    return build_generic_labels(time_count)


def _label_to_cells(common: dict[str, Any]) -> dict[str, list[str]]:
    anno = common["anno"]
    expr_df = common["expr_df"]
    celltype_col = _pick_col(anno, ["cell_type", "celltype", "type", "annotation", "cluster", "label"])
    cellid_col = _pick_col(anno, ["cell_id", "cellid", "barcode", "cell", "obs_names", "id"])
    if celltype_col is None or cellid_col is None:
        raise RuntimeError(f"Could not identify anno cell-type/cell-id columns: {list(anno.columns)}")

    expr_cols = set(expr_df.columns.astype(str))
    label2cells: dict[str, list[str]] = {}
    for label in pd.unique(anno[celltype_col].astype(str)):
        cells = anno.loc[anno[celltype_col].astype(str) == str(label), cellid_col].astype(str).tolist()
        label2cells[str(label)] = [cell for cell in cells if cell in expr_cols]
    return label2cells


def _prepare_keep_masks(
    time_labels: list[str],
    tg_ids: list[str],
    expr_df: pd.DataFrame,
    label2cells: dict[str, list[str]],
    fb2name: dict[str, str],
    min_expr_frac: float,
    max_expr_frac: float,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    expr_idx = set(expr_df.index.astype(str))
    tg_expr_key: list[str | None] = []
    for tg in map(str, tg_ids):
        if tg in expr_idx:
            tg_expr_key.append(tg)
        else:
            symbol = fb2name.get(tg)
            tg_expr_key.append(str(symbol) if symbol is not None and str(symbol) in expr_idx else None)

    tg_expr_key_arr = np.asarray(tg_expr_key, dtype=object)
    valid_tg_mask = np.asarray([value is not None for value in tg_expr_key_arr], dtype=bool)

    keep_masks: list[np.ndarray | None] = [None] * len(time_labels)
    frac_list: list[np.ndarray | None] = [None] * len(time_labels)
    for time_idx, label in enumerate(time_labels):
        cells = label2cells.get(str(label), [])
        if len(cells) == 0:
            continue
        sub = expr_df.loc[tg_expr_key_arr[valid_tg_mask].tolist(), cells]
        frac_valid = (sub.values > 0).mean(axis=1)
        fracs = np.full(len(tg_ids), np.nan, dtype=float)
        fracs[valid_tg_mask] = frac_valid
        keep = np.isfinite(fracs) & (fracs >= min_expr_frac) & (fracs <= max_expr_frac)
        keep_masks[time_idx] = keep
        frac_list[time_idx] = fracs
    return keep_masks, frac_list


def _activity_table(
    tf_id: str,
    tf_symbol: str,
    tf_idx: int,
    time_labels: list[str],
    keep_masks: list[np.ndarray | None],
    label2cells: dict[str, list[str]],
    expr_df: pd.DataFrame,
    out_tf_np: np.ndarray,
    out_tg_np: np.ndarray,
    centrality_mode: str,
    topk: int,
    threshold: float,
) -> pd.DataFrame:
    norm_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get_norm_mats(time_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if time_idx not in norm_cache:
            norm_cache[time_idx] = (
                _l2norm_rows(out_tf_np[:, time_idx, :]),
                _l2norm_rows(out_tg_np[:, time_idx, :]),
            )
        return norm_cache[time_idx]

    def mean_expr_tf_for_label(label: str) -> float:
        cells = label2cells.get(str(label), [])
        if len(cells) == 0:
            return float("nan")
        if tf_id in expr_df.index:
            return float(np.asarray(expr_df.loc[tf_id, cells]).mean())
        if tf_symbol in expr_df.index:
            return float(np.asarray(expr_df.loc[tf_symbol, cells]).mean())
        return float("nan")

    rows: list[dict[str, Any]] = []
    for time_idx, label in enumerate(time_labels):
        keep = keep_masks[time_idx]
        if keep is None:
            rows.append(
                {
                    "t": time_idx,
                    "time_label": str(label),
                    "time_num": time_idx,
                    "centrality": np.nan,
                    "mean_expr": np.nan,
                    "activity": np.nan,
                    "expression_weighted_activity": np.nan,
                }
            )
            continue
        htf_tn, htg_tn = get_norm_mats(time_idx)
        tf_vec = htf_tn[tf_idx]
        cos = htg_tn @ tf_vec
        prob = (cos + 1.0) / 2.0
        centrality = centrality_from_prob(prob[keep], mode=centrality_mode, topk=topk, thresh=threshold)
        mean_expr = mean_expr_tf_for_label(str(label))
        time_match = re.search(r"(\d+)$", str(label))
        time_num = int(time_match.group(1)) if time_match else time_idx
        rows.append(
            {
                "t": time_idx,
                "time_label": str(label),
                "time_num": time_num,
                "centrality": centrality,
                "mean_expr": mean_expr,
                "activity": centrality,
                "expression_weighted_activity": centrality * mean_expr,
            }
        )
    return pd.DataFrame(rows).sort_values(["time_num", "t"], kind="mergesort").reset_index(drop=True)


def _top_tg_table(
    tf_idx: int,
    time_idx: int,
    label: str,
    keep: np.ndarray | None,
    fracs: np.ndarray | None,
    topn: int,
    user_thresh: float,
    tg_ids: list[str],
    fb2name: dict[str, str],
    out_tf_np: np.ndarray,
    out_tg_np: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    if keep is None or fracs is None:
        return pd.DataFrame(), []

    htf_tn = _l2norm_rows(out_tf_np[:, time_idx, :])
    htg_tn = _l2norm_rows(out_tg_np[:, time_idx, :])
    tf_vec = htf_tn[tf_idx]
    cos_all = htg_tn @ tf_vec

    cand_idx = np.where(keep)[0]
    if cand_idx.size == 0:
        return pd.DataFrame(), []
    cos_cand = cos_all[cand_idx]
    order = np.lexsort((cand_idx, -cos_cand))
    top_idx = cand_idx[order[: min(topn, len(order))]]

    rows_top: list[dict[str, Any]] = []
    for rank, tg_idx in enumerate(top_idx.tolist(), start=1):
        tg_id = str(tg_ids[tg_idx])
        rows_top.append(
            {
                "t": int(time_idx),
                "time_label": str(label),
                "rank": int(rank),
                "tg_id": tg_id,
                "cos_sim": float(cos_all[tg_idx]),
                "expr_frac_>0": float(fracs[tg_idx]) if np.isfinite(fracs[tg_idx]) else np.nan,
                "expr_ge_5pct": bool(np.isfinite(fracs[tg_idx]) and fracs[tg_idx] >= 0.05),
                "expr_le_90pct": bool(np.isfinite(fracs[tg_idx]) and fracs[tg_idx] <= 0.90),
                "expr_ge_user_thresh": bool(np.isfinite(fracs[tg_idx]) and fracs[tg_idx] >= user_thresh),
            }
        )

    table = pd.DataFrame(rows_top)
    if table.empty:
        return table, []
    table_named = table.copy()
    table_named["tg_name"] = table_named["tg_id"].map(fb2name).fillna(table_named["tg_id"])
    table_named.insert(4, "tg_id_original", table_named["tg_id"])
    table_named["tg_id"] = table_named["tg_name"]
    table_named = table_named.drop(columns=["tg_name"])
    target_names = _make_unique_labels_hash(table_named["tg_id"].astype(str).tolist())
    return table_named, target_names


def _safe_tag_from_tf_list(tf_symbols: list[str], max_len: int = 120) -> str:
    tag = "_".join(str(value) for value in tf_symbols)
    tag = re.sub(r"[^A-Za-z0-9_-]+", "_", tag)
    if len(tag) > max_len:
        tag = tag[:max_len].rstrip("_")
    return tag


def _make_unique_labels_hash(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        output.append(name if count == 1 else f"{name}#{count}")
    return output


def _graphs_from_inference(inference: dict[str, Any]) -> list[Any]:
    for key in ["graphs", "graphs_lineage", "graphs_somatic", "graphs_germline"]:
        value = inference.get(key)
        if isinstance(value, list) and value:
            return value
    raise RuntimeError("Inference payload does not expose lineage graphs required for RE selection.")


def _regulon_rows(inference: dict[str, Any], time: float, tf_idx: int, target_ids: list[str],
                  refs: dict[str, Any], settings: dict[str, Any], tg_meta: dict[str, Any],
                  re_coord: dict[str, np.ndarray], sigma: float = 0.7) -> list[dict[str, Any]]:
    if not target_ids:
        return []
    graph = _graphs_from_inference(inference)[int(np.floor(time))]
    _, valid, _ = _select_valid_cols(graph["TF"].x, graph["RE"].x, graph["TG"].x)
    frac = (valid != 0).float().mean(dim=1)
    ire = torch.where(frac >= settings["active_frac_re"])[0]
    if not len(ire):
        return []
    normalized = {}
    for key in ("tf", "re", "tg"):
        values = inference[f"out_{key}"]
        if float(time).is_integer():
            sampled = values[:, int(time), :]
        else:
            stage = torch.arange(values.size(1), dtype=values.dtype, device=values.device)
            weight = torch.exp(-0.5 * ((stage - time) / sigma) ** 2)
            sampled = torch.einsum("ntd,t->nd", values, weight / weight.sum())
        normalized[key] = F.normalize(sampled, dim=1)
    lookup = {key: i for i, key in enumerate(refs["tg_ids"])}
    rows = []
    for target in target_ids:
        if target not in lookup:
            raise ValueError(f"Unknown consensus target ID {target!r}.")
        selected = _select_composite_re_rows(
            time_idx=0, tf_global=tf_idx, tg_global=lookup[target], ire=ire,
            re_embed=normalized["re"][ire], tf_all_norm=normalized["tf"], tg_all_norm=normalized["tg"],
            refs=refs, tg_meta_bundle=tg_meta, re_coord=re_coord,
            act_re_frac=frac[ire].detach().cpu().numpy(), settings=settings, include_score_columns=True)
        for row in selected:
            row["time_idx"] = float(time)
        rows.extend(selected)
    return rows


def _plot_activity_on_ax(
    ax: Any,
    df_activity: pd.DataFrame,
    tf_symbol: str,
    tf_id: str,
    fs_title: int,
    fs_other: int,
) -> None:
    for column, label, color in (("centrality", "Network centrality", "#d97924"),
                                 ("mean_expr", "RNA expression", "#7154a1")):
        values = df_activity[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        scaled = np.full_like(values, np.nan)
        if finite.any():
            low, high = values[finite].min(), values[finite].max()
            scaled[finite] = (values[finite] - low) / (high - low) if high > low else 0.0
        ax.plot(df_activity["time_num"].values, scaled, marker="o", linewidth=2, label=label, color=color)
    ax.set_title(f"{tf_symbol} (tf_id={tf_id})", fontsize=fs_title)
    ax.set_xlabel("Time Point", fontsize=fs_other)
    ax.set_ylabel("Scaled value (0–1)", fontsize=fs_other)
    ax.legend(frameon=False, fontsize=fs_other)
    ax.tick_params(axis="both", which="major", labelsize=fs_other)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_regulon_network_on_ax(
    ax: Any,
    tf_symbol: str,
    targets: list[str],
    target_re_names: list[list[str]],
    time_idx: int,
    network_seed: int,
    fs_title: int,
    fs_gene: int,
    fs_tf_center: int,
) -> None:
    count = len(targets)
    ax.set_title(f"{tf_symbol} regulon @ t={time_idx} (TopTG={count})", fontsize=fs_title)
    if count == 0:
        ax.text(0.5, 0.5, "No targets", ha="center", va="center", transform=ax.transAxes, fontsize=fs_title)
        ax.set_axis_off()
        return

    seed = int(network_seed + abs(hash(tf_symbol)) % 10_000)
    random.seed(seed)
    np.random.seed(seed)

    r_out_mean = 4.8 + 0.55 * np.sqrt(count) + 1.0
    r_out_jit = 0.35
    r_in_mean = max(2.0, 0.55 * r_out_mean)
    r_in_jit = 0.25
    noise_theta = 0.10 if count <= 80 else 0.05
    tg_node_size = 260 if count <= 40 else (180 if count <= 120 else 90)
    re_node_size = 80 if count <= 120 else 45

    colors = {
        "left": "#f46d6d",
        "right": "#f0a21b",
        "bottom": "#7db5ff",
        "top": "#d78fdc",
    }

    def tg_color_by_angle(theta: float) -> str:
        angle = theta % (2.0 * np.pi)
        if np.pi / 4 <= angle < 3 * np.pi / 4:
            return colors["top"]
        if 3 * np.pi / 4 <= angle < 5 * np.pi / 4:
            return colors["left"]
        if 5 * np.pi / 4 <= angle < 7 * np.pi / 4:
            return colors["bottom"]
        return colors["right"]

    graph = nx.DiGraph()
    graph.add_node(tf_symbol, kind="tf")
    theta_base = np.linspace(0, 2 * np.pi, count, endpoint=False)
    positions: dict[str, tuple[float, float]] = {tf_symbol: (0.0, 0.0)}
    tg_colors: dict[str, str] = {}

    for idx, tg_name in enumerate(targets):
        theta = theta_base[idx] + np.random.uniform(-noise_theta, noise_theta)
        radius_tg = np.random.normal(r_out_mean, r_out_jit)
        x_tg, y_tg = radius_tg * np.cos(theta), radius_tg * np.sin(theta)
        selected_re_names = target_re_names[idx] if idx < len(target_re_names) else []
        re_nodes: list[str] = []
        for re_idx, re_label in enumerate(selected_re_names):
            radius_re = np.random.normal(r_in_mean, r_in_jit)
            theta_re = theta + np.random.uniform(-0.12, 0.12)
            x_re, y_re = radius_re * np.cos(theta_re), radius_re * np.sin(theta_re)
            re_key = f"RE__{idx}_{re_idx}"
            graph.add_node(re_key, kind="re", label=str(re_label))
            positions[re_key] = (x_re, y_re)
            re_nodes.append(re_key)

        graph.add_node(tg_name, kind="tg")
        positions[tg_name] = (x_tg, y_tg)
        if re_nodes:
            for re_name in re_nodes:
                graph.add_edge(tf_symbol, re_name)
                graph.add_edge(re_name, tg_name)
        else:
            graph.add_edge(tf_symbol, tg_name)
        tg_colors[tg_name] = tg_color_by_angle(theta)

    nx.draw_networkx_edges(graph, positions, width=0.8, edge_color="black", alpha=0.5, arrows=False, ax=ax)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[tf_symbol],
        node_size=900,
        node_color="#ffd1b8",
        edgecolors="#444444",
        linewidths=1.8,
        node_shape="h",
        ax=ax,
    )

    re_nodes_all = [node for node in graph.nodes() if graph.nodes[node]["kind"] == "re"]
    rng = np.random.default_rng(0)
    if re_nodes_all:
        re_square_set = set(rng.choice(re_nodes_all, size=len(re_nodes_all) // 2, replace=False))
        re_square = list(re_square_set)
        re_diamond = [node for node in re_nodes_all if node not in re_square_set]
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=re_square,
            node_size=re_node_size,
            node_color="#e6b0b0",
            edgecolors="#e6b0b0",
            node_shape="s",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=re_diamond,
            node_size=re_node_size,
            node_color="#e6b0b0",
            edgecolors="#e6b0b0",
            node_shape="D",
            ax=ax,
        )

    tg_nodes = [node for node in graph.nodes() if graph.nodes[node]["kind"] == "tg"]
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=tg_nodes,
        node_size=tg_node_size,
        node_color=[tg_colors[node] for node in tg_nodes],
        edgecolors="#ffffff",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={tf_symbol: tf_symbol.upper()},
        font_size=fs_tf_center,
        font_weight="bold",
        ax=ax,
    )
    for node in tg_nodes:
        x_pos, y_pos = positions[node]
        label = str(graph.nodes[node].get("label", node))
        ax.text(x_pos, y_pos, label, fontsize=fs_gene, fontweight="normal", ha="center", va="center", zorder=13)
    for node in re_nodes_all:
        x_pos, y_pos = positions[node]
        label = str(graph.nodes[node].get("label", node))
        ax.text(x_pos, y_pos, label, fontsize=max(8, fs_gene - 2), fontweight="normal", ha="center", va="center", zorder=13)

    ax.set_aspect("equal")
    ax.set_axis_off()


def _single_tf_context(config: dict[str, Any]) -> dict[str, Any]:
    inference = run_inference(config)
    lineage = str(inference.get("lineage", config.get("lineage", ""))).lower()
    common = inference.get("artifacts", {}).get("common")
    if not isinstance(common, dict):
        common = prepare_common_inputs(config)

    maps = inference.get("maps")
    if not isinstance(maps, dict):
        raise RuntimeError("Checkpoint/artifact did not expose node maps required for single-TF analysis.")

    time_count = int(inference["out_tf"].size(1))
    return {
        "lineage": lineage,
        "inference": inference,
        "common": common,
        "maps": maps,
        "time_count": time_count,
        "time_labels": _time_labels(lineage, inference, time_count),
    }


def run_single_tf_analysis_from_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = _single_tf_settings(config)
    if not settings["tf_symbols"]:
        raise ValueError(
            "Select TFs with --tf TF_NAME, or set single_tf.tf_symbols in the config."
        )

    context = _single_tf_context(config)
    output_dir = ensure_directory(resolve_config_path(config, settings["output_dir"]))
    common = context["common"]
    inference = context["inference"]
    maps = context["maps"]
    expr_df = common["expr_df"]
    input_paths = common["input_paths"]
    time_labels = context["time_labels"]
    label2cells = _label_to_cells(common)
    fb2name = build_fbgn_to_name(common)
    tf_ids = [str(value) for value in maps["tf"]]
    tg_ids = [str(value) for value in maps["tg"]]
    re_names = [str(value) for value in maps["re"]] if "re" in maps else [str(value) for value in common["re_list"]]
    tf_display_names = [str(fb2name.get(tf_id, tf_id)) for tf_id in tf_ids]
    tg_display_names = [str(fb2name.get(tg_id, tg_id)) for tg_id in tg_ids]
    net_settings = _network_settings(config)
    tg_meta_bundle = _load_tg_meta(input_paths.tg_list)
    re_coord = _build_re_coordinate_arrays(re_names)

    keep_masks, frac_list = _prepare_keep_masks(
        time_labels=time_labels,
        tg_ids=tg_ids,
        expr_df=expr_df,
        label2cells=label2cells,
        fb2name=fb2name,
        min_expr_frac=settings["min_expr_frac"],
        max_expr_frac=settings["max_expr_frac"],
    )

    out_tf_np = inference["out_tf"].detach().cpu().numpy()
    out_tg_np = inference["out_tg"].detach().cpu().numpy()

    results: list[dict[str, Any]] = []
    saved_tables: list[str] = []
    for tf_symbol in settings["tf_symbols"]:
        tf_id, tf_idx = resolve_tf_id_and_index(tf_symbol, tf_ids, tf_display_names, common)
        df_activity = _activity_table(
            tf_id=tf_id,
            tf_symbol=tf_symbol,
            tf_idx=tf_idx,
            time_labels=time_labels,
            keep_masks=keep_masks,
            label2cells=label2cells,
            expr_df=expr_df,
            out_tf_np=out_tf_np,
            out_tg_np=out_tg_np,
            centrality_mode=settings["centrality_mode"],
            topk=settings["topk"],
            threshold=settings["threshold"],
        )
        time_idx_top = resolve_time_index(
            settings["time_for_top"],
            time_labels,
            context["time_count"],
            tf_symbol=tf_symbol,
        )
        df_top, targets = _top_tg_table(
            tf_idx=tf_idx,
            time_idx=time_idx_top,
            label=str(time_labels[time_idx_top]),
            keep=keep_masks[time_idx_top],
            fracs=frac_list[time_idx_top],
            topn=resolve_topn(settings["topn_tg"], tf_symbol),
            user_thresh=settings["user_thresh"],
            tg_ids=tg_ids,
            fb2name=fb2name,
            out_tf_np=out_tf_np,
            out_tg_np=out_tg_np,
        )
        refs = {"tf_ids": tf_ids, "tf_names": tf_display_names, "tg_ids": tg_ids,
                "tg_names": tg_display_names, "re_names": re_names}
        re_rows = _regulon_rows(inference, float(time_idx_top), tf_idx,
                                df_top["tg_id_original"].astype(str).tolist() if len(df_top) else [],
                                refs, net_settings, tg_meta_bundle, re_coord)
        target_re_names = [[str(row["RE_name"]) for row in re_rows if row["TG_id"] == key]
                           for key in df_top.get("tg_id_original", [])]
        consensus_report = None
        spec = settings["consensus"]
        if spec.get("enabled", False):
            from .consensus import consensus_times, stage_rankings, export_consensus

            times = consensus_times(spec, time_labels)
            tables = stage_rankings(
                out_tf_np, out_tg_np, tf_idx, tg_ids, tg_display_names, frac_list, times, spec,
                min_fraction=settings["min_expr_frac"], max_fraction=settings["max_expr_frac"],
                parents=inference.get("parents_list", inference.get("parents_somatic")))
            consensus_report = export_consensus(
                config, spec, tf_symbol, tf_id, tf_idx, tables, output_dir,
                re_selector=lambda time, ids: _regulon_rows(
                    inference, time, tf_idx, ids, refs, net_settings, tg_meta_bundle, re_coord,
                    sigma=float(spec.get("sigma_time", 0.7))))
            saved_tables.extend(consensus_report[key] for key in ["all_targets", "top_targets", "stage_ranks", "regulon_re"]
                                if consensus_report[key])

        if settings["save_tables"]:
            activity_path = output_dir / f"{safe_label(tf_symbol)}_activity.tsv"
            df_activity.to_csv(activity_path, sep="\t", index=False)
            saved_tables.append(str(activity_path))
            top_path = output_dir / f"{safe_label(tf_symbol)}_top_tg.tsv"
            df_top.to_csv(top_path, sep="\t", index=False)
            saved_tables.append(str(top_path))

        results.append(
            {
                "tf": str(tf_symbol),
                "tf_id": str(tf_id),
                "tf_idx": int(tf_idx),
                "df_activity": df_activity,
                "df_top": df_top,
                "t_top": int(time_idx_top),
                "targets": targets,
                "target_re_names": target_re_names,
                "consensus": consensus_report,
            }
        )

    tf_tag = _safe_tag_from_tf_list(settings["tf_symbols"])
    saved_figures: list[str] = []
    if settings["save_fig"]:
        for result in results:
            consensus = result["consensus"]
            if consensus is None:
                continue
            fig, ax = plt.subplots(figsize=(9, 9))
            _draw_regulon_network_on_ax(
                ax, result["tf"], consensus["target_names"], consensus["target_res"], "consensus",
                settings["network_seed"], settings["fs_title"], settings["fs_gene"], settings["fs_tf_center"])
            ax.set_title(f"{result['tf']} consensus regulon ({consensus['rank_method']})")
            path = output_dir / f"{safe_label(result['tf'])}_consensus_regulon.png"
            fig.savefig(path, dpi=settings["network_dpi"], bbox_inches="tight")
            plt.close(fig)
            saved_figures.append(str(path))

    n_items = len(results)
    ncols = 2
    nrows = int(math.ceil(n_items / ncols))

    fig_regulon, axes_regulon = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 7.5 * nrows), squeeze=False)
    for idx, result in enumerate(results):
        ax = axes_regulon[idx // ncols][idx % ncols]
        _draw_regulon_network_on_ax(
            ax,
            result["tf"],
            result["targets"],
            result["target_re_names"],
            result["t_top"],
            network_seed=settings["network_seed"],
            fs_title=settings["fs_title"],
            fs_gene=settings["fs_gene"],
            fs_tf_center=settings["fs_tf_center"],
        )
    for idx in range(n_items, nrows * ncols):
        axes_regulon[idx // ncols][idx % ncols].set_axis_off()
    fig_regulon.suptitle("Regulon network (TopTG) per TF", fontsize=settings["fs_title"], y=1.02)
    fig_regulon.tight_layout()
    if settings["save_fig"]:
        if len(results) == 1:
            regulon_name = f"{safe_label(results[0]['tf'])}_regulon.tif"
        else:
            regulon_name = f"{tf_tag}_regulon.tif"
        regulon_path = output_dir / regulon_name
        fig_regulon.savefig(regulon_path, dpi=settings["network_dpi"], bbox_inches="tight")
        saved_figures.append(str(regulon_path))
        if len(results) == 1:
            named_copy = (
                f"{safe_label(results[0]['tf'])}_t{int(results[0]['t_top'])}_"
                f"network_allFilteredTG_n{len(results[0]['targets'])}.tif"
            )
            named_copy_path = output_dir / named_copy
            fig_regulon.savefig(named_copy_path, dpi=settings["network_dpi"], bbox_inches="tight")
            saved_figures.append(str(named_copy_path))
    plt.close(fig_regulon)

    report = {
        "lineage": context["lineage"],
        "output_dir": str(output_dir),
        "checkpoint_path": inference["checkpoint_path"],
        "tf_symbols": settings["tf_symbols"],
        "saved_tables": saved_tables,
        "saved_figures": saved_figures,
        "results": [
            {
                "tf": str(result["tf"]),
                "tf_id": str(result["tf_id"]),
                "tf_idx": int(result["tf_idx"]),
                "t_top": int(result["t_top"]),
                "top_targets": result["targets"],
                "consensus": result["consensus"],
            }
            for result in results
        ],
        "settings": settings,
    }
    save_checkpoint_metadata(output_dir / "single_tf_report.json", report)
    return report


def run_single_tf_analysis(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        loaded = load_yaml_config(config)
        report = run_single_tf_analysis_from_config(loaded)
        report["config_path"] = str(Path(config).resolve())
        return report
    return run_single_tf_analysis_from_config(config)
