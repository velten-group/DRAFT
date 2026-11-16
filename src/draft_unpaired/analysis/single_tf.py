"""Single-TF activity analysis for unpaired TF-TG trajectories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from draft_paired.analysis.gene_keys import match_symbol
from draft_paired.analysis.repro import build_generic_labels, safe_label
from draft_paired.input_schema import map_gene_names, read_gene_table

from ..checkpoint import save_checkpoint_metadata
from ..infer import run_inference
from ..models.gnn import cosine_abs_prob
from ..preprocess import prepare_common_inputs
from ..utils import ensure_directory, load_yaml_config, resolve_config_path


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _single_tf_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config, "single_tf")
    lineage = str(config.get("lineage", "unpaired"))
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
        "consensus": _section(section, "consensus"),
        "topk": int(section.get("topk", 500)),
        "threshold": float(section.get("threshold", 0.5)),
        "min_expr_frac": float(section.get("min_expr_frac", 0.1)),
        "max_expr_frac": float(section.get("max_expr_frac", 0.8)),
        "time_for_top": section.get("time_for_top", 0),
        "topn_tg": section.get("topn_tg", 50),
        "user_thresh": float(section.get("user_thresh", 0.20)),
        "save_fig": bool(section.get("save_fig", True)),
        "save_tables": bool(section.get("save_tables", True)),
        "activity_dpi": int(section.get("activity_dpi", 300)),
        "network_dpi": int(section.get("network_dpi", 300)),
    }


def _cluster_ids(config: dict[str, Any], time_count: int) -> list[str]:
    reproduce_cfg = _section(config, "reproduce")
    provided = reproduce_cfg.get("cluster_ids")
    if isinstance(provided, list) and len(provided) == time_count:
        return [str(value) for value in provided]
    return [f"t{idx}" for idx in range(time_count)]


def centrality_from_prob(
    values: np.ndarray,
    mode: str = "topk_sum",
    topk: int = 500,
    thresh: float = 0.5,
) -> float:
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


def resolve_time_index(time_spec: Any, labels: list[str], time_count: int) -> int:
    if isinstance(time_spec, (int, np.integer)):
        value = int(time_spec)
        if not 0 <= value < time_count:
            raise ValueError(f"time={value} is outside valid range [0, {time_count - 1}].")
        return value
    if isinstance(time_spec, str):
        if time_spec in labels:
            return labels.index(time_spec)
        match = re.search(r"(\d+)$", time_spec)
        if match:
            value = int(match.group(1))
            if 0 <= value < time_count:
                return value
    raise ValueError(f"Could not resolve single_tf.time_for_top={time_spec!r}.")


def _label_to_cells(common: Any) -> dict[str, list[str]]:
    anno = common.anno
    expr_cols = set(common.expr_df.columns.astype(str))
    mapping: dict[str, list[str]] = {}
    for label in pd.unique(anno["cell_type"].astype(str)):
        cells = anno.loc[anno["cell_type"].astype(str) == str(label), "cell_id"].astype(str).tolist()
        mapping[str(label)] = [cell for cell in cells if cell in expr_cols]
    return mapping


def _keep_masks(
    labels: list[str],
    tg_names: list[str],
    expr_df: pd.DataFrame,
    label2cells: dict[str, list[str]],
    min_expr_frac: float,
    max_expr_frac: float,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    keep_masks: list[np.ndarray | None] = [None] * len(labels)
    frac_list: list[np.ndarray | None] = [None] * len(labels)
    for time_idx, label in enumerate(labels):
        cells = label2cells.get(str(label), [])
        if not cells:
            continue
        sub = expr_df.loc[tg_names, cells]
        fracs = (sub.to_numpy(dtype=float) > 0).mean(axis=1)
        keep = (fracs >= min_expr_frac) & (fracs <= max_expr_frac)
        keep_masks[time_idx] = keep
        frac_list[time_idx] = fracs
    return keep_masks, frac_list


def _activity_table(
    tf_name: str,
    tf_idx: int,
    labels: list[str],
    keep_masks: list[np.ndarray | None],
    label2cells: dict[str, list[str]],
    expr_df: pd.DataFrame,
    prob_mats: list[np.ndarray],
    settings: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for time_idx, label in enumerate(labels):
        keep = keep_masks[time_idx]
        cells = label2cells.get(str(label), [])
        if keep is None or not cells:
            rows.append(
                {
                    "t": int(time_idx),
                    "time_label": str(label),
                    "time_num": int(time_idx),
                    "centrality": np.nan,
                    "mean_expr": np.nan,
                    "activity": np.nan,
                    "expression_weighted_activity": np.nan,
                }
            )
            continue
        probs = prob_mats[time_idx][tf_idx]
        centrality = centrality_from_prob(
            probs[keep],
            mode=settings["centrality_mode"],
            topk=settings["topk"],
            thresh=settings["threshold"],
        )
        mean_expr = float(np.asarray(expr_df.loc[tf_name, cells], dtype=float).mean()) if tf_name in expr_df.index else np.nan
        rows.append(
            {
                "t": int(time_idx),
                "time_label": str(label),
                "time_num": int(time_idx),
                "centrality": float(centrality),
                "mean_expr": float(mean_expr),
                "activity": float(centrality),
                "expression_weighted_activity": float(centrality * mean_expr) if np.isfinite(mean_expr) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _top_tg_table(
    tf_idx: int,
    tf_name: str,
    time_idx: int,
    label: str,
    keep: np.ndarray | None,
    fracs: np.ndarray | None,
    tg_names: list[str],
    prob_mats: list[np.ndarray],
    topn: int,
    user_thresh: float,
) -> tuple[pd.DataFrame, list[str]]:
    if keep is None or fracs is None:
        return pd.DataFrame(), []
    probs = prob_mats[time_idx][tf_idx]
    cand_idx = np.where(keep)[0]
    if cand_idx.size == 0:
        return pd.DataFrame(), []
    ordered = cand_idx[np.argsort(-probs[cand_idx], kind="mergesort")]
    top_idx = ordered[: min(topn, len(ordered))]
    rows = []
    for rank, tg_idx in enumerate(top_idx.tolist(), start=1):
        rows.append(
            {
                "tf_name": tf_name,
                "t": int(time_idx),
                "time_label": str(label),
                "rank": int(rank),
                "tg_name": str(tg_names[tg_idx]),
                "score": float(probs[tg_idx]),
                "expr_frac_>0": float(fracs[tg_idx]),
                "expr_ge_user_thresh": bool(fracs[tg_idx] >= user_thresh),
            }
        )
    table = pd.DataFrame(rows)
    targets = table["tg_name"].astype(str).tolist() if not table.empty else []
    return table, targets


def _plot_activity(ax: Any, activity: pd.DataFrame, tf_name: str) -> None:
    from draft_paired.analysis.single_tf import _plot_activity_on_ax

    _plot_activity_on_ax(ax, activity, tf_name, tf_name, 18, 12)


def _draw_regulon(ax: Any, tf_name: str, targets: list[str], time_idx: int) -> None:
    ax.set_title(f"{tf_name} regulon @ t={time_idx}")
    if not targets:
        ax.text(0.5, 0.5, "No targets", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    graph = nx.DiGraph()
    graph.add_node(tf_name, kind="tf")
    for target in targets:
        graph.add_node(target, kind="tg")
        graph.add_edge(tf_name, target)

    angle = np.linspace(0, 2 * np.pi, len(targets), endpoint=False)
    positions = {tf_name: (0.0, 0.0)}
    for idx, target in enumerate(targets):
        positions[target] = (float(np.cos(angle[idx]) * 3.5), float(np.sin(angle[idx]) * 3.5))

    nx.draw_networkx_edges(graph, positions, ax=ax, width=0.8, alpha=0.5, arrows=False)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[tf_name],
        node_size=900,
        node_color="#ffd1b8",
        edgecolors="#444444",
        linewidths=1.5,
        node_shape="h",
        ax=ax,
    )
    tg_nodes = [node for node in graph.nodes if node != tf_name]
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=tg_nodes,
        node_size=260 if len(targets) <= 50 else 180,
        node_color="#7db5ff",
        edgecolors="#ffffff",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, labels={tf_name: tf_name}, font_size=14, font_weight="bold", ax=ax)
    for target in tg_nodes:
        x_pos, y_pos = positions[target]
        ax.text(x_pos, y_pos, target, ha="center", va="center", fontsize=10)
    ax.set_aspect("equal")
    ax.set_axis_off()


def run_single_tf_analysis_from_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = _single_tf_settings(config)
    if not settings["tf_symbols"]:
        raise ValueError(
            "Select TFs with --tf TF_NAME, or set single_tf.tf_symbols in the config."
        )

    inference = run_inference(config)
    common = prepare_common_inputs(config)
    output_dir = ensure_directory(resolve_config_path(config, settings["output_dir"]))

    tf_names = [str(value) for value in common.tf_list]
    tg_names = [str(value) for value in common.tg_list]
    time_count = int(inference["out_tf"].size(1))
    labels = build_generic_labels(
        time_count,
        time_names=inference.get("time_names"),
        cluster_ids=_cluster_ids(config, time_count),
    )
    label2cells = _label_to_cells(common)
    keep_masks, frac_list = _keep_masks(
        labels=labels,
        tg_names=tg_names,
        expr_df=common.expr_df,
        label2cells=label2cells,
        min_expr_frac=settings["min_expr_frac"],
        max_expr_frac=settings["max_expr_frac"],
    )
    prob_mats, _ = cosine_abs_prob(inference["out_tf"], inference["out_tg"])
    prob_np = [item.detach().cpu().numpy().astype(np.float32, copy=False) for item in prob_mats]

    tf_table = read_gene_table(common.input_paths.tf_list)
    requested_ids = map_gene_names(pd.Series(settings["tf_symbols"], dtype=str), tf_table)
    results: list[dict[str, Any]] = []
    saved_tables: list[str] = []
    saved_figures: list[str] = []
    for tf_symbol, requested_id in zip(settings["tf_symbols"], requested_ids):
        # Exact spelling first; case is ignored only where that names one TF.
        tf_idx = match_symbol(requested_id, tf_names)
        if tf_idx is None:
            raise RuntimeError(f"Could not resolve TF '{tf_symbol}' inside the TF graph namespace.")
        tf_name = tf_names[tf_idx]

        activity = _activity_table(
            tf_name=tf_name,
            tf_idx=tf_idx,
            labels=labels,
            keep_masks=keep_masks,
            label2cells=label2cells,
            expr_df=common.expr_df,
            prob_mats=prob_np,
            settings=settings,
        )
        time_idx_top = resolve_time_index(settings["time_for_top"], labels, time_count)
        top_table, targets = _top_tg_table(
            tf_idx=tf_idx,
            tf_name=tf_name,
            time_idx=time_idx_top,
            label=str(labels[time_idx_top]),
            keep=keep_masks[time_idx_top],
            fracs=frac_list[time_idx_top],
            tg_names=tg_names,
            prob_mats=prob_np,
            topn=int(settings["topn_tg"]),
            user_thresh=settings["user_thresh"],
        )
        consensus = None
        if settings["consensus"].get("enabled", False):
            from draft_paired.analysis.consensus import consensus_times, stage_rankings, export_consensus

            spec = settings["consensus"]
            tables = stage_rankings(
                inference["out_tf"].detach().cpu().numpy(), inference["out_tg"].detach().cpu().numpy(),
                tf_idx, tg_names, tg_names, frac_list, consensus_times(spec, labels), spec,
                min_fraction=settings["min_expr_frac"], max_fraction=settings["max_expr_frac"],
                absolute_cosine=True, parents=inference.get("parents_list"))
            consensus = export_consensus(config, spec, tf_name, tf_name, tf_idx, tables, output_dir)
            saved_tables.extend(consensus[key] for key in ("all_targets", "top_targets", "stage_ranks"))

        if settings["save_tables"]:
            activity_path = output_dir / f"{safe_label(tf_name)}_activity.tsv"
            top_path = output_dir / f"{safe_label(tf_name)}_top_tg.tsv"
            activity.to_csv(activity_path, sep="\t", index=False)
            top_table.to_csv(top_path, sep="\t", index=False)
            saved_tables.extend([str(activity_path), str(top_path)])

        if settings["save_fig"]:
            if consensus is not None:
                fig, ax = plt.subplots(figsize=(9, 9))
                _draw_regulon(ax, tf_name, consensus["target_names"], "consensus")
                ax.set_title(f"{tf_name} consensus regulon ({consensus['rank_method']})")
                path = output_dir / f"{safe_label(tf_name)}_consensus_regulon.png"
                fig.savefig(path, dpi=settings["network_dpi"], bbox_inches="tight")
                plt.close(fig)
                saved_figures.append(str(path))
            fig_act, ax_act = plt.subplots(figsize=(6.0, 4.5))
            _plot_activity(ax_act, activity, tf_name)
            act_path = output_dir / f"{safe_label(tf_name)}_activity.png"
            fig_act.tight_layout()
            fig_act.savefig(act_path, dpi=settings["activity_dpi"], bbox_inches="tight")
            plt.close(fig_act)
            saved_figures.append(str(act_path))

            fig_reg, ax_reg = plt.subplots(figsize=(7.0, 7.0))
            _draw_regulon(ax_reg, tf_name, targets, time_idx_top)
            reg_path = output_dir / f"{safe_label(tf_name)}_regulon.png"
            fig_reg.tight_layout()
            fig_reg.savefig(reg_path, dpi=settings["network_dpi"], bbox_inches="tight")
            plt.close(fig_reg)
            saved_figures.append(str(reg_path))

        results.append(
            {
                "tf": tf_name,
                "tf_idx": int(tf_idx),
                "t_top": int(time_idx_top),
                "top_targets": targets,
                "consensus": consensus,
            }
        )

    report = {
        "lineage": str(config.get("lineage", "unpaired")),
        "output_dir": str(output_dir),
        "checkpoint_path": inference["checkpoint_path"],
        "tf_symbols": settings["tf_symbols"],
        "saved_tables": saved_tables,
        "saved_figures": saved_figures,
        "results": results,
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


__all__ = ["run_single_tf_analysis", "run_single_tf_analysis_from_config"]
