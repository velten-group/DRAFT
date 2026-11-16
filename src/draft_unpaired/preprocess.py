"""Preprocessing for unpaired TF-TG lineage training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from draft_paired.input_schema import (
    map_gene_names,
    read_cell_annotation,
    read_gene_table,
    read_table,
    select_matrix,
)
from draft_paired.lineage import load_tree_table

from .io import InputPaths, strip_copy_suffix


@dataclass(frozen=True)
class CommonInputs:
    input_paths: InputPaths
    anno: pd.DataFrame
    tree_df: pd.DataFrame
    tf_list: list[str]
    tg_list: list[str]
    expr_df: pd.DataFrame
    prior_df: pd.DataFrame
    node_tf_scope: dict[str, list[str]]
    node_tg_scope: dict[str, list[str]]
    maps: dict[str, list[str]]
    device: torch.device
    summary_lines: list[str]


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_name_list(path: Path) -> list[str]:
    return pd.unique(read_gene_table(path).gene_id).tolist()


def _annotation_table(path: Path) -> pd.DataFrame:
    frame = read_cell_annotation(path)
    if "time" not in frame:
        frame["time"] = ""
    frame = frame[["cell_id", "time", "cell_type"] + [column for column in frame if column not in {"cell_id", "time", "cell_type"}]].copy()
    frame["cell_id"] = frame["cell_id"].astype(str).str.strip()
    frame["cell_type"] = frame["cell_type"].astype(str).str.strip()
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    frame = frame[frame["cell_id"].ne("") & frame["cell_type"].ne("")].drop_duplicates(subset=["cell_id"], keep="first")
    return frame.reset_index(drop=True)


def _clean_var_names(values: list[str]) -> list[str]:
    return [strip_copy_suffix(str(value)) for value in values]


def _extract_expr_subset(adata: ad.AnnData, genes: list[str], cells: list[str], layer: str | None = None) -> pd.DataFrame:
    obs_names = [str(value) for value in adata.obs_names.tolist()]
    var_names = _clean_var_names(adata.var_names.tolist())
    cell_to_idx = {cell: idx for idx, cell in enumerate(obs_names)}
    gene_to_idx: dict[str, int] = {}
    for idx, gene in enumerate(var_names):
        if gene not in gene_to_idx:
            gene_to_idx[gene] = idx

    valid_cells = [cell for cell in cells if cell in cell_to_idx]
    valid_genes = [gene for gene in genes if gene in gene_to_idx]
    if not valid_cells:
        raise ValueError("No annotation cells were found in the RNA h5ad file.")
    if not valid_genes:
        raise ValueError("None of the requested TF/TG genes were found in the RNA h5ad file.")

    cell_idx = [cell_to_idx[cell] for cell in valid_cells]
    gene_idx = [gene_to_idx[gene] for gene in valid_genes]
    matrix, _ = select_matrix(adata, layer)
    subset = matrix[cell_idx, :][:, gene_idx]
    if sp.issparse(subset):
        subset = subset.toarray()
    values = np.asarray(subset, dtype=np.float32)
    return pd.DataFrame(values.T, index=valid_genes, columns=valid_cells)


def _prepare_node_scope_table(
    path: Path,
    node_names: set[str],
    gene_set: set[str],
    gene_kind: str,
    gene_table: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    frame = read_table(path, header=None).fillna("")
    if frame.shape[1] < 2:
        raise ValueError(f"{path} must have at least two columns for node scope.")

    first_row = [str(value).strip() for value in frame.iloc[0].tolist()]
    known_headers = {
        "node_id",
        "cell_type",
        "cluster",
        "tf",
        "tg",
        "gene",
        "name",
        "regulator",
        "source",
        "target",
    }
    if any(value.lower() in known_headers for value in first_row):
        frame = frame.iloc[1:].copy()
        frame.columns = first_row
    else:
        frame = frame.copy()
        frame.columns = [f"col{idx}" for idx in range(frame.shape[1])]

    columns = {str(column).lower(): str(column) for column in frame.columns}
    node_col = columns.get("node_id") or columns.get("cell_type") or columns.get("cluster")
    if gene_kind == "TF":
        gene_col = (
            columns.get("tf")
            or columns.get("gene")
            or columns.get("name")
            or columns.get("regulator")
            or columns.get("source")
        )
    else:
        gene_col = (
            columns.get("tg")
            or columns.get("gene")
            or columns.get("name")
            or columns.get("target")
        )
    if node_col is None or gene_col is None:
        node_col = str(frame.columns[0])
        gene_col = str(frame.columns[1])

    scope = pd.DataFrame()
    scope["node_id"] = frame[node_col].astype(str).str.strip()
    scope["gene"] = frame[gene_col].astype(str).map(strip_copy_suffix).str.strip()
    if gene_table is not None:
        scope["gene"] = map_gene_names(scope["gene"], gene_table)
    scope = scope[scope["node_id"].isin(node_names) & scope["gene"].isin(gene_set)].copy()
    if scope.empty:
        raise ValueError(f"{path} did not contain any valid {gene_kind} scope entries after filtering.")

    grouped: dict[str, list[str]] = {}
    for node_id, group in scope.groupby("node_id", sort=False):
        grouped[str(node_id)] = pd.unique(group["gene"].astype(str)).tolist()
    return grouped


def _prepare_prior_table(
    path: Path,
    node_names: set[str],
    tf_set: set[str],
    tg_set: set[str],
    score_threshold: float,
    tf_table: pd.DataFrame | None = None,
    tg_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = read_table(path).fillna("")
    columns = {str(column).lower(): str(column) for column in frame.columns}

    node_col = columns.get("node_id") or columns.get("cell_type") or columns.get("cluster")
    tf_col = columns.get("tf") or columns.get("source")
    tg_col = columns.get("tg") or columns.get("target")
    score_col = columns.get("score")
    if node_col is None or tf_col is None or tg_col is None:
        raise ValueError(
            f"{path} must contain node_id/cell_type, TF, and TG columns. Columns={list(frame.columns)}"
        )

    prior = pd.DataFrame()
    prior["node_id"] = frame[node_col].astype(str).str.strip()
    prior["TF"] = frame[tf_col].astype(str).map(strip_copy_suffix).str.strip()
    prior["TG"] = frame[tg_col].astype(str).map(strip_copy_suffix).str.strip()
    if tf_table is not None:
        prior["TF"] = map_gene_names(prior["TF"], tf_table)
    if tg_table is not None:
        prior["TG"] = map_gene_names(prior["TG"], tg_table)
    if score_col is not None:
        prior["score"] = pd.to_numeric(frame[score_col], errors="coerce").fillna(1.0)
    else:
        prior["score"] = 1.0

    prior = prior[prior["score"] > float(score_threshold)].copy()
    prior = prior[
        prior["node_id"].isin(node_names) & prior["TF"].isin(tf_set) & prior["TG"].isin(tg_set)
    ].copy()
    prior = prior.sort_values("score", ascending=False).drop_duplicates(["node_id", "TF", "TG"], keep="first")
    return prior.reset_index(drop=True)


def prepare_common_inputs(
    config: dict[str, Any], *, device: torch.device | None = None,
) -> CommonInputs:
    paths = InputPaths.from_config(config)
    device = device if device is not None else pick_device()

    anno = _annotation_table(paths.annotation)
    tree_df = load_tree_table(paths.tree, strict=int(config.get("config_version", 1)) >= 2, sort=False)

    node_names = tree_df["node_id"].astype(str).tolist()
    anno = anno[anno["cell_type"].isin(node_names)].copy()
    if anno.empty:
        raise ValueError("Annotation has no cells whose cell_type matches tree.node_id.")

    tf_table = read_gene_table(paths.tf_list)
    tg_table = read_gene_table(paths.tg_list)
    tf_list = pd.unique(tf_table.gene_id).tolist()
    tg_list = pd.unique(tg_table.gene_id).tolist()
    adata = ad.read_h5ad(paths.rna_h5ad)
    rna_layer = config.get("data", {}).get("rna_layer")
    expr_df = _extract_expr_subset(adata, list(dict.fromkeys(tf_list + tg_list)), anno["cell_id"].tolist(), rna_layer)

    tf_list = [gene for gene in tf_list if gene in expr_df.index]
    tg_list = [gene for gene in tg_list if gene in expr_df.index]
    if not tf_list:
        raise ValueError("After intersecting with RNA genes, TF list became empty.")
    if not tg_list:
        raise ValueError("After intersecting with RNA genes, TG list became empty.")

    expr_df = expr_df.loc[list(dict.fromkeys(tf_list + tg_list)), :]
    anno = anno[anno["cell_id"].isin(expr_df.columns)].copy().reset_index(drop=True)
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    prior_score_threshold = float(data_cfg.get("prior_score_threshold", 0.1))
    prior_df = _prepare_prior_table(
        paths.tf_tg_prior,
        set(node_names),
        set(tf_list),
        set(tg_list),
        score_threshold=prior_score_threshold,
        tf_table=tf_table, tg_table=tg_table,
    )
    if prior_df.empty:
        raise ValueError("TF-TG prior became empty after filtering to the provided nodes/TF/TG lists.")

    node_tf_scope = (
        _prepare_node_scope_table(paths.node_tf_scope, set(node_names), set(tf_list), "TF", tf_table)
        if paths.node_tf_scope is not None
        else {}
    )
    node_tg_scope = (
        _prepare_node_scope_table(paths.node_tg_scope, set(node_names), set(tg_list), "TG", tg_table)
        if paths.node_tg_scope is not None
        else {}
    )
    if paths.node_tf_scope is not None:
        missing_tf_scope = [node_id for node_id in node_names if len(node_tf_scope.get(node_id, [])) == 0]
        if missing_tf_scope:
            raise ValueError(
                f"{paths.node_tf_scope} is missing TF scope entries for nodes: {', '.join(missing_tf_scope)}"
            )
    if paths.node_tg_scope is not None:
        missing_tg_scope = [node_id for node_id in node_names if len(node_tg_scope.get(node_id, [])) == 0]
        if missing_tg_scope:
            raise ValueError(
                f"{paths.node_tg_scope} is missing TG scope entries for nodes: {', '.join(missing_tg_scope)}"
            )

    tf_symbols = dict(zip(tf_table.gene_id, tf_table.gene_name))
    tg_symbols = dict(zip(tg_table.gene_id, tg_table.gene_name))
    maps = {
        "tf": list(tf_list),
        "tg": list(tg_list),
        "tf_ids": list(tf_list),
        "tf_names": [tf_symbols[gene] for gene in tf_list],
        "tg_ids": list(tg_list),
        "tg_names": [tg_symbols[gene] for gene in tg_list],
    }
    summary_lines = [
        f"RNA h5ad               : {paths.rna_h5ad.name}",
        f"RNA layer              : {select_matrix(adata, rna_layer)[1]}",
        f"Annotation             : {paths.annotation.name} | cells={len(anno)}",
        f"Tree                   : {paths.tree.name} | nodes={len(tree_df)}",
        f"TF list                : {len(tf_list)}",
        f"TG list                : {len(tg_list)}",
        f"TF-TG prior            : {paths.tf_tg_prior.name} | edges={len(prior_df)} | score>{prior_score_threshold:g}",
        (
            f"TF node scope          : {paths.node_tf_scope.name} | explicit"
            if paths.node_tf_scope is not None
            else "TF node scope          : prior-derived fallback"
        ),
        (
            f"TG node scope          : {paths.node_tg_scope.name} | explicit"
            if paths.node_tg_scope is not None
            else "TG node scope          : prior-derived fallback"
        ),
        f"Node order             : {', '.join(node_names)}",
        f"Device                 : {device}",
    ]

    return CommonInputs(
        input_paths=paths,
        anno=anno,
        tree_df=tree_df,
        tf_list=tf_list,
        tg_list=tg_list,
        expr_df=expr_df,
        prior_df=prior_df,
        node_tf_scope=node_tf_scope,
        node_tg_scope=node_tg_scope,
        maps=maps,
        device=device,
        summary_lines=summary_lines,
    )


__all__ = ["CommonInputs", "pick_device", "prepare_common_inputs"]
