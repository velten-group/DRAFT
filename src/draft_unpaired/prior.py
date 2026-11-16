"""Prepare staged TF--TG candidates from annotated, independently sampled ATAC.

Combine peak-matrix accessibility with supplied TF--RE and RE--TG relations
to select candidate TF--TG edges for each tree node.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from draft_paired.input_schema import (
    canonical_cell_annotation,
    find_column,
    map_gene_names,
    read_cell_annotation,
    read_gene_table,
    read_table,
    select_matrix,
)
from draft_paired.lineage import load_tree_table
from draft_paired.utils import load_yaml_config, resolve_config_path


def _assay_annotation(adata, config, key):
    value = config.get("data", {}).get(key)
    if value:
        annotation = read_cell_annotation(resolve_config_path(config, value))
    else:
        frame = adata.obs.copy()
        frame["cell_id"] = adata.obs_names.astype(str)
        annotation = canonical_cell_annotation(frame)
    if not annotation.cell_id.isin(adata.obs_names).all():
        raise ValueError(f"{key} contains cells absent from its own H5AD file.")
    return annotation


def prepare_stage_prior(
    config_or_path: dict[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    config = (
        copy.deepcopy(config_or_path)
        if isinstance(config_or_path, dict)
        else load_yaml_config(config_or_path)
    )
    data = config.get("data", {})
    settings = config.get("prior_preparation", {})
    method = settings.get("method", "accessible_peak_path")
    if method == "scmtni":
        from .scmtni_prior import prepare_scmtni_prior

        return prepare_scmtni_prior(config, output_dir, force=force)
    if method != "accessible_peak_path":
        raise ValueError(f"Unknown prior_preparation.method: {method!r}")
    if settings.get("min_accessible_fraction") is None:
        raise ValueError(
            "Pass --set prior_preparation.min_accessible_fraction=VALUE for your ATAC data."
        )
    fraction = float(settings["min_accessible_fraction"])
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("min_accessible_fraction must be between 0 and 1.")
    score_threshold = float(data.get("prior_score_threshold", 0.1))
    if not np.isfinite(score_threshold) or score_threshold >= 1:
        raise ValueError(
            "Binary prepared candidates require prior_score_threshold < 1."
        )
    required = (
        "rna_h5ad",
        "atac_h5ad",
        "tree",
        "tf_list",
        "tg_list",
        "tf_re_prior",
        "re_tg_prior",
    )
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(f"Prior preparation requires data keys: {', '.join(missing)}")
    paths = {key: resolve_config_path(config, data[key]) for key in required}
    for key in ("node_tf_scope", "node_tg_scope", "annotation", "atac_annotation"):
        if data.get(key):
            path = resolve_config_path(config, data[key])
            if path.exists():
                paths[key] = path
    out = resolve_config_path(config, output_dir)
    names = (
        "tf_tg_prior.tsv",
        "node_tf_scope.tsv",
        "node_tg_scope.tsv",
        "prior_report.json",
        "unpaired.yaml",
        "rna_annotation.tsv",
        "atac_annotation.tsv",
    )
    collisions = [out / name for name in names if (out / name).exists()]
    if collisions and not force:
        raise FileExistsError(
            f"{collisions[0]} already exists; use --force to overwrite prepared files."
        )
    if any((out / name).resolve() in set(paths.values()) for name in names):
        raise ValueError("Prepared outputs must not overwrite input files.")
    tree = load_tree_table(paths["tree"], strict=True)
    rna = ad.read_h5ad(paths["rna_h5ad"], backed="r")
    try:
        if not rna.obs_names.is_unique or not rna.var_names.is_unique:
            raise ValueError("RNA cell and gene identifiers must be unique.")
        rna_annotation = _assay_annotation(rna, config, "annotation")
        for node in tree.node_id:
            if not rna_annotation.cell_type.eq(node).any():
                raise ValueError(f"No annotated RNA cells for tree node {node!r}.")
    finally:
        rna.file.close()
    tf_table, tg_table = (read_gene_table(paths[key]) for key in ("tf_list", "tg_list"))
    relations = []
    input_counts = {}
    for key, gene_kind, genes in (
        ("tf_re_prior", "TF", tf_table),
        ("re_tg_prior", "TG", tg_table),
    ):
        frame = read_table(paths[key])
        re_col = find_column(frame, ("RE", "peak", "region"))
        gene_col = find_column(
            frame,
            (
                gene_kind,
                "source" if gene_kind == "TF" else "target",
                "gene",
                "gene_name",
            ),
        )
        if re_col is None or gene_col is None:
            raise ValueError(f"{paths[key]} requires RE and {gene_kind} columns.")
        relation = pd.DataFrame(
            {
                "RE": frame[re_col].str.strip(),
                gene_kind: map_gene_names(frame[gene_col], genes),
            }
        )
        relation = relation[relation[gene_kind].isin(genes.gene_id)].drop_duplicates()
        input_counts[key] = {
            "input_rows": len(frame),
            "gene_matched_rows": len(relation),
        }
        relations.append(relation)
    tf_re, re_tg = relations
    atac = ad.read_h5ad(paths["atac_h5ad"])
    if not atac.obs_names.is_unique or not atac.var_names.is_unique:
        raise ValueError("ATAC cell and peak identifiers must be unique.")
    annotation = _assay_annotation(atac, config, "atac_annotation")
    matrix, used_layer = select_matrix(atac, data.get("atac_layer", "counts"))
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(
            "ATAC accessibility must contain finite nonnegative counts/binary values."
        )
    cell_indices = {str(cell): idx for idx, cell in enumerate(atac.obs_names)}
    annotation = annotation[annotation.cell_id.isin(cell_indices)]
    peaks = np.asarray(atac.var_names.astype(str))
    peak_set = set(peaks)
    for key, relation in zip(("tf_re_prior", "re_tg_prior"), relations):
        input_counts[key]["peak_matched_rows"] = int(relation.RE.isin(peak_set).sum())
    # Match region names literally; do not silently combine nearby genomic regions.
    tf_re = tf_re[tf_re.RE.isin(peak_set)]
    re_tg = re_tg[re_tg.RE.isin(peak_set)]
    if tf_re.empty or re_tg.empty:
        raise ValueError(
            "TF--RE/RE--TG relations have no usable ATAC peaks; RE names must match ATAC var_names."
        )
    outputs, node_report = [], []
    for node in tree.node_id:
        cells = annotation.loc[annotation.cell_type == node, "cell_id"]
        if cells.empty:
            raise ValueError(f"No annotated ATAC cells for tree node {node!r}.")
        idx = [cell_indices[cell] for cell in cells]
        detected = np.asarray((matrix[idx, :] > 0).mean(axis=0)).ravel()
        active = set(peaks[(detected > 0) & (detected >= fraction)])
        joined = tf_re[tf_re.RE.isin(active)].merge(re_tg, on="RE", how="inner")
        candidates = (
            joined.groupby(["TF", "TG"], sort=True)
            .RE.nunique()
            .reset_index(name="n_supported_RE")
        )
        if candidates.empty:
            raise ValueError(
                f"No TF--RE--TG paths through accessible peaks for node {node!r}."
            )
        candidates.insert(0, "node_id", node)
        candidates["score"] = 1.0
        outputs.append(candidates[["node_id", "TF", "TG", "score", "n_supported_RE"]])
        node_report.append(
            {
                "node_id": node,
                "rna_cells": int(rna_annotation.cell_type.eq(node).sum()),
                "atac_cells": len(cells),
                "accessible_RE": len(active),
                "candidate_edges": len(candidates),
            }
        )
    prior = pd.concat(outputs, ignore_index=True)
    scopes, scope_policy = {}, {}
    for kind, genes in (("TF", tf_table), ("TG", tg_table)):
        key = f"node_{kind.lower()}_scope"
        if key in paths:
            from .preprocess import _prepare_node_scope_table

            grouped = _prepare_node_scope_table(
                paths[key], set(tree.node_id), set(genes.gene_id), kind, genes
            )
            if any(not grouped.get(node) for node in tree.node_id):
                raise ValueError(
                    f"{paths[key]} needs scope entries for every tree node."
                )
            scopes[kind] = pd.DataFrame(
                [(node, gene) for node in tree.node_id for gene in grouped[node]],
                columns=["node_id", kind],
            )
            scope_policy[kind] = "preserved supplied scope"
        else:
            scopes[kind] = prior[["node_id", kind]].drop_duplicates()
            scope_policy[kind] = (
                "candidate-derived coverage, not expression-derived activity"
            )
    # Resolve data paths before writing a directly usable training config.
    prepared = copy.deepcopy(config)
    prepared["base_dir"] = str(resolve_config_path(config, "."))
    prepared_data = prepared.setdefault("data", {})
    prepared_data.update(
        annotation=str(out / "rna_annotation.tsv"),
        atac_annotation=str(out / "atac_annotation.tsv"),
        tf_tg_prior=str(out / "tf_tg_prior.tsv"),
        node_tf_scope=str(out / "node_tf_scope.tsv"),
        node_tg_scope=str(out / "node_tg_scope.tsv"),
    )
    prepared.setdefault("prior_preparation", {})["enabled"] = False
    report = {
        "method": "accessible_peak_path",
        "min_accessible_fraction": fraction,
        "atac_layer": used_layer,
        "cell_matching": "ATAC annotation to ATAC obs_names; RNA cell IDs are independent",
        "score_policy": "binary candidate presence; source scores are not used as training weights",
        "scope_policy": scope_policy,
        "not_scmtni_reproduction": "No BAM peak calling, promoter assignment, motif rescanning, top-density selection or percentile scoring is performed.",
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "relation_counts": input_counts,
        "nodes": node_report,
        "outputs": {name: str(out / name) for name in names},
    }
    out.mkdir(parents=True, exist_ok=True)
    prior.to_csv(out / "tf_tg_prior.tsv", sep="\t", index=False)
    rna_annotation.to_csv(out / "rna_annotation.tsv", sep="\t", index=False)
    annotation.to_csv(out / "atac_annotation.tsv", sep="\t", index=False)
    for kind in ("TF", "TG"):
        scopes[kind].to_csv(
            out / f"node_{kind.lower()}_scope.tsv", sep="\t", index=False
        )
    (out / "prior_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "unpaired.yaml").write_text(
        yaml.safe_dump(prepared, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {
        "reused": False,
        "output_dir": str(out),
        "config_path": str(out / "unpaired.yaml"),
        "report": report,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_training_config(config_path: str | Path):
    """Resolve optional preprocessing once, then use its config for every step."""
    config = load_yaml_config(config_path)
    settings = config.get("prior_preparation", {})
    if not settings.get("enabled", False):
        return config_path, None
    output = settings.get("output_dir", f"result/{config.get('lineage', 'lineage1')}/prepared_prior")
    result = prepare_stage_prior(config, output, force=bool(settings.get("force", False)))
    return result["config_path"], result
