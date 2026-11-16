"""scMTNI source-compatible prior construction from called peaks and motif hits.

Reference: Roy-lab/scMTNI c4d4850629f3529cc7a8b4dae54d91daf0702809,
Scripts/genPriorNetwork. Input group labels, called peaks and motif sites
define the stage-specific networks; rankEdges supplies the scoring rule.
"""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import os
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from draft_paired.input_schema import (
    canonical_cell_annotation,
    map_gene_names,
    read_cell_annotation,
    read_gene_table,
    read_table,
)
from draft_paired.lineage import load_tree_table
from draft_paired.utils import resolve_config_path

REFERENCE_COMMIT = "c4d4850629f3529cc7a8b4dae54d91daf0702809"
REFERENCE_URL = (
    f"https://github.com/Roy-lab/scMTNI/tree/{REFERENCE_COMMIT}/Scripts/genPriorNetwork"
)


def bed_rows(path: Path, kind: str):
    """Read headerless BED rows in file order with their genomic coordinates."""
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not fields or fields[0].startswith(("#", "track ", "browser ")):
                continue
            minimum = {"peak": 10, "motif": 6, "promoter": 4}[kind]
            if len(fields) < minimum:
                raise ValueError(
                    f"{path}:{line}: {kind} needs at least {minimum} tab-separated columns."
                )
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line}: BED coordinates must be integers."
                ) from exc
            if start < 0 or end <= start:
                raise ValueError(
                    f"{path}:{line}: BED must use nonempty, nonnegative half-open intervals."
                )
            yield fields[0], start, end, fields


class IntervalIndex:
    """Find every half-open overlap, including nested/overlapping peak intervals."""

    def __init__(self, records):
        groups = defaultdict(list)
        for idx, (chrom, start, end, *_) in enumerate(records):
            groups[chrom].append((start, end, idx))
        self.groups = {}
        for chrom, values in groups.items():
            values.sort()
            starts, ends, indices = (
                np.asarray(v, dtype=np.int64) for v in zip(*values)
            )
            self.groups[chrom] = starts, ends, np.maximum.accumulate(ends), indices

    def overlaps(self, chrom: str, start: int, end: int) -> list[int]:
        if chrom not in self.groups:
            return []
        starts, ends, prefix_end, indices = self.groups[chrom]
        lo = int(np.searchsorted(prefix_end, start, side="right"))
        hi = int(np.searchsorted(starts, end, side="left"))
        return sorted(indices[lo:hi][ends[lo:hi] > start].tolist())


def source_scores(
    network: dict[tuple[str, str], float], regulators: set[str], targets: set[str]
):
    """Apply mapMot2Gene rounding, density filtering and rankEdges incr scoring.

    The C++ `incr` rule assigns larger percentiles to lower values in an
    ascending sort.
    The R 1:0 slice retains the first row when round(0.2*TF*TG) is zero.
    """
    raw = [(tf, tg, float(f"{score:.6f}")) for (tf, tg), score in network.items()]
    filtered = [row for row in raw if row[0] in regulators and row[1] in targets]
    budget = round(0.2 * len({r[0] for r in filtered}) * len({r[1] for r in filtered}))
    top = sorted(filtered, key=lambda row: -row[2])[: max(1, budget)]
    by_tf = defaultdict(dict)
    for tf, tg, value in top:
        by_tf[tf][tg] = abs(value)
    ranked = []
    for tf in sorted(by_tf):
        groups = defaultdict(list)
        for tg, value in by_tf[tf].items():
            groups[value].append(tg)
        size = len(by_tf[tf])
        left = 0
        for value in sorted(groups):
            genes = sorted(groups[value])
            count = len(genes)
            score = (size - left - count + (count / 2 if count > 1 else 1)) / size
            score = float(f"{score:.6g}")  # C++ ostream default precision.
            ranked.extend((tf, tg, score) for tg in genes)
            left += count
    return raw, filtered, top, ranked, budget


def map_motifs(
    peaks, motif_file: Path, promoter_file: Path, motif_map: dict[str, list[str]]
):
    """Equivalent to both bedtools intersects followed by mapMot2Gene.makeNet."""
    index = IntervalIndex(peaks)
    peak_genes = [set() for _ in peaks]
    for chrom, start, end, fields in bed_rows(promoter_file, "promoter"):
        if not fields[-1].strip():
            raise ValueError(
                f"{promoter_file}: final column must contain the gene name."
            )
        for idx in index.overlaps(chrom, start, end):
            peak_genes[idx].add(fields[-1])
    peak_tf = [{} for _ in peaks]
    motif_rows = 0
    for chrom, start, end, fields in bed_rows(motif_file, "motif"):
        motif_rows += 1
        value = float(fields[-2])
        if not np.isfinite(value):
            raise ValueError(f"{motif_file}: motif scores must be finite.")
        tfs = motif_map.get(fields[-3], [])
        if not tfs:
            continue
        for idx in index.overlaps(chrom, start, end):
            if not peak_genes[idx]:
                continue
            for tf in tfs:
                previous = peak_tf[idx].get(tf)
                if previous is None or value > previous:
                    peak_tf[idx][tf] = value
    networks = defaultdict(dict)
    for idx, (_, _, _, node, _) in enumerate(peaks):
        # Stable density sorting retains the input order for tied values.
        for tf, value in peak_tf[idx].items():
            for gene in peak_genes[idx]:
                key = tf, gene
                if key not in networks[node] or value > networks[node][key]:
                    networks[node][key] = value
    return networks, peak_tf, peak_genes, motif_rows


def _annotation(adata, data, key, paths, config):
    if data.get(key):
        paths[key] = resolve_config_path(config, data[key])
        return read_cell_annotation(paths[key])
    frame = adata.obs.copy()
    frame["cell_id"] = adata.obs_names.astype(str)
    return canonical_cell_annotation(frame)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_scmtni_prior(config: dict[str, Any], output_dir, *, force=False):
    from .preprocess import _prepare_node_scope_table

    config = copy.deepcopy(config)
    data, settings = config.get("data", {}), config.get("prior_preparation", {})
    required = (
        "rna_h5ad",
        "atac_h5ad",
        "tree",
        "tf_list",
        "tg_list",
        "stage_peaks",
        "motif_instances",
        "motif_to_tf",
        "promoters",
    )
    if any(not data.get(key) for key in required):
        raise ValueError(
            "scMTNI mode requires data: "
            + ", ".join(required)
            + ". stage_peaks must identify original per-node MACS2 narrowPeak files; h5ad detection fractions are not equivalent."
        )
    if "min_accessible_fraction" in settings:
        raise ValueError(
            "scMTNI mode uses called stage peaks, not min_accessible_fraction."
        )
    if not data.get("genome_assembly"):
        raise ValueError(
            "Declare data.genome_assembly; peaks, motif hits and promoters must use the same assembly."
        )
    paths = {key: resolve_config_path(config, data[key]) for key in required}
    tree = load_tree_table(paths["tree"], strict=True)
    manifest = read_table(paths["stage_peaks"])
    if (
        not {"node_id", "peak_file"}.issubset(manifest.columns)
        or manifest.node_id.duplicated().any()
        or set(manifest.node_id) != set(tree.node_id)
    ):
        raise ValueError(
            "stage_peaks requires exactly one node_id,peak_file row per tree node."
        )
    for node, file in zip(manifest.node_id, manifest.peak_file):
        paths[f"peak:{node}"] = resolve_config_path(config, file)
    for key in ("node_tf_scope", "node_tg_scope", "annotation", "atac_annotation"):
        if data.get(key):
            paths[key] = resolve_config_path(config, data[key])
    if (
        not all(key in paths for key in ("node_tf_scope", "node_tg_scope"))
        and settings.get("shared_gene_universe") is not True
    ):
        raise ValueError(
            "Supply node_tf_scope/node_tg_scope or explicitly set prior_preparation.shared_gene_universe: true. Zero RNA values cannot recover per-node input gene universes."
        )
    input_meta = {
        key: {"path": str(path), "sha256": _hash(path)} for key, path in paths.items()
    }
    implementation_hashes = {
        "scmtni_prior": _hash(Path(__file__)),
        "input_schema": _hash(Path(read_gene_table.__code__.co_filename)),
        "lineage": _hash(Path(load_tree_table.__code__.co_filename)),
        "scope_loader": _hash(Path(_prepare_node_scope_table.__code__.co_filename)),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "inputs": input_meta,
                "config": config,
                "implementation": implementation_hashes,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    out = resolve_config_path(config, output_dir)
    if any(path.is_relative_to(out) for path in paths.values()):
        raise ValueError("Keep source inputs outside the prepared output directory.")
    report_path = out / "prior_report.json"
    if report_path.exists() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("fingerprint") != fingerprint:
            raise ValueError(
                "Prepared prior inputs/settings changed; use a new output directory or --force."
            )
        for relative, expected in report["output_sha256"].items():
            path = out / relative
            if not path.is_file() or _hash(path) != expected:
                raise ValueError(f"Prepared prior output changed or is missing: {path}")
        return {
            "output_dir": str(out),
            "config_path": str(out / "unpaired.yaml"),
            "report": report,
            "reused": True,
        }
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(
            f"{out} is not empty; choose a new directory or use --force."
        )
    rna, atac = (
        ad.read_h5ad(paths["rna_h5ad"], backed="r"),
        ad.read_h5ad(paths["atac_h5ad"], backed="r"),
    )
    try:
        for assay in (rna, atac):
            if not assay.obs_names.is_unique or not assay.var_names.is_unique:
                raise ValueError("H5AD cell and feature identifiers must be unique.")
        rna_annotation = _annotation(rna, data, "annotation", paths, config)
        atac_annotation = _annotation(atac, data, "atac_annotation", paths, config)
        counts = {}
        for label, assay, annotation in (
            ("RNA", rna, rna_annotation),
            ("ATAC", atac, atac_annotation),
        ):
            if not annotation.cell_id.isin(assay.obs_names).all():
                raise ValueError(
                    f"{label} annotation contains cells absent from its h5ad."
                )
            counts[label] = annotation.cell_type.value_counts().to_dict()
            if any(not counts[label].get(node) for node in tree.node_id):
                raise ValueError(f"Every tree node requires annotated {label} cells.")
        gene_tables = {
            kind: read_gene_table(paths[f"{kind.lower()}_list"])
            for kind in ("TF", "TG")
        }
        scopes = {}
        for kind, genes in gene_tables.items():
            genes = genes[genes.gene_id.isin(rna.var_names)].copy()
            gene_tables[kind] = genes
            key = f"node_{kind.lower()}_scope"
            scopes[kind] = (
                _prepare_node_scope_table(
                    paths[key], set(tree.node_id), set(genes.gene_id), kind, genes
                )
                if key in paths
                else {node: genes.gene_id.tolist() for node in tree.node_id}
            )
            if any(not scopes[kind].get(node) for node in tree.node_id):
                raise ValueError(f"Missing RNA-matched {kind} scope for a tree node.")
    finally:
        rna.file.close()
        atac.file.close()
    mapping = {}
    with paths["motif_to_tf"].open(encoding="utf-8-sig") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 2 or not row[0] or not row[1]:
                raise ValueError(
                    "motif_to_tf must be headerless motif_ID<TAB>TF1::TF2."
                )
            mapping[row[0]] = row[1].split("::")
    peaks = []
    for node in tree.node_id:
        seen = set()
        for chrom, start, end, fields in bed_rows(paths[f"peak:{node}"], "peak"):
            if fields[3] in seen:
                raise ValueError(f"Duplicate peak ID within node {node}: {fields[3]}")
            seen.add(fields[3])
            peaks.append((chrom, start, end, node, fields[3]))
        if not seen:
            raise ValueError(f"No called peaks for {node}.")
    networks, peak_tf, peak_genes, motif_rows = map_motifs(
        peaks, paths["motif_instances"], paths["promoters"], mapping
    )

    # Resolve symbols to the shared RNA identifiers without changing score order.
    def aliases(kind, values):
        unique = list(dict.fromkeys(values))
        mapped = map_gene_names(
            pd.Series(unique, dtype=str), gene_tables[kind]
        ).tolist()
        if len(set(mapped)) != len(mapped):
            raise ValueError(
                f"Multiple {kind} names map to the same RNA ID; use a one-to-one symbol/ID mapping."
            )
        return dict(zip(unique, mapped))

    tf_alias = aliases("TF", [tf for net in networks.values() for tf, _ in net])
    tg_alias = aliases("TG", [tg for net in networks.values() for _, tg in net])
    stage_results, prior_rows, node_reports = [], [], []
    threshold = float(data.get("prior_score_threshold", 0.1))
    for node in tree.node_id:
        network = {
            (tf_alias[tf], tg_alias[tg]): score
            for (tf, tg), score in networks[node].items()
        }
        result = source_scores(
            network, set(scopes["TF"][node]), set(scopes["TG"][node])
        )
        raw, filtered, _top, ranked, budget = result
        if not ranked or not any(value > threshold for _, _, value in ranked):
            raise ValueError(
                f"No scMTNI-scored candidates survive DRAFT's score filter at node {node}."
            )
        stage_results.append(result)
        prior_rows.extend((node, tf, tg, value) for tf, tg, value in ranked)
        node_reports.append(
            {
                "node_id": node,
                "rna_cells": counts["RNA"][node],
                "atac_cells": counts["ATAC"][node],
                "motif_edges": len(raw),
                "scope_filtered_edges": len(filtered),
                "source_density_budget": budget,
                "ranked_edges": len(ranked),
                "training_edges": sum(value > threshold for _, _, value in ranked),
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    output_files = []

    def save_table(relative, frame, *, header=True, float_format=None):
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(
            path, sep="\t", index=False, header=header, float_format=float_format
        )
        output_files.append(relative)

    save_table(
        "tf_tg_prior.tsv",
        pd.DataFrame(prior_rows, columns=["node_id", "TF", "TG", "score"]),
    )
    for kind in ("TF", "TG"):
        save_table(
            f"node_{kind.lower()}_scope.tsv",
            pd.DataFrame(
                [(node, gene) for node in tree.node_id for gene in scopes[kind][node]],
                columns=["node_id", kind],
            ),
        )
    save_table("rna_annotation.tsv", rna_annotation)
    save_table("atac_annotation.tsv", atac_annotation)
    for idx, result in enumerate(stage_results):
        for label, rows in zip(
            ("motif_max", "filtered", "top_density", "percentile"), result[:4]
        ):
            save_table(
                f"networks/stage_{idx:03d}/{label}.tsv",
                pd.DataFrame(rows),
                header=False,
                float_format="%.6g" if label == "percentile" else "%.6f",
            )
    tf_rows, tg_rows = [], []
    for idx, (chrom, start, end, node, peak_id) in enumerate(peaks):
        tf_rows.extend(
            (node, peak_id, chrom, start, end, tf_alias.get(tf, tf), value)
            for tf, value in peak_tf[idx].items()
        )
        tg_rows.extend(
            (node, peak_id, chrom, start, end, tg_alias.get(tg, tg))
            for tg in peak_genes[idx]
        )
    save_table(
        "TF_RE.tsv",
        pd.DataFrame(
            tf_rows,
            columns=["node_id", "RE", "chr", "start", "end", "TF", "motif_score"],
        ),
    )
    save_table(
        "RE_TG.tsv",
        pd.DataFrame(tg_rows, columns=["node_id", "RE", "chr", "start", "end", "TG"]),
    )
    prepared = copy.deepcopy(config)
    prepared["base_dir"] = str(resolve_config_path(config, "."))
    prepared.setdefault("prior_preparation", {})["enabled"] = False
    prepared["data"].update(
        tf_tg_prior=str(out / "tf_tg_prior.tsv"),
        annotation=str(out / "rna_annotation.tsv"),
        node_tf_scope=str(out / "node_tf_scope.tsv"),
        node_tg_scope=str(out / "node_tg_scope.tsv"),
    )
    (out / "unpaired.yaml").write_text(
        yaml.safe_dump(prepared, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    output_files.append("unpaired.yaml")
    report = {
        "method": "scmtni",
        "reference_commit": REFERENCE_COMMIT,
        "reference_url": REFERENCE_URL,
        "implementation_sha256": implementation_hashes,
        "fingerprint": fingerprint,
        "inputs": input_meta,
        "genome_assembly": data["genome_assembly"],
        "peak_source": "provided per-node MACS2 narrowPeak; no h5ad detection proxy",
        "motif_rows_scanned": motif_rows,
        "nodes": node_reports,
        "rank_behavior": "literal source rankEdges incr: lower absolute retained scores receive higher percentiles; no correction of source behavior",
        "tie_policy": "source Python set iteration, stable density sort; boundary ties depend on Python hash seed",
        "python_version": platform.python_version(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
        "scope_policy": "provided scopes or explicitly declared shared measured RNA gene universe; no expression cutoff inferred",
        "limitations": "Starts after peak calling and motif scanning. Does not reproduce BAM processing, MACS2 or LIGER, and does not replace DRAFT with the scMTNI Bayesian model.",
        "output_sha256": {name: _hash(out / name) for name in output_files},
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(out),
        "config_path": str(out / "unpaired.yaml"),
        "report": report,
        "reused": False,
    }
