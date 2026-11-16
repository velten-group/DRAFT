"""Shared RNA/ATAC-to-prior preparation for paired and independent assays.

Select ATAC peak features by accessibility, retrieve JASPAR motif predictions,
and link regions to annotated TSS windows to construct stage-specific priors.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from draft_unpaired.scmtni_prior import IntervalIndex, bed_rows

from .input_schema import (
    canonical_cell_annotation,
    read_cell_annotation,
    read_gene_table,
    read_table,
    select_matrix,
)
from .lineage import load_tree_table
from .prior_resources import RESOURCE_CATALOG, download_cached, open_motif_track, sha256
from .utils import emit_progress, load_yaml_config, resolve_config_path


def canonical_chrom(value):
    value = str(value).strip()
    if value in {"M", "MT", "chrMT", "chrM"}:
        return "chrM"
    return value if value.startswith("chr") else f"chr{value}"


def peak_regions(adata):
    """Return canonical BED coordinates without modifying the H5AD file."""
    columns = {str(c).lower(): c for c in adata.var.columns}
    chrom_col = next(
        (columns[k] for k in ("chr", "chrom", "chromosome") if k in columns), None
    )
    coordinate_columns = chrom_col and "start" in columns and "end" in columns
    regions = []
    for i, name in enumerate(adata.var_names):
        match = re.fullmatch(r"([^:\s]+):(\d+)-(\d+)", str(name))
        if match is None:
            match = re.fullmatch(r"(.+)-(\d+)-(\d+)", str(name))
        if match:
            chrom, start, end = match.groups()
        elif coordinate_columns:
            row = adata.var.iloc[i]
            chrom, start, end = (
                row[chrom_col],
                row[columns["start"]],
                row[columns["end"]],
            )
        else:
            raise ValueError(
                f"ATAC feature {name!r} needs genomic coordinates in its name or var chr/start/end columns."
            )
        if float(start) % 1 or float(end) % 1:
            raise ValueError(f"Noninteger ATAC coordinates for {name!r}.")
        start, end = int(float(start)), int(float(end))
        if start < 0 or end <= start:
            raise ValueError(f"Invalid half-open ATAC coordinates for {name!r}.")
        regions.append((canonical_chrom(chrom), start, end))
    if len(set(regions)) != len(regions):
        raise ValueError(
            "ATAC features must have unique genomic intervals; merge duplicate regions first."
        )
    return regions


def _annotation(adata, config, key, fallback=None):
    value = config.get("data", {}).get(key) or fallback
    if value:
        frame = read_cell_annotation(resolve_config_path(config, value))
    else:
        frame = adata.obs.copy()
        frame["cell_id"] = adata.obs_names.astype(str)
        frame = canonical_cell_annotation(frame)
    if not frame.cell_id.isin(adata.obs_names).all():
        raise ValueError(f"{key} contains cell IDs absent from its own assay.")
    return frame


def _base_id(value):
    return re.sub(r"\.\d+(?=_|$)", "", str(value))


def gene_annotation(path):
    """Read gene bounds and transcript TSSs from GTF; convert to zero-based BED."""
    genes = {}
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in {"gene", "transcript"}:
                continue
            attributes = dict(re.findall(r'(\S+)\s+"([^"]*)"', fields[8]))
            if "gene_id" not in attributes or fields[6] not in {"+", "-"}:
                continue
            identifier = _base_id(attributes["gene_id"])
            start, end = int(fields[3]) - 1, int(fields[4])
            if start < 0 or end <= start:
                raise ValueError("Invalid GTF coordinates.")
            item = genes.setdefault(
                identifier,
                {
                    "gene_id": identifier,
                    "gene_name": attributes.get("gene_name", identifier),
                    "chr": canonical_chrom(fields[0]),
                    "start": start,
                    "end": end,
                    "strand": fields[6],
                    "tss": set(),
                },
            )
            if (item["chr"], item["strand"]) != (canonical_chrom(fields[0]), fields[6]):
                raise ValueError(f"Ambiguous gene loci in annotation: {identifier}")
            item["start"], item["end"] = (
                min(item["start"], start),
                max(item["end"], end),
            )
            if fields[2] == "transcript":
                item["tss"].add(start if fields[6] == "+" else end - 1)
    if not genes:
        raise ValueError(f"No gene/transcript annotations found in {path}.")
    for item in genes.values():
        if not item["tss"]:
            item["tss"].add(item["start"] if item["strand"] == "+" else item["end"] - 1)
    return genes


def _gene_table(path, rna_names, genes, role):
    supplied = read_gene_table(path)
    rna_names = set(map(str, rna_names))
    rna_aliases, gene_aliases = defaultdict(set), defaultdict(set)
    for name in rna_names:
        rna_aliases[_base_id(name).casefold()].add(name)
    for identifier, gene in genes.items():
        for alias in (identifier, gene["gene_name"]):
            gene_aliases[_base_id(alias).casefold()].add(identifier)
    rows, matched = [], {}
    for row in supplied.itertuples(index=False):
        candidates = gene_aliases.get(_base_id(row.gene_id).casefold(), set())
        if not candidates:
            candidates = gene_aliases.get(_base_id(row.gene_name).casefold(), set())
        eligible = {}
        for identifier in candidates:
            gene = genes[identifier]
            matches = set()
            for alias in (identifier, gene["gene_name"]):
                matches.update(rna_aliases.get(_base_id(alias).casefold(), set()))
            if str(row.gene_id) in rna_names:
                matches = {str(row.gene_id)}
            for rna_id in matches:
                if (
                    rna_id in eligible
                    and eligible[rna_id]["gene_id"] != gene["gene_id"]
                ):
                    raise ValueError(
                        f"Ambiguous annotation for {rna_id!r}; use unambiguous gene IDs."
                    )
                eligible[rna_id] = gene
        if len(eligible) > 1:
            raise ValueError(
                f"Ambiguous RNA gene for {row.gene_id!r}; provide unambiguous gene IDs."
            )
        if eligible:
            rna_id, gene = next(iter(eligible.items()))
            matched[rna_id] = gene
            rows.append(
                {
                    "gene_id": rna_id,
                    "gene_name": gene["gene_name"],
                    **{key: gene[key] for key in ("chr", "start", "end", "strand")},
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates("gene_id") if rows else pd.DataFrame()
    if frame.empty:
        raise ValueError(
            f"No {role} genes match both RNA and the selected genome annotation."
        )
    return frame, matched, {"requested": len(supplied), "matched": len(frame)}


def _tf_aliases(table):
    aliases = defaultdict(set)
    for row in table.itertuples(index=False):
        for name in (row.gene_id, row.gene_name):
            aliases[_base_id(name).casefold()].add(row.gene_id)
    return {
        key: next(iter(values)) for key, values in aliases.items() if len(values) == 1
    }


def _tf_ids(value, aliases):
    names = str(value).split("::")
    return sorted(
        {
            aliases[key]
            for name in names
            if (key := _base_id(re.sub(r"\(var\.\d+\)$", "", name).strip()).casefold())
            in aliases
        }
    )


def _merged_regions(regions):
    merged = []
    for chrom, start, end in sorted(set(regions)):
        if merged and merged[-1][0] == chrom and start <= merged[-1][2]:
            merged[-1] = (chrom, merged[-1][1], max(end, merged[-1][2]))
        else:
            merged.append((chrom, start, end))
    return merged


def motif_subset(config, genome, regions, aliases, cache, minimum):
    """Cache just the requested ATAC-region/TF subset of the public BED7 track."""
    resources = config.get("resources", {})
    if resources.get("motif_instances") or resources.get("motif_to_tf"):
        if not all(resources.get(key) for key in ("motif_instances", "motif_to_tf")):
            raise ValueError(
                "Provide both resources.motif_instances and resources.motif_to_tf."
            )
        hits, mapping = (
            resolve_config_path(config, resources[key])
            for key in ("motif_instances", "motif_to_tf")
        )
        return (
            hits,
            mapping,
            {
                "kind": "local_motif_hits",
                "hits_sha256": sha256(hits),
                "mapping_sha256": sha256(mapping),
            },
        )
    source = resources.get("motif_bigbed", RESOURCE_CATALOG[genome]["motif_url"])
    if not str(source).startswith(("https://", "http://")):
        source = str(resolve_config_path(config, source))
    signature = {
        "source": source,
        "regions": sorted(regions),
        "aliases": aliases,
        "minimum_score": minimum,
    }
    if not source.startswith(("https://", "http://")):
        signature["sha256"] = sha256(Path(source))
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    folder = cache / "motif_subsets" / digest
    hits, mapping, metadata = (
        folder / "motif_hits.bed.gz",
        folder / "motif_to_tf.tsv",
        folder / "resource.json",
    )
    if metadata.exists():
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if (
            hits.exists()
            and mapping.exists()
            and sha256(hits) == record["hits_sha256"]
            and sha256(mapping) == record["mapping_sha256"]
        ):
            return hits, mapping, record
        raise ValueError(
            f"Cached motif subset changed: {folder}; remove it to retrieve again."
        )
    folder.mkdir(parents=True, exist_ok=True)
    queries = _merged_regions(regions)
    query_index = IntervalIndex(queries)
    motif_tfs, count = defaultdict(set), 0
    try:
        emit_progress(
            "START",
            "motif-resources",
            f"{genome}: querying {len(queries)} region intervals",
        )
        with (
            open_motif_track(source) as (reader, stream),
            gzip.open(hits, "wt", encoding="utf-8") as out,
        ):
            chroms = reader.chroms()
            missing = sorted({chrom for chrom, _, _ in queries} - set(chroms))
            for i, (chrom, start, end) in enumerate(queries):
                if chrom not in chroms:
                    continue
                if end > chroms[chrom]:
                    raise ValueError(
                        f"ATAC region exceeds {genome} chromosome bounds: {chrom}:{start}-{end}"
                    )
                for row in reader.records(chrom, start, end):
                    if len(row) != 6:
                        raise ValueError(
                            "Expected JASPAR BED7 records: coordinates, matrix ID, score, strand, TF name."
                        )
                    left, right, motif, score, strand, names = row
                    tfs = _tf_ids(names, aliases)
                    if not tfs or float(score) < minimum:
                        continue
                    # BigBed queries can also return sites that only touch the
                    # interval edge; keep each site once, under its first region.
                    owners = query_index.overlaps(chrom, left, right)
                    if not owners or owners[0] != i:
                        continue
                    out.write(f"{chrom}\t{left}\t{right}\t{motif}\t{score}\t{strand}\n")
                    motif_tfs[motif].update(tfs)
                    count += 1
                if (i + 1) % 100 == 0:
                    emit_progress(
                        "PROGRESS",
                        "motif-resources",
                        f"{i + 1}/{len(queries)} intervals",
                    )
            downloaded = stream.bytes_downloaded if stream is not None else None
            identity = stream.identity if stream is not None else None
        if not count:
            raise ValueError(
                "No motif predictions match the supplied TFs and ATAC regions; check gene names, assembly and score settings."
            )
        mapping.write_text(
            "".join(
                f"{motif}\t{'::'.join(sorted(tfs))}\n"
                for motif, tfs in sorted(motif_tfs.items())
            ),
            encoding="utf-8",
        )
        record = {
            "kind": "JASPAR_BED7_subset",
            "source": source,
            "resource_identity": identity,
            "downloaded_bytes": downloaded,
            "hits": count,
            "missing_chromosomes": missing,
            "hits_sha256": sha256(hits),
            "mapping_sha256": sha256(mapping),
        }
        metadata.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return hits, mapping, record
    except Exception:
        hits.unlink(missing_ok=True)
        mapping.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
        raise


def _accessible(matrix, indices, minimum, feature_count):
    detected = np.zeros(feature_count, dtype=np.int64)
    indices = sorted(indices)
    for start in range(0, len(indices), 512):
        values = matrix[indices[start : start + 512], :]
        data = values.data if sp.issparse(values) else np.asarray(values)
        if not np.isfinite(data).all() or (data < 0).any():
            raise ValueError(
                "ATAC values must be finite and nonnegative; select counts/binary data."
            )
        detected += np.asarray((values > 0).sum(axis=0)).ravel().astype(np.int64)
    return np.flatnonzero((detected > 0) & (detected / len(indices) >= minimum))


def prepare_prior(config_or_path, output_dir, *, mode=None, force=False):
    """Prepare standard regulatory priors and a model-input config for either mode."""
    config = (
        load_yaml_config(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else copy.deepcopy(config_or_path)
    )
    mode = mode or config.get("input_mode")
    if mode not in {"paired", "unpaired"}:
        raise ValueError("Specify input_mode: paired or unpaired, or pass --mode.")
    data, settings = config.get("data", {}), config.get("prior_preparation", {})
    if settings.get("method", "auto_motif") != "auto_motif":
        raise ValueError(
            "The shared automatic-prior interface requires method: auto_motif; use the mode-specific command for supplied-prior or scMTNI routes."
        )
    genome = data.get("genome_assembly")
    if genome not in RESOURCE_CATALOG:
        raise ValueError(
            "Automatic resources support hg38, mm10 and dm6; pass --genome or --set data.genome_assembly=ASSEMBLY."
        )
    minimum = float(settings.get("min_accessible_fraction", 0.05))
    motif_minimum = float(settings.get("min_motif_score", 500))
    flank = int(settings.get("promoter_flank", 5000))
    if (
        not 0 <= minimum <= 1
        or not np.isfinite(motif_minimum)
        or motif_minimum < 0
        or flank < 0
    ):
        raise ValueError(
            "Invalid accessibility fraction, motif score or promoter window."
        )
    if float(data.get("prior_score_threshold", 0.1)) >= 1:
        raise ValueError(
            "Binary prepared candidates require prior_score_threshold < 1."
        )
    required = ("rna_h5ad", "atac_h5ad", "tf_list", "tg_list", "tree")
    if any(not data.get(key) for key in required):
        raise ValueError(
            f"Automatic prior preparation needs data: {', '.join(required)}."
        )
    paths = {key: resolve_config_path(config, data[key]) for key in required}
    for key in ("annotation", "atac_annotation", "node_tf_scope", "node_tg_scope"):
        if data.get(key):
            paths[key] = resolve_config_path(config, data[key])
    resources = config.get("resources", {})
    local_resources = {
        key: str(resolve_config_path(config, resources[key]))
        for key in ("gene_annotation", "motif_instances", "motif_to_tf", "motif_bigbed")
        if resources.get(key)
        and not str(resources[key]).startswith(("https://", "http://"))
    }
    fingerprint_data = {
        "mode": mode,
        "genome": genome,
        "settings": settings,
        "catalog": RESOURCE_CATALOG[genome],
        "resources": resources,
        "inputs": {key: sha256(path) for key, path in paths.items()},
        "local_resources": {
            key: sha256(Path(path)) for key, path in local_resources.items()
        },
        "implementation_sha256": sha256(Path(__file__)),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, sort_keys=True).encode()
    ).hexdigest()
    out = resolve_config_path(config, output_dir)
    report_path, config_path = out / "prior_report.json", out / "model_inputs.yaml"
    if report_path.exists() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("fingerprint") != fingerprint:
            raise ValueError(
                "Prior inputs/settings changed; choose another output directory or use --force."
            )
        if any(
            not (out / name).is_file() or sha256(out / name) != digest
            for name, digest in report["output_sha256"].items()
        ):
            raise ValueError(
                "Prepared prior output changed; choose another output directory or use --force."
            )
        return {
            "output_dir": str(out),
            "config_path": str(config_path),
            "report": report,
            "reused": True,
        }
    input_config = (
        Path(config_or_path).expanduser().resolve()
        if isinstance(config_or_path, (str, Path))
        else None
    )
    allowed = (
        {input_config}
        if input_config is not None
        and input_config.name in {"inputs.yaml", "command_config.yaml"}
        and input_config.parent == out
        else set()
    )
    if (
        out.exists()
        and any(path.resolve() not in allowed for path in out.iterdir())
        and not force
    ):
        raise FileExistsError(
            f"{out} is nonempty; choose another output directory or use --force."
        )
    tree = load_tree_table(paths["tree"], strict=True)
    for name in [
        "TF.tsv",
        "TG.tsv",
        "RE.txt",
        "TF_RE.tsv",
        "RE_TG.tsv",
        "tf_tg_prior.tsv",
        "rna_annotation.tsv",
        "atac_annotation.tsv",
        "model_inputs.yaml",
    ]:
        if (out / name).resolve() in set(paths.values()) | {
            Path(path) for path in local_resources.values()
        }:
            raise ValueError("Prepared outputs must not overwrite input files.")
    cache = resolve_config_path(
        config, resources.get("cache_dir", "data/resources/cache")
    )
    if "gene_annotation" in local_resources:
        annotation_path = Path(local_resources["gene_annotation"])
        annotation_record = {
            "path": str(annotation_path),
            "sha256": sha256(annotation_path),
        }
    else:
        emit_progress(
            "START",
            "gene-resources",
            f"{genome}: {RESOURCE_CATALOG[genome]['annotation_version']}",
        )
        annotation_path, annotation_record = download_cached(
            RESOURCE_CATALOG[genome]["annotation_url"], cache / "annotations"
        )
    genes = gene_annotation(annotation_path)
    rna, atac = (
        ad.read_h5ad(paths["rna_h5ad"], backed="r"),
        ad.read_h5ad(paths["atac_h5ad"], backed="r"),
    )
    try:
        if (
            not rna.obs_names.is_unique
            or not rna.var_names.is_unique
            or not atac.obs_names.is_unique
        ):
            raise ValueError("Assay cell/gene identifiers must be unique.")
        rna_anno = _annotation(rna, config, "annotation")
        atac_anno = _annotation(
            atac,
            config,
            "atac_annotation",
            data.get("annotation") if mode == "paired" else None,
        )
        if mode == "paired":
            common = set(rna_anno.cell_id) & set(atac_anno.cell_id)
            rna_anno = rna_anno[rna_anno.cell_id.isin(common)].copy()
            atac_anno = atac_anno[atac_anno.cell_id.isin(common)].copy()
            labels = rna_anno.set_index("cell_id").cell_type
            if any(
                labels[cell] != node
                for cell, node in zip(atac_anno.cell_id, atac_anno.cell_type)
            ):
                raise ValueError(
                    "Paired RNA/ATAC annotations disagree on a cell's tree node."
                )
        regions = peak_regions(atac)
        tf_table, _, tf_counts = _gene_table(
            paths["tf_list"], rna.var_names, genes, "TF"
        )
        tg_table, tg_genes, tg_counts = _gene_table(
            paths["tg_list"], rna.var_names, genes, "TG"
        )
        matrix, layer = select_matrix(atac, data.get("atac_layer", "counts"))
        cell_index = {str(name): i for i, name in enumerate(atac.obs_names)}
        active, node_report = {}, []
        for node in tree.node_id:
            rna_cells = rna_anno.loc[rna_anno.cell_type.eq(node), "cell_id"]
            atac_cells = atac_anno.loc[atac_anno.cell_type.eq(node), "cell_id"]
            if rna_cells.empty or atac_cells.empty:
                raise ValueError(
                    f"Both assays need annotated cells for tree node {node!r}; paired inputs also need matching cells."
                )
            active[node] = _accessible(
                matrix, [cell_index[cell] for cell in atac_cells], minimum, len(regions)
            )
            node_report.append(
                {
                    "node_id": node,
                    "rna_cells": len(rna_cells),
                    "atac_cells": len(atac_cells),
                    "accessible_RE": len(active[node]),
                }
            )
    finally:
        rna.file.close()
        atac.file.close()
    active_union = set(map(int, np.concatenate(list(active.values()))))
    if not active_union:
        raise ValueError("No ATAC regions pass stage accessibility selection.")
    aliases = _tf_aliases(tf_table)
    scopes = {}
    for role, table in (("TF", tf_table), ("TG", tg_table)):
        key = f"node_{role.lower()}_scope"
        if key not in paths:
            scopes[role] = {node: set(table.gene_id) for node in tree.node_id}
            continue
        frame = read_table(paths[key])
        if not {"node_id", role}.issubset(frame.columns):
            raise ValueError(f"{key} requires node_id and {role} columns.")
        role_aliases = _tf_aliases(table)
        scopes[role] = {
            node: {
                role_aliases[_base_id(value).casefold()]
                for value in frame.loc[frame.node_id.eq(node), role]
                if _base_id(value).casefold() in role_aliases
            }
            for node in tree.node_id
        }
        if any(not values for values in scopes[role].values()):
            raise ValueError(f"{key} must contain measured genes for every tree node.")
    hits, motif_map, motif_record = motif_subset(
        config,
        genome,
        [regions[i] for i in sorted(active_union)],
        aliases,
        cache,
        motif_minimum,
    )
    mapping = {}
    for line in Path(motif_map).read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError("Motif mapping must contain motif_ID<TAB>TF1::TF2.")
        mapping[fields[0]] = _tf_ids(fields[1], aliases)
    index = IntervalIndex(regions)
    peak_tf, peak_tg = [{} for _ in regions], [set() for _ in regions]
    promoters = []
    for tg, gene in tg_genes.items():
        for tss in sorted(gene["tss"]):
            left, right = max(0, tss - flank), tss + flank + 1
            promoters.append((gene["chr"], left, right, tg))
            for i in index.overlaps(gene["chr"], left, right):
                if i in active_union:
                    peak_tg[i].add(tg)
    for chrom, left, right, fields in bed_rows(Path(hits), "motif"):
        score = float(fields[-2])
        if not np.isfinite(score) or score < motif_minimum:
            continue
        for i in index.overlaps(canonical_chrom(chrom), left, right):
            if i not in active_union:
                continue
            for tf in mapping.get(fields[-3], []):
                peak_tf[i][tf] = max(score, peak_tf[i].get(tf, -np.inf))
    outputs = []

    def save(name, frame, *, header=True):
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, sep="\t", index=False, header=header)
        outputs.append(name)

    re_names = [f"{chrom}:{start}-{end}" for chrom, start, end in regions]
    tf_rows, tg_rows, prior_rows, manifest = [], [], [], []
    for stage_idx, (node, selected) in enumerate(active.items()):
        counts = defaultdict(int)
        tf_start, tg_start = len(tf_rows), len(tg_rows)
        for i in selected:
            for tf, score in sorted(peak_tf[i].items()):
                if tf not in scopes["TF"][node]:
                    continue
                tf_rows.append((node, tf, re_names[i], score))
                for tg in sorted(peak_tg[i] & scopes["TG"][node]):
                    counts[tf, tg] += 1
            tg_rows.extend(
                (node, re_names[i], tg)
                for tg in sorted(peak_tg[i] & scopes["TG"][node])
            )
        if not counts:
            raise ValueError(
                f"No TF-RE-TG candidates at {node!r}; check resources, TF identifiers and promoter/accessibility settings."
            )
        prior_rows.extend(
            (node, tf, tg, 1.0, support) for (tf, tg), support in sorted(counts.items())
        )
        node_report[stage_idx]["TF_TG_candidates"] = len(counts)
        node_report[stage_idx]["TF_RE_candidates"] = len(tf_rows) - tf_start
        node_report[stage_idx]["RE_TG_candidates"] = len(tg_rows) - tg_start
        name = f"stage_peaks/stage_{stage_idx:03d}.bed"
        manifest.append((node, str(out / name)))
    out.mkdir(parents=True, exist_ok=True)
    save("TF.tsv", tf_table)
    save("TG.tsv", tg_table)
    save("rna_annotation.tsv", rna_anno)
    save("atac_annotation.tsv", atac_anno)
    save("promoters.bed", pd.DataFrame(promoters), header=False)
    (out / "RE.txt").write_text("\n".join(re_names) + "\n", encoding="utf-8")
    outputs.append("RE.txt")
    for stage_idx, selected in enumerate(active.values()):
        save(
            f"stage_peaks/stage_{stage_idx:03d}.bed",
            pd.DataFrame([(*regions[i], re_names[i]) for i in selected]),
            header=False,
        )
    staged_tf = pd.DataFrame(tf_rows, columns=["node_id", "TF", "RE", "motif_score"])
    staged_tg = pd.DataFrame(tg_rows, columns=["node_id", "RE", "TG"])
    save("TF_RE_by_stage.tsv", staged_tf)
    save("RE_TG_by_stage.tsv", staged_tg)
    save(
        "TF_RE.tsv",
        staged_tf[["TF", "RE", "motif_score"]].drop_duplicates(["TF", "RE"]),
    )
    save("RE_TG.tsv", staged_tg[["RE", "TG"]].drop_duplicates())
    save(
        "tf_tg_prior.tsv",
        pd.DataFrame(
            prior_rows, columns=["node_id", "TF", "TG", "score", "n_supported_RE"]
        ),
    )
    save("stage_peaks.tsv", pd.DataFrame(manifest, columns=["node_id", "peak_file"]))
    for kind, table in (("TF", tf_table), ("TG", tg_table)):
        save(
            f"node_{kind.lower()}_scope.tsv",
            pd.DataFrame(
                [
                    (node, gene)
                    for node in tree.node_id
                    for gene in sorted(scopes[kind][node])
                ],
                columns=["node_id", kind],
            ),
        )
    prepared = copy.deepcopy(config)
    prepared["base_dir"] = str(resolve_config_path(config, "."))
    prepared["input_mode"] = mode
    prepared["prior_preparation"] = {"enabled": False}
    prepared.setdefault("data", {}).update(
        tf_list=str(out / "TF.tsv"),
        tg_list=str(out / "TG.tsv"),
        re_list=str(out / "RE.txt"),
        annotation=str(out / "rna_annotation.tsv"),
        atac_annotation=str(out / "atac_annotation.tsv"),
        tf_re_prior=str(out / "TF_RE.tsv"),
        re_tg_prior=str(out / "RE_TG.tsv"),
        stage_tf_re_prior=str(out / "TF_RE_by_stage.tsv"),
        stage_re_tg_prior=str(out / "RE_TG_by_stage.tsv"),
        tf_tg_prior=str(out / "tf_tg_prior.tsv"),
        canonicalize_atac_peaks=True,
        node_tf_scope=str(out / "node_tf_scope.tsv"),
        node_tg_scope=str(out / "node_tg_scope.tsv"),
    )
    config_path.write_text(yaml.safe_dump(prepared, sort_keys=False), encoding="utf-8")
    outputs.append("model_inputs.yaml")
    motif_matched = {tf for item in peak_tf for tf in item}
    tf_counts.update(
        motif_matched=len(motif_matched),
        without_matched_motif=sorted(set(tf_table.gene_id) - motif_matched),
    )
    report = {
        "method": "auto_motif",
        "input_mode": mode,
        "genome_assembly": genome,
        "fingerprint": fingerprint,
        "catalog": RESOURCE_CATALOG[genome],
        "annotation_resource": annotation_record,
        "motif_resource": motif_record,
        "atac_layer": layer,
        "peak_source": "existing ATAC feature intervals selected by per-node accessibility",
        "score_policy": "binary candidate presence; maximum motif score retained as auxiliary TF-RE evidence",
        "scope_policy": "Provided node scopes when supplied, otherwise RNA-matched global TF/TG universe; no expression threshold inferred",
        "min_accessible_fraction": minimum,
        "min_motif_score": motif_minimum,
        "promoter_flank": flank,
        "gene_coverage": {"TF": tf_counts, "TG": tg_counts},
        "nodes": node_report,
        "limitations": "Accessible regions are selected from existing ATAC matrix features; raw-read peak calling is not performed.",
        "output_sha256": {name: sha256(out / name) for name in outputs},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    emit_progress("DONE", "prior-preparation", str(config_path))
    return {
        "output_dir": str(out),
        "config_path": str(config_path),
        "report": report,
        "reused": False,
    }
