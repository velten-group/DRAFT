"""Receptor-centrality communication on a shared, receiver-stage time axis."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..checkpoint import save_checkpoint_metadata
from ..input_schema import read_cell_annotation, read_gene_table, read_table
from ..utils import ensure_directory, load_yaml_config, resolve_config_path
from .ccc_alignment import network_alignment
from .gene_keys import GeneKeys, as_request, clean_symbol, resolve_symbols

DIRECTIONS = {
    "hub_to_germline": ("hub", "germ"),
    "hub_to_soma": ("hub", "soma"),
    "soma_to_germline": ("soma", "germ"),
    "germline_to_soma": ("germ", "soma"),
}


def receptor_percentiles(
    tf: np.ndarray,
    tg: np.ndarray,
    active_tf: np.ndarray,
    active_tg: np.ndarray,
    block_size: int = 4096,
    *,
    absolute_cosine: bool = False,
) -> np.ndarray:
    """Mean right-inclusive percentile over active TFs, for every TG.

    The reference distribution contains all active TF x active TG pairs;
    expression gating of the receptor is applied later and independently.
    """
    tf, tg = np.asarray(tf, float), np.asarray(tg, float)
    if (
        tf.ndim != 2
        or tg.ndim != 2
        or tf.shape[1] != tg.shape[1]
        or not np.isfinite(tf).all()
        or not np.isfinite(tg).all()
    ):
        raise ValueError("Receiver embeddings must be finite matrices of equal width.")
    it, ig = np.flatnonzero(active_tf), np.flatnonzero(active_tg)
    if not len(it) or not len(ig):
        raise ValueError(
            "Receiver percentile reference needs active TFs and TGs at every stage."
        )
    tf = tf[it] / np.maximum(np.linalg.norm(tf[it], axis=1, keepdims=True), 1e-12)
    tg = tg / np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
    reference_scores = tf @ tg[ig].T
    if absolute_cosine:
        reference_scores = np.abs(reference_scores)
    reference = np.sort(reference_scores.ravel())
    result = np.empty(len(tg))
    for start in range(0, len(tg), block_size):
        scores = tf @ tg[start : start + block_size].T
        if absolute_cosine:
            scores = np.abs(scores)
        result[start : start + block_size] = (
            np.searchsorted(reference, scores, side="right") / reference.size
        ).mean(axis=0)
    return result


def timing_call(
    curve: np.ndarray, labels: list[str], early_count: int, min_range: float = 0.1
) -> dict[str, Any]:
    curve = np.asarray(curve, float)
    if len(curve) != len(labels) or not 0 < early_count < len(curve):
        raise ValueError(
            "early_stage_count must split the receiver stages into nonempty early and late windows."
        )
    if not np.isfinite(curve).all() or np.any(curve < 0):
        raise ValueError("Communication scores must be finite and nonnegative.")
    if curve.max() == 0:
        return {
            "status": "unscored",
            "peak_index": None,
            "peak_stage": None,
            "dynamic_range": 0.0,
            "earliness_score": 0.0,
        }
    peak, maximum = int(np.argmax(curve)), float(curve.max())
    dynamic = float((maximum - curve.min()) / maximum)
    return {
        "status": "untimed"
        if dynamic < min_range
        else ("early" if peak < early_count else "late"),
        "peak_index": peak,
        "peak_stage": labels[peak],
        "dynamic_range": dynamic,
        "earliness_score": float(
            (curve[:early_count].max() - curve[early_count:].max()) / maximum
        ),
    }


def stage_nodes(
    groups: list[str], parallel: list[str] | None = None
) -> list[list[str]]:
    if len(set(groups)) != len(groups) or len(groups) < 2:
        raise ValueError(
            "Each lineage requires at least two distinct stage groups in developmental order."
        )
    if not parallel:
        return [[g] for g in groups]
    if (
        len(set(parallel)) != len(parallel)
        or len(parallel) < 2
        or set(parallel) != set(groups[-len(parallel) :])
    ):
        raise ValueError(
            "parallel_groups must name the distinct terminal groups at the end of stage_groups."
        )
    return [[g] for g in groups[: -len(parallel)]] + [list(parallel)]


def aligned_positions(
    table: pd.DataFrame, soma_groups: list[str], nodes: list[list[str]], germ_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map sender positions to the receiver stages in both alignment directions."""
    if germ_count < 2 or len(nodes) < 2:
        raise ValueError("Alignment requires at least two receiver stages per lineage.")
    old_s = 1 + table.soma_tau.to_numpy(float) * (len(soma_groups) - 1)
    germ = 1 + table.germ_tau.to_numpy(float) * (germ_count - 1)
    if len(nodes[-1]) > 1:
        # Parallel terminal groups share an endpoint, reached linearly
        # from their preceding stage.
        last_single = len(nodes) - 1
        anchor = float(np.interp(last_single, old_s, germ))
        soma = np.interp(
            old_s, [1, last_single, len(soma_groups)], [1, last_single, len(nodes)]
        )
        terminal = soma >= last_single
        germ[terminal] = anchor + (soma[terminal] - last_single) * (germ_count - anchor)
    else:
        soma = old_s
    # An inverse of a plateau uses the mean source position, with fixed ends.
    inverse = (
        pd.DataFrame({"germ": germ, "soma": soma})
        .groupby("germ", sort=True)
        .soma.mean()
    )
    inverse.iloc[0], inverse.iloc[-1] = 1.0, float(len(nodes))
    soma_at_germ = np.interp(
        np.arange(1, germ_count + 1), inverse.index, inverse.values
    )
    germ_at_soma = np.interp(np.arange(1, len(nodes) + 1), soma, germ)
    return soma_at_germ, germ_at_soma


def _collapse(frame: pd.DataFrame, nodes: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {i: frame[groups].mean(axis=1) for i, groups in enumerate(nodes)},
        index=frame.index,
    )


def _unpaired_reference_names(
    inference: dict[str, Any], paths: Any
) -> dict[str, list[str]]:
    """Resolve gene symbols in graph row order from the saved maps and gene tables."""
    maps = inference.get("maps") or {}
    refs = {}
    for role in ("tf", "tg"):
        count = inference[f"out_{role}"].shape[0]
        ids = maps.get(f"{role}_ids", maps.get(role, []))
        if len(ids) != count:
            raise ValueError(
                f"Unpaired graph maps must identify every {role.upper()} row."
            )
        ids = [str(value) for value in ids]
        names = maps.get(f"{role}_names")
        if names is None:
            table = read_gene_table(getattr(paths, f"{role}_list"))
            symbols = dict(zip(table.gene_id, table.gene_name))
            names = [symbols.get(value, value) for value in ids]
        if len(names) != count:
            raise ValueError(
                f"Unpaired {role.upper()} symbols must match graph row order."
            )
        refs[f"{role}_ids"] = ids
        refs[f"{role}_names"] = [str(value) for value in names]
    return refs


def _receiver_scores(
    config: dict[str, Any],
    name: str,
    spec: dict[str, Any],
    output_dir: Path,
    keys: GeneKeys | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Receptor scores indexed by gene keys (lowercase unless case separates genes).

    Without keys, the case rule comes from this lineage's own gene names.
    """
    frame, groups, meta, vocabulary = _raw_receiver_scores(config, name, spec)
    return _keyed_receiver_scores(
        frame, groups, meta, keys or GeneKeys(vocabulary), name, output_dir
    )


def _keyed_receiver_scores(
    frame: pd.DataFrame,
    groups: list[str],
    meta: dict[str, Any],
    keys: GeneKeys,
    name: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    frame = frame.copy()
    frame.index = keys.keys(frame.index)
    if meta["source"] == "checkpoint":
        source = output_dir / f"{name}_receptor_centrality.tsv"
        frame.to_csv(source, sep="\t", index_label="gene_name")
        meta["path"] = str(source)
    values = frame.to_numpy(float)
    if (
        frame.index.has_duplicates
        or frame.index.isna().any()
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
    ):
        raise ValueError(
            "Full receptor scores require unique gene symbols and finite percentiles in [0, 1]."
        )
    return frame, groups, meta


def _raw_receiver_scores(
    config: dict[str, Any], name: str, spec: dict[str, Any]
) -> tuple[pd.DataFrame, list[str], dict[str, Any], list[str]]:
    """Receptor scores indexed by gene symbols as spelled in their source."""
    groups = [str(g) for g in spec.get("stage_groups", [])]
    input_mode = spec.get("input_mode", "paired")
    if spec.get("receptor_scores"):
        source = resolve_config_path(config, spec["receptor_scores"])
        frame = pd.read_csv(source, sep="\t", index_col=0)
        frame = frame.loc[frame.index.notna()].copy()
        frame.index = frame.index.astype(str).map(clean_symbol)
        vocabulary = list(frame.index)
        if not groups:
            groups = [str(c) for c in frame.columns]
        frame.columns = frame.columns.astype(str)
        columns = [str(c) for c in spec.get("stage_columns", groups)]
        if len(columns) != len(groups):
            raise ValueError(
                "stage_columns must match stage_groups in length and model order."
            )
        frame = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
        frame.columns = groups
        meta = {
            "source": "precomputed",
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "provenance": spec.get("score_provenance", {}),
        }
    else:
        if not spec.get("config"):
            raise ValueError(
                f"ccc.{name} requires config or receptor_scores (full integer-stage percentiles)."
            )
        from .network import (
            _dense_tf_whitelist,
            _load_tf_filter,
            _load_tg_meta,
            _network_settings,
            _pick_reference_names,
            _select_valid_cols,
        )

        config_path = resolve_config_path(config, spec["config"])
        lineage_cfg = load_yaml_config(config_path)
        model_mode = lineage_cfg.get("input_mode")
        input_mode = (
            spec.get("input_mode")
            or model_mode
            or (
                "unpaired"
                if lineage_cfg.get("data", {}).get("tf_tg_prior")
                else "paired"
            )
        )
        if input_mode not in {"paired", "unpaired"}:
            raise ValueError("CCC input_mode must be paired or unpaired.")
        if model_mode is not None and model_mode != input_mode:
            raise ValueError(
                f"ccc.{name}.input_mode must match the training config ({model_mode})."
            )
        if input_mode == "paired":
            from ..infer import run_inference
            from ..io import InputPaths
        else:
            from draft_unpaired.infer import run_inference
            from draft_unpaired.io import InputPaths
        lineage_cfg.setdefault(
            "base_dir", str(Path(config.get("base_dir", ".")).resolve())
        )
        inference = run_inference(lineage_cfg)
        graphs = (
            inference.get("graphs")
            or inference.get("graphs_lineage")
            or inference.get("graphs_somatic")
            or inference.get("graphs_germline")
        )
        tf = inference["out_tf"].detach().cpu().numpy()
        tg = inference["out_tg"].detach().cpu().numpy()
        paths = InputPaths.from_config(lineage_cfg)
        filter_path = (
            resolve_config_path(
                lineage_cfg, _network_settings(lineage_cfg)["tf_filter_file"]
            )
            if input_mode == "paired"
            else paths.tf_filter_file or paths.tf_list
        )
        if not filter_path.exists():
            filter_path = paths.tf_list
        tf_filter = _load_tf_filter(filter_path)
        if input_mode == "paired":
            refs = _pick_reference_names(
                inference.get("maps"),
                paths,
                len(tf),
                len(tg),
                int(inference["out_re"].size(0)),
                tf_filter,
                _load_tg_meta(paths.tg_list),
            )
        else:
            refs = _unpaired_reference_names(inference, paths)
        whitelist = _dense_tf_whitelist(
            refs["tf_ids"], refs["tf_names"], tf_filter
        ).numpy()
        if not groups:
            groups = [
                str(g)
                for g in (
                    inference.get("time_names")
                    or inference.get("somatic_names")
                    or inference.get("germline_types")
                    or []
                )
            ]
            if spec.get("stage_prefix"):
                groups = [str(spec["stage_prefix"]) + group for group in groups]
        if len(groups) != tf.shape[1] or len(graphs) != tf.shape[1]:
            raise ValueError(
                f"ccc.{name}.stage_groups must match the model's {tf.shape[1]} integer stages."
            )
        fraction = float(
            spec.get(
                "active_fraction",
                lineage_cfg.get("network", {})
                .get("dense", {})
                .get("active_thr", 0.05 if input_mode == "paired" else 0.30),
            )
        )
        if not 0 < fraction <= 1:
            raise ValueError("Receiver active_fraction must be in (0, 1].")
        vocabulary = [clean_symbol(g) for g in refs["tf_names"] + refs["tg_names"]]
        if input_mode == "unpaired":
            local = GeneKeys(vocabulary)
            tf_keys, tg_keys = local.keys(refs["tf_names"]), local.keys(refs["tg_names"])
            genes = local.request(dict.fromkeys(tf_keys + tg_keys))
            _, detection = load_group_expression(config, groups, genes)
            tf_detection = detection.reindex(tf_keys)
            tg_detection = detection.reindex(tg_keys)
        columns = []
        for t, graph in enumerate(graphs):
            if input_mode == "paired":
                vx, _, gx = _select_valid_cols(
                    graph["TF"].x, graph["RE"].x, graph["TG"].x
                )
                active_tf = (
                    (vx != 0).float().mean(dim=1).detach().cpu().numpy() >= fraction
                ) & whitelist
                active_tg = (gx != 0).float().mean(
                    dim=1
                ).detach().cpu().numpy() >= fraction
            else:
                active_tf = (tf_detection.iloc[:, t].to_numpy() >= fraction) & whitelist
                active_tg = tg_detection.iloc[:, t].to_numpy() >= fraction
            columns.append(
                receptor_percentiles(
                    tf[:, t],
                    tg[:, t],
                    active_tf,
                    active_tg,
                    absolute_cosine=input_mode == "unpaired",
                )
            )
        frame = pd.DataFrame(
            np.asarray(columns).T,
            index=[clean_symbol(g) for g in refs["tg_names"]],
            columns=groups,
        )
        meta = {
            "source": "checkpoint",
            "config_path": str(config_path),
            "checkpoint_path": inference.get("checkpoint_path"),
            "active_fraction": fraction,
            "tf_filter_path": str(filter_path),
            "stage_groups": groups,
            "definition": (
                "right-inclusive percentile among active TF x TG absolute cosine scores; mean over active TFs"
                if input_mode == "unpaired"
                else "right-inclusive percentile among active TF x TG cosine scores; mean over active TFs"
            ),
        }
        if input_mode == "unpaired":
            meta["activity_inputs"] = {
                "rna_h5ad": config.get("data", {}).get("rna_h5ad"),
                "stage_groups": groups,
                "cell_counts": detection.attrs.get("cell_counts", {}),
                "missing_genes": detection.attrs.get("missing_genes", []),
            }
    meta["input_mode"] = input_mode
    return frame, groups, meta, vocabulary


def load_group_expression(
    config: dict[str, Any],
    groups: list[str],
    genes: list[str],
    *,
    allow_layer_fallback: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import anndata as ad
    from scipy import sparse

    # RNA symbols match the gene keys without regard to case unless case
    # separates genes; a list of plain keys uses its own case rule.
    genes = as_request(genes)
    data = config.get("data", {})
    if data.get("lineages"):
        means, fractions, counts, missing, covered = [], [], {}, set(), set()
        for name, source in data["lineages"].items():
            prefix = str(source.get("stage_prefix", f"{name}::"))
            selected = [group for group in groups if group.startswith(prefix)]
            if not selected:
                continue
            if covered.intersection(selected):
                raise ValueError("CCC lineage group prefixes must be distinct.")
            native_config = load_yaml_config(
                resolve_config_path(config, source["config"])
            )
            native_config.setdefault("base_dir", config.get("base_dir", "."))
            native_data = native_config.setdefault("data", {})
            native_data.setdefault(
                "cell_id_normalization",
                "exact"
                if int(native_config.get("config_version", 1)) >= 2
                else "sample_suffix",
            )
            mode = native_config.get("input_mode") or (
                "unpaired" if native_data.get("tf_tg_prior") else "paired"
            )
            if mode == "unpaired":
                native_data.pop("atac_h5ad", None)
            for field in ("rna_layer", "gene_symbol_column", "cell_id_normalization"):
                if field in data:
                    native_data[field] = data[field]
            native_groups = [group[len(prefix) :] for group in selected]
            mean, frac = load_group_expression(
                native_config, native_groups, genes, allow_layer_fallback=True
            )
            mapping = dict(zip(native_groups, selected))
            counts.update(
                {
                    mapping[group]: count
                    for group, count in mean.attrs.get("cell_counts", {}).items()
                }
            )
            missing.update(mean.attrs.get("missing_genes", []))
            mean.attrs, frac.attrs = {}, {}
            means.append(mean.rename(columns=mapping))
            fractions.append(frac.rename(columns=mapping))
            covered.update(selected)
        if covered != set(groups):
            raise ValueError(
                f"CCC RNA sources do not cover groups: {sorted(set(groups) - covered)}"
            )
        mean = pd.concat(means, axis=1).reindex(columns=groups)
        mean.attrs = {"cell_counts": counts, "missing_genes": sorted(missing)}
        return mean, pd.concat(fractions, axis=1).reindex(columns=groups)
    if not data.get("rna_h5ad"):
        raise ValueError(
            "Receptor-centrality CCC requires data.rna_h5ad with normalized RNA expression."
        )
    rna = ad.read_h5ad(resolve_config_path(config, data["rna_h5ad"]))
    normalization = data.get("cell_id_normalization", "sample_suffix")
    if normalization == "sample_suffix":
        from ..preprocess import norm_id

        normalize = norm_id
    elif normalization == "exact":
        normalize = str
    else:
        raise ValueError("data.cell_id_normalization must be sample_suffix or exact.")
    rna_ids = pd.Index([normalize(c) for c in rna.obs_names])
    if not rna_ids.is_unique:
        raise ValueError("RNA cell IDs collide after the configured normalization.")
    group_col = str(data.get("group_column", "cell_type"))
    if data.get("annotation"):
        annotation_path = resolve_config_path(config, data["annotation"])
        annotation = (
            read_cell_annotation(annotation_path)
            if "cell_id_column" not in data and "group_column" not in data
            else read_table(annotation_path)
        )
        cell_col = str(data.get("cell_id_column", "cell_id"))
        if cell_col not in annotation or group_col not in annotation:
            raise ValueError(f"CCC annotation requires {cell_col} and {group_col}.")
        annotation[cell_col] = annotation[cell_col].map(normalize)
        if annotation[cell_col].duplicated().any():
            raise ValueError("CCC requires unique RNA and annotation cell IDs.")
        membership = annotation.set_index(cell_col)[group_col].reindex(rna_ids)
    else:
        if group_col not in rna.obs:
            raise ValueError(
                f"RNA obs requires {group_col}, or provide data.annotation."
            )
        membership = rna.obs[group_col]
    if data.get("atac_h5ad"):
        atac = ad.read_h5ad(resolve_config_path(config, data["atac_h5ad"]), backed="r")
        try:
            atac_ids = pd.Index([normalize(c) for c in atac.obs_names])
            if not atac_ids.is_unique:
                raise ValueError(
                    "ATAC cell IDs collide after the configured normalization."
                )
            membership = membership.where(rna_ids.isin(atac_ids))
        finally:
            atac.file.close()
    layer = data.get("rna_layer", "lognorm")
    if layer is not None and layer not in rna.layers:
        if allow_layer_fallback:
            layer = None
        else:
            raise ValueError(
                f"RNA layer {layer!r} is missing. Use --set data.rna_layer=null when X is normalized."
            )
    symbols = pd.Series(rna.var_names.astype(str), index=rna.var_names)
    symbol_column = data.get("gene_symbol_column")
    if symbol_column:
        if symbol_column not in rna.var:
            raise ValueError(f"RNA var has no gene symbol column {symbol_column!r}.")
        symbols = rna.var[symbol_column].astype(str)
    else:
        lookup = {}
        for key in ("tg_list", "tf_list"):
            if not data.get(key):
                continue
            annot = read_gene_table(resolve_config_path(config, data[key]))
            id_col = next(
                (c for c in ("FBgn_ID", "FBgn", "gene_id", "id") if c in annot), None
            )
            if id_col and "gene_name" in annot:
                current = (
                    annot.dropna(subset=[id_col, "gene_name"])
                    .set_index(id_col)
                    .gene_name.to_dict()
                )
                for gene_id, name in current.items():
                    if gene_id in lookup and lookup[gene_id].lower() != name.lower():
                        raise ValueError(
                            f"Conflicting gene symbols for RNA ID {gene_id!r} in TF/TG annotations."
                        )
                    lookup[gene_id] = name
        symbols = symbols.map(lambda key: lookup.get(key, key))
    symbols = resolve_symbols(symbols, genes)
    keep = symbols.isin(set(genes)).to_numpy()
    matrix = (rna.X if layer is None else rna.layers[layer])[:, keep]
    mean, frac = {}, {}
    for group in dict.fromkeys(groups):
        cells = np.flatnonzero((membership == group).to_numpy())
        if not len(cells):
            raise ValueError(
                f"No RNA cells match CCC stage group {group!r}; check stage_groups and annotation IDs."
            )
        values = matrix[cells]
        observed = values.data if sparse.issparse(values) else np.asarray(values)
        if not np.isfinite(observed).all() or np.any(observed < 0):
            raise ValueError(
                "CCC expression must be finite, nonnegative normalized counts."
            )
        mean[group] = np.asarray(values.mean(axis=0)).ravel()
        frac[group] = np.asarray((values > 0).mean(axis=0)).ravel()
    names = symbols.iloc[np.flatnonzero(keep)].tolist()
    if len(names) != len(set(names)):
        raise ValueError(
            "RNA gene symbols are not unique; resolve duplicate symbols before CCC."
        )
    order = list(genes)
    mean_frame = pd.DataFrame(mean, index=names).reindex(order, fill_value=0.0)
    mean_frame.attrs["missing_genes"] = sorted(set(order) - set(names))
    mean_frame.attrs["cell_counts"] = {g: int((membership == g).sum()) for g in mean}
    return mean_frame, pd.DataFrame(frac, index=names).reindex(order, fill_value=0.0)


def _pair_scores(
    lr: pd.DataFrame,
    mean: pd.DataFrame,
    frac: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
    nodes: dict[str, list[list[str]]],
    labels: dict[str, list[str]],
    positions: dict[str, np.ndarray],
    section: dict[str, Any],
    native: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs, curves = [], []
    ligand_gate = float(section.get("ligand_min_frac", 0.05))
    receptor_gate = float(section.get("receptor_gate", 0.05))
    min_range = float(section.get("dynamic_range_min", 0.1))
    if not (0 <= ligand_gate <= 1 and 0 <= receptor_gate <= 1 and 0 <= min_range <= 1):
        raise ValueError("Expression gates and dynamic_range_min must lie in [0, 1].")
    expr = {key: _collapse(mean, groups) for key, groups in nodes.items()}
    detect = {key: _collapse(frac, groups) for key, groups in nodes.items()}
    centrality = {key: _collapse(frame, nodes[key]) for key, frame in scores.items()}
    directions = section.get("_direction_specs", DIRECTIONS)
    requested = section.get("directions", list(directions))
    for direction in requested:
        if direction not in directions:
            raise ValueError(f"Unknown CCC direction {direction!r}.")
        sender, receiver = directions[direction]
        if sender not in nodes:
            raise ValueError(f"{direction} requires ccc.hub_group.")
        n = len(nodes[receiver])
        receiver_spec = section.get(receiver, {})
        early_count = int(
            receiver_spec.get("early_stage_count", 6 if receiver == "germ" else 4)
        )
        for ligand, receptor in lr[["ligand", "receptor"]].itertuples(
            index=False, name=None
        ):
            if (
                detect[sender].loc[ligand].max() < ligand_gate
                or detect[receiver].loc[receptor].max() < receptor_gate
                or receptor not in centrality[receiver].index
            ):
                continue
            ligand_values = expr[sender].loc[ligand].to_numpy(float)
            ligand_curve = (
                np.full(n, ligand_values[0])
                if sender == "hub"
                else np.interp(
                    positions[direction],
                    np.arange(1, len(ligand_values) + 1),
                    ligand_values,
                )
            )
            gate = detect[receiver].loc[receptor].to_numpy(float) >= receptor_gate
            profiles = {
                "DRAFT": ligand_curve
                * centrality[receiver].loc[receptor].to_numpy(float),
                "LR_expression": ligand_curve
                * expr[receiver].loc[receptor].to_numpy(float),
            }
            for method, tables in (native or {}).items():
                frame = tables.get((ligand, receptor))
                if frame is None:
                    profiles[method] = np.zeros(n)
                    continue
                matrix = np.array(
                    [
                        [
                            frame.loc[snode, rnode].to_numpy(float).mean()
                            for rnode in nodes[receiver]
                        ]
                        for snode in nodes[sender]
                    ]
                )
                profiles[method] = (
                    matrix[0]
                    if sender == "hub"
                    else np.array(
                        [
                            np.interp(
                                positions[direction][i],
                                np.arange(1, len(matrix) + 1),
                                matrix[:, i],
                            )
                            for i in range(n)
                        ]
                    )
                )
            for method, profile in profiles.items():
                curve = profile * gate
                identity = {
                    "direction": direction,
                    "method": method,
                    "ligand": ligand,
                    "receptor": receptor,
                }
                pairs.append(
                    {
                        **identity,
                        **timing_call(curve, labels[receiver], early_count, min_range),
                    }
                )
                curves.extend(
                    {
                        **identity,
                        "stage_index": i,
                        "stage": labels[receiver][i],
                        "score": float(value),
                        "receptor_detected": bool(gate[i]),
                    }
                    for i, value in enumerate(curve)
                )
    return pd.DataFrame(pairs), pd.DataFrame(curves)


def _native_baselines(
    config: dict[str, Any], groups: list[str], keys: GeneKeys | None = None
) -> tuple[dict, dict]:
    keys = keys or GeneKeys()
    native, provenance = {}, {}
    for method, spec in config.get("ccc", {}).get("baselines", {}).items():
        if method in {"DRAFT", "LR_expression"}:
            raise ValueError(
                "External baseline names must differ from DRAFT and LR_expression."
            )
        path = resolve_config_path(config, spec["edges"])
        data = pd.read_csv(path, sep="\t")
        for column in (spec["sender"], spec["receiver"]):
            data[column] = data[column].astype(str)
            if not set(data[column]) & set(groups):
                raise ValueError(
                    f"{method} {column} groups do not overlap the RNA stage groups."
                )
        if "key" in spec:
            pair = (
                data[spec["key"]]
                .astype(str)
                .str.split(spec.get("key_separator", "_"), n=1, expand=True)
            )
            if pair.shape[1] != 2:
                raise ValueError(
                    f"Cannot split {method} LR keys; provide ligand/receptor column mappings instead."
                )
            data["__ligand"], data["__receptor"] = pair[0], pair[1]
        else:
            data["__ligand"] = data[spec.get("ligand", "ligand")].astype(str)
            data["__receptor"] = data[spec.get("receptor", "receptor")].astype(str)
        data[spec["score"]] = pd.to_numeric(data[spec["score"]], errors="raise")
        data["__ligand"] = keys.series(data["__ligand"])
        data["__receptor"] = keys.series(data["__receptor"])
        values = data[spec["score"]].to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{method} scores must be finite and nonnegative.")
        native[method] = {
            (ligand, receptor): part.pivot_table(
                index=spec["sender"],
                columns=spec["receiver"],
                values=spec["score"],
                aggfunc="max",
            )
            .reindex(index=groups, columns=groups)
            .fillna(0.0)
            for (ligand, receptor), part in data.groupby(["__ligand", "__receptor"])
        }
        provenance[method] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return native, provenance


def literature_benchmark(
    pairs: pd.DataFrame,
    literature: pd.DataFrame,
    seed: int = 2025,
    draws: int = 1000,
    keys: GeneKeys | None = None,
) -> pd.DataFrame:
    """Balanced non-literature draws shared by every method of a direction."""
    from sklearn.metrics import roc_auc_score

    if draws < 1:
        raise ValueError("benchmark_draws must be positive.")
    keys = keys or GeneKeys(pairs.ligand, pairs.receptor)
    literature = literature.copy()
    for col in ("ligand", "receptor"):
        literature[col] = keys.series(literature[col].astype(str))
    keys = set(
        literature[["direction", "ligand", "receptor"]].itertuples(
            index=False, name=None
        )
    )
    rng, results = np.random.default_rng(seed), []
    for direction, data in pairs.groupby("direction", sort=False):
        reference = data[data.method == "DRAFT"].set_index(["ligand", "receptor"])
        positive = np.array([(direction, a, b) in keys for a, b in reference.index])
        ip, ine = np.flatnonzero(positive), np.flatnonzero(~positive)
        enough = len(ip) > 1 and len(ine) >= len(ip)
        samples = (
            [rng.choice(ine, len(ip), replace=False) for _ in range(draws)]
            if enough
            else []
        )
        for method, part in data.groupby("method", sort=False):
            part = part.set_index(["ligand", "receptor"]).reindex(reference.index)
            scores = part.earliness_score.to_numpy(float)
            auc = (
                np.mean(
                    [
                        roc_auc_score(
                            np.r_[np.ones(len(ip)), np.zeros(len(ip))],
                            scores[np.r_[ip, sample]],
                        )
                        for sample in samples
                    ]
                )
                if enough
                else np.nan
            )
            results.append(
                {
                    "direction": direction,
                    "method": method,
                    "literature_count": len(ip),
                    "other_count": len(ine),
                    "literature_early": int(
                        (part.status.to_numpy()[ip] == "early").sum()
                    ),
                    "balanced_auroc": float(auc),
                    "draws": draws if enough else 0,
                    "status": "evaluated" if enough else "insufficient_reference_pairs",
                }
            )
    return pd.DataFrame(results)


def run_receptor_ccc(config: dict[str, Any]) -> dict[str, Any]:
    from .ccc import _ccc_settings, _set_seed
    from .ccc_config import normalize_ccc_config

    config = normalize_ccc_config(config)
    settings, section = _ccc_settings(config), config.get("ccc", {})
    aliases = section.get("_lineage_aliases", {})
    directions = section.get("_direction_specs", DIRECTIONS)
    output = ensure_directory(resolve_config_path(config, settings["output_dir"]))
    _set_seed(settings["seed"])
    scores, groups, metadata = {}, {}, {}
    raw = {
        name: _raw_receiver_scores(config, aliases.get(name, name), section.get(name, {}))
        for name in ("soma", "germ")
    }
    # One case rule for both lineages: genes that differ only by case in
    # either gene table (such as Drosophila Tor and tor) stay separate.
    keys = GeneKeys(*(item[3] for item in raw.values()))
    for name, (frame, stage_groups, meta, _) in raw.items():
        scores[name], groups[name], metadata[name] = _keyed_receiver_scores(
            frame, stage_groups, meta, keys, aliases.get(name, name), output
        )
    nodes = {
        key: stage_nodes(value, section.get(key, {}).get("parallel_groups"))
        for key, value in groups.items()
    }
    labels = {
        key: section.get(key, {}).get("stage_labels")
        or [
            "/".join(
                group.removeprefix(str(section.get(key, {}).get("stage_prefix", "")))
                for group in node
            )
            for node in value
        ]
        for key, value in nodes.items()
    }
    if any(len(labels[key]) != len(nodes[key]) for key in nodes):
        raise ValueError(
            "stage_labels must have one label per receiver time point after merging parallel groups."
        )
    if section.get("hub_group"):
        nodes["hub"] = [[str(section["hub_group"])]]
    lr = pd.read_csv(resolve_config_path(config, settings["ccc_csv"]))
    lr = lr.rename(
        columns={"source_genesymbol": "ligand", "target_genesymbol": "receptor"}
    )
    if not {"ligand", "receptor"}.issubset(lr.columns):
        raise ValueError(
            "LR resource requires ligand/receptor or source_genesymbol/target_genesymbol columns."
        )
    lr = lr[["ligand", "receptor"]].dropna().astype(str)
    symbols = pd.unique(lr.to_numpy().ravel())
    ambiguous = sorted(s for s in symbols if keys(s) is None)
    lr = lr.apply(keys.series).dropna().drop_duplicates()
    common_genes = set(scores["soma"].index) & set(scores["germ"].index)
    lr = lr[lr.ligand.isin(common_genes) & lr.receptor.isin(common_genes)]
    if lr.empty:
        raise ValueError("No ligand-receptor pairs overlap both lineage gene sets.")
    genes = keys.request(sorted(set(lr.ligand) | set(lr.receptor)))
    all_groups = [g for value in nodes.values() for node in value for g in node]
    mean, frac = load_group_expression(config, all_groups, genes)
    alignment, alignment_report = network_alignment(config, output)
    soma_at_germ, germ_at_soma = aligned_positions(
        alignment, groups["soma"], nodes["soma"], len(nodes["germ"])
    )
    positions = {
        name: soma_at_germ if (sender, receiver) == ("soma", "germ") else germ_at_soma
        for name, (sender, receiver) in directions.items()
        if sender != "hub"
    }
    native, baseline_provenance = _native_baselines(
        config, list(dict.fromkeys(all_groups)), keys
    )
    pairs, curves = _pair_scores(
        lr, mean, frac, scores, nodes, labels, positions, section, native
    )
    if pairs.empty:
        raise ValueError(
            "No LR candidates pass the expression gates in the requested directions."
        )
    pairs_path, curves_path = output / "ccc_pairs.tsv", output / "ccc_curves.tsv"
    pairs.to_csv(pairs_path, sep="\t", index=False)
    curves.to_csv(curves_path, sep="\t", index=False)
    benchmark_path = None
    if section.get("literature_pairs"):
        literature = pd.read_csv(
            resolve_config_path(config, section["literature_pairs"]), sep="\t"
        )
        benchmark = literature_benchmark(
            pairs,
            literature,
            settings["seed"],
            int(section.get("benchmark_draws", 1000)),
            keys,
        )
        benchmark_path = output / "ccc_benchmark.tsv"
        benchmark.to_csv(benchmark_path, sep="\t", index=False)
    mean.to_csv(output / "group_mean_expression.tsv", sep="\t", index_label="gene_name")
    frac.to_csv(
        output / "group_detection_fraction.tsv", sep="\t", index_label="gene_name"
    )
    pd.DataFrame(
        {
            f"{aliases.get('soma', 'soma')}_stage": labels["soma"],
            f"{aliases.get('germ', 'germ')}_position": germ_at_soma,
        }
    ).to_csv(output / "ccc_stage_alignment.tsv", sep="\t", index=False)
    figures = []
    if section.get("save_fig", True):
        import matplotlib.pyplot as plt

        for direction, group in curves[curves.method == "DRAFT"].groupby(
            "direction", sort=False
        ):
            matrix = group.pivot(
                index=["ligand", "receptor"], columns="stage_index", values="score"
            )
            values = matrix.to_numpy(float)
            values = values / np.maximum(values.max(axis=1, keepdims=True), 1e-12)
            fig, ax = plt.subplots(figsize=(10, max(3, min(18, len(matrix) * 0.2))))
            ax.imshow(
                values,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            receiver = directions[direction][1]
            ax.set_xticks(
                np.arange(len(labels[receiver])),
                labels[receiver],
                rotation=45,
                ha="right",
            )
            ax.set_yticks(
                np.arange(len(matrix)),
                [f"{a}–{b}" for a, b in matrix.index],
                fontsize=7,
            )
            ax.set_title(f"{direction}: DRAFT score / pair maximum")
            path = output / f"{direction}_heatmap.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            figures.append(str(path))
    report = {
        "analysis": "ccc",
        "score_mode": "receptor_centrality",
        "output_dir": str(output),
        "chosen_alignment": "DTW",
        "alignment": alignment_report,
        "receiver_scores": {
            aliases.get(key, key): value for key, value in metadata.items()
        },
        "expression_inputs": {
            "lineages": config.get("data", {}).get("lineages"),
            "rna_h5ad": config.get("data", {}).get("rna_h5ad"),
            "atac_h5ad": config.get("data", {}).get("atac_h5ad"),
            "cell_id_normalization": config.get("data", {}).get(
                "cell_id_normalization", "sample_suffix"
            ),
            "cell_counts": mean.attrs.get("cell_counts", {}),
            "missing_genes": mean.attrs.get("missing_genes", []),
        },
        # LR symbols skipped because they match genes that differ only by case.
        "lr_symbols_ambiguous_by_case": ambiguous,
        "method_profile": config.get("method_profile"),
        "ligand_min_frac": float(section.get("ligand_min_frac", 0.05)),
        "receptor_gate": float(section.get("receptor_gate", 0.05)),
        "dynamic_range_min": float(section.get("dynamic_range_min", 0.1)),
        "stage_nodes": {aliases.get(key, key): value for key, value in nodes.items()},
        "seed": settings["seed"],
        "pairs_path": str(pairs_path),
        "curves_path": str(curves_path),
        "figures": figures,
        "baselines": baseline_provenance,
        "benchmark_path": str(benchmark_path) if benchmark_path else None,
        "candidates_per_direction": pairs[pairs.method == "DRAFT"]
        .groupby("direction")
        .size()
        .to_dict(),
    }
    if aliases:
        report["lineage_order"] = list(aliases.values())
    save_checkpoint_metadata(output / "ccc_report.json", report)
    return report
