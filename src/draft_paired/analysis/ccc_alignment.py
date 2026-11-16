"""Align lineage stages using regulatory-network fingerprints and DTW."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gene_keys import GeneKeys


def fingerprint_dtw(
    soma: np.ndarray,
    germ: np.ndarray,
    *,
    band: int | None = 6,
    curvature: float = 1.0,
    n_iter: int = 50,
    entropy_weight: float = 0.0,
    entropy_soma: np.ndarray | None = None,
    entropy_germ: np.ndarray | None = None,
) -> np.ndarray:
    """Return normalized second-lineage time for each first-lineage fingerprint.

    Fingerprints are standardized by the caller. Entropy is an explicit
    sensitivity option; weight zero never reads or validates entropy arrays.
    """
    from .ccc import dtw_path, smooth_warp_d2_monotone

    soma, germ = np.asarray(soma, float), np.asarray(germ, float)
    if (
        soma.ndim != 2
        or germ.ndim != 2
        or soma.shape[1] != germ.shape[1]
        or min(len(soma), len(germ)) < 2
    ):
        raise ValueError(
            "DTW needs at least two grid points per lineage and equal fingerprint widths."
        )
    if not np.isfinite(soma).all() or not np.isfinite(germ).all():
        raise ValueError("Network fingerprints must be finite.")
    if not np.isfinite(entropy_weight) or entropy_weight < 0:
        raise ValueError("dtw_ent_w must be finite and nonnegative.")
    if not np.isfinite(curvature) or curvature < 0 or n_iter < 1:
        raise ValueError("DTW curvature must be nonnegative and dtw_n_iter positive.")
    cost = ((soma[:, None, :] - germ[None, :, :]) ** 2).sum(axis=2)
    if entropy_weight:
        es, eg = np.asarray(entropy_soma, float), np.asarray(entropy_germ, float)
        if es.shape != (len(soma),) or eg.shape != (len(germ),):
            raise ValueError(
                "Nonzero entropy weight requires one entropy value per grid point."
            )
        if not np.isfinite(es).all() or not np.isfinite(eg).all():
            raise ValueError("Entropy curves must be finite when dtw_ent_w is nonzero.")
        # Scale both costs by their means when combining them.
        extra = (es[:, None] - eg[None, :]) ** 2
        cost = cost / (cost.mean() + 1e-12) + entropy_weight * extra / (
            extra.mean() + 1e-12
        )
    path = dtw_path(cost, band)
    matches: list[list[float]] = [[] for _ in soma]
    for i, j in path:
        matches[i].append(j / (len(germ) - 1))
    warp = np.asarray([np.mean(values) for values in matches])
    warp[0], warp[-1] = 0.0, 1.0
    return smooth_warp_d2_monotone(warp, curvature, n_iter, enforce_endpoints=True)


def network_alignment(
    config: dict[str, Any], output_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from ..utils import resolve_config_path
    from .ccc import (
        _ccc_settings,
        _normalize_bipartite,
        _normalize_time_to_01,
        _restrict_to_common,
        _tf_centrality,
        _top_r_svals,
        _zscore_cols,
        _zscore_rows,
        prepare_entropy_on_net_grid,
    )

    settings = _ccc_settings(config)
    section = config.get("ccc", {})
    aliases = section.get("_lineage_aliases", {})
    columns = {f"{name}_tau": f"{key}_tau" for key, name in aliases.items()}
    if settings["alignment_method"].lower() != "dtw":
        raise ValueError("Receptor-centrality CCC uses ccc.alignment_method: dtw.")
    stored = section.get("alignment_path")
    if stored:
        source = resolve_config_path(config, stored)
        table = pd.read_csv(source, sep="\t")
        table = table.rename(columns=columns)
        if table.columns.duplicated().any():
            raise ValueError("Alignment contains ambiguous lineage coordinate columns.")
        if not {"soma_tau", "germ_tau"}.issubset(table.columns):
            raise ValueError(
                "Stored alignment requires soma_tau and germ_tau columns in [0, 1]."
            )
        report = {
            "source": "stored",
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "settings": section.get("alignment_provenance", {}),
        }
    else:
        from .ccc_networks import alignment_series

        series = [
            alignment_series(config, lineage, settings) for lineage in ("soma", "germ")
        ]
        # Genes match across lineages without regard to case unless case
        # separates genes in either lineage (for example H and h).
        keys = GeneKeys(*(names for item in series for names in (item.tf_names, item.tg_names)))
        tf_keys = [keys.keys(item.tf_names) for item in series]
        tg_keys = [keys.keys(item.tg_names) for item in series]
        tf_common = sorted(set(tf_keys[0]) & set(tf_keys[1]))
        tg_common = sorted(set(tg_keys[0]) & set(tg_keys[1]))
        if min(len(tf_common), len(tg_common)) < 2:
            raise ValueError(
                "Spectral alignment needs at least two shared TFs and TGs."
            )
        features, time_grids, centralities = [], [], []
        for item, item_tfs, item_tgs in zip(series, tf_keys, tg_keys):
            fingerprints, times, centrality = [], [], []

            def collect(
                time,
                matrix,
                item_tfs=item_tfs,
                item_tgs=item_tgs,
                fingerprints=fingerprints,
                times=times,
                centrality=centrality,
            ):
                shared = _restrict_to_common(
                    [matrix], item_tfs, item_tgs, tf_common, tg_common
                )[0]
                normalized = _normalize_bipartite(shared, settings)
                fingerprints.append(_top_r_svals(normalized, settings["finger_r"]))
                times.append(time)
                if settings["dtw_ent_w"]:
                    centrality.append(_tf_centrality(normalized))

            item.visit(collect)
            if len(times) < 2:
                raise ValueError(
                    "Alignment requires at least two network timepoints per lineage."
                )
            order = np.argsort(times)
            features.append(_zscore_cols(np.vstack(fingerprints)[order]))
            time_grids.append(np.asarray(times)[order])
            if settings["dtw_ent_w"]:
                centralities.append(np.vstack(centrality)[order])
        ts, tg = time_grids
        es = eg = None
        if settings["dtw_ent_w"]:
            cs, cg = [
                pd.DataFrame(
                    _zscore_rows(centrality.T),
                    index=tf_common,
                    columns=times,
                )
                for centrality, times in zip(centralities, (ts, tg))
            ]
            es, eg = prepare_entropy_on_net_grid(
                cs,
                cg,
                _normalize_time_to_01(ts),
                _normalize_time_to_01(tg),
                resolve_config_path(config, settings["tf_filter_path"]),
            )
        warp = fingerprint_dtw(
            features[0],
            features[1],
            band=None if settings["dtw_band"] is None else int(settings["dtw_band"]),
            curvature=settings["dtw_lam_d2"],
            n_iter=settings["dtw_n_iter"],
            entropy_weight=settings["dtw_ent_w"],
            entropy_soma=es,
            entropy_germ=eg,
        )
        table = pd.DataFrame({"soma_tau": _normalize_time_to_01(ts), "germ_tau": warp})
        table["t_soma_net"] = ts
        table["t_germ_warped"] = tg.min() + warp * (tg.max() - tg.min())
        source = output_dir / settings["warp_dtw_tsv"]
        exported = table.rename(columns={value: key for key, value in columns.items()})
        if aliases:
            exported = exported.rename(
                columns={
                    "t_soma_net": f"t_{aliases['soma']}_net",
                    "t_germ_warped": f"t_{aliases['germ']}_warped",
                }
            )
        exported.to_csv(source, sep="\t", index=False)
        report = {
            "source": "computed",
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "shared_tf_count": len(tf_common),
            "shared_tg_count": len(tg_common),
            "settings": {
                k: settings[k]
                for k in (
                    "finger_r",
                    "dtw_ent_w",
                    "dtw_band",
                    "dtw_lam_d2",
                    "dtw_n_iter",
                )
            },
            "network_sources": {
                aliases.get(key, key): item.provenance
                for key, item in zip(("soma", "germ"), series)
            },
        }
    values = table[["soma_tau", "germ_tau"]].to_numpy(float)
    if (
        len(values) < 2
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
        or np.any(np.diff(values[:, 0]) <= 0)
        or np.any(np.diff(values[:, 1]) < 0)
        or not np.allclose(values[[0, -1]], [[0, 0], [1, 1]], atol=1e-8)
    ):
        raise ValueError(
            "Alignment must be finite, monotone, and fixed at (0,0) and (1,1)."
        )
    return table, report
