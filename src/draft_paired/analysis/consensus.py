"""Window-based target-rank aggregation shared by paired and unpaired workflows."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..checkpoint import save_checkpoint_metadata
from ..utils import resolve_config_path
from .repro import safe_label


def consensus_times(spec: dict[str, Any], labels: list[str]) -> list[float]:
    points = spec.get("time_points")
    if points is not None and spec.get("time_window") is not None:
        raise ValueError("Set either consensus.time_points or time_window, not both.")
    if points is None:
        window = spec.get("time_window")
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError(
                "Consensus requires time_points or a two-endpoint time_window."
            )
        start, end = [float(v) for v in window]
        interpolation = int(spec.get("n_interp", 4))
        if (
            not start.is_integer()
            or not end.is_integer()
            or start > end
            or interpolation < 0
        ):
            raise ValueError(
                "Consensus window endpoints must be ordered integer stage indices; n_interp must be nonnegative."
            )
        points = np.arange(
            int(start) * (interpolation + 1), int(end) * (interpolation + 1) + 1
        ) / (interpolation + 1)
    if len(points) == 0:
        raise ValueError("Consensus needs at least one stage.")
    times = [
        float(labels.index(v)) if isinstance(v, str) and v in labels else float(v)
        for v in points
    ]
    if (
        len(set(times)) != len(times)
        or not np.isfinite(times).all()
        or min(times) < 0
        or max(times) > len(labels) - 1
    ):
        raise ValueError(
            "Consensus times must be distinct and inside the model stage range (zero-based)."
        )
    return sorted(times)


def sample_embeddings(values: np.ndarray, time: float, sigma: float) -> np.ndarray:
    time = float(time)
    if time.is_integer():
        return values[:, int(time), :]
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Consensus interpolation sigma must be positive.")
    weights = np.exp(-0.5 * ((np.arange(values.shape[1]) - time) / sigma) ** 2)
    weights /= weights.sum()
    return np.einsum("ntd,t->nd", values, weights, optimize=True)


def stage_rankings(
    tf: np.ndarray,
    tg: np.ndarray,
    tf_idx: int,
    tg_ids: list[str],
    tg_names: list[str],
    fractions: list[np.ndarray | None],
    times: list[float],
    spec: dict[str, Any],
    *,
    min_fraction: float,
    max_fraction: float,
    absolute_cosine: bool = False,
    parents: list[list[int]] | None = None,
) -> dict[str, pd.DataFrame]:
    fractional = any(not t.is_integer() for t in times)
    if fractional and parents is not None:
        # Whole-lineage Gaussian interpolation is only defined on a chain.
        edges = [
            (p, i)
            for i, ps in enumerate(parents)
            for p in ps
            if p is not None and p >= 0
        ]
        if len(edges) != len(parents) - 1 or any(
            sum(p == i for p, _ in edges) > 1 for i in range(len(parents))
        ):
            raise ValueError(
                "Fractional consensus times require a linear lineage; use integer time_points on a branching lineage."
            )
    tables = {}
    for time in times:
        left = int(np.floor(time))
        frac = fractions[left]
        columns = ["tg_id", "tg_name", "tg_idx", "rank", "score"]
        if frac is None:
            tables[f"{time:g}"] = pd.DataFrame(columns=columns)
            continue
        mask = np.isfinite(frac) & (frac >= min_fraction) & (frac <= max_fraction)
        indices = np.flatnonzero(mask)
        tv = sample_embeddings(
            tf[tf_idx : tf_idx + 1], time, float(spec.get("sigma_time", 0.7))
        )[0]
        gv = sample_embeddings(tg, time, float(spec.get("sigma_time", 0.7)))
        score = (gv @ tv) / np.maximum(
            np.linalg.norm(gv, axis=1) * np.linalg.norm(tv), 1e-12
        )
        if absolute_cosine:
            score = np.abs(score)
        if not np.isfinite(score).all():
            raise ValueError("Consensus model scores must be finite.")
        order = indices[np.lexsort((np.asarray(tg_ids)[indices], -score[indices]))]
        limit = spec.get("stage_topn")
        if isinstance(limit, dict):
            limit = limit.get(f"{time:g}", limit.get(time))
        if limit is not None:
            if int(limit) < 1:
                raise ValueError("consensus.stage_topn must be positive or null.")
            order = order[: int(limit)]
        tables[f"{time:g}"] = pd.DataFrame(
            {
                "tg_id": [str(tg_ids[i]) for i in order],
                "tg_name": [str(tg_names[i]) for i in order],
                "tg_idx": order,
                "rank": np.arange(1, len(order) + 1),
                "score": score[order],
            }
        )
    return tables


def _cluster_targets(
    path: Path,
    tf_name: str,
    tf_id: str,
    tf_idx: int,
    tables: dict[str, pd.DataFrame],
    requested: Any,
) -> tuple[set[str], Any]:
    edges = pd.read_csv(path)
    if "edge_cluster" not in edges:
        raise ValueError("Consensus cluster table requires edge_cluster.")
    if "tf_id" in edges:
        edges = edges[edges.tf_id.astype(str) == tf_id]
    elif "tf_idx" in edges:
        edges = edges[pd.to_numeric(edges.tf_idx, errors="raise") == tf_idx]
    elif "tf_name" in edges:
        edges = edges[edges.tf_name.astype(str).str.lower() == tf_name.lower()]
    else:
        raise ValueError("Consensus cluster table requires tf_id, tf_idx, or tf_name.")
    edges = edges.copy()
    catalog = pd.concat(list(tables.values()), ignore_index=True).drop_duplicates(
        "tg_id"
    )
    if "tg_id" in edges:
        edges["canonical_id"] = edges.tg_id.astype(str)
    elif "tg_idx" in edges:
        lookup = catalog.set_index("tg_idx").tg_id.to_dict()
        edges["canonical_id"] = edges.tg_idx.map(lookup)
    elif "tg_name" in edges:
        if catalog.tg_name.str.lower().duplicated().any():
            raise ValueError(
                "Gene symbols are ambiguous; use IDs or indices in the cluster table."
            )
        lookup = dict(zip(catalog.tg_name.str.lower(), catalog.tg_id))
        edges["canonical_id"] = edges.tg_name.astype(str).str.lower().map(lookup)
    else:
        raise ValueError("Consensus cluster table requires tg_id, tg_idx, or tg_name.")
    edges = edges.dropna(subset=["canonical_id"])
    if edges.empty:
        raise ValueError(f"No eligible consensus edges overlap {tf_name} in {path}.")
    if requested is None:
        present = pd.concat(
            [table[["tg_id"]] for table in tables.values()], ignore_index=True
        )
        overlap = present.merge(
            edges[["canonical_id", "edge_cluster"]].drop_duplicates(),
            left_on="tg_id",
            right_on="canonical_id",
        )
        if overlap.empty:
            raise ValueError(
                "No selected stage targets overlap the edge-cluster table."
            )
        counts = overlap.groupby("edge_cluster").size()
        requested = min(counts.index[counts == counts.max()], key=str)
    selected = edges[edges.edge_cluster.astype(str) == str(requested)]
    if selected.empty:
        raise ValueError(
            f"TF {tf_name} has no eligible targets in edge cluster {requested}."
        )
    return set(selected.canonical_id), str(requested)


def aggregate_ranks(
    tables: dict[str, pd.DataFrame],
    *,
    method: str = "geometric_mean",
    rrf_k: float = 60.0,
    allowed: set[str] | None = None,
    exclude_pattern: str | None = None,
) -> pd.DataFrame:
    if (
        method not in {"geometric_mean", "normalized_geometric_mean", "rrf"}
        or rrf_k < 0
    ):
        raise ValueError(
            "Consensus rank_method must be geometric_mean, normalized_geometric_mean, or rrf; rrf_k must be nonnegative."
        )
    catalog = (
        pd.concat(list(tables.values()), ignore_index=True)
        .drop_duplicates("tg_id")
        .set_index("tg_id")
    )
    if allowed is not None:
        catalog = catalog.loc[catalog.index.isin(allowed)]
    if exclude_pattern:
        catalog = catalog.loc[
            ~catalog.tg_name.astype(str).str.match(exclude_pattern, case=False)
        ]
    if catalog.empty:
        raise ValueError(
            "No consensus targets remain after eligibility and cluster/name filters."
        )
    ranks, normalized, scores, raw = [], [], [], []
    for time, table in tables.items():
        if table.tg_id.duplicated().any():
            raise ValueError(f"Stage {time} contains duplicate consensus target IDs.")
        stage = table.set_index("tg_id").reindex(catalog.index)
        r = stage["rank"].to_numpy(float)
        if np.any(r[np.isfinite(r)] <= 0):
            raise ValueError("Consensus ranks must be positive.")
        raw.append(r)
        ranks.append(np.where(np.isfinite(r), r, len(table) + 1))
        normalized.append(np.where(np.isfinite(r), (r - 0.5) / max(1, len(table)), 1.0))
        scores.append(stage.score.to_numpy(float))
    r, n, s, original = map(np.asarray, (ranks, normalized, scores, raw))
    result = catalog[["tg_name", "tg_idx"]].copy()
    result["geometric_mean_rank"] = np.exp(np.log(r).mean(axis=0))
    result["geometric_mean_normalized_rank"] = np.exp(np.log(n).mean(axis=0))
    result["rrf_score"] = np.where(
        np.isfinite(original), 1 / (rrf_k + original), 0
    ).mean(axis=0)
    result["stage_count"] = np.isfinite(original).sum(axis=0)
    result["mean_score"] = np.nanmean(s, axis=0)
    result["best_time"] = np.asarray(list(tables))[
        np.argmin(np.where(np.isfinite(original), original, np.inf), axis=0)
    ]
    for i, time in enumerate(tables):
        result[f"rank_t{time}"] = original[i]
    column = {
        "geometric_mean": "geometric_mean_rank",
        "normalized_geometric_mean": "geometric_mean_normalized_rank",
        "rrf": "rrf_score",
    }[method]
    order = (
        [column]
        + (
            ["geometric_mean_normalized_rank"]
            if column != "geometric_mean_normalized_rank"
            else []
        )
        + ["mean_score", "tg_name", "tg_id"]
    )
    result = result.reset_index().sort_values(
        order,
        ascending=[method != "rrf"]
        + ([True] if column != "geometric_mean_normalized_rank" else [])
        + [False, True, True],
        kind="stable",
    )
    result.insert(0, "consensus_rank", np.arange(1, len(result) + 1))
    return result


def export_consensus(
    config: dict[str, Any],
    spec: dict[str, Any],
    tf_name: str,
    tf_id: str,
    tf_idx: int,
    tables: dict[str, pd.DataFrame],
    output: Path,
    re_selector: Callable[[float, list[str]], list[dict]] | None = None,
) -> dict[str, Any]:
    allowed, cluster = None, None
    cluster_source = spec.get("edge_cluster_file")
    if cluster_source:
        path = resolve_config_path(config, cluster_source)
        allowed, cluster = _cluster_targets(
            path, tf_name, tf_id, tf_idx, tables, spec.get("edge_cluster")
        )
    elif spec.get("edge_cluster") is not None:
        raise ValueError("edge_cluster requires consensus.edge_cluster_file.")
    method = str(spec.get("rank_method", "geometric_mean"))
    ranked = aggregate_ranks(
        tables,
        method=method,
        rrf_k=float(spec.get("rrf_k", 60)),
        allowed=allowed,
        exclude_pattern=spec.get("exclude_gene_pattern"),
    )
    topn = int(spec.get("topn", 20))
    if topn < 1:
        raise ValueError("Consensus topn must be positive.")
    selected = ranked.head(topn)
    prefix = output / f"{safe_label(tf_name)}_consensus"
    all_path, top_path = (
        Path(f"{prefix}_all_targets.tsv"),
        Path(f"{prefix}_top_targets.tsv"),
    )
    ranked.to_csv(all_path, sep="\t", index=False)
    selected.to_csv(top_path, sep="\t", index=False)
    long = pd.concat(
        [table.assign(time=float(time)) for time, table in tables.items()],
        ignore_index=True,
    )
    rank_path = Path(f"{prefix}_stage_ranks.tsv")
    long.to_csv(rank_path, sep="\t", index=False)
    re_path, target_res = None, [[] for _ in range(len(selected))]
    if re_selector is not None:
        evidence = []
        for time, table in tables.items():
            ids = [key for key in selected.tg_id if key in set(table.tg_id)]
            evidence.extend(re_selector(float(time), ids))
        if evidence:
            frame = pd.DataFrame(evidence)
            combined = frame.groupby(["TG_id", "RE_name"], as_index=False).agg(
                selected_stage_count=("time_idx", "nunique"),
                mean_local_score=("score_triplet_local", "mean"),
            )
            combined = combined.sort_values(
                ["TG_id", "selected_stage_count", "mean_local_score", "RE_name"],
                ascending=[True, False, False, True],
            )
            n_re = int(spec.get("topn_re_per_tg", 2))
            if n_re < 1:
                raise ValueError("Consensus topn_re_per_tg must be positive.")
            combined = combined.groupby("TG_id", sort=False).head(n_re)
            re_path = Path(f"{prefix}_regulon_re.tsv")
            combined.to_csv(re_path, sep="\t", index=False)
            target_res = [
                combined.loc[combined.TG_id == key, "RE_name"].tolist()
                for key in selected.tg_id
            ]
    report = {
        "tf": tf_name,
        "tf_id": tf_id,
        "rank_method": method,
        "times": [float(t) for t in tables],
        "candidate_count": len(ranked),
        "edge_cluster": cluster,
        "cluster_source": str(resolve_config_path(config, cluster_source))
        if cluster_source
        else None,
        "cluster_sha256": hashlib.sha256(
            resolve_config_path(config, cluster_source).read_bytes()
        ).hexdigest()
        if cluster_source
        else None,
        "missing_rank": "stage list length + 1; missing normalized rank is 1; missing RRF contribution is 0",
        "eligibility_scope": "left integer-stage expression fractions at fractional grid points",
        "all_targets": str(all_path),
        "top_targets": str(top_path),
        "stage_ranks": str(rank_path),
        "regulon_re": str(re_path) if re_path else None,
        "target_ids": selected.tg_id.tolist(),
        "target_names": selected.tg_name.tolist(),
        "target_res": target_res,
        "settings": spec,
        "method_profile": config.get("method_profile"),
    }
    save_checkpoint_metadata(Path(f"{prefix}_report.json"), report)
    return report
