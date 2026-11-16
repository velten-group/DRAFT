"""Stream interpolated network matrices from checkpoints when aligning lineages."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ..utils import load_yaml_config, resolve_config_path


@dataclass
class AlignmentSeries:
    tf_names: list[str]
    tg_names: list[str]
    visit: Callable[[Callable[[float, Any], None]], None]
    provenance: dict[str, Any]


def alignment_series(
    config: dict[str, Any], lineage: str, settings: dict[str, Any]
) -> AlignmentSeries:
    from .ccc import EDGE_GLOBS, _load_network_series

    spec = config.get("ccc", {}).get(lineage, {})
    configured_dir = spec.get("network_dir") or config.get("ccc", {}).get(
        f"network_dir_{lineage}"
    )
    if configured_dir and any(
        next(resolve_config_path(config, configured_dir).glob(pattern), None)
        is not None
        for pattern in EDGE_GLOBS
    ):
        times, matrices, tfs, tgs = _load_network_series(
            resolve_config_path(config, configured_dir),
            resolve_config_path(config, settings[f"tf_annot_{lineage}"]),
            resolve_config_path(config, settings[f"tg_annot_{lineage}"]),
            keep_case=True,
        )

        def visit_saved(consume):
            for time, matrix in zip(times, matrices):
                consume(float(time), matrix)

        return AlignmentSeries(
            tfs,
            tgs,
            visit_saved,
            {"source": "exported", "network_dir": str(configured_dir)},
        )

    if not spec.get("config"):
        # Load an externally supplied network series.
        directory = resolve_config_path(config, settings[f"network_dir_{lineage}"])
        times, matrices, tfs, tgs = _load_network_series(
            directory,
            resolve_config_path(config, settings[f"tf_annot_{lineage}"]),
            resolve_config_path(config, settings[f"tg_annot_{lineage}"]),
            keep_case=True,
        )

        def visit_external(consume):
            for time, matrix in zip(times, matrices):
                consume(float(time), matrix)

        return AlignmentSeries(
            tfs,
            tgs,
            visit_external,
            {"source": "exported", "network_dir": str(directory)},
        )

    config_path = resolve_config_path(config, spec["config"])
    model_config = load_yaml_config(config_path)
    mode = (
        spec.get("input_mode")
        or model_config.get("input_mode")
        or ("unpaired" if model_config.get("data", {}).get("tf_tg_prior") else "paired")
    )
    if (
        mode not in {"paired", "unpaired"}
        or model_config.get("input_mode", mode) != mode
    ):
        raise ValueError(
            "Alignment input_mode must match the paired/unpaired training config."
        )
    model_config.setdefault("base_dir", config.get("base_dir", "."))
    return checkpoint_series(model_config, mode=mode, config_path=str(config_path))


def checkpoint_series(
    model_config: dict[str, Any],
    *,
    mode: str | None = None,
    config_path: str | None = None,
) -> AlignmentSeries:
    from .gene_keys import clean_symbol

    mode = (
        mode
        or model_config.get("input_mode")
        or ("unpaired" if model_config.get("data", {}).get("tf_tg_prior") else "paired")
    )
    if mode not in {"paired", "unpaired"}:
        raise ValueError("Network model mode must be paired or unpaired.")
    module = importlib.import_module(f"draft_{mode}.analysis.network")
    context = module._network_context(model_config)
    if mode == "paired":
        tf_ids, tf_names = context["refs"]["tf_ids"], context["refs"]["tf_names"]
        tg_names = context["refs"]["tg_names"]
        tf_export, tg_export = tf_names, tg_names
    else:
        tf_ids = context["tf_names"]
        tf_export, tg_export = context["tf_names"], context["tg_names"]
        tf_names = context["common"].maps.get("tf_names", tf_export)
        tg_names = context["common"].maps.get("tg_names", tg_export)

    def clean(names):
        return [clean_symbol(name) for name in names]

    def positions(names):
        """Exact spelling first; ignoring case only where that stays unique."""
        exact = dict(zip(clean(names), range(len(names))))
        folded: dict[str, set[int]] = {}
        for name, index in exact.items():
            folded.setdefault(name.lower(), set()).add(index)

        def lookup(values):
            cleaned = pd.Series(values, dtype=str).map(clean_symbol)
            found = cleaned.map(exact)
            missing = found.isna()
            if missing.any():
                unique = {k: next(iter(v)) for k, v in folded.items() if len(v) == 1}
                found = found.where(~missing, cleaned.str.lower().map(unique))
            if found.isna().any():
                raise ValueError(
                    f"Exported network genes are missing from the model: {sorted(set(cleaned[found.isna()]))[:5]}"
                )
            return found.to_numpy(dtype=int)

        return lookup

    # Names keep their case; alignment decides how case is compared across lineages.
    tfs, tgs = clean(tf_names), clean(tg_names)
    tf_rows, tg_cols = positions(tf_export), positions(tg_export)
    dense_settings = context["dense_settings"]

    def visit_model(consume):
        def frame_consumer(label, frame):
            if "TF_idx" in frame and "TG_idx" in frame:
                offset = int(dense_settings["use_one_based"])
                rows = frame.TF_idx.to_numpy(dtype=int) - offset
                cols = frame.TG_idx.to_numpy(dtype=int) - offset
            else:
                rows = tf_rows(frame.TF)
                cols = tg_cols(frame.TG)
            values = frame.prob.to_numpy(dtype=np.float32)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(
                    "Alignment network weights must be finite and nonnegative."
                )
            matrix = sparse.coo_matrix(
                (values, (rows, cols)), shape=(len(tfs), len(tgs))
            ).tocsr()
            matrix.eliminate_zeros()
            consume(float(label), matrix)

        kwargs = {
            "output_dir": None,
            "out_tf": context["inference"]["out_tf"],
            "out_tg": context["inference"]["out_tg"],
            "tf_names": tf_export,
            "tg_names": tg_export,
            "settings": dense_settings,
            "frame_consumer": frame_consumer,
        }
        if mode == "paired":
            kwargs.update(
                lineage=context["lineage"],
                graphs_sel=context["graphs_sel"],
                tf_ids=tf_ids,
                tf_filter=context["tf_filter"],
            )
        else:
            kwargs.update(
                common=context["common"],
                graphs=context["graphs"],
                time_names=context["time_names"],
                allowed_tfs=context["allowed_tfs"],
            )
        module._export_dense_networks(**kwargs)

    return AlignmentSeries(
        tfs,
        tgs,
        visit_model,
        {
            "source": "checkpoint",
            "input_mode": mode,
            "config_path": config_path,
            "checkpoint_path": context["inference"]["checkpoint_path"],
            "settings": dense_settings,
        },
    )
