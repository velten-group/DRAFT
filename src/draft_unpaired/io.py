"""Input helpers for draft_unpaired."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from draft_paired.io import resolve_copy_suffix_path, save_text_report, strip_copy_suffix


@dataclass(frozen=True)
class InputPaths:
    base_dir: Path
    tf_list: Path
    tg_list: Path
    rna_h5ad: Path
    annotation: Path
    tree: Path
    tf_tg_prior: Path
    tf_filter_file: Path | None
    node_tf_scope: Path | None
    node_tg_scope: Path | None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "InputPaths":
        data = config.get("data", {})
        base_dir = Path(str(config.get("base_dir", "."))).expanduser().resolve()
        lineage = str(config.get("lineage", "unpaired")).strip() or "unpaired"

        def _resolve(key: str, *defaults: str) -> Path:
            candidates = [str(data[key])] if key in data else [item for item in defaults if item]
            if int(config.get("config_version", 1)) >= 2 and key not in data:
                candidates = candidates[:1]
            fallback: Path | None = None
            for candidate in candidates:
                target = (base_dir / candidate).resolve()
                if fallback is None:
                    fallback = target
                if target.exists():
                    return target
                copy = resolve_copy_suffix_path(target)
                if copy is not None:
                    return copy.resolve()
            return fallback if fallback is not None else base_dir.resolve()

        def _resolve_optional(key: str) -> Path | None:
            raw = str(data.get(key, "")).strip()
            if not raw:
                return None
            target = (base_dir / raw).resolve()
            if target.exists():
                return target
            copy = resolve_copy_suffix_path(target)
            if copy is not None:
                return copy.resolve()
            return None

        tf_filter_value = data.get("tf_filter_file")
        tf_filter_path = _resolve("tf_filter_file", str(tf_filter_value)) if tf_filter_value else None
        return cls(
            base_dir=base_dir,
            tf_list=_resolve("tf_list", f"data/raw/{lineage}/{lineage}_TF.tsv", "data/raw/unpaired/unpaired_TF.tsv"),
            tg_list=_resolve("tg_list", f"data/raw/{lineage}/{lineage}_TG.tsv", "data/raw/unpaired/unpaired_TG.tsv"),
            rna_h5ad=_resolve("rna_h5ad", f"data/raw/{lineage}/{lineage}_rna.h5ad", "data/raw/unpaired/unpaired_rna.h5ad"),
            annotation=_resolve(
                "annotation",
                f"data/raw/{lineage}/{lineage}_cell_annotation.csv",
                "data/raw/unpaired/unpaired_cell_annotation.csv",
            ),
            tree=_resolve("tree", f"data/raw/{lineage}/{lineage}_tree.csv", "data/raw/unpaired/unpaired_tree.csv"),
            tf_tg_prior=_resolve(
                "tf_tg_prior",
                f"data/raw/{lineage}/{lineage}_tf_tg_prior_all_stages.tsv",
                "data/raw/unpaired/unpaired_tf_tg_prior_all_stages.tsv",
            ),
            tf_filter_file=tf_filter_path,
            node_tf_scope=_resolve_optional("node_tf_scope"),
            node_tg_scope=_resolve_optional("node_tg_scope"),
        )


def summarize_input_paths(paths: InputPaths) -> dict[str, str]:
    payload = {
        "base_dir": str(paths.base_dir),
        "tf_list": str(paths.tf_list),
        "tg_list": str(paths.tg_list),
        "rna_h5ad": str(paths.rna_h5ad),
        "annotation": str(paths.annotation),
        "tree": str(paths.tree),
        "tf_tg_prior": str(paths.tf_tg_prior),
    }
    if paths.tf_filter_file is not None:
        payload["tf_filter_file"] = str(paths.tf_filter_file)
    if paths.node_tf_scope is not None:
        payload["node_tf_scope"] = str(paths.node_tf_scope)
    if paths.node_tg_scope is not None:
        payload["node_tg_scope"] = str(paths.node_tg_scope)
    return payload


def has_explicit_tree_config(config: dict[str, Any]) -> bool:
    data = config.get("data", {})
    if not isinstance(data, dict):
        return False
    return bool(str(data.get("tree", "")).strip())


__all__ = [
    "InputPaths",
    "has_explicit_tree_config",
    "save_text_report",
    "strip_copy_suffix",
    "summarize_input_paths",
]
