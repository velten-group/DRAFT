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
        lineage = str(config.get("lineage", "")).strip() or "lineage1"
        folder = str(config.get("lineage_data_dir") or f"data/{lineage}")

        def _resolve(key: str, name: str) -> Path:
            """Path from the config, or the file `name` in the lineage folder."""
            target = (base_dir / str(data[key] if key in data else f"{folder}/{name}")).resolve()
            if not target.exists():
                copy = resolve_copy_suffix_path(target)
                if copy is not None:
                    return copy.resolve()
            return target

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
        tf_filter_path = _resolve("tf_filter_file", "") if tf_filter_value else None
        return cls(
            base_dir=base_dir,
            tf_list=_resolve("tf_list", "TF.tsv"),
            tg_list=_resolve("tg_list", "TG.tsv"),
            rna_h5ad=_resolve("rna_h5ad", "rna.h5ad"),
            annotation=_resolve("annotation", "rna_cells.tsv"),
            tree=_resolve("tree", "tree.tsv"),
            tf_tg_prior=_resolve("tf_tg_prior", "tf_tg_prior.tsv"),
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
