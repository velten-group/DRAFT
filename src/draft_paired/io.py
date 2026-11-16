"""Input path helpers for draft_paired workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class InputPaths:
    base_dir: Path
    tf_list: Path
    tg_list: Path
    re_list: Path
    rna_h5ad: Path
    atac_h5ad: Path
    re_tg_prior: Path
    tf_re_prior: Path
    annotation: Path
    tree: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "InputPaths":
        data = config.get("data", {})
        base_dir = Path(config.get("base_dir", ".")).expanduser().resolve()
        lineage_value = str(config.get("lineage", "")).strip().lower()
        if int(config.get("config_version", 1)) >= 2:
            lineage_prefix = lineage_value or "lineage1"
        elif lineage_value in {"soma", "somatic"}:
            lineage_prefix = "soma"
        elif lineage_value in {"germ", "germline"}:
            lineage_prefix = "germ"
        else:
            lineage_prefix = None

        def _resolve(key: str, *defaults: str) -> Path:
            candidates = [str(data[key])] if key in data else [item for item in defaults if item]
            if int(config.get("config_version", 1)) >= 2 and key not in data:
                candidates = candidates[:1]
            resolved_fallback: Path | None = None
            for candidate in candidates:
                target = (base_dir / candidate).resolve()
                if resolved_fallback is None:
                    resolved_fallback = target
                if target.exists():
                    return target
                copy = resolve_copy_suffix_path(target)
                if copy is not None:
                    return copy.resolve()
            return resolved_fallback if resolved_fallback is not None else base_dir.resolve()

        return cls(
            base_dir=base_dir,
            tf_list=_resolve("tf_list", "data/raw/TF.tsv", "TF.tsv"),
            tg_list=_resolve("tg_list", "data/raw/TG.tsv", "TG.tsv"),
            re_list=_resolve("re_list", "data/raw/RE.txt", "RE.txt"),
            rna_h5ad=_resolve(
                "rna_h5ad",
                f"data/raw/{lineage_prefix}/{lineage_prefix}_rna.h5ad" if lineage_prefix else "",
                "data/raw/A_rna_focus.h5ad",
                "A_rna_focus.h5ad",
            ),
            atac_h5ad=_resolve(
                "atac_h5ad",
                f"data/raw/{lineage_prefix}/{lineage_prefix}_atac.h5ad" if lineage_prefix else "",
                "data/raw/B_atac_focus.h5ad",
                "B_atac_focus.h5ad",
            ),
            re_tg_prior=_resolve("re_tg_prior", "data/raw/RE_TG_100kb.tsv", "RE_TG_100kb.tsv"),
            tf_re_prior=_resolve("tf_re_prior", "data/raw/TF_RE_top3pct_edges.tsv", "TF_RE_top3pct_edges.tsv"),
            annotation=_resolve(
                "annotation",
                f"data/raw/{lineage_prefix}/{lineage_prefix}_cell_annotation.csv" if lineage_prefix else "",
                "data/raw/cell_annotation.csv",
                "cell_annotation.csv",
            ),
            tree=_resolve(
                "tree",
                f"data/raw/{lineage_prefix}/{lineage_prefix}_tree.csv" if lineage_prefix else "",
                "data/raw/tree.csv",
                "tree.csv",
            ),
        )


def strip_copy_suffix(value: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", str(value).strip())


def resolve_copy_suffix_path(path: str | Path) -> Path | None:
    target = Path(path).expanduser()
    parent = target.parent
    if not parent.exists():
        return None

    suffix = target.suffix
    stem = strip_copy_suffix(target.stem)
    pattern = re.compile(rf"^{re.escape(stem)}(?: ?\(\d+\))?{re.escape(suffix)}$")

    for candidate in sorted(parent.iterdir(), key=lambda item: item.name):
        if candidate.is_file() and pattern.match(candidate.name):
            return candidate.resolve()
    return None


def save_text_report(path: str | Path, content: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def summarize_input_paths(paths: InputPaths) -> dict[str, str]:
    return {
        "base_dir": str(paths.base_dir),
        "tf_list": str(paths.tf_list),
        "tg_list": str(paths.tg_list),
        "re_list": str(paths.re_list),
        "rna_h5ad": str(paths.rna_h5ad),
        "atac_h5ad": str(paths.atac_h5ad),
        "re_tg_prior": str(paths.re_tg_prior),
        "tf_re_prior": str(paths.tf_re_prior),
        "annotation": str(paths.annotation),
        "tree": str(paths.tree),
    }


def has_explicit_tree_config(config: dict[str, Any]) -> bool:
    data = config.get("data", {})
    if not isinstance(data, dict):
        return False
    value = data.get("tree")
    return bool(str(value).strip()) if value is not None else False
