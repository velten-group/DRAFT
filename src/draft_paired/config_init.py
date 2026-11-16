"""Template config generation for draft_paired."""

from __future__ import annotations

from pathlib import Path

from .utils import read_packaged_text, write_config_text, write_packaged_text

DEFAULT_MIN_ACCESSIBLE_FRACTION = 0.05


def build_auto_config(*, mode: str, genome: str, lineage="lineage1"):
    """Return automatic-prior defaults without writing a config file."""
    import yaml

    from .prior_resources import RESOURCE_CATALOG

    if mode not in {"paired", "unpaired"} or genome not in RESOURCE_CATALOG:
        raise ValueError("Choose paired/unpaired and one of hg38, mm10, dm6.")
    template = (
        "draft_lineage_template.yaml"
        if mode == "paired"
        else "draft_unpaired_template.yaml"
    )
    config = yaml.safe_load(
        read_packaged_text(f"draft_{mode}", f"templates/{template}")
    )
    config.update(lineage=lineage, input_mode=mode, base_dir=".")
    config["data"] = {
        "rna_h5ad": "${lineage_data_dir}/rna.h5ad",
        "rna_layer": "lognorm",
        "atac_h5ad": "${lineage_data_dir}/atac.h5ad",
        "atac_layer": "counts",
        "tf_list": "${lineage_data_dir}/TF.tsv",
        "tg_list": "${lineage_data_dir}/TG.tsv",
        "annotation": None,
        "atac_annotation": None,
        "tree": "${lineage_data_dir}/tree.tsv",
        "genome_assembly": genome,
    }
    config["prior_preparation"] = {
        "enabled": True,
        "method": "auto_motif",
        "min_accessible_fraction": DEFAULT_MIN_ACCESSIBLE_FRACTION,
        "min_motif_score": 500,
        "promoter_flank": 5000,
    }
    config["resources"] = {"cache_dir": "data/resources/cache"}
    return config


def init_auto_config(
    *, mode: str, genome: str, lineage="lineage1", output=None, force=False
):
    """Configure shared automatic prior preparation without existing edge tables."""
    import yaml

    config = build_auto_config(mode=mode, genome=genome, lineage=lineage)
    target = output or f"configs/{lineage}.yaml"
    path = write_config_text(
        yaml.safe_dump(config, sort_keys=False), target, overwrite=force
    )
    return {"output_path": str(path), "input_mode": mode}


_TEMPLATES = {
    "lineage": (
        "templates/draft_lineage_template.yaml",
        "configs/draft_lineage_template.yaml",
    ),
    "ccc": ("templates/draft_ccc_template.yaml", "configs/draft_ccc_template.yaml"),
    "simulation": (
        "templates/draft_simulation_template.yaml",
        "configs/draft_simulation_template.yaml",
    ),
    "simulation-prior": (
        "templates/draft_simulation_prior_template.yaml",
        "configs/draft_simulation_prior_template.yaml",
    ),
}


def init_config(
    *,
    kind: str = "lineage",
    output: str | Path | None = None,
    force: bool = False,
    profile: str | None = None,
    lineage: str | None = None,
    example: str | None = None,
) -> dict[str, str]:
    if kind not in _TEMPLATES:
        allowed = ", ".join(sorted(_TEMPLATES))
        raise ValueError(
            f"Unsupported paired config kind: {kind}. Expected one of: {allowed}."
        )

    relative_path, default_name = _TEMPLATES[kind]
    if example is not None:
        if (
            (kind == "lineage" and example not in {"fly_soma", "fly_germ"})
            or (kind == "ccc" and example != "fly_ccc")
            or kind not in {"lineage", "ccc"}
        ):
            raise ValueError(
                "Use fly_soma/fly_germ for lineage configs or fly_ccc for CCC."
            )
        relative_path = f"templates/draft_example_{example}.yaml"
        default_name = f"configs/examples/{example}.yaml"
    target = Path(output) if output is not None else Path(default_name)
    if profile is not None or lineage is not None:
        if kind != "lineage":
            raise ValueError("--profile and --lineage apply only to lineage configs.")
        content = read_packaged_text("draft_paired", relative_path)
        if profile is not None:
            from .config_profiles import profile_config

            content = profile_config(content, profile, "paired", lineage)
        else:
            import yaml

            config = yaml.safe_load(content)
            config["lineage"] = lineage
            content = yaml.safe_dump(config, sort_keys=False)
        written = write_config_text(content, target, overwrite=force)
    else:
        written = write_packaged_text(
            "draft_paired", relative_path, target, overwrite=force
        )
    return {
        "kind": kind,
        "output_path": str(written),
    }


__all__ = ["init_config"]
