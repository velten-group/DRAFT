"""Template config generation for draft_unpaired."""

from __future__ import annotations

from pathlib import Path

from draft_paired.utils import (
    read_packaged_text,
    write_config_text,
    write_packaged_text,
)


def init_config(
    *,
    kind: str = "lineage",
    output: str | Path | None = None,
    force: bool = False,
    profile: str | None = None,
    lineage: str | None = None,
    example: str | None = None,
) -> dict[str, str]:
    template_names = {
        "lineage": "draft_unpaired_template.yaml",
        "motif": "draft_motif_template.yaml",
        "staged-prior": "draft_staged_prior_template.yaml",
    }
    if kind not in template_names:
        raise ValueError(
            "Choose lineage, motif or staged-prior for the unpaired input layout."
        )
    if example is not None and kind != "lineage":
        raise ValueError(
            "Application examples already define their input layout; omit --kind."
        )
    if example is not None and example not in {"mouse_development", "mouse_motifs"}:
        raise ValueError(
            "Choose mouse_development or mouse_motifs for unpaired application settings."
        )
    package = "draft_paired" if example else "draft_unpaired"
    template = (
        f"templates/draft_example_{example}.yaml"
        if example
        else f"templates/{template_names[kind]}"
    )
    target = (
        Path(output)
        if output is not None
        else Path(
            f"configs/examples/{example}.yaml"
            if example
            else f"configs/{template_names[kind]}"
        )
    )
    if profile is not None or lineage is not None:
        from draft_paired.config_profiles import profile_config

        content = read_packaged_text(package, template)
        if profile is not None:
            content = profile_config(content, profile, "unpaired", lineage)
        else:
            import yaml

            config = yaml.safe_load(content)
            config["lineage"] = lineage
            content = yaml.safe_dump(config, sort_keys=False)
        written = write_config_text(content, target, overwrite=force)
    else:
        written = write_packaged_text(
            package,
            template,
            target,
            overwrite=force,
        )
    return {
        "kind": "unpaired",
        "output_path": str(written),
    }


__all__ = ["init_config"]
