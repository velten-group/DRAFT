"""Named parameter sets for training and downstream analyses."""

from __future__ import annotations

from typing import Any

from .utils import read_packaged_text


def profile_config(
    text: str, profile: str, family: str, lineage: str | None = None
) -> str:
    import yaml

    catalog = yaml.safe_load(
        read_packaged_text("draft_paired", "templates/method_profiles.yaml")
    )["profiles"]
    if profile not in catalog:
        raise ValueError(
            f"Unknown config profile {profile!r}; choose from {', '.join(catalog)}."
        )
    config = yaml.safe_load(text)
    if lineage is not None:
        config["lineage"] = lineage

    def merge(target: dict[str, Any], overlay: dict[str, Any]) -> None:
        for key, value in overlay.items():
            if isinstance(value, dict):
                merge(target.setdefault(key, {}), value)
            else:
                target[key] = value

    merge(config, catalog[profile][family])
    config["method_profile"] = {"name": profile}
    return (
        "# Generated parameter profile. Edit input paths and lineage topology before running.\n"
        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    )
