"""Resolve lineage IDs, directions and input specifications for two-series CCC."""

from __future__ import annotations

import copy
import re
from typing import Any


def normalize_ccc_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("ccc", {})
    if "lineages" not in section:
        return config
    result = copy.deepcopy(config)
    section = result["ccc"]
    lineages = section.pop("lineages")
    if not isinstance(lineages, dict) or len(lineages) != 2:
        raise ValueError(
            "ccc.lineages must contain exactly two named lineage specifications."
        )
    names = section.get("lineage_order", list(lineages))
    if not isinstance(names, list) or len(names) != 2 or set(names) != set(lineages):
        raise ValueError(
            "ccc.lineage_order must list both lineage IDs once, in alignment-axis order."
        )
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", n) or n == "hub" for n in names):
        raise ValueError(
            "CCC lineage IDs must start with a letter and use letters, digits, _ or -; hub is reserved."
        )
    if "soma" in section or "germ" in section:
        raise ValueError(
            "Use ccc.lineages or the soma/germ key layout, not both."
        )
    aliases = dict(zip(("soma", "germ"), names))
    role = {name: key for key, name in aliases.items()}
    for key, name in aliases.items():
        spec = lineages[name]
        if not isinstance(spec, dict):
            raise ValueError(f"ccc.lineages.{name} must be a mapping.")
        if spec.get("input_mode", "paired") not in {"paired", "unpaired"}:
            raise ValueError(
                f"Unknown input_mode for {name}; choose paired or unpaired."
            )
        if "early_stage_count" not in spec:
            raise ValueError(
                f"Define ccc.lineages.{name}.early_stage_count for your biological early window."
            )
        section[key] = spec
        for field, target in (
            ("network_dir", "network_dir"),
            ("tf_list", "tf_annot"),
            ("tg_list", "tg_annot"),
        ):
            if field in spec:
                section[f"{target}_{key}"] = spec[field]
        if not section.get("alignment_path") and not (spec.get("config") or spec.get("network_dir")):
            raise ValueError(
                f"ccc.lineages.{name} requires config or network_dir when computing an alignment."
            )
    directions = {
        f"{names[0]}_to_{names[1]}": ("soma", "germ"),
        f"{names[1]}_to_{names[0]}": ("germ", "soma"),
    }
    if len(directions) != 2:
        raise ValueError(
            "Lineage IDs produce ambiguous direction names; choose distinct simple IDs."
        )
    if section.get("hub_group"):
        directions.update(
            {f"hub_to_{name}": ("hub", key) for key, name in aliases.items()}
        )
    requested = section.get("directions", list(directions))
    normalized = []
    for item in requested:
        if isinstance(item, dict):
            sender, receiver = item.get("sender"), item.get("receiver")
            if (
                sender not in {*role, "hub"}
                or receiver not in role
                or sender == receiver
            ):
                raise ValueError(f"Invalid CCC sender/receiver: {item!r}.")
            item = f"{sender}_to_{receiver}"
        if item not in directions:
            raise ValueError(
                f"Unknown CCC direction {item!r}; choose from {list(directions)}."
            )
        normalized.append(item)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("CCC directions must be nonempty and distinct.")
    section["directions"] = normalized
    section["_lineage_aliases"] = aliases
    section["_direction_specs"] = directions
    section.setdefault("warp_dtw_tsv", f"time_warp_{names[0]}_to_{names[1]}.tsv")
    return result
