"""Command-line choices applied to generated run configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .utils import emit_progress, load_yaml_config, resolve_config_path

SECTIONS = {
    "single-tf": "single_tf",
    "edge-cluster": "edge_cluster",
    "topic": "topic",
    "continuous-activity": "continuous_activity",
    "network-export": "network",
    "reproduce": "reproduce",
    "ccc": "ccc",
}


def add_run_options(parser, command, *, output=True):
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key; repeat for more settings",
    )
    if output:
        parser.add_argument("--output", help="Output directory for this command")
    if command in {"single-tf", "continuous-activity"}:
        parser.add_argument("--tf", nargs="+", help="TF identifiers or gene symbols")
    if command == "single-tf":
        parser.add_argument("--stage", help="Tree node name or zero-based stage index")
        parser.add_argument(
            "--consensus-stages",
            nargs="+",
            help="Stages to combine into a consensus regulon",
        )
        parser.add_argument(
            "--edge-cluster-file",
            help="Edge-cluster CSV for restricting consensus targets",
        )
        parser.add_argument(
            "--edge-cluster", type=int, help="Cluster ID to use for consensus targets"
        )
    if command == "topic":
        parser.add_argument("--organism", help="Organism for optional topic enrichment")
        parser.add_argument(
            "--gene-sets", nargs="+", help="Enrichment gene-set libraries"
        )


def apply_overrides(config, entries):
    result = copy.deepcopy(config)
    for entry in entries or []:
        key, separator, raw = entry.partition("=")
        parts = [part.strip() for part in key.split(".")]
        if not separator or any(not part.strip() for part in parts):
            raise ValueError(f"Expected --set KEY=VALUE, got {entry!r}.")
        node = result
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            if not isinstance(node[part], dict):
                raise TypeError(f"Cannot set {key}: {part} is not a mapping.")
            node = node[part]
        try:
            node[parts[-1]] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML value for {key}.") from exc
    return result


def relocate_run(config, output):
    """Keep source inputs and move the run's outputs and downstream references."""
    root = str(Path(output).expanduser().resolve())
    previous = config.get("result_root", f"result/{config.get('lineage', 'lineage')}")
    prefixes = [
        str(previous).replace("\\", "/").rstrip("/"),
        str(resolve_config_path(config, previous)).replace("\\", "/").rstrip("/"),
    ]
    inputs = {
        key: copy.deepcopy(config[key])
        for key in ("data", "resources")
        if key in config
    }

    def move(value):
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            for prefix in prefixes:
                if normalized == prefix or normalized.startswith(prefix + "/"):
                    return root + normalized[len(prefix) :]
        return value

    relocated = move(config)
    relocated.update(inputs)
    relocated["result_root"] = root
    relocated["output_dir"] = str(Path(root) / "train")
    relocated.setdefault("checkpoints", {})["dir"] = str(Path(root) / "model")
    return relocated


def command_config(args, *, config=None):
    """Save effective command settings alongside outputs, keeping source YAML intact."""
    command = args.command
    entries = getattr(args, "set_values", [])
    selections = [
        getattr(args, key, None)
        for key in (
            "tf",
            "stage",
            "consensus_stages",
            "edge_cluster_file",
            "edge_cluster",
            "organism",
            "gene_sets",
        )
    ]
    if config is None and not (
        entries
        or getattr(args, "output", None)
        or any(value is not None for value in selections)
    ):
        return args.config
    config = apply_overrides(
        config if config is not None else load_yaml_config(args.config), entries
    )
    config["base_dir"] = str(Path(config.get("base_dir", ".")).expanduser().resolve())
    section = SECTIONS.get(command)
    values = config.setdefault(section, {}) if section else config
    if getattr(args, "tf", None) is not None:
        values["tf_symbols" if command == "single-tf" else "tf_list"] = args.tf
    if getattr(args, "stage", None) is not None:
        values["time_for_top"] = args.stage
    if command == "single-tf":
        if args.consensus_stages is not None:
            consensus = values.setdefault("consensus", {})
            consensus.update(enabled=True, time_points=args.consensus_stages)
            consensus.pop("time_window", None)
        for argument, field in (
            ("edge_cluster_file", "edge_cluster_file"),
            ("edge_cluster", "edge_cluster"),
        ):
            value = getattr(args, argument, None)
            if value is not None:
                values.setdefault("consensus", {}).update(
                    enabled=True, **{field: value}
                )
    if command == "topic":
        for argument, field in (
            ("organism", "enrich_organism"),
            ("gene_sets", "enrich_db"),
        ):
            value = getattr(args, argument, None)
            if value is not None:
                values.update(do_enrich=True, **{field: value})
    if command in {"train", "pipeline"}:
        if args.output:
            config = relocate_run(config, args.output)
        destination = resolve_config_path(
            config,
            config.get("checkpoints", {}).get(
                "dir", f"result/{config.get('lineage', 'lineage')}/model"
            ),
        ).parent
    elif command in {"preprocess", "prepare-prior"}:
        destination = Path(args.output).expanduser().resolve()
    else:
        if args.output:
            previous = values.get("output_dir")
            values["output_dir"] = str(Path(args.output).expanduser().resolve())
            if command == "network-export":
                for key in ("edge_cluster", "topic"):
                    if config.get(key, {}).get("network_dir") == previous:
                        config[key]["network_dir"] = values["output_dir"]
        destination = resolve_config_path(
            config,
            values.get(
                "output_dir", f"result/{config.get('lineage', 'lineage')}/{section}"
            ),
        )
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (
        "inputs.yaml"
        if command in {"preprocess", "prepare-prior"}
        else "run_config.yaml"
    )
    source = getattr(args, "config", None)
    if source and target == Path(source).expanduser().resolve():
        target = destination / "command_config.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    emit_progress("SETTINGS", command, str(target))
    return str(target)
