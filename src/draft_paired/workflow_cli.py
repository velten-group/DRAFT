"""Prepare regulatory priors, construct stage graphs and train DRAFT models."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cli_options import add_run_options, apply_overrides, command_config
from .utils import emit_progress

_INPUT_OPTIONS = {
    "rna_h5ad": ("--rna", "RNA H5AD path"),
    "atac_h5ad": ("--atac", "ATAC H5AD path"),
    "tree": ("--tree", "Lineage tree table"),
    "tf_list": ("--tf-list", "TF gene table"),
    "tg_list": ("--tg-list", "TG gene table"),
    "annotation": ("--rna-labels", "RNA cell-label table"),
    "atac_annotation": ("--atac-labels", "ATAC cell-label table"),
    "tf_re_prior": ("--tf-re", "TF-RE relation table"),
    "re_tg_prior": ("--re-tg", "RE-TG relation table"),
    "re_list": ("--re-list", "Paired RE list; otherwise derived from ATAC"),
}
_RESOURCE_OPTIONS = {
    "gene_annotation": "--gene-annotation",
    "motif_instances": "--motif-instances",
    "motif_to_tf": "--motif-to-tf",
    "motif_bigbed": "--motif-bigbed",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="draft")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init-config", help="Configure automatic priors from RNA/ATAC and gene lists"
    )
    init.add_argument("--mode", choices=["paired", "unpaired"], required=True)
    init.add_argument("--genome", choices=["hg38", "mm10", "dm6"], required=True)
    init.add_argument("--lineage", default="lineage1")
    init.add_argument("--output")
    init.add_argument("--force", action="store_true")
    for name, help_text in [
        ("prepare-prior", "Prepare TF-RE, RE-TG and stage-specific TF-TG priors"),
        ("preprocess", "Prepare priors and stage graphs for the configured input mode"),
    ]:
        command = commands.add_parser(name, help=help_text)
        if name in {"preprocess", "prepare-prior"}:
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--config", help="Existing preprocessing YAML")
            source.add_argument(
                "--input-dir",
                help="Discover input files and generate the preprocessing config",
            )
            command.add_argument(
                "--lineage", help="Lineage name; defaults to the input-directory name"
            )
            command.add_argument(
                "--genome",
                choices=["hg38", "mm10", "dm6"],
                help="Required for motif-generated priors",
            )
            command.add_argument(
                "--prior-source",
                choices=["auto", "provided", "motif"],
                help="Use supplied TF-RE/RE-TG relations or generate motif-based priors",
            )
            command.add_argument(
                "--min-accessible-fraction",
                type=float,
                help="Override the accessibility rule when preparing priors",
            )
            for field, (flag, description) in _INPUT_OPTIONS.items():
                command.add_argument(
                    flag, dest=field, help=description
                )
            for field, flag in _RESOURCE_OPTIONS.items():
                command.add_argument(
                    flag,
                    dest=field,
                    help="Local motif/annotation resource",
                )
        else:
            command.add_argument("--config", required=True)
        command.add_argument("--output", required=True, help="Directory for prepared files")
        command.add_argument(
            "--mode", choices=["paired", "unpaired"], help="Overrides config input_mode"
        )
        command.add_argument("--force", action="store_true", help="Replace existing prepared files")
        add_run_options(command, name, output=False)
    train = commands.add_parser(
        "train", help="Train from prepared paired or unpaired stage graphs"
    )
    train.add_argument(
        "--config",
        required=True,
        help="Training YAML written by preprocess, or a config with data.graphs",
    )
    add_run_options(train, "train")
    return parser


def _input_config(args, parser):
    from .directory_inputs import prepare_directory_config
    from .utils import load_yaml_config

    files = {field: getattr(args, field) for field in _INPUT_OPTIONS}
    resources = {field: getattr(args, field) for field in _RESOURCE_OPTIONS}
    if args.input_dir:
        if args.mode is None:
            parser.error("--input-dir requires --mode paired or unpaired.")
        path = prepare_directory_config(
            args.input_dir,
            args.output,
            mode=args.mode,
            lineage=args.lineage,
            genome=args.genome,
            prior_source=args.prior_source or "auto",
            min_accessible_fraction=args.min_accessible_fraction,
            files=files,
            resources=resources,
            force=args.force,
        )
        if args.set_values:
            import yaml

            config = apply_overrides(load_yaml_config(path), args.set_values)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return str(path), args.mode
    config = apply_overrides(load_yaml_config(args.config), args.set_values)
    mode = args.mode or config.get("input_mode")
    if mode not in {"paired", "unpaired"}:
        parser.error("Pass --mode paired/unpaired or use a config with input_mode.")
    options = [
        *files.values(),
        *resources.values(),
        args.lineage,
        args.genome,
        args.prior_source,
        args.min_accessible_fraction,
        *args.set_values,
    ]
    if not any(value is not None for value in options):
        return args.config, mode
    for key, value in files.items():
        if value is not None:
            config.setdefault("data", {})[key] = str(Path(value).expanduser().resolve())
    for key, value in resources.items():
        if value is not None:
            config.setdefault("resources", {})[key] = str(
                Path(value).expanduser().resolve()
            )
    if args.lineage:
        config["lineage"] = args.lineage
    if args.genome:
        config.setdefault("data", {})["genome_assembly"] = args.genome
    if args.prior_source and args.prior_source != "auto":
        config.setdefault("prior_preparation", {}).update(
            enabled=args.prior_source == "motif" or mode == "unpaired",
            method="auto_motif"
            if args.prior_source == "motif"
            else "accessible_peak_path",
        )
    if args.min_accessible_fraction is not None:
        config.setdefault("prior_preparation", {})["min_accessible_fraction"] = (
            args.min_accessible_fraction
        )
    return command_config(args, config=config), mode


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init-config":
        from .config_init import init_auto_config

        result = init_auto_config(
            mode=args.mode,
            genome=args.genome,
            lineage=args.lineage,
            output=args.output,
            force=args.force,
        )
        emit_progress("DONE", "init-config", result["output_path"])
        return 0
    if args.command == "prepare-prior":
        from .prior_preparation import prepare_prior

        source, mode = _input_config(args, parser)
        result = prepare_prior(source, args.output, mode=mode, force=args.force)
        emit_progress("DONE", "prepare-prior", result["config_path"])
        return 0
    if args.command == "preprocess":
        from .workflow import preprocess_lineage

        source, mode = _input_config(args, parser)
        result = preprocess_lineage(source, mode, args.output, force=args.force)
        emit_progress("DONE", "preprocess", result["graph_path"])
        emit_progress(
            "NEXT", "train", f'draft train --config "{result["config_path"]}"'
        )
        return 0
    from .workflow import train_graph_model

    args.config = command_config(args)
    emit_progress("START", "train", args.config)
    report = train_graph_model(args.config)
    emit_progress("DONE", "train", report["training"]["checkpoint_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
