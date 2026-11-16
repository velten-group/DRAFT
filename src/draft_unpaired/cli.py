"""Command line interface for draft_unpaired workflows."""

from __future__ import annotations

import argparse
from typing import Any

from draft_paired.cli_options import add_run_options, command_config

from .utils import emit_progress


def _result_location(command: str, result: dict[str, Any]) -> str | None:
    if command == "train":
        training = result.get("training", {})
        if isinstance(training, dict):
            return training.get("checkpoint_path") or training.get("artifact_path")
    if command == "reproduce":
        return result.get("output_dir")
    if command == "network-export":
        strict = result.get("strict", {})
        if isinstance(strict, dict):
            return strict.get("output_dir")
    if command in {"single-tf", "edge-cluster", "topic"}:
        return result.get("output_dir")
    if command == "continuous-activity":
        return result.get("figure_path") or result.get("table_path")
    if command == "pipeline":
        network = result.get("network", {})
        if isinstance(network, dict):
            strict = network.get("strict", {})
            if isinstance(strict, dict):
                return strict.get("output_dir")
        return result.get("output_dir")
    if command == "init-config":
        return result.get("output_path")
    return result.get("output_dir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="draft_unpaired")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Prepare stage priors when enabled, stage graphs and a shared training config")
    preprocess.add_argument("--config", required=True, help="Unpaired input YAML")
    preprocess.add_argument("--output", required=True, help="Directory for graphs.pt and train.yaml")
    preprocess.add_argument("--force", action="store_true", help="Replace existing prepared outputs")

    for name, help_text in [
        ("train", "Run unpaired lineage model training"),
        ("reproduce", "Export embeddings and smoothness summaries"),
        ("network-export", "Export stage-specific TF-TG networks"),
        ("single-tf", "Run single-TF activity and regulon analysis"),
        ("edge-cluster", "Run edge-cluster analysis from exported networks"),
        ("topic", "Run topic analysis from exported networks"),
        ("continuous-activity", "Plot interpolated TF activity from a trained model"),
        ("pipeline", "Run the unified unpaired pipeline"),
    ]:
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--config", required=True, help="Path to a YAML config file")
        add_run_options(subparser, name)
        if name == "network-export":
            subparser.add_argument("--dense", action="store_true", help="Also save interpolated network tables")

    prior_parser = subparsers.add_parser("prepare-prior", help="Build staged TF-TG candidates from annotated ATAC and TF-RE/RE-TG relations")
    prior_parser.add_argument("--config", required=True, help="Unpaired YAML with optional ATAC preparation inputs")
    prior_parser.add_argument("--output", required=True, help="Directory for the prior, scopes, report and training config")
    prior_parser.add_argument("--force", action="store_true", help="Overwrite previously prepared output files")

    init_parser = subparsers.add_parser(
        "init-config",
        help="Write the packaged unpaired YAML template to the current directory",
    )
    init_parser.add_argument(
        "--kind", choices=["lineage", "motif", "staged-prior"], default="lineage",
        help="RNA/ATAC with relation tables, supplied called peaks (scMTNI), or an existing stage prior",
    )
    init_parser.add_argument(
        "--output",
        help="Output path for the generated config file",
    )
    init_parser.add_argument("--profile", choices=["methods"], help="Named unpaired parameter profile")
    init_parser.add_argument("--lineage", help="User-defined lineage identifier")
    init_parser.add_argument("--example", choices=["mouse_development", "mouse_motifs"], help="Application presets for manuscript reproduction")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )

    add_run_options(preprocess, "preprocess", output=False)
    add_run_options(prior_parser, "prepare-prior", output=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "init-config":
        try:
            args.config = command_config(args)
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))

    if args.command == "preprocess":
        from draft_paired.workflow import preprocess_lineage

        emit_progress("START", "preprocess", args.config)
        result = preprocess_lineage(args.config, "unpaired", args.output, force=args.force)
        emit_progress("DONE", "preprocess", result["graph_path"])
        emit_progress("NEXT", "train", f"draft train --config \"{result['config_path']}\"")
        return 0

    if args.command == "prepare-prior":
        from .prior import prepare_stage_prior

        emit_progress("START", "prepare-prior", args.config)
        result = prepare_stage_prior(args.config, args.output, force=args.force)
        emit_progress("DONE", "prepare-prior", result["config_path"])
        return 0

    if args.command == "train":
        from .train import train_lineage_model

        emit_progress("START", "train", args.config)
        result = train_lineage_model(args.config)
        emit_progress("DONE", "train", _result_location(args.command, result))
        return 0

    if args.command == "reproduce":
        from .reproduce import reproduce_lineage_model

        emit_progress("START", "reproduce", args.config)
        result = reproduce_lineage_model(args.config)
        emit_progress("DONE", "reproduce", _result_location(args.command, result))
        return 0

    if args.command == "network-export":
        from .analysis.network import export_lineage_networks

        emit_progress("START", "network-export", args.config)
        result = export_lineage_networks(args.config, dense=args.dense)
        emit_progress("DONE", "network-export", _result_location(args.command, result))
        return 0

    if args.command == "single-tf":
        from .analysis.single_tf import run_single_tf_analysis

        emit_progress("START", "single-tf", args.config)
        result = run_single_tf_analysis(args.config)
        emit_progress("DONE", "single-tf", _result_location(args.command, result))
        return 0

    if args.command == "edge-cluster":
        from .analysis.edge_cluster import run_edge_cluster

        emit_progress("START", "edge-cluster", args.config)
        result = run_edge_cluster(args.config)
        emit_progress("DONE", "edge-cluster", _result_location(args.command, result))
        return 0

    if args.command == "topic":
        from .analysis.topic import run_topic_analysis

        emit_progress("START", "topic", args.config)
        result = run_topic_analysis(args.config)
        emit_progress("DONE", "topic", _result_location(args.command, result))
        return 0

    if args.command == "continuous-activity":
        from .analysis.continuous_activity import run_continuous_activity

        emit_progress("START", "continuous-activity", args.config)
        result = run_continuous_activity(args.config)
        emit_progress("DONE", "continuous-activity", _result_location(args.command, result))
        return 0

    if args.command == "pipeline":
        from .pipeline import run_draft_pipeline

        emit_progress("START", "pipeline", args.config)
        result = run_draft_pipeline(args.config)
        emit_progress("DONE", "pipeline", _result_location(args.command, result))
        return 0

    if args.command == "init-config":
        from .config_init import init_config

        emit_progress("START", "init-config", "unpaired")
        result = init_config(kind=args.kind, output=args.output, force=bool(args.force), profile=args.profile,
                             lineage=args.lineage, example=args.example)
        emit_progress("DONE", "init-config", _result_location(args.command, result))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
