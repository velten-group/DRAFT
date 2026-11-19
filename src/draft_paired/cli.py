"""Command line interface for draft_paired workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .ccc_inputs import add_ccc_options, ccc_config_from_args
from .cli_options import add_run_options, apply_overrides, command_config
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
    if command == "single-tf":
        return result.get("output_dir")
    if command == "edge-cluster":
        return result.get("output_dir")
    if command == "topic":
        return result.get("output_dir")
    if command == "continuous-activity":
        return result.get("figure_path") or result.get("table_path")
    if command == "ccc":
        return result.get("output_dir")
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


def build_parser(prog: str = "draft_paired") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Prepare stage graphs and a shared training config")
    preprocess.add_argument("--config", required=True, help="Paired input YAML")
    preprocess.add_argument("--output", required=True, help="Directory for graphs.pt and train.yaml")
    preprocess.add_argument("--force", action="store_true", help="Replace existing prepared outputs")

    train_parser = subparsers.add_parser("train", help="Run lineage model training")
    train_parser.add_argument(
        "--config", required=True, help="Path to a YAML config file"
    )

    simulation_parser = subparsers.add_parser(
        "simulation", help="Run the bundled simulated RNA/ATAC example"
    )
    simulation_parser.add_argument(
        "--config", help="Path to a simulation YAML config file"
    )
    simulation_parser.add_argument(
        "--prior-grn",
        help="Candidate TF-TG CSV/TSV; generate RNA/ATAC with scMultiSim before training",
    )
    simulation_parser.add_argument(
        "--output", help="Simulation training/output directory"
    )
    simulation_parser.add_argument(
        "--rscript", help="Path to Rscript for prior-GRN generation"
    )
    simulation_parser.add_argument(
        "--r-library", help="Optional R package library directory"
    )

    reproduce_parser = subparsers.add_parser(
        "reproduce", help="Export embeddings and summaries across stages"
    )
    reproduce_parser.add_argument(
        "--config", required=True, help="Path to a YAML config file"
    )

    network_parser = subparsers.add_parser(
        "network-export", help="Export stage-specific TF-TG networks and supporting regions"
    )
    network_parser.add_argument(
        "--config", required=True, help="Path to a YAML config file"
    )
    network_parser.add_argument("--dense", action="store_true", help="Also save interpolated network tables")

    single_tf_parser = subparsers.add_parser(
        "single-tf", help="Run single-TF activity and regulon analysis"
    )
    single_tf_parser.add_argument(
        "--config", required=True, help="Path to a YAML config file"
    )

    edge_parser = subparsers.add_parser(
        "edge-cluster", help="Run edge-cluster analysis from exported networks"
    )
    edge_parser.add_argument(
        "--config", required=True, help="Path to a YAML config file"
    )

    topic_parser = subparsers.add_parser(
        "topic", help="Run LDA topic analysis from exported networks"
    )
    topic_parser.add_argument(
        "--config", required=True, help="Path to a YAML config file"
    )

    continuous_parser = subparsers.add_parser(
        "continuous-activity",
        help="Plot interpolated TF activity from a trained model",
    )
    continuous_parser.add_argument(
        "--config", required=True, help="Path to a YAML config file"
    )

    ccc_parser = subparsers.add_parser(
        "ccc",
        help="Run DTW-aligned ligand-expression x receptor-centrality CCC",
    )
    ccc_parser.add_argument(
        "--config", help="Existing communication config"
    )
    add_ccc_options(ccc_parser)

    init_parser = subparsers.add_parser(
        "init-config",
        help="Write a packaged YAML template config to the current directory",
    )
    init_parser.add_argument(
        "--kind",
        choices=["lineage", "ccc", "simulation", "simulation-prior"],
        default="lineage",
        help="Which paired template to generate",
    )
    init_parser.add_argument(
        "--output",
        help="Output path for the generated config file",
    )
    init_parser.add_argument(
        "--profile",
        choices=["methods"],
        help="Named lineage parameter profile",
    )
    init_parser.add_argument(
        "--lineage", help="Lineage name for the generated lineage config"
    )
    init_parser.add_argument(
        "--example", choices=["fly_soma", "fly_germ", "fly_ccc"],
        help="Application presets for manuscript reproduction",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the unified DRAFT pipeline: train, reproduce embeddings, export networks, and any enabled downstream analyses",
    )
    pipeline_parser.add_argument(
        "--config", required=True, help="Path to a YAML config file"
    )

    for command, command_parser in (
        ("train", train_parser), ("reproduce", reproduce_parser),
        ("network-export", network_parser), ("single-tf", single_tf_parser),
        ("edge-cluster", edge_parser), ("topic", topic_parser),
        ("continuous-activity", continuous_parser), ("ccc", ccc_parser),
        ("pipeline", pipeline_parser),
    ):
        add_run_options(command_parser, command)
    add_run_options(preprocess, "preprocess", output=False)
    add_run_options(simulation_parser, "simulation", output=False)
    return parser


def main(argv: list[str] | None = None, prog: str = "draft_paired") -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    if args.command not in {"init-config", "simulation"}:
        try:
            config = ccc_config_from_args(args) if args.command == "ccc" else None
            args.config = command_config(args, config=config)
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))

    if args.command == "preprocess":
        from .workflow import preprocess_lineage

        emit_progress("START", "preprocess", args.config)
        result = preprocess_lineage(args.config, "paired", args.output, force=args.force)
        emit_progress("DONE", "preprocess", result["graph_path"])
        emit_progress("NEXT", "train", f"draft train --config \"{result['config_path']}\"")
        return 0

    if args.command == "simulation":
        from .simulation import run_simulation
        from .utils import load_yaml_config

        if not args.config and not args.prior_grn:
            parser.error("simulation requires --config or --prior-grn")
        config = (
            load_yaml_config(args.config)
            if args.config
            else {
                "simulation": {
                    "data_seeds": [101],
                    "model_seeds": [0],
                    "output_dir": args.output or "result/simulation_from_prior",
                },
                "generator": {
                    "output_dir": str(
                        Path(args.output or "result/simulation_from_prior")
                        / "generated_data"
                    )
                },
            }
        )
        if args.prior_grn:
            config.setdefault("simulation", {})["prior_grn"] = args.prior_grn
        config = apply_overrides(config, args.set_values)
        if args.output:
            config.setdefault("simulation", {})["output_dir"] = args.output
        if args.rscript:
            config.setdefault("generator", {})["rscript"] = args.rscript
        if args.r_library:
            config.setdefault("generator", {})["r_library"] = args.r_library

        emit_progress("START", "simulation", args.config)
        result = run_simulation(config)
        emit_progress("DONE", "simulation", _result_location(args.command, result))
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
        emit_progress(
            "DONE", "continuous-activity", _result_location(args.command, result)
        )
        return 0

    if args.command == "ccc":
        from .ccc_workflow import run_draft_ccc_workflow

        emit_progress("START", "ccc", args.config)
        result = run_draft_ccc_workflow(args.config)
        emit_progress("DONE", "ccc", _result_location(args.command, result))
        return 0

    if args.command == "init-config":
        from .config_init import init_config

        emit_progress("START", "init-config", args.kind)
        result = init_config(
            kind=args.kind,
            output=args.output,
            force=bool(args.force),
            profile=args.profile,
            lineage=args.lineage,
            example=args.example,
        )
        emit_progress("DONE", "init-config", _result_location(args.command, result))
        return 0

    if args.command == "pipeline":
        from .pipeline import run_draft_pipeline

        emit_progress("START", "pipeline", args.config)
        result = run_draft_pipeline(args.config)
        emit_progress("DONE", "pipeline", _result_location(args.command, result))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
