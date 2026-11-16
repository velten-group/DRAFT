"""Build a communication run from two saved lineage configurations."""

from __future__ import annotations

from pathlib import Path

from .utils import load_yaml_config


def add_ccc_options(parser):
    parser.add_argument("--lineage1", help="First lineage's training config")
    parser.add_argument("--lineage2", help="Second lineage's training config")
    parser.add_argument("--name1", help="Optional first lineage identifier")
    parser.add_argument("--name2", help="Optional second lineage identifier")
    parser.add_argument("--lr", help="Ligand-receptor CSV")
    parser.add_argument(
        "--early1",
        type=int,
        help="Number of initial stages in the first lineage's early window",
    )
    parser.add_argument(
        "--early2",
        type=int,
        help="Number of initial stages in the second lineage's early window",
    )
    parser.add_argument(
        "--rna", help="Optional combined RNA H5AD, instead of the models' RNA inputs"
    )
    parser.add_argument("--labels", help="Cell-label table for --rna")
    parser.add_argument(
        "--groups1", nargs="+", help="First lineage's groups in --rna, in model order"
    )
    parser.add_argument(
        "--groups2", nargs="+", help="Second lineage's groups in --rna, in model order"
    )


def build_ccc_config(
    training_configs, lr, early_counts, *, output="result/ccc", names=None
):
    paths = [str(Path(path).expanduser().resolve()) for path in training_configs]
    if len(paths) != 2 or len(early_counts) != 2:
        raise ValueError("CCC requires two training configs and two early windows.")
    models = [load_yaml_config(path) for path in paths]
    names = names or [
        str(model.get("lineage", f"lineage{i}")) for i, model in enumerate(models, 1)
    ]
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("Choose distinct lineage names with --name1 and --name2.")
    if any(count is None or count < 1 for count in early_counts):
        raise ValueError("Provide positive --early1 and --early2 stage counts.")
    specs, sources = {}, {}
    for name, path, count in zip(names, paths, early_counts):
        prefix = f"{name}::"
        specs[name] = {
            "config": path,
            "stage_prefix": prefix,
            "early_stage_count": count,
        }
        sources[name] = {"config": path, "stage_prefix": prefix}
    return {
        "config_version": 2,
        "base_dir": str(Path.cwd()),
        "data": {"lineages": sources},
        "ccc": {
            "lineage_order": names,
            "lineages": specs,
            "output_dir": str(Path(output).expanduser().resolve()),
            "ccc_csv": str(Path(lr).expanduser().resolve()),
            "alignment_method": "dtw",
            "dtw_ent_w": 0.0,
            "score_mode": "receptor_centrality",
        },
    }


def ccc_config_from_args(args):
    direct = (
        args.lineage1,
        args.lineage2,
        args.lr,
        args.early1,
        args.early2,
        args.rna,
        args.labels,
        args.groups1,
        args.groups2,
        args.name1,
        args.name2,
    )
    if args.config:
        if any(value is not None for value in direct):
            raise ValueError(
                "Use --config or the direct lineage-input options; use --set with an existing config."
            )
        return None
    if not all((args.lineage1, args.lineage2, args.lr)):
        raise ValueError("CCC requires --config or --lineage1, --lineage2 and --lr.")
    models = [load_yaml_config(path) for path in (args.lineage1, args.lineage2)]
    names = [
        args.name1 or str(models[0].get("lineage", "lineage1")),
        args.name2 or str(models[1].get("lineage", "lineage2")),
    ]
    config = build_ccc_config(
        [args.lineage1, args.lineage2],
        args.lr,
        [args.early1, args.early2],
        output=args.output or "result/ccc",
        names=names,
    )
    if args.rna:
        if not args.groups1 or not args.groups2:
            raise ValueError(
                "--rna requires --groups1 and --groups2 in model stage order."
            )
        config["data"] = {
            "rna_h5ad": str(Path(args.rna).expanduser().resolve()),
            "group_column": "cell_type",
            "cell_id_normalization": "exact",
            "rna_layer": "lognorm",
        }
        if args.labels:
            config["data"]["annotation"] = str(Path(args.labels).expanduser().resolve())
        for name, groups in zip(names, (args.groups1, args.groups2)):
            config["ccc"]["lineages"][name].pop("stage_prefix")
            config["ccc"]["lineages"][name]["stage_groups"] = groups
    elif any(value is not None for value in (args.labels, args.groups1, args.groups2)):
        raise ValueError("--labels and --groups1/--groups2 accompany --rna.")
    return config
