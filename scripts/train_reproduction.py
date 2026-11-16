"""Train the three application models directly from prepared PyG bundles."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Directory containing mouse_graphs.pt, dro_germ_graphs.pt and soma_graphs.pt",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "mouse", "dro_germ", "soma"],
        default="all",
        help="Train all three datasets or select one",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result/reproduction"),
        help="Output root, with a separate directory per dataset",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda or a device such as cuda:0"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate graph inputs without starting training",
    )
    args = parser.parse_args(argv)

    import yaml

    from draft_paired.graph_input import load_graph_artifacts
    from draft_paired.utils import load_yaml_config
    from draft_paired.workflow import train_graph_model

    datasets = (
        ["mouse", "dro_germ", "soma"] if args.dataset == "all" else [args.dataset]
    )
    graph_dir, output = args.graph_dir.resolve(), args.output.resolve()
    # Check all requested file paths before training any model.
    for name in datasets:
        if not (graph_dir / f"{name}_graphs.pt").is_file():
            parser.error(f"Missing graph bundle: {graph_dir / f'{name}_graphs.pt'}")
    results = []
    for name in datasets:
        config = load_yaml_config(REPO / "configs/reproduction" / f"{name}.yaml")
        config["base_dir"] = str(REPO)
        config["data"] = {"graphs": str(graph_dir / f"{name}_graphs.pt")}
        destination = output / name
        config["output_dir"] = str(destination / "train")
        config["checkpoints"] = {"dir": str(destination / "model")}
        config["artifacts"] = {"dir": str(destination / "train/artifacts")}
        config["reproduce"] = {"output_dir": str(destination / "reproduce")}
        config["training"]["device"] = "cpu" if args.check_only else args.device
        family = "unpaired" if name == "mouse" else "paired"
        if args.check_only:
            artifacts = load_graph_artifacts(config, family)
            result = {
                "dataset": name,
                "graphs": str(graph_dir / f"{name}_graphs.pt"),
                "stages": len(artifacts["graphs_lineage"]),
                "node_types": artifacts["graphs_lineage"][0].node_types,
                "status": "input_validated",
            }
            del artifacts
        else:
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / "config.yaml"
            # Save the effective config for loading this checkpoint.
            if (
                path.exists()
                and yaml.safe_load(path.read_text(encoding="utf-8")) != config
            ):
                raise FileExistsError(
                    f"{path} belongs to a different run; choose another --output directory."
                )
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"Training {name} from {config['data']['graphs']}", flush=True)
            report = train_graph_model(path)
            result = {
                "dataset": name,
                "config": str(path),
                "checkpoint": report["training"]["checkpoint_path"],
                "status": "training_completed",
            }
        results.append(result)
        print(json.dumps(result), flush=True)
        gc.collect()
    return results


if __name__ == "__main__":
    main()
