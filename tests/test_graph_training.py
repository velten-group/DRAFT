import copy
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from torch_geometric.data import HeteroData

from draft_paired.graph_input import load_graph_artifacts


def graph_bundle(family, branching=False):
    torch.manual_seed(17)
    graphs = []
    maps = {"tf": ["TF1", "TF2"], "tg": ["G1", "G2", "G3"]}
    if family == "paired":
        maps["re"] = ["R1", "R2", "R3"]
    for _ in range(3):
        graph = HeteroData()
        for kind, names in maps.items():
            graph[kind.upper()].x = torch.rand(len(names), 4)
        if family == "paired":
            tr = torch.tensor([[0, 1, 0], [0, 1, 2]])
            rt = torch.tensor([[0, 1, 2], [0, 1, 2]])
            relations = {
                ("TF", "combine", "RE"): tr,
                ("RE", "combined_by", "TF"): tr.flip(0),
                ("RE", "regulate", "TG"): rt,
                ("TG", "regulated_by", "RE"): rt.flip(0),
            }
        else:
            tr = torch.tensor([[0, 1, 0], [0, 1, 2]])
            relations = {
                ("TF", "regulate", "TG"): tr,
                ("TG", "regulated_by", "TF"): tr.flip(0),
            }
        for edge, indices in relations.items():
            graph[edge].edge_index = indices
        graphs.append(graph)
    return {
        "graphs": graphs,
        "time_names": ["early", "middle", "late"],
        "topo_order": [0, 1, 2],
        "parents_list": [[], [0], [0 if branching else 1]],
        "in_channels": 4,
        "T_sel": 3,
        "maps": maps,
    }


def settings(root, bundle):
    return {
        "config_version": 2,
        "lineage": "another_lineage",
        "base_dir": str(root),
        "data": {"graphs": str(bundle)},
        "output_dir": "train",
        "checkpoints": {"dir": "model"},
        "model": {
            "hidden_channels": 3,
            "out_channels": 3,
            "gat_heads": 1,
            "dropout_rate": 0,
        },
        "training": {
            "num_epochs": 2,
            "seed": 2025,
            "device": "cpu",
            "auto_lambda_smooth": False,
            "lambda_smooth_init": 0,
            "log_every": 2,
        },
    }


@pytest.mark.parametrize("family", ["paired", "unpaired"])
def test_graph_training_matches_direct_training_and_reloads_without_raw_inputs(
    tmp_path, monkeypatch, family
):
    import importlib

    trainer = importlib.import_module(f"draft_{family}.train")
    infer = importlib.import_module(f"draft_{family}.infer")
    bundle = graph_bundle(family, branching=True)
    path = tmp_path / "graphs.pt"
    torch.save(bundle, path)
    before = path.read_bytes()
    config = settings(tmp_path / "direct", path)
    direct = {
        "graphs_lineage": copy.deepcopy(bundle["graphs"]),
        "topo_order": bundle["topo_order"],
        "parents_list": bundle["parents_list"],
        "time_names": bundle["time_names"],
        "maps": bundle["maps"],
        "in_channels": 4,
        "model_kind": "multi_parent",
        "topology_kind": "branching",
        "ordered_tree": pd.DataFrame(
            {
                "node_id": bundle["time_names"],
                "parent_id": ["", "early", "early"],
                "stage_id": range(3),
            }
        ),
    }
    train_core = (
        trainer._train_tree_artifacts
        if family == "paired"
        else trainer._train_artifacts
    )
    expected = train_core(direct, config)
    expected_ck = torch.load(
        expected["checkpoint_path"], map_location="cpu", weights_only=False
    )
    config = settings(tmp_path / "graph_mode", path)
    cfg_path = tmp_path / "graph.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    def no_copy(_):
        raise AssertionError("A supplied graph file must not be copied or rewritten.")

    monkeypatch.setattr(trainer, "_graph_cpu_copy", no_copy)
    result = trainer.train_lineage_model(cfg_path)
    ck = torch.load(
        result["training"]["checkpoint_path"], map_location="cpu", weights_only=False
    )
    assert all(
        torch.equal(ck["state_dict"][key], value)
        for key, value in expected_ck["state_dict"].items()
    )
    assert Path(result["training"]["artifact_path"]) == path
    assert path.read_bytes() == before
    assert result["input_paths"] == {"graphs": str(path)}
    output = infer.run_inference(config)
    assert output["out_tf"].shape == (2, 3, 3)


def test_chain_graph_training_matches_germ_training(tmp_path):
    from draft_paired.train import _train_germ_artifacts, _train_tree_artifacts

    bundle = graph_bundle("paired")
    cfg = settings(tmp_path / "germ", tmp_path / "unused.pt")
    cfg["training"]["checkpoint_selection"] = "last"
    cfg["training"]["lr"] = 0.0005
    germ_artifacts = {
        "graphs_germline": copy.deepcopy(bundle["graphs"]),
        "topo_order": [0, 1, 2],
        "parent": [-1, 0, 1],
    }
    germ = _train_germ_artifacts(germ_artifacts, cfg)
    cfg["base_dir"] = str(tmp_path / "tree")
    tree = _train_tree_artifacts(
        {
            "graphs_lineage": bundle["graphs"],
            "topo_order": [0, 1, 2],
            "parents_list": bundle["parents_list"],
            "time_names": bundle["time_names"],
            "in_channels": 4,
        },
        cfg,
    )
    germ_ck = torch.load(germ["checkpoint_path"], map_location="cpu", weights_only=False)
    tree_ck = torch.load(tree["checkpoint_path"], map_location="cpu", weights_only=False)
    assert all(
        torch.equal(value, tree_ck["state_dict"][key])
        for key, value in germ_ck["model_state_dict"].items()
    )


def test_reproduction_command_trains_all_three_graph_cases(tmp_path, monkeypatch):
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "train_reproduction", repo / "scripts/train_reproduction.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    graph_dir, fake_repo = tmp_path / "graphs", tmp_path / "repo"
    graph_dir.mkdir()
    presets = fake_repo / "configs/reproduction"
    presets.mkdir(parents=True)
    for name in ["mouse", "dro_germ", "soma"]:
        family = "unpaired" if name == "mouse" else "paired"
        path = graph_dir / f"{name}_graphs.pt"
        torch.save(graph_bundle(family, branching=name != "dro_germ"), path)
        cfg = settings(tmp_path / "unused", path)
        (presets / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(runner, "REPO", fake_repo)
    checked = runner.main(["--graph-dir", str(graph_dir), "--check-only"])
    assert all(item["status"] == "input_validated" for item in checked)
    result = runner.main(
        [
            "--graph-dir",
            str(graph_dir),
            "--output",
            str(tmp_path / "output"),
            "--device",
            "cpu",
        ]
    )
    assert [item["dataset"] for item in result] == ["mouse", "dro_germ", "soma"]
    assert all(Path(item["checkpoint"]).is_file() for item in result)


@pytest.mark.parametrize("fault", ["parent", "mapping", "features", "family"])
def test_graph_input_rejects_inconsistent_bundles(tmp_path, fault):
    bundle = graph_bundle("paired")
    if fault == "parent":
        bundle["parents_list"][0] = [2]
    elif fault == "mapping":
        bundle["maps"].pop("re")
    elif fault == "features":
        bundle["graphs"][0]["TF"].x[0, 0] = float("nan")
    path = tmp_path / "input.pt"
    torch.save(bundle, path)
    with pytest.raises(ValueError):
        load_graph_artifacts(
            settings(tmp_path, path), "unpaired" if fault == "family" else "paired"
        )
