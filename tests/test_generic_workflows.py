import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch_geometric.data import HeteroData

from draft_paired.analysis.network import _dense_network_settings, _network_settings
from draft_paired.analysis.single_tf import _single_tf_settings
from draft_paired.train import (
    _checkpoint_selection,
    _resolve_tree_hyperparameters,
    _train_tree_artifacts,
)


def test_lineage_ids_do_not_select_scientific_defaults():
    configs = [
        {"config_version": 2, "lineage": name}
        for name in ["soma", "germ", "unpaired", "lineage1", "other_species"]
    ]
    for resolver in [
        _resolve_tree_hyperparameters,
        _network_settings,
        _dense_network_settings,
        _single_tf_settings,
    ]:
        results = []
        for config in configs:
            settings = resolver(config)
            results.append(
                {key: value for key, value in settings.items() if key != "output_dir"}
            )
        assert all(value == results[0] for value in results)
    assert all(_checkpoint_selection(c) == "best_loss" for c in configs)
    assert _checkpoint_selection({"lineage": "germ"}) == "last"
    assert _resolve_tree_hyperparameters({"lineage": "germ"})["lr"] == 5e-4


def test_generic_enrichment_does_not_select_a_species():
    from draft_paired.analysis.topic import _topic_settings
    from draft_unpaired.analysis.topic import _patched_config

    for config in [{"config_version": 2}, _patched_config({"config_version": 2})]:
        settings = _topic_settings(config)
        assert not settings["do_enrich"]
        assert settings["enrich_db"] is None and settings["enrich_organism"] is None


def test_unpaired_topic_follows_the_configured_gene_input_paths(tmp_path):
    from draft_paired.analysis.topic import _topic_settings
    from draft_unpaired.config_init import init_config

    path = tmp_path / "lineage.yaml"
    init_config(output=path)
    config = yaml.safe_load(path.read_text())
    config["data"].update(tf_list="data/new/TF.tsv", tg_list="data/new/TG.tsv")
    settings = _topic_settings(config)
    assert settings["tf_list_file"] == "data/new/TF.tsv"
    assert settings["tg_list_file"] == "data/new/TG.tsv"


@pytest.mark.parametrize(
    "mode,example", [("paired", "fly_germ"), ("unpaired", "mouse_development")]
)
def test_generic_and_application_config_generation_are_independent_of_ids(
    mode, example, tmp_path
):
    if mode == "paired":
        from draft_paired.config_init import init_config
    else:
        from draft_unpaired.config_init import init_config
    configs = []
    for name in ["soma", "germ", "lineage1"]:
        path = tmp_path / f"{name}.yaml"
        init_config(lineage=name, output=path)
        config = yaml.safe_load(path.read_text())
        assert config["lineage"] == name and config["config_version"] == 2
        assert config["topic"]["enrich_organism"] is None
        assert all(
            not config[key]["enabled"]
            for key in ["topic", "single_tf", "continuous_activity", "edge_cluster"]
        )
        configs.append(
            {key: value for key, value in config.items() if key != "lineage"}
        )
    assert configs[0] == configs[1] == configs[2]
    case = []
    for name in ["germ", "renamed"]:
        path = tmp_path / f"case_{name}.yaml"
        init_config(example=example, lineage=name, output=path)
        cfg = yaml.safe_load(path.read_text())
        assert cfg["application_profile"] == example
        from draft_paired.utils import load_yaml_config

        resolved = load_yaml_config(path)
        assert (
            Path(resolved["reproduce"]["artifact_path"])
            == Path(resolved["artifacts"]["dir"]) / resolved["artifacts"]["filename"]
        )
        case.append({key: value for key, value in cfg.items() if key != "lineage"})
    assert case[0] == case[1]
    if mode == "paired":
        assert (
            case[0]["training"]["lr"] == 5e-4
            and case[0]["training"]["checkpoint_selection"] == "last"
        )
    else:
        assert case[0]["training"]["branch_weight_scales"] == [7, 1]


def small_artifacts(branching):
    torch.manual_seed(71)
    graphs = []
    for _ in range(3):
        graph = HeteroData()
        for kind, n in [("TF", 2), ("RE", 3), ("TG", 3)]:
            graph[kind].x = torch.rand(n, 4)
        tr = torch.tensor([[0, 1, 0], [0, 1, 2]])
        rt = torch.tensor([[0, 1, 2], [0, 1, 2]])
        for key, edges in {
            ("TF", "combine", "RE"): tr,
            ("RE", "combined_by", "TF"): tr.flip(0),
            ("RE", "regulate", "TG"): rt,
            ("TG", "regulated_by", "RE"): rt.flip(0),
        }.items():
            graph[key].edge_index = edges
        graphs.append(graph)
    return {
        "graphs_lineage": graphs,
        "topo_order": [0, 1, 2],
        "parents_list": [[], [0], [0 if branching else 1]],
        "time_names": ["stage0", "stage1", "stage2"],
        "in_channels": 4,
        "model_kind": "multi_parent",
        "topology_kind": "branching" if branching else "linear",
    }


@pytest.mark.parametrize("branching", [False, True])
def test_renaming_preserves_training_state_and_predictions(tmp_path, branching):
    from draft_paired.models import build_model

    artifacts = small_artifacts(branching)
    base = {
        "config_version": 2,
        "model": {
            "hidden_channels": 3,
            "out_channels": 3,
            "gat_heads": 1,
            "dropout_rate": 0,
        },
        "training": {
            "num_epochs": 3,
            "auto_lambda_smooth": False,
            "lambda_smooth_init": 0,
            "log_every": 3,
        },
    }
    states, outputs = [], []
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name in ["germ", "lineage1"]:
            cfg = copy.deepcopy(base)
            cfg.update(lineage=name, base_dir=str(tmp_path / name))
            report = _train_tree_artifacts(artifacts, cfg)
            checkpoint = torch.load(
                report["checkpoint_path"], map_location="cpu", weights_only=False
            )
            states.append(checkpoint["state_dict"])
            model = build_model(
                "tree",
                in_channels=4,
                hidden_channels=3,
                out_channels=3,
                heads=1,
                dropout_rate=0,
            )
            model.load_state_dict(states[-1])
            model.eval()
            with torch.no_grad():
                outputs.append(
                    model(
                        artifacts["graphs_lineage"],
                        [0, 1, 2],
                        artifacts["parents_list"],
                    )
                )
        assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
        assert all(torch.equal(a, b) for a, b in zip(*outputs))
    finally:
        torch.set_num_threads(previous)


def test_named_lineage_models_match_generic_models():
    from draft_paired.models import build_model, build_model_for_lineage

    artifacts = small_artifacts(False)
    kwargs = dict(
        in_channels=4, hidden_channels=3, out_channels=3, heads=1, dropout_rate=0
    )
    for name, kind, parents in [
        ("soma", "tree", [[], [0], [1]]),
        ("germ", "chain", [-1, 0, 1]),
    ]:
        torch.manual_seed(19)
        named = build_model_for_lineage(name, **kwargs)
        torch.manual_seed(19)
        generic = build_model(kind, **kwargs)
        assert all(
            torch.equal(a, b)
            for a, b in zip(named.state_dict().values(), generic.state_dict().values())
        )
        named.eval()
        generic.eval()
        with torch.no_grad():
            assert all(
                torch.equal(a, b)
                for a, b in zip(
                    named(artifacts["graphs_lineage"], [0, 1, 2], parents),
                    generic(artifacts["graphs_lineage"], [0, 1, 2], parents),
                )
            )


def test_non_fly_gene_id_and_symbol_inputs_preserve_graph_data(tmp_path):
    from draft_paired.tutorial_data import prepare_demo_data
    from draft_paired.utils import load_yaml_config
    from draft_paired.preprocess import prepare_common_inputs
    import anndata as ad

    path = prepare_demo_data(tmp_path / "data", mode="paired")
    cfg = load_yaml_config(path)
    before = prepare_common_inputs(cfg)
    assert len(before["tf_list"]) == 50 and len(before["tg_list"]) == 150
    rna = ad.read_h5ad(cfg["data"]["rna_h5ad"])
    lookup = dict(zip(rna.var_names, rna.var.gene_name))
    rna.var_names = rna.var.gene_name
    rna.write_h5ad(cfg["data"]["rna_h5ad"])
    for key in ["tf_list", "tg_list"]:
        frame = pd.read_csv(cfg["data"][key], sep="\t")
        frame.drop(columns="gene_id").to_csv(cfg["data"][key], sep="\t", index=False)
    prior = pd.read_csv(cfg["data"]["re_tg_prior"], sep="\t")
    prior.TG = prior.TG.map(lookup)
    prior.to_csv(cfg["data"]["re_tg_prior"], sep="\t", index=False)
    after = prepare_common_inputs(cfg)
    np.testing.assert_array_equal(
        before["expr_df"].to_numpy(), after["expr_df"].to_numpy()
    )
    for key in ["edge_index_tf_re", "edge_index_re_tg"]:
        assert set(map(tuple, before[key].T.tolist())) == set(
            map(tuple, after[key].T.tolist())
        )
