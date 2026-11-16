from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch_geometric.data import HeteroData

from draft_paired.analysis.ccc import run_ccc_analysis_from_config
from draft_paired.analysis.ccc_timing import _receiver_scores, receptor_percentiles
from draft_paired.tutorial_data import prepare_demo_ccc, prepare_demo_data
from draft_paired.utils import load_yaml_config
from draft_paired.workflow import preprocess_lineage
from draft_paired.workflow_cli import main as train_main


def test_absolute_percentiles_rank_negative_cosines_by_strength():
    tf = np.array([[1.0, 0.0]])
    tg = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(
        receptor_percentiles(tf, tg, [True], [True] * 3), [1 / 3, 2 / 3, 1]
    )
    np.testing.assert_allclose(
        receptor_percentiles(
            tf, tg, [True], [True] * 3, block_size=1, absolute_cosine=True
        ),
        [1, 1 / 3, 1],
    )


def test_unpaired_receiver_uses_graph_order_and_rna_activity(tmp_path, monkeypatch):
    import draft_paired.analysis.ccc_timing as timing
    from draft_unpaired import infer

    # Tables include an unmeasured gene and differ from graph row order.
    pd.DataFrame(
        {"gene_id": ["tf_extra", "tf_b", "tf_a"], "gene_name": ["Extra", "B", "A"]}
    ).to_csv(tmp_path / "TF.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"gene_id": ["tg_z", "tg_y", "tg_x"], "gene_name": ["Z", "Y", "X"]}
    ).to_csv(tmp_path / "TG.tsv", sep="\t", index=False)
    pd.DataFrame({"gene_name": ["A"]}).to_csv(
        tmp_path / "filter.tsv", sep="\t", index=False
    )
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input_mode": "unpaired",
                "base_dir": str(tmp_path),
                "data": {
                    "tf_list": "TF.tsv",
                    "tg_list": "TG.tsv",
                    "tf_filter_file": "filter.tsv",
                },
                "network": {"dense": {"active_thr": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    graphs = []
    for _ in range(2):
        graph = HeteroData()
        graph["TF"].x = torch.zeros((2, 5))
        graph["TG"].x = torch.zeros((3, 5))
        graphs.append(graph)
    monkeypatch.setattr(
        infer,
        "run_inference",
        lambda _: {
            "out_tf": torch.tensor([[[1.0, 0.0]] * 2, [[0.0, 1.0]] * 2]),
            "out_tg": torch.tensor(
                [[[-1.0, 0.0]] * 2, [[0.0, 1.0]] * 2, [[1.0, 0.0]] * 2]
            ),
            "graphs": graphs,
            "maps": {"tf": ["tf_a", "tf_b"], "tg": ["tg_x", "tg_y", "tg_z"]},
            "time_names": ["s0", "s1"],
        },
    )
    fractions = pd.DataFrame(
        {"s0": [1.0, 1.0, 1.0, 1.0, 1.0], "s1": [1.0, 1.0, 1.0, 0.0, 0.0]},
        index=["a", "b", "x", "y", "z"],
    )

    def expression(_config, groups, genes):
        assert groups == ["s0", "s1"]
        assert set(genes) == set(fractions.index)
        return fractions, fractions

    monkeypatch.setattr(timing, "load_group_expression", expression)
    frame, groups, meta = _receiver_scores(
        {}, "lineage1", {"config": str(config_path)}, tmp_path
    )
    assert groups == ["s0", "s1"] and list(frame.index) == ["x", "y", "z"]
    np.testing.assert_allclose(frame.to_numpy(), [[1, 1], [1 / 3, 0], [1, 1]])
    assert meta["input_mode"] == "unpaired" and "absolute cosine" in meta["definition"]
    assert Path(meta["path"]).is_file()


@pytest.mark.parametrize(
    "modes", [("paired", "paired"), ("unpaired", "unpaired"), ("paired", "unpaired")]
)
def test_ccc_reads_real_checkpoints_and_matches_score_table_reuse(tmp_path, modes):
    import importlib

    paths = []
    for i, mode in enumerate(modes, start=1):
        source = prepare_demo_data(
            tmp_path / f"lineage{i}", mode=mode, lineage=f"lineage{i}"
        )
        source_config = load_yaml_config(source)
        source_config["network"]["dense"].update(
            keep_id_cols=i == 2, use_one_based=True, add_prefix=True
        )
        source.write_text(yaml.safe_dump(source_config), encoding="utf-8")
        prepared = preprocess_lineage(source, mode, tmp_path / f"prepared{i}")
        assert train_main(["train", "--config", prepared["config_path"]]) == 0
        main = importlib.import_module(f"draft_{mode}.cli").main
        assert main(["network-export", "--config", prepared["config_path"]]) == 0
        assert not (tmp_path / f"lineage{i}/result/network_dense").exists()
        paths.append(prepared["config_path"])
    config = load_yaml_config(prepare_demo_ccc(paths, tmp_path / "ccc"))
    config["ccc"]["save_fig"] = False
    for spec in config["ccc"]["lineages"].values():
        spec.pop("input_mode")  # Infer each mode from its generated train.yaml.
    direct = run_ccc_analysis_from_config(config)
    assert direct["chosen_alignment"] == "DTW"
    from draft_paired.cli import main as ccc_main

    original_configs = [Path(path).read_bytes() for path in paths]
    cli_output = tmp_path / "direct_cli"
    assert (
        ccc_main(
            [
                "ccc",
                "--lineage1",
                paths[0],
                "--lineage2",
                paths[1],
                "--lr",
                config["ccc"]["ccc_csv"],
                "--early1",
                "2",
                "--early2",
                "2",
                "--output",
                str(cli_output),
                "--set",
                "ccc.save_fig=false",
            ]
        )
        == 0
    )
    assert [Path(path).read_bytes() for path in paths] == original_configs
    pd.testing.assert_frame_equal(
        pd.read_csv(direct["curves_path"], sep="\t"),
        pd.read_csv(cli_output / "ccc_curves.tsv", sep="\t"),
        check_exact=False,
        atol=1e-6,
        rtol=1e-6,
    )
    assert all(
        item["source"] == "checkpoint"
        for item in direct["alignment"]["network_sources"].values()
    )
    for i in (1, 2):
        assert not (tmp_path / f"lineage{i}/result/network_dense").exists()

    # Explicit file export and streamed model alignment use the same networks.
    for name, mode, path in zip(("lineage1", "lineage2"), modes, paths):
        main = importlib.import_module(f"draft_{mode}.cli").main
        assert main(["network-export", "--config", path, "--dense"]) == 0
        model_config = load_yaml_config(path)
        config["ccc"]["lineages"][name]["network_dir"] = model_config["network"][
            "dense"
        ]["output_dir"]
    config["ccc"]["output_dir"] = str(tmp_path / "from_files")
    from_files = run_ccc_analysis_from_config(config)
    assert all(
        item["source"] == "exported"
        for item in from_files["alignment"]["network_sources"].values()
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(direct["curves_path"], sep="\t"),
        pd.read_csv(from_files["curves_path"], sep="\t"),
        check_exact=False,
        atol=1e-6,
        rtol=1e-6,
    )
    for name, mode in zip(("lineage1", "lineage2"), modes):
        metadata = direct["receiver_scores"][name]
        assert metadata["source"] == "checkpoint" and metadata["input_mode"] == mode
        assert (
            "absolute cosine" in metadata["definition"]
            if mode == "unpaired"
            else "absolute cosine" not in metadata["definition"]
        )
        spec = config["ccc"]["lineages"][name]
        spec["receptor_scores"] = metadata["path"]
        spec.pop("config")
    config["ccc"]["output_dir"] = str(tmp_path / "reused")
    config["ccc"]["alignment_path"] = direct["alignment"]["path"]
    imported = run_ccc_analysis_from_config(config)
    a = pd.read_csv(direct["curves_path"], sep="\t")
    b = pd.read_csv(imported["curves_path"], sep="\t")
    assert not a.empty and np.isfinite(a.score).all()
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, rtol=1e-12)
