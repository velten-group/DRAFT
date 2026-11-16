from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData


@pytest.mark.parametrize("family", ["paired", "unpaired"])
def test_single_tf_exports_window_consensus_and_independent_activity(
    tmp_path: Path, monkeypatch, family
):
    names = ["a", "b", "CG123"]
    ids = ["a", "b", "c"]
    cells = ["c0", "c1", "c2", "c3"]
    expr = pd.DataFrame(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0]],
        index=["tfa", "tfb", *ids],
        columns=cells,
    )
    annotation = pd.DataFrame({"cell_id": cells, "cell_type": ["G0", "G0", "G1", "G1"]})
    tg = pd.DataFrame(
        {
            "FBgn_ID": ids,
            "gene_name": names,
            "chr": ["chr1"] * 3,
            "start": [1, 10, 20],
            "end": [5, 15, 25],
        }
    )
    tg_path = tmp_path / "TG.tsv"
    tg.to_csv(tg_path, sep="\t", index=False)
    tf = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])
    target = torch.tensor(
        [[[0.9, 0.1], [0.8, 0.2]], [[0.8, 0.2], [0.9, 0.1]], [[0.1, 0.9], [0.2, 0.8]]]
    )
    graphs = []
    for _ in range(2):
        graph = HeteroData()
        graph["TF"].x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        graph["RE"].x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        graph["TG"].x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        graphs.append(graph)
    inference = {
        "out_tf": tf,
        "out_tg": target,
        "out_re": tf.clone(),
        "graphs": graphs,
        "time_names": ["G0", "G1"],
        "parents_list": [[], [0]],
        "checkpoint_path": str(tmp_path / "model.pt"),
    }
    config = {
        "base_dir": str(tmp_path),
        "lineage": "germ",
        "single_tf": {
            "tf_symbols": ["tfa"],
            "time_for_top": 0,
            "topn_tg": 2,
            "save_fig": False,
            "consensus": {
                "enabled": True,
                "time_window": [0, 1],
                "n_interp": 1,
                "topn": 2,
                "exclude_gene_pattern": "^CG[0-9]+",
            },
        },
    }
    if family == "paired":
        import draft_paired.analysis.single_tf as module

        inference["maps"] = {
            "tf": ["tfa", "tfb"],
            "tg": ids,
            "re": ["chr1:1-5", "chr1:20-25"],
        }
        common = {
            "anno": annotation,
            "expr_df": expr,
            "tg_df": tg,
            "tf_df": pd.DataFrame(
                {"FBgn_ID": ["tfa", "tfb"], "gene_name": ["tfa", "tfb"]}
            ),
            "input_paths": SimpleNamespace(tg_list=tg_path),
        }
        monkeypatch.setattr(
            module,
            "_single_tf_context",
            lambda _: {
                "lineage": "germ",
                "inference": inference,
                "common": common,
                "maps": inference["maps"],
                "time_count": 2,
                "time_labels": ["G0", "G1"],
            },
        )
    else:
        import draft_unpaired.analysis.single_tf as module

        tf_path = tmp_path / "TF.tsv"
        pd.DataFrame(
            {"gene_id": ["tfa", "tfb"], "gene_name": ["RegA", "RegB"]}
        ).to_csv(tf_path, sep="\t", index=False)
        config["single_tf"]["tf_symbols"] = ["RegA"]
        common = SimpleNamespace(
            input_paths=SimpleNamespace(tf_list=tf_path),
            tf_list=["tfa", "tfb"],
            tg_list=ids,
            expr_df=expr,
            anno=annotation,
            annotation=annotation,
        )
        monkeypatch.setattr(module, "run_inference", lambda _: inference)
        monkeypatch.setattr(module, "prepare_common_inputs", lambda _: common)
    report = module.run_single_tf_analysis_from_config(config)
    consensus = report["results"][0]["consensus"]
    assert consensus["times"] == [0, 0.5, 1]
    assert "CG123" not in consensus["target_names"]
    assert len(consensus["target_ids"]) == 2
    assert Path(consensus["top_targets"]).exists()
    activity = pd.read_csv(
        tmp_path / "result/germ/single_tf/tfa_activity.tsv", sep="\t"
    )
    np.testing.assert_allclose(activity.activity, activity.centrality)
    assert "expression_weighted_activity" in activity
    if family == "paired":
        assert Path(consensus["regulon_re"]).exists()
