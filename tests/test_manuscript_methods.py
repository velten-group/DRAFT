from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from draft_paired.analysis.ccc import (
    _ccc_settings,
    dtw_path,
    run_ccc_analysis_from_config,
)
from draft_paired.analysis.ccc_alignment import fingerprint_dtw, network_alignment
from draft_paired.analysis.ccc_timing import (
    aligned_positions,
    receptor_percentiles,
    stage_nodes,
    timing_call,
)
from draft_paired.analysis.repro import summarize_smoothness


def test_zero_entropy_dtw_ignores_entropy_and_keeps_endpoints():
    class UnusableEntropy:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Entropy must not be read at zero weight")

    warp = fingerprint_dtw(
        np.zeros((4, 2)),
        np.zeros((7, 2)),
        entropy_soma=UnusableEntropy(),
        entropy_germ=UnusableEntropy(),
    )
    assert np.isfinite(warp).all()
    assert warp[0] == 0 and warp[-1] == 1
    assert np.all(np.diff(warp) >= 0)
    assert _ccc_settings({})["alignment_method"] == "dtw"
    assert _ccc_settings({})["dtw_ent_w"] == 0


def test_dtw_validates_unreachable_band_and_explicit_entropy():
    with pytest.raises(ValueError, match="No DTW path"):
        dtw_path(np.ones((2, 8)), band=0)
    with pytest.raises(ValueError, match="entropy"):
        fingerprint_dtw(np.zeros((3, 2)), np.zeros((4, 2)), entropy_weight=0.1)


def test_network_fingerprints_are_repeatable_and_ignore_rounding_noise():
    from scipy import sparse

    from draft_paired.analysis.ccc import (
        _normalize_bipartite,
        _top_r_svals,
        _zscore_cols,
    )

    matrices = [
        _normalize_bipartite(
            sparse.random(
                60, 200, density=0.1, random_state=seed, dtype=np.float32, format="csr"
            ),
            _ccc_settings({}),
        )
        for seed in range(4)
    ]
    first = np.vstack([_top_r_svals(m, 10) for m in matrices])
    for _ in range(3):
        np.testing.assert_array_equal(
            np.vstack([_top_r_svals(m, 10) for m in matrices]), first
        )
    # A constant column that differs only by float32 rounding carries no signal.
    noisy = np.column_stack(
        [np.float32(1) + np.float32(1e-7) * np.arange(4), first[:, 1]]
    ).astype(np.float32)
    z = _zscore_cols(noisy)
    assert np.all(z[:, 0] == 0)
    np.testing.assert_allclose(z[:, 1], _zscore_cols(first[:, [1]])[:, 0])


def test_empty_dense_stage_is_retained_in_alignment_grid(tmp_path: Path):
    from draft_paired.analysis.ccc import _load_network_series

    pd.DataFrame({"gene_name": ["a", "b"]}).to_csv(
        tmp_path / "tf.tsv", sep="\t", index=False
    )
    pd.DataFrame({"gene_name": ["x", "y"]}).to_csv(
        tmp_path / "tg.tsv", sep="\t", index=False
    )
    for t in range(3):
        rows = [] if t == 1 else [{"TF": "a", "TG": "x", "prob": 0.5}]
        pd.DataFrame(rows, columns=["TF", "TG", "prob"]).to_csv(
            tmp_path / f"TF_TG_dense_t{t}.tsv", sep="\t", index=False
        )
    times, matrices, _, _ = _load_network_series(
        tmp_path, tmp_path / "tf.tsv", tmp_path / "tg.tsv"
    )
    np.testing.assert_array_equal(times, [0, 1, 2])
    assert matrices[1].nnz == 0


@pytest.mark.parametrize("extension,separator", [("tsv", "\t"), ("csv", ",")])
def test_alignment_maps_exported_ids_and_symbols_to_the_same_network(
    tmp_path: Path, extension: str, separator: str
):
    from draft_paired.analysis.ccc import _load_network_series

    tf_path, tg_path = tmp_path / f"tf.{extension}", tmp_path / f"tg.{extension}"
    pd.DataFrame(
        {"gene_id": ["id_a", "id_b"], "gene_name": ["A", "B"]}
    ).to_csv(tf_path, sep=separator, index=False)
    pd.DataFrame(
        {"gene_id": ["id_x", "id_y"], "gene_name": ["X", "Y"]}
    ).to_csv(tg_path, sep=separator, index=False)
    loaded = []
    for label, tfs, tgs in [
        ("ids", ["TF:id_a", "TF:id_b"], ["TG:id_y", "TG:id_x"]),
        ("symbols", ["TF:A", "TF:B"], ["TG:Y", "TG:X"]),
    ]:
        directory = tmp_path / label
        directory.mkdir()
        for time in range(2):
            pd.DataFrame(
                {"TF": tfs, "TG": tgs, "prob": [0.25, 0.75]}
            ).to_csv(directory / f"TF_TG_dense_t{time}.tsv", sep="\t", index=False)
        times, matrices, names_tf, names_tg = _load_network_series(
            directory, tf_path, tg_path
        )
        np.testing.assert_array_equal(times, [0, 1])
        assert names_tf == ["a", "b"] and names_tg == ["x", "y"]
        loaded.append([matrix.toarray() for matrix in matrices])
    np.testing.assert_array_equal(loaded[0], loaded[1])
    np.testing.assert_array_equal(loaded[0][0], [[0, 0.25], [0.75, 0]])


def test_receiver_percentiles_use_full_active_reference_and_right_inclusive_ties():
    tf = np.array([[1, 0], [0, 1]])
    tg = np.array([[1, 0], [0, 1], [-1, 0]])
    result = receptor_percentiles(
        tf, tg, np.array([True, True]), np.array([True, True, False]), block_size=1
    )
    np.testing.assert_allclose(result, [0.75, 0.75, 0.25])
    with pytest.raises(ValueError, match="active TFs"):
        receptor_percentiles(
            tf, tg, np.array([False, False]), np.array([True, True, False])
        )


def test_peak_calls_distinguish_zero_flat_early_and_late():
    labels = ["GSC", "Gb", "Sc1", "Sc2"]
    assert timing_call(np.zeros(4), labels, 2)["status"] == "unscored"
    assert timing_call(np.ones(4), labels, 2)["status"] == "untimed"
    early = timing_call(np.array([1, 0.8, 0.4, 0.2]), labels, 2)
    late = timing_call(np.array([0.1, 0.2, 1, 0.5]), labels, 2)
    assert early["status"] == "early" and early["earliness_score"] == pytest.approx(0.6)
    assert late["status"] == "late" and late["earliness_score"] == pytest.approx(-0.8)


def test_parallel_terminal_groups_share_one_receiver_time():
    groups = ["stem", "early", "late", "SdA", "SdB"]
    nodes = stage_nodes(groups, ["SdA", "SdB"])
    assert nodes[-1] == ["SdA", "SdB"] and len(nodes) == 4
    table = pd.DataFrame(
        {"soma_tau": np.linspace(0, 1, 17), "germ_tau": np.linspace(0, 1, 17) ** 0.8}
    )
    soma, germ = aligned_positions(table, groups, nodes, 15)
    assert len(soma) == 15 and len(germ) == 4
    assert soma[0] == 1 and soma[-1] == 4
    assert germ[0] == 1 and germ[-1] == 15
    assert np.all(np.diff(soma) >= 0) and np.all(np.diff(germ) >= 0)


def test_branch_smoothness_avoids_cross_branch_steps_and_constant_is_one():
    # The last two nodes are sibling endpoints, not consecutive stages.
    values = np.array([0, 1, 2, 10])
    dist = np.abs(values[:, None] - values[None, :])
    summary = summarize_smoothness(
        dist, [[], [0], [1], [1]], ["root", "shared", "A", "B"]
    )
    assert [b["indices"] for b in summary["branches"]] == [[0, 1, 2], [0, 1, 3]]
    assert summary["overall_smoothness"] == pytest.approx(0.5)
    assert summarize_smoothness(np.zeros((3, 3)))["overall_smoothness"] == 1
    with pytest.raises(ValueError, match="cycle"):
        summarize_smoothness(dist, [[1], [0], [1], [1]])


def test_germline_checkpoint_and_learning_rate_defaults():
    from draft_paired.train import _checkpoint_selection, _resolve_tree_hyperparameters

    assert _checkpoint_selection({"lineage": "germ"}) == "last"
    assert _checkpoint_selection({"lineage": "soma"}) == "best_loss"
    assert (
        _checkpoint_selection(
            {"lineage": "germ", "training": {"checkpoint_selection": "best_loss"}}
        )
        == "best_loss"
    )
    assert _resolve_tree_hyperparameters({"lineage": "germ"})["lr"] == 5e-4


def test_activity_plot_keeps_network_and_rna_peaks_separate():
    import matplotlib.pyplot as plt

    from draft_paired.analysis.single_tf import _plot_activity_on_ax

    fig, ax = plt.subplots()
    try:
        data = pd.DataFrame(
            {"time_num": [0, 1, 2], "centrality": [3, 2, 1], "mean_expr": [1, 2, 3]}
        )
        _plot_activity_on_ax(ax, data, "Mondo", "id", 12, 10)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [1, 0.5, 0])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [0, 0.5, 1])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("selection", ["last", "best_loss"])
def test_tree_training_saves_selected_checkpoint(tmp_path: Path, selection):
    import torch
    from torch_geometric.data import HeteroData

    from draft_paired.checkpoint import torch_load_compat
    from draft_paired.train import _train_tree_artifacts

    graphs = []
    generator = torch.Generator().manual_seed(7)
    for _ in range(3):
        graph = HeteroData()
        for key, n in (("TF", 2), ("RE", 2), ("TG", 3)):
            graph[key].x = torch.rand((n, 4), generator=generator)
        tr, rt = torch.tensor([[0, 1], [0, 1]]), torch.tensor([[0, 1, 1], [0, 1, 2]])
        for (a, rel, b), edges in {
            ("TF", "combine", "RE"): tr,
            ("RE", "combined_by", "TF"): tr.flip(0),
            ("RE", "regulate", "TG"): rt,
            ("TG", "regulated_by", "RE"): rt.flip(0),
        }.items():
            graph[a, rel, b].edge_index = edges
        graphs.append(graph)
    artifacts = {
        "graphs_lineage": graphs,
        "topo_order": [0, 1, 2],
        "parents_list": [[], [0], [1]],
        "time_names": ["G0", "G1", "G2"],
        "in_channels": 4,
        "maps": {
            "tf": ["tfa", "tfb"],
            "re": ["chr1:1-5", "chr1:6-10"],
            "tg": ["lig", "rec", "tg3"],
        },
    }
    config = {
        "lineage": "germ",
        "base_dir": str(tmp_path),
        "model": {
            "hidden_channels": 2,
            "out_channels": 2,
            "gat_heads": 1,
            "dropout_rate": 0,
        },
        "training": {"num_epochs": 3, "checkpoint_selection": selection},
    }
    report = _train_tree_artifacts(artifacts, config)
    saved = torch_load_compat(report["checkpoint_path"], map_location="cpu")
    expected_epoch = 3 if selection == "last" else report["best_epoch"]
    assert saved["epoch"] == expected_epoch
    assert saved["meta"]["checkpoint_selection"] == selection
    assert report["selected_epoch"] == expected_epoch
    if selection == "last":
        import yaml

        from draft_paired.analysis.ccc_timing import _receiver_scores

        pd.DataFrame({"gene_name": ["tfa", "tfb"]}).to_csv(
            tmp_path / "TF.tsv", sep="\t", index=False
        )
        pd.DataFrame(
            {
                "gene_name": ["lig", "rec", "tg3"],
                "FBgn_ID": ["lig", "rec", "tg3"],
                "chr": ["chr1"] * 3,
                "start": [1, 2, 3],
                "end": [4, 5, 6],
            }
        ).to_csv(tmp_path / "TG.tsv", sep="\t", index=False)
        config["data"] = {"tf_list": "TF.tsv", "tg_list": "TG.tsv", "tree": "tree.csv"}
        path = tmp_path / "lineage.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        scores, stages, provenance = _receiver_scores(
            {"base_dir": str(tmp_path)}, "germ", {"config": str(path)}, tmp_path
        )
        assert list(scores.index) == ["lig", "rec", "tg3"]
        assert stages == ["G0", "G1", "G2"]
        assert scores.shape == (3, 3) and np.isfinite(scores.to_numpy()).all()
        assert provenance["source"] == "checkpoint"


def test_literature_benchmark_uses_same_candidates_for_methods():
    from draft_paired.analysis.ccc_timing import literature_benchmark

    rows = []
    for method, values in (
        ("DRAFT", [1, 0.8, -1, -0.5]),
        ("LR_expression", [-1, -0.8, 1, 0.5]),
    ):
        for i, value in enumerate(values):
            rows.append(
                {
                    "method": method,
                    "direction": "soma_to_germline",
                    "ligand": f"l{i}",
                    "receptor": "r",
                    "earliness_score": value,
                    "status": "early" if value > 0 else "late",
                }
            )
    literature = pd.DataFrame(
        {
            "direction": ["soma_to_germline"] * 2,
            "ligand": ["l0", "l1"],
            "receptor": ["r"] * 2,
        }
    )
    result = literature_benchmark(pd.DataFrame(rows), literature, draws=10).set_index(
        "method"
    )
    assert result.loc["DRAFT", "balanced_auroc"] == 1
    assert result.loc["LR_expression", "balanced_auroc"] == 0


def test_full_ccc_from_rna_and_percentiles_uses_all_directions_without_entropy(
    tmp_path: Path, monkeypatch
):
    import anndata as ad

    import draft_paired.analysis.ccc as ccc_module

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "The DTW workflow must not run the warping or entropy selection"
        )

    for name in [
        "learn_monotone_warp_continuous",
        "prepare_entropy_on_net_grid",
        "pick_alignment_method",
        "entropy_corr_score",
    ]:
        monkeypatch.setattr(ccc_module, name, forbidden)
    genes = ["lig", "rec", "x", "y"]
    stages = {"soma": ["S0", "SdA", "SdB"], "germ": ["G0", "G1", "G2"]}
    for name, groups in stages.items():
        pd.DataFrame(
            [[0.8, 0.7, 0.6], [0.9, 0.4, 0.2], [0.4, 0.5, 0.6], [0.3, 0.5, 0.6]],
            index=genes,
            columns=groups,
        ).to_csv(tmp_path / f"{name}.tsv", sep="\t")
        directory = tmp_path / name
        directory.mkdir()
        for t in range(3):
            rows = [
                {"TF": tf, "TG": gene, "prob": 0.1 + (i + 1) * (j + 1 + t) / 50}
                for i, tf in enumerate(["tfa", "tfb", "tfc"])
                for j, gene in enumerate(genes)
            ]
            pd.DataFrame(rows).to_csv(
                directory / f"TF_TG_dense_t{t}.tsv", sep="\t", index=False
            )
    pd.DataFrame({"gene_name": ["tfa", "tfb", "tfc"]}).to_csv(
        tmp_path / "TF.tsv", sep="\t", index=False
    )
    pd.DataFrame({"gene_name": genes}).to_csv(
        tmp_path / "TG.tsv", sep="\t", index=False
    )
    pd.DataFrame({"ligand": ["lig"], "receptor": ["rec"]}).to_csv(
        tmp_path / "lr.csv", index=False
    )
    group_order = ["Hub", *stages["soma"], *stages["germ"]]
    expr = np.array(
        [
            [2, 1, 1, 1],
            [1, 1, 1, 1],
            [3, 1, 1, 1],
            [3, 1, 1, 1],
            [1, 0, 1, 1],
            [2, 1, 1, 1],
            [3, 2, 1, 1],
        ],
        dtype=float,
    )
    rna = ad.AnnData(
        expr,
        obs=pd.DataFrame({"cell_type": group_order}, index=[f"c{i}" for i in range(7)]),
        var=pd.DataFrame(index=genes),
    )
    rna.write_h5ad(tmp_path / "rna.h5ad")
    section = {
        "output_dir": "out",
        "ccc_csv": "lr.csv",
        "hub_group": "Hub",
        "save_fig": False,
        "tf_annot_soma": "TF.tsv",
        "tf_annot_germ": "TF.tsv",
        "tg_annot_soma": "TG.tsv",
        "tg_annot_germ": "TG.tsv",
        "network_dir_soma": "soma",
        "network_dir_germ": "germ",
    }
    for name, groups in stages.items():
        section[name] = {
            "receptor_scores": f"{name}.tsv",
            "stage_groups": groups,
            "early_stage_count": 1,
        }
    section["soma"]["parallel_groups"] = ["SdA", "SdB"]
    pd.DataFrame(
        [
            {
                "source": sender,
                "target": receiver,
                "ligand": "lig",
                "receptor": "rec",
                "prob": 1.0,
            }
            for sender in group_order
            for receiver in group_order
        ]
    ).to_csv(tmp_path / "baseline.tsv", sep="\t", index=False)
    section["baselines"] = {
        "CellChat": {
            "edges": "baseline.tsv",
            "sender": "source",
            "receiver": "target",
            "score": "prob",
        }
    }
    config = {
        "base_dir": str(tmp_path),
        "data": {"rna_h5ad": "rna.h5ad", "rna_layer": None},
        "ccc": section,
    }
    report = run_ccc_analysis_from_config(config)
    assert report["chosen_alignment"] == "DTW"
    assert len(report["candidates_per_direction"]) == 4
    assert report["alignment"]["settings"]["dtw_ent_w"] == 0
    curves = pd.read_csv(report["curves_path"], sep="\t")
    gated = curves[(curves.direction == "hub_to_germline") & (curves.stage_index == 0)]
    assert (gated.score == 0).all()
    hub = curves[(curves.direction == "hub_to_germline") & (curves.method == "DRAFT")]
    np.testing.assert_allclose(hub.score, [0, 0.8, 0.4])
    assert set(curves.method) == {"DRAFT", "LR_expression", "CellChat"}
    native = curves[
        (curves.direction == "hub_to_germline") & (curves.method == "CellChat")
    ]
    np.testing.assert_allclose(native.score, [0, 1, 1])
    bad = tmp_path / "invalid_alignment.tsv"
    pd.DataFrame({"soma_tau": [0, 1], "germ_tau": [1, 0]}).to_csv(
        bad, sep="\t", index=False
    )
    config["ccc"]["alignment_path"] = str(bad)
    with pytest.raises(ValueError, match="monotone"):
        network_alignment(config, tmp_path / "out")
