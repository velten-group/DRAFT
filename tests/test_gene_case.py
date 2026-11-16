"""Gene symbols that differ only by case (Drosophila H/h, Tor/tor) stay separate."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from draft_paired.analysis.ccc import _load_network_series
from draft_paired.analysis.ccc_timing import load_group_expression
from draft_paired.analysis.gene_keys import GeneKeys, resolve_symbols


def test_case_is_ignored_except_where_it_separates_genes():
    keys = GeneKeys(["H", "h", "Notch", "Tor"], ["tor", "Dl", "TF:Mad"])
    assert keys.keys(["H", "h", "NOTCH", "notch", "Tor", "tor", "TOR", "dl", "mad"]) == [
        "H",
        "h",
        "notch",
        "notch",
        "Tor",
        "tor",
        None,
        "dl",
        "mad",
    ]
    assert keys.spelling("notch") == "Notch"
    # Without a vocabulary every symbol is compared in lowercase.
    assert GeneKeys().keys(["Notch", " TOR "]) == ["notch", "tor"]


def test_single_tf_lookup_respects_case_pairs():
    import pytest

    from draft_paired.analysis.single_tf import resolve_tf_id_and_index

    ids = ["FBgn_H", "FBgn_h", "FBgn_Sa", "FBgn_sa", "FBgn_ovo"]
    names = ["H", "h", "Sa", "sa", "ovo"]
    common = {"tf_df": pd.DataFrame({"gene_name": names, "FBgn_ID": ids})}
    assert resolve_tf_id_and_index("h", ids, names, common) == ("FBgn_h", 1)
    assert resolve_tf_id_and_index("H", ids, names, common) == ("FBgn_H", 0)
    assert resolve_tf_id_and_index("OVO", ids, names, common) == ("FBgn_ovo", 4)
    with pytest.raises(ValueError, match="differ only by case"):
        resolve_tf_id_and_index("SA", ids, names, common)


def test_continuous_activity_keeps_case_pairs_apart(tmp_path: Path):
    import yaml

    from draft_paired.cli import main

    pd.DataFrame({"gene_id": ["FB_H", "FB_h"], "gene_name": ["H", "h"]}).to_csv(
        tmp_path / "TF.tsv", sep="\t", index=False
    )
    network = tmp_path / "network"
    network.mkdir()
    for time in (1, 2, 3):
        pd.DataFrame(
            {"TF": ["H", "h"], "TG": ["x", "x"], "prob": [0.1 * time, 0.5]}
        ).to_csv(network / f"TF_TG_dense_t{time}.tsv", sep="\t", index=False)
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base_dir": str(tmp_path),
                "data": {"tf_list": "TF.tsv"},
                "continuous_activity": {"network_dir": "network", "save_fig": False},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "activity"
    command = ["continuous-activity", "--config", str(path), "--output", str(output)]
    assert main([*command, "--tf", "H", "h"]) == 0
    table = pd.read_csv(output / "multi_tf_activity.tsv", sep="\t")
    np.testing.assert_allclose(
        table.set_index(["tf_query", "time"]).centrality_raw.unstack().to_numpy(),
        [[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]],
    )


def test_other_tables_keep_the_gene_table_spelling():
    # The gene table has only Tor (key "tor"); RNA also measures torso (tor).
    request = GeneKeys(["Tor", "Notch"]).request(["tor", "notch"])
    resolved = resolve_symbols(pd.Series(["tor", "Tor", "NOTCH"]), request)
    assert pd.isna(resolved[0]) and resolved[1:].tolist() == ["tor", "notch"]
    # With both spellings in the gene table, each RNA symbol keeps its own key.
    request = GeneKeys(["Tor", "tor"]).request(["Tor", "tor"])
    resolved = resolve_symbols(pd.Series(["tor", "Tor", "TOR"]), request)
    assert resolved[:2].tolist() == ["tor", "Tor"] and pd.isna(resolved[2])


def test_network_series_keeps_case_pairs_apart(tmp_path: Path):
    pd.DataFrame({"gene_name": ["H", "h", "Dl"]}).to_csv(
        tmp_path / "tf.tsv", sep="\t", index=False
    )
    pd.DataFrame({"gene_name": ["Tor", "tor", "x"]}).to_csv(
        tmp_path / "tg.tsv", sep="\t", index=False
    )
    edges = pd.DataFrame(
        {
            "TF": ["H", "h", "dl", "H"],
            "TG": ["Tor", "tor", "X", "TOR"],  # TOR matches two genes and is skipped
            "prob": [0.1, 0.2, 0.3, 0.4],
        }
    )
    for t in range(2):
        edges.to_csv(tmp_path / f"TF_TG_dense_t{t}.tsv", sep="\t", index=False)
    _, matrices, tfs, tgs = _load_network_series(
        tmp_path, tmp_path / "tf.tsv", tmp_path / "tg.tsv"
    )
    assert tfs == ["H", "h", "dl"] and tgs == ["Tor", "tor", "x"]
    np.testing.assert_allclose(
        matrices[0].toarray(), [[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]], rtol=1e-6
    )
    _, _, tfs, tgs = _load_network_series(
        tmp_path, tmp_path / "tf.tsv", tmp_path / "tg.tsv", keep_case=True
    )
    assert tfs == ["H", "h", "Dl"] and tgs == ["Tor", "tor", "x"]


def test_receptor_ccc_scores_case_pairs_as_separate_receptors(tmp_path: Path):
    from draft_paired.analysis.ccc_timing import run_receptor_ccc

    groups = {"lin1": ["a1", "a2", "a3"], "lin2": ["b1", "b2", "b3"]}
    genes = ["Dl", "Tor", "tor"]
    for name, stages in groups.items():
        pd.DataFrame(
            [[0.5, 0.5, 0.5], [0.9, 0.6, 0.3], [0.2, 0.5, 0.8]],
            index=genes,
            columns=stages,
        ).to_csv(tmp_path / f"{name}.tsv", sep="\t", index_label="gene_name")
    rna = ad.AnnData(
        np.ones((6, 3)),
        obs=pd.DataFrame(
            {"cell_type": groups["lin1"] + groups["lin2"]},
            index=[f"c{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=genes),
    )
    rna.layers["lognorm"] = rna.X.copy()
    rna.write_h5ad(tmp_path / "rna.h5ad")
    pd.DataFrame({"lin1_tau": [0, 0.5, 1], "lin2_tau": [0, 0.5, 1]}).to_csv(
        tmp_path / "warp.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {"ligand": ["Dl", "Dl", "DL"], "receptor": ["Tor", "tor", "TOR"]}
    ).to_csv(tmp_path / "LR.csv", index=False)
    config = {
        "base_dir": str(tmp_path),
        "data": {
            "rna_h5ad": "rna.h5ad",
            "group_column": "cell_type",
            "cell_id_normalization": "exact",
        },
        "ccc": {
            "lineage_order": ["lin1", "lin2"],
            "lineages": {
                name: {
                    "receptor_scores": f"{name}.tsv",
                    "stage_groups": stages,
                    "early_stage_count": 1,
                }
                for name, stages in groups.items()
            },
            "ccc_csv": "LR.csv",
            "alignment_path": "warp.tsv",
            "output_dir": "out",
            "save_fig": False,
        },
    }
    report = run_receptor_ccc(config)
    pairs = pd.read_csv(report["pairs_path"], sep="\t")
    draft = pairs[pairs.method == "DRAFT"].set_index(["direction", "receptor"])
    assert set(draft.ligand) == {"dl"}
    assert draft.loc[("lin1_to_lin2", "Tor"), "status"] == "early"
    assert draft.loc[("lin1_to_lin2", "tor"), "status"] == "late"
    assert report["lr_symbols_ambiguous_by_case"] == ["TOR"]


def test_rna_expression_reads_case_pairs_separately(tmp_path: Path):
    rna = ad.AnnData(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        obs=pd.DataFrame({"cell_type": ["A", "B"]}, index=["c1", "c2"]),
        var=pd.DataFrame(index=["H", "h", "Notch"]),
    )
    rna.layers["lognorm"] = rna.X.copy()
    rna.write_h5ad(tmp_path / "rna.h5ad")
    config = {"base_dir": str(tmp_path), "data": {"rna_h5ad": "rna.h5ad"}}
    genes = GeneKeys(["H", "h", "Notch"]).request(["H", "h", "notch"])
    mean, _ = load_group_expression(config, ["A", "B"], genes)
    assert list(mean.index) == ["H", "h", "notch"]
    np.testing.assert_allclose(mean.to_numpy(), [[1, 4], [2, 5], [3, 6]])
