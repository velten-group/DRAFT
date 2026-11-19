from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from draft_paired.input_schema import map_gene_names, read_gene_table
from draft_paired.lineage import load_tree_table
from draft_paired.preprocess import prepare_common_inputs as paired_inputs
from draft_unpaired.preprocess import prepare_common_inputs as unpaired_inputs
from draft_unpaired.prior import prepare_stage_prior


def inputs(root: Path, sparse: bool = False) -> dict:
    rna_cells = ["r0", "r1", "r2", "r3"]
    genes = ["idTF1", "idTF2", "idTG1", "idTG2"]
    rna = ad.AnnData(
        np.ones((4, 4)),
        obs=pd.DataFrame(index=rna_cells),
        var=pd.DataFrame(index=genes),
    )
    rna.layers["lognorm"] = np.arange(16, dtype=float).reshape(4, 4)
    rna.write_h5ad(root / "rna.h5ad")
    peaks = ["chr1:1-10", "chr1:11-20", "chr1:21-30"]
    counts = np.array([[2, 0, 1], [0, 0, 3], [0, 2, 0], [0, 1, 0]], dtype=float)
    atac = ad.AnnData(
        sp.csr_matrix(counts) if sparse else counts,
        obs=pd.DataFrame(index=["a0", "a1", "a2", "a3"]),
        var=pd.DataFrame(index=peaks),
    )
    atac.layers["counts"] = atac.X.copy()
    atac.write_h5ad(root / "atac.h5ad")
    atac.obs_names = rna_cells
    atac.write_h5ad(root / "paired_atac.h5ad")
    (root / "RE.txt").write_text("\n".join(peaks), encoding="utf-8")
    for kind in ("TF", "TG"):
        pd.DataFrame(
            {
                "gene_name": [f"{kind}1", f"{kind}2"],
                "gene_id": [f"id{kind}1", f"id{kind}2"],
            }
        ).to_csv(root / f"{kind}.csv", index=False)
    pd.DataFrame(
        {"cell_id": rna_cells, "node_id": ["early", "early", "late", "late"]}
    ).to_csv(root / "rna_annotation.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "cell_id": ["a0", "a1", "a2", "a3"],
            "node_id": ["early", "early", "late", "late"],
        }
    ).to_csv(root / "atac_annotation.csv", index=False)
    pd.DataFrame(
        {"node_id": ["early", "late"], "parent_id": ["", "early"], "stage_id": [0, 1]}
    ).to_csv(root / "tree.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"TF": ["TF1", "idTF2", "TF1"], "RE": peaks, "score": [9, 8, 3]}
    ).to_csv(root / "TF_RE.csv", index=False)
    pd.DataFrame({"RE": peaks, "TG": ["TG1", "idTG2", "idTG1"]}).to_csv(
        root / "RE_TG.tsv", sep="\t", index=False
    )
    return {
        "config_version": 2,
        "lineage": "arbitrary",
        "base_dir": str(root),
        "data": {
            "rna_h5ad": "rna.h5ad",
            "rna_layer": "lognorm",
            "atac_h5ad": "atac.h5ad",
            "atac_layer": "counts",
            "annotation": "rna_annotation.tsv",
            "atac_annotation": "atac_annotation.csv",
            "tree": "tree.tsv",
            "tf_list": "TF.csv",
            "tg_list": "TG.csv",
            "re_list": "RE.txt",
            "tf_re_prior": "TF_RE.csv",
            "re_tg_prior": "RE_TG.tsv",
        },
        "prior_preparation": {"min_accessible_fraction": 0.5},
    }


@pytest.mark.parametrize("sparse", [False, True])
def test_independent_atac_cells_generate_staged_prior_and_shared_training_inputs(
    tmp_path, sparse
):
    from draft_paired.utils import load_yaml_config
    from draft_unpaired.graph_builder import build_lineage_graph_artifacts

    cfg = inputs(tmp_path, sparse)
    result = prepare_stage_prior(cfg, "prepared")
    prior = pd.read_csv(tmp_path / "prepared/tf_tg_prior.tsv", sep="\t")
    assert prior[["node_id", "TF", "TG"]].values.tolist() == [
        ["early", "idTF1", "idTG1"],
        ["late", "idTF2", "idTG2"],
    ]
    assert prior.n_supported_RE.tolist() == [2, 1]
    assert prior.score.tolist() == [1, 1]
    assert result["report"]["nodes"][0]["atac_cells"] == 2
    unpaired = unpaired_inputs(load_yaml_config(result["config_path"]))
    cfg["data"]["atac_h5ad"] = "paired_atac.h5ad"
    paired = paired_inputs(cfg)
    pd.testing.assert_frame_equal(
        paired["expr_df"], unpaired.expr_df, check_dtype=False
    )
    assert paired["tf_list"] == unpaired.tf_list == ["idTF1", "idTF2"]
    assert unpaired.maps["tf_names"] == ["TF1", "TF2"]
    assert paired["edge_index_re_tg"].shape[1] == 3
    artifacts = build_lineage_graph_artifacts(load_yaml_config(result["config_path"]))
    assert artifacts["time_names"] == ["early", "late"]
    with pytest.raises(FileExistsError):
        prepare_stage_prior(cfg, "prepared")


def test_layer_selection_falls_back_to_x_and_accepts_same_gene_tables(
    tmp_path,
):
    from draft_paired.utils import load_yaml_config

    cfg = inputs(tmp_path)
    result = prepare_stage_prior(cfg, "prepared")
    prepared = load_yaml_config(result["config_path"])
    normalized = unpaired_inputs(prepared)
    prepared["data"].pop("rna_layer")
    x_values = unpaired_inputs(prepared)
    assert not np.array_equal(normalized.expr_df.values, x_values.expr_df.values)
    assert (x_values.expr_df.values == 1).all()
    prepared["data"]["rna_layer"] = "absent_preferred_layer"
    pd.testing.assert_frame_equal(x_values.expr_df, unpaired_inputs(prepared).expr_df)


@pytest.mark.parametrize(
    "content",
    ["id1\nid2\n", "gene_id\nid1\nid2\n", "gene_id\tgene_name\nid1\tA\nid2\tB\n"],
)
def test_gene_tables_and_headerless_lists_do_not_lose_first_gene(tmp_path, content):
    path = tmp_path / "genes.tsv"
    path.write_text(content)
    assert read_gene_table(path).gene_id.tolist() == ["id1", "id2"]


def test_prepared_prior_cli_trains_and_exports_with_shared_csv_gene_tables(tmp_path):
    import yaml

    from draft_unpaired.cli import main

    cfg = inputs(tmp_path)
    cfg.update(
        output_dir=str(tmp_path / "train"),
        artifacts={"dir": str(tmp_path / "artifacts")},
        checkpoints={"dir": str(tmp_path / "model")},
        model={"hidden_channels": 4, "out_channels": 4, "dropout_rate": 0},
        training={
            "num_epochs": 2,
            "seed": 1,
            "log_every": 1,
            "auto_lambda_smooth": False,
            "lambda_smooth_init": 0,
        },
        network={
            "output_dir": str(tmp_path / "network"),
            "min_score_keep": 0,
            "top_k_global": 4,
        },
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "prepared"
    assert main(["prepare-prior", "--config", str(path), "--output", str(out)]) == 0
    assert main(["train", "--config", str(out / "unpaired.yaml")]) == 0
    assert main(["network-export", "--config", str(out / "unpaired.yaml")]) == 0
    frame = pd.read_csv(tmp_path / "network/t00_TF_TG.tsv", sep="\t")
    assert not frame.empty


def test_ambiguous_symbols_require_ids(tmp_path):
    path = tmp_path / "genes.csv"
    path.write_text("gene_id,gene_name\nid1,A\nid2,A\n")
    table = read_gene_table(path)
    with pytest.raises(ValueError, match="Ambiguous"):
        map_gene_names(pd.Series(["A"]), table)
    assert map_gene_names(pd.Series(["id1"]), table).tolist() == ["id1"]


def test_prior_preparation_preserves_supplied_scopes(tmp_path):
    cfg = inputs(tmp_path)
    scope = pd.DataFrame(
        {
            "node_id": ["early", "early", "late", "late"],
            "TF": ["TF1", "TF2", "TF1", "TF2"],
        }
    )
    scope.to_csv(tmp_path / "tf_scope.csv", index=False)
    cfg["data"]["node_tf_scope"] = "tf_scope.csv"
    result = prepare_stage_prior(cfg, "prepared")
    actual = pd.read_csv(tmp_path / "prepared/node_tf_scope.tsv", sep="\t")
    assert actual.TF.tolist() == ["idTF1", "idTF2", "idTF1", "idTF2"]
    assert result["report"]["scope_policy"]["TF"] == "preserved supplied scope"


def test_generic_annotation_uses_canonical_groups_when_other_columns_coexist(
    tmp_path,
):
    cfg = inputs(tmp_path)
    cfg["data"]["atac_h5ad"] = "paired_atac.h5ad"
    annotation = pd.read_csv(tmp_path / "rna_annotation.tsv", sep="\t")
    annotation["cell_type"] = annotation["node_id"]
    annotation["GEX_latent_time_bins_annotated"] = "unused_group"
    annotation.to_csv(tmp_path / "rna_annotation.tsv", sep="\t", index=False)
    assert paired_inputs(cfg)["anno"].cell_type.tolist() == [
        "early",
        "early",
        "late",
        "late",
    ]


def test_tree_errors_are_explicit_and_atac_preparation_requires_group_correspondence(
    tmp_path,
):
    cfg = inputs(tmp_path)
    pd.DataFrame(
        {"node_id": ["early", "late"], "parent_id": ["", "unknown"], "stage_id": [0, 1]}
    ).to_csv(tmp_path / "bad_tree.csv", index=False)
    with pytest.raises(ValueError, match="unknown parent"):
        load_tree_table(tmp_path / "bad_tree.csv", strict=True)
    cfg["prior_preparation"].clear()
    with pytest.raises(ValueError, match="min_accessible_fraction"):
        prepare_stage_prior(cfg, "prepared")
    cfg["prior_preparation"]["min_accessible_fraction"] = 0.5
    pd.DataFrame({"cell_id": ["a0"], "cell_type": ["early"]}).to_csv(
        tmp_path / "atac_annotation.csv", index=False
    )
    with pytest.raises(ValueError, match="late"):
        prepare_stage_prior(cfg, "prepared")
    assert not (tmp_path / "prepared").exists()
