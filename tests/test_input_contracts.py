from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from draft_paired.analysis.ccc_timing import _receiver_scores, load_group_expression


def test_real_barcode_conventions_and_paired_cell_universe(tmp_path: Path):
    rna = ad.AnnData(
        np.array([[1.0, 2.0], [100.0, 100.0], [3.0, 4.0]]),
        obs=pd.DataFrame(index=["c1-1#S1", "c2-1#S1", "c3-1#S1"]),
        var=pd.DataFrame(index=["FB1", "FB2"]),
    )
    rna.layers["lognorm"] = rna.X.copy()
    rna.write_h5ad(tmp_path / "rna.h5ad")
    ad.AnnData(
        np.ones((2, 1)), obs=pd.DataFrame(index=["c1-1#s01", "c3-1#S1"])
    ).write_h5ad(tmp_path / "atac.h5ad")
    pd.DataFrame(
        {
            "cell_id": ["c1-1_sample1", "c2-1_sample1", "c3-1_sample1"],
            "cell_type": ["A", "A", "B"],
        }
    ).to_csv(tmp_path / "anno.csv", index=False)
    pd.DataFrame({"FBgn_ID": ["FB1"], "gene_name": ["Marker"]}).to_csv(
        tmp_path / "TF.tsv", sep="\t", index=False
    )
    pd.DataFrame({"FBgn_ID": ["FB2"], "gene_name": ["Target"]}).to_csv(
        tmp_path / "TG.tsv", sep="\t", index=False
    )
    config = {
        "base_dir": str(tmp_path),
        "data": {
            "rna_h5ad": "rna.h5ad",
            "atac_h5ad": "atac.h5ad",
            "annotation": "anno.csv",
            "tf_list": "TF.tsv",
            "tg_list": "TG.tsv",
        },
    }
    mean, frac = load_group_expression(config, ["A", "B"], ["marker", "target"])
    np.testing.assert_allclose(mean.to_numpy(), [[1, 3], [2, 4]])
    assert mean.attrs["cell_counts"] == {"A": 1, "B": 1}
    assert mean.attrs["missing_genes"] == []
    assert (frac == 1).all().all()
    config["data"]["cell_id_normalization"] = "exact"
    with pytest.raises(ValueError, match="No RNA cells"):
        load_group_expression(config, ["A", "B"], ["marker", "target"])


def test_imported_integer_scores_can_map_explicit_stage_columns(tmp_path: Path):
    path = tmp_path / "scores.tsv"
    pd.DataFrame(
        {
            "gene": ["rec", None],
            "tg_id": ["FB1", "FB2"],
            "t1": [0.4, 0.1],
            "t2": [0.8, 0.2],
        }
    ).to_csv(path, sep="\t", index=False)
    frame, groups, meta = _receiver_scores(
        {"base_dir": str(tmp_path)},
        "germ",
        {
            "receptor_scores": str(path),
            "stage_groups": ["G0", "G1"],
            "stage_columns": ["t1", "t2"],
        },
        tmp_path,
    )
    assert groups == ["G0", "G1"]
    np.testing.assert_allclose(frame.to_numpy(), [[0.4, 0.8]])
    assert meta["sha256"]


def test_five_percent_boundary_is_inclusive_in_network_activity():
    import torch
    from torch_geometric.data import HeteroData

    from draft_paired.analysis.network import (
        _dense_active_masks,
        _dense_network_settings,
        _network_settings,
    )

    graph = HeteroData()
    graph["RE"].x = torch.ones((1, 20))
    graph["TF"].x = torch.zeros((2, 20))
    graph["TG"].x = torch.zeros((2, 20))
    graph["TF"].x[0, 0] = graph["TG"].x[0, 0] = 1
    settings = _dense_network_settings({})
    assert settings["active_thr"] == _network_settings({})["active_frac_tf_tg"] == 0.05
    tf, tg, _ = _dense_active_masks([graph], settings["active_thr"])
    assert tf.tolist() == tg.tolist() == [[True, False]]


@pytest.mark.parametrize(
    "libraries", ["GO_Biological_Process_2021", ["GO_Biological_Process_2021"]]
)
def test_species_and_custom_enrichment_background_are_preserved(
    tmp_path: Path, libraries
):
    from draft_paired.analysis.topic import _enrichment_inputs

    calls = []

    def library(**kwargs):
        calls.append(kwargs)
        return {"term": ["a", "b", "c"]}

    settings = {
        "enrich_db": libraries,
        "enrich_organism": "Mouse",
        "enrich_background": "network_targets",
    }
    sets, background, provenance = _enrichment_inputs(
        SimpleNamespace(get_library=library),
        settings,
        pd.DataFrame({"tg_name": ["a", "b", "b"]}),
        tmp_path,
    )
    assert sets == {"term": ["a", "b", "c"]} and background == ["a", "b"]
    assert calls[0]["name"] == "GO_Biological_Process_2021"
    assert calls[0]["organism"] == "Mouse"
    assert provenance["background_gene_count"] == 2 and provenance["library_sha256"]


@pytest.mark.parametrize("background", [None, "network_targets"])
def test_enrichment_accepts_cli_library_lists_without_losing_same_named_terms(
    tmp_path: Path, background
):
    from draft_paired.analysis.topic import _enrichment_inputs

    calls = []

    def library(**kwargs):
        calls.append(kwargs["name"])
        return {"shared_term": [kwargs["name"]]}

    gene_sets, _, _ = _enrichment_inputs(
        SimpleNamespace(get_library=library),
        {
            "enrich_db": ["library_A", "library_B"],
            "enrich_organism": "Human",
            "enrich_background": background,
        },
        pd.DataFrame({"tg_name": ["gene1", "gene2"]}),
        tmp_path,
    )
    if background is None:
        assert gene_sets == ["library_A", "library_B"]
    else:
        assert calls == ["library_A", "library_B"]
        assert gene_sets == {
            "library_A__shared_term": ["library_A"],
            "library_B__shared_term": ["library_B"],
        }


def test_enrichment_errors_are_reported_instead_of_silently_dropped(
    tmp_path: Path, monkeypatch
):
    import sys

    from draft_paired.analysis.topic import _topic_settings, run_enrichment_per_topic

    calls = []

    def unavailable(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("library unavailable")

    monkeypatch.setitem(sys.modules, "gseapy", SimpleNamespace(enrichr=unavailable))
    settings = _topic_settings({"topic": {"save_fig": False}})
    edges = pd.DataFrame(
        {"tf_idx": [0, 0, 0], "tg_idx": [0, 1, 2], "tg_name": ["a", "b", "c"]}
    )
    result = run_enrichment_per_topic(
        np.ones((3, 1)),
        np.ones((1, 2)),
        [0, 1],
        [(0, 0), (0, 1), (0, 2)],
        edges,
        tmp_path,
        settings,
    )
    assert calls[0]["organism"] == "Fly"
    assert (
        result["status"] == "partial" and result["errors"]["0"] == "library unavailable"
    )
