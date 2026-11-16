from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from draft_paired.analysis.consensus import (
    aggregate_ranks,
    consensus_times,
    export_consensus,
    sample_embeddings,
    stage_rankings,
)


def _tables():
    return {
        "0": pd.DataFrame(
            {
                "tg_id": ["a", "b", "c"],
                "tg_name": ["A", "B", "CG123"],
                "tg_idx": [0, 1, 2],
                "rank": [1, 2, 3],
                "score": [0.9, 0.8, 0.7],
            }
        ),
        "1": pd.DataFrame(
            {
                "tg_id": ["b", "c", "a"],
                "tg_name": ["B", "CG123", "A"],
                "tg_idx": [1, 2, 0],
                "rank": [1, 2, 3],
                "score": [0.9, 0.8, 0.7],
            }
        ),
    }


def test_geometric_consensus_ranks_stable_targets_and_records_missing_stage_penalty():
    tables = _tables()
    result = aggregate_ranks(tables)
    assert result.tg_id.tolist() == ["b", "a", "c"]
    assert result.iloc[0].geometric_mean_rank == pytest.approx(np.sqrt(2))
    tables["1"] = tables["1"].head(2)
    result = aggregate_ranks(tables).set_index("tg_id")
    assert result.loc["a", "geometric_mean_rank"] == pytest.approx(np.sqrt(3))
    assert result.loc["a", "stage_count"] == 1
    rrf = aggregate_ranks(tables, method="rrf", rrf_k=60).set_index("tg_id")
    assert rrf.loc["a", "rrf_score"] == pytest.approx(0.5 / 61)


def test_dense_window_preserves_integer_embeddings_and_rejects_branch_interpolation():
    times = consensus_times({"time_window": [0, 1], "n_interp": 4}, ["GSC", "Gb", "Sp"])
    np.testing.assert_allclose(times, [0, 0.2, 0.4, 0.6, 0.8, 1])
    values = np.array([[[1.0], [3.0]]])
    assert sample_embeddings(values, 0, 0.7)[0, 0] == 1
    assert sample_embeddings(values, 1, 0.7)[0, 0] == 3
    assert sample_embeddings(values, 0.5, 0.7)[0, 0] == pytest.approx(2)
    tf, tg = np.ones((1, 3, 2)), np.ones((3, 3, 2))
    with pytest.raises(ValueError, match="linear lineage"):
        stage_rankings(
            tf,
            tg,
            0,
            ["a", "b", "c"],
            ["A", "B", "C"],
            [np.ones(3) * 0.2] * 3,
            [0.5],
            {},
            min_fraction=0.1,
            max_fraction=0.7,
            parents=[[], [0], [0]],
        )


def test_expression_upper_bound_is_respected_in_consensus_rankings():
    tf = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    tg = np.array(
        [[[0.8, 0.2], [0.7, 0.3]], [[1.0, 0.0], [1.0, 0.0]], [[0.2, 0.8], [0.1, 0.9]]]
    )
    tables = stage_rankings(
        tf,
        tg,
        0,
        ["a", "housekeeping", "c"],
        ["A", "housekeeping", "C"],
        [np.array([0.2, 0.9, 0.3])] * 2,
        [0.0, 0.5, 1.0],
        {},
        min_fraction=0.1,
        max_fraction=0.7,
    )
    assert all("housekeeping" not in set(table.tg_id) for table in tables.values())
    assert all(table.iloc[0].tg_id == "a" for table in tables.values())


def test_consensus_export_filters_cluster_and_combines_real_re_evidence(tmp_path: Path):
    cluster = tmp_path / "clusters.csv"
    pd.DataFrame(
        {"tf_idx": [0, 0, 0], "tg_idx": [0, 1, 2], "edge_cluster": [0, 1, 1]}
    ).to_csv(cluster, index=False)

    def select(time, ids):
        assert ids == ["b"]
        return [
            {
                "TG_id": "b",
                "RE_name": "shared",
                "time_idx": time,
                "score_triplet_local": 0.5,
            },
            {
                "TG_id": "b",
                "RE_name": f"only_{time:g}",
                "time_idx": time,
                "score_triplet_local": 0.8 + time / 10,
            },
        ]

    report = export_consensus(
        {"base_dir": str(tmp_path)},
        {
            "edge_cluster_file": "clusters.csv",
            "exclude_gene_pattern": "^CG[0-9]+",
            "topn": 1,
        },
        "TF",
        "tf_id",
        0,
        _tables(),
        tmp_path,
        select,
    )
    assert report["edge_cluster"] == "1"
    assert report["target_ids"] == ["b"]
    assert report["target_res"] == [["shared", "only_1"]]
    assert Path(report["stage_ranks"]).exists()
    assert report["rank_method"] == "geometric_mean"
