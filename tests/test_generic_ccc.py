import copy

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from draft_paired.analysis.ccc import run_ccc_analysis_from_config
from draft_paired.analysis.ccc_config import normalize_ccc_config
from draft_paired.analysis.ccc_timing import _receiver_scores


def test_generic_lineage_names_preserve_ccc_scores_and_alignment(tmp_path):
    genes = ["lig", "rec", "x"]
    groups = {"soma": ["S0", "S1", "S2"], "germ": ["G0", "G1", "G2"]}
    for key, stages in groups.items():
        pd.DataFrame(
            [[0.8, 0.6, 0.4], [0.9, 0.5, 0.2], [0.2, 0.4, 0.8]],
            index=genes,
            columns=stages,
        ).to_csv(tmp_path / f"{key}.tsv", sep="\t")
    stages = groups["soma"] + groups["germ"]
    ad.AnnData(
        np.ones((6, 3)),
        obs=pd.DataFrame({"cell_type": stages}, index=[f"c{i}" for i in range(6)]),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(tmp_path / "rna.h5ad")
    pd.DataFrame({"ligand": ["lig", "x"], "receptor": ["rec", "rec"]}).to_csv(
        tmp_path / "lr.csv", index=False
    )
    alignment = pd.DataFrame({"soma_tau": [0, 0.5, 1], "germ_tau": [0, 0.5, 1]})
    alignment.to_csv(tmp_path / "named_path.tsv", sep="\t", index=False)
    alignment.rename(columns={"soma_tau": "Alpha_tau", "germ_tau": "Beta_tau"}).to_csv(
        tmp_path / "generic_path.tsv", sep="\t", index=False
    )
    named = {
        "base_dir": str(tmp_path),
        "data": {
            "rna_h5ad": "rna.h5ad",
            "rna_layer": None,
            "group_column": "cell_type",
        },
        "ccc": {
            "ccc_csv": "lr.csv",
            "alignment_path": "named_path.tsv",
            "save_fig": False,
            "output_dir": "named",
            "directions": ["soma_to_germline", "germline_to_soma"],
            **{
                key: {
                    "receptor_scores": f"{key}.tsv",
                    "stage_groups": value,
                    "early_stage_count": 1,
                }
                for key, value in groups.items()
            },
        },
    }
    reference = run_ccc_analysis_from_config(named)
    generic = copy.deepcopy(named)
    section = generic["ccc"]
    section.update(output_dir="generic", alignment_path="generic_path.tsv")
    section["lineages"] = {"Alpha": section.pop("soma"), "Beta": section.pop("germ")}
    section["lineages"]["Beta"]["input_mode"] = (
        "unpaired"  # Imported scores are independent of input mode.
    )
    section["directions"] = [
        {"sender": "Alpha", "receiver": "Beta"},
        {"sender": "Beta", "receiver": "Alpha"},
    ]
    current = run_ccc_analysis_from_config(generic)
    mapping = {"Alpha_to_Beta": "soma_to_germline", "Beta_to_Alpha": "germline_to_soma"}
    for key in ["pairs_path", "curves_path"]:
        a, b = pd.read_csv(reference[key], sep="\t"), pd.read_csv(current[key], sep="\t")
        b.direction = b.direction.map(mapping)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert set(current["receiver_scores"]) == {"Alpha", "Beta"}
    assert set(current["stage_nodes"]) == {"Alpha", "Beta"}
    assert pd.read_csv(
        tmp_path / "generic/ccc_stage_alignment.tsv", sep="\t"
    ).columns.tolist() == ["Alpha_stage", "Beta_position"]


def test_generic_ccc_validates_ids_and_early_window():
    with pytest.raises(ValueError, match="exactly two"):
        normalize_ccc_config({"ccc": {"lineages": {"one": {}}}})
    with pytest.raises(ValueError, match="early_stage_count"):
        normalize_ccc_config({"ccc": {"lineages": {"Alpha": {}, "Beta": {}}}})
    with pytest.raises(ValueError, match="reserved"):
        normalize_ccc_config({"ccc": {"lineages": {"hub": {}, "Beta": {}}}})


def test_checkpoint_mode_must_match_its_training_config(tmp_path):
    import yaml

    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump({"input_mode": "unpaired"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must match the training config"):
        _receiver_scores(
            {}, "Alpha", {"input_mode": "paired", "config": str(path)}, tmp_path
        )
