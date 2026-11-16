import importlib

import pandas as pd
import pytest
import yaml


@pytest.mark.parametrize("family", ["paired", "unpaired"])
def test_continuous_activity_accepts_tf_ids_from_command(tmp_path, family):
    pd.DataFrame({"gene_id": ["id_A"], "gene_name": ["A"]}).to_csv(
        tmp_path / "TF.tsv", sep="\t", index=False
    )
    network = tmp_path / "network"
    network.mkdir()
    scores = [0.8, 0.6, 0.4, 0.2]
    for time, score in enumerate(scores, start=1):
        pd.DataFrame({"TF": ["TF:id_A"], "TG": ["TG:x"], "prob": [score]}).to_csv(
            network / f"TF_TG_dense_t{time}.tsv", sep="\t", index=False
        )
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
    original = path.read_bytes()
    output = tmp_path / "activity"
    assert (
        importlib.import_module(f"draft_{family}.cli").main(
            [
                "continuous-activity",
                "--config",
                str(path),
                "--tf",
                "id_A",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    table = pd.read_csv(output / "multi_tf_activity.tsv", sep="\t")
    assert table.tf_query.tolist() == ["A"] * 4
    assert table.centrality_raw.tolist() == scores
    assert path.read_bytes() == original
