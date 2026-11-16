from pathlib import Path

import pytest
import yaml

from draft_paired.cli_options import apply_overrides
from draft_paired.tutorial_data import prepare_demo_data
from draft_paired.utils import load_yaml_config
from draft_paired.workflow_cli import main


def test_dotted_options_preserve_source_and_parse_yaml_types():
    original = {"data": {"rna_layer": "lognorm"}, "single_tf": {"tf_symbols": ["A"]}}
    effective = apply_overrides(
        original,
        [
            "data.rna_layer=null",
            "single_tf.tf_symbols=[A, B]",
            "single_tf.save_fig=false",
            "training.num_epochs=2",
        ],
    )
    assert original == {
        "data": {"rna_layer": "lognorm"},
        "single_tf": {"tf_symbols": ["A"]},
    }
    assert effective["data"]["rna_layer"] is None
    assert effective["single_tf"] == {"tf_symbols": ["A", "B"], "save_fig": False}
    assert effective["training"]["num_epochs"] == 2
    with pytest.raises(ValueError, match="KEY=VALUE"):
        apply_overrides(original, ["data.rna_layer"])
    with pytest.raises(TypeError, match="not a mapping"):
        apply_overrides(original, ["data.rna_layer.name=x"])


def test_separate_rna_inputs_allow_shared_barcodes_and_stage_names(tmp_path):
    import anndata as ad
    import numpy as np
    import pandas as pd

    from draft_paired.analysis.ccc_timing import load_group_expression
    from draft_paired.ccc_inputs import build_ccc_config

    paths = []
    for name, values in (
        ("One", [[1.0, 0.0], [2.0, 3.0]]),
        ("Two", [[10.0, 1.0], [20.0, 0.0]]),
    ):
        rna = tmp_path / f"{name}.h5ad"
        ad.AnnData(
            np.array(values),
            obs=pd.DataFrame({"cell_type": ["early", "late"]}, index=["c0", "c1"]),
            var=pd.DataFrame({"gene_name": ["lig", "rec"]}, index=["id1", "id2"]),
        ).write_h5ad(rna)
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "lineage": name,
                    "input_mode": "unpaired",
                    "config_version": 2,
                    "data": {
                        "rna_h5ad": str(rna),
                        "atac_h5ad": "unmatched_atac.h5ad",
                        "gene_symbol_column": "gene_name",
                        "rna_layer": "lognorm",
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    config = build_ccc_config(paths, tmp_path / "lr.csv", [1, 1])
    groups = ["One::early", "Two::early", "One::late", "Two::late"]
    mean, fraction = load_group_expression(config, groups, ["lig", "rec"])
    np.testing.assert_array_equal(mean.to_numpy(), [[1, 10, 2, 20], [0, 1, 3, 0]])
    np.testing.assert_array_equal(fraction.to_numpy(), [[1, 1, 1, 1], [0, 1, 1, 0]])
    assert mean.attrs["cell_counts"] == dict.fromkeys(groups, 1)


@pytest.mark.parametrize("mode", ["paired", "unpaired"])
def test_preprocessing_file_options_and_training_output_keep_original_configs(
    tmp_path, mode
):
    source = prepare_demo_data(tmp_path / "source", mode=mode)
    original = source.read_bytes()
    config = load_yaml_config(source)
    prepared = tmp_path / "prepared"
    assert (
        main(
            [
                "preprocess",
                "--config",
                str(source),
                "--mode",
                mode,
                "--rna",
                config["data"]["rna_h5ad"],
                "--output",
                str(prepared),
                "--set",
                "training.num_epochs=1",
            ]
        )
        == 0
    )
    assert source.read_bytes() == original
    training = prepared / "train.yaml"
    before = training.read_bytes()
    output = tmp_path / "my_run"
    assert main(["train", "--config", str(training), "--output", str(output)]) == 0
    assert training.read_bytes() == before
    runtime = load_yaml_config(output / "run_config.yaml")
    assert runtime["data"]["graphs"] == str(prepared / "graphs.pt")
    assert runtime["checkpoints"]["dir"] == str(output / "model")
    assert Path(runtime["edge_cluster"]["network_dir"]) == output / "network"
    assert list((output / "model").glob("*.pt"))


def test_topic_and_tf_named_options_override_yaml_without_changes(
    tmp_path, monkeypatch
):
    from draft_paired.analysis import single_tf, topic
    from draft_paired.cli import main as paired_main

    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "base_dir": str(tmp_path),
                "single_tf": {"consensus": {"time_window": [0, 1]}},
            }
        ),
        encoding="utf-8",
    )
    before = source.read_bytes()
    seen = []

    def capture(path):
        seen.append(load_yaml_config(path))
        return {"output_dir": str(tmp_path)}

    monkeypatch.setattr(single_tf, "run_single_tf_analysis", capture)
    monkeypatch.setattr(topic, "run_topic_analysis", capture)
    assert (
        paired_main(
            [
                "single-tf",
                "--config",
                str(source),
                "--tf",
                "A",
                "B",
                "--stage",
                "early",
                "--consensus-stages",
                "early",
                "late",
                "--output",
                str(tmp_path / "tf"),
            ]
        )
        == 0
    )
    assert seen[-1]["single_tf"]["consensus"] == {
        "enabled": True,
        "time_points": ["early", "late"],
    }
    assert (
        paired_main(
            [
                "topic",
                "--config",
                str(source),
                "--organism",
                "Human",
                "--gene-sets",
                "my_library",
                "--output",
                str(tmp_path / "topics"),
            ]
        )
        == 0
    )
    assert seen[-1]["topic"]["enrich_organism"] == "Human"
    assert seen[-1]["topic"]["enrich_db"] == ["my_library"]
    assert source.read_bytes() == before
