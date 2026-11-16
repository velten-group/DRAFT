import importlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from draft_paired.analysis.ccc_alignment import network_alignment
from draft_paired.tutorial_data import prepare_demo_data
from draft_paired.utils import load_yaml_config
from draft_paired.workflow import preprocess_lineage
from draft_paired.workflow_cli import main as train_main


@pytest.mark.parametrize("family", ["paired", "unpaired"])
def test_training_outputs_feed_clustering_and_tf_regulons(tmp_path: Path, family):
    source = prepare_demo_data(tmp_path / "inputs", mode=family)
    config = load_yaml_config(source)
    config["edge_cluster"]["save_fig"] = False
    config["topic"]["save_fig"] = False
    config["network"]["dense"]["enabled"] = (
        True  # Stage-only export is the default even with dense.enabled set.
    )
    config["continuous_activity"].update(tf_list=["TF1"], save_fig=False)
    config["single_tf"].update(
        tf_symbols=["choose_a_tf"],
        time_for_top="lineage1_stage_0",
        save_fig=False,
        min_expr_frac=0,
        max_expr_frac=1,
    )
    source.write_text(yaml.safe_dump(config))
    prepared = preprocess_lineage(source, family, tmp_path / "prepared")
    original = Path(prepared["config_path"]).read_bytes()
    assert train_main(["train", "--config", prepared["config_path"]]) == 0
    main = importlib.import_module(f"draft_{family}.cli").main
    for command in [
        "reproduce",
        "network-export",
        "edge-cluster",
        "topic",
        "single-tf",
        "continuous-activity",
    ]:
        options = []
        if command == "single-tf":
            stages = pd.read_csv(config["data"]["tree"]).node_id.tolist()
            options = [
                "--tf",
                "TF1",
                "--stage",
                stages[0],
                "--consensus-stages",
                *stages[:2],
            ]
        elif command == "continuous-activity":
            options = ["--tf", "TF1"]
        assert main([command, "--config", prepared["config_path"], *options]) == 0
    assert Path(prepared["config_path"]).read_bytes() == original
    output = tmp_path / "inputs/result"
    assert not (output / "network_dense").exists()
    activity = pd.read_csv(
        output / "continuous_activity/multi_tf_activity.tsv", sep="\t"
    )
    assert activity.centrality_raw.max() > 0
    alignment_output = output / "alignment"
    alignment_output.mkdir()
    alignment_config = {
        "ccc": {
            lineage: {"config": prepared["config_path"]} for lineage in ("soma", "germ")
        }
    }
    alignment, report = network_alignment(alignment_config, alignment_output)
    assert len(alignment) > 1
    assert not (output / "network_dense").exists()
    assert report["shared_tf_count"] >= 2 and report["shared_tg_count"] >= 2
    assert list((output / "edge_cluster").glob("edge_clusters_*.csv"))
    assert (output / "topic/topic_x_cluster_lda.csv").is_file()
    targets = list((output / "single_tf").glob("*_top_tg.tsv"))
    assert len(targets) == 1
    assert not pd.read_csv(targets[0], sep="\t").empty
    assert list((output / "single_tf").glob("*_consensus_all_targets.tsv"))
    effective = load_yaml_config(output / "single_tf/run_config.yaml")
    assert effective["single_tf"]["tf_symbols"] == ["TF1"]
