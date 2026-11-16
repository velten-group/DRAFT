import importlib
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("family", ["paired", "unpaired"])
def test_pipeline_creates_cluster_input_before_consensus(
    tmp_path: Path, monkeypatch, family
):
    package = f"draft_{family}"
    pipeline = importlib.import_module(f"{package}.pipeline")
    cluster_file = tmp_path / "clusters.csv"
    calls = []

    def record(name):
        def run(config):
            calls.append(name)
            return {"status": "completed"}

        return run

    for name, function in (
        ("train", "train_lineage_model"),
        ("reproduce", "reproduce_lineage_model"),
    ):
        monkeypatch.setattr(pipeline, function, record(name))
    monkeypatch.setattr(
        importlib.import_module(f"{package}.analysis.network"),
        "export_lineage_networks",
        record("network"),
    )

    def cluster(config):
        cluster_file.write_text("tf_idx,tg_idx,edge_cluster\n0,0,1\n", encoding="utf-8")
        calls.append("edge_cluster")
        return {"edge_clusters_path": str(cluster_file)}

    def consensus(config):
        assert Path(config["single_tf"]["consensus"]["edge_cluster_file"]).exists()
        calls.append("single_tf")
        return {"status": "completed"}

    monkeypatch.setattr(
        importlib.import_module(f"{package}.analysis.edge_cluster"),
        "run_edge_cluster",
        cluster,
    )
    monkeypatch.setattr(
        importlib.import_module(f"{package}.analysis.single_tf"),
        "run_single_tf_analysis",
        consensus,
    )
    config = {
        "lineage": "example",
        "pipeline": {"output_dir": str(tmp_path / "output")},
        "edge_cluster": {"enabled": True},
        "single_tf": {
            "enabled": True,
            "consensus": {"edge_cluster_file": str(cluster_file)},
        },
        "method_profile": {"name": "methods", "provenance": "test"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = pipeline.run_draft_pipeline(path)
    assert calls == ["train", "reproduce", "network", "edge_cluster", "single_tf"]
    assert result["method_profile"] == config["method_profile"]
