import importlib
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from scmtni_cases import write_inputs as motif_inputs
from test_shared_inputs import inputs

from draft_paired.utils import load_yaml_config
from draft_paired.workflow import (
    graph_input_mode,
    preprocess_lineage,
    train_graph_model,
)
from draft_paired.workflow_cli import main as train_main


def configure_training(config):
    config.update(
        output_dir="train",
        checkpoints={"dir": "model"},
        model={
            "hidden_channels": 3,
            "out_channels": 3,
            "gat_heads": 1,
            "dropout_rate": 0,
        },
        training={
            "num_epochs": 2,
            "seed": 1,
            "log_every": 2,
            "auto_lambda_smooth": False,
            "lambda_smooth_init": 0,
            "device": "cpu",
        },
    )


@pytest.mark.parametrize("family", ["paired", "unpaired"])
def test_raw_inputs_preprocess_then_shared_training_without_raw_files(
    tmp_path, monkeypatch, family
):
    raw = tmp_path / "raw"
    raw.mkdir()
    config = inputs(raw)
    configure_training(config)
    config["data"]["atac_h5ad"] = "paired_atac.h5ad"
    if family == "unpaired":
        pd.DataFrame(
            {"node_id": ["early", "late"], "TF": ["TF1", "TF2"], "TG": ["TG1", "TG2"]}
        ).to_csv(raw / "prior.tsv", sep="\t", index=False)
        config["data"]["tf_tg_prior"] = "prior.tsv"
    path = tmp_path / "inputs.yaml"
    path.write_text(yaml.safe_dump(config))
    original_config = path.read_bytes()
    builder = importlib.import_module(f"draft_{family}.graph_builder")
    expected = builder.build_lineage_graph_artifacts(config, device=torch.device("cpu"))
    command = importlib.import_module(f"draft_{family}.cli").main
    out = tmp_path / "prepared"
    assert command(["preprocess", "--config", str(path), "--output", str(out)]) == 0
    assert not (raw / "model").exists()
    assert graph_input_mode(out / "graphs.pt") == family
    payload = torch.load(out / "graphs.pt", map_location="cpu", weights_only=False)
    for before, after in zip(expected["graphs_lineage"], payload["graphs"]):
        for kind in before.node_types:
            assert after[kind].x.device.type == "cpu"
            assert torch.equal(before[kind].x, after[kind].x)
        for edge in before.edge_types:
            assert torch.equal(before[edge].edge_index, after[edge].edge_index)
    before_bundle = (out / "graphs.pt").read_bytes()
    with pytest.raises(FileExistsError):
        preprocess_lineage(path, family, out)
    assert (out / "graphs.pt").read_bytes() == before_bundle
    assert path.read_bytes() == original_config
    # Graph-only training resolves input paths independently of the working directory.
    for assay in ("rna.h5ad", "paired_atac.h5ad"):
        (raw / assay).unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert train_main(["train", "--config", str(out / "train.yaml")]) == 0
    report = __import__("json").loads((raw / "train/train_report.json").read_text())
    assert Path(report["training"]["checkpoint_path"]).is_file()
    assert report["input_paths"] == {"graphs": str(out / "graphs.pt")}
    assert (out / "graphs.pt").read_bytes() == before_bundle


@pytest.mark.parametrize("method", ["scmtni", "accessible_peak_path"])
def test_unpaired_preprocess_includes_prior_preparation_once(
    tmp_path, monkeypatch, method
):
    raw = tmp_path / "raw"
    raw.mkdir()
    config = motif_inputs(raw) if method == "scmtni" else inputs(raw)
    configure_training(config)
    config["prior_preparation"].update(method=method, enabled=True)
    path = tmp_path / "motifs.yaml"
    path.write_text(yaml.safe_dump(config))
    out = tmp_path / "prepared"
    result = preprocess_lineage(path, "unpaired", out)
    assert (out / "prior/tf_tg_prior.tsv").is_file()
    prepared = load_yaml_config(result["config_path"])
    assert prepared["prior_preparation"] == {"enabled": False}
    assert graph_input_mode(out / "graphs.pt") == "unpaired"

    def no_rebuild(*args, **kwargs):
        raise AssertionError("Training must not rerun prior preparation.")

    import draft_unpaired.prior

    monkeypatch.setattr(draft_unpaired.prior, "prepare_stage_prior", no_rebuild)
    assert train_main(["train", "--config", str(out / "train.yaml")]) == 0


def test_shared_training_rejects_raw_inputs_and_inconsistent_graph_metadata(tmp_path):
    from test_graph_training import graph_bundle, settings

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"data": {"rna_h5ad": "absent.h5ad"}}))
    with pytest.raises(ValueError, match="run paired or unpaired preprocess"):
        train_graph_model(path)
    payload = graph_bundle("paired")
    payload["input_mode"] = "unpaired"
    graph_path = tmp_path / "graphs.pt"
    torch.save(payload, graph_path)
    path.write_text(yaml.safe_dump(settings(tmp_path, graph_path)))
    with pytest.raises(ValueError, match="does not match"):
        train_graph_model(path)
    config = load_yaml_config(path)
    config["prior_preparation"] = {"enabled": True}
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="belongs in preprocess"):
        train_graph_model(path)
