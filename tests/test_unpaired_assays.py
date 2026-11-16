import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from test_preprocessing_workflow import configure_training
from test_shared_inputs import inputs

from draft_paired.utils import load_yaml_config
from draft_paired.workflow_cli import main as train_main
from draft_unpaired.cli import main
from draft_unpaired.prior import prepare_stage_prior


def independent_inputs(root):
    config = inputs(root)
    rna = ad.read_h5ad(root / "rna.h5ad")
    rna.obs["node_id"] = ["early", "early", "late", "late"]
    rna.write_h5ad(root / "rna.h5ad")
    atac = ad.read_h5ad(root / "atac.h5ad")
    counts = np.vstack([atac.X, [2, 0, 1]])
    atac = ad.AnnData(
        counts,
        obs=pd.DataFrame(
            {"node_id": ["early", "early", "late", "late", "early"]},
            index=["a0", "a1", "a2", "a3", "a4"],
        ),
        var=atac.var.copy(),
    )
    atac.layers["counts"] = counts.copy()
    atac.write_h5ad(root / "atac.h5ad")
    assert not set(rna.obs_names) & set(atac.obs_names)
    assert rna.n_obs != atac.n_obs
    config["data"].update(annotation=None, atac_annotation=None)
    return config


def test_demo_inputs_use_independent_multiomics_and_preprocess_to_graphs(tmp_path):
    from draft_paired.tutorial_data import prepare_demo_data
    from draft_paired.workflow import preprocess_lineage

    path = prepare_demo_data(tmp_path / "inputs", mode="unpaired")
    config = load_yaml_config(path)
    assert "tf_tg_prior" not in config["data"]
    rna = ad.read_h5ad(config["data"]["rna_h5ad"])
    atac = ad.read_h5ad(config["data"]["atac_h5ad"])
    assert rna.n_obs != atac.n_obs
    assert set(rna.obs_names).isdisjoint(atac.obs_names)
    result = preprocess_lineage(path, "unpaired", tmp_path / "prepared")
    report = json.loads((tmp_path / "prepared/prior/prior_report.json").read_text())
    assert all(row["rna_cells"] > 0 and row["atac_cells"] > 0 for row in report["nodes"])
    assert all(row["candidate_edges"] > 0 for row in report["nodes"])
    graphs = torch.load(result["graph_path"], weights_only=False)
    assert graphs["input_mode"] == "unpaired"
    assert set(graphs["graphs"][0].node_types) == {"TF", "TG"}


def test_default_unpaired_inputs_use_independent_assays_and_atac_changes_candidates(
    tmp_path,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    supplied = independent_inputs(raw)
    path = tmp_path / "unpaired.yaml"
    assert (
        main(["init-config", "--lineage", "independent_lineage", "--output", str(path)])
        == 0
    )
    config = load_yaml_config(path)
    assert config["data"].get("atac_h5ad")
    assert "tf_tg_prior" not in config["data"]
    assert config["prior_preparation"]["enabled"]
    config["base_dir"] = str(raw)
    config["data"].update(supplied["data"])
    config["prior_preparation"]["min_accessible_fraction"] = 0.5
    configure_training(config)
    path.write_text(yaml.safe_dump(config))
    out = tmp_path / "prepared"
    assert main(["preprocess", "--config", str(path), "--output", str(out)]) == 0
    report = json.loads((out / "prior/prior_report.json").read_text())
    assert [(row["rna_cells"], row["atac_cells"]) for row in report["nodes"]] == [
        (2, 3),
        (2, 2),
    ]
    prior = pd.read_csv(out / "prior/tf_tg_prior.tsv", sep="\t")
    assert prior[["node_id", "TF", "TG"]].values.tolist() == [
        ["early", "idTF1", "idTG1"],
        ["late", "idTF2", "idTG2"],
    ]
    bundle = torch.load(out / "graphs.pt", weights_only=False)
    assert bundle["time_names"] == ["early", "late"]
    assert bundle["parents_list"] == [[], [0]]
    edge = ("TF", "regulate", "TG")
    assert bundle["graphs"][0][edge].edge_index.tolist() == [[0], [0]]

    atac = ad.read_h5ad(raw / "atac.h5ad")
    values = atac.layers["counts"].copy()
    values[atac.obs.node_id.eq("early"), :] = [0, 2, 0]
    atac.layers["counts"] = values
    atac.write_h5ad(raw / "atac.h5ad")
    changed = tmp_path / "changed"
    assert main(["preprocess", "--config", str(path), "--output", str(changed)]) == 0
    bundle = torch.load(changed / "graphs.pt", weights_only=False)
    assert bundle["graphs"][0][edge].edge_index.tolist() == [[1], [1]]
    # The training stage uses only the saved input, even after both assays disappear.
    (raw / "rna.h5ad").unlink()
    (raw / "atac.h5ad").unlink()
    assert train_main(["train", "--config", str(out / "train.yaml")]) == 0
    training = json.loads((raw / "train/train_report.json").read_text())
    assert Path(training["training"]["checkpoint_path"]).is_file()


@pytest.mark.parametrize("assay", ["rna", "atac"])
def test_unpaired_groups_must_cover_tree_nodes_within_each_assay(tmp_path, assay):
    config = independent_inputs(tmp_path)
    path = tmp_path / f"{assay}.h5ad"
    adata = ad.read_h5ad(path)
    adata.obs["node_id"] = adata.obs.node_id.astype(str).replace("late", "unmapped")
    adata.write_h5ad(path)
    with pytest.raises(ValueError, match=f"No annotated {assay.upper()} cells.*late"):
        prepare_stage_prior(config, "prepared")


def test_staged_prior_reuse_remains_an_explicit_input_layout(tmp_path):
    path = tmp_path / "staged.yaml"
    assert main(["init-config", "--kind", "staged-prior", "--output", str(path)]) == 0
    staged = load_yaml_config(path)
    assert staged["data"].get("tf_tg_prior")
    assert not staged["data"].get("atac_h5ad")
    assert not staged.get("prior_preparation", {}).get("enabled", False)
    main(["init-config", "--output", str(tmp_path / "default.yaml")])
    default = load_yaml_config(tmp_path / "default.yaml")
    for section in ["model", "training", "network"]:
        assert default[section] == staged[section]


def test_existing_pipeline_can_prepare_independent_assays_without_repeating_preparation(
    tmp_path,
):
    from draft_unpaired.pipeline import run_draft_pipeline

    config = independent_inputs(tmp_path)
    configure_training(config)
    config["prior_preparation"].update(enabled=True, method="accessible_peak_path")
    config["network"] = {
        "output_dir": "network",
        "min_score_keep": 0,
        "top_k_global": 4,
    }
    config["reproduce"] = {"output_dir": "reproduce"}
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config))
    result = run_draft_pipeline(path)
    assert result["status"] == "completed"
    assert result["prior_preparation"]["reused"] is False
    assert (tmp_path / "network/t00_TF_TG.tsv").is_file()
