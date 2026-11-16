import hashlib
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest
import torch
from test_auto_prior import inputs as motif_inputs
from test_shared_inputs import inputs as supplied_inputs

from draft_paired.directory_inputs import prepare_directory_config
from draft_paired.utils import load_yaml_config
from draft_paired.workflow_cli import main


def _raw(root, family, embedded=False):
    root.mkdir()
    supplied_inputs(root)
    if family == "paired":
        (root / "atac.h5ad").write_bytes((root / "paired_atac.h5ad").read_bytes())
        pd.read_csv(root / "rna_annotation.tsv", sep="\t").to_csv(
            root / "atac_annotation.csv", index=False
        )
        (root / "RE.txt").unlink()
    if embedded:
        for assay in ("rna", "atac"):
            data = ad.read_h5ad(root / f"{assay}.h5ad")
            data.obs["cell_type"] = ["early", "early", "late", "late"]
            data.write_h5ad(root / f"{assay}.h5ad")
        (root / "rna_annotation.tsv").unlink()
        (root / "atac_annotation.csv").unlink()


@pytest.mark.parametrize("family", ["paired", "unpaired"])
@pytest.mark.parametrize("embedded", [False, True])
def test_directory_command_builds_graphs_and_ready_config_without_editing_yaml(
    tmp_path, family, embedded, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    raw, output = tmp_path / "samples", tmp_path / "prepared"
    _raw(raw, family, embedded)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in raw.iterdir()
    }
    assert (
        main(
            [
                "preprocess",
                "--input-dir",
                str(raw),
                "--mode",
                family,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    config = load_yaml_config(output / "inputs.yaml")
    train = load_yaml_config(output / "train.yaml")
    assert config["lineage"] == "samples"
    assert config["input_mode"] == family
    assert train["data"]["graphs"] == str(output / "graphs.pt")
    assert train["training"]["num_epochs"] == 600
    assert train["prior_preparation"] == {"enabled": False}
    if family == "paired":
        assert Path(config["data"]["re_list"]).read_text().splitlines() == [
            "chr1:1-10",
            "chr1:11-20",
            "chr1:21-30",
        ]
        assert not (raw / "RE.txt").exists()
    else:
        assert config["prior_preparation"]["min_accessible_fraction"] == 0.05
        assert (output / "prior/tf_tg_prior.tsv").is_file()
    graphs = torch.load(output / "graphs.pt", weights_only=False)
    assert graphs["parents_list"] == [[], [0]]
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in raw.iterdir()
    } == before
    original = (output / "inputs.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        main(
            [
                "preprocess",
                "--input-dir",
                str(raw),
                "--mode",
                family,
                "--output",
                str(output),
            ]
        )
    assert (output / "inputs.yaml").read_bytes() == original


@pytest.mark.parametrize("family", ["paired", "unpaired"])
def test_directory_command_builds_motif_priors_with_custom_local_resources(
    tmp_path, family, monkeypatch
):
    raw, output = tmp_path / "samples", tmp_path / "prepared"
    motif_inputs(raw, family, "hg38")

    def no_download(*args, **kwargs):
        raise AssertionError(
            "Local resources must avoid motif and annotation downloads."
        )

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    assert (
        main(
            [
                "preprocess",
                "--input-dir",
                str(raw),
                "--mode",
                family,
                "--genome",
                "hg38",
                "--lineage",
                "lineage1",
                "--output",
                str(output),
                "--gene-annotation",
                str(raw / "annotation.gtf.gz"),
                "--motif-instances",
                str(raw / "motifs.bed"),
                "--motif-to-tf",
                str(raw / "motif_to_tf.tsv"),
            ]
        )
        == 0
    )
    config = load_yaml_config(output / "inputs.yaml")
    assert config["prior_preparation"]["method"] == "auto_motif"
    assert config["data"]["genome_assembly"] == "hg38"
    assert (output / "prior/TF_RE.tsv").is_file()
    assert (output / "prior/RE_TG.tsv").is_file()
    assert (output / "graphs.pt").is_file()


def test_directory_discovery_rejects_missing_genome_and_incomplete_priors(tmp_path):
    raw = tmp_path / "samples"
    _raw(raw, "unpaired")
    (raw / "TF_RE.csv").unlink()
    with pytest.raises(ValueError, match="only one prior"):
        prepare_directory_config(raw, tmp_path / "output", mode="unpaired")
    (raw / "RE_TG.tsv").unlink()
    with pytest.raises(ValueError, match="--genome"):
        prepare_directory_config(raw, tmp_path / "output", mode="unpaired")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("family", ["paired", "unpaired"])
def test_prior_only_directory_command_uses_local_resources_and_settings(
    tmp_path, family
):
    raw, output = tmp_path / "samples", tmp_path / "prior"
    raw.mkdir()
    motif_inputs(raw, family, "hg38")
    assert (
        main(
            [
                "prepare-prior",
                "--input-dir",
                str(raw),
                "--mode",
                family,
                "--genome",
                "hg38",
                "--output",
                str(output),
                "--gene-annotation",
                str(raw / "annotation.gtf.gz"),
                "--motif-instances",
                str(raw / "motifs.bed"),
                "--motif-to-tf",
                str(raw / "motif_to_tf.tsv"),
                "--set",
                "prior_preparation.min_motif_score=0",
            ]
        )
        == 0
    )
    assert (output / "TF_RE.tsv").is_file()
    assert (output / "model_inputs.yaml").is_file()
    assert (
        load_yaml_config(output / "inputs.yaml")["prior_preparation"]["min_motif_score"]
        == 0
    )


def test_explicit_file_paths_resolve_discovery_ambiguity_and_custom_names(tmp_path):
    raw, output = tmp_path / "samples", tmp_path / "prepared"
    _raw(raw, "unpaired")
    pd.read_csv(raw / "TF.csv").to_csv(raw / "TF.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="Ambiguous tf_list"):
        prepare_directory_config(raw, output, mode="unpaired")
    (raw / "rna.h5ad").rename(raw / "expression.h5ad")
    with pytest.raises(FileNotFoundError, match="Missing rna_h5ad"):
        prepare_directory_config(raw, output, mode="unpaired")
    path = prepare_directory_config(
        raw,
        output,
        mode="unpaired",
        min_accessible_fraction=0.2,
        files={"tf_list": raw / "TF.csv", "rna_h5ad": raw / "expression.h5ad"},
    )
    config = load_yaml_config(path)
    assert config["data"]["rna_h5ad"] == str(raw / "expression.h5ad")
    assert config["data"]["tf_list"] == str(raw / "TF.csv")
    assert config["prior_preparation"]["min_accessible_fraction"] == 0.2


def test_directory_command_requires_mode_and_accepts_config_overrides(
    tmp_path, monkeypatch
):
    with pytest.raises(SystemExit) as missing:
        main(
            [
                "preprocess",
                "--input-dir",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
            ]
        )
    assert missing.value.code == 2
    from draft_paired import workflow

    source = tmp_path / "config.yaml"
    source.write_text("input_mode: unpaired\n", encoding="utf-8")
    original = source.read_bytes()
    captured = []

    def preprocess(config, mode, output, *, force):
        captured.append(load_yaml_config(config))
        return {"graph_path": str(tmp_path / "graphs.pt"), "config_path": str(config)}

    monkeypatch.setattr(workflow, "preprocess_lineage", preprocess)
    assert (
        main(
            [
                "preprocess",
                "--config",
                str(source),
                "--genome",
                "hg38",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert captured[0]["data"]["genome_assembly"] == "hg38"
    assert source.read_bytes() == original
