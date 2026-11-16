import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scmtni_cases import write_inputs

from draft_unpaired.prior import prepare_stage_prior
from draft_unpaired.scmtni_prior import IntervalIndex, source_scores


def test_source_scores_match_official_cpp_and_r_golden():
    golden = json.loads(
        (Path(__file__).parent / "fixtures/scmtni_reference.json").read_text()
    )
    for case in golden["scoring_cases"]:
        network = {(tf, tg): score for tf, tg, score in case["input"]}
        _, _, top, ranked, _ = source_scores(
            network, {tf for tf, _ in network}, {tg for _, tg in network}
        )
        assert [list(r) for r in top] == case["top"]
        assert [list(r) for r in ranked] == case["ranked"]


def test_interval_overlaps_match_half_open_bedtools_semantics():
    records = [("chr1", 0, 100), ("chr1", 10, 20), ("chr1", 50, 60), ("chr2", 0, 100)]
    index = IntervalIndex(records)
    assert index.overlaps("chr1", 20, 50) == [0]
    assert index.overlaps("chr1", 15, 55) == [0, 1, 2]
    assert index.overlaps("chr1", 100, 101) == []
    assert index.overlaps("chrX", 0, 1) == []


def test_strict_mapping_obs_annotations_and_scores_match_upstream_golden(tmp_path):
    cfg = write_inputs(tmp_path)
    result = prepare_stage_prior(cfg, "prepared")
    assert result["report"]["method"] == "scmtni"
    assert not result["reused"]
    frame = pd.read_csv(tmp_path / "prepared/tf_tg_prior.tsv", sep="\t")
    golden = json.loads(
        (Path(__file__).parent / "fixtures/scmtni_reference.json").read_text()
    )
    assert frame.values.tolist() == golden["staged_prior"]
    assert prepare_stage_prior(cfg, "prepared")["reused"]
    relations = pd.read_csv(tmp_path / "prepared/TF_RE.tsv", sep="\t")
    # The high-scoring motif at a touching endpoint must not enter either node.
    assert relations.motif_score.max() == 99
    assert result["report"]["nodes"][0]["rna_cells"] == 2
    (tmp_path / "prepared/tf_tg_prior.tsv").write_text("changed")
    with pytest.raises(ValueError, match="output changed"):
        prepare_stage_prior(cfg, "prepared")


def test_gzipped_motifs_and_changed_inputs_are_audited(tmp_path):
    cfg = write_inputs(tmp_path)
    with gzip.open(tmp_path / "motifs.bed.gz", "wt") as handle:
        handle.write((tmp_path / "motifs.bed").read_text())
    cfg["data"]["motif_instances"] = "motifs.bed.gz"
    prepare_stage_prior(cfg, "prepared")
    (tmp_path / "promoters.bed").write_text(
        (tmp_path / "promoters.bed").read_text() + "chr1\t80\t81\tnew\t0\t+\tG3\n"
    )
    with pytest.raises(ValueError, match="inputs/settings changed"):
        prepare_stage_prior(cfg, "prepared")


@pytest.mark.parametrize(
    "missing", ["stage_peaks", "motif_instances", "genome_assembly"]
)
def test_strict_mode_never_substitutes_a_detection_fraction(tmp_path, missing):
    cfg = write_inputs(tmp_path)
    cfg["data"].pop(missing)
    with pytest.raises(ValueError):
        prepare_stage_prior(cfg, "prepared")
    assert not (tmp_path / "prepared").exists()


def test_scope_assumptions_must_be_explicit(tmp_path):
    cfg = write_inputs(tmp_path)
    cfg["prior_preparation"].pop("shared_gene_universe")
    with pytest.raises(ValueError, match="input gene universes"):
        prepare_stage_prior(cfg, "prepared")


def test_stage_specific_input_gene_universes_filter_before_density(tmp_path):
    cfg = write_inputs(tmp_path)
    cfg["prior_preparation"].pop("shared_gene_universe")
    for kind, values in (("TF", ["TFc", "TFa"]), ("TG", ["G1", "G2"])):
        path = tmp_path / f"node_{kind.lower()}.tsv"
        pd.DataFrame({"node_id": ["early", "late"], kind: values}).to_csv(
            path, sep="\t", index=False
        )
        cfg["data"][f"node_{kind.lower()}_scope"] = path.name
    result = prepare_stage_prior(cfg, "prepared")
    assert all(n["scope_filtered_edges"] == 1 for n in result["report"]["nodes"])
    prior = pd.read_csv(tmp_path / "prepared/tf_tg_prior.tsv", sep="\t")
    assert prior.values.tolist() == [
        ["early", "TFc", "G1", 1],
        ["late", "TFa", "G2", 1],
    ]


def test_strict_mode_rejects_detection_proxy_and_new_example_is_packaged(tmp_path):
    from draft_paired.utils import load_yaml_config
    from draft_unpaired.config_init import init_config

    cfg = write_inputs(tmp_path)
    cfg["prior_preparation"]["min_accessible_fraction"] = 0.05
    with pytest.raises(ValueError, match="called stage peaks"):
        prepare_stage_prior(cfg, "prepared")
    path = tmp_path / "mouse.yaml"
    init_config(example="mouse_motifs", lineage="another_lineage", output=path)
    generated = load_yaml_config(path)
    assert generated["lineage"] == "another_lineage"
    assert generated["prior_preparation"]["enabled"]
    assert generated["prior_preparation"]["method"] == "scmtni"
    assert "stage_peaks" in generated["data"]


def test_generic_motif_config_uses_own_resources_and_trains(tmp_path):
    from draft_paired.utils import load_yaml_config
    from draft_paired.workflow_cli import main as train_main
    from draft_unpaired.cli import main

    source = write_inputs(tmp_path / "inputs")
    path = tmp_path / "motif.yaml"
    assert (
        main(
            [
                "init-config",
                "--kind",
                "motif",
                "--lineage",
                "lineage2",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    config = load_yaml_config(path)
    assert config["lineage"] == "lineage2"
    assert config["data"]["genome_assembly"] is None
    assert config["topic"]["enrich_organism"] is None
    assert config["training"]["branch_weight_scales"] is None
    assert "application_profile" not in config
    assert "mouse" not in json.dumps(config).lower()
    assert config["prior_preparation"]["method"] == "scmtni"
    assert "min_accessible_fraction" not in config["prior_preparation"]
    config["base_dir"] = source["base_dir"]
    config["data"].update(source["data"])
    config["data"]["genome_assembly"] = "custom_build"
    config["prior_preparation"]["shared_gene_universe"] = True
    for key in ["model", "training", "output_dir", "checkpoints"]:
        config[key] = source[key]
    config["training"]["device"] = "cpu"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "prepared"
    assert main(["preprocess", "--config", str(path), "--output", str(output)]) == 0
    prior = pd.read_csv(output / "prior/tf_tg_prior.tsv", sep="\t")
    assert set(prior.node_id) == {"early", "late"}
    assert train_main(["train", "--config", str(output / "train.yaml")]) == 0
    report = json.loads((Path(source["output_dir"]) / "train_report.json").read_text())
    assert Path(report["training"]["checkpoint_path"]).is_file()


def test_automatic_prior_is_used_by_training_and_all_pipeline_steps(tmp_path):
    from draft_unpaired.pipeline import run_draft_pipeline
    from draft_unpaired.train import train_lineage_model

    cfg = write_inputs(tmp_path)
    cfg["prior_preparation"]["enabled"] = True
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    train = train_lineage_model(path)
    assert train["config_path"] == str(tmp_path / "prepared/unpaired.yaml")
    assert train["prior_preparation"]["reused"] is False
    result = run_draft_pipeline(path)
    assert result["prior_preparation"]["reused"] is True
    assert result["training"]["config_path"] == result["config_path"]
    assert (tmp_path / "network/t00_TF_TG.tsv").exists()
    assert (tmp_path / "reproduce").exists()


def test_rank_edges_source_orientation_and_ties_are_preserved():
    # Literal C++ incr: score=1 ranks above 3; tied-only programs rank 0.5.
    network = {(f"TF{i}", f"G{j}"): float(j) for i in range(10) for j in range(20)}
    _, _, _, ranked, _ = source_scores(
        network, {f"TF{i}" for i in range(10)}, {f"G{j}" for j in range(20)}
    )
    scores = {(tf, tg): score for tf, tg, score in ranked}
    assert scores["TF0", "G16"] == 1
    assert scores["TF0", "G19"] == 0.25
    assert np.isfinite(list(scores.values())).all()
