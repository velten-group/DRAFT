import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from draft_paired import simulation


def test_simulation_cli_template_matches_requested_protocol(tmp_path: Path):
    from draft_paired.cli import main

    path = tmp_path / "simulation.yaml"
    assert main(["init-config", "--kind", "simulation", "--output", str(path)]) == 0
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["training"]["num_epochs"] == 600
    assert cfg["training"]["lr"] == 0.001
    assert cfg["training"]["lr_decay_start"] == 560
    assert cfg["training"]["lr_decay_end"] == 800
    assert cfg["training"]["final_lr"] == 0.0001
    assert cfg["training"]["gradient_clip_norm"] == 1
    assert cfg["training"]["checkpoint_selection"] == "last"
    assert cfg["training"]["reconstruction_reduction"] == "mean"
    assert cfg["training"]["smooth_weight"] == 0
    assert cfg["model"]["hidden_channels"] == cfg["model"]["out_channels"] == 32
    assert cfg["simulation"]["data_seeds"] == [101, 102, 103]
    assert cfg["simulation"]["model_seeds"] == [0, 1, 2, 3, 4]
    assert cfg["simulation"]["module_attention_beta"] == 0
    with pytest.raises(ValueError, match="module_attention_beta must be 0"):
        simulation._effective_config({"simulation": {"module_attention_beta": 1}})


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_bundled_main_observations_and_priors_are_complete(seed, monkeypatch):
    original = simulation._bundle_bytes

    def training_only(config, name):
        assert not name.startswith("truth_")
        return original(config, name)

    monkeypatch.setattr(simulation, "_bundle_bytes", training_only)
    inputs, provenance = simulation.load_simulation_inputs({}, seed)
    assert inputs["rna_counts"].shape == (5, 200, 200)
    assert inputs["atac_counts"].shape == (5, 900, 200)
    assert inputs["candidate"].sum() == 1800
    assert provenance["generation"]["candidate_inside"] == 9
    assert provenance["generation"]["candidate_outside"] == 3
    assert provenance["generation"]["seed"] == seed
    assert np.all(inputs["candidate"].sum(1) == 12)
    assert not any("truth" in key or "module" in key for key in inputs)


def test_simulation_normalizes_each_assay_without_changing_paired_cells():
    counts = np.array([[[0, 1, 0], [0, 3, 2]]], dtype=float)
    actual = simulation.log_counts_per_10000(counts)
    np.testing.assert_allclose(actual[0, :, 0], [0, 0])
    np.testing.assert_allclose(actual[0, :, 1], np.log1p([2500, 7500]), rtol=1e-6)
    np.testing.assert_allclose(actual[0, :, 2], np.log1p([0, 10000]), rtol=1e-6)


def test_simulation_rejects_tampered_observations(tmp_path: Path):
    resource = simulation.resources.files("draft_paired").joinpath(
        "examples", "simulation_9plus3"
    )
    (tmp_path / "manifest.json").write_bytes(
        resource.joinpath("manifest.json").read_bytes()
    )
    (tmp_path / "inputs_101.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        simulation.load_simulation_inputs(
            {"simulation": {"data_dir": str(tmp_path)}}, 101
        )


def test_draft_simulation_trains_reloads_exports_without_opening_truth(
    tmp_path: Path, monkeypatch
):
    original = simulation._bundle_bytes

    def training_only(config, name):
        assert not name.startswith("truth_")
        return original(config, name)

    monkeypatch.setattr(simulation, "_bundle_bytes", training_only)
    config = {
        "simulation": {
            "data_seeds": [101],
            "model_seeds": [0],
            "output_dir": str(tmp_path / "run"),
            "evaluate": False,
            "save_figures": False,
            "device": "cpu",
            "threads": 1,
        },
        "training": {"num_epochs": 2, "log_every": 1},
        "model": {
            "hidden_channels": 4,
            "out_channels": 4,
            "gat_heads": 1,
            "dropout_rate": 0,
        },
    }
    result = simulation.run_simulation(config)
    output = Path(result["runs"][0]["output_dir"])
    report = json.loads((output / "simulation_report.json").read_text(encoding="utf-8"))
    assert report["method"] == "DRAFT" and report["truth_used_in_training"] is False
    assert report["training"]["selected_epoch"] == 2
    assert report["training"]["shared_projections"] == ["res", "gru_skip", "fc"]
    assert report["training"]["negative_generator_seed"] == 9031
    history = pd.read_csv(output / "training_history.tsv", sep="\t")
    assert history.learning_rate.tolist() == [0.001, 0.001]
    assert history.gradient_clip_max_norm.tolist() == [1, 1]
    assert report["explicit_smoothness_weight"] == 0
    assert report["evaluation"] is None and report["truth_provenance"] is None
    assert not (output / "per_stage_metrics.tsv").exists()
    assert len(list((output / "network").glob("T*_TF_TG.tsv"))) == 5
    assert len(pd.read_csv(output / "network/T0_TF_TG.tsv", sep="\t")) == 1800
    with np.load(output / "relations.npz", allow_pickle=False) as scores:
        logits = scores["logits"]
        assert logits.dtype == np.float64
        assert logits.shape == (5, 150, 50)
        tf = np.loadtxt(output / "embeddings/embedding_T0_TF.txt")
        tg = np.loadtxt(output / "embeddings/embedding_T0_TG.txt")
        np.testing.assert_allclose(logits[0], tg @ tf.T, atol=1e-5)
        flat = scores["probabilities"][:, scores["candidate"]]
        expected = np.linalg.norm(flat[0].astype(float) - flat[1].astype(float))
    distances = pd.read_csv(output / "candidate_distances.tsv", sep="\t", index_col=0)
    assert distances.iloc[0, 1] == pytest.approx(expected)
    with pytest.raises(FileExistsError, match="choose a new output_dir"):
        simulation.run_simulation(config)


def test_simulation_evaluation_covers_all_candidates_without_sampling():
    config = simulation._effective_config({})
    inputs, _ = simulation.load_simulation_inputs(config, 101)
    truth, _ = simulation._bundle_array(config, 101, "truth")
    logits = truth["truth"].astype(float) * 2 - 1
    stage, provenance = simulation.evaluate_simulation(config, 101, inputs, logits)
    assert stage.auroc.tolist() == [1] * 5
    assert stage.positive_pairs.tolist() == stage.negative_pairs.tolist() == [900] * 5
    assert provenance["file"] == "truth_101.npz"


def test_main_schedule_decays_from_epoch_560_to_800():
    from draft_paired.simulation_training import learning_rate

    settings = simulation._effective_config({})["training"]
    rates = [learning_rate(i, settings) for i in range(1, 801)]
    assert rates[:560] == [0.001] * 560
    assert rates[560] == pytest.approx(0.001 * (1 - 0.9 / 240))
    assert rates[-1] == pytest.approx(0.0001)
    assert np.all(np.diff(rates[559:]) < 0)


def test_main_model_preserves_cross_type_projection_sharing():
    from draft_paired.simulation_training import MainSimulationGRN

    model = MainSimulationGRN(in_channels=4, hidden_channels=2, out_channels=2)
    for name in ("res", "gru_skip", "fc"):
        assert getattr(model, f"{name}_tf") is getattr(model, f"{name}_re")
        assert getattr(model, f"{name}_tf") is getattr(model, f"{name}_tg")
    assert model.gru_tf is not model.gru_re
    assert not any(name.endswith("table") for name, _ in model.named_buffers())


def test_main_sampler_uses_independent_uniform_negative_stream(monkeypatch):
    import torch.nn.functional as functional

    from draft_paired.simulation_training import reconstruction

    observed = []

    def capture(logits, labels):
        observed.append((logits.detach().clone(), labels.detach().clone()))
        return logits.sum()

    monkeypatch.setattr(functional, "binary_cross_entropy_with_logits", capture)
    a = torch.arange(1.0, 21.0).reshape(10, 2)
    b = torch.arange(1.0, 11.0).reshape(5, 2)
    edges = torch.tensor([[0, 2, 5], [1, 3, 4]])
    generator = torch.Generator().manual_seed(9031)
    expected_generator = torch.Generator().manual_seed(9031)
    labels = torch.zeros((10, 5))
    labels[edges[0], edges[1]] = 1
    pos = torch.nonzero(labels.flatten()).flatten()
    neg = torch.nonzero(labels.flatten() == 0).flatten()
    expected = torch.cat(
        [pos, neg[torch.randperm(len(neg), generator=expected_generator)[: len(pos)]]]
    )
    before = torch.get_rng_state().clone()
    reconstruction(a, b, edges, generator)
    torch.testing.assert_close(torch.get_rng_state(), before)
    torch.testing.assert_close(observed[0][0], (a @ b.T).flatten()[expected])
    assert observed[0][1].tolist() == [1, 1, 1, 0, 0, 0]
