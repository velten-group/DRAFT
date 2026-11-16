import json
import hashlib
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from draft_paired import simulation
from draft_paired import simulation_generate as generation


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_bundled_design_reproduces_supplied_truth_and_prior_arrays(seed, tmp_path):
    """Verify module memberships, sampled order, rewiring and RE grouping."""
    settings = {
        "design": "main_9plus3",
        "stages": 5,
        "active_fraction": 0.5,
        "change_fraction": 0.1,
        "tfs_per_region": 2,
    }
    data = resources.files("draft_paired").joinpath("examples/simulation_9plus3")
    with (
        data.joinpath(f"inputs_{seed}.npz").open("rb") as stream,
        np.load(stream) as archived,
    ):
        candidate = archived["candidate"]
        tr, rt = archived["tf_re"], archived["re_tg"]
    with (
        data.joinpath(f"truth_{seed}.npz").open("rb") as stream,
        np.load(stream) as archived,
    ):
        truth = archived["truth"]
    actual, a_tr, a_rt, events = generation.design_from_prior(candidate, seed, settings)
    np.testing.assert_array_equal(actual, truth)
    np.testing.assert_array_equal(a_tr, tr)
    np.testing.assert_array_equal(a_rt, rt)
    assert len(events) == 360
    if seed == 101:
        generation._write_design(tmp_path / "main", candidate, seed, settings)
        # Expected hashes of source-archive members.
        expected = {
            "candidate_prior.csv": "b8dd1e6d804983ba5a2e3a474b581f30776024b50b42bb15dc73a7983ddee9e5",
            "gene_modules.tsv": "de99a87484f48d3b4fdf4d92dcd9577203cc0e0831d9da52cb25d1f82c95ee76",
            "generator_only/potential_distance.csv": "49555ae832461fa5de14a13ff93c0bbd0847a614f6ec86886b7c100062f3b11b",
        }
        for name, checksum in expected.items():
            assert (
                hashlib.sha256((tmp_path / "main" / name).read_bytes()).hexdigest()
                == checksum
            )


def test_main_prior_restores_axes_and_rejects_wrong_seed():
    config = {
        "simulation": {"prior_grn": "bundled:main_9plus3"},
        "generator": {"design": "auto"},
    }
    candidate, _ = generation.read_prior(config)
    canonical, design = generation._generation_design(config, candidate)
    assert design == "main_9plus3"
    assert canonical.columns.tolist() == [str(i) for i in range(1, 51)]
    assert canonical.index.tolist() == [str(i) for i in range(51, 201)]
    settings = {
        "design": design,
        "stages": 5,
        "active_fraction": 0.5,
        "change_fraction": 0.1,
        "tfs_per_region": 2,
    }
    generation.design_from_prior(canonical.to_numpy(), 101, settings)
    with pytest.raises(ValueError, match="does not match.*seed 102"):
        generation.design_from_prior(canonical.to_numpy(), 102, settings)
    shuffled = candidate.sample(frac=1, random_state=0).sample(
        frac=1, axis=1, random_state=0
    )
    pd.testing.assert_frame_equal(
        generation._generation_design(config, shuffled)[0], canonical
    )


def test_main_preset_cannot_silently_accept_uniform_generation_settings():
    with pytest.raises(ValueError, match="main_9plus3 requires"):
        generation._main_9plus3_design(
            101, {"stages": 3, "active_fraction": 0.5, "tfs_per_region": 2}
        )
    with pytest.raises(ValueError, match="generator.design"):
        generation._generation_design({"generator": {"design": "typo"}}, pd.DataFrame())


def prior_config(tmp_path: Path):
    path = tmp_path / "prior.tsv"
    pd.DataFrame(
        [(f"TF{j}", f"g{i}") for i in range(3) for j in range(4)], columns=["TF", "TG"]
    ).to_csv(path, sep="\t", index=False)
    return simulation._effective_config(
        {
            "simulation": {
                "prior_grn": str(path),
                "data_seeds": [101],
                "model_seeds": [0],
                "output_dir": str(tmp_path / "training"),
                "threads": 1,
                "save_figures": False,
            },
            "generator": {
                "output_dir": str(tmp_path / "generated"),
                "stages": 3,
                "cells_per_stage": 4,
            },
            "training": {"num_epochs": 2, "log_every": 1},
            "model": {
                "hidden_channels": 2,
                "out_channels": 2,
                "gat_heads": 1,
                "dropout_rate": 0,
            },
        }
    )


@pytest.fixture
def fake_native_r(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        folder = Path(command[3])
        genes, cells, stages = int(command[6]), int(command[7]), int(command[8])
        prior = pd.read_csv(folder / "multiome/tf_re_prior.csv", index_col=0)
        names = [f"cell{i}" for i in range(cells)]
        for t in range(stages):
            x = (np.arange(genes * cells).reshape(genes, cells) + t) % 5
            a = (np.arange(len(prior) * cells).reshape(len(prior), cells) + t) % 3
            pd.DataFrame(
                x, index=[str(i + 1) for i in range(genes)], columns=names
            ).to_csv(folder / f"counts_time_{t}.csv")
            pd.DataFrame(a, index=prior.index, columns=names).to_csv(
                folder / "multiome" / f"atac_time_{t}.csv"
            )
        pd.DataFrame(
            {"package": ["scMultiSim", "R"], "version": ["1.2.0", "test-stub"]}
        ).to_csv(folder / "generator_only/versions.tsv", sep="\t", index=False)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(generation, "find_rscript", lambda config: "fake-Rscript")
    monkeypatch.setattr(generation.subprocess, "run", run)
    return calls


def test_prior_edge_lists_and_matrices_keep_gene_identity(tmp_path: Path):
    path = tmp_path / "edges.csv"
    pd.DataFrame(
        {
            "source": ["NA", "TF2", "NA", "TF2", "TF2"],
            "target": ["g1", "g1", "g2", "g2", "g3"],
            "score": [1, 0, 2, 1, 1],
        }
    ).to_csv(path, index=False)
    config = {"simulation": {"prior_grn": str(path)}}
    matrix, meta = generation.read_prior(config)
    assert matrix.columns.tolist() == ["NA", "TF2"]
    assert matrix.index.tolist() == ["g1", "g2", "g3"]
    assert matrix.to_numpy().sum() == 4 and meta["sha256"]
    second = tmp_path / "matrix.csv"
    matrix.astype(int).to_csv(second)
    actual, _ = generation.read_prior({"simulation": {"prior_grn": str(second)}})
    pd.testing.assert_frame_equal(actual, matrix)


@pytest.mark.parametrize(
    "data,error",
    [
        ({"TF": ["a", "b"], "TG": ["b", "c"]}, "disjoint"),
        ({"TF": ["a", "b"], "TG": ["x", "y"], "weight": [-1, 2]}, "nonnegative"),
        ({"TF": ["a", ""], "TG": ["x", "y"]}, "valid"),
    ],
)
def test_invalid_prior_contracts_fail_before_generation(tmp_path: Path, data, error):
    path = tmp_path / "prior.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        generation.read_prior({"simulation": {"prior_grn": str(path)}})


def test_prior_subsampling_has_exact_degrees_and_one_way_turnover(tmp_path: Path):
    cfg = prior_config(tmp_path)
    candidate = np.ones((20, 8), bool)
    settings = {**cfg["generator"], "stages": 5, "change_fraction": 0.1}
    truth, tr, rt, events = generation.design_from_prior(candidate, 101, settings)
    assert np.all(truth.sum(2) == 4)
    assert np.all((truth[1:] & ~truth[:-1]).sum((1, 2)) == 8)
    assert events.groupby("time").size().tolist() == [8] * 4
    assert not np.any(truth & ~candidate)
    assert np.max(np.sum(truth[1:] != truth[:-1], axis=0)) <= 1
    np.testing.assert_array_equal(rt.astype(int) @ tr.astype(int) > 0, candidate)
    again = generation.design_from_prior(candidate, 101, settings)
    np.testing.assert_array_equal(again[0], truth)
    static = generation.design_from_prior(
        candidate, 101, {**settings, "active_fraction": 1, "change_fraction": 0}
    )
    assert static[0].all() and static[3].empty
    with pytest.raises(ValueError, match="exceeds"):
        generation.design_from_prior(candidate, 101, {**settings, "change_fraction": 1})


def test_generation_cache_reuses_valid_outputs_and_rejects_tampering(
    tmp_path: Path, fake_native_r
):
    cfg = prior_config(tmp_path)
    root = generation.generate_from_prior(cfg)
    assert len(fake_native_r) == 1
    generation.generate_from_prior(cfg)
    assert len(fake_native_r) == 1
    cfg["simulation"]["data_dir"] = str(root)
    data, provenance = simulation.load_simulation_inputs(cfg, 101)
    assert data["rna_counts"].shape == (3, 7, 4)
    assert data["atac_counts"].shape == (3, 6, 4)
    assert data["tf_ids"].tolist() == ["TF0", "TF1", "TF2", "TF3"]
    assert provenance["generator"] == "scMultiSim 1.2.0"
    with (root / "inputs_101.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        generation.generate_from_prior(cfg)


def test_generation_does_not_overwrite_another_prior_or_unmarked_directory(
    tmp_path: Path, fake_native_r
):
    cfg = prior_config(tmp_path)
    root = generation.generate_from_prior(cfg)
    cfg["generator"]["active_fraction"] = 0.6
    with pytest.raises(ValueError, match="different prior/settings/code"):
        generation.generate_from_prior(cfg)
    cfg["generator"]["output_dir"] = str(tmp_path)
    with pytest.raises(FileExistsError, match="unmarked"):
        generation.generate_from_prior(cfg)
    assert (root / "manifest.json").exists()


def test_prior_to_training_supports_variable_shapes_and_separate_truth(
    tmp_path: Path, fake_native_r, monkeypatch
):
    cfg = prior_config(tmp_path)
    original = simulation._bundle_array

    def trace(config, seed, kind):
        if kind == "truth":
            assert (
                Path(cfg["simulation"]["output_dir"])
                / "seed_101/model_0/model/final_epoch_0002.pt"
            ).exists()
        return original(config, seed, kind)

    monkeypatch.setattr(simulation, "_bundle_array", trace)
    result = simulation.run_simulation(cfg)
    output = Path(result["runs"][0]["output_dir"])
    assert len(fake_native_r) == 1
    assert len(list((output / "network").glob("T*_TF_TG.tsv"))) == 3
    metrics = pd.read_csv(output / "per_stage_metrics.tsv", sep="\t")
    assert metrics.candidate_pairs.tolist() == [12] * 3
    report = json.loads((output / "simulation_report.json").read_text(encoding="utf-8"))
    assert report["truth_used_in_training"] is False
    assert report["training"]["selected_epoch"] == 2


def test_native_failure_is_reported_without_falling_back_to_bundled_data(
    tmp_path: Path, monkeypatch
):
    cfg = prior_config(tmp_path)
    monkeypatch.setattr(generation, "find_rscript", lambda config: "fake-Rscript")

    def fail(command, **kwargs):
        kwargs["stdout"].write("there is no package called scMultiSim\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(generation.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="scMultiSim generation failed"):
        generation.generate_from_prior(cfg)
    assert not (Path(cfg["generator"]["output_dir"]) / "manifest.json").exists()


def test_prior_only_cli_supplies_the_complete_generation_workflow(
    tmp_path: Path, monkeypatch
):
    from draft_paired.cli import main

    captured = []

    def run(config):
        captured.append(simulation._effective_config(config))
        return {"output_dir": str(tmp_path)}

    monkeypatch.setattr(simulation, "run_simulation", run)
    assert (
        main(
            [
                "simulation",
                "--prior-grn",
                "prior.tsv",
                "--output",
                str(tmp_path),
                "--rscript",
                "Rscript",
                "--r-library",
                str(tmp_path / "R"),
            ]
        )
        == 0
    )
    cfg = captured[0]
    assert cfg["training"]["num_epochs"] == 600
    assert cfg["simulation"]["prior_grn"] == "prior.tsv"
    assert cfg["simulation"]["data_seeds"] == [101]
    assert cfg["generator"]["output_dir"] == str(tmp_path / "generated_data")


@pytest.mark.parametrize(
    "key,value",
    [("stages", 2.5), ("observation_depth", float("nan")), ("cif_sigma", 0)],
)
def test_invalid_generation_settings_fail_before_running_r(
    tmp_path: Path, monkeypatch, key, value
):
    cfg = prior_config(tmp_path)
    cfg["generator"][key] = value
    monkeypatch.setattr(
        generation,
        "find_rscript",
        lambda config: pytest.fail("R must not run for invalid settings"),
    )
    with pytest.raises(ValueError):
        generation.generate_from_prior(cfg)


def test_prior_config_generation_provides_only_network_input_and_600_epochs(
    tmp_path: Path,
):
    import yaml

    from draft_paired.cli import main

    path = tmp_path / "prior.yaml"
    assert (
        main(["init-config", "--kind", "simulation-prior", "--output", str(path)]) == 0
    )
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["simulation"]["prior_grn"] == "bundled:main_9plus3"
    assert cfg["generator"]["cells_per_stage"] == 200
    assert cfg["training"]["num_epochs"] == 600


def test_single_class_generated_truth_is_reported_without_inventing_auroc(
    tmp_path: Path, fake_native_r
):
    cfg = prior_config(tmp_path)
    cfg["generator"].update(active_fraction=1, change_fraction=0)
    root = generation.generate_from_prior(cfg)
    cfg["simulation"]["data_dir"] = str(root)
    inputs, _ = simulation.load_simulation_inputs(cfg, 101)
    metrics, _ = simulation.evaluate_simulation(cfg, 101, inputs, np.ones((3, 3, 4)))
    assert metrics.auroc.isna().all()
    assert metrics.status.tolist() == ["one_class_reference"] * 3
