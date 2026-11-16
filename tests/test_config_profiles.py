from pathlib import Path

import yaml


def test_paired_methods_profile_sets_export_settings_and_example_defaults(
    tmp_path: Path,
):
    from draft_paired.cli import main

    path = tmp_path / "germ.yaml"
    assert (
        main(
            [
                "init-config",
                "--profile",
                "methods",
                "--example",
                "fly_germ",
                "--lineage",
                "germ",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["lineage"] == "germ"
    assert cfg["training"]["lr"] == 5e-4
    assert cfg["training"]["checkpoint_selection"] == "last"
    assert cfg["network"]["active_frac_tf_tg"] == 0.05
    assert cfg["network"]["dense"]["active_thr"] == 0.05
    assert cfg["topic"]["top_n_per_cluster"] == 10000
    assert cfg["topic"]["enrich_db"] == "GO_Biological_Process_2018"
    assert cfg["method_profile"]["name"] == "methods"
    assert (
        main(
            [
                "init-config",
                "--profile",
                "methods",
                "--example",
                "fly_germ",
                "--lineage",
                "germ",
                "--output",
                str(path),
            ]
        )
        == 0
    )


def test_generic_profile_leaves_enrichment_resources_for_the_user(tmp_path: Path):
    from draft_paired.config_init import init_config
    from draft_unpaired.config_init import init_config as unpaired_config

    for family, configure in (("paired", init_config), ("unpaired", unpaired_config)):
        path = tmp_path / f"{family}.yaml"
        configure(profile="methods", lineage="my_lineage", output=path)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["method_profile"] == {"name": "methods"}
        assert cfg["topic"]["enrich_db"] is None
        assert cfg["topic"]["enrich_organism"] is None


def test_unpaired_profile_and_template_use_thirty_percent_and_50000_edges(
    tmp_path: Path,
):
    from draft_unpaired.config_init import init_config

    for profile in (None, "methods"):
        path = tmp_path / f"unpaired_{profile}.yaml"
        init_config(profile=profile, example="mouse_development", output=path)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["network"]["active_frac_tf_tg"] == 0.30
        assert cfg["network"]["top_k_global"] == 50000
        assert cfg["network"]["min_score_keep"] == 0.85
        assert cfg["network"]["dense"]["top_k_global"] == 50000
        assert cfg["topic"]["enrich_db"] == "GO_Biological_Process_2021"
