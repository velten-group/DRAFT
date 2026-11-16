from __future__ import annotations

from pathlib import Path


def test_imports() -> None:
    import draft_paired  # noqa: F401
    import draft_unpaired  # noqa: F401


def test_paired_parser_includes_init_config() -> None:
    from draft_paired.cli import build_parser

    help_text = build_parser().format_help()
    assert "init-config" in help_text
    assert "pipeline" in help_text


def test_unpaired_parser_includes_init_config() -> None:
    from draft_unpaired.cli import build_parser

    help_text = build_parser().format_help()
    assert "init-config" in help_text
    assert "pipeline" in help_text


def test_paired_init_config_writes_lineage_template(tmp_path: Path) -> None:
    from draft_paired.cli import main

    output_path = tmp_path / "paired.yaml"
    exit_code = main(["init-config", "--output", str(output_path)])

    assert exit_code == 0
    text = output_path.read_text(encoding="utf-8")
    assert "lineage:" in text
    assert "network:" in text
    assert "tf_symbols: []" in text
    assert "lineage: lineage1" in text


def test_paired_init_config_writes_ccc_template(tmp_path: Path) -> None:
    from draft_paired.cli import main

    output_path = tmp_path / "ccc.yaml"
    exit_code = main(["init-config", "--kind", "ccc", "--output", str(output_path)])

    assert exit_code == 0
    text = output_path.read_text(encoding="utf-8")
    assert "ccc_result_root:" in text
    assert "alignment_method:" in text


def test_unpaired_init_config_writes_template(tmp_path: Path) -> None:
    from draft_unpaired.cli import main

    output_path = tmp_path / "unpaired.yaml"
    exit_code = main(["init-config", "--output", str(output_path)])

    assert exit_code == 0
    text = output_path.read_text(encoding="utf-8")
    assert "lineage: lineage1" in text
    assert "rna_h5ad:" in text and "atac_h5ad:" in text
    assert "tf_re_prior:" in text and "re_tg_prior:" in text
    assert "enabled: true" in text
    assert "tf_symbols: []" in text


def test_init_config_defaults_to_configs_directory(tmp_path: Path, monkeypatch) -> None:
    from draft_paired.cli import main as paired_main
    from draft_unpaired.cli import main as unpaired_main

    monkeypatch.chdir(tmp_path)

    assert paired_main(["init-config"]) == 0
    assert paired_main(["init-config", "--kind", "ccc"]) == 0
    assert unpaired_main(["init-config"]) == 0

    assert (tmp_path / "configs" / "draft_lineage_template.yaml").exists()
    assert (tmp_path / "configs" / "draft_ccc_template.yaml").exists()
    assert (tmp_path / "configs" / "draft_unpaired_template.yaml").exists()


def test_init_config_is_idempotent_for_same_template(tmp_path: Path, monkeypatch) -> None:
    from draft_paired.cli import main as paired_main

    monkeypatch.chdir(tmp_path)

    assert paired_main(["init-config"]) == 0
    assert paired_main(["init-config"]) == 0


def test_init_config_ignores_trailing_newline_differences(tmp_path: Path, monkeypatch) -> None:
    from draft_unpaired.cli import main as unpaired_main

    monkeypatch.chdir(tmp_path)

    assert unpaired_main(["init-config"]) == 0
    config_path = tmp_path / "configs" / "draft_unpaired_template.yaml"
    text = config_path.read_text(encoding="utf-8").rstrip("\n")
    config_path.write_text(text, encoding="utf-8")

    assert unpaired_main(["init-config"]) == 0
