from pathlib import Path
import importlib.util

import pytest


def test_downstream_example_executes_with_synthetic_data_and_outputs(tmp_path):
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "validate_tutorials", repo / "dev/validate_tutorials.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    filename = "downstream.ipynb"
    result = module.execute_tutorial(repo / "docs" / filename, tmp_path)
    assert result["png_outputs"] > 0
    assert result["data_source"] == "synthetic_demo"
    assert (tmp_path / filename).exists()
