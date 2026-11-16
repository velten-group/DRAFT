"""Execute the downstream-analysis example with an isolated kernel and workspace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def execute_tutorial(notebook: Path, output: Path) -> dict:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    output = output.resolve()
    kernel = output / "kernels" / "draft-tutorial"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "DRAFT tutorial validation",
                "language": "python",
            }
        )
    )
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        DRAFT_TUTORIAL_OUTPUT_DIR=str(output / "work"),
        IPYTHONDIR=str(output / "ipython"),
        JUPYTER_RUNTIME_DIR=str(output / "runtime"),
        MPLCONFIGDIR=str(output / "matplotlib"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env["PYTHONPATH"] = str(repo / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    manager = AsyncKernelManager(
        kernel_name="draft-tutorial",
        connection_file=str(output / "connection.json"),
        kernel_spec_manager=KernelSpecManager(
            kernel_dirs=[str(kernel.parent)], ensure_native_kernel=False
        ),
    )
    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    client = NotebookClient(
        nb, km=manager, timeout=300, resources={"metadata": {"path": str(repo)}}
    )
    try:
        executed = client.execute(env=env, cleanup_kc=True)
    finally:
        nbformat.write(nb, output / notebook.name)
    errors = [
        o
        for c in executed.cells
        for o in c.get("outputs", [])
        if o.output_type == "error"
    ]
    images = sum(
        "image/png" in o.get("data", {})
        for c in executed.cells
        for o in c.get("outputs", [])
    )
    assert not errors and images > 0
    report = {
        "notebook": notebook.name,
        "code_cells": sum(c.cell_type == "code" for c in executed.cells),
        "png_outputs": images,
        "data_source": "synthetic_demo",
        "output_dir": str(output),
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    notebook = repo / "docs/downstream.ipynb"
    print(json.dumps(execute_tutorial(notebook, args.output / notebook.stem), indent=2))
