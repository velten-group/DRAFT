"""Build the sdist and wheel, check them with twine, install the wheel in a fresh environment and run smoke tests."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command: list[str], *, cwd: Path | None = None) -> None:
    pretty = " ".join(str(part) for part in command)
    print(f"[RUN] {pretty}")
    subprocess.run(command, cwd=str(cwd) if cwd is not None else None, check=True)


def venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_command(venv_dir: Path, name: str) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def build_artifacts(repo_root: Path, artifact_dir: Path) -> list[Path]:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "build", "twine"])
    run([sys.executable, "-m", "build", "--outdir", str(artifact_dir)], cwd=repo_root)
    artifacts = sorted(path for path in artifact_dir.iterdir() if path.is_file())
    if not artifacts:
        raise RuntimeError(f"No build artifacts were produced in {artifact_dir}.")
    run([sys.executable, "-m", "twine", "check", *[str(path) for path in artifacts]])
    return artifacts


def install_and_smoke(artifact_dir: Path, work_dir: Path) -> None:
    venv_dir = work_dir / "venv"
    run([sys.executable, "-m", "venv", str(venv_dir)])

    python_path = venv_python(venv_dir)
    paired_cli = venv_command(venv_dir, "draft_paired")
    unpaired_cli = venv_command(venv_dir, "draft_unpaired")

    run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])

    wheel_candidates = sorted(artifact_dir.glob("*.whl"))
    if not wheel_candidates:
        raise RuntimeError(f"No wheel artifact found in {artifact_dir}.")
    wheel_path = wheel_candidates[0]
    run([str(python_path), "-m", "pip", "install", str(wheel_path)])

    smoke_dir = work_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    run([str(paired_cli), "--help"])
    run([str(unpaired_cli), "--help"])
    run([str(paired_cli), "init-config", "--output", str(smoke_dir / "draft_lineage_template.yaml")])
    run([str(paired_cli), "init-config", "--kind", "ccc", "--output", str(smoke_dir / "draft_ccc_template.yaml")])
    run([str(paired_cli), "init-config", "--kind", "simulation", "--output", str(smoke_dir / "draft_simulation_template.yaml")])
    run([str(paired_cli), "init-config", "--kind", "simulation-prior", "--output", str(smoke_dir / "draft_simulation_prior_template.yaml")])
    run([str(paired_cli), "init-config", "--example", "fly_germ", "--lineage", "lineage2", "--output", str(smoke_dir / "fly_germ.yaml")])
    run([str(unpaired_cli), "init-config", "--example", "mouse_development", "--lineage", "soma", "--output", str(smoke_dir / "mouse.yaml")])
    run([str(python_path), "-c", "from importlib import resources; r=resources.files('draft_paired'); assert r.joinpath('R/simulate_prior.R').read_bytes(); assert r.joinpath('R/positive_kinetics.R').read_bytes(); assert r.joinpath('R/prior_trial_context.R').read_bytes(); assert r.joinpath('R/check_simulator.R').read_bytes(); assert r.joinpath('examples/simulation_9plus3/prior.tsv').read_bytes()"])
    run([str(python_path), "-c", "from importlib import resources; import json, hashlib; root=resources.files('draft_paired').joinpath('examples', 'simulation_9plus3'); manifest=json.loads(root.joinpath('manifest.json').read_text(encoding='utf-8')); assert len(manifest['seeds']) == 3; assert all(hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() == sha for seed in manifest['seeds'].values() for name, sha in seed['packaged_files'].items())"])
    run([str(unpaired_cli), "init-config", "--output", str(smoke_dir / "draft_unpaired_template.yaml")])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local release validation chain: build, twine check, clean venv install, and CLI smoke."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of this script.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Optional parent directory for temporary release-check runs.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    base_work_root = (args.work_root.resolve() if args.work_root is not None else repo_root / ".release-check")
    base_work_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(base_work_root)))
    artifact_dir = run_dir / "dist"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] repo_root={repo_root}")
    print(f"[INFO] work_dir={run_dir}")

    artifacts = build_artifacts(repo_root, artifact_dir)
    install_and_smoke(artifact_dir, run_dir)

    print("[DONE] Release check completed successfully.")
    print("[INFO] Artifacts:")
    for artifact in artifacts:
        print(f"  - {artifact}")
    print(f"[INFO] Smoke output directory: {run_dir / 'smoke'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
