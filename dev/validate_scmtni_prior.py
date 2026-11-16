"""Compare prior scores with pinned upstream Python, R and C++ programs.

Read the supplied upstream source directory and write reference results as JSON.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from draft_unpaired.prior import prepare_stage_prior
from draft_unpaired.scmtni_prior import REFERENCE_COMMIT, source_scores


def rows(path):
    return [line.split("\t") for line in path.read_text().splitlines() if line]


def edges(path):
    return [[tf, tg, float(score)] for tf, tg, score in rows(path)]


def run_script(path, args):
    with (
        patch.object(sys, "argv", [str(path), *map(str, args)]),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        runpy.run_path(str(path), run_name="__main__")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--rank-exe", type=Path, required=True)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "scmtni_cases", repo / "tests/scmtni_cases.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = module.write_inputs(out / "inputs")
    root = Path(cfg["base_dir"])
    reference = args.reference_dir.resolve()
    source_files = [
        "mapMot2Gene.py",
        "filterpriornetwork.py",
        "filtertop20Pedges.R",
        "rankEdges_Source/Framework.C",
        "rankEdges_Source/Framework.H",
    ]
    hashes = {
        name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
        for name in source_files
    }
    known = repo / "tests/fixtures/scmtni_reference.json"
    if known.exists():
        expected = json.loads(known.read_text())["source_sha256"]
        if hashes != expected:
            raise ValueError(
                "Upstream sources differ from the pinned, audited version."
            )
    rdata, input_dir = out / "rdata", out / "filtered"
    rdata.mkdir(exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    (out / "filtered_top0.2").mkdir(exist_ok=True)
    names = ["early", "late"]
    for node in names:
        peaks = rows(root / f"{node}.narrowPeak")
        # Independent Cartesian BED overlap oracle for this tiny fixture.
        for bed, label in (("motifs.bed", "motifs"), ("promoters.bed", "genes")):
            overlaps = []
            for peak in peaks:
                for feature in rows(root / bed):
                    if peak[0] == feature[0] and max(
                        int(peak[1]), int(feature[1])
                    ) < min(int(peak[2]), int(feature[2])):
                        overlaps.append("\t".join(peak + feature))
            (out / f"{node}_{label}.tsv").write_text("\n".join(overlaps) + "\n")
        mapped = out / f"{node}_raw.tsv"
        run_script(
            reference / "mapMot2Gene.py",
            [
                "--mot2tf",
                root / "motif_to_tf.tsv",
                "--mot2peak",
                out / f"{node}_motifs.tsv",
                "--peak2gene",
                out / f"{node}_genes.tsv",
                "--outfile",
                mapped,
            ],
        )
        for kind in ("TF", "TG"):
            table = pd.read_csv(root / f"{kind}.tsv", sep="\t")
            (out / f"{kind}.txt").write_text("\n".join(table.gene_id) + "\n")
        run_script(
            reference / "filterpriornetwork.py",
            [
                "--regfile",
                out / "TF.txt",
                "--genefile",
                out / "TG.txt",
                "--netfile",
                mapped,
                "--outfile",
                input_dir / f"{node}_network.txt",
            ],
        )
    cases = [
        [["single", "G", 2.1234567]],
        [["A", "G1", 5], ["A", "G2", 5], ["B", "G1", 5], ["B", "G2", -3]],
        [[f"TF{i}", f"G{j}", float(j)] for i in range(10) for j in range(20)],
        [[f"TF{i}", f"G{j}", float(j // 2)] for i in range(10) for j in range(20)],
        [[f"TF{i}", f"G{j}", -float(j)] for i in range(2) for j in range(10)],
    ]
    for i, case in enumerate(cases):
        name = f"case{i}"
        names.append(name)
        (input_dir / f"{name}_network.txt").write_text(
            "".join(f"{tf}\t{tg}\t{score:.6f}\n" for tf, tg, score in case)
        )
    (rdata / "celltype_order.txt").write_text("\n".join(names) + "\n")
    subprocess.run(
        [
            args.rscript,
            "--vanilla",
            str(reference / "filtertop20Pedges.R"),
            str(rdata).replace("\\", "/") + "/",
            str(input_dir).replace("\\", "/") + "/",
        ],
        check=True,
        stdout=(out / "R.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    ranked = {}
    for name in names:
        path = out / "filtered_top0.2" / f"{name}_network.txt"
        dest = out / f"{name}_percentile.tsv"
        subprocess.run(
            [str(args.rank_exe.resolve()), str(path), str(dest), "incr"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        ranked[name] = edges(dest)
    scoring = []
    for i, case in enumerate(cases):
        expected_top = edges(out / "filtered_top0.2" / f"case{i}_network.txt")
        network = {(tf, tg): value for tf, tg, value in case}
        _, _, actual_top, actual_ranked, _ = source_scores(
            network, {tf for tf, _ in network}, {tg for _, tg in network}
        )
        assert [list(row) for row in actual_top] == expected_top
        assert [list(row) for row in actual_ranked] == ranked[f"case{i}"]
        scoring.append(
            {"input": case, "top": expected_top, "ranked": ranked[f"case{i}"]}
        )
    prepared = prepare_stage_prior(cfg, "prepared")
    for i, node in enumerate(("early", "late")):
        folder = root / "prepared/networks" / f"stage_{i:03d}"
        assert edges(folder / "motif_max.tsv") == edges(out / f"{node}_raw.tsv")
        assert edges(folder / "filtered.tsv") == edges(
            input_dir / f"{node}_network.txt"
        )
        assert edges(folder / "top_density.tsv") == edges(
            out / "filtered_top0.2" / f"{node}_network.txt"
        )
        assert edges(folder / "percentile.tsv") == ranked[node]
    staged = [[node, *row] for node in ("early", "late") for row in ranked[node]]
    assert (
        pd.read_csv(root / "prepared/tf_tg_prior.tsv", sep="\t").values.tolist()
        == staged
    )
    result = {
        "reference_commit": REFERENCE_COMMIT,
        "source_sha256": hashes,
        "scoring_cases": scoring,
        "staged_prior": staged,
        "validation": "actual upstream Python mapping/filtering, R density filter, compiled C++ percentile program; all intermediate tables equal",
        "prior_report": prepared["config_path"],
    }
    (out / "scmtni_reference.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["validation"])


if __name__ == "__main__":
    main()
