"""Package the bundled simulation example (src/draft_paired/examples/simulation_9plus3).

Reads the main simulation design (datasets linear/main, seeds 101-103) from the DRAFT-analysis deposit:
draft_analysis_data_simulation.tar.gz and its manifest.tsv, which give every file's SHA-256 and size. Stores the
observations and candidate priors for training, with the generating truth in a separate evaluation bundle.

    python dev/package_simulation_example.py draft_analysis_data_simulation.tar.gz manifest.tsv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


def package_example(archive: Path, deposit_manifest: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with deposit_manifest.open(encoding="utf-8") as fh:
        expected = {
            row["path"]: {"sha256": row["sha256"], "size": int(row["bytes"])}
            for row in csv.DictReader(fh, delimiter="\t")
        }
    with tarfile.open(archive) as source:
        manifest = {
            "schema": "draft_simulation_9plus3_v1",
            "training_uses": "observed RNA/ATAC counts and public TF-RE/RE-TG priors only",
            "seeds": {},
        }
        for seed in (101, 102, 103):
            prefix = f"simulation/datasets/linear/main/seed_{seed}/"

            def read(name: str) -> bytes:
                key = prefix + name
                data = source.extractfile(key).read()
                digest = hashlib.sha256(data).hexdigest()
                if (
                    digest != expected[key]["sha256"]
                    or len(data) != expected[key]["size"]
                ):
                    raise ValueError(f"Source checksum mismatch: {key}")
                return data

            def matrix(name: str) -> pd.DataFrame:
                frame = pd.read_csv(io.BytesIO(read(name)), index_col=0)
                frame.index = frame.index.astype(str)
                frame.columns = frame.columns.astype(str)
                return frame

            settings = json.loads(read("settings.json"))
            params = {key: value for key, value in settings.items() if key != "case"}
            assert (params["candidate_inside"], params["candidate_outside"]) == (9, 3)
            assert (
                params["num_tf"],
                params["num_genes"],
                params["times"],
                params["cells"],
            ) == (50, 200, 5, 200)
            candidate = matrix("candidate_prior.csv")
            tf_re = matrix("multiome/tf_re_prior.csv")
            re_tg = matrix("multiome/re_tg_prior.csv")
            tfs, tgs, regions = (
                candidate.columns.tolist(),
                candidate.index.tolist(),
                tf_re.index.tolist(),
            )
            assert (
                tf_re.columns.tolist() == tfs
                and re_tg.index.tolist() == tgs
                and re_tg.columns.tolist() == regions
            )
            genes = tfs + tgs
            rna, atac, cells, truth = [], [], [], []
            excluded = None
            for time in range(5):
                x, a = (
                    matrix(f"counts_time_{time}.csv"),
                    matrix(f"multiome/atac_time_{time}.csv"),
                )
                extra_genes = [g for g in x.index if g not in genes]
                assert all(g.startswith("background_") for g in extra_genes)
                if excluded is None:
                    excluded = extra_genes
                assert extra_genes == excluded
                x = x.loc[genes]
                assert x.index.tolist() == genes and a.index.tolist() == regions
                assert x.columns.tolist() == a.columns.tolist()
                for frame in (x, a):
                    values = frame.to_numpy()
                    assert (
                        np.isfinite(values).all()
                        and (values >= 0).all()
                        and (values == np.floor(values)).all()
                    )
                    assert values.max() <= np.iinfo(np.uint32).max
                rna.append(x.to_numpy(dtype=np.uint32))
                atac.append(a.to_numpy(dtype=np.uint32))
                cells.append(x.columns.to_numpy(dtype=str))
                y = matrix(f"grn_time_{time}.csv").loc[tgs, tfs].to_numpy(bool)
                truth.append(y)
            candidate_array = candidate.to_numpy(bool)
            tr, rt = tf_re.to_numpy(bool), re_tg.to_numpy(bool)
            np.testing.assert_array_equal(
                rt.astype(int) @ tr.astype(int) > 0, candidate_array
            )
            assert np.all(candidate_array.sum(1) == 12) and np.all(
                np.asarray(truth).sum(2) == 6
            )
            assert not np.any(np.asarray(truth) & ~candidate_array)
            # Validate module memberships against the simulation design.
            modules = (
                pd.read_csv(
                    io.BytesIO(read("gene_modules.tsv")), sep="\t", dtype={"gene": str}
                )
                .set_index("gene")
                .module
            )
            same = (
                modules.loc[tgs].to_numpy()[:, None]
                == modules.loc[tfs].to_numpy()[None, :]
            )
            assert np.all((candidate_array & same).sum(1) == 9)
            assert np.all((candidate_array & ~same).sum(1) == 3)
            assert np.all((np.asarray(truth) & same).sum(2) == 6)
            paths = [output / f"inputs_{seed}.npz", output / f"truth_{seed}.npz"]
            np.savez_compressed(
                paths[0],
                rna_counts=np.stack(rna),
                atac_counts=np.stack(atac),
                tf_re=tr,
                re_tg=rt,
                candidate=candidate_array,
                tf_ids=np.asarray(tfs),
                tg_ids=np.asarray(tgs),
                re_ids=np.asarray(regions),
                times=np.arange(5),
                cell_ids=np.stack(cells),
            )
            np.savez_compressed(
                paths[1],
                truth=np.stack(truth),
                tf_ids=np.asarray(tfs),
                tg_ids=np.asarray(tgs),
                times=np.arange(5),
            )
            manifest["seeds"][str(seed)] = {
                "generation": {"params": params},
                "packaged_files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                },
                "dimensions": {
                    "stages": 5,
                    "cells_per_stage": 200,
                    "tf": 50,
                    "tg": 150,
                    "re": len(regions),
                    "candidate_pairs": int(candidate_array.sum()),
                },
            }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, help="draft_analysis_data_simulation.tar.gz")
    parser.add_argument("deposit_manifest", type=Path, help="manifest.tsv of the deposit")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src/draft_paired/examples/simulation_9plus3",
    )
    args = parser.parse_args()
    result = package_example(args.archive, args.deposit_manifest, args.output)
    print(json.dumps({"output": str(args.output), "data_seeds": list(result["seeds"])}))
