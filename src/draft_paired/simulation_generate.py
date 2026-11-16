"""Generate paired observations from a candidate TF-TG prior with scMultiSim."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import emit_progress, resolve_config_path

SCHEMA = "draft_simulation_prior_v1"
R_FILES = (
    "simulate_prior.R",
    "positive_kinetics.R",
    "prior_trial_context.R",
    "check_simulator.R",
)


def read_prior(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read an unsigned edge list or TG-row/TF-column matrix, preserving names."""
    name = str(config["simulation"]["prior_grn"])
    if name == "bundled:main_9plus3":
        content = (
            resources.files("draft_paired")
            .joinpath("examples/simulation_9plus3/prior.tsv")
            .read_bytes()
        )
        separator = "\t"
    else:
        path = resolve_config_path(config, name)
        content = path.read_bytes()
        separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    header = next(
        csv.reader(io.StringIO(content.decode("utf-8-sig")), delimiter=separator)
    )
    stripped = [c.strip() for c in header]
    if len(set(stripped)) != len(stripped):
        raise ValueError("Prior GRN has duplicate column labels.")
    frame = pd.read_csv(
        io.BytesIO(content), sep=separator, dtype=str, keep_default_na=False
    )
    frame.columns = frame.columns.str.strip()
    columns = {c.lower(): c for c in frame.columns}
    source = next(
        (
            columns[k]
            for k in ("tf", "source", "regulator", "regulator.gene")
            if k in columns
        ),
        None,
    )
    target = next(
        (columns[k] for k in ("tg", "target", "regulated.gene") if k in columns), None
    )
    if source is not None and target is not None:
        if frame[[source, target]].isna().any().any():
            raise ValueError("Prior GRN contains missing TF/TG names.")
        edges = frame[[source, target]].copy()
        edges.columns = ["TF", "TG"]
        score = next(
            (
                columns[k]
                for k in ("weight", "score", "regulator.effect")
                if k in columns
            ),
            None,
        )
        if score:
            values = pd.to_numeric(frame[score], errors="raise").to_numpy(float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(
                    "Prior weights must be finite and nonnegative; this generator uses unsigned topology."
                )
            edges = edges.loc[values > 0]
        edges = edges.apply(lambda c: c.str.strip()).drop_duplicates()
        if edges.empty or (edges == "").any().any():
            raise ValueError("Prior GRN contains no valid nonzero candidate edges.")
        tfs, tgs = pd.unique(edges.TF).tolist(), pd.unique(edges.TG).tolist()
        ti, gi = {v: i for i, v in enumerate(tfs)}, {v: i for i, v in enumerate(tgs)}
        array = np.zeros((len(tgs), len(tfs)), dtype=bool)
        array[[gi[v] for v in edges.TG], [ti[v] for v in edges.TF]] = True
        matrix = pd.DataFrame(array, index=tgs, columns=tfs)
    else:
        if frame.shape[1] < 2 or frame.iloc[:, 0].isna().any():
            raise ValueError(
                "Expected TF/TG edge-list columns, or a matrix with TG names in its first column."
            )
        index = frame.iloc[:, 0].str.strip()
        if index.duplicated().any() or (index == "").any():
            raise ValueError("Prior matrix requires distinct nonempty TG row labels.")
        values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Prior matrix entries must be finite and nonnegative.")
        matrix = pd.DataFrame(values > 0, index=index, columns=frame.columns[1:])
        matrix = matrix.loc[matrix.any(axis=1), matrix.any(axis=0)]
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(
            "The paired simulation requires at least two TFs and two TGs with candidate edges."
        )
    if set(matrix.index) & set(matrix.columns):
        raise ValueError(
            "This bipartite TF-TG generator requires disjoint TF and TG identifiers; TF-to-TF/self regulation is not supported."
        )
    matrix.index.name = None
    matrix.columns.name = None
    return matrix, {
        "source": name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "tf": len(matrix.columns),
        "tg": len(matrix),
        "candidate_pairs": int(matrix.to_numpy().sum()),
        "weight_policy": "positive values indicate candidates; generating effects are binary",
    }


def design_from_prior(
    candidate: np.ndarray, seed: int, settings: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Sample truth independently of model training, within the provided candidates."""
    if settings.get("design") == "main_9plus3":
        states, expected, tr, rt, events, _ = _main_9plus3_design(seed, settings)
        if not np.array_equal(candidate, expected):
            raise ValueError(
                f"Prior does not match the original main 9+3 candidate network for data seed {seed}; "
                "use the matching prior/seed, or generator.design: candidate_uniform for a new experiment."
            )
        return states, tr, rt, events
    fraction = float(settings["active_fraction"])
    turnover = float(settings["change_fraction"])
    stages, chunk = int(settings["stages"]), int(settings["tfs_per_region"])
    if not 0 < fraction <= 1 or not 0 <= turnover <= 1 or stages < 2 or chunk < 1:
        raise ValueError(
            "Require 0<active_fraction<=1, 0<=change_fraction<=1, stages>=2 and tfs_per_region>=1."
        )
    candidate = np.asarray(candidate, bool)
    if candidate.ndim != 2 or not candidate.any(axis=1).all():
        raise ValueError("Every simulated TG requires at least one candidate TF.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 811]))
    initial = np.zeros_like(candidate)
    for g, row in enumerate(candidate):
        eligible = np.flatnonzero(row)
        count = (
            len(eligible)
            if fraction == 1
            else min(
                max(1, int(round(len(eligible) * fraction))), max(1, len(eligible) - 1)
            )
        )
        initial[g, rng.choice(eligible, count, replace=False)] = True
    states, events = [initial], []
    changes = round(turnover * int(initial.sum()))
    for t in range(1, stages):
        previous = states[-1]
        old = [np.flatnonzero(previous[g] & initial[g]) for g in range(len(candidate))]
        new = [
            np.flatnonzero(candidate[g] & ~previous[g] & ~initial[g])
            for g in range(len(candidate))
        ]
        slots = np.repeat(
            np.arange(len(candidate)), [min(len(a), len(b)) for a, b in zip(old, new)]
        )
        if changes > len(slots):
            raise ValueError(
                f"Requested one-way turnover exceeds available edges at stage {t}; reduce change_fraction or stages."
            )
        allocation = np.bincount(
            slots[rng.choice(len(slots), changes, replace=False)],
            minlength=len(candidate),
        )
        state = previous.copy()
        for g in np.flatnonzero(allocation):
            remove = rng.choice(old[g], allocation[g], replace=False)
            add = rng.choice(new[g], allocation[g], replace=False)
            state[g, remove], state[g, add] = False, True
            events.extend((t, int(g), int(a), int(b)) for a, b in zip(remove, add))
        states.append(state)
    mapper = np.random.default_rng(np.random.SeedSequence([seed, 913]))
    motif, distance = [], []
    for g, row in enumerate(candidate):
        eligible = mapper.permutation(np.flatnonzero(row))
        for start in range(0, len(eligible), chunk):
            tr, rg = np.zeros(candidate.shape[1], bool), np.zeros(len(candidate), bool)
            tr[eligible[start : start + chunk]], rg[g] = True, True
            motif.append(tr)
            distance.append(rg)
    tr, rt = np.asarray(motif), np.asarray(distance).T
    assert np.array_equal(rt.astype(int) @ tr.astype(int) > 0, candidate)
    return (
        np.stack(states),
        tr,
        rt,
        pd.DataFrame(
            events, columns=["time", "target_index", "old_tf_index", "new_tf_index"]
        ),
    )


def _main_9plus3_design(seed: int, settings: dict[str, Any]):
    """Construct the bundled developmental design with deterministic draws."""
    if (
        int(settings["stages"]) != 5
        or float(settings["active_fraction"]) != 0.5
        or int(settings["tfs_per_region"]) != 2
    ):
        raise ValueError(
            "main_9plus3 requires stages=5, active_fraction=0.5 and tfs_per_region=2."
        )
    ntf, ntg, active, active_in = 50, 150, 6, 6
    rng = np.random.default_rng(seed)
    dynamics = np.random.default_rng(np.random.SeedSequence([seed, 811]))
    tf_module = rng.permutation(np.arange(ntf) % 3)
    tg_module = rng.permutation(np.arange(ntg) % 3)
    same = tg_module[:, None] == tf_module[None, :]
    candidate, initial = np.zeros_like(same), np.zeros_like(same)
    for g in range(ntg):
        for local, count, number in (
            (True, 9, active_in),
            (False, 3, active - active_in),
        ):
            chosen = rng.choice(np.flatnonzero(same[g] == local), count, replace=False)
            candidate[g, chosen] = True
            initial[g, dynamics.choice(chosen, number, replace=False)] = True
    changes = round(float(settings["change_fraction"]) * ntg * active)
    per_stratum = {True: round(changes * active_in / active)}
    per_stratum[False] = changes - per_stratum[True]
    states, events = [initial], []
    for t in range(1, 5):
        previous = states[-1]
        state = previous.copy()
        for local, count in per_stratum.items():
            allowed = candidate & (same == local)
            old = [
                np.flatnonzero(previous[g] & initial[g] & allowed[g])
                for g in range(ntg)
            ]
            new = [
                np.flatnonzero(~previous[g] & ~initial[g] & allowed[g])
                for g in range(ntg)
            ]
            capacity = np.array([min(len(a), len(b)) for a, b in zip(old, new)])
            slots = np.repeat(np.arange(ntg), capacity)
            if count > len(slots):
                raise ValueError("Change rate exceeds one-way capacity in a stratum")
            allocation = np.bincount(
                slots[dynamics.choice(slots.size, count, replace=False)], minlength=ntg
            )
            for g in np.flatnonzero(allocation):
                n = int(allocation[g])
                removed = dynamics.choice(old[g], n, replace=False)
                added = dynamics.choice(new[g], n, replace=False)
                state[g, removed], state[g, added] = False, True
                events.extend(
                    (t, int(g), int(a), int(b)) for a, b in zip(removed, added)
                )
        states.append(state)
    states = np.stack(states)
    if not np.all(states.sum(2) == active) or not np.all(
        (states & same).sum(2) == active_in
    ):
        raise AssertionError("Active degrees/module mixture drifted")
    mapper = np.random.default_rng(np.random.SeedSequence([seed, 913]))
    motif, distance = [], []
    for g in range(ntg):
        chosen = mapper.permutation(np.flatnonzero(candidate[g]))
        for chunk in np.array_split(chosen, len(chosen) // 2):
            tr, rg = np.zeros(ntf, bool), np.zeros(ntg, bool)
            tr[chunk], rg[g] = True, True
            motif.append(tr)
            distance.append(rg)
    return (
        states,
        candidate,
        np.array(motif),
        np.array(distance).T,
        pd.DataFrame(
            events, columns=["time", "target_index", "old_tf_index", "new_tf_index"]
        ),
        pd.DataFrame(
            {
                "gene": [str(i + 1) for i in range(ntf + ntg)],
                "module": np.r_[tf_module, tg_module],
            }
        ),
    )


def _generation_design(config: dict[str, Any], candidate: pd.DataFrame):
    """Select a simulation design and order its TF/TG identifiers."""
    design = str(config["generator"].get("design", "auto"))
    if design == "auto":
        design = (
            "main_9plus3"
            if config["simulation"]["prior_grn"] == "bundled:main_9plus3"
            else "candidate_uniform"
        )
    if design not in {"main_9plus3", "candidate_uniform"}:
        raise ValueError(
            "generator.design must be auto, main_9plus3 or candidate_uniform."
        )
    if design == "main_9plus3":
        tfs, tgs = [str(i) for i in range(1, 51)], [str(i) for i in range(51, 201)]
        if set(candidate.columns) != set(tfs) or set(candidate.index) != set(tgs):
            raise ValueError(
                "main_9plus3 requires original TF IDs 1-50 and TG IDs 51-200."
            )
        candidate = candidate.reindex(index=tgs, columns=tfs)
    return candidate, design


def find_rscript(config: dict[str, Any]) -> str:
    value = config["generator"].get("rscript") or os.environ.get("RSCRIPT")
    if value:
        path = shutil.which(str(value)) or str(Path(value).expanduser())
        if Path(path).is_file():
            return path
        raise FileNotFoundError(f"Rscript executable not found: {value}")
    found = shutil.which("Rscript")
    if found:
        return found
    if sys.platform == "win32":
        directory = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "R"
        candidates = sorted(directory.glob("R-*/bin/Rscript.exe"))
        if candidates:
            return str(candidates[-1])
    raise FileNotFoundError(
        "Rscript is required for prior-GRN generation. Install R and Bioconductor 3.20 scMultiSim 1.2.0, or set generator.rscript / --rscript."
    )


def _write_design(
    folder: Path, candidate: np.ndarray, seed: int, settings: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth, tr, rt, events = design_from_prior(candidate, seed, settings)
    audit, multiome = folder / "generator_only", folder / "multiome"
    audit.mkdir(parents=True, exist_ok=True)
    multiome.mkdir(exist_ok=True)
    ntg, ntf = candidate.shape
    genes = [str(i + 1) for i in range(ntf + ntg)]
    regions = [f"RE_{i + 1}" for i in range(len(tr))]
    for array, index, columns, name in (
        (tr, regions, genes[:ntf], audit / "potential_motif.csv"),
        (rt.T, regions, genes[ntf:], audit / "potential_distance.csv"),
        (tr, regions, genes[:ntf], multiome / "tf_re_prior.csv"),
        (rt, genes[ntf:], regions, multiome / "re_tg_prior.csv"),
        (candidate, genes[ntf:], genes[:ntf], folder / "candidate_prior.csv"),
    ):
        pd.DataFrame(array.astype(int), index=index, columns=columns).to_csv(
            name, lineterminator="\n"
        )
    for t, state in enumerate(truth):
        full = np.zeros((ntf + ntg, ntf + ntg), int)
        full[ntf:, :ntf] = state
        pd.DataFrame(full, index=genes, columns=genes).to_csv(
            folder / f"grn_time_{t}.csv", lineterminator="\n"
        )
    if settings.get("design") == "main_9plus3":
        # Save gene modules alongside the generated observations.
        _, _, _, _, _, modules = _main_9plus3_design(seed, settings)
        modules.to_csv(
            folder / "gene_modules.tsv", sep="\t", index=False, lineterminator="\n"
        )
        activity = []
        for t, state in enumerate(truth):
            pd.DataFrame(
                state.astype(int), index=genes[ntf:], columns=genes[:ntf]
            ).to_csv(audit / f"weights_time_{t}.csv", lineterminator="\n")
            activity.append(
                np.sum((rt.T @ state.astype(float)) * tr, axis=1) / tr.sum(1)
            )
        pd.DataFrame(np.array(activity).T, index=regions).to_csv(
            audit / "region_activity.csv", lineterminator="\n"
        )
    events.to_csv(audit / "transitions.tsv", sep="\t", index=False, lineterminator="\n")
    return truth, tr, rt


def _pack_generated(
    folder: Path,
    root: Path,
    seed: int,
    candidate: pd.DataFrame,
    truth: np.ndarray,
    tr: np.ndarray,
    rt: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    ntg, ntf = candidate.shape
    genes = [str(i + 1) for i in range(ntf + ntg)]
    regions = [f"RE_{i + 1}" for i in range(len(tr))]
    rna, atac, cells = [], [], []
    for t in range(len(truth)):
        x = pd.read_csv(folder / f"counts_time_{t}.csv", index_col=0)
        a = pd.read_csv(folder / "multiome" / f"atac_time_{t}.csv", index_col=0)
        x.index, a.index = x.index.astype(str), a.index.astype(str)
        if x.columns.tolist() != a.columns.tolist() or a.index.tolist() != regions:
            raise ValueError("Generated RNA/ATAC paired axes differ.")
        x = x.loc[genes]
        if (
            x.shape[1] != settings["cells_per_stage"]
            or a.shape[1] != settings["cells_per_stage"]
        ):
            raise ValueError(
                "scMultiSim did not produce the requested equal stage cell counts."
            )
        for frame in (x, a):
            values = frame.to_numpy(float)
            if (
                not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values != np.floor(values))
            ):
                raise ValueError("scMultiSim returned invalid observed counts.")
        rna.append(x.to_numpy(dtype=np.float64))
        atac.append(a.to_numpy(dtype=np.float64))
        cells.append(x.columns.to_numpy(dtype=str))
    inputs = dict(
        rna_counts=np.stack(rna),
        atac_counts=np.stack(atac),
        tf_re=tr,
        re_tg=rt,
        candidate=candidate.to_numpy(bool),
        tf_ids=np.asarray(candidate.columns, str),
        tg_ids=np.asarray(candidate.index, str),
        re_ids=np.asarray(regions),
        times=np.arange(len(truth)),
        cell_ids=np.stack(cells),
    )
    references = dict(
        truth=truth,
        tf_ids=inputs["tf_ids"],
        tg_ids=inputs["tg_ids"],
        times=inputs["times"],
    )
    files = {}
    for kind, arrays in (("inputs", inputs), ("truth", references)):
        final = root / f"{kind}_{seed}.npz"
        temporary = root / f"{kind}_{seed}.pending.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(final)
        files[final.name] = hashlib.sha256(final.read_bytes()).hexdigest()
    versions = (
        pd.read_csv(folder / "generator_only/versions.tsv", sep="\t")
        .set_index("package")
        .version.to_dict()
    )
    return {
        "packaged_files": files,
        "generation": {"params": dict(settings, seed=seed)},
        "versions": versions,
        "dimensions": {
            "stages": len(truth),
            "cells_per_stage": settings["cells_per_stage"],
            "tf": ntf,
            "tg": ntg,
            "re": len(tr),
            "candidate_pairs": int(candidate.to_numpy().sum()),
        },
        "raw_directory": str(folder),
    }


def generate_from_prior(config: dict[str, Any]) -> Path:
    candidate, prior = read_prior(config)
    candidate, design = _generation_design(config, candidate)
    options = config["generator"]
    scientific = {
        k: v
        for k, v in options.items()
        if k not in {"rscript", "r_library", "output_dir"}
    }
    scientific["design"] = design
    # Validate candidate priors before creating outputs or invoking R.
    if design == "main_9plus3":
        for seed in config["simulation"]["data_seeds"]:
            design_from_prior(candidate.to_numpy(bool), seed, scientific)
    for key, minimum in (("stages", 2), ("cells_per_stage", 2), ("tfs_per_region", 1)):
        value = scientific[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value != int(value)
            or value < minimum
        ):
            raise ValueError(f"generator.{key} must be an integer at least {minimum}.")
        scientific[key] = int(value)
    for key in (
        "active_fraction",
        "change_fraction",
        "intrinsic_noise",
        "observation_depth",
        "diff_cif_fraction",
        "atac_effect",
        "cif_sigma",
    ):
        if not np.isfinite(float(scientific[key])):
            raise ValueError(f"generator.{key} must be finite.")
    if (
        not 0 <= scientific["diff_cif_fraction"] <= 1
        or not 0 <= scientific["atac_effect"] <= 1
        or scientific["intrinsic_noise"] < 0
        or scientific["cif_sigma"] <= 0
    ):
        raise ValueError(
            "Invalid scMultiSim noise, CIF fraction or ATAC effect settings."
        )
    if (
        not 0 < scientific["active_fraction"] <= 1
        or not 0 <= scientific["change_fraction"] <= 1
    ):
        raise ValueError("Require 0<active_fraction<=1 and 0<=change_fraction<=1.")
    if (
        int(scientific["cells_per_stage"]) < 2
        or float(scientific["observation_depth"]) <= 0
    ):
        raise ValueError(
            "Generation needs at least two cells per stage and positive observation_depth."
        )
    code = {
        name: resources.files("draft_paired").joinpath(f"R/{name}").read_bytes()
        for name in R_FILES
    }
    identity = {
        "schema": SCHEMA,
        "prior_sha256": prior["sha256"],
        "settings": scientific,
        "python_generator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "r_code": {
            name: hashlib.sha256(value).hexdigest() for name, value in code.items()
        },
    }
    root = resolve_config_path(config, options["output_dir"])
    identity_file, manifest_file = (
        root / "generation_identity.json",
        root / "manifest.json",
    )
    if identity_file.exists():
        if json.loads(identity_file.read_text(encoding="utf-8")) != identity:
            raise ValueError(
                "Generation cache uses a different prior/settings/code; choose a new generator.output_dir."
            )
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite an unmarked generation directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    identity_file.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    manifest = (
        json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest_file.exists()
        else {
            "schema": SCHEMA,
            "prior": prior,
            "generator": "scMultiSim 1.2.0",
            "design": design,
            "seeds": {},
        }
    )
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("prior", {}).get("sha256") != prior["sha256"]
    ):
        raise ValueError("Generation manifest does not match the supplied prior.")
    executable = None
    for seed in config["simulation"]["data_seeds"]:
        saved = manifest["seeds"].get(str(seed))
        if saved is not None:
            for name, digest in saved["packaged_files"].items():
                if (
                    not (root / name).exists()
                    or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
                ):
                    raise ValueError(f"Generated data checksum changed: {root / name}")
            emit_progress("REUSE", "simulation:generate", f"data seed={seed}")
            continue
        executable = executable or find_rscript(config)
        folder = root / f"seed_{seed}"
        truth, tr, rt = _write_design(
            folder, candidate.to_numpy(bool), seed, scientific
        )
        scripts = root / "generator_code"
        scripts.mkdir(exist_ok=True)
        for name, value in code.items():
            (scripts / name).write_bytes(value)
        command = [
            executable,
            "--vanilla",
            str(scripts / "simulate_prior.R"),
            str(folder),
            str(seed),
            str(candidate.shape[1]),
            str(sum(candidate.shape)),
            str(scientific["cells_per_stage"]),
            str(scientific["stages"]),
        ]
        command += [
            str(scientific[k])
            for k in (
                "intrinsic_noise",
                "observation_depth",
                "diff_cif_fraction",
                "atac_effect",
                "cif_sigma",
            )
        ]
        env = os.environ.copy()
        if options.get("r_library"):
            env["R_LIBS"] = str(resolve_config_path(config, options["r_library"])) + (
                os.pathsep + env["R_LIBS"] if env.get("R_LIBS") else ""
            )
        log = folder / "scmultisim.log"
        emit_progress(
            "GENERATE",
            "simulation:scMultiSim",
            f"seed={seed} TF={candidate.shape[1]} TG={candidate.shape[0]} log={log}",
        )
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        if process.returncode:
            tail = "\n".join(
                log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            )
            raise RuntimeError(
                f"scMultiSim generation failed. Requires R, Bioconductor 3.20 scMultiSim 1.2.0, ape and digest. See {log}\n{tail}"
            )
        manifest["seeds"][str(seed)] = _pack_generated(
            folder, root, seed, candidate, truth, tr, rt, scientific
        )
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root
