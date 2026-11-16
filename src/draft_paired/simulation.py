"""Network inference from bundled or generated simulated RNA/ATAC data."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform

from .analysis.repro import (
    plot_smoothness_heatmap,
    save_embeddings_by_label,
    summarize_smoothness,
)
from .checkpoint import save_checkpoint_metadata, torch_load_compat
from .preprocess import pick_device_by_free_mem
from .simulation_training import MainSimulationGRN, train_main_simulation
from .utils import (
    emit_progress,
    load_yaml_config,
    read_packaged_text,
    resolve_config_path,
)


def _bundle_bytes(config: dict[str, Any], name: str) -> bytes:
    directory = config.get("simulation", {}).get("data_dir")
    if directory:
        return (resolve_config_path(config, directory) / name).read_bytes()
    return (
        resources.files("draft_paired")
        .joinpath(f"examples/simulation_9plus3/{name}")
        .read_bytes()
    )


def _bundle_array(
    config: dict[str, Any], seed: int, kind: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(_bundle_bytes(config, "manifest.json"))
    if manifest.get("schema") not in {
        "draft_simulation_9plus3_v1",
        "draft_simulation_prior_v1",
    }:
        raise ValueError("Unsupported simulation data schema.")
    record = manifest["seeds"].get(str(seed))
    if record is None:
        raise ValueError(f"Simulation data seed {seed} is not present in this bundle.")
    name = f"{kind}_{seed}.npz"
    content = _bundle_bytes(config, name)
    digest = hashlib.sha256(content).hexdigest()
    if digest != record["packaged_files"][name]:
        raise ValueError(f"Simulation input checksum mismatch: {name}")
    with np.load(io.BytesIO(content), allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    return arrays, {
        "file": name,
        "sha256": digest,
        "data_seed": seed,
        "generator": manifest.get("generator"),
        "prior": manifest.get("prior"),
        "versions": record.get("versions"),
        "generation": record["generation"]["params"],
        "dimensions": record["dimensions"],
    }


def load_simulation_inputs(
    config: dict[str, Any], seed: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read observations and public priors only; never open the truth bundle."""
    values, provenance = _bundle_array(config, seed, "inputs")
    required = {
        "rna_counts",
        "atac_counts",
        "tf_re",
        "re_tg",
        "candidate",
        "tf_ids",
        "tg_ids",
        "re_ids",
        "times",
        "cell_ids",
    }
    if set(values) != required:
        raise ValueError(
            f"Unexpected simulation training input fields: {set(values) ^ required}"
        )
    rna, atac = values["rna_counts"], values["atac_counts"]
    ntf, ntg, nre = len(values["tf_ids"]), len(values["tg_ids"]), len(values["re_ids"])
    stages = len(values["times"])
    if rna.ndim != 3 or atac.ndim != 3 or min(ntf, ntg, nre, stages, rna.shape[-1]) < 2:
        raise ValueError(
            "Simulation inputs require at least two TFs, TGs, REs, stages and paired cells per stage."
        )
    cells = rna.shape[-1]
    if rna.shape != (stages, ntf + ntg, cells) or atac.shape != (stages, nre, cells):
        raise ValueError("Simulation count axes differ from the TF/TG/RE/stage labels.")
    for name in ("rna_counts", "atac_counts"):
        counts = values[name]
        if (
            not np.isfinite(counts).all()
            or np.any(counts < 0)
            or np.any(counts != np.floor(counts))
        ):
            raise ValueError(
                "Simulation observations must be finite nonnegative counts."
            )
    for key, count in (("tf_ids", ntf), ("tg_ids", ntg), ("re_ids", nre)):
        if values[key].shape != (count,) or len(set(values[key].astype(str))) != count:
            raise ValueError(f"Invalid simulation {key}.")
    if set(values["tf_ids"].astype(str)) & set(values["tg_ids"].astype(str)):
        raise ValueError("Simulation TF and TG identifiers overlap.")
    if not np.array_equal(values["times"], np.arange(stages)) or values[
        "cell_ids"
    ].shape != (stages, cells):
        raise ValueError("Invalid simulation stage/cell labels.")
    if any(len(set(row.astype(str))) != cells for row in values["cell_ids"]):
        raise ValueError("Duplicated paired cell IDs within a simulation stage.")
    for key, shape in (
        ("tf_re", (nre, ntf)),
        ("re_tg", (ntg, nre)),
        ("candidate", (ntg, ntf)),
    ):
        if values[key].shape != shape or not np.isin(values[key], [0, 1]).all():
            raise ValueError(f"Invalid binary simulation prior {key}.")
        values[key] = values[key].astype(bool)
    indirect = values["re_tg"].astype(np.int32) @ values["tf_re"].astype(np.int32) > 0
    if not np.array_equal(indirect, values["candidate"]):
        raise ValueError(
            "Simulation candidates differ from paths through the public RE prior."
        )
    return values, provenance


def log_counts_per_10000(counts: np.ndarray) -> np.ndarray:
    """Normalize each cell separately, preserving node-by-cell correspondence."""
    values = np.asarray(counts, dtype=np.float32)
    library = values.sum(axis=1, keepdims=True)
    return np.log1p(
        np.divide(values * 10000, library, out=np.zeros_like(values), where=library > 0)
    )


def simulation_graphs(
    inputs: dict[str, np.ndarray], device: torch.device
) -> dict[str, Any]:
    from torch_geometric.data import HeteroData

    rna = log_counts_per_10000(inputs["rna_counts"])
    # Normalize ATAC counts in the stored floating-point precision.
    atac = []
    for counts in inputs["atac_counts"]:
        recovered = np.expm1(np.log1p(counts.astype(np.float32)).astype(np.float64))
        library = recovered.sum(0, keepdims=True)
        atac.append(
            np.log1p(
                np.divide(
                    recovered * 1e4,
                    library,
                    out=np.zeros_like(recovered),
                    where=library > 0,
                )
            ).astype(np.float32)
        )
    atac = np.stack(atac)
    tr = torch.tensor(
        np.vstack(np.nonzero(inputs["tf_re"])).astype(np.int64), device=device
    ).flip(0)
    rt = torch.tensor(
        np.vstack(np.nonzero(inputs["re_tg"])).astype(np.int64), device=device
    ).flip(0)
    graphs = []
    ntf, stages = len(inputs["tf_ids"]), len(rna)
    for stage in range(len(rna)):
        graph = HeteroData()
        graph["TF"].x = torch.tensor(rna[stage, :ntf], device=device)
        graph["TG"].x = torch.tensor(rna[stage, ntf:], device=device)
        graph["RE"].x = torch.tensor(atac[stage], device=device)
        for relation, edges in {
            ("TF", "combine", "RE"): tr,
            ("RE", "combined_by", "TF"): tr.flip(0),
            ("RE", "regulate", "TG"): rt,
            ("TG", "regulated_by", "RE"): rt.flip(0),
        }.items():
            graph[relation].edge_index = edges
        graphs.append(graph)
    return {
        "graphs_lineage": graphs,
        "topo_order": list(range(stages)),
        "parents_list": [[]] + [[i] for i in range(stages - 1)],
        "time_names": [f"T{i}" for i in range(stages)],
        "in_channels": rna.shape[-1],
        "model_kind": "main_simulation_shared_beta0",
        "topology_kind": "linear",
        "maps": {
            name: inputs[key].astype(str).tolist()
            for name, key in (("tf", "tf_ids"), ("tg", "tg_ids"), ("re", "re_ids"))
        },
    }


def _effective_config(config: dict[str, Any]) -> dict[str, Any]:
    import yaml

    effective = yaml.safe_load(
        read_packaged_text("draft_paired", "templates/draft_simulation_template.yaml")
    )
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(effective.get(key), dict):
            effective[key].update(copy.deepcopy(value))
        else:
            effective[key] = copy.deepcopy(value)
    if effective["simulation"].get("prior_grn"):
        effective["lineage"] = config.get("lineage", "simulation_from_prior")
        defaults = yaml.safe_load(
            read_packaged_text(
                "draft_paired", "templates/draft_simulation_prior_template.yaml"
            )
        )["generator"]
        effective["generator"] = {**defaults, **effective.get("generator", {})}
    sim, training = effective["simulation"], effective["training"]
    if float(sim["module_attention_beta"]) != 0.0:
        raise ValueError(
            "This example uses ordinary paired DRAFT GAT: module_attention_beta must be 0."
        )
    if float(training["smooth_weight"]) != 0:
        raise ValueError("The main simulation uses smooth_weight: 0.")
    if (
        training["checkpoint_selection"] != "last"
        or training["reconstruction_reduction"] != "mean"
    ):
        raise ValueError(
            "The simulation requires final-epoch selection and mean reconstruction loss."
        )
    if int(training["num_epochs"]) < 1 or int(training["log_every"]) < 1:
        raise ValueError("Simulation num_epochs and log_every must be positive.")
    if int(training["num_epochs"]) > int(training["lr_decay_end"]):
        raise ValueError("Simulation epoch budget exceeds lr_decay_end.")
    if float(training["gradient_clip_norm"]) != 1:
        raise ValueError("The main simulation uses global gradient clipping at norm 1.")
    if int(effective["model"]["hidden_channels"]) != int(
        effective["model"]["out_channels"]
    ):
        raise ValueError(
            "The main simulation uses matching hidden and output dimensions."
        )
    for key in ("data_seeds", "model_seeds"):
        seeds = sim[key]
        if (
            not isinstance(seeds, list)
            or not seeds
            or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)
        ):
            raise ValueError(
                f"simulation.{key} must be a nonempty list of distinct nonnegative integers."
            )
    if int(sim["threads"]) < 1:
        raise ValueError("simulation.threads must be positive.")
    return effective


def _reload_embeddings(
    training: dict[str, Any], artifacts: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, ...]:
    saved = torch_load_compat(training["checkpoint_path"], map_location=device)
    if saved.get("meta", {}).get("model_kind") != "main_simulation_shared_beta0":
        raise ValueError(
            "Checkpoint does not use the main simulation shared-projection model."
        )
    hp = saved["hp"]
    model = MainSimulationGRN(
        in_channels=artifacts["in_channels"],
        hidden_channels=hp["hidden_channels"],
        out_channels=hp["out_channels"],
        heads=hp["gat_heads"],
        dropout_rate=hp["dropout_rate"],
    ).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(
            artifacts["graphs_lineage"],
            artifacts["topo_order"],
            [-1] + list(range(len(artifacts["graphs_lineage"]) - 1)),
        )


def evaluate_simulation(
    config: dict[str, Any], seed: int, inputs: dict[str, np.ndarray], logits: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Called after checkpoint selection/inference; truth is evaluation-only."""
    from sklearn.metrics import roc_auc_score

    reference, provenance = _bundle_array(config, seed, "truth")
    for key in ("tf_ids", "tg_ids", "times"):
        if not np.array_equal(reference[key], inputs[key]):
            raise ValueError(f"Simulation truth {key} differs from training inputs.")
    truth = reference["truth"]
    mask = inputs["candidate"]
    if (
        truth.shape != logits.shape
        or not np.isin(truth, [0, 1]).all()
        or np.any(truth.astype(bool) & ~mask)
    ):
        raise ValueError(
            "Simulation truth must be binary and confined to the fixed candidate set."
        )
    rows = []
    for stage in range(len(inputs["times"])):
        y = truth[stage, mask]
        rows.append(
            {
                "stage": f"T{stage}",
                "candidate_pairs": int(mask.sum()),
                "positive_pairs": int(y.sum()),
                "negative_pairs": int((~y.astype(bool)).sum()),
                "auroc": float(roc_auc_score(y, logits[stage, mask]))
                if len(np.unique(y)) == 2
                else np.nan,
                "status": "scored" if len(np.unique(y)) == 2 else "one_class_reference",
            }
        )
    return pd.DataFrame(rows), provenance


def plot_simulation_outputs(
    smoothness: dict[str, Any],
    labels: list[str],
    output: Path,
    stage_metrics: pd.DataFrame | None = None,
) -> list[str]:
    import matplotlib.pyplot as plt

    heatmap = output / "candidate_distances.png"
    plot_smoothness_heatmap(
        smoothness["dist_mat_norm"],
        labels,
        smoothness["overall_smoothness"],
        "DRAFT candidate networks",
        output_path=heatmap,
        title_fs=14,
        other_fs=12,
        cbar_label="Normalized distance",
    )
    paths = [str(heatmap)]
    if stage_metrics is not None and stage_metrics.auroc.notna().any():
        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
        ax.plot(labels, stage_metrics.auroc, color="#d97924", marker="o", linewidth=2)
        ax.set(
            xlabel="Stage",
            ylabel="Candidate-pair AUROC",
            ylim=(0, 1),
            title=f"DRAFT simulation (mean AUROC {stage_metrics.auroc.mean():.4f})",
        )
        ax.spines[["top", "right"]].set_visible(False)
        path = output / "candidate_auroc.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _run_one(
    config: dict[str, Any],
    seed: int,
    model_seed: int,
    inputs: dict[str, np.ndarray],
    provenance: dict[str, Any],
    root: Path,
    device: torch.device,
) -> dict[str, Any]:
    output = root / f"seed_{seed}" / f"model_{model_seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Simulation output already exists; choose a new output_dir: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    run_config = copy.deepcopy(config)
    run_config.update(
        lineage=str(config.get("lineage", "simulation")),
        artifacts={"dir": str(output / "artifacts"), "filename": "graphs.pt"},
        checkpoints={"dir": str(output / "model")},
    )
    run_config["training"]["seed"] = model_seed
    run_config["input_provenance"] = provenance
    save_checkpoint_metadata(output / "effective_config.json", run_config)
    save_checkpoint_metadata(output / "input_provenance.json", provenance)
    artifacts = simulation_graphs(inputs, device)
    training = train_main_simulation(artifacts, run_config)
    save_checkpoint_metadata(output / "train_report.json", training)
    out_tf, out_re, out_tg = _reload_embeddings(training, artifacts, device)
    labels = artifacts["time_names"]
    embedding_dir = output / "embeddings"
    saved_embeddings = save_embeddings_by_label(
        out_tf, out_re, out_tg, labels, embedding_dir
    )
    logits = np.stack(
        [
            (out_tg[:, t].double() @ out_tf[:, t].double().T).detach().cpu().numpy()
            for t in range(len(labels))
        ]
    )
    probabilities = np.exp(-np.logaddexp(0, -logits))
    np.savez_compressed(
        output / "relations.npz",
        logits=logits,
        probabilities=probabilities,
        candidate=inputs["candidate"],
        times=inputs["times"],
        tf_ids=inputs["tf_ids"],
        tg_ids=inputs["tg_ids"],
    )
    mask = inputs["candidate"]
    target, tf = np.nonzero(mask)
    network_dir = output / "network"
    network_dir.mkdir()
    for stage in range(len(labels)):
        pd.DataFrame(
            {
                "TF": inputs["tf_ids"][tf],
                "TG": inputs["tg_ids"][target],
                "logit": logits[stage, target, tf],
                "probability": probabilities[stage, target, tf],
            }
        ).to_csv(network_dir / f"T{stage}_TF_TG.tsv", sep="\t", index=False)
    distances = squareform(
        pdist(probabilities[:, mask].astype(np.float64), metric="euclidean")
    )
    smoothness = summarize_smoothness(distances)
    pd.DataFrame(distances, index=labels, columns=labels).to_csv(
        output / "candidate_distances.tsv", sep="\t"
    )
    pd.DataFrame(smoothness["dist_mat_norm"], index=labels, columns=labels).to_csv(
        output / "candidate_distances_normalized.tsv", sep="\t"
    )
    evaluation, truth_provenance, stage_metrics = None, None, None
    if config["simulation"]["evaluate"]:
        stage_metrics, truth_provenance = evaluate_simulation(
            config, seed, inputs, logits
        )
        stage_metrics.to_csv(output / "per_stage_metrics.tsv", sep="\t", index=False)
        evaluation = {
            "mean_auroc": float(stage_metrics.auroc.mean())
            if stage_metrics.auroc.notna().any()
            else None,
            "score": "TF-TG inner-product logits",
            "scope": "all fixed candidate pairs; no score threshold or negative subsampling",
            "per_stage_metrics": str(output / "per_stage_metrics.tsv"),
        }
    figures = (
        plot_simulation_outputs(smoothness, labels, output, stage_metrics)
        if config["simulation"]["save_figures"]
        else []
    )
    report = {
        "method": "DRAFT",
        "data_seed": seed,
        "model_seed": model_seed,
        "output_dir": str(output),
        "device": str(device),
        "training": training,
        "module_attention_beta": 0.0,
        "explicit_smoothness_weight": training["lambda_smooth_used"],
        "normalization": "RNA and ATAC separately: log1p(counts per 10000), per cell over selected nodes",
        "input_provenance": provenance,
        "truth_provenance": truth_provenance,
        "truth_used_in_training": False,
        "evaluation": evaluation,
        "smoothness": {
            key: value for key, value in smoothness.items() if key != "dist_mat_norm"
        },
        "smoothness_score_space": f"sigmoid TF-TG logits on all {int(mask.sum())} fixed candidates",
        "saved_embeddings": saved_embeddings,
        "figures": figures,
        "relations": str(output / "relations.npz"),
        "checkpoint_sha256": hashlib.sha256(
            Path(training["checkpoint_path"]).read_bytes()
        ).hexdigest(),
        "status": "completed",
    }
    save_checkpoint_metadata(output / "simulation_report.json", report)
    return report


def run_simulation(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Train, reload, export and optionally evaluate simulated regulatory networks."""
    loaded = load_yaml_config(config) if isinstance(config, (str, Path)) else config
    effective = _effective_config(loaded)
    sim = effective["simulation"]
    root = resolve_config_path(effective, sim["output_dir"])
    if sim["device"] == "auto":
        device = pick_device_by_free_mem()
    else:
        device = torch.device(sim["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "Simulation requested CUDA but this PyTorch runtime has no CUDA device."
        )
    # Check output ownership before generation or training.
    for seed in sim["data_seeds"]:
        for model_seed in sim["model_seeds"]:
            path = root / f"seed_{seed}" / f"model_{model_seed}"
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(
                    f"Simulation output already exists; choose a new output_dir: {path}"
                )
    if sim.get("prior_grn"):
        from .simulation_generate import generate_from_prior

        sim["data_dir"] = str(generate_from_prior(effective))
    datasets = {
        seed: load_simulation_inputs(effective, seed) for seed in sim["data_seeds"]
    }
    root.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    reports = []
    try:
        torch.set_num_threads(int(sim["threads"]))
        for seed, (inputs, provenance) in datasets.items():
            for model_seed in sim["model_seeds"]:
                emit_progress(
                    "START",
                    "simulation:DRAFT",
                    f"data={seed} model={model_seed} epochs={effective['training']['num_epochs']}",
                )
                reports.append(
                    _run_one(
                        effective, seed, model_seed, inputs, provenance, root, device
                    )
                )
    finally:
        torch.set_num_threads(previous_threads)
    rows = [
        {
            "data_seed": r["data_seed"],
            "model_seed": r["model_seed"],
            "mean_auroc": r["evaluation"]["mean_auroc"] if r["evaluation"] else None,
            "smoothness": r["smoothness"]["overall_smoothness"],
            "output_dir": r["output_dir"],
        }
        for r in reports
    ]
    pd.DataFrame(rows).to_csv(root / "runs.tsv", sep="\t", index=False)
    report = {
        "method": "DRAFT",
        "output_dir": str(root),
        "run_count": len(reports),
        "config": effective,
        "runs": rows,
        "status": "completed",
    }
    save_checkpoint_metadata(root / "simulation_report.json", report)
    return report
