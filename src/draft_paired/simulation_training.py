"""Training model and optimizer for the simulated RNA/ATAC example."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .models.gnn import GermlineLineageGNN
from .train import _graph_cpu_copy, _save_soma_checkpoint
from .utils import emit_progress


class MainSimulationGRN(GermlineLineageGNN):
    """Graph attention with recurrent state and shared output projections."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Share projections after initializing all node-role layers.
        super().__init__(*args, **kwargs)
        for layer in ("res", "gru_skip", "fc"):
            setattr(self, f"{layer}_re", getattr(self, f"{layer}_tf"))
            setattr(self, f"{layer}_tg", getattr(self, f"{layer}_tf"))


def learning_rate(epoch: int, settings: dict[str, Any]) -> float:
    start, end = int(settings["lr_decay_start"]), int(settings["lr_decay_end"])
    initial, final = float(settings["lr"]), float(settings["final_lr"])
    if not 1 <= epoch <= end or not 0 < start < end or not 0 < final <= initial:
        raise ValueError("Invalid main simulation learning-rate schedule.")
    if epoch <= start:
        return initial
    fraction = (epoch - start) / (end - start)
    return initial * (1 - (1 - final / initial) * fraction)


def reconstruction(
    a: torch.Tensor, b: torch.Tensor, edges: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Sample non-edges uniformly without replacement using a separate RNG."""
    logits = a @ b.T
    labels = torch.zeros_like(logits)
    labels[edges[0], edges[1]] = 1
    pos = torch.nonzero(labels.reshape(-1)).flatten()
    neg = torch.nonzero(labels.reshape(-1) == 0).flatten()
    selected = torch.cat(
        [pos, neg[torch.randperm(len(neg), generator=generator)[: len(pos)]]]
    )
    return F.binary_cross_entropy_with_logits(
        logits.reshape(-1)[selected], labels.reshape(-1)[selected]
    )


def train_main_simulation(
    artifacts: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    settings = config["training"]
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    graphs = artifacts["graphs_lineage"]
    device = graphs[0]["TF"].x.device
    shape = config["model"]
    model = MainSimulationGRN(
        in_channels=artifacts["in_channels"],
        hidden_channels=shape["hidden_channels"],
        out_channels=shape["out_channels"],
        heads=shape["gat_heads"],
        dropout_rate=shape["dropout_rate"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    generator = torch.Generator().manual_seed(seed + 9031)
    order = artifacts["topo_order"]
    parents = [-1] + list(range(len(graphs) - 1))
    output = Path(config["artifacts"]["dir"]).parent
    artifact_path = Path(config["artifacts"]["dir"]) / "graphs.pt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "graphs": [_graph_cpu_copy(g) for g in graphs],
            "topo_order": order,
            "parents_list": artifacts["parents_list"],
            "parent": parents,
            "time_names": artifacts["time_names"],
            "maps": artifacts["maps"],
            "in_channels": artifacts["in_channels"],
            "T_sel": len(graphs),
            "model_kind": "main_simulation_shared_beta0",
        },
        artifact_path,
    )
    history = []
    model.train()
    for epoch in range(1, int(settings["num_epochs"]) + 1):
        optimizer.zero_grad()
        tf, re, tg = model(graphs, order, parents)
        loss = sum(
            reconstruction(
                tf[:, t],
                re[:, t],
                graphs[t]["TF", "combine", "RE"].edge_index,
                generator,
            )
            + reconstruction(
                re[:, t],
                tg[:, t],
                graphs[t]["RE", "regulate", "TG"].edge_index,
                generator,
            )
            for t in range(len(graphs))
        ) / (2 * len(graphs))
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Nonfinite main simulation loss at epoch {epoch}."
            )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(settings["gradient_clip_norm"]),
            error_if_nonfinite=True,
        )
        rate = learning_rate(epoch, settings)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        value = float(loss.detach())
        history.append(
            {
                "epoch": epoch,
                "reconstruction": value,
                "learning_rate": rate,
                "gradient_norm_before_clip": float(norm),
                "gradient_clip_max_norm": settings["gradient_clip_norm"],
            }
        )
        if (
            epoch == 1
            or epoch % int(settings["log_every"]) == 0
            or epoch == settings["num_epochs"]
        ):
            emit_progress(
                "PROGRESS",
                "simulation:DRAFT",
                f"epoch={epoch}/{settings['num_epochs']} loss={value:.6f} lr={rate:.6g}",
            )
    frame = pd.DataFrame(history)
    frame.to_csv(output / "training_history.tsv", sep="\t", index=False)
    checkpoint = (
        Path(config["checkpoints"]["dir"])
        / f"final_epoch_{settings['num_epochs']:04d}.pt"
    )
    hp = {**settings, **shape}
    meta = {
        "model_kind": "main_simulation_shared_beta0",
        "module_attention_beta": 0.0,
        "shared_projections": ["res", "gru_skip", "fc"],
        "checkpoint_selection": "last",
        "selected_epoch": settings["num_epochs"],
        "seed": seed,
        "in_channels": artifacts["in_channels"],
        "input_provenance": config.get("input_provenance"),
    }
    _save_soma_checkpoint(
        checkpoint,
        model,
        optimizer,
        settings["num_epochs"],
        hp,
        meta,
        extra={
            "negative_rng_state": generator.get_state(),
            "sampler": "main_uniform_without_replacement",
        },
    )
    return {
        "checkpoint_path": str(checkpoint),
        "artifact_path": str(artifact_path),
        "hp": hp,
        "checkpoint_selection": "last",
        "selected_epoch": settings["num_epochs"],
        "seed": seed,
        "last_total_loss": history[-1]["reconstruction"],
        "lambda_smooth_used": 0.0,
        "negative_sampling": "uniform_without_replacement_separate_generator",
        "negative_generator_seed": seed + 9031,
        "shared_projections": ["res", "gru_skip", "fc"],
        "training_history": str(output / "training_history.tsv"),
        "model_kind": "main_simulation_shared_beta0",
    }
