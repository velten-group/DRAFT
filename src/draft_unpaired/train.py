"""Training entrypoint for unpaired TF-TG lineage models."""

from __future__ import annotations

import copy
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import save_checkpoint_metadata
from .graph_builder import build_lineage_graph_artifacts
from .io import save_text_report, summarize_input_paths
from .models.gnn import build_model, cosine_abs_prob
from .utils import emit_progress, ensure_directory, load_yaml_config, resolve_config_path


UNPAIRED_HP_DEFAULTS: dict[str, Any] = {
    "hidden_channels": 64,
    "out_channels": 64,
    "dropout_rate": 0.40,
    "gat_heads": 2,
    "neg_sampling_ratio": 1.0,
    "num_epochs": 600,
    "lr": 2e-4,
    "weight_decay": 1e-3,
    "auto_lambda_smooth": True,
    "lambda_smooth_init": 300.0,
    "log_every": 10,
}


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _resolve_hyperparameters(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = _section(config, "model")
    training_cfg = _section(config, "training")
    hp = dict(UNPAIRED_HP_DEFAULTS)
    hp["hidden_channels"] = int(model_cfg.get("hidden_channels", hp["hidden_channels"]))
    hp["out_channels"] = int(model_cfg.get("out_channels", hp["out_channels"]))
    hp["dropout_rate"] = float(model_cfg.get("dropout_rate", hp["dropout_rate"]))
    hp["gat_heads"] = int(model_cfg.get("gat_heads", hp["gat_heads"]))
    hp["neg_sampling_ratio"] = float(training_cfg.get("neg_sampling_ratio", hp["neg_sampling_ratio"]))
    hp["num_epochs"] = int(training_cfg.get("num_epochs", hp["num_epochs"]))
    hp["lr"] = float(training_cfg.get("lr", hp["lr"]))
    hp["weight_decay"] = float(training_cfg.get("weight_decay", hp["weight_decay"]))
    hp["auto_lambda_smooth"] = bool(training_cfg.get("auto_lambda_smooth", hp["auto_lambda_smooth"]))
    hp["lambda_smooth_init"] = float(
        training_cfg.get("lambda_smooth_init", hp["lambda_smooth_init"])
    )
    hp["log_every"] = int(training_cfg.get("log_every", hp["log_every"]))
    return hp


def _resolve_seed(config: dict[str, Any], default: int = 4412) -> int:
    return int(_section(config, "training").get("seed", default))


def _resolve_branch_weight_scales(config: dict[str, Any], branch_count: int) -> list[float] | None:
    training_cfg = _section(config, "training")
    raw = training_cfg.get("branch_weight_scales")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("training.branch_weight_scales must be a list when provided.")
    values = [float(item) for item in raw]
    if branch_count > 0 and len(values) != branch_count:
        raise ValueError(
            f"training.branch_weight_scales has {len(values)} items, expected {branch_count}."
        )
    return values


def _configure_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _graph_cpu_copy(graph: Any) -> Any:
    graph_copy = copy.deepcopy(graph)
    graph_copy = graph_copy.to("cpu")
    for node_type in graph_copy.node_types:
        if hasattr(graph_copy[node_type], "x") and graph_copy[node_type].x is not None:
            graph_copy[node_type].x = graph_copy[node_type].x.contiguous()
    for edge_type in graph_copy.edge_types:
        graph_copy[edge_type].edge_index = graph_copy[edge_type].edge_index.contiguous()
    return graph_copy


def build_labels_for_graph(
    pos_tf: torch.Tensor,
    pos_tg: torch.Tensor,
    n_tf: int,
    n_tg: int,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.zeros(n_tf * n_tg, dtype=torch.float32, device=device)
    if pos_tf.numel() > 0:
        linear_idx = (pos_tf * n_tg + pos_tg).long()
        labels[linear_idx] = 1.0
    return labels


def temporal_smoothness_loss(
    probs_per_t: list[torch.Tensor],
    parents_list: list[list[int]],
) -> torch.Tensor:
    if len(probs_per_t) <= 1:
        return probs_per_t[0].new_tensor(0.0)

    smooth = probs_per_t[0].new_tensor(0.0)
    pair_count = 0
    for time_idx, parents in enumerate(parents_list):
        for parent_idx in parents:
            if parent_idx is None or parent_idx < 0:
                continue
            smooth = smooth + F.mse_loss(probs_per_t[time_idx], probs_per_t[parent_idx])
            pair_count += 1
    if pair_count == 0:
        return probs_per_t[0].new_tensor(0.0)
    return smooth / pair_count


def branch_smoothness_loss(
    probs_per_t: list[torch.Tensor],
    branch_pairs: list[tuple[int, int]],
) -> torch.Tensor:
    if len(probs_per_t) <= 1 or not branch_pairs:
        return probs_per_t[0].new_tensor(0.0)
    losses: list[torch.Tensor] = []
    for parent_idx, child_idx in branch_pairs:
        if parent_idx < 0 or child_idx < 0:
            continue
        losses.append(F.mse_loss(probs_per_t[child_idx], probs_per_t[parent_idx]))
    if not losses:
        return probs_per_t[0].new_tensor(0.0)
    return torch.stack(losses, dim=0).mean()


def _branch_paths_from_tree(tree_df: Any, time_names: list[str]) -> list[list[str]]:
    if tree_df is None:
        return []
    names = [str(value) for value in time_names]
    parents_by_name: dict[str, list[str]] = {name: [] for name in names}
    children_by_name: dict[str, list[str]] = {name: [] for name in names}
    order_index = {name: idx for idx, name in enumerate(names)}

    rows = tree_df.to_dict("records") if hasattr(tree_df, "to_dict") else []
    for row in rows:
        node_name = str(row.get("node_id", "")).strip()
        if node_name not in parents_by_name:
            continue
        raw_parent = str(row.get("parent_id", "")).strip()
        parent_names = [
            item.strip()
            for item in raw_parent.replace(";", "|").replace(",", "|").split("|")
            if item.strip()
        ]
        parent_names = [item for item in parent_names if item in parents_by_name]
        parents_by_name[node_name] = parent_names
        for parent_name in parent_names:
            children_by_name[parent_name].append(node_name)

    leaves = [name for name in names if len(children_by_name.get(name, [])) == 0]
    if not leaves:
        return []

    paths: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for leaf_name in sorted(leaves, key=lambda item: order_index[item]):
        path = [leaf_name]
        current = leaf_name
        visited = {leaf_name}
        while parents_by_name.get(current):
            parent_name = parents_by_name[current][0]
            if parent_name in visited:
                break
            path.append(parent_name)
            visited.add(parent_name)
            current = parent_name
        path.reverse()
        key = tuple(path)
        if len(path) > 1 and key not in seen:
            seen.add(key)
            paths.append(path)
    return paths


def _branch_pairs_from_paths(branch_paths: list[list[str]], time_names: list[str]) -> list[list[tuple[int, int]]]:
    index_by_name = {str(name): idx for idx, name in enumerate(time_names)}
    pairs_by_branch: list[list[tuple[int, int]]] = []
    for path in branch_paths:
        pairs: list[tuple[int, int]] = []
        for idx in range(len(path) - 1):
            parent_name = str(path[idx])
            child_name = str(path[idx + 1])
            if parent_name not in index_by_name or child_name not in index_by_name:
                continue
            pairs.append((index_by_name[parent_name], index_by_name[child_name]))
        if pairs:
            pairs_by_branch.append(pairs)
    return pairs_by_branch


def _resolve_branch_weights(
    branch_losses: list[torch.Tensor],
    branch_scales: list[float] | None = None,
) -> torch.Tensor:
    if not branch_losses:
        raise ValueError("branch_losses must not be empty.")
    values = np.asarray([float(loss.detach()) for loss in branch_losses], dtype=float)
    eps = 1e-12
    raw = np.zeros_like(values, dtype=float)
    positive = values > eps
    if not positive.any():
        raw[:] = 1.0
    else:
        raw[positive] = 1.0 / values[positive]
        raw[~positive] = raw[positive].max(initial=1.0)
    if branch_scales is not None:
        scale_array = np.asarray(branch_scales, dtype=float)
        if scale_array.shape != raw.shape:
            raise ValueError("branch_scales shape does not match branch_losses.")
        raw = raw * np.clip(scale_array, a_min=0.0, a_max=None)
    raw = raw / max(raw.sum(), eps)
    return branch_losses[0].new_tensor(raw.tolist())


def _artifact_filename(config: dict[str, Any]) -> str:
    artifacts_cfg = _section(config, "artifacts")
    explicit = str(artifacts_cfg.get("filename", "")).strip()
    if explicit:
        return explicit
    lineage = str(config.get("lineage", "unpaired")).strip().lower() or "unpaired"
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in lineage)
    return f"{safe}_graphs.pt"


def _save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    hp: dict[str, Any],
    meta: dict[str, Any],
    extra: dict[str, Any],
    state_dict: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "state_dict": state_dict if state_dict is not None else model.state_dict(),
        "optimizer": optimizer_state if optimizer_state is not None else optimizer.state_dict(),
        "hp": hp,
        "meta": meta,
        "extra": extra,
        "versions": {"python": sys.version, "torch": torch.__version__},
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        },
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(payload, output_path)
    return output_path


def _train_artifacts(artifacts: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    hp = _resolve_hyperparameters(config)
    seed = _resolve_seed(config)
    _configure_seed(seed)

    graphs = artifacts["graphs_lineage"]
    topo_order = list(artifacts["topo_order"])
    parents_list = [list(item) for item in artifacts["parents_list"]]
    time_names = list(artifacts["time_names"])
    ordered_tree = artifacts.get("ordered_tree")
    if not graphs:
        raise ValueError("No lineage graphs were constructed from the unpaired inputs.")

    device = graphs[0]["TF"].x.device
    in_channels = int(artifacts["in_channels"])
    lineage_name = str(config.get("lineage", "unpaired"))

    artifact_dir = ensure_directory(
        resolve_config_path(
            config,
            _section(config, "artifacts").get("dir", f"result/{lineage_name}/train/artifacts"),
        )
    )
    artifact_path = artifact_dir / _artifact_filename(config)
    if artifacts.get("source_graph_path"):
        artifact_path = Path(artifacts["source_graph_path"])
    else:
        artifact_payload = {
            "graphs": [_graph_cpu_copy(graph) for graph in graphs],
            "topo_order": topo_order,
            "parents_list": parents_list,
            "time_names": time_names,
            "in_channels": in_channels,
            "T_sel": int(len(graphs)),
            "seed": int(seed),
            "maps": artifacts.get("maps", {}),
            "lineage": lineage_name,
        }
        torch.save(artifact_payload, artifact_path)
    emit_progress("PROGRESS", f"train:{lineage_name}", f"artifact={artifact_path.as_posix()}")

    model = build_model(
        in_channels=in_channels,
        hidden_channels=hp["hidden_channels"],
        out_channels=hp["out_channels"],
        heads=hp["gat_heads"],
        dropout_rate=hp["dropout_rate"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    checkpoint_dir = ensure_directory(
        resolve_config_path(
            config,
            _section(config, "checkpoints").get("dir", f"result/{lineage_name}/model"),
        )
    )

    lambda_smooth = float(hp["lambda_smooth_init"])
    lambda_fixed = not bool(hp["auto_lambda_smooth"])
    branch_paths = _branch_paths_from_tree(ordered_tree, time_names)
    branch_pairs = _branch_pairs_from_paths(branch_paths, time_names)
    branch_weight_scales = _resolve_branch_weight_scales(config, len(branch_pairs))
    branch_weights: torch.Tensor | None = None
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_sims: list[np.ndarray] | None = None
    last_total_loss = float("nan")
    last_bce_loss = float("nan")
    last_smooth_loss = float("nan")

    for epoch in range(1, hp["num_epochs"] + 1):
        model.train()
        out_tf, out_tg = model(graphs, topo_order, parents_list)
        sims, probs_per_t = cosine_abs_prob(out_tf, out_tg)

        bce_total = out_tf.new_tensor(0.0)
        for time_idx, graph in enumerate(graphs):
            edge_index = graph["TF", "regulate", "TG"].edge_index
            pos_tf = edge_index[0]
            pos_tg = edge_index[1]
            labels = build_labels_for_graph(
                pos_tf=pos_tf,
                pos_tg=pos_tg,
                n_tf=int(out_tf.size(0)),
                n_tg=int(out_tg.size(0)),
                device=device,
            )
            probs_flat = probs_per_t[time_idx].reshape(-1)

            pos_idx = (labels == 1).nonzero(as_tuple=False).view(-1)
            neg_idx = (labels == 0).nonzero(as_tuple=False).view(-1)

            if (
                pos_idx.numel() > 0
                and neg_idx.numel() > 0
                and float(hp["neg_sampling_ratio"]) > 0
            ):
                k = min(int(neg_idx.numel()), int(pos_idx.numel() * hp["neg_sampling_ratio"]))
                if k > 0:
                    perm = torch.randperm(int(neg_idx.numel()), device=device)[:k]
                    idx = torch.cat([pos_idx, neg_idx[perm]], dim=0)
                else:
                    idx = pos_idx
            else:
                idx = pos_idx

            if idx.numel() > 0:
                bce_total = bce_total + F.binary_cross_entropy(probs_flat[idx], labels[idx])

        if branch_pairs:
            branch_losses = [branch_smoothness_loss(probs_per_t, pairs) for pairs in branch_pairs]
            if branch_weights is None:
                branch_weights = _resolve_branch_weights(branch_losses, branch_weight_scales)
            smooth_loss = branch_losses[0].new_tensor(0.0)
            for idx, branch_loss in enumerate(branch_losses):
                smooth_loss = smooth_loss + (branch_weights[idx] * branch_loss)
        else:
            smooth_loss = temporal_smoothness_loss(probs_per_t, parents_list)
        if not lambda_fixed:
            with torch.no_grad():
                smooth_value = float(smooth_loss.detach())
                bce_value = float(bce_total.detach())
                lambda_smooth = (
                    float(hp["lambda_smooth_init"])
                    if smooth_value <= 1e-12
                    else (bce_value / smooth_value)
                )
            lambda_fixed = True

        total_loss = bce_total + (lambda_smooth * smooth_loss)
        total_value = float(total_loss.detach())
        state_before_step: dict[str, Any] | None = None
        optimizer_before_step: dict[str, Any] | None = None
        if total_value < best_loss:
            state_before_step = copy.deepcopy(model.state_dict())
            optimizer_before_step = copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        bce_value = float(bce_total.detach())
        smooth_value = float((lambda_smooth * smooth_loss).detach())
        last_total_loss = total_value
        last_bce_loss = bce_value
        last_smooth_loss = smooth_value

        if epoch == 1 or epoch % hp["log_every"] == 0:
            smooth_pct = (100.0 * smooth_value / total_value) if total_value > 0 else 0.0
            emit_progress(
                "PROGRESS",
                f"train:{lineage_name}",
                f"epoch={epoch}/{hp['num_epochs']} loss={total_value:.4f} smooth_pct={smooth_pct:5.1f}%",
            )

        if total_value < best_loss:
            best_loss = total_value
            best_epoch = epoch
            best_state = state_before_step if state_before_step is not None else copy.deepcopy(model.state_dict())
            best_optimizer_state = (
                optimizer_before_step
                if optimizer_before_step is not None
                else copy.deepcopy(optimizer.state_dict())
            )
            best_sims = [item.detach().cpu().numpy().astype(np.float32, copy=False) for item in sims]

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        with torch.no_grad():
            out_tf, out_tg = model(graphs, topo_order, parents_list)
            sims_best, _ = cosine_abs_prob(out_tf, out_tg)
            best_sims = [item.detach().cpu().numpy().astype(np.float32, copy=False) for item in sims_best]

    meta = {
        "artifact_path": artifact_path.as_posix(),
        "in_channels": in_channels,
        "topo_order": topo_order,
        "parents_list": parents_list,
        "time_names": time_names,
        "branch_paths": branch_paths,
        "branch_weight_scales": branch_weight_scales or [],
        "seed": int(seed),
        "lineage_name": lineage_name,
        "maps": artifacts.get("maps", {}),
    }
    extra = {
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "best_snapshot_mode": "pre_step",
        "last_total_loss": float(last_total_loss),
        "last_bce_loss": float(last_bce_loss),
        "last_smooth_loss": float(last_smooth_loss),
        "lambda_smooth": float(lambda_smooth),
        "smoothness_mode": "branch" if branch_pairs else "parent_pair",
        "branch_paths": branch_paths,
        "branch_weight_scales": branch_weight_scales or [],
        "branch_weights": branch_weights.detach().cpu().tolist() if branch_weights is not None else [],
        "sims_best": best_sims,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"{stamp}.pt"
    _save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=best_epoch if best_epoch > 0 else hp["num_epochs"],
        hp=hp,
        meta=meta,
        extra=extra,
        state_dict=best_state,
        optimizer_state=best_optimizer_state,
    )

    return {
        "lineage": lineage_name,
        "seed": int(seed),
        "hp": hp,
        "artifact_path": artifact_path.as_posix(),
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "timestamp": stamp,
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "last_total_loss": float(last_total_loss),
        "last_bce_loss": float(last_bce_loss),
        "last_smooth_loss": float(last_smooth_loss),
        "lambda_smooth": float(lambda_smooth),
        "time_names": time_names,
        "topo_order": topo_order,
        "parents_list": parents_list,
        "smoothness_mode": "branch" if branch_pairs else "parent_pair",
        "branch_paths": branch_paths,
        "branch_weight_scales": branch_weight_scales or [],
        "branch_weights": branch_weights.detach().cpu().tolist() if branch_weights is not None else [],
        "in_channels": int(in_channels),
    }


def train_lineage_model(config_path: str | Path) -> dict[str, Any]:
    from .prior import prepare_training_config

    requested_config = config_path
    config_path, preparation = prepare_training_config(config_path)
    config = load_yaml_config(config_path)
    lineage_name = str(config.get("lineage", "unpaired"))
    output_dir = ensure_directory(
        resolve_config_path(config, config.get("output_dir", f"result/{lineage_name}/train"))
    )

    artifacts = build_lineage_graph_artifacts(config)
    common = artifacts.get("common")
    save_text_report(output_dir / "preprocess_summary.txt", "\n".join(artifacts["summary_lines"]) + "\n")

    report: dict[str, Any] = {
        "lineage": lineage_name,
        "method_profile": config.get("method_profile"),
        "config_path": str(Path(config_path).resolve()),
        "requested_config_path": str(Path(requested_config).resolve()),
        "prior_preparation": {"report_path": str(Path(preparation["output_dir"]) / "prior_report.json"), "reused": preparation["reused"]} if preparation else None,
        "input_paths": {"graphs": artifacts["source_graph_path"]} if common is None else summarize_input_paths(common.input_paths),
        "num_nodes": artifacts["num_nodes"],
        "maps_sizes": {
            "tf": len(artifacts.get("maps", {}).get("tf", [])),
            "tg": len(artifacts.get("maps", {}).get("tg", [])),
        },
        "device": str(artifacts["device"]),
        "summary_lines": artifacts["summary_lines"],
        "lineage_count": len(artifacts["graphs_lineage"]),
        "in_channels": artifacts["in_channels"],
        "time_names": artifacts["time_names"],
        "topo_order": artifacts["topo_order"],
        "parents_list": artifacts["parents_list"],
    }
    report["training"] = _train_artifacts(artifacts, config)
    report["status"] = "training_completed"
    save_checkpoint_metadata(output_dir / "train_report.json", report)
    return report


__all__ = ["train_lineage_model"]
