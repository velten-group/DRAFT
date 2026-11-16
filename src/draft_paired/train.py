"""Training entrypoints for draft_paired lineage models."""

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
import torch.nn as nn

from .checkpoint import save_checkpoint_metadata
from .graph_builder import build_lineage_graph_artifacts, unify_feature_dim
from .io import save_text_report, summarize_input_paths
from .models.gnn import build_model, build_model_for_lineage
from .utils import emit_progress, ensure_directory, load_yaml_config, resolve_config_path


TREE_HP_DEFAULTS: dict[str, Any] = {
    "hidden_channels": 64,
    "out_channels": 64,
    "dropout_rate": 0.40,
    "gat_heads": 2,
    "neg_sampling_ratio": 1.0,
    "dedup_neg": True,
    "num_epochs": 600,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "auto_lambda_smooth": True,
    "lambda_smooth_init": 0.0,
    "log_every": 10,
}

SOMA_HP_DEFAULTS: dict[str, Any] = {"USE_SOMATIC": True, **TREE_HP_DEFAULTS}
GERM_HP_DEFAULTS: dict[str, Any] = {**TREE_HP_DEFAULTS, "lr": 5e-4}


def _default_hp_for_lineage(lineage: str) -> dict[str, Any]:
    lineage_key = str(lineage).strip().lower()
    if lineage_key in {"germ", "germline"}:
        return GERM_HP_DEFAULTS
    return SOMA_HP_DEFAULTS


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_soma_hyperparameters(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = _section(config, "model")
    training_cfg = _section(config, "training")
    hp = dict(SOMA_HP_DEFAULTS)
    hp["hidden_channels"] = int(model_cfg.get("hidden_channels", hp["hidden_channels"]))
    hp["out_channels"] = int(model_cfg.get("out_channels", hp["out_channels"]))
    hp["dropout_rate"] = float(model_cfg.get("dropout_rate", hp["dropout_rate"]))
    hp["gat_heads"] = int(model_cfg.get("gat_heads", hp["gat_heads"]))
    hp["neg_sampling_ratio"] = float(
        training_cfg.get("neg_sampling_ratio", hp["neg_sampling_ratio"])
    )
    hp["dedup_neg"] = _as_bool(training_cfg.get("dedup_neg", hp["dedup_neg"]))
    hp["num_epochs"] = int(training_cfg.get("num_epochs", hp["num_epochs"]))
    hp["lr"] = float(training_cfg.get("lr", hp["lr"]))
    hp["weight_decay"] = float(training_cfg.get("weight_decay", hp["weight_decay"]))
    hp["auto_lambda_smooth"] = _as_bool(
        training_cfg.get("auto_lambda_smooth", hp["auto_lambda_smooth"])
    )
    hp["lambda_smooth_init"] = float(
        training_cfg.get("lambda_smooth_init", hp["lambda_smooth_init"])
    )
    hp["log_every"] = int(training_cfg.get("log_every", hp["log_every"]))
    return hp


def _resolve_germ_hyperparameters(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = _section(config, "model")
    training_cfg = _section(config, "training")
    hp = dict(GERM_HP_DEFAULTS)
    hp["hidden_channels"] = int(model_cfg.get("hidden_channels", hp["hidden_channels"]))
    hp["out_channels"] = int(model_cfg.get("out_channels", hp["out_channels"]))
    hp["dropout_rate"] = float(model_cfg.get("dropout_rate", hp["dropout_rate"]))
    hp["gat_heads"] = int(model_cfg.get("gat_heads", hp["gat_heads"]))
    hp["neg_sampling_ratio"] = float(
        training_cfg.get("neg_sampling_ratio", hp["neg_sampling_ratio"])
    )
    hp["dedup_neg"] = _as_bool(training_cfg.get("dedup_neg", hp["dedup_neg"]))
    hp["num_epochs"] = int(training_cfg.get("num_epochs", hp["num_epochs"]))
    hp["lr"] = float(training_cfg.get("lr", hp["lr"]))
    hp["weight_decay"] = float(training_cfg.get("weight_decay", hp["weight_decay"]))
    hp["auto_lambda_smooth"] = _as_bool(
        training_cfg.get("auto_lambda_smooth", hp["auto_lambda_smooth"])
    )
    hp["lambda_smooth_init"] = float(
        training_cfg.get("lambda_smooth_init", hp["lambda_smooth_init"])
    )
    hp["log_every"] = int(training_cfg.get("log_every", hp["log_every"]))
    return hp


def _resolve_tree_hyperparameters(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = _section(config, "model")
    training_cfg = _section(config, "training")
    # Use explicit training settings independently of the lineage name.
    hp = dict(
        TREE_HP_DEFAULTS if int(config.get("config_version", 1)) >= 2
        else _default_hp_for_lineage(str(config.get("lineage", "")))
    )
    hp["hidden_channels"] = int(model_cfg.get("hidden_channels", hp["hidden_channels"]))
    hp["out_channels"] = int(model_cfg.get("out_channels", hp["out_channels"]))
    hp["dropout_rate"] = float(model_cfg.get("dropout_rate", hp["dropout_rate"]))
    hp["gat_heads"] = int(model_cfg.get("gat_heads", hp["gat_heads"]))
    hp["neg_sampling_ratio"] = float(
        training_cfg.get("neg_sampling_ratio", hp["neg_sampling_ratio"])
    )
    hp["dedup_neg"] = _as_bool(training_cfg.get("dedup_neg", hp["dedup_neg"]))
    hp["num_epochs"] = int(training_cfg.get("num_epochs", hp["num_epochs"]))
    hp["lr"] = float(training_cfg.get("lr", hp["lr"]))
    hp["weight_decay"] = float(training_cfg.get("weight_decay", hp["weight_decay"]))
    hp["auto_lambda_smooth"] = _as_bool(
        training_cfg.get("auto_lambda_smooth", hp["auto_lambda_smooth"])
    )
    hp["lambda_smooth_init"] = float(
        training_cfg.get("lambda_smooth_init", hp["lambda_smooth_init"])
    )
    hp["log_every"] = int(training_cfg.get("log_every", hp["log_every"]))
    return hp


def _checkpoint_selection(config: dict[str, Any]) -> str:
    selection = str(_section(config, "training").get("checkpoint_selection", "auto")).lower()
    if selection == "auto":
        config_v1_germ = int(config.get("config_version", 1)) < 2 and str(config.get("lineage", "")).lower() in {"germ", "germline"}
        selection = "last" if config_v1_germ else "best_loss"
    if selection not in {"last", "best_loss"}:
        raise ValueError("training.checkpoint_selection must be auto, last, or best_loss.")
    return selection


def _resolve_seed(config: dict[str, Any], default: int = 2025) -> int:
    training_cfg = _section(config, "training")
    return int(training_cfg.get("seed", default))


def _configure_soma_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _configure_germ_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _configure_seed(seed: int, include_python_random: bool = True) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if include_python_random:
        random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_negative_pairs(
    n_src: int,
    n_dst: int,
    pos_is: torch.Tensor,
    pos_js: torch.Tensor,
    k: int,
    device: torch.device | str,
    dedup: bool = True,
    max_trials: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = n_src * n_dst
    k = int(min(k, max(0, total - pos_is.numel())))
    if k == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    if pos_is.numel() > 0:
        pos_lin = pos_is * n_dst + pos_js
    else:
        pos_lin = torch.empty(0, dtype=torch.long, device=device)
    got = torch.empty(0, dtype=torch.long, device=device)
    remain = k
    for _ in range(max_trials):
        need = int(remain * 1.3) + 16
        rand_lin = torch.randint(0, total, (need,), device=device)
        if pos_lin.numel() > 0:
            rand_lin = rand_lin[~torch.isin(rand_lin, pos_lin)]
        if dedup:
            rand_lin = torch.unique(rand_lin)
        got = (
            rand_lin
            if got.numel() == 0
            else (
                torch.unique(torch.cat([got, rand_lin], dim=0))
                if dedup
                else torch.cat([got, rand_lin], dim=0)
            )
        )
        if got.numel() >= k:
            got = got[:k]
            break
        remain = k - got.numel()
    return got // n_dst, got % n_dst


def pairwise_scores(
    h_src: torch.Tensor, h_dst: torch.Tensor, is_idx: torch.Tensor, js_idx: torch.Tensor
) -> torch.Tensor:
    if is_idx.numel() == 0:
        return torch.empty(0, device=h_src.device)
    return (h_src[is_idx] * h_dst[js_idx]).sum(dim=1)


def frob_diff_of_outer(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    ata = a.T @ a
    btb = b.T @ b
    ctc = c.T @ c
    dtd = d.T @ d
    return (ata * btb).sum() + (ctc * dtd).sum() - 2.0 * ((a.T @ c) * (b.T @ d)).sum()


def _graph_cpu_copy(graph: Any) -> Any:
    graph_copy = copy.deepcopy(graph)
    graph_copy = graph_copy.to("cpu")
    for node_type in graph_copy.node_types:
        if hasattr(graph_copy[node_type], "x") and graph_copy[node_type].x is not None:
            graph_copy[node_type].x = graph_copy[node_type].x.contiguous()
    for edge_type in graph_copy.edge_types:
        graph_copy[edge_type].edge_index = graph_copy[edge_type].edge_index.contiguous()
    return graph_copy


def _save_soma_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    hp: dict[str, Any],
    meta: dict[str, Any],
    extra: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": int(epoch),
        "state_dict": state_dict if state_dict is not None else model.state_dict(),
        "optimizer": (
            optimizer_state
            if optimizer_state is not None
            else (optimizer.state_dict() if optimizer is not None else None)
        ),
        "hp": hp,
        "meta": meta,
        "versions": {"python": sys.version, "torch": torch.__version__},
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        },
        "extra": extra or {},
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, output_path)
    return output_path


def _save_tree_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    hp: dict[str, Any],
    meta: dict[str, Any],
    extra: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> Path:
    return _save_soma_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        hp=hp,
        meta=meta,
        extra=extra,
        state_dict=state_dict,
        optimizer_state=optimizer_state,
    )


def _train_soma_artifacts(artifacts: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    hp = _resolve_soma_hyperparameters(config)
    seed = _resolve_seed(config, default=2025)
    _configure_soma_seed(seed)

    graphs_sel = artifacts["graphs_somatic"]
    if len(graphs_sel) == 0:
        raise ValueError("No somatic graphs were constructed. Cannot start soma training.")

    topo_order = list(artifacts["topo_somatic"])
    parents_list = artifacts["parents_somatic"]
    time_names = artifacts["somatic_names"]

    t_sel = len(graphs_sel)
    device = graphs_sel[0]["TF"].x.device
    in_channels = graphs_sel[0]["TF"].x.size(1)
    unify_feature_dim(graphs_sel, in_channels)

    artifact_dir = ensure_directory(
        resolve_config_path(config, _section(config, "artifacts").get("dir", "result/soma/train/artifacts"))
    )
    artifact_path = artifact_dir / "soma_graphs.pt"
    artifact_payload = {
        "graphs": [_graph_cpu_copy(graph) for graph in graphs_sel],
        "topo_order": list(topo_order),
        "parents_list": parents_list,
        "time_names": list(time_names),
        "in_channels": int(in_channels),
        "T_sel": int(t_sel),
        "seed": int(seed),
    }
    if "maps" in artifacts:
        artifact_payload["maps"] = artifacts["maps"]
    torch.save(artifact_payload, artifact_path)
    emit_progress("PROGRESS", "train:soma", f"artifact={artifact_path.as_posix()}")

    model = build_model_for_lineage(
        "soma",
        in_channels=in_channels,
        hidden_channels=hp["hidden_channels"],
        out_channels=hp["out_channels"],
        heads=hp["gat_heads"],
        dropout_rate=hp["dropout_rate"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    bce = nn.BCEWithLogitsLoss()

    meta = {
        "USE_SOMATIC": True,
        "artifact_path": artifact_path.as_posix(),
        "in_channels": int(in_channels),
        "topo_order": list(topo_order),
        "parents_list": parents_list,
        "time_names": list(time_names),
        "T_sel": int(t_sel),
        "seed": int(seed),
    }
    if "maps" in artifacts:
        meta["maps"] = artifacts["maps"]

    checkpoint_dir = ensure_directory(
        resolve_config_path(config, _section(config, "checkpoints").get("dir", "result/soma/model"))
    )

    lambda_smooth_used = float(hp["lambda_smooth_init"])
    lambda_fixed = not hp.get("auto_lambda_smooth", True)

    best_loss = float("inf")
    best_epoch = -1
    last_total_loss = float("nan")
    best_state_dict: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None

    for epoch in range(1, hp["num_epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        out_tf, out_re, out_tg = model(graphs_sel, topo_order, parents_list)

        data_loss = torch.tensor(0.0, device=device)
        sum_loss_tr = 0.0
        sum_loss_rt = 0.0
        smooth_accum = torch.tensor(0.0, device=device)
        n_smooth_terms = 0

        for t in range(t_sel):
            h_tf = out_tf[:, t, :]
            h_re = out_re[:, t, :]
            h_tg = out_tg[:, t, :]

            e_tr = graphs_sel[t][("TF", "combine", "RE")].edge_index.to(device)
            if e_tr.numel() > 0:
                pos_i, pos_j = e_tr[0], e_tr[1]
                n_src, n_dst = h_tf.size(0), h_re.size(0)
                k_neg = int(hp["neg_sampling_ratio"] * max(1, pos_i.numel()))
                neg_i, neg_j = sample_negative_pairs(
                    n_src, n_dst, pos_i, pos_j, k_neg, device, dedup=hp["dedup_neg"]
                )
                logits = torch.cat(
                    [
                        pairwise_scores(h_tf, h_re, pos_i, pos_j),
                        pairwise_scores(h_tf, h_re, neg_i, neg_j),
                    ],
                    dim=0,
                )
                labels = torch.cat(
                    [
                        torch.ones(pos_i.numel(), device=device),
                        torch.zeros(neg_i.numel(), device=device),
                    ],
                    dim=0,
                )
                loss_tr = bce(logits, labels)
                data_loss = data_loss + loss_tr
                sum_loss_tr += loss_tr.item()

            e_rt = graphs_sel[t][("RE", "regulate", "TG")].edge_index.to(device)
            if e_rt.numel() > 0:
                pos_i, pos_j = e_rt[0], e_rt[1]
                n_src, n_dst = h_re.size(0), h_tg.size(0)
                k_neg = int(hp["neg_sampling_ratio"] * max(1, pos_i.numel()))
                neg_i, neg_j = sample_negative_pairs(
                    n_src, n_dst, pos_i, pos_j, k_neg, device, dedup=hp["dedup_neg"]
                )
                logits = torch.cat(
                    [
                        pairwise_scores(h_re, h_tg, pos_i, pos_j),
                        pairwise_scores(h_re, h_tg, neg_i, neg_j),
                    ],
                    dim=0,
                )
                labels = torch.cat(
                    [
                        torch.ones(pos_i.numel(), device=device),
                        torch.zeros(neg_i.numel(), device=device),
                    ],
                    dim=0,
                )
                loss_rt = bce(logits, labels)
                data_loss = data_loss + loss_rt
                sum_loss_rt += loss_rt.item()

            parents_t = [p for p in (parents_list[t] or []) if p is not None and p >= 0]
            for parent_idx in parents_t:
                h_tf_p = out_tf[:, parent_idx, :]
                h_re_p = out_re[:, parent_idx, :]
                h_tg_p = out_tg[:, parent_idx, :]
                smooth_accum = smooth_accum + frob_diff_of_outer(h_tf, h_re, h_tf_p, h_re_p)
                smooth_accum = smooth_accum + frob_diff_of_outer(h_re, h_tg, h_re_p, h_tg_p)
                n_smooth_terms += 2

        smooth_avg = smooth_accum / max(1, n_smooth_terms)
        if not lambda_fixed:
            if n_smooth_terms > 0 and smooth_avg.item() > 0 and data_loss.item() > 0:
                lambda_smooth_used = (
                    data_loss.detach() / (smooth_avg.detach() + 1e-12)
                ).item()
            else:
                lambda_smooth_used = float(hp["lambda_smooth_init"])
            lambda_fixed = True

        total_loss = data_loss + (lambda_smooth_used * smooth_avg)
        total_loss.backward()
        optimizer.step()

        total_val = float(total_loss.detach())
        last_total_loss = total_val

        if epoch % hp["log_every"] == 0 or epoch == 1:
            smooth_contrib = float(lambda_smooth_used * smooth_avg.detach())
            frac = (smooth_contrib / total_val) if total_val > 0 else 0.0
            emit_progress(
                "PROGRESS",
                "train:soma",
                f"epoch={epoch}/{hp['num_epochs']} loss={total_val:.4f} smooth_pct={frac*100:5.1f}%",
            )

        if total_val < best_loss:
            best_loss = total_val
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = checkpoint_dir / f"{stamp}.pt"
    _save_soma_checkpoint(
        save_path,
        model,
        optimizer,
        best_epoch if best_epoch > 0 else hp["num_epochs"],
        hp,
        meta,
        extra={
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "last_total_loss": float(last_total_loss),
            "lambda_smooth_used": float(lambda_smooth_used),
        },
        state_dict=best_state_dict,
        optimizer_state=best_optimizer_state,
    )

    return {
        "lineage": "soma",
        "seed": int(seed),
        "hp": hp,
        "artifact_path": artifact_path.as_posix(),
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "checkpoint_path": save_path.as_posix(),
        "timestamp": stamp,
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "last_total_loss": float(last_total_loss),
        "lambda_smooth_used": float(lambda_smooth_used),
        "T_sel": int(t_sel),
        "in_channels": int(in_channels),
        "topo_order": list(topo_order),
        "parents_list": parents_list,
        "time_names": list(time_names),
    }


def _tree_artifact_filename(config: dict[str, Any]) -> str:
    artifacts_cfg = _section(config, "artifacts")
    explicit = str(artifacts_cfg.get("filename", "")).strip()
    if explicit:
        return explicit
    lineage = str(config.get("lineage", "lineage")).strip().lower() or "lineage"
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in lineage)
    return f"{safe}_graphs.pt"


def _train_tree_artifacts(artifacts: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    hp = _resolve_tree_hyperparameters(config)
    selection = _checkpoint_selection(config)
    seed = _resolve_seed(config, default=2025)
    _configure_seed(seed)

    graphs_sel = artifacts["graphs_lineage"]
    if len(graphs_sel) == 0:
        raise ValueError("No lineage graphs were constructed from the provided tree.")

    topo_order = list(artifacts["topo_order"])
    parents_list = [list(parents) for parents in artifacts["parents_list"]]
    time_names = list(artifacts["time_names"])
    model_kind = str(artifacts.get("model_kind", "multi_parent"))
    topology_kind = str(artifacts.get("topology_kind", "unknown"))

    t_sel = len(graphs_sel)
    device = graphs_sel[0]["TF"].x.device
    in_channels = int(artifacts["in_channels"])
    unify_feature_dim(graphs_sel, in_channels)

    artifact_dir = ensure_directory(
        resolve_config_path(config, _section(config, "artifacts").get("dir", f"result/{config.get('lineage', 'lineage')}/train/artifacts"))
    )
    artifact_path = artifact_dir / _tree_artifact_filename(config)
    if artifacts.get("source_graph_path"):
        artifact_path = Path(artifacts["source_graph_path"])
    else:
        artifact_payload = {
            "graphs": [_graph_cpu_copy(graph) for graph in graphs_sel],
            "topo_order": topo_order,
            "parents_list": parents_list,
            "time_names": time_names,
            "in_channels": int(in_channels),
            "T_sel": int(t_sel),
            "seed": int(seed),
            "model_kind": model_kind,
            "topology_kind": topology_kind,
        }
        if "maps" in artifacts:
            artifact_payload["maps"] = artifacts["maps"]
        torch.save(artifact_payload, artifact_path)
    emit_progress("PROGRESS", f"train:{config.get('lineage', 'lineage')}", f"artifact={artifact_path.as_posix()}")

    model = build_model(
        "tree",
        in_channels=in_channels,
        hidden_channels=hp["hidden_channels"],
        out_channels=hp["out_channels"],
        heads=hp["gat_heads"],
        dropout_rate=hp["dropout_rate"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    bce = nn.BCEWithLogitsLoss()

    meta = {
        "artifact_path": artifact_path.as_posix(),
        "in_channels": int(in_channels),
        "topo_order": topo_order,
        "parents_list": parents_list,
        "time_names": time_names,
        "T_sel": int(t_sel),
        "seed": int(seed),
        "model_kind": model_kind,
        "topology_kind": topology_kind,
        "lineage_name": str(config.get("lineage", "lineage")),
    }
    if "maps" in artifacts:
        meta["maps"] = artifacts["maps"]

    checkpoint_dir = ensure_directory(
        resolve_config_path(config, _section(config, "checkpoints").get("dir", f"result/{config.get('lineage', 'lineage')}/model"))
    )

    lambda_smooth_used = float(hp["lambda_smooth_init"])
    lambda_fixed = not hp.get("auto_lambda_smooth", True)

    best_loss = float("inf")
    best_epoch = -1
    last_total_loss = float("nan")
    best_state_dict: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    task_label = f"train:{config.get('lineage', 'lineage')}"

    for epoch in range(1, hp["num_epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        out_tf, out_re, out_tg = model(graphs_sel, topo_order, parents_list)

        data_loss = torch.tensor(0.0, device=device)
        smooth_accum = torch.tensor(0.0, device=device)
        n_smooth_terms = 0

        for t in range(t_sel):
            h_tf = out_tf[:, t, :]
            h_re = out_re[:, t, :]
            h_tg = out_tg[:, t, :]

            e_tr = graphs_sel[t][("TF", "combine", "RE")].edge_index.to(device)
            if e_tr.numel() > 0:
                pos_i, pos_j = e_tr[0], e_tr[1]
                n_src, n_dst = h_tf.size(0), h_re.size(0)
                k_neg = int(hp["neg_sampling_ratio"] * max(1, pos_i.numel()))
                neg_i, neg_j = sample_negative_pairs(
                    n_src, n_dst, pos_i, pos_j, k_neg, device, dedup=hp["dedup_neg"]
                )
                logits = torch.cat(
                    [
                        pairwise_scores(h_tf, h_re, pos_i, pos_j),
                        pairwise_scores(h_tf, h_re, neg_i, neg_j),
                    ],
                    dim=0,
                )
                labels = torch.cat(
                    [
                        torch.ones(pos_i.numel(), device=device),
                        torch.zeros(neg_i.numel(), device=device),
                    ],
                    dim=0,
                )
                data_loss = data_loss + bce(logits, labels)

            e_rt = graphs_sel[t][("RE", "regulate", "TG")].edge_index.to(device)
            if e_rt.numel() > 0:
                pos_i, pos_j = e_rt[0], e_rt[1]
                n_src, n_dst = h_re.size(0), h_tg.size(0)
                k_neg = int(hp["neg_sampling_ratio"] * max(1, pos_i.numel()))
                neg_i, neg_j = sample_negative_pairs(
                    n_src, n_dst, pos_i, pos_j, k_neg, device, dedup=hp["dedup_neg"]
                )
                logits = torch.cat(
                    [
                        pairwise_scores(h_re, h_tg, pos_i, pos_j),
                        pairwise_scores(h_re, h_tg, neg_i, neg_j),
                    ],
                    dim=0,
                )
                labels = torch.cat(
                    [
                        torch.ones(pos_i.numel(), device=device),
                        torch.zeros(neg_i.numel(), device=device),
                    ],
                    dim=0,
                )
                data_loss = data_loss + bce(logits, labels)

            parents_t = [p for p in parents_list[t] if p is not None and p >= 0]
            for parent_idx in parents_t:
                h_tf_p = out_tf[:, parent_idx, :]
                h_re_p = out_re[:, parent_idx, :]
                h_tg_p = out_tg[:, parent_idx, :]
                smooth_accum = smooth_accum + frob_diff_of_outer(h_tf, h_re, h_tf_p, h_re_p)
                smooth_accum = smooth_accum + frob_diff_of_outer(h_re, h_tg, h_re_p, h_tg_p)
                n_smooth_terms += 2

        smooth_avg = smooth_accum / max(1, n_smooth_terms)
        if not lambda_fixed:
            if n_smooth_terms > 0 and smooth_avg.item() > 0 and data_loss.item() > 0:
                lambda_smooth_used = (
                    data_loss.detach() / (smooth_avg.detach() + 1e-12)
                ).item()
            else:
                lambda_smooth_used = float(hp["lambda_smooth_init"])
            lambda_fixed = True

        total_loss = data_loss + (lambda_smooth_used * smooth_avg)
        total_loss.backward()
        total_val = float(total_loss.detach())
        if total_val < best_loss:
            best_loss = total_val
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        optimizer.step()

        last_total_loss = total_val

        if epoch % hp["log_every"] == 0 or epoch == 1:
            smooth_contrib = float(lambda_smooth_used * smooth_avg.detach())
            frac = (smooth_contrib / total_val) if total_val > 0 else 0.0
            emit_progress(
                "PROGRESS",
                task_label,
                f"epoch={epoch}/{hp['num_epochs']} loss={total_val:.4f} smooth_pct={frac*100:5.1f}%",
            )

    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = checkpoint_dir / f"{stamp}.pt"
    selected_epoch = hp["num_epochs"] if selection == "last" else best_epoch
    if selection == "last":
        best_state_dict = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    meta["checkpoint_selection"] = selection
    meta["selected_epoch"] = int(selected_epoch)
    _save_tree_checkpoint(
        save_path,
        model,
        optimizer,
        selected_epoch,
        hp,
        meta,
        extra={
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "last_total_loss": float(last_total_loss),
            "lambda_smooth_used": float(lambda_smooth_used),
        },
        state_dict=best_state_dict,
        optimizer_state=best_optimizer_state,
    )

    return {
        "lineage": str(config.get("lineage", "lineage")),
        "checkpoint_selection": selection,
        "selected_epoch": int(selected_epoch),
        "seed": int(seed),
        "hp": hp,
        "artifact_path": artifact_path.as_posix(),
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "checkpoint_path": save_path.as_posix(),
        "timestamp": stamp,
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "last_total_loss": float(last_total_loss),
        "lambda_smooth_used": float(lambda_smooth_used),
        "T_sel": int(t_sel),
        "in_channels": int(in_channels),
        "topo_order": topo_order,
        "parents_list": parents_list,
        "time_names": time_names,
        "model_kind": model_kind,
        "topology_kind": topology_kind,
    }


def _train_germ_artifacts(artifacts: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    hp = _resolve_germ_hyperparameters(config)
    seed = _resolve_seed(config, default=2025)
    _configure_germ_seed(seed)

    graphs_sel = artifacts["graphs_germline"]
    if len(graphs_sel) == 0:
        raise ValueError("No germline graphs were constructed. Cannot start germ training.")

    t_sel = len(graphs_sel)
    device = graphs_sel[0]["TF"].x.device
    topo_order = list(artifacts.get("topo_order", list(range(t_sel))))
    if "parent" in artifacts:
        parent = list(artifacts["parent"])
    else:
        parent = [-1] + list(range(t_sel - 1)) if t_sel > 0 else []

    in_channels = graphs_sel[0]["TF"].x.size(1)
    unify_feature_dim(graphs_sel, in_channels)

    model = build_model_for_lineage(
        "germ",
        in_channels=in_channels,
        hidden_channels=hp["hidden_channels"],
        out_channels=hp["out_channels"],
        heads=hp["gat_heads"],
        dropout_rate=hp["dropout_rate"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    bce = nn.BCEWithLogitsLoss()

    lambda_smooth_used = float(hp["lambda_smooth_init"])
    lambda_fixed = not hp.get("auto_lambda_smooth", True)
    last_total_loss = float("nan")

    for epoch in range(1, hp["num_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        out_tf, out_re, out_tg = model(graphs_sel, topo_order, parent)

        data_loss = torch.tensor(0.0, device=device)
        sum_loss_tr = 0.0
        sum_loss_rt = 0.0
        smooth_accum = torch.tensor(0.0, device=device)
        n_smooth_terms = 0

        for t in range(t_sel):
            h_tf = out_tf[:, t, :]
            h_re = out_re[:, t, :]
            h_tg = out_tg[:, t, :]

            e_tr = graphs_sel[t][("TF", "combine", "RE")].edge_index.to(device)
            if e_tr.numel() > 0:
                pos_i, pos_j = e_tr[0], e_tr[1]
                n_src, n_dst = h_tf.size(0), h_re.size(0)
                k_neg = int(hp["neg_sampling_ratio"] * max(1, pos_i.numel()))
                neg_i, neg_j = sample_negative_pairs(
                    n_src, n_dst, pos_i, pos_j, k_neg, device, dedup=hp["dedup_neg"]
                )
                logits = torch.cat(
                    [
                        pairwise_scores(h_tf, h_re, pos_i, pos_j),
                        pairwise_scores(h_tf, h_re, neg_i, neg_j),
                    ],
                    dim=0,
                )
                labels = torch.cat(
                    [
                        torch.ones(pos_i.numel(), device=device),
                        torch.zeros(neg_i.numel(), device=device),
                    ],
                    dim=0,
                )
                loss_tr = bce(logits, labels)
                data_loss = data_loss + loss_tr
                sum_loss_tr += loss_tr.item()

            e_rt = graphs_sel[t][("RE", "regulate", "TG")].edge_index.to(device)
            if e_rt.numel() > 0:
                pos_i, pos_j = e_rt[0], e_rt[1]
                n_src, n_dst = h_re.size(0), h_tg.size(0)
                k_neg = int(hp["neg_sampling_ratio"] * max(1, pos_i.numel()))
                neg_i, neg_j = sample_negative_pairs(
                    n_src, n_dst, pos_i, pos_j, k_neg, device, dedup=hp["dedup_neg"]
                )
                logits = torch.cat(
                    [
                        pairwise_scores(h_re, h_tg, pos_i, pos_j),
                        pairwise_scores(h_re, h_tg, neg_i, neg_j),
                    ],
                    dim=0,
                )
                labels = torch.cat(
                    [
                        torch.ones(pos_i.numel(), device=device),
                        torch.zeros(neg_i.numel(), device=device),
                    ],
                    dim=0,
                )
                loss_rt = bce(logits, labels)
                data_loss = data_loss + loss_rt
                sum_loss_rt += loss_rt.item()

            if parent[t] is not None and parent[t] >= 0:
                parent_idx = parent[t]
                h_tf_p = out_tf[:, parent_idx, :]
                h_re_p = out_re[:, parent_idx, :]
                h_tg_p = out_tg[:, parent_idx, :]
                smooth_accum = smooth_accum + frob_diff_of_outer(h_tf, h_re, h_tf_p, h_re_p)
                smooth_accum = smooth_accum + frob_diff_of_outer(h_re, h_tg, h_re_p, h_tg_p)
                n_smooth_terms += 2

        smooth_avg = smooth_accum / max(1, n_smooth_terms)
        if not lambda_fixed:
            if n_smooth_terms > 0 and smooth_avg.item() > 0 and data_loss.item() > 0:
                lambda_smooth_used = (
                    data_loss.detach() / (smooth_avg.detach() + 1e-12)
                ).item()
            else:
                lambda_smooth_used = float(hp["lambda_smooth_init"])
            lambda_fixed = True

        total_loss = data_loss + (lambda_smooth_used * smooth_avg)
        total_loss.backward()
        optimizer.step()

        last_total_loss = float(total_loss.detach())

        if epoch % hp["log_every"] == 0 or epoch == 1:
            smooth_contrib = float(lambda_smooth_used * smooth_avg.detach())
            total_val = float((data_loss + lambda_smooth_used * smooth_avg).detach())
            frac = (smooth_contrib / total_val) if total_val > 0 else 0.0
            emit_progress(
                "PROGRESS",
                "train:germ",
                f"epoch={epoch}/{hp['num_epochs']} loss={total_val:.4f} smooth_pct={frac*100:5.1f}%",
            )

    checkpoint_dir = ensure_directory(
        resolve_config_path(config, _section(config, "checkpoints").get("dir", "result/germ/model"))
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = checkpoint_dir / f"{stamp}.pt"
    checkpoint = {
        "timestamp": stamp,
        "seed": int(seed),
        "hp": hp,
        "in_channels": int(in_channels),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lambda_smooth_used": float(lambda_smooth_used),
        "topo_order": topo_order,
        "parent": parent,
    }
    torch.save(checkpoint, save_path)

    return {
        "lineage": "germ",
        "seed": int(seed),
        "hp": hp,
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "checkpoint_path": save_path.as_posix(),
        "timestamp": stamp,
        "lambda_smooth_used": float(lambda_smooth_used),
        "last_total_loss": float(last_total_loss),
        "T_sel": int(t_sel),
        "in_channels": int(in_channels),
        "topo_order": topo_order,
        "parent": parent,
    }


def train_lineage_model(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    lineage = str(config.get("lineage", "unknown")).lower()
    output_dir = ensure_directory(resolve_config_path(config, config.get("output_dir", f"result/{lineage}/train")))

    artifacts = build_lineage_graph_artifacts(config)
    common = artifacts.get("common")

    save_text_report(
        output_dir / "preprocess_summary.txt", "\n".join(artifacts["summary_lines"]) + "\n"
    )

    report: dict[str, Any] = {
        "lineage": lineage,
        "method_profile": config.get("method_profile"),
        "config_path": str(Path(config_path).resolve()),
        "input_paths": {"graphs": artifacts["source_graph_path"]} if common is None else summarize_input_paths(common["input_paths"]),
        "num_nodes": artifacts["num_nodes"],
        "maps_sizes": {key: len(value) for key, value in artifacts["maps"].items()},
        "device": str(artifacts["device"]),
        "summary_lines": artifacts["summary_lines"],
    }

    if artifacts["mode"] == "tree":
        report["lineage_count"] = len(artifacts["graphs_lineage"])
        report["in_channels"] = artifacts["in_channels"]
        report["time_names"] = artifacts["time_names"]
        report["topo_order"] = artifacts["topo_order"]
        report["parents_list"] = artifacts["parents_list"]
        report["topology_kind"] = artifacts["topology_kind"]
        report["model_kind"] = artifacts["model_kind"]
        report["training"] = _train_tree_artifacts(artifacts, config)
    elif artifacts["mode"] == "soma":
        report["germline_count"] = len(artifacts["graphs_germline"])
        report["somatic_count"] = len(artifacts["graphs_somatic"])
        report["in_channels_germline"] = artifacts["in_channels_germline"]
        report["in_channels_somatic"] = artifacts["in_channels_somatic"]
        report["germline_names"] = artifacts["germline_names"]
        report["somatic_names"] = artifacts["somatic_names"]
        report["training"] = _train_soma_artifacts(artifacts, config)
    else:
        report["germline_count"] = len(artifacts["graphs_germline"])
        report["in_channels"] = artifacts["in_channels"]
        report["germline_types"] = artifacts["germline_types"]
        report["topo_order"] = artifacts["topo_order"]
        report["parent"] = artifacts["parent"]
        report["training"] = _train_germ_artifacts(artifacts, config)

    report["status"] = "training_completed"
    save_checkpoint_metadata(output_dir / "train_report.json", report)
    return report
