"""Unified unpaired DRAFT pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint import save_checkpoint_metadata
from .reproduce import reproduce_lineage_model
from .train import train_lineage_model
from .utils import ensure_directory, load_yaml_config, resolve_config_path


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _analysis_enabled(config: dict[str, Any], key: str) -> bool:
    return bool(_section(config, key).get("enabled", False))


def run_draft_pipeline(config_path: str | Path) -> dict[str, Any]:
    from .prior import prepare_training_config

    requested_config = config_path
    config_path, preparation = prepare_training_config(config_path)
    from .analysis.continuous_activity import run_continuous_activity
    from .analysis.edge_cluster import run_edge_cluster
    from .analysis.network import export_lineage_networks
    from .analysis.single_tf import run_single_tf_analysis
    from .analysis.topic import run_topic_analysis

    config = load_yaml_config(config_path)
    lineage_name = str(config.get("lineage", "unpaired")).strip() or "unpaired"
    pipeline_cfg = _section(config, "pipeline")
    output_dir = ensure_directory(
        resolve_config_path(config, pipeline_cfg.get("output_dir", f"result/{lineage_name}/pipeline"))
    )

    training = train_lineage_model(config_path)
    reproduction = reproduce_lineage_model(config_path)
    network = export_lineage_networks(config_path)

    analyses: dict[str, Any] = {}
    executed: list[str] = []
    skipped: list[str] = []
    if _analysis_enabled(config, "edge_cluster"):
        analyses["edge_cluster"] = run_edge_cluster(config)
        executed.append("edge_cluster")
    else:
        skipped.append("edge_cluster")
    if _analysis_enabled(config, "single_tf"):
        analyses["single_tf"] = run_single_tf_analysis(config)
        executed.append("single_tf")
    else:
        skipped.append("single_tf")
    if _analysis_enabled(config, "topic"):
        analyses["topic"] = run_topic_analysis(config)
        executed.append("topic")
    else:
        skipped.append("topic")
    if _analysis_enabled(config, "continuous_activity"):
        analyses["continuous_activity"] = run_continuous_activity(config)
        executed.append("continuous_activity")
    else:
        skipped.append("continuous_activity")

    report = {
        "lineage": lineage_name,
        "method_profile": config.get("method_profile"),
        "config_path": str(Path(config_path).resolve()),
        "requested_config_path": str(Path(requested_config).resolve()),
        "prior_preparation": {"report_path": str(Path(preparation["output_dir"]) / "prior_report.json"), "reused": preparation["reused"]} if preparation else None,
        "output_dir": str(output_dir),
        "training": training,
        "reproduction": reproduction,
        "network": network,
        "analyses": analyses,
        "executed_analyses": executed,
        "skipped_analyses": skipped,
        "status": "completed",
    }
    save_checkpoint_metadata(output_dir / "pipeline_report.json", report)
    return report


__all__ = ["run_draft_pipeline"]
