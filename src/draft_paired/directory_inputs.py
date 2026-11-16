"""Discover lineage inputs and configure preprocessing without editing YAML."""

from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from .config_init import DEFAULT_MIN_ACCESSIBLE_FRACTION, build_auto_config
from .input_schema import canonical_cell_annotation
from .utils import emit_progress, read_packaged_text, write_config_text

_FILES = {
    "rna_h5ad": (("rna",), (".h5ad",)),
    "atac_h5ad": (("atac",), (".h5ad",)),
    "tree": (("tree",), (".tsv", ".csv", ".txt")),
    "tf_list": (("TF",), (".tsv", ".csv", ".txt")),
    "tg_list": (("TG",), (".tsv", ".csv", ".txt")),
    "annotation": (
        ("rna_cells", "rna_annotation", "cell_annotation", "annotation", "cells"),
        (".tsv", ".csv", ".txt"),
    ),
    "atac_annotation": (("atac_cells", "atac_annotation"), (".tsv", ".csv", ".txt")),
    "tf_re_prior": (("TF_RE", "TF_RE_prior"), (".tsv", ".csv", ".txt")),
    "re_tg_prior": (("RE_TG", "RE_TG_prior"), (".tsv", ".csv", ".txt")),
    "re_list": (("RE",), (".txt",)),
}


def _find(root, key, prefixes, explicit=None, *, required=False):
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{key} is not a file: {path}")
        return path
    stems, suffixes = _FILES[key]
    names = {
        f"{prefix}{stem}{suffix}".casefold()
        for stem in stems
        for suffix in suffixes
        for prefix in ("", *(f"{value}_" for value in prefixes))
    }
    found = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name.casefold() in names
    )
    if len(found) > 1:
        raise ValueError(
            f"Ambiguous {key}: {', '.join(str(path) for path in found)}. "
            "Use an explicit file option or --config to select the intended input."
        )
    if found:
        return found[0].resolve()
    if required:
        examples = ", ".join(f"{stems[0]}{suffix}" for suffix in suffixes)
        raise FileNotFoundError(
            f"Missing {key} under {root}; expected {examples}, or supply its file option."
        )
    return None


def _metadata(path):
    import anndata as ad

    data = ad.read_h5ad(path, backed="r")
    try:
        frame = data.obs.copy()
        frame["cell_id"] = data.obs_names.astype(str)
        return frame, list(map(str, data.var_names))
    finally:
        data.file.close()


def prepare_directory_config(
    input_dir,
    output_dir,
    *,
    mode,
    lineage=None,
    genome=None,
    prior_source="auto",
    min_accessible_fraction=None,
    files=None,
    resources=None,
    force=False,
):
    """Resolve input files and save the effective config alongside graph inputs."""
    root = Path(input_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")
    if root == out:
        raise ValueError(
            "Choose an output directory different from the input directory."
        )
    if mode not in {"paired", "unpaired"}:
        raise ValueError("Directory preprocessing requires --mode paired or unpaired.")
    lineage = lineage or root.name
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", lineage):
        raise ValueError(
            "Pass --lineage with a name starting with a letter and using letters, digits, _ or -."
        )
    if genome is not None and genome not in {"hg38", "mm10", "dm6"}:
        raise ValueError("Choose genome hg38, mm10 or dm6.")
    if prior_source not in {"auto", "provided", "motif"}:
        raise ValueError("Choose prior_source auto, provided or motif.")
    minimum = (
        DEFAULT_MIN_ACCESSIBLE_FRACTION
        if min_accessible_fraction is None
        else float(min_accessible_fraction)
    )
    if not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError("min_accessible_fraction must be between 0 and 1.")
    for target in (out / "graphs.pt", out / "train.yaml"):
        if target.exists() and not force:
            raise FileExistsError(
                f"{target} already exists; choose another --output or use --force."
            )

    files = files or {}
    unknown = set(files) - set(_FILES)
    if unknown:
        raise ValueError(f"Unknown input fields: {sorted(unknown)}")
    prefixes = {root.name, lineage}
    required = {"rna_h5ad", "atac_h5ad", "tree", "tf_list", "tg_list"}
    paths = {
        key: _find(root, key, prefixes, files.get(key), required=key in required)
        for key in _FILES
        if prior_source != "motif"
        or key not in {"tf_re_prior", "re_tg_prior", "re_list"}
    }
    supplied = sum(paths.get(key) is not None for key in ("tf_re_prior", "re_tg_prior"))
    if prior_source == "provided" and supplied != 2:
        raise ValueError(
            "Provided priors require both TF_RE and RE_TG relation tables."
        )
    if prior_source == "auto" and supplied == 1:
        raise ValueError(
            "Found only one prior relation table. Supply both TF_RE and RE_TG, or choose --prior-source motif with --genome."
        )
    method = (
        "provided"
        if prior_source == "provided" or (prior_source == "auto" and supplied == 2)
        else "motif"
    )
    if method == "motif" and genome is None:
        raise ValueError(
            "No complete supplied priors selected. Pass --genome hg38, mm10 or dm6 to generate priors from motifs."
        )

    resources = {
        key: value for key, value in (resources or {}).items() if value is not None
    }
    if resources and method != "motif":
        raise ValueError("Motif resources require --prior-source motif and --genome.")
    if bool(resources.get("motif_instances")) != bool(resources.get("motif_to_tf")):
        raise ValueError("Provide both --motif-instances and --motif-to-tf.")
    if resources.get("motif_bigbed") and resources.get("motif_instances"):
        raise ValueError("Choose motif BigBed or BED plus mapping, not both.")
    for key, value in resources.items():
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{key} is not a file: {path}")
        resources[key] = str(path)

    if method == "motif":
        config = build_auto_config(mode=mode, genome=genome, lineage=lineage)
        config["resources"].update(resources)
        config["prior_preparation"]["min_accessible_fraction"] = minimum
    else:
        template = (
            "draft_lineage_template.yaml"
            if mode == "paired"
            else "draft_unpaired_template.yaml"
        )
        config = yaml.safe_load(
            read_packaged_text(f"draft_{mode}", f"templates/{template}")
        )
        if mode == "unpaired":
            config["prior_preparation"]["min_accessible_fraction"] = minimum
    config.update(
        lineage=lineage,
        input_mode=mode,
        base_dir=str(Path.cwd().resolve()),
        lineage_data_dir=str(root),
    )
    config["data"].update(
        {key: str(path) if path is not None else None for key, path in paths.items()}
    )
    if genome is not None:
        config["data"]["genome_assembly"] = genome

    generated = {}
    if mode == "paired" and method == "provided":
        config["data"]["annotation"] = config["data"].get("annotation") or config[
            "data"
        ].get("atac_annotation")
        if config["data"]["annotation"] is None:
            frame, _ = _metadata(paths["rna_h5ad"])
            target = out / "input_rna_cells.tsv"
            generated[target] = canonical_cell_annotation(
                frame, str(paths["rna_h5ad"])
            ).to_csv(sep="\t", index=False)
            config["data"]["annotation"] = str(target)
        if config["data"].get("atac_annotation") is None:
            config["data"]["atac_annotation"] = config["data"]["annotation"]
        if config["data"].get("re_list") is None:
            _, peaks = _metadata(paths["atac_h5ad"])
            target = out / "input_RE.txt"
            generated[target] = "\n".join(peaks) + "\n"
            config["data"]["re_list"] = str(target)

    config_path = out / "inputs.yaml"
    for target in (*generated, config_path):
        if target.exists() and not force:
            raise FileExistsError(
                f"{target} already exists; choose another --output or use --force."
            )
    protected = {path for path in paths.values() if path is not None} | {
        Path(value) for value in resources.values()
    }
    if protected.intersection(
        {*generated, config_path, out / "graphs.pt", out / "train.yaml"}
    ):
        raise ValueError("Preprocessing outputs must not overwrite source inputs.")
    out.mkdir(parents=True, exist_ok=True)
    for target, content in generated.items():
        target.write_text(content, encoding="utf-8")
    write_config_text(
        yaml.safe_dump(config, sort_keys=False), config_path, overwrite=force
    )
    emit_progress(
        "DONE", "inputs", f"{config_path} (lineage={lineage}, priors={method})"
    )
    return config_path


def preprocess_directory(input_dir, output_dir, **options):
    from .workflow import preprocess_lineage

    config_path = prepare_directory_config(input_dir, output_dir, **options)
    result = preprocess_lineage(
        config_path, options["mode"], output_dir, force=options.get("force", False)
    )
    result["input_config_path"] = str(config_path)
    return result
