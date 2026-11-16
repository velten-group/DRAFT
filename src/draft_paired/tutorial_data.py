"""Small, explicitly synthetic inputs for executable workflow tutorials.

Use subsets of bundled simulated observations and candidate priors.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path


def prepare_demo_data(
    output_dir: str | Path, *, mode: str, lineage: str = "lineage1", seed: int = 101
) -> Path:
    import anndata as ad
    import numpy as np
    import pandas as pd
    import yaml

    if mode not in {"paired", "unpaired"} or seed not in {101, 102, 103}:
        raise ValueError("Choose paired/unpaired and a bundled simulation seed.")
    content = (
        resources.files("draft_paired")
        .joinpath(f"examples/simulation_9plus3/inputs_{seed}.npz")
        .read_bytes()
    )
    identity = {
        "kind": "synthetic_tutorial_v2",
        "mode": mode,
        "lineage": lineage,
        "seed": seed,
        "source_sha256": hashlib.sha256(content).hexdigest(),
    }
    root = Path(output_dir).resolve()
    marker, config_path = root / "tutorial_data.json", root / "workflow.yaml"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(
                "Tutorial data identity changed; choose a different output directory."
            )
        return config_path
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-tutorial directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    import io

    with np.load(io.BytesIO(content), allow_pickle=False) as source:
        raw_rna = np.concatenate([x[:, :16].T for x in source["rna_counts"]]).astype(
            "float32"
        )
        raw_atac = np.concatenate([x[:, :16].T for x in source["atac_counts"]]).astype(
            "float32"
        )
        tr, rt, candidate = source["tf_re"], source["re_tg"], source["candidate"]
    ntg, ntf = candidate.shape
    ids = [f"gene_{i + 1}" for i in range(ntf + ntg)]
    symbols = [f"TF{i + 1}" for i in range(ntf)] + [f"TG{i + 1}" for i in range(ntg)]
    stage_names = [f"{lineage}_stage_{t}" for t in range(5)]
    cells = [f"{lineage}_s{t}_cell{i}" for t in range(5) for i in range(16)]
    annotation = pd.DataFrame(
        {
            "cell_id": cells,
            "cell_type": np.repeat(stage_names, 16),
            "time": np.repeat(np.arange(5), 16),
        }
    )
    obs = annotation.set_index("cell_id")
    rna = ad.AnnData(
        raw_rna, obs=obs, var=pd.DataFrame({"gene_name": symbols}, index=ids)
    )
    rna.layers["lognorm"] = np.log1p(
        raw_rna * 10000 / np.maximum(raw_rna.sum(1, keepdims=True), 1)
    )
    rna.write_h5ad(root / "rna.h5ad")
    annotation.to_csv(root / "annotation.csv", index=False)
    pd.DataFrame(
        {
            "node_id": stage_names,
            "parent_id": [""] + stage_names[:-1],
            "stage_id": range(5),
        }
    ).to_csv(root / "tree.csv", index=False)
    pd.DataFrame({"gene_id": ids[:ntf], "gene_name": symbols[:ntf]}).to_csv(
        root / "TF.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "gene_id": ids[ntf:],
            "gene_name": symbols[ntf:],
            "chr": "chrDemo",
            "start": np.arange(ntg) * 10000 + 1000,
            "end": np.arange(ntg) * 10000 + 2000,
            "strand": "+",
        }
    ).to_csv(root / "TG.tsv", sep="\t", index=False)
    from .utils import read_packaged_text

    family_template = "lineage" if mode == "paired" else "unpaired"
    cfg = yaml.safe_load(
        read_packaged_text(
            f"draft_{mode}", f"templates/draft_{family_template}_template.yaml"
        )
    )
    cfg.update(
        lineage=lineage,
        base_dir=str(root),
        lineage_data_dir=str(root),
        result_root=str(root / "result"),
    )
    paths = {
        "rna_h5ad": "rna.h5ad",
        "annotation": "annotation.csv",
        "tree": "tree.csv",
        "tf_list": "TF.tsv",
        "tg_list": "TG.tsv",
    }
    target = np.argmax(rt, axis=0)
    regions = [
        f"chrDemo:{int(g) * 10000 + 1100 + int(r % 6) * 100}-{int(g) * 10000 + 1150 + int(r % 6) * 100}"
        for r, g in enumerate(target)
    ]
    atac_obs = obs.copy()
    if mode == "unpaired":
        selected = np.concatenate([np.arange(t * 16, (t + 1) * 16, 2) for t in range(5)])
        raw_atac = raw_atac[selected]
        atac_obs = obs.iloc[selected].copy()
        atac_obs.index = [f"{lineage}_atac_{i}" for i in range(len(selected))]
        cfg["prior_preparation"]["min_accessible_fraction"] = 0.05
    atac = ad.AnnData(
        np.log1p(raw_atac), obs=atac_obs, var=pd.DataFrame(index=regions)
    )
    atac.layers["counts"] = raw_atac
    atac.write_h5ad(root / "atac.h5ad")
    atac.obs.reset_index(names="cell_id").to_csv(root / "atac_annotation.csv", index=False)
    (root / "RE.txt").write_text("\n".join(regions) + "\n")
    re, tf = np.nonzero(tr)
    pd.DataFrame(
        {"TF": np.asarray(symbols[:ntf])[tf], "RE": np.asarray(regions)[re]}
    ).to_csv(root / "TF_RE.tsv", sep="\t", index=False)
    tg, re = np.nonzero(rt)
    pd.DataFrame(
        {"TG": np.asarray(ids[ntf:])[tg], "RE": np.asarray(regions)[re]}
    ).to_csv(root / "RE_TG.tsv", sep="\t", index=False)
    paths.update(
        atac_h5ad="atac.h5ad",
        atac_annotation="atac_annotation.csv",
        tf_re_prior="TF_RE.tsv",
        re_tg_prior="RE_TG.tsv",
    )
    if mode == "paired":
        paths["re_list"] = "RE.txt"
    cfg["data"].update({key: str(root / value) for key, value in paths.items()})
    cfg["model"].update(hidden_channels=8, out_channels=8, dropout_rate=0)
    cfg["training"].update(
        num_epochs=2, log_every=1, auto_lambda_smooth=False, lambda_smooth_init=0
    )
    cfg["network"].update(top_k_global=200, min_score_keep=0)
    cfg["network"]["dense"].update(
        n_interp=1, top_k_global=200, min_score_keep=0, rescue_m_max=0
    )
    if mode == "paired":
        cfg["network"]["strict_score_mode"] = "score_cos_prob"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    marker.write_text(json.dumps(identity, indent=2))
    return config_path


def prepare_demo_ccc(config_paths: list[str | Path], output_dir: str | Path) -> Path:
    """Prepare RNA, labels and ligand-receptor inputs for two paired demos."""
    import anndata as ad
    import pandas as pd
    import yaml
    from .utils import load_yaml_config

    if len(config_paths) != 2:
        raise ValueError("Provide exactly two lineage configs.")
    configs = [load_yaml_config(path) for path in config_paths]
    if len({c["lineage"] for c in configs}) != 2:
        raise ValueError("The two lineage identifiers must be distinct.")
    root = Path(output_dir).resolve()
    identity = {
        "kind": "synthetic_ccc_tutorial_v1",
        "configs": [
            hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in config_paths
        ],
    }
    marker = root / "tutorial_data.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(
                "CCC tutorial inputs changed; choose a new output directory."
            )
        return root / "ccc.yaml"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Choose a new CCC tutorial output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    datasets = [ad.read_h5ad(c["data"]["rna_h5ad"]) for c in configs]
    rna = ad.concat(datasets, merge="same")
    rna.write_h5ad(root / "rna.h5ad")
    rna.obs.reset_index().rename(columns={"index": "cell_id"}).to_csv(
        root / "annotation.csv", index=False
    )
    pd.DataFrame({"ligand": ["TG1", "TG3"], "receptor": ["TG2", "TG4"]}).to_csv(
        root / "LR.csv", index=False
    )
    lineages = {}
    for path, cfg in zip(config_paths, configs):
        tree = pd.read_csv(cfg["data"]["tree"])
        lineages[cfg["lineage"]] = {
            "input_mode": "paired",
            "config": str(Path(path).resolve()),
            "tf_list": cfg["data"]["tf_list"],
            "tg_list": cfg["data"]["tg_list"],
            "stage_groups": tree.node_id.tolist(),
            "early_stage_count": 2,
        }
    config = {
        "base_dir": str(root),
        "data": {
            "rna_h5ad": str(root / "rna.h5ad"),
            "annotation": str(root / "annotation.csv"),
            "gene_symbol_column": "gene_name",
            "rna_layer": "lognorm",
            "group_column": "cell_type",
            "cell_id_normalization": "exact",
        },
        "ccc": {
            "lineage_order": list(lineages),
            "lineages": lineages,
            "ccc_csv": str(root / "LR.csv"),
            "output_dir": str(root / "result"),
            "save_fig": True,
        },
    }
    path = root / "ccc.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    marker.write_text(json.dumps(identity, indent=2))
    return path
