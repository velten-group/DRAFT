"""Small, explicit BED fixtures shared by tests and the upstream-source audit."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def write_inputs(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    tfs = ["TFa", "TFb", "TFc", "TFd", "TFe"]
    tgs = ["G1", "G2", "G3", "G4"]
    obs = pd.DataFrame(
        {"cell_type": ["early", "early", "late", "late"]},
        index=["r0", "r1", "r2", "r3"],
    )
    ad.AnnData(
        np.arange(36, dtype=float).reshape(4, 9),
        obs=obs,
        var=pd.DataFrame(index=tfs + tgs),
    ).write_h5ad(root / "rna.h5ad")
    obs.index = ["a0", "a1", "a2", "a3"]
    ad.AnnData(
        np.ones((4, 3)),
        obs=obs,
        var=pd.DataFrame(index=["chr1:0-100", "chr1:200-300", "chr1:400-500"]),
    ).write_h5ad(root / "atac.h5ad")
    for kind, genes in (("TF", tfs), ("TG", tgs)):
        pd.DataFrame({"gene_id": genes, "gene_name": genes}).to_csv(
            root / f"{kind}.tsv", sep="\t", index=False
        )
    pd.DataFrame(
        {"node_id": ["early", "late"], "parent_id": ["", "early"], "stage_id": [0, 1]}
    ).to_csv(root / "tree.csv", index=False)
    for node, intervals in (
        ("early", [(0, 100), (200, 300)]),
        ("late", [(0, 100), (400, 500)]),
    ):
        (root / f"{node}.narrowPeak").write_text(
            "".join(
                f"chr1\t{a}\t{b}\t{node}_p{i}\t10\t.\t1\t1\t1\t5\n"
                for i, (a, b) in enumerate(intervals)
            )
        )
    pd.DataFrame(
        {
            "node_id": ["early", "late"],
            "peak_file": ["early.narrowPeak", "late.narrowPeak"],
        }
    ).to_csv(root / "stage_peaks.tsv", sep="\t", index=False)
    (root / "motif_to_tf.tsv").write_text(
        "M1\tTFa::TFb\nM2\tTFc\nM3\tTFa\nM4\tTFd\nM5\tTFe\nMU\tUnmeasuredTF\n"
    )
    (root / "motifs.bed").write_text(
        "chr1\t10\t15\tM1\t2.0000001\t+\nchr1\t20\t25\tM2\t5\t-\nchr1\t30\t35\tM3\t8.0000004\t+\nchr1\t40\t45\tMU\t99\t+\nchr1\t100\t105\tM4\t1000\t+\nchr1\t210\t215\tM4\t10\t+\nchr1\t220\t225\tM5\t6\t-\nchr1\t410\t415\tM4\t11\t+\nchr1\t420\t425\tM5\t6\t+\n"
    )
    (root / "promoters.bed").write_text(
        "chr1\t50\t60\ttss1\t0\t+\tG1\nchr1\t70\t80\ttss2\t0\t-\tG2\nchr1\t80\t90\ttss3\t0\t+\tG1\nchr1\t260\t270\ttss4\t0\t+\tG3\nchr1\t450\t460\ttss5\t0\t+\tG4\n"
    )
    return {
        "config_version": 2,
        "lineage": "lineage1",
        "base_dir": str(root),
        "data": {
            "rna_h5ad": "rna.h5ad",
            "atac_h5ad": "atac.h5ad",
            "rna_layer": None,
            "tf_list": "TF.tsv",
            "tg_list": "TG.tsv",
            "tree": "tree.csv",
            "stage_peaks": "stage_peaks.tsv",
            "motif_instances": "motifs.bed",
            "motif_to_tf": "motif_to_tf.tsv",
            "promoters": "promoters.bed",
            "genome_assembly": "mm10",
        },
        "prior_preparation": {
            "method": "scmtni",
            "shared_gene_universe": True,
            "output_dir": "prepared",
        },
        "output_dir": str(root / "train"),
        "artifacts": {"dir": str(root / "artifacts")},
        "checkpoints": {"dir": str(root / "model")},
        "model": {"hidden_channels": 4, "out_channels": 4, "dropout_rate": 0},
        "training": {
            "num_epochs": 2,
            "seed": 1,
            "log_every": 1,
            "auto_lambda_smooth": False,
            "lambda_smooth_init": 0,
        },
        "network": {
            "output_dir": str(root / "network"),
            "min_score_keep": 0,
            "top_k_global": 20,
        },
        "reproduce": {"output_dir": str(root / "reproduce")},
        "pipeline": {"output_dir": str(root / "pipeline")},
    }
