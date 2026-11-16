import gzip
import io
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
import yaml

from draft_paired.prior_preparation import gene_annotation, peak_regions, prepare_prior
from draft_paired.prior_resources import RESOURCE_CATALOG, HTTPRangeReader, sha256
from draft_paired.utils import load_yaml_config
from draft_paired.workflow_cli import main


def inputs(root, mode, genome):
    root.mkdir(parents=True, exist_ok=True)
    chrom = "chr2L" if genome == "dm6" else "chr1"
    genes = ["IDTF1", "IDTF2", "IDTG1", "IDTG2"]
    groups = ["early", "early", "late", "late"]
    rna_cells = [f"r{i}" for i in range(4)]
    rna = ad.AnnData(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        obs=pd.DataFrame({"node_id": groups}, index=rna_cells),
        var=pd.DataFrame(index=genes),
    )
    rna.layers["lognorm"] = np.log1p(rna.X)
    rna.write_h5ad(root / "rna.h5ad")
    counts = np.array([[2, 0, 1], [3, 0, 1], [0, 2, 1], [0, 4, 1]], dtype=np.float32)
    atac_cells = rna_cells if mode == "paired" else [f"a{i}" for i in range(4)]
    if mode == "unpaired":
        counts = np.vstack([counts, [0, 1, 1]])
        groups = groups + ["late"]
        atac_cells.append("a4")
    atac = ad.AnnData(
        sp.csr_matrix(counts),
        obs=pd.DataFrame({"node_id": groups}, index=atac_cells),
        var=pd.DataFrame(
            index=[f"{chrom}-100-130", f"{chrom}-1000-1030", f"{chrom}-10000-10020"]
        ),
    )
    atac.layers["counts"] = atac.X.copy()
    atac.write_h5ad(root / "atac.h5ad")
    (root / "TF.tsv").write_text("gene_id\nIDTF1\nIDTF2\n")
    (root / "TG.tsv").write_text("gene_id\nIDTG1\nIDTG2\n")
    pd.DataFrame(
        {"node_id": ["early", "late"], "parent_id": ["", "early"], "stage_id": [0, 1]}
    ).to_csv(root / "tree.tsv", sep="\t", index=False)
    raw_chrom = chrom.removeprefix("chr")
    lines = []
    for gene, symbol, start, end, strand in [
        ("IDTF1.2", "Alpha", 5001, 5100, "+"),
        ("IDTF2.3", "Beta", 6001, 6100, "+"),
        ("IDTG1.1", "Target1", 111, 250, "+"),
        ("IDTG2.4", "Target2", 901, 1011, "-"),
    ]:
        for feature in ["gene", "transcript"]:
            lines.append(
                f'{raw_chrom}\tsource\t{feature}\t{start}\t{end}\t.\t{strand}\t.\tgene_id "{gene}"; gene_name "{symbol}";\n'
            )
    with gzip.open(root / "annotation.gtf.gz", "wt") as handle:
        handle.writelines(lines)
    (root / "motifs.bed").write_text(
        f"{chrom}\t105\t110\tM1\t800\t+\n{chrom}\t1005\t1010\tM2\t900\t-\n{chrom}\t10005\t10010\tM1\t700\t+\n"
    )
    (root / "motif_to_tf.tsv").write_text("M1\tAlpha\nM2\tBeta\n")
    return chrom


def config(root, mode, genome):
    path = root.parent / f"{mode}.yaml"
    assert (
        main(["init-config", "--mode", mode, "--genome", genome, "--output", str(path)])
        == 0
    )
    cfg = load_yaml_config(path)
    cfg["base_dir"] = str(root)
    cfg["data"].update(
        rna_h5ad="rna.h5ad",
        atac_h5ad="atac.h5ad",
        tf_list="TF.tsv",
        tg_list="TG.tsv",
        tree="tree.tsv",
    )
    cfg["resources"].update(
        gene_annotation="annotation.gtf.gz",
        motif_instances="motifs.bed",
        motif_to_tf="motif_to_tf.tsv",
    )
    cfg["prior_preparation"].update(min_accessible_fraction=0.5, promoter_flank=10)
    cfg["model"].update(hidden_channels=3, out_channels=3, gat_heads=1, dropout_rate=0)
    cfg["training"].update(
        num_epochs=2,
        auto_lambda_smooth=False,
        lambda_smooth_init=0,
        device="cpu",
        log_every=2,
    )
    cfg.update(output_dir="train", checkpoints={"dir": "model"})
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


@pytest.mark.parametrize("genome", ["hg38", "mm10", "dm6"])
@pytest.mark.parametrize("mode", ["paired", "unpaired"])
def test_shared_auto_prior_preprocess_and_training_for_all_genomes(
    tmp_path, genome, mode
):
    root = tmp_path / "inputs"
    chrom = inputs(root, mode, genome)
    path, cfg = config(root, mode, genome)
    before = {name: sha256(root / name) for name in ["rna.h5ad", "atac.h5ad"]}
    prior_dir = tmp_path / "prior_only"
    assert (
        main(["prepare-prior", "--config", str(path), "--output", str(prior_dir)]) == 0
    )
    prior = pd.read_csv(prior_dir / "tf_tg_prior.tsv", sep="\t")
    assert prior[["node_id", "TF", "TG"]].values.tolist() == [
        ["early", "IDTF1", "IDTG1"],
        ["late", "IDTF2", "IDTG2"],
    ]
    tf_re = pd.read_csv(prior_dir / "TF_RE.tsv", sep="\t")
    assert len(tf_re) == 3
    assert f"{chrom}:10000-10020" in set(tf_re.RE)
    tg_re = pd.read_csv(prior_dir / "RE_TG.tsv", sep="\t")
    assert len(tg_re) == 2
    report = json.loads((prior_dir / "prior_report.json").read_text())
    assert "existing ATAC feature intervals" in report["peak_source"]
    assert report["nodes"][1]["atac_cells"] == (2 if mode == "paired" else 3)
    assert prepare_prior(cfg, prior_dir)["reused"]
    output = tmp_path / "prepared"
    assert main(["preprocess", "--config", str(path), "--output", str(output)]) == 0
    bundle = torch.load(output / "graphs.pt", map_location="cpu", weights_only=False)
    assert bundle["input_mode"] == mode
    assert bundle["parents_list"] == [[], [0]]
    if mode == "paired":
        relation = ("TF", "combine", "RE")
        assert bundle["graphs"][0][relation].edge_index.tolist() == [[0, 0], [0, 2]]
        assert bundle["graphs"][1][relation].edge_index.tolist() == [[0, 1], [2, 1]]
        assert bundle["maps"]["re"][0] == f"{chrom}:100-130"
    else:
        relation = ("TF", "regulate", "TG")
        assert bundle["graphs"][0][relation].edge_index.tolist() == [[0], [0]]
        assert bundle["graphs"][1][relation].edge_index.tolist() == [[1], [1]]
    assert before == {name: sha256(root / name) for name in before}
    for name in before:
        (root / name).unlink()
    assert main(["train", "--config", str(output / "train.yaml")]) == 0
    training = json.loads((root / "train/train_report.json").read_text())
    assert Path(training["training"]["checkpoint_path"]).is_file()


def test_gene_coordinates_and_peak_coordinate_columns_are_unambiguous(tmp_path):
    inputs(tmp_path, "unpaired", "dm6")
    annotation = gene_annotation(tmp_path / "annotation.gtf.gz")
    assert annotation["IDTG1"]["tss"] == {110}
    assert annotation["IDTG2"]["tss"] == {1010}
    atac = ad.read_h5ad(tmp_path / "atac.h5ad")
    expected = peak_regions(atac)
    atac.var["chr"], atac.var["start"], atac.var["end"] = zip(*expected)
    atac.var_names = ["peak_a", "peak_b", "peak_c"]
    assert peak_regions(atac) == expected
    atac.var.loc["peak_b", ["start", "end"]] = [100, 130]
    with pytest.raises(ValueError, match="unique genomic intervals"):
        peak_regions(atac)


def test_auto_prior_rejects_wrong_pairing_and_does_not_replace_source_methods(tmp_path):
    root = tmp_path / "inputs"
    inputs(root, "unpaired", "hg38")
    _path, cfg = config(root, "paired", "hg38")
    with pytest.raises(ValueError, match="paired inputs also need matching cells"):
        prepare_prior(cfg, tmp_path / "prior")
    assert not (tmp_path / "prior").exists()
    cfg["prior_preparation"]["method"] = "scmtni"
    with pytest.raises(ValueError, match="requires method: auto_motif"):
        prepare_prior(cfg, tmp_path / "prior")


class Response(io.BytesIO):
    status = 206


def urlopen_for_bytes(content, calls):
    def open_request(request, timeout=None):
        calls.append(request.full_url)
        if request.full_url.endswith(".gtf.gz"):
            response = Response(content["gtf"])
            response.status = 200
            response.headers = {"Content-Length": str(len(content["gtf"]))}
            return response
        start, end = map(
            int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"]).groups()
        )
        blob = content["bb"]
        end = min(end, len(blob) - 1)
        response = Response(blob[start : end + 1])
        response.headers = {
            "Content-Range": f"bytes {start}-{end}/{len(blob)}",
            "ETag": '"fixture"',
        }
        return response

    return open_request


def test_http_ranges_are_bounded_and_reject_ignored_ranges(monkeypatch):
    calls = []
    blob = bytes(range(256)) * 4096
    monkeypatch.setattr(
        "urllib.request.urlopen", urlopen_for_bytes({"bb": blob}, calls)
    )
    with HTTPRangeReader(
        "https://test/motifs.bb", block_size=1024, cache_blocks=2
    ) as reader:
        reader.seek(100000)
        assert reader.read(3000) == blob[100000:103000]
        reader.seek(-10, io.SEEK_END)
        assert reader.read(20) == blob[-10:]
        assert len(reader.blocks) <= 2
        assert reader.bytes_downloaded < 10000

    def ignored(request, timeout=None):
        response = Response(blob)
        response.status, response.headers = 200, {}
        return response

    monkeypatch.setattr("urllib.request.urlopen", ignored)
    with pytest.raises(OSError, match="byte ranges"):
        HTTPRangeReader("https://test/motifs.bb")


def test_automatic_resource_selection_downloads_only_a_subset_and_reuses_cache(
    tmp_path, monkeypatch
):
    pybigtools = pytest.importorskip("pybigtools")
    root = tmp_path / "inputs"
    inputs(root, "unpaired", "hg38")
    _path, cfg = config(root, "unpaired", "hg38")
    bb = tmp_path / "motifs.bb"
    monkeypatch.chdir(tmp_path)
    writer = pybigtools.open("motifs.bb", "w")
    writer.write(
        {"chr1": 20000},
        [
            ("chr1", 105, 110, "M1\t800\t+\tAlpha"),
            ("chr1", 1005, 1010, "M2\t900\t-\tBeta"),
            ("chr1", 10005, 10010, "M1\t700\t+\tAlpha"),
        ],
    )
    cfg["resources"] = {"cache_dir": str(tmp_path / "cache")}
    monkeypatch.setitem(
        RESOURCE_CATALOG,
        "hg38",
        {
            "annotation_version": "fixture",
            "annotation_url": "https://test/annotation.gtf.gz",
            "motif_version": "fixture",
            "motif_url": "https://test/motifs.bb",
        },
    )
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        urlopen_for_bytes(
            {"gtf": (root / "annotation.gtf.gz").read_bytes(), "bb": bb.read_bytes()},
            calls,
        ),
    )
    result = prepare_prior(cfg, tmp_path / "prior")
    assert result["report"]["motif_resource"]["kind"] == "JASPAR_BED7_subset"
    assert any(url.endswith(".gtf.gz") for url in calls)
    assert any(url.endswith(".bb") for url in calls)

    def offline(*args, **kwargs):
        raise AssertionError(
            "Cached resources must be usable without downloading again."
        )

    monkeypatch.setattr("urllib.request.urlopen", offline)
    prepare_prior(cfg, tmp_path / "another_prior")
    (tmp_path / "prior/TF_RE.tsv").write_text("changed")
    with pytest.raises(ValueError, match="output changed"):
        prepare_prior(cfg, tmp_path / "prior")
