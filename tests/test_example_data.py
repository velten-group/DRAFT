"""The committed example inputs in data/ match dev/make_example_data.py and the README commands."""

import importlib.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
TABLES = ["rna_cells.tsv", "atac_cells.tsv", "tree.tsv", "TF.tsv", "TG.tsv", "TF_RE.tsv", "RE_TG.tsv"]
# The source distribution leaves out data/; these checks need a git checkout.
pytestmark = pytest.mark.skipif(not (REPO / "data/lineage1").is_dir(), reason="example data not present")


def _generator():
    spec = importlib.util.spec_from_file_location("make_example_data", REPO / "dev/make_example_data.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_committed_example_matches_generator(tmp_path):
    generator = _generator()
    for lineage, seed in (("lineage1", 101), ("lineage2", 102)):
        generator.write_lineage(tmp_path / lineage, lineage, seed, 50)
        for name in TABLES:
            pd.testing.assert_frame_equal(
                pd.read_csv(REPO / "data" / lineage / name, sep="\t", dtype=str, keep_default_na=False),
                pd.read_csv(tmp_path / lineage / name, sep="\t", dtype=str, keep_default_na=False),
            )
        for name, layer in (("rna.h5ad", "lognorm"), ("atac.h5ad", "counts")):
            committed, fresh = ad.read_h5ad(REPO / "data" / lineage / name), ad.read_h5ad(tmp_path / lineage / name)
            assert list(committed.obs_names) == list(fresh.obs_names)
            assert list(committed.var_names) == list(fresh.var_names)
            np.testing.assert_allclose(committed.X.toarray(), fresh.X.toarray())
            np.testing.assert_allclose(committed.layers[layer].toarray(), fresh.layers[layer].toarray())


def test_motif_example_has_inputs_but_no_priors():
    folder = REPO / "data/motif_example"
    for name in ("rna.h5ad", "atac.h5ad", "rna_cells.tsv", "atac_cells.tsv", "tree.tsv", "TF.tsv", "TG.tsv"):
        assert (folder / name).is_file(), name
    assert not list(folder.glob("TF_RE*")) and not list(folder.glob("RE_TG*"))
    rna, atac = ad.read_h5ad(folder / "rna.h5ad"), ad.read_h5ad(folder / "atac.h5ad")
    assert list(rna.obs_names) == list(atac.obs_names)
    assert "lognorm" in rna.layers and "counts" in atac.layers
    assert all(name.count(":") == 1 for name in atac.var_names)


def test_readme_example_names_exist():
    for lineage in ("lineage1", "lineage2"):
        folder = REPO / "data" / lineage
        assert "TF1" in set(pd.read_csv(folder / "TF.tsv", sep="\t").gene_name)
        assert {"stage1", "stage2"} <= set(pd.read_csv(folder / "tree.tsv", sep="\t").node_id)
        tgs = set(pd.read_csv(folder / "TG.tsv", sep="\t").gene_name)
        pairs = pd.read_csv(REPO / "data/ligand_receptor.csv")
        assert set(pairs.ligand) | set(pairs.receptor) <= tgs
