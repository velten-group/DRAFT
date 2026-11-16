# Motif-prior example

A small Drosophila (dm6) input without `TF_RE.tsv` / `RE_TG.tsv`, for trying
motif-based prior preparation. It is a placeholder, not a dataset for biological
analysis; use your own files for a real run.

```bash
draft preprocess --input-dir data/motif_example --mode paired --genome dm6 --output prepared/motif_example
```

DRAFT downloads the Ensembl dm6 gene annotation and the JASPAR 2024 motif sites
overlapping the ATAC peaks (internet needed; cached in `data/resources/cache/`), and
writes the candidate relations to `prepared/motif_example/prior/` (`TF_RE.tsv`,
`RE_TG.tsv`) together with `graphs.pt` and `train.yaml` for training.

Contents: 120 germline cells in three stages (`Germline_0`, `Germline_5`,
`Germline_10`) from fly testis multiome data (GEO GSE277132), 761 TFs, 500 target
genes and the ATAC peaks within 20 kb of them (`dev/make_motif_example.py`).
