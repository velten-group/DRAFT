# Network export and downstream analyses

Run these after [training](../README.md#2-train), from the same `DRAFT/`
directory. Commands use `draft_unpaired`; use `draft_paired` for a paired
model. Paths, TF and stage names are those of the bundled example in `data/`;
replace them with your own. If you trained with `--output result/my_run`, pass
`result/my_run/run_config.yaml` instead of `prepared/lineage1/train.yaml`.

## Paired and unpaired outputs

| Command | Paired | Unpaired |
|---|---|---|
| `network-export` | TF-TG edges plus supporting TF-RE-TG relations | TF-TG edges |
| `edge-cluster`, `topic` | TF-TG edges | TF-TG edges |
| `single-tf` | TF activity, target genes and supporting REs | TF activity and target genes |
| `ccc` | Either mode, or one of each | Either mode, or one of each |

## Export embeddings and networks

```bash
draft_unpaired reproduce --config prepared/lineage1/train.yaml
draft_unpaired network-export --config prepared/lineage1/train.yaml
```

| Output | Contents |
|---|---|
| `result/lineage1/reproduce/` | Embeddings, smoothness summaries and figures |
| `result/lineage1/network/` | `t*_TF_TG.tsv`: one TF-TG edge table per stage |

Each stage keeps TF-TG edges scoring at least `network.min_score_keep`. If
fewer than 1% of the stage's TF-TG pairs reach it, that stage keeps its top 1%
instead and a warning is printed; `--set network.min_edge_fraction=0` turns
this off (`network.dense.*` for interpolated networks).
Paired export needs the [TG coordinates](input_formats.md#tf-and-tg-lists).
Add `--dense` to also save interpolated networks in `result/lineage1/network_dense/`
(DTW and continuous activity compute them in memory otherwise).

## Cluster edges and regulatory topics

```bash
draft_unpaired edge-cluster --config prepared/lineage1/train.yaml
draft_unpaired topic --config prepared/lineage1/train.yaml
```

Both read `result/lineage1/network/`.

| Output | Main files |
|---|---|
| `result/lineage1/edge_cluster/` | `edge_clusters_*.csv` (edge → cluster), stage-pattern tables, TF/TG summaries, figures |
| `result/lineage1/topic/` | `topic_x_cluster_lda.csv`, `topic_x_cluster_prop_lda.csv`, figures |

**Optional enrichment.** Install `pip install -e ".[full,enrich]"`, then pass an
organism and an [Enrichr library](https://gseapy.readthedocs.io/en/latest/gseapy_example.html#enrichr-gene-set-libraries):

```bash
draft_unpaired topic --config prepared/lineage1/train.yaml --organism Human --gene-sets GO_Biological_Process_2023
```

This adds `topic_T*_enrich_*.csv` and figures to `result/lineage1/topic/`.

## Select TFs and export regulons

```bash
draft_unpaired single-tf --config prepared/lineage1/train.yaml --tf TF1 TF2 --stage stage1
```

`--tf` takes IDs or gene symbols; `--stage` takes a node name from your tree.
For each TF, `result/lineage1/single_tf/` receives `*_activity.tsv`,
`*_top_tg.tsv` and regulon figures.

### Consensus regulons

Combine target evidence over several stages:

```bash
draft_unpaired single-tf --config prepared/lineage1/train.yaml --tf TF1 --stage stage1 --consensus-stages stage1 stage2 stage3
```

This adds `*_consensus_all_targets.tsv`, `*_consensus_top_targets.tsv`,
`*_consensus_stage_ranks.tsv`, `*_consensus_regulon.png` and, for paired
models, `*_consensus_regulon_re.tsv`. To keep only targets in one edge cluster,
add `--edge-cluster-file result/lineage1/edge_cluster/edge_clusters_<name>.csv --edge-cluster 1`.

### TF activity over time

```bash
draft_unpaired continuous-activity --config prepared/lineage1/train.yaml --tf TF1 TF2
```

Writes `multi_tf_activity.tsv` and `Figure_MultiTF_Centrality.pdf` to
`result/lineage1/continuous_activity/`.

## Compare two lineages

### Inputs

| Input | Option |
|---|---|
| Two training configs | `--lineage1`, `--lineage2` |
| Ligand-receptor CSV ([format](input_formats.md#communication-between-two-lineages)) | `--lr` |
| Number of early stages in each lineage, leaving at least one later stage | `--early1`, `--early2` |

RNA and cell labels are read from each training config. Ligand and receptor
genes must be in both lineages' TG lists. Both datasets may reuse cell IDs and
stage names.

### Run DTW and CCC

```bash
draft_paired ccc --lineage1 prepared/lineage1/train.yaml --lineage2 prepared/lineage2/train.yaml --lr data/ligand_receptor.csv --early1 1 --early2 1 --output result/ccc
```

The command aligns the two network series with DTW, then scores each
ligand-receptor pair at the aligned stages. `--name1`/`--name2` rename the
lineages in the outputs.

| Output in `result/ccc/` | Contents |
|---|---|
| `time_warp_lineage1_to_lineage2.tsv` | DTW alignment path |
| `ccc_stage_alignment.tsv` | Stage-to-aligned-position mapping |
| `ccc_curves.tsv` | Scores by direction, pair and stage |
| `ccc_pairs.tsv` | Timing summary per pair and direction |
| `ccc_report.json`, `run_config.yaml` | Settings and provenance; rerun with `draft_paired ccc --config result/ccc/run_config.yaml` |

### Options

- **One combined RNA file:** add `--rna data/ccc/combined_rna.h5ad --labels data/ccc/cells.tsv --groups1 L1_early L1_middle L1_late --groups2 L2_early L2_middle L2_late`,
  with groups in model stage order and different group names in the two
  lineages. Use `--set data.gene_symbol_column=gene_name`
  if symbols are in RNA `var`, or `--set data.rna_layer=null` to use `.X`.
- **Reuse an alignment:** `--set ccc.alignment_path=result/ccc/time_warp_lineage1_to_lineage2.tsv`.
- **Reuse receptor scores:** `--set ccc.lineages.LINEAGE.receptor_scores=path/to/scores.tsv`
  ([format](input_formats.md#communication-between-two-lineages)).
- **Parallel terminal groups:** for example
  `--set 'ccc.lineages.lineage1.parallel_groups=[lineage1::late_A,lineage1::late_B]'`
  merges `late_A` and `late_B` into one endpoint (drop the `lineage1::` prefix
  with combined RNA).

## Command-line settings

`--output DIR` selects the output folder, and repeatable `--set KEY=VALUE`
changes any config value (YAML types, for example
`--set single_tf.save_fig=false`). The effective settings are saved as
`run_config.yaml` in the output folder. `--help` lists each command's options.

## Interactive example

[downstream.ipynb](downstream.ipynb) runs export, clustering and CCC on a small
simulated example, or on your own saved models. Install
`pip install -e ".[full,tutorial]"`, open the notebook, select that environment
as the kernel and run the cells in order. Results go to `result/downstream_example/`.
