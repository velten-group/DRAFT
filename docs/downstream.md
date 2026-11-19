# Downstream analyses

Outputs and options of the commands that follow [training](../README.md#2-train).
Each command takes the training config, `--config prepared/lineage1/train.yaml`,
and works the same for paired and unpaired models. Paths, TF and stage names
below are those of the example in `data/`.

| Command | Output in `result/lineage1/` | Paired | Unpaired |
|---|---|---|---|
| `draft reproduce` | `reproduce/` | Embeddings per stage, temporal smoothness | Same |
| `draft network-export` | `network/` | TF-TG edges per stage, with supporting TF-RE-TG relations | TF-TG edges per stage |
| `draft edge-cluster` | `edge_cluster/` | Edge clusters by stage pattern | Same |
| `draft topic` | `topic/` | Regulatory topics | Same |
| `draft single-tf` | `single_tf/` | TF activity, target genes, regulons with supporting REs | TF activity, target genes, regulons |
| `draft continuous-activity` | `continuous_activity/` | TF activity along interpolated time | Same |
| `draft ccc` | `result/ccc/` | Two lineages of either mode | Same |

## `draft reproduce`

Writes `embedding_<stage>_<TF|RE|TG>.txt` (paired; TF and TG for unpaired),
the stage-by-stage distance matrix of the TF-TG networks
(`tf_tg_prob_frobenius_dist_matrix.tsv`) and its heatmap
(`smoothness_tf_tg_heatmap.png`).

## `draft network-export`

Writes `t<stage>_TF_TG.tsv`, one TF-TG edge table per stage (paired also
`t<stage>_TF_RE_TG.tsv`), and per-stage counts of active genes and kept edges.

- Each stage keeps the TF-TG edges scoring at least `network.min_score_keep`.
  If fewer than 1% of the stage's TF-TG pairs reach it, that stage keeps its
  top 1% instead and a warning is printed; `--set network.min_edge_fraction=0`
  turns this off (`network.dense.*` for interpolated networks).
- `--dense` also saves interpolated networks in `network_dense/`
  (`draft ccc` and `draft continuous-activity` compute them in memory otherwise).
- Paired export needs the [TG coordinates](input_formats.md#tf-and-tg-lists).

## `draft edge-cluster` and `draft topic`

Both read `network/`.

| Output | Main files |
|---|---|
| `edge_cluster/` | `edge_clusters_*.csv` (edge → cluster), stage-pattern tables, TF/TG summaries, figures |
| `topic/` | `topic_x_cluster_lda.csv`, `topic_x_cluster_prop_lda.csv`, figures |

**GO enrichment of topics.** Install `pip install -e ".[full,enrich]"`, then
pass an organism and an [Enrichr library](https://gseapy.readthedocs.io/en/latest/gseapy_example.html#enrichr-gene-set-libraries):

```bash
draft topic --config prepared/lineage1/train.yaml --organism Human --gene-sets GO_Biological_Process_2023
```

This adds `topic_T*_enrich_*.csv` and figures to `topic/`.

## `draft single-tf`

`--tf` takes one or more TF IDs or gene symbols; `--stage` a node name from
your tree. For each TF, `single_tf/` receives `<TF>_activity.tsv`,
`<TF>_top_tg.tsv` and regulon figures.

`--consensus-stages stage1 stage2 ...` combines target evidence over these
stages and adds `<TF>_consensus_all_targets.tsv`,
`<TF>_consensus_top_targets.tsv`, `<TF>_consensus_stage_ranks.tsv`,
`<TF>_consensus_regulon.png` and, for paired models,
`<TF>_consensus_regulon_re.tsv`. To keep only targets in one edge cluster, add
`--edge-cluster-file result/lineage1/edge_cluster/edge_clusters_<name>.csv --edge-cluster 1`.

## `draft continuous-activity`

```bash
draft continuous-activity --config prepared/lineage1/train.yaml --tf TF1 TF2
```

Writes `multi_tf_activity.tsv` and `Figure_MultiTF_Centrality.pdf` to
`continuous_activity/`.

## `draft ccc`

| Input | Option |
|---|---|
| Two training configs | `--lineage1`, `--lineage2` |
| Ligand-receptor CSV ([format](input_formats.md#communication-between-two-lineages)) | `--lr` |
| Number of early stages in each lineage, leaving at least one later stage | `--early1`, `--early2` |

RNA and cell labels are read from each training config. Both datasets may
reuse cell IDs and stage names. The command aligns the two network series
with DTW, then scores each ligand-receptor pair at the aligned stages.
`--name1`/`--name2` rename the lineages in the outputs.

| Output in `result/ccc/` | Contents |
|---|---|
| `time_warp_lineage1_to_lineage2.tsv` | DTW alignment path |
| `ccc_stage_alignment.tsv` | Stage-to-aligned-position mapping |
| `ccc_curves.tsv` | Scores by direction, pair and stage |
| `ccc_pairs.tsv` | Timing summary per pair and direction |
| `ccc_report.json`, `run_config.yaml` | Settings and inputs; run again with `draft ccc --config result/ccc/run_config.yaml` |

Options:

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

## Settings of every command

`--output DIR` selects the output folder, and repeatable `--set KEY=VALUE`
changes any config value (YAML types, for example
`--set single_tf.save_fig=false`). The effective settings are saved as
`run_config.yaml` in the output folder; after `draft train --output result/my_run`,
pass `result/my_run/run_config.yaml` to the later commands in place of
`train.yaml`. `draft <command> --help` lists each command's options.

## Interactive example

[downstream.ipynb](downstream.ipynb) runs export, clustering and CCC on a small
simulated example, or on your own saved models. Install
`pip install -e ".[full,tutorial]"`, open the notebook, select that environment
as the kernel and run the cells in order. Results go to `result/downstream_example/`.
