# Input guide

The files `draft preprocess` reads, its options, and the inputs of the other
commands. Paired and unpaired data use the same files.

## Input files

Put each lineage in its own folder (`data/lineage1/`, `data/lineage2/`); files
are read in place. The repository's `data/lineage1/` and `data/lineage2/` hold
a small example in these formats.

| File | Required | Contents | One row per |
|---|---|---|---|
| `rna.h5ad` | Yes | [RNA expression](#rna-and-atac-h5ad-files) | Cell (columns: genes) |
| `atac.h5ad` | Yes | [ATAC accessibility](#rna-and-atac-h5ad-files) | Cell (columns: peaks) |
| `rna_cells.tsv`, `atac_cells.tsv` | Yes, or `cell_type` in the H5AD `obs` | [Tree node of each cell](#cell-labels) | Cell |
| `tree.tsv` | Yes | [Lineage tree](#lineage-tree) | Tree node |
| `TF.tsv`, `TG.tsv` | Yes | [TFs and target genes](#tf-and-tg-lists) | Gene |
| `TF_RE.tsv`, `RE_TG.tsv` | No; without them DRAFT [builds motif-based priors](#when-regulatory-priors-are-unavailable) | [Candidate TF-peak and peak-gene relations](#regulatory-priors) | Relation |

What differs between the modes:

| | Paired | Unpaired |
|---|---|---|
| Cells | RNA and ATAC share cell IDs; one label table serves both | Each assay has its own cells; every tree node needs cells in both |
| Tree | A node may have several parents | One parent per node |
| `TG.tsv` | With supplied priors, also gene coordinates (for network export) | Gene IDs and names |
| Priors | TF-RE and RE-TG edges of the graph; `RE.txt` (optional, one peak per line) selects the peaks used as REs, otherwise all ATAC peaks | Combined with the peaks open at each tree node into TF-TG candidates |

Tables: `.tsv`/`.txt` are tab-separated and `.csv` comma-separated, UTF-8,
with a header row unless stated otherwise.

## RNA and ATAC H5AD files

| Field | RNA | ATAC |
|---|---|---|
| `.X` or a layer | Cells x genes | Cells x peaks |
| `.obs_names` | Unique cell IDs | Unique cell IDs |
| `.var_names` | Unique gene IDs or symbols | Unique peak IDs |
| `.obs['cell_type']` | Optional cell labels | Optional cell labels |

- **RNA:** finite, nonnegative normalized values (for example log-normalized).
  The `lognorm` layer is used if present, otherwise `.X`
  (`--set data.rna_layer=null` forces `.X`).
- **ATAC:** finite, nonnegative counts or binary values, from the `counts`
  layer or `.X` (`--set data.atac_layer=null`). Dense or sparse.
- **Peak IDs:** `chr1:100000-100200` (zero-based, half-open). For motif-based
  priors, `chr1-100000-100200` or `chr`, `start`, `end` columns in ATAC `.var`
  also work. Use the assembly selected with `--genome`.

[AnnData format reference](https://anndata.readthedocs.io/en/stable/fileformat-prose.html).

## Cell labels

| Column | Meaning |
|---|---|
| `cell_id` | Cell ID from that H5AD's `.obs_names` |
| `cell_type` | Tree `node_id` of the cell (`node_id` is also accepted as the column name) |
| `time` | Optional; the tree's `stage_id` defines graph order |

```tsv
cell_id	cell_type
rna_01	early
rna_02	middle
rna_03	late_A
rna_04	late_B
```

Without a label table, preprocessing reads `.obs['cell_type']` or
`.obs['node_id']` from the H5AD.

## Lineage tree

| Column | Meaning |
|---|---|
| `node_id` | Unique node name, as used in the cell labels |
| `parent_id` | Parent `node_id`; empty for the root. Paired: several parents separated by `\|` |
| `stage_id` | Integer order; a parent is smaller than its child, parallel nodes may share a value |
| `branch` | Optional branch label |

```tsv
node_id	parent_id	stage_id	branch
early		0	shared
middle	early	1	shared
late_A	middle	2	A
late_B	middle	2	B
```

## TF and TG lists

One gene per row, with `gene_id` (matching RNA `.var_names`) and `gene_name`
(symbol; may equal `gene_id`):

```tsv
gene_id	gene_name
TP53	TP53
STAT3	STAT3
```

A single `gene_id` or `gene_name` column, or a headerless one-gene-per-line
list, also works. Priors may use IDs or unambiguous symbols from these tables.

- **Gene coordinates** (paired, with supplied priors): add `chr`, `start`,
  `end` and `strand` to `TG.tsv` (zero-based, half-open, same assembly as
  ATAC; the TSS is `start` on `+` and `end` on `-`). Motif-based priors add
  them automatically.

  ```tsv
  gene_id	gene_name	chr	start	end	strand
  gene_1	GENE1	chr1	100000	100500	+
  gene_2	GENE2	chr2	200000	200800	-
  ```

- **For CCC**, include the ligand and receptor genes in both lineages' TG lists.

## Regulatory priors

Two tables of candidate relations; `RE` is an ATAC peak ID, `TF`/`TG` come from
the gene lists.

```tsv
TF	RE
TP53	chr1:100000-100200
STAT3	chr2:200000-200300
```

```tsv
RE	TG
chr1:100000-100200	CDKN1A
chr2:200000-200300	MYC
```

## Preprocessing options

```bash
draft preprocess --input-dir data/lineage1 --mode paired --output prepared/lineage1
```

### File names

DRAFT finds the inputs in the folder by name:

- Names are matched case-insensitively and may start with a prefix such as
  `lineage1_`.
- Tables may end in `.tsv`, `.csv` or `.txt`.
- If several files match the same input, DRAFT stops with an error.

| Input | Accepted names | Option for another file |
|---|---|---|
| RNA, ATAC | `rna.h5ad`, `atac.h5ad` | `--rna`, `--atac` |
| Cell labels | `rna_cells`, `rna_annotation`, `cell_annotation`, `annotation` or `cells`; `atac_cells` or `atac_annotation` | `--rna-labels`, `--atac-labels` |
| Tree, gene lists | `tree`, `TF`, `TG` | `--tree`, `--tf-list`, `--tg-list` |
| Priors | `TF_RE` or `TF_RE_prior`, `RE_TG` or `RE_TG_prior` | `--tf-re`, `--re-tg` |
| RE list (paired) | `RE.txt` | `--re-list` |

The lineage name is the folder name; `--lineage NAME` sets another one
(letters, digits, `_` or `-`, starting with a letter). `--force` overwrites an
existing output folder.

### Prior source

With `--prior-source auto` (the default):

| Prior files in the folder | Result |
|---|---|
| `TF_RE` and `RE_TG` | These priors are used |
| Neither | Motif-based priors are built; `--genome` is required |
| Only one of them | Error |

`--prior-source provided` or `--prior-source motif` forces one source.
`--min-accessible-fraction` (0–1, default 0.05) is the fraction of a node's
cells in which a peak must be open to count at that node; it applies to
motif-based priors and to unpaired TF-TG candidates.

### Outputs

| Output | Contents |
|---|---|
| `graphs.pt` | Model input: one graph per tree node in topological order, with the node names, parents and row identifiers |
| `train.yaml` | Config of `draft train` and every later command |
| `inputs.yaml` | Resolved input files and settings |
| `prior/` | Priors built by DRAFT (motif-based priors; unpaired TF-TG candidates per node) |

`graphs.pt` is a Python pickle: load only files you trust.

### Settings

`--set KEY=VALUE` changes a config value on any command, for example
`--set training.num_epochs=300` on `draft preprocess` or `draft train`. To
change several settings, edit `prepared/lineage1/inputs.yaml` and preprocess
again from it: `draft preprocess --config prepared/lineage1/inputs.yaml --output prepared/lineage1_v2`.

## When regulatory priors are unavailable

Preprocess with `--genome hg38`, `mm10` or `dm6` (paired or unpaired). DRAFT
selects open peaks per node, assigns pre-scanned motif sites to TFs and links
peaks to genes by TSS windows. The [`data/motif_example/`](../data/motif_example/README.md)
folder is a small fly input for trying this:

```bash
draft preprocess --input-dir data/motif_example --mode paired --genome dm6 --output prepared/motif_example
```

The output folder also receives `prior/TF_RE.tsv` (`TF,RE,motif_score`),
`prior/RE_TG.tsv` (`RE,TG`), the same relations per node
(`prior/*_by_stage.tsv`) and the per-node TF-TG candidates
(`prior/tf_tg_prior.tsv`). `draft prepare-prior` with the same options builds
only the priors.

### Automatic motif downloads

On first use (internet required), DRAFT queries JASPAR 2024 pre-scanned motif
sites for your peaks and TFs and downloads the gene annotation. Files are
cached in `data/resources/cache/` (`--set resources.cache_dir=PATH` changes it).

| Assembly | Motif sites | Gene annotation |
|---|---|---|
| hg38 | [JASPAR 2024 BigBed](https://mencius-pub.uio.no/JASPAR/JASPAR_TFBSs/2024/JASPAR2024_hg38.bb) | [GENCODE 44 GTF](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz) |
| mm10 | [JASPAR 2024 BigBed](https://mencius-pub.uio.no/JASPAR/JASPAR_TFBSs/2024/JASPAR2024_mm10.bb) | [GENCODE M25 GTF](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz) |
| dm6 | [JASPAR 2024 BigBed](https://mencius-pub.uio.no/JASPAR/JASPAR_TFBSs/2024/JASPAR2024_dm6.bb) | [Ensembl 112 GTF](https://ftp.ensembl.org/pub/release-112/gtf/drosophila_melanogaster/Drosophila_melanogaster.BDGP6.46.112.gtf.gz) |

TF names must match the JASPAR motif-to-TF names. The same GTF's `gene_id`
and `gene_name` attributes can be used to build `TF.tsv` and `TG.tsv`.

### Use your own motif resources

```bash
draft preprocess --input-dir data/lineage1 --mode paired --genome hg38 --prior-source motif --motif-instances data/resources/my_motif_hits.bed.gz --motif-to-tf data/resources/my_motif_to_tf.tsv --gene-annotation data/resources/my_genes.gtf.gz --output prepared/lineage1
```

Give all three files for offline use (`--gene-annotation` is otherwise downloaded).

- **Motif sites** (`.bed` or `.bed.gz`, headerless BED6): chromosome,
  zero-based start, end, motif ID, score, strand. Change the score cutoff with
  `--set prior_preparation.min_motif_score=VALUE`.

  ```text
  chr1	100020	100032	motif_A	800	+
  ```

- **Motif-to-TF table** (headerless, 2 columns): motif ID, TF name or ID from
  the TF list; several TFs as `TF1::TF2`.

  ```text
  motif_A	TP53
  ```

- **Gene annotation:** standard GTF (`.gtf` or `.gtf.gz`) for the same
  assembly, with `gene` or `transcript` rows and `gene_id`/`gene_name`
  attributes.
- **Alternative:** `--motif-bigbed FILE.bb` in the JASPAR BED7 BigBed layout
  (its TF-name column provides the mapping) replaces the two motif files.

Supply genome-wide motif hits; scan PWM/MEME libraries against the reference
genome first.

## Other inputs

### Called peaks per tree node

<details>
<summary>Unpaired priors from per-node peak calls, following the scMTNI rules</summary>

In addition to the files of the lineage folder:

| File | Format | Example |
|---|---|---|
| Stage-peak manifest | TSV `node_id,peak_file`, one row per tree node | `data/lineage1/stage_peaks.tsv` |
| Called peaks | Headerless narrowPeak (10 columns), one file per node | `data/lineage1/peaks/early.narrowPeak` |
| Motif sites and mapping | As [above](#use-your-own-motif-resources) | `data/resources/motif_hits.bed.gz`, `motif_to_tf.tsv` |
| Promoters | Headerless BED4 `chrom,start,end,gene_name` with TG symbols | `data/resources/promoters.bed` |
| Gene scopes | Optional TSV `node_id,TF` and `node_id,TG`: the genes measured at each node; otherwise set a shared gene universe | `data/lineage1/node_tf_scope.tsv` |

```bash
draft_unpaired init-config --kind motif --lineage lineage1 --output configs/unpaired_motifs.yaml
draft preprocess --mode unpaired --config configs/unpaired_motifs.yaml --set data.genome_assembly=hg38 --set data.motif_instances=data/resources/motif_hits.bed.gz --set data.motif_to_tf=data/resources/motif_to_tf.tsv --set data.promoters=data/resources/promoters.bed --set prior_preparation.shared_gene_universe=true --output prepared/lineage1
```

The config reads the files of `data/lineage1/`, including `stage_peaks.tsv`.
Scores follow the
[scMTNI prior generation](https://github.com/Roy-lab/scMTNI/tree/c4d4850629f3529cc7a8b4dae54d91daf0702809/Scripts/genPriorNetwork).
Peak-file paths resolve from the working directory; use one genome assembly
for all files.

</details>

### Simulation priors

A CSV/TSV with `TF,TG` columns, one candidate edge per row (optional
nonnegative `score` or `weight`; positive marks a candidate). A matrix with TG
names in the first column and TF names as headers also works. TF and TG names
must not overlap, with at least two connected genes in each role.

```tsv
TF	TG
TF1	TG1
TF1	TG2
TF2	TG1
TF2	TG2
```

To simulate RNA/ATAC data from such a prior and train on it (needs R 4.4 or
newer):

```bash
Rscript scripts/install_simulation_r.R .R-library
draft simulation --prior-grn data/simulation/prior.tsv --output result/my_simulation --r-library .R-library
```

### Communication between two lineages

**Ligand-receptor CSV** (`--lr`), one pair per row:

```csv
ligand,receptor
CXCL12,CXCR4
TGFB1,TGFBR1
```

`source_genesymbol,target_genesymbol` columns (as in the
[LIANA resource](https://raw.githubusercontent.com/saezlab/liana-py/main/src/liana/resource/omni_resource.csv))
also work. Curated pairs: [CellTalkDB](https://xomics.com.cn/celltalkdb/download.php)
(human, mouse), [FlyPhoneDB2](https://www.flyrnai.org/tools/fly_phone_v2/web/annotation)
(Drosophila). Both genes of a pair must be in both lineages' TG lists and in
the RNA.

Symbols match without regard to case (`CXCR4` = `Cxcr4`), except for genes
that differ only by case in your TF/TG lists, such as Drosophila `Tor` and
`tor`: each spelling is its own gene, and other spellings (`TOR`) are skipped
and listed under `lr_symbols_ambiguous_by_case` in `ccc_report.json`. Output
tables write symbols in lowercase except for these genes.

**Receptor-score table** (optional, to reuse scores): first column gene symbol,
then one column per stage with percentiles in `[0,1]`:

```tsv
gene_name	early	middle	late
CXCR4	0.8	0.6	0.4
```

**Alignment table** (optional, to reuse a DTW path): columns
`<lineage1>_tau` and `<lineage2>_tau` with positions in `[0,1]`, as written to
`result/ccc/`. How to pass both tables is described in
[`draft ccc`](downstream.md#draft-ccc).
