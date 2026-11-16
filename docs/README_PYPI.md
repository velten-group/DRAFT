# DRAFT

DRAFT infers dynamic gene regulatory networks from paired or unpaired
single-cell RNA and ATAC data along a lineage tree, and compares two lineages
by developmental alignment (DTW) and cell-cell communication (CCC).

> **Status:** DRAFT is being packaged for PyPI. Until the release, install it
> from the GitHub repository as shown below. Commands and options may still change.

## Installation

Python 3.10 or newer:

```bash
git clone https://github.com/velten-group/DRAFT.git
cd DRAFT
pip install -e ".[full]"
```

## Quick start

Bundled simulation example (results in `result/simulation/`):

```bash
draft_paired init-config --kind simulation
draft_paired simulation --config configs/draft_simulation_template.yaml
```

Preprocess and train one lineage; the repository's `data/lineage1/` holds a small
example to replace with your own files (`--mode paired` for matched cells):

```bash
draft preprocess --input-dir data/lineage1 --mode unpaired --output prepared/lineage1
draft train --config prepared/lineage1/train.yaml
```

## Documentation

- [README](https://github.com/velten-group/DRAFT#readme): tutorial and
  [manuscript reproduction](https://github.com/velten-group/DRAFT#reproducing-the-manuscript)
- [Input guide](https://github.com/velten-group/DRAFT/blob/main/docs/input_formats.md)
- [Network export and downstream analyses](https://github.com/velten-group/DRAFT/blob/main/docs/downstream.md)
  and the [interactive example](https://github.com/velten-group/DRAFT/blob/main/docs/downstream.ipynb)
- Manuscript analyses and figures: [DRAFT-analysis](https://github.com/velten-group/DRAFT-analysis)

## License and contact

MIT. Questions and bug reports: [GitHub issues](https://github.com/velten-group/DRAFT/issues).
