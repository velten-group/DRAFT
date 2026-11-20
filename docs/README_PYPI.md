# DRAFT

DRAFT infers dynamic gene regulatory networks from paired or unpaired
single-cell RNA and ATAC data along a lineage tree, and compares two lineages
by developmental alignment (DTW) and cell-cell communication (CCC).

> **Status:** early release; commands and options may still change.

## Installation

Python 3.10 or newer:

```bash
pip install "draft-grn[full]"
draft simulation      # quick start on bundled simulated data; results in result/simulation/
```

Optional extras: `[enrich]` adds GO enrichment of regulatory topics, `[tutorial]`
runs the interactive example. The tutorial's example inputs (`data/`) and the
manuscript reproduction script are in the GitHub repository; to use them, install
from a clone:

```bash
git clone https://github.com/velten-group/DRAFT.git
cd DRAFT
pip install -e ".[full]"
```

## Documentation

- [README](https://github.com/velten-group/DRAFT#readme): tutorial on example
  data (preprocess, train, networks, regulons, two-lineage comparison) and
  [manuscript reproduction](https://github.com/velten-group/DRAFT#reproducing-the-manuscript)
- [Input guide](https://github.com/velten-group/DRAFT/blob/main/docs/input_formats.md)
- [Downstream analyses](https://github.com/velten-group/DRAFT/blob/main/docs/downstream.md)
- Manuscript analyses and figures: [DRAFT-analysis](https://github.com/velten-group/DRAFT-analysis)

## Citation

Zhang Z, Santhi Sekhar P, van Nierop y Sanchez P, Lohmann I, Velten B. DRAFT: graph-recurrent learning of
dynamic gene regulatory networks along single-cell multi-omics trajectories. bioRxiv (2026).
[doi:10.64898/2026.11.17.767145](https://doi.org/10.64898/2026.11.17.767145)

## License and contact

MIT. Questions and bug reports: [GitHub issues](https://github.com/velten-group/DRAFT/issues).
