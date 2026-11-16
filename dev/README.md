# Maintainer tools

Scripts for maintaining this repository. They are not needed to use DRAFT.

| Script | Purpose |
|---|---|
| `release_check.py` | Builds the sdist and wheel, checks them with twine, installs the wheel in a fresh environment and runs the command-line smoke tests (used by CI) |
| `validate_tutorials.py` | Executes `docs/downstream.ipynb` on synthetic data in an isolated workspace |
| `make_example_data.py` | Writes the example inputs `data/lineage1/`, `data/lineage2/` and `data/ligand_receptor.csv` |
| `make_motif_example.py` | Writes `data/motif_example/` from the fly inputs of the DRAFT-analysis deposit |
| `package_simulation_example.py` | Writes the bundled simulation example (`src/draft_paired/examples/`) |
| `validate_scmtni_prior.py` | Compares DRAFT's scMTNI prior scores with the upstream programs |
