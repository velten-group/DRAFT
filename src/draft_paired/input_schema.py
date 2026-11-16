"""Shared file conventions for paired and unpaired data preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import strip_copy_suffix


def read_table(path: str | Path, *, header: int | None = 0) -> pd.DataFrame:
    """CSV uses commas; TSV/TXT uses tabs. Keep identifiers as strings."""
    path = Path(path)
    frame = pd.read_csv(
        path,
        sep="," if path.suffix.lower() == ".csv" else "\t",
        header=header,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if header is not None:
        frame.columns = [str(c).strip() for c in frame.columns]
        if frame.columns.duplicated().any():
            raise ValueError(f"{path} contains duplicate column labels.")
    return frame


def find_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    columns = {str(c).strip().lower(): str(c) for c in frame.columns}
    return next((columns[n.lower()] for n in names if n.lower() in columns), None)


def read_gene_table(path: str | Path) -> pd.DataFrame:
    """Read gene_id/gene_name tables or one-column gene lists."""
    raw = read_table(path, header=None)
    if raw.empty:
        raise ValueError(f"{path} contains no genes.")
    id_names = ("gene_id", "ensembl_id", "id", "fbgn_id", "fbgn", "fbgnid")
    symbol_names = ("gene_name", "symbol", "gene", "tf", "tg")
    first = [str(v).strip() for v in raw.iloc[0]]
    known = set(id_names + symbol_names)
    if any(value.lower() in known for value in first):
        frame = raw.iloc[1:].copy()
        frame.columns = first
    elif raw.shape[1] == 1:
        frame = raw.copy()
        frame.columns = ["gene_id"]
    else:
        raise ValueError(
            f"{path} needs gene_id/gene_name headers or a one-column gene list."
        )
    if frame.columns.duplicated().any():
        raise ValueError(f"{path} contains duplicate gene-table columns.")
    id_col = find_column(frame, id_names)
    name_col = find_column(frame, symbol_names)
    if id_col is None and name_col is None:
        raise ValueError(f"{path} needs a gene_id or gene_name column.")
    ids = frame[id_col or name_col].astype(str).map(strip_copy_suffix)
    names = frame[name_col or id_col].astype(str).map(strip_copy_suffix)
    if len(ids) == 0 or ids.eq("").any() or names.eq("").any():
        raise ValueError(f"{path} contains empty gene identifiers/names.")
    frame["gene_id"] = ids
    frame["gene_name"] = names
    return frame.reset_index(drop=True)


def map_gene_names(values: pd.Series, genes: pd.DataFrame) -> pd.Series:
    """IDs match literally; symbols map to IDs only when unambiguous."""
    ids = set(genes.gene_id)
    groups = genes.groupby("gene_name", sort=False).gene_id.agg(lambda x: set(x))
    aliases = {
        name: next(iter(matches))
        for name, matches in groups.items()
        if len(matches) == 1
    }
    cleaned = values.astype(str).map(strip_copy_suffix)
    ambiguous = (
        set(cleaned)
        & {name for name, matches in groups.items() if len(matches) > 1} - ids
    )
    if ambiguous:
        raise ValueError(
            f"Ambiguous gene symbols; use gene IDs: {sorted(ambiguous)[:10]}"
        )
    return cleaned.map(
        lambda value: value if value in ids else aliases.get(value, value)
    )


def read_cell_annotation(path: str | Path) -> pd.DataFrame:
    return canonical_cell_annotation(read_table(path), str(path))


def canonical_cell_annotation(
    frame: pd.DataFrame, source: str = "AnnData.obs"
) -> pd.DataFrame:
    frame = frame.copy()
    cell = find_column(
        frame,
        ("cell_id", "cell", "barcode", "barcodes", "Unnamed: 0", "index", "name", "id"),
    )
    group = find_column(
        frame,
        (
            "cell_type",
            "node_id",
            "GEX_latent_time_bins_annotated",
            "nt_time_bins_annotated",
            "annotation",
            "cluster",
            "label",
            "labels",
        ),
    )
    time = find_column(
        frame,
        ("time", "stage_id", "timepoint", "GEX_latent_time_bins", "latent_time_bins"),
    )
    if cell is None or group is None:
        raise ValueError(
            f"{source} requires cell_id and cell_type (or node_id) columns."
        )
    rename = {cell: "cell_id", group: "cell_type"}
    if time is not None:
        rename[time] = "time"
    frame = frame.rename(columns=rename)
    for column in ("cell_id", "cell_type"):
        frame[column] = frame[column].astype(str).str.strip()
    if frame[["cell_id", "cell_type"]].eq("").any().any():
        raise ValueError(f"{source} contains empty cell IDs or node labels.")
    if frame.cell_id.duplicated().any():
        raise ValueError(f"{source} contains duplicate cell IDs.")
    return frame


def select_matrix(adata: Any, layer: str | None) -> tuple[Any, str]:
    """A preferred layer falls back to X; null explicitly selects X."""
    if layer is not None and layer in adata.layers:
        return adata.layers[layer], layer
    return adata.X, "X"
