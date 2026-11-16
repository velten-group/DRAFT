"""Gene-symbol keys that ignore case except where case separates genes."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping

import pandas as pd

_PREFIX = re.compile(r"^(tf:|tg:)", re.IGNORECASE)


def clean_symbol(value: object) -> str:
    """Strip a byte-order mark, surrounding spaces and a TF:/TG: prefix; keep the case."""
    return _PREFIX.sub("", str(value).replace("﻿", "").strip())


class GeneKeys:
    """Match gene symbols without regard to case unless case distinguishes genes.

    Symbols are compared in lowercase, so tables written in different case
    conventions still match. When the vocabulary has spellings that differ
    only by case, such as the Drosophila genes ``H`` (Hairless) and ``h``
    (hairy), each spelling is its own key. A symbol that reaches such a group
    only by ignoring case, such as ``HDC`` for ``Hdc``/``hdc``, is ambiguous
    and has no key.
    """

    def __init__(self, *vocabularies: Iterable[object]):
        spellings: dict[str, set[str]] = defaultdict(set)
        for names in vocabularies:
            for name in names:
                symbol = clean_symbol(name)
                spellings[symbol.lower()].add(symbol)
        self.case_sensitive = {
            low for low, forms in spellings.items() if len(forms) > 1
        }
        self._forms = {
            low: next(iter(forms))
            for low, forms in spellings.items()
            if low not in self.case_sensitive
        }
        self._exact = {form for low in self.case_sensitive for form in spellings[low]}

    def __call__(self, name: object) -> str | None:
        symbol = clean_symbol(name)
        lower = symbol.lower()
        if lower not in self.case_sensitive:
            return lower
        return symbol if symbol in self._exact else None

    def keys(self, names: Iterable[object]) -> list[str | None]:
        return [self(name) for name in names]

    def series(self, names: pd.Series) -> pd.Series:
        return names.map(self)

    def spelling(self, key: str) -> str:
        """The vocabulary spelling of a key (the key itself when unknown)."""
        return self._forms.get(key, key)

    def request(self, genes: Iterable[str]) -> RequestedGenes:
        return RequestedGenes(self, genes)


class RequestedGenes(Mapping):
    """Gene keys to look up in another table, with their gene-table spellings."""

    def __init__(self, rule: GeneKeys, genes: Iterable[str]):
        self.rule = rule
        self._spellings = {str(g): rule.spelling(str(g)) for g in dict.fromkeys(genes)}

    def __getitem__(self, key: str) -> str:
        return self._spellings[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._spellings)

    def __len__(self) -> int:
        return len(self._spellings)


def as_request(genes: Iterable[str]) -> RequestedGenes:
    """Plain gene lists use the case rule of the listed keys themselves."""
    if isinstance(genes, RequestedGenes):
        return genes
    genes = [str(g) for g in genes]
    return GeneKeys(genes).request(genes)


def match_symbol(symbol: object, candidates: Iterable[object]) -> int | None:
    """Position of a requested gene: exact spelling first, else ignoring case when unique."""
    wanted = clean_symbol(symbol)
    values = [clean_symbol(value) for value in candidates]
    if wanted in values:
        return values.index(wanted)
    folded = [i for i, value in enumerate(values) if value.lower() == wanted.lower()]
    spellings = sorted({values[i] for i in folded})
    if len(spellings) > 1:
        raise ValueError(
            f"'{wanted}' matches genes that differ only by case ({', '.join(spellings)}); "
            "use the exact spelling or a gene ID."
        )
    return folded[0] if folded else None


def resolve_symbols(symbols: pd.Series, genes: RequestedGenes) -> pd.Series:
    """Key each symbol of another gene table (such as RNA genes) to a requested gene.

    When several symbols reach one key, the one spelled as in the gene table is
    kept; any other duplicates remain for the caller to report.
    """
    cleaned = symbols.astype(str).map(clean_symbol)
    resolved = cleaned.map(genes.rule)
    shared = resolved.isin(set(genes)) & resolved.duplicated(keep=False)
    if shared.any():
        preferred = shared & cleaned.eq(resolved.map(lambda key: genes.get(key)))
        winners = resolved[preferred]
        unique_winners = set(winners[~winners.duplicated(keep=False)])
        resolved = resolved.mask(shared & ~preferred & resolved.isin(unique_winners))
    return resolved
