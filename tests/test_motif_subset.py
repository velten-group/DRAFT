"""Motif retrieval keeps sites inside the queried ATAC regions and skips sites that only touch them."""

import contextlib
import gzip

from draft_paired import prior_preparation


def test_motif_subset_skips_sites_that_only_touch_a_region(tmp_path, monkeypatch):
    class Reader:
        def chroms(self):
            return {"chr1": 10_000}

        def records(self, chrom, start, end):
            return [
                (start - 12, start, "MA0001", 900, "+", "TP53"),  # ends where the region starts
                (start + 5, start + 17, "MA0002", 900, "-", "TP53"),
            ]

    @contextlib.contextmanager
    def fake_track(_source):
        yield Reader(), None

    monkeypatch.setattr(prior_preparation, "open_motif_track", fake_track)
    hits, mapping, record = prior_preparation.motif_subset(
        {"resources": {}}, "hg38", [("chr1", 100, 200)], {"tp53": "ENSG_TP53"}, tmp_path, 0
    )
    with gzip.open(hits, "rt") as handle:
        rows = [line.split("\t") for line in handle.read().splitlines()]
    assert [row[3] for row in rows] == ["MA0002"]
    assert mapping.read_text().splitlines() == ["MA0002\tENSG_TP53"]
    assert record["hits"] == 1
