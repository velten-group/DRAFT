"""Stage score cutoff: min_score_keep, lowered only when fewer than min_edge_fraction of pairs reach it."""

import numpy as np
import pytest
import torch

from draft_paired.analysis.network import lowered_cutoff_warning, min_edge_fraction, stage_score_cutoff


def test_configured_cutoff_applies_when_enough_pairs_reach_it():
    scores = np.linspace(0, 1, 1000)
    assert stage_score_cutoff(scores, 0.85, 0.01) == 0.85
    assert stage_score_cutoff(torch.tensor(scores, dtype=torch.float32), 0.85, 0.01) == 0.85
    assert lowered_cutoff_warning([0.85, 0.85], 0.85, 0.01, "network") is None


def test_cutoff_is_lowered_to_keep_the_top_fraction():
    scores = np.linspace(0, 0.5, 1000)
    cutoff = stage_score_cutoff(scores, 0.85, 0.01)
    assert int((scores >= cutoff).sum()) == 10
    assert stage_score_cutoff(scores, 0.85, 0) == 0.85
    warning = lowered_cutoff_warning([cutoff, 0.85], 0.85, 0.01, "network")
    assert "1/2 stages" in warning and "network.min_edge_fraction=0" in warning
    with pytest.raises(ValueError, match="between 0 and 1"):
        min_edge_fraction({"min_edge_fraction": 2})
