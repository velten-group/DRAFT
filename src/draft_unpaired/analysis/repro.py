"""Label and smoothness helpers reused by draft_unpaired."""

from draft_paired.analysis.repro import (
    build_generic_labels,
    make_unique_labels,
    plot_smoothness_heatmap,
    prob_cross_blockwise,
    prob_norm2_blockwise,
    safe_label,
    summarize_smoothness,
    tf_tg_prob_frobenius_dist_matrix,
)

__all__ = [
    "build_generic_labels",
    "make_unique_labels",
    "plot_smoothness_heatmap",
    "prob_cross_blockwise",
    "prob_norm2_blockwise",
    "safe_label",
    "summarize_smoothness",
    "tf_tg_prob_frobenius_dist_matrix",
]
