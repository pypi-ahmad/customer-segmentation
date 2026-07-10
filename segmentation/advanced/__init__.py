"""Advanced segmentation extensions (learning path beyond notebooks 01–03).

Does not modify the baseline pipeline in segmentation.preprocess / selection.
"""

from segmentation.advanced.features_plus import build_rfm_plus, load_retail_transactions
from segmentation.advanced.whales import split_whales
from segmentation.advanced.temporal import (
    build_customer_features_asof,
    evaluate_future_outcomes,
    time_split_cutoff,
)
from segmentation.advanced.hybrid import fit_clv_proxy, hybrid_segment_table
from segmentation.advanced.soft_hierarchical import (
    fit_gmm_soft,
    hierarchical_two_level,
    soft_summary,
)
from segmentation.advanced.scoring import SegmentScorer, Playbook, default_playbooks
from segmentation.advanced.stability_time import rolling_refit_stability

__all__ = [
    "build_rfm_plus",
    "load_retail_transactions",
    "split_whales",
    "build_customer_features_asof",
    "evaluate_future_outcomes",
    "time_split_cutoff",
    "fit_clv_proxy",
    "hybrid_segment_table",
    "fit_gmm_soft",
    "hierarchical_two_level",
    "soft_summary",
    "SegmentScorer",
    "Playbook",
    "default_playbooks",
    "rolling_refit_stability",
]
