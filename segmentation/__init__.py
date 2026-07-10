"""Production-oriented unsupervised customer segmentation helpers."""

from segmentation.metrics import internal_metrics, production_score, stability_ari
from segmentation.preprocess import (
    build_rfm,
    clean_transactions,
    prepare_matrix,
    winsorize_frame,
)
from segmentation.selection import (
    fit_model,
    select_best_config,
    survey_algorithms,
    sweep_k,
)

__all__ = [
    "build_rfm",
    "clean_transactions",
    "prepare_matrix",
    "winsorize_frame",
    "internal_metrics",
    "production_score",
    "stability_ari",
    "fit_model",
    "select_best_config",
    "survey_algorithms",
    "sweep_k",
]
