"""Separate ultra-high-value customers before clustering the core base."""

from __future__ import annotations

import pandas as pd


def split_whales(
    features: pd.DataFrame,
    *,
    value_col: str = "Monetary",
    upper_q: float = 0.99,
    id_col: str = "customer_id",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Split whales (top value tail) from core customers.

    Returns (core_df, whale_df, threshold).
    Whales get their own managed segment in production; clustering the rest
    avoids centroid collapse toward extremes.
    """
    thr = float(features[value_col].quantile(upper_q))
    whales = features.loc[features[value_col] >= thr].copy()
    core = features.loc[features[value_col] < thr].copy()
    whales["is_whale"] = True
    core["is_whale"] = False
    return core, whales, thr
