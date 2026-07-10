"""Feature engineering and leakage-safe preprocessing for segmentation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def winsorize_frame(
    df: pd.DataFrame,
    cols: Iterable[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clip heavy tails per column using train quantiles.

    Returns (clipped_frame, bounds_df with columns lower/upper).
    """
    out = df.copy()
    bounds = {}
    for c in cols:
        lo = float(out[c].quantile(lower_q))
        hi = float(out[c].quantile(upper_q))
        if hi <= lo:
            hi = lo + 1e-9
        out[c] = out[c].clip(lo, hi)
        bounds[c] = {"lower": lo, "upper": hi}
    return out, pd.DataFrame(bounds).T


def clean_transactions(
    raw: pd.DataFrame,
    *,
    invoice_col: str,
    customer_col: str,
    qty_col: str,
    price_col: str,
    date_col: str,
) -> pd.DataFrame:
    """Standard retail clean: drop cancels, missing customers, non-positive lines."""
    tx = raw.copy()
    tx[invoice_col] = tx[invoice_col].astype(str)
    tx[date_col] = pd.to_datetime(tx[date_col], errors="coerce")
    is_cancel = tx[invoice_col].str.startswith("C")
    mask = (
        (~is_cancel)
        & tx[customer_col].notna()
        & (tx[qty_col] > 0)
        & (tx[price_col] > 0)
        & tx[date_col].notna()
    )
    cleaned = tx.loc[mask].copy()
    cleaned["line_total"] = cleaned[qty_col] * cleaned[price_col]
    return cleaned


def build_rfm(
    tx: pd.DataFrame,
    *,
    customer_col: str,
    invoice_col: str,
    date_col: str,
    amount_col: str = "line_total",
) -> pd.DataFrame:
    """Aggregate one row per customer: Recency, Frequency, Monetary."""
    snapshot = tx[date_col].max()
    rfm = (
        tx.groupby(customer_col)
        .agg(
            Recency=(date_col, lambda s: (snapshot - s.max()).days),
            Frequency=(invoice_col, "nunique"),
            Monetary=(amount_col, "sum"),
        )
        .reset_index()
    )
    return rfm


def rfm_quantile_scores(rfm: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Classic RFM 1..n_bins scores (higher = better). Recency inverted."""
    out = rfm.copy()
    # Recency: lower days => higher score
    out["R_score"] = pd.qcut(
        -out["Recency"].rank(method="first"),
        q=n_bins,
        labels=list(range(1, n_bins + 1)),
    ).astype(int)
    out["F_score"] = pd.qcut(
        out["Frequency"].rank(method="first"),
        q=n_bins,
        labels=list(range(1, n_bins + 1)),
    ).astype(int)
    out["M_score"] = pd.qcut(
        out["Monetary"].rank(method="first"),
        q=n_bins,
        labels=list(range(1, n_bins + 1)),
    ).astype(int)
    out["RFM_score"] = out["R_score"] + out["F_score"] + out["M_score"]
    return out


def prepare_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    winsorize: bool = True,
    log_cols: list[str] | None = None,
    scaler: str = "robust",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[np.ndarray, pd.DataFrame, object, dict]:
    """Winsorize → optional log1p → scale. Returns X, working_df, fitted_scaler, meta."""
    work = df[feature_cols].copy()
    meta: dict = {"feature_cols": list(feature_cols), "log_cols": [], "bounds": None}

    if winsorize:
        work, bounds = winsorize_frame(work, feature_cols, lower_q=lower_q, upper_q=upper_q)
        meta["bounds"] = bounds

    log_cols = list(log_cols or [])
    for c in log_cols:
        if c not in work.columns:
            continue
        # only log positive-ish columns
        if (work[c] < 0).any():
            continue
        work[c] = np.log1p(work[c].astype(float))
        meta["log_cols"].append(c)

    if scaler == "robust":
        sc = RobustScaler()
    elif scaler == "standard":
        sc = StandardScaler()
    else:
        raise ValueError(f"unknown scaler: {scaler}")

    X = sc.fit_transform(work.to_numpy(dtype=float))
    meta["scaler"] = scaler
    return X, work, sc, meta
