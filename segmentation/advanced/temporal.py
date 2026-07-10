"""Time-based holdout evaluation for segments (future value / retention)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from segmentation.advanced.features_plus import build_rfm_plus


def time_split_cutoff(
    tx: pd.DataFrame,
    *,
    train_frac: float = 0.7,
    date_col: str = "invoice_date",
) -> pd.Timestamp:
    """Pick a cutoff so ~train_frac of the time span is history."""
    dmin, dmax = tx[date_col].min(), tx[date_col].max()
    span = (dmax - dmin).days
    return dmin + pd.Timedelta(days=int(span * train_frac))


def build_customer_features_asof(tx: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """RFM+ features using only history ≤ asof (no leakage)."""
    return build_rfm_plus(tx, asof=asof)


def future_outcomes(
    tx: pd.DataFrame,
    customers: pd.Series,
    *,
    cutoff: pd.Timestamp,
    horizon_days: int = 90,
    customer_col: str = "customer_id",
) -> pd.DataFrame:
    """Outcomes in (cutoff, cutoff+horizon] for each customer id."""
    end = cutoff + pd.Timedelta(days=horizon_days)
    fut = tx.loc[
        (~tx["is_cancel"])
        & (tx["quantity"] > 0)
        & (tx["unit_price"] > 0)
        & (tx["invoice_date"] > cutoff)
        & (tx["invoice_date"] <= end)
        & tx[customer_col].notna()
    ]
    mon = fut.groupby(customer_col)["line_total"].sum().rename("future_monetary")
    freq = fut.groupby(customer_col)["invoice"].nunique().rename("future_orders")
    out = pd.DataFrame({customer_col: customers.unique()})
    out = out.merge(mon, on=customer_col, how="left")
    out = out.merge(freq, on=customer_col, how="left")
    out["future_monetary"] = out["future_monetary"].fillna(0.0)
    out["future_orders"] = out["future_orders"].fillna(0).astype(int)
    out["retained"] = (out["future_orders"] > 0).astype(int)
    return out


def evaluate_future_outcomes(
    features_at_cutoff: pd.DataFrame,
    labels: np.ndarray | pd.Series,
    tx: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon_days: int = 90,
    customer_col: str = "customer_id",
    segment_col: str = "segment",
) -> pd.DataFrame:
    """Per-segment future monetary / retention / lift vs portfolio mean."""
    feat = features_at_cutoff.copy()
    feat[segment_col] = np.asarray(labels)
    outcomes = future_outcomes(
        tx,
        feat[customer_col],
        cutoff=cutoff,
        horizon_days=horizon_days,
        customer_col=customer_col,
    )
    m = feat[[customer_col, segment_col]].merge(outcomes, on=customer_col, how="left")
    g = m.groupby(segment_col, dropna=False)
    summary = g.agg(
        n=(customer_col, "count"),
        mean_future_monetary=("future_monetary", "mean"),
        median_future_monetary=("future_monetary", "median"),
        total_future_monetary=("future_monetary", "sum"),
        retention_rate=("retained", "mean"),
        mean_future_orders=("future_orders", "mean"),
    ).reset_index()
    port_mean = m["future_monetary"].mean()
    port_ret = m["retained"].mean()
    summary["lift_future_monetary"] = summary["mean_future_monetary"] / max(port_mean, 1e-9)
    summary["lift_retention"] = summary["retention_rate"] / max(port_ret, 1e-9)
    summary["pct_customers"] = 100 * summary["n"] / summary["n"].sum()
    summary["pct_future_value"] = 100 * summary["total_future_monetary"] / max(
        summary["total_future_monetary"].sum(), 1e-9
    )
    return summary.sort_values("mean_future_monetary", ascending=False)


def separation_score(summary: pd.DataFrame) -> dict:
    """How well segments separate future value (higher better)."""
    if len(summary) < 2:
        return {"future_value_ratio_max_min": np.nan, "retention_gap": np.nan}
    fv = summary["mean_future_monetary"].astype(float)
    ret = summary["retention_rate"].astype(float)
    # floor tiny means so empty-future segments do not explode the ratio
    floored = fv.clip(lower=max(fv.median() * 0.01, 1.0))
    return {
        "future_value_ratio_max_min": float(fv.max() / floored.min()),
        "retention_gap": float(ret.max() - ret.min()),
        "top_segment_future_value_share": float(summary["pct_future_value"].iloc[0]),
    }
