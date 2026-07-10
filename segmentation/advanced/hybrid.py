"""Hybrid unsupervised structure + supervised CLV/churn proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split


def fit_clv_proxy(
    X: np.ndarray,
    y_monetary: np.ndarray,
    y_retained: np.ndarray,
    *,
    seed: int = 42,
) -> dict:
    """Train simple proxies for next-period monetary and retention.

    Returns fitted models + holdout metrics (honest quality of proxies themselves).
    """
    # regression for future monetary
    Xtr, Xte, ytr, yte, rtr, rte = train_test_split(
        X,
        y_monetary,
        y_retained,
        test_size=0.25,
        random_state=seed,
    )
    reg = GradientBoostingRegressor(random_state=seed, max_depth=3, n_estimators=80)
    reg.fit(Xtr, ytr)
    pred_m = reg.predict(Xte)
    r2 = float(r2_score(yte, pred_m)) if np.var(yte) > 1e-12 else np.nan

    clf = GradientBoostingClassifier(random_state=seed, max_depth=3, n_estimators=80)
    # need both classes
    if len(np.unique(rtr)) < 2:
        auc = np.nan
        clf = None
    else:
        clf.fit(Xtr, rtr)
        proba = clf.predict_proba(Xte)[:, 1]
        try:
            auc = float(roc_auc_score(rte, proba))
        except ValueError:
            auc = np.nan

    # refit on full data for scoring
    reg_full = GradientBoostingRegressor(random_state=seed, max_depth=3, n_estimators=80)
    reg_full.fit(X, y_monetary)
    clf_full = None
    if len(np.unique(y_retained)) >= 2:
        clf_full = GradientBoostingClassifier(random_state=seed, max_depth=3, n_estimators=80)
        clf_full.fit(X, y_retained)

    return {
        "regressor": reg_full,
        "classifier": clf_full,
        "holdout_r2_future_monetary": r2,
        "holdout_auc_retention": auc,
    }


def hybrid_segment_table(
    features: pd.DataFrame,
    cluster_labels: np.ndarray,
    pred_monetary: np.ndarray,
    pred_retention: np.ndarray | None = None,
    *,
    customer_col: str = "customer_id",
    n_value_bands: int = 3,
) -> pd.DataFrame:
    """Combine unsupervised cluster with predicted value bands.

    Hybrid label: ``{cluster}|V{band}`` where band is high/mid/low by predicted monetary.
    """
    out = features[[customer_col]].copy() if customer_col in features.columns else pd.DataFrame(index=features.index)
    if customer_col not in out.columns:
        out[customer_col] = features.index
    out["cluster"] = cluster_labels
    out["pred_future_monetary"] = pred_monetary
    if pred_retention is not None:
        out["pred_retention"] = pred_retention
    # quantile bands 0=low, 1=mid, 2=high
    try:
        out["value_band"] = pd.qcut(
            out["pred_future_monetary"].rank(method="first"),
            q=n_value_bands,
            labels=[f"V{i}" for i in range(n_value_bands)],
        ).astype(str)
    except ValueError:
        out["value_band"] = "V0"
    out["hybrid_segment"] = out["cluster"].astype(str) + "|" + out["value_band"]
    return out
