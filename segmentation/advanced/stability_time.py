"""Rolling re-fit stability across successive as-of dates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler

from segmentation.advanced.features_plus import RFM_PLUS_CLUSTER_COLS, build_rfm_plus
from segmentation.preprocess import winsorize_frame


def _matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    work = df[cols].copy()
    work, _ = winsorize_frame(work, cols)
    for c in ["Frequency", "Monetary", "AvgOrderValue", "MonetaryLast90", "MonetaryPrev90", "NUniqueStock"]:
        if c in work.columns:
            work[c] = np.log1p(work[c].clip(lower=0).astype(float))
    return RobustScaler().fit_transform(work.to_numpy(dtype=float))


def rolling_refit_stability(
    tx: pd.DataFrame,
    *,
    n_points: int = 4,
    n_clusters: int = 4,
    seed: int = 42,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Re-fit KMeans at several as-of dates; ARI on overlapping customers.

    Returns a table of consecutive ARI / size drift — production monitoring style.
    """
    cols = feature_cols or [c for c in RFM_PLUS_CLUSTER_COLS]
    dmin, dmax = tx["invoice_date"].min(), tx["invoice_date"].max()
    # as-of dates spanning last portion of history
    cutoffs = pd.date_range(dmin + (dmax - dmin) * 0.5, dmax, periods=n_points)

    fitted: list[tuple[pd.Timestamp, pd.DataFrame, np.ndarray]] = []
    for asof in cutoffs:
        feat = build_rfm_plus(tx, asof=asof)
        if len(feat) < 50:
            continue
        use_cols = [c for c in cols if c in feat.columns]
        X = _matrix(feat, use_cols)
        lab = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(X)
        fitted.append((pd.Timestamp(asof), feat[["customer_id"]].assign(label=lab), lab))

    rows = []
    for i in range(1, len(fitted)):
        t0, df0, _ = fitted[i - 1]
        t1, df1, _ = fitted[i]
        m = df0.merge(df1, on="customer_id", suffixes=("_prev", "_curr"))
        if len(m) < 20:
            continue
        ari = adjusted_rand_score(m["label_prev"], m["label_curr"])
        # size distribution L1 drift
        p0 = m["label_prev"].value_counts(normalize=True).sort_index()
        p1 = m["label_curr"].value_counts(normalize=True).sort_index()
        idx = sorted(set(p0.index) | set(p1.index))
        drift = float(sum(abs(p0.get(j, 0) - p1.get(j, 0)) for j in idx) / 2)
        rows.append(
            {
                "asof_prev": t0,
                "asof_curr": t1,
                "n_overlap": len(m),
                "ari": ari,
                "size_l1_drift": drift,
            }
        )
    return pd.DataFrame(rows)
