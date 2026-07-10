"""Soft GMM membership and two-level hierarchical segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def fit_gmm_soft(
    X: np.ndarray,
    n_components: int = 4,
    *,
    seed: int = 42,
) -> dict:
    """Fit GMM and return hard labels + responsibility matrix."""
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=seed,
        n_init=5,
        max_iter=300,
        reg_covar=1e-5,
    )
    hard = gmm.fit_predict(X)
    proba = gmm.predict_proba(X)
    sil = (
        float(silhouette_score(X, hard))
        if len(set(hard.tolist())) >= 2
        else np.nan
    )
    return {"model": gmm, "labels": hard, "proba": proba, "silhouette": sil, "bic": float(gmm.bic(X))}


def soft_summary(proba: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster mean max-probability (confidence) and entropy."""
    rows = []
    ent = -np.sum(proba * np.log(proba + 1e-12), axis=1)
    maxp = proba.max(axis=1)
    for c in sorted(set(labels.tolist())):
        m = labels == c
        rows.append(
            {
                "cluster": c,
                "n": int(m.sum()),
                "mean_confidence": float(maxp[m].mean()),
                "mean_entropy": float(ent[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def hierarchical_two_level(
    X: np.ndarray,
    *,
    seed: int = 42,
    value_axis: np.ndarray | None = None,
    vip_quantile: float = 0.85,
) -> dict:
    """Level-1 VIP by value quantile (default top 15%); Level-2 KMeans on the rest.

    Using a **policy quantile** for VIP avoids the common failure mode where k=2
    puts most customers into a large "high-ish" blob. Rest is sub-segmented into 3.
    """
    if value_axis is None:
        value_axis = X.sum(axis=1)
    value_axis = np.asarray(value_axis, dtype=float)
    thr = float(np.quantile(value_axis, vip_quantile))
    vip_mask = value_axis >= thr
    rest_mask = ~vip_mask

    labels = np.zeros(len(X), dtype=int)  # 0 = VIP
    labels[rest_mask] = -1
    level2_model = None
    X_rest = X[rest_mask]
    if len(X_rest) >= 30:
        km2 = KMeans(n_clusters=3, random_state=seed, n_init=30, max_iter=500)
        l2 = km2.fit_predict(X_rest)
        labels[rest_mask] = l2 + 1  # 1,2,3
        level2_model = km2
    else:
        labels[rest_mask] = 1

    name_map = {0: "L1_VIP"}
    for c in sorted(set(labels.tolist())):
        if c > 0:
            name_map[c] = f"L2_Core_{c}"
    return {
        "labels": labels,
        "level1_model": None,
        "level2_model": level2_model,
        "vip_threshold": thr,
        "vip_quantile": vip_quantile,
        "name_map": name_map,
        "n_vip": int(vip_mask.sum()),
        "n_rest": int(rest_mask.sum()),
    }
