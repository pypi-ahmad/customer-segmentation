"""Internal cluster validation and production-oriented composite scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette / DB / CH with noise handling (label < 0 ignored for geometry)."""
    labels = np.asarray(labels)
    mask = labels >= 0
    n_noise = int((~mask).sum())
    n = int(mask.sum())
    uniq = sorted(set(labels[mask].tolist())) if n else []
    out = {
        "n_clusters": len(uniq),
        "n_noise": n_noise,
        "noise_pct": 100.0 * n_noise / len(labels) if len(labels) else np.nan,
        "largest_cluster_pct": np.nan,
        "min_cluster_pct": np.nan,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    if not uniq:
        return out
    sizes = pd.Series(labels[mask]).value_counts()
    out["largest_cluster_pct"] = 100.0 * float(sizes.iloc[0]) / n
    out["min_cluster_pct"] = 100.0 * float(sizes.iloc[-1]) / n
    if len(uniq) >= 2 and n > len(uniq):
        out["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
        out["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
        out["calinski_harabasz"] = float(calinski_harabasz_score(X[mask], labels[mask]))
    return out


def production_score(row: pd.Series | dict, *, ch_scale: float = 5000.0) -> float:
    """Composite score for production model selection.

    Emphasizes Silhouette, penalizes imbalance / noise / tiny clusters,
    mildly rewards CH and lower DB. Prefer 3–6 clusters slightly for actionability
    when geometry is comparable (marketing-usable granularity).
    """
    if isinstance(row, dict):
        row = pd.Series(row)
    if pd.isna(row.get("silhouette")):
        return -1e9

    sil = float(row["silhouette"])
    db = float(row["davies_bouldin"]) if pd.notna(row.get("davies_bouldin")) else 2.0
    ch = float(row["calinski_harabasz"]) if pd.notna(row.get("calinski_harabasz")) else 0.0
    noise = float(row.get("noise_pct", 0.0) or 0.0)
    largest = float(row.get("largest_cluster_pct", 100.0) or 100.0)
    min_pct = float(row.get("min_cluster_pct", 0.0) or 0.0)
    k = int(row.get("n_clusters", 0) or 0)
    stab = float(row["stability_ari"]) if pd.notna(row.get("stability_ari")) else 0.0

    score = 1.0 * sil
    score += 0.15 * min(ch / max(ch_scale, 1.0), 1.5)
    score -= 0.08 * db
    score += 0.12 * stab

    # hard soft-penalties
    if noise > 15:
        score -= 0.25 * ((noise - 15) / 50.0)
    if largest > 70:
        score -= 0.35 * ((largest - 70) / 30.0)
    if min_pct < 3 and k >= 2:
        score -= 0.2
    # actionability: slight preference for 3–5 segments when sil is strong
    if 3 <= k <= 5:
        score += 0.03
    elif k == 2:
        score += 0.0
    elif k > 8:
        score -= 0.05 * (k - 8)
    return float(score)


def stability_ari(
    X: np.ndarray,
    labels_full: np.ndarray,
    fit_fn,
    *,
    n_boot: int = 8,
    sample_frac: float = 0.7,
    seed: int = 42,
) -> float:
    """Mean ARI between full labels and fits on bootstrap subsamples (projected).

    fit_fn(X_sub) -> labels_sub aligned to X_sub rows.
    We compare labels on the subsample indices only.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    if n < 50:
        return np.nan
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=max(20, int(sample_frac * n)), replace=False)
        try:
            lab_sub = np.asarray(fit_fn(X[idx]))
        except Exception:
            continue
        # map: only non-noise for ARI when possible
        a = labels_full[idx]
        b = lab_sub
        if len(set(a.tolist())) < 2 or len(set(b.tolist())) < 2:
            continue
        scores.append(adjusted_rand_score(a, b))
    if not scores:
        return np.nan
    return float(np.mean(scores))
