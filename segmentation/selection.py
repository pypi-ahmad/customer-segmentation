"""Algorithm survey, k-sweeps, and production model selection."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.mixture import GaussianMixture

from segmentation.metrics import internal_metrics, production_score, stability_ari

SEED = 42


def fit_model(
    name: str,
    X: np.ndarray,
    *,
    n_clusters: int = 4,
    random_state: int = SEED,
) -> np.ndarray:
    """Fit a named algorithm and return integer labels."""
    name = name.lower()
    if name in {"kmeans", "km"}:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=30,
            max_iter=500,
            algorithm="lloyd",
        )
        return model.fit_predict(X)
    if name in {"minibatch_kmeans", "mbkmeans"}:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=20,
            batch_size=min(2048, max(256, len(X) // 5)),
            max_iter=300,
        )
        return model.fit_predict(X)
    if name in {"hclust", "agglomerative", "ward"}:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        return model.fit_predict(X)
    if name == "birch":
        model = Birch(n_clusters=n_clusters)
        return model.fit_predict(X)
    if name in {"gmm", "gaussian_mixture"}:
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_state,
            n_init=5,
            max_iter=300,
            reg_covar=1e-5,
        )
        return model.fit_predict(X)
    raise ValueError(f"unsupported model: {name}")


def sweep_k(
    name: str,
    X: np.ndarray,
    ks: range | list[int] = range(2, 9),
    *,
    random_state: int = SEED,
    compute_stability: bool = True,
) -> pd.DataFrame:
    """Sweep n_clusters and return metrics + production score."""
    rows = []
    for k in ks:
        labels = fit_model(name, X, n_clusters=int(k), random_state=random_state)
        m = internal_metrics(X, labels)
        row = {"model": name, "n_clusters": int(k), **m}

        if compute_stability and m["n_clusters"] >= 2 and not np.isnan(m["silhouette"]):

            def _fit_sub(Xs, kk=int(k), nm=name):
                return fit_model(nm, Xs, n_clusters=kk, random_state=random_state)

            row["stability_ari"] = stability_ari(
                X, labels, _fit_sub, n_boot=6, sample_frac=0.7, seed=random_state
            )
        else:
            row["stability_ari"] = np.nan

        # inertia for kmeans only
        if name in {"kmeans", "minibatch_kmeans"}:
            km = KMeans(
                n_clusters=int(k),
                random_state=random_state,
                n_init=30,
                max_iter=500,
            ).fit(X)
            row["inertia"] = float(km.inertia_)
        else:
            row["inertia"] = np.nan

        if name == "gmm":
            gmm = GaussianMixture(
                n_components=int(k),
                covariance_type="full",
                random_state=random_state,
                n_init=5,
                max_iter=300,
                reg_covar=1e-5,
            ).fit(X)
            row["bic"] = float(gmm.bic(X))
            row["aic"] = float(gmm.aic(X))
        else:
            row["bic"] = np.nan
            row["aic"] = np.nan

        row["production_score"] = production_score(row)
        rows.append(row)
    return pd.DataFrame(rows)


def survey_algorithms(
    X: np.ndarray,
    *,
    models: list[str] | None = None,
    n_clusters: int = 4,
    random_state: int = SEED,
) -> pd.DataFrame:
    """Quick multi-algorithm survey at a fixed k (for shortlisting families)."""
    models = models or ["kmeans", "minibatch_kmeans", "hclust", "birch", "gmm"]
    rows = []
    for name in models:
        try:
            labels = fit_model(name, X, n_clusters=n_clusters, random_state=random_state)
            m = internal_metrics(X, labels)
            row = {"model": name, "n_clusters": n_clusters, "error": "", **m}
            row["production_score"] = production_score(row)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 — survey must continue
            rows.append(
                {
                    "model": name,
                    "n_clusters": n_clusters,
                    "error": str(exc),
                    "silhouette": np.nan,
                    "davies_bouldin": np.nan,
                    "calinski_harabasz": np.nan,
                    "largest_cluster_pct": np.nan,
                    "noise_pct": np.nan,
                    "production_score": -1e9,
                }
            )
    return pd.DataFrame(rows).sort_values("production_score", ascending=False)


def select_best_config(
    sweeps: dict[str, pd.DataFrame],
    *,
    min_k: int = 2,
    max_k: int = 8,
    require_largest_lt: float = 90.0,
) -> tuple[str, int, pd.Series, pd.DataFrame]:
    """Pick best (model, k) across per-model sweeps using production_score."""
    frames = []
    for name, df in sweeps.items():
        d = df.copy()
        d["model"] = name
        frames.append(d)
    all_df = pd.concat(frames, ignore_index=True)
    viable = all_df.loc[
        all_df["silhouette"].notna()
        & (all_df["n_clusters"] >= min_k)
        & (all_df["n_clusters"] <= max_k)
        & (all_df["noise_pct"].fillna(0) <= 20)
        & (all_df["largest_cluster_pct"] <= require_largest_lt)
        & (all_df["min_cluster_pct"].fillna(0) >= 2)
    ].copy()
    if viable.empty:
        viable = all_df.loc[all_df["silhouette"].notna()].copy()
    viable = viable.sort_values("production_score", ascending=False)
    best = viable.iloc[0]
    return str(best["model"]), int(best["n_clusters"]), best, viable


def top_two_families(
    survey: pd.DataFrame,
    *,
    min_sil: float = 0.05,
) -> list[str]:
    """Return up to two distinct model families from a fixed-k survey."""
    ok = survey.loc[
        (survey.get("error", pd.Series([""] * len(survey))).fillna("") == "")
        & survey["silhouette"].notna()
        & (survey["silhouette"] >= min_sil)
        & (survey["largest_cluster_pct"] <= 90)
        & (survey["n_clusters"] >= 2)
    ].sort_values("production_score", ascending=False)
    names: list[str] = []
    for m in ok["model"].tolist():
        if m not in names:
            names.append(m)
        if len(names) == 2:
            break
    if len(names) < 2:
        # fall back to top rows even if weak
        for m in survey.sort_values("production_score", ascending=False)["model"]:
            if m not in names:
                names.append(m)
            if len(names) == 2:
                break
    return names


def labels_for_config(model: str, X: np.ndarray, n_clusters: int, random_state: int = SEED) -> np.ndarray:
    return fit_model(model, X, n_clusters=n_clusters, random_state=random_state)


# type alias for external notebooks
FitFn = Callable[[np.ndarray], np.ndarray]
