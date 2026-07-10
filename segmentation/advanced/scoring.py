"""Lightweight scoring API + campaign playbooks."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from segmentation.preprocess import winsorize_frame


@dataclass
class Playbook:
    segment: str
    action: str
    channel: str
    budget_priority: str  # high / medium / low
    success_metric: str
    offer_hint: str


def default_playbooks() -> dict[str, Playbook]:
    return {
        "VIP": Playbook(
            segment="VIP",
            action="Protect with high-touch retention",
            channel="Account manager + email",
            budget_priority="high",
            success_metric="Retention rate, repeat AOV",
            offer_hint="Early access / non-discount perks first",
        ),
        "Growth": Playbook(
            segment="Growth",
            action="Raise frequency and category breadth",
            channel="Email / CRM automation",
            budget_priority="medium",
            success_metric="Orders next 90d, AOV",
            offer_hint="Bundles, loyalty stamps",
        ),
        "AtRisk": Playbook(
            segment="AtRisk",
            action="Win-back sequence",
            channel="Email + SMS",
            budget_priority="medium",
            success_metric="Reactivation rate",
            offer_hint="Modest time-boxed incentive",
        ),
        "LowValue": Playbook(
            segment="LowValue",
            action="Low-cost lifecycle only",
            channel="Email automation",
            budget_priority="low",
            success_metric="Cost per activated order",
            offer_hint="Avoid deep discounts",
        ),
        "Whale": Playbook(
            segment="Whale",
            action="Dedicated key-account management",
            channel="Human AM",
            budget_priority="high",
            success_metric="Account retention, NPS",
            offer_hint="Service SLAs, not coupons",
        ),
    }


class SegmentScorer:
    """Fit winsor + robust scale + KMeans; score new customers.

    Persists bounds/scaler/centroids for production-style inference demos.
    """

    def __init__(
        self,
        feature_cols: list[str],
        *,
        n_clusters: int = 4,
        log_cols: list[str] | None = None,
        seed: int = 42,
        whale_threshold: float | None = None,
        monetary_col: str = "Monetary",
    ):
        self.feature_cols = list(feature_cols)
        self.n_clusters = n_clusters
        self.log_cols = list(log_cols or [])
        self.seed = seed
        self.whale_threshold = whale_threshold
        self.monetary_col = monetary_col
        self.bounds_: pd.DataFrame | None = None
        self.scaler_: RobustScaler | None = None
        self.model_: KMeans | None = None
        self.segment_names_: dict[int, str] = {}

    def _transform(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        work = df[self.feature_cols].copy()
        if fit:
            work, self.bounds_ = winsorize_frame(work, self.feature_cols)
        else:
            assert self.bounds_ is not None
            for c in self.feature_cols:
                lo, hi = self.bounds_.loc[c, "lower"], self.bounds_.loc[c, "upper"]
                work[c] = work[c].clip(lo, hi)
        for c in self.log_cols:
            if c in work.columns:
                work[c] = np.log1p(work[c].clip(lower=0).astype(float))
        arr = work.to_numpy(dtype=float)
        if fit:
            self.scaler_ = RobustScaler()
            return self.scaler_.fit_transform(arr)
        assert self.scaler_ is not None
        return self.scaler_.transform(arr)

    def fit(self, features: pd.DataFrame) -> "SegmentScorer":
        X = self._transform(features, fit=True)
        self.model_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            n_init=30,
            max_iter=500,
        )
        labels = self.model_.fit_predict(X)
        # name by mean monetary if available
        if self.monetary_col in features.columns:
            tmp = features[[self.monetary_col]].copy()
            tmp["lab"] = labels
            order = tmp.groupby("lab")[self.monetary_col].mean().sort_values(ascending=False)
            names = ["VIP", "Growth", "AtRisk", "LowValue", "Seg4", "Seg5"]
            self.segment_names_ = {
                int(c): names[i] if i < len(names) else f"Seg{i}"
                for i, c in enumerate(order.index)
            }
        else:
            self.segment_names_ = {i: f"Seg{i}" for i in range(self.n_clusters)}
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert self.model_ is not None
        out = pd.DataFrame(index=features.index)
        if "customer_id" in features.columns:
            out["customer_id"] = features["customer_id"].values
        # whales first
        is_whale = np.zeros(len(features), dtype=bool)
        if self.whale_threshold is not None and self.monetary_col in features.columns:
            is_whale = features[self.monetary_col].to_numpy() >= self.whale_threshold
        X = self._transform(features, fit=False)
        hard = self.model_.predict(X)
        # distance to assigned centroid as confidence proxy (lower = better)
        centers = self.model_.cluster_centers_
        dist = np.linalg.norm(X - centers[hard], axis=1)
        conf = 1.0 / (1.0 + dist)
        names = []
        actions = []
        books = default_playbooks()
        for i in range(len(features)):
            if is_whale[i]:
                names.append("Whale")
                pb = books["Whale"]
            else:
                nm = self.segment_names_.get(int(hard[i]), f"Seg{hard[i]}")
                names.append(nm)
                pb = books.get(nm, books["LowValue"])
            actions.append(pb.action)
        out["segment"] = names
        out["cluster_id"] = hard
        out["is_whale"] = is_whale
        out["confidence"] = conf
        out["recommended_action"] = actions
        return out

    def playbook_table(self) -> pd.DataFrame:
        rows = [asdict(pb) for pb in default_playbooks().values()]
        return pd.DataFrame(rows)
