# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python (customer-segmentation-project)
#     language: python
#     name: customer-segmentation-project
# ---

# %% [markdown]
# # 04 — Advanced path: RFM+ features, whale split, richer representation
#
# ## How to read this notebook (learning design)
#
# Notebooks **01–03 are unchanged** and remain the **baseline / learning track**.
# This notebook starts the **advanced track**. We do **not** overwrite old results.
#
# | Track | Notebooks | Purpose |
# |-------|-----------|---------|
# | Baseline | 01, 02, 03 | Classic RFM / spend, survey, production_score |
# | Advanced | **04**, 05, 06 | Features+, whales, future eval, hybrid, soft/hier, scoring |
#
# ## Why we are doing this
#
# With only Recency / Frequency / Monetary, many customers that *feel* different
# (steady monthly buyer vs one big spike; high cancel rate; shrinking spend) look
# almost the same in 3D space. That caps Silhouette and business lift.
#
# **Improvements in this notebook**
#
# 1. **RFM+ features** — tenure, AOV, inter-purchase gaps, cancel rate, 90-day trends  
# 2. **Whale split** — top 1% Monetary managed separately so they do not own the centroids  
# 3. **Compare** baseline 3-feature clustering vs RFM+ on the **core** base (same algo)
#
# Unsupervised only for labeling: still **no** segment ground truth / accuracy.

# %%
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from segmentation.advanced.features_plus import (
    RFM_BASIC_COLS,
    RFM_PLUS_CLUSTER_COLS,
    build_rfm_plus,
    load_retail_transactions,
)
from segmentation.advanced.whales import split_whales
from segmentation.metrics import internal_metrics
from segmentation.preprocess import prepare_matrix

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

print("Advanced track notebook 04 | root:", ROOT)

# %% [markdown]
# ## 1. Load real transactions (Online Retail II)
#
# Same UCI source as notebook 01 — we only **add** an advanced feature layer.

# %%
tx, meta = load_retail_transactions(ROOT, prefer="ii")
print(meta)
print("tx shape", tx.shape, "date", tx["invoice_date"].min(), "→", tx["invoice_date"].max())
display(tx.head(3))

# %% [markdown]
# ## 2. Build RFM+ (why each block exists)
#
# | Feature group | Why |
# |---------------|-----|
# | R, F, M | Classic value/engagement |
# | Tenure, inter-purchase mean/std | Distinguishes one-timers vs steady buyers |
# | AOV + std | Basket quality / volatility |
# | Cancel rate | Friction / return behavior |
# | Unique stock & countries | Breadth of engagement |
# | Last90 / Prev90 monetary + trend | **Trajectory**, not only lifetime total |
#
# All features use history only (full sample as-of = max date here).

# %%
feat = build_rfm_plus(tx)
print("Customers with RFM+:", len(feat))
display(feat[RFM_PLUS_CLUSTER_COLS].describe().T.head(20))

# %% [markdown]
# ## 3. Whale split (production pattern)
#
# **Why:** Top ~1% Monetary customers are operationally different (key accounts) and
# statistically different (they drag KMeans centers). We **remove them from the
# clustering matrix** and label them `Whale` by rule.

# %%
core, whales, thr = split_whales(feat, value_col="Monetary", upper_q=0.99)
print(f"Whale threshold Monetary ≥ {thr:,.2f}")
print(f"Core n={len(core):,} | Whales n={len(whales):,} ({100*len(whales)/len(feat):.2f}%)")
print(f"Whale share of lifetime Monetary: {100*whales['Monetary'].sum()/feat['Monetary'].sum():.1f}%")
display(whales[["customer_id", "Monetary", "Frequency", "Recency"]].sort_values("Monetary", ascending=False).head())

# %% [markdown]
# ## 4. Side-by-side: baseline RFM (3 cols) vs RFM+ on **core** only
#
# Same algorithm (KMeans k=4, RobustScaler pipeline) so the lift is from **features**,
# not from shopping a different estimator.

# %%
def fit_kmeans_block(df: pd.DataFrame, cols: list[str], k: int = 4, log_cols=None):
    log_cols = log_cols or [c for c in cols if c in {
        "Frequency", "Monetary", "AvgOrderValue", "StdOrderValue",
        "NUniqueStock", "MonetaryLast90", "MonetaryPrev90", "FreqLast90",
    }]
    X, work, sc, meta = prepare_matrix(df, cols, winsorize=True, log_cols=log_cols, scaler="robust")
    # choose k by silhouette among 2..6
    rows = []
    best = None
    for kk in range(2, 7):
        lab = KMeans(n_clusters=kk, random_state=SEED, n_init=30, max_iter=500).fit_predict(X)
        m = internal_metrics(X, lab)
        m["k"] = kk
        rows.append(m)
        if best is None or m["silhouette"] > best["silhouette"]:
            best = {**m, "k": kk, "labels": lab, "X": X}
    return pd.DataFrame(rows), best

basic_cols = [c for c in RFM_BASIC_COLS if c in core.columns]
plus_cols = [c for c in RFM_PLUS_CLUSTER_COLS if c in core.columns]

sweep_b, best_b = fit_kmeans_block(core, basic_cols)
sweep_p, best_p = fit_kmeans_block(core, plus_cols)

print("=== Baseline RFM (3 features) k-sweep on core ===")
display(sweep_b)
print("=== RFM+ k-sweep on core ===")
display(sweep_p)

print("\nPREFERRED baseline: k=%d Sil=%.4f CH=%.1f" % (best_b["k"], best_b["silhouette"], best_b["calinski_harabasz"]))
print("PREFERRED RFM+:     k=%d Sil=%.4f CH=%.1f" % (best_p["k"], best_p["silhouette"], best_p["calinski_harabasz"]))

# %% [markdown]
# ## 5. Profiles + PCA (RFM+ preferred)

# %%
core = core.copy()
core["seg_basic"] = best_b["labels"]
core["seg_plus"] = best_p["labels"]

prof = (
    core.groupby("seg_plus")
    .agg(
        n=("customer_id", "count"),
        R=("Recency", "mean"),
        F=("Frequency", "mean"),
        M=("Monetary", "mean"),
        tenure=("TenureDays", "mean"),
        aov=("AvgOrderValue", "mean"),
        trend90=("MonetaryTrend90", "mean"),
        cancel=("CancelRate", "mean"),
    )
    .reset_index()
)
prof["pct"] = 100 * prof["n"] / prof["n"].sum()
display(prof.sort_values("M", ascending=False))

Z = PCA(2, random_state=SEED).fit_transform(best_p["X"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(Z[:, 0], Z[:, 1], c=best_b["labels"], s=8, cmap="tab10", alpha=0.7)
axes[0].set_title(f"PCA — baseline RFM k={best_b['k']} (Sil={best_b['silhouette']:.3f})")
axes[1].scatter(Z[:, 0], Z[:, 1], c=best_p["labels"], s=8, cmap="tab10", alpha=0.7)
axes[1].set_title(f"PCA — RFM+ k={best_p['k']} (Sil={best_p['silhouette']:.3f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. What improved and what to remember
#
# - **RFM+** should improve **business interpretability** (trend, cancel, tenure) even when
#   Silhouette gains are modest — more axes → more nuanced segments.  
# - **Whales** are not “noise”; they are a **policy segment** with a different playbook.  
# - Notebooks 01–03 stay the simple story; this notebook is the **feature engineering upgrade**.  
# - Next (**05**): prove segments on **future** monetary/retention (time holdout) and hybrid CLV.

# %%
cmp = pd.DataFrame([
    {"pipeline": "baseline_RFM_3", "k": best_b["k"], "silhouette": best_b["silhouette"],
     "ch": best_b["calinski_harabasz"], "db": best_b["davies_bouldin"], "n_features": len(basic_cols)},
    {"pipeline": "RFM_plus_core", "k": best_p["k"], "silhouette": best_p["silhouette"],
     "ch": best_p["calinski_harabasz"], "db": best_p["davies_bouldin"], "n_features": len(plus_cols)},
    {"pipeline": "whales_rule", "k": 1, "silhouette": np.nan, "ch": np.nan, "db": np.nan,
     "n_features": 0},
])
print("Summary comparison (core customers only for clustering metrics):")
display(cmp)
print(f"Whales held out: n={len(whales)} threshold={thr:.2f}")
