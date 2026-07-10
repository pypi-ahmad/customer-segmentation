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
# # 05 — Advanced path: temporal holdout evaluation + hybrid CLV / churn proxies
#
# ## Why this notebook exists
#
# Silhouette answers: *do points form blobs?*  
# Marketing answers: *do segments differ on **future** spend and return?*
#
# Here we:
#
# 1. Cut time into **history** (features only ≤ cutoff) and **future** (next 90 days)  
# 2. Cluster with **baseline RFM** vs **RFM+** on history (no leakage)  
# 3. Score each segment on **future monetary**, **retention**, and **lift**  
# 4. Fit simple **CLV / retention proxies** and build **hybrid** labels (cluster + value band)
#
# Baseline notebooks 01–03 are **not** modified.

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

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from segmentation.advanced.features_plus import (
    RFM_BASIC_COLS,
    RFM_PLUS_CLUSTER_COLS,
    load_retail_transactions,
)
from segmentation.advanced.hybrid import fit_clv_proxy, hybrid_segment_table
from segmentation.advanced.temporal import (
    build_customer_features_asof,
    evaluate_future_outcomes,
    future_outcomes,
    separation_score,
    time_split_cutoff,
)
from segmentation.advanced.whales import split_whales
from segmentation.metrics import internal_metrics
from segmentation.preprocess import prepare_matrix

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Data + time cutoff (anti-leakage)
#
# Features for customer *i* use only invoices with `date ≤ cutoff`.  
# Outcomes use invoices in `(cutoff, cutoff+90d]`.

# %%
tx, meta = load_retail_transactions(ROOT, prefer="ii")
print(meta)
cutoff = time_split_cutoff(tx, train_frac=0.75)  # keyword-only style
horizon = 90
print("Cutoff (as-of):", cutoff)
print("Future window: (", cutoff, ",", cutoff + pd.Timedelta(days=horizon), "]")

hist_feat = build_customer_features_asof(tx, cutoff)
print("Customers with history features:", len(hist_feat))
# need some purchase history
hist_feat = hist_feat.loc[hist_feat["Frequency"] >= 1].copy()
core, whales, thr = split_whales(hist_feat, upper_q=0.99)
print(f"Core={len(core)} whales={len(whales)} thr={thr:.2f}")

# %% [markdown]
# ## 2. Cluster on history only — baseline RFM vs RFM+

# %%
def cluster_core(df, cols, k=None):
    log_cols = [c for c in cols if any(x in c for x in [
        "Frequency", "Monetary", "AvgOrder", "StdOrder", "NUnique", "FreqLast", "Stock"
    ])]
    X, _, _, _ = prepare_matrix(df, cols, winsorize=True, log_cols=log_cols, scaler="robust")
    if k is None:
        best_k, best_sil, best_lab = 4, -1, None
        for kk in range(2, 7):
            lab = KMeans(kk, random_state=SEED, n_init=25, max_iter=400).fit_predict(X)
            sil = internal_metrics(X, lab)["silhouette"]
            if sil > best_sil:
                best_k, best_sil, best_lab = kk, sil, lab
        k, labels = best_k, best_lab
    else:
        labels = KMeans(k, random_state=SEED, n_init=25, max_iter=400).fit_predict(X)
    m = internal_metrics(X, labels)
    m["k"] = k
    return labels, m, X

cols_b = [c for c in RFM_BASIC_COLS if c in core.columns]
cols_p = [c for c in RFM_PLUS_CLUSTER_COLS if c in core.columns]
lab_b, met_b, Xb = cluster_core(core, cols_b)
lab_p, met_p, Xp = cluster_core(core, cols_p)

print("Baseline RFM history metrics:", met_b)
print("RFM+ history metrics:", met_p)

# %% [markdown]
# ## 3. Future outcomes by segment (the “is it awesome?” test)

# %%
sum_b = evaluate_future_outcomes(core, lab_b, tx, cutoff=cutoff, horizon_days=horizon)
sum_p = evaluate_future_outcomes(core, lab_p, tx, cutoff=cutoff, horizon_days=horizon)
print("=== Future metrics — baseline RFM segments ===")
display(sum_b)
print("separation", separation_score(sum_b))
print("\n=== Future metrics — RFM+ segments ===")
display(sum_p)
print("separation", separation_score(sum_p))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(sum_b["segment"].astype(str), sum_b["mean_future_monetary"], color="C0")
axes[0].set_title("Future £ / customer — baseline RFM")
axes[0].set_xlabel("segment")
axes[1].bar(sum_p["segment"].astype(str), sum_p["mean_future_monetary"], color="C1")
axes[1].set_title("Future £ / customer — RFM+")
axes[1].set_xlabel("segment")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. How to read the lift numbers
#
# - **`lift_future_monetary` > 1** → segment spends more next 90d than portfolio average  
# - **`retention_rate`** → fraction with ≥1 order in the future window  
# - **`future_value_ratio_max_min`** → how many times richer the top segment is vs the bottom  
#
# If RFM+ increases that ratio or top-segment future value share, the **feature upgrade
# is working for business**, even when Silhouette only moves a little.

# %%
sep_b, sep_p = separation_score(sum_b), separation_score(sum_p)
cmp = pd.DataFrame([
    {"pipeline": "baseline_RFM", **met_b, **sep_b,
     "top_seg_future_share": sum_b["pct_future_value"].iloc[0]},
    {"pipeline": "RFM_plus", **met_p, **sep_p,
     "top_seg_future_share": sum_p["pct_future_value"].iloc[0]},
])
print("Head-to-head (history geometry + future separation):")
display(cmp)

# %% [markdown]
# ## 5. Hybrid CLV / retention proxies
#
# Unsupervised clusters group *similar histories*.  
# Supervised proxies rank *expected future value*.  
# **Hybrid** = cluster id + predicted value band → finer CRM targeting without
# pretending clusters have ground-truth labels.

# %%
outcomes = future_outcomes(tx, core["customer_id"], cutoff=cutoff, horizon_days=horizon)
y_m = core.merge(outcomes, on="customer_id", how="left")["future_monetary"].fillna(0).to_numpy()
y_r = core.merge(outcomes, on="customer_id", how="left")["retained"].fillna(0).to_numpy().astype(int)

proxy = fit_clv_proxy(Xp, y_m, y_r, seed=SEED)
print("Holdout R² (future monetary):", proxy["holdout_r2_future_monetary"])
print("Holdout AUC (retention):", proxy["holdout_auc_retention"])

pred_m = proxy["regressor"].predict(Xp)
pred_r = (
    proxy["classifier"].predict_proba(Xp)[:, 1]
    if proxy["classifier"] is not None
    else np.full(len(Xp), np.nan)
)
hybrid = hybrid_segment_table(core, lab_p, pred_m, pred_r)
display(hybrid.head(10))
print("Hybrid segment counts:")
display(hybrid["hybrid_segment"].value_counts().head(15))

# hybrid future quality: group by hybrid label
core_h = core.copy()
core_h["hybrid_segment"] = hybrid["hybrid_segment"].values
# map hybrid to evaluate — use string labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
hy_codes = le.fit_transform(hybrid["hybrid_segment"])
sum_h = evaluate_future_outcomes(core, hy_codes, tx, cutoff=cutoff, horizon_days=horizon)
sum_h["hybrid_name"] = sum_h["segment"].map(lambda i: le.inverse_transform([int(i)])[0])
print("Top hybrid segments by future monetary:")
display(sum_h.head(8))
print("Hybrid separation", separation_score(sum_h))

# %% [markdown]
# ## 6. Takeaways
#
# | Question | Where answered |
# |----------|----------------|
# | Did we leak future into features? | No — as-of cutoff |
# | Are segments useful commercially? | Future £ and retention tables |
# | Did RFM+ help vs baseline RFM? | Head-to-head comparison frame |
# | How do we target inside a cluster? | Hybrid `cluster|Vband` |
#
# Next (**06**): soft GMM membership, hierarchical VIP→core, rolling stability, scoring API + playbooks.

# %%
print("DONE notebook 05")
print(cmp.to_string(index=False))
