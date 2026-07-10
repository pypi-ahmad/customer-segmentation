# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (customer-segmentation-project)
#     language: python
#     name: customer-segmentation-project
# ---

# %% [markdown]
# # Online Retail (UCI 352) — Production-Grade Unsupervised Segmentation (RFM)
#
# Single-year predecessor to Online Retail II. Column names differ:
# `InvoiceNo`, `UnitPrice`, `CustomerID`.
#
# Pipeline: clean → RFM → winsorize + log1p + RobustScaler → survey → k-sweep +
# stability → production_score selection → profiles.
#
# **Unsupervised only** — no accuracy / confusion matrix.

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
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from segmentation.metrics import internal_metrics, production_score
from segmentation.preprocess import build_rfm, clean_transactions, prepare_matrix, rfm_quantile_scores
from segmentation.selection import (
    labels_for_config,
    select_best_config,
    survey_algorithms,
    sweep_k,
    top_two_families,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
sns.set_theme(style="whitegrid", context="notebook")

import pycaret
import sklearn

print("Python:", sys.version.split()[0])
print("pycaret:", pycaret.__version__, "| sklearn:", sklearn.__version__)

# %% [markdown]
# ## 1. Data (UCI 352)

# %%
retail = fetch_ucirepo(id=352)
raw = retail.data.original.copy() if retail.data.original is not None else retail.data.features.copy()
if "InvoiceNo" not in raw.columns and retail.data.ids is not None:
    raw = pd.concat(
        [retail.data.ids.reset_index(drop=True), retail.data.features.reset_index(drop=True)],
        axis=1,
    )
print("Shape:", raw.shape)
print("Columns:", list(raw.columns))
for c in ["InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]:
    assert c in raw.columns, c
display(raw.head(3))

# %% [markdown]
# ## 2. EDA

# %%
raw = raw.copy()
raw["InvoiceNo"] = raw["InvoiceNo"].astype(str)
raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
print("Missing %:\n", (100 * raw.isna().mean()).round(2))
print("Cancels:", raw["InvoiceNo"].str.startswith("C").sum())
print("Date range:", raw["InvoiceDate"].min(), "→", raw["InvoiceDate"].max())

daily = (
    raw.dropna(subset=["InvoiceDate"])
    .assign(d=lambda x: x["InvoiceDate"].dt.date)
    .groupby("d")["InvoiceNo"]
    .nunique()
)
fig, ax = plt.subplots(figsize=(11, 3.2))
daily.plot(ax=ax)
ax.set_title("Distinct invoices per day")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. RFM + production preprocessing

# %%
tx = clean_transactions(
    raw,
    invoice_col="InvoiceNo",
    customer_col="CustomerID",
    qty_col="Quantity",
    price_col="UnitPrice",
    date_col="InvoiceDate",
)
print(f"Raw: {len(raw):,} | Cleaned: {len(tx):,} | Customers: {tx['CustomerID'].nunique():,}")

rfm = build_rfm(
    tx,
    customer_col="CustomerID",
    invoice_col="InvoiceNo",
    date_col="InvoiceDate",
)
rfm["CustomerID"] = rfm["CustomerID"].astype(int)
print("RFM shape:", rfm.shape)
display(rfm[["Recency", "Frequency", "Monetary"]].describe())
print("Skew:\n", rfm[["Recency", "Frequency", "Monetary"]].skew())

display(rfm_quantile_scores(rfm)[["R_score", "F_score", "M_score", "RFM_score"]].head())

feat_cols = ["Recency", "Frequency", "Monetary"]
X, work, scaler, meta = prepare_matrix(
    rfm,
    feat_cols,
    winsorize=True,
    log_cols=["Frequency", "Monetary"],
    scaler="robust",
)
print("Meta:", {k: meta[k] for k in ["scaler", "log_cols"]})
print("Winsor bounds:\n", meta["bounds"])

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, c in zip(axes, feat_cols):
    sns.histplot(work[c], bins=40, ax=ax)
    ax.set_title(f"Processed {c}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Survey + deep k-sweep + stability

# %%
survey = survey_algorithms(
    X,
    models=["kmeans", "minibatch_kmeans", "hclust", "birch", "gmm"],
    n_clusters=4,
    random_state=SEED,
)
display(survey)
best_two = top_two_families(survey)
print("Best 2 families:", best_two)

sweeps = {}
for fam in best_two:
    print(f"\n=== Sweep {fam} ===")
    sw = sweep_k(fam, X, ks=range(2, 9), random_state=SEED, compute_stability=True)
    sweeps[fam] = sw
    display(sw.sort_values("production_score", ascending=False))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    axes[0].plot(sw["n_clusters"], sw["silhouette"], "o-")
    axes[0].set_title(f"{fam} Silhouette")
    axes[1].plot(sw["n_clusters"], sw["production_score"], "o-", color="C1")
    axes[1].set_title(f"{fam} production_score")
    plt.tight_layout()
    plt.show()

model_a, k_a, best_row, ranking = select_best_config(sweeps)
display(ranking.head(10))
configs = [(model_a, k_a, best_row)]
other = [m for m in best_two if m != model_a]
if other:
    sw_o = sweeps[other[0]].sort_values("production_score", ascending=False).iloc[0]
    configs.append((other[0], int(sw_o["n_clusters"]), sw_o))

print("Configs:")
for m, k, r in configs:
    print(
        f"  {m} k={k} | Sil={r['silhouette']:.4f} | stab={r.get('stability_ari', np.nan):.3f} | "
        f"prod={r['production_score']:.4f}"
    )

# %% [markdown]
# ## 5. Final models

# %%
results = {}
for model_name, k, row in configs:
    labels = labels_for_config(model_name, X, k, random_state=SEED)
    m = internal_metrics(X, labels)
    m["production_score"] = production_score({**m, "stability_ari": row.get("stability_ari", np.nan)})
    print(f"\n### {model_name} k={k}", m)

    Z = PCA(2, random_state=SEED).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=10, alpha=0.75)
    ax.set_title(f"PCA — {model_name} k={k} (Sil={m['silhouette']:.3f})")
    ax.legend(*sc.legend_elements(), title="cluster")
    plt.tight_layout()
    plt.show()

    tmp = rfm.copy()
    tmp["cluster"] = labels
    prof = (
        tmp.groupby("cluster")
        .agg(
            n=("CustomerID", "count"),
            mean_R=("Recency", "mean"),
            mean_F=("Frequency", "mean"),
            mean_M=("Monetary", "mean"),
            total_M=("Monetary", "sum"),
            median_M=("Monetary", "median"),
        )
        .reset_index()
    )
    prof["pct_n"] = 100 * prof["n"] / prof["n"].sum()
    prof["pct_M"] = 100 * prof["total_M"] / prof["total_M"].sum()
    prof = prof.sort_values("mean_M", ascending=False)
    display(prof)

    results[f"{model_name}_k{k}"] = {
        "model": model_name,
        "k": k,
        "labels": labels,
        "metrics": m,
        "profile": prof,
        "row": row,
    }

# %% [markdown]
# ## 6. Comparison & manager brief

# %%
cmp = pd.DataFrame(
    [
        {
            "config": key,
            "model": r["model"],
            "k": r["k"],
            "silhouette": r["metrics"]["silhouette"],
            "davies_bouldin": r["metrics"]["davies_bouldin"],
            "calinski_harabasz": r["metrics"]["calinski_harabasz"],
            "largest_pct": r["metrics"]["largest_cluster_pct"],
            "stability_ari": r["row"].get("stability_ari", np.nan),
            "production_score": r["metrics"]["production_score"],
        }
        for key, r in results.items()
    ]
).sort_values("production_score", ascending=False)
display(cmp)
pref = results[cmp.iloc[0]["config"]]
print("PREFERRED:", cmp.iloc[0]["config"])

prof = pref["profile"]
print("\nMANAGER BRIEF — Online Retail 352 (production pipeline)")
print("=" * 64)
print(
    f"{pref['model']} k={pref['k']} | Sil={pref['metrics']['silhouette']:.3f} | "
    f"DB={pref['metrics']['davies_bouldin']:.3f} | CH={pref['metrics']['calinski_harabasz']:.1f} | "
    f"stab={pref['row'].get('stability_ari', float('nan')):.3f}"
)
print(f"Customers: {len(rfm):,}")
for _, pr in prof.iterrows():
    print(
        f"Segment {int(pr['cluster'])}: n={int(pr['n'])} ({pr['pct_n']:.1f}% cust, {pr['pct_M']:.1f}% rev) | "
        f"R={pr['mean_R']:.0f}d F={pr['mean_F']:.1f} M=£{pr['mean_M']:.0f} (median £{pr['median_M']:.0f})"
    )
print(
    "\nActions: protect high-M/low-R segments; grow mid-frequency; automate low-value lapsed."
)
