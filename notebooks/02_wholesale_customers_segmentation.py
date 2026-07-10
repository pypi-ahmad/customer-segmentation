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
# # Wholesale Customers — Production-Grade Unsupervised Segmentation
#
# Segment 440 wholesale accounts by **annual category spend** with a production pipeline:
# winsorize → log1p → RobustScaler → multi-algorithm survey → k-sweep + stability → profiles.
#
# `Channel` and `Region` are **held out** of clustering and used only as an external sanity check.
#
# **Unsupervised:** no segment labels, no accuracy/confusion matrix.

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
from segmentation.preprocess import prepare_matrix
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
# ## 1. Data (UCI 292)

# %%
wholesale = fetch_ucirepo(id=292)
df = wholesale.data.original.copy()
print("Source: ucimlrepo id=292 original | shape:", df.shape)
print("Columns:", list(df.columns))
display(df.head())

deli = [c for c in df.columns if c.lower().startswith("delic")][0]
print("Delicatessen column:", repr(deli))
SPEND = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", deli]
HOLD = ["Channel", "Region"]

# %% [markdown]
# ## 2. EDA

# %%
print("Missing:\n", df.isna().sum())
display(df[SPEND].describe())
cmp = pd.DataFrame(
    {
        "mean": df[SPEND].mean(),
        "median": df[SPEND].median(),
        "skew": df[SPEND].skew(),
    }
)
display(cmp)

fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
for ax, c in zip(axes.ravel(), SPEND):
    sns.histplot(df[c], bins=25, ax=ax)
    ax.set_title(f"{c} skew={df[c].skew():.2f}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Production preprocessing
#
# Category spends are heavily right-skewed. We **winsorize** (1–99%), **log1p** all six,
# then **RobustScaler**. Channel/Region stay out of the matrix.

# %%
X, work, scaler, meta = prepare_matrix(
    df,
    SPEND,
    winsorize=True,
    log_cols=SPEND,
    scaler="robust",
)
print("Meta:", {k: meta[k] for k in ["scaler", "log_cols"]})
print("Winsor bounds:\n", meta["bounds"])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(pd.DataFrame(X, columns=SPEND).corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation — processed spend features")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Survey + deep k-sweep

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
    print(f"  {m} k={k} prod={r['production_score']:.4f} sil={r['silhouette']:.4f}")

# %% [markdown]
# ## 5. Final fits, PCA, profiles

# %%
results = {}
for model_name, k, row in configs:
    labels = labels_for_config(model_name, X, k, random_state=SEED)
    m = internal_metrics(X, labels)
    m["production_score"] = production_score({**m, "stability_ari": row.get("stability_ari", np.nan)})
    print(f"\n### {model_name} k={k}", m)

    Z = PCA(2, random_state=SEED).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=28, alpha=0.85)
    ax.set_title(f"PCA — {model_name} k={k}")
    ax.legend(*sc.legend_elements(), title="cluster")
    plt.tight_layout()
    plt.show()

    tmp = df.copy()
    tmp["cluster"] = labels
    tmp["row_spend"] = tmp[SPEND].sum(axis=1)
    g = tmp.groupby("cluster")
    prof = g[SPEND].mean()
    prof.insert(0, "n", g.size())
    prof["mean_total"] = g["row_spend"].mean()
    prof["total_spend"] = g["row_spend"].sum()
    prof = prof.reset_index()
    prof["pct_n"] = 100 * prof["n"] / prof["n"].sum()
    prof["pct_spend"] = 100 * prof["total_spend"] / prof["total_spend"].sum()
    prof = prof.sort_values("mean_total", ascending=False)
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
# ## 6. Comparison + Channel/Region sanity check

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
pref_key = cmp.iloc[0]["config"]
pref = results[pref_key]
print("PREFERRED:", pref_key)

sanity = df[HOLD].copy()
sanity["cluster"] = pref["labels"]
print("\nCluster × Channel (counts):")
display(pd.crosstab(sanity["cluster"], sanity["Channel"], margins=True))
print("Cluster × Channel (row %):")
display(pd.crosstab(sanity["cluster"], sanity["Channel"], normalize="index").mul(100).round(1))
print("Cluster × Region (row %):")
display(pd.crosstab(sanity["cluster"], sanity["Region"], normalize="index").mul(100).round(1))

for cl in sorted(set(pref["labels"])):
    sub = sanity.loc[sanity["cluster"] == cl]
    mode = sub["Channel"].value_counts(normalize=True)
    print(
        f"Cluster {cl}: n={len(sub)}, dominant Channel={mode.index[0]} "
        f"({100*mode.iloc[0]:.1f}%) — descriptive only"
    )

# %% [markdown]
# ## 7. Manager brief

# %%
prof = pref["profile"]
print("MANAGER BRIEF — Wholesale Customers (production pipeline)")
print("=" * 64)
print(
    f"{pref['model']} k={pref['k']} | Sil={pref['metrics']['silhouette']:.3f} | "
    f"DB={pref['metrics']['davies_bouldin']:.3f} | CH={pref['metrics']['calinski_harabasz']:.1f} | "
    f"stab={pref['row'].get('stability_ari', float('nan')):.3f}"
)
for _, pr in prof.iterrows():
    top = max(SPEND, key=lambda c: pr[c])
    print(
        f"Segment {int(pr['cluster'])}: n={int(pr['n'])} ({pr['pct_n']:.1f}% accounts, "
        f"{pr['pct_spend']:.1f}% spend) | mean total≈{pr['mean_total']:.0f} | top category {top}≈{pr[top]:.0f}"
    )
    print("  → Align assortment, logistics SLA, and account coverage to category mix.")

print(
    "\nSanity: Channel/Region crosstabs above are descriptive only — not clustering accuracy."
)
