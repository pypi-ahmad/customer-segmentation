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
# # Online Retail II — Production-Grade Unsupervised Segmentation (RFM)
#
# **Goal.** Discover actionable customer segments from real UCI Online Retail II
# transactions using a **production-minded** unsupervised pipeline:
#
# 1. Clean + RFM engineering  
# 2. **Winsorize** heavy tails, **log1p**, **RobustScaler**  
# 3. Multi-algorithm survey (KMeans, MiniBatchKMeans, Ward hierarchical, Birch, **GMM**)  
# 4. Per-family **k-sweep** with Silhouette / DB / CH + **bootstrap stability (ARI)**  
# 5. Select with a **production_score** (geometry + balance + stability + actionability)  
# 6. Business profiles + manager brief  
#
# ## Unsupervised only
#
# There is **no ground-truth segment label**. Metrics are internal cluster validation
# and business profiles — **not** accuracy or confusion matrices.

# %%
from __future__ import annotations

import random
import sys
import warnings
import zipfile
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

# project root on path
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from segmentation.metrics import internal_metrics, production_score
from segmentation.preprocess import build_rfm, clean_transactions, prepare_matrix, rfm_quantile_scores
from segmentation.selection import (
    fit_model,
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

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

import pycaret
import sklearn

print("Python:", sys.version.split()[0])
print("pycaret:", pycaret.__version__, "| sklearn:", sklearn.__version__)
print("Kernel: customer-segmentation-project")
print("Project root:", ROOT)

# %% [markdown]
# ## 1. Data acquisition (UCI 502)
#
# Prefer `ucimlrepo`; if UCI marks the dataset non-importable, use the official static zip.

# %%
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_PATH = DATA_DIR / "online_retail_ii.zip"
XLSX_PATH = DATA_DIR / "online_retail_II.xlsx"


def load_online_retail_ii() -> tuple[pd.DataFrame, str]:
    try:
        retail_ii = fetch_ucirepo(id=502)
        if retail_ii.data.features is not None and len(retail_ii.data.features):
            return retail_ii.data.features.copy(), "ucimlrepo features"
        if retail_ii.data.original is not None and len(retail_ii.data.original):
            return retail_ii.data.original.copy(), "ucimlrepo original"
    except Exception as exc:
        print(f"ucimlrepo(id=502) unavailable ({type(exc).__name__}); using official UCI zip.")

    if not XLSX_PATH.exists() or XLSX_PATH.stat().st_size < 1_000_000:
        import urllib.request

        print("Downloading", UCI_ZIP_URL)
        urllib.request.urlretrieve(UCI_ZIP_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
    xl = pd.ExcelFile(XLSX_PATH)
    frames = [pd.read_excel(XLSX_PATH, sheet_name=s) for s in xl.sheet_names]
    raw = pd.concat(frames, ignore_index=True)
    return raw, f"UCI static zip; sheets={xl.sheet_names}"


raw, source = load_online_retail_ii()
print("Source:", source)
print("Shape:", raw.shape)
print("Columns:", list(raw.columns))
display(raw.head(3))

# %% [markdown]
# ## 2. EDA (transaction level)

# %%
raw = raw.copy()
raw["Invoice"] = raw["Invoice"].astype(str)
raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
print("Missing %:\n", (100 * raw.isna().mean()).round(2))
print("Cancels:", raw["Invoice"].str.startswith("C").sum())
print("Date range:", raw["InvoiceDate"].min(), "→", raw["InvoiceDate"].max())

daily = raw.dropna(subset=["InvoiceDate"]).assign(d=lambda x: x["InvoiceDate"].dt.date)
daily = daily.groupby("d")["Invoice"].nunique()
fig, ax = plt.subplots(figsize=(11, 3.2))
daily.plot(ax=ax)
ax.set_title("Distinct invoices per day")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. RFM engineering + production preprocessing
#
# **Research-backed steps** used here:
#
# - Aggregate RFM at customer grain (standard CRM practice)
# - **Winsorize** at 1%/99% so KMeans/GMM are not dominated by extreme spenders
# - **log1p** on Frequency & Monetary (heavy right skew in retail)
# - **RobustScaler** (median/IQR) — more stable than StandardScaler with residual outliers
# - Optional **RFM quantile scores** for interpretability (classic 1–5 scores)

# %%
tx = clean_transactions(
    raw,
    invoice_col="Invoice",
    customer_col="Customer ID",
    qty_col="Quantity",
    price_col="Price",
    date_col="InvoiceDate",
)
print(f"Raw rows: {len(raw):,} | Cleaned: {len(tx):,} | Customers: {tx['Customer ID'].nunique():,}")

rfm = build_rfm(
    tx,
    customer_col="Customer ID",
    invoice_col="Invoice",
    date_col="InvoiceDate",
)
rfm["Customer ID"] = rfm["Customer ID"].astype(int)
print("RFM shape:", rfm.shape)
display(rfm[["Recency", "Frequency", "Monetary"]].describe())
print("Skew raw:\n", rfm[["Recency", "Frequency", "Monetary"]].skew())

rfm_scored = rfm_quantile_scores(rfm, n_bins=5)
display(rfm_scored[["R_score", "F_score", "M_score", "RFM_score"]].describe())

feat_cols = ["Recency", "Frequency", "Monetary"]
X, work, scaler, meta = prepare_matrix(
    rfm,
    feat_cols,
    winsorize=True,
    log_cols=["Frequency", "Monetary"],
    scaler="robust",
    lower_q=0.01,
    upper_q=0.99,
)
print("Preprocess meta:", {k: meta[k] for k in ["scaler", "log_cols"]})
print("Winsor bounds:\n", meta["bounds"])
print("X shape:", X.shape)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, c in zip(axes, feat_cols):
    sns.histplot(work[c], bins=40, ax=ax)
    ax.set_title(f"Processed {c}")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(4.5, 3.8))
sns.heatmap(pd.DataFrame(X, columns=feat_cols).corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation (scaled RFM)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Algorithm survey (fixed k=4 shortlist)
#
# Shortlist **families** quickly, then deep-tune k per family. Includes **GMM**, which
# often fits RFM better than spherical KMeans when clusters differ in variance.

# %%
survey = survey_algorithms(
    X,
    models=["kmeans", "minibatch_kmeans", "hclust", "birch", "gmm"],
    n_clusters=4,
    random_state=SEED,
)
print("=== Fixed-k=4 survey ===")
display(survey)
best_two = top_two_families(survey)
print("Best 2 families for deep tune:", best_two)

# %% [markdown]
# ## 5. Deep k-sweep + stability for the best 2 families
#
# For each family we sweep `k = 2..8` and compute:
#
# - Silhouette, Davies–Bouldin, Calinski–Harabasz  
# - **Bootstrap ARI stability** (subsample 70%, 6 boots)  
# - **production_score** (geometry + balance + stability + slight preference for 3–5 segments)

# %%
sweeps = {}
for fam in best_two:
    print(f"\n=== Sweep {fam} ===")
    sw = sweep_k(fam, X, ks=range(2, 9), random_state=SEED, compute_stability=True)
    sweeps[fam] = sw
    display(sw.sort_values("production_score", ascending=False))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(sw["n_clusters"], sw["silhouette"], "o-", color="C0")
    axes[0].set_title(f"{fam}: Silhouette")
    axes[1].plot(sw["n_clusters"], sw["production_score"], "o-", color="C1")
    axes[1].set_title(f"{fam}: production_score")
    if sw["inertia"].notna().any():
        axes[2].plot(sw["n_clusters"], sw["inertia"], "o-", color="C2")
        axes[2].set_title(f"{fam}: inertia (elbow)")
    else:
        axes[2].plot(sw["n_clusters"], sw["stability_ari"], "o-", color="C2")
        axes[2].set_title(f"{fam}: stability ARI")
    for ax in axes:
        ax.set_xlabel("k")
    plt.tight_layout()
    plt.show()

# Global best + second-best distinct families for side-by-side delivery
model_a, k_a, best_row, ranking = select_best_config(sweeps)
print("\n=== Global ranking (top 10) ===")
display(ranking.head(10))

# Ensure two configs for comparison: best overall + best of the other family
configs = [(model_a, k_a, best_row)]
other = [m for m in best_two if m != model_a]
if other:
    sw_o = sweeps[other[0]].sort_values("production_score", ascending=False).iloc[0]
    configs.append((other[0], int(sw_o["n_clusters"]), sw_o))
else:
    # second row of ranking if same family
    if len(ranking) > 1:
        r2 = ranking.iloc[1]
        configs.append((str(r2["model"]), int(r2["n_clusters"]), r2))

print("Configs to implement properly:")
for m, k, r in configs:
    print(
        f"  {m} k={k} | Sil={r['silhouette']:.4f} | DB={r['davies_bouldin']:.3f} | "
        f"CH={r['calinski_harabasz']:.1f} | stab={r.get('stability_ari', np.nan):.3f} | "
        f"prod={r['production_score']:.4f}"
    )

# %% [markdown]
# ## 6. Final models — fit, PCA, profiles

# %%
results = {}
for model_name, k, row in configs:
    labels = labels_for_config(model_name, X, k, random_state=SEED)
    m = internal_metrics(X, labels)
    m["production_score"] = production_score({**m, "stability_ari": row.get("stability_ari", np.nan)})
    print(f"\n### {model_name} k={k}")
    print(m)

    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=10, alpha=0.75)
    ax.set_title(f"PCA — {model_name} k={k} (Sil={m['silhouette']:.3f})")
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    ax.legend(*sc.legend_elements(), title="cluster")
    plt.tight_layout()
    plt.show()

    tmp = rfm.copy()
    tmp["cluster"] = labels
    prof = (
        tmp.groupby("cluster")
        .agg(
            n=("Customer ID", "count"),
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

    # name segments heuristically
    names = {}
    for _, pr in prof.iterrows():
        cl = int(pr["cluster"])
        if pr["mean_R"] <= prof["mean_R"].median() and pr["mean_M"] >= prof["mean_M"].median():
            names[cl] = "Champions / Loyal high-value"
        elif pr["mean_R"] > prof["mean_R"].quantile(0.66):
            names[cl] = "At-risk / Lapsed"
        elif pr["mean_F"] <= prof["mean_F"].median():
            names[cl] = "Promising / Low frequency"
        else:
            names[cl] = "Core / Steady"
    print("Heuristic labels:", names)

    results[f"{model_name}_k{k}"] = {
        "model": model_name,
        "k": k,
        "labels": labels,
        "metrics": m,
        "profile": prof,
        "names": names,
        "row": row,
    }

# %% [markdown]
# ## 7. Side-by-side comparison & preferred production model

# %%
cmp = []
for key, res in results.items():
    m = res["metrics"]
    cmp.append(
        {
            "config": key,
            "model": res["model"],
            "k": res["k"],
            "silhouette": m["silhouette"],
            "davies_bouldin": m["davies_bouldin"],
            "calinski_harabasz": m["calinski_harabasz"],
            "largest_pct": m["largest_cluster_pct"],
            "min_pct": m["min_cluster_pct"],
            "stability_ari": res["row"].get("stability_ari", np.nan),
            "production_score": m["production_score"],
        }
    )
cmp_df = pd.DataFrame(cmp).sort_values("production_score", ascending=False)
display(cmp_df)
preferred_key = cmp_df.iloc[0]["config"]
pref = results[preferred_key]
print("PREFERRED PRODUCTION CONFIG:", preferred_key)

# %% [markdown]
# ## 8. Manager brief (preferred model)

# %%
prof = pref["profile"]
print("MANAGER BRIEF — Online Retail II (production pipeline)")
print("=" * 64)
print(
    f"Model: {pref['model']} with k={pref['k']} | "
    f"Silhouette={pref['metrics']['silhouette']:.3f} | "
    f"DB={pref['metrics']['davies_bouldin']:.3f} | "
    f"CH={pref['metrics']['calinski_harabasz']:.1f} | "
    f"stability_ARI={pref['row'].get('stability_ari', float('nan')):.3f}"
)
print(f"Customers segmented: {len(rfm):,}")
print()
actions = {
    "Champions / Loyal high-value": "VIP retention, early access, dedicated service — protect revenue concentration.",
    "At-risk / Lapsed": "Win-back sequence with modest incentive timed to recency decay.",
    "Promising / Low frequency": "Cross-sell / frequency drivers (bundles, loyalty stamps).",
    "Core / Steady": "Efficient lifecycle automation; test moderate upsell.",
}
for _, pr in prof.iterrows():
    cl = int(pr["cluster"])
    name = pref["names"].get(cl, f"Segment {cl}")
    act = actions.get(name, "Define a test-and-learn offer for this profile.")
    print(
        f"[{name}] cluster={cl}: n={int(pr['n'])} ({pr['pct_n']:.1f}% cust, {pr['pct_M']:.1f}% revenue) | "
        f"R={pr['mean_R']:.0f}d F={pr['mean_F']:.1f} M=£{pr['mean_M']:.0f} (median £{pr['median_M']:.0f})"
    )
    print(f"  → {act}")

# %% [markdown]
# ## 9. Production notes & limitations
#
# - **Preprocessing** (winsorize + log + robust scale) is part of the model artifact — reapply the same bounds at scoring time.
# - **Stability ARI** is a subsample self-consistency check, not a guarantee of future-period stability.
# - Re-fit periodically (e.g. monthly) as the customer base drifts.
# - No supervised accuracy exists for segments; validate with **campaign A/B tests**.
