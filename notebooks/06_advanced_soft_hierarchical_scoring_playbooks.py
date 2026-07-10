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
# # 06 — Advanced path: soft segments, hierarchy, time stability, scoring API
#
# ## Why
#
# Production CRM rarely wants a single hard integer forever:
#
# - **Soft membership** — probability of being VIP vs Core (GMM responsibilities)  
# - **Hierarchy** — first VIP vs rest, then sub-segment the rest  
# - **Stability over time** — re-fit at successive as-of dates; track ARI / size drift  
# - **Scoring API** — `customer features → segment + confidence + action`  
# - **Playbooks** — named KPI-linked actions for A/B design  
#
# Notebooks 01–03 remain the baseline learning path.

# %%
from __future__ import annotations

import json
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

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from segmentation.advanced.features_plus import (
    RFM_PLUS_CLUSTER_COLS,
    build_rfm_plus,
    load_retail_transactions,
)
from segmentation.advanced.scoring import SegmentScorer, default_playbooks
from segmentation.advanced.soft_hierarchical import (
    fit_gmm_soft,
    hierarchical_two_level,
    soft_summary,
)
from segmentation.advanced.stability_time import rolling_refit_stability
from segmentation.advanced.whales import split_whales
from segmentation.preprocess import prepare_matrix

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Features (RFM+) + whale holdout

# %%
tx, meta = load_retail_transactions(ROOT, prefer="ii")
print(meta)
feat = build_rfm_plus(tx)
core, whales, thr = split_whales(feat, upper_q=0.99)
print(f"core={len(core)} whales={len(whales)} thr={thr:.2f}")

cols = [c for c in RFM_PLUS_CLUSTER_COLS if c in core.columns]
log_cols = [c for c in cols if any(k in c for k in ["Frequency", "Monetary", "AvgOrder", "StdOrder", "NUnique", "FreqLast"])]
X, _, _, _ = prepare_matrix(core, cols, winsorize=True, log_cols=log_cols, scaler="robust")
print("X", X.shape)

# %% [markdown]
# ## 2. Soft GMM segments
#
# Hard labels force a single assignment. **Responsibilities** let ops treat a customer
# as 70% VIP / 30% Growth — useful for gradual offers and confidence thresholds.

# %%
soft = fit_gmm_soft(X, n_components=4, seed=SEED)
print("GMM Silhouette (hard):", soft["silhouette"], "BIC:", soft["bic"])
display(soft_summary(soft["proba"], soft["labels"]))

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(soft["proba"].max(axis=1), bins=30, color="C2")
ax.set_title("Distribution of max soft membership (confidence)")
ax.set_xlabel("max cluster probability")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Hierarchical two-level cut
#
# **Level 1:** VIP vs rest (protect revenue concentration).  
# **Level 2:** split the rest into 3 actionable cores (growth / steady / lapsed-ish).
#
# This matches how budgets are set in practice (VIP program vs mass CRM).

# %%
# value axis = monetary from features aligned to core rows
val = core["Monetary"].to_numpy()
hier = hierarchical_two_level(X, seed=SEED, value_axis=val)
print("VIP n=", hier["n_vip"], "rest n=", hier["n_rest"])
print("name_map", hier["name_map"])
core = core.copy()
core["hier_label"] = hier["labels"]
core["hier_name"] = core["hier_label"].map(hier["name_map"])
display(
    core.groupby("hier_name").agg(
        n=("customer_id", "count"),
        mean_M=("Monetary", "mean"),
        mean_R=("Recency", "mean"),
        mean_F=("Frequency", "mean"),
        mean_trend=("MonetaryTrend90", "mean"),
    ).sort_values("mean_M", ascending=False)
)

# %% [markdown]
# ## 4. Rolling time stability (monitoring)
#
# Bootstrap ARI (notebook 02-style production_score) checks resample stability.  
# **Rolling as-of re-fits** check whether segments still mean the same thing as time moves.

# %%
stab = rolling_refit_stability(tx, n_points=4, n_clusters=4, seed=SEED)
print("Rolling re-fit stability:")
display(stab)
if len(stab):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(len(stab)), stab["ari"], "o-", label="ARI")
    ax.plot(range(len(stab)), 1 - stab["size_l1_drift"], "s--", label="1 - size_drift")
    ax.set_ylim(0, 1.05)
    ax.set_title("Segment stability across successive as-of dates")
    ax.legend()
    plt.tight_layout()
    plt.show()
    print("Mean ARI:", float(stab["ari"].mean()))

# %% [markdown]
# ## 5. Scoring API demo
#
# Production pattern: fit once on training features, `predict` for new rows.
# Whales short-circuit to the Whale playbook.

# %%
scorer = SegmentScorer(
    cols,
    n_clusters=4,
    log_cols=log_cols,
    seed=SEED,
    whale_threshold=thr,
    monetary_col="Monetary",
)
# fit on full feature table (including whales) so scaler sees full support
scorer.fit(feat)
scored = scorer.predict(feat.head(20))
print("Sample scores:")
display(scored)
print("\nPlaybooks:")
display(scorer.playbook_table())

# export a tiny artifact for demos
art_dir = ROOT / "artifacts"
art_dir.mkdir(exist_ok=True)
sample_path = art_dir / "sample_scores.csv"
scorer.predict(feat).to_csv(sample_path, index=False)
print("Wrote", sample_path, "rows", len(feat))

# %% [markdown]
# ## 6. A/B-style experiment design (tutorial, not a live test)
#
# For each playbook, define **who**, **what**, **metric**. That is how you turn
# segments into scientific marketing rather than pretty PCA plots.

# %%
ab = []
for name, pb in default_playbooks().items():
    ab.append({
        "segment": name,
        "hypothesis": f"If we apply '{pb.action}' to {name}, then {pb.success_metric} improves vs holdout.",
        "channel": pb.channel,
        "budget_priority": pb.budget_priority,
        "primary_metric": pb.success_metric,
        "offer_hint": pb.offer_hint,
        "design": "50/50 random split within segment; 4-week window; guardrail: margin",
    })
ab_df = pd.DataFrame(ab)
display(ab_df)
ab_df.to_csv(art_dir / "ab_experiment_design.csv", index=False)

# %% [markdown]
# ## 7. How the full advanced track improves on 01–03
#
# | Layer | 01–03 baseline | Advanced 04–06 | Why better for production |
# |-------|----------------|----------------|---------------------------|
# | Features | R,F,M (or 6 spends) | RFM+ trajectories, cancels, tenure | Separates real behaviors |
# | Outliers | Winsorize only | Whale **policy** segment | Ops + geometry |
# | Evaluation | Silhouette / CH / DB | **Future £ & retention** | Business truth |
# | Labels | Hard only | Soft GMM + hierarchy | Budgeting reality |
# | Monitoring | Seed stability | Rolling as-of ARI | Drift detection |
# | Delivery | Notebook tables | **Scorer + playbooks + A/B design** | Runnable CRM loop |
#
# Keep 01–03 for teaching fundamentals. Use 04–06 when you need **portfolio-grade / production-minded** depth.

# %%
print("DONE notebook 06")
print("Artifacts:", list(art_dir.glob("*")))
