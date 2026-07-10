# Customer Segmentation

**Unsupervised clustering of real retail customers across three UCI datasets** — production-minded preprocessing (winsorize + log + RobustScaler), multi-algorithm survey (KMeans / MiniBatch / Ward / Birch / **GMM**), k-sweeps with **bootstrap stability (ARI)**, and a composite **production_score** for model selection.

| Audience | Start here |
|----------|------------|
| **Portfolio / technical reviewers** | [Architecture & design decisions](#architecture--design-decisions) → [Real evidence](#real-evidence-from-executed-notebooks) → [Limitations](#honest-limitations) |
| **Hands-on users** | [Quick start](#quick-start) → [Operate the notebooks](#operate-the-notebooks) → [Troubleshooting](#troubleshooting) → [Extend the project](#extend-the-project) |
| **Tutorial learners** | [What this project teaches](#what-this-project-teaches) → [Concepts](#concepts-unsupervised-segmentation) → [Implementation flow](#implementation-flow-step-by-step) → notebooks in order |

**License:** MIT for code and notebooks ([LICENSE](LICENSE)). Datasets remain under UCI / CC BY 4.0 terms — not re-licensed by this repo.

**Community:** [Contributing](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md) · [Open an issue](https://github.com/pypi-ahmad/customer-segmentation/issues/new/choose)

---

## Why this exists

Customer segmentation answers: *Who are our distinct customer groups, and what should we do differently for each?*

Most retail data has **no labeled “segment” column**. You cannot train a classifier and report accuracy. Instead you:

1. Engineer behavioral features (classically **RFM**: Recency, Frequency, Monetary).
2. Cluster in feature space with unsupervised algorithms.
3. Judge quality with **internal** cluster metrics and **business-readable** profiles.
4. Act on segments (VIP retention, reactivation, automation tiers).

This repository implements that loop **three times** on real UCI data so you can compare:

| Notebook | Dataset | Nature | Feature design |
|----------|---------|--------|----------------|
| `notebooks/01_online_retail_ii_segmentation.ipynb` | [Online Retail II (UCI 502)](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | ~1.07M transactions, Dec 2009–Dec 2011 | Engineered RFM per customer |
| `notebooks/02_wholesale_customers_segmentation.ipynb` | [Wholesale Customers (UCI 292)](https://archive.ics.uci.edu/dataset/292/wholesale+customers) | 440 customer-level rows | Log annual spend across 6 categories |
| `notebooks/03_online_retail_segmentation.ipynb` | [Online Retail (UCI 352)](https://archive.ics.uci.edu/dataset/352/online+retail) | ~542k transactions, Dec 2010–Dec 2011 | Engineered RFM (different column names than II) |

Each notebook is **tutorial-style** (markdown teaches *why* before *how*) and **fully executed** (real numbers, plots, and segment profiles in the saved `.ipynb`).

---

## What this project is *not*

- **Not supervised learning.** No accuracy, F1, confusion matrix, or classification report as model scores.
- **Not an LLM / RAG / chat product.** No Ollama, no embeddings, no prompts — classical tabular clustering only.
- **Not a revenue forecast.** Monetary/spend means are **descriptive segment value**, not predicted future sales.
- **Not black-box AutoML.** Algorithm survey + sklearn re-fits live in `segmentation/`; selection uses an explicit **production_score**, not a hidden leaderboard.

---

## Repository map

```text
Customer Segmentation/
├── LICENSE                 # MIT (code/notebooks)
├── README.md               # this file
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── pyproject.toml          # uv project + dependencies
├── uv.lock
├── .python-version         # 3.13.13
├── segmentation/           # production helper package
│   ├── preprocess.py       # clean, RFM, winsorize, log, RobustScaler
│   ├── metrics.py          # Silhouette/DB/CH, production_score, stability ARI
│   └── selection.py        # survey, k-sweep, fit_model (incl. GMM)
├── data/                   # local UCI cache (large; gitignored)
└── notebooks/
    ├── 01_online_retail_ii_segmentation.{py,ipynb}
    ├── 02_wholesale_customers_segmentation.{py,ipynb}
    └── 03_online_retail_segmentation.{py,ipynb}
```

- **`segmentation/`** is the reusable production core imported by notebooks.
- **`.py`** files are [jupytext](https://jupytext.readthedocs.io/) sources; **`.ipynb`** are fully executed.

---

# For portfolio evaluators

## Architecture & design decisions

### High-level pipeline (production)

```text
Data (UCI) → EDA → RFM / spend features
    → Winsorize 1–99% → log1p (skewed cols) → RobustScaler
    → Family survey @ k=4 (KMeans, MiniBatchKMeans, Ward, Birch, GMM)
    → Deep k-sweep (2..8) + bootstrap stability ARI
    → production_score ranking → top configs
    → PCA + business profiles + manager brief
    → (NB2) Channel/Region sanity crosstab only
```

### Decision table (what we chose and why)

| Decision | Choice | Tradeoff / research basis |
|----------|--------|---------------------------|
| Learning setup | Unsupervised clustering | No labels → internal metrics + profiles only |
| Outliers | **Winsorize 1%/99%** before log | KMeans/GMM distorted by retail mega-buyers; common RFM practice |
| Transform | **log1p** on F/M or all spends | Right-skew (mean ≫ median) on real UCI pulls |
| Scale | **RobustScaler** (median/IQR) | More stable than StandardScaler with residual tails |
| Model families | KMeans++, MiniBatch, Ward, Birch, **GMM** | GMM allows unequal variance; MiniBatch scales |
| k selection | Sweep 2–8 + Silhouette/DB/CH + **stability ARI** | Elbow alone is under-specified for ops |
| Selection objective | **`production_score`** | Geometry + balance + stability + slight 3–5 segment bias for actionability |
| Hold-outs (NB2) | `Channel`/`Region` never in features | Sanity check only — not accuracy |
| Packaging | `segmentation/` library | Notebooks stay tutorials; logic is testable/reusable |
| Python | **3.13.13** + uv | Locked deps; PyCaret pin optional for env parity |

### Ranking rubric for the survey (best 2)

A model is **viable** only if all hold:

- no fit error  
- Silhouette is finite  
- **2 ≤ n_clusters ≤ 25** (excludes 1-blob and micro-fragmentation like 500 affinity clusters)  
- **noise ≤ 30%** (DBSCAN/OPTICS)  
- **largest cluster ≤ 85%** of mass (rejects “almost everything in one cluster”)

Among viable models, a composite score prefers higher Silhouette, higher Calinski–Harabasz, lower Davies–Bouldin, and slightly penalizes extreme imbalance.

If fewer than two models pass, the notebook **relaxes once** (noise ≤ 50%, largest ≤ 95%) and logs a warning rather than inventing winners.

### Why “survey then re-implement” instead of only PyCaret?

1. **Survey** answers: *which families of algorithms even make sense on this matrix?*  
2. **Re-implement** answers: *given the winner family, what k/eps should we actually ship?*  

PyCaret’s default `num_clusters=4` is only a survey default. Part 2 sweeps `k ∈ [2, 10]` (or density hyperparameters) with elbow + Silhouette plots.

### Preprocessing consistency

Features enter PyCaret **already log-transformed and scaled** with `normalize=False`, so Part 1 and Part 2 share the same matrix. Double-normalization would silently change distances and make “best algorithm” claims non-comparable.

---

## Reproducibility

| Control | How it is enforced |
|---------|-------------------|
| Python version | `.python-version` → `3.13.13` |
| Dependencies | `uv.lock` + `pyproject.toml` (`pycaret==4.0.0a8`, etc.) |
| Kernel name | `customer-segmentation-project` |
| Randomness | `SEED = 42` for NumPy / sklearn / PyCaret `session_id` |
| Data provenance | UCI ids documented; 502 falls back to official static zip URL |
| Execution | `jupyter nbconvert --execute` with pinned kernel and long timeout |
| Source of truth for results | Executed notebook outputs (this README quotes them) |

### Environment snapshot (verified on last full run)

| Package | Version |
|---------|---------|
| Python | **3.13.13** |
| pycaret | **4.0.0a8** |
| scikit-learn | **1.9.0** |
| pandas | **3.0.3** |
| numpy | **2.5.1** |

Reproduce the stack:

```bash
cd "/home/ahmad/AI/Customer Segmentation"
uv sync
uv run python -c "import sys,pycaret,sklearn; print(sys.version.split()[0], pycaret.__version__, sklearn.__version__)"
```

---

## Real evidence (from executed notebooks)

All figures below come from a **full re-execution after the production upgrade** (0 error cells; plots embedded). They are **not** copied from blogs.

### Improvement vs prior baseline (same data, better pipeline)

| Notebook | Prior preferred | Prior Silhouette | **New preferred** | **New Silhouette** | Other gains |
|----------|-----------------|------------------|-------------------|--------------------|-------------|
| 01 Retail II | KMeans k=2 | 0.419 | **KMeans k=2** | **0.427** | CH 5793→**6290**, stability ARI **0.968**, revenue conc. **91.3%** |
| 02 Wholesale | HClust k=2 | 0.258 | **MiniBatchKMeans k=2** | 0.252 | CH 135→**149**, Channel purity **97.1%** Horeca, stab **0.756** |
| 03 Retail 352 | KMeans k=3 | 0.416 | **KMeans k=3** | **0.426** | CH 4395→**4917**, stability ARI **0.974**, VIP rev **82.2%** |

Wholesale Silhouette is essentially flat (−0.006) while **Channel recovery and CH improved** — production_score intentionally balances geometry with operational structure, not Silhouette alone.

### Execution health (latest run)

| Notebook | Code cells w/ outputs | Errors |
|----------|----------------------|--------|
| 01 Online Retail II | 9/9 | 0 |
| 02 Wholesale | 8/8 | 0 |
| 03 Online Retail | 7/7 | 0 |

---

### Notebook 1 — Online Retail II (UCI 502)

| Stage | Real output |
|-------|-------------|
| Source | Official UCI zip (`online_retail_II.xlsx`) |
| Raw / cleaned / customers | **1,067,371** / **805,549** / **5,878** |
| Preprocess | Winsorize 1–99% → log1p(F,M) → RobustScaler |
| Best families | **kmeans**, **minibatch_kmeans** |
| Preferred | **KMeans k=2** |

| Config | Silhouette ↑ | DB ↓ | CH ↑ | Stability ARI | production_score |
|--------|--------------|------|------|---------------|------------------|
| **KMeans k=2** | **0.427** | **0.870** | **6290** | **0.968** | **0.662** |
| MiniBatchKMeans k=2 | 0.427 | 0.872 | 6288 | 0.946 | 0.659 |

| Segment | n | % cust | % revenue | R (d) | F | M mean | Label |
|---------|---|--------|-----------|-------|---|--------|-------|
| Champions | 2,705 | 46.0% | **91.3%** | 70 | 11.5 | **£5,990** | VIP protect |
| At-risk / Lapsed | 3,173 | 54.0% | 8.7% | 311 | 1.8 | £486 | Win-back |

---

### Notebook 2 — Wholesale Customers (UCI 292)

| Stage | Real output |
|-------|-------------|
| Source | `ucimlrepo` id=292 original — **440×8** |
| Deli column | **`Delicassen`** |
| Preprocess | Winsorize → log1p(all 6 spends) → RobustScaler |
| Preferred | **MiniBatchKMeans k=2** (Sil 0.252, CH 149, stab 0.756) |

| Segment | n | % accounts | % spend | Mean total | Top category |
|---------|---|------------|---------|------------|--------------|
| 1 | 201 | 45.7% | **57.4%** | ~41,732 | Grocery ≈ 13,709 |
| 0 | 239 | 54.3% | 42.6% | ~26,073 | Fresh ≈ 14,826 |

**Channel sanity (descriptive only):** Cluster 0 → Channel 1 (Horeca) **97.1%**; Cluster 1 → Channel 2 (Retail) **67.2%**.

---

### Notebook 3 — Online Retail (UCI 352)

| Stage | Real output |
|-------|-------------|
| Raw / cleaned / customers | **541,909** / **397,884** / **4,338** |
| Preferred | **KMeans k=3** |

| Config | Silhouette | DB | CH | Stability ARI | production_score |
|--------|------------|----|----|---------------|------------------|
| **KMeans k=3** | **0.426** | **0.801** | **4917** | **0.974** | **0.656** |
| MiniBatchKMeans k=3 | 0.426 | — | — | 0.914 | 0.649 |

| Segment | n | % cust | % revenue | R (d) | F | M mean |
|---------|---|--------|-----------|-------|---|--------|
| VIP | 1,397 | 32.2% | **82.2%** | 29 | 9.5 | **£5,245** |
| Mid | 1,956 | 45.1% | 12.4% | 54 | 2.0 | £566 |
| Lapsed / low | 985 | 22.7% | 5.4% | 254 | 1.4 | £485 |

---

## Honest limitations

1. **Internal metrics ≠ business lift.** Silhouette / production_score optimize geometry and stability, not campaign ROI. Validate with A/B tests.
2. **production_score is a design choice.** Weights (stability, 3–5 segment preference, imbalance penalties) are documented in `segmentation/metrics.py` and can be tuned.
3. **Wholesale Silhouette can trade off against Channel purity.** This run slightly lowered Silhouette while improving CH and Channel recovery — reported honestly.
4. **UCI 502 Python import is broken** in `ucimlrepo`; notebooks use the official UCI static zip for the same dataset id.
5. **Bootstrap ARI** measures self-consistency on subsamples, not future-period stability. Re-fit on a rolling window in production.
6. **Winsor bounds and scaler** are part of the scoring artifact — must be persisted for inference on new customers.
7. **No causal claims.** Manager actions are hypotheses.
8. **MIT covers code only**; UCI data stays under original terms.
9. **Distribution shift** across years/verticals is expected — do not ship 2011 UK segments unchanged into a new market.

---

# For hands-on users

## Prerequisites

- Linux (verified), macOS/Windows with `uv` should work with minor path changes  
- [uv](https://docs.astral.sh/uv/) installed  
- Network access for UCI / `ucimlrepo` (and zip for dataset 502 if not cached)  
- ~2 GB free disk for env + data + notebook outputs  

## Quick start

```bash
cd "/home/ahmad/AI/Customer Segmentation"

# 1. Install locked dependencies into .venv
uv sync

# 2. Register Jupyter kernel used by notebooks
uv run python -m ipykernel install --user \
  --name customer-segmentation-project \
  --display-name "Python (customer-segmentation-project)"

# 3. Optional: verify stack
uv run python -c "
import sys, pycaret, sklearn
print(sys.version.split()[0], pycaret.__version__, sklearn.__version__)
from pycaret.clustering import ClusteringExperiment
print('ClusteringExperiment OK')
"

# 4. Open Jupyter and select kernel: customer-segmentation-project
uv run jupyter lab
# or: uv run jupyter notebook
```

Open notebooks in order:

1. `notebooks/02_wholesale_customers_segmentation.ipynb` (fastest smoke test, 440 rows)  
2. `notebooks/01_online_retail_ii_segmentation.ipynb` (largest; needs Retail II file under `data/`)  
3. `notebooks/03_online_retail_segmentation.ipynb`  

## Data acquisition

### Automatic (preferred)

- **292 / 352:** notebooks call `ucimlrepo.fetch_ucirepo` live.  
- **502:** notebook tries `fetch_ucirepo(id=502)`, then loads:

```text
https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
```

into `data/online_retail_II.xlsx`.

### Manual (if 502 download fails)

1. Download: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip  
2. Dataset page: https://archive.ics.uci.edu/dataset/502/online+retail+ii  
3. Place file as either:

```text
data/online_retail_ii.zip
```

or extract:

```text
data/online_retail_II.xlsx
```

## Operate the notebooks

### Interactive

```bash
uv run jupyter lab
# Kernel → customer-segmentation-project
# Run All
```

### Headless re-execution (CI-style)

```bash
NB=02_wholesale_customers_segmentation   # or 01_... / 03_...

uv run jupytext --to ipynb "notebooks/${NB}.py"

uv run jupyter nbconvert --to notebook --execute "notebooks/${NB}.ipynb" \
  --output "${NB}.ipynb" \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=customer-segmentation-project
```

**Timeouts:** 3600s is usually enough for NB2; use **7200s** for NB1/NB3 (RFM + multi-algorithm survey).

### Do not set `MPLBACKEND=Agg` for nbconvert

Agg prevents plot images from embedding in the `.ipynb`. Notebooks force `matplotlib_inline` so executed outputs include charts.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `DatasetNotFoundError` for id 502 | UCI marks 502 non-importable via `ucimlrepo` | Expected; ensure zip/xlsx under `data/` or allow network for static URL |
| `Pycaret only supports python 3.9–3.11` | Wrong PyCaret major (3.3.x) | Use this repo’s lock: `pycaret==4.0.0a8` via `uv sync` |
| Kernel not found | Kernel not registered | Re-run `ipykernel install --name customer-segmentation-project` |
| Empty plot outputs after execute | Backend was Agg / no inline | Unset `MPLBACKEND`; re-execute with project notebooks (inline set in setup cell) |
| `ModuleNotFoundError: openpyxl` | Missing Excel reader for 502 | `uv add openpyxl` (already in `pyproject.toml`) |
| Survey picks weird density model | Noise / micro-clusters | Check viability table in notebook; hard filters should drop them — report if not |
| Very slow `ap` / `optics` / `sc` | Algorithm scaling | Survey still records times; Part 2 only re-implements best 2 |
| Column errors on Retail II vs I | Different schemas | II uses `Invoice`/`Price`/`Customer ID`; 352 uses `InvoiceNo`/`UnitPrice`/`CustomerID` |
| Wholesale `Delicatessen` KeyError | Spelling | Live column is **`Delicassen`** — notebooks resolve it dynamically |
| Memory pressure on NB1 | 1M+ transaction rows | Use project machine with ≥8 GB RAM; free frames after RFM if extending |

## Extend the project

Practical extension points (smallest first):

1. **Change seed / k range** — edit `SEED` and `range(2, 11)` in the notebook `.py`, reconvert, re-execute.  
2. **Add RFM+ features** — e.g. average basket size, tenure, country one-hot (watch dimensionality).  
3. **Swap survey candidates** — `candidate_ids` list in Part 1.  
4. **Export segment tables** — `rfm.assign(cluster=labels).to_csv("data/segments_nb1.csv")`.  
5. **CLI wrapper** (optional) — add a small Typer/Click entrypoint that runs one notebook via `nbconvert` for ops users.  
6. **Compare windows** — treat NB1 vs NB3 as longitudinal study of the same retailer with different horizons.  
7. **Stability** — bootstrap subsample clustering and measure adjusted Rand index between runs.

Keep the contract: **no fake supervised metrics**, and any external labels stay out of the feature matrix unless you intentionally switch problem type.

---

# For tutorial learners

## What this project teaches

1. How **unsupervised** problems differ from classification.  
2. How to build **RFM** from messy transactions (cancels, missing IDs, prices).  
3. Why **log transforms** and **scaling** matter for distance-based clustering.  
4. How to **survey** algorithms quickly, then **own** the final fit.  
5. How to read **Silhouette / Davies–Bouldin / Calinski–Harabasz** without treating them as accuracy.  
6. How to write **manager-facing** segment stories from cluster means.  

Work through notebooks **02 → 01 → 03** if you are new (smallest → largest → related RFM variant).

---

## Concepts: unsupervised segmentation

### Supervised vs unsupervised (critical)

| | Supervised (e.g. churn) | This project |
|--|-------------------------|--------------|
| Label | Known (churned / not) | **None** for “segment id” |
| Goal | Predict label | **Discover** structure |
| Metrics | Accuracy, AUC, F1 | Silhouette, DB, CH + profiles |
| Leakage risk | Future features into train | Using Channel as a feature then “validating” on Channel |

If you report accuracy here, you either invented labels or mis-framed the problem.

### RFM features

| Feature | Definition used here | Intuition |
|---------|----------------------|-----------|
| **Recency** | Days since last purchase relative to dataset max date | Lower is “hotter” |
| **Frequency** | Distinct invoice count | Engagement |
| **Monetary** | Sum of quantity × unit price | Historical value |

Cleaning before RFM:

- Drop cancellation invoices (`Invoice` / `InvoiceNo` starts with `C`).  
- Drop missing customer IDs.  
- Keep only positive quantity and price.  

### Why log1p on Frequency and Monetary?

Retail spend is **right-skewed** (few customers with huge totals). Euclidean clustering on raw pounds will chase outliers. In this run:

- NB1: Frequency skew **12.64**, Monetary **25.31** → log1p  
- NB3: Frequency **12.07**, Monetary **19.32** → log1p  

Recency is usually less extreme and is kept on the day scale, then all three axes are standardized.

### Internal cluster metrics (what the numbers mean)

| Metric | Direction | Intuition |
|--------|-----------|-----------|
| **Silhouette** | Higher better (≈ −1…1) | Cohesion vs separation |
| **Davies–Bouldin** | Lower better | Average “similarity” of each cluster to its nearest other cluster |
| **Calinski–Harabasz** | Higher better | Between-cluster dispersion over within-cluster dispersion |

None of these guarantee marketing ROI. A high Silhouette with a useless 90/10 split is still a bad business segmentation — that is why we filter on **largest cluster %** and read **profiles**.

### Algorithm families (survey menu)

| Id | Idea | When it struggles |
|----|------|-------------------|
| `kmeans` | Spherical blobs, needs k | Non-spherical shapes, outliers |
| `hclust` | Bottom-up merges | Large N (memory/time), choice of linkage |
| `birch` | Hierarchical CF-tree | Very irregular densities |
| `dbscan` / `optics` | Density + noise label −1 | Parameter sensitivity; large noise mass |
| `sc` (spectral) | Graph embedding then cluster | Costly; can collapse mass |
| `meanshift` | Mode seeking | Bandwidth choice; can yield 1 cluster |
| `ap` | Exemplar messages | Often too many micro-clusters |

### Implementation flow (step by step)

```text
1. Setup          versions, seed, kernel name
2. Load data      UCI / zip — verify columns for this dataset only
3. EDA            missingness, distributions, time/country (tx data)
4. Features       RFM or log spend; never feed hold-out labels
5. Scale          StandardScaler on the clustering matrix
6. Survey         PyCaret create_model loop → viability table → best 2
7. Proper fit     sklearn sweeps for each of the 2 winners
8. Visualize      PCA 2D colored by cluster
9. Profile        means in original units + revenue/spend share
10. Compare       which of the two is more actionable
11. Sanity (NB2)  crosstab vs Channel/Region (descriptive only)
12. Manager brief plain-language actions per segment
```

### Reading a manager brief

Good briefs answer four questions:

1. How many segments?  
2. How large is each (customers and value share)?  
3. What behavior defines each (R/F/M or category mix)?  
4. What is **one concrete action** per segment?  

Example from this run (NB3): VIP ~30% of customers ≈ 82% revenue → protect with high-touch retention; mid-tier → growth offers; lapsed → cheap reactivation.

---

## Datasets & citations

| UCI id | Name | Access in this repo |
|--------|------|---------------------|
| 502 | Online Retail II | Official zip (see [manual download](#manual-if-502-download-fails)) |
| 292 | Wholesale customers | `ucimlrepo` |
| 352 | Online Retail | `ucimlrepo` |

- Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D  
- Abreu, N. (2014). *Wholesale customers*. UCI Machine Learning Repository. https://doi.org/10.24432/C5030X  
- Chen, D., Sain, S. L., & Guo, K. (2012). *Online Retail*. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33  

---

## License

Project **source code and notebooks**: [MIT License](LICENSE) © 2026 Ahmad.

**Datasets and any third-party data files** are not covered by that MIT grant; use them under their original UCI / CC BY terms.

---

## Summary for reviewers in one paragraph

This portfolio project demonstrates an end-to-end unsupervised segmentation practice: real multi-source UCI data, leakage-aware feature design (RFM / log spend; Channel held out), a reproducible `uv`+Python 3.13 environment, a deliberate **PyCaret survey → sklearn re-implementation** architecture, hard quality filters against degenerate clusters, fully executed notebooks with embedded plots, and **honest reporting** of internal metrics plus business profiles without inventing classification accuracy. The strongest real finding across retail notebooks is classic power-law value concentration (≈30–45% of customers drive ≈80–90% of observed revenue), which is exactly the kind of structure segmentation is meant to surface for retention and budget allocation.
