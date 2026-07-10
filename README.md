# Customer Segmentation

**Unsupervised clustering of real retail customers across three UCI datasets** — survey many algorithms with PyCaret, re-implement the best two with scikit-learn, and explain segments in business language.

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
- **Not AutoML as the deliverable.** PyCaret is a **survey tool** that narrows algorithms quickly; the production-style answer is the **sklearn re-implementation** with real hyperparameter sweeps.

---

## Repository map

```text
Customer Segmentation/
├── LICENSE                 # MIT (code/notebooks)
├── README.md               # this file
├── pyproject.toml          # uv project + dependencies
├── uv.lock                 # locked resolves
├── .python-version         # 3.13.13
├── data/                   # local UCI cache (large; gitignored)
│   ├── online_retail_ii.zip
│   └── online_retail_II.xlsx
└── notebooks/
    ├── 01_online_retail_ii_segmentation.{py,ipynb}
    ├── 02_wholesale_customers_segmentation.{py,ipynb}
    └── 03_online_retail_segmentation.{py,ipynb}
```

- **`.py`** files are [jupytext](https://jupytext.readthedocs.io/) percent-format sources (easy to diff and re-convert).
- **`.ipynb`** files are the executed deliverables with outputs and plots.

---

# For portfolio evaluators

## Architecture & design decisions

### High-level pipeline

```text
┌──────────────┐   ┌─────────┐   ┌──────────────────┐   ┌─────────────────┐
│ Data load    │ → │ EDA     │ → │ Feature engineer │ → │ Scale features  │
│ UCI / zip    │   │ skew,   │   │ RFM or log spend │   │ StandardScaler  │
└──────────────┘   │ missing │   └──────────────────┘   └────────┬────────┘
                   └─────────┘                                    │
         ┌────────────────────────────────────────────────────────┘
         ▼
┌────────────────────────────┐     best 2 algorithms
│ Part 1: PyCaret survey     │ ──────────────────────────────┐
│ ClusteringExperiment       │                               ▼
│ models + labels + metrics  │     ┌─────────────────────────────────────┐
└────────────────────────────┘     │ Part 2: sklearn re-implementation │
                                   │ k / eps sweep (elbow + silhouette)│
                                   │ final Silhouette / DB / CH          │
                                   │ PCA scatter + business profiles     │
                                   └─────────────────────────────────────┘
```

### Decision table (what we chose and why)

| Decision | Choice | Tradeoff |
|----------|--------|----------|
| Learning setup | Unsupervised clustering | No labels → cannot claim “accuracy”; must use internal metrics + profiles |
| Survey tool | **PyCaret 4.0.0a8** `ClusteringExperiment` | Fast multi-algorithm scan; API is alpha; metrics pulled via sklearn |
| Final models | **scikit-learn** re-fits of the survey winners | More control, honest k-sweeps, reproducible estimators |
| Feature scale | `StandardScaler` after optional `log1p` | Euclidean methods need comparable axes; log reduces heavy-tail domination |
| RFM vs raw transactions | Aggregate to customer grain before clustering | Transaction-level clustering is not “customer segments” |
| Hold-outs (NB2) | `Channel`, `Region` never enter the feature matrix | Used only as **external sanity check**, not training targets |
| Best-2 ranking | Hard filters then composite score | Drops noise-dominated / single-blob solutions even if Silhouette looks high |
| Python runtime | **3.13.13** + uv | Prefer modern Python; forces PyCaret 4 alpha (3.3.x blocks ≥3.12) |
| Notebooks as product | Fully executed `.ipynb` | Reviewers see real numbers without re-running; sources stay as jupytext `.py` |

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

All figures below were produced on a full re-execution of the three notebooks (0 error cells; plots embedded). They are **not** copied from blogs.

### Execution health

| Notebook | Code cells with outputs | Embedded plots | Approx. size | Errors |
|----------|-------------------------|----------------|--------------|--------|
| 01 Online Retail II | 12/12 | 9 | ~836 KB | 0 |
| 02 Wholesale | 9/9 | 9 | ~784 KB | 0 |
| 03 Online Retail | 8/8 | 9 | ~768 KB | 0 |

---

### Notebook 1 — Online Retail II (UCI 502)

| Stage | Real output |
|-------|-------------|
| Source | Official UCI zip (`online_retail_II.xlsx`, sheets 2009–2010 + 2010–2011) — `ucimlrepo` import unavailable for id 502 |
| Raw rows | **1,067,371** |
| Cleaned transactions | **805,549** (drop cancels, missing `Customer ID`, non-positive qty/price) |
| Customers | **5,878** RFM rows |
| Skew | Frequency skew **12.64**, Monetary skew **25.31** → `log1p` applied |
| Survey best 2 | **kmeans**, **hclust** |
| Preferred final model | **KMeans, k = 2** |

**Final internal metrics**

| Model | Params | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Largest % |
|-------|--------|--------------|------------------|---------------------|-----------|
| **KMeans (preferred)** | k=2 | **0.419** | **0.889** | **5793** | 54.5% |
| Hierarchical | k=2 | 0.399 | 0.846 | 5066 | 68.7% |

**Business profiles (KMeans, original RFM units)**

| Segment | n | % customers | % revenue | Avg Recency | Avg Frequency | Avg Monetary |
|---------|---|-------------|-----------|-------------|---------------|--------------|
| 0 — high value / recent | 2,674 | 45.5% | **90.2%** | 62 d | 11.6 | **£5,988** |
| 1 — low value / stale | 3,204 | 54.5% | 9.8% | 316 d | 1.9 | £541 |

Interpretation: nearly half the base produces ~90% of observed revenue. Retention budget belongs on Segment 0; Segment 1 is automation / win-back, not high-touch spend.

---

### Notebook 2 — Wholesale Customers (UCI 292)

| Stage | Real output |
|-------|-------------|
| Source | `fetch_ucirepo(id=292).data.original` |
| Shape | **440 × 8** |
| Deli column (live spelling) | **`Delicassen`** (not “Delicatessen”) |
| Clustering features | Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen |
| Held out | `Channel`, `Region` |
| Survey best 2 | **birch**, **hclust** |
| Preferred final model | **Agglomerative clustering, k = 2** |

**Final internal metrics**

| Model | Params | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Largest % |
|-------|--------|--------------|------------------|---------------------|-----------|
| Birch | k=2 | 0.222 | 1.645 | 124 | 53.4% |
| **HClust (preferred)** | k=2 | **0.258** | **1.600** | **135** | 59.5% |

**Profiles (original spend units)**

| Segment | n | % accounts | % spend | Mean total spend | Strongest mean category |
|---------|---|------------|---------|------------------|-------------------------|
| 0 | 178 | 40.5% | 49.0% | ~40,275 | Grocery ≈ 14,247 |
| 1 | 262 | 59.5% | 51.0% | ~28,437 | Fresh ≈ 15,016 |

**External sanity check (not accuracy)**

| Cluster | n | Dominant Channel | Share of cluster |
|---------|---|------------------|------------------|
| 0 | 178 | Channel **2** (Retail) | **73.6%** |
| 1 | 262 | Channel **1** (Horeca) | **95.8%** |

Spend-only clusters recover a strong Channel structure without ever training on Channel. That supports interpretability; it does **not** mean the clustering “achieved 95% accuracy.”

---

### Notebook 3 — Online Retail (UCI 352)

| Stage | Real output |
|-------|-------------|
| Source | `fetch_ucirepo(id=352).data.original` |
| Columns | `InvoiceNo`, `UnitPrice`, `CustomerID` (differs from dataset II) |
| Raw rows | **541,909** |
| Cleaned | **397,884** |
| Customers | **4,338** |
| Skew | Frequency **12.07**, Monetary **19.32** → `log1p` |
| Survey best 2 | **kmeans**, **hclust** |
| Preferred final model | **KMeans, k = 3** |

**Final internal metrics**

| Model | Params | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Largest % |
|-------|--------|--------------|------------------|---------------------|-----------|
| **KMeans (preferred)** | **k=3** | **0.416** | **0.825** | **4395** | 46.9% |
| Hierarchical | k=2 | 0.400 | 0.892 | 3691 | 64.8% |

**Profiles (KMeans k=3)**

| Segment | n | % customers | % revenue | Avg Recency | Avg Frequency | Avg Monetary |
|---------|---|-------------|-----------|-------------|---------------|--------------|
| 1 — VIP | 1,323 | 30.5% | **81.6%** | 29 d | 9.8 | **£5,494** |
| 0 — mid | 2,036 | 46.9% | 14.1% | 54 d | 2.0 | £615 |
| 2 — lapsed / low | 979 | 22.6% | 4.4% | 254 d | 1.4 | £398 |

Single-year data yields a **three-way** split under Silhouette (VIP / mid / lapsed), whereas the longer Online Retail II window preferred a coarser **two-way** split in this run. Same retailer family, different observation window and feature realizations — numbers are not interchangeable.

---

## Honest limitations

1. **Internal metrics ≠ business lift.** Silhouette can favor geometry that is not the most actionable marketing cut. Always read profiles and mass balance.
2. **k often lands at 2.** That can be statistically preferred but operationally coarse. Notebooks report the metric-driven choice; stakeholders may still request k=3–5 for campaign design.
3. **UCI 502 Python import is broken** in `ucimlrepo` (dataset exists but is not importable). The notebook uses the official UCI static zip for the **same dataset id**.
4. **PyCaret 4.0.0a8 is alpha.** Chosen because PyCaret 3.3.x hard-fails on Python ≥ 3.12. Pin and re-verify after upgrades.
5. **Density methods** can report high Silhouette on non-noise points while labeling most of the base as noise. Hard filters drop those from best-2 contention.
6. **No causal claims.** Segment actions in the manager briefs are hypotheses for A/B testing, not proven treatments.
7. **MIT does not cover data.** UCI datasets remain under their original licenses (typically CC BY 4.0).
8. **Distribution shift.** Segments from 2009–2011 UK gift retail will not automatically transfer to another vertical or year.

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
