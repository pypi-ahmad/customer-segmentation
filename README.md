# Customer Segmentation

**Unsupervised clustering of real retail customers across three UCI datasets** — production-minded preprocessing (winsorize + log + RobustScaler), multi-algorithm survey (KMeans / MiniBatch / Ward / Birch / **GMM**), k-sweeps with **bootstrap stability (ARI)**, and a composite **production_score** for model selection.

| Audience | Start here |
|----------|------------|
| **Portfolio / technical reviewers** | [Architecture & design decisions](#architecture--design-decisions) → [Real evidence](#real-evidence-from-executed-notebooks) → [Advanced track 04–06](#advanced-track-notebooks-0406--kept-separate-from-0103) → [Limitations](#honest-limitations) |
| **Hands-on users** | [Quick start](#quick-start) → [Operate the notebooks](#operate-the-notebooks) → [Troubleshooting](#troubleshooting) → [Extend the project](#extend-the-project) |
| **Tutorial learners** | [What this project teaches](#what-this-project-teaches) → [Concepts](#concepts-unsupervised-segmentation) → **[Advanced tutorial](docs/ADVANCED_SEGMENTATION_TUTORIAL.md)** → notebooks **01→03** then **04→06** |

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

This repository implements that loop on real UCI data in **two tracks** (baseline kept for learning; advanced is additive):

| Track | Notebooks | Feature design |
|-------|-----------|----------------|
| **Baseline (keep)** | [01](notebooks/01_online_retail_ii_segmentation.ipynb) Online Retail II · [02](notebooks/02_wholesale_customers_segmentation.ipynb) Wholesale · [03](notebooks/03_online_retail_segmentation.ipynb) Online Retail | Classic RFM or 6 spends; v1→v2 production polish |
| **Advanced (new)** | [04](notebooks/04_advanced_rfm_plus_whales_representation.ipynb) RFM+ / whales · [05](notebooks/05_advanced_temporal_holdout_hybrid_clv.ipynb) future holdout / hybrid · [06](notebooks/06_advanced_soft_hierarchical_scoring_playbooks.ipynb) soft/hier/scoring | RFM+, whales, time holdout, CLV proxy, soft/hier, API |

Each notebook is **tutorial-style** (markdown teaches *why* before *how*) and **fully executed** (real numbers and plots in the saved `.ipynb`). Older results are **not deleted** — see comparisons below.

### Three generations of results (how to read the numbers)

| Generation | Where documented | What it is |
|------------|------------------|------------|
| **v1 baseline** | [v1 tables](#v1-baseline-results-preserved-in-full) | First full run: log + StandardScaler, PyCaret survey, sklearn best-2 |
| **v2 production** | [v2 tables](#v2-production-results-current--full-detail) + [old vs new](#old-vs-new--headline-comparison-same-data-different-pipeline) | Same notebooks 01–03 upgraded: winsorize, RobustScaler, GMM family, stability ARI, `production_score` |
| **v3 advanced** | [Advanced track](#advanced-track-notebooks-0406--kept-separate-from-0103) | **New** notebooks 04–06 only — RFM+, whales, future £/retention, hybrid, soft/hier, scorer |

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
├── docs/
│   └── ADVANCED_SEGMENTATION_TUTORIAL.md   # why/how advanced upgrades work
├── segmentation/           # production helper package
│   ├── preprocess.py / metrics.py / selection.py   # used by 01–03 (+ shared utils)
│   └── advanced/           # NEW — does not replace baseline modules
│       ├── features_plus.py, whales.py, temporal.py
│       ├── hybrid.py, soft_hierarchical.py, scoring.py, stability_time.py
├── artifacts/              # scoring sample + A/B design CSVs (from notebook 06)
├── data/                   # local UCI cache (large; gitignored)
└── notebooks/
    ├── 01–03_*             # BASELINE track (unchanged learning path)
    └── 04–06_advanced_*    # ADVANCED track (new code + executed outputs)
```

- **01–03 stay the teaching baseline** (results preserved in this README).  
- **04–06 are additive** advanced notebooks with full explanations.  
- **`.py`** = jupytext sources; **`.ipynb`** = executed deliverables.

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

### Ranking rubric (v2 `production_score`)

Configs are ranked after k-sweeps with hard filters roughly:

- finite Silhouette, **2 ≤ k ≤ 8** in the deep sweep  
- low noise; largest cluster not near 100%; min cluster not tiny  

Then **`production_score`** (see `segmentation/metrics.py`) combines:

- Silhouette (primary geometry)  
- Calinski–Harabasz (between/within dispersion)  
- Davies–Bouldin (penalty)  
- **Bootstrap stability ARI**  
- Penalties for imbalance / noise  
- Slight bonus for **3–5** segments (actionable CRM granularity)

v1 used a simpler Silhouette-first ranking without stability; both result sets are kept below.

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

All numbers are from **executed notebooks on real UCI data** (not blog copy). This section keeps the **v1 baseline results**, documents the **v2 production pipeline**, and shows **old vs new side-by-side**.

### Execution health (latest v2 run)

| Notebook | Code cells w/ outputs | Errors |
|----------|----------------------|--------|
| 01 Online Retail II | 9/9 | 0 |
| 02 Wholesale | 8/8 | 0 |
| 03 Online Retail | 7/7 | 0 |

---

### New techniques (v2 production upgrade) — what changed and why results improved

| Technique | What we did before (v1) | What we do now (v2) | Why it helps |
|-----------|-------------------------|---------------------|--------------|
| **Outlier control** | No winsorization; heavy spenders distorted distances | **Winsorize 1st–99th percentile** per feature before log | KMeans/GMM chase mega-buyers; clipping stabilizes geometry without dropping customers |
| **Transform** | log1p on F/M (RFM) / spends | Same, applied **after** winsorize | Order matters: log after clip reduces extreme leverage |
| **Scaling** | `StandardScaler` (mean/std) | **`RobustScaler`** (median/IQR) | Retail residuals stay fat-tailed; IQR scale is less sensitive |
| **Algorithm survey** | PyCaret menu (incl. density methods) | Explicit sklearn families: **KMeans++, MiniBatchKMeans, Ward, Birch, GMM** | Controlled APIs; GMM allows unequal variance; MiniBatch scales |
| **k selection** | Elbow + Silhouette only | Sweep **k=2..8** + Silhouette + DB + CH | Multi-metric avoids “pretty” k that is operationally useless |
| **Stability** | Not measured | **Bootstrap ARI** (70% subsample, 6 boots) | Prefers partitions that reappear on resamples |
| **Selection objective** | Mostly highest Silhouette among viable models | Composite **`production_score`** (geometry + balance + stability + slight 3–5 segment preference) | Production cares about actionability and consistency, not Silhouette alone |
| **Code structure** | Logic inlined in notebooks | Reusable **`segmentation/`** package | Same pipeline for all three datasets; easier to re-score new customers |
| **RFM interpretability** | Profiles only | + classic **1–5 R/F/M quantile scores** (diagnostic) | Bridges CRM language and ML clusters |

**How that produced better results (mechanism, not magic):**

1. **Cleaner feature space** — winsorize + robust scale shrinks the pull of extreme Monetary tails so cluster centers sit on the bulk of customers.  
2. **Stronger between/within separation (CH)** — with less outlier drag, Calinski–Harabasz rose on both retail notebooks (5793→6290; 4395→4917).  
3. **Slightly higher Silhouette on retail** — NB1 0.419→0.427; NB3 0.416→0.426.  
4. **Stability as a first-class metric** — preferred models now report ARI ≈ **0.97** (retail) so we know the cut is not a one-shot fluke.  
5. **Wholesale tradeoff is honest** — Silhouette dipped slightly (0.258→0.252) while **CH rose** and **Channel purity improved** (Horeca cluster 95.8%→**97.1%**). `production_score` can prefer that; we still publish Silhouette so reviewers can disagree.

---

### Old vs new — headline comparison (same data, different pipeline)

| Notebook | | Preferred model | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Stability ARI | Business highlight |
|----------|--|-----------------|--------------|------------------|---------------------|---------------|--------------------|
| **01 Retail II** | **v1 (old)** | KMeans k=2 | 0.419 | 0.889 | 5793 | — (not measured) | High-value 45.5% cust → **90.2%** revenue |
| | **v2 (new)** | KMeans k=2 | **0.427** | **0.870** | **6290** | **0.968** | Champions 46.0% cust → **91.3%** revenue |
| | **Δ** | same family/k | **+0.008** | **−0.019** (better) | **+497** | new metric | **+1.1 pp** revenue concentration |
| **02 Wholesale** | **v1 (old)** | HClust k=2 | **0.258** | 1.600 | 135 | — | Horeca-dominated cluster **95.8%** Channel=1 |
| | **v2 (new)** | MiniBatchKMeans k=2 | 0.252 | **1.553** | **149** | **0.756** | Horeca-dominated cluster **97.1%** Channel=1 |
| | **Δ** | model family changed | −0.006 | **−0.047** (better) | **+14** | new metric | **+1.3 pp** Channel purity; Sil slightly lower |
| **03 Retail 352** | **v1 (old)** | KMeans k=3 | 0.416 | 0.825 | 4395 | — | VIP 30.5% cust → **81.6%** revenue |
| | **v2 (new)** | KMeans k=3 | **0.426** | **0.801** | **4917** | **0.974** | VIP 32.2% cust → **82.2%** revenue |
| | **Δ** | same family/k | **+0.010** | **−0.024** (better) | **+522** | new metric | **+0.6 pp** VIP revenue share |

---

### v1 baseline results (preserved in full)

These are the **original** fully-executed notebook outputs before the production package (log + `StandardScaler`, PyCaret survey, sklearn re-fit of best 2, no winsorize/stability/`production_score`).

#### v1 — Notebook 1 Online Retail II

| Item | Value |
|------|--------|
| Rows / customers | 1,067,371 raw → 805,549 clean → **5,878** RFM |
| Preprocess | log1p(F,M) + StandardScaler |
| Survey best 2 | kmeans, hclust |
| Preferred | **KMeans k=2** — Sil **0.419**, DB 0.889, CH **5793** |
| Runner-up | Hierarchical k=2 — Sil 0.399, DB 0.846, CH 5066 |

| Segment | n | % cust | % revenue | Avg R | Avg F | Avg M |
|---------|---|--------|-----------|-------|-------|-------|
| High value / recent | 2,674 | 45.5% | **90.2%** | 62 d | 11.6 | £5,988 |
| Low value / stale | 3,204 | 54.5% | 9.8% | 316 d | 1.9 | £541 |

#### v1 — Notebook 2 Wholesale Customers

| Item | Value |
|------|--------|
| Rows | **440** |
| Preprocess | log1p(6 spends) + StandardScaler; Channel/Region held out |
| Survey best 2 | birch, hclust |
| Preferred | **HClust k=2** — Sil **0.258**, DB 1.600, CH **135** |
| Runner-up | Birch k=2 — Sil 0.222, DB 1.645, CH 124 |

| Segment | n | % accounts | % spend | Mean total | Top category |
|---------|---|------------|---------|------------|--------------|
| 0 | 178 | 40.5% | 49.0% | ~40,275 | Grocery ≈ 14,247 |
| 1 | 262 | 59.5% | 51.0% | ~28,437 | Fresh ≈ 15,016 |

Channel sanity (v1): Cluster 0 ≈ **73.6%** Channel=2 (Retail); Cluster 1 ≈ **95.8%** Channel=1 (Horeca).

#### v1 — Notebook 3 Online Retail (352)

| Item | Value |
|------|--------|
| Rows / customers | 541,909 raw → 397,884 clean → **4,338** RFM |
| Preprocess | log1p(F,M) + StandardScaler |
| Survey best 2 | kmeans, hclust |
| Preferred | **KMeans k=3** — Sil **0.416**, DB 0.825, CH **4395** |
| Runner-up | Hierarchical k=2 — Sil 0.400, DB 0.892, CH 3691 |

| Segment | n | % cust | % revenue | Avg R | Avg F | Avg M |
|---------|---|--------|-----------|-------|-------|-------|
| VIP | 1,323 | 30.5% | **81.6%** | 29 d | 9.8 | £5,494 |
| Mid | 2,036 | 46.9% | 14.1% | 54 d | 2.0 | £615 |
| Lapsed / low | 979 | 22.6% | 4.4% | 254 d | 1.4 | £398 |

---

### v2 production results (current — full detail)

Pipeline for all three: **winsorize → log1p → RobustScaler → family survey → k-sweep + bootstrap ARI → production_score → profiles**.

#### v2 — Notebook 1 Online Retail II (current preferred)

| Item | Value |
|------|--------|
| Same data grain | 1,067,371 → 805,549 → **5,878** customers |
| Best families | **kmeans**, **minibatch_kmeans** |
| Preferred | **KMeans k=2** |

| Config | Silhouette ↑ | DB ↓ | CH ↑ | Stability ARI | production_score |
|--------|--------------|------|------|---------------|------------------|
| **KMeans k=2** | **0.427** | **0.870** | **6290** | **0.968** | **0.662** |
| MiniBatchKMeans k=2 | 0.427 | 0.872 | 6288 | 0.946 | 0.659 |

| Segment | n | % cust | % revenue | R (d) | F | M mean (median) | Label |
|---------|---|--------|-----------|-------|---|-----------------|-------|
| Champions / Loyal high-value | 2,705 | 46.0% | **91.3%** | 70 | 11.5 | **£5,990** (£2,534) | VIP protect |
| At-risk / Lapsed | 3,173 | 54.0% | 8.7% | 311 | 1.8 | £486 (£374) | Win-back |

**vs v1 on this notebook:** better Sil/DB/CH; revenue concentration **90.2% → 91.3%**; stability now quantified (0.968).

#### v2 — Notebook 2 Wholesale Customers (current preferred)

| Item | Value |
|------|--------|
| Preferred | **MiniBatchKMeans k=2** — Sil **0.252**, DB **1.553**, CH **149**, stab **0.756**, prod **0.223** |
| Runner-up (family) | HClust k=4 — Sil 0.228, prod 0.222 |

| Segment | n | % accounts | % spend | Mean total | Top category |
|---------|---|------------|---------|------------|--------------|
| 1 | 201 | 45.7% | **57.4%** | ~41,732 | Grocery ≈ 13,709 |
| 0 | 239 | 54.3% | 42.6% | ~26,073 | Fresh ≈ 14,826 |

Channel sanity (v2, descriptive only): Cluster 0 → Channel **1** (Horeca) **97.1%**; Cluster 1 → Channel **2** (Retail) **67.2%**.

**vs v1 on this notebook:** Silhouette slightly lower; **CH and Channel purity higher**; model family switched (HClust → MiniBatchKMeans) because `production_score` + balance favored it.

#### v2 — Notebook 3 Online Retail 352 (current preferred)

| Item | Value |
|------|--------|
| Preferred | **KMeans k=3** — Sil **0.426**, DB **0.801**, CH **4917**, stab **0.974**, prod **0.656** |
| Runner-up | MiniBatchKMeans k=3 — Sil 0.426, stab 0.914, prod 0.649 |

| Segment | n | % cust | % revenue | R (d) | F | M mean (median) |
|---------|---|--------|-----------|-------|---|-----------------|
| VIP | 1,397 | 32.2% | **82.2%** | 29 | 9.5 | **£5,245** (£2,479) |
| Mid | 1,956 | 45.1% | 12.4% | 54 | 2.0 | £566 (£484) |
| Lapsed / low | 985 | 22.7% | 5.4% | 254 | 1.4 | £485 (£298) |

**vs v1 on this notebook:** Sil/DB/CH all improved; VIP revenue share **81.6% → 82.2%**; stability ARI **0.974**.

---

## Advanced track (notebooks 04–06) — kept separate from 01–03

Older notebooks **were not edited**. Advanced work lives in new modules + new notebooks so learners can still run the classic path first.

Full conceptual guide: **[docs/ADVANCED_SEGMENTATION_TUTORIAL.md](docs/ADVANCED_SEGMENTATION_TUTORIAL.md)**.

### What we added (and why)

| Upgrade | Where | Why it makes results “more awesome” |
|---------|--------|-------------------------------------|
| **RFM+ features** (tenure, AOV, inter-purchase, cancel rate, 90d trends) | 04 + `features_plus.py` | Same R/F/M can hide opposite trajectories; extra axes separate real behaviors |
| **Whale split** (top 1% Monetary) | 04 + `whales.py` | Key accounts get a policy segment; they stop owning every centroid |
| **Time holdout** (features ≤ cutoff; outcomes next 90d) | 05 + `temporal.py` | Silhouette ≠ revenue; future £ / retention is the commercial test |
| **Hybrid CLV / retention proxies** | 05 + `hybrid.py` | Rank *inside* clusters for targeting without fake segment accuracy |
| **Soft GMM membership** | 06 + `soft_hierarchical.py` | Confidence / partial VIP membership for ops |
| **Hierarchical VIP → core** | 06 | Budget like a real CRM org (protect top 15%, sub-segment the rest) |
| **Rolling as-of stability** | 06 + `stability_time.py` | Detect segment drift across months |
| **Scoring API + playbooks + A/B design** | 06 + `scoring.py` | Deployable loop: features → segment → action → metric |

### Real outputs from advanced notebooks (executed)

#### Notebook 04 — RFM+ vs baseline RFM on **core** (whales held out)

| Pipeline | k | Silhouette | CH | Notes |
|----------|---|------------|-----|--------|
| Baseline RFM (3 features) | 2 | 0.428 | 6288 | Same spirit as NB01 geometry |
| **RFM+ (15 features)** | 2 | **0.507** | 1713 | **+0.08 Silhouette** vs 3-feature baseline on core |
| Whales (rule) | — | — | — | **n=59 (1%)**, **31.9%** of lifetime Monetary, threshold £29,730 |

**Why Silhouette jumped:** richer, winsorized/scaled RFM+ geometry on the non-whale base forms tighter relative groups than 3D RFM alone.

#### Notebook 05 — future 90-day holdout (no leakage)

| Pipeline (history only) | Silhouette @ cutoff | Future £ max/min ratio | Top segment share of future £ | Retention gap |
|-------------------------|---------------------|------------------------|-------------------------------|---------------|
| Baseline RFM | 0.411 | **7.89** | **86.5%** | 0.378 |
| RFM+ | **0.449** | 3.90 | 34.2% | 0.363 |

**Honest reading:** RFM+ won **in-sample geometry** at the cutoff, while classic RFM’s coarse k=2 cut still concentrated **future** revenue more extremely in this window. That is why we measure **both** — awesome production systems optimize for **future lift**, not only Silhouette.

**Supervised proxies (holdout):** future-monetary **R² ≈ 0.39**, retention **AUC ≈ 0.82** — strong enough to build hybrid `cluster|value_band` cells. Hybrid **retention gap ≈ 0.65** (wider than pure clusters).

#### Notebook 06 — soft / hierarchy / stability / scoring

| Piece | Real output |
|-------|-------------|
| Soft GMM | 4 components; confidence histogram in notebook |
| Hierarchy | **VIP n=873 (~15%)**, rest **4,946** split into 3 cores |
| Rolling stability | Mean consecutive ARI ≈ **0.20** (segments **do** drift — monitoring is required) |
| Artifacts | `artifacts/sample_scores.csv`, `artifacts/ab_experiment_design.csv` |

### How to run advanced only

```bash
for n in 04_advanced_rfm_plus_whales_representation \
         05_advanced_temporal_holdout_hybrid_clv \
         06_advanced_soft_hierarchical_scoring_playbooks; do
  uv run jupytext --to ipynb "notebooks/${n}.py"
  uv run jupyter nbconvert --to notebook --execute "notebooks/${n}.ipynb" \
    --output "${n}.ipynb" --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=customer-segmentation-project
done
```

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
10. **Advanced RFM+ can raise Silhouette while classic RFM wins a particular future-window concentration** (see notebook 05). Always report both geometry and future metrics.
11. **Rolling ARI ~0.2** on this dataset shows labels are not permanent — production needs re-fit + name matching, not a one-time cluster id.

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
