# Advanced Customer Segmentation Tutorial

**Learning path:** keep notebooks **01–03** as the baseline story. Use **04–06** for production-minded upgrades.

This document explains *why* each upgrade exists and *how* it improves results — without deleting or rewriting the older notebooks.

---

## Map of the two tracks

| Track | Notebooks | Code | What you learn |
|-------|-----------|------|----------------|
| **Baseline** | `01`, `02`, `03` | `segmentation/preprocess.py`, `metrics.py`, `selection.py` | RFM/spend, survey, k-sweep, production_score, profiles |
| **Advanced** | `04`, `05`, `06` | `segmentation/advanced/*` | RFM+, whales, future eval, hybrid CLV, soft/hier, scoring API |

Baseline results (v1 and v2 in the main README) stay valid historical evidence. Advanced notebooks **add** a deeper evaluation story.

---

## Conceptual ladder (why each step)

### 1. Better features (notebook 04)

**Problem.** Classic RFM compresses a customer into three numbers. Two customers with similar R/F/M can differ in:

- trend (spend rising vs dying),
- rhythm (steady vs one-off),
- friction (cancels/returns),
- breadth (many products vs one SKU).

**What we do.** Build **RFM+**: tenure, AOV, inter-purchase stats, cancel rate, 90-day monetary trend, etc.

**Why results improve.** Clustering is only as good as the geometry of the feature space. More *behaviorally meaningful* axes often separate future value better than a fancier algorithm on 3D RFM.

### 2. Whale policy segment (notebook 04)

**Problem.** Top ~1% Monetary customers dominate Euclidean geometry and need different ops (key accounts).

**What we do.** Split them by rule (`Monetary ≥ q99`) **before** clustering the core.

**Why results improve.** Core centroids describe “normal” customers; whales get an explicit **Whale** playbook instead of distorting every cluster mean.

### 3. Temporal holdout evaluation (notebook 05)

**Problem.** Silhouette measures blob quality *today*, not whether segments predict *tomorrow*.

**What we do.** Cutoff date: features use history ≤ cutoff; metrics use next 90 days of spend/retention.

**Why results improve.** You can claim “segment A is 5× future value of segment B” with **no label leakage**. That is the commercial definition of better.

### 4. Hybrid CLV / churn proxies (notebook 05)

**Problem.** Unsupervised groups similar histories; marketers also need **ranking** inside a group.

**What we do.** Train simple models to predict next-period monetary and retention from history features; combine `cluster|value_band`.

**Why results improve.** Hybrid cells (e.g. Core + high predicted value) are more actionable than cluster id alone — without inventing fake “segment accuracy.”

### 5. Soft membership & hierarchy (notebook 06)

**Problem.** Hard labels are brittle; budgets are hierarchical (VIP program vs mass).

**What we do.** GMM responsibilities; level-1 VIP vs rest; level-2 subsegments on the rest.

**Why results improve.** Matches how organizations allocate budget and allows confidence thresholds (only treat high-probability VIP).

### 6. Rolling stability & scoring API (notebook 06)

**Problem.** One-shot notebook fits drift when new months arrive.

**What we do.** Re-fit at successive as-of dates; report ARI / size drift; ship a `SegmentScorer` with playbooks and a sample A/B design table.

**Why results improve.** Production is about **repeatable scoring + monitoring**, not a single PCA screenshot.

---

## How to run the advanced track

```bash
cd "/home/ahmad/AI/Customer Segmentation"
uv sync
uv run python -m ipykernel install --user --name customer-segmentation-project

# convert + execute (order matters pedagogically)
for n in \
  04_advanced_rfm_plus_whales_representation \
  05_advanced_temporal_holdout_hybrid_clv \
  06_advanced_soft_hierarchical_scoring_playbooks
do
  uv run jupytext --to ipynb "notebooks/${n}.py"
  uv run jupyter nbconvert --to notebook --execute "notebooks/${n}.ipynb" \
    --output "${n}.ipynb" \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=customer-segmentation-project
done
```

Data: same UCI Online Retail II cache under `data/` as notebook 01.

Artifacts written by notebook 06:

- `artifacts/sample_scores.csv` — full scored base  
- `artifacts/ab_experiment_design.csv` — playbook A/B design  

---

## What “better” means here

| Metric | Baseline 01–03 | Advanced 04–06 |
|--------|----------------|----------------|
| Silhouette / CH / DB | Yes | Yes (plus RFM+ comparison) |
| Future monetary / retention | No | **Yes (notebook 05)** |
| Stability over time | Bootstrap only (v2) | **Rolling as-of ARI (06)** |
| Deployable scoring | Manual | **`SegmentScorer`** |
| Ops playbooks | Manager paragraph | **Structured playbooks + A/B table** |

Expect: future-value **separation** and **actionability** to improve more dramatically than Silhouette. That is intentional.

---

## Integrity rules

- Do not edit notebooks 01–03 when extending advanced material.  
- Never use future invoices in features (temporal module enforces as-of).  
- Never report classification accuracy for unsupervised segment ids.  
- Whales and Channel/Region are policy/sanity tools, not magic accuracy labels.
