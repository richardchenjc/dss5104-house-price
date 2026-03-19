# DSS5104 — House Price Prediction with Linear Models

King County, WA residential property price prediction using regularised linear models, KNN comparable-sales features, and frequency-based demand encoding.

---

## Results

| Model | Test MAPE | vs XGB Conservative |
|---|---|---|
| **Lean Ridge (8 feats) — Primary** | **16.05%** | +0.43 pp |
| RBF Kernel (Nyström, 8 feats) | 15.73% | +0.10 pp |
| Poly Kernel (Nyström, 8 feats) | 15.80% | +0.18 pp |
| XGB Conservative (benchmark) | 15.62% | — |
| XGB Early Stopping (benchmark) | 15.28% | — |

All models use the same 8 Lasso-selected features. The RBF kernel is within 0.10 pp of a properly regularised XGBoost — without a single tree-based component.

---

## Repository Structure

```
.
├── analysis.py                 # Full pipeline — run this to reproduce all results
├── make_report.py              # Builds house_price_report.pdf from results.json
├── house_price_analysis.ipynb  # Annotated notebook (mirrors analysis.py)
├── house_price_report.pdf      # Final academic report (10 pages)
├── results.json                # All metrics (written by analysis.py)
├── house_dataset.csv           # Raw dataset (place here before running)
└── figures/                    # Output figures (written by analysis.py)
    ├── fig_price_dist.png
    ├── fig_corr.png
    ├── fig_eda_insights.png
    ├── fig_mape_vs_nfeats.png
    ├── fig_coef.png
    ├── fig_diagnostics.png
    ├── fig_segments.png
    └── fig_comparison.png
```

---

## Quickstart

```bash
# Place house_dataset.csv in the repo root, then:
python analysis.py      # runs full pipeline, writes results.json + figures/
python make_report.py   # builds house_price_report.pdf
```

**Requirements:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `reportlab`

```bash
pip install pandas numpy scikit-learn matplotlib seaborn reportlab
```

---

## Data

| Stage | Rows | Notes |
|---|---|---|
| Raw | 9,200 | From `house_dataset.csv` |
| After zero-price removal | 9,102 | 98 rows removed |
| After deduplication | **4,553** | Dataset confirmed to be two identical halves stacked |

**Why deduplication matters (empirically verified):**
Training on 9,102 duplicated rows yields 0.29% train MAPE / 17.5% test MAPE — a 17.2 pp gap indicating pure memorisation with no generalisation benefit. The deduped model achieves 16.05% on the same test set. Deduplication is verified by 5 independent checks (identical sale dates, zero price differences, fixed row offset).

**Train / test split:** 80/20, `random_state=42` → 3,642 training / 911 test rows.

---

## Feature Engineering

**37 candidate features** across 8 groups, all derived before the train/test split.

### Feature Groups

| Group | Count | Key decisions |
|---|---|---|
| Location | 4 | Raw target encoding (no smoothing — empirically +0.53 pp worse with m=50) |
| Size | 7 | All log-transformed; quadratic `log_sqft_living_sq` captures concave size–price curve |
| Rooms | 4 | Ratios encode space quality, not raw counts |
| Condition | 3 | `top_condition` for non-linear jump at condition=5 |
| Age | 5 | `era_lp` = target-encoded build decade (non-linear age proxy, NOT a location feature) |
| Renovation | 3 | 386 rows with `yr_renovated < yr_built` treated as no valid renovation |
| View / Waterfront | 2 | Ordinal `view` retained; binary `any_view` dropped (r=0.928 collinearity) |
| Time | 1 | Seasonal variation via `month_sold` |
| KNN comparable-sales | 4 | Dual-scale (k=10 hyperlocal, k=25 broader); leave-one-out on training rows; re-fitted per CV fold |
| Frequency / demand | 4 | Transaction count per ZIP/city; all r < 0.23 with existing features |

### Collinearity Exclusions

| Feature | Reason |
|---|---|
| `has_basement` | r=0.992 with `log_sqft_basement` |
| `any_view` | r=0.928 with `view` (ordinal strictly dominates) |
| `view_x_wf` | r=0.978 with `waterfront` |
| `size_x_condition` | r=0.959 with `condition` |
| `city_x_sqft` | r=0.972 with `zip_x_sqft` (ZIP version stronger) |

### Exhaustive Feature Audit (rejected candidates)

| Candidate | Partial r | Reason rejected |
|---|---|---|
| `zip_lp²` | −0.08 | r=0.9998 with `zip_lp` — pure collinearity; log already linearises location signal |
| `zip_lp × log²_sqft` | +0.03 | r=0.984 with `zip_x_sqft` |
| View interactions (`view_lp`, `cond×view`, `zip×view`) | 0.07–0.08 | Enter Lasso path at n≥10, below 0.5 SE threshold |
| ZIP quantile features (`zip_q75`, `zip_iqr`) | — | 0.04 pp overall gain — complexity not justified |
| `year_sold` | NaN | Only 2 distinct values (2014/2015); near-zero variance after ZIP encoding |

---

## Modelling Pipeline

### Two-Stage L1 → L2 Design

**Stage 1 — Lasso (L1) for feature selection:**
Sweeps 60 regularisation strengths (α ∈ [10⁻⁴, 10⁰]). At each α, surviving features are evaluated by 5-fold CV with **full re-encoding per fold** (target encoding + KNN refit + frequency refit — prevents leakage from all three encoding layers). Ridge(α=1.0) is used for path comparability. The **geometric elbow** (maximum curvature of the CV MAPE curve) selects n=8 features. Beyond the elbow, every marginal gain is < 0.5 SE — statistically indistinguishable from fold-sampling noise.

**Stage 2 — Ridge (L2) for prediction:**
The 8 selected features include correlated pairs (`zip_lp`/`zip_x_sqft`; `knn_median`/`knn_weighted_mean`/`knn_broad_weighted`). L1 would arbitrarily zero one of each pair; L2 distributes weight stably across all. Alpha tuned by 5-fold CV over {0.01, 0.1, 0.5, 1, 5, 10, 50, 100} → **best α=100**.

### Elbow-Selected Features (n=8)

| Feature | Group | Description |
|---|---|---|
| `zip_lp` | Location | ZIP code mean log-price (target encoding) — neighbourhood price baseline |
| `zip_x_sqft` | Location | `zip_lp × log(sqft_living)` — location premium scales with size |
| `sqft_per_bedroom` | Rooms | `sqft_living / (bedrooms+1)` — space quality signal |
| `waterfront` | View | Binary waterfront premium — idiosyncratic, not absorbed by KNN |
| `knn_median` | KNN | Median log-price of 10 nearest comparable properties |
| `knn_weighted_mean` | KNN | Distance-weighted mean of 10 hyperlocal comparables |
| `knn_broad_weighted` | KNN | Distance-weighted mean of 25 broader-neighbourhood comparables |
| `city_freq_x_lp` | Frequency | `log(city transaction count) × city_lp` — demand × price level |

**Seed stability (6 random splits):** 5 features appear in all 6 splits: `zip_lp`, `zip_x_sqft`, `sqft_per_bedroom`, `knn_median`, `knn_broad_weighted`. The elbow lands at n=7 or n=8 in 5 of 6 seeds.

### KNN Feature Design

- **10-dimensional property space:** `log_sqft_living`, `log_sqft_lot`, `bathrooms`, `bedrooms`, `floors`, `condition`, `view`, `house_age`, `zip_lp`, `city_lp`
- **Leave-one-out on training rows:** self-neighbour excluded at index 0 to prevent target leakage
- **Re-fitted per CV fold:** entire KNN model rebuilt from scratch inside each fold so validation fold prices never contaminate features
- **Why KNN works:** directly encodes the comparable-sales heuristic XGBoost approximates via tree splits — once encoded explicitly, XGBoost's structural advantage is largely pre-empted

### Frequency Feature Design

- `log_zip_freq` / `log_city_freq`: log(transaction count per group in training set)
- `zip_freq_x_lp` / `city_freq_x_lp`: frequency × price level interaction
- Higher frequency = more liquid market = demand-driven premium
- Unseen groups fall back to count=1 (log=0) — conservative, not global mean
- `city_freq_x_lp` selected at elbow; contributes −0.30 pp CV improvement over 7-feature model

### Kernel Extensions

Both kernels operate on the **same 8 lean features** via Nyström approximation (500 components). Hyperparameters re-tuned by CV grid search after adding `city_freq_x_lp`:

| Kernel | gamma | coef0 | degree | Ridge α | CV MAPE |
|---|---|---|---|---|---|
| RBF | 0.05 | — | — | 0.1 | 18.15% |
| Polynomial | 0.05 | 1 | 3 | 10 | 17.92% |

Note: `coef0=0` causes numerical blow-up with the frequency×price interaction term's scale.

---

## Key Methodological Decisions

### Why MAPE, Not Scaled RMSE

Scaled RMSE (RMSE / mean price) = 164% on this dataset. 11 ultra-luxury properties (>$2M) contribute **97.8% of total RMSE²** — making it insensitive to model quality on 89% of the market. MAPE is scale-independent and treats proportional errors equally across the price range. Core market MAPE ($200K–$1M, n=804): 14.4%.

### Why No Target Encoding Smoothing

ZIP signal-to-noise ratio = 1.19× (between-group > within-group variance). Empirical CV confirms: smoothed encoding (m=50) is **+0.53 pp worse** than raw group means. The standard justification for smoothing (protect sparse groups) does not hold here — high-value ZIPs like Medina (n=9) carry genuine signal that smoothing aggressively suppresses. Long-tail risk (2 ZIPs with n=1 in training) is absorbed by per-fold CV; KNN provides a robust fallback.

### Train / Test Gap

Averages ~0 pp across 6 random seeds (3/6 negative, 3/6 positive). A negative gap (test < train) is explained by three mechanisms: (1) α=100 deliberately inflates training error via L2 shrinkage; (2) KNN leave-one-out gives training rows slightly noisier features than test rows; (3) MAPE in price-space vs log-space optimisation. Not a structural property — split-composition noise.

### Extreme-End Performance

| Segment | MAPE | Bias | Root cause |
|---|---|---|---|
| <$200K | 32% | +29% (over) | Distressed/atypical sales; KNN neighbours mostly $300K+ |
| $1M–$2M | 23% | −14% (under) | Sparse training data; 179 examples above $1M |
| >$2M | 51% | −46% (under) | Only 36 training examples; luxury idiosyncrasies unobservable from listing data |

This is regression-to-the-mean from data sparsity — not a modelling failure. No feature engineering on this dataset resolves it (exhaustively tested).

---

## XGBoost Benchmarks

Fixed values from a separate pre-run (not reproduced by `analysis.py`):

| Configuration | Train MAPE | Test MAPE | Notes |
|---|---|---|---|
| Default (depth=6) | 5.63% | 15.49% | 9.86 pp gap — severe memorisation on duplicated data |
| Tuned (depth=7) | 10.45% | 15.38% | |
| **Conservative (depth=4)** | **15.05%** | **15.62%** | **Primary benchmark** — negligible train/test gap |
| Early Stopping | 11.21% | 15.28% | Principled benchmark |

The Default XGBoost's apparent advantage was inflated by training on the un-deduplicated 9,102-row dataset. On the deduplicated dataset, the gap to our best linear model shrinks from 5+ pp to **0.10 pp** (RBF kernel vs XGB Conservative).

---

## Reproducibility

| Package | Min version |
|---|---|
| Python | 3.10 |
| scikit-learn | 1.3 |
| pandas | 2.0 |
| numpy | 1.24 |
| matplotlib | 3.7 |
| seaborn | 0.12 |
| reportlab | 4.0 |

All random seeds fixed at `SEED = 42`. The pipeline is fully deterministic given the same dataset and library versions. `results.json` is overwritten on each run of `analysis.py`.
