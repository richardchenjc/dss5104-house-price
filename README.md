# DSS5104 — House Price Prediction with Linear Models

King County, WA residential property price prediction using a regularised linear model with KNN comparable-sales features, street direction encodings, and frequency-based demand encoding.

---

## Results

| Model | Test MAPE | vs XGB Conservative |
|---|---|---|
| **Lean Ridge (12 feats) — Primary** | **15.98%** | +0.36 pp |
| XGB Conservative (benchmark) | 15.62% | — |
| XGB Early Stopping (benchmark) | 15.28% | — |

The Lean Ridge model trails a properly regularised XGBoost by only **0.36 pp** — within typical cross-validation variance — while offering full coefficient-level interpretability.

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
    ├── fig_eda_insights.png        — age curve, condition premium, city prices
    ├── fig_eda_size_view_reno.png  — size, view, waterfront, renovation EDA (2×3 grid)
    ├── fig_eda_direction.png       — ZIP × direction price heatmap
    ├── fig_mape_vs_nfeats.png      — Lasso regularisation path
    ├── fig_coef.png                — Ridge standardised coefficients
    ├── fig_diagnostics.png         — predicted vs actual, residuals
    ├── fig_segments.png            — MAPE by price segment
    └── fig_comparison.png          — Lean Ridge vs XGBoost benchmarks
```

---

## Quickstart

```bash
# Place house_dataset.csv in the repo root, then:
python analysis.py      # full pipeline — ~2 min; writes results.json + figures/
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
| After deduplication | **4,553** | Dataset is two identical halves stacked |

**Why deduplication matters (empirically verified):** Training on 9,102 duplicated rows yields 0.29% train / 17.5% test MAPE — a 17.2 pp gap from pure memorisation with no generalisation benefit. The deduped model achieves 15.98% on the same test set.

**Train / test split:** 80/20, `random_state=42` → 3,642 training / 911 test rows.

---

## EDA → Feature Engineering Mapping

Every feature group has explicit EDA motivation:

| EDA Finding | Feature decision |
|---|---|
| Right-skewed price distribution | Log-transform target → `log_price` |
| U-shaped age–price curve | `log_house_age` + `is_new` |
| Non-linear condition jump at score 5 | `top_condition` binary + `cond_x_age` |
| 6× city price range | Target encoding → `city_lp`, `zip_lp` |
| Concave size–price curve | `log_sqft_living` + `log_sqft_living_sq` |
| sqft/bedroom r=0.59 with log-price | `sqft_per_bedroom` space-quality ratio |
| Non-linear view premium ladder | Ordinal `view` feature |
| 159% waterfront median premium | Binary `waterfront` flag |
| Renovation recency paradox | `recent_reno` over raw `was_renovated` |
| ZIP × direction spread avg 26% | `zip_dir_lp`, `zip_dir_x_sqft` |

---

## Feature Engineering

**40 candidate features** across 11 groups, all derived before the train/test split.

### All Groups

| Group | Count | Features |
|---|---|---|
| Location | 4 | `city_lp`, `zip_lp`, `zip_city_diff`, `zip_x_sqft` |
| Size | 7 | Log-transformed sqft (living, above, basement, lot) + quadratic + ratios |
| Rooms | 4 | `bathrooms`, `floors`, `sqft_per_bedroom`, `bath_bed_ratio` |
| Condition | 3 | `condition`, `top_condition`, `cond_x_age` |
| Age | 5 | `era_lp`, `house_age`, `log_house_age`, `effective_age`, `is_new` |
| Renovation | 3 | `was_renovated`, `recent_reno`, `reno_lag` |
| View / Waterfront | 2 | `view`, `waterfront` |
| Time | 1 | `month_sold` |
| KNN comparable-sales | 4 | `knn_median`, `knn_weighted_mean`, `knn_broad_weighted`, `knn_local_vs_broad` |
| Frequency / demand | 4 | `log_zip_freq`, `log_city_freq`, `zip_freq_x_lp`, `city_freq_x_lp` |
| **Street direction** | **3** | **`dir_lp`, `zip_dir_lp`, `zip_dir_x_sqft`** |

### Street Direction Features

Extracted via regex from street address (96.4% coverage). Format varies: suffix (`'Densmore Ave N'`, `'170th Pl NE'`) or prefix after house number (`'NE 88th St'`). Compound directions (NE/NW/SE/SW) take priority over cardinal (N/S/E/W); last match taken to avoid street-name false hits.

EDA motivation: within-ZIP directional price spreads average **26%** in non-waterfront ZIPs — `r(zip_dir_lp, waterfront) = 0.05`, confirming direction and waterfront are independent signals. All three direction features survive Lasso path elbow; combined CV improvement −0.26 pp.

### Collinearity Exclusions

| Feature | Reason |
|---|---|
| `has_basement` | r=0.992 with `log_sqft_basement` |
| `any_view` | r=0.928 with `view` |
| `view_x_wf` | r=0.978 with `waterfront` |
| `size_x_condition` | r=0.959 with `condition` |
| `city_x_sqft` | r=0.972 with `zip_x_sqft` |

---

## Modelling Pipeline

### Two-Stage L1 → L2 Design

**Stage 1 — Lasso (L1) for feature selection:**
- Sweeps 60 α values (10⁻⁴ to 10⁰)
- Each surviving feature set evaluated by **5-fold CV with full re-encoding per fold** (all four encoding layers refit — prevents leakage)
- Geometric elbow selects **n=12** features
- CV drops from 19.7% (n=1) to 17.94% (n=12), then flattens — post-elbow gains all < 0.5 SE

**Stage 2 — Ridge (L2) for prediction:**
- The 12 features include correlated pairs; Ridge distributes weight stably across all
- Alpha tuned over {0.01, 0.1, 0.5, 1, 5, 10, 50, 100} → **best α=100**

### Elbow-Selected Features (n=12, ordered by |coefficient|)

| Rank | Feature | Group | Coef |
|---|---|---|---|
| 1 | `zip_dir_x_sqft` | Direction | +0.109 |
| 2 | `zip_dir_lp` | Direction | +0.108 |
| 3 | `knn_broad_weighted` | KNN | +0.088 |
| 4 | `knn_weighted_mean` | KNN | +0.060 |
| 5 | `log_sqft_above` | Size | +0.056 |
| 6 | `city_freq_x_lp` | Frequency | +0.053 |
| 7 | `sqft_per_bedroom` | Rooms | +0.050 |
| 8 | `knn_median` | KNN | +0.044 |
| 9 | `view` | View | +0.028 |
| 10 | `condition` | Condition | +0.026 |
| 11 | `dir_lp` | Direction | +0.024 |
| 12 | `waterfront` | View | +0.024 |

All 12 coefficients are positive. The two direction interaction features (`zip_dir_x_sqft`, `zip_dir_lp`) rank 1–2, followed by KNN comparable-sales features.

### KNN Design (10-dimensional property space)

Features: `log_sqft_living`, `log_sqft_lot`, `bathrooms`, `bedrooms`, `floors`, `condition`, `view`, `house_age`, `zip_lp`, `city_lp`

Including `zip_lp` and `city_lp` in the distance space ensures neighbours are similar in both **physical attributes and neighbourhood price level** — a 3-bed house in Medina ($1.5M median) is not comparable to a 3-bed house in Auburn ($280K median) even though they're physically similar.

Distance weights: `w_i = 1/d_i`, normalised to sum to 1 — closer comparables receive proportionally higher weight.

---

## Key Methodological Decisions

### Why MAPE, Not Scaled RMSE
Scaled RMSE = 164% on this dataset. 11 ultra-luxury properties (>$2M) contribute 97.8% of RMSE² — insensitive to model quality for 89% of the market. Core market MAPE ($200K–$1M, n=804): ~14%.

### Why No Target Encoding Smoothing
ZIP SNR = 1.19× (between-group > within-group variance). Empirical CV: smoothed encoding (m=50) is +0.53 pp *worse* than raw group means.

### Extreme-End Performance

| Segment | MAPE | Bias | Root cause |
|---|---|---|---|
| <$200K | 32.5% | +29% (over) | Sparse training data; KNN neighbours mostly $300K+ |
| $1M–$2M | 23.3% | −14% (under) | Idiosyncratic luxury factors |
| >$2M | 51.3% | −46% (under) | Only 36 training examples; unobservable features |

Regression-to-the-mean from data sparsity — not a modelling failure.

---

## Five Methodological Contributions

1. **Data integrity audit** — deduplication reduced apparent XGBoost advantage from 5+ pp to ~1 pp
2. **KNN comparable-sales features** — −0.87 pp CV; closed gap from 2.0 pp to under 1 pp
3. **Frequency demand proxy** (`city_freq_x_lp`) — −0.30 pp CV
4. **Street direction features** (`zip_dir_lp`, `zip_dir_x_sqft`) — −0.26 pp CV; EDA-motivated
5. **Interpretability-first design** — dominant features encode heuristics human appraisers already use

---

## XGBoost Benchmarks

Fixed values from a separate pre-run (not reproduced by `analysis.py`):

| Configuration | Train MAPE | Test MAPE | Notes |
|---|---|---|---|
| Default (depth=6) | 5.63% | 15.49% | 9.86 pp gap — severe memorisation |
| Tuned (depth=7) | 10.45% | 15.38% | |
| **Conservative (depth=4)** | **15.05%** | **15.62%** | **Primary benchmark** |
| Early Stopping | 11.21% | 15.28% | |

Gap to Lean Ridge: **+0.36 pp** vs Conservative XGBoost.

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

All random seeds fixed at `SEED = 42`. Pipeline is fully deterministic. `results.json` is overwritten on each run of `analysis.py`.
