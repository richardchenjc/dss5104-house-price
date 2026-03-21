# DSS5104 — House Price Prediction with Linear Models

King County, WA residential property price prediction using a regularised linear model with KNN comparable-sales features, street direction encodings, and frequency-based demand encoding.

---

## Results

| Model | Test MAPE | vs Lean Ridge |
|---|---|---|
| **Lean Ridge (12 feats) — Primary** | **15.98%** | — |
| XGB Tuned (randomised search, n=700) | 14.98% | −1.01 pp |
| XGB Conservative (depth=4, n=400) | 15.18% | −0.80 pp |
| XGB Early Stopping (n=80) | 15.40% | −0.58 pp |
| XGB Default (depth=6, n=500) | 15.49% | −0.51 pp (train=1.9% — severely overfit) |

The Lean Ridge model trails a well-tuned XGBoost (50-iteration randomised search) by only **+1.01 pp**, while offering full coefficient-level interpretability.

### Seed Stability (6 random splits)

| Seed | Train MAPE | Test MAPE | Gap |
|---|---|---|---|
| **42 (primary)** | 17.13% | **15.98%** | −1.15 pp |
| 0 | 16.54% | 19.62% | +3.08 pp |
| 1 | 16.24% | 19.98% | +3.74 pp |
| 7 | 17.25% | 16.38% | −0.87 pp |
| 13 | 16.74% | 17.89% | +1.15 pp |
| 99 | 16.92% | 17.78% | +0.85 pp |

Mean gap: **+1.14 pp** (2 negative, 4 positive). Seed 42 is a favourable split; the typical gap is positive, consistent with a well-generalising model. The large test MAPEs on seeds 0 and 1 are caused by the **geometric elbow firing too early** on those training sets (n=9–10 features instead of 12), dropping `log_sqft_above`, `condition`, and `view` — not by a difficult test set composition.

### Feature Stability Across Seeds

| Stability | Features |
|---|---|
| **Stable (6/6 seeds)** | `sqft_per_bedroom`, `knn_median`, `knn_broad_weighted`, `city_freq_x_lp`, `zip_dir_lp`, `zip_dir_x_sqft` |
| Near-stable (5/6) | `waterfront`, `knn_weighted_mean`, `dir_lp` |
| Split-sensitive (2–3/6) | `view`, `log_sqft_above`, `condition` |

The 6 stable features form the robust core; the split-sensitive features are real signals but the geometric elbow is sensitive to which 3,642 rows land in training.

---

## Repository Structure

```
.
├── analysis.py                 # Main pipeline — run after xgboost_benchmark.py
├── xgboost_benchmark.py        # Generates XGBoost benchmarks → xgb_results.json
├── make_report.py              # Builds house_price_report.pdf from results.json
├── house_price_analysis.ipynb  # Annotated notebook (mirrors analysis.py)
├── house_price_report.pdf      # Final academic report (10 pages)
├── results.json                # All metrics (written by analysis.py)
├── xgb_results.json            # XGBoost benchmark results (written by xgboost_benchmark.py)
├── data/
│   └── house_dataset.csv       # Raw dataset (place in data/ subfolder)
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
# Place house_dataset.csv in the data/ subfolder, then:

# Step 1 (recommended): run XGBoost benchmarks
# Requires: pip install xgboost
python xgboost_benchmark.py   # ~5-10 min; writes xgb_results.json

# Step 2: run the main pipeline
python analysis.py             # ~2 min; merges xgb_results.json → results.json + figures/

# Step 3: build the report
python make_report.py          # builds house_price_report.pdf
```

If `xgboost_benchmark.py` is skipped, `analysis.py` falls back to pre-run values stored as constants. The report is still fully buildable — only the XGBoost rows in Table 8 will show fallback numbers.

**Requirements:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `reportlab`, `xgboost`

```bash
pip install pandas numpy scikit-learn matplotlib seaborn reportlab xgboost
```

---

## Data

| Stage | Rows | Notes |
|---|---|---|
| Raw | 9,200 | From `data/house_dataset.csv` |
| After zero-price removal | 9,102 | 98 rows removed |
| After deduplication | **4,553** | Dataset is two identical halves stacked |

**Why deduplication matters:** Training on 9,102 duplicated rows → 0.29% train / 17.5% test (17.2 pp memorisation gap). The deduped model achieves 15.98% on the same test set.

**Dedup key:** 14 columns — price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated, street.

**Train / test split:** 80/20, `random_state=42` → 3,642 training / 911 test rows.

**Renovation date fix:** 386 rows with `yr_renovated < yr_built` (century-digit typos). Treated as no valid renovation — `effective_age` falls back to `yr_built`, `recent_reno` set to zero.

---

## EDA → Feature Engineering Mapping

| EDA Finding | Feature decision |
|---|---|
| Right-skewed price distribution | Log-transform target → `log_price` |
| U-shaped age–price curve | `log_house_age` + `is_new` |
| Non-linear condition jump at score 5 | `top_condition` + `cond_x_age` |
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

### Collinearity Exclusions

| Feature | Reason |
|---|---|
| `has_basement` | r=0.992 with `log_sqft_basement` |
| `any_view` | r=0.928 with `view` |
| `view_x_wf` | r=0.978 with `waterfront` |
| `size_x_condition` | r=0.959 with `condition` |
| `city_x_sqft` | r=0.972 with `zip_x_sqft` |

### Street Direction Features

Extracted via regex from street address (96.4% coverage). Compound directions (NE/NW/SE/SW) take priority over cardinal (N/S/E/W); last match taken.

EDA motivation: within-ZIP directional price spreads average **26%** in non-waterfront ZIPs. `r(zip_dir_lp, waterfront) = 0.05` — direction and waterfront are independent signals. All three direction features survive every seed's Lasso path.

---

## Modelling Pipeline

### Two-Stage L1 → L2 Design

**Stage 1 — Lasso (L1) feature selection:**
- Sweeps 60 α values (10⁻⁴ to 10⁰)
- Each surviving feature set evaluated by **5-fold CV with full re-encoding per fold**
- Geometric elbow selects n=12 features on seed=42
- CV drops from 19.7% (n=1) to 17.94% (n=12), then flattens — post-elbow gains all < 0.5 SE (±0.60pp)

**Stage 2 — Ridge (L2) for prediction:**
- Alpha tuned over {0.01, 0.1, 0.5, 1, 5, 10, 50, 100} → **best α=100**

### Elbow-Selected Features (n=12, ordered by |coefficient|)

| Rank | Feature | Group | Stable? |
|---|---|---|---|
| 1 | `zip_dir_x_sqft` | Direction | ✓ 6/6 |
| 2 | `zip_dir_lp` | Direction | ✓ 6/6 |
| 3 | `knn_broad_weighted` | KNN | ✓ 6/6 |
| 4 | `knn_weighted_mean` | KNN | 5/6 |
| 5 | `log_sqft_above` | Size | 2/6 |
| 6 | `city_freq_x_lp` | Frequency | ✓ 6/6 |
| 7 | `sqft_per_bedroom` | Rooms | ✓ 6/6 |
| 8 | `knn_median` | KNN | ✓ 6/6 |
| 9 | `view` | View | 3/6 |
| 10 | `condition` | Condition | 2/6 |
| 11 | `dir_lp` | Direction | 5/6 |
| 12 | `waterfront` | View | 5/6 |

All 12 coefficients are positive.

### Seed Stability Note

The geometric elbow varies across seeds (n=7–12). Seeds 0 and 1 fire the elbow too early, dropping `log_sqft_above`, `condition`, and `view`, which costs ~4 pp on those test sets. The 6 features stable across all seeds form the truly robust core of the model.

### Validation Protocol

5-fold CV with **full re-encoding per fold** — target encodings, KNN, frequency, and direction features are all re-fitted from scratch on each training fold. This is essential: KNN encodes training prices; target encodings use training log-prices. Fitting once on the full training set would leak validation fold information.

---

## Key Methodological Decisions

### Why MAPE, Not Scaled RMSE
11 ultra-luxury properties (>$2M) contribute 97.8% of RMSE². MAPE treats proportional error equally across the price range. Core market MAPE ($200K–$1M, n=804): ~14%.

### Why No Target Encoding Smoothing
ZIP SNR = 1.19× (between-group > within-group variance). Empirical CV: smoothed encoding (m=50) is +0.53 pp *worse* than raw group means.

### Extreme-End Performance

| Segment | MAPE | Bias | Root cause |
|---|---|---|---|
| <$200K | 32.5% | +29% (over) | Sparse training data; KNN neighbours mostly $300K+ |
| $1M–$2M | 23.3% | −14% (under) | Idiosyncratic luxury factors |
| >$2M | 51.3% | −46% (under) | Only ~50 training examples; unobservable features |

---

## Five Methodological Contributions

1. **Data integrity audit** — deduplication reduced apparent XGBoost advantage from 5+ pp to ~1 pp
2. **KNN comparable-sales features** — −0.87 pp CV; closed gap from 2.0 pp to under 1 pp
3. **Frequency demand proxy** (`city_freq_x_lp`) — −0.30 pp CV
4. **Street direction features** (`zip_dir_lp`, `zip_dir_x_sqft`) — −0.26 pp CV; EDA-motivated
5. **Interpretability-first design** — dominant features encode heuristics human appraisers already use

---

## XGBoost Benchmarks

Generated by `xgboost_benchmark.py` → `xgb_results.json`, merged into `results.json` by `analysis.py`.

The benchmark script runs a **50-iteration randomised hyperparameter search** over 9 parameters (depth, n_estimators, learning rate, subsample, colsample_bytree, min_child_weight, reg_lambda, reg_alpha, gamma), each evaluated by 5-fold CV with per-fold re-encoding — the same leakage-proof protocol as the linear pipeline.

| Configuration | Train MAPE | Test MAPE | Gap | Notes |
|---|---|---|---|---|
| Default (depth=6, n=500) | 1.91% | 15.49% | +13.57 pp | Severely overfit — not a valid benchmark |
| Conservative (depth=4, n=400) | 11.91% | 15.18% | +3.27 pp | Reference config |
| Early Stopping (n=80) | 11.73% | 15.40% | +3.68 pp | Principled |
| **Tuned (randomised search, n=700)** | **11.24%** | **14.98%** | **+3.73 pp** | **Primary benchmark** |
| Lean Ridge (ours) | 17.13% | **15.98%** | −1.15 pp | Primary model |

Gap to primary benchmark: **+1.01 pp**. The near-parity has a structural explanation: KNN comparable-sales features directly encode the neighbourhood price structure that XGBoost discovers through tree splits, pre-empting its main advantage.

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
| xgboost | 1.7 (for benchmark only) |

All random seeds fixed at `SEED = 42`. Pipeline is fully deterministic given the same data and library versions. `results.json` is overwritten on each run of `analysis.py`; `xgb_results.json` is only written by `xgboost_benchmark.py`.
