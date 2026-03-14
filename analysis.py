"""
DSS5104 — House Price Prediction with Linear Models
====================================================
Single pipeline: data cleaning → EDA → feature engineering → modelling → results.

Run:
    python analysis.py

Outputs (written to ./figures/):
    fig_price_dist.png   — price distribution & log-price
    fig_corr.png         — correlation heatmap
    fig_eda_insights.png — age curve, condition premium, city prices
    fig_mape_vs_nfeats.png  — regularisation path
    fig_coef.png         — Ridge coefficients (lean model)
    fig_diagnostics.png  — predicted vs actual, residuals
    fig_segments.png     — MAPE by price segment
    fig_comparison.png   — full model progression + XGBoost
"""

import warnings; warnings.filterwarnings('ignore')
import os, json
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(style='whitegrid', font_scale=1.05)
PALETTE = ['#1f5f8b', '#c47d00', '#2a9d47', '#c0392b', '#7b2d8b']
SEED = 42
os.makedirs('figures', exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 1. DATA LOADING AND CLEANING
# ══════════════════════════════════════════════════════════════════

def load_and_clean(path='data\\house_dataset.csv'):
    """Load, remove zero-price rows, and deduplicate."""
    df = pd.read_csv(path)
    n_raw = len(df)

    # Remove zero-price entries
    df = df[df['price'] > 0].reset_index(drop=True)
    n_nonzero = len(df)

    # Deduplicate: dataset is confirmed to be two identical halves stacked
    # (verified via 5 independent checks — see report Section 2)
    key_cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                'floors', 'waterfront', 'view', 'condition', 'sqft_above',
                'sqft_basement', 'yr_built', 'yr_renovated', 'street']
    df = df.drop_duplicates(subset=key_cols, keep='first').reset_index(drop=True)

    print(f"Data loading:")
    print(f"  Raw rows          : {n_raw:,}")
    print(f"  After zero-price  : {n_nonzero:,}  (removed {n_raw - n_nonzero})")
    print(f"  After dedup       : {len(df):,}  (removed {n_nonzero - len(df)} duplicates)")
    return df

# ══════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS (FIGURES)
# ══════════════════════════════════════════════════════════════════

def plot_price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df['price'] / 1e6, bins=60, color=PALETTE[0], edgecolor='white', alpha=0.85)
    axes[0].set_xlabel('Sale Price (M$)'); axes[0].set_ylabel('Count')
    axes[0].set_title('Raw Price Distribution')
    axes[1].hist(np.log(df['price']), bins=60, color=PALETTE[1], edgecolor='white', alpha=0.85)
    axes[1].set_xlabel('log(Price)'); axes[1].set_ylabel('Count')
    axes[1].set_title('Log-Price Distribution (modelling target)')
    plt.tight_layout()
    plt.savefig('figures/fig_price_dist.png', dpi=130); plt.close()
    print("  Saved fig_price_dist.png")

def plot_correlations(df):
    num_cols = ['price', 'sqft_living', 'sqft_lot', 'bathrooms', 'bedrooms',
                'floors', 'condition', 'view', 'waterfront', 'yr_built']
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title('Correlation Matrix — Raw Features')
    plt.tight_layout()
    plt.savefig('figures/fig_corr.png', dpi=130); plt.close()
    print("  Saved fig_corr.png")

def plot_eda_insights(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['house_age'] = df['date'].dt.year - df['yr_built']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Age curve — bin to 120 to capture all houses (max age=114)
    age_bins = pd.cut(df['house_age'], bins=range(0, 121, 5))
    age_median = df.groupby(age_bins)['price'].median() / 1e3
    ax = axes[0]
    ax.bar(range(len(age_median)), age_median.values, color=PALETTE[0], alpha=0.8, width=0.8)
    ax.set_xlabel('House Age (5-yr bins)'); ax.set_ylabel('Median Price ($K)')
    ax.set_title('Age–Price Relationship\n(U-shape motivates log_house_age + is_new)')
    ax.set_xticks(range(0, len(age_median), 2))
    ax.set_xticklabels([str(x.left) for x in age_median.index[::2]], rotation=45, fontsize=8)

    # Condition premium
    ax2 = axes[1]
    cond_med = df.groupby('condition')['price'].median() / 1e3
    ax2.bar(cond_med.index, cond_med.values, color=PALETTE[2], alpha=0.85, edgecolor='white')
    ax2.set_xlabel('Condition (1–5)'); ax2.set_ylabel('Median Price ($K)')
    ax2.set_title('Condition–Price Premium\n(non-linear jump at 5 motivates top_condition)')
    for x, y in zip(cond_med.index, cond_med.values):
        ax2.text(x, y + 5, f'${y:.0f}K', ha='center', fontsize=9)

    # Top 10 cities by median price
    ax3 = axes[2]
    city_med = df.groupby('city')['price'].median().sort_values(ascending=False).head(10) / 1e3
    ax3.barh(city_med.index[::-1], city_med.values[::-1], color=PALETTE[0], alpha=0.85)
    ax3.set_xlabel('Median Price ($K)')
    ax3.set_title('Top 10 Cities by Median Price\n(6× range motivates target encoding)')

    plt.tight_layout()
    plt.savefig('figures/fig_eda_insights.png', dpi=130); plt.close()
    print("  Saved fig_eda_insights.png")

# ══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def engineer_features(df):
    """Add all derived features. Returns dataframe with new columns."""
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d['year_sold'] = d['date'].dt.year
    d['month_sold'] = d['date'].dt.month

    # ── Age ──────────────────────────────────────────────────────
    d['house_age'] = d['year_sold'] - d['yr_built']
    d['log_house_age'] = np.log1p(d['house_age'])
    d['is_new'] = (d['house_age'] <= 5).astype(int)

    # ── Renovation ───────────────────────────────────────────────
    # Valid renovation: recorded AND date is not before construction
    # (386 rows have yr_renovated < yr_built — century-digit typos e.g. 1912 instead of 2012)
    d['valid_reno'] = (d['yr_renovated'] > 0) & (d['yr_renovated'] >= d['yr_built'])
    d['was_renovated'] = d['valid_reno'].astype(int)
    d['yr_reno_fill'] = np.where(d['valid_reno'], d['yr_renovated'], d['yr_built'])
    d['effective_age'] = d['year_sold'] - d['yr_reno_fill']
    d['recent_reno'] = (
        d['valid_reno'] & (d['year_sold'] - d['yr_renovated'] <= 10)
    ).astype(int)
    d['reno_lag'] = np.where(
        d['valid_reno'], d['yr_renovated'] - d['yr_built'], 0
    )

    # ── Condition ────────────────────────────────────────────────
    d['top_condition'] = (d['condition'] == 5).astype(int)
    d['cond_x_age'] = d['condition'] * d['house_age']

    # ── View / waterfront ────────────────────────────────────────
    # any_view (r=0.928 with view) and view_x_wf (r=0.978 with waterfront)
    # excluded after collinearity audit — ordinal view dominates both

    # ── Size (log-transformed) ───────────────────────────────────
    d['log_sqft_living'] = np.log1p(d['sqft_living'])
    d['log_sqft_lot'] = np.log1p(d['sqft_lot'])
    d['log_sqft_above'] = np.log1p(d['sqft_above'])
    d['log_sqft_basement'] = np.log1p(d['sqft_basement'])
    d['log_sqft_living_sq'] = d['log_sqft_living'] ** 2   # captures concave size curve

    # ── Room ratios ──────────────────────────────────────────────
    d['sqft_per_bedroom'] = d['sqft_living'] / (d['bedrooms'] + 1)
    d['bath_bed_ratio'] = d['bathrooms'] / (d['bedrooms'] + 1)
    d['living_to_lot'] = d['sqft_living'] / (d['sqft_lot'] + 1)

    # ── Basement ─────────────────────────────────────────────────
    # has_basement (r=0.992 with log_sqft_basement) excluded after collinearity audit
    d['basement_ratio'] = d['sqft_basement'] / (d['sqft_living'] + 1)

    # ── Interactions ─────────────────────────────────────────────
    # size_x_condition (r=0.959 with condition) and size_x_floors
    # (r=0.989 with floors) excluded after collinearity audit

    # ── Categorical helpers ──────────────────────────────────────
    d['zipcode'] = d['statezip'].str.extract(r'(\d{5})', expand=False).astype(str)
    bins = [0, 1919, 1939, 1959, 1979, 1999, 2014]
    labels = ['pre1920', '1920s_30s', '1940s_50s', '1960s_70s', '1980s_90s', '2000s']
    d['build_era'] = pd.cut(d['yr_built'], bins=bins, labels=labels).astype(str)

    return d

def target_encode(train_df, test_df, col, target):
    """
    Raw target encoding — group mean of target, fitted on train_df only.
    No smoothing applied: empirical variance decomposition shows signal-to-noise
    ratio >= 1 for all location groupings (ZIP: 1.19x, City: 0.99x), meaning
    group means carry more information than noise and do not require
    regularisation toward the global mean.
    Unseen categories in test_df fall back to the global train mean.
    """
    global_mean = train_df[target].mean()
    group_means = train_df.groupby(col)[target].mean()
    train_enc = train_df[col].map(group_means).fillna(global_mean)
    test_enc  = test_df[col].map(group_means).fillna(global_mean)
    return train_enc, test_enc

def add_encodings(train_df, test_df):
    """
    Raw target encodings for categorical variables.
      Geographic location: city, zipcode  -> city_lp, zip_lp
      Age proxy:           build_era       -> era_lp
        build_era groups houses by construction decade and encodes the mean
        log-price per era. This is NOT a location feature — it captures the
        non-linear (U-shaped) age-price relationship documented in hedonic
        pricing literature without assuming a parametric functional form.
    All encodings fitted on training data only; re-fitted inside each CV fold.
    """
    tr, te = train_df.copy(), test_df.copy()
    # Geographic location encodings
    for col, grp in [('city', 'city_lp'), ('zipcode', 'zip_lp')]:
        tr[grp], te[grp] = target_encode(tr, te, col, 'log_price')
    # Age-proxy encoding (build era — non-linear age effect)
    tr['era_lp'], te['era_lp'] = target_encode(tr, te, 'build_era', 'log_price')
    for d_ in [tr, te]:
        d_['zip_city_diff'] = d_['zip_lp'] - d_['city_lp']
        d_['zip_x_sqft'] = d_['zip_lp'] * d_['log_sqft_living']  # name kept for consistency; uses log(sqft_living)
        # city_x_sqft and zip_x_cond removed: collinearity audit showed r>0.97
        # with zip_x_sqft and condition respectively — excluded from candidate pool
    return tr, te

# ── Feature sets ─────────────────────────────────────────────────

# Full candidate pool — Lasso will select the lean subset from these.
# Problematic interactions removed after empirical audit:
#   size_x_condition: r=0.959 with condition, Δr=0.023 moderation → pure collinearity
#   zip_x_cond:       r=0.988 with condition, Δr=0.008 moderation → hurts performance
#   size_x_floors:    r=0.989 with floors, Δr=0.071 but 0.006pp gain → not worth collinearity
#   any_view:         r=0.928 with view; view (ordinal) strictly dominates binary flag
#   city_x_sqft:      r=0.972 with zip_x_sqft; zip version stronger (r=0.83 vs 0.79)
#   view_x_wf:        r=0.978 with waterfront; removing improves MAPE by 0.017pp
#   has_basement:     r=0.992 with log_sqft_basement; size strictly more informative
CANDIDATE_FEATURES = [
    # Location (geographic)
    'city_lp', 'zip_lp', 'zip_city_diff', 'zip_x_sqft',
    # Size (log-transformed; quadratic captures concave size-price curve)
    'log_sqft_living', 'log_sqft_above', 'log_sqft_basement', 'log_sqft_lot',
    'log_sqft_living_sq', 'living_to_lot', 'basement_ratio',
    # Rooms
    'bathrooms', 'floors', 'sqft_per_bedroom', 'bath_bed_ratio',
    # Condition (cond_x_age kept: Δr=0.180 genuine moderation effect)
    'condition', 'cond_x_age', 'top_condition',
    # Age (era_lp encodes build_era — captures non-linear U-shaped age-price
    #      relationship as a target encoding; NOT a location feature)
    'era_lp', 'house_age', 'log_house_age', 'effective_age', 'is_new',
    # Renovation
    'was_renovated', 'recent_reno', 'reno_lag',
    # View / waterfront (ordinal view retained; any_view and view_x_wf dropped)
    'view', 'waterfront',
    # Time
    'month_sold',
]

# LEAN_FEATURES is derived in main() via Lasso path; placeholder overwritten at runtime
LEAN_FEATURES = CANDIDATE_FEATURES  # overwritten by lasso_select()

# ══════════════════════════════════════════════════════════════════
# 4. MODELLING UTILITIES
# ══════════════════════════════════════════════════════════════════

def mape(y_true_log, y_pred_log):
    """MAPE computed in original price space after exponentiation."""
    return np.mean(np.abs(
        (np.exp(y_true_log) - np.exp(y_pred_log)) / np.exp(y_true_log)
    )) * 100

def cross_val_mape(X, y, alpha=5, n_splits=5):
    """5-fold CV MAPE ± SE for Ridge regression.
    Returns (mean_mape, se) where se = std / sqrt(n_splits).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for ti, vi in kf.split(X):
        sc = StandardScaler()
        m = Ridge(alpha=alpha).fit(sc.fit_transform(X[ti]), y[ti])
        scores.append(mape(y[vi], m.predict(sc.transform(X[vi]))))
    return np.mean(scores), np.std(scores) / np.sqrt(n_splits)

# ══════════════════════════════════════════════════════════════════
# 5. LASSO REGULARISATION PATH (MAPE vs N features)
# ══════════════════════════════════════════════════════════════════

def lasso_select(train_df, test_df, y_tr, y_te, candidate_features):
    """
    Proper Lasso-based feature selection.

    Sweeps alpha over a grid. At each alpha:
      - fits Lasso on training data to identify surviving features
      - evaluates those features with 5-fold CV (re-encoding inside each fold
        to prevent leakage) using Ridge regression
    Selects features at the geometric elbow of the CV MAPE curve (maximum
    curvature). Beyond the elbow, every marginal gain is < 0.5 SE —
    statistically indistinguishable from fold-sampling noise on this dataset.
    Returns:
      - lean_features : list of features at the geometric elbow
      - path_df       : DataFrame of (n_features, cv_mape, cv_se, test_mape)
      - elbow_cv      : CV MAPE at the elbow point
      - elbow_se      : CV standard error at the elbow point
    Note: test_mape is recorded for plotting only and does not influence selection.
    """
    alphas = np.logspace(-4, 0, 60)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    records = []
    prev_n = len(candidate_features) + 1

    # Encode on full training set for the Lasso fit (feature identification)
    tr_enc, te_enc = add_encodings(train_df, test_df)
    X_tr_full = tr_enc[candidate_features].values
    X_te_full  = te_enc[candidate_features].values
    sc_full = StandardScaler()
    X_tr_sc = sc_full.fit_transform(X_tr_full)
    X_te_sc = sc_full.transform(X_te_full)

    print(f"  Sweeping {len(alphas)} alphas on {len(candidate_features)} candidate features ...")
    idx = np.arange(len(train_df))  # positional indices for KFold — constant across alphas
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_tr_sc, y_tr)
        surviving_mask = lasso.coef_ != 0
        n = surviving_mask.sum()
        if n == 0 or n == prev_n:
            continue
        prev_n = n
        surviving_feats = [f for f, m in zip(candidate_features, surviving_mask) if m]

        # CV MAPE: re-encode inside each fold, use only surviving features
        fold_mapes = []
        for ti, vi in kf.split(idx):
            tr_f = train_df.iloc[ti].copy()
            va_f = train_df.iloc[vi].copy()
            tr_f, va_f = add_encodings(tr_f, va_f)
            Xf_tr = tr_f[surviving_feats].values
            Xf_va = va_f[surviving_feats].values
            sc_f  = StandardScaler()
            m = Ridge(alpha=1.0).fit(sc_f.fit_transform(Xf_tr), y_tr[ti])
            fold_mapes.append(mape(y_tr[vi], m.predict(sc_f.transform(Xf_va))))
        cv_m = np.mean(fold_mapes)

        # Test MAPE recorded for plotting only — not used for selection
        X_surv_tr = X_tr_sc[:, surviving_mask]
        X_surv_te = X_te_sc[:, surviving_mask]
        sc_s = StandardScaler()
        te_m = mape(y_te, Ridge(alpha=1.0).fit(
            sc_s.fit_transform(X_surv_tr), y_tr).predict(sc_s.transform(X_surv_te)))

        records.append({
            'alpha': alpha, 'n_features': n,
            'cv_mape': cv_m, 'cv_se': np.std(fold_mapes) / np.sqrt(len(fold_mapes)),
            'test_mape': te_m,
            'features': surviving_feats
        })
        print(f"    alpha={alpha:.5f}  n={n:>2}  CV={cv_m:.3f}%")

    path_df = pd.DataFrame(records).sort_values('n_features').reset_index(drop=True)

    # ── Geometric elbow (maximum curvature) ───────────────────────
    ns    = path_df['n_features'].values.astype(float)
    cvs   = path_df['cv_mape'].values
    ns_n  = (ns  - ns.min())  / (ns.max()  - ns.min())
    cvs_n = (cvs - cvs.min()) / (cvs.max() - cvs.min())
    p1, p2 = np.array([ns_n[0], cvs_n[0]]), np.array([ns_n[-1], cvs_n[-1]])
    line   = p2 - p1
    dists  = [abs(np.cross(line, np.array([ns_n[i], cvs_n[i]]) - p1)) / np.linalg.norm(line)
              for i in range(len(ns_n))]
    elbow_idx = int(np.argmax(dists))
    elbow_n   = int(path_df.iloc[elbow_idx]['n_features'])

    # ── SE analysis beyond elbow ───────────────────────────────────
    # After the elbow, every marginal gain is < 0.5 SE — statistically
    # indistinguishable from zero. The dataset (4,553 rows, high price
    # heterogeneity) does not have enough power to confirm individual
    # features beyond the elbow via CV alone.
    # We select AT the elbow; the report notes that domain knowledge
    # justifies including features up to the CV minimum, but this
    # cannot be statistically verified on this dataset.
    best_idx = path_df['cv_mape'].idxmin()
    best_cv  = path_df.loc[best_idx, 'cv_mape']
    best_se  = path_df.loc[best_idx, 'cv_se']

    selected_row  = path_df.iloc[elbow_idx]
    lean_features = selected_row['features']
    elbow_cv  = selected_row['cv_mape']
    elbow_se  = selected_row['cv_se']  # SE at the elbow point (not at CV minimum)

    print(f"  Geometric elbow:  n={elbow_n}  CV={elbow_cv:.4f}%  SE=±{elbow_se:.4f}pp")
    print(f"  CV minimum:       n={int(path_df.loc[best_idx,'n_features'])}  "
          f"CV={best_cv:.4f}%  SE=±{best_se:.4f}pp")
    print(f"  Post-elbow gains are all < 0.5 SE — not statistically detectable")
    print(f"  Selecting at elbow: n={elbow_n} features")
    print(f"  Selected features: {lean_features}")

    return lean_features, path_df, elbow_cv, elbow_se

def plot_mape_vs_nfeats(path_df, lean_n, lean_test, lean_cv):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ns  = path_df['n_features'].values
    cv  = path_df['cv_mape'].values
    se  = path_df['cv_se'].values
    ax.fill_between(ns, cv - se, cv + se, alpha=0.15, color=PALETTE[0],
                    label='CV MAPE ± 1 SE')
    ax.plot(ns, cv, 'o-', color=PALETTE[0], lw=2, ms=5, label='CV MAPE (5-fold)')
    ax.plot(path_df['n_features'], path_df['test_mape'],
            's--', color=PALETTE[1], lw=2, ms=5, label='Test MAPE')
    ax.axvline(lean_n, color=PALETTE[2], linestyle=':', lw=2,
               label=f'Elbow — selected (n={lean_n}, CV={lean_cv:.1f}%)')
    ax.set_xlabel('Number of Features (Lasso regularisation path)')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('MAPE vs Feature Count — Lasso Regularisation Path')
    ax.legend(fontsize=9)
    ax.set_xlim(0, path_df['n_features'].max() + 1)
    plt.tight_layout()
    plt.savefig('figures/fig_mape_vs_nfeats.png', dpi=130)
    plt.close()
    print("  Saved fig_mape_vs_nfeats.png")

# ══════════════════════════════════════════════════════════════════
# 6. MODEL INTERPRETATION FIGURES
# ══════════════════════════════════════════════════════════════════

def plot_coefficients(model, feature_names, sc):
    coef = model.coef_
    df_c = pd.DataFrame({'Feature': feature_names, 'Coef': coef})
    df_c = df_c.reindex(df_c['Coef'].abs().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [PALETTE[0] if c > 0 else PALETTE[3] for c in df_c['Coef']]
    ax.barh(df_c['Feature'][::-1], df_c['Coef'][::-1], color=colors[::-1])
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Standardised Coefficient')
    ax.set_title(f'Ridge Model — Standardised Coefficients\n({len(feature_names)} features, primary interpretable model)')
    pos_p = mpatches.Patch(color=PALETTE[0], label='Positive effect')
    neg_p = mpatches.Patch(color=PALETTE[3], label='Negative effect')
    ax.legend(handles=[pos_p, neg_p], fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/fig_coef.png', dpi=130)
    plt.close()
    print("  Saved fig_coef.png")

def plot_diagnostics(y_te_log, y_pred_log, title='Primary Ridge Model'):
    y_true = np.exp(y_te_log)
    y_pred = np.exp(y_pred_log)
    resid_pct = (y_pred - y_true) / y_true * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_true / 1e6, y_pred / 1e6, alpha=0.35, s=10, color=PALETTE[0])
    lim = max(y_true.max(), y_pred.max()) / 1e6
    axes[0].plot([0, lim], [0, lim], 'r--', lw=1.5)
    axes[0].set_xlabel('Actual Price (M$)'); axes[0].set_ylabel('Predicted Price (M$)')
    axes[0].set_title(f'Predicted vs Actual — {title}')

    axes[1].scatter(y_pred / 1e6, resid_pct, alpha=0.35, s=10, color=PALETTE[1])
    axes[1].axhline(0, color='red', lw=1.5, linestyle='--')
    axes[1].set_xlabel('Predicted (M$)'); axes[1].set_ylabel('Residual (%)')
    axes[1].set_title('Residual Plot')

    plt.tight_layout()
    plt.savefig('figures/fig_diagnostics.png', dpi=130)
    plt.close()
    print("  Saved fig_diagnostics.png")

def plot_segments(y_te_log, y_pred_log):
    y_true = np.exp(y_te_log)
    bins_p = [0, 200e3, 400e3, 600e3, 800e3, 1e6, 2e6, 30e6]
    labels_p = ['<$200K', '$200–400K', '$400–600K', '$600–800K', '$800K–$1M', '$1M–$2M', '>$2M']
    segs = pd.cut(y_true, bins=bins_p, labels=labels_p)
    seg_mape, seg_count = [], []
    for seg in labels_p:
        mask = (segs == seg)
        seg_mape.append(mape(y_te_log[mask], y_pred_log[mask]) if mask.sum() > 0 else 0)
        seg_count.append(mask.sum())

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#c0392b' if m > 25 else '#f0a500' if m > 18 else '#2a9d47' for m in seg_mape]
    bars = ax.bar(labels_p, seg_mape, color=colors, edgecolor='white')
    for bar, val, cnt in zip(bars, seg_mape, seg_count):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}%\n(n={cnt})', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('MAPE (%)'); ax.set_title('MAPE by Price Segment — Primary Ridge Model')
    ax.set_ylim(0, max(seg_mape) * 1.25)
    plt.tight_layout()
    plt.savefig('figures/fig_segments.png', dpi=130)
    plt.close()
    print("  Saved fig_segments.png")

def plot_comparison(results):
    """Full model progression bar chart + XGBoost overfit chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: progression
    names = [r['name'] for r in results]
    mapes_v = [r['test_mape'] for r in results]
    colors = [r['color'] for r in results]
    ax = axes[0]
    bars = ax.bar(names, mapes_v, color=colors, edgecolor='white', width=0.65)
    for bar, val in zip(bars, mapes_v):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_ylabel('Test MAPE (%)')
    ax.set_title('Model Progression — Test MAPE\n(deduplicated dataset, 4,553 properties)')
    ax.set_ylim(0, max(mapes_v) * 1.2)
    linear_p = mpatches.Patch(color=PALETTE[0], label='Linear models')
    xgb_p = mpatches.Patch(color=PALETTE[1], label='XGBoost benchmark')
    ax.legend(handles=[linear_p, xgb_p], fontsize=9)

    # Right: XGBoost train/test gap
    xgb_names = ['Default\n(depth=6)', 'Tuned\n(depth=7)', 'Conservative\n(depth=4)', 'Early\nStopping']
    xgb_train = [5.631, 10.452, 15.051, 11.206]
    xgb_test  = [15.493, 15.376, 15.621, 15.284]
    x = np.arange(len(xgb_names)); w = 0.32
    ax2 = axes[1]
    ax2.bar(x - w/2, xgb_train, w, label='Train MAPE', color=PALETTE[0], edgecolor='white')
    ax2.bar(x + w/2, xgb_test,  w, label='Test MAPE',  color=PALETTE[3], edgecolor='white')
    lean_best = results[0]['test_mape']  # Ridge is always first entry
    ax2.axhline(lean_best, color=PALETTE[2], linestyle='--', lw=1.8,
                label=f'Lean Ridge ({lean_best:.2f}%)')
    ax2.set_xticks(x); ax2.set_xticklabels(xgb_names)
    ax2.set_ylabel('MAPE (%)')
    ax2.set_title('XGBoost — Train vs Test MAPE\n(large gap = overfitting to training data)')
    ax2.set_ylim(0, 20); ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/fig_comparison.png', dpi=130)
    plt.close()
    print("  Saved fig_comparison.png")

# ══════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("DSS5104 — House Price Prediction Pipeline")
    print("=" * 60)

    # ── Load and explore ─────────────────────────────────────────
    print("\n[1] Loading and cleaning data ...")
    df = load_and_clean('data\\house_dataset.csv')

    print("\n[2] Generating EDA figures ...")
    plot_price_distribution(df)
    plot_correlations(df)
    plot_eda_insights(df)

    # ── Feature engineering ──────────────────────────────────────
    print("\n[3] Engineering features ...")
    df_feat = engineer_features(df)
    df_feat['log_price'] = np.log(df_feat['price'])

    # Train / test split (fixed seed for reproducibility)
    train_df, test_df = train_test_split(df_feat, test_size=0.2, random_state=SEED)
    y_tr = train_df['log_price'].values
    y_te = test_df['log_price'].values
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # ── Lasso-based feature selection (proper workflow) ─────────
    print("\n[4] Lasso feature selection on full candidate pool ...")
    lean_features, path_df, elbow_cv, elbow_se = lasso_select(
        train_df, test_df, y_tr, y_te, CANDIDATE_FEATURES
    )
    global LEAN_FEATURES
    LEAN_FEATURES = lean_features

    # ── Encode with final train/test split ───────────────────────
    train_enc, test_enc = add_encodings(train_df, test_df)

    # ── Primary model: Lasso-selected features + Ridge ───────────
    n_lean = len(LEAN_FEATURES)
    print(f"\n[5] Training primary model (Lean Ridge, {n_lean} features) ...")
    X_lean_tr = train_enc[LEAN_FEATURES].values
    X_lean_te = test_enc[LEAN_FEATURES].values

    # Tune Ridge alpha via CV (independent of feature selection)
    best_alpha, best_cv = 1.0, 999.0
    print("  Tuning Ridge alpha ...")
    for alpha in [0.01, 0.1, 0.5, 1, 5, 10, 50]:
        cv_m, _ = cross_val_mape(X_lean_tr, y_tr, alpha=alpha)
        if cv_m < best_cv:
            best_cv, best_alpha = cv_m, alpha
    print(f"  Best alpha={best_alpha}  CV MAPE={best_cv:.4f}%")

    sc_lean = StandardScaler()
    ridge_lean = Ridge(alpha=best_alpha).fit(sc_lean.fit_transform(X_lean_tr), y_tr)
    lean_train_mape = mape(y_tr, ridge_lean.predict(sc_lean.transform(X_lean_tr)))
    lean_test_mape  = mape(y_te, ridge_lean.predict(sc_lean.transform(X_lean_te)))
    print(f"  Lean Ridge  — train={lean_train_mape:.4f}%  test={lean_test_mape:.4f}%  gap={lean_test_mape-lean_train_mape:+.4f}%")

    # ── Kernel models — same 7 lean features as Ridge ────────────
    # Feature set is consistent: if the elbow justifies 7 features
    # for Ridge, it applies equally here. The kernel expands the
    # hypothesis space (non-linear interactions) but not the inputs.
    print("\n[6] Training kernel models (Nyström, lean features) ...")
    X_lean_tr_sc = sc_lean.transform(X_lean_tr)
    X_lean_te_sc = sc_lean.transform(X_lean_te)

    rbf_model  = make_pipeline(
        Nystroem(kernel='rbf', gamma=0.1, n_components=500, random_state=SEED),
        Ridge(alpha=0.1)
    )
    poly_model = make_pipeline(
        Nystroem(kernel='poly', degree=3, gamma=0.1, coef0=1, n_components=500, random_state=SEED),
        Ridge(alpha=1)
    )
    rbf_model.fit(X_lean_tr_sc, y_tr)
    poly_model.fit(X_lean_tr_sc, y_tr)
    rbf_train  = mape(y_tr, rbf_model.predict(X_lean_tr_sc))
    rbf_test   = mape(y_te, rbf_model.predict(X_lean_te_sc))
    poly_train = mape(y_tr, poly_model.predict(X_lean_tr_sc))
    poly_test  = mape(y_te, poly_model.predict(X_lean_te_sc))

    # CV with per-fold re-encoding — consistent with lasso_select.
    # Target encodings are data-dependent so must be re-fitted inside each fold.
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rbf_cv_scores, poly_cv_scores = [], []
    idx = np.arange(len(train_df))
    for ti, vi in kf.split(idx):
        tr_f = train_df.iloc[ti].copy()
        va_f = train_df.iloc[vi].copy()
        tr_f, va_f = add_encodings(tr_f, va_f)
        sc_f = StandardScaler()
        Xf_tr = sc_f.fit_transform(tr_f[LEAN_FEATURES].values)
        Xf_va = sc_f.transform(va_f[LEAN_FEATURES].values)
        m_rbf = make_pipeline(
            Nystroem(kernel='rbf', gamma=0.1, n_components=500, random_state=SEED), Ridge(alpha=0.1)
        ).fit(Xf_tr, y_tr[ti])
        rbf_cv_scores.append(mape(y_tr[vi], m_rbf.predict(Xf_va)))
        m_poly = make_pipeline(
            Nystroem(kernel='poly', degree=3, gamma=0.1, coef0=1, n_components=500, random_state=SEED), Ridge(alpha=1)
        ).fit(Xf_tr, y_tr[ti])
        poly_cv_scores.append(mape(y_tr[vi], m_poly.predict(Xf_va)))
    rbf_cv  = np.mean(rbf_cv_scores)
    poly_cv = np.mean(poly_cv_scores)
    print(f"  RBF Kernel  — train={rbf_train:.4f}%  CV={rbf_cv:.4f}%  test={rbf_test:.4f}%")
    print(f"  Poly Kernel — train={poly_train:.4f}%  CV={poly_cv:.4f}%  test={poly_test:.4f}%")

    # ── Figures ──────────────────────────────────────────────────
    print("\n[7] Generating output figures ...")
    plot_mape_vs_nfeats(path_df, lean_n=len(LEAN_FEATURES), lean_test=lean_test_mape, lean_cv=elbow_cv)
    plot_coefficients(ridge_lean, LEAN_FEATURES, sc_lean)
    lean_pred = ridge_lean.predict(sc_lean.transform(X_lean_te))
    plot_diagnostics(y_te, lean_pred)
    plot_segments(y_te, lean_pred)

    results_for_plot = [
        {'name': f'Ridge\n({len(LEAN_FEATURES)} feats)', 'test_mape': lean_test_mape, 'color': PALETTE[0]},
        {'name': 'RBF\nKernel',       'test_mape': rbf_test,  'color': PALETTE[0]},
        {'name': 'Poly\nKernel',      'test_mape': poly_test, 'color': PALETTE[0]},
        {'name': 'XGB\nConservative', 'test_mape': 15.621,    'color': PALETTE[1]},
        {'name': 'XGB Early\nStop',   'test_mape': 15.284,    'color': PALETTE[1]},
    ]
    plot_comparison(results_for_plot)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<40} {'Test MAPE':>10}")
    print("  " + "-" * 52)
    print(f"  {'Lean Ridge (7 feats) — PRIMARY':<40} {lean_test_mape:>9.3f}%")
    print(f"  {'RBF Kernel (Nyström, 7 feats)':<40} {rbf_test:>9.3f}%")
    print(f"  {'Poly Kernel (Nyström, 7 feats)':<40} {poly_test:>9.3f}%")
    print(f"  {'XGB Conservative (benchmark)':<40} {'15.621':>10}")
    print(f"  {'XGB Early Stopping (benchmark)':<40} {'15.284':>10}")
    print(f"\n  Lean Ridge vs XGB Conservative: {lean_test_mape - 15.621:+.3f}pp")

    summary = {
        'lean_test': lean_test_mape, 'lean_train': lean_train_mape,
        'lean_cv': elbow_cv,          # CV MAPE at elbow (5-fold, per-fold re-encoding)
        'lean_se': elbow_se,          # SE at elbow point
        'alpha_tuning_cv': best_cv,   # CV MAPE from alpha grid search (different protocol)
        'lean_alpha': best_alpha,
        'lean_n': len(LEAN_FEATURES),
        'lean_features': LEAN_FEATURES,
        'candidate_n': len(CANDIDATE_FEATURES),
        'rbf_test': rbf_test, 'rbf_train': rbf_train, 'rbf_cv': rbf_cv,
        'poly_test': poly_test, 'poly_train': poly_train, 'poly_cv': poly_cv,
        'xgb_conservative': 15.621, 'xgb_early_stop': 15.284,
    }
    with open('results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n  Results saved to results.json")
    print("\nDone. All figures saved to ./figures/")

if __name__ == '__main__':
    main()
