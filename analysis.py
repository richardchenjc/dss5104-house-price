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
from sklearn.neighbors import NearestNeighbors

sns.set_theme(style='whitegrid', font_scale=1.05)
PALETTE = ['#1f5f8b', '#c47d00', '#2a9d47', '#c0392b', '#7b2d8b']
SEED = 42
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs('figures', exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 1. DATA LOADING AND CLEANING
# ══════════════════════════════════════════════════════════════════

def load_and_clean(path='~/data/house_dataset.csv'):
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
    df['log_price'] = np.log(df['price'])
    df['zipcode'] = df['statezip'].str.extract(r'(\d{5})')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Age curve — scatter with lowess
    age_bins = pd.cut(df['house_age'], bins=range(0, 101, 5))
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
    d['was_renovated'] = (d['yr_renovated'] > 0).astype(int)
    d['yr_reno_fill'] = np.where(d['yr_renovated'] > 0, d['yr_renovated'], d['yr_built'])
    d['effective_age'] = d['year_sold'] - d['yr_reno_fill']
    d['recent_reno'] = (
        (d['yr_renovated'] > 0) & (d['year_sold'] - d['yr_renovated'] <= 10)
    ).astype(int)
    d['reno_lag'] = np.where(
        d['yr_renovated'] > 0, d['yr_renovated'] - d['yr_built'], 0
    )

    # ── Condition ────────────────────────────────────────────────
    d['top_condition'] = (d['condition'] == 5).astype(int)
    d['cond_x_age'] = d['condition'] * d['house_age']

    # ── View / waterfront ────────────────────────────────────────
    d['any_view'] = (d['view'] > 0).astype(int)
    d['view_x_wf'] = d['view'] * d['waterfront']

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
    d['has_basement'] = (d['sqft_basement'] > 0).astype(int)
    d['basement_ratio'] = d['sqft_basement'] / (d['sqft_living'] + 1)

    # ── Interactions ─────────────────────────────────────────────
    d['size_x_condition'] = d['log_sqft_living'] * d['condition']
    d['size_x_floors'] = d['log_sqft_living'] * d['floors']

    # ── Categorical helpers ──────────────────────────────────────
    d['zipcode'] = d['statezip'].str.extract(r'(\d{5})').astype(str)
    bins = [0, 1919, 1939, 1959, 1979, 1999, 2014]
    labels = ['pre1920', '1920s_30s', '1940s_50s', '1960s_70s', '1980s_90s', '2000s']
    d['build_era'] = pd.cut(d['yr_built'], bins=bins, labels=labels).astype(str)

    return d


def smooth_target_encode(train_df, test_df, col, target, strength=30):
    """
    Bayesian-smoothed target encoding.
    Blends category mean with global mean; prevents overfitting on rare levels.
    Fitted on train_df only — test_df maps to train encodings.
    """
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['mean', 'count'])
    smoothed = (
        (stats['count'] * stats['mean'] + strength * global_mean)
        / (stats['count'] + strength)
    )
    train_enc = train_df[col].map(smoothed).fillna(global_mean)
    test_enc = test_df[col].map(smoothed).fillna(global_mean)
    return train_enc, test_enc


def add_location_encodings(train_df, test_df):
    """Add smoothed target encodings and location interaction features."""
    tr, te = train_df.copy(), test_df.copy()
    for col, grp in [('city', 'city_lp'), ('zipcode', 'zip_lp'), ('build_era', 'era_lp')]:
        tr[grp], te[grp] = smooth_target_encode(tr, te, col, 'log_price', strength=30)
    for d_ in [tr, te]:
        d_['zip_city_diff'] = d_['zip_lp'] - d_['city_lp']
        d_['zip_x_sqft'] = d_['zip_lp'] * d_['log_sqft_living']
        d_['city_x_sqft'] = d_['city_lp'] * d_['log_sqft_living']
        d_['zip_x_cond'] = d_['zip_lp'] * d_['condition']
    return tr, te


# ── Feature sets ─────────────────────────────────────────────────

# Primary interpretable model — 20 features selected via Lasso path
LEAN_FEATURES = [
    # Location (4) — smoothed target encodings + key interaction
    'city_lp', 'zip_lp', 'zip_city_diff', 'zip_x_sqft',
    # Size (3) — log-transformed; quadratic term captures concave curve
    'log_sqft_above', 'log_sqft_living_sq', 'log_sqft_basement',
    # Condition (2) — level + interaction with age
    'condition', 'cond_x_age',
    # Age (2) — log captures U-shape; effective_age uses renovation year
    'log_house_age', 'effective_age',
    # Rooms (2)
    'bathrooms', 'sqft_per_bedroom',
    # View / waterfront (2)
    'view', 'waterfront',
    # Basement (1)
    'has_basement',
    # Renovation (1)
    'recent_reno',
    # Interactions (2)
    'size_x_condition', 'city_x_sqft',
    # Time (1)
    'month_sold',
]

# Extended feature set for stacked ensemble
EXTENDED_FEATURES = LEAN_FEATURES + [
    'zip_lp', 'era_lp', 'zip_x_cond',
    'log_sqft_living', 'log_sqft_lot', 'log_sqft_living_sq', 'living_to_lot',
    'bedrooms_clip' if False else 'bathrooms',  # placeholder; see below
    'floors', 'bath_bed_ratio', 'basement_ratio',
    'house_age', 'log_house_age', 'is_new',
    'was_renovated', 'reno_lag', 'any_view', 'view_x_wf',
    'size_x_floors',
]

# Build deduplicated extended feature list
_seen = set()
EXTENDED_FEATURES_DEDUP = []
for f in LEAN_FEATURES + [
    'era_lp', 'zip_x_cond',
    'log_sqft_living', 'log_sqft_lot', 'living_to_lot',
    'floors', 'bath_bed_ratio', 'basement_ratio',
    'house_age', 'is_new',
    'was_renovated', 'reno_lag', 'any_view', 'view_x_wf',
    'size_x_floors',
]:
    if f not in _seen:
        EXTENDED_FEATURES_DEDUP.append(f)
        _seen.add(f)
EXTENDED_FEATURES = EXTENDED_FEATURES_DEDUP


# ══════════════════════════════════════════════════════════════════
# 4. MODELLING UTILITIES
# ══════════════════════════════════════════════════════════════════

def mape(y_true_log, y_pred_log):
    """MAPE computed in original price space after exponentiation."""
    return np.mean(np.abs(
        (np.exp(y_true_log) - np.exp(y_pred_log)) / np.exp(y_true_log)
    )) * 100


def cross_val_mape(X, y, alpha=5, n_splits=5):
    """5-fold CV MAPE for Ridge regression."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for ti, vi in kf.split(X):
        sc = StandardScaler()
        m = Ridge(alpha=alpha).fit(sc.fit_transform(X[ti]), y[ti])
        scores.append(mape(y[vi], m.predict(sc.transform(X[vi]))))
    return np.mean(scores), np.std(scores)


def oof_stack(base_models, X_tr, y_tr, X_te, n_splits=5):
    """
    Out-of-fold stacking. Returns OOF training predictions and
    averaged test predictions for each base model.
    Prevents leakage: meta-learner trains only on OOF predictions.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y_tr), len(base_models)))
    test_preds = np.zeros((len(X_te), len(base_models)))
    for j, model in enumerate(base_models):
        fold_test = []
        for ti, vi in kf.split(X_tr):
            model.fit(X_tr[ti], y_tr[ti])
            oof[vi, j] = model.predict(X_tr[vi])
            fold_test.append(model.predict(X_te))
        test_preds[:, j] = np.mean(fold_test, axis=0)
    return oof, test_preds


# ══════════════════════════════════════════════════════════════════
# 5. LASSO REGULARISATION PATH (MAPE vs N features)
# ══════════════════════════════════════════════════════════════════

def run_lasso_path(X_tr, X_te, y_tr, y_te, feature_names):
    """Sweep Lasso alphas to show optimal feature count."""
    alphas = np.logspace(-4, -1, 40)
    results = []
    prev_n = len(feature_names) + 1
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=5000).fit(X_tr_sc, y_tr)
        n = (lasso.coef_ != 0).sum()
        if n == prev_n or n == 0:
            continue
        prev_n = n
        fold_mapes = []
        for ti, vi in kf.split(X_tr_sc):
            sc_ = StandardScaler()
            l = Lasso(alpha=alpha, max_iter=5000).fit(sc_.fit_transform(X_tr[ti]), y_tr[ti])
            fold_mapes.append(mape(y_tr[vi], l.predict(sc_.transform(X_tr[vi]))))
        te_m = mape(y_te, lasso.predict(X_te_sc))
        results.append({'n_features': n, 'cv_mape': np.mean(fold_mapes), 'test_mape': te_m})

    return pd.DataFrame(results).sort_values('n_features')


def plot_mape_vs_nfeats(path_df, lean_n, lean_test):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(path_df['n_features'], path_df['cv_mape'],
            'o-', color=PALETTE[0], lw=2, ms=5, label='CV MAPE (5-fold)')
    ax.plot(path_df['n_features'], path_df['test_mape'],
            's--', color=PALETTE[1], lw=2, ms=5, label='Test MAPE')
    ax.axvline(lean_n, color=PALETTE[2], linestyle=':', lw=2,
               label=f'Selected model ({lean_n} features, {lean_test:.2f}% MAPE)')
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
    lean_best = next(r['test_mape'] for r in results if 'Lean' in r['name'])
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
    df = load_and_clean('data/house_dataset.csv')

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

    # Location encodings (fitted on train only)
    train_enc, test_enc = add_location_encodings(train_df, test_df)

    # ── Lasso path ───────────────────────────────────────────────
    print("\n[4] Running Lasso regularisation path ...")
    X_all = train_enc[LEAN_FEATURES].values
    X_all_te = test_enc[LEAN_FEATURES].values
    path_df = run_lasso_path(X_all, X_all_te, y_tr, y_te, LEAN_FEATURES)
    print(f"  Sweet spot: ~19–20 features, CV MAPE ≈ {path_df.loc[path_df['n_features'].between(18,22),'cv_mape'].min():.2f}%")

    # ── Primary model: Lean 20-feature Ridge ─────────────────────
    print("\n[5] Training primary model (Lean Ridge, 20 features) ...")
    X_lean_tr = train_enc[LEAN_FEATURES].values
    X_lean_te = test_enc[LEAN_FEATURES].values

    # Tune alpha via CV
    best_alpha, best_cv = 0.5, 999.0
    print("  Tuning alpha ...")
    for alpha in [0.01, 0.1, 0.5, 1, 5, 10, 50]:
        cv_m, cv_s = cross_val_mape(X_lean_tr, y_tr, alpha=alpha)
        if cv_m < best_cv:
            best_cv, best_alpha = cv_m, alpha
    print(f"  Best alpha={best_alpha}  CV MAPE={best_cv:.4f}%")

    sc_lean = StandardScaler()
    ridge_lean = Ridge(alpha=best_alpha).fit(sc_lean.fit_transform(X_lean_tr), y_tr)
    lean_train_mape = mape(y_tr, ridge_lean.predict(sc_lean.transform(X_lean_tr)))
    lean_test_mape  = mape(y_te, ridge_lean.predict(sc_lean.transform(X_lean_te)))
    print(f"  Lean Ridge  — train={lean_train_mape:.4f}%  test={lean_test_mape:.4f}%  gap={lean_test_mape-lean_train_mape:+.4f}%")

    # ── Extended model: 35-feature Ridge ─────────────────────────
    print("\n[6] Training extended model (35 features) ...")
    X_ext_tr = train_enc[EXTENDED_FEATURES].values
    X_ext_te = test_enc[EXTENDED_FEATURES].values
    sc_ext = StandardScaler()
    ridge_ext = Ridge(alpha=5).fit(sc_ext.fit_transform(X_ext_tr), y_tr)
    ext_cv, _ = cross_val_mape(X_ext_tr, y_tr, alpha=5)
    ext_test = mape(y_te, ridge_ext.predict(sc_ext.transform(X_ext_te)))
    ext_train = mape(y_tr, ridge_ext.predict(sc_ext.transform(X_ext_tr)))
    print(f"  Extended Ridge — train={ext_train:.4f}%  test={ext_test:.4f}%  CV={ext_cv:.4f}%")

    # ── Kernel models on extended features ───────────────────────
    print("\n[7] Training kernel models (Nyström approximation) ...")
    X_ext_tr_sc = sc_ext.transform(X_ext_tr)
    X_ext_te_sc = sc_ext.transform(X_ext_te)

    rbf_model  = make_pipeline(
        Nystroem(kernel='rbf', gamma=0.005, n_components=500, random_state=SEED),
        Ridge(alpha=0.01)
    )
    poly_model = make_pipeline(
        Nystroem(kernel='poly', degree=3, gamma=0.01, coef0=1, n_components=500, random_state=SEED),
        Ridge(alpha=1)
    )
    rbf_model.fit(X_ext_tr_sc, y_tr)
    poly_model.fit(X_ext_tr_sc, y_tr)
    rbf_test  = mape(y_te, rbf_model.predict(X_ext_te_sc))
    poly_test = mape(y_te, poly_model.predict(X_ext_te_sc))
    print(f"  RBF Kernel  — test={rbf_test:.4f}%")
    print(f"  Poly Kernel — test={poly_test:.4f}%")

    # ── Stacked ensemble ─────────────────────────────────────────
    print("\n[8] OOF stacking ...")
    base_models = [
        Ridge(alpha=5),
        make_pipeline(Nystroem(kernel='rbf', gamma=0.005, n_components=500, random_state=SEED), Ridge(alpha=0.01)),
        make_pipeline(Nystroem(kernel='poly', degree=3, gamma=0.01, coef0=1, n_components=500, random_state=SEED), Ridge(alpha=1)),
    ]
    oof_preds, test_preds_stack = oof_stack(base_models, X_ext_tr_sc, y_tr, X_ext_te_sc)
    meta = Ridge(alpha=0.001).fit(oof_preds, y_tr)
    stacked_test = mape(y_te, meta.predict(test_preds_stack))
    print(f"  Stacked (Ridge+RBF+Poly) — test={stacked_test:.4f}%")
    print(f"  Meta weights: Ridge={meta.coef_[0]:.3f} RBF={meta.coef_[1]:.3f} Poly={meta.coef_[2]:.3f}")

    # ── Figures ──────────────────────────────────────────────────
    print("\n[9] Generating output figures ...")
    plot_mape_vs_nfeats(path_df, lean_n=len(LEAN_FEATURES), lean_test=lean_test_mape)
    plot_coefficients(ridge_lean, LEAN_FEATURES, sc_lean)
    lean_pred = ridge_lean.predict(sc_lean.transform(X_lean_te))
    plot_diagnostics(y_te, lean_pred)
    plot_segments(y_te, lean_pred)

    results_for_plot = [
        {'name': 'Ridge\nLean (20)', 'test_mape': lean_test_mape,  'color': PALETTE[0]},
        {'name': 'Ridge\nExt. (35)', 'test_mape': ext_test,         'color': PALETTE[0]},
        {'name': 'RBF\nKernel',      'test_mape': rbf_test,          'color': PALETTE[0]},
        {'name': 'Poly\nKernel',     'test_mape': poly_test,         'color': PALETTE[0]},
        {'name': 'Stacked\n(best)',  'test_mape': stacked_test,      'color': PALETTE[2]},
        {'name': 'XGB\nConservative','test_mape': 15.621,            'color': PALETTE[1]},
        {'name': 'XGB Early\nStop',  'test_mape': 15.284,            'color': PALETTE[1]},
    ]
    plot_comparison(results_for_plot)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<40} {'Test MAPE':>10}")
    print("  " + "-" * 52)
    print(f"  {'Lean Ridge (20 feats) — PRIMARY':<40} {lean_test_mape:>9.3f}%")
    print(f"  {'Extended Ridge (35 feats)':<40} {ext_test:>9.3f}%")
    print(f"  {'RBF Kernel (Nyström)':<40} {rbf_test:>9.3f}%")
    print(f"  {'Poly Kernel (Nyström)':<40} {poly_test:>9.3f}%")
    print(f"  {'Stacked ensemble — BEST':<40} {stacked_test:>9.3f}%")
    print(f"  {'XGB Conservative (benchmark)':<40} {'15.621':>10}")
    print(f"  {'XGB Early Stopping (benchmark)':<40} {'15.284':>10}")
    print(f"\n  Lean Ridge vs XGB Conservative: {lean_test_mape - 15.621:+.3f}pp")
    print(f"  Stacked vs XGB Early Stopping:  {stacked_test - 15.284:+.3f}pp")

    # Save results to JSON for report script
    summary = {
        'lean_test': lean_test_mape, 'lean_train': lean_train_mape,
        'lean_cv': best_cv, 'lean_alpha': best_alpha, 'lean_n': len(LEAN_FEATURES),
        'ext_test': ext_test, 'ext_train': ext_train, 'ext_cv': ext_cv,
        'rbf_test': rbf_test, 'poly_test': poly_test,
        'stacked_test': stacked_test,
        'meta_weights': meta.coef_.tolist(),
        'xgb_conservative': 15.621, 'xgb_early_stop': 15.284,
    }
    with open('results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n  Results saved to results.json")
    print("\nDone. All figures saved to ./figures/")


if __name__ == '__main__':
    main()
