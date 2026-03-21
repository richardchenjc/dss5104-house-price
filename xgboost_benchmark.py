"""
DSS5104 — XGBoost Benchmark Runner
===================================
Trains XGBoost on the same train/test split as analysis.py and writes
results to xgb_results.json, which analysis.py merges into results.json.

Workflow:
    python xgboost_benchmark.py    # ~5-10 min
    python analysis.py             # picks up xgb_results.json automatically
    python make_report.py

Requires: xgboost (pip install xgboost)
Reads:    data/house_dataset.csv
Writes:   xgb_results.json

Structure
---------
Step 1: Reference configurations (Default, Conservative, Early Stopping)
        — retained for report Table 8 comparison continuity.

Step 2: Randomised hyperparameter search (50 iterations, 5-fold CV)
        — "well-tuned XGBoost" as required by assignment instructions.
        — CV uses per-fold re-encoding (same protocol as analysis.py) so
          target/KNN/direction encodings never leak from validation rows.
        — Best CV configuration is retrained on full training set.

The tuned model's test MAPE is stored as xgb_tuned and used as the
primary benchmark in the report (Section 8, Table 8).
"""

import warnings; warnings.filterwarnings('ignore')
import json, time
import numpy as np
from sklearn.model_selection import train_test_split, KFold

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit(
        "xgboost not installed.\n"
        "Run: pip install xgboost\n"
        "Then: python xgboost_benchmark.py"
    )

# ── Import shared utilities from analysis.py ────────────────────────────────
# Reuses load_and_clean, engineer_features, add_encodings, mape, SEED,
# CANDIDATE_FEATURES, lasso_select — guarantees identical split and features.
exec(open('analysis.py', encoding='utf-8').read())

print("\n" + "=" * 65)
print("DSS5104 — XGBoost Benchmark with Hyperparameter Search")
print("=" * 65)

# ── Data (identical split to analysis.py) ────────────────────────────────────
print("\n[1] Loading data and engineering features ...")
df       = load_and_clean('data/house_dataset.csv')
df_feat  = engineer_features(df)
df_feat['log_price'] = np.log(df_feat['price'])

train_df, test_df = train_test_split(df_feat, test_size=0.2, random_state=SEED)
y_tr = train_df['log_price'].values
y_te = test_df['log_price'].values

# ── Feature matrix — all 40 candidates ───────────────────────────────────────
# XGBoost gets the full candidate pool (not just the Lasso-selected 12).
# It implicitly selects features via tree splits and regularisation.
print("[2] Encoding features ...")
_, _, _, _ = lasso_select(train_df, test_df, y_tr, y_te, CANDIDATE_FEATURES)
train_enc, test_enc = add_encodings(train_df, test_df)
X_tr = train_enc[CANDIDATE_FEATURES].values
X_te = test_enc[CANDIDATE_FEATURES].values
print(f"  Feature matrix: {X_tr.shape[0]} train × {X_tr.shape[1]} features")

# ── CV helper with per-fold re-encoding ──────────────────────────────────────
def cv_mape_xgb(params, n_splits=5):
    """
    5-fold MAPE CV for XGBoost with full re-encoding per fold.
    Mirrors analysis.py's lasso_select CV exactly: target encodings, KNN,
    frequency, and direction features are all re-fitted from scratch on the
    training fold, preventing any leakage from validation rows.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_mapes = []
    for ti, vi in kf.split(np.arange(len(train_df))):
        tr_f = train_df.iloc[ti].copy()
        va_f = train_df.iloc[vi].copy()
        tr_f, va_f = add_encodings(tr_f, va_f)
        m = xgb.XGBRegressor(objective='reg:squarederror', **params)
        m.fit(tr_f[CANDIDATE_FEATURES].values, y_tr[ti], verbose=False)
        fold_mapes.append(mape(y_tr[vi], m.predict(va_f[CANDIDATE_FEATURES].values)))
    return np.mean(fold_mapes), np.std(fold_mapes) / np.sqrt(n_splits)

# ── Step 1: Reference configurations ─────────────────────────────────────────
print("\n[3] Training reference configurations ...")
results = {}

REF_CONFIGS = {
    'Default (depth=6)': dict(
        max_depth=6, n_estimators=500, learning_rate=0.1,
        subsample=1.0, colsample_bytree=1.0,
        random_state=SEED, n_jobs=-1,
    ),
    'Conservative (depth=4)': dict(
        max_depth=4, n_estimators=400, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, reg_lambda=5.0, reg_alpha=1.0,
        random_state=SEED, n_jobs=-1,
    ),
}

for name, params in REF_CONFIGS.items():
    m = xgb.XGBRegressor(objective='reg:squarederror', **params)
    m.fit(X_tr, y_tr, verbose=False)
    tr_m = mape(y_tr, m.predict(X_tr))
    te_m = mape(y_te, m.predict(X_te))
    results[name] = {
        'train_mape': round(tr_m, 3), 'test_mape': round(te_m, 3),
        'gap': round(te_m - tr_m, 3),
        'n_estimators_used': params['n_estimators'], 'params': params,
    }
    print(f"  {name:<32}: train={tr_m:.3f}%  test={te_m:.3f}%  gap={te_m-tr_m:+.3f}pp")

# Early Stopping: find n_estimators on 20% validation holdout, retrain on full set
print("  Training Early Stopping config ...")
val_n = int(len(X_tr) * 0.2)
es_finder = xgb.XGBRegressor(
    objective='reg:squarederror', max_depth=6, n_estimators=1500,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1, early_stopping_rounds=30, eval_metric='rmse',
)
es_finder.fit(X_tr[:-val_n], y_tr[:-val_n],
              eval_set=[(X_tr[-val_n:], y_tr[-val_n:])], verbose=False)
n_opt = es_finder.best_iteration + 1
es_model = xgb.XGBRegressor(
    objective='reg:squarederror', max_depth=6, n_estimators=n_opt,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1,
)
es_model.fit(X_tr, y_tr, verbose=False)
es_tr = mape(y_tr, es_model.predict(X_tr))
es_te = mape(y_te, es_model.predict(X_te))
results['Early Stopping'] = {
    'train_mape': round(es_tr, 3), 'test_mape': round(es_te, 3),
    'gap': round(es_te - es_tr, 3),
    'n_estimators_used': n_opt,
    'params': {'note': f'early_stopping(rounds=30), n_opt={n_opt}'},
}
print(f"  {'Early Stopping':<32}: train={es_tr:.3f}%  test={es_te:.3f}%  "
      f"gap={es_te-es_tr:+.3f}pp  (n_opt={n_opt})")

# ── Step 2: Randomised hyperparameter search ──────────────────────────────────
N_ITER   = 50
N_SPLITS = 5

print(f"\n[4] Randomised search ({N_ITER} iterations × {N_SPLITS}-fold CV w/ re-encoding) ...")
print(f"    Parameter space:")

# Wide search space covering key regularisation trade-offs
PARAM_SPACE = {
    'max_depth':        [3, 4, 5, 6, 7],
    'n_estimators':     [300, 500, 700, 900, 1200, 1500],
    'learning_rate':    [0.005, 0.01, 0.02, 0.05, 0.08, 0.1],
    'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7, 10, 15, 20],
    'reg_lambda':       [0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    'reg_alpha':        [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
    'gamma':            [0.0, 0.1, 0.2, 0.5, 1.0],
}
for k, v in PARAM_SPACE.items():
    print(f"      {k:<22}: {v}")

rng = np.random.RandomState(SEED)
best_cv, best_params, best_iter = 999.0, None, -1
search_log = []
t0 = time.time()

for i in range(N_ITER):
    params = {k: rng.choice(v).item() for k, v in PARAM_SPACE.items()}
    params['random_state'] = SEED
    params['n_jobs']       = -1

    cv_m, cv_se = cv_mape_xgb(params, n_splits=N_SPLITS)
    elapsed = time.time() - t0
    eta     = elapsed / (i + 1) * (N_ITER - i - 1)
    marker  = " ← best" if cv_m < best_cv else ""

    print(f"  [{i+1:>3}/{N_ITER}] CV={cv_m:.4f}% ±{cv_se:.4f}pp  "
          f"d={params['max_depth']} n={params['n_estimators']} "
          f"lr={params['learning_rate']} sub={params['subsample']:.1f}  "
          f"({elapsed:.0f}s, ETA {eta:.0f}s){marker}")

    search_log.append({
        'iter': i + 1, 'cv_mape': round(cv_m, 4),
        'cv_se': round(cv_se, 4), 'params': params,
    })
    if cv_m < best_cv:
        best_cv, best_params, best_iter = cv_m, params.copy(), i + 1

print(f"\n  Search done in {time.time()-t0:.0f}s")
print(f"  Best CV MAPE: {best_cv:.4f}%  at iteration {best_iter}")
print(f"  Best params: {best_params}")

# Retrain best config on full training set, evaluate on held-out test
print("\n[5] Retraining best config on full training set ...")
tuned_model = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
tuned_model.fit(X_tr, y_tr, verbose=False)
tuned_tr = mape(y_tr, tuned_model.predict(X_tr))
tuned_te = mape(y_te, tuned_model.predict(X_te))
print(f"  Tuned XGBoost: train={tuned_tr:.3f}%  test={tuned_te:.3f}%  "
      f"gap={tuned_te-tuned_tr:+.3f}pp")

results['Tuned (randomised search)'] = {
    'train_mape':        round(tuned_tr, 3),
    'test_mape':         round(tuned_te, 3),
    'gap':               round(tuned_te - tuned_tr, 3),
    'cv_mape':           round(best_cv, 4),
    'n_estimators_used': best_params['n_estimators'],
    'params':            best_params,
    'search_iterations': N_ITER,
    'search_cv_splits':  N_SPLITS,
}

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("XGBoost Benchmark Summary")
print("=" * 65)
print(f"  {'Configuration':<34} {'Train':>7}  {'Test':>7}  {'Gap':>8}")
print("  " + "-" * 60)
for name, r in results.items():
    marker = "  ◄ primary benchmark" if name == 'Tuned (randomised search)' else ""
    print(f"  {name:<34} {r['train_mape']:>6.3f}%  "
          f"{r['test_mape']:>6.3f}%  {r['gap']:>+7.3f}pp{marker}")

# ── Write xgb_results.json ────────────────────────────────────────────────────
# Keys used by analysis.py and make_report.py:
#   xgb_conservative — Conservative (depth=4); retained in Table 8 for comparison
#   xgb_early_stop   — Early Stopping; retained in Table 8
#   xgb_tuned        — Best from randomised search; NEW primary benchmark
#   xgb_all          — Per-config details for Table 8
xgb_out = {
    'xgb_conservative': results['Conservative (depth=4)']['test_mape'],
    'xgb_early_stop':   results['Early Stopping']['test_mape'],
    'xgb_tuned':        results['Tuned (randomised search)']['test_mape'],
    'xgb_tuned_cv':     results['Tuned (randomised search)']['cv_mape'],
    'xgb_tuned_params': best_params,
    'xgb_all': {
        name: {
            'train_mape':        r['train_mape'],
            'test_mape':         r['test_mape'],
            'gap':               r['gap'],
            'n_estimators_used': r.get('n_estimators_used', '?'),
        }
        for name, r in results.items()
    },
    'search_log':       search_log,
    'xgboost_version':  xgb.__version__,
}

with open('xgb_results.json', 'w') as f:
    json.dump(xgb_out, f, indent=2)

print(f"\n  Written: xgb_results.json  ({len(json.dumps(xgb_out))//1024} KB)")
print(f"  XGBoost version: {xgb.__version__}")
print(f"\n  Primary benchmark: Tuned (randomised search) = {tuned_te:.3f}% test MAPE")
print(f"  Next: python analysis.py  →  python make_report.py")
