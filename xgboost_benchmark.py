"""
DSS5104 — XGBoost Benchmark Runner
===================================
Trains XGBoost in four configurations on the same train/test split as analysis.py
and writes results to xgb_results.json.

Run ONCE before analysis.py to generate reproducible benchmark figures:
    python xgboost_benchmark.py

Requires: xgboost (pip install xgboost)
Reads:    data/house_dataset.csv
Writes:   xgb_results.json

The four configurations mirror common choices in the literature:
  Default       — standard settings; reveals overfitting risk
  Tuned deeper  — deeper trees; moderately tuned
  Conservative  — heavy regularisation, shallow trees; honest comparator
  Early Stopping — validation-set stopping; principled comparator
"""

import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit(
        "xgboost not installed. Run: pip install xgboost\n"
        "Then re-run: python xgboost_benchmark.py"
    )

# ── Import shared utilities from analysis.py ────────────────────────────────
# We reuse load_and_clean, engineer_features, add_encodings, mape, SEED
# by exec-ing analysis.py (avoids code duplication and guarantees identical split)
exec(open('analysis.py').read())

print("\n" + "=" * 60)
print("DSS5104 — XGBoost Benchmark")
print("=" * 60)

# ── Prepare data (identical split to analysis.py) ────────────────────────────
print("\n[1] Loading data and engineering features ...")
df       = load_and_clean('data/house_dataset.csv')
df_feat  = engineer_features(df)
df_feat['log_price'] = np.log(df_feat['price'])

train_df, test_df = train_test_split(df_feat, test_size=0.2, random_state=SEED)
y_tr = train_df['log_price'].values
y_te = test_df['log_price'].values

# ── Build feature matrix — same encodings as analysis.py ────────────────────
# XGBoost receives all 40 candidate features (not just the lean 12 — it selects its own)
print("[2] Encoding features ...")
_, _  = lasso_select(train_df, test_df, y_tr, y_te, CANDIDATE_FEATURES)
train_enc, test_enc = add_encodings(train_df, test_df)

X_tr = train_enc[CANDIDATE_FEATURES].values
X_te = test_enc[CANDIDATE_FEATURES].values

# ── XGBoost configurations ────────────────────────────────────────────────────
# Fixed random_state for reproducibility; all other params intentional
CONFIGS = {
    'Default (depth=6)': dict(
        max_depth=6, n_estimators=500,
        learning_rate=0.1, subsample=1.0, colsample_bytree=1.0,
        random_state=SEED, n_jobs=-1
    ),
    'Tuned (depth=7)': dict(
        max_depth=7, n_estimators=800,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1
    ),
    'Conservative (depth=4)': dict(
        max_depth=4, n_estimators=400,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, reg_lambda=5.0, reg_alpha=1.0,
        random_state=SEED, n_jobs=-1
    ),
}

print("\n[3] Training XGBoost configurations ...")
results = {}

# ── 1–3: Fixed configurations ─────────────────────────────────────────────────
for name, params in CONFIGS.items():
    model = xgb.XGBRegressor(objective='reg:squarederror', **params)
    model.fit(X_tr, y_tr, verbose=False)
    train_mape = mape(y_tr, model.predict(X_tr))
    test_mape  = mape(y_te, model.predict(X_te))
    n_trees    = model.best_iteration + 1 if hasattr(model, 'best_iteration') and model.best_iteration else params['n_estimators']
    gap        = test_mape - train_mape
    results[name] = {
        'train_mape': round(train_mape, 3),
        'test_mape':  round(test_mape,  3),
        'gap':        round(gap,        3),
        'n_estimators_used': n_trees,
        'params': params,
    }
    print(f"  {name:<30}: train={train_mape:.3f}%  test={test_mape:.3f}%  gap={gap:+.3f}pp")

# ── 4: Early stopping ─────────────────────────────────────────────────────────
# Uses 20% of training set as validation to determine optimal n_estimators
val_size = int(len(X_tr) * 0.2)
X_tr_sub = X_tr[:-val_size];  y_tr_sub = y_tr[:-val_size]
X_val    = X_tr[-val_size:];  y_val    = y_tr[-val_size:]

es_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=6, n_estimators=1000,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1,
    early_stopping_rounds=20,
    eval_metric='rmse'
)
es_model.fit(
    X_tr_sub, y_tr_sub,
    eval_set=[(X_val, y_val)],
    verbose=False
)
n_used     = es_model.best_iteration + 1
# Retrain on full training set with optimal n_estimators
final_es = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=6, n_estimators=n_used,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1
)
final_es.fit(X_tr, y_tr, verbose=False)
es_train = mape(y_tr, final_es.predict(X_tr))
es_test  = mape(y_te, final_es.predict(X_te))
es_gap   = es_test - es_train
results['Early Stopping'] = {
    'train_mape': round(es_train, 3),
    'test_mape':  round(es_test,  3),
    'gap':        round(es_gap,   3),
    'n_estimators_used': n_used,
    'params': {'early_stopping_rounds': 20, 'val_fraction': 0.2},
}
print(f"  {'Early Stopping':<30}: train={es_train:.3f}%  test={es_test:.3f}%  "
      f"gap={es_gap:+.3f}pp  (n={n_used} trees)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("XGBoost Benchmark Summary")
print("=" * 60)
print(f"  {'Configuration':<30} {'Train':>7}  {'Test':>7}  {'Gap':>8}")
print("  " + "-" * 56)
for name, r in results.items():
    print(f"  {name:<30} {r['train_mape']:>6.3f}%  {r['test_mape']:>6.3f}%  {r['gap']:>+7.3f}pp")

# ── Write xgb_results.json ─────────────────────────────────────────────────────
# Keys match what analysis.py expects: xgb_conservative, xgb_early_stop, xgb_all
xgb_out = {
    'xgb_conservative': results['Conservative (depth=4)']['test_mape'],
    'xgb_early_stop':   results['Early Stopping']['test_mape'],
    'xgb_all': {
        name: {
            'train_mape': r['train_mape'],
            'test_mape':  r['test_mape'],
            'gap':        r['gap'],
            'n_estimators_used': r['n_estimators_used'],
        }
        for name, r in results.items()
    },
    'xgboost_version': xgb.__version__,
}
with open('xgb_results.json', 'w') as f:
    json.dump(xgb_out, f, indent=2)
print(f"\n  Results written to xgb_results.json")
print(f"  XGBoost version: {xgb.__version__}")
print("\nDone. Run analysis.py to regenerate the full report.")
