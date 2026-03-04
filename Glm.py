# “””
poisson_toolkit.py

Lightweight toolkit for Poisson frequency modeling in insurance.
Designed for use with pandas, sklearn, and glum.

## Usage:

```
from poisson_toolkit import audit_data, preprocess, univariate_selection, suggest_interactions

report  = audit_data(df)
df_clean, preprocessor = preprocess(df, target='claim_count', exposure='exposure')
selected = univariate_selection(df_clean, target='claim_count', exposure='exposure', top_n=20)
pairs    = suggest_interactions(df_clean, selected, target='claim_count', exposure='exposure', top_n=10)
```

“””

import warnings
import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.preprocessing import LabelEncoder
from glum import GeneralizedLinearRegressor

warnings.filterwarnings(“ignore”)

# ─────────────────────────────────────────────

# 1. DATA AUDIT

# ─────────────────────────────────────────────

def audit_data(df: pd.DataFrame, target: str = None, exposure: str = None) -> pd.DataFrame:
“””
Scan all columns and return a cleaning report.

```
Returns a DataFrame with one row per column containing:
  - dtype, n_missing, pct_missing
  - n_unique, pct_unique
  - suggested_action  ← key output
  - notes

Parameters
----------
df       : raw DataFrame
target   : name of target column (excluded from suggestions)
exposure : name of exposure column (excluded from suggestions)
"""
skip = {c for c in [target, exposure] if c}
rows = []

for col in df.columns:
    s = df[col]
    n = len(s)
    n_missing = int(s.isna().sum())
    pct_missing = round(100 * n_missing / n, 2)
    n_unique = int(s.nunique(dropna=True))
    pct_unique = round(100 * n_unique / n, 2)
    dtype = str(s.dtype)
    action, notes = [], []

    if col in skip:
        action.append("keep_as_is")
        notes.append("target/exposure — do not transform")
    else:
        # ── missing values ──
        if pct_missing > 50:
            action.append("consider_drop")
            notes.append(f"{pct_missing}% missing — high risk")
        elif pct_missing > 0:
            if s.dtype in [np.float64, np.float32, np.int64, np.int32]:
                action.append("impute_median")
            else:
                action.append("impute_mode")

        # ── near-constant ──
        if n_unique == 1:
            action.append("drop_constant")
            notes.append("single unique value — no predictive power")
        elif pct_unique < 0.5 and n_unique <= 2:
            action.append("review_binary")

        # ── high cardinality categoricals ──
        if s.dtype == object or str(s.dtype) == "category":
            if n_unique > 50:
                action.append("high_cardinality_encode")
                notes.append(f"{n_unique} levels — consider grouping rare levels")
            elif n_unique > 1:
                action.append("label_encode_or_woe")
        elif s.dtype in [np.float64, np.float32, np.int64, np.int32]:
            # ── skewness ──
            try:
                skew = float(s.dropna().skew())
                if abs(skew) > 2:
                    action.append("log_transform")
                    notes.append(f"skewness={skew:.2f} — log1p recommended")
            except Exception:
                pass

            # ── outliers (IQR) ──
            try:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                n_outliers = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
                if n_outliers > 0:
                    notes.append(f"{n_outliers} extreme outliers (3×IQR rule)")
                    action.append("cap_outliers")
            except Exception:
                pass

        if not action:
            action.append("ok")

    rows.append({
        "column": col,
        "dtype": dtype,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
        "n_unique": n_unique,
        "pct_unique": pct_unique,
        "suggested_action": " | ".join(action),
        "notes": "; ".join(notes) if notes else "",
    })

report = pd.DataFrame(rows).set_index("column")
print(f"[audit_data] Scanned {len(df.columns)} columns. "
      f"{(report['suggested_action'].str.contains('drop')).sum()} flagged for drop, "
      f"{(report['suggested_action'].str.contains('impute')).sum()} need imputation.")
return report
```

# ─────────────────────────────────────────────

# 2. PREPROCESSING

# ─────────────────────────────────────────────

def preprocess(
df: pd.DataFrame,
target: str,
exposure: str,
drop_threshold_missing: float = 0.5,
cap_outliers_iqr: float = 3.0,
log_skew_threshold: float = 2.0,
max_cardinality: int = 50,
rare_level_threshold: float = 0.01,
) -> tuple:
“””
Clean and encode a DataFrame for Poisson GLM modeling.

```
Steps applied automatically:
  1. Drop near-constant and high-missing columns
  2. Cap outliers (numeric, IQR-based)
  3. Log-transform highly skewed numerics
  4. Impute missing values (median for numeric, mode for categorical)
  5. Group rare categorical levels into 'OTHER'
  6. Label-encode all categoricals

Returns
-------
df_clean   : cleaned DataFrame (target and exposure preserved as-is)
meta       : dict with lists of numeric_cols, cat_cols, dropped_cols, log_transformed
"""
df = df.copy()
skip = {target, exposure}
feature_cols = [c for c in df.columns if c not in skip]
dropped, log_transformed = [], []

# ── Step 1: Drop constants and high-missing ──
for col in feature_cols:
    if df[col].nunique(dropna=True) <= 1:
        df.drop(columns=col, inplace=True)
        dropped.append((col, "constant"))
    elif df[col].isna().mean() > drop_threshold_missing:
        df.drop(columns=col, inplace=True)
        dropped.append((col, f">{int(drop_threshold_missing*100)}% missing"))

feature_cols = [c for c in feature_cols if c in df.columns]

numeric_cols = [c for c in feature_cols if df[c].dtype in
                [np.float64, np.float32, np.int64, np.int32, "int64", "float64"]]
cat_cols = [c for c in feature_cols if c not in numeric_cols]

# ── Step 2: Cap outliers ──
for col in numeric_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - cap_outliers_iqr * iqr, q3 + cap_outliers_iqr * iqr
    df[col] = df[col].clip(lower=lo, upper=hi)

# ── Step 3: Log-transform skewed numerics ──
for col in numeric_cols:
    skew = df[col].dropna().skew()
    if abs(skew) > log_skew_threshold and df[col].min() >= 0:
        df[col] = np.log1p(df[col])
        log_transformed.append(col)

# ── Step 4: Impute missing ──
for col in numeric_cols:
    if df[col].isna().any():
        df[col].fillna(df[col].median(), inplace=True)
for col in cat_cols:
    if df[col].isna().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# ── Step 5: Group rare categorical levels ──
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    rare = freq[freq < rare_level_threshold].index
    if len(rare) > 0:
        df[col] = df[col].where(~df[col].isin(rare), other="OTHER")

# ── Step 6: Drop very high cardinality categoricals ──
for col in cat_cols[:]:
    if df[col].nunique() > max_cardinality:
        df.drop(columns=col, inplace=True)
        dropped.append((col, f"cardinality>{max_cardinality}"))
        cat_cols.remove(col)

# ── Step 7: Label encode ──
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

meta = {
    "numeric_cols": numeric_cols,
    "cat_cols": cat_cols,
    "dropped_cols": dropped,
    "log_transformed": log_transformed,
    "all_features": [c for c in df.columns if c not in skip],
}

print(f"[preprocess] Done. Features kept: {len(meta['all_features'])} | "
      f"Dropped: {len(dropped)} | Log-transformed: {len(log_transformed)}")
return df, meta
```

# ─────────────────────────────────────────────

# 3. UNIVARIATE FEATURE SELECTION

# ─────────────────────────────────────────────

def univariate_selection(
df: pd.DataFrame,
target: str,
exposure: str,
top_n: int = 30,
min_exposure: float = 0.0,
) -> pd.DataFrame:
“””
Rank features by univariate Poisson deviance reduction.

```
For each feature, fits:
    claims ~ feature + offset(log(exposure))   [glum Poisson]
and records the deviance improvement over the null model.

Returns
-------
DataFrame ranked by deviance_reduction (descending), with columns:
  feature | deviance_null | deviance_model | deviance_reduction | pct_reduction | converged
"""
if min_exposure > 0:
    df = df[df[exposure] >= min_exposure].copy()

y = df[target].values
w = df[exposure].values
feature_cols = [c for c in df.columns if c not in {target, exposure}]

# Null model deviance (intercept only)
null_model = GeneralizedLinearRegressor(family="poisson", fit_intercept=True, max_iter=200)
null_model.fit(np.ones((len(y), 1)), y, sample_weight=w)
mu_null = null_model.predict(np.ones((len(y), 1))) * w
dev_null = float(2 * np.sum(np.where(y > 0, y * np.log(y / mu_null.clip(1e-10)), 0) - (y - mu_null)))

results = []
for col in feature_cols:
    try:
        X = df[[col]].values.astype(float)
        model = GeneralizedLinearRegressor(
            family="poisson", fit_intercept=True, max_iter=200
        )
        model.fit(X, y, sample_weight=w)
        mu = model.predict(X) * w
        dev = float(2 * np.sum(np.where(y > 0, y * np.log(y / mu.clip(1e-10)), 0) - (y - mu)))
        dev_red = dev_null - dev
        results.append({
            "feature": col,
            "deviance_null": round(dev_null, 2),
            "deviance_model": round(dev, 2),
            "deviance_reduction": round(dev_red, 2),
            "pct_reduction": round(100 * dev_red / dev_null, 4),
            "converged": True,
        })
    except Exception as e:
        results.append({
            "feature": col,
            "deviance_null": round(dev_null, 2),
            "deviance_model": None,
            "deviance_reduction": None,
            "pct_reduction": None,
            "converged": False,
        })

result_df = (
    pd.DataFrame(results)
    .sort_values("deviance_reduction", ascending=False)
    .reset_index(drop=True)
)

selected = result_df.head(top_n)
print(f"[univariate_selection] Tested {len(feature_cols)} features. "
      f"Top {top_n} selected. Best: '{selected.iloc[0]['feature']}' "
      f"({selected.iloc[0]['pct_reduction']:.3f}% deviance reduction).")
return selected
```

# ─────────────────────────────────────────────

# 4. INTERACTION SUGGESTION

# ─────────────────────────────────────────────

def suggest_interactions(
df: pd.DataFrame,
selected_features: pd.DataFrame,
target: str,
exposure: str,
top_n: int = 10,
max_features_to_combine: int = 15,
) -> pd.DataFrame:
“””
Suggest the most promising pairwise interaction terms.

```
For each pair (A, B) from your top selected features, fits:
    claims ~ A + B + A*B + offset(log(exposure))
and compares deviance to the additive model (A + B only).

Returns
-------
DataFrame of top interactions ranked by interaction_deviance_gain, with columns:
  feature_a | feature_b | dev_additive | dev_interaction | interaction_gain | pct_gain
"""
y = df[target].values
w = df[exposure].values

# Use top N features from selection
top_features = selected_features["feature"].head(max_features_to_combine).tolist()
pairs = list(combinations(top_features, 2))
print(f"[suggest_interactions] Testing {len(pairs)} pairs from top {len(top_features)} features...")

results = []
for feat_a, feat_b in pairs:
    try:
        a = df[feat_a].values.astype(float)
        b = df[feat_b].values.astype(float)
        ab = a * b  # interaction term

        X_add = np.column_stack([a, b])
        X_int = np.column_stack([a, b, ab])

        m_add = GeneralizedLinearRegressor(family="poisson", fit_intercept=True, max_iter=200)
        m_add.fit(X_add, y, sample_weight=w)
        mu_add = m_add.predict(X_add) * w
        dev_add = float(2 * np.sum(np.where(y > 0, y * np.log(y / mu_add.clip(1e-10)), 0) - (y - mu_add)))

        m_int = GeneralizedLinearRegressor(family="poisson", fit_intercept=True, max_iter=200)
        m_int.fit(X_int, y, sample_weight=w)
        mu_int = m_int.predict(X_int) * w
        dev_int = float(2 * np.sum(np.where(y > 0, y * np.log(y / mu_int.clip(1e-10)), 0) - (y - mu_int)))

        gain = dev_add - dev_int
        results.append({
            "feature_a": feat_a,
            "feature_b": feat_b,
            "dev_additive": round(dev_add, 2),
            "dev_interaction": round(dev_int, 2),
            "interaction_gain": round(gain, 2),
            "pct_gain": round(100 * gain / max(abs(dev_add), 1e-10), 4),
        })
    except Exception:
        pass

result_df = (
    pd.DataFrame(results)
    .sort_values("interaction_gain", ascending=False)
    .reset_index(drop=True)
)

top = result_df.head(top_n)
if len(top) > 0:
    print(f"[suggest_interactions] Top interaction: "
          f"'{top.iloc[0]['feature_a']}' × '{top.iloc[0]['feature_b']}' "
          f"(gain: {top.iloc[0]['pct_gain']:.4f}%)")
return top
```

# ─────────────────────────────────────────────

# 5. QUICK SUMMARY REPORT

# ─────────────────────────────────────────────

def modeling_summary(
audit_report: pd.DataFrame,
meta: dict,
selected: pd.DataFrame,
interactions: pd.DataFrame,
top_n_features: int = 10,
top_n_interactions: int = 5,
):
“””
Print a concise modeling summary to the notebook.
“””
print(”=” * 60)
print(”  POISSON FREQUENCY MODELING — PIPELINE SUMMARY”)
print(”=” * 60)

```
print(f"\n[1] DATA AUDIT")
actions = audit_report["suggested_action"].value_counts()
for action, count in actions.items():
    print(f"    {action:<35} {count} columns")

print(f"\n[2] PREPROCESSING")
print(f"    Features retained : {len(meta['all_features'])}")
print(f"    Columns dropped   : {len(meta['dropped_cols'])}")
print(f"    Log-transformed   : {len(meta['log_transformed'])}")
if meta["log_transformed"]:
    print(f"      → {meta['log_transformed']}")

print(f"\n[3] TOP {top_n_features} FEATURES (univariate deviance reduction)")
top_f = selected.head(top_n_features)[["feature", "pct_reduction"]]
for _, row in top_f.iterrows():
    bar = "█" * int((row["pct_reduction"] / top_f["pct_reduction"].max()) * 20)
    print(f"    {row['feature']:<35} {bar}  {row['pct_reduction']:.4f}%")

print(f"\n[4] TOP {top_n_interactions} INTERACTIONS")
for _, row in interactions.head(top_n_interactions).iterrows():
    print(f"    {row['feature_a']} × {row['feature_b']:<30} gain: {row['pct_gain']:.4f}%")

print("\n" + "=" * 60)
```
