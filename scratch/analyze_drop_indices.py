import sys
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42

print("=== Loading X_train_raw_48.npz ===")
data = np.load('scratch/X_train_raw_48.npz')
X_raw = data['X_train_raw']
y = data['y_train']

raw_feats = [
    'cos_wc', 'cos_wfs', 'cos_cc', 'cos_cfs',
    'bm25_c', 'bm25_fs',
    'cos_lsa_c', 'cos_lsa_fs',
    'best_win_tfidf_cos',
    'best_50_cov', 'best_win_pos_norm', 'best_minus_lead',
    'lead_25_cov', 'lead_50_cov', 'lead_80_cov', 'top1_top2_contrast',
    'loc_mismatch', 'loc_conflict', 'loc_c_conflict',
    'fig_mismatch', 'fig_conflict',
    'inst_mismatch',
    'tok_cov_exact', 'tok_cov_stem', 'stem_gain', 'head_cov_c',
    'head_entity_absence', 'num_unit_match_ratio',
    'neg_inversion', 'lead_neg_asymmetry', 'ada_conflict', 'antonym_conflict',
    'refuting_score', 'ordered_bigram_ratio',
    'num_missing_subst', 'num_cov_subst',
    'cap_cov_c', 'cap_cov_fs', 'len_ratio', 'len_t_words', 'len_c_words',
    'v_conf', 'e_conf', 'comp_conf', 'dir_conf',
    'max_unm_idf', 'sum_unm_ratio', 'max_sent_cos'
]

drop_indices = [4, 5, 7, 11, 25, 26, 27, 31, 34, 35, 38, 39, 43, 46, 47]
is_dropped = [i in drop_indices for i in range(len(raw_feats))]

print(f"X_raw shape: {X_raw.shape}")

# 1. Correlations with y_train
corrs = []
for i in range(X_raw.shape[1]):
    col = X_raw[:, i]
    if np.std(col) < 1e-9:
        corrs.append(0.0)
    else:
        r = np.corrcoef(col, y)[0, 1]
        corrs.append(float(r))

# 2. XGBoost Feature Importance
print("Training XGBoost on 48 raw features...")
sample_weights = np.where(y == 0, 3.5, 1.0)
clf = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.035,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss',
    n_jobs=4
)
clf.fit(X_raw, y, sample_weight=sample_weights)
fi = clf.feature_importances_

# 3. Collinearity: find max correlation with another feature
corr_matrix = np.corrcoef(X_raw.T)
max_other_corr = []
for i in range(X_raw.shape[1]):
    row = np.abs(corr_matrix[i])
    row[i] = 0.0 # ignore self
    max_other_corr.append(float(np.max(row)))

df = pd.DataFrame({
    'idx': range(len(raw_feats)),
    'feature': raw_feats,
    'dropped': is_dropped,
    'corr_y': corrs,
    'abs_corr_y': np.abs(corrs),
    'importance': fi,
    'max_collinear': max_other_corr
})

print("\n=== 15 DROPPED FEATURES ANALYSIS ===")
df_dropped = df[df['dropped']].sort_values('importance', ascending=False)
print(df_dropped.to_string(index=False))

print("\n=== 33 KEPT FEATURES TOP 15 BY IMPORTANCE ===")
df_kept = df[~df['dropped']].sort_values('importance', ascending=False).head(15)
print(df_kept.to_string(index=False))

print("\n=== SUMMARY COMPARISON ===")
print(f"Dropped features mean importance: {df[df['dropped']]['importance'].mean():.4f}")
print(f"Kept features mean importance   : {df[~df['dropped']]['importance'].mean():.4f}")
print(f"Dropped features mean abs corr  : {df[df['dropped']]['abs_corr_y'].mean():.4f}")
print(f"Kept features mean abs corr    : {df[~df['dropped']]['abs_corr_y'].mean():.4f}")
print(f"Dropped features mean max coll  : {df[df['dropped']]['max_collinear'].mean():.4f}")
print(f"Kept features mean max coll    : {df[~df['dropped']]['max_collinear'].mean():.4f}")
