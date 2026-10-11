import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Imports
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb

# 2. Base variables
if 'SEED' not in globals():
    SEED = 42
if 'train' not in globals():
    train = pd.read_csv('train.csv')
if 'test' not in globals():
    test = pd.read_csv('test.csv')
if 'y_train' not in globals():
    y_train = train['label'].values
if 'skf' not in globals():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
if 'sample_weights' not in globals():
    sample_weights = np.where(y_train == 0, 2.0, 1.0)

# 3. Arrays
if os.path.exists('scratch/full_rebuild_arrays.npz'):
    _arrs = np.load('scratch/full_rebuild_arrays.npz')
    X_train_46 = _arrs['X_train_46']
    X_test_46  = _arrs['X_test_46']

_scaler = StandardScaler()
X_train_scaled = _scaler.fit_transform(X_train_46)
X_test_scaled  = _scaler.transform(X_test_46)

# 4. Feature names
feature_names = [
    'bm25_score', 'tfidf_sim', 'best_win_tfidf_cos', 'best_win_jaccard',
    'lead_cov', 'max_sent_cov', 'spread_ratio', 'needed_sents',
    'title_len_words', 'content_len_words', 'len_ratio',
    'cos_lsa_c', 'cos_lsa_fs',
    'title_neg_win_pos', 'title_pos_win_neg',
    'num_missing_subst', 'num_cov_subst',
    'cap_ent_overlap', 'loc_conflict', 'has_loc_title', 'loc_coverage',
    'fig_conflict', 'inst_mismatch',
    'fasttext_title_content_sim', 'fasttext_title_lead_sim', 'fasttext_title_best_win_sim',
    'tokoh_conflict', 'organisasi_conflict',
    'tokoh_missing_count', 'organisasi_missing_count',
    'neg_inversion', 'ada_conflict', 'num_missing_exact',
    'num_cov_exact', 'antonym_conflict',
    'lead_keyword_mismatch', 'title_uncovered_keywords_ratio',
    'max_token_idf_in_title', 'mean_token_idf_in_title',
    'min_ngram_jaccard', 'title_word_density_in_lead',
    'title_token_coverage_ratio', 'city_lead_conflict',
    'twin_has_match', 'twin_title_jaccard', 'twin_conflict'
]

# Hyperparameter terbaik hasil 100 trials Optuna di atas X_train_46
study_xgb_best = {"n_estimators": 550, "max_depth": 3, "learning_rate": 0.016014316489551022, "subsample": 0.6, "colsample_bytree": 0.6, "min_child_weight": 4, "reg_lambda": 0.12569828566858568}
study_hgb_best = {"max_iter": 450, "learning_rate": 0.04314623120358049, "max_depth": 5, "l2_regularization": 0.10211938143560412, "max_leaf_nodes": 48}
study_et_best  = {"n_estimators": 200, "max_depth": 18, "min_samples_split": 8, "min_samples_leaf": 2, "max_features": 0.6}
study_lr_best  = {"C": 4.295400261870456, "penalty": "l1"}

print("Hyperparameter Terbaik Hasil HPO di atas X_train_46:")
print("  XGBoost      :", study_xgb_best)
print("  HistGBM      :", study_hgb_best)
print("  ExtraTrees   :", study_et_best)
print("  LogisticReg  :", study_lr_best)

loc_conf = X_train_46[:, feature_names.index('loc_conflict')]
fig_conf = X_train_46[:, feature_names.index('fig_conflict')]
inst_mis = X_train_46[:, feature_names.index('inst_mismatch')]
neg_inv  = X_train_46[:, feature_names.index('neg_inversion')]
ant_conf = X_train_46[:, feature_names.index('antonym_conflict')]
row_conflict = ((loc_conf == 1) | (fig_conf == 1) | (inst_mis == 1) | (neg_inv == 1) | (ant_conf == 1))

print("Feature conflicts computed successfully, count:", row_conflict.sum())
print("ALL PRE-CHECKS PASSED!")
