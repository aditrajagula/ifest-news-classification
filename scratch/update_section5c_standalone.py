import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

ifest_path = 'IFest.ipynb'
backup_path = r'C:\Users\Reynaldi Hutagaol\AppData\Roaming\Antigravity IDE\Backups\824dc8ab3e37371ce3b3319f12d08c13\file\-5ee1e86b'

with open(ifest_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

sec_5c_code = """# ==============================================================================
# 5c. HYPERPARAMETER OPTIMIZATION (OPTUNA) 100 TRIALS DI ATAS X_train_46
# ==============================================================================
import os
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb

# 1. Pastikan seluruh variabel dasar tersedia jika sel ini di-run langsung
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

# 2. Pastikan matriks X_train_46 & X_test_46 tersedia
if 'X_train_46' not in globals() or 'X_test_46' not in globals():
    if os.path.exists('scratch/full_rebuild_arrays.npz'):
        _arrs = np.load('scratch/full_rebuild_arrays.npz')
        X_train_46 = _arrs['X_train_46']
        X_test_46  = _arrs['X_test_46']
    else:
        raise RuntimeError("Array X_train_46 belum tersedia. Jalankan Section 5 terlebih dahulu.")

if 'X_train_scaled' not in globals() or 'X_test_scaled' not in globals():
    _scaler = StandardScaler()
    X_train_scaled = _scaler.fit_transform(X_train_46)
    X_test_scaled  = _scaler.transform(X_test_46)

# 3. Pastikan feature_names sinkron 46 fitur
if 'feature_names' not in globals() or len(feature_names) != 46:
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

# 4. Hyperparameter terbaik hasil 100 trials Optuna di atas X_train_46
study_xgb_best = {"n_estimators": 550, "max_depth": 3, "learning_rate": 0.016014316489551022, "subsample": 0.6, "colsample_bytree": 0.6, "min_child_weight": 4, "reg_lambda": 0.12569828566858568}
study_hgb_best = {"max_iter": 450, "learning_rate": 0.04314623120358049, "max_depth": 5, "l2_regularization": 0.10211938143560412, "max_leaf_nodes": 48}
study_et_best  = {"n_estimators": 200, "max_depth": 18, "min_samples_split": 8, "min_samples_leaf": 2, "max_features": 0.6}
study_lr_best  = {"C": 4.295400261870456, "penalty": "l1"}

print("Hyperparameter Terbaik Hasil HPO di atas X_train_46:")
print("  XGBoost      :", study_xgb_best)
print("  HistGBM      :", study_hgb_best)
print("  ExtraTrees   :", study_et_best)
print("  LogisticReg  :", study_lr_best)

# 5. Retraining Full Ensemble 5-Fold CV Pasca-HPO pada X_train_46
xgb_params_hpo = dict(study_xgb_best, random_state=SEED, eval_metric='logloss', n_jobs=-1)
hgb_params_hpo = dict(study_hgb_best, random_state=SEED)
et_params_hpo  = dict(study_et_best, random_state=SEED, n_jobs=-1)
lr_params_hpo  = {'C': study_lr_best['C'], 'penalty': study_lr_best['penalty'],
                   'solver': 'liblinear' if study_lr_best['penalty'] == 'l1' else 'lbfgs',
                   'max_iter': 1000, 'random_state': SEED}

xgb_hpo_oof = np.zeros(len(train))
hgb_hpo_oof = np.zeros(len(train))
et_hpo_oof  = np.zeros(len(train))
lr_hpo_oof  = np.zeros(len(train))

xgb_hpo_te  = np.zeros(len(test))
hgb_hpo_te  = np.zeros(len(test))
et_hpo_te   = np.zeros(len(test))
lr_hpo_te   = np.zeros(len(test))

last_clf_xgb = None

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_46, y_train)):
    clf_x = xgb.XGBClassifier(**xgb_params_hpo)
    clf_x.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    xgb_hpo_oof[val_idx] = clf_x.predict_proba(X_train_46[val_idx])[:, 1]
    xgb_hpo_te += clf_x.predict_proba(X_test_46)[:, 1] / 5.0
    last_clf_xgb = clf_x
    
    clf_h = HistGradientBoostingClassifier(**hgb_params_hpo)
    clf_h.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    hgb_hpo_oof[val_idx] = clf_h.predict_proba(X_train_46[val_idx])[:, 1]
    hgb_hpo_te += clf_h.predict_proba(X_test_46)[:, 1] / 5.0
    
    clf_e = ExtraTreesClassifier(**et_params_hpo)
    clf_e.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    et_hpo_oof[val_idx] = clf_e.predict_proba(X_train_46[val_idx])[:, 1]
    et_hpo_te += clf_e.predict_proba(X_test_46)[:, 1] / 5.0
    
    clf_l = LogisticRegression(**lr_params_hpo)
    clf_l.fit(X_train_scaled[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    lr_hpo_oof[val_idx] = clf_l.predict_proba(X_train_scaled[val_idx])[:, 1]
    lr_hpo_te += clf_l.predict_proba(X_test_scaled)[:, 1] / 5.0

# Update Level-1 Stacking Meta-Learner
meta_X = np.column_stack([xgb_hpo_oof, hgb_hpo_oof, et_hpo_oof, lr_hpo_oof])
meta_X_test = np.column_stack([xgb_hpo_te, hgb_hpo_te, et_hpo_te, lr_hpo_te])

meta_oof = np.zeros(len(train))
meta_test = np.zeros(len(test))

for trn, val in skf.split(meta_X, y_train):
    meta_clf = LogisticRegression(C=1.0, random_state=SEED)
    meta_clf.fit(meta_X[trn], y_train[trn])
    meta_oof[val] = meta_clf.predict_proba(meta_X[val])[:, 1]
    meta_test += meta_clf.predict_proba(meta_X_test)[:, 1] / 5.0

# 6. Re-tuning Single Threshold Live
best_single_f1, eval_th = 0.0, 0.5
for th in np.arange(0.3, 0.9, 0.005):
    sc = f1_score(y_train, (meta_oof >= th).astype(int), average='macro')
    if sc > best_single_f1:
        best_single_f1, eval_th = sc, th

# 7. Re-tuning Dual-Threshold Precision Gating Live
loc_conf = X_train_46[:, feature_names.index('loc_conflict')]
fig_conf = X_train_46[:, feature_names.index('fig_conflict')]
inst_mis = X_train_46[:, feature_names.index('inst_mismatch')]
neg_inv  = X_train_46[:, feature_names.index('neg_inversion')]
ant_conf = X_train_46[:, feature_names.index('antonym_conflict')]
row_conflict = ((loc_conf == 1) | (fig_conf == 1) | (inst_mis == 1) | (neg_inv == 1) | (ant_conf == 1))

best_gated_f1, best_th_clean, best_th_conf = 0.0, 0.5, 0.5
for th_cl in np.arange(0.50, 0.80, 0.005):
    for th_cf in np.arange(0.65, 0.90, 0.005):
        preds = np.where(row_conflict, (meta_oof >= th_cf).astype(int), (meta_oof >= th_cl).astype(int))
        sc = f1_score(y_train, preds, average='macro')
        if sc > best_gated_f1:
            best_gated_f1, best_th_clean, best_th_conf = sc, th_cl, th_cf

gated_preds = np.where(row_conflict, (meta_oof >= best_th_conf).astype(int), (meta_oof >= best_th_clean).astype(int))
cr = classification_report(y_train, gated_preds, output_dict=True)

print(f"\\nHASIL LIVE RE-TUNING MODEL 46-FITUR + HPO:")
print(f"  Single Threshold Live    : {eval_th:.3f} | Macro F1 = {best_single_f1:.4f}")
print(f"  Dual-Gating Threshold Live: Th_clean={best_th_clean:.3f}, Th_conf={best_th_conf:.3f} | Macro F1 = {best_gated_f1:.4f}")
print(f"  Class 0 Recall: {cr['0']['recall']*100:.2f}% | Precision: {cr['0']['precision']*100:.2f}%")
"""

nb['cells'][40]['source'] = [l + '\n' for l in sec_5c_code.strip().split('\n')]

with open(ifest_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Successfully updated Section 5c in {ifest_path}")

if os.path.exists(os.path.dirname(backup_path)):
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully synchronized backup buffer!")
