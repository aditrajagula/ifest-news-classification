import os
import json
import re
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train['label'].values

data = np.load('features_44_clean.npz')
X_ref = data['X_train']
X_test_ref = data['X_test']

# Load original
nb = json.load(open('ifest_c562eb8.ipynb', encoding='utf-8'))
exec(''.join(nb['cells'][5]['source']))
exec(''.join(nb['cells'][7]['source']))
exec(''.join(nb['cells'][9]['source']))

# X_train is original 45 features, X_test is original 45 features
# Drop head_entity_absence (col 26) -> 44 features
X_base_tr = np.delete(X_train, 26, axis=1)
X_base_te = np.delete(X_test, 26, axis=1)

# comp_conf is col 42 in 44-features matrix
# In X_ref, comp_conf incorporates the nationwide PROVINCE_OF hierarchy
X_opt_tr = X_base_tr.copy()
X_opt_te = X_base_te.copy()
X_opt_tr[:, 42] = X_ref[:, 42]
X_opt_te[:, 42] = X_test_ref[:, 42]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sample_weights = np.ones(len(train))

scaler = StandardScaler()
X_scaled_tr = scaler.fit_transform(X_opt_tr)
X_scaled_te = scaler.transform(X_opt_te)

xgb_oof = np.zeros(len(train))
hgb_oof = np.zeros(len(train))
et_oof = np.zeros(len(train))
lr_oof = np.zeros(len(train))

xgb_te = np.zeros(len(test))
hgb_te = np.zeros(len(test))
et_te = np.zeros(len(test))
lr_te = np.zeros(len(test))

print("Training 5-Fold Ensemble on Optimal Refined Features (44 dimensions)...")
for fold, (trn, val) in enumerate(skf.split(X_opt_tr, y_train)):
    clf_xgb = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.035, subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss')
    clf_xgb.fit(X_opt_tr[trn], y_train[trn], sample_weight=sample_weights[trn])
    xgb_oof[val] = clf_xgb.predict_proba(X_opt_tr[val])[:, 1]
    xgb_te += clf_xgb.predict_proba(X_opt_te)[:, 1] / 5.0
    
    clf_hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.035, max_depth=6, random_state=SEED)
    clf_hgb.fit(X_opt_tr[trn], y_train[trn], sample_weight=sample_weights[trn])
    hgb_oof[val] = clf_hgb.predict_proba(X_opt_tr[val])[:, 1]
    hgb_te += clf_hgb.predict_proba(X_opt_te)[:, 1] / 5.0
    
    clf_et = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=SEED, n_jobs=-1)
    clf_et.fit(X_opt_tr[trn], y_train[trn], sample_weight=sample_weights[trn])
    et_oof[val] = clf_et.predict_proba(X_opt_tr[val])[:, 1]
    et_te += clf_et.predict_proba(X_opt_te)[:, 1] / 5.0
    
    clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf_lr.fit(X_scaled_tr[trn], y_train[trn], sample_weight=sample_weights[trn])
    lr_oof[val] = clf_lr.predict_proba(X_scaled_tr[val])[:, 1]
    lr_te += clf_lr.predict_proba(X_scaled_te)[:, 1] / 5.0

meta_X = np.column_stack([xgb_oof, hgb_oof, et_oof, lr_oof])
meta_X_te = np.column_stack([xgb_te, hgb_te, et_te, lr_te])

meta_oof = np.zeros(len(train))
meta_te = np.zeros(len(test))

for trn, val in skf.split(meta_X, y_train):
    meta_clf = LogisticRegression(C=1.0, random_state=SEED)
    meta_clf.fit(meta_X[trn], y_train[trn])
    meta_oof[val] = meta_clf.predict_proba(meta_X[val])[:, 1]
    meta_te += meta_clf.predict_proba(meta_X_te)[:, 1] / 5.0

print("\n--- Evaluasi Threshold Konsensus (0.780 - 0.830) ---")
best_th, best_f1 = 0.810, 0.0
for th in np.arange(0.780, 0.835, 0.005):
    preds = (meta_oof >= th).astype(int)
    sc = f1_score(y_train, preds, average='macro')
    cm = confusion_matrix(y_train, preds)
    t_c0 = ((meta_te >= th).astype(int) == 0).sum()
    rec_c0 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    prec_c0 = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print(f"Th: {th:.3f} | F1: {sc:.4f} | FP: {cm[1, 0]:3d} | Rec C0: {rec_c0*100:.2f}% | Prec C0: {prec_c0*100:.2f}% | Test C0: {t_c0:3d}")
    if sc > best_f1:
        best_f1, best_th = sc, th

print(f"\n[OPTIMAL REFINED BASELINE] Best Th: {best_th:.3f} | Macro F1: {best_f1:.4f}")
oof_preds = (meta_oof >= best_th).astype(int)
cm_opt = confusion_matrix(y_train, oof_preds)
print("Confusion Matrix:")
print(cm_opt)
print(f"False Positives: {cm_opt[1, 0]} (Berkurang dari baseline 445: {445 - cm_opt[1, 0]} kasus, dari 491: {491 - cm_opt[1, 0]} kasus)")

# Generate submission_baseline_refined.csv
test_preds = (meta_te >= best_th).astype(int)
c0_count = (test_preds == 0).sum()

# If outside 310-350, find consensus threshold
chosen_th = best_th
if not (310 <= c0_count <= 350):
    best_prior_th = chosen_th
    best_prior_f1 = 0.0
    for th_cand in np.arange(0.780, 0.840, 0.002):
        t_c0 = ((meta_te >= th_cand).astype(int) == 0).sum()
        if 310 <= t_c0 <= 350:
            sc_cand = f1_score(y_train, (meta_oof >= th_cand).astype(int), average='macro')
            if sc_cand > best_prior_f1:
                best_prior_f1 = sc_cand
                best_prior_th = th_cand
    if best_prior_f1 > 0:
        print(f"Threshold konsensus prior dipilih: {best_prior_th:.3f} (OOF F1: {best_prior_f1:.4f})")
        chosen_th = best_prior_th
        test_preds = (meta_te >= chosen_th).astype(int)
        c0_count = (test_preds == 0).sum()

sub_file = 'submission_baseline_refined.csv'
sub = pd.DataFrame({'id': test['id'], 'label': test_preds})
sub.to_csv(sub_file, index=False)

c1_count = (sub['label'] == 1).sum()
print(f"\nDistribusi Prediksi {sub_file}:")
print(f"  Kelas 1: {c1_count} ({c1_count/len(sub)*100:.2f}%)")
print(f"  Kelas 0: {c0_count} ({c0_count/len(sub)*100:.2f}%)")

assert len(sub) == 3603, f"Jumlah baris salah: {len(sub)} != 3603"
assert list(sub.columns) == ['id', 'label'], f"Nama kolom salah: {sub.columns}"
assert sub.isnull().sum().sum() == 0, "Ditemukan nilai null pada submission!"
assert set(sub['label'].unique()).issubset({0, 1}), "Label bukan biner {0, 1}!"
print("\nSELURUH ASERSI INTEGRITAS RESMI TERPENUHI DENGAN SUKSES!")

# Simpan probabilitas OOF dan Test untuk stacking masa depan
np.savez('optimal_refined_oof_preds.npz', meta_oof=meta_oof, meta_te=meta_te, y_train=y_train, X_opt_tr=X_opt_tr, X_opt_te=X_opt_te)
