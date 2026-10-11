import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42

print("=== Running Gating on W=2.0x for 43 Features ===")
train = pd.read_csv('train.csv')
y_train = train['label'].values
test = pd.read_csv('test.csv')

cache = np.load(r'C:\Users\Reynaldi Hutagaol\.gemini\antigravity-ide\brain\489beaa2-e2ba-4370-a4e6-0110f42de593\scratch\features_cache.npz')
X_base_tr = cache['X_train'] # 42 features
X_base_te = cache['X_test']

extra = np.load('scratch/extra_features.npz')
city_tr = extra['city_tr'] # 1 feature
city_te = extra['city_te']

X_tr_43 = np.hstack([X_base_tr, city_tr])
X_te_43 = np.hstack([X_base_te, city_te])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sample_weights = np.where(y_train == 0, 2.0, 1.0) # Official W=2.0x

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_43)
X_te_sc = scaler.transform(X_te_43)

xgb_oof = np.zeros(len(y_train))
hgb_oof = np.zeros(len(y_train))
et_oof  = np.zeros(len(y_train))
lr_oof  = np.zeros(len(y_train))

xgb_te = np.zeros(len(X_te_43))
hgb_te = np.zeros(len(X_te_43))
et_te  = np.zeros(len(X_te_43))
lr_te  = np.zeros(len(X_te_43))

for trn, val in skf.split(X_tr_43, y_train):
    clf_xgb = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.035, subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss', n_jobs=4)
    clf_xgb.fit(X_tr_43[trn], y_train[trn], sample_weight=sample_weights[trn])
    xgb_oof[val] = clf_xgb.predict_proba(X_tr_43[val])[:, 1]
    xgb_te += clf_xgb.predict_proba(X_te_43)[:, 1] / 5.0

    clf_hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.035, max_depth=6, random_state=SEED)
    clf_hgb.fit(X_tr_43[trn], y_train[trn], sample_weight=sample_weights[trn])
    hgb_oof[val] = clf_hgb.predict_proba(X_tr_43[val])[:, 1]
    hgb_te += clf_hgb.predict_proba(X_te_43)[:, 1] / 5.0

    clf_et = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=SEED, n_jobs=4)
    clf_et.fit(X_tr_43[trn], y_train[trn], sample_weight=sample_weights[trn])
    et_oof[val] = clf_et.predict_proba(X_tr_43[val])[:, 1]
    et_te += clf_et.predict_proba(X_te_43)[:, 1] / 5.0

    clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf_lr.fit(X_tr_sc[trn], y_train[trn], sample_weight=sample_weights[trn])
    lr_oof[val] = clf_lr.predict_proba(X_tr_sc[val])[:, 1]
    lr_te += clf_lr.predict_proba(X_te_sc)[:, 1] / 5.0

meta_X = np.column_stack([xgb_oof, hgb_oof, et_oof, lr_oof])
meta_X_te = np.column_stack([xgb_te, hgb_te, et_te, lr_te])

meta_oof = np.zeros(len(y_train))
meta_test = np.zeros(len(X_te_43))
fold_ths = []

for trn, val in skf.split(meta_X, y_train):
    meta_clf = LogisticRegression(C=1.0, random_state=SEED)
    meta_clf.fit(meta_X[trn], y_train[trn])
    val_probs = meta_clf.predict_proba(meta_X[val])[:, 1]
    meta_oof[val] = val_probs
    meta_test += meta_clf.predict_proba(meta_X_te)[:, 1] / 5.0

    f_th, f_f1 = 0.810, 0.0
    for th in np.arange(0.750, 0.835, 0.005):
        sc = f1_score(y_train[val], (val_probs >= th).astype(int), average='macro')
        if sc > f_f1:
            f_f1 = sc
            f_th = th
    fold_ths.append(round(float(f_th), 3))

eval_th = round(float(np.median(fold_ths)), 3)
eval_preds = (meta_oof >= eval_th).astype(int)
eval_macro_f1 = f1_score(y_train, eval_preds, average='macro')

# Construct row_conflict from X_base_tr
loc_conf = X_base_tr[:, 18]
fig_conf = X_base_tr[:, 21]
inst_mis = X_base_tr[:, 22]
neg_inv  = X_base_tr[:, 30]
ant_conf = X_base_tr[:, 34]
row_conflict = ((loc_conf == 1) | (fig_conf == 1) | (inst_mis == 1) | (neg_inv == 1) | (ant_conf == 1))

best_th_clean, best_th_conf, best_gate_f1 = 0.5, 0.5, 0.0
for th_cl in np.arange(0.65, 0.80, 0.005):
    for th_cf in np.arange(0.75, 0.90, 0.005):
        preds = np.where(row_conflict, (meta_oof >= th_cf).astype(int), (meta_oof >= th_cl).astype(int))
        sc = f1_score(y_train, preds, average='macro')
        if sc > best_gate_f1:
            best_gate_f1, best_th_clean, best_th_conf = sc, th_cl, th_cf

print(f"Single-Threshold OOF (W=2.0x, 43 Fitur): Th={eval_th:.3f} | Macro F1 = {eval_macro_f1:.4f}")
print(f"Dual-Threshold Gating OOF (W=2.0x)     : Th_clean={best_th_clean:.3f}, Th_conf={best_th_conf:.3f} | Macro F1 = {best_gate_f1:.4f}")
print(f"Delta F1 (Gating vs Single)            : {best_gate_f1 - eval_macro_f1:+.4f}")

# Extract row_conflict_test
loc_conf_te = X_base_te[:, 18]
fig_conf_te = X_base_te[:, 21]
inst_mis_te = X_base_te[:, 22]
neg_inv_te  = X_base_te[:, 30]
ant_conf_te = X_base_te[:, 34]
row_conflict_test = ((loc_conf_te == 1) | (fig_conf_te == 1) | (inst_mis_te == 1) | (neg_inv_te == 1) | (ant_conf_te == 1))

test_preds = np.where(row_conflict_test, (meta_test >= best_th_conf).astype(int), (meta_test >= best_th_clean).astype(int))
sub = pd.DataFrame({'id': test['id'], 'label': test_preds})
vc = sub['label'].value_counts()
print("\nDistribusi Test Set (W=2.0x Gating):")
print(vc)
print(f"Persentase Kelas 0: {vc.get(0, 0)/len(sub)*100:.2f}% ({vc.get(0, 0)} baris)")

np.savez('scratch/gating_w2_results.npz',
         meta_oof=meta_oof, meta_test=meta_test,
         eval_th=eval_th, eval_macro_f1=eval_macro_f1,
         best_th_clean=best_th_clean, best_th_conf=best_th_conf,
         best_gate_f1=best_gate_f1)
print("Saved to scratch/gating_w2_results.npz")
