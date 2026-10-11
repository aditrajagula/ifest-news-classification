import sys
import os
import json
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

print("=" * 85)
print("UJI KOMBINASI FITUR SEKUNDER (NUMBER ATTACHMENT & MODAL EXPANSION)")
print("=" * 85)

train = pd.read_csv('train.csv')
y_train = train['label'].values
test = pd.read_csv('test.csv')

# Load base 43 features + Content-Twin features
cache = np.load(r'C:\Users\Reynaldi Hutagaol\.gemini\antigravity-ide\brain\489beaa2-e2ba-4370-a4e6-0110f42de593\scratch\features_cache.npz')
X_base_tr = cache['X_train']
X_base_te = cache['X_test']

extra = np.load('scratch/extra_features.npz')
city_tr = extra['city_tr']
city_te = extra['city_te']

twin_data = np.load('scratch/content_twin_features.npz')
twin_tr = twin_data['twin_feats_tr'] # (14397, 3)
twin_te = twin_data['twin_feats_te'] # (3603, 3)

X_tr_46 = np.hstack([X_base_tr, city_tr, twin_tr])
X_te_46 = np.hstack([X_base_te, city_te, twin_te])

# Load HPO parameters
with open('scratch/optuna_results.json', 'r') as f:
    opt_res = json.load(f)
hpo_best = opt_res['best_params']

xgb_params = dict(hpo_best['xgb'], random_state=SEED, eval_metric='logloss', n_jobs=-1)
hgb_params = dict(hpo_best['hgb'], random_state=SEED)
et_params  = dict(hpo_best['et'], random_state=SEED, n_jobs=-1)
lr_params  = {'C': hpo_best['lr']['C'], 'penalty': hpo_best['lr']['penalty'],
              'solver': 'liblinear' if hpo_best['lr']['penalty'] == 'l1' else 'lbfgs',
              'max_iter': 1000, 'random_state': SEED}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sample_weights = np.where(y_train == 0, 2.0, 1.0)

# Evaluate function
def eval_stacking(X_tr, X_te, name):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    xgb_oof = np.zeros(len(y_train))
    hgb_oof = np.zeros(len(y_train))
    et_oof  = np.zeros(len(y_train))
    lr_oof  = np.zeros(len(y_train))
    
    xgb_te  = np.zeros(len(X_te))
    hgb_te  = np.zeros(len(X_te))
    et_te   = np.zeros(len(X_te))
    lr_te   = np.zeros(len(X_te))

    for trn, val in skf.split(X_tr, y_train):
        clf_x = xgb.XGBClassifier(**xgb_params)
        clf_x.fit(X_tr[trn], y_train[trn], sample_weight=sample_weights[trn])
        xgb_oof[val] = clf_x.predict_proba(X_tr[val])[:, 1]
        xgb_te += clf_x.predict_proba(X_te)[:, 1] / 5.0
        
        clf_h = HistGradientBoostingClassifier(**hgb_params)
        clf_h.fit(X_tr[trn], y_train[trn], sample_weight=sample_weights[trn])
        hgb_oof[val] = clf_h.predict_proba(X_tr[val])[:, 1]
        hgb_te += clf_h.predict_proba(X_te)[:, 1] / 5.0
        
        clf_e = ExtraTreesClassifier(**et_params)
        clf_e.fit(X_tr[trn], y_train[trn], sample_weight=sample_weights[trn])
        et_oof[val] = clf_e.predict_proba(X_tr[val])[:, 1]
        et_te += clf_e.predict_proba(X_te)[:, 1] / 5.0
        
        clf_l = LogisticRegression(**lr_params)
        clf_l.fit(X_tr_sc[trn], y_train[trn], sample_weight=sample_weights[trn])
        lr_oof[val] = clf_l.predict_proba(X_tr_sc[val])[:, 1]
        lr_te += clf_l.predict_proba(X_te_sc)[:, 1] / 5.0

    meta_X = np.column_stack([xgb_oof, hgb_oof, et_oof, lr_oof])
    meta_X_te = np.column_stack([xgb_te, hgb_te, et_te, lr_te])
    
    meta_oof = np.zeros(len(y_train))
    meta_test = np.zeros(len(X_te))
    fold_ths = []
    
    for trn, val in skf.split(meta_X, y_train):
        meta_clf = LogisticRegression(C=1.0, random_state=SEED)
        meta_clf.fit(meta_X[trn], y_train[trn])
        val_probs = meta_clf.predict_proba(meta_X[val])[:, 1]
        meta_oof[val] = val_probs
        meta_test += meta_clf.predict_proba(meta_X_te)[:, 1] / 5.0
        
        f_th, f_f1 = 0.810, 0.0
        for th in np.arange(0.65, 0.85, 0.005):
            sc = f1_score(y_train[val], (val_probs >= th).astype(int), average='macro')
            if sc > f_f1:
                f_f1 = sc
                f_th = th
        fold_ths.append(round(float(f_th), 3))

    cons_th = round(float(np.median(fold_ths)), 3)
    single_preds = (meta_oof >= cons_th).astype(int)
    single_f1 = f1_score(y_train, single_preds, average='macro')
    cr = classification_report(y_train, single_preds, output_dict=True)

    # Precision Gating
    loc_conf = X_base_tr[:, 18]
    fig_conf = X_base_tr[:, 21]
    inst_mis = X_base_tr[:, 22]
    neg_inv  = X_base_tr[:, 30]
    ant_conf = X_base_tr[:, 34]
    row_conflict = ((loc_conf == 1) | (fig_conf == 1) | (inst_mis == 1) | (neg_inv == 1) | (ant_conf == 1))
    
    best_cl, best_cf, best_gated_f1 = 0.5, 0.5, 0.0
    for th_cl in np.arange(0.60, 0.78, 0.005):
        for th_cf in np.arange(0.70, 0.88, 0.005):
            preds = np.where(row_conflict, (meta_oof >= th_cf).astype(int), (meta_oof >= th_cl).astype(int))
            sc = f1_score(y_train, preds, average='macro')
            if sc > best_gated_f1:
                best_gated_f1, best_cl, best_cf = sc, th_cl, th_cf

    print(f"[{name}]")
    print(f"  Single Th = {cons_th:.3f} | Macro F1 = {single_f1:.4f} | Rec C0 = {cr['0']['recall']:.4f} | Prec C0 = {cr['0']['precision']:.4f}")
    print(f"  Dual-Gating: Th_clean={best_cl:.3f}, Th_conf={best_cf:.3f} | Macro F1 = {best_gated_f1:.4f}")
    
    return {
        'name': name,
        'dim': X_tr.shape[1],
        'single_th': cons_th,
        'single_f1': single_f1,
        'gated_f1': best_gated_f1,
        'th_clean': best_cl,
        'th_conf': best_cf,
        'c0_rec': cr['0']['recall'],
        'c0_prec': cr['0']['precision'],
        'meta_oof': meta_oof,
        'meta_test': meta_test
    }

results = []

# 1. Baseline + Content-Twin (46 Fitur)
r_twin = eval_stacking(X_tr_46, X_te_46, "1. Baseline 43 + Content-Twin (46 Fitur)")
results.append(r_twin)

# Save final 46 features as optimal
np.savez('scratch/final_46_features.npz',
         X_tr_46=X_tr_46, X_te_46=X_te_46,
         meta_oof=r_twin['meta_oof'],
         meta_test=r_twin['meta_test'],
         single_th=r_twin['single_th'],
         single_f1=r_twin['single_f1'],
         th_clean=r_twin['th_clean'],
         th_conf=r_twin['th_conf'],
         gated_f1=r_twin['gated_f1'])

print("\nSaved final 46 features to scratch/final_46_features.npz")
