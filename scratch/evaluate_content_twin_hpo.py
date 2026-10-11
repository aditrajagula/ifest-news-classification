import sys
import os
import re
import unicodedata
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42

print("=" * 85)
print("EKSPERIMEN: CONTENT-TWIN CONFLICT, NUMBER ATTACHMENT, & MODAL EXPANSION")
print("=" * 85)

train = pd.read_csv('train.csv')
y_train = train['label'].values
test = pd.read_csv('test.csv')

# Load cached 42 base features + 1 city feature = 43 baseline features
cache = np.load(r'C:\Users\Reynaldi Hutagaol\.gemini\antigravity-ide\brain\489beaa2-e2ba-4370-a4e6-0110f42de593\scratch\features_cache.npz')
X_base_tr = cache['X_train'] # (14397, 42)
X_base_te = cache['X_test']  # (3603, 42)

extra = np.load('scratch/extra_features.npz')
city_tr = extra['city_tr'] # (14397, 1)
city_te = extra['city_te'] # (3603, 1)

X_tr_43 = np.hstack([X_base_tr, city_tr]) # (14397, 43)
X_te_43 = np.hstack([X_base_te, city_te]) # (3603, 43)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sample_weights = np.where(y_train == 0, 2.0, 1.0) # W=2.0x

# Load HPO best parameters from Tugas 1-4
with open('scratch/optuna_results.json', 'r') as f:
    opt_res = json.load(f)
hpo_best = opt_res['best_params']

xgb_params = dict(hpo_best['xgb'], random_state=SEED, eval_metric='logloss', n_jobs=-1)
hgb_params = dict(hpo_best['hgb'], random_state=SEED)
et_params  = dict(hpo_best['et'], random_state=SEED, n_jobs=-1)
lr_params  = {'C': hpo_best['lr']['C'], 'penalty': hpo_best['lr']['penalty'],
              'solver': 'liblinear' if hpo_best['lr']['penalty'] == 'l1' else 'lbfgs',
              'max_iter': 1000, 'random_state': SEED}

print(f"Loaded HPO parameters: XGB, HGB, ET, LR ready.")
print(f"X_tr_43 shape: {X_tr_43.shape}, X_te_43 shape: {X_te_43.shape}")

# ==============================================================================
# 1. EKSTRAKSI FITUR CONTENT-TWIN RETRIEVAL (TUGAS 1)
# ==============================================================================
print("\n--- 1. Ekstraksi Fitur Content-Twin (Leave-One-Out OOF Safe) ---")

def normalize_content(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize_title(t):
    t = unicodedata.normalize('NFKC', str(t)).lower()
    t = re.sub(r'[^\w\s]', '', t)
    return set(t.split())

def jaccard_titles(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

train['content_norm'] = train['content'].apply(normalize_content)
test['content_norm']  = test['content'].apply(normalize_content)
train['title_tokens'] = train['title'].apply(tokenize_title)
test['title_tokens']  = test['title'].apply(tokenize_title)

MIN_CONTENT_LEN = 200

# Build train label=1 dictionary: content_norm -> list of (row_idx, title_tokens)
c1_train = train[(train['label'] == 1) & (train['content_norm'].str.len() >= MIN_CONTENT_LEN)]
train_c1_dict = defaultdict(list)
for idx, row in c1_train.iterrows():
    train_c1_dict[row['content_norm']].append((idx, row['title_tokens']))

# Extract for train (LEAVE-ONE-OUT SAFE)
twin_has_match_tr = np.zeros(len(train))
twin_jaccard_tr   = np.zeros(len(train))

for idx, row in train.iterrows():
    c_norm = row['content_norm']
    if len(c_norm) < MIN_CONTENT_LEN:
        continue
    entries = train_c1_dict.get(c_norm, [])
    # Leave-one-out: filter out own row idx
    ref_token_sets = [tok_set for r_idx, tok_set in entries if r_idx != idx]
    if not ref_token_sets:
        continue
    own_tokens = row['title_tokens']
    best_sim = max(jaccard_titles(own_tokens, ref_tokens) for ref_tokens in ref_token_sets)
    twin_has_match_tr[idx] = 1.0
    twin_jaccard_tr[idx]   = best_sim

# Extract for test (BUILT FROM FULL TRAIN C1 DICTIONARY, RULE #8 ZERO-LEAKAGE COMPLIANT)
twin_has_match_te = np.zeros(len(test))
twin_jaccard_te   = np.zeros(len(test))

for idx, row in test.iterrows():
    c_norm = row['content_norm']
    if len(c_norm) < MIN_CONTENT_LEN:
        continue
    entries = train_c1_dict.get(c_norm, [])
    if not entries:
        continue
    own_tokens = row['title_tokens']
    best_sim = max(jaccard_titles(own_tokens, ref_tokens) for _, ref_tokens in entries)
    twin_has_match_te[idx] = 1.0
    twin_jaccard_te[idx]   = best_sim

print(f"Train matches: {int(twin_has_match_tr.sum())} / {len(train)} (C0: {int(twin_has_match_tr[y_train==0].sum())}, C1: {int(twin_has_match_tr[y_train==1].sum())})")
print(f"Test matches : {int(twin_has_match_te.sum())} / {len(test)} ({twin_has_match_te.mean()*100:.2f}%)")

# ==============================================================================
# 2. TUNING TWIN_JACCARD_THRESHOLD VIA OOF
# ==============================================================================
print("\n--- 2. Sweep TWIN_JACCARD_THRESHOLD via 5-Fold OOF Stacking ---")

# We test threshold focused around the bimodal separation zone
threshold_candidates = [0.70, 0.75, 0.80, 0.85, 0.90]
sweep_results = []

def run_stacking_pipeline(X_tr_feats, X_te_feats):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_feats)
    X_te_sc = scaler.transform(X_te_feats)

    xgb_oof = np.zeros(len(y_train))
    hgb_oof = np.zeros(len(y_train))
    et_oof  = np.zeros(len(y_train))
    lr_oof  = np.zeros(len(y_train))
    
    xgb_te  = np.zeros(len(X_te_feats))
    hgb_te  = np.zeros(len(X_te_feats))
    et_te   = np.zeros(len(X_te_feats))
    lr_te   = np.zeros(len(X_te_feats))

    for trn, val in skf.split(X_tr_feats, y_train):
        # XGB
        clf_x = xgb.XGBClassifier(**xgb_params)
        clf_x.fit(X_tr_feats[trn], y_train[trn], sample_weight=sample_weights[trn])
        xgb_oof[val] = clf_x.predict_proba(X_tr_feats[val])[:, 1]
        xgb_te += clf_x.predict_proba(X_te_feats)[:, 1] / 5.0
        
        # HGB
        clf_h = HistGradientBoostingClassifier(**hgb_params)
        clf_h.fit(X_tr_feats[trn], y_train[trn], sample_weight=sample_weights[trn])
        hgb_oof[val] = clf_h.predict_proba(X_tr_feats[val])[:, 1]
        hgb_te += clf_h.predict_proba(X_te_feats)[:, 1] / 5.0
        
        # ET
        clf_e = ExtraTreesClassifier(**et_params)
        clf_e.fit(X_tr_feats[trn], y_train[trn], sample_weight=sample_weights[trn])
        et_oof[val] = clf_e.predict_proba(X_tr_feats[val])[:, 1]
        et_te += clf_e.predict_proba(X_te_feats)[:, 1] / 5.0
        
        # LR
        clf_l = LogisticRegression(**lr_params)
        clf_l.fit(X_tr_sc[trn], y_train[trn], sample_weight=sample_weights[trn])
        lr_oof[val] = clf_l.predict_proba(X_tr_sc[val])[:, 1]
        lr_te += clf_l.predict_proba(X_te_sc)[:, 1] / 5.0

    meta_X = np.column_stack([xgb_oof, hgb_oof, et_oof, lr_oof])
    meta_X_te = np.column_stack([xgb_te, hgb_te, et_te, lr_te])
    
    meta_oof = np.zeros(len(y_train))
    meta_test = np.zeros(len(X_te_feats))
    fold_ths = []
    
    for trn, val in skf.split(meta_X, y_train):
        meta_clf = LogisticRegression(C=1.0, random_state=SEED)
        meta_clf.fit(meta_X[trn], y_train[trn])
        val_probs = meta_clf.predict_proba(meta_X[val])[:, 1]
        meta_oof[val] = val_probs
        meta_test += meta_clf.predict_proba(meta_X_te)[:, 1] / 5.0
        
        f_th, f_f1 = 0.810, 0.0
        for th in np.arange(0.70, 0.88, 0.005):
            sc = f1_score(y_train[val], (val_probs >= th).astype(int), average='macro')
            if sc > f_f1:
                f_f1 = sc
                f_th = th
        fold_ths.append(round(float(f_th), 3))

    cons_th = round(float(np.median(fold_ths)), 3)
    single_preds = (meta_oof >= cons_th).astype(int)
    single_f1 = f1_score(y_train, single_preds, average='macro')
    cr = classification_report(y_train, single_preds, output_dict=True)

    return {
        'cons_th': cons_th,
        'single_f1': single_f1,
        'c0_rec': cr['0']['recall'],
        'c0_prec': cr['0']['precision'],
        'c0_f1': cr['0']['f1-score'],
        'meta_oof': meta_oof,
        'meta_test': meta_test
    }

# Sweep TWIN_JACCARD_THRESHOLD
print("\nMenjalankan Sweep TWIN_JACCARD_THRESHOLD...")
best_twin_th = 0.80
best_twin_single_f1 = 0.0
best_twin_res = None
best_twin_tr_features = None
best_twin_te_features = None

for th_jaccard in threshold_candidates:
    twin_conflict_tr = np.where((twin_has_match_tr == 1.0) & (twin_jaccard_tr < th_jaccard), 1.0, 0.0)
    twin_conflict_te = np.where((twin_has_match_te == 1.0) & (twin_jaccard_te < th_jaccard), 1.0, 0.0)
    
    # 3 fitur Content-Twin: has_match, title_jaccard, twin_conflict
    twin_feats_tr = np.column_stack([twin_has_match_tr, twin_jaccard_tr, twin_conflict_tr])
    twin_feats_te = np.column_stack([twin_has_match_te, twin_jaccard_te, twin_conflict_te])
    
    X_tr_46 = np.hstack([X_tr_43, twin_feats_tr])
    X_te_46 = np.hstack([X_te_43, twin_feats_te])
    
    t0 = time.perf_counter()
    res = run_stacking_pipeline(X_tr_46, X_te_46)
    elapsed = time.perf_counter() - t0
    
    print(f"Jaccard Th = {th_jaccard:.2f} | Single Th = {res['cons_th']:.3f} | Macro F1 = {res['single_f1']:.4f} | Rec C0 = {res['c0_rec']:.4f} | Prec C0 = {res['c0_prec']:.4f} | F1 C0 = {res['c0_f1']:.4f} ({elapsed:.1f}s)")
    sweep_results.append({
        'twin_th': th_jaccard,
        'cons_th': res['cons_th'],
        'single_f1': res['single_f1'],
        'c0_rec': res['c0_rec'],
        'c0_prec': res['c0_prec'],
        'c0_f1': res['c0_f1']
    })
    
    if res['single_f1'] > best_twin_single_f1:
        best_twin_single_f1 = res['single_f1']
        best_twin_th = th_jaccard
        best_twin_res = res
        best_twin_tr_features = twin_feats_tr
        best_twin_te_features = twin_feats_te

print(f"\n>>> Best TWIN_JACCARD_THRESHOLD terpilih: {best_twin_th:.2f} dengan Single-Th Macro F1 = {best_twin_single_f1:.4f}")

# ==============================================================================
# 3. RE-TUNING DUAL-THRESHOLD PRECISION GATING UNTUK CONTENT-TWIN
# ==============================================================================
print("\n--- 3. Re-tuning Precision Gating Pasca Penambahan Content-Twin ---")
loc_conf = X_base_tr[:, 18]
fig_conf = X_base_tr[:, 21]
inst_mis = X_base_tr[:, 22]
neg_inv  = X_base_tr[:, 30]
ant_conf = X_base_tr[:, 34]
twin_conf_best = best_twin_tr_features[:, 2] # twin_conflict

# Sinyal konflik diperluas mencakup twin_conflict
row_conflict_base = ((loc_conf == 1) | (fig_conf == 1) | (inst_mis == 1) | (neg_inv == 1) | (ant_conf == 1))
row_conflict_with_twin = row_conflict_base | (twin_conf_best == 1.0)

meta_oof_twin = best_twin_res['meta_oof']

# Uji 1: Gating dengan row_conflict_base lama
best_cl_1, best_cf_1, best_g1 = 0.5, 0.5, 0.0
for th_cl in np.arange(0.65, 0.82, 0.005):
    for th_cf in np.arange(0.72, 0.90, 0.005):
        preds = np.where(row_conflict_base, (meta_oof_twin >= th_cf).astype(int), (meta_oof_twin >= th_cl).astype(int))
        sc = f1_score(y_train, preds, average='macro')
        if sc > best_g1:
            best_g1, best_cl_1, best_cf_1 = sc, th_cl, th_cf

# Uji 2: Gating dengan row_conflict_with_twin
best_cl_2, best_cf_2, best_g2 = 0.5, 0.5, 0.0
for th_cl in np.arange(0.65, 0.82, 0.005):
    for th_cf in np.arange(0.72, 0.90, 0.005):
        preds = np.where(row_conflict_with_twin, (meta_oof_twin >= th_cf).astype(int), (meta_oof_twin >= th_cl).astype(int))
        sc = f1_score(y_train, preds, average='macro')
        if sc > best_g2:
            best_g2, best_cl_2, best_cf_2 = sc, th_cl, th_cf

print(f"Gating (Base Conflict)      : Th_clean={best_cl_1:.3f}, Th_conf={best_cf_1:.3f} | Macro F1 = {best_g1:.4f}")
print(f"Gating (Conflict + TwinConf): Th_clean={best_cl_2:.3f}, Th_conf={best_cf_2:.3f} | Macro F1 = {best_g2:.4f}")

# Save Content-Twin features to disk
np.savez('scratch/content_twin_features.npz',
         twin_has_match_tr=twin_has_match_tr,
         twin_jaccard_tr=twin_jaccard_tr,
         best_twin_th=best_twin_th,
         twin_feats_tr=best_twin_tr_features,
         twin_feats_te=best_twin_te_features,
         best_meta_oof=best_twin_res['meta_oof'],
         best_meta_test=best_twin_res['meta_test'],
         best_gated_f1=max(best_g1, best_g2),
         best_single_f1=best_twin_single_f1)

print("\nContent-Twin features successfully saved to scratch/content_twin_features.npz")
