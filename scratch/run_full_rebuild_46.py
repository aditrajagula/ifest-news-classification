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
import optuna

sys.stdout.reconfigure(encoding='utf-8')
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_TRIALS = 100
MIN_CONTENT_LEN = 200
TWIN_JACCARD_THRESHOLD = 0.90

print("=" * 90)
print("PERBAIKAN KRITIS: FULL REBUILD PIPELINE 46-FITUR + HPO LIVE (TUGAS 1 - 6)")
print("=" * 90)

train = pd.read_csv('train.csv')
y_train = train['label'].values
test = pd.read_csv('test.csv')

# Load cached 42 base features + 1 city feature
cache = np.load(r'C:\Users\Reynaldi Hutagaol\.gemini\antigravity-ide\brain\489beaa2-e2ba-4370-a4e6-0110f42de593\scratch\features_cache.npz')
X_base_tr = cache['X_train'] # (14397, 42)
X_base_te = cache['X_test']  # (3603, 42)

extra = np.load('scratch/extra_features.npz')
city_tr = extra['city_tr'] # (14397, 1)
city_te = extra['city_te'] # (3603, 1)

X_train_43 = np.hstack([X_base_tr, city_tr])
X_test_43  = np.hstack([X_base_te, city_te])
print(f"X_train_43 shape: {X_train_43.shape}, X_test_43 shape: {X_test_43.shape}")

# Functions to build 3 Content-Twin features
def normalize_content(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = re.sub(r'[^\w\s]', '', s)
    return re.sub(r'\s+', ' ', s).strip()

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

# Build train label=1 dictionary: content_norm -> list of (row_idx, title_tokens)
c1_train = train[(train['label'] == 1) & (train['content_norm'].str.len() >= MIN_CONTENT_LEN)]
train_c1_dict = defaultdict(list)
for idx, row in c1_train.iterrows():
    train_c1_dict[row['content_norm']].append((idx, row['title_tokens']))

# Extract 3 features for train (LEAVE-ONE-OUT SAFE)
print("\n--- Ekstraksi 3 Fitur Content-Twin untuk Train (Leave-One-Out) ---")
twin_cols_tr = np.zeros((len(train), 3), dtype=np.float32)

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
    twin_cols_tr[idx, 0] = 1.0       # twin_has_match
    twin_cols_tr[idx, 1] = best_sim  # twin_title_jaccard
    twin_cols_tr[idx, 2] = 1.0 if best_sim < TWIN_JACCARD_THRESHOLD else 0.0 # twin_conflict

# Extract 3 features for test (FULL TRAIN C1 INDEX, ZERO-LEAKAGE)
print("--- Ekstraksi 3 Fitur Content-Twin untuk Test (Full Train C1 Index) ---")
twin_cols_te = np.zeros((len(test), 3), dtype=np.float32)

for idx, row in test.iterrows():
    c_norm = row['content_norm']
    if len(c_norm) < MIN_CONTENT_LEN:
        continue
    entries = train_c1_dict.get(c_norm, [])
    if not entries:
        continue
    own_tokens = row['title_tokens']
    best_sim = max(jaccard_titles(own_tokens, ref_tokens) for _, ref_tokens in entries)
    twin_cols_te[idx, 0] = 1.0       # twin_has_match
    twin_cols_te[idx, 1] = best_sim  # twin_title_jaccard
    twin_cols_te[idx, 2] = 1.0 if best_sim < TWIN_JACCARD_THRESHOLD else 0.0 # twin_conflict

# Build X_train_46 and X_test_46
X_train_46 = np.hstack([X_train_43, twin_cols_tr])
X_test_46  = np.hstack([X_test_43, twin_cols_te])

assert X_train_46.shape == (14397, 46), f"Dimensi train salah: {X_train_46.shape}"
assert X_test_46.shape == (3603, 46), f"Dimensi test salah: {X_test_46.shape}"
print(f"VERIFIKASI DIMENSI SUKSES: X_train_46={X_train_46.shape}, X_test_46={X_test_46.shape}")

# StratifiedKFold & sample_weights W=2.0x
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sample_weights = np.where(y_train == 0, 2.0, 1.0)
assert set(sample_weights) == {1.0, 2.0}, "Sample weights bukan W=2.0x!"
print(f"Sample weights W=2.0x verified: {set(sample_weights)}")

# ==============================================================================
# TUGAS 1: Training CV 5-Fold Baseline 46-Fitur (Default Params)
# ==============================================================================
print("\n" + "="*90)
print("--- TUGAS 1: TRAINING CV 5-FOLD BASELINE 46-FITUR (DEFAULT PARAMS, W=2.0X) ---")
print("="*90)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train_46)
X_te_sc = scaler.transform(X_test_46)

xgb_def_oof = np.zeros(len(train))
hgb_def_oof = np.zeros(len(train))
et_def_oof  = np.zeros(len(train))
lr_def_oof  = np.zeros(len(train))

xgb_def_te  = np.zeros(len(test))
hgb_def_te  = np.zeros(len(test))
et_def_te   = np.zeros(len(test))
lr_def_te   = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_46, y_train)):
    # XGB Default
    clf_x = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.035,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss', n_jobs=-1
    )
    clf_x.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    xgb_def_oof[val_idx] = clf_x.predict_proba(X_train_46[val_idx])[:, 1]
    xgb_def_te += clf_x.predict_proba(X_test_46)[:, 1] / 5.0
    
    # HGB Default
    clf_h = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.035, max_depth=6, random_state=SEED)
    clf_h.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    hgb_def_oof[val_idx] = clf_h.predict_proba(X_train_46[val_idx])[:, 1]
    hgb_def_te += clf_h.predict_proba(X_test_46)[:, 1] / 5.0
    
    # ET Default
    clf_e = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=SEED, n_jobs=-1)
    clf_e.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    et_def_oof[val_idx] = clf_e.predict_proba(X_train_46[val_idx])[:, 1]
    et_def_te += clf_e.predict_proba(X_test_46)[:, 1] / 5.0
    
    # LR Default
    clf_l = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf_l.fit(X_tr_sc[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    lr_def_oof[val_idx] = clf_l.predict_proba(X_tr_sc[val_idx])[:, 1]
    lr_def_te += clf_l.predict_proba(X_te_sc)[:, 1] / 5.0

# Meta Learner Default 46
meta_X_def = np.column_stack([xgb_def_oof, hgb_def_oof, et_def_oof, lr_def_oof])
meta_X_te_def = np.column_stack([xgb_def_te, hgb_def_te, et_def_te, lr_def_te])

meta_oof_def = np.zeros(len(train))
meta_test_def = np.zeros(len(test))
for trn, val in skf.split(meta_X_def, y_train):
    meta_clf = LogisticRegression(C=1.0, random_state=SEED)
    meta_clf.fit(meta_X_def[trn], y_train[trn])
    meta_oof_def[val] = meta_clf.predict_proba(meta_X_def[val])[:, 1]
    meta_test_def += meta_clf.predict_proba(meta_X_te_def)[:, 1] / 5.0

# Evaluate default 46 single threshold
best_def_th, best_def_f1 = 0.5, 0.0
for th in np.arange(0.3, 0.9, 0.005):
    sc = f1_score(y_train, (meta_oof_def >= th).astype(int), average='macro')
    if sc > best_def_f1:
        best_def_f1, best_def_th = sc, th

print(f"Hasil Live Tugas 1: Baseline 46-Fitur (Default Params, W=2.0x):")
print(f"  Single Threshold = {best_def_th:.3f} | Macro F1 OOF = {best_def_f1:.4f}")

# ==============================================================================
# TUGAS 2: HPO Sungguhan via Optuna di atas X_train_46 (100 Trials per Model)
# ==============================================================================
print("\n" + "="*90)
print(f"--- TUGAS 2: HPO SUNGGUHAN VIA OPTUNA DI ATAS X_train_46 ({N_TRIALS} TRIALS PER MODEL) ---")
print("="*90)

study_times = {}

# 1. Objective XGBoost on X_train_46
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
    }
    oof = np.zeros(len(y_train))
    fold_f1s = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_46, y_train)):
        clf = xgb.XGBClassifier(**params, random_state=SEED, eval_metric='logloss', n_jobs=-1)
        clf.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
        preds = clf.predict_proba(X_train_46[val_idx])[:, 1]
        oof[val_idx] = preds
        
        # Intermediate fold evaluation for MedianPruner
        fold_f1 = max(f1_score(y_train[val_idx], (preds >= th).astype(int), average='macro')
                      for th in np.arange(0.50, 0.85, 0.05))
        fold_f1s.append(fold_f1)
        trial.report(float(np.mean(fold_f1s)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    best_f1 = max(f1_score(y_train, (oof >= th).astype(int), average='macro')
                  for th in np.arange(0.40, 0.90, 0.01))
    return best_f1

# 2. Objective HistGradientBoosting on X_train_46
def objective_hgb(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 150, 450, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 5.0),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
    }
    oof = np.zeros(len(y_train))
    fold_f1s = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_46, y_train)):
        clf = HistGradientBoostingClassifier(**params, random_state=SEED)
        clf.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
        preds = clf.predict_proba(X_train_46[val_idx])[:, 1]
        oof[val_idx] = preds
        
        fold_f1 = max(f1_score(y_train[val_idx], (preds >= th).astype(int), average='macro')
                      for th in np.arange(0.50, 0.85, 0.05))
        fold_f1s.append(fold_f1)
        trial.report(float(np.mean(fold_f1s)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    best_f1 = max(f1_score(y_train, (oof >= th).astype(int), average='macro')
                  for th in np.arange(0.40, 0.90, 0.01))
    return best_f1

# 3. Objective ExtraTrees on X_train_46
def objective_et(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 150, 350, step=50),
        'max_depth': trial.suggest_int('max_depth', 8, 18),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.6, 0.8]),
    }
    oof = np.zeros(len(y_train))
    fold_f1s = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_46, y_train)):
        clf = ExtraTreesClassifier(**params, random_state=SEED, n_jobs=-1)
        clf.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
        preds = clf.predict_proba(X_train_46[val_idx])[:, 1]
        oof[val_idx] = preds
        
        fold_f1 = max(f1_score(y_train[val_idx], (preds >= th).astype(int), average='macro')
                      for th in np.arange(0.50, 0.85, 0.05))
        fold_f1s.append(fold_f1)
        trial.report(float(np.mean(fold_f1s)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    best_f1 = max(f1_score(y_train, (oof >= th).astype(int), average='macro')
                  for th in np.arange(0.40, 0.90, 0.01))
    return best_f1

# 4. Objective LogisticRegression on X_train_46
def objective_lr(trial):
    C = trial.suggest_float('C', 0.05, 5.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1'])
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
    
    oof = np.zeros(len(y_train))
    fold_f1s = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_46, y_train)):
        scaler_f = StandardScaler()
        X_tr_fold = scaler_f.fit_transform(X_train_46[trn_idx])
        X_val_fold = scaler_f.transform(X_train_46[val_idx])
        
        clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=SEED)
        clf.fit(X_tr_fold, y_train[trn_idx], sample_weight=sample_weights[trn_idx])
        preds = clf.predict_proba(X_val_fold)[:, 1]
        oof[val_idx] = preds
        
        fold_f1 = max(f1_score(y_train[val_idx], (preds >= th).astype(int), average='macro')
                      for th in np.arange(0.50, 0.85, 0.05))
        fold_f1s.append(fold_f1)
        trial.report(float(np.mean(fold_f1s)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    best_f1 = max(f1_score(y_train, (oof >= th).astype(int), average='macro')
                  for th in np.arange(0.40, 0.90, 0.01))
    return best_f1

def make_callback(name):
    def callback(study, trial):
        if (trial.number + 1) % 10 == 0 or trial.number == N_TRIALS - 1 or trial.number == 0:
            state = "PRUNED" if trial.state == optuna.trial.TrialState.PRUNED else f"val={trial.value:.4f}"
            best = f"{study.best_value:.4f}" if study.best_trial else "N/A"
            print(f"  [{name}] Trial {trial.number+1:>3}/{N_TRIALS} -> {state} | Best: {best}", flush=True)
    return callback

# Run XGBoost Optuna
print("\n1. Mengoptimasi XGBoost pada X_train_46...")
t0 = time.perf_counter()
study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, callbacks=[make_callback("XGBoost")])
study_times['xgb'] = time.perf_counter() - t0
print(f"XGBoost selesai ({study_times['xgb']/60:.2f} m). Best F1: {study_xgb.best_value:.4f} | Params: {study_xgb.best_params}")

# Run HistGBM Optuna
print("\n2. Mengoptimasi HistGradientBoosting pada X_train_46...")
t0 = time.perf_counter()
study_hgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
study_hgb.optimize(objective_hgb, n_trials=N_TRIALS, callbacks=[make_callback("HistGBM")])
study_times['hgb'] = time.perf_counter() - t0
print(f"HistGBM selesai ({study_times['hgb']/60:.2f} m). Best F1: {study_hgb.best_value:.4f} | Params: {study_hgb.best_params}")

# Run ExtraTrees Optuna
print("\n3. Mengoptimasi ExtraTrees pada X_train_46...")
t0 = time.perf_counter()
study_et = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
study_et.optimize(objective_et, n_trials=N_TRIALS, callbacks=[make_callback("ExtraTrees")])
study_times['et'] = time.perf_counter() - t0
print(f"ExtraTrees selesai ({study_times['et']/60:.2f} m). Best F1: {study_et.best_value:.4f} | Params: {study_et.best_params}")

# Run LogisticRegression Optuna
print("\n4. Mengoptimasi LogisticRegression pada X_train_46...")
t0 = time.perf_counter()
study_lr = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
study_lr.optimize(objective_lr, n_trials=N_TRIALS, callbacks=[make_callback("LogisticReg")])
study_times['lr'] = time.perf_counter() - t0
print(f"LogisticRegression selesai ({study_times['lr']/60:.2f} m). Best F1: {study_lr.best_value:.4f} | Params: {study_lr.best_params}")

total_hpo_time = sum(study_times.values())
print(f"\nTotal waktu HPO 4 model: {total_hpo_time/60:.2f} menit ({total_hpo_time:.1f} detik)")

# ==============================================================================
# RETRAIN FULL ENSEMBLE 5-FOLD CV DENGAN HYPERPARAMETER HASIL HPO NYATA
# ==============================================================================
print("\n" + "="*90)
print("--- RETRAIN FULL ENSEMBLE CV 5-FOLD DENGAN HYPERPARAMETER HASIL HPO NYATA ---")
print("="*90)

xgb_best = study_xgb.best_params
hgb_best = study_hgb.best_params
et_best  = study_et.best_params
lr_best  = study_lr.best_params

xgb_params_final = dict(xgb_best, random_state=SEED, eval_metric='logloss', n_jobs=-1)
hgb_params_final = dict(hgb_best, random_state=SEED)
et_params_final  = dict(et_best, random_state=SEED, n_jobs=-1)
lr_params_final  = {'C': lr_best['C'], 'penalty': lr_best['penalty'],
                    'solver': 'liblinear' if lr_best['penalty'] == 'l1' else 'lbfgs',
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
    # XGB
    clf_xgb_live = xgb.XGBClassifier(**xgb_params_final)
    clf_xgb_live.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    xgb_hpo_oof[val_idx] = clf_xgb_live.predict_proba(X_train_46[val_idx])[:, 1]
    xgb_hpo_te += clf_xgb_live.predict_proba(X_test_46)[:, 1] / 5.0
    last_clf_xgb = clf_xgb_live
    
    # HGB
    clf_hgb_live = HistGradientBoostingClassifier(**hgb_params_final)
    clf_hgb_live.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    hgb_hpo_oof[val_idx] = clf_hgb_live.predict_proba(X_train_46[val_idx])[:, 1]
    hgb_hpo_te += clf_hgb_live.predict_proba(X_test_46)[:, 1] / 5.0
    
    # ET
    clf_et_live = ExtraTreesClassifier(**et_params_final)
    clf_et_live.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    et_hpo_oof[val_idx] = clf_et_live.predict_proba(X_train_46[val_idx])[:, 1]
    et_hpo_te += clf_et_live.predict_proba(X_test_46)[:, 1] / 5.0
    
    # LR
    clf_lr_live = LogisticRegression(**lr_params_final)
    clf_lr_live.fit(X_tr_sc[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    lr_hpo_oof[val_idx] = clf_lr_live.predict_proba(X_tr_sc[val_idx])[:, 1]
    lr_hpo_te += clf_lr_live.predict_proba(X_te_sc)[:, 1] / 5.0

# Verifikasi eksplisit parameter aktual model XGBoost
print("\nParameter XGBoost yang benar-benar dipakai di model aktual:")
print(f"  n_estimators actual: {last_clf_xgb.get_params()['n_estimators']}")
print(f"  n_estimators target HPO: {study_xgb.best_params['n_estimators']}")
assert last_clf_xgb.get_params()['n_estimators'] == study_xgb.best_params['n_estimators'], "Model TIDAK memakai hyperparameter HPO!"
print("ASERSI MODEL AKTUAL LOLOS 100%!")

# Meta Learner Final (Stacking Level-1)
meta_X = np.column_stack([xgb_hpo_oof, hgb_hpo_oof, et_hpo_oof, lr_hpo_oof])
meta_X_test = np.column_stack([xgb_hpo_te, hgb_hpo_te, et_hpo_te, lr_hpo_te])

meta_oof = np.zeros(len(train))
meta_test = np.zeros(len(test))

for trn, val in skf.split(meta_X, y_train):
    meta_clf = LogisticRegression(C=1.0, random_state=SEED)
    meta_clf.fit(meta_X[trn], y_train[trn])
    meta_oof[val] = meta_clf.predict_proba(meta_X[val])[:, 1]
    meta_test += meta_clf.predict_proba(meta_X_test)[:, 1] / 5.0

# ==============================================================================
# TUGAS 3: Re-tuning Threshold Live dari meta_oof Baru
# ==============================================================================
print("\n" + "="*90)
print("--- TUGAS 3: RE-TUNING THRESHOLD LIVE DARI meta_oof BARU ---")
print("="*90)

# Single threshold sweep
best_single_f1, eval_th = 0.0, 0.5
for th in np.arange(0.3, 0.9, 0.005):
    sc = f1_score(y_train, (meta_oof >= th).astype(int), average='macro')
    if sc > best_single_f1:
        best_single_f1, eval_th = sc, th

single_preds = (meta_oof >= eval_th).astype(int)
cr_single = classification_report(y_train, single_preds, output_dict=True)

print(f"Single Consensus Threshold: {eval_th:.3f}")
print(f"Single-Threshold Macro F1: {best_single_f1:.4f}")
print(f"  Class 0 Recall: {cr_single['0']['recall']:.4f}, Precision: {cr_single['0']['precision']:.4f}")

# Dual-Threshold Precision Gating Grid Search
loc_conf = X_base_tr[:, 18]
fig_conf = X_base_tr[:, 21]
inst_mis = X_base_tr[:, 22]
neg_inv  = X_base_tr[:, 30]
ant_conf = X_base_tr[:, 34]
row_conflict = ((loc_conf == 1) | (fig_conf == 1) | (inst_mis == 1) | (neg_inv == 1) | (ant_conf == 1))

best_gated_f1, best_th_clean, best_th_conf = 0.0, 0.5, 0.5
for th_cl in np.arange(0.50, 0.80, 0.005):
    for th_cf in np.arange(0.65, 0.90, 0.005):
        preds = np.where(row_conflict, (meta_oof >= th_cf).astype(int), (meta_oof >= th_cl).astype(int))
        sc = f1_score(y_train, preds, average='macro')
        if sc > best_gated_f1:
            best_gated_f1, best_th_clean, best_th_conf = sc, th_cl, th_cf

gated_preds = np.where(row_conflict, (meta_oof >= best_th_conf).astype(int), (meta_oof >= best_th_clean).astype(int))
cr_gated = classification_report(y_train, gated_preds, output_dict=True)

print(f"Dual-Threshold Gating: Th_clean = {best_th_clean:.3f}, Th_conf = {best_th_conf:.3f}")
print(f"Gated Macro F1 OOF Final: {best_gated_f1:.4f}")
print(f"  Class 0 Recall: {cr_gated['0']['recall']:.4f}, Precision: {cr_gated['0']['precision']:.4f}")

# ==============================================================================
# SANITY CHECK ROBUSTNESS SEED 43
# ==============================================================================
print("\n" + "="*90)
print("--- SANITY CHECK ROBUSTNESS ANTAR-SEED (SEED 42 vs SEED 43) ---")
print("="*90)

skf_seed43 = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

xgb_s43 = np.zeros(len(train))
hgb_s43 = np.zeros(len(train))
et_s43  = np.zeros(len(train))
lr_s43  = np.zeros(len(train))

for trn, val in skf_seed43.split(X_train_46, y_train):
    clf_x = xgb.XGBClassifier(**xgb_params_final)
    clf_x.fit(X_train_46[trn], y_train[trn], sample_weight=sample_weights[trn])
    xgb_s43[val] = clf_x.predict_proba(X_train_46[val])[:, 1]
    
    clf_h = HistGradientBoostingClassifier(**hgb_params_final)
    clf_h.fit(X_train_46[trn], y_train[trn], sample_weight=sample_weights[trn])
    hgb_s43[val] = clf_h.predict_proba(X_train_46[val])[:, 1]
    
    clf_e = ExtraTreesClassifier(**et_params_final)
    clf_e.fit(X_train_46[trn], y_train[trn], sample_weight=sample_weights[trn])
    et_s43[val] = clf_e.predict_proba(X_train_46[val])[:, 1]
    
    clf_l = LogisticRegression(**lr_params_final)
    clf_l.fit(X_tr_sc[trn], y_train[trn], sample_weight=sample_weights[trn])
    lr_s43[val] = clf_l.predict_proba(X_tr_sc[val])[:, 1]

meta_X_s43 = np.column_stack([xgb_s43, hgb_s43, et_s43, lr_s43])
meta_oof_s43 = np.zeros(len(train))

for trn, val in skf_seed43.split(meta_X_s43, y_train):
    meta_clf = LogisticRegression(C=1.0, random_state=43)
    meta_clf.fit(meta_X_s43[trn], y_train[trn])
    meta_oof_s43[val] = meta_clf.predict_proba(meta_X_s43[val])[:, 1]

preds_s43 = np.where(row_conflict, (meta_oof_s43 >= best_th_conf).astype(int), (meta_oof_s43 >= best_th_clean).astype(int))
f1_seed43 = f1_score(y_train, preds_s43, average='macro')
delta_seed43 = abs(best_gated_f1 - f1_seed43)

print(f"Seed 42 Macro F1 OOF : {best_gated_f1:.4f}")
print(f"Seed 43 Macro F1 OOF : {f1_seed43:.4f}")
print(f"Selisih Mutlak (|Δ|) : {delta_seed43:.4f} (< 0.010)")
is_robust = (delta_seed43 < 0.010)
print(f"Hasil Robustness: {'ROBUST (PASS)' if is_robust else 'OVERFITTING WARNING (FAIL)'}")

# ==============================================================================
# TUGAS 4: AUDIT WIRING PIPELINE
# ==============================================================================
print("\n" + "="*90)
print("=== AUDIT WIRING PIPELINE (mencegah bug integrasi seperti sebelumnya) ===")
print(f"1. Dimensi X_train yang dipakai model: {X_train_46.shape} (harus 46 kolom)")
print(f"2. sample_weight yang dipakai: {set(sample_weights)} (harus {{1.0, 2.0}})")
print(f"3. XGBoost n_estimators aktual di model: {last_clf_xgb.get_params()['n_estimators']} (harus cocok study_xgb.best_params: {study_xgb.best_params['n_estimators']})")
print(f"4. eval_th dihitung dari meta_oof shape: {meta_oof.shape}, nilai unik threshold: {eval_th}")
print(f"5. meta_test dihitung dari X_test shape: {X_test_46.shape} (harus 46 kolom, BUKAN 43)")
print(f"6. Sanity check seed 43 delta: {delta_seed43:.4f} (< 0.010)")

assert X_train_46.shape[1] == 46, "X_train_46 bukan 46 kolom!"
assert X_test_46.shape[1] == 46, "X_test_46 bukan 46 kolom!"
assert set(sample_weights) == {1.0, 2.0}, "Sample weights bukan {1.0, 2.0}!"
assert last_clf_xgb.get_params()['n_estimators'] == study_xgb.best_params['n_estimators'], "XGB n_estimators tidak cocok dengan hasil HPO!"
assert len(meta_test) == 3603, "meta_test bukan 3603 baris!"
assert delta_seed43 < 0.010, f"Seed 43 delta terlalu besar: {delta_seed43}"
print("SEMUA VERIFIKASI WIRING LOLOS — pipeline benar-benar terintegrasi.")

# ==============================================================================
# TUGAS 5: GENERATE SUBMISSION NYATA & VERIFIKASI
# ==============================================================================
print("\n" + "="*90)
print("--- TUGAS 5: GENERATE SUBMISSION NYATA & VERIFIKASI ---")
print("="*90)

loc_conf_te = X_base_te[:, 18]
fig_conf_te = X_base_te[:, 21]
inst_mis_te = X_base_te[:, 22]
neg_inv_te  = X_base_te[:, 30]
ant_conf_te = X_base_te[:, 34]
row_conflict_test = ((loc_conf_te == 1) | (fig_conf_te == 1) | (inst_mis_te == 1) | (neg_inv_te == 1) | (ant_conf_te == 1))

test_preds = np.where(row_conflict_test, (meta_test >= best_th_conf).astype(int), (meta_test >= best_th_clean).astype(int))

sub = pd.DataFrame({'id': test['id'], 'label': test_preds})
assert len(sub) == 3603, f"Jumlah baris salah: {len(sub)}"
assert list(sub.columns) == ['id', 'label'], f"Nama kolom salah: {sub.columns}"
assert sub['label'].isnull().sum() == 0, "Ditemukan NULL di submission"
assert set(sub['label'].unique()).issubset({0, 1}), "Label bukan biner {0, 1}"

sub.to_csv('submission_baseline.csv', index=False)
sub.to_csv('scratch/submission_rebuild_46.csv', index=False)
print("File submission_baseline.csv BERHASIL digenerate dari model 46-fitur HPO!")
print("Distribusi Submission Final:")
print(sub['label'].value_counts())
c0 = (sub['label'] == 0).sum()
c1 = (sub['label'] == 1).sum()
print(f"  Kelas 0: {c0} ({c0/len(sub)*100:.2f}%)")
print(f"  Kelas 1: {c1} ({c1/len(sub)*100:.2f}%)")

# ==============================================================================
# SAVE ARRAYS & RESULTS UNTUK SINKRONISASI NOTEBOOK
# ==============================================================================
results_dict = {
    'baseline_46_def_th': float(best_def_th),
    'baseline_46_def_f1': float(best_def_f1),
    'best_params': {
        'xgb': study_xgb.best_params,
        'hgb': study_hgb.best_params,
        'et': study_et.best_params,
        'lr': study_lr.best_params
    },
    'best_individual_f1': {
        'xgb': float(study_xgb.best_value),
        'hgb': float(study_hgb.best_value),
        'et': float(study_et.best_value),
        'lr': float(study_lr.best_value)
    },
    'study_times': study_times,
    'total_hpo_time': float(total_hpo_time),
    'final_single_th': float(eval_th),
    'final_single_f1': float(best_single_f1),
    'final_gated_th_clean': float(best_th_clean),
    'final_gated_th_conf': float(best_th_conf),
    'final_gated_f1': float(best_gated_f1),
    'final_c0_recall': float(cr_gated['0']['recall']),
    'final_c0_precision': float(cr_gated['0']['precision']),
    'final_c0_f1': float(cr_gated['0']['f1-score']),
    'seed43_f1': float(f1_seed43),
    'seed43_delta': float(delta_seed43),
    'test_c0_count': int(c0),
    'test_c1_count': int(c1)
}

with open('scratch/full_rebuild_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

np.savez('scratch/full_rebuild_arrays.npz',
         X_train_46=X_train_46,
         X_test_46=X_test_46,
         meta_oof=meta_oof,
         meta_test=meta_test)

print("\nSELURUH HASIL SUKSES DISIMPAN KE scratch/full_rebuild_results.json & .npz")
