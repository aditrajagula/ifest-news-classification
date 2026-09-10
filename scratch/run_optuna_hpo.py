import sys
import os
import time
import json
import numpy as np
import pandas as pd
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

print("=" * 80)
print("HYPERPARAMETER OPTIMIZATION (OPTUNA) - IFEST 2026 NEWS CLASSIFICATION")
print("=" * 80)
print(f"Random Seed: {SEED}")
print(f"Trials per model: {N_TRIALS}")
print("Rules: Strict zero-leakage (Rule #8) - train set CV only, no test data in objectives")
print("Sample Weight: Class 0 = 2.0x (explicitly applied in every objective fit)")
print("Pruner: optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)")
print("=" * 80)

# Load data
train = pd.read_csv('train.csv')
y_train = train['label'].values

cache = np.load(r'C:\Users\Reynaldi Hutagaol\.gemini\antigravity-ide\brain\489beaa2-e2ba-4370-a4e6-0110f42de593\scratch\features_cache.npz')
X_base_tr = cache['X_train'] # (14397, 42)
X_base_te = cache['X_test']  # (3603, 42)

extra = np.load('scratch/extra_features.npz')
city_tr = extra['city_tr'] # (14397, 1)
city_te = extra['city_te'] # (3603, 1)

X_train = np.hstack([X_base_tr, city_tr]) # (14397, 43)
X_test  = np.hstack([X_base_te, city_te]) # (3603, 43)
print(f"Features shape: X_train={X_train.shape}, X_test={X_test.shape}")

# StratifiedKFold and Sample Weights
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sample_weights = np.where(y_train == 0, 2.0, 1.0) # 2.0x weight for class 0

# Baseline default parameters
DEFAULT_PARAMS = {
    'xgb': {
        'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.035,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1, 'reg_lambda': 1.0
    },
    'hgb': {
        'max_iter': 300, 'learning_rate': 0.035, 'max_depth': 6,
        'l2_regularization': 0.0, 'max_leaf_nodes': 31
    },
    'et': {
        'n_estimators': 250, 'max_depth': 12, 'min_samples_split': 2,
        'min_samples_leaf': 1, 'max_features': 'sqrt'
    },
    'lr': {
        'C': 0.5, 'penalty': 'l2', 'solver': 'lbfgs'
    }
}

# Evaluate individual baseline models with default parameters
print("\n--- Evaluating Baseline Individual Models (Default Params) ---")
baseline_individual_f1 = {}

# 1. Default XGB
print("Running default XGBoost...")
xgb_oof_def = np.zeros(len(y_train))
for trn, val in skf.split(X_train, y_train):
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.035, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED, eval_metric='logloss', n_jobs=-1
    )
    clf.fit(X_train[trn], y_train[trn], sample_weight=sample_weights[trn])
    xgb_oof_def[val] = clf.predict_proba(X_train[val])[:, 1]
f1_xgb_def = max(f1_score(y_train, (xgb_oof_def >= th).astype(int), average='macro') for th in np.arange(0.50, 0.90, 0.01))
baseline_individual_f1['xgb'] = f1_xgb_def
print(f"Default XGBoost OOF Macro F1: {f1_xgb_def:.4f}")

# 2. Default HGB
print("Running default HistGradientBoosting...")
hgb_oof_def = np.zeros(len(y_train))
for trn, val in skf.split(X_train, y_train):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.035, max_depth=6, random_state=SEED)
    clf.fit(X_train[trn], y_train[trn], sample_weight=sample_weights[trn])
    hgb_oof_def[val] = clf.predict_proba(X_train[val])[:, 1]
f1_hgb_def = max(f1_score(y_train, (hgb_oof_def >= th).astype(int), average='macro') for th in np.arange(0.50, 0.90, 0.01))
baseline_individual_f1['hgb'] = f1_hgb_def
print(f"Default HistGradientBoosting OOF Macro F1: {f1_hgb_def:.4f}")

# 3. Default ET
print("Running default ExtraTrees...")
et_oof_def = np.zeros(len(y_train))
for trn, val in skf.split(X_train, y_train):
    clf = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=SEED, n_jobs=-1)
    clf.fit(X_train[trn], y_train[trn], sample_weight=sample_weights[trn])
    et_oof_def[val] = clf.predict_proba(X_train[val])[:, 1]
f1_et_def = max(f1_score(y_train, (et_oof_def >= th).astype(int), average='macro') for th in np.arange(0.50, 0.90, 0.01))
baseline_individual_f1['et'] = f1_et_def
print(f"Default ExtraTrees OOF Macro F1: {f1_et_def:.4f}")

# 4. Default LR
print("Running default LogisticRegression...")
lr_oof_def = np.zeros(len(y_train))
for trn, val in skf.split(X_train, y_train):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train[trn])
    X_val_sc = scaler.transform(X_train[val])
    clf = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf.fit(X_tr_sc, y_train[trn], sample_weight=sample_weights[trn])
    lr_oof_def[val] = clf.predict_proba(X_val_sc)[:, 1]
f1_lr_def = max(f1_score(y_train, (lr_oof_def >= th).astype(int), average='macro') for th in np.arange(0.50, 0.90, 0.01))
baseline_individual_f1['lr'] = f1_lr_def
print(f"Default LogisticRegression OOF Macro F1: {f1_lr_def:.4f}")

# ==============================================================================
# TUGAS 1: Optuna Studies per Model
# ==============================================================================

# 1. XGBoost Objective
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.65, 0.95, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.60, 0.90, step=0.05),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.2, 8.0, log=True),
    }
    oof = np.zeros(len(y_train))
    fold_f1s = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        clf = xgb.XGBClassifier(
            **params, random_state=SEED, eval_metric='logloss', n_jobs=-1
        )
        clf.fit(X_train[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
        preds = clf.predict_proba(X_train[val_idx])[:, 1]
        oof[val_idx] = preds
        
        # Intermediate fold validation F1 for pruning
        fold_f1 = max(f1_score(y_train[val_idx], (preds >= th).astype(int), average='macro')
                      for th in np.arange(0.65, 0.85, 0.02))
        fold_f1s.append(fold_f1)
        trial.report(float(np.mean(fold_f1s)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    best_f1 = max(f1_score(y_train, (oof >= th).astype(int), average='macro')
                  for th in np.arange(0.50, 0.90, 0.01))
    return best_f1

# 2. HistGradientBoosting Objective
def objective_hgb(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 150, 450, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 5.0),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
    }
    oof = np.zeros(len(y_train))
    fold_f1s = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        clf = HistGradientBoostingClassifier(
            **params, random_state=SEED
        )
        clf.fit(X_train[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
        preds = clf.predict_proba(X_train[val_idx])[:, 1]
        oof[val_idx] = preds
        
        fold_f1 = max(f1_score(y_train[val_idx], (preds >= th).astype(int), average='macro')
                      for th in np.arange(0.65, 0.85, 0.02))
        fold_f1s.append(fold_f1)
        trial.report(float(np.mean(fold_f1s)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    best_f1 = max(f1_score(y_train, (oof >= th).astype(int), average='macro')
                  for th in np.arange(0.50, 0.90, 0.01))
    return best_f1

# 3. ExtraTrees Objective
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
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        clf = ExtraTreesClassifier(
            **params, random_state=SEED, n_jobs=-1
        )
        clf.fit(X_train[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
        preds = clf.predict_proba(X_train[val_idx])[:, 1]
        oof[val_idx] = preds
        
        fold_f1 = max(f1_score(y_train[val_idx], (preds >= th).astype(int), average='macro')
                      for th in np.arange(0.65, 0.85, 0.02))
        fold_f1s.append(fold_f1)
        trial.report(float(np.mean(fold_f1s)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    best_f1 = max(f1_score(y_train, (oof >= th).astype(int), average='macro')
                  for th in np.arange(0.50, 0.90, 0.01))
    return best_f1

# 4. LogisticRegression Objective
def objective_lr(trial):
    C = trial.suggest_float('C', 0.05, 5.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1'])
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
    
    oof = np.zeros(len(y_train))
    fold_f1s = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        scaler = StandardScaler()
        X_trn_sc = scaler.fit_transform(X_train[trn_idx])
        X_val_sc = scaler.transform(X_train[val_idx])
        
        clf = LogisticRegression(
            C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=SEED
        )
        clf.fit(X_trn_sc, y_train[trn_idx], sample_weight=sample_weights[trn_idx])
        preds = clf.predict_proba(X_val_sc)[:, 1]
        oof[val_idx] = preds
        
        fold_f1 = max(f1_score(y_train[val_idx], (preds >= th).astype(int), average='macro')
                      for th in np.arange(0.65, 0.85, 0.02))
        fold_f1s.append(fold_f1)
        trial.report(float(np.mean(fold_f1s)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    best_f1 = max(f1_score(y_train, (oof >= th).astype(int), average='macro')
                  for th in np.arange(0.50, 0.90, 0.01))
    return best_f1

studies = {}
study_times = {}

def make_callback(model_name):
    def callback(study, trial):
        if (trial.number + 1) % 10 == 0 or trial.number == N_TRIALS - 1 or trial.number == 0:
            state_str = "PRUNED" if trial.state == optuna.trial.TrialState.PRUNED else f"val={trial.value:.4f}"
            best_val = f"{study.best_value:.4f}" if study.best_trial is not None else "N/A"
            print(f"  [{model_name}] Trial {trial.number+1:>3}/{N_TRIALS} -> {state_str} | Best: {best_val}", flush=True)
    return callback

# Optimize XGBoost
print("\n" + "="*80)
print(f"--- 1. OPTIMIZING XGBOOST ({N_TRIALS} Trials) ---")
print("="*80)
t0 = time.perf_counter()
study_xgb = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
)
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, callbacks=[make_callback("XGBoost")], show_progress_bar=False)
t_xgb = time.perf_counter() - t0
study_times['xgb'] = t_xgb
studies['xgb'] = study_xgb
print(f"XGBoost finished in {t_xgb/60:.2f} mins.")
print(f"Best Trial #{study_xgb.best_trial.number} - Best OOF Macro F1: {study_xgb.best_value:.4f}")
print(f"Best Params: {study_xgb.best_params}")
print(f"Completed trials: {len([t for t in study_xgb.trials if t.state == optuna.trial.TrialState.COMPLETE])}, "
      f"Pruned trials: {len([t for t in study_xgb.trials if t.state == optuna.trial.TrialState.PRUNED])}")

# Optimize HistGradientBoosting
print("\n" + "="*80)
print(f"--- 2. OPTIMIZING HISTGRADIENTBOOSTING ({N_TRIALS} Trials) ---")
print("="*80)
t0 = time.perf_counter()
study_hgb = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
)
study_hgb.optimize(objective_hgb, n_trials=N_TRIALS, callbacks=[make_callback("HistGBM")], show_progress_bar=False)
t_hgb = time.perf_counter() - t0
study_times['hgb'] = t_hgb
studies['hgb'] = study_hgb
print(f"HistGradientBoosting finished in {t_hgb/60:.2f} mins.")
print(f"Best Trial #{study_hgb.best_trial.number} - Best OOF Macro F1: {study_hgb.best_value:.4f}")
print(f"Best Params: {study_hgb.best_params}")
print(f"Completed trials: {len([t for t in study_hgb.trials if t.state == optuna.trial.TrialState.COMPLETE])}, "
      f"Pruned trials: {len([t for t in study_hgb.trials if t.state == optuna.trial.TrialState.PRUNED])}")

# Optimize ExtraTrees
print("\n" + "="*80)
print(f"--- 3. OPTIMIZING EXTRATREES ({N_TRIALS} Trials) ---")
print("="*80)
t0 = time.perf_counter()
study_et = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
)
study_et.optimize(objective_et, n_trials=N_TRIALS, callbacks=[make_callback("ExtraTrees")], show_progress_bar=False)
t_et = time.perf_counter() - t0
study_times['et'] = t_et
studies['et'] = study_et
print(f"ExtraTrees finished in {t_et/60:.2f} mins.")
print(f"Best Trial #{study_et.best_trial.number} - Best OOF Macro F1: {study_et.best_value:.4f}")
print(f"Best Params: {study_et.best_params}")
print(f"Completed trials: {len([t for t in study_et.trials if t.state == optuna.trial.TrialState.COMPLETE])}, "
      f"Pruned trials: {len([t for t in study_et.trials if t.state == optuna.trial.TrialState.PRUNED])}")

# Optimize LogisticRegression
print("\n" + "="*80)
print(f"--- 4. OPTIMIZING LOGISTIC REGRESSION ({N_TRIALS} Trials) ---")
print("="*80)
t0 = time.perf_counter()
study_lr = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
)
study_lr.optimize(objective_lr, n_trials=N_TRIALS, callbacks=[make_callback("LogisticReg")], show_progress_bar=False)
t_lr = time.perf_counter() - t0
study_times['lr'] = t_lr
studies['lr'] = study_lr
print(f"LogisticRegression finished in {t_lr/60:.2f} mins.")
print(f"Best Trial #{study_lr.best_trial.number} - Best OOF Macro F1: {study_lr.best_value:.4f}")
print(f"Best Params: {study_lr.best_params}")
print(f"Completed trials: {len([t for t in study_lr.trials if t.state == optuna.trial.TrialState.COMPLETE])}, "
      f"Pruned trials: {len([t for t in study_lr.trials if t.state == optuna.trial.TrialState.PRUNED])}")

total_time_hpo = sum(study_times.values())
print(f"\n>>> Total HPO Execution Time across 4 models: {total_time_hpo/60:.2f} minutes ({total_time_hpo:.1f}s)")

# ==============================================================================
# TUGAS 2: Rebuild Ensemble dengan Hyperparameter Hasil HPO
# ==============================================================================
print("\n" + "="*80)
print("--- TUGAS 2: REBUILD ENSEMBLE DENGAN HYPERPARAMETER HASIL HPO ---")
print("="*80)

xgb_best = study_xgb.best_params
hgb_best = study_hgb.best_params
et_best  = study_et.best_params
lr_best  = study_lr.best_params

# Model parameters
xgb_params = dict(xgb_best, random_state=SEED, eval_metric='logloss', n_jobs=-1)
hgb_params = dict(hgb_best, random_state=SEED)
et_params  = dict(et_best, random_state=SEED, n_jobs=-1)
lr_solver  = 'liblinear' if lr_best['penalty'] == 'l1' else 'lbfgs'
lr_params  = {'C': lr_best['C'], 'penalty': lr_best['penalty'], 'solver': lr_solver, 'max_iter': 1000, 'random_state': SEED}

xgb_hpo_oof = np.zeros(len(y_train))
hgb_hpo_oof = np.zeros(len(y_train))
et_hpo_oof  = np.zeros(len(y_train))
lr_hpo_oof  = np.zeros(len(y_train))

xgb_hpo_te  = np.zeros(len(X_test))
hgb_hpo_te  = np.zeros(len(X_test))
et_hpo_te   = np.zeros(len(X_test))
lr_hpo_te   = np.zeros(len(X_test))

for trn, val in skf.split(X_train, y_train):
    # XGB
    clf_x = xgb.XGBClassifier(**xgb_params)
    clf_x.fit(X_train[trn], y_train[trn], sample_weight=sample_weights[trn])
    xgb_hpo_oof[val] = clf_x.predict_proba(X_train[val])[:, 1]
    xgb_hpo_te += clf_x.predict_proba(X_test)[:, 1] / 5.0
    
    # HGB
    clf_h = HistGradientBoostingClassifier(**hgb_params)
    clf_h.fit(X_train[trn], y_train[trn], sample_weight=sample_weights[trn])
    hgb_hpo_oof[val] = clf_h.predict_proba(X_train[val])[:, 1]
    hgb_hpo_te += clf_h.predict_proba(X_test)[:, 1] / 5.0
    
    # ET
    clf_e = ExtraTreesClassifier(**et_params)
    clf_e.fit(X_train[trn], y_train[trn], sample_weight=sample_weights[trn])
    et_hpo_oof[val] = clf_e.predict_proba(X_train[val])[:, 1]
    et_hpo_te += clf_e.predict_proba(X_test)[:, 1] / 5.0
    
    # LR
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train[trn])
    X_val_sc = scaler.transform(X_train[val])
    X_te_sc = scaler.transform(X_test)
    clf_l = LogisticRegression(**lr_params)
    clf_l.fit(X_tr_sc, y_train[trn], sample_weight=sample_weights[trn])
    lr_hpo_oof[val] = clf_l.predict_proba(X_val_sc)[:, 1]
    lr_hpo_te += clf_l.predict_proba(X_te_sc)[:, 1] / 5.0

# Stacking Meta-Learner
meta_X_hpo = np.column_stack([xgb_hpo_oof, hgb_hpo_oof, et_hpo_oof, lr_hpo_oof])
meta_X_te_hpo = np.column_stack([xgb_hpo_te, hgb_hpo_te, et_hpo_te, lr_hpo_te])

meta_oof_hpo = np.zeros(len(y_train))
meta_test_hpo = np.zeros(len(X_test))
fold_ths_hpo = []

for trn, val in skf.split(meta_X_hpo, y_train):
    meta_clf = LogisticRegression(C=1.0, random_state=SEED)
    meta_clf.fit(meta_X_hpo[trn], y_train[trn])
    val_probs = meta_clf.predict_proba(meta_X_hpo[val])[:, 1]
    meta_oof_hpo[val] = val_probs
    meta_test_hpo += meta_clf.predict_proba(meta_X_te_hpo)[:, 1] / 5.0
    
    f_th, f_f1 = 0.810, 0.0
    for th in np.arange(0.70, 0.88, 0.005):
        sc = f1_score(y_train[val], (val_probs >= th).astype(int), average='macro')
        if sc > f_f1:
            f_f1 = sc
            f_th = th
    fold_ths_hpo.append(round(float(f_th), 3))

cons_th_hpo = round(float(np.median(fold_ths_hpo)), 3)
single_th_preds_hpo = (meta_oof_hpo >= cons_th_hpo).astype(int)
single_th_f1_hpo = f1_score(y_train, single_th_preds_hpo, average='macro')
cr_single = classification_report(y_train, single_th_preds_hpo, output_dict=True)

print(f"Stacking Ensemble HPO - Single Consensus Threshold: {cons_th_hpo:.3f}")
print(f"Stacking Ensemble HPO - Single-Threshold Macro F1: {single_th_f1_hpo:.4f}")
print(f"Class 0 Recall: {cr_single['0']['recall']:.4f}, Precision: {cr_single['0']['precision']:.4f}, F1: {cr_single['0']['f1-score']:.4f}")

# ==============================================================================
# TUGAS 3: Re-tuning Threshold & Precision Gating Pasca-HPO
# ==============================================================================
print("\n" + "="*80)
print("--- TUGAS 3: RE-TUNING THRESHOLD & GATING PASCA-HPO ---")
print("="*80)

# Construct row_conflict from X_base_tr
loc_conf = X_base_tr[:, 18]
fig_conf = X_base_tr[:, 21]
inst_mis = X_base_tr[:, 22]
neg_inv  = X_base_tr[:, 30]
ant_conf = X_base_tr[:, 34]
row_conflict = ((loc_conf == 1) | (fig_conf == 1) | (inst_mis == 1) | (neg_inv == 1) | (ant_conf == 1))

best_th_clean, best_th_conf, best_gate_f1 = 0.5, 0.5, 0.0
for th_cl in np.arange(0.65, 0.82, 0.005):
    for th_cf in np.arange(0.74, 0.90, 0.005):
        preds = np.where(row_conflict, (meta_oof_hpo >= th_cf).astype(int), (meta_oof_hpo >= th_cl).astype(int))
        sc = f1_score(y_train, preds, average='macro')
        if sc > best_gate_f1:
            best_gate_f1, best_th_clean, best_th_conf = sc, th_cl, th_cf

gated_preds_hpo = np.where(row_conflict, (meta_oof_hpo >= best_th_conf).astype(int), (meta_oof_hpo >= best_th_clean).astype(int))
cr_gated = classification_report(y_train, gated_preds_hpo, output_dict=True)

print(f"Re-tuned Dual-Threshold Gating: Th_clean={best_th_clean:.3f}, Th_conf={best_th_conf:.3f}")
print(f"Re-tuned Gated Ensemble Macro F1: {best_gate_f1:.4f}")
print(f"Class 0 Recall: {cr_gated['0']['recall']:.4f}, Precision: {cr_gated['0']['precision']:.4f}, F1: {cr_gated['0']['f1-score']:.4f}")
print(f"Perbandingan dengan Baseline Resmi (0.7126):")
print(f"  Delta F1 vs Baseline: {best_gate_f1 - 0.7126:+.4f}")

# ==============================================================================
# TUGAS 4: Sanity Check Overfitting HPO (CV Seed = 43)
# ==============================================================================
print("\n" + "="*80)
print("--- TUGAS 4: SANITY CHECK OVERFITTING HPO (SEED 43) ---")
print("="*80)

skf_seed43 = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

xgb_s43_oof = np.zeros(len(y_train))
hgb_s43_oof = np.zeros(len(y_train))
et_s43_oof  = np.zeros(len(y_train))
lr_s43_oof  = np.zeros(len(y_train))

for trn, val in skf_seed43.split(X_train, y_train):
    clf_x = xgb.XGBClassifier(**xgb_params)
    clf_x.fit(X_train[trn], y_train[trn], sample_weight=sample_weights[trn])
    xgb_s43_oof[val] = clf_x.predict_proba(X_train[val])[:, 1]
    
    clf_h = HistGradientBoostingClassifier(**hgb_params)
    clf_h.fit(X_train[trn], y_train[trn], sample_weight=sample_weights[trn])
    hgb_s43_oof[val] = clf_h.predict_proba(X_train[val])[:, 1]
    
    clf_e = ExtraTreesClassifier(**et_params)
    clf_e.fit(X_train[trn], y_train[trn], sample_weight=sample_weights[trn])
    et_s43_oof[val] = clf_e.predict_proba(X_train[val])[:, 1]
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train[trn])
    X_val_sc = scaler.transform(X_train[val])
    clf_l = LogisticRegression(**lr_params)
    clf_l.fit(X_tr_sc, y_train[trn], sample_weight=sample_weights[trn])
    lr_s43_oof[val] = clf_l.predict_proba(X_val_sc)[:, 1]

meta_X_s43 = np.column_stack([xgb_s43_oof, hgb_s43_oof, et_s43_oof, lr_s43_oof])
meta_oof_s43 = np.zeros(len(y_train))

for trn, val in skf_seed43.split(meta_X_s43, y_train):
    meta_clf = LogisticRegression(C=1.0, random_state=43)
    meta_clf.fit(meta_X_s43[trn], y_train[trn])
    meta_oof_s43[val] = meta_clf.predict_proba(meta_X_s43[val])[:, 1]

# Evaluate on seed 43 using re-tuned thresholds
preds_s43_single = (meta_oof_s43 >= cons_th_hpo).astype(int)
f1_s43_single = f1_score(y_train, preds_s43_single, average='macro')

preds_s43_gated = np.where(row_conflict, (meta_oof_s43 >= best_th_conf).astype(int), (meta_oof_s43 >= best_th_clean).astype(int))
f1_s43_gated = f1_score(y_train, preds_s43_gated, average='macro')

delta_single = abs(single_th_f1_hpo - f1_s43_single)
delta_gated  = abs(best_gate_f1 - f1_s43_gated)

print(f"Seed 42 Single Th F1: {single_th_f1_hpo:.4f} | Seed 43 Single Th F1: {f1_s43_single:.4f} | Delta: {delta_single:.4f}")
print(f"Seed 42 Gated F1    : {best_gate_f1:.4f} | Seed 43 Gated F1    : {f1_s43_gated:.4f} | Delta: {delta_gated:.4f}")
is_robust = (delta_gated < 0.010)
print(f"Generalization Check: {'ROBUST (PASS)' if is_robust else 'OVERFITTING WARNING (FAIL)'} (criterion: Delta < 0.010)")

# Save all results to disk
results_summary = {
    'baseline_individual': baseline_individual_f1,
    'default_params': DEFAULT_PARAMS,
    'best_params': {
        'xgb': xgb_best,
        'hgb': hgb_best,
        'et': et_best,
        'lr': lr_best
    },
    'best_individual_f1': {
        'xgb': float(study_xgb.best_value),
        'hgb': float(study_hgb.best_value),
        'et': float(study_et.best_value),
        'lr': float(study_lr.best_value)
    },
    'study_times_sec': study_times,
    'trials_count': {
        'xgb': len(study_xgb.trials),
        'hgb': len(study_hgb.trials),
        'et': len(study_et.trials),
        'lr': len(study_lr.trials)
    },
    'pruned_count': {
        'xgb': len([t for t in study_xgb.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'hgb': len([t for t in study_hgb.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'et': len([t for t in study_et.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'lr': len([t for t in study_lr.trials if t.state == optuna.trial.TrialState.PRUNED])
    },
    'total_hpo_time_sec': total_time_hpo,
    'single_consensus_th': float(cons_th_hpo),
    'single_th_macro_f1': float(single_th_f1_hpo),
    'best_th_clean': float(best_th_clean),
    'best_th_conf': float(best_th_conf),
    'best_gate_macro_f1': float(best_gate_f1),
    'baseline_comparison': {
        'official_baseline_gated_f1': 0.7126,
        'gain': float(best_gate_f1 - 0.7126)
    },
    'sanity_check_seed43': {
        'seed42_gated_f1': float(best_gate_f1),
        'seed43_gated_f1': float(f1_s43_gated),
        'delta': float(delta_gated),
        'is_robust': bool(is_robust)
    }
}

with open('scratch/optuna_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

np.savez(
    'scratch/optuna_results.npz',
    meta_oof_hpo=meta_oof_hpo,
    meta_test_hpo=meta_test_hpo,
    xgb_hpo_oof=xgb_hpo_oof,
    hgb_hpo_oof=hgb_hpo_oof,
    et_hpo_oof=et_hpo_oof,
    lr_hpo_oof=lr_hpo_oof,
    xgb_hpo_te=xgb_hpo_te,
    hgb_hpo_te=hgb_hpo_te,
    et_hpo_te=et_hpo_te,
    lr_hpo_te=lr_hpo_te
)
print("\nSuccessfully saved results to scratch/optuna_results.json and scratch/optuna_results.npz")
