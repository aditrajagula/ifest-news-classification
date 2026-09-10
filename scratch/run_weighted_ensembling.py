import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import xgboost as xgb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train['label'].values

# Load 44 optimal features
data = np.load('optimal_refined_oof_preds.npz')
X_train = data['X_opt_tr']
X_test = data['X_opt_te']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sample_weights = np.ones(len(train))

xgb_oof = np.zeros(len(train))
hgb_oof = np.zeros(len(train))
et_oof = np.zeros(len(train))
lr_oof = np.zeros(len(train))

xgb_test = np.zeros(len(test))
hgb_test = np.zeros(len(test))
et_test = np.zeros(len(test))
lr_test = np.zeros(len(test))

print("Melatih 4 Model Dasar dengan Stratified 5-Fold Cross Validation...")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"--- Training Fold {fold+1} (XGBoost, HistGradientBoosting, ExtraTrees, LogisticRegression) ---")
    
    clf_xgb = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.035,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss'
    )
    clf_xgb.fit(X_train[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    xgb_oof[val_idx] = clf_xgb.predict_proba(X_train[val_idx])[:, 1]
    xgb_test += clf_xgb.predict_proba(X_test)[:, 1] / 5.0
    
    clf_hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.035, max_depth=6, random_state=SEED
    )
    clf_hgb.fit(X_train[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    hgb_oof[val_idx] = clf_hgb.predict_proba(X_train[val_idx])[:, 1]
    hgb_test += clf_hgb.predict_proba(X_test)[:, 1] / 5.0
    
    clf_et = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=SEED, n_jobs=-1)
    clf_et.fit(X_train[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    et_oof[val_idx] = clf_et.predict_proba(X_train[val_idx])[:, 1]
    et_test += clf_et.predict_proba(X_test)[:, 1] / 5.0
    
    clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf_lr.fit(X_train_scaled[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    lr_oof[val_idx] = clf_lr.predict_proba(X_train_scaled[val_idx])[:, 1]
    lr_test += clf_lr.predict_proba(X_test_scaled)[:, 1] / 5.0

# Simpan OOF base models untuk reuse cepat
np.savez('base_models_oof.npz',
    xgb_oof=xgb_oof, hgb_oof=hgb_oof, et_oof=et_oof, lr_oof=lr_oof,
    xgb_test=xgb_test, hgb_test=hgb_test, et_test=et_test, lr_test=lr_test
)

# --- REVISI CELL 6: Weighted Ensembling (Menggantikan Meta-Learner yang Overfit) ---

# 1. Agregasi Out-of-Fold (OOF) secara terbobot
# Beri bobot tinggi pada Gradient Boosting karena jauh lebih akurat pada data diskursus
w_xgb  = 0.40
w_hgb  = 0.35
w_et   = 0.15
w_lr   = 0.10

meta_oof = (w_xgb * xgb_oof) + (w_hgb * hgb_oof) + (w_et * et_oof) + (w_lr * lr_oof)
meta_test = (w_xgb * xgb_test) + (w_hgb * hgb_test) + (w_et * et_test) + (w_lr * lr_test)

# 2. Scanning Threshold Halus (Rentang Kritis 0.75 - 0.88)
best_th, best_f1 = 0.810, 0.0
for th in np.linspace(0.75, 0.88, 261):
    preds = (meta_oof >= th).astype(int)
    sc = f1_score(y_train, preds, average='macro')
    if sc > best_f1:
        best_f1, best_th = sc, th

print("\n" + "="*50)
print(f"HASIL KALIBRASI OPTIMAL THRESHOLD: {best_th:.3f} | MACRO F1: {best_f1:.4f}")
print("="*50)

oof_preds = (meta_oof >= best_th).astype(int)
print(classification_report(y_train, oof_preds, digits=4))
print("Confusion Matrix (Optimal Threshold):")
print(confusion_matrix(y_train, oof_preds))

# Analisis Distribusi Test
test_preds = (meta_test >= best_th).astype(int)
c0_count = (test_preds == 0).sum()
c1_count = (test_preds == 1).sum()
print("\n" + "="*50)
print(f"Distribusi Prediksi Data Test pada Best Th ({best_th:.3f}):")
print(f"  Kelas 1: {c1_count} ({c1_count/len(test)*100:.2f}%)")
print(f"  Kelas 0: {c0_count} ({c0_count/len(test)*100:.2f}%)")
print("="*50)
