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

# Load original X_train
nb = json.load(open('ifest_c562eb8.ipynb', encoding='utf-8'))
exec(''.join(nb['cells'][5]['source']))
exec(''.join(nb['cells'][7]['source']))
exec(''.join(nb['cells'][9]['source']))

# X_train is original 45 features
data = np.load('features_44_clean.npz')
X_ref = data['X_train']

def fast_eval(X, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sample_weights = np.ones(len(train))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    xgb_oof = np.zeros(len(train))
    hgb_oof = np.zeros(len(train))
    et_oof = np.zeros(len(train))
    lr_oof = np.zeros(len(train))
    
    for trn, val in skf.split(X, y_train):
        clf_xgb = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.035, subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss')
        clf_xgb.fit(X[trn], y_train[trn], sample_weight=sample_weights[trn])
        xgb_oof[val] = clf_xgb.predict_proba(X[val])[:, 1]
        
        clf_hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.035, max_depth=6, random_state=SEED)
        clf_hgb.fit(X[trn], y_train[trn], sample_weight=sample_weights[trn])
        hgb_oof[val] = clf_hgb.predict_proba(X[val])[:, 1]
        
        clf_et = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=SEED, n_jobs=-1)
        clf_et.fit(X[trn], y_train[trn], sample_weight=sample_weights[trn])
        et_oof[val] = clf_et.predict_proba(X[val])[:, 1]
        
        clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
        clf_lr.fit(X_scaled[trn], y_train[trn], sample_weight=sample_weights[trn])
        lr_oof[val] = clf_lr.predict_proba(X_scaled[val])[:, 1]
        
    meta_X = np.column_stack([xgb_oof, hgb_oof, et_oof, lr_oof])
    meta_oof = np.zeros(len(train))
    for trn, val in skf.split(meta_X, y_train):
        meta_clf = LogisticRegression(C=1.0, random_state=SEED)
        meta_clf.fit(meta_X[trn], y_train[trn])
        meta_oof[val] = meta_clf.predict_proba(meta_X[val])[:, 1]
        
    best_th, best_f1 = 0.810, 0.0
    for th in np.arange(0.780, 0.845, 0.005):
        sc = f1_score(y_train, (meta_oof >= th).astype(int), average='macro')
        if sc > best_f1:
            best_f1, best_th = sc, th
    cm = confusion_matrix(y_train, (meta_oof >= best_th).astype(int))
    rec_c0 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    prec_c0 = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print(f"[{name}] Best Th: {best_th:.3f} | Macro F1: {best_f1:.4f} | FP: {cm[1, 0]} | Rec C0: {rec_c0*100:.2f}% | Prec C0: {prec_c0*100:.2f}%")
    return best_f1, best_th, cm[1, 0]

# 1. Original 45
fast_eval(X_train, "1. Original X_train (45 feats)")

# 2. X_train dropping head_entity_absence (index 26)
X_no_head = np.delete(X_train, 26, axis=1)
fast_eval(X_no_head, "2. X_train drop head_entity_absence (44 feats)")

# 3. Refined 44 features
fast_eval(X_ref, "3. X_ref refined (44 feats)")

# 4. Refined 44 + head_entity_absence (45 feats)
head_col = X_train[:, 26:27]
X_ref_with_head = np.hstack([X_ref[:, :26], head_col, X_ref[:, 26:]])
fast_eval(X_ref_with_head, "4. X_ref with head_entity_absence kept (45 feats)")
