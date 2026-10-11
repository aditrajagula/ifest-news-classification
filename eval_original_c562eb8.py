import os
import random
import re
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train['label'].values

# Load from c562eb8 notebook directly
import json
nb = json.load(open('ifest_c562eb8.ipynb', encoding='utf-8'))

# Cell 5: lexical
exec(''.join(nb['cells'][5]['source']))

# Cell 7: TF-IDF, BM25, SVD
exec(''.join(nb['cells'][7]['source']))

# Let's define the original extract_features from c562eb8
exec(''.join(nb['cells'][9]['source']))

print("Features extracted in c562eb8: shape =", X_train.shape)

# Evaluate original c562eb8
def eval_stacking(X, name):
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
    return best_f1, best_th, cm, meta_oof

eval_stacking(X_train, "Original c562eb8")
