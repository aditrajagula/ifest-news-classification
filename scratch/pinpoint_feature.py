import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb

SEED = 42
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train['label'].values

data = np.load('features_44_clean.npz')
X_ref = data['X_train']

# Load original
import json
nb = json.load(open('ifest_c562eb8.ipynb', encoding='utf-8'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re, random

exec(''.join(nb['cells'][5]['source']))
exec(''.join(nb['cells'][7]['source']))
exec(''.join(nb['cells'][9]['source']))

# X_train is original 45 features
# X_no_head is 44 features (dropping head_entity_absence, col 26)
X_base = np.delete(X_train, 26, axis=1) # 44 cols, F1 = 0.7118

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
    return best_f1

# Let's test swapping each modified feature into X_base one by one:
# 1. Base (X_no_head)
fast_eval(X_base, "Base (X_no_head)")

# 2. Swap v_conf only (Col 40)
X_vconf = X_base.copy()
X_vconf[:, 40] = X_ref[:, 40]
fast_eval(X_vconf, "Swap v_conf only (Col 40)")

# 3. Swap loc features only (Cols 16, 17, 18)
X_loc = X_base.copy()
X_loc[:, 16:19] = X_ref[:, 16:19]
fast_eval(X_loc, "Swap loc features only (Cols 16, 17, 18)")

# 4. Swap comp_conf only (Col 42)
X_comp = X_base.copy()
X_comp[:, 42] = X_ref[:, 42]
fast_eval(X_comp, "Swap comp_conf only (Col 42)")
