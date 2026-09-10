import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb

train = pd.read_csv('train.csv')
y_train = train['label'].values
SEED = 42

# Load baseline 45 features
b45 = np.load('features_45_cache.npz')
X45 = b45['X_train']

print(f"X45 shape: {X45.shape}")

# Feature names in 45 features:
# 0..7: similarities
# 8: best_win_tfidf_cos
# 9..24: discourse & tokens
# 25: head_entity_absence
# 26: num_unit_match_ratio
# 27: neg_inversion
# 28: lead_neg_asymmetry
# ...
# 39: v_conf

def evaluate_features(X, name=""):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    xgb_oof = np.zeros(len(X))
    hgb_oof = np.zeros(len(X))
    et_oof = np.zeros(len(X))
    lr_oof = np.zeros(len(X))
    
    for trn, val in skf.split(X, y_train):
        c_xgb = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.035, subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss')
        c_xgb.fit(X[trn], y_train[trn])
        xgb_oof[val] = c_xgb.predict_proba(X[val])[:, 1]
        
        c_hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.035, max_depth=6, random_state=SEED)
        c_hgb.fit(X[trn], y_train[trn])
        hgb_oof[val] = c_hgb.predict_proba(X[val])[:, 1]
        
        c_et = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=SEED, n_jobs=-1)
        c_et.fit(X[trn], y_train[trn])
        et_oof[val] = c_et.predict_proba(X[val])[:, 1]
        
        c_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
        c_lr.fit(X_scaled[trn], y_train[trn])
        lr_oof[val] = c_lr.predict_proba(X_scaled[val])[:, 1]
        
    meta_X = np.column_stack([xgb_oof, hgb_oof, et_oof, lr_oof])
    meta_oof = np.zeros(len(X))
    for trn, val in skf.split(meta_X, y_train):
        m_lr = LogisticRegression(C=1.0, random_state=SEED)
        m_lr.fit(meta_X[trn], y_train[trn])
        meta_oof[val] = m_lr.predict_proba(meta_X[val])[:, 1]
        
    best_th, best_f1 = 0.810, 0.0
    for th in np.arange(0.780, 0.840, 0.005):
        sc = f1_score(y_train, (meta_oof >= th).astype(int), average='macro')
        if sc > best_f1:
            best_f1, best_th = sc, th
            
    cm = confusion_matrix(y_train, (meta_oof >= best_th).astype(int))
    print(f"[{name}] Best Th: {best_th:.3f} | Macro F1: {best_f1:.4f} | FP: {cm[1,0]} | Recall C0: {cm[0,0]}/{cm[0,0]+cm[0,1]} ({cm[0,0]/(cm[0,0]+cm[0,1])*100:.2f}%)")
    return best_f1, cm

# Test 1: Original 45 features
evaluate_features(X45, "1. Original 45 Features")

# Test 2: Drop column 26 (head_entity_absence in X45: col 8 was inserted, so head_entity_absence is col 26)
# Let's check which column is head_entity_absence
# In extract_features:
# col 0..7
# insert at 8: best_win_tfidf_cos -> now col 8
# items 9..24 (16 items)
# item 25 was head_entity_absence -> in X45, it is index 25+1 = 26!
X_no_head = np.delete(X45, 26, axis=1)
evaluate_features(X_no_head, "2. Drop head_entity_absence only (44 feats)")
