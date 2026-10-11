import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
import sys
sys.stdout.reconfigure(encoding='utf-8')

train = pd.read_csv('train.csv')
y_train = train['label'].values
SEED = 42
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sample_weights = np.where(y_train == 0, 2.0, 1.0)

arrs = np.load('scratch/full_rebuild_arrays.npz')
X_train_46_global = arrs['X_train_46']

# Load strict k-fold features
from scratch.check_twin_leakage import feats_strict
X_train_46_strict = np.hstack([X_train_46_global[:, :43], feats_strict])

study_xgb_best = {'n_estimators': 550, 'max_depth': 3, 'learning_rate': 0.016014316489551022, 'subsample': 0.6, 'colsample_bytree': 0.6, 'min_child_weight': 4, 'reg_lambda': 0.12569828566858568}
xgb_params = dict(study_xgb_best, random_state=SEED, eval_metric='logloss', n_jobs=-1)

def eval_pipeline(X_mat):
    oof = np.zeros(len(train))
    for trn, val in skf.split(X_mat, y_train):
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_mat[trn], y_train[trn], sample_weight=sample_weights[trn])
        oof[val] = clf.predict_proba(X_mat[val])[:, 1]
    
    best_f1, best_th = 0.0, 0.5
    for th in np.arange(0.3, 0.9, 0.01):
        sc = f1_score(y_train, (oof >= th).astype(int), average='macro')
        if sc > best_f1:
            best_f1, best_th = sc, th
    return best_f1, best_th

f1_glob, th_glob = eval_pipeline(X_train_46_global)
f1_strc, th_strc = eval_pipeline(X_train_46_strict)

print(f"Global Twin OOF F1: {f1_glob:.4f} (th={th_glob:.2f})")
print(f"Strict K-Fold Twin OOF F1: {f1_strc:.4f} (th={th_strc:.2f})")
print(f"Difference: {abs(f1_glob - f1_strc):.4f}")
