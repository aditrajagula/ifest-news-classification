import sys
import os
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

print("=== Loading data and features ===")
train = pd.read_csv('train.csv')
y_train = train['label'].values

cache = np.load(r'C:\Users\Reynaldi Hutagaol\.gemini\antigravity-ide\brain\489beaa2-e2ba-4370-a4e6-0110f42de593\scratch\features_cache.npz')
X_base_tr = cache['X_train'] # (14397, 42)
X_base_te = cache['X_test']  # (3603, 42)

extra = np.load('scratch/extra_features.npz')
city_tr = extra['city_tr'] # (14397, 1)
city_te = extra['city_te']

comp = np.load('scratch/comp_conf_features.npz')
comp_tr = comp['comp_tr'] # (14397, 1)
comp_te = comp['comp_te']

configurations = {
    '1. Baseline (42 Fitur)': (X_base_tr, X_base_te),
    '2. Baseline + city_lead_conflict (43 Fitur)': (np.hstack([X_base_tr, city_tr]), np.hstack([X_base_te, city_te])),
    '3. Baseline + comp_conf (43 Fitur)': (np.hstack([X_base_tr, comp_tr]), np.hstack([X_base_te, comp_te])),
    '4. Baseline + city + comp_conf (44 Fitur)': (np.hstack([X_base_tr, city_tr, comp_tr]), np.hstack([X_base_te, city_te, comp_te]))
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def evaluate_config(X_tr, X_te, weight_c0):
    sample_weights = np.where(y_train == 0, weight_c0, 1.0)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    xgb_oof = np.zeros(len(y_train))
    hgb_oof = np.zeros(len(y_train))
    et_oof  = np.zeros(len(y_train))
    lr_oof  = np.zeros(len(y_train))

    xgb_te = np.zeros(len(X_te))
    hgb_te = np.zeros(len(X_te))
    et_te  = np.zeros(len(X_te))
    lr_te  = np.zeros(len(X_te))

    for trn, val in skf.split(X_tr, y_train):
        clf_xgb = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.035, subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss', n_jobs=4)
        clf_xgb.fit(X_tr[trn], y_train[trn], sample_weight=sample_weights[trn])
        xgb_oof[val] = clf_xgb.predict_proba(X_tr[val])[:, 1]
        xgb_te += clf_xgb.predict_proba(X_te)[:, 1] / 5.0

        clf_hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.035, max_depth=6, random_state=SEED)
        clf_hgb.fit(X_tr[trn], y_train[trn], sample_weight=sample_weights[trn])
        hgb_oof[val] = clf_hgb.predict_proba(X_tr[val])[:, 1]
        hgb_te += clf_hgb.predict_proba(X_te)[:, 1] / 5.0

        clf_et = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=SEED, n_jobs=4)
        clf_et.fit(X_tr[trn], y_train[trn], sample_weight=sample_weights[trn])
        et_oof[val] = clf_et.predict_proba(X_tr[val])[:, 1]
        et_te += clf_et.predict_proba(X_te)[:, 1] / 5.0

        clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
        clf_lr.fit(X_tr_sc[trn], y_train[trn], sample_weight=sample_weights[trn])
        lr_oof[val] = clf_lr.predict_proba(X_tr_sc[val])[:, 1]
        lr_te += clf_lr.predict_proba(X_te_sc)[:, 1] / 5.0

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
        for th in np.arange(0.750, 0.835, 0.005):
            sc = f1_score(y_train[val], (val_probs >= th).astype(int), average='macro')
            if sc > f_f1:
                f_f1 = sc
                f_th = th
        fold_ths.append(round(float(f_th), 3))

    cons_th = round(float(np.median(fold_ths)), 3)
    oof_preds = (meta_oof >= cons_th).astype(int)
    macro_f1 = f1_score(y_train, oof_preds, average='macro')
    cr = classification_report(y_train, oof_preds, output_dict=True)

    return {
        'Cons_Th': cons_th,
        'Macro_F1': macro_f1,
        'Rec_0': cr['0']['recall'],
        'Prec_0': cr['0']['precision'],
        'F1_0': cr['0']['f1-score'],
        'meta_oof': meta_oof,
        'meta_test': meta_test
    }

print("\n" + "="*90)
print("=== PENGUJIAN LENGKAP: SAMPLE_WEIGHT 2.0x (BASELINE RESMI AUDIT LAMA) ===")
print("="*90)
results_w2 = []
for name, (X_tr, X_te) in configurations.items():
    print(f"Running W=2.0 for: {name}...")
    res = evaluate_config(X_tr, X_te, weight_c0=2.0)
    print(f"  -> Th={res['Cons_Th']:.3f} | Macro F1={res['Macro_F1']:.4f} | Rec_0={res['Rec_0']:.4f} | Prec_0={res['Prec_0']:.4f} | F1_0={res['F1_0']:.4f}")
    results_w2.append({
        'Konfigurasi': name,
        'Dim': X_tr.shape[1],
        'Weight_C0': 2.0,
        'Consensus_Th': res['Cons_Th'],
        'Macro F1': round(res['Macro_F1'], 4),
        'Recall C0': round(res['Rec_0'], 4),
        'Precision C0': round(res['Prec_0'], 4),
        'F1 C0': round(res['F1_0'], 4)
    })

print("\n" + "="*90)
print("=== PENGUJIAN LENGKAP: SAMPLE_WEIGHT 3.5x (KONFIGURASI REVISI KRITIS) ===")
print("="*90)
results_w35 = []
saved_models = {}
for name, (X_tr, X_te) in configurations.items():
    print(f"Running W=3.5 for: {name}...")
    res = evaluate_config(X_tr, X_te, weight_c0=3.5)
    print(f"  -> Th={res['Cons_Th']:.3f} | Macro F1={res['Macro_F1']:.4f} | Rec_0={res['Rec_0']:.4f} | Prec_0={res['Prec_0']:.4f} | F1_0={res['F1_0']:.4f}")
    saved_models[name] = res
    results_w35.append({
        'Konfigurasi': name,
        'Dim': X_tr.shape[1],
        'Weight_C0': 3.5,
        'Consensus_Th': res['Cons_Th'],
        'Macro F1': round(res['Macro_F1'], 4),
        'Recall C0': round(res['Rec_0'], 4),
        'Precision C0': round(res['Prec_0'], 4),
        'F1 C0': round(res['F1_0'], 4)
    })

print("\n" + "="*90)
print("HASIL EVALUASI LENGKAP W=2.0x:")
print(pd.DataFrame(results_w2).to_string(index=False))
print("\nHASIL EVALUASI LENGKAP W=3.5x:")
print(pd.DataFrame(results_w35).to_string(index=False))

# Save results
np.savez('scratch/reconciliation_results.npz',
         df_w2=pd.DataFrame(results_w2).to_json(),
         df_w35=pd.DataFrame(results_w35).to_json())
print("\nSaved reconciliation results to scratch/reconciliation_results.npz")
