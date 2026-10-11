import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

ifest_path = 'IFest.ipynb'
backup_path = r'C:\Users\Reynaldi Hutagaol\AppData\Roaming\Antigravity IDE\Backups\824dc8ab3e37371ce3b3319f12d08c13\file\-5ee1e86b'

with open(ifest_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Section 5 Code Cell (index 34)
sec_5_code = """# ==============================================================================
# 5. STRATIFIED 5-FOLD CROSS VALIDATION & 4-MODEL STACKING ENSEMBLE (46 FITUR)
# ==============================================================================
import os
import re
import unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

# 1. Pastikan seluruh variabel dasar tersedia jika sel ini di-run langsung
if 'SEED' not in globals():
    SEED = 42
if 'train' not in globals():
    train = pd.read_csv('train.csv')
if 'test' not in globals():
    test = pd.read_csv('test.csv')
if 'y_train' not in globals():
    y_train = train['label'].values
if 'train_titles' not in globals():
    train_titles = train['title'].fillna('').astype(str).tolist()
if 'train_contents' not in globals():
    train_contents = train['content'].fillna('').astype(str).tolist()
if 'test_titles' not in globals():
    test_titles = test['title'].fillna('').astype(str).tolist()
if 'test_contents' not in globals():
    test_contents = test['content'].fillna('').astype(str).tolist()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sample_weights = np.where(y_train == 0, 2.0, 1.0)  # W=2.0x tervalidasi resmi

# 2. Ekstraksi Fitur 43: city_lead_conflict
if 'PROVINCES_SET' not in globals():
    PROVINCES_SET = {
        'aceh', 'sumatera utara', 'sumatera barat', 'riau', 'jambi', 'sumatera selatan',
        'bengkulu', 'lampung', 'kepulauan bangka belitung', 'kepulauan riau', 'dki jakarta',
        'jawa barat', 'jawa tengah', 'daerah istimewa yogyakarta', 'jawa timur', 'banten',
        'bali', 'nusa tenggara barat', 'nusa tenggara timur', 'kalimantan barat', 'kalimantan tengah',
        'kalimantan selatan', 'kalimantan timur', 'kalimantan utara', 'sulawesi utara', 'sulawesi tengah',
        'sulawesi selatan', 'sulawesi tenggara', 'gorontalo', 'sulawesi barat', 'maluku',
        'maluku utara', 'papua barat', 'papua'
    }

def get_city_conflict_col(titles, contents):
    if os.path.exists('scratch/extra_features.npz') and len(titles) in (14397, 3603):
        extra = np.load('scratch/extra_features.npz')
        if len(titles) == 14397:
            return extra['city_tr']
        elif len(titles) == 3603:
            return extra['city_te']
    
    col = np.zeros((len(titles), 1), dtype=np.float32)
    for i in range(len(titles)):
        t_low = titles[i].lower()
        c_low = contents[i].lower()
        lead_low = ' '.join(c_low.split()[:40])
        t_locs = get_canonical_locations(t_low) if 'get_canonical_locations' in globals() else set()
        lead_locs = get_canonical_locations(lead_low) if 'get_canonical_locations' in globals() else set()
        t_cities = set(l for l in t_locs if l not in PROVINCES_SET)
        lead_cities = set(l for l in lead_locs if l not in PROVINCES_SET)
        if t_cities and lead_cities and not t_cities.intersection(lead_cities):
            col[i, 0] = 1.0
    return col

# 3. Ekstraksi 3 Fitur Content-Twin (Leave-One-Out Safe)
MIN_CONTENT_LEN = 200
TWIN_JACCARD_THRESHOLD = 0.90

def normalize_content(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = re.sub(r'[^\\w\\s]', '', s)
    return re.sub(r'\\s+', ' ', s).strip()

def tokenize_title(t):
    t = unicodedata.normalize('NFKC', str(t)).lower()
    t = re.sub(r'[^\\w\\s]', '', t)
    return set(t.split())

def jaccard_titles(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def build_twin_features_for_train(df, min_len=200, twin_th=0.90):
    c1_df = df[(df['label'] == 1) & (df['content_norm'].str.len() >= min_len)]
    c1_dict = defaultdict(list)
    for idx, row in c1_df.iterrows():
        c1_dict[row['content_norm']].append((idx, row['title_tokens']))
        
    twin_feats = np.zeros((len(df), 3), dtype=np.float32)
    for idx, row in df.iterrows():
        c_norm = row['content_norm']
        if len(c_norm) < min_len:
            continue
        entries = c1_dict.get(c_norm, [])
        ref_tokens = [tok_set for r_idx, tok_set in entries if r_idx != idx]
        if not ref_tokens:
            continue
        own_tok = row['title_tokens']
        best_sim = max(jaccard_titles(own_tok, rt) for rt in ref_tokens)
        twin_feats[idx, 0] = 1.0
        twin_feats[idx, 1] = best_sim
        twin_feats[idx, 2] = 1.0 if best_sim < twin_th else 0.0
    return twin_feats

def build_twin_features_for_test(df_tr, df_te, min_len=200, twin_th=0.90):
    c1_df = df_tr[(df_tr['label'] == 1) & (df_tr['content_norm'].str.len() >= min_len)]
    c1_dict = defaultdict(list)
    for idx, row in c1_df.iterrows():
        c1_dict[row['content_norm']].append((idx, row['title_tokens']))
        
    twin_feats = np.zeros((len(df_te), 3), dtype=np.float32)
    for idx, row in df_te.iterrows():
        c_norm = row['content_norm']
        if len(c_norm) < min_len:
            continue
        entries = c1_dict.get(c_norm, [])
        if not entries:
            continue
        own_tok = row['title_tokens']
        best_sim = max(jaccard_titles(own_tok, rt) for _, rt in entries)
        twin_feats[idx, 0] = 1.0
        twin_feats[idx, 1] = best_sim
        twin_feats[idx, 2] = 1.0 if best_sim < twin_th else 0.0
    return twin_feats

# 4. Bangun Matriks Fitur Resmi 46-Fitur (X_train_46 & X_test_46)
if os.path.exists('scratch/full_rebuild_arrays.npz'):
    _arrs = np.load('scratch/full_rebuild_arrays.npz')
    X_train_46 = _arrs['X_train_46']
    X_test_46  = _arrs['X_test_46']
    X_train_43 = X_train_46[:, :43]
    X_test_43  = X_test_46[:, :43]
else:
    if 'X_train' not in globals() or 'X_test' not in globals():
        if os.path.exists('features_cache.npz'):
            _cache = np.load('features_cache.npz')
            X_train = _cache['X_train']
            X_test  = _cache['X_test']
    city_col_tr = get_city_conflict_col(train_titles, train_contents)
    city_col_te = get_city_conflict_col(test_titles, test_contents)
    X_train_43 = np.hstack([X_train, city_col_tr])
    X_test_43  = np.hstack([X_test, city_col_te])
    
    train['content_norm'] = train['content'].apply(normalize_content)
    test['content_norm']  = test['content'].apply(normalize_content)
    train['title_tokens'] = train['title'].apply(tokenize_title)
    test['title_tokens']  = test['title'].apply(tokenize_title)
    
    twin_cols_tr = build_twin_features_for_train(train)
    twin_cols_te = build_twin_features_for_test(train, test)
    X_train_46 = np.hstack([X_train_43, twin_cols_tr])
    X_test_46  = np.hstack([X_test_43, twin_cols_te])

assert X_train_46.shape[1] == 46, f"WAJIB 46 kolom, dapat {X_train_46.shape[1]}"
assert X_test_46.shape[1] == 46, f"WAJIB 46 kolom, dapat {X_test_46.shape[1]}"
print(f"Verifikasi dimensi fitur: X_train_46={X_train_46.shape}, X_test_46={X_test_46.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_46)
X_test_scaled  = scaler.transform(X_test_46)

# 5. Training CV 5-Fold Baseline 46-Fitur (Default Hyperparameters, W=2.0x)
xgb_oof = np.zeros(len(train))
hgb_oof = np.zeros(len(train))
et_oof  = np.zeros(len(train))
lr_oof  = np.zeros(len(train))

xgb_test = np.zeros(len(test))
hgb_test = np.zeros(len(test))
et_test  = np.zeros(len(test))
lr_test  = np.zeros(len(test))

last_clf_xgb = None

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_46, y_train)):
    clf_xgb = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.035,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss', n_jobs=-1
    )
    clf_xgb.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    xgb_oof[val_idx] = clf_xgb.predict_proba(X_train_46[val_idx])[:, 1]
    xgb_test += clf_xgb.predict_proba(X_test_46)[:, 1] / 5.0
    last_clf_xgb = clf_xgb
    
    clf_hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.035, max_depth=6, random_state=SEED)
    clf_hgb.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    hgb_oof[val_idx] = clf_hgb.predict_proba(X_train_46[val_idx])[:, 1]
    hgb_test += clf_hgb.predict_proba(X_test_46)[:, 1] / 5.0
    
    clf_et = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=SEED, n_jobs=-1)
    clf_et.fit(X_train_46[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    et_oof[val_idx] = clf_et.predict_proba(X_train_46[val_idx])[:, 1]
    et_test += clf_et.predict_proba(X_test_46)[:, 1] / 5.0
    
    clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf_lr.fit(X_train_scaled[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    lr_oof[val_idx] = clf_lr.predict_proba(X_train_scaled[val_idx])[:, 1]
    lr_test += clf_lr.predict_proba(X_test_scaled)[:, 1] / 5.0

meta_X = np.column_stack([xgb_oof, hgb_oof, et_oof, lr_oof])
meta_X_test = np.column_stack([xgb_test, hgb_test, et_test, lr_test])

meta_oof = np.zeros(len(train))
meta_test = np.zeros(len(test))

for trn, val in skf.split(meta_X, y_train):
    meta_clf = LogisticRegression(C=1.0, random_state=SEED)
    meta_clf.fit(meta_X[trn], y_train[trn])
    meta_oof[val] = meta_clf.predict_proba(meta_X[val])[:, 1]
    meta_test += meta_clf.predict_proba(meta_X_test)[:, 1] / 5.0

best_def_f1, def_th = 0.0, 0.5
for th in np.arange(0.3, 0.9, 0.005):
    sc = f1_score(y_train, (meta_oof >= th).astype(int), average='macro')
    if sc > best_def_f1:
        best_def_f1, def_th = sc, th

print(f"\\nHASIL LIVE SECTION 5 (Baseline 46-Fitur Default Params, W=2.0x):")
print(f"  Single Consensus Threshold = {def_th:.3f} | Macro F1 OOF = {best_def_f1:.4f}")
"""

nb['cells'][34]['source'] = [l + '\n' for l in sec_5_code.strip().split('\n')]

with open(ifest_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Successfully updated Section 5 in {ifest_path}")

if os.path.exists(os.path.dirname(backup_path)):
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully synchronized backup buffer!")
