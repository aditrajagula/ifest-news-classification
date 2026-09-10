import pandas as pd
import numpy as np
import unicodedata, re
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import sys
sys.stdout.reconfigure(encoding='utf-8')

train = pd.read_csv('train.csv')
SEED = 42

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

train['content_norm'] = [normalize_content(c) for c in train['content'].fillna('').astype(str)]
train['title_tokens'] = [tokenize_title(t) for t in train['title'].fillna('').astype(str)]

# 1. Global (Current implementation in notebook)
def build_twin_global(df, min_len=200, twin_th=0.90):
    c1_df = df[(df['label'] == 1) & (df['content_norm'].str.len() >= min_len)]
    c1_dict = defaultdict(list)
    for idx, row in c1_df.iterrows():
        c1_dict[row['content_norm']].append((idx, row['title_tokens']))
    twin_feats = np.zeros((len(df), 3), dtype=np.float32)
    for idx, row in df.iterrows():
        c_norm = row['content_norm']
        if len(c_norm) < min_len: continue
        entries = c1_dict.get(c_norm, [])
        ref_tokens = [tok_set for r_idx, tok_set in entries if r_idx != idx]
        if not ref_tokens: continue
        best_sim = max(jaccard_titles(row['title_tokens'], rt) for rt in ref_tokens)
        twin_feats[idx, 0] = 1.0
        twin_feats[idx, 1] = best_sim
        twin_feats[idx, 2] = 1.0 if best_sim < twin_th else 0.0
    return twin_feats

# 2. Strict K-Fold Out-of-Fold (Zero Target Leakage)
def build_twin_strict_kfold(df, skf, min_len=200, twin_th=0.90):
    twin_feats = np.zeros((len(df), 3), dtype=np.float32)
    y = df['label'].values
    for fold, (trn_idx, val_idx) in enumerate(skf.split(df, y)):
        trn_df = df.iloc[trn_idx]
        c1_trn = trn_df[(trn_df['label'] == 1) & (trn_df['content_norm'].str.len() >= min_len)]
        c1_dict = defaultdict(list)
        for idx, row in c1_trn.iterrows():
            c1_dict[row['content_norm']].append((idx, row['title_tokens']))
            
        val_df = df.iloc[val_idx]
        for idx, row in val_df.iterrows():
            c_norm = row['content_norm']
            if len(c_norm) < min_len: continue
            entries = c1_dict.get(c_norm, [])
            if not entries: continue
            best_sim = max(jaccard_titles(row['title_tokens'], rt) for _, rt in entries)
            twin_feats[idx, 0] = 1.0
            twin_feats[idx, 1] = best_sim
            twin_feats[idx, 2] = 1.0 if best_sim < twin_th else 0.0
    return twin_feats

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("Computing Global Twin Features...")
feats_global = build_twin_global(train)
print("Computing Strict K-Fold Twin Features...")
feats_strict = build_twin_strict_kfold(train, skf)

diff_count = np.sum(feats_global != feats_strict, axis=0)
print(f"Total rows: {len(train)}")
print(f"Differences between Global and Strict K-Fold:")
print(f"  twin_has_match diffs : {diff_count[0]} rows ({(diff_count[0]/len(train))*100:.2f}%)")
print(f"  twin_jaccard diffs   : {diff_count[1]} rows ({(diff_count[1]/len(train))*100:.2f}%)")
print(f"  twin_conflict diffs  : {diff_count[2]} rows ({(diff_count[2]/len(train))*100:.2f}%)")
