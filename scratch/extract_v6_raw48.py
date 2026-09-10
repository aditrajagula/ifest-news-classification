import sys
import os
import re
import json
import time
import numpy as np
import pandas as pd
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42

print("=== Loading data ===")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train['label'].values

# Load cells from IFest (6).ipynb
with open(r'C:\Users\Reynaldi Hutagaol\Downloads\IFest (6).ipynb', 'r', encoding='utf-8', errors='ignore') as f:
    nb = json.load(f)

# Execute cell 5 (kamus) and cell 7 (tfidf & bm25)
exec_scope = {'pd': pd, 'np': np, 're': re, 'Counter': Counter, 'math': math, 'os': os, 'SEED': SEED, 'train': train, 'test': test, 'y_train': y_train}

# Prepare titles and contents
train_titles = train['title'].fillna('').astype(str).tolist()
train_contents = train['content'].fillna('').astype(str).tolist()
train_fs = [' '.join(c.split()[:40]) for c in train_contents]

test_titles = test['title'].fillna('').astype(str).tolist()
test_contents = test['content'].fillna('').astype(str).tolist()
test_fs = [' '.join(c.split()[:40]) for c in test_contents]

exec_scope['train_titles'] = train_titles
exec_scope['train_contents'] = train_contents
exec_scope['train_fs'] = train_fs

exec_scope['test_titles'] = test_titles
exec_scope['test_contents'] = test_contents
exec_scope['test_fs'] = test_fs

print("=== Executing Imports, Kamus & TF-IDF setup from IFest (6).ipynb ===")
# Cell 1 (imports)
exec(''.join(nb['cells'][1]['source']), exec_scope)

# Cell 5
exec(''.join(nb['cells'][5]['source']), exec_scope)

# Cell 7
exec(''.join(nb['cells'][7]['source']), exec_scope)

# In Cell 9, grab extract_features
cell9_src = ''.join(nb['cells'][9]['source'])
func_end = cell9_src.find('print("Mengekstrak 48 fitur')
extract_func_code = cell9_src[:func_end]
exec(extract_func_code, exec_scope)

print("=== Extracting 48 Raw Features on Train ===")
t0 = time.time()
X_train_raw = exec_scope['extract_features'](
    train_titles, train_contents, train_fs,
    exec_scope['cos_w_c_tr'], exec_scope['cos_w_fs_tr'], exec_scope['cos_c_c_tr'], exec_scope['cos_c_fs_tr'],
    exec_scope['bm25_c_tr'], exec_scope['bm25_fs_tr'],
    exec_scope['cos_lsa_c_tr'], exec_scope['cos_lsa_fs_tr'],
    exec_scope['tr_t_w']
)
print(f"Extraction completed in {time.time() - t0:.2f} seconds! Shape: {X_train_raw.shape}")

np.savez('scratch/X_train_raw_48.npz', X_train_raw=X_train_raw, y_train=y_train)
print("Saved to scratch/X_train_raw_48.npz")
