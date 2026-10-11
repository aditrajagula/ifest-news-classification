import os
import json
import numpy as np
import pandas as pd
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train['label'].values

nb = json.load(open('ifest_c562eb8.ipynb', encoding='utf-8'))
SEED = 42
exec(''.join(nb['cells'][5]['source']))
exec(''.join(nb['cells'][7]['source']))
exec(''.join(nb['cells'][9]['source']))

# X_train is original 45 features
data = np.load('features_44_clean.npz')
X_ref = data['X_train']

print('Orig shape:', X_train.shape, 'Ref shape:', X_ref.shape)

# Let's see differences between original and refined columns
# In refined we dropped head_entity_absence (index 15 in original extract_features)
orig_cols = [
    'cos_wc', 'cos_wfs', 'cos_cc', 'cos_cfs',
    'bm25_c', 'bm25_fs',
    'cos_lsa_c', 'cos_lsa_fs',
    'best_win_tfidf_cos', # inserted at 8
    'best_50_cov', 'best_win_pos_norm', 'best_minus_lead',
    'lead_25_cov', 'lead_50_cov', 'lead_80_cov', 'top1_top2_contrast',
    'loc_mismatch', 'loc_conflict', 'loc_c_conflict',
    'fig_mismatch', 'fig_conflict',
    'inst_mismatch',
    'tok_cov_exact', 'tok_cov_stem', 'stem_gain', 'head_cov_c',
    'head_entity_absence', 'num_unit_match_ratio',
    'neg_inversion', 'lead_neg_asymmetry', 'ada_conflict', 'antonym_conflict',
    'refuting_score', 'ordered_bigram_ratio',
    'num_missing_subst', 'num_cov_subst',
    'cap_cov_c', 'cap_cov_fs', 'len_ratio', 'len_t_words', 'len_c_words',
    'v_conf', 'e_conf', 'comp_conf', 'dir_conf'
]

ref_cols = [c for c in orig_cols if c != 'head_entity_absence']

print(f"Orig count: {len(orig_cols)}, Ref count: {len(ref_cols)}")

for i, col in enumerate(ref_cols):
    orig_idx = orig_cols.index(col)
    diff = np.abs(X_train[:, orig_idx] - X_ref[:, i])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    num_diff = np.sum(diff > 1e-6)
    if num_diff > 0:
        print(f"Col {i:2d} ({col:<22}): {num_diff:5d} differences, max diff = {max_diff:.4f}, mean diff = {mean_diff:.4f}")
    else:
        print(f"Col {i:2d} ({col:<22}): EXACT MATCH")

