import sys
import os
import re
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
print("=== Testing comp_conf extraction with IFest.ipynb dictionaries ===")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Load cell 5 from IFest.ipynb
with open('IFest.ipynb', 'r', encoding='utf-8') as f:
    nb = __import__('json').load(f)

exec_scope = {'re': re, 'pd': pd, 'np': np}
exec(''.join(nb['cells'][5]['source']), exec_scope)

INSTITUTIONS = exec_scope['INSTITUTIONS']
LOC_REGEX = exec_scope['LOC_REGEX']

print(f"Loaded {len(INSTITUTIONS)} INSTITUTIONS and LOC_REGEX from IFest.ipynb")

re_inst_marker = re.compile(r'\b(' + '|'.join(re.escape(m) for m in sorted(INSTITUTIONS, key=len, reverse=True)) + r')\b', re.I)

def extract_comp_conf(titles, contents):
    N = len(titles)
    feat = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        t_lower = titles[i].lower()
        c_lower = contents[i].lower()
        m_inst = list(re_inst_marker.finditer(t_lower))
        m_loc_t = list(LOC_REGEX.finditer(t_lower))
        if m_inst and m_loc_t:
            conf = 0.0
            for inst in m_inst:
                for loc in m_loc_t:
                    if 0 <= loc.start() - inst.end() <= 30 or 0 <= inst.start() - loc.end() <= 30:
                        inst_name = inst.group(0)
                        loc_name_t = loc.group(0)
                        idx_c = c_lower.find(inst_name)
                        if idx_c != -1:
                            surrounding = c_lower[max(0, idx_c-30):min(len(c_lower), idx_c+len(inst_name)+30)]
                            locs_c = [m.group(0) for m in LOC_REGEX.finditer(surrounding)]
                            if locs_c and loc_name_t not in locs_c:
                                conf = 1.0
                                break
                if conf == 1.0:
                    break
            feat[i, 0] = conf
    return feat

train_titles = train['title'].fillna('').astype(str).tolist()
train_contents = train['content'].fillna('').astype(str).tolist()

test_titles = test['title'].fillna('').astype(str).tolist()
test_contents = test['content'].fillna('').astype(str).tolist()

comp_tr = extract_comp_conf(train_titles, train_contents)
comp_te = extract_comp_conf(test_titles, test_contents)

print(f"Train comp_conf positive: {comp_tr.sum()} ({comp_tr.mean()*100:.2f}%)")
print(f"Test comp_conf positive : {comp_te.sum()} ({comp_te.mean()*100:.2f}%)")

y_train = train['label'].values
# Correlation with y_train
r = np.corrcoef(comp_tr[:, 0], y_train)[0, 1]
print(f"Correlation with y_train: {r:.4f}")

# Cross-tab with y_train
ct = pd.crosstab(comp_tr[:, 0], y_train)
print("\nCrosstab comp_conf vs y_train:")
print(ct)

np.savez('scratch/comp_conf_features.npz', comp_tr=comp_tr, comp_te=comp_te)
print("\nSaved to scratch/comp_conf_features.npz")
