import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

print("=== UJI OOF TERISOLASI MODAL EXPANSION ===")
train = pd.read_csv('train.csv')
y_train = train['label'].values

# Check how many extra antonym_conflicts are detected if we add the modal pairs
# Existing EXPANDED_ANTONYM_PAIRS in 42-base features is already in X_base_tr[:, 34]
cache = np.load(r'C:\Users\Reynaldi Hutagaol\.gemini\antigravity-ide\brain\489beaa2-e2ba-4370-a4e6-0110f42de593\scratch\features_cache.npz')
X_base_tr = cache['X_train']
old_antonym_conf = X_base_tr[:, 34]

MODAL_CANDIDATES = [
    ('akan', 'tidak akan'), ('akan', 'tak akan'),
    ('bisa', 'tidak bisa'), ('resmi', 'isu'), ('keputusan', 'rencana')
]

extra_modal_conf = np.zeros(len(train))
for idx, row in train.iterrows():
    t_raw = str(row['title']).lower()
    c_raw = str(row['content']).lower()
    hit = 0.0
    for w1, w2 in MODAL_CANDIDATES:
        if (w1 in t_raw and w2 in c_raw) or (w2 in t_raw and w1 in c_raw):
            hit = 1.0
            break
    extra_modal_conf[idx] = hit

print(f"Extra modal conflict hits: {int(extra_modal_conf.sum())} / {len(train)}")
print(f"  Class 0 hits: {int(extra_modal_conf[y_train==0].sum())} ({extra_modal_conf[y_train==0].mean()*100:.2f}%)")
print(f"  Class 1 hits: {int(extra_modal_conf[y_train==1].sum())} ({extra_modal_conf[y_train==1].mean()*100:.2f}%)")
print(f"Overlap with old antonym_conflict: {((extra_modal_conf == 1) & (old_antonym_conf == 1)).sum()}")
