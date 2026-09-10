import pandas as pd
import numpy as np
import unicodedata, re
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

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

test['content_norm'] = [normalize_content(c) for c in test['content'].fillna('').astype(str)]
test['title_tokens'] = [tokenize_title(t) for t in test['title'].fillna('').astype(str)]

c1_df = train[(train['label'] == 1) & (train['content_norm'].str.len() >= 200)]
c1_dict = defaultdict(list)
for idx, row in c1_df.iterrows():
    c1_dict[row['content_norm']].append((idx, row['title_tokens']))

test_matches = 0
test_conflicts = 0
for idx, row in test.iterrows():
    c_norm = row['content_norm']
    if len(c_norm) < 200: continue
    entries = c1_dict.get(c_norm, [])
    if not entries: continue
    test_matches += 1
    best_sim = max(jaccard_titles(row['title_tokens'], rt) for _, rt in entries)
    if best_sim < 0.90:
        test_conflicts += 1

print(f"Total rows in test.csv: {len(test)}")
print(f"Test rows with content twins in train.csv: {test_matches} ({test_matches/len(test)*100:.2f}%)")
print(f"Test rows with twin conflicts (potential hoax label=0): {test_conflicts} ({test_conflicts/len(test)*100:.2f}%)")
