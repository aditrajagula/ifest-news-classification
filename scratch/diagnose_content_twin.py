import re
import unicodedata
import pandas as pd
import numpy as np

print("=" * 80)
print("=== DIAGNOSTIK MENDALAM POLA CONTENT-TWIN & DISTRIBUSI JACCARD ===")
print("=" * 80)

train = pd.read_csv('train.csv')

def normalize_content(s):
    s = unicodedata.normalize('NFKC', str(s))
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize_title(t):
    t = unicodedata.normalize('NFKC', str(t)).lower()
    t = re.sub(r'[^\w\s]', '', t)
    return set(t.split())

def jaccard_titles(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

train['content_norm'] = train['content'].apply(normalize_content)
train['content_len'] = train['content_norm'].str.len()
train['title_tokens'] = train['title'].apply(tokenize_title)

print(f"Distribusi Panjang content_norm (Karakter):")
print(train['content_len'].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95]))

# 1. Analisis Filter Panjang Minimum (MIN_CONTENT_LEN)
print("\n--- 1. Uji Sensitivitas Panjang Minimum (MIN_CONTENT_LEN) ---")
for min_len in [0, 50, 100, 150, 200, 250, 300]:
    valid_mask = train['content_len'] >= min_len
    train_valid = train[valid_mask]
    
    c1_valid = train_valid[train_valid['label'] == 1]
    c0_valid = train[train['label'] == 0]
    
    c1_content_set = set(c1_valid['content_norm'])
    matched_c0 = c0_valid[c0_valid['content_len'] >= min_len]['content_norm'].isin(c1_content_set).sum()
    pct_c0 = matched_c0 / (train['label'] == 0).sum() * 100
    print(f"MIN_LEN = {min_len:3d} | Match C0: {matched_c0:4d} / 1437 ({pct_c0:.2f}%) | C1 Index Size: {len(c1_content_set)}")

# 2. Analisis Ambiguitas Multi-Judul pada Label=1 (MIN_LEN = 200)
MIN_LEN = 200
print(f"\n--- 2. Analisis Ambiguitas Multi-Judul pada Label=1 (MIN_LEN = {MIN_LEN}) ---")
c1_df = train[(train['label'] == 1) & (train['content_len'] >= MIN_LEN)]
content_to_titles = c1_df.groupby('content_norm')['title'].apply(lambda x: list(set(x))).to_dict()

title_counts = [len(titles) for titles in content_to_titles.values()]
s_counts = pd.Series(title_counts).value_counts().sort_index()
print("Jumlah Judul Berbeda per Satu content_norm di Label=1:")
for count, n_keys in s_counts.items():
    pct = n_keys / len(content_to_titles) * 100
    print(f"  {count} Judul: {n_keys:5d} konten ({pct:5.2f}%)")

# 3. Distribusi Jaccard Similarity (Leave-One-Out Safe)
print(f"\n--- 3. Analisis Distribusi Jaccard Similarity (OOF Leave-One-Out) ---")

# Build mapping from content_norm to list of (row_idx, title_tokens) for label=1
from collections import defaultdict
content_to_label1_entries = defaultdict(list)
for idx, row in c1_df.iterrows():
    content_to_label1_entries[row['content_norm']].append((idx, row['title_tokens']))

# Compute twin_title_jaccard for all rows with leave-one-out
train['twin_has_match'] = 0.0
train['twin_title_jaccard'] = 0.0

for idx, row in train.iterrows():
    c_norm = row['content_norm']
    if len(c_norm) < MIN_LEN:
        continue
    entries = content_to_label1_entries.get(c_norm, [])
    # Leave-one-out: filter out own row idx
    ref_token_sets = [tok_set for r_idx, tok_set in entries if r_idx != idx]
    if not ref_token_sets:
        continue
    own_tokens = row['title_tokens']
    best_sim = max(jaccard_titles(own_tokens, ref_tokens) for ref_tokens in ref_token_sets)
    train.loc[idx, 'twin_has_match'] = 1.0
    train.loc[idx, 'twin_title_jaccard'] = best_sim

# Compare distributions for matched rows: Label 0 vs Label 1
c0_matched = train[(train['label'] == 0) & (train['twin_has_match'] == 1.0)]['twin_title_jaccard']
c1_matched = train[(train['label'] == 1) & (train['twin_has_match'] == 1.0)]['twin_title_jaccard']

print(f"\nTotal Matched Rows (MIN_LEN >= {MIN_LEN}):")
print(f"  Class 0 Matched: {len(c0_matched)} / {(train['label']==0).sum()} ({len(c0_matched)/(train['label']==0).sum()*100:.2f}%)")
print(f"  Class 1 Matched: {len(c1_matched)} / {(train['label']==1).sum()} ({len(c1_matched)/(train['label']==1).sum()*100:.2f}%)")

print("\nDistribusi twin_title_jaccard pada Class 0 Matched (Hoaks / False):")
bins = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
c0_dist = pd.cut(c0_matched, bins=bins, right=False).value_counts().sort_index()
for interval, count in c0_dist.items():
    pct = count / len(c0_matched) * 100
    bar = '#' * int(pct // 2)
    print(f"  {str(interval):<16} : {count:4d} ({pct:5.1f}%) | {bar}")

print("\nDistribusi twin_title_jaccard pada Class 1 Matched (Valid / True):")
c1_dist = pd.cut(c1_matched, bins=bins, right=False).value_counts().sort_index()
for interval, count in c1_dist.items():
    pct = count / len(c1_matched) * 100
    bar = '#' * int(pct // 2)
    print(f"  {str(interval):<16} : {count:4d} ({pct:5.1f}%) | {bar}")

print("\nAnalisis Kunci Pemisah:")
print(f"Persentase Class 0 dengan Jaccard < 0.70: {(c0_matched < 0.70).mean()*100:.2f}% ({(c0_matched < 0.70).sum()} baris)")
print(f"Persentase Class 1 dengan Jaccard < 0.70: {(c1_matched < 0.70).mean()*100:.2f}% ({(c1_matched < 0.70).sum()} baris)")
print(f"Persentase Class 0 dengan Jaccard >= 0.90: {(c0_matched >= 0.90).mean()*100:.2f}%")
print(f"Persentase Class 1 dengan Jaccard >= 0.90: {(c1_matched >= 0.90).mean()*100:.2f}%")
