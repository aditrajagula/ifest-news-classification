import re
import unicodedata
import pandas as pd
import numpy as np

print("=== TUGAS 2: AUDIT & EKSTRAKSI FITUR NUMBER ATTACHMENT CONTEXT WINDOW ===")

train = pd.read_csv('train.csv')

def extract_numbers_tokens(text):
    text = unicodedata.normalize('NFKC', str(text)).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    numbers = [tok for tok in tokens if tok.isdigit() and len(tok) <= 8]
    return tokens, numbers

def get_number_contexts(tokens, num_str, window=5):
    contexts = []
    for i, tok in enumerate(tokens):
        if tok == num_str:
            ctx = set(tokens[max(0, i - window):i]) | set(tokens[i + 1:i + 1 + window])
            contexts.append(ctx)
    return contexts

def jaccard_sets(s1, s2):
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

num_mismatch_flags = []
min_overlaps = []
has_common_nums = []

for idx, row in train.iterrows():
    t_tokens, t_nums = extract_numbers_tokens(row['title'])
    c_tokens, c_nums = extract_numbers_tokens(row['content'])
    
    common_nums = set(t_nums) & set(c_nums)
    if not common_nums:
        has_common_nums.append(0.0)
        num_mismatch_flags.append(0.0)
        min_overlaps.append(1.0)
        continue
        
    has_common_nums.append(1.0)
    # Check context overlap for each common number
    overlaps = []
    for num in common_nums:
        t_ctxs = get_number_contexts(t_tokens, num, window=5)
        c_ctxs = get_number_contexts(c_tokens, num, window=5)
        # For each occurrence in title, find best match in content
        for t_ctx in t_ctxs:
            best_match = max((jaccard_sets(t_ctx, c_ctx) for c_ctx in c_ctxs), default=0.0)
            overlaps.append(best_match)
            
    min_ov = min(overlaps) if overlaps else 1.0
    min_overlaps.append(min_ov)
    # If overlap is very low (< 0.15), flag as mismatch
    num_mismatch_flags.append(1.0 if min_ov < 0.15 else 0.0)

train['has_common_num'] = has_common_nums
train['num_min_overlap'] = min_overlaps
train['num_mismatch'] = num_mismatch_flags

print("\n--- Statistik Fitur Number Attachment Context Window ---")
print(f"Total baris dengan angka yang sama di judul & isi: {int(train['has_common_num'].sum())} / {len(train)}")
c0 = train[train['label'] == 0]
c1 = train[train['label'] == 1]

print(f"Rata-rata num_min_overlap pada Class 0: {c0[c0['has_common_num']==1]['num_min_overlap'].mean():.4f}")
print(f"Rata-rata num_min_overlap pada Class 1: {c1[c1['has_common_num']==1]['num_min_overlap'].mean():.4f}")

print(f"Persentase num_mismatch pada Class 0: {c0['num_mismatch'].mean()*100:.2f}% ({int(c0['num_mismatch'].sum())} baris)")
print(f"Persentase num_mismatch pada Class 1: {c1['num_mismatch'].mean()*100:.2f}% ({int(c1['num_mismatch'].sum())} baris)")
