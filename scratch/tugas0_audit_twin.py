import re
import unicodedata
import pandas as pd
import numpy as np

print("=== TUGAS 0: AUDIT INDEPENDEN POLA CONTENT-TWIN ===")
train = pd.read_csv('train.csv')
total_rows = len(train)
c0_count = (train['label'] == 0).sum()
c1_count = (train['label'] == 1).sum()

print(f"Total train rows: {total_rows}")
print(f"Class 0 count: {c0_count} ({c0_count/total_rows*100:.2f}%)")
print(f"Class 1 count: {c1_count} ({c1_count/total_rows*100:.2f}%)")

def normalize_content(s):
    s = unicodedata.normalize('NFKC', str(s))
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

train['content_norm'] = train['content'].apply(normalize_content)

# Group content_norm from label=1
content_to_label1_titles = train[train['label']==1].groupby('content_norm')['title'].apply(list).to_dict()

# Check label=0
c0_mask = (train['label'] == 0)
has_twin = train[c0_mask]['content_norm'].isin(content_to_label1_titles.keys())
n_twin = has_twin.sum()
pct_twin = n_twin / c0_count * 100

print(f"\n[HASIL AUDIT RESMI TUGAS 0]")
print(f"Baris label 0 dengan content-twin di label 1: {n_twin} dari {c0_count} ({pct_twin:.2f}%)")

# Let's inspect some examples of content-twin pairs
print("\n--- Contoh Pasangan Content-Twin (Temuan Mandiri) ---")
c0_twins = train[c0_mask & train['content_norm'].isin(content_to_label1_titles.keys())]
example_count = 0
for idx, row in c0_twins.head(10).iterrows():
    c0_id = row['id']
    c0_title = row['title']
    norm = row['content_norm']
    c1_titles = content_to_label1_titles[norm]
    print(f"Sample #{example_count+1}: ID {c0_id} (Label 0)")
    print(f"  Title C0: {c0_title}")
    print(f"  Title C1 (Twin): {c1_titles[0]}")
    example_count += 1
    if example_count >= 3:
        break

# Also check test set: how many test rows have content-twin in train label=1?
test = pd.read_csv('test.csv')
test['content_norm'] = test['content'].apply(normalize_content)
test_has_twin = test['content_norm'].isin(content_to_label1_titles.keys())
print(f"\nTest set rows with content-twin in train (label=1): {test_has_twin.sum()} dari {len(test)} ({test_has_twin.mean()*100:.2f}%)")
