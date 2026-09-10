import re
import unicodedata
import pandas as pd
import numpy as np

print("=== TUGAS 3: AUDIT PASANGAN MODAL & STATUS OPERATOR DI TRAIN.CSV ===")

train = pd.read_csv('train.csv')

def clean_text(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    return set(s.split()), s

MODAL_PAIRS = [
    ('akan', 'batal'),
    ('akan', 'tidak akan'),
    ('akan', 'tak akan'),
    ('sudah', 'belum'),
    ('telah', 'belum'),
    ('resmi', 'wacana'),
    ('resmi', 'isu'),
    ('resmi', 'klaim'),
    ('resmi', 'hoaks'),
    ('keputusan', 'mengkaji'),
    ('keputusan', 'rencana'),
    ('fakta', 'hoaks'),
    ('fakta', 'isu'),
    ('bisa', 'mustahil'),
    ('bisa', 'tidak bisa'),
    ('dapat', 'tidak dapat')
]

c0_hits = {pair: 0 for pair in MODAL_PAIRS}
c1_hits = {pair: 0 for pair in MODAL_PAIRS}

c0_df = train[train['label'] == 0]
c1_df = train[train['label'] == 1]

for idx, row in train.iterrows():
    t_tokens, t_raw = clean_text(row['title'])
    c_tokens, c_raw = clean_text(row['content'])
    lbl = row['label']
    
    for w1, w2 in MODAL_PAIRS:
        hit = False
        # If w1 in title and w2 in content or vice versa
        if (' ' in w1 or ' ' in w2):
            if (w1 in t_raw and w2 in c_raw) or (w2 in t_raw and w1 in c_raw):
                hit = True
        else:
            if (w1 in t_tokens and w2 in c_tokens) or (w2 in t_tokens and w1 in c_tokens):
                hit = True
                
        if hit:
            if lbl == 0:
                c0_hits[(w1, w2)] += 1
            else:
                c1_hits[(w1, w2)] += 1

print("\n--- Frekuensi Kemunculan Pasangan Modal (Title vs Content) ---")
results = []
for pair in MODAL_PAIRS:
    n0 = c0_hits[pair]
    n1 = c1_hits[pair]
    pct0 = n0 / len(c0_df) * 100
    pct1 = n1 / len(c1_df) * 100
    ratio = (pct0 / pct1) if pct1 > 0 else 999.0
    results.append({
        'Pasangan Modal': f"{pair[0]} <-> {pair[1]}",
        'Class 0 Hits': n0,
        'Class 0 %': f"{pct0:.2f}%",
        'Class 1 Hits': n1,
        'Class 1 %': f"{pct1:.2f}%",
        'Rasio C0/C1': f"{ratio:.2f}x"
    })

df_modal = pd.DataFrame(results)
print(df_modal.to_string(index=False))
