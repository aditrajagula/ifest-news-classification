import json
import re

nb_path = 'IFest.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== AUDIT TEST LEAKAGE DI IFEST.IPYNB ===")
patterns = [
    r'np\.percentile\([^)]*test',
    r'meta_test\.(mean|std|sum|quantile)',
    r'X_test\.(mean|std|sum)',
    r'test_preds\.(mean|sum)',
    r'fit\(.*test',
    r'fit_transform\(.*test',
]

findings = []
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    for pat in patterns:
        matches = list(re.finditer(pat, src, re.I))
        if matches:
            for m in matches:
                findings.append((i, cell['cell_type'], m.group(0), src))

if not findings:
    print("Tidak ditemukan pola agregasi atau fitting pada data test!")
else:
    print(f"Ditemukan {len(findings)} temuan potensial:")
    for cell_idx, ctype, match_text, full_src in findings:
        print(f"\n[Cell {cell_idx} - {ctype}]: Match '{match_text}'")
        # Print the relevant lines around the match
        for line in full_src.split('\n'):
            if any(term in line for term in ['percentile', 'mean', 'quantile', 'fit_transform', 'fit(']):
                if 'test' in line:
                    print(f"  Line: {line.strip()}")
