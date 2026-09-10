import json
import os
import pandas as pd

print("=== FINAL INTEGRITY & COMPLIANCE VERIFICATION ===")

# 1. JSON and cell structure
nb = json.load(open('IFest.ipynb', encoding='utf-8'))
print(f"Total cells in IFest.ipynb: {len(nb['cells'])}")
assert len(nb['cells']) == 43, f"Expected 43 cells, got {len(nb['cells'])}"

# 2. Check Cell 0 declaration
cell0 = ''.join(nb['cells'][0]['source'])
assert 'content-twin' in cell0.lower(), "content-twin missing in Cell 0 declaration"
assert '63.60%' in cell0, "Audit figure 63.60% missing in Cell 0"
assert '0.8888' in cell0 or '0.8899' in cell0, "Score missing in Cell 0"

# 3. Verify all code cells compile without error
code_cell_count = 0
for idx, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        code = ''.join(c['source'])
        compile(code, f'Cell_{idx}', 'exec')
        code_cell_count += 1
print(f"All {code_cell_count} code cells compiled with zero syntax errors!")

# 4. Backup buffer sync
backup_path = r'C:\Users\Reynaldi Hutagaol\AppData\Roaming\Antigravity IDE\Backups\824dc8ab3e37371ce3b3319f12d08c13\file\-5ee1e86b'
if os.path.exists(backup_path):
    nb_bk = json.load(open(backup_path, encoding='utf-8'))
    assert len(nb_bk['cells']) == 43, "Backup buffer cell count mismatch"
    print("Backup buffer verification: 100% IN SYNC")

# 5. Submission verification
sub = pd.read_csv('submission_baseline.csv')
assert len(sub) == 3603, f"Submission rows mismatch: {len(sub)}"
assert list(sub.columns) == ['id', 'label'], f"Columns mismatch: {list(sub.columns)}"
assert sub['label'].isnull().sum() == 0, "Null values found"
assert set(sub['label'].unique()).issubset({0, 1}), "Invalid label values"
print("Submission verification: ALL 4 INTEGRITY ASSERTIONS PASSED")
print("Distribusi Submission Final:")
print(sub['label'].value_counts())
c0 = (sub['label'] == 0).sum()
c1 = (sub['label'] == 1).sum()
print(f"Kelas 0: {c0} ({c0/len(sub)*100:.2f}%)")
print(f"Kelas 1: {c1} ({c1/len(sub)*100:.2f}%)")

print("\nSELURUH VERIFIKASI & AUDIT LOLOS DENGAN SUKSES 100%!")
