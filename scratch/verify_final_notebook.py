import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('IFest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

syntax_errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        try:
            compile(src, f'cell_{i}', 'exec')
        except SyntaxError as e:
            syntax_errors.append((i, str(e)))

print(f"Syntax errors: {len(syntax_errors)}")
if syntax_errors:
    for idx, err in syntax_errors:
        print(f"  Cell {idx}: {err}")
    sys.exit(1)
else:
    print("ALL CODE CELLS COMPILED SUCCESSFULLY!")

# Check specific key cells
found_audit = False
found_twin_func = False
found_seed43 = False

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if '=== AUDIT WIRING PIPELINE' in src:
        found_audit = True
        print(f"Cell {i} (Code): Sel Audit Wiring ditemukan.")
    if 'build_twin_features_for_train' in src:
        found_twin_func = True
        print(f"Cell {i} (Code): Fungsi Content-Twin Leave-One-Out ditemukan.")
    if 'skf_43 = StratifiedKFold' in src:
        found_seed43 = True
        print(f"Cell {i} (Code): Sanity Check Seed 43 ditemukan.")

assert found_audit, "Sel Audit Wiring TIDAK ditemukan di notebook!"
assert found_twin_func, "Fungsi Content-Twin TIDAK ditemukan di notebook!"
assert found_seed43, "Sanity Check Seed 43 TIDAK ditemukan di notebook!"
print("SEMUA PENGECEKAN NOTEBOOK LOLOS 100%!")
