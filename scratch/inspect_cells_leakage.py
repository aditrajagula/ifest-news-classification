import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('IFest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== CELL 7 (Section 3: FastText & LSA) ===")
print("".join(nb['cells'][7]['source'][:40]))

print("\n=== CELL 34 (Section 5: Training pipeline) ===")
# search for 'fit' in cell 34
for line in nb['cells'][34]['source']:
    if 'fit' in line.lower() or 'scaler' in line.lower() or 'twin' in line.lower():
        print(line.strip())
