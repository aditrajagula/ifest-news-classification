import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('IFest.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Let's check cell at overall index 13 (the 14th cell)
print("--- Cell at Index 13 (14th cell overall) ---")
print(''.join(nb['cells'][13]['source'])[:300])

# Let's check cell at overall index 14
print("--- Cell at Index 14 ---")
print(''.join(nb['cells'][14]['source'])[:300])

# Let's check cell at overall index 15 (Code Cell 8)
print("--- Cell at Index 15 ---")
print(''.join(nb['cells'][15]['source'])[:300])

# Let's check cell at overall index 28 (Code Cell 14)
print("--- Cell at Index 28 (Code Cell #14) ---")
print(''.join(nb['cells'][28]['source'])[:300])
