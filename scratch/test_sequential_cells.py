import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('IFest.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Global namespace simulating Jupyter notebook session
ns = {}

code_idx = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code_idx += 1
        src = ''.join(cell['source'])
        print(f"\n--- Running Overall Index {i} | Code Cell #{code_idx} ---")
        try:
            # We skip long training in sweep or HPO if needed, but let's test fast compile and key variables
            exec(src, ns)
            print(f"Index {i} (Code #{code_idx}): SUCCESS")
        except Exception as e:
            print(f"Index {i} (Code #{code_idx}): FAILED with {type(e).__name__}: {e}")
            break
