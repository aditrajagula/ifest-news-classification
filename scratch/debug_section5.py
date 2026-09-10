import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('IFest.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][34]['source'])
print("Executing Section 5 source with debug...", flush=True)

try:
    exec(src)
    print("Section 5 finished successfully!", flush=True)
except Exception as e:
    import traceback
    print("Section 5 exception:", type(e).__name__, e, flush=True)
    traceback.print_exc()
