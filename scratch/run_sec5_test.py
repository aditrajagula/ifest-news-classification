import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('IFest.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Find Section 5 Code Cell
sec5_code = ''.join(nb['cells'][34]['source'])
print("--- Extracted Section 5 Code ---")
print(sec5_code[:500])

print("\n--- Executing Section 5 Code ---")
try:
    exec(sec5_code, {'__name__': '__main__'})
    print("SUCCESS: Section 5 executed completely without error!")
except Exception as e:
    import traceback
    print(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
