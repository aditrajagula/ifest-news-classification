import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('IFest.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

code_cell_num = 0
for i, c in enumerate(nb['cells']):
    first_line = "".join(c['source']).strip().splitlines()[0] if c['source'] else "EMPTY"
    if c['cell_type'] == 'code':
        code_cell_num += 1
        print(f"Overall Index: {i} | Code Cell #{code_cell_num} | Exec: {c.get('execution_count')} | First line: {first_line[:80]}")
    else:
        print(f"Overall Index: {i} | Markdown      | Exec: None | First line: {first_line[:80]}")
