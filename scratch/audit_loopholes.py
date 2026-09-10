import json
import re

with open('IFest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in notebook: {len(nb['cells'])}")

for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source = "".join(cell['source'])
    first_line = source.split('\n')[0] if source else "(empty)"
    
    # Check for specific patterns:
    # 1. Fit on test or fit_transform on test or concat(train, test)
    fit_on_test = bool(re.search(r'fit.*\(.*test', source, re.I))
    concat_tt = bool(re.search(r'concat.*\[.*train.*test.*\]', source, re.I))
    # 2. Scaler fit on test
    scaler_fit_test = bool(re.search(r'scaler.*fit.*test', source, re.I))
    # 3. Label used in test feature extraction
    label_in_test = bool(re.search(r'label.*test', source, re.I))
    
    flags = []
    if fit_on_test: flags.append("FIT_ON_TEST?")
    if concat_tt: flags.append("CONCAT_TRAIN_TEST?")
    if scaler_fit_test: flags.append("SCALER_ON_TEST?")
    if label_in_test: flags.append("LABEL_IN_TEST?")
    
    flag_str = f" | FLAGS: {', '.join(flags)}" if flags else ""
    print(f"Cell {i:02d} [{ctype[:4]}] (lines: {len(cell['source'])}) {first_line[:50]} {flag_str}")
