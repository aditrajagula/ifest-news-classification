import json

backup_path = r'C:\Users\Reynaldi Hutagaol\AppData\Roaming\Antigravity IDE\Backups\824dc8ab3e37371ce3b3319f12d08c13\file\-5ee1e86b'
ifest_path = r'c:\ifest-news-classification\IFest.ipynb'

# 1. Read backup file
with open(backup_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

meta_header = lines[0]
nb = json.loads(''.join(lines[1:]))

# 2. In cell 11, add skf initialization
cell11_src = nb['cells'][11]['source']
src_text = ''.join(cell11_src)

if 'skf = StratifiedKFold' not in src_text:
    new_source = ['skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n\n']
    for line in cell11_src:
        new_source.append(line)
        if '"""Menjalankan' in line:
            new_source.append('    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n')
    nb['cells'][11]['source'] = new_source
    print('Cell 11 updated with skf definition!')
else:
    print('skf already in Cell 11')

# 3. Save to IFest.ipynb
with open(ifest_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved to IFest.ipynb successfully!')

# 4. Save to backup file
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(meta_header)
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved to backup file successfully!')
