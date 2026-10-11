import json

ifest_path = 'IFest.ipynb'
backup_path = r'C:\Users\Reynaldi Hutagaol\AppData\Roaming\Antigravity IDE\Backups\824dc8ab3e37371ce3b3319f12d08c13\file\-5ee1e86b'

with open(ifest_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

sec_4b_code = """# Evaluasi Error Analysis Manual pada False Negative (Class 0)
import os
import numpy as np
import pandas as pd

# Pastikan variabel dasar tersedia jika sel ini di-run mandiri di kernel baru
if 'SEED' not in globals():
    SEED = 42
if 'train' not in globals():
    train = pd.read_csv('train.csv')
if 'y_train' not in globals():
    y_train = train['label'].values

# Menggunakan meta_oof & eval_th dari pipeline jika sudah ada, atau fallback otomatis ke cache OOF / sweep 4a
if 'meta_oof' in globals() and 'eval_th' in globals():
    meta_oof_fn = meta_oof
    eval_th_fn = eval_th
    src_label = "Stacking Pipeline Resmi (Section 5)"
elif 'meta_oof_sweep' in globals() and 'eval_th_sweep' in globals():
    meta_oof_fn = meta_oof_sweep
    eval_th_fn = eval_th_sweep
    src_label = f"Hasil Sweep Bobot Terbaik (Section 4a, W={globals().get('best_weight', 2.0)}x)"
elif os.path.exists('scratch/full_rebuild_arrays.npz'):
    _cached_data = np.load('scratch/full_rebuild_arrays.npz')
    meta_oof_fn = _cached_data['meta_oof']
    eval_th_fn = 0.575
    src_label = "Prediksi OOF Terverifikasi (46-Fitur HPO)"
else:
    sw_def = np.where(y_train == 0, 2.0, 1.0)
    if 'run_pipeline' in globals():
        meta_oof_fn, eval_th_fn, _ = run_pipeline(sw_def)
        src_label = "Live CV 5-Fold Baseline (W=2.0x)"
    else:
        meta_oof_fn = np.zeros(len(y_train))
        eval_th_fn = 0.5
        src_label = "Initial State"

print(f"Sumber Prediksi OOF: {src_label} (Threshold = {eval_th_fn:.3f})")
oof_preds_check = (meta_oof_fn >= eval_th_fn).astype(int)

fn_mask = (y_train == 0) & (oof_preds_check == 1)
print(f"Total False Negative (label=0, diprediksi=1): {fn_mask.sum()} dari {(y_train == 0).sum()} kasus Class 0")

fn_indices = train.index[fn_mask]
fn_sample = train.loc[fn_indices, ['id', 'title', 'content']].sample(
    n=min(30, fn_mask.sum()), random_state=SEED
)

print("\\n" + "="*80)
print(f"  SAMPEL {len(fn_sample)} FALSE NEGATIVE UNTUK INSPEKSI MANUAL")
print("="*80)

for i, (idx, row) in enumerate(fn_sample.iterrows(), 1):
    print(f"\\n--- Sampel {i} | id={row['id']} ---")
    print(f"JUDUL : {row['title']}")
    print(f"ISI   : {row['content'][:400]}{'...' if len(row['content']) > 400 else ''}")
    print("-"*80)
"""

nb['cells'][13]['source'] = [l + '\n' for l in sec_4b_code.strip().split('\n')]

with open(ifest_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Updated {ifest_path} successfully")

import os
if os.path.exists(os.path.dirname(backup_path)):
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Updated backup buffer successfully")
