import json

with open('IFest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c11 = ''.join(nb['cells'][11]['source'])
target_str = '# Stacking Meta-Learner Cross Validation'
idx = c11.find(target_str)
assert idx != -1, "Target string not found in Cell 11"

# Retain everything before target_str
prefix = c11[:idx]

new_code = '''# --- REVISI CELL 6: Weighted Ensembling (Menggantikan Meta-Learner yang Overfit) ---

# 1. Agregasi Out-of-Fold (OOF) secara terbobot
# Beri bobot tinggi pada Gradient Boosting karena jauh lebih akurat pada data diskursus
w_xgb  = 0.40
w_hgb  = 0.35
w_et   = 0.15
w_lr   = 0.10

meta_oof = (w_xgb * xgb_oof) + (w_hgb * hgb_oof) + (w_et * et_oof) + (w_lr * lr_oof)
meta_test = (w_xgb * xgb_test) + (w_hgb * hgb_test) + (w_et * et_test) + (w_lr * lr_test)

# 2. Scanning Threshold Halus (Rentang Kritis 0.75 - 0.88)
best_th, best_f1 = 0.810, 0.0
for th in np.linspace(0.75, 0.88, 261):
    preds = (meta_oof >= th).astype(int)
    sc = f1_score(y_train, preds, average='macro')
    if sc > best_f1:
        best_f1, best_th = sc, th

print("\\n" + "="*50)
print(f"HASIL KALIBRASI OPTIMAL THRESHOLD: {best_th:.3f} | MACRO F1: {best_f1:.4f}")
print("="*50)

oof_preds = (meta_oof >= best_th).astype(int)
print(classification_report(y_train, oof_preds, digits=4))
print("Confusion Matrix (Optimal Threshold):")
print(confusion_matrix(y_train, oof_preds))
'''

updated_c11 = prefix + new_code

nb['cells'][11]['source'] = [line + '\n' for line in updated_c11.split('\n')[:-1]] + [updated_c11.split('\n')[-1]]

with open('IFest.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("IFest.ipynb Cell 11 successfully replaced with Weighted Ensembling!")
