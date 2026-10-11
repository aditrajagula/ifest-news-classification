import numpy as np
import pandas as pd

test = pd.read_csv('test.csv')
res = np.load('scratch/final_evaluation_results.npz')

# Best model was +City-Level Conflict (43 Fitur)
meta_oof = res['best_meta_oof']
meta_test = res['best_meta_test']
eval_th = float(res['eval_th']) # 0.770
th_clean = float(res['best_th_clean']) # 0.745
th_conf = float(res['best_th_conf']) # 0.785

cache = np.load(r'C:\Users\Reynaldi Hutagaol\.gemini\antigravity-ide\brain\489beaa2-e2ba-4370-a4e6-0110f42de593\scratch\features_cache.npz')
X_base_te = cache['X_test']

# Extract row_conflict_test
# Indices: 18: loc_conflict, 21: fig_conflict, 22: inst_mismatch, 30: neg_inversion, 34: antonym_conflict
loc_conf_te = X_base_te[:, 18]
fig_conf_te = X_base_te[:, 21]
inst_mis_te = X_base_te[:, 22]
neg_inv_te  = X_base_te[:, 30]
ant_conf_te = X_base_te[:, 34]

row_conflict_test = ((loc_conf_te == 1) | (fig_conf_te == 1) | (inst_mis_te == 1) | (neg_inv_te == 1) | (ant_conf_te == 1))
print(f"Test row conflict count: {row_conflict_test.sum()} ({row_conflict_test.mean()*100:.2f}%)")

# Pure Row-Independent Prediction with Gating:
test_preds = np.where(row_conflict_test, (meta_test >= th_conf).astype(int), (meta_test >= th_clean).astype(int))

sub = pd.DataFrame({'id': test['id'], 'label': test_preds})
sub.to_csv('submission_baseline.csv', index=False)
print("Saved submission_baseline.csv successfully!")

print("\n--- Distribusi Prediksi Submission Final ---")
vc = sub['label'].value_counts()
print(vc)
c0 = int(vc.get(0, 0))
c1 = int(vc.get(1, 0))
print(f"Kelas 1: {c1} ({c1/len(sub)*100:.2f}%)")
print(f"Kelas 0: {c0} ({c0/len(sub)*100:.2f}%)")

# Official Assertions
assert len(sub) == 3603, f"Expected 3603 rows, got {len(sub)}"
assert list(sub.columns) == ['id', 'label'], f"Invalid columns: {sub.columns}"
assert sub['label'].isnull().sum() == 0, "Found null values in predictions"
assert set(sub['label'].unique()).issubset({0, 1}), "Labels must be binary (0 or 1)"
print("\nSeluruh 4 asersi integritas resmi lolos 100%!")
