import numpy as np
import pandas as pd

print("=== Generating Final Submission from HPO Model ===")
test = pd.read_csv('test.csv')

cache = np.load(r'C:\Users\Reynaldi Hutagaol\.gemini\antigravity-ide\brain\489beaa2-e2ba-4370-a4e6-0110f42de593\scratch\features_cache.npz')
X_base_te = cache['X_test']

opt_res = np.load('scratch/optuna_results.npz')
meta_test_hpo = opt_res['meta_test_hpo']

# Conflict signals from test set (strictly row-independent)
loc_conf_te = X_base_te[:, 18]
fig_conf_te = X_base_te[:, 21]
inst_mis_te = X_base_te[:, 22]
neg_inv_te  = X_base_te[:, 30]
ant_conf_te = X_base_te[:, 34]
row_conflict_test = ((loc_conf_te == 1) | (fig_conf_te == 1) | (inst_mis_te == 1) | (neg_inv_te == 1) | (ant_conf_te == 1))

# Re-tuned thresholds
th_clean = 0.720
th_conf  = 0.760

test_preds = np.where(row_conflict_test, (meta_test_hpo >= th_conf).astype(int), (meta_test_hpo >= th_clean).astype(int))

sub = pd.DataFrame({
    'id': test['id'],
    'label': test_preds
})

# Assertions
assert len(sub) == 3603, f"Expected 3603 rows, got {len(sub)}"
assert list(sub.columns) == ['id', 'label'], f"Columns mismatch: {sub.columns}"
assert sub['label'].isnull().sum() == 0, "Found null values"
assert set(sub['label'].unique()).issubset({0, 1}), "Invalid labels"
class0_count = (sub['label'] == 0).sum()
class1_count = (sub['label'] == 1).sum()
assert class0_count > 100 and class1_count > 2500, f"Unreasonable class distribution: C0={class0_count}, C1={class1_count}"

sub.to_csv('submission_baseline.csv', index=False)
sub.to_csv('scratch/submission_hpo_gated.csv', index=False)

print("Submission file successfully generated and verified:")
print(f"Shape: {sub.shape}")
print("Class distribution:")
print(sub['label'].value_counts())
class0_ratio = (sub['label'] == 0).mean()
print(f"Class 0 ratio: {class0_ratio*100:.2f}% ({class0_count} rows)")
print(f"Class 1 ratio: {(1-class0_ratio)*100:.2f}% ({class1_count} rows)")
print("All integrity checks passed successfully!")
