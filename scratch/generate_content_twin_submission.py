import numpy as np
import pandas as pd

print("=== GENERATING SUBMISSION FROM FINAL 46-FEATURE CONTENT-TWIN MODEL ===")
test = pd.read_csv('test.csv')

cache = np.load(r'C:\Users\Reynaldi Hutagaol\.gemini\antigravity-ide\brain\489beaa2-e2ba-4370-a4e6-0110f42de593\scratch\features_cache.npz')
X_base_te = cache['X_test']

twin_res = np.load('scratch/final_46_features.npz')
meta_test = twin_res['meta_test']

# Conflict signals from test set (strictly row-independent)
loc_conf_te = X_base_te[:, 18]
fig_conf_te = X_base_te[:, 21]
inst_mis_te = X_base_te[:, 22]
neg_inv_te  = X_base_te[:, 30]
ant_conf_te = X_base_te[:, 34]
row_conflict_test = ((loc_conf_te == 1) | (fig_conf_te == 1) | (inst_mis_te == 1) | (neg_inv_te == 1) | (ant_conf_te == 1))

# Optimal re-tuned thresholds from final 46-feature evaluation
th_clean = float(twin_res['th_clean']) # 0.600
th_conf  = float(twin_res['th_conf'])  # 0.700

test_preds = np.where(row_conflict_test, (meta_test >= th_conf).astype(int), (meta_test >= th_clean).astype(int))

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
class0_ratio = class0_count / len(sub)

sub.to_csv('submission_baseline.csv', index=False)
sub.to_csv('scratch/submission_content_twin.csv', index=False)

print("Submission file successfully generated and verified:")
print(f"Shape: {sub.shape}")
print("Class distribution:")
print(sub['label'].value_counts())
print(f"Class 0 ratio: {class0_ratio*100:.2f}% ({class0_count} rows)")
print(f"Class 1 ratio: {(1-class0_ratio)*100:.2f}% ({class1_count} rows)")
print("All 4 official competition assertions PASSED with 100% success!")
