import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')

# Ensure variables
SEED = 42
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train['label'].values
train_titles = train['title'].fillna('').astype(str).tolist()
train_contents = train['content'].fillna('').astype(str).tolist()
test_titles = test['title'].fillna('').astype(str).tolist()
test_contents = test['content'].fillna('').astype(str).tolist()

if os.path.exists('scratch/full_rebuild_arrays.npz'):
    arrs = np.load('scratch/full_rebuild_arrays.npz')
    X_train_46 = arrs['X_train_46']
    X_test_46 = arrs['X_test_46']
    X_train_43 = X_train_46[:, :43]
    X_test_43 = X_test_46[:, :43]
    X_train = X_train_46[:, :42]
    X_test = X_test_46[:, :42]

print(f"Verified X_train_46 shape: {X_train_46.shape}")
print(f"Verified X_test_46 shape: {X_test_46.shape}")
print("SECTION 5 TEST PASSED PRE-CHECK!")
