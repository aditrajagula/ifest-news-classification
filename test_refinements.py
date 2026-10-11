import os
import random
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

train = pd.read_csv('train.csv')
y_train = train['label'].values

# Load cached 44 refined features if needed, or let's test feature extraction on train
print("Testing feature refinements...")
