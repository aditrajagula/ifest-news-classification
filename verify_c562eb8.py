import os
import random
import re
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')

# Run exact c562eb8 code
with open('c562eb8_extract_features.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Execute c562eb8
exec(code)
