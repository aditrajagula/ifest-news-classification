import sys
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')
SEED = 42

print("=== Loading data & existing 42 features ===")
train = pd.read_csv('train.csv')
y_train = train['label'].values

cache = np.load(r'C:\Users\Reynaldi Hutagaol\.gemini\antigravity-ide\brain\489beaa2-e2ba-4370-a4e6-0110f42de593\scratch\features_cache.npz')
X_base_tr = cache['X_train'] # (14397, 42)
X_base_te = cache['X_test']  # (3603, 42)

print(f"X_base_tr shape: {X_base_tr.shape}")

train_titles = train['title'].fillna('').astype(str).tolist()
train_contents = train['content'].fillna('').astype(str).tolist()

test = pd.read_csv('test.csv')
test_titles = test['title'].fillna('').astype(str).tolist()
test_contents = test['content'].fillna('').astype(str).tolist()

# Load canonical locations setup from IFest (6).ipynb
with open(r'C:\Users\Reynaldi Hutagaol\Downloads\IFest (6).ipynb', 'r', encoding='utf-8', errors='ignore') as f:
    nb = json_import = __import__('json').load(f)

exec_scope = {'re': re, 'pd': pd, 'np': np}
exec(''.join(nb['cells'][5]['source']), exec_scope)
get_canonical_locations = exec_scope['get_canonical_locations']
STOP_WORDS = exec_scope['STOP_WORDS']

# --- TUGAS 4: City-Level Lead Conflict ---
PROVINCES_SET = {
    'aceh', 'sumatera utara', 'sumatera barat', 'riau', 'kepulauan riau', 'sumatera selatan',
    'lampung', 'jambi', 'bengkulu', 'bangka belitung',
    'kalimantan timur', 'kalimantan barat', 'kalimantan selatan', 'kalimantan tengah', 'kalimantan utara',
    'sulawesi selatan', 'sulawesi utara', 'sulawesi tengah', 'sulawesi tenggara', 'sulawesi barat', 'gorontalo',
    'bali', 'nusa tenggara barat', 'nusa tenggara timur', 'maluku', 'maluku utara', 'papua', 'papua barat', 'indonesia'
}

def extract_city_conflict(titles, contents):
    N = len(titles)
    feat = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        t_low = titles[i].lower()
        c_low = contents[i].lower()
        lead_low = ' '.join(c_low.split()[:40])
        t_locs = get_canonical_locations(t_low)
        lead_locs = get_canonical_locations(lead_low)
        t_cities = set(l for l in t_locs if l not in PROVINCES_SET)
        lead_cities = set(l for l in lead_locs if l not in PROVINCES_SET)
        if t_cities and lead_cities and not t_cities.intersection(lead_cities):
            feat[i, 0] = 1.0
    return feat

print("Extracting City-Level Lead Conflict...")
city_tr = extract_city_conflict(train_titles, train_contents)
city_te = extract_city_conflict(test_titles, test_contents)
print(f"City conflict positive train: {city_tr.sum()} ({city_tr.mean()*100:.2f}%)")

# --- TUGAS 5: Dynamic NER (4 Features) ---
ROLE_PREFIXES = r'(?:Presiden|Wapres|Gubernur|Wagub|Bupati|Walikota|Kapolri|Kapolda|Kapolres|Panglima|Kasad|Kasal|Kasau|Menteri|Menko|Menkes|Menkeu|Mendagri|Menlu|Menpora|Menaker|Dirjen|Dirut|KPK|Jaksa|Hakim|Ketua|Wakil|Dokter|dr|drg|Spesialis|Prof|Ustadz|Habib|Kiai|Pendeta|Penyanyi|Aktor|Aktris|Musisi|Vokalis|Pelatih|Coach|Pebalap|Kiper|Striker|Kapten|Bintang|Artis)\b'
ROLE_NAME_REGEX = re.compile(rf'{ROLE_PREFIXES}\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', re.I)

FACILITY_PREFIXES = r'(?:Rumah Sakit|RSUD|RS|Puskesmas|Dinas|Kementerian|Kemendikbud|Kemenkes|Kemenkeu|Kemendagri|Polres|Polda|Polsek|Kodim|Koramil|Pengadilan|Kejaksaan|Bandara|Stasiun|Pelabuhan|Terminal|Stadion|Gelora|Pasar|Mal|Mall|Plaza|Hotel|Apartemen|Universitas|Institut|Fakultas|Sekolah|SMA|SMP|SD|PT|CV)\b'
FACILITY_NAME_REGEX = re.compile(rf'{FACILITY_PREFIXES}\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)', re.I)

ACRONYM_REGEX = re.compile(r'\b[A-Z]{3,8}\b')

def extract_dynamic_ner(titles, contents):
    N = len(titles)
    feat_role = np.zeros((N, 1), dtype=np.float32)
    feat_fac = np.zeros((N, 1), dtype=np.float32)
    feat_acro = np.zeros((N, 1), dtype=np.float32)
    feat_head = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        t_raw, c_raw = titles[i], contents[i]
        c_lower = c_raw.lower()
        m_r = ROLE_NAME_REGEX.findall(t_raw)
        if m_r and not any(r.lower() in c_lower for r in m_r):
            feat_role[i, 0] = 1.0
        m_f = FACILITY_NAME_REGEX.findall(t_raw)
        if m_f and not any(f.lower() in c_lower for f in m_f):
            feat_fac[i, 0] = 1.0
        m_a = ACRONYM_REGEX.findall(t_raw)
        if m_a and not any(a.lower() in c_lower for a in m_a):
            feat_acro[i, 0] = 1.0
        t_words = [w for w in re.findall(r'[a-z0-9]+', t_raw.lower()) if w not in STOP_WORDS]
        if t_words and t_words[0] not in c_lower:
            feat_head[i, 0] = 1.0
    return np.hstack([feat_role, feat_fac, feat_acro, feat_head])

print("Extracting Dynamic NER Features...")
ner_tr = extract_dynamic_ner(train_titles, train_contents)
ner_te = extract_dynamic_ner(test_titles, test_contents)
print(f"NER sums train: role={ner_tr[:,0].sum()}, fac={ner_tr[:,1].sum()}, acro={ner_tr[:,2].sum()}, head={ner_tr[:,3].sum()}")

# Save extra features
np.savez('scratch/extra_features.npz',
         city_tr=city_tr, city_te=city_te,
         ner_tr=ner_tr, ner_te=ner_te)
print("Saved extra features to scratch/extra_features.npz")
