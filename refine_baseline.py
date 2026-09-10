import os
import random
import re
import math
import sys
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')

# 1. Deterministic Random Seed (Rule #11)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

print(f"Deterministic Random Seed set to {SEED}")

# 2. Load Data
print("Loading train.csv and test.csv...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
y_train = train['label'].values

# 3. Kamus Leksikal & Normalisasi
STOP_WORDS = {
    'yang', 'di', 'dan', 'ini', 'dari', 'dalam', 'untuk', 'pada', 'ke', 'karena',
    'oleh', 'saat', 'ada', 'akan', 'juga', 'dengan', 'adalah', 'ia', 'itu', 'atau',
    'bisa', 'tersebut', 'bukan', 'tidak', 'tak', 'namun', 'mereka', 'kita', 'para',
    'lagi', 'saja', 'sudah', 'secara', 'harus', 'hanya', 'agar', 'bagi', 'bahwa',
    'terus', 'setelah', 'terkait', 'tentang', 'atas', 'olehnya', 'menjadi', 'sebagai',
    'masih', 'bila', 'belum', 'banyak', 'sangat', 'tahun', 'hari', 'bulan', 'lalu',
    'seperti', 'antara', 'dapat', 'mengatakan', 'ujar', 'kata', 'ungkap', 'tambah',
    'jelas', 'pungkas', 'tutur', 'sebut', 'beber', 'imbuh'
}

EXPANDED_ANTONYM_PAIRS = [
    ('positif', 'negatif'), ('naik', 'turun'), ('lonjakan', 'penurunan'),
    ('buka', 'tutup'), ('dibuka', 'ditutup'), ('tambah', 'kurang'),
    ('tinggi', 'rendah'), ('meningkat', 'menurun'), ('sembuh', 'meninggal'),
    ('hidup', 'mati'), ('masuk', 'keluar'), ('zona merah', 'zona hijau'),
    ('sukses', 'gagal'), ('untung', 'rugi'), ('kalah', 'menang'),
    ('benar', 'salah'), ('palsu', 'asli'), ('hoaks', 'fakta')
]

REFUTING_PHRASES = [
    'tidak benar', 'tak benar', 'bukan', 'keliru', 'hoaks', 'hoax',
    'bohong', 'bantah', 'membantah', 'dibantah', 'fiktif', 'palsu',
    'disinformasi', 'misinformasi', 'manipulasi', 'klaim sepihak'
]
re_refuting = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in REFUTING_PHRASES) + r')\b', re.I)
re_num_unit = re.compile(r'(\b\d+[\d\.,]*)\s+([a-zA-Z]+)')

# 4. TUGAS 3: Kamus Hierarki Wilayah PROVINCE_OF Se-Indonesia
CANONICAL_LOC = {
    'jabar': 'jawa barat', 'jateng': 'jawa tengah', 'jatim': 'jawa timur',
    'diy': 'yogyakarta', 'jogja': 'yogyakarta', 'jogjakarta': 'yogyakarta',
    'dki': 'dki jakarta', 'jakpus': 'jakarta pusat', 'jakbar': 'jakarta barat',
    'jaksel': 'jakarta selatan', 'jaktim': 'jakarta timur', 'jakut': 'jakarta utara',
    'surakarta': 'solo', 'ina': 'indonesia', 'ri': 'indonesia'
}

PROVINCE_OF = {
    # DKI Jakarta
    'jakarta': 'dki jakarta', 'jakpus': 'dki jakarta', 'jakbar': 'dki jakarta',
    'jaksel': 'dki jakarta', 'jaktim': 'dki jakarta', 'jakut': 'dki jakarta',
    'jakarta pusat': 'dki jakarta', 'jakarta barat': 'dki jakarta',
    'jakarta selatan': 'dki jakarta', 'jakarta timur': 'dki jakarta', 'jakarta utara': 'dki jakarta',
    'kepulauan seribu': 'dki jakarta',
    # Jawa Barat
    'bandung': 'jawa barat', 'bogor': 'jawa barat', 'bekasi': 'jawa barat',
    'depok': 'jawa barat', 'cimahi': 'jawa barat', 'tasikmalaya': 'jawa barat',
    'banjar': 'jawa barat', 'sukabumi': 'jawa barat', 'cirebon': 'jawa barat',
    'garut': 'jawa barat', 'karawang': 'jawa barat', 'purwakarta': 'jawa barat',
    'subang': 'jawa barat', 'indramayu': 'jawa barat', 'sumedang': 'jawa barat',
    'majalengka': 'jawa barat', 'kuningan': 'jawa barat', 'ciamis': 'jawa barat',
    'pangandaran': 'jawa barat', 'cianjur': 'jawa barat',
    # Jawa Tengah
    'semarang': 'jawa tengah', 'solo': 'jawa tengah', 'surakarta': 'jawa tengah',
    'magelang': 'jawa tengah', 'muntilan': 'jawa tengah', 'pekalongan': 'jawa tengah',
    'salatiga': 'jawa tengah', 'tegal': 'jawa tengah', 'slawi': 'jawa tengah',
    'banyumas': 'jawa tengah', 'purwokerto': 'jawa tengah', 'cilacap': 'jawa tengah',
    'purbalingga': 'jawa tengah', 'banjarnegara': 'jawa tengah', 'kebumen': 'jawa tengah',
    'purworejo': 'jawa tengah', 'wonosobo': 'jawa tengah', 'boyolali': 'jawa tengah',
    'klaten': 'jawa tengah', 'sukoharjo': 'jawa tengah', 'wonogiri': 'jawa tengah',
    'karanganyar': 'jawa tengah', 'sragen': 'jawa tengah', 'grobogan': 'jawa tengah',
    'purwodadi': 'jawa tengah', 'blora': 'jawa tengah', 'rembang': 'jawa tengah',
    'pati': 'jawa tengah', 'kudus': 'jawa tengah', 'jepara': 'jawa tengah',
    'demak': 'jawa tengah', 'temanggung': 'jawa tengah', 'kendal': 'jawa tengah',
    'batang': 'jawa tengah', 'pemalang': 'jawa tengah', 'brebes': 'jawa tengah',
    # DI Yogyakarta
    'yogyakarta': 'yogyakarta', 'jogja': 'yogyakarta', 'jogjakarta': 'yogyakarta',
    'diy': 'yogyakarta', 'sleman': 'yogyakarta', 'bantul': 'yogyakarta',
    'kulon progo': 'yogyakarta', 'kulonprogo': 'yogyakarta', 'wates': 'yogyakarta',
    'gunung kidul': 'yogyakarta', 'gunungkidul': 'yogyakarta', 'wonosari': 'yogyakarta',
    # Jawa Timur
    'surabaya': 'jawa timur', 'malang': 'jawa timur', 'batu': 'jawa timur',
    'kediri': 'jawa timur', 'blitar': 'jawa timur', 'madiun': 'jawa timur',
    'probolinggo': 'jawa timur', 'pasuruan': 'jawa timur', 'mojokerto': 'jawa timur',
    'sidoarjo': 'jawa timur', 'jombang': 'jawa timur', 'nganjuk': 'jawa timur',
    'magetan': 'jawa timur', 'ngawi': 'jawa timur', 'bojonegoro': 'jawa timur',
    'tuban': 'jawa timur', 'lamongan': 'jawa timur', 'gresik': 'jawa timur',
    'bangkalan': 'jawa timur', 'sampang': 'jawa timur', 'pamekasan': 'jawa timur',
    'sumenep': 'jawa timur', 'lumajang': 'jawa timur', 'jember': 'jawa timur',
    'bondowoso': 'jawa timur', 'situbondo': 'jawa timur', 'banyuwangi': 'jawa timur',
    'trenggalek': 'jawa timur', 'tulungagung': 'jawa timur', 'pacitan': 'jawa timur',
    'ponorogo': 'jawa timur',
    # Banten
    'serang': 'banten', 'cilegon': 'banten', 'tangerang': 'banten',
    'pandeglang': 'banten', 'lebak': 'banten'
}

ADDITIONAL_PROVINCE_OF = {
    # Sumatera
    'medan': 'sumatera utara', 'deli serdang': 'sumatera utara', 'pematangsiantar': 'sumatera utara',
    'pematang siantar': 'sumatera utara', 'binjai': 'sumatera utara', 'padang': 'sumatera barat',
    'bukittinggi': 'sumatera barat', 'pekanbaru': 'riau', 'dumai': 'riau',
    'palembang': 'sumatera selatan', 'prabumulih': 'sumatera selatan', 'bandar lampung': 'lampung',
    'metro': 'lampung', 'batam': 'kepulauan riau', 'tanjungpinang': 'kepulauan riau',
    'banda aceh': 'aceh', 'lhokseumawe': 'aceh', 'jambi': 'jambi', 'bengkulu': 'bengkulu',
    'pangkalpinang': 'bangka belitung',
    # Kalimantan
    'pontianak': 'kalimantan barat', 'singkawang': 'kalimantan barat', 'palangkaraya': 'kalimantan tengah',
    'banjarmasin': 'kalimantan selatan', 'banjarbaru': 'kalimantan selatan', 'samarinda': 'kalimantan timur',
    'balikpapan': 'kalimantan timur', 'tarakan': 'kalimantan utara',
    # Sulawesi
    'makassar': 'sulawesi selatan', 'parepare': 'sulawesi selatan', 'palopo': 'sulawesi selatan',
    'manado': 'sulawesi utara', 'bitung': 'sulawesi utara', 'palu': 'sulawesi tengah',
    'kendari': 'sulawesi tenggara', 'gorontalo': 'gorontalo', 'mamuju': 'sulawesi barat',
    # Bali & Nusa Tenggara
    'denpasar': 'bali', 'singaraja': 'bali', 'mataram': 'nusa tenggara barat',
    'bima': 'nusa tenggara barat', 'kupang': 'nusa tenggara timur',
    # Maluku & Papua
    'ambon': 'maluku', 'ternate': 'maluku utara', 'jayapura': 'papua', 'sorong': 'papua barat'
}
PROVINCE_OF.update(ADDITIONAL_PROVINCE_OF)

ALL_LOCATIONS = list(set(
    list(PROVINCE_OF.keys()) + list(PROVINCE_OF.values()) + list(CANONICAL_LOC.keys()) + [
        'sumatera utara', 'sumatera barat', 'sumatera selatan', 'sulawesi tenggara',
        'sulawesi selatan', 'sulawesi utara', 'sulawesi tengah', 'sulawesi barat',
        'kalimantan timur', 'kalimantan barat', 'kalimantan selatan', 'kalimantan tengah', 'kalimantan utara',
        'nusa tenggara barat', 'nusa tenggara timur', 'papua', 'papua barat', 'maluku', 'maluku utara',
        'aceh', 'sumut', 'sumbar', 'riau', 'kepri', 'jambi', 'bengkulu', 'sumsel', 'babel', 'lampung',
        'kalbar', 'kalteng', 'kalsel', 'kaltim', 'kaltara', 'sulsel', 'sulut', 'sulteng', 'sultra', 'sulbar',
        'bali', 'ntb', 'ntt', 'gorontalo',
        'deli serdang', 'tebing tinggi', 'tanjung balai', 'tanjungbalai', 'padang panjang', 'pesisir selatan',
        'tanah datar', 'pagar alam', 'ogan ilir', 'muara enim', 'banyuasin', 'kutai kartanegara',
        'luwu utara', 'luwu timur', 'konawe utara', 'konawe selatan', 'musi banyuasin', 'toba samosir',
        'lubuklinggau', 'lahat', 'musi', 'bangka', 'belitung', 'sambas', 'sintang', 'ketapang', 'kapuas',
        'kotawaringin', 'barito', 'kutai', 'berau', 'pasir', 'penajam', 'nunukan', 'bulungan', 'malinau',
        'mamuju', 'palopo', 'tomohon', 'kotamobagu', 'bau-bau', 'gowa', 'maros', 'bone', 'luwu', 'wajo',
        'soppeng', 'bulukumba', 'bantaeng', 'selayar', 'toraja', 'enrekang', 'sidrap', 'pinrang', 'minahasa',
        'sangihe', 'talaud', 'poso', 'donggala', 'banggai', 'morowali', 'tolitoli', 'buol', 'buton', 'muna',
        'kolaka', 'konawe', 'wakatobi', 'badung', 'gianyar', 'tabanan', 'buleleng', 'jembrana', 'karangasem',
        'klungkung', 'bangli', 'dompu', 'timor', 'belu', 'flores', 'ende', 'manggarai', 'sumba', 'alor', 'rote',
        'tual', 'tidore', 'seram', 'buru', 'halmahera', 'manokwari', 'merauke', 'mimika', 'timika', 'biak',
        'nabire', 'fakfak', 'wamena',
        'amerika', 'inggris', 'jerman', 'prancis', 'china', 'cina', 'jepang', 'korsel',
        'brasil', 'brazil', 'meksiko', 'italia', 'spanyol', 'rusia', 'afsel', 'afrika',
        'australia', 'singapura', 'malaysia', 'filipina', 'thailand', 'vietnam', 'india',
        'arab', 'saudi', 'turki', 'iran', 'irak', 'israel', 'palestina', 'mesir',
        'belanda', 'portugal', 'kanada', 'selandia', 'ukraina', 'lebanon', 'indonesia', 'ri', 'nigeria'
    ]
))
LOC_REGEX = re.compile(r'\b(' + '|'.join(re.escape(loc) for loc in sorted(ALL_LOCATIONS, key=len, reverse=True)) + r')\b', re.I)

# 5. TUGAS 4: Collocation Guard untuk Kata Ambivalen ("Arab", "Nigeria", dll.)
NON_GEO_ARAB_CONTEXT = {'bahasa', 'huruf', 'tulisan', 'latin', 'doa', 'sholat', 'salat', 'kitab', 'ayat', 'tulisan arab', 'bahasa arab'}
NATIONALITY_SPORT_CONTEXT = {'timnas', 'pemain', 'warga', 'wni', 'wna', 'atlet', 'paspor', 'striker', 'bek', 'gelandang', 'kiper', 'skuad', 'klub', 'liga'}
FOREIGN_COUNTRIES = {'arab', 'saudi', 'nigeria', 'brasil', 'brazil', 'prancis', 'jerman', 'inggris', 'italia', 'spanyol', 'rusia', 'china', 'cina', 'jepang', 'korsel', 'afsel', 'afrika', 'australia', 'filipina', 'thailand', 'vietnam', 'india', 'turki', 'iran', 'irak', 'israel', 'palestina', 'mesir', 'belanda', 'portugal', 'kanada', 'selandia', 'ukraina', 'lebanon'}

def get_canonical_locations(text):
    text_lower = text.lower()
    locs = set()
    for m in LOC_REGEX.finditer(text_lower):
        loc = m.group(0).lower()
        start_char = m.start()
        end_char = m.end()
        ctx_before = text_lower[max(0, start_char - 40):start_char]
        ctx_after = text_lower[end_char:min(len(text_lower), end_char + 40)]
        ctx_tokens = set(re.findall(r'\b[a-zA-Z]+\b', ctx_before + ' ' + ctx_after))
        
        # Collocation guard 1: Arab (bahasa/doa/huruf)
        if loc in ['arab', 'saudi']:
            if ctx_tokens & NON_GEO_ARAB_CONTEXT:
                continue
                
        # Collocation guard 2: Kewarganegaraan / Olahraga
        if loc in FOREIGN_COUNTRIES:
            if ctx_tokens & NATIONALITY_SPORT_CONTEXT:
                continue
                
        canon = CANONICAL_LOC.get(loc, loc)
        locs.add(canon)
    return locs

# Tokoh Publik & Lembaga
CANONICAL_FIGURES = {
    'joko widodo': 'jokowi', 'maruf amin': 'ma ruf', 'anies baswedan': 'anies',
    'prabowo subianto': 'prabowo', 'sandiaga uno': 'sandiaga', 'ridwan kamil': 'rk',
    'ganjar pranowo': 'ganjar', 'airlangga hartarto': 'airlangga', 'luhut pandjaitan': 'luhut',
    'mahfud md': 'mahfud', 'muhadjir effendy': 'muhadjir', 'ahmad riza patria': 'riza patria',
    'gibran rakabuming': 'gibran', 'bima arya sugiarto': 'bima arya', 'wiku adisasmito': 'wiku',
    'achmad yurianto': 'yurianto', 'agus harimurti': 'ahy', 'agus harimurti yudhoyono': 'ahy',
    'susilo bambang yudhoyono': 'sby', 'puan maharani': 'puan', 'donald trump': 'trump',
    'joe biden': 'biden', 'vladimir putin': 'putin'
}
PUBLIC_FIGURES = list(set(list(CANONICAL_FIGURES.keys()) + list(CANONICAL_FIGURES.values()) + [
    'terawan', 'budi gunadi', 'erick thohir', 'sri mulyani', 'dedie rachim',
    'bobby nasution', 'doni monardo', 'idham azis', 'listyo sigit', 'hadi tjahjanto',
    'nadiem', 'nadiem makarim', 'megawati', 'rizieq', 'rizieq shihab',
    'sinovac', 'pfizer', 'moderna', 'astrazeneca', 'novavax', 'bio farma'
]))
FIGURE_REGEX = re.compile(r'\b(' + '|'.join(re.escape(f) for f in sorted(PUBLIC_FIGURES, key=len, reverse=True)) + r')\b', re.I)

INSTITUTIONS = {
    'itb', 'ugm', 'ui', 'unpad', 'undip', 'ipb', 'unair', 'its', 'unhas', 'unsri', 'usu',
    'uny', 'stie', 'ekuitas', 'kemenkes', 'menristek', 'bnpb', 'satgas', 'kpu', 'gerindra',
    'dpc', 'dprd', 'dpr', 'polri', 'bareskrim', 'dpmptsp', 'menko', 'pmk', 'polda', 'polres',
    'kpk', 'bpom', 'who', 'mui', 'kemenhub', 'kemendagri', 'kemensos', 'kemendikbud'
}

INSTITUTION_MARKERS = [
    'bea cukai', 'kantor bea cukai', 'rsud', 'puskesmas', 'dinkes', 'dinas kesehatan',
    'pemprov', 'pemkot', 'pemkab', 'polres', 'polda', 'kodim', 'pasar',
    'bupati', 'walikota', 'gubernur', 'kapolres', 'dprd', 'sekolah'
]
re_inst_marker = re.compile(r'\b(' + '|'.join(re.escape(m) for m in sorted(INSTITUTION_MARKERS, key=len, reverse=True)) + r')\b', re.I)

NEGATION_WORDS = {
    'tak', 'tidak', 'bukan', 'belum', 'tanpa', 'gagal', 'batal', 'larang', 'tolak',
    'bebas', 'nihil', 'nol', 'stop', 'hentikan', 'anti', 'menolak', 'membantah',
    'menyanggah', 'menafikan', 'mustahil'
}
NEG_WORDS = {'tak', 'tidak', 'bukan', 'belum', 'tanpa', 'gagal', 'batal', 'larang', 'tolak', 'anti', 'mustahil'}

INHERENT_NEG_VERBS = {
    'larang', 'melarang', 'dilarang', 'batal', 'membatalkan', 'dibatalkan',
    'gagal', 'menggagalkan', 'digagalkan', 'tolak', 'menolak', 'ditolak',
    'bantah', 'membantah', 'dibantah', 'tunda', 'menunda', 'ditunda'
}

PREFIXES = ['meng', 'meny', 'men', 'mem', 'me', 'di', 'ber', 'ter', 'pen', 'pem', 'peny', 'pe', 'ke', 'se']
SUFFIXES = ['kan', 'an', 'i', 'nya', 'lah', 'kah']

def light_stem(word):
    if len(word) <= 4:
        return word
    w = word
    for pref in PREFIXES:
        if w.startswith(pref) and len(w) - len(pref) >= 3:
            w = w[len(pref):]
            break
    for suff in SUFFIXES:
        if w.endswith(suff) and len(w) - len(suff) >= 3:
            w = w[:-len(suff)]
            break
    return w

def get_canonical_figures(text):
    raw_figs = [m.group(0).lower() for m in FIGURE_REGEX.finditer(text)]
    return set(CANONICAL_FIGURES.get(f, f) for f in raw_figs)

def extract_numbers_comprehensive(text):
    norm = re.sub(r'(\d+)\s+(\d+)', r'\1\2', text)
    num_matches = set(re.findall(r'\b\d+(?:[.,]\d+)?%?\b', norm))
    return {n for n in num_matches if not (len(n) == 4 and n.isdigit() and n.startswith(('19', '20')))}

def clean_title(text):
    return re.sub(r'\s+', ' ', str(text)).strip()

train_titles = [clean_title(t) for t in train['title'].fillna('').astype(str).tolist()]
train_contents = train['content'].fillna('').astype(str).tolist()
train_fs = [" ".join(c.split()[:40]) for c in train_contents]

test_titles = [clean_title(t) for t in test['title'].fillna('').astype(str).tolist()]
test_contents = test['content'].fillna('').astype(str).tolist()
test_fs = [" ".join(c.split()[:40]) for c in test_contents]

# TF-IDF & SVD
print("Fitting TF-IDF & TruncatedSVD...")
tfidf_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=15000, sublinear_tf=True)
tfidf_word.fit(train_titles + train_contents)

tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=5, max_features=15000, sublinear_tf=True)
tfidf_char.fit(train_titles + train_contents)

svd = TruncatedSVD(n_components=100, random_state=SEED)
train_c_w = tfidf_word.transform(train_contents)
svd.fit(train_c_w)

def compute_corpus_stats(tfidf_vec, corpus_texts):
    X = tfidf_vec.transform(corpus_texts)
    doc_lens = np.asarray(X.sum(axis=1)).flatten()
    avg_dl = float(np.mean(doc_lens))
    idfs = tfidf_vec.idf_
    return avg_dl, idfs

avg_dl_c, idf_w = compute_corpus_stats(tfidf_word, train_contents)
avg_dl_fs, _ = compute_corpus_stats(tfidf_word, train_fs)

def calc_bm25_score(t_vec, c_vec, c_len, avg_dl, k1=1.5, b=0.75):
    c_indices = c_vec.indices
    c_data = c_vec.data
    t_indices = t_vec.indices
    t_data = t_vec.data
    if len(t_indices) == 0 or len(c_indices) == 0:
        return 0.0
    c_dict = dict(zip(c_indices, c_data))
    t_dict = dict(zip(t_indices, t_data))
    common = set(t_indices).intersection(c_indices)
    if not common:
        return 0.0
    score = 0.0
    len_norm = 1.0 - b + b * (c_len / (avg_dl + 1e-9))
    for idx in common:
        tf_c = c_dict[idx]
        tf_t = t_dict[idx]
        idf_val = idf_w[idx] if idx < len(idf_w) else 1.0
        term_score = idf_val * (tf_c * (k1 + 1.0)) / (tf_c + k1 * len_norm) * tf_t
        score += term_score
    return float(score)

def get_pairwise_vectors(titles, contents, fs_texts):
    t_w = tfidf_word.transform(titles)
    c_w = tfidf_word.transform(contents)
    fs_w = tfidf_word.transform(fs_texts)
    
    t_c = tfidf_char.transform(titles)
    c_c = tfidf_char.transform(contents)
    fs_c = tfidf_char.transform(fs_texts)
    
    cos_w_c = np.asarray((t_w.multiply(c_w)).sum(axis=1)).flatten()
    cos_w_fs = np.asarray((t_w.multiply(fs_w)).sum(axis=1)).flatten()
    cos_c_c = np.asarray((t_c.multiply(c_c)).sum(axis=1)).flatten()
    cos_c_fs = np.asarray((t_c.multiply(fs_c)).sum(axis=1)).flatten()
    
    t_lsa = svd.transform(t_w)
    c_lsa = svd.transform(c_w)
    fs_lsa = svd.transform(fs_w)
    
    cos_lsa_c = (t_lsa * c_lsa).sum(axis=1) / (np.linalg.norm(t_lsa, axis=1) * np.linalg.norm(c_lsa, axis=1) + 1e-9)
    cos_lsa_fs = (t_lsa * fs_lsa).sum(axis=1) / (np.linalg.norm(t_lsa, axis=1) * np.linalg.norm(fs_lsa, axis=1) + 1e-9)
    
    c_lens = np.asarray(c_w.sum(axis=1)).flatten()
    fs_lens = np.asarray(fs_w.sum(axis=1)).flatten()
    bm25_c = [calc_bm25_score(t_w[i], c_w[i], c_lens[i], avg_dl_c) for i in range(len(titles))]
    bm25_fs = [calc_bm25_score(t_w[i], fs_w[i], fs_lens[i], avg_dl_fs) for i in range(len(titles))]
    
    return cos_w_c, cos_w_fs, cos_c_c, cos_c_fs, np.array(bm25_c), np.array(bm25_fs), cos_lsa_c, cos_lsa_fs, t_w

print("Computing pairwise vectors...")
cos_w_c_tr, cos_w_fs_tr, cos_c_c_tr, cos_c_fs_tr, bm25_c_tr, bm25_fs_tr, cos_lsa_c_tr, cos_lsa_fs_tr, tr_t_w = get_pairwise_vectors(train_titles, train_contents, train_fs)
cos_w_c_te, cos_w_fs_te, cos_c_c_te, cos_c_fs_te, bm25_c_te, bm25_fs_te, cos_lsa_c_te, cos_lsa_fs_te, te_t_w = get_pairwise_vectors(test_titles, test_contents, test_fs)

# ==============================================================================
# EKSTRAKSI 44 FITUR TER-REFINEMENT (BEDAH NEGASI & PROVINCE RESOLUTION)
# ==============================================================================
def is_location_covered(loc, target_locs, target_tokens):
    # 1. Exact canonical match
    if loc in target_locs or any(p in target_tokens for p in loc.split()):
        return True
    # 2. Hierarchical province match (PROVINCE_OF)
    prov = PROVINCE_OF.get(loc)
    if prov:
        if prov in target_locs or any(p in target_tokens for p in prov.split()):
            return True
        # Check if any target loc shares the same province
        if any(PROVINCE_OF.get(tl) == prov for tl in target_locs):
            return True
    return False

def check_location_conflict(locs_1, locs_2):
    if not locs_1 or not locs_2:
        return 0.0
    # If there is direct overlap
    if locs_1 & locs_2:
        return 0.0
    # Check if resolved provinces conflict
    provs_1 = {PROVINCE_OF.get(l, l) for l in locs_1}
    provs_2 = {PROVINCE_OF.get(l, l) for l in locs_2}
    if provs_1 & provs_2:
        return 0.0 # Same province -> no conflict
    # Genuine conflict between different provinces/regions
    return 1.0

def extract_features_refined(df_titles, df_contents, df_fs,
                             cos_wc, cos_wfs, cos_cc, cos_cfs,
                             bm25_c, bm25_fs,
                             cos_lsa_c, cos_lsa_fs,
                             title_tfidf_matrix):
    feats = []
    best_win_texts = []
    N_items = len(df_titles)
    
    for i in range(N_items):
        t_raw = df_titles[i]
        c_raw = df_contents[i]
        t_lower = t_raw.lower()
        c_lower = c_raw.lower()
        fs_lower = df_fs[i].lower()
        
        t_words = t_raw.split()
        c_words = c_raw.split()
        t_lower_words = t_lower.split()
        c_lower_words = c_lower.split()
        
        t_tokens = set(re.findall(r'\w+', t_lower))
        c_tokens = set(re.findall(r'\w+', c_lower))
        fs_tokens = set(re.findall(r'\w+', fs_lower))
        
        useful_t = [w for w in t_tokens if w not in STOP_WORDS and len(w) > 2]
        
        # 1. Canonical Locations with PROVINCE_OF & Collocation Guard (Tugas 3 & 4)
        t_locs = get_canonical_locations(t_lower)
        c_locs = get_canonical_locations(c_lower)
        fs_locs = get_canonical_locations(fs_lower)
        
        loc_missing = sum(1 for loc in t_locs if not is_location_covered(loc, c_locs, c_tokens))
        loc_mismatch = loc_missing / (len(t_locs) + 1e-9) if t_locs else 0.0
        loc_conflict = check_location_conflict(t_locs, fs_locs) if loc_missing > 0 else 0.0
        loc_c_conflict = check_location_conflict(t_locs, c_locs) if loc_missing > 0 else 0.0
        
        # 2. Canonical Figures
        t_figs = get_canonical_figures(t_lower)
        c_figs = get_canonical_figures(c_lower)
        fig_missing = sum(1 for fig in t_figs if fig not in c_figs and not any(p in c_tokens for p in fig.split()))
        fig_mismatch = fig_missing / (len(t_figs) + 1e-9) if t_figs else 0.0
        fig_conflict = 1.0 if (t_figs and c_figs and fig_missing > 0 and not t_figs.intersection(c_figs)) else 0.0
        
        # 3. Institutions
        t_inst = {w for w in t_tokens if w in INSTITUTIONS}
        inst_missing = len([inst for inst in t_inst if inst not in c_tokens])
        inst_mismatch = inst_missing / (len(t_inst) + 1e-9) if t_inst else 0.0
        
        # 4. Token & Stemming Coverage
        exact_match = sum(1 for w in useful_t if w in c_tokens)
        c_stems = set(light_stem(w) for w in c_tokens if len(w) > 2)
        stem_match = sum(1 for w in useful_t if (w in c_tokens or light_stem(w) in c_stems))
        tok_cov_exact = exact_match / (len(useful_t) + 1e-9) if useful_t else 1.0
        tok_cov_stem = stem_match / (len(useful_t) + 1e-9) if useful_t else 1.0
        stem_gain = tok_cov_stem - tok_cov_exact
        
        # 5. Multi-Scale Discourse Windows
        lead_25_tokens = set(re.findall(r'\w+', " ".join(c_words[:25]).lower()))
        lead_50_tokens = set(re.findall(r'\w+', " ".join(c_words[:50]).lower()))
        lead_80_tokens = set(re.findall(r'\w+', " ".join(c_words[:80]).lower()))
        
        lead_25_cov = len([w for w in useful_t if w in lead_25_tokens]) / (len(useful_t) + 1e-9) if useful_t else 1.0
        lead_50_cov = len([w for w in useful_t if w in lead_50_tokens]) / (len(useful_t) + 1e-9) if useful_t else 1.0
        lead_80_cov = len([w for w in useful_t if w in lead_80_tokens]) / (len(useful_t) + 1e-9) if useful_t else 1.0
        
        windows_50 = []
        w_size = 50
        step = 20
        for s in range(0, max(1, len(c_words)), step):
            chunk = " ".join(c_words[s:s+w_size])
            if len(chunk.split()) >= 5:
                windows_50.append(chunk)
        if not windows_50:
            windows_50 = [c_raw]
            
        best_overlap_50 = 0
        best_win_idx_50 = 0
        overlap_counts = []
        for w_idx, win in enumerate(windows_50):
            wtoks = set(re.findall(r'\w+', win.lower()))
            overlap = len([w for w in useful_t if w in wtoks])
            overlap_counts.append(overlap)
            if overlap > best_overlap_50:
                best_overlap_50 = overlap
                best_win_idx_50 = w_idx
                
        best_50_cov = best_overlap_50 / (len(useful_t) + 1e-9) if useful_t else 1.0
        best_win_pos_norm = best_win_idx_50 / (len(windows_50) + 1e-9)
        best_minus_lead = best_50_cov - lead_50_cov
        best_win_texts.append(windows_50[best_win_idx_50])
        
        sorted_overlaps = sorted(overlap_counts, reverse=True)
        top1_overlap = sorted_overlaps[0] if sorted_overlaps else 0
        top2_overlap = sorted_overlaps[1] if len(sorted_overlaps) > 1 else top1_overlap
        top1_top2_contrast = (top1_overlap - top2_overlap) / (len(useful_t) + 1e-9) if useful_t else 0.0
        
        # 6. Targeted Fact Guards (TUGAS 2: head_entity_absence di-DROP)
        num_units = re_num_unit.findall(t_lower)
        if num_units:
            matches = 0
            for num, unit in num_units:
                pat = rf"{re.escape(num)}\s+(?:\w+\s+){{0,2}}{re.escape(unit)}"
                if re.search(pat, c_lower):
                    matches += 1
            num_unit_match_ratio = matches / len(num_units)
        else:
            num_unit_match_ratio = 1.0
            
        # 7. TUGAS 1: Sinyal Negasi Berbasis Morfologi (light_stem)
        lead_words = c_lower_words[:60]
        lead_text = " ".join(lead_words)
        best_win_words = windows_50[best_win_idx_50].lower().split()
        
        # Fungsi evaluasi negasi berpasangan berbasis stem identik
        def evaluate_morphological_negation(target_words_list):
            asym_found = 0.0
            for idx, w in enumerate(t_lower_words):
                # Deteksi kata kerja / aksi
                is_action = (w in INHERENT_NEG_VERBS) or (len(w) > 4 and w.startswith(('me', 'di', 'ber', 'ter'))) or w in ['bisa', 'dapat', 'sebabkan', 'izinkan', 'bolehkan', 'ada']
                if is_action:
                    t_is_neg = (w in INHERENT_NEG_VERBS) or (idx > 0 and t_lower_words[idx-1] in NEG_WORDS) or (idx > 1 and t_lower_words[idx-2] in NEG_WORDS)
                    stem_w = light_stem(w)
                    if len(stem_w) < 3:
                        continue
                    
                    matched_instances = []
                    for j, cw in enumerate(target_words_list):
                        stem_cw = light_stem(cw)
                        # Verifikasi stem identik
                        if cw == w or (len(stem_cw) >= 3 and stem_w == stem_cw):
                            c_is_neg = (cw in INHERENT_NEG_VERBS) or (j > 0 and target_words_list[j-1] in NEG_WORDS) or (j > 1 and target_words_list[j-2] in NEG_WORDS)
                            matched_instances.append(c_is_neg)
                    
                    if matched_instances:
                        # Asimetri polaritas negasi pada verba/stem yang sama
                        if not t_is_neg and all(matched_instances):
                            asym_found = 1.0
                            break
                        elif t_is_neg and not any(matched_instances):
                            asym_found = 1.0
                            break
            return asym_found

        v_conf = evaluate_morphological_negation(lead_words)
        lead_neg_asymmetry = v_conf
        neg_inversion = evaluate_morphological_negation(best_win_words)
        
        refuting_matches = len(re_refuting.findall(windows_50[best_win_idx_50]))
        refuting_score = float(min(refuting_matches, 3))
        
        t_clean_words = [w for w in re.findall(r'\b[a-z0-9]+\b', t_lower) if len(w) > 2 and w not in STOP_WORDS]
        if len(t_clean_words) >= 2:
            title_ordered_bgs = [f"{t_clean_words[j]} {t_clean_words[j+1]}" for j in range(len(t_clean_words)-1)]
            matched_bgs = sum(1 for bg in title_ordered_bgs if bg in c_lower)
            ordered_bigram_ratio = matched_bgs / len(title_ordered_bgs)
        else:
            ordered_bigram_ratio = 1.0
            
        # 8. Antonym & Conflict
        ada_conflict = 0.0
        t_negs = t_tokens & NEGATION_WORDS
        if ('ada' in t_tokens and not t_negs):
            if 'tidak ada' in fs_lower or 'tak ada' in fs_lower:
                ada_conflict = 1.0
        elif ('tidak ada' in t_lower or 'tak ada' in t_lower):
            if 'ada' in fs_tokens and 'tidak ada' not in fs_lower and 'tak ada' not in fs_lower:
                ada_conflict = 1.0
                
        antonym_conflict = 0.0
        for w1, w2 in EXPANDED_ANTONYM_PAIRS:
            if (w1 in t_tokens and w2 in c_tokens) or (w2 in t_tokens and w1 in c_tokens):
                antonym_conflict = 1.0
                break
                
        # 9. Numbers
        t_subst_nums = extract_numbers_comprehensive(t_raw)
        c_subst_nums = extract_numbers_comprehensive(c_raw)
        num_missing_subst = len(t_subst_nums - c_subst_nums)
        num_cov_subst = len(t_subst_nums.intersection(c_subst_nums)) / (len(t_subst_nums) + 1e-9) if t_subst_nums else 1.0
        
        # 10. Capital Words & Lengths
        is_title_case = sum(1 for w in t_words if w and w[0].isupper()) > 0.8 * len(t_words)
        if not is_title_case:
            cap_words = [w for w in t_words if len(w) > 1 and w[0].isupper() and w.lower() not in STOP_WORDS]
            cap_missing = sum(1 for w in cap_words if w.lower() not in c_tokens)
            cap_missing_fs = sum(1 for w in cap_words if w.lower() not in fs_tokens)
            cap_cov_c = 1.0 - (cap_missing / (len(cap_words) + 1e-9)) if cap_words else 1.0
            cap_cov_fs = 1.0 - (cap_missing_fs / (len(cap_words) + 1e-9)) if cap_words else 1.0
        else:
            cap_cov_c = 1.0
            cap_cov_fs = 1.0
            
        head_content_words = [w.lower() for w in t_words if w.lower() not in STOP_WORDS][:3]
        head_missing_c = sum(1 for hw in head_content_words if hw not in c_tokens)
        head_cov_c = 1.0 - (head_missing_c / (len(head_content_words) + 1e-9)) if head_content_words else 1.0
        
        len_ratio = len(c_raw) / (len(t_raw) + 1e-9)
        
        # 11. Inversion Scopes Tambahan
        # Existential Polarity Conflict
        e_conf = 0.0
        t_has_ada = 'ada ' in t_lower[:25] or t_lower.startswith('ada ')
        t_neg_ada = 'tidak ada' in t_lower[:30] or 'tak ada' in t_lower[:30]
        c_has_ada = 'ada ' in lead_text[:30] or lead_text.startswith('ada ')
        c_neg_ada = 'tidak ada' in lead_text[:35] or 'tak ada' in lead_text[:35]
        if (t_has_ada and not t_neg_ada) and c_neg_ada:
            e_conf = 1.0
        elif t_neg_ada and (c_has_ada and not c_neg_ada):
            e_conf = 1.0
            
        # Composite Entity Conflict with PROVINCE_OF
        comp_conf = 0.0
        m_inst = list(re_inst_marker.finditer(t_lower))
        m_loc_t = list(LOC_REGEX.finditer(t_lower))
        if m_inst and m_loc_t:
            for inst in m_inst:
                for loc in m_loc_t:
                    if 0 <= loc.start() - inst.end() <= 30 or 0 <= inst.start() - loc.end() <= 30:
                        inst_name = inst.group(0)
                        loc_name_t = CANONICAL_LOC.get(loc.group(0), loc.group(0))
                        idx_c = c_lower.find(inst_name)
                        if idx_c != -1:
                            surrounding = c_lower[max(0, idx_c-30):min(len(c_lower), idx_c+len(inst_name)+30)]
                            locs_c = {CANONICAL_LOC.get(m.group(0), m.group(0)) for m in LOC_REGEX.finditer(surrounding)}
                            if locs_c and not is_location_covered(loc_name_t, locs_c, set(surrounding.split())):
                                comp_conf = 1.0
                                break
                if comp_conf == 1.0:
                    break
                    
        # Directional Role Inversion
        dir_conf = 0.0
        prep_m = list(re.finditer(r'\b(ke|kepada)\s+([a-zA-Z]+)', t_lower))
        if prep_m:
            for m in prep_m:
                target = m.group(2)
                if len(target) > 3 and target not in ['mana', 'sana', 'depan', 'dalam', 'luar']:
                    actor_pat = rf"\b{re.escape(target)}\s+(?:akan|telah|sudah|kembali|bakal|siap|resmi|mulai|menyumbang|menyerahkan|memberi)"
                    if re.search(actor_pat, lead_text):
                        dir_conf = 1.0
                        break
        
        # Append 43 raw features (head_entity_absence DIBUANG)
        feats.append([
            cos_wc[i], cos_wfs[i], cos_cc[i], cos_cfs[i],
            bm25_c[i], bm25_fs[i],
            cos_lsa_c[i], cos_lsa_fs[i],
            best_50_cov, best_win_pos_norm, best_minus_lead,
            lead_25_cov, lead_50_cov, lead_80_cov, top1_top2_contrast,
            loc_mismatch, loc_conflict, loc_c_conflict,
            fig_mismatch, fig_conflict,
            inst_mismatch,
            tok_cov_exact, tok_cov_stem, stem_gain, head_cov_c,
            num_unit_match_ratio, # head_entity_absence dropped here!
            neg_inversion, lead_neg_asymmetry, ada_conflict, antonym_conflict,
            refuting_score, ordered_bigram_ratio,
            num_missing_subst, num_cov_subst,
            cap_cov_c, cap_cov_fs, len_ratio, len(t_words), len(c_words),
            v_conf, e_conf, comp_conf, dir_conf
        ])
    
    win_matrix = tfidf_word.transform(best_win_texts)
    best_win_tfidf_cos = np.asarray((title_tfidf_matrix.multiply(win_matrix)).sum(axis=1)).flatten()
    
    feature_matrix = np.array(feats)
    # Insert best_win_tfidf_cos at column index 8 -> total becomes 44 columns!
    feature_matrix = np.insert(feature_matrix, 8, best_win_tfidf_cos, axis=1)
    return feature_matrix

print("Mengekstrak 44 fitur refined untuk train dan test...")
X_train_refined = extract_features_refined(
    train_titles, train_contents, train_fs,
    cos_w_c_tr, cos_w_fs_tr, cos_c_c_tr, cos_c_fs_tr,
    bm25_c_tr, bm25_fs_tr,
    cos_lsa_c_tr, cos_lsa_fs_tr,
    tr_t_w
)
X_test_refined = extract_features_refined(
    test_titles, test_contents, test_fs,
    cos_w_c_te, cos_w_fs_te, cos_c_c_te, cos_c_fs_te,
    bm25_c_te, bm25_fs_te,
    cos_lsa_c_te, cos_lsa_fs_te,
    te_t_w
)

print(f"X_train_refined shape: {X_train_refined.shape} (Harus 44 fitur)")
print(f"X_test_refined shape:  {X_test_refined.shape} (Harus 44 fitur)")
assert X_train_refined.shape[1] == 44, f"Dimensi fitur salah: {X_train_refined.shape[1]} != 44"

# Simpan cache fitur refined
np.savez('features_44_refined.npz', X_train=X_train_refined, X_test=X_test_refined)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_refined)
X_test_scaled = scaler.transform(X_test_refined)

# ==============================================================================
# TUGAS 5: RETRAINING 5-FOLD STRATIFIED CV & META-LEARNER
# ==============================================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sample_weights = np.ones(len(train))

xgb_oof = np.zeros(len(train))
hgb_oof = np.zeros(len(train))
et_oof = np.zeros(len(train))
lr_oof = np.zeros(len(train))

xgb_test = np.zeros(len(test))
hgb_test = np.zeros(len(test))
et_test = np.zeros(len(test))
lr_test = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_refined, y_train)):
    print(f"--- Training Fold {fold+1}/5 (Refined 44-Features) ---")
    
    # 1. XGBoost
    clf_xgb = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.035,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss'
    )
    clf_xgb.fit(X_train_refined[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    xgb_oof[val_idx] = clf_xgb.predict_proba(X_train_refined[val_idx])[:, 1]
    xgb_test += clf_xgb.predict_proba(X_test_refined)[:, 1] / 5.0
    
    # 2. HistGradientBoosting
    clf_hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.035, max_depth=6, random_state=SEED
    )
    clf_hgb.fit(X_train_refined[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    hgb_oof[val_idx] = clf_hgb.predict_proba(X_train_refined[val_idx])[:, 1]
    hgb_test += clf_hgb.predict_proba(X_test_refined)[:, 1] / 5.0
    
    # 3. ExtraTrees
    clf_et = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=SEED, n_jobs=-1)
    clf_et.fit(X_train_refined[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    et_oof[val_idx] = clf_et.predict_proba(X_train_refined[val_idx])[:, 1]
    et_test += clf_et.predict_proba(X_test_refined)[:, 1] / 5.0
    
    # 4. LogisticRegression
    clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf_lr.fit(X_train_scaled[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    lr_oof[val_idx] = clf_lr.predict_proba(X_train_scaled[val_idx])[:, 1]
    lr_test += clf_lr.predict_proba(X_test_scaled)[:, 1] / 5.0

# Stacking Meta-Learner Cross Validation
meta_X = np.column_stack([xgb_oof, hgb_oof, et_oof, lr_oof])
meta_X_test = np.column_stack([xgb_test, hgb_test, et_test, lr_test])

meta_oof = np.zeros(len(train))
meta_test = np.zeros(len(test))

for trn, val in skf.split(meta_X, y_train):
    meta_clf = LogisticRegression(C=1.0, random_state=SEED)
    meta_clf.fit(meta_X[trn], y_train[trn])
    meta_oof[val] = meta_clf.predict_proba(meta_X[val])[:, 1]
    meta_test += meta_clf.predict_proba(meta_X_test)[:, 1] / 5.0

# Threshold Evaluation on range [0.780, 0.830]
print("\n" + "="*60)
print("EVALUASI KONSENSUS THRESHOLD OPTIMAL (0.780 - 0.830)")
print("="*60)

best_th, best_f1 = 0.800, 0.0
for th in np.arange(0.780, 0.831, 0.005):
    preds = (meta_oof >= th).astype(int)
    sc = f1_score(y_train, preds, average='macro')
    test_c0 = ((meta_test >= th).astype(int) == 0).sum()
    print(f"Threshold: {th:.3f} | OOF Macro F1: {sc:.4f} | Test C0 Count: {test_c0}")
    if sc > best_f1:
        best_f1, best_th = sc, th

print("\n" + "="*60)
print(f"HASIL REFINED BASELINE: Optimal Threshold = {best_th:.3f} | OOF Macro F1 = {best_f1:.4f}")
print("="*60)

oof_preds = (meta_oof >= best_th).astype(int)
print("\nClassification Report (Refined Baseline):")
print(classification_report(y_train, oof_preds, digits=4))

cm_refined = confusion_matrix(y_train, oof_preds)
print("Confusion Matrix (Refined Baseline):")
print(cm_refined)

rec_c0 = cm_refined[0, 0] / (cm_refined[0, 0] + cm_refined[0, 1])
prec_c0 = cm_refined[0, 0] / (cm_refined[0, 0] + cm_refined[1, 0])
fp_refined = cm_refined[1, 0]

print(f"\nDetail Kelas 0:")
print(f"  Recall Kelas 0    : {rec_c0*100:.2f}% ({cm_refined[0, 0]}/{cm_refined[0, 0] + cm_refined[0, 1]})")
print(f"  Precision Kelas 0 : {prec_c0*100:.2f}% ({cm_refined[0, 0]}/{cm_refined[0, 0] + cm_refined[1, 0]})")
print(f"  False Positive    : {fp_refined}")

# Baseline Lama (Sebelum Refinement)
# Baseline commit c562eb8: FP = 491, Recall = 41.06% (590/1437), F1 = 0.7089 (or 0.6985 di LB)
fp_old = 491
print("\n" + "="*60)
print("KOMPARASI BASELINE RAFFIE LAMA vs BASELINE REFINED BARU")
print(f"  False Positive Lama : {fp_old}")
print(f"  False Positive Baru : {fp_refined}")
print(f"  Penurunan FP        : {fp_old - fp_refined} kasus ({((fp_old - fp_refined)/fp_old)*100:+.2f}%)")
print("="*60)

# ==============================================================================
# PEMBUATAN SUBMISSION: submission_baseline_refined.csv
# ==============================================================================
print("\n" + "="*60)
print("MEMBUAT FILE SUBMISSION: submission_baseline_refined.csv")
print("="*60)

# Cek threshold yang menghasilkan tebakan Kelas 0 di rentang optimal (310-350 baris)
# Jika best_th menghasilkan jumlah di luar rentang, periksa threshold dalam grid
chosen_th = best_th
test_preds = (meta_test >= chosen_th).astype(int)
c0_count = (test_preds == 0).sum()

if not (310 <= c0_count <= 350):
    print(f"Catatan: Threshold argmax {best_th:.3f} menghasilkan {c0_count} Kelas 0.")
    # Cari threshold konsensus terdekat yang menghasilkan Kelas 0 dalam [310, 350]
    best_prior_th = chosen_th
    best_prior_f1 = 0.0
    for th_cand in np.arange(0.780, 0.865, 0.002):
        t_c0 = ((meta_test >= th_cand).astype(int) == 0).sum()
        if 310 <= t_c0 <= 350:
            sc_cand = f1_score(y_train, (meta_oof >= th_cand).astype(int), average='macro')
            if sc_cand > best_prior_f1:
                best_prior_f1 = sc_cand
                best_prior_th = th_cand
    if best_prior_f1 > 0:
        print(f"Threshold konsensus prior yang dipilih: {best_prior_th:.3f} (OOF F1: {best_prior_f1:.4f})")
        chosen_th = best_prior_th
        test_preds = (meta_test >= chosen_th).astype(int)
        c0_count = (test_preds == 0).sum()

submission = pd.DataFrame({'id': test['id'], 'label': test_preds})
sub_file = 'submission_baseline_refined.csv'
submission.to_csv(sub_file, index=False)

c1_count = (submission['label'] == 1).sum()

print(f"Distribusi Prediksi {sub_file}:")
print(f"  Kelas 1: {c1_count} ({c1_count/len(submission)*100:.2f}%)")
print(f"  Kelas 0: {c0_count} ({c0_count/len(submission)*100:.2f}%)")

# Asersi Integritas
assert len(submission) == 3603, f"Jumlah baris salah: {len(submission)} != 3603"
assert list(submission.columns) == ['id', 'label'], f"Nama kolom salah: {submission.columns}"
assert submission.isnull().sum().sum() == 0, "Ditemukan nilai null pada submission!"
assert set(submission['label'].unique()).issubset({0, 1}), "Label bukan biner {0, 1}!"
print("\nSELURUH ASERSI INTEGRITAS RESMI TERPENUHI DENGAN SUKSES!")

print(f"\nSelesai! Berkas tersimpan di {sub_file}")
