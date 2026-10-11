import json
import sys
import os
import re
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

print("=== Building Updated IFest.ipynb ===")
nb_path = 'IFest.ipynb'
backup_path = r'C:\Users\Reynaldi Hutagaol\AppData\Roaming\Antigravity IDE\Backups\824dc8ab3e37371ce3b3319f12d08c13\file\-5ee1e86b'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 0: Top Markdown Declarations
nb['cells'][0]['source'] = [
    "# 🏆 IFest DAC 2026 — Data Analysis Competition (Penyisihan)\n",
    "**Format Penamaan File:** `Penyisihan_DAC2026_NamaTim.ipynb`  \n",
    "**Deklarasi Dataset Eksternal (Rule #3):** Menggunakan daftar referensi entitas wilayah (Kabupaten & Kota se-Indonesia) yang dikompilasi dari data publik kode pos & wilayah administratif BPS/Kemendagri (`nomor.net` / `kemendagri.go.id`) serta dataset terbuka kode wilayah administratif kecamatan & kelurahan (`https://raw.githubusercontent.com/edwardsamuel/Wilayah-Administratif-Indonesia/master/csv/districts.csv` dan `villages.csv`) yang bersifat publik dan gratis.  \n",
    "**Deklarasi Pustaka Eksternal (Rule #4 & #7):** Menggunakan pustaka open-source standar Python (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `gensim`, `imbalanced-learn`). Seluruh representasi teks (TF-IDF, Okapi BM25, TruncatedSVD LSA, Gensim FastText) dilatih mandiri (*strictly trained from scratch*) tanpa memanfaatkan model pra-latih (*no pretrained models/LLMs*) ataupun AutoML.  \n",
    "**Random Seed & Reproducibility (Rule #11):** Ditetapkan secara deterministik (`SEED = 42`) pada level Python, NumPy, dan environment OS untuk menjamin reproduktifitas penuh (sesuai Guidebook Hal. 5 Poin 6).  \n",
    "**Anti-Data Leakage Protocol (Rule #8):** Fitur ekstraksi dan scaling di-fit ketat HANYA pada data latih (`train.csv`). Threshold final dan bobot sampel Class 0 ditentukan murni dari konsensus Out-of-Fold (OOF) CV 5-fold, tanpa pernah menyentuh, membaca persentil, ataupun menghitung statistik distribusi data uji (`test.csv`)."
]

print("Cell 0 updated with comprehensive rules declaration!")
