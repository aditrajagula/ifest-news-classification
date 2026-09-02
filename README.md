# 📰 News Title-Content Incongruence Detection (IFest DAC)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F1%2099.38%25-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repository ini berisi solusi Machine Learning untuk mendeteksi ketidaksesuaian (*incongruence / clickbait*) antara judul dan isi berita berbahasa Indonesia pada kompetisi **IFest DAC**.

---

## 📌 Problem Statement
Membangun model klasifikasi biner untuk menentukan apakah isi berita mendukung judulnya:
- **Label 1 (Sesuai)**: Informasi utama pada judul dibahas dan didukung oleh isi berita.
- **Label 0 (Tidak Sesuai)**: Informasi pada judul tidak didukung, memiliki konteks berbeda, atau peristiwanya bertentangan.
- **Metrik Evaluasi**: **Macro F1-Score** $\left(\frac{F_{1,\text{class 0}} + F_{1,\text{class 1}}}{2}\right)$.

---

## 🔍 Key Findings & Methodology

### 1. Dataset Structural Insight
- **Class Imbalance**: 90% data berlabel 1 (12.960 sampel) dan 10% data berlabel 0 (1.440 sampel).
- **Data Generation Pattern**:
  - Judul yang muncul **1 kali** (80% data uji / 2.880 baris) adalah artikel asli $\rightarrow$ **100% Label 1**.
  - Judul yang muncul **2 kali** (20% data uji / 360 pasang judul ganda) merupakan pasangan *hard negative* $\rightarrow$ Tepat **1 Label 1 dan 1 Label 0**.

### 2. Pairwise Ranking Strategy
Pada pasangan judul ganda di data uji, model membandingkan skor relevansi isi A vs isi B:
\text{Prediksi} = \begin{cases} \text{Isi } A = 1, \text{Isi } B = 0 & \text{jika } P(\text{Match} \mid \text{Judul}, A) \ge P(\text{Match} \mid \text{Judul}, B) \\ \text{Isi } A = 0, \text{Isi } B = 1 & \text{sebaliknya} \end{cases}

### 3. Feature Engineering (NLP Statistik Non-Pretrained)
- **Word TF-IDF Cosine Similarity** (Unigram & Bigram).
- **Char TF-IDF Cosine Similarity** (3–5 n-grams).
- **Lexical Overlap & Jaccard Index**.
- **Token Count Features**.

---

## 📊 Hasil Evaluasi (5-Fold Cross Validation)

| Metode | Precision (0) | Recall (0) | F1-Score (0) | F1-Score (1) | **Macro F1-Score** | Akurasi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Logistic Regression Standar) | 0.9040 | 0.7389 | 0.8131 | 0.9813 | **0.8972** | 96.60% |
| **Proposed Method (+ Pairwise Ranking)** | **0.9889** | **0.9889** | **0.9889** | **0.9988** | **0.9938** | **99.78%** |

---

## 📂 Struktur File
`	ext
├── IFest.ipynb              # Notebook lengkap (EDA, Feature Engineering, CV, Submission)
├── train.csv                # Data latih (14.400 baris)
├── test.csv                 # Data uji (3.600 baris)
├── sample_submission.csv    # Template submission
├── submission_baseline.csv  # File hasil prediksi model (Macro F1 CV: 99.38%)
├── requirements.txt         # Dependensi Python
├── .gitignore               # Pengabaian file cache & temporary
└── README.md                # Dokumentasi proyek
`

---

## 🚀 Cara Menjalankan (Quickstart)

### 1. Install Dependensi
`ash
pip install -r requirements.txt
`

### 2. Jalankan Notebook
Buka dan jalankan seluruh cell di IFest.ipynb menggunakan Jupyter Notebook atau VS Code / Colab.
Output submission akan otomatis diekspor ke submission_baseline.csv.
