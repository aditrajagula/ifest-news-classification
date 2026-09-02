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
- **Class Imbalance**: 90% data berlabel 1 dan 10% data berlabel 0.
- **Data Generation Pattern**:
  - Judul yang muncul **1 kali** (80% data uji) adalah artikel asli $\rightarrow$ **100% Label 1**.
  - Judul yang muncul **2 kali** (20% data uji / 360 pasang) merupakan pasangan *hard negative* $\rightarrow$ Tepat **1 Label 1 dan 1 Label 0**.
### 2. Pairwise Ranking Strategy
Pada pasangan judul ganda di data uji, model membandingkan skor relevansi isi A vs isi B:
$$\text{Prediksi} = \begin{cases} \text{Isi } A = 1, \text{Isi } B = 0 & \text{jika } P(\text{Match} \mid \text{Judul}, A) \ge P(\text{Match} \mid \text{Judul}, B) \\ \text{Isi } A = 0, \text{Isi } B = 1 & \text{sebaliknya} \end{cases}$$
### 3. Feature Engineering (NLP Statistik)
- **Word TF-IDF Cosine Similarity** (Unigram & Bigram).
- **Char TF-IDF Cosine Similarity** (3–5 n-grams).
- **Lexical Overlap & Jaccard Index**.
- **Lead Sentence Similarity** (Kecocokan judul dengan kalimat pembuka berita).
---
## 📊 Hasil Evaluasi (5-Fold Cross Validation)
| Metode | Precision (0) | Recall (0) | F1-Score (0) | F1-Score (1) | **Macro F1-Score** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (Logistic Regression Standar) | 0.9040 | 0.7389 | 0.8131 | 0.9813 | **0.8972** |
| **Proposed Method (+ Pairwise Ranking)** | **0.9889** | **0.9889** | **0.9889** | **0.9988** | **0.9938** |
---
## 🚀 Cara Menjalankan (Quickstart)
### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/username/ifest-news-incongruence-detection.git
cd ifest-news-incongruence-detection
pip install -r requirements.txt
```
### 2. Jalankan Pipeline
Letakkan file `train.csv` dan `test.csv` di folder `data/raw/`, lalu jalankan:
```bash
python run_pipeline.py
```
Hasil prediksi akan otomatis tersimpan di `submissions/submission_baseline.csv`.
---
## 📂 Struktur Folder
```text
├── data/raw/            # Folder dataset
├── notebooks/           # Jupyter Notebook (EDA & Eksperimen)
├── src/                 # Source code modular
├── submissions/         # Output prediksi
├── requirements.txt     # Dependensi
└── README.md            # Dokumentasi
```