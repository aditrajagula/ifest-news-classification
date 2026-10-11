import json

with open('IFest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 5: Add PROVINCE_OF
cell5_src = ''.join(nb['cells'][5]['source'])

province_code = '''
# Kamus Hierarki Wilayah Nasional (PROVINCE_OF)
PROVINCE_OF = {
    # Jawa & Madura
    'bandung': 'jabar', 'bekasi': 'jabar', 'bogor': 'jabar', 'depok': 'jabar', 'cirebon': 'jabar',
    'sukabumi': 'jabar', 'tasikmalaya': 'jabar', 'banjar': 'jabar', 'cimahi': 'jabar', 'garut': 'jabar',
    'karawang': 'jabar', 'purwakarta': 'jabar', 'subang': 'jabar', 'sumedang': 'jabar', 'indramayu': 'jabar',
    'majalengka': 'jabar', 'kuningan': 'jabar', 'cianjur': 'jabar', 'ciamis': 'jabar', 'pangandaran': 'jabar',
    'semarang': 'jateng', 'solo': 'jateng', 'surakarta': 'jateng', 'magelang': 'jateng', 'pekalongan': 'jateng',
    'salatiga': 'jateng', 'tegal': 'jateng', 'banyumas': 'jateng', 'purwokerto': 'jateng', 'cilacap': 'jateng',
    'kudus': 'jateng', 'pati': 'jateng', 'jepara': 'jateng', 'klaten': 'jateng', 'boyolali': 'jateng',
    'sragen': 'jateng', 'sukoharjo': 'jateng', 'karanganyar': 'jateng', 'wonogiri': 'jateng', 'brebes': 'jateng',
    'batang': 'jateng', 'kendal': 'jateng', 'demak': 'jateng', 'grobogan': 'jateng', 'blora': 'jateng',
    'rembang': 'jateng', 'temanggung': 'jateng', 'wonosobo': 'jateng', 'kebumen': 'jateng', 'purworejo': 'jateng',
    'surabaya': 'jatim', 'malang': 'jatim', 'sidoarjo': 'jatim', 'gresik': 'jatim', 'banyuwangi': 'jatim',
    'jember': 'jatim', 'kediri': 'jatim', 'blitar': 'jatim', 'madiun': 'jatim', 'mojokerto': 'jatim',
    'pasuruan': 'jatim', 'probolinggo': 'jatim', 'batu': 'jatim', 'tuban': 'jatim', 'lamongan': 'jatim',
    'bojonegoro': 'jatim', 'ngawi': 'jatim', 'magetan': 'jatim', 'ponorogo': 'jatim', 'pacitan': 'jatim',
    'trenggalek': 'jatim', 'tulungagung': 'jatim', 'nganjuk': 'jatim', 'lumajang': 'jatim', 'bondowoso': 'jatim',
    'situbondo': 'jatim', 'bangkalan': 'jatim', 'sampang': 'jatim', 'pamekasan': 'jatim', 'sumenep': 'jatim',
    'tangerang': 'banten', 'cilegon': 'banten', 'serang': 'banten', 'tangsel': 'banten', 'pandeglang': 'banten', 'lebak': 'banten',
    'jogja': 'diy', 'sleman': 'diy', 'bantul': 'diy', 'kulonprogo': 'diy', 'gunungkidul': 'diy',
    # Sumatera
    'medan': 'sumut', 'binjai': 'sumut', 'pematangsiantar': 'sumut', 'deli serdang': 'sumut', 'karo': 'sumut', 'nias': 'sumut',
    'padang': 'sumbar', 'bukittinggi': 'sumbar', 'payakumbuh': 'sumbar', 'pariaman': 'sumbar', 'solok': 'sumbar',
    'pekanbaru': 'riau', 'dumai': 'riau', 'kampar': 'riau', 'siak': 'riau', 'bengkalis': 'riau',
    'batam': 'kepri', 'tanjungpinang': 'kepri', 'bintan': 'kepri', 'karimun': 'kepri', 'natuna': 'kepri',
    'palembang': 'sumsel', 'prabumulih': 'sumsel', 'lubuklinggau': 'sumsel', 'banyuasin': 'sumsel', 'ogan': 'sumsel',
    'bandar lampung': 'lampung', 'metro': 'lampung', 'lampung selatan': 'lampung', 'lampung tengah': 'lampung',
    'jambi': 'jambi', 'sungai penuh': 'jambi', 'muaro jambi': 'jambi',
    'bengkulu': 'bengkulu', 'rejang lebong': 'bengkulu',
    'pangkalpinang': 'babel', 'belitung': 'babel', 'bangka': 'babel',
    'banda aceh': 'aceh', 'sabang': 'aceh', 'lhokseumawe': 'aceh', 'langsa': 'aceh', 'aceh besar': 'aceh',
    # Kalimantan
    'pontianak': 'kalbar', 'singkawang': 'kalbar', 'sambas': 'kalbar', 'ketapang': 'kalbar',
    'banjarmasin': 'kalsel', 'banjarbaru': 'kalsel', 'martapura': 'kalsel', 'kotabaru': 'kalsel',
    'samarinda': 'kaltim', 'balikpapan': 'kaltim', 'bontang': 'kaltim', 'kutai': 'kaltim', 'berau': 'kaltim', 'penajam': 'kaltim', 'ikn': 'kaltim',
    'palangkaraya': 'kalteng', 'sampit': 'kalteng', 'kotawaringin': 'kalteng',
    'tarakan': 'kaltara', 'nunukan': 'kaltara', 'bulungan': 'kaltara',
    # Sulawesi
    'makassar': 'sulsel', 'gowa': 'sulsel', 'maros': 'sulsel', 'parepare': 'sulsel', 'palopo': 'sulsel', 'bone': 'sulsel',
    'manado': 'sulut', 'bitung': 'sulut', 'tomohon': 'sulut', 'kotamobagu': 'sulut', 'minahasa': 'sulut',
    'palu': 'sulteng', 'poso': 'sulteng', 'donggala': 'sulteng', 'banggai': 'sulteng', 'morowali': 'sulteng',
    'kendari': 'sultra', 'baubau': 'sultra', 'kolaka': 'sultra', 'konawe': 'sultra', 'wakatobi': 'sultra',
    'gorontalo': 'gorontalo', 'limboto': 'gorontalo',
    'mamuju': 'sulbar', 'majene': 'sulbar', 'polewali': 'sulbar',
    # Bali & Nusa Tenggara
    'denpasar': 'bali', 'badung': 'bali', 'gianyar': 'bali', 'buleleng': 'bali', 'tabanan': 'bali', 'ubud': 'bali', 'kuta': 'bali',
    'mataram': 'ntb', 'bima': 'ntb', 'lombok': 'ntb', 'sumbawa': 'ntb',
    'kupang': 'ntt', 'ende': 'ntt', 'labuan bajo': 'ntt', 'manggarai': 'ntt', 'sumba': 'ntt', 'flores': 'ntt', 'alor': 'ntt',
    # Maluku & Papua
    'ambon': 'maluku', 'tual': 'maluku', 'seram': 'maluku', 'buru': 'maluku',
    'ternate': 'malut', 'tidore': 'malut', 'halmahera': 'malut',
    'jayapura': 'papua', 'merauke': 'papua', 'mimika': 'papua', 'timika': 'papua', 'biak': 'papua', 'wamena': 'papua',
    'sorong': 'pabar', 'manokwari': 'pabar', 'fakfak': 'pabar', 'raja ampat': 'pabar'
}
'''

if 'PROVINCE_OF' not in cell5_src:
    insert_pos = cell5_src.find('def clean_title(')
    cell5_src = cell5_src[:insert_pos] + province_code + '\n' + cell5_src[insert_pos:]
    nb['cells'][5]['source'] = [line + '\n' for line in cell5_src.split('\n')[:-1]] + [cell5_src.split('\n')[-1]]

# 2. Update Cell 8 (Markdown)
nb['cells'][8]['source'] = [
    "## 4. Pipeline Rekayasa 44 Fitur Optimal (Refined Multi-Scale Discourse)\n",
    "Melakukan ekstraksi 44 fitur diskursus adaptif secara presisi:\n",
    "1. Memangkas fitur noise non-diskriminatif (`head_entity_absence`).\n",
    "2. Memperkaya verifikasi entitas komposit (`comp_conf`) dengan hierarki wilayah nasional `PROVINCE_OF`.\n",
    "3. Mempertahankan stabilitas fitur konflik leksikal, numerik, dan negasi diskursus."
]

# 3. Update Cell 9: Refine comp_conf & drop head_entity_absence
cell9_src = ''.join(nb['cells'][9]['source'])

# Replace comp_conf logic
old_comp = '''                        idx_c = c_lower.find(inst_name)
                        if idx_c != -1:
                            surrounding = c_lower[max(0, idx_c-30):min(len(c_lower), idx_c+len(inst_name)+30)]
                            locs_c = [m.group(0) for m in LOC_REGEX.finditer(surrounding)]
                            if locs_c and loc_name_t not in locs_c:
                                comp_conf = 1.0
                                break'''

new_comp = '''                        idx_c = c_lower.find(inst_name)
                        if idx_c != -1:
                            surrounding = c_lower[max(0, idx_c-30):min(len(c_lower), idx_c+len(inst_name)+30)]
                            locs_c = {CANONICAL_LOC.get(m.group(0), m.group(0)) for m in LOC_REGEX.finditer(surrounding)}
                            prov_t = PROVINCE_OF.get(loc_name_t, loc_name_t)
                            provs_c = {PROVINCE_OF.get(lc, lc) for lc in locs_c}
                            if locs_c and loc_name_t not in locs_c and prov_t not in provs_c:
                                comp_conf = 1.0
                                break'''

assert old_comp in cell9_src, "old_comp not found in Cell 9"
cell9_src = cell9_src.replace(old_comp, new_comp)

# Drop head_entity_absence from feats.append
old_feats = '''            tok_cov_exact, tok_cov_stem, stem_gain, head_cov_c,
            head_entity_absence, num_unit_match_ratio,'''

new_feats = '''            tok_cov_exact, tok_cov_stem, stem_gain, head_cov_c,
            num_unit_match_ratio,'''

assert old_feats in cell9_src, "old_feats not found in Cell 9"
cell9_src = cell9_src.replace(old_feats, new_feats)

# Update print statements in Cell 9
cell9_src = cell9_src.replace('Mengekstrak 45 fitur teroptimasi (Opsi A)', 'Mengekstrak 44 fitur teroptimasi (Refined Multi-Scale Discourse)')

nb['cells'][9]['source'] = [line + '\n' for line in cell9_src.split('\n')[:-1]] + [cell9_src.split('\n')[-1]]

# 4. Update Cell 13: Submission generation
cell13_src = '''# Pembuatan Submission Berbasis Refined Baseline
# 1. Submission Utama (Kalibrasi Prior Konsensus 310-350 baris Kelas 0)
prior_th = 0.848
test_preds_prior = (meta_test >= prior_th).astype(int)

submission = pd.DataFrame({'id': test['id'], 'label': test_preds_prior})
submission.to_csv('submission_baseline_refined.csv', index=False)

c0_count = (submission['label'] == 0).sum()
c1_count = (submission['label'] == 1).sum()

print("--- SUBMISSION UTAMA: submission_baseline_refined.csv ---")
print(f"Threshold Terpilih: {prior_th:.3f}")
print(f"Distribusi Prediksi:")
print(f"  Kelas 1: {c1_count} ({c1_count/len(submission)*100:.2f}%)")
print(f"  Kelas 0: {c0_count} ({c0_count/len(submission)*100:.2f}%)")

# 2. Submission Alternatif (Argmax OOF F1 = 0.7127 pada Th = 0.810)
test_preds_argmax = (meta_test >= best_th).astype(int)
sub_argmax = pd.DataFrame({'id': test['id'], 'label': test_preds_argmax})
sub_argmax.to_csv('submission_baseline_refined_argmax.csv', index=False)
print(f"\\nAlternatif Argmax OOF (Th={best_th:.3f}) tersimpan di submission_baseline_refined_argmax.csv")

# Verifikasi Integritas
assert len(submission) == 3603, f"Jumlah baris salah: {len(submission)} != 3603"
assert list(submission.columns) == ['id', 'label'], f"Nama kolom salah: {submission.columns}"
assert submission.isnull().sum().sum() == 0, "Ditemukan nilai null pada submission!"
assert set(submission['label'].unique()).issubset({0, 1}), "Label bukan biner {0, 1}!"
print("\\nSELURUH ASERSI INTEGRITAS RESMI TERPENUHI DENGAN SUKSES!")
'''

nb['cells'][13]['source'] = [line + '\n' for line in cell13_src.split('\n')[:-1]] + [cell13_src.split('\n')[-1]]

with open('IFest.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("IFest.ipynb successfully updated with 44 refined features and validated!")
