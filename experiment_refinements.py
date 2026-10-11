import os
import random
import re
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train['label'].values

import json
nb = json.load(open('ifest_c562eb8.ipynb', encoding='utf-8'))

# Run cell 5 (lexical)
exec(''.join(nb['cells'][5]['source']))

# TUGAS 3: Tambahkan PROVINCE_OF dan ADDITIONAL_PROVINCE_OF
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

# Tambahkan seluruh lokasi baru ke ALL_LOCATIONS & LOC_REGEX
for k in PROVINCE_OF:
    if k not in ALL_LOCATIONS:
        ALL_LOCATIONS.append(k)
LOC_REGEX = re.compile(r'\b(' + '|'.join(re.escape(loc) for loc in sorted(ALL_LOCATIONS, key=len, reverse=True)) + r')\b', re.I)

# TUGAS 4: Collocation Guard
NON_GEO_ARAB_CONTEXT = {'bahasa', 'huruf', 'tulisan', 'latin', 'doa', 'sholat', 'salat', 'kitab', 'ayat'}
NATIONALITY_SPORT_CONTEXT = {'timnas', 'pemain', 'warga', 'wni', 'wna', 'atlet', 'paspor', 'striker', 'bek', 'gelandang', 'kiper'}

def get_canonical_locations_guarded(text):
    text_lower = text.lower()
    raw_locs = []
    for m in LOC_REGEX.finditer(text_lower):
        loc = m.group(0).lower()
        start_char = m.start()
        end_char = m.end()
        ctx_before = text_lower[max(0, start_char - 30):start_char]
        ctx_after = text_lower[end_char:min(len(text_lower), end_char + 30)]
        ctx_tokens = set(re.findall(r'\b[a-zA-Z]+\b', ctx_before + ' ' + ctx_after))
        
        # Collocation guard for 'arab'
        if loc in ['arab', 'saudi'] and (ctx_tokens & NON_GEO_ARAB_CONTEXT):
            continue
        # Collocation guard for 'nigeria'
        if loc in ['nigeria'] and (ctx_tokens & NATIONALITY_SPORT_CONTEXT):
            continue
            
        raw_locs.append(loc)
        
    return set(CANONICAL_LOC.get(loc, loc) for loc in raw_locs)

# Run cell 7 (TF-IDF, SVD, BM25)
exec(''.join(nb['cells'][7]['source']), globals())

# Ekstraksi Fitur dengan 4 Perbaikan Bedah Sesuai Prompt
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
        t_tokens = set(re.findall(r'\w+', t_lower))
        c_tokens = set(re.findall(r'\w+', c_lower))
        fs_tokens = set(re.findall(r'\w+', fs_lower))
        
        useful_t = [w for w in t_tokens if w not in STOP_WORDS and len(w) > 2]
        
        # 1. Canonical Locations (Tugas 3: PROVINCE_OF & Tugas 4: Collocation Guard)
        t_locs = get_canonical_locations_guarded(t_lower)
        c_locs = get_canonical_locations_guarded(c_lower)
        fs_locs = get_canonical_locations_guarded(fs_lower)
        
        # Location missing: jika loc ada di PROVINCE_OF, provinsinya juga dicek!
        loc_missing = 0
        for loc in t_locs:
            prov = PROVINCE_OF.get(loc)
            is_present = (loc in c_locs) or any(p in c_tokens for p in loc.split())
            if not is_present and prov:
                is_present = (prov in c_locs) or any(p in c_tokens for p in prov.split())
            if not is_present:
                loc_missing += 1
                
        loc_mismatch = loc_missing / (len(t_locs) + 1e-9) if t_locs else 0.0
        
        # Conflict: jika salah satu provinsinya sama -> no conflict
        def check_conflict(locs_a, locs_b):
            if not locs_a or not locs_b or loc_missing == 0:
                return 0.0
            if locs_a.intersection(locs_b):
                return 0.0
            provs_a = {PROVINCE_OF.get(l, l) for l in locs_a}
            provs_b = {PROVINCE_OF.get(l, l) for l in locs_b}
            if provs_a.intersection(provs_b):
                return 0.0
            return 1.0
            
        loc_conflict = check_conflict(t_locs, fs_locs)
        loc_c_conflict = check_conflict(t_locs, c_locs)
        
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
            
        # Sinyal Negasi
        t_negs = t_tokens & NEGATION_WORDS
        best_w_tokens = set(re.findall(r'\w+', windows_50[best_win_idx_50].lower()))
        win_negs = best_w_tokens & NEGATION_WORDS
        lead_negs = lead_50_tokens & NEGATION_WORDS
        neg_inversion = 1.0 if (bool(t_negs) != bool(win_negs)) else 0.0
        lead_neg_asymmetry = 1.0 if (bool(t_negs) != bool(lead_negs)) else 0.0
        
        refuting_matches = len(re_refuting.findall(windows_50[best_win_idx_50]))
        refuting_score = float(min(refuting_matches, 3))
        
        t_clean_words = [w for w in re.findall(r'\b[a-z0-9]+\b', t_lower) if len(w) > 2 and w not in STOP_WORDS]
        if len(t_clean_words) >= 2:
            title_ordered_bgs = [f"{t_clean_words[j]} {t_clean_words[j+1]}" for j in range(len(t_clean_words)-1)]
            matched_bgs = sum(1 for bg in title_ordered_bgs if bg in c_lower)
            ordered_bigram_ratio = matched_bgs / len(title_ordered_bgs)
        else:
            ordered_bigram_ratio = 1.0
            
        # 7. Antonym & Conflict
        ada_conflict = 0.0
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
                
        # 8. Numbers
        t_subst_nums = extract_numbers_comprehensive(t_raw)
        c_subst_nums = extract_numbers_comprehensive(c_raw)
        num_missing_subst = len(t_subst_nums - c_subst_nums)
        num_cov_subst = len(t_subst_nums.intersection(c_subst_nums)) / (len(t_subst_nums) + 1e-9) if t_subst_nums else 1.0
        
        # 9. Capital Words & Lengths
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
        
        # 10. TUGAS 1: Verb Negation Scope Conflict dengan light_stem (Ganti w[:5])
        t_lower_words = t_lower.split()
        lead_words = c_lower.split()[:60]
        lead_text = " ".join(lead_words)
        
        v_conf = 0.0
        for idx, w in enumerate(t_lower_words):
            is_action = (w in INHERENT_NEG_VERBS) or (len(w) > 4 and w.startswith(('me', 'di', 'ber', 'ter'))) or w in ['bisa', 'dapat', 'sebabkan', 'izinkan', 'bolehkan']
            if is_action:
                t_is_neg = (w in INHERENT_NEG_VERBS) or (idx > 0 and t_lower_words[idx-1] in NEG_WORDS) or (idx > 1 and t_lower_words[idx-2] in NEG_WORDS)
                stem_w = light_stem(w)
                c_instances = []
                for j, cw in enumerate(lead_words):
                    stem_cw = light_stem(cw)
                    # VERIFIKASI STEM DENGAN light_stem (GANTI w[:5] == cw[:5])
                    if cw == w or (len(stem_w) >= 3 and len(stem_cw) >= 3 and stem_w == stem_cw):
                        c_neg = (cw in INHERENT_NEG_VERBS) or (j > 0 and lead_words[j-1] in NEG_WORDS) or (j > 1 and lead_words[j-2] in NEG_WORDS)
                        c_instances.append(c_neg)
                if c_instances:
                    if not t_is_neg and all(c_instances):
                        v_conf = 1.0
                        break
                    elif t_is_neg and not any(c_instances):
                        v_conf = 1.0
                        break
                        
        # 10b. Existential Polarity Conflict
        e_conf = 0.0
        t_has_ada = 'ada ' in t_lower[:25] or t_lower.startswith('ada ')
        t_neg_ada = 'tidak ada' in t_lower[:30] or 'tak ada' in t_lower[:30]
        c_has_ada = 'ada ' in lead_text[:30] or lead_text.startswith('ada ')
        c_neg_ada = 'tidak ada' in lead_text[:35] or 'tak ada' in lead_text[:35]
        if (t_has_ada and not t_neg_ada) and c_neg_ada:
            e_conf = 1.0
        elif t_neg_ada and (c_has_ada and not c_neg_ada):
            e_conf = 1.0
            
        # 10c. Composite Entity Conflict (dengan PROVINCE_OF)
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
                            prov_t = PROVINCE_OF.get(loc_name_t, loc_name_t)
                            provs_c = {PROVINCE_OF.get(lc, lc) for lc in locs_c}
                            if locs_c and loc_name_t not in locs_c and prov_t not in provs_c:
                                comp_conf = 1.0
                                break
                if comp_conf == 1.0:
                    break
                    
        # 10d. Directional Role Inversion
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
        
        # 43 features (head_entity_absence di-drop)
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
            num_unit_match_ratio, # head_entity_absence dropped here
            neg_inversion, lead_neg_asymmetry, ada_conflict, antonym_conflict,
            refuting_score, ordered_bigram_ratio,
            num_missing_subst, num_cov_subst,
            cap_cov_c, cap_cov_fs, len_ratio, len(t_words), len(c_words),
            v_conf, e_conf, comp_conf, dir_conf
        ])
    
    win_matrix = tfidf_word.transform(best_win_texts)
    best_win_tfidf_cos = np.asarray((title_tfidf_matrix.multiply(win_matrix)).sum(axis=1)).flatten()
    
    feature_matrix = np.array(feats)
    feature_matrix = np.insert(feature_matrix, 8, best_win_tfidf_cos, axis=1) # 44 columns
    return feature_matrix

print("Mengekstrak 44 fitur refined terintegrasi...")
X_train_ref = extract_features_refined(
    train_titles, train_contents, train_fs,
    cos_w_c_tr, cos_w_fs_tr, cos_c_c_tr, cos_c_fs_tr,
    bm25_c_tr, bm25_fs_tr,
    cos_lsa_c_tr, cos_lsa_fs_tr,
    tr_t_w
)
X_test_ref = extract_features_refined(
    test_titles, test_contents, test_fs,
    cos_w_c_te, cos_w_fs_te, cos_c_c_te, cos_c_fs_te,
    bm25_c_te, bm25_fs_te,
    cos_lsa_c_te, cos_lsa_fs_te,
    te_t_w
)
print(f"X_train_ref shape: {X_train_ref.shape} | X_test_ref shape: {X_test_ref.shape}")

# Simpan cache
np.savez('features_44_perfect.npz', X_train=X_train_ref, X_test=X_test_ref)

# Evaluasi 4-Model Stacking
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ref)
X_test_scaled = scaler.transform(X_test_ref)

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

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_ref, y_train)):
    clf_xgb = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.035, subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss')
    clf_xgb.fit(X_train_ref[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    xgb_oof[val_idx] = clf_xgb.predict_proba(X_train_ref[val_idx])[:, 1]
    xgb_test += clf_xgb.predict_proba(X_test_ref)[:, 1] / 5.0
    
    clf_hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.035, max_depth=6, random_state=SEED)
    clf_hgb.fit(X_train_ref[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    hgb_oof[val_idx] = clf_hgb.predict_proba(X_train_ref[val_idx])[:, 1]
    hgb_test += clf_hgb.predict_proba(X_test_ref)[:, 1] / 5.0
    
    clf_et = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=SEED, n_jobs=-1)
    clf_et.fit(X_train_ref[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    et_oof[val_idx] = clf_et.predict_proba(X_train_ref[val_idx])[:, 1]
    et_test += clf_et.predict_proba(X_test_ref)[:, 1] / 5.0
    
    clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf_lr.fit(X_train_scaled[trn_idx], y_train[trn_idx], sample_weight=sample_weights[trn_idx])
    lr_oof[val_idx] = clf_lr.predict_proba(X_train_scaled[val_idx])[:, 1]
    lr_test += clf_lr.predict_proba(X_test_scaled)[:, 1] / 5.0

meta_X = np.column_stack([xgb_oof, hgb_oof, et_oof, lr_oof])
meta_X_test = np.column_stack([xgb_test, hgb_test, et_test, lr_test])

meta_oof = np.zeros(len(train))
meta_test = np.zeros(len(test))

for trn, val in skf.split(meta_X, y_train):
    meta_clf = LogisticRegression(C=1.0, random_state=SEED)
    meta_clf.fit(meta_X[trn], y_train[trn])
    meta_oof[val] = meta_clf.predict_proba(meta_X[val])[:, 1]
    meta_test += meta_clf.predict_proba(meta_X_test)[:, 1] / 5.0

print("\n--- Evaluasi Threshold Konsensus (0.780 - 0.840) ---")
best_th, best_f1 = 0.810, 0.0
for th in np.arange(0.780, 0.841, 0.005):
    preds = (meta_oof >= th).astype(int)
    sc = f1_score(y_train, preds, average='macro')
    test_c0 = ((meta_test >= th).astype(int) == 0).sum()
    cm = confusion_matrix(y_train, preds)
    print(f"Th: {th:.3f} | F1: {sc:.4f} | FP: {cm[1, 0]:3d} | Recall C0: {cm[0, 0]}/{cm[0, 0]+cm[0, 1]} ({cm[0, 0]/(cm[0, 0]+cm[0, 1])*100:.2f}%) | Test C0: {test_c0:3d}")
    if sc > best_f1:
        best_f1, best_th = sc, th

print(f"\n[HASIL REFINED BASELINE] Optimal: Th={best_th:.3f} | F1={best_f1:.4f}")
oof_preds = (meta_oof >= best_th).astype(int)
cm_opt = confusion_matrix(y_train, oof_preds)
print("Confusion Matrix:")
print(cm_opt)
print(f"FP: {cm_opt[1, 0]} (Penurunan FP dari baseline lama 491: {491 - cm_opt[1, 0]} kasus)")
