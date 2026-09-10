def extract_features(df_titles, df_contents, df_fs,
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
        
        # 1. Canonical Locations
        t_locs = get_canonical_locations(t_lower)
        c_locs = get_canonical_locations(c_lower)
        fs_locs = get_canonical_locations(fs_lower)
        loc_missing = sum(1 for loc in t_locs if loc not in c_locs and not any(p in c_tokens for p in loc.split()))
        loc_mismatch = loc_missing / (len(t_locs) + 1e-9) if t_locs else 0.0
        loc_conflict = 1.0 if (t_locs and fs_locs and loc_missing > 0 and not t_locs.intersection(fs_locs)) else 0.0
        loc_c_conflict = 1.0 if (t_locs and c_locs and loc_missing > 0 and not t_locs.intersection(c_locs)) else 0.0
        
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
        
        # 5. Multi-Scale Discourse Windows (25, 50, 80 token)
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
        
        # 6. Targeted Fact Guards
        head_ent = ""
        for j, w in enumerate(t_words[:4]):
            clean_w = re.sub(r'\W+', '', w)
            if clean_w and clean_w.lower() not in STOP_WORDS and len(clean_w) > 2:
                if w[0].isupper():
                    if j+1 < len(t_words) and t_words[j+1][0].isupper():
                        next_w = re.sub(r'\W+', '', t_words[j+1])
                        head_ent = f"{clean_w.lower()} {next_w.lower()}"
                    else:
                        head_ent = clean_w.lower()
                break
        head_entity_absence = 1.0 if (head_ent and head_ent not in c_lower) else 0.0
        
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
                
        # 8. Numbers Comprehensive
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
        
        # 10. Adaptive Discourse Inversion Scopes (4 Fitur Baru Penekan False Negatives)
        t_lower_words = t_lower.split()
        lead_words = c_lower.split()[:60]
        lead_text = " ".join(lead_words)
        
        # 10a. Verb Negation Scope Conflict
        v_conf = 0.0
        for idx, w in enumerate(t_lower_words):
            is_action = (w in INHERENT_NEG_VERBS) or (len(w) > 4 and w.startswith(('me', 'di', 'ber', 'ter'))) or w in ['bisa', 'dapat', 'sebabkan', 'izinkan', 'bolehkan']
            if is_action:
                t_is_neg = (w in INHERENT_NEG_VERBS) or (idx > 0 and t_lower_words[idx-1] in NEG_WORDS) or (idx > 1 and t_lower_words[idx-2] in NEG_WORDS)
                c_instances = []
                for j, cw in enumerate(lead_words):
                    if cw == w or (len(w) >= 5 and len(cw) >= 5 and w[:5] == cw[:5]):
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
            
        # 10c. Composite Entity Conflict
        comp_conf = 0.0
        m_inst = list(re_inst_marker.finditer(t_lower))
        m_loc_t = list(LOC_REGEX.finditer(t_lower))
        if m_inst and m_loc_t:
            for inst in m_inst:
                for loc in m_loc_t:
                    if 0 <= loc.start() - inst.end() <= 30 or 0 <= inst.start() - loc.end() <= 30:
                        inst_name = inst.group(0)
                        loc_name_t = loc.group(0)
                        idx_c = c_lower.find(inst_name)
                        if idx_c != -1:
                            surrounding = c_lower[max(0, idx_c-30):min(len(c_lower), idx_c+len(inst_name)+30)]
                            locs_c = [m.group(0) for m in LOC_REGEX.finditer(surrounding)]
                            if locs_c and loc_name_t not in locs_c:
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
            head_entity_absence, num_unit_match_ratio,
            neg_inversion, lead_neg_asymmetry, ada_conflict, antonym_conflict,
            refuting_score, ordered_bigram_ratio,
            num_missing_subst, num_cov_subst,
            cap_cov_c, cap_cov_fs, len_ratio, len(t_words), len(c_words),
            v_conf, e_conf, comp_conf, dir_conf
        ])
    
    # Ultra-fast Batch TF-IDF transform for best focal window cosine
    win_matrix = tfidf_word.transform(best_win_texts)
    best_win_tfidf_cos = np.asarray((title_tfidf_matrix.multiply(win_matrix)).sum(axis=1)).flatten()
    
    feature_matrix = np.array(feats)
    # Insert best_win_tfidf_cos at column index 8
    feature_matrix = np.insert(feature_matrix, 8, best_win_tfidf_cos, axis=1)
    return feature_matrix

print("Mengekstrak 45 fitur teroptimasi (Opsi A) untuk data train dan test...")
X_train = extract_features(
    train_titles, train_contents, train_fs,
    cos_w_c_tr, cos_w_fs_tr, cos_c_c_tr, cos_c_fs_tr,
    bm25_c_tr, bm25_fs_tr,
    cos_lsa_c_tr, cos_lsa_fs_tr,
    tr_t_w
)
X_test = extract_features(
    test_titles, test_contents, test_fs,
    cos_w_c_te, cos_w_fs_te, cos_c_c_te, cos_c_fs_te,
    bm25_c_te, bm25_fs_te,
    cos_lsa_c_te, cos_lsa_fs_te,
    te_t_w
)
y_train = train['label'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")