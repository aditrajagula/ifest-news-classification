def extract_features(df_titles, df_contents, df_fs,
                     cos_wc, cos_wfs, cos_cc, cos_cfs,
                     bm25_c, bm25_fs,
                     cos_lsa_c, cos_lsa_fs,
                     title_tfidf_matrix):
    feats = []
    for i in range(len(df_titles)):
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
        
        # 1. Canonical Location Matching
        t_locs = get_canonical_locations(t_lower)
        c_locs = get_canonical_locations(c_lower)
        fs_locs = get_canonical_locations(fs_lower)
        loc_missing = sum(1 for loc in t_locs if loc not in c_locs and not any(p in c_tokens for p in loc.split()))
        loc_mismatch = loc_missing / (len(t_locs) + 1e-9) if t_locs else 0.0
        loc_conflict = 1.0 if (t_locs and fs_locs and loc_missing > 0 and not t_locs.intersection(fs_locs)) else 0.0
        loc_c_conflict = 1.0 if (t_locs and c_locs and loc_missing > 0 and not t_locs.intersection(c_locs)) else 0.0
        
        # 2. Canonical Figure Matching
        t_figs = get_canonical_figures(t_lower)
        c_figs = get_canonical_figures(c_lower)
        fig_missing = sum(1 for fig in t_figs if fig not in c_figs and not any(p in c_tokens for p in fig.split()))
        fig_mismatch = fig_missing / (len(t_figs) + 1e-9) if t_figs else 0.0
        fig_conflict = 1.0 if (t_figs and c_figs and fig_missing > 0 and not t_figs.intersection(c_figs)) else 0.0
        
        # 3. Institution Matching
        t_inst = {w for w in t_tokens if w in INSTITUTIONS}
        inst_missing = len([inst for inst in t_inst if inst not in c_tokens])
        inst_mismatch = inst_missing / (len(t_inst) + 1e-9) if t_inst else 0.0
        
        # 4. Morphology & Token Coverage
        exact_match = sum(1 for w in useful_t if w in c_tokens)
        c_stems = set(light_stem(w) for w in c_tokens if len(w) > 2)
        stem_match = sum(1 for w in useful_t if (w in c_tokens or light_stem(w) in c_stems))
        tok_cov_exact = exact_match / (len(useful_t) + 1e-9) if useful_t else 1.0
        tok_cov_stem = stem_match / (len(useful_t) + 1e-9) if useful_t else 1.0
        stem_gain = tok_cov_stem - tok_cov_exact
        
        # 5. Best Window Alignment & IDF-Weighted Window Cosine
        windows = []
        w_size = 25
        step = 12
        for start in range(0, len(c_words), step):
            chunk = " ".join(c_words[start:start+w_size])
            if len(chunk.split()) >= 5:
                windows.append(chunk)
        if not windows:
            windows = [c_raw]
            
        best_overlap = 0
        best_window = windows[0]
        best_win_idx = 0
        for w_idx, win in enumerate(windows):
            wtoks = set(re.findall(r'\w+', win.lower()))
            overlap = len([w for w in useful_t if w in wtoks])
            if overlap > best_overlap:
                best_overlap = overlap
                best_window = win
                best_win_idx = w_idx
                
        best_win_cov = best_overlap / (len(useful_t) + 1e-9) if useful_t else 1.0
        win_vec = tfidf_word.transform([best_window])
        best_win_tfidf_cos = float((title_tfidf_matrix[i].multiply(win_vec)).sum())
        
        # 6. Sentence-Level Spread & Fact Fragmentation
        sentences = [s.strip().lower() for s in SENT_SPLIT.split(c_raw) if len(s.strip().split()) >= 3]
        if not sentences:
            sentences = [c_lower]
        sent_token_sets = [set(re.findall(r'\w+', s)) for s in sentences]
        
        max_single_sent_cov = 0.0
        for stoks in sent_token_sets:
            cov = len([w for w in useful_t if w in stoks]) / (len(useful_t) + 1e-9)
            if cov > max_single_sent_cov:
                max_single_sent_cov = cov
                
        lead_stoks = sent_token_sets[0] if len(sent_token_sets) > 0 else set()
        if len(sent_token_sets) > 1:
            lead_stoks = lead_stoks.union(sent_token_sets[1])
        lead_cov = len([w for w in useful_t if w in lead_stoks]) / (len(useful_t) + 1e-9) if useful_t else 1.0
        
        covered = set()
        needed_sents = 0
        sorted_sents = sorted(sent_token_sets, key=lambda st: len(st.intersection(useful_t)), reverse=True)
        target_tokens = 0.75 * len(useful_t)
        for st in sorted_sents:
            new_covered = covered.union(st.intersection(useful_t))
            if len(new_covered) > len(covered):
                covered = new_covered
                needed_sents += 1
            if len(covered) >= target_tokens:
                break
        spread_ratio = needed_sents / (len(sentences) + 1e-9)
        
        # 7. Directional Negation & Expanded Antonyms
        best_w_tokens = set(re.findall(r'\w+', best_window.lower()))
        t_negs = t_tokens.intersection(NEGATION_WORDS)
        win_negs = best_w_tokens.intersection(NEGATION_WORDS)
        neg_inversion = 1.0 if (bool(t_negs) != bool(win_negs)) else 0.0
        title_neg_win_pos = 1.0 if (bool(t_negs) and not bool(win_negs)) else 0.0
        title_pos_win_neg = 1.0 if (not bool(t_negs) and bool(win_negs)) else 0.0
        
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
                
        # 8. Comprehensive Numbers
        t_subst_nums = extract_numbers_comprehensive(t_raw)
        c_subst_nums = extract_numbers_comprehensive(c_raw)
        num_missing_subst = len(t_subst_nums - c_subst_nums)
        num_cov_subst = len(t_subst_nums.intersection(c_subst_nums)) / (len(t_subst_nums) + 1e-9) if t_subst_nums else 1.0
        
        # 9. Generic Capitalized Entity N-grams
        words = t_raw.split()
        cap_entities = []
        for j in range(len(words)-1):
            if words[j] and words[j+1] and words[j][0].isupper() and words[j+1][0].isupper():
                w1_c = re.sub(r'\W+', '', words[j]).lower()
                w2_c = re.sub(r'\W+', '', words[j+1]).lower()
                if w1_c not in STOP_WORDS and w2_c not in STOP_WORDS and len(w1_c)>1 and len(w2_c)>1:
                    cap_entities.append(f"{w1_c} {w2_c}")
        generic_entity_missing = 0.0
        if cap_entities:
            generic_entity_missing = sum(1 for ent in cap_entities if ent not in c_lower) / len(cap_entities)
            
        # 10. Title-Case aware Capital Word checking
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
            
        # 11. Bigrams & Head token
        t_content_words = [w.lower() for w in t_words if w.lower() not in STOP_WORDS and len(w) > 2]
        t_bigrams = [f"{t_content_words[j]} {t_content_words[j+1]}" for j in range(len(t_content_words)-1)]
        bg_cov_c = sum(1 for bg in t_bigrams if bg in c_lower) / (len(t_bigrams) + 1e-9) if t_bigrams else 1.0
        
        head_content_words = [w.lower() for w in t_words if w.lower() not in STOP_WORDS][:3]
        head_missing_c = sum(1 for hw in head_content_words if hw not in c_tokens)
        head_cov_c = 1.0 - (head_missing_c / (len(head_content_words) + 1e-9)) if head_content_words else 1.0
        
        len_ratio = len(c_raw) / (len(t_raw) + 1e-9)
        
        feats.append([
            cos_wc[i], cos_wfs[i], cos_cc[i], cos_cfs[i],
            bm25_c[i], bm25_fs[i],
            cos_lsa_c[i], cos_lsa_fs[i],
            best_win_tfidf_cos, best_win_cov, best_win_idx,
            max_single_sent_cov, lead_cov, spread_ratio, needed_sents,
            loc_mismatch, loc_conflict, loc_c_conflict,
            fig_mismatch, fig_conflict,
            inst_mismatch,
            tok_cov_exact, tok_cov_stem, stem_gain, bg_cov_c, head_cov_c,
            generic_entity_missing,
            neg_inversion, title_neg_win_pos, title_pos_win_neg, ada_conflict, antonym_conflict,
            num_missing_subst, num_cov_subst,
            cap_cov_c, cap_cov_fs, len_ratio, len(t_words), len(c_words)
        ])
    return np.array(feats)

print("Mengekstrak 39 fitur teroptimasi untuk data train dan test...")
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
