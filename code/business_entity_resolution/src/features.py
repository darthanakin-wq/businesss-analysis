import math
import re
from typing import Dict, List, Tuple, Set, Any, Optional
import pandas as pd

try:
    import rapidfuzz
    from rapidfuzz.distance import Levenshtein, JaroWinkler
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

import difflib  # Always needed for _lcs_ratio

# No config imports needed - features are self-contained


def _safe_jaccard(set1: Set, set2: Set) -> float:
    if not set1 and not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def _safe_overlap(set1: Set, set2: Set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    min_len = min(len(set1), len(set2))
    return intersection / min_len if min_len > 0 else 0.0


def _safe_containment(set1: Set, set2: Set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    return max(intersection / len(set1), intersection / len(set2))


def _safe_len_ratio(s1: str, s2: str) -> float:
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


def _ngrams(s: str, n: int = 3) -> Set[str]:
    return {s[i:i+n] for i in range(max(1, len(s) - n + 1))}


def _normalized_levenshtein(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
        
    if HAS_RAPIDFUZZ:
        return Levenshtein.normalized_distance(s1, s2)
    else:
        # Fallback using difflib
        return 1.0 - difflib.SequenceMatcher(None, s1, s2).ratio()


def _jaro_winkler(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 0.0
        
    if HAS_RAPIDFUZZ:
        return JaroWinkler.normalized_similarity(s1, s2)
    else:
        # Fallback using difflib
        return difflib.SequenceMatcher(None, s1, s2).ratio()


def _lcs_ratio(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 0.0
        
    if HAS_RAPIDFUZZ:
        # If rapidfuzz available, use difflib anyway for LCS approx or difflib
        matcher = difflib.SequenceMatcher(None, s1, s2)
        match = matcher.find_longest_match(0, len(s1), 0, len(s2))
        lcs_len = match.size
    else:
        matcher = difflib.SequenceMatcher(None, s1, s2)
        match = matcher.find_longest_match(0, len(s1), 0, len(s2))
        lcs_len = match.size
        
    max_len = max(len(s1), len(s2))
    return lcs_len / max_len if max_len > 0 else 0.0


def _extract_numbers(s: str) -> Set[str]:
    return set(re.findall(r'\d+', s))


def _extract_alpha(s: str) -> Set[str]:
    return set(re.findall(r'[a-zA-Z]+', s))


def compute_pair_features(s1_row: pd.Series, candidate_row: pd.Series, meta: dict = None) -> dict:
    """
    Compute similarity features for a candidate pair.
    
    Args:
        s1_row: pd.Series representing the first entity (from S1).
        candidate_row: pd.Series representing the second entity (candidate).
        meta: Optional dict containing metadata (bi_encoder_score, etc.)
    
    Returns:
        dict: A dictionary of feature names and their computed values.
    """
    if meta is None:
        meta = {}
        
    features = {}
    
    # Safely get strings — use the correct column names from preprocessing.py
    n1 = str(s1_row.get('name_norm', ''))
    n2 = str(candidate_row.get('name_norm', ''))
    n1 = '' if n1 == 'nan' else n1
    n2 = '' if n2 == 'nan' else n2
    
    a1 = str(s1_row.get('addr_norm', ''))
    a2 = str(candidate_row.get('addr_norm', ''))
    a1 = '' if a1 == 'nan' else a1
    a2 = '' if a2 == 'nan' else a2
    
    c1 = f"{n1} {a1}".strip()
    c2 = f"{n2} {a2}".strip()

    t_n1 = set(n1.split()) if n1 else set()
    t_n2 = set(n2.split()) if n2 else set()
    
    t_a1 = set(a1.split()) if a1 else set()
    t_a2 = set(a2.split()) if a2 else set()

    # 1. Name Features
    features['name_jaccard_token'] = _safe_jaccard(t_n1, t_n2)
    features['name_jaccard_ngram'] = _safe_jaccard(_ngrams(n1, 3), _ngrams(n2, 3))
    features['name_levenshtein'] = _normalized_levenshtein(n1, n2)
    features['name_jaro_winkler'] = _jaro_winkler(n1, n2)
    features['name_tfidf_cosine'] = meta.get('name_tfidf_cosine', 0.0)
    
    s_n1 = " ".join(sorted(list(t_n1)))
    s_n2 = " ".join(sorted(list(t_n2)))
    features['name_sorted_match'] = 1.0 if s_n1 and s_n2 and s_n1 == s_n2 else 0.0
    
    features['name_lcs_ratio'] = _lcs_ratio(n1, n2)
    features['name_token_overlap'] = _safe_overlap(t_n1, t_n2)
    
    first_n1 = n1.split()[0] if n1 else ""
    first_n2 = n2.split()[0] if n2 else ""
    features['name_first_token_match'] = 1.0 if first_n1 and first_n2 and first_n1 == first_n2 else 0.0
    
    # Acronym match
    acronym_1 = "".join([t[0] for t in n1.split() if t])
    acronym_2 = "".join([t[0] for t in n2.split() if t])
    acronym_match_score = 0.0
    if n1 and n2:
        if acronym_1 == n2.replace(" ", "") or acronym_2 == n1.replace(" ", ""):
            acronym_match_score = 1.0
    features['name_acronym_match'] = acronym_match_score
    
    features['name_containment'] = _safe_containment(t_n1, t_n2)
    features['name_len_ratio'] = _safe_len_ratio(n1, n2)
    
    # 2. Address Features
    features['addr_jaccard_token'] = _safe_jaccard(t_a1, t_a2)
    features['addr_levenshtein'] = _normalized_levenshtein(a1, a2)
    
    # Use addr_postal from preprocessing
    p1 = str(s1_row.get('addr_postal', ''))
    p2 = str(candidate_row.get('addr_postal', ''))
    p1 = '' if p1 == 'nan' else p1
    p2 = '' if p2 == 'nan' else p2
    
    features['addr_postal_match'] = 1.0 if p1 and p2 and p1 == p2 else 0.0
    features['addr_postal_present'] = 1.0 if p1 and p2 else 0.0
    
    features['addr_number_match'] = _safe_jaccard(_extract_numbers(a1), _extract_numbers(a2))
    features['addr_tfidf_cosine'] = meta.get('addr_tfidf_cosine', 0.0)
    features['addr_alpha_jaccard'] = _safe_jaccard(_extract_alpha(a1), _extract_alpha(a2))
    features['addr_token_overlap'] = _safe_overlap(t_a1, t_a2)
    features['addr_len_ratio'] = _safe_len_ratio(a1, a2)
    
    # Has landmark — check if original address had landmark patterns
    orig_a1 = str(s1_row.get('business_address', ''))
    orig_a2 = str(candidate_row.get('business_address', ''))
    import re
    lm_pattern = re.compile(r'\b(?:near|opp|opposite|behind|beside|next to)\b', re.IGNORECASE)
    features['addr_has_landmark'] = 1.0 if lm_pattern.search(orig_a1) or lm_pattern.search(orig_a2) else 0.0

    # 3. Combined Features
    t_c1 = set(c1.split()) if c1 else set()
    t_c2 = set(c2.split()) if c2 else set()
    features['combined_jaccard'] = _safe_jaccard(t_c1, t_c2)
    features['combined_levenshtein'] = _normalized_levenshtein(c1, c2)
    
    # Use country_norm from preprocessing
    ctry1 = str(s1_row.get('country_norm', ''))
    ctry2 = str(candidate_row.get('country_norm', ''))
    ctry1 = '' if ctry1 == 'nan' else ctry1
    ctry2 = '' if ctry2 == 'nan' else ctry2
    features['country_match'] = 1.0 if ctry1 and ctry2 and ctry1 == ctry2 else 0.0
    
    # name_in_addr
    name_in_addr_flag = 0.0
    if len(t_n1.intersection(t_a2)) > 0 or len(t_n2.intersection(t_a1)) > 0:
        name_in_addr_flag = 1.0
    features['name_in_addr'] = name_in_addr_flag
    
    l_c1 = len(t_c1)
    l_c2 = len(t_c2)
    features['total_token_ratio'] = min(l_c1, l_c2) / max(l_c1, l_c2) if max(l_c1, l_c2) > 0 else 0.0
    
    # source_indicator — determine from entity_id prefix
    cand_eid = str(candidate_row.get('entity_id', ''))
    features['source_indicator'] = 1.0 if cand_eid.startswith('S3') else 0.0

    # 4. Meta Features
    features['bi_encoder_score'] = float(meta.get('bi_encoder_score', 0.0))
    features['cross_encoder_score'] = float(meta.get('cross_encoder_score', 0.0))
    features['blocking_layers_count'] = float(meta.get('blocking_layers_count', 1.0))
    features['faiss_rank'] = float(meta.get('faiss_rank', 999.0))
    features['num_candidates'] = float(meta.get('num_candidates', 1.0))
    features['best_candidate_score'] = float(meta.get('best_candidate_score', 0.0))
    features['score_margin'] = float(meta.get('score_margin', 0.0))

    return features


def compute_batch_features(s1_df: pd.DataFrame, candidates_df: pd.DataFrame, pairs: List[Tuple[Any, Any]], meta_map: Dict[Tuple[Any, Any], Dict] = None) -> pd.DataFrame:
    """
    Vectorized/batch computation of features for many pairs.
    
    Args:
        s1_df: DataFrame of S1 entities. Index or an 'id' column should exist.
        candidates_df: DataFrame of candidate entities.
        pairs: List of (s1_entity_id, candidate_entity_id) tuples.
        meta_map: Dict mapping (s1_id, candidate_id) to meta dicts.
    
    Returns:
        pd.DataFrame: DataFrame where each row corresponds to a pair and columns are features.
    """
    if meta_map is None:
        meta_map = {}
        
    s1_df = s1_df.set_index('id') if 'id' in s1_df.columns else s1_df
    candidates_df = candidates_df.set_index('id') if 'id' in candidates_df.columns else candidates_df

    feature_rows = []
    
    for s1_id, cand_id in pairs:
        try:
            s1_row = s1_df.loc[s1_id]
            cand_row = candidates_df.loc[cand_id]
            
            # loc might return DataFrame if index is not unique
            if isinstance(s1_row, pd.DataFrame):
                s1_row = s1_row.iloc[0]
            if isinstance(cand_row, pd.DataFrame):
                cand_row = cand_row.iloc[0]
                
            meta = meta_map.get((s1_id, cand_id), {})
            
            feats = compute_pair_features(s1_row, cand_row, meta)
            feats['s1_entity_id'] = s1_id
            feats['candidate_entity_id'] = cand_id
            
            feature_rows.append(feats)
        except KeyError:
            # If id not found, skip or log
            continue
            
    return pd.DataFrame(feature_rows)
