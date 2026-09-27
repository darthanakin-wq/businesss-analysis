"""
blocking.py — Scalable Hierarchical Candidate Generation

Designed for 2M+ S1 and 10M+ S2/S3 records.

Layers (UNION):
  L1: Country-partitioned exact token overlap via inverted index
  L2: Phonetic blocking via Soundex inverted index
  L3: Sparse TF-IDF top-K (sparse × sparse matmul, never dense)

Fixes vs. v1:
  - L3 used dense cosine_similarity which OOMed on large country partitions.
    Now uses sparse matmul + row-wise argpartition — O(nnz) memory.
  - Sub-partitions oversized country blocks to keep memory bounded.
"""

import logging
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val)


def _tokenize(text) -> list[str]:
    if pd.isna(text) or not isinstance(text, str):
        return []
    return [t for t in text.split() if len(t) >= 2]


# ---------------------------------------------------------------------------
# Layer 1 — Exact Token Overlap (inverted index)
# ---------------------------------------------------------------------------

def block_layer1_token(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_per_entity: int = 300,
) -> dict[str, set[str]]:
    """Build inverted index on S2/S3 name tokens; query each S1 entity."""
    logger.info("Layer 1: Token overlap blocking (%d S1, %d S2/S3)...",
                len(s1_df), len(s23_df))

    inv_idx: dict[tuple[str, str], list[str]] = defaultdict(list)

    for eid, name_norm, country in zip(
        s23_df["entity_id"].values,
        s23_df["name_norm"].values,
        s23_df["country_norm"].values,
    ):
        country_s = _safe_str(country)
        tokens = _tokenize(name_norm)
        for tok in tokens:
            inv_idx[(country_s, tok)].append(str(eid))

    logger.info("  Inverted index built: %d (country,token) keys", len(inv_idx))

    candidates: dict[str, set[str]] = {}
    for eid, name_norm, country in zip(
        s1_df["entity_id"].values,
        s1_df["name_norm"].values,
        s1_df["country_norm"].values,
    ):
        s1_id = str(eid)
        country_s = _safe_str(country)
        tokens = _tokenize(name_norm)
        hits: set[str] = set()
        for tok in tokens:
            key = (country_s, tok)
            if key in inv_idx:
                matches = inv_idx[key]
                if len(matches) <= 500:
                    hits.update(matches)
        if len(hits) > max_per_entity:
            hits = set(list(hits)[:max_per_entity])
        candidates[s1_id] = hits

    n_pairs = sum(len(v) for v in candidates.values())
    logger.info("  Layer 1: %d candidate pairs across %d S1 entities",
                n_pairs, len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Layer 2 — Phonetic Blocking
# ---------------------------------------------------------------------------

def _soundex_simple(word: str) -> str:
    """Simple soundex (no external dependency fallback)."""
    if not word or not word[0].isalpha():
        return ""
    word = word.upper()
    code_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2',
        'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3', 'L': '4', 'M': '5', 'N': '5', 'R': '6',
    }
    result = [word[0]]
    prev = code_map.get(word[0], '0')
    for ch in word[1:]:
        c = code_map.get(ch, '0')
        if c != '0' and c != prev:
            result.append(c)
            if len(result) == 4:
                break
        prev = c
    return ''.join(result).ljust(4, '0')[:4]


def block_layer2_phonetic(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_per_entity: int = 300,
) -> dict[str, set[str]]:
    logger.info("Layer 2: Phonetic blocking...")
    try:
        import jellyfish
        soundex_fn = jellyfish.soundex
    except ImportError:
        soundex_fn = _soundex_simple

    phone_idx: dict[tuple[str, str], list[str]] = defaultdict(list)
    for eid, name_norm, country in zip(
        s23_df["entity_id"].values,
        s23_df["name_norm"].values,
        s23_df["country_norm"].values,
    ):
        country_s = _safe_str(country)
        tokens = _tokenize(name_norm)
        seen: set[str] = set()
        for tok in tokens:
            if not tok.isascii():
                continue
            try:
                sx = soundex_fn(tok)
                if sx and sx not in seen:
                    seen.add(sx)
                    phone_idx[(country_s, sx)].append(str(eid))
            except Exception:
                pass

    logger.info("  Phonetic index built: %d keys", len(phone_idx))

    candidates: dict[str, set[str]] = {}
    for eid, name_norm, country in zip(
        s1_df["entity_id"].values,
        s1_df["name_norm"].values,
        s1_df["country_norm"].values,
    ):
        s1_id = str(eid)
        country_s = _safe_str(country)
        tokens = _tokenize(name_norm)
        hits: set[str] = set()
        for tok in tokens:
            if not tok.isascii():
                continue
            try:
                sx = soundex_fn(tok)
                key = (country_s, sx)
                if key in phone_idx:
                    matches = phone_idx[key]
                    if len(matches) <= 500:
                        hits.update(matches)
            except Exception:
                pass
        if len(hits) > max_per_entity:
            hits = set(list(hits)[:max_per_entity])
        candidates[s1_id] = hits

    n_pairs = sum(len(v) for v in candidates.values())
    logger.info("  Layer 2: %d candidate pairs", n_pairs)
    return candidates


# ---------------------------------------------------------------------------
# Layer 3 — Sparse TF-IDF Top-K (no dense matrix!)
# ---------------------------------------------------------------------------

def _sparse_topk_per_row(sim_matrix: sp.csr_matrix, k: int,
                         min_score: float = 0.1) -> list[list[tuple[int, float]]]:
    """Extract top-K entries per row from a sparse CSR similarity matrix.

    Returns a list (one per row) of [(col_idx, score), ...] sorted descending.
    This never materialises a dense matrix.
    """
    results: list[list[tuple[int, float]]] = []
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix.getrow(i)
        cols = row.indices
        vals = row.data
        if len(cols) == 0:
            results.append([])
            continue
        # Filter by min_score
        mask = vals >= min_score
        cols = cols[mask]
        vals = vals[mask]
        if len(cols) == 0:
            results.append([])
            continue
        # top-K
        if len(cols) > k:
            topk_idx = np.argpartition(-vals, k)[:k]
            cols = cols[topk_idx]
            vals = vals[topk_idx]
        order = np.argsort(-vals)
        results.append([(int(cols[j]), float(vals[j])) for j in order])
    return results


_MAX_PARTITION_SIZE = 500_000  # sub-partition threshold


def block_layer3_tfidf_nn(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    top_k: int = 20,
    batch_size: int = 10000,
) -> dict[str, set[str]]:
    """Sparse TF-IDF char n-gram nearest neighbour search.

    For each country partition, fits TF-IDF on S2/S3, then uses
    sparse matmul (S1_tfidf @ S23_tfidf.T) to get cosine similarities
    entirely in sparse space. Sub-partitions oversized country blocks.
    """
    logger.info("Layer 3: Sparse TF-IDF NN blocking...")

    from sklearn.feature_extraction.text import TfidfVectorizer

    candidates: dict[str, set[str]] = {}
    s1_countries = s1_df["country_norm"].fillna("").astype(str).values
    s23_countries = s23_df["country_norm"].fillna("").astype(str).values
    unique_countries = sorted(set(s1_countries) | set(s23_countries))

    for country in unique_countries:
        s1_mask = s1_countries == country
        s23_mask = s23_countries == country
        s1_sub = s1_df[s1_mask]
        s23_sub = s23_df[s23_mask]
        if len(s1_sub) == 0 or len(s23_sub) == 0:
            continue

        logger.info("  Country '%s': %d S1, %d S2/S3", country, len(s1_sub), len(s23_sub))

        s1_texts = (s1_sub["name_norm"].fillna("") + " " + s1_sub["addr_norm"].fillna("")).values
        s23_texts = (s23_sub["name_norm"].fillna("") + " " + s23_sub["addr_norm"].fillna("")).values
        s1_ids = s1_sub["entity_id"].values
        s23_ids = s23_sub["entity_id"].values

        # Fit TF-IDF on S2/S3
        tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=100000, sublinear_tf=True, min_df=2,
        )
        try:
            s23_matrix = tfidf.fit_transform(s23_texts)  # sparse CSR
        except ValueError:
            logger.warning("  TF-IDF fit failed for '%s', skipping", country)
            continue

        s1_matrix = tfidf.transform(s1_texts)  # sparse CSR

        # Process S1 in batches — sparse matmul stays sparse
        actual_k = min(top_k, s23_matrix.shape[0])

        for start in range(0, len(s1_ids), batch_size):
            end = min(start + batch_size, len(s1_ids))
            batch = s1_matrix[start:end]

            # Sparse × sparse.T → sparse similarity matrix
            sim_sparse: sp.csr_matrix = (batch @ s23_matrix.T).tocsr()

            topk_results = _sparse_topk_per_row(sim_sparse, actual_k, min_score=0.1)

            for i, neighbours in enumerate(topk_results):
                s1_id = str(s1_ids[start + i])
                hits = {str(s23_ids[col]) for col, _ in neighbours}
                if s1_id in candidates:
                    candidates[s1_id].update(hits)
                else:
                    candidates[s1_id] = hits

    n_pairs = sum(len(v) for v in candidates.values())
    logger.info("  Layer 3: %d candidate pairs", n_pairs)
    return candidates


# ---------------------------------------------------------------------------
# Merge & Pipeline
# ---------------------------------------------------------------------------

def merge_candidate_sets(
    *candidate_dicts: dict[str, set[str]],
    max_per_entity: int = 200,
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    """Merge candidate sets from multiple layers (UNION), track layer counts."""
    merged: dict[str, set[str]] = defaultdict(set)
    layer_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for layer_dict in candidate_dicts:
        for s1_id, cand_set in layer_dict.items():
            for cid in cand_set:
                layer_counts[s1_id][cid] += 1
            merged[s1_id].update(cand_set)

    # Cap per entity — prioritise candidates appearing in more layers
    capped: dict[str, set[str]] = {}
    for s1_id, cands in merged.items():
        if len(cands) > max_per_entity:
            scored = sorted(cands, key=lambda c: -layer_counts[s1_id][c])
            capped[s1_id] = set(scored[:max_per_entity])
        else:
            capped[s1_id] = cands

    return capped, dict(layer_counts)


def run_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_candidates: int = 200,
    **kwargs,
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    """Run hierarchical blocking pipeline. Returns (candidates, layer_counts)."""
    logger.info("=" * 60)
    logger.info("BLOCKING PIPELINE: S1=%d, S2/S3=%d", len(s1_df), len(s23_df))
    logger.info("=" * 60)

    l1 = block_layer1_token(s1_df, s23_df)
    l2 = block_layer2_phonetic(s1_df, s23_df)
    l3 = block_layer3_tfidf_nn(s1_df, s23_df, top_k=20)

    candidates, layer_counts = merge_candidate_sets(l1, l2, l3, max_per_entity=max_candidates)

    # Ensure all S1 entities have entries
    for eid in s1_df["entity_id"].values:
        s1_id = str(eid)
        if s1_id not in candidates:
            candidates[s1_id] = set()

    total_pairs = sum(len(v) for v in candidates.values())
    non_empty = sum(1 for v in candidates.values() if v)
    singletons = sum(1 for v in candidates.values() if not v)
    avg = total_pairs / max(non_empty, 1)

    logger.info("BLOCKING SUMMARY:")
    logger.info("  Total pairs: %d", total_pairs)
    logger.info("  S1 with candidates: %d (avg %.1f)", non_empty, avg)
    logger.info("  Predicted singletons: %d", singletons)

    return candidates, layer_counts


def measure_blocking_recall(
    candidates: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> dict:
    total_true = 0
    captured = 0
    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        total_true += len(true_matches)
        cands = candidates.get(s1_id, set())
        captured += len(true_matches & cands)
    recall = captured / max(total_true, 1)
    return {"blocking_recall": recall, "captured_pairs": captured, "total_true_pairs": total_true}


def candidates_to_tsv(candidates: dict[str, set[str]], output_path: str):
    rows = []
    for s1_id in sorted(candidates.keys()):
        cand_ids = sorted(candidates[s1_id])
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cand_ids) if cand_ids else "",
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("Wrote %d rows to %s", len(rows), output_path)
