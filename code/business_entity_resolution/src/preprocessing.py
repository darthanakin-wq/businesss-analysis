"""
preprocessing.py — Business name & address normalization (RAM-efficient).

Changes vs. v1:
  - Drops unused list-valued columns (name_tokens, addr_tokens, addr_numbers,
    addr_alpha_tokens, name_dba_parts, combined_text) that were wasting RAM
    at 10M+ rows.
  - Wires in Indic-script transliteration (transliteration.py) so that
    Devanagari/Kannada business names get matched to their Latin equivalents.
  - Only keeps scalar columns needed downstream:
    name_norm, name_sorted, name_acronym, addr_norm, addr_postal, country_norm
"""

import pandas as pd
import re
import unicodedata
import logging
from typing import Union

from config import LEGAL_SUFFIX_MAP, NOISE_WORDS, ADDRESS_ABBREV_MAP, COUNTRY_MAP

logger = logging.getLogger(__name__)

# Pre-compile regex patterns
DBA_PATTERN = re.compile(r'\b(?:dba|d/b/a|trading as|t/a)\b', re.IGNORECASE)
PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
MULTIPLE_SPACES_PATTERN = re.compile(r'\s+')
LANDMARK_PATTERN = re.compile(
    r'\b(?:near|opp|opposite|behind|beside|next to)\b.*?(?:,|$)', re.IGNORECASE
)
POSTAL_CODE_PATTERN = re.compile(r'\b(\d{5}(?:-\d{4})?|\d{6})\b')

# Lazy-import transliteration (installed alongside)
_transliterate_fn = None

def _get_transliterate():
    """Lazy-load transliteration to avoid circular imports."""
    global _transliterate_fn
    if _transliterate_fn is None:
        try:
            from transliteration import normalize_for_matching
            _transliterate_fn = normalize_for_matching
        except ImportError:
            logger.warning("transliteration module not found — Indic scripts won't be handled")
            _transliterate_fn = lambda x: x  # passthrough
    return _transliterate_fn


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def normalize_business_name(name: Union[str, float]) -> dict:
    """Return dict with keys: normalized, sorted_tokens, acronym."""
    result = {'normalized': '', 'sorted_tokens': '', 'acronym': ''}

    if pd.isna(name) or not isinstance(name, str) or not name.strip():
        return result

    # Transliterate Indic scripts to Latin *before* everything else
    translit = _get_transliterate()
    name = translit(name)

    # Unicode NFKD
    name = unicodedata.normalize('NFKD', name)
    name = name.lower()

    # Split off DBA parts (we extract them but don't store a list column)
    dba_splits = DBA_PATTERN.split(name)
    if len(dba_splits) > 1:
        name = dba_splits[0]

    name = name.replace('&', ' and ')
    name = PUNCTUATION_PATTERN.sub(' ', name)

    tokens = name.split()
    processed = []
    for tok in tokens:
        tok = LEGAL_SUFFIX_MAP.get(tok, tok)
        if tok not in NOISE_WORDS:
            processed.append(tok)

    normalized = ' '.join(processed)
    result['normalized'] = normalized
    if processed:
        result['sorted_tokens'] = ' '.join(sorted(processed))
        result['acronym'] = ''.join(t[0] for t in processed if t)
    return result


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------

def normalize_address(address: Union[str, float]) -> dict:
    """Return dict with keys: normalized, postal_code."""
    result = {'normalized': '', 'postal_code': ''}

    if pd.isna(address) or not isinstance(address, str) or not address.strip():
        return result

    # Transliterate Indic scripts
    translit = _get_transliterate()
    address = translit(address)

    address = unicodedata.normalize('NFKD', address)
    address = address.lower()

    # Remove landmark references
    address = LANDMARK_PATTERN.sub(' ', address)

    # Extract postal code
    postal_matches = POSTAL_CODE_PATTERN.findall(address)
    if postal_matches:
        result['postal_code'] = postal_matches[0]

    # Expand abbreviations
    tokens = address.split()
    expanded = []
    for tok in tokens:
        clean = tok.rstrip(',')
        if clean in ADDRESS_ABBREV_MAP:
            tok = tok.replace(clean, ADDRESS_ABBREV_MAP[clean])
        expanded.append(tok)
    address = ' '.join(expanded)

    # Remove punctuation, collapse spaces
    address = PUNCTUATION_PATTERN.sub(' ', address)
    address = MULTIPLE_SPACES_PATTERN.sub(' ', address).strip()

    result['normalized'] = address
    return result


# ---------------------------------------------------------------------------
# Country normalization
# ---------------------------------------------------------------------------

def normalize_country(country: Union[str, float]) -> str:
    if pd.isna(country) or not isinstance(country, str) or not country.strip():
        return ''
    country = country.lower().strip()
    return COUNTRY_MAP.get(country, country)


# ---------------------------------------------------------------------------
# DataFrame-level preprocessing (RAM-efficient)
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized scalar columns. Drops list-valued columns to save RAM."""
    logger.info("Preprocessing dataframe with %d records", len(df))
    df = df.copy()

    # Name normalization — only keep scalar columns
    if 'business_name' in df.columns:
        logger.info("Normalizing business names...")
        name_results = df['business_name'].apply(normalize_business_name)
        df['name_norm'] = name_results.apply(lambda x: x['normalized'])
        df['name_sorted'] = name_results.apply(lambda x: x['sorted_tokens'])
        df['name_acronym'] = name_results.apply(lambda x: x['acronym'])
        # NOT creating: name_tokens, name_dba_parts (list columns — RAM waste)

    # Address normalization — only keep scalar columns
    if 'business_address' in df.columns:
        logger.info("Normalizing business addresses...")
        addr_results = df['business_address'].apply(normalize_address)
        df['addr_norm'] = addr_results.apply(lambda x: x['normalized'])
        df['addr_postal'] = addr_results.apply(lambda x: x['postal_code'])
        # NOT creating: addr_tokens, addr_numbers, addr_alpha_tokens (list columns)

    # Country normalization
    if 'country' in df.columns:
        logger.info("Normalizing countries...")
        df['country_norm'] = df['country'].apply(normalize_country)

    # NOT creating combined_text (not used by any downstream module)

    logger.info("Preprocessing complete.")
    return df


def load_and_preprocess(source_path: str) -> pd.DataFrame:
    """Load a TSV file and run preprocess_dataframe on it."""
    logger.info("Loading data from %s", source_path)
    df = pd.read_csv(source_path, sep='\t', dtype=str)
    return preprocess_dataframe(df)
