import logging
import re
import pandas as pd
from typing import Optional, Union
from unidecode import unidecode

logger = logging.getLogger(__name__)

# Script Unicode ranges
SCRIPT_RANGES = {
    'devanagari': (0x0900, 0x097F),
    'kannada': (0x0C80, 0x0CFF),
    'telugu': (0x0C00, 0x0C7F),
    'tamil': (0x0B80, 0x0BFF),
    'bengali': (0x0980, 0x09FF),
    'gujarati': (0x0A80, 0x0AFF),
}

# Phonetic suffix/word mapping (keys lowercase)
PHONETIC_MAP = {
    'phaarmaasyuutiklz': 'pharmaceuticals',
    'entarpraaijej': 'enterprises',
    'entarapraaizez': 'enterprises',
    'intarneshnal': 'intl',
    'intarnaeshnal': 'intl',
    'knsaltants': 'consultants',
    'teksnolaajii': 'technology',
    'teksnolaajee': 'technology',
    'indasriiz': 'industries',
    'indastriij': 'industries',
    'solushanz': 'solutions',
    'soluushnz': 'solutions',
    'vyaapaari': 'trading',
    'sarvisez': 'services',
    'sarvisej': 'services',
    'praiveet': 'pvt',
    'praaivet': 'pvt',
    'andaited': 'united',
    'limiited': 'ltd',
    'kampani': 'co',
    'kampnii': 'co',
    'limited': 'ltd',
    'limitd': 'ltd',
    'nyu': 'new'
}

# Compile a regex pattern to replace words. Use word boundaries \b
# We sort by length descending to match longer strings first.
PATTERN_STR = r'\b(' + '|'.join(re.escape(k) for k in sorted(PHONETIC_MAP.keys(), key=len, reverse=True)) + r')\b'
REPLACE_PATTERN = re.compile(PATTERN_STR, flags=re.IGNORECASE)


def detect_script(text: str) -> str:
    """
    Detects the dominant script in the given text based on Unicode blocks.
    Returns 'latin', 'devanagari', 'kannada', 'telugu', 'tamil', 'bengali', 'gujarati', 'mixed', or 'other'.
    """
    if not isinstance(text, str):
        return 'other'

    counts = {
        'devanagari': 0,
        'kannada': 0,
        'telugu': 0,
        'tamil': 0,
        'bengali': 0,
        'gujarati': 0,
        'latin': 0,
        'other': 0
    }
    
    total_alpha = 0

    for char in text:
        if not char.isalpha():
            continue
            
        total_alpha += 1
        code_point = ord(char)
        
        script_found = False
        for script, (start, end) in SCRIPT_RANGES.items():
            if start <= code_point <= end:
                counts[script] += 1
                script_found = True
                break
                
        if not script_found:
            # Check for Latin/ASCII
            if 'a' <= char.lower() <= 'z':
                counts['latin'] += 1
            else:
                counts['other'] += 1

    if total_alpha == 0:
        return 'other'

    # If >50% of alpha chars are in one script, return that script
    for script, count in counts.items():
        if count / total_alpha > 0.5:
            return script

    return 'mixed'


def transliterate_indic(text: str) -> str:
    """
    Uses unidecode to convert any non-ASCII text to ASCII approximation.
    Applies phonetic canonicalization for Hindi/Indian business terms.
    """
    if not isinstance(text, str):
        return text if pd.notna(text) else ''

    # Unidecode converts to ASCII approximation
    ascii_text = unidecode(text).lower()

    # Replace phonetic business terms
    ascii_text = REPLACE_PATTERN.sub(lambda m: PHONETIC_MAP[m.group(1).lower()], ascii_text)

    # Collapse multiple spaces and strip
    ascii_text = re.sub(r'\s+', ' ', ascii_text).strip()

    return ascii_text


def normalize_for_matching(text: str) -> str:
    """
    Main entry point: detects script and transliterates if non-Latin or mixed.
    """
    if not isinstance(text, str):
        return text if pd.notna(text) else ''

    script = detect_script(text)

    if script == 'latin':
        return re.sub(r'\s+', ' ', text.lower()).strip()
    else:
        return transliterate_indic(text)


def batch_transliterate(series: pd.Series) -> pd.Series:
    """
    Vectorized batch version using .apply() for DataFrame columns.
    Returns a new Series with transliterated values.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
        
    logger.info(f"Batch transliterating series of length {len(series)}")
    return series.apply(normalize_for_matching)
