"""
config.py — Portable configuration for Business Entity Resolution Pipeline.

Paths are resolved via environment variables with sensible fallbacks.
Set BER_DATA_DIR and BER_OUTPUT_DIR to override defaults.
"""

import os
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — portable via env vars
# ---------------------------------------------------------------------------

_DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, os.pardir, os.pardir,  # src -> business_entity_resolution -> code -> amazon-er-2026
    "dataset", "student_resource", "student_resource", "dataset"
)
_DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, os.pardir, os.pardir,
    "output"
)

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", os.path.normpath(_DEFAULT_DATA)))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", os.path.normpath(_DEFAULT_OUTPUT)))

# Train files
TRAIN_SOURCE1 = DATA_DIR / "train" / "train_source1.tsv"
TRAIN_SOURCE2 = DATA_DIR / "train" / "train_source2.tsv"
TRAIN_SOURCE3 = DATA_DIR / "train" / "train_source3.tsv"
TRAIN_GROUND_TRUTH = DATA_DIR / "train" / "train_ground_truth.tsv"

# Test files
TEST_SOURCE1 = DATA_DIR / "test" / "test_source1.tsv"
TEST_SOURCE2 = DATA_DIR / "test" / "test_source2.tsv"
TEST_SOURCE3 = DATA_DIR / "test" / "test_source3.tsv"

# Output files
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
MODEL_PATH = OUTPUT_DIR / "matcher_model.pkl"
THRESHOLD_PATH = OUTPUT_DIR / "threshold.txt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCKING_MAX_CANDIDATES = 200
MATCH_THRESHOLD = 0.65
SINGLETON_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.15  # margin-over-next-best gating
FAISS_TOP_K = 50
LSH_NUM_PERM = 128
LSH_THRESHOLD = 0.2
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
NGRAM_SIZE = 3

# ---------------------------------------------------------------------------
# Normalization dictionaries
# ---------------------------------------------------------------------------

LEGAL_SUFFIX_MAP = {
    # English
    'corporation': 'corp', 'corp': 'corp',
    'incorporated': 'inc', 'inc': 'inc',
    'limited': 'ltd', 'ltd': 'ltd',
    'private': 'pvt', 'pvt': 'pvt',
    'company': 'co', 'co': 'co',
    'international': 'intl', 'intl': 'intl',
    'limited liability company': 'llc', 'llc': 'llc',
    'limited liability partnership': 'llp', 'llp': 'llp',
    'public limited company': 'plc', 'plc': 'plc',
    'enterprises': 'enterprises', 'enterprise': 'enterprises',
    'solutions': 'solutions', 'solution': 'solutions',
    'technologies': 'tech', 'technology': 'tech', 'tech': 'tech',
    'services': 'services', 'service': 'services',
    'industries': 'industries', 'industry': 'industries',
    'associates': 'assoc', 'association': 'assoc', 'assoc': 'assoc',
    'consultants': 'consultants', 'consulting': 'consultants',
    'pharmaceuticals': 'pharma', 'pharmaceutical': 'pharma', 'pharma': 'pharma',
    'trading': 'trading', 'traders': 'trading',
    # French / German
    'societe anonyme': 'sa', 'sa': 'sa',
    'sarl': 'sarl',
    'societe par actions simplifiee': 'sas', 'sas': 'sas',
    'gesellschaft mit beschrankter haftung': 'gmbh', 'gmbh': 'gmbh',
}

ADDRESS_ABBREV_MAP = {
    'road': 'rd', 'rd': 'rd',
    'street': 'st', 'st': 'st',
    'avenue': 'ave', 'ave': 'ave',
    'boulevard': 'blvd', 'blvd': 'blvd',
    'drive': 'dr', 'dr': 'dr',
    'lane': 'ln', 'ln': 'ln',
    'highway': 'hwy', 'hwy': 'hwy',
    'apartment': 'apt', 'apt': 'apt',
    'suite': 'ste', 'ste': 'ste',
    'floor': 'fl', 'fl': 'fl',
    'building': 'bldg', 'bldg': 'bldg',
    'place': 'pl', 'pl': 'pl',
    'court': 'ct', 'ct': 'ct',
    'circle': 'cir', 'cir': 'cir',
    'parkway': 'pkwy', 'pkwy': 'pkwy',
    'terrace': 'ter', 'ter': 'ter',
    'route': 'rte', 'rte': 'rte',
    'nagar': 'ngr', 'ngr': 'ngr',
    'marg': 'mrg', 'mrg': 'mrg',
    'district': 'dist', 'dist': 'dist',
    'sector': 'sec', 'sec': 'sec',
    'phase': 'ph', 'ph': 'ph',
}

NOISE_WORDS = {'the', 'of', 'a', 'an', 'and', 'or', 'for', 'in', 'at', 'to', 'by'}

COUNTRY_MAP = {
    'united states': 'us', 'usa': 'us', 'us': 'us', 'u.s.': 'us', 'u.s.a.': 'us',
    'india': 'india', 'in': 'india', 'ind': 'india',
    'france': 'france', 'fr': 'france', 'fra': 'france',
}

# ---------------------------------------------------------------------------
# Logging & utilities
# ---------------------------------------------------------------------------

def setup_logging(log_file: bool = False, level=logging.INFO):
    """Configure logging."""
    handlers = [logging.StreamHandler()]
    if log_file:
        ensure_directories()
        handlers.append(logging.FileHandler(OUTPUT_DIR / "pipeline.log", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

def ensure_directories():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
