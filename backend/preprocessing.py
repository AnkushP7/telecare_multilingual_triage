"""IndicNLP Preprocessing & Code-Mix Detection Module.

Implements text normalization for Indian regional languages, code-mix detection
for Hinglish, Tanglish, Devanagari Hindi, and Tamil script inputs, phonetic cleaning,
and tokenization matching the telecare triage pipeline.
"""
import re
import unicodedata
from typing import Dict, List, Tuple


# Indic script Unicode ranges
DEVANAGARI_RANGE = (0x0900, 0x097F)
TAMIL_RANGE = (0x0B80, 0x0BFF)

# Phonetic normalization map for common informal transliterations
PHONETIC_NORMALIZATION = {
    # Hinglish variations
    r"\brha\b": "raha",
    r"\brhi\b": "rahi",
    r"\brhe\b": "rahe",
    r"\bnhi\b": "nahi",
    r"\bnhin\b": "nahin",
    r"\bbht\b": "bahut",
    r"\bbohot\b": "bahut",
    r"\bbahot\b": "bahut",
    r"\bthoda\b": "thoda",
    r"\bthodi\b": "thoda",
    r"\bkch\b": "kuch",
    r"\bsir\b": "sar",
    r"\bseena\b": "chest",
    r"\bseene\b": "chest",
    r"\bpet\b": "stomach",
    r"\bgale\b": "gala",
    r"\bdard\b": "dard",
    r"\bdrd\b": "dard",
    # Tanglish variations
    r"\biruku\b": "irukku",
    r"\birukudhu\b": "irukku",
    r"\bkaachal\b": "kaichal",
    r"\bkaychal\b": "kaichal",
    r"\bthala\b": "thalai",
    r"\bnenjula\b": "nenju",
    r"\bkonjam\b": "konjam",
    r"\bknjam\b": "konjam",
    r"\bromba\b": "romba",
    r"\bvalikuthu\b": "valikkuthu",
}

# Vocabulary anchors for language identification
HINDI_MARKERS = [
    "hai", "hain", "ho", "raha", "rahi", "rahe", "din", "dino", "se", "dard",
    "mein", "me", "bahut", "thoda", "sar", "gala", "saans", "ulti", "bukhar",
    "pet", "badan", "kamzor", "kamzori", "zor", "aaya", "aayi", "lag", "bhi",
    "aur", "kal", "subah", "raat", "dikkat", "kharash", "chakkar"
]

TAMIL_MARKERS = [
    "irukku", "irukirathu", "naal", "naala", "enakku", "ennakku", "konjam",
    "romba", "vali", "kaichal", "suthudhu", "suthuradhu", "ah", "um", "nenju",
    "thalai", "thala", "inniku", "kaalaila", "vidradhula", "kastam", "thondai",
    "vayiru", "vaanthi", "vaandhi", "varudhu", "mudhuguvalli", "athigama"
]

ENGLISH_CLINICAL_MARKERS = [
    "the", "and", "with", "since", "morning", "pain", "fever", "vomiting",
    "cough", "breath", "breathing", "severe", "moderate", "mild", "hours",
    "days", "yesterday", "weakness", "dizziness", "chest", "headache",
    "pressure", "throat", "stomach", "heavy", "difficulty", "feeling"
]


def indic_unicode_normalize(text: str) -> str:
    """Normalize unicode representations (NFC) and clean whitespace."""
    text = unicodedata.normalize("NFC", text)
    # Remove control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def phonetic_clean(text: str) -> str:
    """Clean common informal SMS/telecare transliteration spellings."""
    cleaned = text
    for pattern, replacement in PHONETIC_NORMALIZATION.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def detect_script(text: str) -> Dict[str, int]:
    """Count character frequencies in native Indic scripts."""
    devanagari_count = sum(1 for ch in text if DEVANAGARI_RANGE[0] <= ord(ch) <= DEVANAGARI_RANGE[1])
    tamil_count = sum(1 for ch in text if TAMIL_RANGE[0] <= ord(ch) <= TAMIL_RANGE[1])
    latin_count = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    return {
        "devanagari": devanagari_count,
        "tamil": tamil_count,
        "latin": latin_count
    }


def detect_code_mix(text: str) -> Dict:
    """Comprehensive code-mix and dialect detection engine."""
    cleaned = indic_unicode_normalize(text)
    lower = cleaned.lower()
    tokens = re.findall(r"[\w']+", lower)
    total_tokens = max(len(tokens), 1)

    scripts = detect_script(cleaned)

    # Word-level pattern hits
    hindi_hits = sum(1 for t in tokens if t in HINDI_MARKERS)
    tamil_hits = sum(1 for t in tokens if t in TAMIL_MARKERS)
    english_hits = sum(1 for t in tokens if t in ENGLISH_CLINICAL_MARKERS)

    # Substring checks for multi-token phrases
    if "ho raha" in lower or "ho rha" in lower or "lag raha" in lower:
        hindi_hits += 2
    if "saans lene" in lower or "seene mein" in lower or "seene me" in lower:
        hindi_hits += 2
    if "nenju vali" in lower or "thala vali" in lower or "gala pain" in lower:
        tamil_hits += 2

    # Dialect classification
    scores = {
        "English": english_hits,
        "Hinglish": hindi_hits,
        "Tanglish": tamil_hits,
        "Tamil Script": scripts["tamil"] * 2,
        "Devanagari Script": scripts["devanagari"] * 2
    }

    if scripts["devanagari"] > 3 and scripts["latin"] == 0:
        label = "Hindi (Devanagari Script)"
        mix_ratio = "Monolingual Indic Script"
    elif scripts["tamil"] > 3 and scripts["latin"] == 0:
        label = "Tamil (Tamil Script)"
        mix_ratio = "Monolingual Indic Script"
    elif tamil_hits >= 2 and english_hits >= 1:
        label = "Tanglish / English mixed"
        mix_ratio = f"{round((tamil_hits / total_tokens) * 100)}% Tamil, {round((english_hits / total_tokens) * 100)}% English"
    elif tamil_hits >= 1:
        label = "Tanglish (Tamil-English code-mixed)"
        mix_ratio = "Tamil-English Code-Mixed"
    elif hindi_hits >= 2 and english_hits >= 1:
        label = "Hinglish / English mixed"
        mix_ratio = f"{round((hindi_hits / total_tokens) * 100)}% Hindi, {round((english_hits / total_tokens) * 100)}% English"
    elif hindi_hits >= 1:
        label = "Hinglish (Hindi-English code-mixed)"
        mix_ratio = "Hindi-English Code-Mixed"
    elif english_hits >= 2:
        label = "English (Clinical)"
        mix_ratio = "Standard Clinical English"
    else:
        label = "Code-mixed vernacular"
        mix_ratio = "Multi-dialect colloquial"

    return {
        "label": label,
        "mix_ratio": mix_ratio,
        "scores": scores,
        "scripts": scripts,
        "token_count": len(tokens),
        "clean_text": cleaned
    }


def indic_tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Tokenize text preserving exact character offsets for IOB BIO tagging."""
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.group(0), match.start(), match.end()))
    return tokens
