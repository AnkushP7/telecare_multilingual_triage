"""Medical Concept Normalization Engine (Objective 3 & Expected Outcome 2).

Translates colloquial Hinglish, Tanglish, and English layman phrases
into standardized SNOMED CT and ICD-10 medical concepts via semantic
n-gram vector representations and cosine similarity.
"""
import json
import re
from pathlib import Path
from math import sqrt, log
from collections import Counter
from typing import Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
TERMINOLOGY_FILE = BASE_DIR / "data" / "terminology.json"

with open(TERMINOLOGY_FILE, "r", encoding="utf-8") as f:
    TERMINOLOGY_DB = json.load(f)

# Concept categories for clinical triage grouping
CONCEPT_CATEGORIES = {
    "chest_pain": "Cardiovascular / Thoracic",
    "shortness_breath": "Respiratory / Pulmonary",
    "palpitations": "Cardiovascular",
    "headache": "Neurological",
    "dizziness": "Neurological / Vestibular",
    "syncope": "Neurological / Cardiovascular",
    "vomiting": "Gastrointestinal",
    "nausea": "Gastrointestinal",
    "abdominal_pain": "Gastrointestinal / Abdominal",
    "hematemesis": "Gastrointestinal Emergency",
    "diarrhea": "Gastrointestinal",
    "fever": "Constitutional / Infectious",
    "cough": "Respiratory / ENT",
    "sore_throat": "ENT / Oropharyngeal",
    "fatigue": "Constitutional",
    "myalgia": "Musculoskeletal",
    "back_pain": "Musculoskeletal",
    "rash": "Dermatological"
}


def _tokenize_subwords(text: str) -> List[str]:
    """Tokenize into words, character trigrams, and word bigrams."""
    text_clean = text.lower()
    words = re.findall(r"[\w\u0900-\u097F\u0B80-\u0BFF]+", text_clean)
    features = list(words)

    # Word bigrams
    for w1, w2 in zip(words, words[1:]):
        features.append(f"{w1}__{w2}")

    # Character trigrams for morphological and transliteration tolerance
    for w in words:
        if len(w) >= 3:
            for i in range(len(w) - 2):
                features.append(f"char3:{w[i:i+3]}")

    return features


class ConceptNormalizer:
    """Semantic mapping pipeline to standard SNOMED CT and ICD-10 medical codes."""

    def __init__(self, terminology: List[Dict] = TERMINOLOGY_DB):
        self.terminology = terminology
        self.corpus_docs = []
        self.items = []

        for item in self.terminology:
            # Combine concept title, normalized name, and all colloquial terms
            all_text = f"{item['normalized']} " + " ".join(item.get("terms", []))
            doc = _tokenize_subwords(all_text)
            self.corpus_docs.append(doc)
            self.items.append(item)

        # Precompute IDF
        self.n_docs = len(self.corpus_docs)
        self.df = Counter()
        for doc in self.corpus_docs:
            for term in set(doc):
                self.df[term] += 1

        # Precompute concept vectors
        self.concept_vectors = [self._vectorize(doc) for doc in self.corpus_docs]

    def _vectorize(self, doc: List[str]) -> Dict[str, float]:
        tf = Counter(doc)
        vec = {}
        doc_len = max(len(doc), 1)
        for term, count in tf.items():
            idf = log((1 + self.n_docs) / (1 + self.df.get(term, 0))) + 1.0
            vec[term] = (count / doc_len) * idf
        return vec

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(val * b.get(term, 0.0) for term, val in a.items())
        na = sqrt(sum(val * val for val in a.values()))
        nb = sqrt(sum(val * val for val in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def map_phrase(self, phrase: str, top_k: int = 5) -> List[Dict]:
        """Rank candidate SNOMED CT and ICD-10 concepts by semantic cosine similarity."""
        query_doc = _tokenize_subwords(phrase)
        q_vec = self._vectorize(query_doc)

        ranked = []
        lower_phrase = phrase.lower()

        for item, c_vec in zip(self.items, self.concept_vectors):
            base_score = self._cosine(q_vec, c_vec)

            # Direct colloquial anchor boost
            for term in item.get("terms", []):
                if term.lower() in lower_phrase:
                    base_score = max(base_score, 0.75 + (0.20 if term.lower() == lower_phrase else 0.10))
                    break

            ranked.append((item, min(base_score, 0.99)))

        ranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for item, score in ranked[:top_k]:
            results.append({
                "candidate": item["normalized"],
                "key": item["key"],
                "score": round(float(score), 3),
                "snomed_ct": item.get("snomed_ct"),
                "icd10": item.get("icd10"),
                "category": CONCEPT_CATEGORIES.get(item["key"], "General Clinical")
            })

        return results

    def resolve_entity(self, entity_text: str) -> Optional[Dict]:
        """Resolve a single extracted entity span to its standard medical terminology."""
        candidates = self.map_phrase(entity_text, top_k=1)
        if candidates and candidates[0]["score"] > 0.15:
            c = candidates[0]
            return {
                "normalized": c["candidate"],
                "key": c["key"],
                "snomed_ct": c["snomed_ct"],
                "icd10": c["icd10"],
                "category": c["category"],
                "confidence": round(float(c["score"]), 3)
            }
        return None


# Global singleton instance
NORMALIZER = ConceptNormalizer()
