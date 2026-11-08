"""Comprehensive Verification Test Suite for Project Review 2 Deliverables.

Validates:
- Objective 1 & Methodology Phase 1: Curated & Tagged IOB Dataset
- Objective 2 & Expected Outcome 1: Transformer-CRF Engine with F1 > 88%
- Objective 3 & Expected Outcome 2: SNOMED CT and ICD-10 Medical Concept Normalization
- Objective 4 & Expected Outcome 3: Urgency Stratification & Faster Intake Report (>50% time saved)
- Expected Outcome 4: End-to-End Pipeline & Web Demonstration
"""
import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "backend"))

from services import analyze
from normalization import NORMALIZER


def test_objective_1_dataset():
    print("\n--- [TEST 1] Objective 1: Dataset Curation & Tagging ---")
    data_path = BASE / "backend" / "data" / "annotated_clinical_codemixed_dataset.json"
    assert data_path.exists(), "Dataset JSON file is missing!"
    data = json.loads(data_path.read_text(encoding="utf-8"))

    train_count = len(data["train"])
    dev_count = len(data["dev"])
    test_count = len(data["test"])
    total = train_count + dev_count + test_count

    print(f"[OK] Found curated dataset with {total} annotated clinical sentences")
    print(f"  - Train split: {train_count}")
    print(f"  - Dev split  : {dev_count}")
    print(f"  - Test split : {test_count}")

    # Check CoNLL files
    for split in ["train", "dev", "test"]:
        conll_file = BASE / "backend" / "data" / f"{split}.conll"
        assert conll_file.exists(), f"CoNLL split {split}.conll missing!"
    print("[OK] CoNLL 2003 / IOB2 standard split files verified")


def test_objective_2_outcome_1_transformer_crf():
    print("\n--- [TEST 2] Objective 2 & Outcome 1: Transformer-CRF & High Precision Tagger ---")
    model_path = BASE / "backend" / "data" / "transformer_crf_model.pt"
    eval_path = BASE / "backend" / "data" / "evaluation_metrics.json"

    assert model_path.exists(), "PyTorch Transformer-CRF model weights missing!"
    assert eval_path.exists(), "Model evaluation metrics missing!"

    metrics = json.loads(eval_path.read_text(encoding="utf-8"))
    overall_f1 = metrics["overall"]["f1_score"]
    p = metrics["overall"]["precision"]
    r = metrics["overall"]["recall"]

    print(f"[OK] Model weights: {model_path.stat().st_size / 1024:.1f} KB")
    print(f"[OK] Overall Precision : {p}%")
    print(f"[OK] Overall Recall    : {r}%")
    print(f"[OK] Overall F1-Score  : {overall_f1}%")
    assert overall_f1 >= 88.0, f"Outcome 1 F1 threshold failed: {overall_f1}% < 88.0%"
    print(f"[OK] EXCEEDS TARGET THRESHOLD (F1 > 88.0%): PASSED ({overall_f1}%)")

    for ent, s in metrics["per_entity"].items():
        print(f"  - {ent:<10}: F1 = {s['f1_score']}% (P: {s['precision']}%, R: {s['recall']}%)")


def test_objective_3_outcome_2_concept_normalization():
    print("\n--- [TEST 3] Objective 3 & Outcome 2: SNOMED CT and ICD-10 Normalization ---")
    term_path = BASE / "backend" / "data" / "terminology.json"
    terms = json.loads(term_path.read_text(encoding="utf-8"))

    # Verify every concept has official SNOMED CT and ICD-10
    missing_codes = [t["normalized"] for t in terms if not t.get("snomed_ct") or not t.get("icd10")]
    assert len(missing_codes) == 0, f"Missing terminology codes: {missing_codes}"
    print(f"[OK] All {len(terms)} clinical concepts mapped to official SNOMED CT and ICD-10 codes")

    # Test colloquial vernacular phrase mappings
    test_cases = [
        ("gala dard", "Sore throat", "267102003", "R07.0"),
        ("chest heavy", "Chest pain", "29857009", "R07.9"),
        ("sar dard", "Headache", "25064002", "R51.9"),
        ("nenju vali", "Chest pain", "29857009", "R07.9"),
        ("saans lene mein dikkat", "Shortness of breath", "267036007", "R06.02"),
        ("ulti me blood", "Blood in vomit (Hematemesis)", "44903000", "K92.0")
    ]

    for phrase, expected_norm, expected_snomed, expected_icd in test_cases:
        top_candidates = NORMALIZER.map_phrase(phrase, top_k=1)
        top = top_candidates[0]
        assert top["candidate"] == expected_norm, f"Mapping failed for '{phrase}': got {top['candidate']}, expected {expected_norm}"
        assert top["snomed_ct"] == expected_snomed, f"SNOMED CT mismatch: {top['snomed_ct']} vs {expected_snomed}"
        assert top["icd10"] == expected_icd, f"ICD-10 mismatch: {top['icd10']} vs {expected_icd}"
        print(f"[OK] '{phrase}' -> {top['candidate']} (SNOMED CT: {top['snomed_ct']} | ICD-10: {top['icd10']}) [Sim: {top['score']}]")


def test_objective_4_outcome_3_faster_intake_review():
    print("\n--- [TEST 4] Objective 4 & Outcome 3: Triage Stratification & Faster Intake Review ---")
    query = "mujhe zor se chest pain ho rha hai aur sar dard bhi"
    res = analyze(query)

    print(f"[OK] Query: \"{query}\"")
    print(f"  - Dialect: {res['code_mix']['label']}")
    print(f"  - Triage Level: {res['triage']['level']} (Score: {res['triage']['score']})")
    print(f"  - Action: {res['triage']['action']}")
    print(f"  - Summary: {res['summary']}")

    assert res["triage"]["level"] == "High", f"Expected High priority for chest pain: got {res['triage']['level']}"
    assert "Chest pain / pressure" in res["triage"]["flags"], "Expected chest pain escalation flag"

    report = res["intake_report"]
    metrics = report["time_metrics"]
    reduction = metrics["reduction_percentage"]

    print(f"[OK] Doctor Intake Review Time Metrics:")
    print(f"  - Baseline Unstructured Review Time: {metrics['baseline_unstructured_seconds']} seconds")
    print(f"  - Structured Telecare Review Time  : {metrics['structured_telecare_seconds']} seconds")
    print(f"  - Clinician Time Saved             : {metrics['time_saved_seconds']} seconds")
    print(f"  - Review Time Reduction Rate       : {reduction}%")
    assert reduction >= 50.0, f"Expected review time reduction >= 50%: got {reduction}%"
    print(f"[OK] EXCEEDS 50% REVIEW TIME REDUCTION (Outcome 3): PASSED ({reduction}%)")


def test_end_to_end_review_samples():
    print("\n--- [TEST 5] End-to-End Test Across Indian Vernacular Dialects ---")
    samples = [
        ("2 din se sar dard aur vomiting lag raha hai, body bahut weak hai.", "Medium", "Hinglish"),
        ("Since morning chest heavy hai aur saans lene mein thodi dikkat ho rahi hai.", "High", "Hinglish / English mixed"),
        ("Enakku 3 naal ah gala pain irukku, fever um konjam high ah irukku.", "Medium", "Tanglish"),
        ("Kal se halka cough hai, no fever and severity mild.", "Low", "Hinglish / English mixed"),
        ("Achanak severe abdominal pain ho raha hai aur ulti me blood aaya hai.", "High", "Hinglish"),
        ("Nenju vali romba athigama irukku, saans vidradhula kastam irukku 2 hours ah.", "High", "Tanglish")
    ]

    for text, expected_level, dialect_prefix in samples:
        r = analyze(text)
        print(f"[OK] [{r['triage']['level'].upper():<6}] ({r['code_mix']['label']}) \"{text[:45]}...\"")
        assert r["triage"]["level"] == expected_level, f"Level mismatch for '{text}': got {r['triage']['level']} expected {expected_level}"


if __name__ == "__main__":
    test_objective_1_dataset()
    test_objective_2_outcome_1_transformer_crf()
    test_objective_3_outcome_2_concept_normalization()
    test_objective_4_outcome_3_faster_intake_review()
    test_end_to_end_review_samples()
    print("\n=================================================================")
    print("ALL OBJECTIVES & EXPECTED OUTCOMES SUCCESSFULLY VERIFIED (100%)")
    print("=================================================================\n")
