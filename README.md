# Mixed Multilingual Symptom Extraction & Triage Support for Telecare
**Project Review 2 | 22BML0008 · 22BML0075 · 24BML0099**

An automated clinical Natural Language Processing (NLP) framework designed to extract, normalize, and categorize clinical entities from noisy, code-mixed patient text/speech (Hinglish, Tanglish, and regional dialects), mapping colloquial complaints into standardized SNOMED CT and ICD-10 terms, performing automated risk stratification (Low, Medium, High), and generating structured doctor intake reports that reduce preliminary triage review times by over 50%.

---

## 1. System Architecture & End-to-End Pipeline

Aligned directly with the Review 2 Architecture & Flow Diagrams:

```
[Patient Input] ➔ [Preprocessing]   ➔ [Transformer-CRF] ➔ [Normalization] ➔ [Triage Classifier] ➔ [Doctor Dashboard]
  (Voice/Text)     (IndicNLP/Mix)         (NER Tagging)       (SNOMED / ICD)      (Urgency Score)       (SOAP Summary)
```

```
[Raw Input Query] ➔ [Code-Mix Detection] ➔ [BIO Token Tagging] ➔ [Cosine Mapping] ➔ [Risk Assessment] ➔ [Intake Reduction]
```

---

## 2. Implementation of Project Objectives & Expected Outcomes

### Objective 1: Dataset Curation & Tagging (Methodology Phase 1)
- **Dataset File:** `backend/data/annotated_clinical_codemixed_dataset.json`
- **Standard Splits:** `train.conll`, `dev.conll`, `test.conll` (CoNLL 2003 / IOB2 schema)
- **Entities Tagged:** `B-SYMPTOM`, `I-SYMPTOM`, `B-DURATION`, `I-DURATION`, `B-SEVERITY`, `I-SEVERITY`, `B-SITE`, `I-SITE`, `O`
- **Corpus Content:** Combines Indian code-mixed telecare samples (Hinglish, Tanglish) with clinical benchmark entities (NCBI, MedMentions). Total 155 annotated clinical sentences partitioned into 70% Train (108), 15% Dev (23), and 15% Test (24).

### Objective 2 & Expected Outcome 1: High Precision Tagger (F1 > 88%)
- **Architecture:** Multilingual Contextual Sequence Encoder paired with a Linear Chain Conditional Random Field (CRF) layer using dynamic programming log-sum-exp forward loss and Viterbi optimal sequence decoding (`backend/transformer_crf.py`).
- **Trained Model Checkpoint:** `backend/data/transformer_crf_model.pt`
- **Verified Evaluation Metrics (`backend/data/evaluation_metrics.json`):**
  - **Overall Precision:** **93.83%**
  - **Overall Recall:** **95.00%**
  - **Overall F1-Score:** **94.41%** (Exceeds the >88.0% benchmark target: **PASSED**)
  - **SYMPTOM:** F1 = 92.54% (Precision: 91.18%, Recall: 93.94%)
  - **DURATION:** F1 = 97.30% (Precision: 100.0%, Recall: 94.74%)
  - **SEVERITY:** F1 = 91.89% (Precision: 89.47%, Recall: 94.44%)
  - **SITE:** F1 = 100.0% (Precision: 100.0%, Recall: 100.0%)

### Objective 3 & Expected Outcome 2: Normalized Clinical Vocabulary
- **Mapping Pipeline:** `backend/normalization.py`
- **Knowledge Base:** `backend/data/terminology.json`
- **Terminology Standards:** Official **SNOMED CT International Edition Concept IDs** and **ICD-10-CM codes**.
- **Colloquial Vernacular Semantic Mappings:**
  - `'gala dard'` / `'gala pain'` ➔ **Sore throat** (`SNOMED CT: 267102003`, `ICD-10: R07.0`)
  - `'chest heavy'` / `'nenju vali'` ➔ **Chest pain** (`SNOMED CT: 29857009`, `ICD-10: R07.9`)
  - `'sar dard'` / `'thala vali'` ➔ **Headache** (`SNOMED CT: 25064002`, `ICD-10: R51.9`)
  - `'saans lene mein dikkat'` ➔ **Shortness of breath** (`SNOMED CT: 267036007`, `ICD-10: R06.02`)
  - `'pet me bahut dard'` ➔ **Abdominal pain** (`SNOMED CT: 21522001`, `ICD-10: R10.9`)
  - `'ulti me blood'` ➔ **Blood in vomit (Hematemesis)** (`SNOMED CT: 44903000`, `ICD-10: K92.0`)
  - `'chakkar'` / `'thala suthudhu'` ➔ **Dizziness** (`SNOMED CT: 404640003`, `ICD-10: R42`)
  - `'badan kamzor'` / `'tired'` ➔ **Fatigue / weakness** (`SNOMED CT: 84229001`, `ICD-10: R53.83`)

### Objective 4 & Expected Outcome 3: Faster Intake Review (Structured Telecare Report Generator)
- **Report Engine:** `backend/report_generator.py`
- **Review Time Reduction:**
  - **Baseline Unstructured Free-Text Review:** ~120 – 167 seconds
  - **Structured Telecare Triage Review:** ~45 – 53 seconds
  - **Time Saved:** **65.6% – 68.3% review time reduction** (Exceeds the 50% target in Expected Outcome 3: **PASSED**).
- **Clinical SOAP Formatting:**
  - **Subjective:** Raw patient vernacular narrative, IndicNLP dialect breakdown.
  - **Objective:** Structured entity table with SNOMED CT and ICD-10 codes.
  - **Assessment:** Urgency priority level (`High`, `Medium`, `Low`), risk gauge score (0-100), red flag escalation indicators.
  - **Plan:** Automated clinical guidance (Immediate ER escalation vs. urgent teleconsultation vs. routine care).
- **Print & Export:** Interactive doctor intake sheet modal with 1-click printable / downloadable summary sheet.

### Expected Outcome 4: Working Web Prototype
- **FastAPI Backend:** `backend/app.py`
- **High-Compatibility Server:** `backend/server.py` (zero-dependency Python standard library server)
- **Doctor Portal Interface:** `frontend/dist/index.html` and React source `frontend/src/main.jsx`
- **Interactive UI Features:**
  - Web Speech API voice capture button (`en-IN` recognition)
  - Pre-loaded demo cases including the exact Review 2 case (`mujhe zor se chest pain ho rha hai aur sar dard bhi`)
  - Circular risk gauge (`95 - 100` score ring)
  - Interactive BIO token tagging cloud
  - Ranked top-5 cosine candidates with SNOMED CT & ICD-10 badges
  - Modal viewing model evaluation metrics (F1 > 88%)
  - Modal viewing and printing structured Doctor Intake Sheets

---

## 3. Running the Prototype Locally

### Option A: Standard Server (Recommended - Runs Out-of-the-Box)

```bash
cd telecare_prototype
python backend/server.py
```
Open **`http://localhost:8000`** in any web browser.

### Option B: FastAPI / Uvicorn Server

```bash
cd telecare_prototype
uvicorn backend.app:app --reload --port 8000
```

---

## 4. Running the Complete Verification Suite

To verify all 4 objectives and expected outcomes via automated tests:

```bash
cd telecare_prototype
python test_all_objectives.py
```

To run the interactive smoke test:

```bash
cd telecare_prototype
python smoke_test.py
```

To re-train the Transformer-CRF model and re-evaluate on test data:

```bash
cd telecare_prototype
python backend/train_ner.py
```

---

## 5. API Endpoints

- `GET /api/health`: Service health and pipeline architecture status.
- `GET /api/demo-cases`: Pre-configured clinical demo cases across dialects.
- `POST /api/analyze`: Full NLP pipeline analysis (code-mix detection, Transformer-CRF BIO tags, SNOMED CT/ICD-10 normalization, urgency stratification, and intake review metrics).
- `GET /api/model-evaluation`: Full test set evaluation report showing Precision, Recall, and F1 > 88%.
- `GET /api/terminology`: Standardized clinical ontology database with SNOMED CT and ICD-10 codes.
- `POST /api/intake-report`: Structured clinical SOAP summary generator.
