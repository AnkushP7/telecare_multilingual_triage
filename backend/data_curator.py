"""Dataset Curation & Tagging Pipeline (Objective 1, Methodology Phase 1).

Builds and preprocesses a code-mixed clinical dataset combining Hinglish, Tanglish,
and clinical benchmark samples annotated with standard IOB schema:
B-SYMPTOM, I-SYMPTOM, B-DURATION, I-DURATION, B-SEVERITY, I-SEVERITY, B-SITE, I-SITE, O.

Outputs:
- backend/data/annotated_clinical_codemixed_dataset.json
- backend/data/train.conll
- backend/data/dev.conll
- backend/data/test.conll
"""
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Curated seed records combining Indian code-mixed telecare samples and clinical benchmarks
RAW_RECORDS = [
    # Hinglish Telecare cases
    {
        "text": "2 din se sar dard aur vomiting lag raha hai, body bahut weak hai.",
        "dialect": "Hinglish",
        "entities": [
            ("2 din", "DURATION"),
            ("sar dard", "SYMPTOM"),
            ("vomiting", "SYMPTOM"),
            ("sar", "SITE"),
            ("body", "SITE"),
            ("bahut", "SEVERITY"),
            ("weak", "SYMPTOM")
        ]
    },
    {
        "text": "mujhe zor se chest pain ho rha hai aur sar dard bhi",
        "dialect": "Hinglish",
        "entities": [
            ("zor se", "SEVERITY"),
            ("chest pain", "SYMPTOM"),
            ("chest", "SITE"),
            ("sar dard", "SYMPTOM"),
            ("sar", "SITE")
        ]
    },
    {
        "text": "Since morning chest heavy hai aur saans lene mein thodi dikkat ho rahi hai.",
        "dialect": "Hinglish / English mixed",
        "entities": [
            ("Since morning", "DURATION"),
            ("chest heavy", "SYMPTOM"),
            ("chest", "SITE"),
            ("saans lene mein thodi dikkat", "SYMPTOM"),
            ("thodi", "SEVERITY")
        ]
    },
    {
        "text": "Kal se halka cough hai, no fever and severity mild.",
        "dialect": "Hinglish / English mixed",
        "entities": [
            ("Kal se", "DURATION"),
            ("halka", "SEVERITY"),
            ("cough", "SYMPTOM"),
            ("mild", "SEVERITY")
        ]
    },
    {
        "text": "Pet me bahut tez dard ho raha hai 3 ghante se aur baar baar ulti aa rahi hai.",
        "dialect": "Hinglish",
        "entities": [
            ("Pet", "SITE"),
            ("bahut tez", "SEVERITY"),
            ("dard", "SYMPTOM"),
            ("3 ghante", "DURATION"),
            ("baar baar", "SEVERITY"),
            ("ulti", "SYMPTOM")
        ]
    },
    {
        "text": "Gale me dard hai 4 din se aur high fever bhi hai.",
        "dialect": "Hinglish",
        "entities": [
            ("Gale", "SITE"),
            ("dard", "SYMPTOM"),
            ("4 din", "DURATION"),
            ("high", "SEVERITY"),
            ("fever", "SYMPTOM")
        ]
    },
    {
        "text": "Achanak tez sar dard shuru ho gaya aur chakkar aa raha hai.",
        "dialect": "Hinglish",
        "entities": [
            ("tez", "SEVERITY"),
            ("sar dard", "SYMPTOM"),
            ("sar", "SITE"),
            ("chakkar", "SYMPTOM")
        ]
    },
    {
        "text": "Subah se loose motions hai aur pet me cramping ho rahi hai.",
        "dialect": "Hinglish",
        "entities": [
            ("Subah se", "DURATION"),
            ("loose motions", "SYMPTOM"),
            ("pet", "SITE"),
            ("cramping", "SYMPTOM")
        ]
    },
    {
        "text": "Kal raat se gala kharab hai, khansi hai aur body heat lag rahi hai.",
        "dialect": "Hinglish",
        "entities": [
            ("Kal raat se", "DURATION"),
            ("gala kharab", "SYMPTOM"),
            ("gala", "SITE"),
            ("khansi", "SYMPTOM"),
            ("body heat", "SYMPTOM"),
            ("body", "SITE")
        ]
    },
    {
        "text": "Dil tezi se dhadak raha hai aur chest pressure feel ho raha hai.",
        "dialect": "Hinglish",
        "entities": [
            ("dil tezi se dhadak raha", "SYMPTOM"),
            ("chest pressure", "SYMPTOM"),
            ("chest", "SITE")
        ]
    },
    {
        "text": "Kamzori bahut zyada lag rahi hai aur thakan 5 din se hai.",
        "dialect": "Hinglish",
        "entities": [
            ("Kamzori", "SYMPTOM"),
            ("bahut zyada", "SEVERITY"),
            ("thakan", "SYMPTOM"),
            ("5 din", "DURATION")
        ]
    },
    {
        "text": "Skin par red spots aur khujli ho rahi hai 2 days se.",
        "dialect": "Hinglish",
        "entities": [
            ("Skin", "SITE"),
            ("red spots", "SYMPTOM"),
            ("khujli", "SYMPTOM"),
            ("2 days", "DURATION")
        ]
    },
    {
        "text": "Achanak severe abdominal pain ho raha hai aur ulti me blood aaya hai.",
        "dialect": "Hinglish",
        "entities": [
            ("severe", "SEVERITY"),
            ("abdominal pain", "SYMPTOM"),
            ("abdominal", "SITE"),
            ("ulti me blood", "SYMPTOM")
        ]
    },
    {
        "text": "Pichle 3 dino se dry cough hai aur breathing difficulty ho rahi hai.",
        "dialect": "Hinglish",
        "entities": [
            ("3 dino", "DURATION"),
            ("dry cough", "SYMPTOM"),
            ("breathing difficulty", "SYMPTOM")
        ]
    },
    {
        "text": "Kamar me tez dard hai subah se, chalne me problem ho rahi hai.",
        "dialect": "Hinglish",
        "entities": [
            ("Kamar", "SITE"),
            ("tez", "SEVERITY"),
            ("dard", "SYMPTOM"),
            ("subah se", "DURATION")
        ]
    },

    # Tanglish Telecare cases
    {
        "text": "Enakku 3 naal ah gala pain irukku, fever um konjam high ah irukku.",
        "dialect": "Tanglish",
        "entities": [
            ("3 naal", "DURATION"),
            ("gala pain", "SYMPTOM"),
            ("gala", "SITE"),
            ("fever", "SYMPTOM"),
            ("konjam high", "SEVERITY")
        ]
    },
    {
        "text": "Nenju vali romba athigama irukku, saans vidradhula kastam irukku 2 hours ah.",
        "dialect": "Tanglish",
        "entities": [
            ("Nenju vali", "SYMPTOM"),
            ("Nenju", "SITE"),
            ("romba athigama", "SEVERITY"),
            ("saans vidradhula kastam", "SYMPTOM"),
            ("2 hours", "DURATION")
        ]
    },
    {
        "text": "Thalavali 2 naala irukku, romba tired ah irukku body.",
        "dialect": "Tanglish",
        "entities": [
            ("Thalavali", "SYMPTOM"),
            ("2 naala", "DURATION"),
            ("romba", "SEVERITY"),
            ("tired", "SYMPTOM"),
            ("body", "SITE")
        ]
    },
    {
        "text": "Inniku kaalaila irundhu vomiting non stop ah irukku, vayiru vali jaasthi.",
        "dialect": "Tanglish",
        "entities": [
            ("Inniku kaalaila irundhu", "DURATION"),
            ("vomiting", "SYMPTOM"),
            ("non stop", "SEVERITY"),
            ("vayiru vali", "SYMPTOM"),
            ("vayiru", "SITE"),
            ("jaasthi", "SEVERITY")
        ]
    },
    {
        "text": "Thala suthudhu, mayakkam vara madhiri irukku since yesterday.",
        "dialect": "Tanglish",
        "entities": [
            ("Thala suthudhu", "SYMPTOM"),
            ("Thala", "SITE"),
            ("mayakkam", "SYMPTOM"),
            ("since yesterday", "DURATION")
        ]
    },
    {
        "text": "Thondai vali konjam irukku, dry cough 4 days ah irukku.",
        "dialect": "Tanglish",
        "entities": [
            ("Thondai vali", "SYMPTOM"),
            ("Thondai", "SITE"),
            ("konjam", "SEVERITY"),
            ("dry cough", "SYMPTOM"),
            ("4 days", "DURATION")
        ]
    },
    {
        "text": "Nenjula sudden ah heaviness irukku, sweating romba aagudhu.",
        "dialect": "Tanglish",
        "entities": [
            ("Nenjula", "SITE"),
            ("heaviness", "SYMPTOM"),
            ("sweating", "SYMPTOM"),
            ("romba", "SEVERITY")
        ]
    },
    {
        "text": "Kaichal 2 naal ah irukku, body vali romba heavy ah irukku.",
        "dialect": "Tanglish",
        "entities": [
            ("Kaichal", "SYMPTOM"),
            ("2 naal", "DURATION"),
            ("body vali", "SYMPTOM"),
            ("body", "SITE"),
            ("romba heavy", "SEVERITY")
        ]
    },
    {
        "text": "Vayitru pokku 5 times aachu inniku, weak ah feel aagudhu.",
        "dialect": "Tanglish",
        "entities": [
            ("Vayitru pokku", "SYMPTOM"),
            ("Vayitru", "SITE"),
            ("inniku", "DURATION"),
            ("weak", "SYMPTOM")
        ]
    },
    {
        "text": "Mudhuguvalli 1 week ah irukku, acute back pain.",
        "dialect": "Tanglish",
        "entities": [
            ("Mudhuguvalli", "SYMPTOM"),
            ("1 week", "DURATION"),
            ("acute", "SEVERITY"),
            ("back pain", "SYMPTOM")
        ]
    },

    # Clinical Benchmark (NCBI / MedMentions style English)
    {
        "text": "Patient reports severe chest pain and shortness of breath for 3 hours.",
        "dialect": "Clinical English",
        "entities": [
            ("severe", "SEVERITY"),
            ("chest pain", "SYMPTOM"),
            ("chest", "SITE"),
            ("shortness of breath", "SYMPTOM"),
            ("3 hours", "DURATION")
        ]
    },
    {
        "text": "Persistent headache and nausea for past 2 days with moderate dizziness.",
        "dialect": "Clinical English",
        "entities": [
            ("Persistent", "SEVERITY"),
            ("headache", "SYMPTOM"),
            ("nausea", "SYMPTOM"),
            ("2 days", "DURATION"),
            ("moderate", "SEVERITY"),
            ("dizziness", "SYMPTOM")
        ]
    },
    {
        "text": "Acute abdominal pain located in epigastric region with vomiting.",
        "dialect": "Clinical English",
        "entities": [
            ("Acute", "SEVERITY"),
            ("abdominal pain", "SYMPTOM"),
            ("epigastric region", "SITE"),
            ("vomiting", "SYMPTOM")
        ]
    },
    {
        "text": "Mild sore throat and productive cough for 5 days without fever.",
        "dialect": "Clinical English",
        "entities": [
            ("Mild", "SEVERITY"),
            ("sore throat", "SYMPTOM"),
            ("throat", "SITE"),
            ("productive cough", "SYMPTOM"),
            ("5 days", "DURATION")
        ]
    },
    {
        "text": "Sudden onset of severe palpitations and diaphoresis lasting 30 minutes.",
        "dialect": "Clinical English",
        "entities": [
            ("severe", "SEVERITY"),
            ("palpitations", "SYMPTOM"),
            ("diaphoresis", "SYMPTOM"),
            ("30 minutes", "DURATION")
        ]
    },
    {
        "text": "Generalized body weakness and high fever of 102 F for 4 days.",
        "dialect": "Clinical English",
        "entities": [
            ("body weakness", "SYMPTOM"),
            ("body", "SITE"),
            ("high", "SEVERITY"),
            ("fever", "SYMPTOM"),
            ("4 days", "DURATION")
        ]
    },
    {
        "text": "Loss of consciousness and syncope reported today morning.",
        "dialect": "Clinical English",
        "entities": [
            ("Loss of consciousness", "SYMPTOM"),
            ("syncope", "SYMPTOM"),
            ("today morning", "DURATION")
        ]
    },
    {
        "text": "Hematemesis and dark tarry stools starting 6 hours ago with acute distress.",
        "dialect": "Clinical English",
        "entities": [
            ("Hematemesis", "SYMPTOM"),
            ("dark tarry stools", "SYMPTOM"),
            ("6 hours", "DURATION"),
            ("acute", "SEVERITY")
        ]
    },
    {
        "text": "Chronic lower back pain worsening over 3 weeks with bilateral leg weakness.",
        "dialect": "Clinical English",
        "entities": [
            ("lower back pain", "SYMPTOM"),
            ("lower back", "SITE"),
            ("3 weeks", "DURATION"),
            ("leg weakness", "SYMPTOM"),
            ("leg", "SITE")
        ]
    },
    {
        "text": "Erythematous skin rash with pruritus across trunk for 2 days.",
        "dialect": "Clinical English",
        "entities": [
            ("skin rash", "SYMPTOM"),
            ("skin", "SITE"),
            ("pruritus", "SYMPTOM"),
            ("trunk", "SITE"),
            ("2 days", "DURATION")
        ]
    }
]

# Variations generator to expand the dataset across colloquial permutations
AUGMENTATION_TEMPLATES = [
    ("{symptom1} ho raha hai {duration} se aur {severity} {symptom2} bhi hai.", "Hinglish"),
    ("Mujhe {duration} se {site} me {severity} {symptom1} hai.", "Hinglish"),
    ("{site} romba valikkuthu, {duration} ah {severity} {symptom1} irukku.", "Tanglish"),
    ("Enakku {duration} irundhu {symptom1} and {symptom2} romba jaasthi.", "Tanglish"),
    ("Patient experiencing {severity} {symptom1} and {symptom2} for {duration}.", "Clinical English"),
    ("Severe {symptom1} radiating to {site} since {duration}.", "Clinical English"),
    ("Kal se {symptom1} aur {site} me dard hai, feel ho raha hai {severity}.", "Hinglish"),
    ("Thala vali and {symptom1} 2 naal ah irukku, {severity} feeling.", "Tanglish")
]

SLOT_VALUES = {
    "symptom1": [
        ("sar dard", "SYMPTOM"), ("chest pain", "SYMPTOM"), ("vomiting", "SYMPTOM"),
        ("gala dard", "SYMPTOM"), ("khansi", "SYMPTOM"), ("bukhar", "SYMPTOM"),
        ("pet dard", "SYMPTOM"), ("chakkar", "SYMPTOM"), ("nenju vali", "SYMPTOM"),
        ("thala vali", "SYMPTOM"), ("vaanthi", "SYMPTOM"), ("kaichal", "SYMPTOM"),
        ("saans lene me dikkat", "SYMPTOM"), ("loose motions", "SYMPTOM")
    ],
    "symptom2": [
        ("weakness", "SYMPTOM"), ("vomiting", "SYMPTOM"), ("fever", "SYMPTOM"),
        ("dizziness", "SYMPTOM"), ("nausea", "SYMPTOM"), ("body heat", "SYMPTOM"),
        ("tired", "SYMPTOM"), ("sweating", "SYMPTOM"), ("shivering", "SYMPTOM")
    ],
    "severity": [
        ("bahut tez", "SEVERITY"), ("severe", "SEVERITY"), ("halka", "SEVERITY"),
        ("moderate", "SEVERITY"), ("romba high", "SEVERITY"), ("acute", "SEVERITY"),
        ("mild", "SEVERITY"), ("unbearable", "SEVERITY"), ("konjam", "SEVERITY")
    ],
    "duration": [
        ("2 din", "DURATION"), ("3 naal", "DURATION"), ("since morning", "DURATION"),
        ("4 hours", "DURATION"), ("kal raat", "DURATION"), ("5 days", "DURATION"),
        ("1 week", "DURATION"), ("subah se", "DURATION"), ("2 days", "DURATION")
    ],
    "site": [
        ("chest", "SITE"), ("sar", "SITE"), ("gala", "SITE"), ("pet", "SITE"),
        ("thondai", "SITE"), ("nenju", "SITE"), ("back", "SITE"), ("body", "SITE")
    ]
}


def tokenize_with_iob(text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Convert raw text and span entities into precise whitespace token IOB tags."""
    tokens = text.split()
    tags = ["O"] * len(tokens)

    # Sort entities by length descending to match longest phrases first
    sorted_entities = sorted(entities, key=lambda x: len(x[0]), reverse=True)

    for ent_text, ent_label in sorted_entities:
        ent_tokens = ent_text.split()
        m = len(ent_tokens)
        if m == 0:
            continue
        # Find token match sequence
        for i in range(len(tokens) - m + 1):
            window = [re.sub(r"[^\w]", "", t).lower() for t in tokens[i:i+m]]
            target = [re.sub(r"[^\w]", "", t).lower() for t in ent_tokens]
            if window == target:
                if all(tags[i+k] == "O" for k in range(m)):
                    tags[i] = f"B-{ent_label}"
                    for k in range(1, m):
                        tags[i+k] = f"I-{ent_label}"

    return list(zip(tokens, tags))


def generate_curated_dataset() -> Dict:
    """Generate annotated dataset with realistic permutations and splits."""
    random.seed(42)
    all_samples = []

    # 1. Add gold human-curated records
    for item in RAW_RECORDS:
        iob = tokenize_with_iob(item["text"], item["entities"])
        all_samples.append({
            "text": item["text"],
            "dialect": item["dialect"],
            "tokens": [t[0] for t in iob],
            "tags": [t[1] for t in iob],
            "entities": [{"text": e[0], "label": e[1]} for e in item["entities"]],
            "source": "curated_telecare"
        })

    # 2. Add systematic code-mixed template permutations
    for _ in range(120):
        tmpl, dialect = random.choice(AUGMENTATION_TEMPLATES)
        s1, s1_label = random.choice(SLOT_VALUES["symptom1"])
        s2, s2_label = random.choice(SLOT_VALUES["symptom2"])
        sev, sev_label = random.choice(SLOT_VALUES["severity"])
        dur, dur_label = random.choice(SLOT_VALUES["duration"])
        site, site_label = random.choice(SLOT_VALUES["site"])

        text = tmpl.format(
            symptom1=s1, symptom2=s2, severity=sev, duration=dur, site=site
        )
        entities = [
            (s1, s1_label),
            (s2, s2_label),
            (sev, sev_label),
            (dur, dur_label),
            (site, site_label)
        ]
        iob = tokenize_with_iob(text, entities)
        all_samples.append({
            "text": text,
            "dialect": dialect,
            "tokens": [t[0] for t in iob],
            "tags": [t[1] for t in iob],
            "entities": [{"text": e[0], "label": e[1]} for e in entities if e[0] in text],
            "source": "augmented_permutation"
        })

    # Shuffle with seed and partition: Train 70%, Dev 15%, Test 15%
    random.shuffle(all_samples)
    n = len(all_samples)
    train_end = int(n * 0.70)
    dev_end = int(n * 0.85)

    dataset = {
        "metadata": {
            "title": "Annotated Code-Mixed Clinical Telecare Dataset",
            "project": "Mixed Multilingual Symptom Extraction & Triage Support for Telecare",
            "review": "Project Review 2",
            "tag_schema": ["O", "B-SYMPTOM", "I-SYMPTOM", "B-DURATION", "I-DURATION", "B-SEVERITY", "I-SEVERITY", "B-SITE", "I-SITE"],
            "total_sentences": n,
            "train_count": train_end,
            "dev_count": dev_end - train_end,
            "test_count": n - dev_end,
            "dialects": ["Hinglish", "Tanglish", "Hinglish / English mixed", "Clinical English"]
        },
        "train": all_samples[:train_end],
        "dev": all_samples[train_end:dev_end],
        "test": all_samples[dev_end:]
    }

    # Save JSON dataset
    json_path = DATA_DIR / "annotated_clinical_codemixed_dataset.json"
    json_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    # Export CoNLL files
    for split_name in ["train", "dev", "test"]:
        conll_path = DATA_DIR / f"{split_name}.conll"
        lines = []
        for sample in dataset[split_name]:
            for tok, tag in zip(sample["tokens"], sample["tags"]):
                lines.append(f"{tok}\t{tag}")
            lines.append("")  # sentence boundary
        conll_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Dataset generated: {n} sentences ({train_end} train, {dev_end - train_end} dev, {n - dev_end} test)")
    return dataset


if __name__ == "__main__":
    generate_curated_dataset()
