"""FastAPI Server for Mixed Multilingual Symptom Extraction & Triage Support.
Project Review 2 | 22BML0008 22BML0075 24BML0099
"""
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from services import analyze

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"

app = FastAPI(
    title="Mixed Multilingual Symptom Extraction & Triage Support for Telecare",
    version="2.0.0",
    description="End-to-End Multilingual Clinical NLP, Transformer-CRF, SNOMED CT/ICD-10 Normalization & Triage"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=2000)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "telecare-triage-prototype",
        "phase": "Project Review 2",
        "pipeline": "IndicNLP -> Transformer-CRF -> SNOMED/ICD -> Triage -> Doctor Summary"
    }


@app.get("/api/demo-cases")
def demo_cases():
    path = DATA_DIR / "demo_cases.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Demo cases not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/analyze")
def analyze_text(payload: AnalyzeRequest):
    return analyze(payload.text)


@app.get("/api/model-evaluation")
def model_evaluation():
    path = DATA_DIR / "evaluation_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model evaluation metrics not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/terminology")
def terminology_concepts():
    path = DATA_DIR / "terminology.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Terminology database not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/intake-report")
def get_intake_report(payload: AnalyzeRequest):
    result = analyze(payload.text)
    return result.get("intake_report", {})


# Serve frontend static build when frontend/dist exists
DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
