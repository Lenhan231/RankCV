"""
CV-Job Matching API
Run:  uvicorn src.app:app --reload   (from the project root)
Docs: http://127.0.0.1:8000/docs
"""

import re
import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List

# ---------------------------------------------------------------------------
# Load model artifacts (once at startup)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

try:
    tfidf             = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    skills_db         = joblib.load(os.path.join(MODEL_DIR, "skills_db.joblib"))
    skill_name_to_id  = joblib.load(os.path.join(MODEL_DIR, "skill_name_to_id.joblib"))
    title_profiles    = joblib.load(os.path.join(MODEL_DIR, "title_skill_profiles.joblib"))
except FileNotFoundError:
    raise RuntimeError("Model artifacts not found. Run train_model.py first.")

# Pre-compile skill patterns sorted longest-first to avoid short tokens
# swallowing longer skill names (e.g. "r" matching before "r studio")
_SKILL_PATTERNS = [
    (re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE), sid)
    for name, sid in sorted(skill_name_to_id.items(), key=lambda x: -len(x[0]))
]

# ---------------------------------------------------------------------------
# Education ladder
# ---------------------------------------------------------------------------
_EDU_KEYWORDS: list[tuple[str, int]] = [
    ("phd", 4), ("ph.d", 4), ("doctorate", 4), ("doctoral", 4),
    ("master", 3), ("mba", 3), ("m.s", 3), ("m.sc", 3), ("postgraduate", 3),
    ("bachelor", 2), ("b.s", 2), ("b.sc", 2), ("undergraduate", 2), ("college degree", 2),
    ("associate", 1),
    ("high school", 0), ("diploma", 0), ("ged", 0),
]

_EXP_PATTERNS = [
    re.compile(r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience", re.IGNORECASE),
    re.compile(r"(\d+)\s*\+?\s*yrs?\s+(?:of\s+)?experience", re.IGNORECASE),
    re.compile(r"experience\s+(?:of\s+)?(\d+)\s*\+?\s*years?", re.IGNORECASE),
    re.compile(r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:work|industry|relevant)", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> set[int]:
    found: set[int] = set()
    for pattern, sid in _SKILL_PATTERNS:
        if pattern.search(text):
            found.add(sid)
    return found


def extract_edu_level(text: str) -> int:
    t = text.lower()
    best = -1
    for kw, level in _EDU_KEYWORDS:
        if kw in t:
            best = max(best, level)
    return best


def extract_years(text: str) -> float:
    found: list[float] = []
    for p in _EXP_PATTERNS:
        for m in p.finditer(text):
            found.append(float(m.group(1)))
    return max(found) if found else 0.0


def tfidf_similarity(a: str, b: str) -> float:
    vecs = tfidf.transform([a, b])
    return float(cosine_similarity(vecs[0], vecs[1])[0][0])


def score_skill_match(cv_skills: set[int], job_skills: set[int],
                      cv_text: str, job_text: str) -> float:
    if job_skills:
        return len(cv_skills & job_skills) / len(job_skills) * 100.0
    # Fallback to TF-IDF similarity when no structured skills found in job text
    return tfidf_similarity(cv_text, job_text) * 100.0


def score_experience(cv_text: str, job_text: str) -> float:
    job_yrs = extract_years(job_text)
    cv_yrs  = extract_years(cv_text)
    if job_yrs <= 0 and cv_yrs <= 0:
        return 70.0          # neither specifies — neutral
    if job_yrs <= 0:
        return 85.0          # CV has experience but job doesn't require a number
    if cv_yrs <= 0:
        return 40.0          # job requires years but CV doesn't mention any
    return min(cv_yrs / job_yrs * 100.0, 100.0)


def score_education(cv_text: str, job_text: str) -> float:
    job_edu = extract_edu_level(job_text)
    cv_edu  = extract_edu_level(cv_text)
    if job_edu < 0:
        return 80.0          # job doesn't mention education
    if cv_edu < 0:
        return 40.0          # CV doesn't mention education
    if cv_edu >= job_edu:
        return 100.0
    # Partial credit — 1 level below = 70, 2 below = 45, etc.
    gap = job_edu - cv_edu
    return max(0.0, 100.0 - gap * 25.0)


def build_recommendation(overall: float, strengths: list[str],
                          missing: list[str]) -> str:
    top_strengths = ", ".join(strengths[:3]) if strengths else "your current skills"
    top_missing   = ", ".join(missing[:4])   if missing   else "none identified"

    if overall >= 80:
        return (
            f"Excellent match! Your profile strongly aligns with this role. "
            f"Key strengths: {top_strengths}. Apply with confidence."
        )
    if overall >= 65:
        return (
            f"Good match. Highlight your experience in {top_strengths}. "
            f"Consider addressing gaps in: {top_missing}."
        )
    if overall >= 45:
        return (
            f"Moderate match. Your strengths in {top_strengths} are relevant, "
            f"but acquiring these skills would improve your fit: {top_missing}."
        )
    return (
        f"Low match. This role requires significant skill development. "
        f"Key gaps: {top_missing}. "
        f"Consider roles better suited to your current profile or invest in upskilling."
    )

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    cv_text: str
    job_text: str

    @field_validator("cv_text", "job_text")
    @classmethod
    def min_length(cls, v: str) -> str:
        if len(v.strip()) < 50:
            raise ValueError("minimum 50 characters")
        return v


class MatchResponse(BaseModel):
    overall_score: float
    skill_match: float
    experience_match: float
    education_match: float
    missing_skills: List[str]
    strengths: List[str]
    recommendation: str

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CV–Job Match API",
    description="Score how well a CV matches a job description using skills data from 1.6 M real job postings.",
    version="1.0.0",
)


@app.post("/match", response_model=MatchResponse, summary="Score CV against a job description")
def match(req: MatchRequest) -> MatchResponse:
    cv_text  = req.cv_text
    job_text = req.job_text

    # --- Structured skill extraction ---
    cv_skills  = extract_skills(cv_text)
    job_skills = extract_skills(job_text)

    # --- Component scores ---
    skill_match      = score_skill_match(cv_skills, job_skills, cv_text, job_text)
    experience_match = score_experience(cv_text, job_text)
    education_match  = score_education(cv_text, job_text)
    text_sim         = tfidf_similarity(cv_text, job_text) * 100.0

    # --- Weighted overall score ---
    # skill_match has highest weight; text_sim acts as a context-awareness bonus
    overall_score = round(
        skill_match      * 0.45 +
        experience_match * 0.20 +
        education_match  * 0.15 +
        text_sim         * 0.20,
        1,
    )

    # --- Missing skills & strengths ---
    missing_ids  = job_skills - cv_skills
    matched_ids  = cv_skills  & job_skills

    missing_skills = sorted(skills_db[sid]["name"] for sid in missing_ids if sid in skills_db)
    strengths      = sorted(skills_db[sid]["name"] for sid in matched_ids  if sid in skills_db)

    recommendation = build_recommendation(overall_score, strengths, missing_skills)

    return MatchResponse(
        overall_score    = overall_score,
        skill_match      = round(skill_match,      1),
        experience_match = round(experience_match, 1),
        education_match  = round(education_match,  1),
        missing_skills   = missing_skills,
        strengths        = strengths,
        recommendation   = recommendation,
    )


@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "skills_loaded": len(skills_db),
        "job_title_profiles": len(title_profiles),
        "tfidf_vocab_size": len(tfidf.vocabulary_),
    }
