"""
Train CV-Job matching model artifacts from job postings data.

Produces model/ directory with:
  - tfidf_vectorizer.joblib   : TF-IDF trained on real job-title + skills corpus
  - skills_db.joblib          : {skill_id: {name, type}}
  - skill_name_to_id.joblib   : {lowercase_skill_name: skill_id}
  - title_skill_profiles.joblib : {job_title_short: {skill_id: frequency}}
"""

import re
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Project root = parent of this file's directory (src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("Loading skills_dim.csv ...")
skills_df = pd.read_csv(os.path.join(DATA_DIR, "skills_dim.csv"))

print("Loading skills_job_dim.csv ...")
skills_job_df = pd.read_csv(os.path.join(DATA_DIR, "skills_job_dim.csv"))

print("Loading job_postings_fact.csv (titles only) ...")
jobs_df = pd.read_csv(
    os.path.join(DATA_DIR, "job_postings_fact.csv"),
    usecols=["job_id", "job_title_short", "job_title"],
)
print(f"  Loaded {len(jobs_df):,} job postings")

# ---------------------------------------------------------------------------
# 2. Build skills lookup structures
# ---------------------------------------------------------------------------
skills_db: dict[int, dict] = {}
for _, row in skills_df.iterrows():
    skills_db[int(row["skill_id"])] = {
        "name": row["skills"].strip().lower(),
        "type": row["type"].strip(),
    }

# Map lowercase skill name → skill_id (longest names matched first to avoid
# short tokens like "r" swallowing longer ones)
skill_name_to_id: dict[str, int] = {
    v["name"]: k for k, v in sorted(skills_db.items(), key=lambda x: -len(x[1]["name"]))
}

print(f"  Skills in DB: {len(skills_db)}")

# ---------------------------------------------------------------------------
# 3. Build per-job-title skill frequency profiles
#    frequency = fraction of postings with that title that require the skill
# ---------------------------------------------------------------------------
print("Building job-title → skill frequency profiles ...")
job_skills = (
    skills_job_df.groupby("job_id")["skill_id"].apply(list).reset_index()
)
job_skills.columns = ["job_id", "skill_ids"]
merged = jobs_df.merge(job_skills, on="job_id", how="inner")

title_skill_profiles: dict[str, dict[int, float]] = {}
for title, group in merged.groupby("job_title_short"):
    counter: dict[int, int] = {}
    total = len(group)
    for skill_list in group["skill_ids"]:
        for sid in skill_list:
            counter[int(sid)] = counter.get(int(sid), 0) + 1
    title_skill_profiles[title] = {sid: cnt / total for sid, cnt in counter.items()}

print(f"  Job title profiles built: {len(title_skill_profiles)}")

# ---------------------------------------------------------------------------
# 4. Train TF-IDF vectorizer on job-title + skills text
#    Sampled to 150 k rows for speed; shuffled to cover all titles
# ---------------------------------------------------------------------------
print("Building TF-IDF corpus ...")

def _row_to_text(row) -> str:
    title = str(row["job_title"]) if not pd.isna(row["job_title"]) else ""
    sids = row["skill_ids"]
    skill_tokens = " ".join(skills_db[sid]["name"] for sid in sids if sid in skills_db)
    return f"{title} {skill_tokens}"

SAMPLE = min(150_000, len(merged))
sample_df = merged.sample(n=SAMPLE, random_state=42)
corpus = sample_df.apply(_row_to_text, axis=1).tolist()

print(f"  Fitting TF-IDF on {SAMPLE:,} documents ...")
tfidf = TfidfVectorizer(
    max_features=8_000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    sublinear_tf=True,
)
tfidf.fit(corpus)
print(f"  Vocabulary size: {len(tfidf.vocabulary_):,}")

# ---------------------------------------------------------------------------
# 5. Save artifacts
# ---------------------------------------------------------------------------
joblib.dump(tfidf,                os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(skills_db,            os.path.join(MODEL_DIR, "skills_db.joblib"))
joblib.dump(skill_name_to_id,     os.path.join(MODEL_DIR, "skill_name_to_id.joblib"))
joblib.dump(title_skill_profiles, os.path.join(MODEL_DIR, "title_skill_profiles.joblib"))

print("\nAll artifacts saved to model/")
print("  tfidf_vectorizer.joblib")
print("  skills_db.joblib")
print("  skill_name_to_id.joblib")
print("  title_skill_profiles.joblib")
