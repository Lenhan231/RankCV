# CV–Job Match API

A REST API that scores how well a candidate's CV matches a job description, built on skills data extracted from **1.6 million real job postings** (the [Luke Barousse data-jobs dataset](https://www.lukebarousse.com/sql)).

The service returns a structured breakdown — overall score plus skill, experience, and education sub-scores — along with the candidate's matched strengths, missing skills, and a written recommendation.

---

## How it works

The pipeline has two stages:

**1. Data export & training** (`src/data.py` → `src/train_model.py`)

| Source | Used for |
|---|---|
| `skills_dim.csv` (262 skills) | Keyword-based skill extraction from free text |
| `skills_job_dim.csv` + `job_postings_fact.csv` (1.6 M rows) | TF-IDF vectorizer trained on the real job title + skills corpus |
| Aggregated profiles | `job_title_short` → skill-frequency table |

**2. Scoring API** (`src/app.py`)

For each request the service:
- Extracts known skills from both the CV and the job text (regex over the 262-skill dictionary, longest-match-first).
- Computes sub-scores for skills, experience (regex on "N years"), and education (degree-level ladder).
- Computes a TF-IDF cosine similarity between the two texts as a context signal.
- Combines them into a weighted overall score:

```
overall = skill_match×0.45 + experience_match×0.20 + education_match×0.15 + tfidf_similarity×0.20
```

The `recommendation` text is template-based (deterministic, no external API needed).

---

## Project structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data.py          # export CSVs from MotherDuck → data/
│   ├── train_model.py   # build model artifacts → model/
│   └── app.py           # FastAPI service
├── data/                # source CSVs (gitignored — regenerable)
└── model/               # trained artifacts (committed)
```

---

## Setup

```bash
pip install -r requirements.txt
```

### (Optional) Regenerate the data and retrain

The CSVs are not committed (the largest is ~256 MB). To rebuild them you need a free [MotherDuck](https://motherduck.com/) account — `data.py` triggers a browser SSO login on first run.

```bash
python src/data.py          # exports the 4 CSVs into data/
python src/train_model.py   # rebuilds model/ artifacts
```

The trained `model/` artifacts are already committed, so this step is **optional** — the API runs out of the box.

---

## Run the API

```bash
uvicorn src.app:app --reload
```

- Interactive docs: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

---

## API reference

### `POST /match`

**Request**

```json
{
  "cv_text": "string (minimum 50 characters)",
  "job_text": "string (minimum 50 characters)"
}
```

**Response**

```json
{
  "overall_score": 0,
  "skill_match": 0,
  "experience_match": 0,
  "education_match": 0,
  "missing_skills": ["string"],
  "strengths": ["string"],
  "recommendation": "string"
}
```

**Example**

```bash
curl -X POST http://127.0.0.1:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "cv_text": "Data analyst with 4 years experience in SQL, Python, Tableau, Excel and Power BI. Bachelor degree in Computer Science.",
    "job_text": "Data Analyst, 3+ years experience. Required: SQL, Python, Tableau, Excel. Nice to have: Power BI, Spark. Bachelor degree required."
  }'
```

```json
{
  "overall_score": 81.2,
  "skill_match": 85.7,
  "experience_match": 100.0,
  "education_match": 100.0,
  "missing_skills": ["spark"],
  "strengths": ["azure", "excel", "power bi", "python", "sql", "tableau"],
  "recommendation": "Excellent match! Your profile strongly aligns with this role. ..."
}
```

---

## Notes & limitations

- Skill matching is dictionary-based over 262 known tech skills; skills outside that set are not detected.
- Experience and education are parsed heuristically from text patterns.
- The dataset is dominated by data-related roles (Data Analyst, Data Scientist, Data Engineer, etc.), so accuracy is highest for those job families.
