"""
Export the source tables from the shared MotherDuck database into local CSVs.
Output goes to the project-level data/ directory.

Usage:  python src/data.py
(A browser SSO login to MotherDuck will be triggered on first run.)
"""

import os
import duckdb

# Project root = parent of this file's directory (src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def out(name: str) -> str:
    # Forward slashes keep the path valid inside the DuckDB SQL string on Windows
    return os.path.join(DATA_DIR, name).replace("\\", "/")


con = duckdb.connect()

con.execute("""
ATTACH 'md:_share/data_jobs/87603155-cdc7-4c80-85ad-3a6b0d760d93' AS data_jobs;
""")

for table in ["skills_dim", "skills_job_dim", "company_dim", "job_postings_fact"]:
    print(f"Exporting {table} ...")
    con.execute(f"""
        COPY (
            SELECT *
            FROM data_jobs.main.{table}
        ) TO '{out(table + ".csv")}' (HEADER, DELIMITER ',');
    """)

print(f"\nAll tables exported to {DATA_DIR}")
