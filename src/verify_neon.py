"""Verify Neon connection and compare with local FAISS cache."""

import pickle
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from neon_db import _discover_project_id, _resolve_database_url

DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "summaries_newslens.csv"
PKL_PATH = DATA_DIR / "documents.pkl"


def main():
    url = _resolve_database_url()
    project_id = _discover_project_id()

    print("=== NEON DATABASE (summaries table) ===")
    print(f"Project ID: {project_id}")

    stats = pd.read_sql(
        """
        SELECT COUNT(*) AS cnt,
               MAX(id) AS max_id,
               MAX(summary_date) AS latest_date,
               MAX(created_at) AS latest_created
        FROM summaries
        WHERE headline IS NOT NULL AND summary IS NOT NULL
        """,
        url,
    )
    print(stats.to_string(index=False))

    latest = pd.read_sql(
        """
        SELECT id, headline, summary_date, created_at, content_hash
        FROM summaries
        ORDER BY id DESC
        LIMIT 3
        """,
        url,
    )
    print("\nLatest 3 rows in Neon:")
    for _, row in latest.iterrows():
        print(
            f"  id={row.id} date={row.summary_date} "
            f"hash={str(row.content_hash)[:16]}... | {row.headline[:60]}"
        )

    neon_ids = set(pd.read_sql("SELECT id FROM summaries", url)["id"])

    print("\n=== LOCAL FILES ===")
    print(f"CSV exists: {CSV_PATH.exists()} ({CSV_PATH})")
    print(f"FAISS cache exists: {PKL_PATH.exists()}")

    if PKL_PATH.exists():
        with open(PKL_PATH, "rb") as f:
            docs = pickle.load(f)
        print(f"\n=== LOCAL FAISS CACHE ===")
        print(f"Documents in index: {len(docs)}")
        dates = [d.get("summary_date", "") for d in docs if d.get("summary_date")]
        ids = [d.get("id") for d in docs]
        print(f"Latest summary_date in index: {max(dates) if dates else 'N/A'}")
        print(f"Max id in index: {max(ids) if ids else 'N/A'}")
        with_hash = sum(1 for d in docs if d.get("content_hash"))
        print(f"Docs with content_hash (Neon field): {with_hash}/{len(docs)}")

        index_ids = set(ids)
        print(
            f"\nSync status: neon={len(neon_ids)} index={len(index_ids)} "
            f"missing={len(neon_ids - index_ids)} extra={len(index_ids - neon_ids)}"
        )
        if neon_ids == index_ids and with_hash == len(docs):
            print("RESULT: Index matches Neon summaries table.")
        else:
            print("RESULT: Index is OUT OF SYNC with Neon — rebuild required.")


if __name__ == "__main__":
    main()
