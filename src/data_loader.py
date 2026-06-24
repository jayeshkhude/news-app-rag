"""
data_loader.py
Loads summaries from the Neon PostgreSQL `summaries` table.
"""

import pandas as pd
from typing import List, Dict

from config import get_config_bool


def load_dataset() -> pd.DataFrame:
    """Load summaries exclusively from Neon."""
    if not get_config_bool("USE_NEON", default=True):
        raise RuntimeError(
            "Local CSV is disabled. Set USE_NEON=true and configure Neon in .env."
        )

    from neon_db import _resolve_database_url, neon_configured

    if not neon_configured():
        raise ValueError(
            "Neon is not configured. Set DATABASE_URL or NEON_API_KEY in .env."
        )

    query = """
        SELECT id, topic, headline, summary, category, importance_score,
               sources, article_links, created_at, summary_date, thumbnail_url,
               links_hash, content_hash
        FROM summaries
        WHERE headline IS NOT NULL
          AND summary IS NOT NULL
        ORDER BY id
    """
    df = pd.read_sql(query, _resolve_database_url())

    for col in ["headline", "summary", "category"]:
        df[col] = df[col].astype(str).str.strip()

    return df.dropna(subset=["headline", "summary"]).reset_index(drop=True)


def load_summaries_since(since_id: int = 0) -> pd.DataFrame:
    """Load summaries with id greater than since_id (for incremental sync)."""
    from neon_db import _resolve_database_url, neon_configured

    if not neon_configured():
        raise ValueError("Neon is not configured.")

    query = f"""
        SELECT id, topic, headline, summary, category, importance_score,
               sources, article_links, created_at, summary_date, thumbnail_url,
               links_hash, content_hash
        FROM summaries
        WHERE headline IS NOT NULL
          AND summary IS NOT NULL
          AND id > {int(since_id)}
        ORDER BY id
    """
    df = pd.read_sql(query, _resolve_database_url())

    for col in ["headline", "summary", "category"]:
        df[col] = df[col].astype(str).str.strip()

    return df.dropna(subset=["headline", "summary"]).reset_index(drop=True)


def get_neon_stats() -> Dict:
    """Return row count and latest summary date from Neon."""
    import pandas as pd
    from neon_db import _resolve_database_url

    stats = pd.read_sql(
        """
        SELECT COUNT(*) AS count,
               MAX(id) AS max_id,
               MAX(summary_date) AS latest_date
        FROM summaries
        WHERE headline IS NOT NULL AND summary IS NOT NULL
        """,
        _resolve_database_url(),
    )
    row = stats.iloc[0]
    return {
        "count": int(row["count"]),
        "max_id": int(row["max_id"]),
        "latest_date": str(row["latest_date"]) if row["latest_date"] else "",
    }


def build_documents(df: pd.DataFrame) -> List[Dict]:
    """Convert each row into a document dict for embedding and retrieval."""
    documents = []
    for idx, row in df.iterrows():
        text_parts = []
        if row.get("topic") and str(row["topic"]).strip():
            text_parts.append(f"Topic: {str(row['topic']).strip()}")
        if row.get("headline") and str(row["headline"]).strip():
            text_parts.append(f"Headline: {str(row['headline']).strip()}")
        if row.get("summary") and str(row["summary"]).strip():
            text_parts.append(f"Summary: {str(row['summary']).strip()}")
        if row.get("category") and str(row["category"]).strip():
            text_parts.append(f"Category: {str(row['category']).strip()}")
        if row.get("summary_date") and str(row["summary_date"]).strip():
            text_parts.append(f"Date: {str(row['summary_date']).strip()}")
        if row.get("sources") and str(row["sources"]).strip():
            text_parts.append(f"Sources: {str(row['sources']).strip()}")

        doc_id = int(row["id"]) if "id" in row and pd.notna(row["id"]) else int(idx)
        documents.append({
            "id": doc_id,
            "text": "\n".join(text_parts),
            "headline": row.get("headline", ""),
            "summary": row.get("summary", ""),
            "category": row.get("category", "other"),
            "importance": int(row.get("importance_score", 0))
            if pd.notna(row.get("importance_score", 0))
            else 0,
            "topic": row.get("topic", ""),
            "sources": row.get("sources", "[]"),
            "article_links": row.get("article_links", "[]"),
            "created_at": str(row.get("created_at", "")),
            "summary_date": str(row.get("summary_date", "")),
            "thumbnail_url": row.get("thumbnail_url", ""),
            "links_hash": row.get("links_hash", ""),
            "content_hash": row.get("content_hash", ""),
        })
    return documents


def filter_by_category(documents: List[Dict], category: str) -> List[Dict]:
    return [d for d in documents if d["category"].lower() == category.lower()]


def get_categories(documents: List[Dict]) -> List[str]:
    return sorted(set(d["category"] for d in documents))


if __name__ == "__main__":
    stats = get_neon_stats()
    df = load_dataset()
    print(f"Neon project summaries table: {stats['count']} rows")
    print(f"Latest news date: {stats['latest_date']}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
