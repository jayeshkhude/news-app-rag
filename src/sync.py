"""
sync.py
Keep the local FAISS index in sync with the growing Neon summaries table.
"""

from typing import Dict

from data_loader import build_documents, get_neon_stats, load_dataset, load_summaries_since
from vectorstore import VectorStore


def _full_rebuild(vs: VectorStore, neon_stats: Dict) -> Dict:
    df = load_dataset()
    docs = build_documents(df)
    vs.build(docs)
    vs.save()
    return {
        "status": "rebuilt",
        "count": len(vs.documents),
        "added": len(docs),
        "latest_date": neon_stats["latest_date"],
    }


def ensure_index_synced(vs: VectorStore, force_rebuild: bool = False) -> Dict:
    """
    Sync the vector index with Neon.
    - New rows only: incremental embed + append (fast)
    - Missing/corrupt index or deletions: full rebuild
    """
    neon_stats = get_neon_stats()

    if force_rebuild or vs.index is None or not vs.documents:
        result = _full_rebuild(vs, neon_stats)
        result["added"] = result["count"]
        return result

    index_count = len(vs.documents)
    index_ids = {d["id"] for d in vs.documents}

    if (
        index_count == neon_stats["count"]
        and index_ids
        and max(index_ids) == neon_stats["max_id"]
    ):
        return {
            "status": "synced",
            "count": index_count,
            "added": 0,
            "latest_date": neon_stats["latest_date"],
        }

    # Growing database — append only new summaries
    if neon_stats["count"] > index_count and index_ids:
        since_id = max(index_ids)
        df = load_summaries_since(since_id)
        new_docs = build_documents(df)
        if new_docs:
            added = vs.add_documents(new_docs)
            vs.save()
            return {
                "status": "updated",
                "count": len(vs.documents),
                "added": added,
                "latest_date": neon_stats["latest_date"],
            }

    # Count mismatch, deletions, or stale data — full rebuild
    return _full_rebuild(vs, neon_stats)
