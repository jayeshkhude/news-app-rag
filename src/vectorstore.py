"""
vectorstore.py
Embeds documents using sentence-transformers and builds a FAISS index.
Supports save/load so you don't re-embed every run.
"""

import os
import json
import pickle
import re
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Keywords that signal the user wants recent/latest news
RECENCY_KEYWORDS = re.compile(
    r"\b(latest|recent|newest|today|yesterday|this week|this month|"
    r"current|breaking|just happened|now|new|last few days|today's|todays)\b",
    re.IGNORECASE,
)


def is_recency_query(query: str) -> bool:
    """Return True if the query is asking about recent/latest news."""
    return bool(RECENCY_KEYWORDS.search(query))


INDEX_DIR = Path(__file__).parent.parent / "data"
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
DOCS_FILE = INDEX_DIR / "documents.pkl"


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents: List[Dict] = []
        self.dimension = self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------ #
    # Build                                                                #
    # ------------------------------------------------------------------ #

    def build(self, documents: List[Dict], batch_size: int = 64) -> None:
        """Embed all documents and build a flat L2 FAISS index."""
        print(f"Embedding {len(documents)} documents...")
        texts = [d["text"] for d in documents]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via inner product
        )

        # FAISS IndexFlatIP = inner product on normalised vectors = cosine
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.documents = documents
        print(f"Index built with {self.index.ntotal} vectors (dim={self.dimension})")

    def add_documents(self, documents: List[Dict], batch_size: int = 64) -> int:
        """Embed and append new documents to an existing index."""
        if not documents:
            return 0
        if self.index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        existing_ids = {d["id"] for d in self.documents}
        new_docs = [d for d in documents if d["id"] not in existing_ids]
        if not new_docs:
            return 0

        print(f"Adding {len(new_docs)} new documents to index...")
        texts = [d["text"] for d in new_docs]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(new_docs) > 32,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(new_docs)
        print(f"Index now has {self.index.ntotal} vectors")
        return len(new_docs)

    # ------------------------------------------------------------------ #
    # Retrieve                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _recency_score(doc: Dict, now: datetime, half_life_days: float = 7.0) -> float:
        """
        Return a recency score in [0, 1] based on the document's summary_date.
        Uses exponential decay: score = exp(-age_days * ln2 / half_life_days)
        A 7-day half-life means a 7-day-old article gets 0.5, 14-day-old gets 0.25, etc.
        """
        date_str = doc.get("summary_date", "")
        if not date_str or date_str in ("None", "NaT", ""):
            return 0.0
        try:
            doc_date = datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            return 0.0
        age_days = max((now - doc_date).days, 0)
        import math
        return math.exp(-age_days * math.log(2) / half_life_days)

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: str = None,
        recency_boost: float = 0.0,
    ) -> List[Tuple[Dict, float]]:
        """
        Return top_k (document, score) pairs for the query.
        Optionally filter by category after retrieval (over-fetch to compensate).

        recency_boost: float in [0, 1]. When > 0, the final score is:
            final = (1 - recency_boost) * cosine_score + recency_boost * recency_score
        This lets recent articles rank higher for time-sensitive queries.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        # When recency or category filtering is active, over-fetch to get
        # enough candidates for re-ranking / filtering.
        need_rerank = recency_boost > 0 or category_filter
        fetch_k = min(top_k * 10 if need_rerank else top_k, self.index.ntotal)

        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, fetch_k)
        scores, indices = scores[0], indices[0]

        now = datetime.utcnow()

        candidates = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            doc = self.documents[idx]
            if category_filter and doc["category"].lower() != category_filter.lower():
                continue

            if recency_boost > 0:
                rec = self._recency_score(doc, now)
                final_score = (1 - recency_boost) * float(score) + recency_boost * rec
            else:
                final_score = float(score)

            candidates.append((doc, final_score))

        # Re-sort by final blended score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_k]

    # ------------------------------------------------------------------ #
    # Persist                                                              #
    # ------------------------------------------------------------------ #

    def save(self, index_path: str = None, docs_path: str = None) -> None:
        ip = Path(index_path) if index_path else INDEX_FILE
        dp = Path(docs_path) if docs_path else DOCS_FILE
        faiss.write_index(self.index, str(ip))
        with open(dp, "wb") as f:
            pickle.dump(self.documents, f)
        print(f"Saved index -> {ip}  |  docs -> {dp}")

    def load(self, index_path: str = None, docs_path: str = None) -> bool:
        ip = Path(index_path) if index_path else INDEX_FILE
        dp = Path(docs_path) if docs_path else DOCS_FILE
        if not ip.exists() or not dp.exists():
            return False
        self.index = faiss.read_index(str(ip))
        with open(dp, "rb") as f:
            self.documents = pickle.load(f)
        print(f"Loaded index ({self.index.ntotal} vectors) from {ip}")
        return True


if __name__ == "__main__":
    from data_loader import load_dataset, build_documents

    df = load_dataset()
    docs = build_documents(df)

    vs = VectorStore()
    vs.build(docs)
    vs.save()

    results = vs.search("elections in India", top_k=3)
    for doc, score in results:
        print(f"[{score:.4f}] {doc['headline'][:80]}")
