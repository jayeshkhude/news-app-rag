"""
rag_pipeline.py
RAG pipeline using Groq (free) as the LLM backend.
Model: llama-3.1-8b-instant — fast and free on Groq.
"""

import os
from typing import List, Dict, Tuple, Optional

from groq import Groq

from config import get_config
from vectorstore import VectorStore, is_recency_query

GROQ_API_KEY = get_config("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TOP_K = int(get_config("TOP_K", "5") or "5")

SYSTEM_PROMPT = """You are an expert Indian news analyst assistant.
You answer questions strictly based on the provided news summaries.
If the answer cannot be found in the provided context, say so clearly.
Always cite which article(s) you are drawing from using [Article N] notation.
Keep answers concise, factual, and well-structured.

IMPORTANT: When the user asks about "latest", "recent", or "new" news,
always prioritize and highlight the articles with the most recent dates.
Explicitly mention the dates of the articles you cite so the user knows
how current the information is."""


def build_context(results: List[Tuple[Dict, float]]) -> str:
    parts = []
    for i, (doc, score) in enumerate(results, 1):
        text_lines = [
            f"[Article {i}]",
            f"Topic: {doc.get('topic', 'N/A')}",
            f"Category: {doc.get('category', 'N/A')}",
            f"Headline: {doc.get('headline', 'N/A')}",
            f"Summary: {doc.get('summary', 'N/A')}"
        ]
        if doc.get("summary_date"):
            text_lines.append(f"Date: {doc['summary_date']}")
        if doc.get("sources") and doc["sources"] != "[]":
            text_lines.append(f"Reporting Outlets: {doc['sources']}")
        if doc.get("importance"):
            text_lines.append(f"Importance Score: {doc['importance']}")
        
        text_lines.append(f"Relevance Score: {score:.3f}")
        parts.append("\n".join(text_lines))
    return "\n\n---\n\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    return f"""Below are relevant Indian news summaries retrieved for your query.

=== RETRIEVED CONTEXT ===
{context}

=== USER QUESTION ===
{query}

Answer based only on the context above. Cite using [Article N] notation."""


class RAGPipeline:
    def __init__(self, vector_store: VectorStore, top_k: int = DEFAULT_TOP_K):
        self.vs = vector_store
        self.top_k = top_k
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = DEFAULT_MODEL

    def query(
        self,
        question: str,
        category_filter: Optional[str] = None,
        top_k: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict:
        k = top_k or self.top_k

        # Auto-sync with Neon before each query (fast check; embeds only new rows)
        from sync import ensure_index_synced
        sync = ensure_index_synced(self.vs)

        # 1. Retrieve — apply recency boost when the query is about latest news
        recency = 0.5 if is_recency_query(question) else 0.0
        results = self.vs.search(
            question, top_k=k, category_filter=category_filter,
            recency_boost=recency,
        )
        if not results:
            return {
                "question": question,
                "answer": "No relevant articles found for your query.",
                "sources": [],
                "sync": sync,
            }

        # 2. Build prompt
        context = build_context(results)
        prompt = build_prompt(question, context)

        if verbose:
            print(f"\n--- Retrieved {len(results)} articles ---")
            for doc, score in results:
                print(f"  [{score:.3f}] {doc['headline'][:70]}")

        # 3. Call Groq (free)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        answer = response.choices[0].message.content

        # 4. Package sources
        sources = [
            {
                "rank": i + 1,
                "headline": doc.get("headline", ""),
                "category": doc.get("category", ""),
                "score": round(score, 4),
                "topic": doc.get("topic", ""),
                "summary": doc.get("summary", ""),
                "summary_date": doc.get("summary_date", ""),
                "sources_list": doc.get("sources", "[]"),
                "article_links": doc.get("article_links", "[]"),
                "thumbnail_url": doc.get("thumbnail_url", ""),
            }
            for i, (doc, score) in enumerate(results)
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "sync": sync,
        }


def print_response(response: Dict) -> None:
    print("\n" + "=" * 60)
    print(f"QUESTION: {response['question']}")
    print("=" * 60)
    print(f"\nANSWER:\n{response['answer']}")
    print("\nSOURCES:")
    for src in response["sources"]:
        print(f"  [{src['rank']}] ({src['category']}) {src['headline'][:70]}")
    print("=" * 60)
