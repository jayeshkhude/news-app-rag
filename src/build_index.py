"""
build_index.py
One-time script to embed all documents and save the FAISS index.
Run this before chatbot.py if you want to pre-build the index.

Usage:
    python src/build_index.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset, build_documents
from vectorstore import VectorStore


def main():
    print("=== Building FAISS index for Indian News RAG ===\n")
    print("Data source: Neon `summaries` table\n")

    # 1. Load data
    df = load_dataset()
    print(f"Loaded {len(df)} articles")
    print(f"Categories: {df['category'].value_counts().to_dict()}\n")

    # 2. Build documents
    docs = build_documents(df)

    # 3. Embed + index
    vs = VectorStore(model_name="all-MiniLM-L6-v2")
    vs.build(docs)

    # 4. Save
    vs.save()
    print("\nIndex saved successfully.")

    # 5. Quick sanity check
    print("\n=== Sanity check: top 3 results for 'cricket match India' ===")
    results = vs.search("cricket match India", top_k=3)
    for doc, score in results:
        print(f"  [{score:.4f}] [{doc['category']}] {doc['headline'][:70]}")

    print("\nAll done. You can now run: python src/chatbot.py")


if __name__ == "__main__":
    main()
