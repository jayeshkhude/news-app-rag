# Indian News RAG

A Retrieval-Augmented Generation (RAG) system built on 1,269 Indian news summaries.
Ask questions in natural language — the system retrieves the most relevant articles
and uses Claude to generate a grounded answer.

---

## How it works

```
Your Question
     │
     ▼
[Sentence Transformer]  ←  all-MiniLM-L6-v2
     │  query embedding
     ▼
[FAISS Index]           ←  1,269 article embeddings
     │  top-k most similar articles
     ▼
[Prompt Builder]        ←  formats retrieved context
     │
     ▼
[Claude Sonnet]         ←  generates grounded answer
     │
     ▼
Answer + Sources
```

---

## Setup

```bash
# 1. Clone / download the project
cd indian_news_rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 4. Build the vector index (run once)
python src/build_index.py

# 5. Start the chatbot
python src/chatbot.py
```

---

## Project structure

```
indian_news_rag/
├── data/
│   ├── summaries_newslens.csv    ← original dataset (1,269 rows)
│   ├── faiss_index.bin           ← generated after build_index.py
│   └── documents.pkl             ← generated after build_index.py
│
├── src/
│   ├── data_loader.py            ← load CSV, build document dicts
│   ├── vectorstore.py            ← embed + FAISS index + search
│   ├── rag_pipeline.py           ← retrieve → prompt → Claude
│   ├── build_index.py            ← one-time index builder
│   └── chatbot.py                ← interactive CLI
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Dataset columns

| Column           | Description                          |
|------------------|--------------------------------------|
| headline         | Article headline                     |
| summary          | 2–5 sentence summary                 |
| category         | politics / sports / world / economy / other / science_tech / society |
| importance_score | Importance score (0 = standard)      |

---

## Example questions

```
What happened in the Tamil Nadu elections?
Tell me about India's cricket performance recently
What are the latest developments in the Indian economy?
/category sports
Who won the recent IPL matches?
/category world
What is happening between India and Pakistan?
```

---

## Chatbot commands

| Command              | Description                          |
|----------------------|--------------------------------------|
| `/category <name>`   | Filter retrieval to one category     |
| `/clear`             | Remove category filter               |
| `/topk <n>`          | Set number of retrieved articles     |
| `/sources`           | Toggle showing source headlines      |
| `/quit`              | Exit                                 |

---

## Tech stack

| Component       | Library / Model                      |
|-----------------|--------------------------------------|
| Embeddings      | `sentence-transformers` (MiniLM-L6)  |
| Vector search   | `faiss-cpu` (IndexFlatIP)            |
| LLM             | Claude Sonnet 4.6 via `anthropic`    |
| Data            | `pandas`                             |
| CLI             | `colorama`                           |
