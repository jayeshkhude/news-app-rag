"""
app.py
Web UI for Indian News RAG.
Run: streamlit run src/app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from chatbot import setup
from data_loader import get_neon_stats
from neon_db import _discover_project_id, neon_configured

CATEGORIES = [
    "All categories",
    "politics",
    "sports",
    "world",
    "economy",
    "other",
    "science_tech",
    "society",
]

EXAMPLES = [
    "What happened in Indian cricket recently?",
    "Tell me about the latest Indian economy news",
    "What are the developments in Tamil Nadu politics?",
    "What science and tech news is trending in India?",
]


@st.cache_resource
def load_rag():
    return setup()


def render_sources_ui(sources):
    if not sources:
        return

    st.markdown("### Cited News Sources")
    for s in sources:
        date_str = f" ({s['summary_date']})" if s.get("summary_date") else ""
        title = f"[{s['rank']}] **{s['category'].upper()}** — {s['headline']}{date_str}"

        with st.expander(title, expanded=False):
            col1, col2 = st.columns([1, 4]) if s.get("thumbnail_url") else (None, st.container())

            if col1 is not None and s.get("thumbnail_url"):
                col1.image(s["thumbnail_url"], use_container_width=True)

            with col2:
                if s.get("topic"):
                    st.markdown(f"**Topic:** {s['topic']}")
                st.markdown(f"**Summary:** {s['summary']}")

                try:
                    outlets = json.loads(s.get("sources_list", "[]"))
                    if outlets:
                        st.markdown(f"**Reporting Outlets:** {', '.join(outlets)}")
                except Exception:
                    pass

                try:
                    links = json.loads(s.get("article_links", "[]"))
                    if links:
                        link_markdowns = []
                        for link in links:
                            source_name = link.get("source", "Link")
                            url = link.get("link", "#")
                            link_markdowns.append(f"[{source_name}]({url})")
                        st.markdown(f"**Original Articles:** {' | '.join(link_markdowns)}")
                except Exception:
                    pass


def main():
    st.set_page_config(
        page_title="Indian News RAG",
        page_icon="📰",
        layout="wide",
    )

    st.title("Indian News RAG")
    st.caption("Live Neon `summaries` table — auto-syncs new articles on each query")

    with st.sidebar:
        st.header("Settings")
        category = st.selectbox("Category filter", CATEGORIES)
        top_k = st.slider("Articles to retrieve", 1, 10, 5)
        show_sources = st.checkbox("Show sources", value=True)

        st.divider()
        st.subheader("Neon Database")

        if neon_configured():
            try:
                project = _discover_project_id()
                stats = get_neon_stats()
                st.success("Connected to Neon")
                st.info(
                    f"**Project:** newslens (`{project}`)\n\n"
                    f"**Table:** `summaries`\n\n"
                    f"**Articles:** {stats['count']}\n\n"
                    f"**Latest news:** {stats['latest_date']}"
                )
                st.caption("Search index builds automatically on your first question.")
            except Exception as e:
                st.error(f"Neon connection failed: {e}")
        else:
            st.error("Neon not configured. Add secrets in Streamlit Cloud settings.")

        if st.button("Sync / Rebuild Index from Neon", use_container_width=True):
            with st.spinner("Fetching latest summaries from Neon and embedding..."):
                try:
                    st.cache_resource.clear()
                    setup(rebuild=True)
                    st.success("Index rebuilt from Neon!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to sync with Neon: {e}")

        st.divider()
        st.subheader("Example questions")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True, key=example):
                st.session_state["pending_question"] = example

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources_ui(message["sources"])

    question = st.session_state.pop("pending_question", None)
    if prompt := (question or st.chat_input("Ask about Indian news...")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                try:
                    cat_filter = None if category == "All categories" else category
                    resp = load_rag().query(
                        prompt,
                        category_filter=cat_filter,
                        top_k=int(top_k),
                    )
                    answer = resp["answer"]
                    sources = resp.get("sources", []) if show_sources else []
                    sync = resp.get("sync", {})
                    if sync.get("added", 0) > 0:
                        st.caption(
                            f"Indexed {sync['added']} new article(s) from Neon "
                            f"(total: {sync['count']})"
                        )
                except Exception as e:
                    answer = f"**Error:** {e}"
                    sources = []

            st.markdown(answer)
            if sources:
                render_sources_ui(sources)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
