"""
chatbot.py
Interactive CLI chatbot for querying Indian news summaries via RAG.
Run: python src/chatbot.py
"""

import sys
import os
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset, build_documents, get_categories
from vectorstore import VectorStore
from rag_pipeline import RAGPipeline, print_response


BANNER = f"""
{Fore.GREEN}==========================================================
|       Indian News RAG - Live Neon summaries + FAISS       |
=========================================================={Style.RESET_ALL}

Commands:
  {Fore.CYAN}/category <name>{Style.RESET_ALL}   filter by category (politics, sports, world, economy, other, science_tech, society)
  {Fore.CYAN}/clear{Style.RESET_ALL}             remove active category filter
  {Fore.CYAN}/topk <n>{Style.RESET_ALL}          set number of retrieved articles (default: 5)
  {Fore.CYAN}/sync{Style.RESET_ALL}              fetch new summaries from Neon
  {Fore.CYAN}/quit{Style.RESET_ALL}              exit

Just type your question and press Enter.
"""


def setup(rebuild: bool = False) -> RAGPipeline:
    """Load vector store and sync with Neon (incremental when possible)."""
    from sync import ensure_index_synced

    vs = VectorStore()
    if not rebuild:
        vs.load()

    result = ensure_index_synced(vs, force_rebuild=rebuild)

    if result["status"] == "synced":
        print(
            f"{Fore.GREEN}Index in sync with Neon "
            f"({result['count']} articles, latest: {result['latest_date']})"
            f"{Style.RESET_ALL}"
        )
    elif result["status"] == "updated":
        print(
            f"{Fore.GREEN}Added {result['added']} new articles from Neon "
            f"(now {result['count']} total, latest: {result['latest_date']})"
            f"{Style.RESET_ALL}"
        )
    else:
        print(
            f"{Fore.GREEN}Index rebuilt from Neon "
            f"({result['count']} articles, latest: {result['latest_date']})"
            f"{Style.RESET_ALL}"
        )

    return RAGPipeline(vs, top_k=5)


def run_chatbot():
    print(BANNER)
    rag = setup()

    category_filter = None
    show_sources = True
    top_k = 5
    categories = ["politics", "sports", "world", "economy", "other", "science_tech", "society"]

    while True:
        try:
            # Prompt line
            filter_label = f" [{Fore.CYAN}{category_filter}{Style.RESET_ALL}]" if category_filter else ""
            raw = input(f"\n{Fore.GREEN}You{filter_label}{Style.RESET_ALL} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
            break

        if not raw:
            continue

        # Handle commands
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "/quit":
                print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break

            elif cmd == "/category":
                if len(parts) < 2:
                    print(f"Available: {', '.join(categories)}")
                elif parts[1].lower() in categories:
                    category_filter = parts[1].lower()
                    print(f"{Fore.GREEN}Filter set to: {category_filter}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Unknown category. Choose from: {', '.join(categories)}{Style.RESET_ALL}")

            elif cmd == "/clear":
                category_filter = None
                print(f"{Fore.GREEN}Category filter cleared.{Style.RESET_ALL}")

            elif cmd == "/topk":
                try:
                    top_k = int(parts[1])
                    print(f"{Fore.GREEN}Top-k set to {top_k}{Style.RESET_ALL}")
                except (IndexError, ValueError):
                    print("Usage: /topk <number>")

            elif cmd == "/sync":
                print(f"{Fore.YELLOW}Checking Neon for new summaries...{Style.RESET_ALL}")
                from sync import ensure_index_synced
                result = ensure_index_synced(rag.vs)
                if result["added"]:
                    print(f"{Fore.GREEN}Added {result['added']} new articles (total: {result['count']}){Style.RESET_ALL}")
                else:
                    print(f"{Fore.GREEN}Already up to date ({result['count']} articles){Style.RESET_ALL}")

            elif cmd == "/sources":
                show_sources = not show_sources
                print(f"{Fore.GREEN}Sources: {'ON' if show_sources else 'OFF'}{Style.RESET_ALL}")

            else:
                print(f"{Fore.RED}Unknown command: {cmd}{Style.RESET_ALL}")
            continue

        # Normal query
        print(f"\n{Fore.YELLOW}Searching...{Style.RESET_ALL}")
        try:
            resp = rag.query(
                raw,
                category_filter=category_filter,
                top_k=top_k,
                verbose=False,
            )
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            continue

        # Print answer
        print(f"\n{Fore.BLUE}Assistant:{Style.RESET_ALL}")
        print(resp["answer"])

        # Print sources
        if show_sources and resp["sources"]:
            print(f"\n{Fore.CYAN}Sources used:{Style.RESET_ALL}")
            for src in resp["sources"]:
                print(f"  [{src['rank']}] ({src['category']}) {src['headline'][:65]}")

        if "usage" in resp:
            u = resp["usage"]
            print(f"\n{Fore.CYAN}Tokens:{Style.RESET_ALL} {u['input_tokens']} in / {u['output_tokens']} out")


if __name__ == "__main__":
    run_chatbot()
