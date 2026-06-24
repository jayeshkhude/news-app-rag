"""
config.py
Load settings from Streamlit secrets (cloud) or .env (local).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def get_config(key: str, default: str = "") -> str:
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return os.getenv(key, default)


def get_config_bool(key: str, default: bool = True) -> bool:
    value = get_config(key, str(default)).lower()
    return value in {"1", "true", "yes", "on"}
