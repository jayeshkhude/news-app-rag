"""
neon_db.py
Connect to Neon PostgreSQL for the summaries table.
"""

import os
from functools import lru_cache
from typing import Optional

import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()

NEON_API_KEY = os.getenv("NEON_API_KEY", "")
NEON_PROJECT_ID = os.getenv("NEON_PROJECT_ID", "royal-shadow-76392601")
NEON_DATABASE = os.getenv("NEON_DATABASE", "neondb")
NEON_ROLE = os.getenv("NEON_ROLE", "neondb_owner")
DATABASE_URL = os.getenv("DATABASE_URL", "")


def neon_configured() -> bool:
    return bool(DATABASE_URL or NEON_API_KEY)


@lru_cache(maxsize=1)
def _resolve_database_url() -> str:
    if DATABASE_URL:
        return DATABASE_URL

    if not NEON_API_KEY:
        raise ValueError(
            "Set DATABASE_URL or NEON_API_KEY in .env to load data from Neon."
        )

    project_id = NEON_PROJECT_ID or _discover_project_id()
    branch_id = _get_primary_branch_id(project_id)
    return _fetch_connection_uri(project_id, branch_id)


def _neon_headers() -> dict:
    return {
        "Authorization": f"Bearer {NEON_API_KEY}",
        "Accept": "application/json",
    }


def _discover_project_id() -> str:
    response = requests.get(
        "https://console.neon.tech/api/v2/projects",
        headers=_neon_headers(),
        timeout=30,
    )
    response.raise_for_status()
    projects = response.json().get("projects", [])
    if not projects:
        raise ValueError("No Neon projects found for this API key.")
    if len(projects) == 1:
        return projects[0]["id"]
    raise ValueError(
        "Multiple Neon projects found. Set NEON_PROJECT_ID in .env."
    )


def _get_primary_branch_id(project_id: str) -> str:
    response = requests.get(
        f"https://console.neon.tech/api/v2/projects/{project_id}/branches",
        headers=_neon_headers(),
        timeout=30,
    )
    response.raise_for_status()
    branches = response.json().get("branches", [])
    primary = next((b for b in branches if b.get("primary")), None)
    if not primary:
        raise ValueError(f"No primary branch found for project {project_id}.")
    return primary["id"]


def _fetch_connection_uri(project_id: str, branch_id: str) -> str:
    response = requests.get(
        f"https://console.neon.tech/api/v2/projects/{project_id}/connection_uri"
        f"?branch_id={branch_id}&database_name={NEON_DATABASE}&role_name={NEON_ROLE}",
        headers=_neon_headers(),
        timeout=30,
    )
    response.raise_for_status()
    uri = response.json().get("uri")
    if not uri:
        raise ValueError("Neon API did not return a database connection URI.")
    return uri


def get_neon_connection():
    return psycopg2.connect(_resolve_database_url())
