"""Shared Earth Engine setup for the offline-fixture preparation scripts."""

from __future__ import annotations

import os
from pathlib import Path

import ee
from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
API_DEADLINE_MS = 120_000
MAX_RETRIES = 4


def initialize_earth_engine() -> None:
    """Initialize EE with bounded requests and the project from package .env."""

    load_dotenv(PACKAGE_DIR / ".env")
    project = os.environ.get("EE_PROJECT", "").strip()
    if not project:
        raise SystemExit(
            f"EE_PROJECT is missing; set it in {PACKAGE_DIR / '.env'} or the environment"
        )
    ee.Initialize(project=project)
    ee.data.setDeadline(API_DEADLINE_MS)
    ee.data.setMaxRetries(MAX_RETRIES)
