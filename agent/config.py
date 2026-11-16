"""Settings (.env) + YAML config loading.

Decision: one cached loader for each config file so every module reads the SAME numbers,
and nothing is hard-coded in code (spec §3: "every number in this spec is read from config").
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")


class Settings(BaseModel):
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    embed_model: str = "gemini-embedding-2"
    embed_dim: int = 768
    anthropic_api_key: str = ""
    anthropic_model: str = ""
    price_in_per_m: float = 0.30
    price_out_per_m: float = 2.50
    database_url: str = ""
    test_database_url: str = ""
    hunter_api_key: str = ""
    poc_source: str = "hunter"
    my_email: str = ""
    bot_email: str = ""
    google_credentials_file: str = "credentials.json"
    google_token_file: str = "token.json"
    growth_lead_name: str = "Anu"
    growth_lead_tz: str = "Asia/Kolkata"
    reviewer_id: str = "anu"
    review_mode: str = "off"
    user_agent: str = ""
    chromium_path: str = ""
    alert_channel: str = "email"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    crm_provider: str = "postgres"
    hubspot_token: str = ""


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return None
    # strip inline "# comment" that people copy from .env.example
    v = v.split(" #")[0].strip()
    return v or None


@lru_cache
def settings() -> Settings:
    data: dict[str, Any] = {}
    for field in Settings.model_fields:
        v = _env(field.upper())
        if v is not None:
            data[field] = v
    return Settings(**data)


@lru_cache
def load_yaml(name: str) -> dict:
    """Load config/<name>.yaml. Missing file = loud error, never a silent default."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def icp() -> dict:
    return load_yaml("icp_config")


def page_map() -> dict:
    return load_yaml("page_map")


def sprinto_facts() -> dict:
    return load_yaml("sprinto_facts")


def feedback_cfg() -> dict:
    return load_yaml("feedback_config")


def outreach_cfg() -> dict:
    return load_yaml("outreach_config")


def retry_table() -> dict:
    return load_yaml("retry_table")


def poc_overrides() -> dict:
    with (CONFIG_DIR / "poc_overrides.json").open() as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def require(*names: str) -> None:
    """Fail loudly (with the exact missing names) instead of running half-configured."""
    s = settings()
    missing = [n for n in names if not getattr(s, n.lower(), "")]
    if missing:
        raise RuntimeError(
            "Missing configuration in .env: " + ", ".join(m.upper() for m in missing)
            + ". See .env.example."
        )
