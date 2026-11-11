"""Settings read from the environment (.env), with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "corpus" / "raw"
MANUAL_DIR = DATA_DIR / "corpus" / "manual"
NOTES_DIR = DATA_DIR / "corpus" / "notes"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
VECTOR_DIR = INDEX_DIR / "chroma"
PLOTS_DIR = DATA_DIR / "plots"
GEO_CACHE_DIR = DATA_DIR / "cache" / "geo"
RUNS_DIR = ROOT / "runs"


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "none")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "60"))   # per attempt; the SDK default is 600 s
    llm_retries: int = int(os.getenv("LLM_RETRIES", "2"))
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    max_retries: int = int(os.getenv("RAG_MAX_RETRIES", "2"))
    retrieval: str = os.getenv("RAG_RETRIEVAL", "hybrid")  # hybrid, bm25 or vector
    ee_project: str = os.getenv("EE_PROJECT", "")
    ee_timeout_s: float = float(os.getenv("EE_TIMEOUT_S", "120"))   # per Earth Engine request

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)


settings = Settings()
