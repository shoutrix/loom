"""
Pydantic-based configuration for Loom.

Reads from environment variables and/or a .env file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


_ENV_FILE = Path(__file__).parent / ".env"


class LLMSettings(BaseSettings):
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    pro_model: str = "gemini-2.5-pro"
    flash_model: str = "gemini-2.0-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 3072
    temperature_pro: float = 0.3
    temperature_flash: float = 0.1
    max_output_tokens_pro: int = 65536
    max_output_tokens_flash: int = 8192
    retry_max_attempts: int = 5
    retry_base_delay: float = 2.0
    retry_max_delay: float = 60.0

    model_config = {"env_prefix": "LOOM_LLM_", "extra": "ignore", "env_file": str(_ENV_FILE), "env_file_encoding": "utf-8"}


class RateLimitSettings(BaseSettings):
    max_calls_per_minute: int = 300
    max_daily_llm_calls: int = 5000
    circuit_breaker_threshold: int = 3
    circuit_breaker_pause_seconds: float = 60.0

    model_config = {"env_prefix": "LOOM_RATE_", "extra": "ignore"}


class GraphSettings(BaseSettings):
    entity_embedding_similarity_threshold: float = 0.85
    fuzzy_match_threshold: float = 0.6
    entropy_min_threshold: float = 1.5
    entropy_min_name_length: int = 6
    community_dirty_size_ratio: float = 2.0
    deep_scan_interval: int = 5
    pruning_interval: int = 50

    model_config = {"env_prefix": "LOOM_GRAPH_", "extra": "ignore"}


class SearchSettings(BaseSettings):
    chunk_top_k: int = 15
    proposition_top_k: int = 20
    entity_top_k: int = 10
    bm25_top_k: int = 15
    proposition_boost: float = 1.3
    rrf_k: int = 60

    model_config = {"env_prefix": "LOOM_SEARCH_", "extra": "ignore"}


class Neo4jSettings(BaseSettings):
    uri: str = ""
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    enabled: bool = False

    model_config = {"env_prefix": "LOOM_NEO4J_", "extra": "ignore"}


class Settings(BaseSettings):
    vault_dir: Path = Path("vault")
    data_dir: Path = Path("data")
    wal_path: Path = Path("data/wal.jsonl")
    snapshot_path: Path = Path("data/snapshot.json")
    semantic_scholar_api_key: str = ""
    active_workspace: str = "default"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)

    model_config = {"env_prefix": "LOOM_", "extra": "ignore", "env_file": str(_ENV_FILE), "env_file_encoding": "utf-8"}

    def ensure_dirs(self) -> None:
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def for_workspace(self, workspace_id: str) -> Settings:
        """Return a copy of settings with paths scoped to a workspace subdirectory."""
        import copy
        ws = copy.copy(self)
        ws_data = self.data_dir / workspace_id
        ws.data_dir = ws_data
        ws.vault_dir = self.vault_dir / workspace_id
        ws.wal_path = ws_data / "wal.jsonl"
        ws.snapshot_path = ws_data / "snapshot.json"
        ws.active_workspace = workspace_id
        ws.ensure_dirs()
        return ws


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
