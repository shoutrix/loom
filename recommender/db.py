"""
SQLite schema + connection helpers for recommender-enabled workspaces.

One database per feed workspace at `<workspace_data_dir>/feed.db`. Holds the
profile, items, runs, ratings, and an embedding cache. Schema is created on
first connection (no migration framework yet -- v0).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feed_profile (
  workspace_id TEXT PRIMARY KEY,
  description TEXT,
  seed_topics TEXT,                    -- JSON list[str]
  pos_centroid BLOB,                   -- numpy float32 bytes
  neg_centroid BLOB,
  pos_count INT DEFAULT 0,
  neg_count INT DEFAULT 0,
  ranker_stage INT DEFAULT 0,
  ranker_weights BLOB,                 -- pickled sklearn / numpy
  ranker_calibrator BLOB,
  config TEXT,                         -- JSON
  created_at TEXT,
  last_run_at TEXT
);

CREATE TABLE IF NOT EXISTS feed_item (
  id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  kind TEXT NOT NULL,                  -- 'paper' | 'blog'
  source TEXT,                         -- 'arxiv' | 'semantic_scholar' | 'rss' | 'hn' | ...
  external_id TEXT,
  url TEXT,
  title TEXT,
  authors TEXT,                        -- JSON list[str]
  published_at TEXT,                   -- ISO timestamp
  abstract TEXT,
  features TEXT,                       -- JSON dict[str, float]
  llm_score REAL,
  final_score REAL,
  calibrated_prob REAL,
  status TEXT DEFAULT 'surfaced',      -- 'surfaced' | 'read' | 'saved' | 'skipped' | 'ingested'
  exploration INT DEFAULT 0,           -- 0 = exploitation, 1 = exploration
  fetched_at TEXT,
  run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_feed_item_run ON feed_item(run_id);
CREATE INDEX IF NOT EXISTS idx_feed_item_workspace_status ON feed_item(workspace_id, status);
CREATE INDEX IF NOT EXISTS idx_feed_item_url ON feed_item(url);

CREATE TABLE IF NOT EXISTS feed_run (
  id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  window TEXT,
  window_start TEXT,
  window_end TEXT,
  candidate_count INT,
  surfaced_count INT,
  digest_path TEXT,
  ranker_stage INT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS feed_rating (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id TEXT NOT NULL,
  workspace_id TEXT NOT NULL,
  rating INT NOT NULL,
  note TEXT,
  rated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_feed_rating_workspace ON feed_rating(workspace_id);

CREATE TABLE IF NOT EXISTS embedding_cache (
  item_id TEXT NOT NULL,
  model_name TEXT NOT NULL,
  vector BLOB NOT NULL,
  PRIMARY KEY (item_id, model_name)
);
"""


def db_path_for(workspace_data_dir: Path) -> Path:
    return workspace_data_dir / "feed.db"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.executescript(SCHEMA_SQL)
    return conn


@contextmanager
def session(path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()
