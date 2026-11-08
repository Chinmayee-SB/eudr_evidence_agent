"""Durable storage for supervisor runs.

LangGraph saves the whole case file after every node, keyed by the case's thread id. Nothing in the
supervisor reads or writes the database itself: the graph does it. So a case paused for the officer
survives a closed browser or a restart, and resumes exactly where it stopped.

Backends, same interface (Postgres holds case state only; documents and traces stay as files):
  SQLite   (default)          one file, runs/checkpoints.sqlite, no server; fine for one machine
  Postgres (DATABASE_URL set) shared and durable, with a connection pool, for several users or app instances
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from eudr_agent.config import RUNS_DIR

SQLITE_PATH = RUNS_DIR / "checkpoints.sqlite"
POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))  # most connections open at once
POOL_TIMEOUT_S = 10  # how long a request waits for a free connection before failing


@contextmanager
def open_checkpointer(sqlite_path: Path | None = None) -> Iterator[Any]:
    """Yield a LangGraph checkpointer; closes its connection on exit."""
    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as e:
            raise RuntimeError('DATABASE_URL is set: install the extras with  pip install -e ".[postgres]"') from e
        # A pool keeps a few connections open and lends one to each request, so several officers can work at
        # once without waiting on a single connection. The three settings are what PostgresSaver requires.
        with ConnectionPool(database_url, min_size=1, max_size=POOL_MAX_SIZE, open=True, timeout=POOL_TIMEOUT_S,
                            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}) as pool:
            saver = PostgresSaver(pool)
            saver.setup()  # creates the tables on first use; safe to repeat
            yield saver
        return

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = sqlite_path or SQLITE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # the desk serves each request on its own thread, so the connection must not be tied to one
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()
