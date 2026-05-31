"""
SQLite-backed cache for financial data.

Stores serialised DataFrames and dicts with TTL expiry.
Avoids redundant API calls during development and analysis.
"""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import (
    DB_PATH,
    FUNDAMENTAL_DATA_TTL_DAYS,
    MARKET_DATA_TTL_HOURS,
)

logger = logging.getLogger(__name__)


class FinancialCache:
    """
    Key-value cache backed by SQLite.

    Keys follow the pattern: "<source>:<ticker>:<data_type>"
    e.g. "yfinance:AAPL:prices", "av:MSFT:income_statement"

    DataFrames are stored as pickle blobs; plain dicts as JSON.
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS cache (
        key         TEXT PRIMARY KEY,
        value       BLOB NOT NULL,
        value_type  TEXT NOT NULL,   -- 'pickle' or 'json'
        expires_at  TEXT NOT NULL,
        created_at  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at);
    """

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing / expired."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, value_type, expires_at FROM cache WHERE key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None

        value_blob, value_type, expires_at = row
        if datetime.fromisoformat(expires_at) < datetime.utcnow():
            logger.debug("Cache expired for key: %s", key)
            self.delete(key)
            return None

        return self._deserialise(value_blob, value_type)

    def set(
        self,
        key: str,
        value: Any,
        ttl_hours: float | None = None,
        ttl_days: float | None = None,
    ) -> None:
        """Store *value* under *key* with given TTL."""
        if ttl_days is not None:
            ttl_hours = ttl_days * 24
        if ttl_hours is None:
            ttl_hours = MARKET_DATA_TTL_HOURS  # default

        expires_at = (datetime.utcnow() + timedelta(hours=ttl_hours)).isoformat()
        blob, vtype = self._serialise(value)

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache
                   (key, value, value_type, expires_at, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, blob, vtype, expires_at, datetime.utcnow().isoformat()),
            )
        logger.debug("Cached %s (expires %s)", key, expires_at)

    def delete(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def clear_expired(self) -> int:
        """Remove all expired entries. Returns count deleted."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM cache WHERE expires_at < ?",
                (datetime.utcnow().isoformat(),),
            )
            return cur.rowcount

    def clear_all(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM cache")

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE expires_at < ?",
                (datetime.utcnow().isoformat(),),
            ).fetchone()[0]
        return {"total": total, "expired": expired, "valid": total - expired}

    # ── Convenience wrappers with sensible TTLs ────────────────────────────────

    def get_prices(self, ticker: str) -> Any | None:
        return self.get(f"yf:{ticker}:prices")

    def set_prices(self, ticker: str, data: Any) -> None:
        self.set(f"yf:{ticker}:prices", data, ttl_hours=MARKET_DATA_TTL_HOURS)

    def get_fundamentals(self, ticker: str, source: str = "yf") -> Any | None:
        return self.get(f"{source}:{ticker}:fundamentals")

    def set_fundamentals(self, ticker: str, data: Any, source: str = "yf") -> None:
        self.set(
            f"{source}:{ticker}:fundamentals",
            data,
            ttl_days=FUNDAMENTAL_DATA_TTL_DAYS,
        )

    def get_macro(self) -> Any | None:
        return self.get("fred:macro:snapshot")

    def set_macro(self, data: Any) -> None:
        self.set("fred:macro:snapshot", data, ttl_hours=MARKET_DATA_TTL_HOURS)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as conn:
            for stmt in self._CREATE_TABLE.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _serialise(value: Any) -> tuple[bytes, str]:
        try:
            return json.dumps(value).encode(), "json"
        except (TypeError, ValueError):
            return pickle.dumps(value), "pickle"

    @staticmethod
    def _deserialise(blob: bytes, value_type: str) -> Any:
        if value_type == "json":
            return json.loads(blob)
        return pickle.loads(blob)  # noqa: S301
