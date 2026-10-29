"""Scoped local memories with directional, code-enforced read permissions.

This module deliberately has no AstrBot dependency. Every operation requires a
trusted Scope from the caller; no identity is inferred from memory contents.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import re
import sqlite3
import threading
import time


def _identity(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(f"A nonempty, stable {label} is required")


@dataclass(frozen=True)
class Scope:
    """Trusted platform identity; group_id=None represents a private chat."""

    platform: str
    bot_id: str
    user_id: str
    group_id: str | None = None

    def __post_init__(self) -> None:
        _identity(self.platform, "platform")
        _identity(self.bot_id, "bot ID")
        _identity(self.user_id, "user ID")
        if self.group_id is not None:
            _identity(self.group_id, "group ID")


@dataclass(frozen=True)
class Memory:
    id: int
    text: str
    kind: str
    group_id: str | None
    created_at: float


def _terms(text: str) -> Counter[str]:
    """A bounded, dependency-free lexical index suitable for short Chinese text."""
    terms: Counter[str] = Counter()
    for part in re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.casefold()):
        if "\u3400" <= part[0] <= "\u9fff":
            terms.update(part)
            terms.update(part[i : i + 2] for i in range(len(part) - 1))
        else:
            terms[part] += 1
    return terms


class MemoryStore:
    """Thread-safe SQLite storage with the same ACL for recall and deletion.

    A DM can read its own user's records in every group. A group can read only
    records from the same user in that same group. Platform and bot namespaces
    never mix. A private record uses the empty internal group key; public Scope
    values disallow that key, so private/group identities cannot collide.

    Zero retention_days disables time expiry; record limits still apply. If a
    platform supplies no message ID, identical texts deduplicate within a
    one-minute timestamp bucket. This fallback is intentionally not permanent.
    """

    MAX_RESULTS = 50
    MAX_CANDIDATES = 1000
    MAX_QUERY_CHARS = 512
    MAX_RECALL_CHARS = 20000

    def __init__(
        self,
        path: str | Path,
        retention_days: float = 180,
        max_records_per_user: int = 1000,
        max_text_chars: int = 1200,
    ) -> None:
        if (
            isinstance(retention_days, bool)
            or not isinstance(retention_days, (int, float))
            or not math.isfinite(retention_days)
            or retention_days < 0
        ):
            raise ValueError("retention_days must be a finite nonnegative number")
        for label, value in (
            ("max_records_per_user", max_records_per_user),
            ("max_text_chars", max_text_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.retention_days = float(retention_days)
        self.max_records_per_user = min(max_records_per_user, 100000)
        self.max_text_chars = min(max_text_chars, 20000)
        self._lock = threading.RLock()
        self._closed = False
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(path), timeout=5, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        with self._connection:
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS directional_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    bot_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('message', 'fact')),
                    created_at REAL NOT NULL,
                    dedupe_key TEXT NOT NULL,
                    UNIQUE(platform, bot_id, user_id, group_id, dedupe_key)
                )"""
            )
            self._connection.execute(
                """CREATE INDEX IF NOT EXISTS idx_directional_owner_scope
                ON directional_memories
                (platform, bot_id, user_id, group_id, created_at DESC, id DESC)"""
            )
            self._connection.execute(
                """CREATE INDEX IF NOT EXISTS idx_directional_expiry
                ON directional_memories(created_at)"""
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemoryStore is closed")

    @staticmethod
    def _validate_scope(scope: Scope) -> None:
        if not isinstance(scope, Scope):
            raise ValueError("An explicit trusted Scope is required")
        # Also validate values when a caller has bypassed frozen dataclass guards.
        scope.__post_init__()

    @staticmethod
    def _timestamp(now: float | None) -> float:
        timestamp = time.time() if now is None else now
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (float, int))
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("A finite nonnegative timestamp is required")
        return float(timestamp)

    @staticmethod
    def _owner(scope: Scope) -> tuple[str, str, str]:
        return scope.platform, scope.bot_id, scope.user_id

    @classmethod
    def _visibility(cls, scope: Scope) -> tuple[str, tuple[str, ...]]:
        clause = "platform=? AND bot_id=? AND user_id=?"
        args = cls._owner(scope)
        if scope.group_id is not None:
            clause += " AND group_id=?"
            args += (scope.group_id,)
        return clause, args

    def add(
        self,
        scope: Scope,
        text: str,
        kind: str = "message",
        message_id: str = "",
        now: float | None = None,
    ) -> bool:
        self._validate_scope(scope)
        if not isinstance(text, str):
            raise ValueError("Memory text must be a string")
        if kind not in ("message", "fact"):
            raise ValueError("Memory kind must be message or fact")
        if not isinstance(message_id, str) or len(message_id) > 1024:
            raise ValueError("message_id must be a string of at most 1024 characters")
        timestamp = self._timestamp(now)
        text = text.strip()[: self.max_text_chars]
        if not text:
            return False
        if message_id:
            fingerprint = "event:" + hashlib.sha256(message_id.encode()).hexdigest()
        else:
            raw = f"{kind}\0{int(timestamp // 60)}\0{text}"
            fingerprint = "fallback:" + hashlib.sha256(raw.encode()).hexdigest()
        with self._lock:
            self._ensure_open()
            with self._connection:
                if self.retention_days:
                    self._connection.execute(
                        "DELETE FROM directional_memories WHERE created_at < ?",
                        (timestamp - self.retention_days * 86400,),
                    )
                cursor = self._connection.execute(
                    """INSERT OR IGNORE INTO directional_memories
                    (platform, bot_id, user_id, group_id, text, kind, created_at, dedupe_key)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (*self._owner(scope), scope.group_id or "", text, kind,
                     timestamp, fingerprint),
                )
                inserted = cursor.rowcount == 1
                self._connection.execute(
                    """DELETE FROM directional_memories
                    WHERE platform=? AND bot_id=? AND user_id=? AND id NOT IN (
                        SELECT id FROM directional_memories
                        WHERE platform=? AND bot_id=? AND user_id=?
                        ORDER BY created_at DESC, id DESC LIMIT ?
                    )""",
                    (*self._owner(scope), *self._owner(scope),
                     self.max_records_per_user),
                )
                return inserted

    def search(
        self,
        scope: Scope,
        query: str = "",
        limit: int = 5,
        max_chars: int = 1200,
        now: float | None = None,
    ) -> list[Memory]:
        self._validate_scope(scope)
        if not isinstance(query, str):
            raise ValueError("Recall query must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("Recall limit must be an integer")
        if isinstance(max_chars, bool) or not isinstance(max_chars, int):
            raise ValueError("Recall character budget must be an integer")
        timestamp = self._timestamp(now)
        limit = max(0, min(limit, self.MAX_RESULTS))
        max_chars = max(0, min(max_chars, self.MAX_RECALL_CHARS))
        if not limit or not max_chars:
            return []
        where, args = self._visibility(scope)
        if self.retention_days:
            where += " AND created_at >= ?"
            args += (timestamp - self.retention_days * 86400,)
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT id, text, kind, group_id, created_at "
                f"FROM directional_memories WHERE {where} "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (*args, min(self.max_records_per_user, self.MAX_CANDIDATES)),
            ).fetchall()
        query_terms = _terms(query[: self.MAX_QUERY_CHARS])

        def relevance(row: sqlite3.Row) -> tuple[bool, float, bool, float, int]:
            overlap = query_terms.keys() & _terms(row["text"]).keys()
            score = sum(2.0 if len(term) > 1 else 0.25 for term in overlap)
            return bool(overlap), score, row["kind"] == "fact", row["created_at"], row["id"]

        rows.sort(key=relevance, reverse=True)
        result = []
        remaining = max_chars
        for row in rows[:limit]:
            memory = Memory(row["id"], row["text"], row["kind"],
                            row["group_id"] or None, row["created_at"])
            text = memory.text[:remaining]
            if not text:
                break
            result.append(replace(memory, text=text))
            remaining -= len(text)
            if remaining <= 0:
                break
        return result

    def clear_visible(self, scope: Scope) -> int:
        """Delete only what this same caller is permitted to read."""
        self._validate_scope(scope)
        where, args = self._visibility(scope)
        with self._lock:
            self._ensure_open()
            with self._connection:
                cursor = self._connection.execute(
                    f"DELETE FROM directional_memories WHERE {where}", args
                )
                return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True
