from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np


class MemoryStore:
    """SQLite-backed internal memory with a human-readable JSON diary mirror."""

    _ACTIVE = "active"

    def __init__(self, legacy_path: Path, *, diary_limit: int = 360) -> None:
        self.legacy_path = Path(legacy_path)
        self.db_path = self.legacy_path.with_suffix(".sqlite3")
        self.diary_limit = max(20, min(2000, int(diary_limit)))
        self._write_lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._migrate_legacy_json()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id INTEGER REFERENCES memory_items(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL CHECK(kind IN ('diary', 'fact')),
                    scope TEXT NOT NULL DEFAULT 'user',
                    slot TEXT NOT NULL DEFAULT 'diary',
                    summary TEXT NOT NULL,
                    subject TEXT NOT NULL DEFAULT '',
                    predicate TEXT NOT NULL DEFAULT '',
                    object TEXT NOT NULL DEFAULT '',
                    policy TEXT NOT NULL DEFAULT 'append',
                    status TEXT NOT NULL DEFAULT 'active',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.45,
                    freshness TEXT NOT NULL DEFAULT 'unknown',
                    source_type TEXT NOT NULL DEFAULT 'unknown',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    topics_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    last_verified_at TEXT NOT NULL DEFAULT '',
                    expires_at TEXT NOT NULL DEFAULT '',
                    last_accessed_at TEXT NOT NULL DEFAULT '',
                    recall_count INTEGER NOT NULL DEFAULT 0,
                    superseded_by INTEGER REFERENCES memory_items(id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_active
                    ON memory_items(status, kind, slot, created_at);
                CREATE INDEX IF NOT EXISTS idx_memory_relation
                    ON memory_items(slot, subject, predicate, status);
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id INTEGER NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
                    model_key TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY(memory_id, model_key)
                );
                CREATE TABLE IF NOT EXISTS memory_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_sources (
                    id INTEGER PRIMARY KEY,
                    transcript TEXT NOT NULL DEFAULT '',
                    scope TEXT NOT NULL DEFAULT 'user',
                    archive_key TEXT NOT NULL DEFAULT '',
                    archive_revision INTEGER NOT NULL DEFAULT 0,
                    preferences_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS memory_archive_jobs (
                    key TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    transcript TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    cancelled INTEGER NOT NULL DEFAULT 0,
                    committed_id INTEGER
                );
                CREATE TABLE IF NOT EXISTS memory_source_links (
                    memory_id INTEGER NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
                    source_id INTEGER NOT NULL REFERENCES memory_sources(id) ON DELETE CASCADE,
                    PRIMARY KEY(memory_id, source_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_source ON memory_source_links(source_id);
                INSERT OR IGNORE INTO memory_sources(id)
                    SELECT id FROM memory_items WHERE kind='diary';
                INSERT OR IGNORE INTO memory_source_links(memory_id, source_id)
                    SELECT id, id FROM memory_items WHERE kind='diary';
                INSERT OR IGNORE INTO memory_source_links(memory_id, source_id)
                    SELECT id, parent_id FROM memory_items WHERE kind='fact' AND parent_id IS NOT NULL;
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(memory_items)")}
            if "scope" not in columns:
                # Old summaries do not reliably identify a work. Preserve them as legacy.
                connection.execute("ALTER TABLE memory_items ADD COLUMN scope TEXT NOT NULL DEFAULT 'legacy'")
                connection.execute("UPDATE memory_items SET scope='user' WHERE slot='user_profile' OR source_type='manual'")
            source_columns = {row[1] for row in connection.execute("PRAGMA table_info(memory_sources)")}
            for name, definition in (("scope", "TEXT NOT NULL DEFAULT 'legacy'"),
                                     ("archive_key", "TEXT NOT NULL DEFAULT ''"),
                                     ("archive_revision", "INTEGER NOT NULL DEFAULT 0"),
                                     ("preferences_json", "TEXT NOT NULL DEFAULT '[]'")):
                if name not in source_columns:
                    connection.execute(f"ALTER TABLE memory_sources ADD COLUMN {name} {definition}")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_items(scope, status, kind)")
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
                    "USING fts5(summary, topics, tokenize='unicode61')"
                )
            except sqlite3.OperationalError:
                pass

    @staticmethod
    def _loads_list(value: object) -> list[object]:
        if isinstance(value, list):
            return value
        try:
            parsed = json.loads(str(value or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    def _clamp(value: object, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clean(value: object, limit: int = 500) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @classmethod
    def _row_to_entry(cls, row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": int(row["id"]),
            "scope": str(row["scope"]),
            "schema_version": 3,
            "timestamp": str(row["created_at"]),
            "created_at": str(row["created_at"]),
            "last_verified_at": str(row["last_verified_at"]),
            "summary": str(row["summary"]),
            "source_type": str(row["source_type"]),
            "sources": cls._loads_list(row["sources_json"]),
            "confidence": float(row["confidence"]),
            "freshness": str(row["freshness"]),
            "expires_at": str(row["expires_at"]),
            "topics": cls._loads_list(row["topics_json"]),
            "kind": str(row["kind"]),
            "slot": str(row["slot"]),
            "subject": str(row["subject"]),
            "predicate": str(row["predicate"]),
            "object": str(row["object"]),
            "policy": str(row["policy"]),
            "status": str(row["status"]),
            "pinned": bool(row["pinned"]),
            "importance": float(row["importance"]),
            "recall_count": int(row["recall_count"]),
        }

    def _meta(self, connection: sqlite3.Connection, key: str) -> str:
        row = connection.execute("SELECT value FROM memory_meta WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else ""

    def _set_meta(self, connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            "INSERT INTO memory_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _migrate_legacy_json(self) -> None:
        with self._connect() as connection:
            if self._meta(connection, "legacy_json_migrated"):
                return
            count = int(connection.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0])
            raw: object = []
            if count == 0 and self.legacy_path.is_file():
                try:
                    raw = json.loads(self.legacy_path.read_text(encoding="utf-8"))
                except Exception:
                    raw = []
            if isinstance(raw, list):
                for entry in raw[-self.diary_limit :]:
                    if isinstance(entry, dict) and self._clean(entry.get("summary")):
                        self._insert_diary(connection, {**entry, "scope": entry.get("scope") or (
                            "user" if entry.get("source_type") == "manual" else "legacy")})
            self._set_meta(connection, "legacy_json_migrated", self._now_iso())

    def _insert_fts(self, connection: sqlite3.Connection, item_id: int, summary: str, topics: Iterable[object]) -> None:
        try:
            connection.execute(
                "INSERT INTO memory_fts(rowid, summary, topics) VALUES(?, ?, ?)",
                (item_id, summary, " ".join(str(topic) for topic in topics)),
            )
        except sqlite3.OperationalError:
            pass

    def _insert_diary(self, connection: sqlite3.Connection, entry: dict[str, object]) -> int:
        summary = self._clean(entry.get("summary"))
        topics = self._loads_list(entry.get("topics"))[:8]
        sources = self._loads_list(entry.get("sources"))[:8]
        created_at = self._clean(entry.get("created_at") or entry.get("timestamp"), 80) or self._now_iso()
        cursor = connection.execute(
            """
            INSERT INTO memory_items(
                kind, slot, summary, policy, pinned, importance, confidence,
                freshness, source_type, sources_json, topics_json, created_at,
                last_verified_at, expires_at, scope
            ) VALUES('diary', 'diary', ?, 'append', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary,
                self._clamp(entry.get("importance"), 0.55),
                self._clamp(entry.get("confidence"), 0.45),
                self._clean(entry.get("freshness") or "unknown", 16),
                self._clean(entry.get("source_type") or "unknown", 32),
                json.dumps(sources, ensure_ascii=False),
                json.dumps(topics, ensure_ascii=False),
                created_at,
                self._clean(entry.get("last_verified_at"), 80),
                self._clean(entry.get("expires_at"), 80),
                str(entry.get("scope") or "user"),
            ),
        )
        item_id = int(cursor.lastrowid)
        connection.execute("INSERT INTO memory_sources(id, scope) VALUES(?, ?)",
                           (item_id, str(entry.get("scope") or "user")))
        connection.execute("INSERT INTO memory_source_links VALUES(?, ?)", (item_id, item_id))
        self._insert_fts(connection, item_id, summary, topics)
        return item_id

    @classmethod
    def _normalize_fact(cls, fact: object, diary: dict[str, object]) -> dict[str, object] | None:
        if not isinstance(fact, dict):
            return None
        subject = cls._clean(fact.get("subject"), 80)
        predicate = cls._clean(fact.get("predicate"), 80)
        object_text = cls._clean(fact.get("object"), 240)
        if not subject or not predicate or not object_text:
            return None
        slot = cls._clean(fact.get("slot") or "knowledge", 40).lower()
        allowed_slots = {"user_profile", "relationship", "current_state", "knowledge"}
        if slot not in allowed_slots:
            slot = "knowledge"
        if slot == "user_profile" and subject.casefold() in {"user", "用户", "哥哥", "主人"}:
            subject = "用户"
        pinned = bool(fact.get("pinned", False))
        policy = cls._clean(fact.get("policy"), 16).lower()
        if pinned:
            policy = "pinned"
        elif policy not in {"replace", "append"}:
            policy = "replace" if slot == "current_state" else "append"
        freshness = cls._clean(fact.get("freshness") or "unknown", 16).lower()
        if freshness not in {"stable", "volatile", "unknown"}:
            freshness = "unknown"
        expires_at = cls._clean(fact.get("expires_at"), 80)
        if freshness == "volatile" and not expires_at:
            expires_at = (datetime.now().astimezone() + timedelta(days=7)).isoformat(timespec="seconds")
        return {
            "slot": slot,
            "subject": subject,
            "predicate": predicate,
            "object": object_text,
            "summary": f"{subject}{predicate}{object_text}",
            "policy": policy,
            "pinned": pinned,
            "importance": cls._clamp(fact.get("importance"), 0.65),
            "confidence": cls._clamp(fact.get("confidence"), cls._clamp(diary.get("confidence"), 0.45)),
            "freshness": freshness,
            "expires_at": expires_at,
        }

    def _insert_fact(
        self,
        connection: sqlite3.Connection,
        parent_id: int,
        fact: dict[str, object],
        diary: dict[str, object],
    ) -> int | None:
        normalized = self._normalize_fact(fact, diary)
        if normalized is None:
            return None
        now = self._now_iso()
        scope = str(diary.get("scope") or "user")
        cte, scope_params = self._recall_cte((scope,), now)
        duplicate = connection.execute(
            cte + """
            SELECT id FROM recallable
            WHERE kind='fact' AND status='active' AND slot=? AND subject=?
              AND predicate=? AND object=? AND (expires_at='' OR expires_at>?)
            LIMIT 1
            """,
            (
                *scope_params,
                normalized["slot"],
                normalized["subject"],
                normalized["predicate"],
                normalized["object"],
                now,
            ),
        ).fetchone()
        if duplicate:
            return int(duplicate[0])

        status = self._ACTIVE
        existing = connection.execute(
            cte + """
            SELECT id, pinned FROM recallable
            WHERE kind='fact' AND status='active' AND slot=? AND subject=? AND predicate=?
              AND (expires_at='' OR expires_at>?)
            ORDER BY id DESC
            """,
            (*scope_params, normalized["slot"], normalized["subject"], normalized["predicate"], now),
        ).fetchall()
        if normalized["policy"] == "replace":
            if any(bool(row["pinned"]) for row in existing) and not normalized["pinned"]:
                status = "conflicted"

        cursor = connection.execute(
            """
            INSERT INTO memory_items(
                parent_id, kind, slot, summary, subject, predicate, object,
                policy, status, pinned, importance, confidence, freshness,
                source_type, sources_json, topics_json, created_at,
                last_verified_at, expires_at, scope
            ) VALUES(?, 'fact', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parent_id,
                normalized["slot"],
                normalized["summary"],
                normalized["subject"],
                normalized["predicate"],
                normalized["object"],
                normalized["policy"],
                status,
                1 if normalized["pinned"] else 0,
                normalized["importance"],
                normalized["confidence"],
                normalized["freshness"],
                self._clean(diary.get("source_type") or "unknown", 32),
                json.dumps(self._loads_list(diary.get("sources"))[:8], ensure_ascii=False),
                json.dumps(self._loads_list(diary.get("topics"))[:8], ensure_ascii=False),
                self._clean(diary.get("created_at") or diary.get("timestamp"), 80) or self._now_iso(),
                self._clean(diary.get("last_verified_at"), 80),
                normalized["expires_at"],
                scope,
            ),
        )
        item_id = int(cursor.lastrowid)
        self._insert_fts(connection, item_id, str(normalized["summary"]), self._loads_list(diary.get("topics")))
        if status == self._ACTIVE and normalized["policy"] in {"replace", "pinned"}:
            replace_ids = [int(row["id"]) for row in existing if not bool(row["pinned"]) or normalized["pinned"]]
            if replace_ids:
                placeholders = ",".join("?" for _ in replace_ids)
                connection.execute(
                    f"UPDATE memory_items SET status='superseded', superseded_by=? WHERE id IN ({placeholders})",
                    (item_id, *replace_ids),
                )
        return item_id

    def prepare_archive(self, transcript: str, scope: str, *, source_key: str = "") -> dict:
        """Persist the immutable input version before starting background inference."""
        key = source_key or self._text_hash(scope + "\0" + transcript)
        with self._write_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM memory_archive_jobs WHERE key=?", (key,)).fetchone()
            if row is None:
                connection.execute("INSERT INTO memory_archive_jobs(key, scope, transcript) VALUES(?, ?, ?)",
                                   (key, scope, transcript))
            elif row["transcript"] != transcript or row["scope"] != scope:
                connection.execute(
                    "UPDATE memory_archive_jobs SET scope=?, transcript=?, revision=revision+1, "
                    "committed_id=NULL, cancelled=0 WHERE key=?", (scope, transcript, key))
            return dict(connection.execute("SELECT * FROM memory_archive_jobs WHERE key=?", (key,)).fetchone())

    def invalidate_archive(self, source_key: str) -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute("UPDATE memory_archive_jobs SET revision=revision+1, cancelled=1, "
                               "committed_id=NULL WHERE key=?", (source_key,))

    def archive_is_current(self, source_key: str, revision: int = 1) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT revision, cancelled FROM memory_archive_jobs WHERE key=?", (source_key,)).fetchone()
            # A live session has a source key before its first archive is prepared.
            return row is None or (row["revision"] == revision and not row["cancelled"])

    def add_diary(
        self, entry: dict[str, object], facts: Iterable[object] = (), *, source_transcript: str = "",
        archive: dict | None = None,
    ) -> int | None:
        with self._write_lock:
            with self._connect() as connection:
                # The version check and every derived write share one SQLite write transaction.
                connection.execute("BEGIN IMMEDIATE")
                if archive is not None:
                    job = connection.execute("SELECT * FROM memory_archive_jobs WHERE key=?", (archive["key"],)).fetchone()
                    if (job is None or job["cancelled"] or job["revision"] != archive["revision"]
                            or job["transcript"] != source_transcript
                            or job["scope"] != str(entry.get("scope") or "user")):
                        return None
                    if job["committed_id"] is not None:
                        return int(job["committed_id"])
                parent_id = self._insert_diary(connection, entry)
                connection.execute(
                    "UPDATE memory_sources SET transcript=?, archive_key=?, archive_revision=?, preferences_json=? WHERE id=?",
                    (source_transcript, archive["key"] if archive else "", archive["revision"] if archive else 0,
                     json.dumps(entry.get("personalization") or [], ensure_ascii=False), parent_id)
                )
                for fact in list(facts)[:12]:
                    fact_id = self._insert_fact(connection, parent_id, fact, entry)
                    if fact_id is not None:
                        # A deduplicated fact can also invalidate this newer summary later.
                        connection.execute(
                            "INSERT OR IGNORE INTO memory_source_links VALUES(?, ?)", (fact_id, parent_id)
                        )
                self._trim(connection)
                if archive is not None:
                    connection.execute("UPDATE memory_archive_jobs SET committed_id=? WHERE key=?",
                                       (parent_id, archive["key"]))
            self._write_legacy_mirror()
            return parent_id

    def _trim(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT id FROM memory_items WHERE kind='diary' AND source_type!='screen_plot' ORDER BY id DESC LIMIT -1 OFFSET ?",
            (self.diary_limit,),
        ).fetchall()
        if rows:
            # Keep fact lifecycles and their sources even when the diary UI rolls over.
            connection.executemany(
                "UPDATE memory_items SET parent_id=NULL WHERE parent_id=? AND kind='fact'",
                [(int(row[0]),) for row in rows],
            )
            connection.executemany("DELETE FROM memory_items WHERE id = ?", [(int(row[0]),) for row in rows])
            connection.execute(
                "DELETE FROM memory_sources WHERE preferences_json='[]' "
                "AND id NOT IN (SELECT source_id FROM memory_source_links)"
            )

    def list_diaries(self, limit: int = 0) -> list[dict[str, object]]:
        # Legacy plot mirrors stay in SQLite, but are not interaction diaries.
        query = "SELECT * FROM memory_items WHERE kind='diary' AND source_type!='screen_plot' ORDER BY id"
        params: tuple[object, ...] = ()
        if limit > 0:
            query = "SELECT * FROM (SELECT * FROM memory_items WHERE kind='diary' AND source_type!='screen_plot' ORDER BY id DESC LIMIT ?) ORDER BY id"
            params = (int(limit),)
        with self._connect() as connection:
            return [self._row_to_entry(row) for row in connection.execute(query, params).fetchall()]

    def _write_legacy_mirror(self) -> None:
        entries = self.list_diaries()
        payload = [
            {key: value for key, value in entry.items() if key in {
                "schema_version", "timestamp", "created_at", "last_verified_at", "summary", "scope",
                "source_type", "sources", "confidence", "freshness", "expires_at", "topics",
            }}
            for entry in entries
        ]
        self.legacy_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.legacy_path.with_suffix(self.legacy_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.legacy_path)

    @staticmethod
    def _fts_terms(query: str) -> list[str]:
        raw = str(query or "").lower()
        terms = re.findall(r"[a-z0-9][a-z0-9._+-]{1,}", raw)
        for segment in re.findall(r"[\u4e00-\u9fff]+", raw):
            if len(segment) <= 4:
                terms.append(segment)
            else:
                terms.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        return list(dict.fromkeys(term for term in terms if term))[:16]

    @staticmethod
    def _recall_cte(scopes: Iterable[str], now: str) -> tuple[str, tuple]:
        scopes = tuple(dict.fromkeys(scopes))
        placeholders = ",".join("?" for _ in scopes) or "NULL"
        # Scope and lifecycle checks precede candidate LIMIT, FTS ranking and embeddings.
        return """
            WITH current_sources AS (
                SELECT s.* FROM memory_sources s LEFT JOIN memory_archive_jobs j ON j.key=s.archive_key
                WHERE s.archive_key='' OR (j.revision=s.archive_revision AND j.cancelled=0)
            ), valid AS (
                SELECT m.* FROM memory_items m WHERE m.status='active'
                AND (m.expires_at='' OR m.expires_at>?)
                AND (NOT EXISTS (SELECT 1 FROM memory_source_links l WHERE l.memory_id=m.id)
                     OR EXISTS (SELECT 1 FROM memory_source_links l JOIN current_sources s ON s.id=l.source_id
                                WHERE l.memory_id=m.id))
            ), blocked_sources AS (
                SELECT DISTINCT l.source_id FROM memory_source_links l JOIN memory_items f ON f.id=l.memory_id
                WHERE f.kind='fact' AND f.id NOT IN (SELECT id FROM valid)
            ), recallable AS (
                SELECT m.* FROM valid m WHERE m.scope IN (""" + placeholders + """)
                AND (m.kind!='diary' OR NOT EXISTS (
                    SELECT 1 FROM memory_source_links l JOIN blocked_sources b ON b.source_id=l.source_id
                    WHERE l.memory_id=m.id))
            )
        """, (now, *scopes)

    def personalizations(self, *, scopes: Iterable[str] = ("user",)) -> list[dict]:
        scopes = tuple(dict.fromkeys(scopes))
        if not scopes:
            return []
        cte, params = self._recall_cte(scopes, self._now_iso())
        with self._connect() as connection:
            rows = connection.execute(cte + "SELECT scope, preferences_json FROM current_sources ORDER BY id", params)
            return [item for row in rows for item in self._loads_list(row["preferences_json"])
                    if isinstance(item, dict) and item.get("scope", row["scope"]) in scopes]

    def read_evidence(self, memory_id: int, *, scopes: Iterable[str] = ("user",)) -> dict[str, object]:
        """Read retained source text without reviving superseded or expired facts."""
        now = self._now_iso()
        scopes = tuple(scopes)
        cte, params = self._recall_cte(scopes, now)
        with self._connect() as connection:
            row = connection.execute(
                cte + "SELECT * FROM recallable WHERE id=? AND source_type!='screen_plot'",
                (*params, memory_id),
            ).fetchone()
            if row is None:
                # Do not disclose out-of-scope IDs, even through error status.
                limited = connection.execute(cte + "SELECT id FROM valid WHERE id=? AND kind='diary' AND scope IN ("
                                             + (",".join("?" for _ in scopes) or "NULL") + ")",
                                             (*params, memory_id, *scopes)).fetchone()
                if limited:
                    return {"status": "lifecycle_limited", "sources": []}
                return {"status": "not_found", "sources": []}
            entry = self._row_to_entry(row)
            sources = connection.execute(
                cte + "SELECT s.id, s.transcript, s.archive_revision, "
                "s.id IN (SELECT source_id FROM blocked_sources) AS restricted FROM current_sources s "
                "JOIN memory_source_links l ON l.source_id=s.id WHERE l.memory_id=? AND s.scope IN ("
                + (",".join("?" for _ in scopes) or "NULL") + ") ORDER BY s.id",
                (*params, memory_id, *scopes),
            ).fetchall()
            visible = [{"source_id": int(source["id"]), "revision": source["archive_revision"], "transcript": source["transcript"]}
                       for source in sources if not source["restricted"] and source["transcript"]]
            return {"status": "success", "memory": entry, "sources": visible,
                    "source_status": "ok" if visible else (
                        "lifecycle_limited" if any(s["restricted"] for s in sources) else "unavailable")}

    def candidates(self, query: str, *, limit: int | None = 80,
                   scopes: Iterable[str] = ("user",)) -> list[dict[str, object]]:
        now = self._now_iso()
        cte, params = self._recall_cte(scopes, now)
        with self._connect() as connection:
            rows = connection.execute(
                cte + """
                SELECT * FROM recallable WHERE source_type!='screen_plot'
                ORDER BY pinned DESC, id DESC
                LIMIT ?
                """,
                (*params, -1 if limit is None else max(20, int(limit))),
            ).fetchall()
            by_id = {int(row["id"]): self._row_to_entry(row) for row in rows}
            terms = self._fts_terms(query)
            if terms:
                like_terms = terms[:8]
                where = " OR ".join("summary LIKE ? OR topics_json LIKE ?" for _ in like_terms)
                like_params: list[str] = []
                for term in like_terms:
                    like_params.extend((f"%{term}%", f"%{term}%"))
                for row in connection.execute(
                    cte + "SELECT * FROM recallable WHERE source_type!='screen_plot' AND (" + where + ") LIMIT 60",
                    (*params, *like_params),
                ).fetchall():
                    entry = self._row_to_entry(row)
                    entry["fts_score"] = max(float(entry.get("fts_score") or 0.0), 0.65)
                    by_id[int(row["id"])] = entry
                try:
                    match = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
                    fts_rows = connection.execute(
                        cte + "SELECT m.* FROM memory_fts JOIN recallable m ON m.id=memory_fts.rowid "
                        "WHERE memory_fts MATCH ? AND m.source_type!='screen_plot' ORDER BY bm25(memory_fts) LIMIT 40",
                        (*params, match),
                    ).fetchall()
                    for rank, fts_row in enumerate(fts_rows):
                        entry = self._row_to_entry(fts_row)
                        entry["fts_score"] = max(0.2, 1.0 - rank / max(1, len(fts_rows)))
                        by_id[entry["id"]] = entry
                except sqlite3.OperationalError:
                    pass
            return list(by_id.values())

    def mark_recalled(self, item_ids: Iterable[int]) -> None:
        ids = list(dict.fromkeys(int(value) for value in item_ids if int(value) > 0))
        if not ids:
            return
        with self._connect() as connection:
            connection.executemany(
                "UPDATE memory_items SET recall_count=recall_count+1, last_accessed_at=? WHERE id=?",
                [(self._now_iso(), item_id) for item_id in ids],
            )

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def load_embedding(self, memory_id: int, model_key: str, text: str) -> np.ndarray | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT text_hash, dimensions, vector FROM memory_embeddings WHERE memory_id=? AND model_key=?",
                (int(memory_id), str(model_key)),
            ).fetchone()
        if not row or str(row["text_hash"]) != self._text_hash(text):
            return None
        vector = np.frombuffer(bytes(row["vector"]), dtype=np.float32)
        if vector.size != int(row["dimensions"]):
            return None
        return vector.copy()

    def save_embedding(self, memory_id: int, model_key: str, text: str, vector: np.ndarray) -> None:
        stored = np.asarray(vector, dtype=np.float32).reshape(-1)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_embeddings(memory_id, model_key, text_hash, dimensions, vector)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, model_key) DO UPDATE SET
                    text_hash=excluded.text_hash,
                    dimensions=excluded.dimensions,
                    vector=excluded.vector
                """,
                (int(memory_id), str(model_key), self._text_hash(text), int(stored.size), stored.tobytes()),
            )
