"""Append-only evidence and local embedding cache, separate from prompt windows."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path


class HistoryArchive:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS evidence (key TEXT PRIMARY KEY, kind TEXT, text TEXT, payload TEXT, created TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT, model TEXT, vector TEXT, PRIMARY KEY(key,model))")

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def add(self, kind: str, text: str, payload: dict | None = None, *, key: str = ""):
        if not text.strip():
            return
        payload = payload or {}
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        key = key or hashlib.sha256((kind + text + serialized).encode()).hexdigest()
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?)",
                       (key, kind, text, serialized, datetime.now().isoformat()))

    def rows(self):
        with self.connect() as db:
            return [dict(key=k, kind=kind, text=text, payload=json.loads(payload), created=created)
                    for k, kind, text, payload, created in db.execute("SELECT * FROM evidence ORDER BY rowid")]

    def similarities(self, query, texts, fallback):
        """Reuse the existing router; persist vectors across process restarts."""
        router = getattr(fallback, "__self__", None)
        if not callable(getattr(router, "embeddings", None)):
            return fallback(query, texts) if callable(fallback) else None
        import numpy as np
        query_vector = router.embeddings([query])
        if query_vector is None:
            return None
        model = router.cache_identity
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        with self.connect() as db:
            cached = {key: json.loads(vector) for key, vector in db.execute(
                "SELECT key,vector FROM vectors WHERE model=?", (model,))}
            missing = list(dict.fromkeys(i for i, key in enumerate(keys) if key not in cached))
            for start in range(0, len(missing), 32):
                indexes = missing[start:start + 32]
                vectors = router.embeddings([texts[i] for i in indexes])
                if vectors is None:
                    return None
                for i, vector in zip(indexes, vectors):
                    cached[keys[i]] = vector.tolist()
                    db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?)",
                               (keys[i], model, json.dumps(cached[keys[i]])))
        return [max(0.0, min(1.0, float(np.asarray(cached[key]) @ query_vector[0]))) for key in keys]
