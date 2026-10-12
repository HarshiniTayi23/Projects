"""SQLite persistence. One file, no server: easy to deploy as a single container.

Every row is scoped to a workspace id (an anonymous browser cookie), so
several judges using the public demo never see each other's documents.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    filename TEXT NOT NULL,
    kind TEXT NOT NULL,              -- 'bill' | 'contract' | 'unknown'
    provider TEXT,
    text TEXT NOT NULL,
    parsed TEXT,                     -- JSON: Bill or ContractFacts
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_ws ON documents(workspace);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    document_id TEXT,
    provider TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    embedding TEXT,                  -- JSON list[float]
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_ws ON facts(workspace);

CREATE TABLE IF NOT EXISTS audits (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL,            -- 'running' | 'done' | 'error'
    result TEXT,                     -- JSON summary + findings
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audits_ws ON audits(workspace);

CREATE TABLE IF NOT EXISTS actions (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    audit_id TEXT,
    type TEXT NOT NULL,              -- 'letter' | 'reminder' | 'call_script'
    status TEXT NOT NULL,            -- 'pending' | 'approved' | 'rejected'
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    filename TEXT NOT NULL,
    mime TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_actions_ws ON actions(workspace);

CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    audit_id TEXT,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    mode TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    cost_usd REAL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_ws ON traces(workspace);

CREATE TABLE IF NOT EXISTS permissions (
    workspace TEXT NOT NULL,
    skill TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    PRIMARY KEY (workspace, skill)
);

CREATE TABLE IF NOT EXISTS usage (
    day TEXT NOT NULL,
    workspace TEXT NOT NULL,
    kind TEXT NOT NULL,              -- 'llm_call' | 'audit'
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_day ON usage(day, kind);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.Lock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # A single shared connection keeps ':memory:' databases alive for tests;
        # the lock serialises access from FastAPI's thread pool.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.tx() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        with self.tx() as conn:
            return conn.execute(sql, params).rowcount

    # ---- documents -------------------------------------------------------
    def add_document(
        self, workspace: str, filename: str, kind: str, provider: str | None, text: str, parsed: dict[str, Any] | None
    ) -> str:
        doc_id = new_id()
        self.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)",
            (doc_id, workspace, filename, kind, provider, text, json.dumps(parsed) if parsed else None, time.time()),
        )
        return doc_id

    def get_document(self, workspace: str, doc_id: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM documents WHERE workspace=? AND id=?", (workspace, doc_id))
        if row and row["parsed"]:
            row["parsed"] = json.loads(row["parsed"])
        return row

    def list_documents(self, workspace: str) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT d.id, d.filename, d.kind, d.provider, d.created, "
            "(SELECT a.id FROM audits a WHERE a.document_id=d.id AND a.status='done' "
            " ORDER BY a.created DESC LIMIT 1) AS last_audit_id "
            "FROM documents d WHERE d.workspace=? ORDER BY d.created",
            (workspace,),
        )
        return rows

    # ---- facts -----------------------------------------------------------
    def add_fact(
        self,
        workspace: str,
        document_id: str | None,
        provider: str,
        key: str,
        value: str,
        embedding: list[float] | None,
    ) -> str:
        fact_id = new_id()
        self.execute(
            "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?)",
            (
                fact_id,
                workspace,
                document_id,
                provider,
                key,
                value,
                json.dumps(embedding) if embedding else None,
                time.time(),
            ),
        )
        return fact_id

    def list_facts(self, workspace: str) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM facts WHERE workspace=? ORDER BY provider, created", (workspace,))
        for row in rows:
            row["embedding"] = json.loads(row["embedding"]) if row["embedding"] else None
        return rows

    def update_fact(self, workspace: str, fact_id: str, value: str, embedding: list[float] | None) -> bool:
        return (
            self.execute(
                "UPDATE facts SET value=?, embedding=? WHERE workspace=? AND id=?",
                (value, json.dumps(embedding) if embedding else None, workspace, fact_id),
            )
            > 0
        )

    def delete_fact(self, workspace: str, fact_id: str) -> bool:
        return self.execute("DELETE FROM facts WHERE workspace=? AND id=?", (workspace, fact_id)) > 0

    # ---- audits ----------------------------------------------------------
    def create_audit(self, workspace: str, document_id: str) -> str:
        audit_id = new_id()
        self.execute(
            "INSERT INTO audits VALUES (?,?,?,?,?,?)", (audit_id, workspace, document_id, "running", None, time.time())
        )
        return audit_id

    def finish_audit(self, audit_id: str, status: str, result: dict[str, Any]) -> None:
        self.execute("UPDATE audits SET status=?, result=? WHERE id=?", (status, json.dumps(result), audit_id))

    def get_audit(self, workspace: str, audit_id: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM audits WHERE workspace=? AND id=?", (workspace, audit_id))
        if row and row["result"]:
            row["result"] = json.loads(row["result"])
        return row

    def list_audits(self, workspace: str) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM audits WHERE workspace=? AND status='done' ORDER BY created", (workspace,))
        for row in rows:
            row["result"] = json.loads(row["result"]) if row["result"] else None
        return rows

    # ---- actions ---------------------------------------------------------
    def add_action(
        self, workspace: str, audit_id: str | None, type_: str, title: str, content: str, filename: str, mime: str
    ) -> str:
        action_id = new_id()
        self.execute(
            "INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?,?)",
            (action_id, workspace, audit_id, type_, "pending", title, content, filename, mime, time.time()),
        )
        return action_id

    def list_actions(self, workspace: str) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM actions WHERE workspace=? ORDER BY created", (workspace,))

    def get_action(self, workspace: str, action_id: str) -> dict[str, Any] | None:
        return self.query_one("SELECT * FROM actions WHERE workspace=? AND id=?", (workspace, action_id))

    def set_action_status(self, workspace: str, action_id: str, status: str) -> bool:
        return (
            self.execute("UPDATE actions SET status=? WHERE workspace=? AND id=?", (status, workspace, action_id)) > 0
        )

    # ---- traces ----------------------------------------------------------
    def add_trace(self, workspace: str, audit_id: str | None, step: dict[str, Any]) -> None:
        self.execute(
            "INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                new_id(),
                workspace,
                audit_id,
                step["agent"],
                step["model"],
                step["mode"],
                step["prompt_tokens"],
                step["completion_tokens"],
                step["latency_ms"],
                step.get("cost_usd"),
                time.time(),
            ),
        )

    def list_traces(self, workspace: str, audit_id: str | None = None) -> list[dict[str, Any]]:
        if audit_id:
            return self.query(
                "SELECT * FROM traces WHERE workspace=? AND audit_id=? ORDER BY created", (workspace, audit_id)
            )
        return self.query("SELECT * FROM traces WHERE workspace=? ORDER BY created DESC LIMIT 200", (workspace,))

    # ---- permissions -----------------------------------------------------
    def get_permissions(self, workspace: str) -> dict[str, bool]:
        rows = self.query("SELECT skill, enabled FROM permissions WHERE workspace=?", (workspace,))
        return {r["skill"]: bool(r["enabled"]) for r in rows}

    def set_permission(self, workspace: str, skill: str, enabled: bool) -> None:
        self.execute(
            "INSERT INTO permissions VALUES (?,?,?) ON CONFLICT(workspace, skill) DO UPDATE SET enabled=excluded.enabled",
            (workspace, skill, int(enabled)),
        )

    # ---- usage / budgets -------------------------------------------------
    def record_usage(self, workspace: str, kind: str) -> None:
        now = time.time()
        self.execute(
            "INSERT INTO usage VALUES (?,?,?,?)", (time.strftime("%Y-%m-%d", time.gmtime(now)), workspace, kind, now)
        )

    def count_usage_today(self, kind: str) -> int:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        row = self.query_one("SELECT COUNT(*) AS n FROM usage WHERE day=? AND kind=?", (day, kind))
        return int(row["n"]) if row else 0

    def count_usage_since(self, workspace: str, kind: str, since: float) -> int:
        row = self.query_one(
            "SELECT COUNT(*) AS n FROM usage WHERE workspace=? AND kind=? AND ts>=?", (workspace, kind, since)
        )
        return int(row["n"]) if row else 0

    # ---- wipe ------------------------------------------------------------
    def wipe_workspace(self, workspace: str) -> None:
        with self.tx() as conn:
            for table in ("documents", "facts", "audits", "actions", "traces", "permissions"):
                conn.execute(f"DELETE FROM {table} WHERE workspace=?", (workspace,))  # noqa: S608 - fixed table names
