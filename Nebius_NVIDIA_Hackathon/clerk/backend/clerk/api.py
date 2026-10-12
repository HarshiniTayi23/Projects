"""HTTP API and static frontend hosting."""

from __future__ import annotations

import json
import logging
import queue
import re
import secrets
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__, ingest, vault
from .agents.llm_agents import answer_question
from .agents.runtime import SKILLS, AgentContext
from .agents.tribunal import run_tribunal
from .config import Settings, load_settings
from .db import Database
from .llm import LLMError, LLMProvider, build_provider
from .schemas import Bill
from .search import SearchClient, build_search

log = logging.getLogger("clerk")
COOKIE = "clerk_ws"
_WS_RE = re.compile(r"^[a-f0-9]{32}$")
HEARTBEAT_SECONDS = 15.0


class FactUpdate(BaseModel):
    value: str = Field(min_length=1, max_length=500)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class PermissionUpdate(BaseModel):
    skill: str
    enabled: bool


def create_app(
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
    search: SearchClient | None = None,
    db: Database | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    llm = llm or build_provider(settings)
    search = search or build_search(settings)
    db = db or Database(settings.db_path)

    app = FastAPI(title="Clerk", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.settings, app.state.llm, app.state.search, app.state.db = settings, llm, search, db

    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def workspace_cookie(request: Request, call_next: Any) -> Response:
        ws = request.cookies.get(COOKIE, "")
        fresh = not _WS_RE.match(ws)
        if fresh:
            ws = secrets.token_hex(16)
        request.state.workspace = ws
        response: Response = await call_next(request)
        if fresh:
            response.set_cookie(
                COOKIE,
                ws,
                max_age=60 * 60 * 24 * 90,
                httponly=True,
                samesite="lax",
                secure=request.url.scheme == "https",
            )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    def ctx_for(request: Request, audit_id: str | None = None, emit: Any = None) -> AgentContext:
        ws = request.state.workspace
        ctx = AgentContext(
            settings=settings,
            llm=llm,
            search=search,
            db=db,
            workspace=ws,
            audit_id=audit_id,
            permissions=db.get_permissions(ws),
        )
        if emit is not None:
            ctx.emit = emit
        return ctx

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "mode": llm.mode}

    @app.get("/api/config")
    def config(request: Request) -> dict[str, Any]:
        perms = db.get_permissions(request.state.workspace)
        return {
            "mode": llm.mode,
            "search_mode": search.mode,
            "models": {role: llm.model_for(role) for role in ("super", "nano", "vision")},
            "skills": [{"id": k, "description": v, "enabled": perms.get(k, True)} for k, v in SKILLS.items()],
            "limits": {
                "max_upload_bytes": settings.max_upload_bytes,
                "audits_per_hour": settings.audits_per_workspace_per_hour,
            },
        }

    # ------------------------------------------------------------- documents
    @app.get("/api/documents")
    def list_documents(request: Request) -> list[dict[str, Any]]:
        return db.list_documents(request.state.workspace)

    @app.post("/api/documents", status_code=201)
    def upload(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"File is larger than {settings.max_upload_bytes // 1024} KB.")
        if not data:
            raise HTTPException(400, "The file is empty.")
        try:
            return ingest.ingest(ctx_for(request), file.filename or "document", data)
        except ingest.IngestError as exc:
            raise HTTPException(400, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/documents/{doc_id}")
    def get_document(request: Request, doc_id: str) -> dict[str, Any]:
        doc = db.get_document(request.state.workspace, doc_id)
        if not doc:
            raise HTTPException(404, "Document not found.")
        return doc

    @app.get("/api/documents/{doc_id}/audit/stream")
    def audit_stream(request: Request, doc_id: str) -> StreamingResponse:
        ws = request.state.workspace
        doc = db.get_document(ws, doc_id)
        if not doc:
            raise HTTPException(404, "Document not found.")
        if doc["kind"] != "bill" or not doc["parsed"]:
            raise HTTPException(400, "Only bills can be audited. Agreements are remembered in the Vault instead.")
        since = time.time() - 3600
        if db.count_usage_since(ws, "audit", since) >= settings.audits_per_workspace_per_hour:
            raise HTTPException(429, "You've run a lot of audits this hour. Please wait a bit and try again.")
        db.record_usage(ws, "audit")
        bill = Bill.from_dict(doc["parsed"])
        audit_id = db.create_audit(ws, doc_id)
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        ctx = ctx_for(request, audit_id=audit_id, emit=events.put)

        def worker() -> None:
            try:
                summary = run_tribunal(ctx, bill)
                db.finish_audit(audit_id, "done", summary)
                events.put({"type": "done", "audit_id": audit_id})
            except Exception as exc:  # surface any failure to the browser, never hang the stream
                log.exception("audit %s failed", audit_id)
                db.finish_audit(audit_id, "error", {"error": str(exc)})
                message = str(exc) if isinstance(exc, LLMError) else "Something went wrong during the audit."
                events.put({"type": "error", "message": message, "audit_id": audit_id})
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def stream() -> Iterator[str]:
            yield f"data: {json.dumps({'type': 'start', 'audit_id': audit_id})}\n\n"
            while True:
                try:
                    event = events.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if event is None:
                    return
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/audits/{audit_id}")
    def get_audit(request: Request, audit_id: str) -> dict[str, Any]:
        audit = db.get_audit(request.state.workspace, audit_id)
        if not audit:
            raise HTTPException(404, "Audit not found.")
        return audit

    @app.get("/api/scoreboard")
    def scoreboard(request: Request) -> dict[str, Any]:
        latest: dict[str, dict[str, Any]] = {}
        for audit in db.list_audits(request.state.workspace):
            latest[audit["document_id"]] = audit["result"] or {}
        return {
            "recoverable_total": round(sum(r.get("recoverable_total", 0) for r in latest.values()), 2),
            "upheld": sum(r.get("upheld", 0) for r in latest.values()),
            "dismissed": sum(r.get("dismissed", 0) for r in latest.values()),
            "bills_audited": len(latest),
        }

    # ----------------------------------------------------------------- vault
    @app.get("/api/vault/facts")
    def facts(request: Request) -> list[dict[str, Any]]:
        return [{k: v for k, v in f.items() if k != "embedding"} for f in db.list_facts(request.state.workspace)]

    @app.patch("/api/vault/facts/{fact_id}")
    def edit_fact(request: Request, fact_id: str, body: FactUpdate) -> dict[str, bool]:
        if not vault.update_fact(ctx_for(request), fact_id, body.value.strip()):
            raise HTTPException(404, "Fact not found.")
        return {"ok": True}

    @app.delete("/api/vault/facts/{fact_id}")
    def delete_fact(request: Request, fact_id: str) -> dict[str, bool]:
        if not db.delete_fact(request.state.workspace, fact_id):
            raise HTTPException(404, "Fact not found.")
        return {"ok": True}

    @app.get("/api/vault/search")
    def search_vault(request: Request, q: str) -> list[dict[str, Any]]:
        return vault.search(ctx_for(request), q[:300])

    @app.post("/api/vault/ask")
    def ask(request: Request, body: Question) -> dict[str, Any]:
        ctx = ctx_for(request)
        hits = vault.search(ctx, body.question)
        try:
            answer = answer_question(ctx, body.question, hits)
        except LLMError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"answer": answer, "facts": hits}

    @app.get("/api/vault/export")
    def export_vault(request: Request) -> Response:
        rows = facts(request)
        return Response(
            json.dumps(rows, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="clerk-vault.json"'},
        )

    # --------------------------------------------------------------- actions
    @app.get("/api/actions")
    def actions(request: Request) -> list[dict[str, Any]]:
        return db.list_actions(request.state.workspace)

    @app.post("/api/actions/{action_id}/{decision}")
    def decide(request: Request, action_id: str, decision: str) -> dict[str, Any]:
        status = {"approve": "approved", "reject": "rejected"}.get(decision)
        if status is None:
            raise HTTPException(404, "Unknown decision.")
        if not db.set_action_status(request.state.workspace, action_id, status):
            raise HTTPException(404, "Action not found.")
        return {"ok": True, "status": status}

    @app.get("/api/actions/{action_id}/download")
    def download(request: Request, action_id: str) -> Response:
        action = db.get_action(request.state.workspace, action_id)
        if not action:
            raise HTTPException(404, "Action not found.")
        if action["status"] != "approved":
            raise HTTPException(403, "Approve this action before downloading it.")
        return Response(
            action["content"],
            media_type=action["mime"],
            headers={"Content-Disposition": f'attachment; filename="{action["filename"]}"'},
        )

    # ----------------------------------------------------------- trace/perms
    @app.get("/api/traces")
    def traces(request: Request, audit_id: str | None = None) -> list[dict[str, Any]]:
        return db.list_traces(request.state.workspace, audit_id)

    @app.put("/api/permissions")
    def set_permission(request: Request, body: PermissionUpdate) -> dict[str, bool]:
        if body.skill not in SKILLS:
            raise HTTPException(404, "Unknown skill.")
        db.set_permission(request.state.workspace, body.skill, body.enabled)
        return {"ok": True}

    # ---------------------------------------------------------- demo/samples
    @app.post("/api/demo/load")
    def demo(request: Request) -> list[dict[str, Any]]:
        ws = request.state.workspace
        db.wipe_workspace(ws)
        try:
            return ingest.load_demo(ctx_for(request))
        except LLMError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/samples")
    def samples() -> list[str]:
        return [p.name for p in ingest.demo_files()]

    @app.get("/api/samples/{name}")
    def sample(name: str) -> FileResponse:
        match = next((p for p in ingest.demo_files() if p.name == name), None)
        if match is None:
            raise HTTPException(404, "Sample not found.")
        return FileResponse(match, media_type="text/plain", filename=match.name)

    @app.delete("/api/workspace")
    def wipe(request: Request) -> dict[str, bool]:
        db.wipe_workspace(request.state.workspace)
        return {"ok": True}

    # ---------------------------------------------------------------- static
    static = settings.static_dir
    if static and Path(static).is_dir():
        root = Path(static).resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> Response:
            if path.startswith("api/"):
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            candidate = (root / path).resolve()
            if path and candidate.is_file() and root in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(root / "index.html")

    return app


def main() -> None:  # pragma: no cover - used by `python -m clerk.api`
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)  # noqa: S104


if __name__ == "__main__":  # pragma: no cover
    main()
