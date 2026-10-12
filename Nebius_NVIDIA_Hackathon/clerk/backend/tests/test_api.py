"""HTTP API: every endpoint, error paths, workspace isolation and static hosting."""

import io
import json
from dataclasses import replace

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from clerk.api import create_app
from tests.conftest import demo_text, read_events


def upload(client, name, content, mime="text/plain"):
    return client.post("/api/documents", files={"file": (name, content, mime)})


def load_demo(client):
    r = client.post("/api/demo/load")
    assert r.status_code == 200
    return {d["filename"]: d for d in r.json()}


def test_health_and_config_set_workspace_cookie(client):
    assert client.get("/api/health").json()["mode"] == "offline"
    r = client.get("/api/config")
    body = r.json()
    assert len(client.cookies["clerk_ws"]) == 32
    assert body["mode"] == "offline" and body["search_mode"] == "offline"
    assert body["models"]["super"].startswith("offline-sim/")
    assert {s["id"] for s in body["skills"]} >= {"web_research", "draft_letters", "ai_auditor"}


def test_invalid_workspace_cookie_is_replaced(client):
    client.cookies.set("clerk_ws", "../../etc/passwd")
    r = client.get("/api/health")
    assert len(r.cookies["clerk_ws"]) == 32


def test_demo_load_and_document_listing(client):
    docs = load_demo(client)
    assert len(docs) == 6
    assert docs["01_northwind_agreement.txt"]["facts_remembered"] == 7
    listed = client.get("/api/documents").json()
    assert [d["kind"] for d in listed].count("bill") == 3
    detail = client.get(f"/api/documents/{docs['11_northwind_march_statement.txt']['id']}").json()
    assert detail["parsed"]["total"] == 89.23 and "NORTHWIND" in detail["text"]
    # loading the demo again resets rather than duplicates
    assert len(load_demo(client)) == 6 and len(client.get("/api/documents").json()) == 6


def test_full_demo_audits_stream_and_score(client):
    docs = load_demo(client)
    expected = {
        "11_northwind_march_statement.txt": (39.99, 2, 1),
        "12_brightside_april_statement.txt": (34.50, 3, 0),
        "13_lakeview_may_statement.txt": (0.0, 0, 4),
    }
    for name, (total, upheld, dismissed) in expected.items():
        events = read_events(client, docs[name]["id"])
        types = [e["type"] for e in events]
        assert types[0] == "start" and types[-1] == "done"
        assert "error" not in types
        summary = next(e["summary"] for e in events if e["type"] == "summary")
        assert (summary["recoverable_total"], summary["upheld"], summary["dismissed"]) == (total, upheld, dismissed)
        audit = client.get(f"/api/audits/{events[-1]['audit_id']}").json()
        assert audit["status"] == "done" and audit["result"]["recoverable_total"] == total
        traces = client.get("/api/traces", params={"audit_id": events[-1]["audit_id"]}).json()
        assert traces and all(t["mode"] == "offline" for t in traces)
    board = client.get("/api/scoreboard").json()
    assert board == {"recoverable_total": 74.49, "upheld": 5, "dismissed": 5, "bills_audited": 3}
    # re-auditing a bill replaces, not adds to, its score
    read_events(client, docs["11_northwind_march_statement.txt"]["id"])
    assert client.get("/api/scoreboard").json()["recoverable_total"] == 74.49


def test_audit_errors(client):
    docs = load_demo(client)
    assert client.get("/api/documents/nope/audit/stream").status_code == 404
    r = client.get(f"/api/documents/{docs['01_northwind_agreement.txt']['id']}/audit/stream")
    assert r.status_code == 400 and "Only bills" in r.json()["detail"]
    assert client.get("/api/audits/nope").status_code == 404


def test_audit_rate_limit(settings, db):
    with TestClient(create_app(replace(settings, audits_per_workspace_per_hour=1), db=db)) as c:
        doc = load_demo(c)["12_brightside_april_statement.txt"]
        read_events(c, doc["id"])
        r = c.get(f"/api/documents/{doc['id']}/audit/stream")
        assert r.status_code == 429


def test_actions_need_approval_before_download(client):
    docs = load_demo(client)
    read_events(client, docs["12_brightside_april_statement.txt"]["id"])
    actions = {a["type"]: a for a in client.get("/api/actions").json()}
    assert set(actions) == {"letter", "call_script", "reminder"}
    assert all(a["status"] == "pending" for a in actions.values())
    letter = actions["letter"]
    assert "$34.50" in letter["content"]
    assert client.get(f"/api/actions/{letter['id']}/download").status_code == 403
    assert client.post(f"/api/actions/{letter['id']}/approve").json()["status"] == "approved"
    r = client.get(f"/api/actions/{letter['id']}/download")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert r.text == letter["content"]
    reminder = actions["reminder"]
    assert "BEGIN:VCALENDAR" in reminder["content"] and "DTSTART;VALUE=DATE:20260502" in reminder["content"]
    client.post(f"/api/actions/{reminder['id']}/reject")
    assert client.get(f"/api/actions/{reminder['id']}/download").status_code == 403
    assert client.post(f"/api/actions/{reminder['id']}/launch").status_code == 404
    assert client.post("/api/actions/nope/approve").status_code == 404


def test_permissions_turn_off_letters_and_reminders(client):
    docs = load_demo(client)
    for skill in ("draft_letters", "reminders"):
        assert client.put("/api/permissions", json={"skill": skill, "enabled": False}).status_code == 200
    assert client.put("/api/permissions", json={"skill": "launch_rockets", "enabled": True}).status_code == 404
    events = read_events(client, docs["12_brightside_april_statement.txt"]["id"])
    assert not [e for e in events if e["type"] == "action"]
    assert client.get("/api/actions").json() == []
    skills = {s["id"]: s["enabled"] for s in client.get("/api/config").json()["skills"]}
    assert skills["draft_letters"] is False and skills["web_research"] is True


def test_editing_memory_changes_the_audit(client):
    docs = load_demo(client)
    facts = client.get("/api/vault/facts").json()
    rate = next(f for f in facts if f["provider"] == "Northwind Fiber" and f["key"] == "monthly_rate")
    assert rate["value"] == "39.99" and "embedding" not in rate
    assert client.patch(f"/api/vault/facts/{rate['id']}", json={"value": "69.99"}).status_code == 200
    events = read_events(client, docs["11_northwind_march_statement.txt"]["id"])
    summary = next(e["summary"] for e in events if e["type"] == "summary")
    assert summary["recoverable_total"] == 9.99  # only the add-on remains
    assert client.patch("/api/vault/facts/nope", json={"value": "1"}).status_code == 404
    assert client.patch(f"/api/vault/facts/{rate['id']}", json={"value": ""}).status_code == 422


def test_vault_search_ask_export_delete(client):
    load_demo(client)
    hits = client.get("/api/vault/search", params={"q": "late fee lakeview"}).json()
    assert hits and hits[0]["provider"] == "Lakeview Apartments"
    answer = client.post("/api/vault/ask", json={"question": "What is my internet late fee?"}).json()
    assert "late fee cap" in answer["answer"] and answer["facts"]
    export = client.get("/api/vault/export")
    assert "attachment" in export.headers["content-disposition"] and len(json.loads(export.text)) == 21
    fact_id = json.loads(export.text)[0]["id"]
    assert client.delete(f"/api/vault/facts/{fact_id}").status_code == 200
    assert client.delete(f"/api/vault/facts/{fact_id}").status_code == 404
    assert len(client.get("/api/vault/facts").json()) == 20


def test_ask_with_empty_vault(client):
    r = client.post("/api/vault/ask", json={"question": "anything?"}).json()
    assert "don't have anything" in r["answer"]


def test_workspaces_are_isolated(settings, db):
    app = create_app(settings, db=db)
    with TestClient(app) as alice, TestClient(app) as bob:
        docs = load_demo(alice)
        read_events(alice, docs["11_northwind_march_statement.txt"]["id"])
        bob.get("/api/health")
        assert alice.cookies["clerk_ws"] != bob.cookies["clerk_ws"]
        assert bob.get("/api/documents").json() == []
        assert bob.get("/api/vault/facts").json() == []
        assert bob.get("/api/actions").json() == []
        assert bob.get(f"/api/documents/{docs['11_northwind_march_statement.txt']['id']}").status_code == 404
        action = alice.get("/api/actions").json()[0]
        assert bob.post(f"/api/actions/{action['id']}/approve").status_code == 404


def test_wipe_workspace(client):
    load_demo(client)
    assert client.delete("/api/workspace").json() == {"ok": True}
    assert client.get("/api/documents").json() == []
    assert client.get("/api/vault/facts").json() == []


def test_upload_text_bill_and_contract(client):
    r = upload(client, "agreement.txt", demo_text("02_brightside_agreement.txt"))
    assert r.status_code == 201 and r.json()["kind"] == "contract" and r.json()["facts_remembered"] == 7
    r = upload(client, "../../bill.txt", demo_text("12_brightside_april_statement.txt"))
    assert r.status_code == 201 and r.json()["filename"] == "bill.txt" and r.json()["line_items"] == 4
    r = upload(client, "note.txt", "Remember to buy milk")
    assert r.status_code == 201 and r.json()["kind"] == "unknown"


def make_pdf(text: str) -> bytes:
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf)
    y = 800
    for line in text.splitlines():
        pdf.drawString(40, y, line)
        y -= 14
    pdf.save()
    return buf.getvalue()


def test_upload_pdf_bill(client):
    upload(client, "a.txt", demo_text("01_northwind_agreement.txt"))
    r = upload(client, "statement.pdf", make_pdf(demo_text("11_northwind_march_statement.txt")), "application/pdf")
    assert r.status_code == 201, r.text
    assert r.json()["kind"] == "bill" and r.json()["line_items"] == 3
    events = read_events(client, r.json()["id"])
    assert next(e["summary"] for e in events if e["type"] == "summary")["recoverable_total"] == 39.99


def test_upload_errors(settings, db):
    with TestClient(create_app(replace(settings, max_upload_bytes=1000, max_document_chars=200), db=db)) as c:
        assert upload(c, "big.txt", "x" * 1001).status_code == 413
        assert upload(c, "empty.txt", "").status_code == 400
        assert upload(c, "blank.txt", "   \n ").json()["detail"] == "The file is empty."
        assert "too long" in upload(c, "long.txt", "x" * 500).json()["detail"]
        assert upload(c, "virus.exe", "MZ").status_code == 400
        assert "PDF" in upload(c, "broken.pdf", "not a pdf", "application/pdf").json()["detail"]
        r = upload(c, "photo.png", b"\x89PNG\r\n", "image/png")
        assert r.status_code == 400 and "live mode" in r.json()["detail"]
        assert (
            upload(c, "scan.pdf", make_pdf(""), "application/pdf").json()["detail"].startswith("This PDF has no text")
        )


def test_samples(client):
    names = client.get("/api/samples").json()
    assert "11_northwind_march_statement.txt" in names
    r = client.get("/api/samples/11_northwind_march_statement.txt")
    assert r.status_code == 200 and "NORTHWIND" in r.text
    assert client.get("/api/samples/..%2Fapi.py").status_code == 404


def test_static_frontend_and_spa_fallback(settings, db, tmp_path):
    (tmp_path / "index.html").write_text("<html>clerk</html>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path.parent / "secret.txt").write_text("secret")
    with TestClient(create_app(replace(settings, static_dir=tmp_path), db=db)) as c:
        assert c.get("/").text == "<html>clerk</html>"
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/vault").text == "<html>clerk</html>"
        assert "secret" not in c.get("/..%2Fsecret.txt").text
        assert c.get("/api/does-not-exist").status_code == 404
        assert c.get("/api/health").json()["ok"] is True


def test_audit_failure_is_reported_not_hung(client, monkeypatch):
    from clerk.llm.base import LLMError

    def boom(ctx, bill, today=None):
        raise LLMError("Token Factory returned 503")

    monkeypatch.setattr("clerk.api.run_tribunal", boom)
    docs = load_demo(client)
    events = read_events(client, docs["11_northwind_march_statement.txt"]["id"])
    assert events[-1]["type"] == "error" and "503" in events[-1]["message"]
    assert client.get(f"/api/audits/{events[-1]['audit_id']}").json()["status"] == "error"


def test_unexpected_crash_hides_internals(client, monkeypatch):
    def boom(ctx, bill, today=None):
        raise KeyError("secret internal detail")

    monkeypatch.setattr("clerk.api.run_tribunal", boom)
    docs = load_demo(client)
    events = read_events(client, docs["11_northwind_march_statement.txt"]["id"])
    assert events[-1] == {
        "type": "error",
        "message": "Something went wrong during the audit.",
        "audit_id": events[-1]["audit_id"],
    }
