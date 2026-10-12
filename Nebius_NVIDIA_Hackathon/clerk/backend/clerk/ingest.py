"""Turning uploaded files into documents: extract text, run the Reader, remember agreements."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from . import vault
from .agents.llm_agents import read_document
from .agents.runtime import AgentContext

TEXT_EXT = {".txt", ".md", ".text"}
PDF_EXT = {".pdf"}
IMAGE_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
DEMO_DIR = Path(__file__).parent / "demo"


class IngestError(ValueError):
    """A problem with the uploaded file that the user can fix."""


def clean_filename(name: str | None) -> str:
    base = Path(name or "document").name.strip() or "document"
    return base[:120]


def extract_text(filename: str, data: bytes, max_chars: int) -> tuple[str | None, bytes | None, str]:
    """Returns (text, image_bytes, mime). Exactly one of text/image is set."""
    ext = Path(filename).suffix.lower()
    if ext in TEXT_EXT:
        text = data.decode("utf-8", errors="replace")
    elif ext in PDF_EXT:
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages[:10])
        except (PdfReadError, ValueError, KeyError) as exc:
            raise IngestError("That PDF couldn't be opened. Is it password-protected or damaged?") from exc
        if not text.strip():
            raise IngestError("This PDF has no text layer (it's a scan). Upload a photo of it instead.")
    elif ext in IMAGE_EXT:
        return None, data, IMAGE_EXT[ext]
    else:
        raise IngestError("Clerk reads .txt, .pdf, .png, .jpg and .webp files.")
    text = text.replace("\x00", "").strip()
    if not text:
        raise IngestError("The file is empty.")
    if len(text) > max_chars:
        raise IngestError(f"That document is too long ({len(text):,} characters; the limit is {max_chars:,}).")
    return text, None, "text/plain"


def ingest(ctx: AgentContext, filename: str, data: bytes) -> dict[str, Any]:
    filename = clean_filename(filename)
    text, image, mime = extract_text(filename, data, ctx.settings.max_document_chars)
    if image is not None and ctx.llm.mode != "live":
        raise IngestError(
            "Reading photos needs live mode (Nemotron vision on Nebius). Upload a text or PDF bill instead."
        )
    result = read_document(ctx, text=text, image=image, mime=mime)
    parsed: dict[str, Any] | None = None
    if result.bill:
        parsed = result.bill.to_dict()
    elif result.contract:
        parsed = result.contract.to_dict()
    doc_id = ctx.db.add_document(ctx.workspace, filename, result.kind, result.provider, result.text, parsed)
    remembered = 0
    if result.contract and ctx.allowed("remember_documents"):
        remembered = len(vault.remember_contract(ctx, doc_id, result.contract))
    return {
        "id": doc_id,
        "filename": filename,
        "kind": result.kind,
        "provider": result.provider,
        "facts_remembered": remembered,
        "line_items": len(result.bill.line_items) if result.bill else 0,
    }


def demo_files() -> list[Path]:
    return sorted(DEMO_DIR.glob("*.txt"))


def load_demo(ctx: AgentContext) -> list[dict[str, Any]]:
    return [ingest(ctx, path.name, path.read_bytes()) for path in demo_files()]
