"""MCP client side: connects the agent to the reservation MCP server.

The MCP SDK is async, while Streamlit and the agent loop are synchronous. The
bridge therefore runs one asyncio event loop in a background thread that keeps
the stdio connection to the server open for the whole session, and exposes
plain blocking `schemas()` / `call()` methods.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MCPBridgeError(RuntimeError):
    pass


def mcp_tool_to_openai(tool: Any) -> dict[str, Any]:
    """Convert an MCP Tool definition into the OpenAI/Llama function-calling format."""
    schema = dict(getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.pop("title", None)
    schema["properties"] = {k: _simplify(v) for k, v in schema["properties"].items()}
    description = " ".join((tool.description or "").split())
    return {"type": "function", "function": {"name": tool.name, "description": description, "parameters": schema}}


def _simplify(prop: dict[str, Any]) -> dict[str, Any]:
    """Flatten `anyOf: [X, null]` (how Optional is encoded) into a single type.
    Small models follow flat schemas far more reliably than anyOf unions."""
    prop = {k: v for k, v in prop.items() if k not in ("title", "default")}
    variants = prop.pop("anyOf", None)
    if variants:
        types = [v.get("type") for v in variants if v.get("type") and v.get("type") != "null"]
        if len(types) == 1:
            prop["type"] = types[0]
        elif "string" in types:
            prop["type"] = "string"  # e.g. int | str -> accept as string
        elif types:
            prop["type"] = types[0]
    return prop


class MCPToolBridge:
    """Blocking facade over an MCP stdio client session."""

    def __init__(self, command: list[str] | None = None, env: dict[str, str] | None = None, timeout: float = 20.0):
        self.command = command or [sys.executable, "-m", "mcp_server.server"]
        self.env = {**os.environ, **(env or {})}
        self.env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-bridge", daemon=True)
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._schemas: list[dict[str, Any]] = []
        self._tool_names: set[str] = set()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "MCPToolBridge":
        self._thread.start()
        self._run(self._connect(), timeout=self.timeout + 10)
        return self

    def close(self) -> None:
        if self._stack is not None:
            try:
                self._run(self._stack.aclose(), timeout=5)
            except Exception as exc:  # shutting down; nothing useful to do
                log.debug("MCP close error: %s", exc)
            self._stack = None
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def __enter__(self) -> "MCPToolBridge":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------ ToolProvider
    def schemas(self) -> list[dict[str, Any]]:
        return self._schemas

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if self._session is None:
            raise MCPBridgeError("MCP session is not started")
        try:
            result = self._run(self._session.call_tool(name, arguments), timeout=self.timeout)
        except FutureTimeout:
            return json.dumps({"ok": False, "error": "TIMEOUT", "message": f"{name} took too long. Try again."})

        text = "\n".join(getattr(c, "text", "") for c in (getattr(result, "content", None) or [])).strip()
        is_error = bool(getattr(result, "is_error", False) or getattr(result, "isError", False))
        if is_error:
            # Typically argument validation errors raised by the MCP server.
            return json.dumps({"ok": False, "error": "TOOL_ERROR", "message": text[:500] or "Tool failed."})
        return text or json.dumps({"ok": False, "error": "EMPTY_RESULT", "message": "Tool returned nothing."})

    # ------------------------------------------------------------ internals
    async def _connect(self) -> None:
        params = StdioServerParameters(command=self.command[0], args=self.command[1:], env=self.env, cwd=str(PROJECT_ROOT))
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        listed = await self._session.list_tools()
        self._schemas = [mcp_tool_to_openai(t) for t in listed.tools]
        self._tool_names = {t.name for t in listed.tools}
        log.info("MCP connected: %d tools", len(self._schemas))

    def _run(self, coro: Any, timeout: float) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            raise
