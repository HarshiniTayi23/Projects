"""Wire together LLM client + MCP bridge + system prompt into a ready Agent."""
from __future__ import annotations

from agent.agent import Agent, ChatModel
from agent.config import settings
from agent.llm_client import LLMClient
from agent.mcp_bridge import MCPToolBridge
from agent.prompts import build_system_prompt


def build_agent(llm: ChatModel | None = None, bridge: MCPToolBridge | None = None) -> tuple[Agent, MCPToolBridge]:
    bridge = bridge or MCPToolBridge().start()
    agent = Agent(
        llm=llm or LLMClient(),
        tools=bridge,
        system_prompt=lambda: build_system_prompt(tz=settings.timezone),  # re-rendered each turn: date stays fresh
    )
    return agent, bridge
