# GoodFoods Reservation Agent

A conversational reservation assistant for the GoodFoods restaurant chain. A small open-weight LLM (Llama 3.x 8B class) decides user intent and calls reservation tools exposed through a Model Context Protocol (MCP) server. The agent loop is written from scratch, without agent frameworks, and served through a Streamlit chat UI.

> Status: planning. See the docs below.

## Documents

- [What to learn before building](docs/LEARNING_PATH.md)
- [Step-by-step, testable build plan](docs/BUILD_PLAN.md)

## Stack

- Python 3.11+
- LLM: Llama 3.x 8B via an OpenAI-compatible API
- Tool protocol: MCP (official Python SDK)
- Storage: SQLite
- Frontend: Streamlit
- Tests: pytest
