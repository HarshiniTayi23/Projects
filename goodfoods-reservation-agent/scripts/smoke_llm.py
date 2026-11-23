"""Phase 1/2 manual check against the real provider (needs LLM_API_KEY in .env).

    python -m scripts.smoke_llm
"""
from agent.agent import Agent
from agent.llm_client import LLMClient
from tests.fakes import ToyTools

client = LLMClient()
print("Plain reply:", client.chat([{"role": "user", "content": "Say hello in 3 words"}]).content)

for question in ["What time is it in Mumbai?", "Tell me a joke"]:
    tools = ToyTools()
    agent = Agent(client, tools, "You are a helpful assistant. Use tools only when needed.")
    turn = agent.chat(question)
    print(f"\n> {question}\n  tools used: {tools.invocations or 'none'}\n  reply: {turn.reply}")
