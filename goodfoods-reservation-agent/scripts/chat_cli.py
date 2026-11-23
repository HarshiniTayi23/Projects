"""Chat with the agent in the terminal (needs LLM_API_KEY).

    python -m scripts.chat_cli            add --debug to print tool calls
"""
import sys

from agent.factory import build_agent

debug = "--debug" in sys.argv
agent, bridge = build_agent()
print("GoodFoods assistant ready. Type 'quit' to exit.\n")
try:
    while True:
        text = input("you> ").strip()
        if text.lower() in ("quit", "exit"):
            break
        turn = agent.chat(text)
        if debug:
            for ev in turn.tool_events:
                print(f"   [tool] {ev.name}({ev.arguments}) -> {ev.result[:160]}")
        print(f"gia> {turn.reply}\n")
except (KeyboardInterrupt, EOFError):
    pass
finally:
    bridge.close()
