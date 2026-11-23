"""Phase 5 check: start the MCP server over stdio and list/call its tools.

    python -m scripts.list_tools
"""
import json

from agent.mcp_bridge import MCPToolBridge

with MCPToolBridge() as bridge:
    for schema in bridge.schemas():
        fn = schema["function"]
        print(f"- {fn['name']}({', '.join(fn['parameters']['properties'])})")
    print("\nsearch_restaurants(cuisine='Italian') ->")
    print(json.dumps(json.loads(bridge.call("search_restaurants", {"cuisine": "Italian"}))["restaurants"][:2], indent=2))
