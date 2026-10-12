"""List the models your Nebius Token Factory key can use, highlighting NVIDIA ones.

    NEBIUS_API_KEY=... python scripts/list_models.py

Copy the exact IDs into CLERK_MODEL_SUPER / CLERK_MODEL_NANO / CLERK_MODEL_VISION / CLERK_MODEL_EMBED.
"""

from __future__ import annotations

import os
import sys

import httpx


def main() -> int:
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        print("Set NEBIUS_API_KEY first.", file=sys.stderr)
        return 1
    base = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    response = httpx.get(base.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if response.status_code >= 400:
        print(f"Token Factory returned {response.status_code}: {response.text[:300]}", file=sys.stderr)
        return 1
    ids = sorted(m["id"] for m in response.json().get("data", []))
    nvidia = [i for i in ids if "nvidia" in i.lower() or "nemotron" in i.lower()]
    print("NVIDIA / Nemotron models:")
    for model in nvidia:
        print(f"  {model}")
    print(f"\nAll models ({len(ids)}):")
    for model in ids:
        print(f"  {model}")
    print(
        "\nSuggested mapping (check the model cards):\n"
        "  CLERK_MODEL_SUPER  = the Nemotron 3 Super model (reasoning: Skeptic, Advocate)\n"
        "  CLERK_MODEL_NANO   = a Nemotron Nano model (fast: Reader, AI Auditor, Ask)\n"
        "  CLERK_MODEL_VISION = a Nemotron model that accepts images, e.g. a Nano Omni / VL model (photos of bills)\n"
        "  CLERK_MODEL_EMBED  = an NVIDIA embedding model if listed (optional)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
