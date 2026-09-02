"""Anthropic tool-use chat loop for the 'Ask AI' tab.

Generic on purpose: this module knows nothing about futures data or USDA
reports — app.py supplies a `tool_dispatch(name, input) -> dict` callback that
does the actual lookups, so this file only owns the Claude conversation loop.
"""

from __future__ import annotations

import json

MODEL = "claude-sonnet-5"
MAX_TOOL_ROUNDS = 6

TOOLS = [
    {
        "name": "get_live_curve",
        "description": "The current live futures curve for a product: upcoming contract tickers, "
                        "expirations, and last prices, nearest month first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_code": {"type": "string", "enum": ["ZC", "ZS", "ZW", "KE"],
                                 "description": "ZC=corn, ZS=soybeans, ZW=Chicago/SRW wheat, KE=KC/HRW wheat"},
                "n_contracts": {"type": "integer", "description": "how many months out to return, default 8"},
            },
            "required": ["product_code"],
        },
    },
    {
        "name": "get_contract_summary",
        "description": "Summary stats for one specific contract's full trading life: first/last "
                        "session, all-time high/low and the dates they occurred, latest settlement.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_code": {"type": "string", "enum": ["ZC", "ZS", "ZW", "KE"]},
                "ticker": {"type": "string", "description": "e.g. ZCZ6 for Dec 2026 corn"},
            },
            "required": ["product_code", "ticker"],
        },
    },
    {
        "name": "get_monthly_high_low",
        "description": "One row per calendar month a contract traded: the session high/low made "
                        "that month and the date each occurred.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_code": {"type": "string", "enum": ["ZC", "ZS", "ZW", "KE"]},
                "ticker": {"type": "string"},
            },
            "required": ["product_code", "ticker"],
        },
    },
    {
        "name": "get_price_on_date",
        "description": "A contract's settlement price on, or the nearest session before, a given date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_code": {"type": "string", "enum": ["ZC", "ZS", "ZW", "KE"]},
                "ticker": {"type": "string"},
                "target_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["product_code", "ticker", "target_date"],
        },
    },
    {
        "name": "get_seasonal_stats",
        "description": "How a contract has behaved seasonally: at several fixed points before "
                        "expiration (90/60/30/14/7 calendar days out), returns the plain average/min/max "
                        "settlement price across its prior-year analogs (e.g. this Dec corn vs. the last "
                        "4 Decembers), plus a 'harmonic_fit' value at each checkpoint — a smooth Fourier "
                        "regression fit across all those years that separates each year's own price level "
                        "from the shared seasonal shape, so it isn't skewed the way a plain average can be "
                        "by one outlier year (a drought spike, a trade-war shock). Prefer harmonic_fit over "
                        "average when asked for 'the' seasonal price or a smoothed/typical seasonal level; "
                        "use average/min/max when asked for the historical range itself.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_code": {"type": "string", "enum": ["ZC", "ZS", "ZW", "KE"]},
                "ticker": {"type": "string", "description": "the reference contract, e.g. ZCZ6"},
                "years_back": {"type": "integer", "description": "prior contract years to include, default 4, max 5"},
            },
            "required": ["product_code", "ticker"],
        },
    },
    {
        "name": "get_wasde_dates",
        "description": "USDA WASDE report release dates within a date range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "get_nass_dates",
        "description": "USDA NASS report release dates within a date range, optionally filtered to "
                        "one report type (Crop Production, Grain Stocks, Acreage, Prospective Plantings, "
                        "Winter Wheat & Canola Seedings, Small Grains Summary, Crop Progress).",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                "report_type": {"type": "string"},
            },
            "required": ["start_date", "end_date"],
        },
    },
]


def run_chat(client, messages: list[dict], tool_dispatch, system_prompt: str) -> str:
    """Send `messages` (ending in the new user turn) to Claude, executing any
    tool calls it makes, and return the final assistant text for this turn."""
    messages = list(messages)
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(
            model=MODEL, max_tokens=1500, system=system_prompt,
            tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                result = tool_dispatch(block.name, block.input)
            except Exception as e:
                result = {"error": str(e)}
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })
        messages.append({"role": "user", "content": tool_results})

    return ("I hit the tool-call limit for this turn without reaching a final answer — "
            "try breaking the question into smaller parts.")
