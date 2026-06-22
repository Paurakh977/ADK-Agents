"""LiteLLM agent with tools + ADK tool-confirmation (terminal UI).

╔══════════════════════════════════════════════════════════════════╗
║            THREE TOOL-CONFIRMATION PATTERNS IN ADK              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. BOOLEAN — FunctionTool(fn, require_confirmation=True)        ║
║     • The function itself is UNTOUCHED.                          ║
║     • ADK wraps it and asks for Y/N before calling it.           ║
║     • Used for: get_weather                                      ║
║                                                                  ║
║  2. ADVANCED (hint + payload) — inside the function body         ║
║     • The function calls tool_context.request_confirmation(      ║
║           hint="...", payload={...})  on first invocation.       ║
║     • ADK pauses, collects user's filled-in payload, resumes.    ║
║     • On second invocation tool_context.tool_confirmation holds  ║
║       .confirmed (bool) + .payload (the filled dict).            ║
║     • Used for: reimburse_expense                                ║
║                                                                  ║
║  3. DYNAMIC BOOLEAN — require_confirmation=<async fn>            ║
║     • A provider function decides at runtime if confirmation     ║
║       is needed (e.g. only when amount > 1000).                  ║
║     • Used for: send_email (only if recipient is external)       ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  HOW collect_user_input WORKS (LongRunningFunctionTool HITL)     ║
║                                                                  ║
║  1. Agent calls _collect_user_input_fn with message + options    ║
║     OR message + response_schema.                                ║
║  2. The function returns None and sets skip_summarization=True.  ║
║     ADK marks its call ID in event.long_running_tool_ids.        ║
║  3. _run_prompt() detects the call ID in long_running_tool_ids.  ║
║  4. _prompt_user_input() prints the terminal form:               ║
║       • options list  → numbered menu, user picks one            ║
║       • response_schema → one prompt per property, required /    ║
║         optional + type shown, Enter=skip optional fields        ║
║  5. We resume by calling runner.run_async() again on the SAME    ║
║     session with a FunctionResponse for _collect_user_input_fn.  ║
║     The response payload is {"result": <collected_data>}.        ║
║  6. The agent continues with the user's answers in context.      ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  HOW THE CONFIRMATION LOOP WORKS (terminal "UI")                 ║
║                                                                  ║
║  1. Agent wants to call a tool.                                  ║
║  2. ADK emits an adk_request_confirmation FunctionCall event.    ║
║  3. _run_prompt() detects it in the event stream.                ║
║  4. _prompt_user_confirmation() prints the terminal UI:          ║
║       • For boolean  → just asks Y/N                             ║
║       • For advanced → asks Y/N + prompts for each payload field ║
║  5. We resume by calling runner.run_async() again on the SAME    ║
║     session with a FunctionResponse for adk_request_confirmation.║
║  6. The agent continues from exactly where it stopped.           ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import string
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional
import httpx

# ── ADK logging (only show LLM Request / Response blocks) ──────────────────
class _LLMBlockOnly(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return msg.startswith("LLM Request") or msg.startswith("LLM Response")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s -\n%(message)s",
)
_adk_log = logging.getLogger("google_adk")
_adk_log.setLevel(logging.DEBUG)
_adk_log.addFilter(_LLMBlockOnly())

os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
# ───────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.adk.tools.long_running_tool import LongRunningFunctionTool


load_dotenv(override=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def _collect_user_input_fn(
    message: str,
    tool_context: ToolContext,
    options: Optional[list] = None,
    response_schema: Optional[dict] = None,
) -> Optional[str]:
    """Collect input from the user via a dynamic terminal form.

    This is a LongRunningFunctionTool — it acts as a SIGNAL only.
    The function returns None immediately; the actual human interaction
    is handled by the application layer (_prompt_user_input) which detects
    this call via event.long_running_tool_ids, presents the UI, and sends
    back a FunctionResponse with the collected data.

    Use `options` for simple button/choice selection (list of strings).
    Use `response_schema` (JSON Schema, type:object) to render a multi-field
    form where each property becomes an input field.

    Example — simple choice:
        options=["Low", "Medium", "High"]

    Example — structured form:
        response_schema={
            "type": "object",
            "properties": {
                "title":    {"type": "string",  "description": "Brief summary"},
                "priority": {"type": "string",  "enum": ["LOW","MEDIUM","HIGH"]},
                "notify":   {"type": "boolean", "description": "Email notification"}
            },
            "required": ["title", "priority"]
        }
    """
    # Signal the framework to skip summarization — the UI layer handles display.
    tool_context.actions.skip_summarization = True
    # Return None: real data comes back from the FunctionResponse the UI sends.
    return None


# Wrap as LongRunningFunctionTool so ADK marks its call ID in
# event.long_running_tool_ids, letting us detect and intercept it.
collect_user_input = LongRunningFunctionTool(_collect_user_input_fn)


def get_current_time(timezone: str = "UTC") -> dict:
    """Return the current date and time for a given timezone.

    Args:
        timezone: IANA timezone name, e.g. 'America/New_York', 'Asia/Kathmandu', 'UTC'.
    """
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return {
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": timezone,
            "day_of_week": now.strftime("%A"),
        }
    except Exception as e:
        return {"error": str(e)}


def calculate(expression: str) -> dict:
    """Evaluate a safe mathematical expression.

    Supports: +, -, *, /, **, %, sqrt(), sin(), cos(), tan(), log(), log10(),
              abs(), round(), floor(), ceil(), pi, e

    Args:
        expression: Math expression string, e.g. 'sqrt(144) + 2 ** 8'
    """
    allowed = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "abs": abs, "round": round, "floor": math.floor,
        "ceil": math.ceil, "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


def get_weather(city: str) -> dict:
    """Fetch current weather for a city using the free Open-Meteo + geocoding API.

    Args:
        city: City name, e.g. 'Kathmandu', 'New York', 'Tokyo'
    """
    try:
        geo = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        ).json()

        if not geo.get("results"):
            return {"error": f"City '{city}' not found."}

        r = geo["results"][0]
        lat, lon, name = r["latitude"], r["longitude"], r["name"]

        weather = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,apparent_temperature,"
                    "relative_humidity_2m,wind_speed_10m,weathercode"
                ),
                "timezone": "auto",
            },
            timeout=10,
        ).json()

        cur = weather["current"]
        code = cur.get("weathercode", 0)
        descriptions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
            55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
            85: "Snow showers", 95: "Thunderstorm", 99: "Thunderstorm with hail",
        }
        return {
            "city": name,
            "temperature_c": cur["temperature_2m"],
            "feels_like_c": cur["apparent_temperature"],
            "humidity_pct": cur["relative_humidity_2m"],
            "wind_speed_kmh": cur["wind_speed_10m"],
            "description": descriptions.get(code, f"Weather code {code}"),
        }
    except Exception as exc:
        return {"error": str(exc)}


def reimburse_expense(
    employee_name: str,
    amount: float,
    description: str,
    tool_context: ToolContext,
) -> dict:
    """Submit an expense reimbursement request.  Always requires finance approval.

    This is the ADVANCED CONFIRMATION pattern:
      • The function itself calls tool_context.request_confirmation(hint, payload)
        on the FIRST invocation (when tool_context.tool_confirmation is None).
      • ADK pauses, asks the user, collects the filled payload, then re-invokes
        the function with tool_context.tool_confirmation populated.
      • On the SECOND invocation we read .confirmed and .payload and do the work.

    Args:
        employee_name: Full name of the employee requesting reimbursement.
        amount:        Claimed expense amount in USD.
        description:   What the expense was for.
    """
    tool_confirmation = tool_context.tool_confirmation

    if not tool_confirmation:
        tool_context.request_confirmation(
            hint=(
                f"Finance approval required.\n"
                f"  Employee   : {employee_name}\n"
                f"  Claimed    : ${amount:.2f}\n"
                f"  Description: {description}\n"
                f"Please fill in the approval details below."
            ),
            payload={
                "approved_amount":    amount,
                "cost_center":        "GENERAL",
                "requires_receipt":   True,
            },
        )
        return {"status": "AWAITING_FINANCE_APPROVAL"}

    if not tool_confirmation.confirmed:
        return {
            "status": "REJECTED",
            "message": f"Expense request for {employee_name} was rejected by finance.",
        }

    payload          = tool_confirmation.payload
    approved_amount  = float(payload.get("approved_amount", 0))
    cost_center      = str(payload.get("cost_center", "GENERAL"))
    requires_receipt = bool(payload.get("requires_receipt", True))

    if approved_amount <= 0:
        return {
            "status": "REJECTED",
            "message": "Approved amount is $0 — request denied.",
        }

    return {
        "status":           "APPROVED",
        "employee":         employee_name,
        "claimed_amount":   amount,
        "approved_amount":  approved_amount,
        "cost_center":      cost_center,
        "requires_receipt": requires_receipt,
        "message": (
            f"${approved_amount:.2f} approved for {employee_name} "
            f"(cost centre: {cost_center}). "
            + ("Receipt required." if requires_receipt else "No receipt needed.")
        ),
    }


def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web using the free DuckDuckGo Instant Answer API.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 5).
    """
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
            timeout=10,
        ).json()
        abstract = resp.get("AbstractText") or resp.get("Answer") or ""
        raw = resp.get("RelatedTopics", [])
        results = []
        for item in raw:
            if len(results) >= max_results:
                break
            if "Text" in item and "FirstURL" in item:
                results.append({"title": item["Text"][:120], "url": item["FirstURL"]})
            for sub in item.get("Topics", []):
                if len(results) >= max_results:
                    break
                if "Text" in sub and "FirstURL" in sub:
                    results.append({"title": sub["Text"][:120], "url": sub["FirstURL"]})
        return {"query": query, "abstract": abstract, "results": results}
    except Exception as exc:
        return {"query": query, "error": str(exc)}


async def _only_for_external(
    recipient: str,
    subject: str,
    body: str,
) -> bool:
    """Confirmation provider for send_email — only external recipients need Y/N."""
    return "@company.com" not in recipient.lower()


def send_email(recipient: str, subject: str, body: str) -> dict:
    """Send an email (simulated).  Requires confirmation for external recipients.

    Args:
        recipient: Email address to send to.
        subject:   Email subject line.
        body:      Email body text.
    """
    return {
        "status":    "sent",
        "recipient": recipient,
        "subject":   subject,
        "message":   f"Email sent to {recipient}.",
    }


def generate_random_data(
    data_type: str = "string",
    count: int = 1,
    min_val: float = 0,
    max_val: float = 100,
    length: int = 12,
) -> dict:
    """Generate random data of various types.

    Args:
        data_type: One of 'string', 'number', 'integer', 'uuid', 'password', 'hex'
        count:     How many values to generate (1–20).
        min_val:   Minimum value for numeric types.
        max_val:   Maximum value for numeric types.
        length:    Character length for string/password/hex types.
    """
    import uuid as _uuid
    count = max(1, min(count, 20))
    results = []
    for _ in range(count):
        if data_type == "integer":
            results.append(random.randint(int(min_val), int(max_val)))
        elif data_type == "number":
            results.append(round(random.uniform(min_val, max_val), 4))
        elif data_type == "uuid":
            results.append(str(_uuid.uuid4()))
        elif data_type == "password":
            chars = string.ascii_letters + string.digits + "!@#$%^&*"
            results.append("".join(random.choices(chars, k=length)))
        elif data_type == "hex":
            results.append("".join(random.choices("0123456789abcdef", k=length)))
        else:
            chars = string.ascii_letters + string.digits
            results.append("".join(random.choices(chars, k=length)))
    return {"data_type": data_type, "count": count, "results": results}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL + AGENT
# ══════════════════════════════════════════════════════════════════════════════

model = LiteLlm(
    model="openai/mercury-2",
    api_key=os.getenv("MERCURY_API_KEY"),
    api_base="https://api.inceptionlabs.ai/v1",
    max_tokens=8000,
    stream=True,
    extra_body={"reasoning_effort": "high"},
)

root_agent = LlmAgent(
    name="root_agent",
    model=model,
    instruction=(
        "You are a helpful assistant with access to tools. "
        "Use them whenever they help answer the user's question accurately.\n\n"
        "## collect_user_input — Interactive Human Input Tool\n"
        "Use this tool whenever you need to gather additional information from the user "
        "interactively — for example, to fill in missing details before calling another tool.\n\n"
        "You must call it with either:\n"
        "  • `options` (list of strings) — for a simple choice menu, e.g. [\"Low\", \"Medium\", \"High\"]\n"
        "  • `response_schema` (JSON Schema object) — for a multi-field form. Each property "
        "becomes one input field. Mark required fields in the 'required' array.\n\n"
        "Always include a clear `message` explaining what you need and why.\n\n"
        "Example — collecting ticket details:\n"
        "  collect_user_input(\n"
        "    message='Please fill in the support ticket details:',\n"
        "    response_schema={\n"
        "      'type': 'object',\n"
        "      'properties': {\n"
        "        'title':    {'type': 'string', 'description': 'Brief summary'},\n"
        "        'priority': {'type': 'string', 'enum': ['LOW','MEDIUM','HIGH']},\n"
        "        'notify':   {'type': 'boolean', 'description': 'Email notification (optional)'}\n"
        "      },\n"
        "      'required': ['title', 'priority']\n"
        "    }\n"
        "  )\n"
    ),
    tools=[
        get_current_time,
        calculate,
        web_search,
        generate_random_data,

        # ── Human-in-the-loop input collection ──────────────────────────────
        # The LLM calls this with message + options OR message + response_schema.
        # The terminal UI collects the answer and sends it back as a
        # FunctionResponse, then the agent continues with the data in context.
        collect_user_input,

        # ── PATTERN 1: Boolean confirmation ─────────────────────────────────
        FunctionTool(get_weather, require_confirmation=True),

        # ── PATTERN 2: Advanced confirmation (hint + payload) ────────────────
        reimburse_expense,

        # ── PATTERN 3: Dynamic boolean confirmation ──────────────────────────
        FunctionTool(send_email, require_confirmation=_only_for_external),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL UI — collect_user_input handler
# ══════════════════════════════════════════════════════════════════════════════

def _coerce_value(raw: str, prop_schema: dict):
    """Coerce a raw string into the type declared in a JSON Schema property."""
    typ = prop_schema.get("type", "string")
    if typ == "boolean":
        return raw.lower() in ("true", "1", "yes", "y")
    if typ == "integer":
        return int(raw)
    if typ == "number":
        return float(raw)
    # string / anything else — return as-is
    return raw


async def _prompt_user_input(fc: types.FunctionCall) -> types.Content:
    """
    Terminal UI for the collect_user_input LongRunningFunctionTool.

    Called when we detect the function call ID in event.long_running_tool_ids.
    Handles two modes:

      • options (list)         → numbered menu, user picks by number or text
      • response_schema (dict) → field-by-field form, required/optional shown,
                                 type + enum values displayed as hints

    Returns a Content(FunctionResponse) that resumes the agent with the
    collected data under {"result": <value>}.

    ── What fc.args contains ─────────────────────────────────────────────────
    {
      "message":         "Please provide the following details…",
      "options":         ["Option A", "Option B"],   ← OR
      "response_schema": {                           ← but not both
          "type": "object",
          "properties": {
              "field_name": {
                  "type":        "string",
                  "description": "Human hint",
                  "enum":        ["A", "B"]          ← optional
              },
              ...
          },
          "required": ["field_name", ...]
      }
    }
    ──────────────────────────────────────────────────────────────────────────
    """
    args            = fc.args or {}
    message         = args.get("message", "Please provide the requested information.")
    options: list   = args.get("options") or []
    schema: dict    = args.get("response_schema") or {}

    SEP = "─" * 62
    print(f"\n{SEP}")
    print("  📝  USER INPUT REQUIRED")
    print(SEP)
    print(f"  {message}")
    print(SEP)

    result = None  # will hold the collected value to send back

    # ── MODE A: simple options list ──────────────────────────────────────────
    if options:
        print()
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        print()
        while True:
            raw = input("  Your choice (number or text): ").strip()
            if not raw:
                print("  Please enter a number or type your choice.")
                continue
            # Accept numeric index
            try:
                idx = int(raw)
                if 1 <= idx <= len(options):
                    result = options[idx - 1]
                    break
                else:
                    print(f"  Please enter a number between 1 and {len(options)}.")
            except ValueError:
                # Accept verbatim text if it matches one of the options (case-insensitive)
                matches = [o for o in options if o.lower() == raw.lower()]
                if matches:
                    result = matches[0]
                    break
                else:
                    # Accept free text as-is (agent asked for a choice but user typed something)
                    result = raw
                    break

    # ── MODE B: JSON Schema form ─────────────────────────────────────────────
    elif schema.get("type") == "object" and schema.get("properties"):
        properties: dict = schema["properties"]
        required_fields: list = schema.get("required") or []
        collected: dict = {}

        print()
        for field_name, prop in properties.items():
            is_required = field_name in required_fields
            field_type  = prop.get("type", "string")
            description = prop.get("description", "")
            enum_vals   = prop.get("enum")

            # Build a hint string for the prompt line
            hint_parts = [field_type]
            if enum_vals:
                hint_parts.append(f"one of: {', '.join(str(v) for v in enum_vals)}")
            if description:
                hint_parts.append(description)
            hint = " | ".join(hint_parts)

            req_marker = " [REQUIRED]" if is_required else " [optional, Enter to skip]"
            prompt_str = f"  {field_name}{req_marker}\n    ({hint})\n  > "

            while True:
                raw = input(prompt_str).strip()

                if raw == "":
                    if is_required:
                        print(f"  ⚠  '{field_name}' is required — please enter a value.")
                        continue
                    else:
                        # Skip optional field — don't include it in the result
                        break
                else:
                    # Validate enum if present
                    if enum_vals and raw not in enum_vals:
                        # Try case-insensitive match
                        ci = [v for v in enum_vals if str(v).lower() == raw.lower()]
                        if ci:
                            raw = str(ci[0])
                        else:
                            print(f"  ⚠  Must be one of: {', '.join(str(v) for v in enum_vals)}")
                            continue
                    # Coerce type
                    try:
                        collected[field_name] = _coerce_value(raw, prop)
                    except (ValueError, TypeError):
                        print(f"  ⚠  Could not parse '{raw}' as {field_type}. Storing as string.")
                        collected[field_name] = raw
                    break

        result = collected

    # ── FALLBACK: plain text input ───────────────────────────────────────────
    else:
        result = input("  Your answer: ").strip()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  ✓ Input captured:")
    if isinstance(result, dict):
        for k, v in result.items():
            print(f"    {k} = {v!r}")
    else:
        print(f"    {result!r}")
    print(f"{SEP}\n")

    # ── Build FunctionResponse ────────────────────────────────────────────────
    #
    # Rules for LongRunningFunctionTool resume:
    #   id   → fc.id   (the ID of the _collect_user_input_fn FunctionCall)
    #   name → "_collect_user_input_fn"  (the actual tool function name)
    #   response → {"result": <collected_value>}
    #
    # The agent receives this as the tool's return value and can use the data
    # to continue its reasoning (e.g. call another tool with the collected info).
    #
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=fc.id,
                    name="_collect_user_input_fn",
                    response={"result": result},
                )
            )
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL UI — tool-confirmation handler (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

async def _prompt_user_confirmation(fc: types.FunctionCall) -> types.Content:
    """
    Terminal "UI" for tool confirmation.  Called whenever ADK emits an
    adk_request_confirmation FunctionCall event.

    Handles both confirmation types:
      • Boolean  (payload_template is None)  → just asks Y/N
      • Advanced (payload_template is dict)  → asks Y/N, then prompts for
                                               each field in the payload

    Returns the Content(FunctionResponse) that resumes the agent.
    """
    args            = fc.args or {}
    original_call   = args.get("originalFunctionCall", {})
    tool_conf_args  = args.get("toolConfirmation", {})

    tool_name        = original_call.get("name", "unknown_tool")
    tool_args        = original_call.get("args", {})
    hint             = tool_conf_args.get("hint", "This tool requires your confirmation.")
    payload_template: dict | None = tool_conf_args.get("payload") or None

    confirmation_type = "ADVANCED (hint + payload)" if payload_template else "BOOLEAN (Y/N only)"

    SEP = "─" * 62
    print(f"\n{SEP}")
    print(f"  ⚠  TOOL CONFIRMATION REQUIRED  [{confirmation_type}]")
    print(SEP)
    print(f"  Tool : {tool_name}")
    if tool_args:
        for k, v in tool_args.items():
            print(f"  Arg  : {k} = {v!r}")
    print(f"  Hint : {hint}")
    if payload_template:
        print(f"\n  Expected payload fields (you will fill these in if approved):")
        for k, v in payload_template.items():
            print(f"    • {k:20s}  (default: {v!r})")
    print(SEP)

    while True:
        answer = input("  Approve? [y/n]: ").strip().lower()
        if answer in ("y", "yes", "n", "no"):
            break
        print("  Please type  y  or  n")
    confirmed = answer in ("y", "yes")

    response_payload: dict = {}
    if confirmed and payload_template:
        print(f"\n  Fill in approval details  (press Enter to keep the default):")
        for key, default_val in payload_template.items():
            raw = input(f"    {key} [{default_val!r}]: ").strip()
            if raw == "":
                response_payload[key] = default_val
            else:
                try:
                    if isinstance(default_val, bool):
                        response_payload[key] = raw.lower() in ("true", "1", "yes", "y")
                    elif isinstance(default_val, int):
                        response_payload[key] = int(raw)
                    elif isinstance(default_val, float):
                        response_payload[key] = float(raw)
                    else:
                        response_payload[key] = raw
                except ValueError:
                    print(f"    (could not convert '{raw}' to {type(default_val).__name__}, using as string)")
                    response_payload[key] = raw

    verdict = "✓ APPROVED" if confirmed else "✗ REJECTED"
    print(f"\n  {verdict}")
    if response_payload:
        print(f"  Payload sent back to tool:")
        for k, v in response_payload.items():
            print(f"    {k} = {v!r}")
    print(f"{SEP}\n")

    response_data: dict = {"confirmed": confirmed}
    if response_payload:
        response_data["payload"] = response_payload

    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=fc.id,
                    name="adk_request_confirmation",
                    response=response_data,
                )
            )
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

async def _run_prompt(
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    *,
    prompt: str | None = None,
    new_message: types.Content | None = None,
) -> None:
    """
    Run one turn.

    Handles two kinds of mid-turn interrupts automatically:

    1. adk_request_confirmation  — tool-confirmation (boolean / advanced)
       Detected via: part.function_call.name == "adk_request_confirmation"

    2. collect_user_input (LongRunningFunctionTool HITL)
       Detected via: part.function_call.id in event.long_running_tool_ids
       The tool function name is "_collect_user_input_fn".

    When either is detected, the handler prompts the user, then resumes
    by calling run_async again on the SAME session with the FunctionResponse.
    """
    if new_message is None:
        assert prompt is not None, "Either prompt or new_message must be provided"
        new_message = types.Content(
            role="user", parts=[types.Part.from_text(text=prompt)]
        )

    # These will be set if we detect an interrupt during the event loop
    confirmation_fc: types.FunctionCall | None = None   # adk_request_confirmation
    user_input_fc:   types.FunctionCall | None = None   # _collect_user_input_fn

    saw_partial = False

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        if not event.content:
            continue

        # IDs of any LongRunningFunctionTool calls in this event
        long_running_ids: set[str] = set(event.long_running_tool_ids or [])

        for part in event.content.parts:
            fc = getattr(part, "function_call", None)

            if fc:
                fc_name = getattr(fc, "name", None)
                fc_id   = getattr(fc, "id",   None)

                # ── Detect tool-confirmation request ─────────────────────────
                if fc_name == "adk_request_confirmation":
                    confirmation_fc = fc
                    continue

                # ── Detect collect_user_input (LongRunningFunctionTool) ──────
                # The function is named _collect_user_input_fn; ADK puts its
                # call ID in event.long_running_tool_ids to signal "pending".
                if fc_id and fc_id in long_running_ids:
                    user_input_fc = fc
                    continue

            # ── Stream text to terminal ──────────────────────────────────────
            text = getattr(part, "text", None)
            if not text:
                continue
            if event.partial:
                sys.stdout.write(text)
                sys.stdout.flush()
                saw_partial = True
            elif not saw_partial:
                sys.stdout.write(text)
                sys.stdout.flush()

    # ── Handle pending interrupts (in priority order) ────────────────────────

    if user_input_fc:
        # ── collect_user_input: show form, send answer, resume ───────────────
        input_content = await _prompt_user_input(user_input_fc)
        print("Agent: ", end="", flush=True)
        await _run_prompt(
            runner, user_id, session_id,
            new_message=input_content,
        )

    elif confirmation_fc:
        # ── adk_request_confirmation: show Y/N (+ payload), resume ──────────
        confirmation_content = await _prompt_user_confirmation(confirmation_fc)
        print("Agent: ", end="", flush=True)
        await _run_prompt(
            runner, user_id, session_id,
            new_message=confirmation_content,
        )

    else:
        print()  # newline after streamed response


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    app_name = "litellm_demo"
    user_id  = "user_1"
    runner   = InMemoryRunner(agent=root_agent, app_name=app_name)
    session  = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )

    tools_info = (
        "get_current_time, calculate, web_search, generate_random_data | "
        "HITL INPUT: collect_user_input [options or schema form] | "
        "CONFIRMATION: get_weather [boolean], reimburse_expense [hint+payload], "
        "send_email [dynamic boolean]"
    )
    print(f"\nAgent ready — tools: {tools_info}")
    print("Type /quit to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt.lower() in ("/quit", "/exit"):
            break

        print("Agent: ", end="", flush=True)
        await _run_prompt(runner, user_id, session.id, prompt=prompt)
        print()


if __name__ == "__main__":
    asyncio.run(main())