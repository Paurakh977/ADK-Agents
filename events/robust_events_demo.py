# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License").
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

"""
Multi-agent, multi-tool ADK demo (non-streaming) using LiteLLM Mercury.

This file is a COMPLETE, EXHAUSTIVE event inspector. Every field defined in
the ADK Event class — content, actions, metadata, control signals, errors,
agent routing, thoughts tokens, node_info, grounding, citations, etc. —
is printed and explained for every event in the stream.

Architecture (an "agent team"):

    root_agent  ("coordinator")
      |-- greeting_agent   -> tool: say_hello
      |-- farewell_agent   -> tool: say_goodbye
      `-- (handles weather / capital / unit-pref itself)
            -> tools: get_weather_stateful, get_capital_city,
                      set_temperature_unit_preference, get_last_state_key

Special REPL commands (type instead of a message):
    events  -> Re-dump every raw event from the last turn
    state   -> Print current session.state as JSON
    history -> Print every event EVER recorded in session.events
    tools   -> Print agent manifest with all tools and signatures
    stats   -> Print cumulative session statistics
    exit / quit

Docs consulted:
  https://adk.dev/events/
  https://adk.dev/tutorials/agent-team/
  https://adk.dev/sessions/state/
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
import warnings
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

load_dotenv()

APP_NAME = "mercury_multi_agent_events_demo"
USER_ID = "user_1"
SESSION_ID = "session_001"

# ANSI colour helpers (terminal only; degrade gracefully if piped)
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_MAG = "\033[35m"
_RESET = "\033[0m"


def _h(colour: str, text: str) -> str:
    """Wrap text in colour codes."""
    return f"{colour}{text}{_RESET}"


def _annotate_state_key(key: str) -> str:
    """Annotate a state key with its scope prefix per ADK docs."""
    if key.startswith("app:"):
        return f"{key}  [APP-SCOPED — persists across users]"
    elif key.startswith("user:"):
        return f"{key}  [USER-SCOPED — persists across sessions]"
    elif key.startswith("temp:"):
        return f"{key}  [TEMP — current invocation only]"
    else:
        return f"{key}  [SESSION-SCOPED]"


# ─────────────────────────────────────────────────────────────────────────────
# CUMULATIVE SESSION STATS
# ─────────────────────────────────────────────────────────────────────────────
_session_stats = {
    "total_turns": 0,
    "total_tokens": 0,
    "total_thought_tokens": 0,
    "total_tool_calls": 0,
    "agent_invocation_counts": {},
    "tool_invocation_counts": {},
    "transfer_log": [],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────


def make_mercury_model() -> LiteLlm:
    """One fresh LiteLlm Mercury instance per agent (avoids shared mutable state)."""
    return LiteLlm(
        model="openai/mercury-2",
        api_key=os.getenv("MERCURY_API_KEY"),
        api_base="https://api.inceptionlabs.ai/v1",
        max_tokens=8000,
        stream=False,
        extra_body={"reasoning_effort": "high"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOOLS
# ─────────────────────────────────────────────────────────────────────────────


def get_capital_city(country: str) -> dict:
    """Looks up the capital city of a country.

    Args:
        country: The country name, e.g. "Japan", "France", "Nepal".

    Returns:
        dict: {"status": "success", "capital": "..."} or error dict.
    """
    print(f"    {_h(_BLUE, '[tool]')} get_capital_city(country={country!r})")
    db = {
        "japan": "Tokyo",
        "france": "Paris",
        "nepal": "Kathmandu",
        "usa": "Washington, D.C.",
        "united states": "Washington, D.C.",
        "uk": "London",
        "germany": "Berlin",
        "india": "New Delhi",
    }
    key = country.strip().lower()
    if key in db:
        return {"status": "success", "capital": db[key]}
    return {"status": "error", "error_message": f"No data for '{country}'."}


def get_weather_stateful(city: str, tool_context: ToolContext) -> dict:
    """Retrieves a mock weather report, honouring the user's unit preference.

    Args:
        city: City name, e.g. "London", "Kathmandu".

    Returns:
        dict: {"status": "success", "report": "..."} or error dict.
    """
    unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")
    print(
        f"    {_h(_BLUE, '[tool]')} get_weather_stateful(city={city!r}) | unit from state={unit!r}"
    )

    db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
        "kathmandu": {"temp_c": 22, "condition": "partly cloudy"},
        "berlin": {"temp_c": 12, "condition": "overcast"},
        "paris": {"temp_c": 17, "condition": "clear"},
    }
    key = city.lower().replace(" ", "")
    if key not in db:
        return {"status": "error", "error_message": f"No weather data for '{city}'."}

    temp_c = db[key]["temp_c"]
    if unit == "Fahrenheit":
        val, sym = (temp_c * 9 / 5) + 32, "°F"
    else:
        val, sym = temp_c, "°C"

    report = (
        f"The weather in {city.title()} is {db[key]['condition']} "
        f"with a temperature of {val:.0f}{sym}."
    )
    # Write to session state — shows up as state_delta on next event
    tool_context.state["last_city_checked"] = city
    return {"status": "success", "report": report}


def set_temperature_unit_preference(unit: str, tool_context: ToolContext) -> dict:
    """Sets the user's preferred temperature unit (Celsius / Fahrenheit).

    Args:
        unit: "Celsius" or "Fahrenheit".

    Returns:
        dict: {"status": "success", "message": "..."}
    """
    normalized = "Fahrenheit" if unit.strip().lower().startswith("f") else "Celsius"
    print(
        f"    {_h(_BLUE, '[tool]')} set_temperature_unit_preference(unit={normalized!r})"
    )
    tool_context.state["user_preference_temperature_unit"] = normalized
    return {"status": "success", "message": f"Preference set to {normalized}."}


def say_hello(name: str | None = None) -> str:
    """Greets the user by name (optional).

    Args:
        name: Optional name to personalise the greeting.

    Returns:
        str: A greeting message.
    """
    print(f"    {_h(_BLUE, '[tool]')} say_hello(name={name!r})")
    return f"Hello, {name}!" if name else "Hello there!"


def say_goodbye() -> str:
    """Says goodbye at the end of a conversation."""
    print(f"    {_h(_BLUE, '[tool]')} say_goodbye()")
    return "Goodbye! Have a great day."


def get_last_state_key(key: str, tool_context: ToolContext) -> dict:
    """Retrieves the current value of any session state key.

    Args:
        key: The session state key to read (e.g. 'user_preference_temperature_unit',
             'last_city_checked', 'last_agent_response').

    Returns:
        dict: {"status": "success", "key": key, "value": <value>} or error dict.
    """
    print(f"    {_h(_BLUE, '[tool]')} get_last_state_key(key={key!r})")
    val = tool_context.state.get(key, "__NOT_FOUND__")
    if val == "__NOT_FOUND__":
        return {
            "status": "error",
            "error_message": f"Key '{key}' not in session state.",
        }
    return {"status": "success", "key": key, "value": val}


# ─────────────────────────────────────────────────────────────────────────────
# 3. AGENTS
# ─────────────────────────────────────────────────────────────────────────────
# 3. AGENTS
# ─────────────────────────────────────────────────────────────────────────────

greeting_agent = LlmAgent(
    name="greeting_agent",
    model=make_mercury_model(),
    description="Handles simple greetings and hellos using the 'say_hello' tool.",
    instruction=(
        "You are the Greeting Agent. Your ONLY task is to provide a friendly "
        "greeting using the 'say_hello' tool. If the user gave their name, pass "
        "it to the tool. Do not do anything else."
    ),
    tools=[say_hello],
)

farewell_agent = LlmAgent(
    name="farewell_agent",
    model=make_mercury_model(),
    description="Handles farewells and goodbyes using the 'say_goodbye' tool.",
    instruction=(
        "You are the Farewell Agent. Your ONLY task is to provide a polite "
        "goodbye using the 'say_goodbye' tool. Do not do anything else."
    ),
    tools=[say_goodbye],
)

root_agent = LlmAgent(
    name="root_agent",
    model=make_mercury_model(),
    description=(
        "Main coordinator. Handles weather/capital lookups itself; "
        "delegates greetings/farewells to specialist sub-agents."
    ),
    instruction=(
        "You are the main coordinating assistant. "
        "You have tools: 'get_weather_stateful', 'get_capital_city', "
        "'set_temperature_unit_preference', 'get_last_state_key'. "
        "Delegate greetings (hi/hello) to 'greeting_agent'. "
        "Delegate farewells (bye/goodbye) to 'farewell_agent'. "
        "Handle weather, capital city, and unit-preference yourself. "
        "Use 'get_last_state_key' to look up any value from session state when the user asks."
    ),
    tools=[
        get_weather_stateful,
        get_capital_city,
        set_temperature_unit_preference,
        get_last_state_key,
    ],
    sub_agents=[greeting_agent, farewell_agent],
    output_key="last_agent_response",
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXHAUSTIVE EVENT INSPECTOR
#    Covers every documented field from https://adk.dev/events/
#    plus every field visible in real Mercury event objects.
# ─────────────────────────────────────────────────────────────────────────────


def _safe_get(obj, *attrs, default=None):
    """Safely drill into nested attributes without raising AttributeError."""
    for attr in attrs:
        if obj is None:
            return default
        obj = getattr(obj, attr, default)
    return obj


def _classify_payload(event) -> str:
    """Return a short human-readable payload category string."""
    calls = event.get_function_calls() or []
    responses = event.get_function_responses() or []

    if calls:
        names = ", ".join(c.name for c in calls)
        return f"TOOL CALL REQUEST  [{names}]"
    if responses:
        names = ", ".join(r.name for r in responses)
        return f"TOOL RESULT        [{names}]"

    parts = _safe_get(event, "content", "parts", default=[])
    if parts:
        # Check all parts for text
        texts = [p.text for p in parts if getattr(p, "text", None)]
        if texts:
            kind = "STREAMING TEXT CHUNK" if event.partial else "COMPLETE TEXT MESSAGE"
            return kind
        # Check for thought / reasoning parts
        thoughts = [p for p in parts if getattr(p, "thought", None)]
        if thoughts:
            return "THOUGHT / REASONING CHUNK"
        return "OTHER CONTENT (non-text parts)"

    # No content at all — might be a pure control/state event
    actions = getattr(event, "actions", None)
    if actions:
        if getattr(actions, "transfer_to_agent", None):
            return "CONTROL: AGENT TRANSFER"
        if getattr(actions, "escalate", None):
            return "CONTROL: ESCALATE (loop terminate)"
        if getattr(actions, "state_delta", None) or getattr(
            actions, "artifact_delta", None
        ):
            return "STATE / ARTIFACT UPDATE ONLY"
    return "EMPTY / PURE SIGNAL"


def describe_event(event, index: int) -> None:
    """
    Print a structured, exhaustive breakdown of ONE ADK Event.

    Covers (per https://adk.dev/events/ + observed Mercury fields):
      ┌─ Identity & Routing
      │   id, invocation_id, author, branch, timestamp, node_info, interaction_id,
      │   isolation_scope, model_version
      ├─ Payload Classification
      │   get_function_calls(), get_function_responses(), text parts, thought parts
      ├─ All Content Parts (every item in event.content.parts)
      │   text, function_call (name+args), function_response (name+result), thought
      ├─ Streaming / Completion State
      │   partial, finish_reason, turn_complete, turn_complete_reason, interrupted
      ├─ Actions (side-effects & control-flow)  [EventActions.*]
      │   state_delta, artifact_delta,
      │   transfer_to_agent, escalate, skip_summarization,
      │   requested_auth_configs, requested_tool_confirmations,
      │   compaction, end_of_agent, agent_state,
      │   rewind_before_invocation_id, route, render_ui_widgets, set_model_response
      ├─ Errors
      │   error_code, error_message
      ├─ Token Usage
      │   prompt_token_count, candidates_token_count, thoughts_token_count,
      │   cached_content_token_count, total_token_count
      ├─ Misc Metadata
      │   long_running_tool_ids, grounding_metadata, custom_metadata,
      │   citation_metadata, cache_metadata, avg_logprobs, logprobs_result,
      │   live_session_resumption_update, live_session_id, go_away,
      │   input_transcription, output_transcription, output
      └─ Final Response?
          is_final_response()
    """
    SEP = "─" * 60

    # ── Header ──────────────────────────────────────────────────────────────
    payload_label = _classify_payload(event)
    colour = _GREEN if event.is_final_response() else _CYAN
    print(f"\n  {_h(_BOLD + colour, f'┌─ Event #{index}  [{payload_label}]')}")
    print(f"  {_h(colour, SEP)}")

    # ── 1. IDENTITY & ROUTING ───────────────────────────────────────────────
    print(f"  {_h(_BOLD, '│ ── IDENTITY & ROUTING')} ─────────────────────────────")
    print(f"  │  id              = {event.id}")
    print(f"  │  invocation_id   = {event.invocation_id}")
    print(f"  │  author          = {_h(_YELLOW, str(event.author))}")
    print(f"  │  branch          = {getattr(event, 'branch', None)}")

    ts = getattr(event, "timestamp", None)
    if ts:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        print(f"  │  timestamp       = {ts:.4f}  ({dt})")
    else:
        print(f"  │  timestamp       = None")

    # node_info — tells you the agent hierarchy path for multi-agent systems
    node_info = getattr(event, "node_info", None)
    if node_info is not None:
        path = getattr(node_info, "path", "–")
        output_for = getattr(node_info, "output_for", None)
        msg_output = getattr(node_info, "message_as_output", None)
        print(f"  │  node_info.path          = {_h(_MAG, str(path))}")
        print(f"  │  node_info.output_for    = {output_for}")
        print(f"  │  node_info.message_as_output = {msg_output}")
    else:
        print(f"  │  node_info       = None")

    interaction_id = getattr(event, "interaction_id", None)
    isolation_scope = getattr(event, "isolation_scope", None)
    model_version = getattr(event, "model_version", None)
    print(f"  │  interaction_id  = {interaction_id}")
    print(f"  │  isolation_scope = {isolation_scope}")
    print(f"  │  model_version   = {model_version}")

    # ── 2. PAYLOAD CLASSIFICATION ───────────────────────────────────────────
    print(f"  {_h(_BOLD, '│ ── PAYLOAD CLASSIFICATION')} ──────────────────────────")
    print(f"  │  payload type    = {_h(_YELLOW, payload_label)}")

    # ── 3. CONTENT — ALL PARTS ─────────────────────────────────────────────
    print(f"  {_h(_BOLD, '│ ── CONTENT (all parts)')} ──────────────────────────────")
    content = getattr(event, "content", None)
    if content is None:
        print("  │  content         = None  (pure control / state event)")
    else:
        role = getattr(content, "role", "–")
        parts = getattr(content, "parts", []) or []
        print(f"  │  content.role    = {role}")
        print(f"  │  content.parts   = {len(parts)} part(s)")

        for pi, part in enumerate(parts):
            print(f"  │    Part [{pi}]:")
            text = getattr(part, "text", None)
            thought = getattr(part, "thought", None)
            fn_call = getattr(part, "function_call", None)
            fn_resp = getattr(part, "function_response", None)
            inline_d = getattr(part, "inline_data", None)
            file_d = getattr(part, "file_data", None)
            exec_res = getattr(part, "executable_code", None)
            code_res = getattr(part, "code_execution_result", None)

            if text is not None:
                preview = text.replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:200] + "…"
                label = "THOUGHT" if thought else "TEXT"
                print(f"  │      type        = {label}")
                print(f"  │      text        = {preview!r}")

            elif fn_call is not None:
                print(f"  │      type        = FUNCTION CALL")
                print(f"  │      name        = {fn_call.name!r}")
                args = dict(fn_call.args or {})
                print(f"  │      args        = {json.dumps(args, default=str)}")
                fn_id = getattr(fn_call, "id", None)
                print(f"  │      call_id     = {fn_id}")

            elif fn_resp is not None:
                print(f"  │      type        = FUNCTION RESPONSE")
                print(f"  │      name        = {fn_resp.name!r}")
                resp = getattr(fn_resp, "response", None)
                print(f"  │      response    = {json.dumps(resp, default=str)}")
                fn_id = getattr(fn_resp, "id", None)
                print(f"  │      call_id     = {fn_id}")

            elif inline_d is not None:
                print(
                    f"  │      type        = INLINE_DATA  [{getattr(inline_d, 'mime_type', '?')}]"
                )

            elif file_d is not None:
                print(
                    f"  │      type        = FILE_DATA  [{getattr(file_d, 'mime_type', '?')}]"
                )

            elif exec_res is not None:
                print(f"  │      type        = EXECUTABLE_CODE")
                print(f"  │      code        = {getattr(exec_res, 'code', '?')!r}")

            elif code_res is not None:
                print(f"  │      type        = CODE_EXECUTION_RESULT")
                print(f"  │      output      = {getattr(code_res, 'output', '?')!r}")

            else:
                print(f"  │      type        = UNKNOWN PART STRUCTURE")
                print(f"  │      raw         = {part!r}")

    # ── 4. STREAMING / COMPLETION STATE ────────────────────────────────────
    print(f"  {_h(_BOLD, '│ ── STREAMING / COMPLETION')} ───────────────────────────")
    print(f"  │  partial             = {event.partial}")
    print(f"  │  finish_reason       = {getattr(event, 'finish_reason', None)}")
    print(f"  │  turn_complete       = {getattr(event, 'turn_complete', None)}")
    print(f"  │  turn_complete_reason= {getattr(event, 'turn_complete_reason', None)}")
    print(f"  │  interrupted         = {getattr(event, 'interrupted', None)}")

    # ── 5. ACTIONS (side-effects & control-flow) ────────────────────────────
    print(f"  {_h(_BOLD, '│ ── ACTIONS (EventActions.*)')} ──────────────────────────")
    actions = getattr(event, "actions", None)
    if actions is None:
        print("  │  actions = None")
    else:
        # ── 5a. State delta ─────────────────────────────────────────────────
        sd = getattr(actions, "state_delta", None)
        if sd:
            annotated = {_annotate_state_key(k): v for k, v in sd.items()}
            print(
                f"  │  {_h(_GREEN, '✓ state_delta')}         = {json.dumps(annotated, default=str)}"
            )
        else:
            print(f"  │  state_delta          = {{}}  (no state changes)")

        # ── 5b. Artifact delta ───────────────────────────────────────────────
        ad = getattr(actions, "artifact_delta", None)
        if ad:
            print(f"  │  {_h(_GREEN, '✓ artifact_delta')}      = {ad}")
        else:
            print(f"  │  artifact_delta       = {{}}  (no artifact changes)")

        # ── 5c. Control-flow signals ─────────────────────────────────────────
        transfer = getattr(actions, "transfer_to_agent", None)
        escalate = getattr(actions, "escalate", None)
        skip_sum = getattr(actions, "skip_summarization", None)

        if transfer:
            print(
                f"  │  {_h(_RED, '⟶ transfer_to_agent')}  = {_h(_YELLOW, str(transfer))}"
            )
        else:
            print(f"  │  transfer_to_agent   = None  (no delegation)")

        if escalate:
            print(
                f"  │  {_h(_RED, '⚠ escalate')}            = {escalate}  (loop will TERMINATE)"
            )
        else:
            print(f"  │  escalate            = {escalate}")

        if skip_sum:
            print(
                f"  │  {_h(_YELLOW, 'skip_summarization')} = {skip_sum}  (raw tool result → user)"
            )
        else:
            print(f"  │  skip_summarization  = {skip_sum}")

        # ── 5d. Auth & tool confirmations ───────────────────────────────────
        rac = getattr(actions, "requested_auth_configs", {}) or {}
        if rac:
            print(
                f"  │  {_h(_MAG, '✦ requested_auth_configs')} ({len(rac)} config(s)):"
            )
            for call_id, auth_cfg in rac.items():
                print(f"  │      function_call_id = {call_id}")
                print(f"  │      auth_config      = {auth_cfg!r}")
        else:
            print(f"  │  requested_auth_configs = {{}}  (no auth requested)")

        rtc = getattr(actions, "requested_tool_confirmations", {}) or {}
        if rtc:
            print(f"  │  {_h(_MAG, '✦ requested_tool_confirmations')} ({len(rtc)}):")
            for call_id, confirmation in rtc.items():
                print(f"  │      function_call_id = {call_id}")
                print(f"  │      confirmation     = {confirmation!r}")
        else:
            print(f"  │  requested_tool_confirmations = {{}}  (no confirmations)")

        # ── 5e. Advanced / internal action fields ───────────────────────────
        compaction = getattr(actions, "compaction", None)
        end_of_agent = getattr(actions, "end_of_agent", None)
        agent_state = getattr(actions, "agent_state", None)
        rewind = getattr(actions, "rewind_before_invocation_id", None)
        route = getattr(actions, "route", None)
        render_ui = getattr(actions, "render_ui_widgets", None)
        set_model_r = getattr(actions, "set_model_response", None)

        if compaction is not None:
            # C1: Structured compaction output
            print(f"  │  compaction.start_timestamp = {compaction.start_timestamp:.4f}")
            print(f"  │  compaction.end_timestamp   = {compaction.end_timestamp:.4f}")
            duration = compaction.end_timestamp - compaction.start_timestamp
            print(f"  │  compaction.duration_sec    = {duration:.2f}s")
            cc = getattr(compaction, "compacted_content", None)
            if cc:
                parts = getattr(cc, "parts", []) or []
                print(f"  │  compaction.content_parts   = {len(parts)} part(s)")

        if end_of_agent is not None:
            print(f"  │  {_h(_MAG, 'end_of_agent')}              = {end_of_agent}")

        if agent_state is not None:
            # C6: agent_state checkpoint data
            agent_state_str = json.dumps(agent_state, default=str)[:200]
            print(f"  │  {_h(_CYAN, '✦ agent_state (checkpoint)')} = {agent_state_str}")

        if rewind is not None:
            print(f"  │  {_h(_RED, 'rewind_before_invocation_id')}  = {rewind}")
        if route is not None:
            print(f"  │  route                           = {route}")

        if render_ui is not None:
            # C4: render_ui_widgets detail
            print(
                f"  │  {_h(_MAG, '✦ render_ui_widgets')} ({len(render_ui)} widget(s)):"
            )
            for i, w in enumerate(render_ui):
                print(f"  │      [{i}] {w!r}")

        if set_model_r is not None:
            # C5: set_model_response JSON truncated
            smr_str = json.dumps(set_model_r, default=str)[:200]
            print(f"  │  {_h(_CYAN, '✦ set_model_response')} = {smr_str}")

    # ── 6. ERRORS ───────────────────────────────────────────────────────────
    err_code = getattr(event, "error_code", None)
    err_msg = getattr(event, "error_message", None)
    if err_code or err_msg:
        print(f"  {_h(_BOLD, '│ ── ERRORS')} ─────────────────────────────────────────")
        print(f"  │  {_h(_RED, 'error_code')}    = {err_code}")
        print(f"  │  {_h(_RED, 'error_message')} = {err_msg}")

    # ── 7. TOKEN USAGE ──────────────────────────────────────────────────────
    usage = getattr(event, "usage_metadata", None)
    if usage:
        print(f"  {_h(_BOLD, '│ ── TOKEN USAGE')} ────────────────────────────────────")
        prompt = getattr(usage, "prompt_token_count", 0) or 0
        candidates = getattr(usage, "candidates_token_count", 0) or 0
        thoughts = getattr(usage, "thoughts_token_count", 0) or 0
        cached = getattr(usage, "cached_content_token_count", 0) or 0
        total = getattr(usage, "total_token_count", 0) or 0
        print(f"  │  prompt_token_count         = {prompt}")
        print(f"  │  candidates_token_count     = {candidates}")
        print(
            f"  │  thoughts_token_count       = {_h(_CYAN, str(thoughts))}   ← reasoning tokens"
        )
        print(f"  │  cached_content_token_count = {cached}")
        print(f"  │  total_token_count          = {_h(_BOLD, str(total))}")

    # ── 8. LONG-RUNNING TOOL IDS ────────────────────────────────────────────
    lrti = getattr(event, "long_running_tool_ids", None)
    if lrti:
        print(
            f"  {_h(_BOLD, '│ ── LONG-RUNNING TOOL IDS')} ───────────────────────────"
        )
        print(f"  │  long_running_tool_ids = {lrti}")

    # ── 9. GROUNDING METADATA ───────────────────────────────────────────────
    gmd = getattr(event, "grounding_metadata", None)
    if gmd:
        print(
            f"  {_h(_BOLD, '│ ── GROUNDING METADATA')} ──────────────────────────────"
        )
        print(f"  │  grounding_metadata = {gmd}")

    # ── 10. CITATION METADATA ───────────────────────────────────────────────
    cit = getattr(event, "citation_metadata", None)
    if cit:
        print(
            f"  {_h(_BOLD, '│ ── CITATION METADATA')} ───────────────────────────────"
        )
        print(f"  │  citation_metadata = {cit}")

    # ── 11. CACHE METADATA ──────────────────────────────────────────────────
    cache_md = getattr(event, "cache_metadata", None)
    if cache_md:
        print(
            f"  {_h(_BOLD, '│ ── CACHE METADATA')} ──────────────────────────────────"
        )
        print(f"  │  cache_metadata = {cache_md}")

    # ── 12. LOG PROBABILITY FIELDS ──────────────────────────────────────────
    avg_logp = getattr(event, "avg_logprobs", None)
    logp_res = getattr(event, "logprobs_result", None)
    if avg_logp is not None:
        print(
            f"  {_h(_BOLD, '│ ── LOG PROBS')} ───────────────────────────────────────"
        )
        print(f"  │  avg_logprobs    = {avg_logp}")
        print(f"  │  logprobs_result = {logp_res}")

    # ── 13. CUSTOM / LIVE SESSION METADATA ──────────────────────────────────
    custom_md = getattr(event, "custom_metadata", None)
    lsru = getattr(event, "live_session_resumption_update", None)
    ls_id = getattr(event, "live_session_id", None)
    go_away = getattr(event, "go_away", None)
    in_trans = getattr(event, "input_transcription", None)
    out_trans = getattr(event, "output_transcription", None)
    output = getattr(event, "output", None)

    has_extras = any(
        x is not None
        for x in [custom_md, lsru, ls_id, go_away, in_trans, out_trans, output]
    )
    if has_extras:
        print(
            f"  {_h(_BOLD, '│ ── MISC / LIVE / CUSTOM')} ──────────────────────────────"
        )
        if custom_md is not None:
            print(f"  │  custom_metadata                 = {custom_md}")
        if lsru is not None:
            print(f"  │  live_session_resumption_update  = {lsru}")
        if ls_id is not None:
            print(f"  │  live_session_id                 = {ls_id}")
        if go_away is not None:
            print(f"  │  go_away                         = {go_away}")
        if in_trans is not None:
            print(f"  │  input_transcription             = {in_trans}")
        if out_trans is not None:
            print(f"  │  output_transcription            = {out_trans}")
        if output is not None:
            print(f"  │  output                          = {output}")

    # ── 14. FINAL VERDICT ───────────────────────────────────────────────────
    is_final = event.is_final_response()
    verdict = (
        _h(_GREEN, "YES — this is the user-facing final response")
        if is_final
        else _h(_DIM, "no  — intermediate event")
    )
    print(f"  {_h(_BOLD, '│ ── is_final_response()')} ────────────────────────────")
    print(f"  │  {verdict}")
    print(f"  {_h(colour, '└' + SEP)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4b. AGENT MANIFEST (tools command)
# ─────────────────────────────────────────────────────────────────────────────


def print_agent_manifest(agent, parent_name=None, visited=None) -> None:
    """Print a comprehensive manifest of every agent in the system."""
    if visited is None:
        visited = set()
        print(
            f"\n  {_h(_BOLD + _MAG, '╔═ AGENT MANIFEST ══════════════════════════════════════')}"
        )

    agent_name = getattr(agent, "name", str(agent))
    if agent_name in visited:
        return
    visited.add(agent_name)

    label = f"[ROOT]" if parent_name is None else f"[SUB-AGENT of {parent_name}]"
    print(f"  {_h(_BOLD, f'┌─ Agent: {agent_name}  {label}')}")

    # Model info
    model = getattr(agent, "model", None)
    if model is not None:
        if hasattr(model, "model"):
            model_str = f"LiteLlm(model={model.model!r})"
        else:
            model_str = str(model)
    else:
        model_str = "None"
    print(f"  │  model        = {model_str}")

    # Description
    desc = getattr(agent, "description", None)
    print(f"  │  description  = {desc}")

    # Output key
    output_key = getattr(agent, "output_key", None)
    print(f"  │  output_key   = {output_key}")

    # Sub-agents
    sub_agents = getattr(agent, "sub_agents", []) or []
    sub_names = [getattr(sa, "name", str(sa)) for sa in sub_agents]
    print(f"  │  sub_agents   = {sub_names}")

    # Tools
    tools = getattr(agent, "tools", []) or []
    print(f"  │  tools ({len(tools)}):")
    if not tools:
        print(f"  │    none")
    else:
        for i, tool in enumerate(tools):
            # Get underlying function
            fn = getattr(tool, "func", tool)
            sig = None
            try:
                sig = inspect.signature(fn)
                sig_str = str(sig)
            except (ValueError, TypeError):
                sig_str = "(...)"

            # Get return annotation
            ret = ""
            if sig is not None:
                try:
                    if sig.return_annotation is not inspect.Parameter.empty:
                        ret = f" → {sig.return_annotation}"
                except Exception:
                    pass

            # Get docstring first line
            doc = inspect.getdoc(fn) or ""
            first_line = doc.split("\n")[0].strip() if doc else ""

            fn_name = getattr(fn, "__name__", str(fn))
            print(f"  │    [{i}] {fn_name}{sig_str}{ret}")
            if first_line:
                print(f'  │        """{first_line}"""')

    print(f"  {_h(_BOLD, '└' + '─' * 60)}")

    # Recurse into sub-agents
    for sa in sub_agents:
        print_agent_manifest(sa, parent_name=agent_name, visited=visited)

    # Print closing border only for root
    if parent_name is None:
        print(
            f"  {_h(_BOLD + _MAG, '╚═════════════════════════════════════════════════════════')}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. TURN SUMMARY  (printed after all per-event blocks)
# ─────────────────────────────────────────────────────────────────────────────


def print_turn_summary(
    all_events: list,
    state_before: dict | None = None,
    state_after: dict | None = None,
) -> None:
    """High-level summary of the entire turn's event stream."""
    total = len(all_events)
    finals = [e for e in all_events if e.is_final_response()]
    authors = list(
        dict.fromkeys(e.author for e in all_events)
    )  # order-preserving unique

    tool_calls = []
    tool_results = []
    transfers = []
    state_deltas = {}

    for e in all_events:
        for c in e.get_function_calls() or []:
            tool_calls.append((e.author, c.name, dict(c.args or {})))
        for r in e.get_function_responses() or []:
            tool_results.append((e.author, r.name, r.response))
        a = getattr(e, "actions", None)
        if a:
            t = getattr(a, "transfer_to_agent", None)
            if t:
                transfers.append((e.author, t))
            sd = getattr(a, "state_delta", None)
            if sd:
                state_deltas.update(sd)

    total_tokens = 0
    thought_tokens = 0
    for e in all_events:
        u = getattr(e, "usage_metadata", None)
        if u:
            total_tokens += getattr(u, "total_token_count", 0) or 0
            thought_tokens += getattr(u, "thoughts_token_count", 0) or 0

    print(
        f"\n  {_h(_BOLD + _MAG, '╔═ TURN SUMMARY ══════════════════════════════════════')}"
    )
    print(f"  ║  Total events       : {total}")
    print(f"  ║  Authors seen       : {' → '.join(_h(_YELLOW, a) for a in authors)}")
    print(f"  ║  Final responses    : {len(finals)}")
    print(f"  ║  Tool calls         : {len(tool_calls)}")
    for author, name, args in tool_calls:
        print(f"  ║    [{author}] {_h(_BLUE, name)} {json.dumps(args, default=str)}")
    print(f"  ║  Tool results       : {len(tool_results)}")
    for author, name, resp in tool_results:
        status = (resp or {}).get("status", "?")
        print(f"  ║    [{author}] {name} → status={status}")
    print(f"  ║  Agent transfers    : {len(transfers)}")
    for src, dst in transfers:
        print(f"  ║    {_h(_YELLOW, src)} ──⟶  {_h(_GREEN, dst)}")
    print(f"  ║  State keys written : {list(state_deltas.keys()) or 'none'}")
    if state_deltas:
        annotated = {_annotate_state_key(k): v for k, v in state_deltas.items()}
        print(f"  ║    {json.dumps(annotated, default=str)}")

    # State diff section
    if state_before is not None and state_after is not None:
        added = {k: v for k, v in state_after.items() if k not in state_before}
        changed = {
            k: (state_before[k], state_after[k])
            for k in state_after
            if k in state_before and state_before[k] != state_after[k]
        }
        removed = {k: v for k, v in state_before.items() if k not in state_after}

        print(f"  ║  State DIFF (vs. turn start):")
        print(f"  ║    + added:   {json.dumps(added, default=str) or '{}'}")
        print(f"  ║    ~ changed: {json.dumps(changed, default=str) or '{}'}")
        print(f"  ║    - removed: {json.dumps(removed, default=str) or '{}'}")
        print(f"  ║  Full state now: {json.dumps(state_after, default=str)}")

    print(
        f"  ║  Total tokens used  : {total_tokens}  (of which thoughts: {thought_tokens})"
    )
    print(f"  {_h(_MAG, '╚═══════════════════════════════════════════════════')}")


# ─────────────────────────────────────────────────────────────────────────────
# 5b. TURN FLOW DIAGRAM (ASCII arrow diagram)
# ─────────────────────────────────────────────────────────────────────────────


def print_turn_flow(all_events: list) -> None:
    """Print a visual flow diagram showing the exact sequence of events."""
    if not all_events:
        return

    print(
        f"\n  {_h(_BOLD + _MAG, '╔═ TURN FLOW DIAGRAM ══════════════════════════════════════')}"
    )
    print(f"  ║")
    print(f"  ║  USER")
    print(f"  ║   │")
    print(f"  ║   ▼")

    current_agent = None
    depth = 0
    indent = "  ║  "

    for event in all_events:
        author = getattr(event, "author", "unknown")

        # Track agent transitions
        if author != current_agent:
            if current_agent is None:
                # First agent
                current_agent = author
                usage = getattr(event, "usage_metadata", None)
                thoughts = (
                    getattr(usage, "thoughts_token_count", 0) or 0 if usage else 0
                )
                thinks_str = (
                    f"  ── THINKS ({thoughts} thought tokens)" if thoughts > 0 else ""
                )
                print(f"  ║  [{author}]{thinks_str}")
                print(f"  ║   │")
            else:
                # New agent (transfer happened)
                current_agent = author

        # Process events
        calls = event.get_function_calls() or []
        responses = event.get_function_responses() or []
        actions = getattr(event, "actions", None)
        transfer = getattr(actions, "transfer_to_agent", None) if actions else None
        state_delta = getattr(actions, "state_delta", None) if actions else None
        is_final = event.is_final_response()

        # Tool calls
        for call in calls:
            args = dict(call.args or {})
            args_str = json.dumps(args, default=str)
            if len(args_str) > 60:
                args_str = args_str[:57] + "..."
            print(f"  ║   ├──TOOL CALL──▶  {call.name}({args_str})")
            print(f"  ║   │                    ▼")

        # Tool responses
        for resp in responses:
            resp_data = resp.response or {}
            status = resp_data.get("status", str(resp_data)[:60])
            print(f"  ║   │               Tool Result: status={status}")

        # Transfer
        if transfer:
            print(f"  ║   │")
            print(f"  ║   ├──TRANSFER──▶  {transfer}")
            print(f"  ║   │                    │")
            print(f"  ║   │               [{transfer}]")
            print(f"  ║   │                    │")

        # State delta
        if state_delta:
            sd_str = json.dumps(dict(state_delta), default=str)
            if len(sd_str) > 80:
                sd_str = sd_str[:77] + "..."
            print(f"  ║   │  STATE WRITE: {sd_str}")

        # Error
        if getattr(event, "error_code", None):
            err_code = event.error_code
            err_msg = getattr(event, "error_message", "") or ""
            print(f"  ║   └──ERROR: {err_code}: {err_msg[:60]}")

        # Final response
        if is_final:
            parts = _safe_get(event, "content", "parts", default=[]) or []
            texts = [p.text for p in parts if getattr(p, "text", None)]
            if texts:
                text = " ".join(texts)
                if len(text) > 80:
                    text = text[:77] + "..."
                print(f'  ║   └──FINAL RESPONSE: "{text}"')
            else:
                print(f"  ║   └──FINAL RESPONSE (no text content)")

    print(f"  ║")
    print(
        f"  {_h(_MAG, '╚══════════════════════════════════════════════════════════╝')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. AGENT CALLER
# ─────────────────────────────────────────────────────────────────────────────


async def call_agent(
    runner: Runner, user_input: str, session_id: str, user_id: str
) -> dict:
    global _session_stats
    content = types.Content(role="user", parts=[types.Part(text=user_input)])

    # Snapshot state before turn
    session_before = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    state_before = dict(session_before.state) if session_before else {}

    final_response_text = None
    final_event = None
    all_events = []

    print(f"\n  {_h(_BOLD, '>>> Event stream for this turn:')}")
    index = 0

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        all_events.append(event)
        describe_event(event, index)
        index += 1

        if event.is_final_response():
            final_event = event
            parts = _safe_get(event, "content", "parts", default=[]) or []
            texts = [p.text for p in parts if getattr(p, "text", None)]
            if texts:
                final_response_text = " ".join(texts)
            elif getattr(getattr(event, "actions", None), "escalate", None):
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No message.'}"
                )

    # Snapshot state after turn
    session_after = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    state_after = dict(session_after.state) if session_after else {}

    print_turn_summary(all_events, state_before=state_before, state_after=state_after)
    print_turn_flow(all_events)

    # Update cumulative session stats
    _session_stats["total_turns"] += 1
    for e in all_events:
        u = getattr(e, "usage_metadata", None)
        if u:
            _session_stats["total_tokens"] += getattr(u, "total_token_count", 0) or 0
            _session_stats["total_thought_tokens"] += (
                getattr(u, "thoughts_token_count", 0) or 0
            )
        for c in e.get_function_calls() or []:
            _session_stats["total_tool_calls"] += 1
            tool_name = c.name
            _session_stats["tool_invocation_counts"][tool_name] = (
                _session_stats["tool_invocation_counts"].get(tool_name, 0) + 1
            )
        author = getattr(e, "author", None)
        if author:
            _session_stats["agent_invocation_counts"][author] = (
                _session_stats["agent_invocation_counts"].get(author, 0) + 1
            )
        a = getattr(e, "actions", None)
        if a:
            t = getattr(a, "transfer_to_agent", None)
            if t:
                _session_stats["transfer_log"].append(
                    (e.author, t, _session_stats["total_turns"])
                )

    return {
        "type": "Event",
        "author": final_event.author if final_event else "unknown",
        "final_response_text": final_response_text,
        "final_event": final_event,
        "all_events": all_events,
        "content": getattr(final_event, "content", None) if final_event else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────


async def create_session() -> InMemorySessionService:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={"user_preference_temperature_unit": "Celsius"},
    )
    return session_service


async def main():
    try:
        session_service = await create_session()
        runner = Runner(
            session_service=session_service,
            app_name=APP_NAME,
            agent=root_agent,
        )
        print(_h(_BOLD + _GREEN, "Session created successfully!"))
        print(
            f"Agent team: root_agent → [greeting_agent, farewell_agent]\n"
            f"Commands: 'events' | 'state' | 'history' | 'tools' | 'stats' | 'exit'\n"
            f"Try: 'hi', 'weather in London', 'capital of Japan', 'bye'\n"
        )
    except Exception as e:
        print(f"Error creating session: {e}")
        return

    last_response = None

    while True:
        user_input = input(_h(_BOLD, "\nYou: ")).strip()

        if user_input.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        if not user_input:
            continue

        # ── Debug commands ─────────────────────────────────────────────────
        if user_input.lower() == "events":
            if not last_response:
                print("No events yet.")
                continue
            evts = last_response["all_events"]
            print(f"\n--- Re-dumping {len(evts)} events from last turn ---")
            for i, ev in enumerate(evts):
                describe_event(ev, i)
            continue

        if user_input.lower() == "state":
            session = await session_service.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
            )
            print(f"\n{_h(_BOLD, '--- session.state ---')}")
            print(json.dumps(dict(session.state), indent=2, default=str))
            continue

        if user_input.lower() == "history":
            session = await session_service.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
            )
            evts = getattr(session, "events", [])
            print(f"\n{_h(_BOLD, f'--- session.events ({len(evts)} total) ---')}")
            for i, ev in enumerate(evts):
                describe_event(ev, i)
            continue

        if user_input.lower() == "tools":
            print_agent_manifest(root_agent)
            continue

        if user_input.lower() == "stats":
            print(
                f"\n  {_h(_BOLD + _MAG, '╔═ CUMULATIVE SESSION STATS ══════════════════════════')}"
            )
            print(f"  ║  Turns completed      : {_session_stats['total_turns']}")
            print(
                f"  ║  Total tokens used    : {(_session_stats['total_tokens']):,}  (thoughts: {(_session_stats['total_thought_tokens']):,})"
            )
            print(f"  ║  Total tool calls     : {_session_stats['total_tool_calls']}")
            print(f"  ║  Agent invocations:")
            for agent_name, count in sorted(
                _session_stats["agent_invocation_counts"].items()
            ):
                print(f"  ║    {agent_name:<24} : {count} turns")
            print(f"  ║  Tool invocations:")
            for tool_name, count in sorted(
                _session_stats["tool_invocation_counts"].items()
            ):
                print(f"  ║    {tool_name:<24} : {count}")
            print(f"  ║  Agent transfers:")
            if _session_stats["transfer_log"]:
                for src, dst, turn in _session_stats["transfer_log"]:
                    print(
                        f"  ║    Turn {turn}: {_h(_YELLOW, src)} ──▶ {_h(_GREEN, dst)}"
                    )
            else:
                print(f"  ║    (none)")
            print(
                f"  {_h(_MAG, '╚═══════════════════════════════════════════════════')}"
            )
            continue

        # ── Normal turn ─────────────────────────────────────────────────────
        try:
            last_response = await call_agent(
                runner=runner,
                user_input=user_input,
                session_id=SESSION_ID,
                user_id=USER_ID,
            )
            print(
                f"\n{_h(_BOLD + _GREEN, 'Assistant')} "
                f"({last_response['author']}): "
                f"{last_response['final_response_text']}"
            )
        except Exception as e:
            print(f"{_h(_RED, 'Error calling agent:')} {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    print(_h(_BOLD, "Starting Mercury Multi-Agent Events Demo (exhaustive inspector)…"))
    asyncio.run(main())
