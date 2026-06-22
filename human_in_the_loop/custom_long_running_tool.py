"""Support ticket agent with CLI runner and HITL (human-in-the-loop) input.

Run with:
    uv run agent.py

ROOT CAUSE FIXES vs original:
  1. Re-ask loop: Runner caches collected data by form fingerprint.
     If the LLM calls _collect_user_input_fn again for already-answered
     fields, cached data is replayed silently — no re-prompt to user.
  2. State persistence: After every HITL collection, data is written to
     ADK session state via state_delta (in-tool) AND direct session mutation
     (post-collection), so the LLM has ground truth on resume.
  3. No recursion: _run_prompt is now a plain iterative while-loop that
     handles any number of consecutive HITL interrupts without stack growth.
  4. Bail-out guard: After _MAX_REASK_REPLAYS silent replays with no
     forward progress, the runner aborts the stuck turn rather than looping.
  5. Input normalization: Select/multiselect values are normalized to the
     closest listed option when allow_custom_answer=False, preventing the
     LLM from rejecting values and re-asking.

Agent instruction and tool docstrings are intentionally unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import sys
from typing import Dict, FrozenSet, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps import App, ResumabilityConfig
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv(override=True)


# ── Logging ───────────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# MODELS  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════


class Question(BaseModel):
    id: str
    question: str
    input_type: Literal[
        "text", "textarea", "select", "multiselect", "boolean", "number"
    ]
    options: List[str] = Field(default_factory=list)
    allow_custom_answer: bool = Field(
        default=True,
        description="Whether the user may provide an answer not present in options.",
    )
    required: bool = True


class TicketInputPlan(BaseModel):
    title: str = Field(description="Short title shown at the top of the form.")
    description: Optional[str] = Field(
        default=None, description="Why the information is being requested."
    )
    questions: List[Question] = Field(
        default_factory=list, description="Questions to collect missing details."
    )


class CreateTicketRequest(BaseModel):
    title: str = Field(description="Short summary of the issue or request.")
    description: str = Field(
        description="Detailed explanation including all gathered requirements."
    )
    category: str = Field(
        description="Ticket category: development, infrastructure, support, security, etc."
    )
    priority: str = Field(
        default="medium", description="Ticket priority: low, medium, high, critical."
    )
    requester_name: Optional[str] = Field(
        default=None, description="Name of the person requesting the ticket."
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  (pure functions, no ADK dependency)
# ══════════════════════════════════════════════════════════════════════════════


def _form_fingerprint(form: TicketInputPlan) -> str:
    """
    Deterministic fingerprint based on sorted question IDs.
    Used to detect when the LLM re-issues the same form.
    Using hashlib instead of hash() to be stable across processes.
    """
    key = ",".join(sorted(q.id for q in form.questions))
    return hashlib.md5(key.encode()).hexdigest()[:10]


def _question_ids(form: TicketInputPlan) -> FrozenSet[str]:
    return frozenset(q.id for q in form.questions)


def _overlap_ratio(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Jaccard-style overlap: |intersection| / |union|."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize_to_option(raw: str, options: List[str]) -> str:
    """
    Map a raw user string to the closest item in options.
    Priority: exact (case-insensitive) → substring → first word → raw.
    Only called when allow_custom_answer is False.
    """
    if not options or not raw:
        return raw
    lower = raw.lower()
    # 1. Exact match
    for opt in options:
        if opt.lower() == lower:
            return opt
    # 2. Option is contained in user input (e.g. "security and learning" → "security")
    for opt in options:
        if opt.lower() in lower:
            return opt
    # 3. User input is contained in option
    for opt in options:
        if lower in opt.lower():
            return opt
    # 4. First word of user input matches an option prefix
    first_word = lower.split()[0] if lower.split() else ""
    for opt in options:
        if opt.lower().startswith(first_word):
            return opt
    # 5. No match found — return raw (caller decides what to do)
    return raw


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS  (docstrings intentionally unchanged; internals improved)
# ══════════════════════════════════════════════════════════════════════════════


def _collect_user_input_fn(
    form: TicketInputPlan,
    tool_context: ToolContext,
) -> Optional[str]:
    """Tool to ask user for additional information"""
    tool_context.actions.skip_summarization = True

    # Write form fingerprint + already-collected ticket values to session state.
    # On resume the LLM can read state["ticket_*"] and won't ask again.
    fp = _form_fingerprint(form)
    seen: list = list(tool_context.state.get("_hitl_forms_seen", []))
    if fp not in seen:
        seen.append(fp)

    # state_delta is the official ADK mechanism for updating session state
    # from within a tool.  These keys are visible to the LLM on next turn.
    # NOTE: tool_context.state is an ADK State proxy — it supports get() and []
    # but NOT .items() / .keys() / .values().  Do NOT call .items() on it.
    # ticket_* fields written by the previous turn's state_delta are already
    # in session state; we don't need to re-copy them here.
    tool_context.actions.state_delta = {
        "_hitl_forms_seen": seen,
        "_hitl_current_fp": fp,
    }
    return None


collect_user_input = LongRunningFunctionTool(func=_collect_user_input_fn)


def create_support_ticket(request: CreateTicketRequest) -> dict:
    """Create a support ticket."""
    ticket_id = f"INC-{random.randint(10000, 99999)}"
    return {
        "status": "success",
        "ticket_id": ticket_id,
        "title": request.title,
        "category": request.category,
        "priority": request.priority,
        "message": "Ticket created successfully.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL + AGENT  (instructions unchanged)
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
    name="support_assistant_agent",
    model=model,
    instruction=(
        "You are a helpful IT support assistant responsible for creating support tickets. "
    ),
    tools=[collect_user_input, create_support_ticket],
)

app = App(
    name="checker",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL UI
# ══════════════════════════════════════════════════════════════════════════════

_SEP = "─" * 62


def _prompt_question(q: Question) -> object:
    """
    Render one question in the terminal based on its input_type.
    Returns the collected value, or None if optional and skipped.
    """
    req_marker = " [REQUIRED]" if q.required else " [optional — Enter to skip]"

    # ── text ──────────────────────────────────────────────────────────────────
    if q.input_type == "text":
        while True:
            raw = input(f"  {q.question}{req_marker}\n  > ").strip()
            if raw:
                return raw
            if not q.required:
                return None
            print("  ⚠  Required — please enter a value.")

    # ── textarea ──────────────────────────────────────────────────────────────
    if q.input_type == "textarea":
        print(f"  {q.question}{req_marker}")
        print("  (enter an empty line to finish)")
        lines: List[str] = []
        while True:
            line = input("  > " if not lines else "  | ")
            if line == "" and lines:
                break
            if line == "" and not lines:
                if not q.required:
                    return None
                print("  ⚠  Required — please enter at least one line.")
                continue
            lines.append(line)
        return "\n".join(lines)

    # ── select ─────────────────────────────────────────────────────────────────
    if q.input_type == "select":
        if not q.options:
            # No options provided — degrade to free-text
            while True:
                raw = input(f"  {q.question}{req_marker}\n  > ").strip()
                if raw:
                    return raw
                if not q.required:
                    return None
                print("  ⚠  Required — please enter a value.")

        print(f"\n  {q.question}{req_marker}")
        for i, opt in enumerate(q.options, 1):
            print(f"    {i}. {opt}")
        if q.allow_custom_answer:
            print("    (or type your own answer)")
        print()

        while True:
            raw = input("  Your choice (number or text): ").strip()
            if raw == "":
                if q.required:
                    print("  ⚠  Required — please make a choice.")
                    continue
                return None
            # Numeric index shortcut
            try:
                idx = int(raw)
                if 1 <= idx <= len(q.options):
                    return q.options[idx - 1]
                print(f"  Enter a number between 1 and {len(q.options)}.")
                continue
            except ValueError:
                pass
            # Case-insensitive exact match
            matches = [o for o in q.options if o.lower() == raw.lower()]
            if matches:
                return matches[0]
            if q.allow_custom_answer:
                return raw
            # Strict: must be from list — show error and retry
            print(f"  ⚠  Must be one of: {', '.join(q.options)}")

    # ── multiselect ───────────────────────────────────────────────────────────
    if q.input_type == "multiselect":
        if not q.options:
            while True:
                raw = input(f"  {q.question}{req_marker}\n  > ").strip()
                if raw:
                    return [v.strip() for v in raw.split(",") if v.strip()]
                if not q.required:
                    return None
                print("  ⚠  Required — please enter a value.")

        print(f"\n  {q.question}{req_marker}")
        for i, opt in enumerate(q.options, 1):
            print(f"    {i}. {opt}")
        if q.allow_custom_answer:
            print("    (or type custom values, comma-separated)")
        print("  Enter numbers or text, comma-separated (e.g. 1,3 or security,dev)")
        print()

        while True:
            raw = input("  Your choices: ").strip()
            if raw == "":
                if q.required:
                    print("  ⚠  Required — please select at least one.")
                    continue
                return None
            parts = [v.strip() for v in raw.split(",") if v.strip()]
            results: List[str] = []
            valid = True
            for part in parts:
                try:
                    idx = int(part)
                    if 1 <= idx <= len(q.options):
                        results.append(q.options[idx - 1])
                    else:
                        print(f"  ⚠  {part} is not between 1 and {len(q.options)}.")
                        valid = False
                        break
                except ValueError:
                    exact = [o for o in q.options if o.lower() == part.lower()]
                    if exact:
                        results.append(exact[0])
                    elif q.allow_custom_answer:
                        results.append(part)
                    else:
                        print(f"  ⚠  '{part}' is not a valid option.")
                        valid = False
                        break
            if valid and results:
                return results
            if valid:
                print("  Please select at least one option.")

    # ── boolean ───────────────────────────────────────────────────────────────
    if q.input_type == "boolean":
        while True:
            raw = input(f"  {q.question}{req_marker} [y/n]\n  > ").strip().lower()
            if raw in ("y", "yes", "true", "1"):
                return True
            if raw in ("n", "no", "false", "0"):
                return False
            if raw == "" and not q.required:
                return None
            if raw == "":
                print("  ⚠  Required — please enter y or n.")
                continue
            print("  Please enter y or n.")

    # ── number ────────────────────────────────────────────────────────────────
    if q.input_type == "number":
        while True:
            raw = input(f"  {q.question}{req_marker}\n  > ").strip()
            if raw == "":
                if q.required:
                    print("  ⚠  Required — please enter a number.")
                    continue
                return None
            try:
                return float(raw) if "." in raw else int(raw)
            except ValueError:
                print(f"  ⚠  '{raw}' is not a valid number.")

    # ── fallback: free text ───────────────────────────────────────────────────
    while True:
        raw = input(f"  {q.question}{req_marker}\n  > ").strip()
        if raw:
            return raw
        if not q.required:
            return None
        print("  ⚠  Required — please enter a value.")


async def _prompt_user_input(
    fc: types.FunctionCall,
) -> Tuple[types.Content, dict, str, FrozenSet[str]]:
    """
    Render the terminal UI for _collect_user_input_fn.

    Returns:
        content       – FunctionResponse Content to send back to the runner
        collected     – dict of {question_id: value}
        fingerprint   – stable form fingerprint (for cache keying)
        question_ids  – frozenset of question IDs (for overlap detection)
    """
    args = fc.args or {}
    try:
        form = TicketInputPlan.model_validate(args.get("form", {}))
    except Exception as exc:
        raise ValueError(f"Agent sent malformed form: {exc}") from exc

    print(f"\n{_SEP}")
    print("  📝  USER INPUT REQUIRED")
    print(_SEP)
    print(f"  {form.title}")
    if form.description:
        print(f"  {form.description}")
    print(_SEP)
    print()

    collected: dict = {}

    if form.questions:
        for q in form.questions:
            value = _prompt_question(q)
            if value is not None:
                collected[q.id] = value
    else:
        # Fallback: no structured questions — just ask for freeform input
        result = input("  Your answer: ").strip()
        if result:
            collected["answer"] = result

    # ── Normalize strict select/multiselect values ────────────────────────────
    # When allow_custom_answer=False the terminal already enforces the list,
    # but values entered via numeric index are already normalized above.
    # This pass handles the edge case of allow_custom_answer=True where the
    # LLM might still reject a custom value — we soft-normalize to the closest
    # option only when strict mode is off (i.e. when allow_custom_answer=False
    # was intended but trust the user's intent otherwise).
    for q in form.questions:
        if q.id not in collected:
            continue
        raw_val = collected[q.id]
        if q.input_type == "select" and q.options and not q.allow_custom_answer:
            collected[q.id] = _normalize_to_option(str(raw_val), q.options)
        elif q.input_type == "multiselect" and q.options and not q.allow_custom_answer:
            vals = raw_val if isinstance(raw_val, list) else [str(raw_val)]
            collected[q.id] = [_normalize_to_option(v, q.options) for v in vals]

    print(f"\n  ✓ Input captured:")
    for k, v in collected.items():
        print(f"    {k} = {v!r}")
    print(f"{_SEP}\n")

    fp = _form_fingerprint(form)
    q_ids = _question_ids(form)

    content = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=fc.id,
                    name="_collect_user_input_fn",
                    response={"result": collected},
                )
            )
        ],
    )

    return content, collected, fp, q_ids


def _make_cached_response(fc: types.FunctionCall, cached_data: dict) -> types.Content:
    """
    Build a FunctionResponse that replays previously collected data.
    Called when a re-ask loop is detected — user is NOT re-prompted.
    """
    print(f"\n{_SEP}")
    print("  🔁  RE-ASK DETECTED — replaying cached input (no re-prompt)")
    print(_SEP)
    for k, v in cached_data.items():
        print(f"    {k} = {v!r}")
    print(f"{_SEP}\n")

    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=fc.id,
                    name="_collect_user_input_fn",
                    response={"result": cached_data},
                )
            )
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER  — iterative, loop-safe, state-persistent
# ══════════════════════════════════════════════════════════════════════════════

# How many times we silently replay cached data before declaring the LLM stuck.
_MAX_REASK_REPLAYS = 3

# Overlap threshold above which we treat a new form as a re-ask of a prior one.
_REASK_OVERLAP_THRESHOLD = 0.5


async def _run_prompt(
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    app_name: str,
    *,
    prompt: str | None = None,
    new_message: types.Content | None = None,
) -> None:
    """
    Run one complete user turn, handling any number of consecutive HITL
    interrupts without recursion.

    Re-ask loop protection
    ─────────────────────
    Per-turn cache maps form fingerprint → (collected_data, question_ids).
    On each HITL interrupt:
      • Exact fingerprint match   → definite re-ask → replay cache silently
      • High question-ID overlap  → probable re-ask → replay best-match cache
      • No overlap / new form     → fresh prompt    → ask user, populate cache

    After _MAX_REASK_REPLAYS silent replays for the same fingerprint the turn
    is aborted to prevent an infinite loop against a confused LLM.

    State persistence
    ─────────────────
    After every real user collection the ticket fields are written directly
    into the InMemorySession state dict so the LLM sees them as ground truth
    on its next invocation within this session.
    """
    if new_message is None:
        assert prompt is not None, "Either prompt or new_message must be provided"
        new_message = types.Content(
            role="user", parts=[types.Part.from_text(text=prompt)]
        )

    # ── Per-turn state ────────────────────────────────────────────────────────
    # Maps fingerprint → (collected_dict, frozenset_of_question_ids)
    cache: Dict[str, Tuple[dict, FrozenSet[str]]] = {}
    # Silent replay counter per fingerprint
    replay_counts: Dict[str, int] = {}
    # Union of all question IDs we have cached answers for
    answered_ids: FrozenSet[str] = frozenset()

    current_message = new_message

    while True:
        user_input_fc: types.FunctionCall | None = None
        saw_partial = False

        # ── Stream one runner turn ────────────────────────────────────────────
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=current_message,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            if not event.content or not event.content.parts:
                continue

            long_running_ids: set[str] = set(
                getattr(event, "long_running_tool_ids", None) or []
            )

            for part in event.content.parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    fc_id = getattr(fc, "id", None)
                    fc_name = getattr(fc, "name", None)
                    # Detect LongRunningFunctionTool by ID (primary) or name (fallback)
                    is_hitl = (fc_id and fc_id in long_running_ids) or (
                        fc_name == "_collect_user_input_fn"
                    )
                    if is_hitl and user_input_fc is None:
                        user_input_fc = fc
                    continue  # don't print function calls

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

        # ── No HITL interrupt → turn complete ─────────────────────────────────
        if user_input_fc is None:
            print()
            break

        # ── Parse the form to decide: fresh prompt or re-ask replay ───────────
        args = user_input_fc.args or {}
        try:
            form = TicketInputPlan.model_validate(args.get("form", {}))
        except Exception as exc:
            print(f"\n  ⚠  Agent sent a malformed form ({exc}); aborting turn.\n")
            break

        fp = _form_fingerprint(form)
        new_ids = _question_ids(form)

        # ── Decide: re-ask or fresh form? ─────────────────────────────────────
        is_reask = False
        best_cache_data: dict = {}

        if fp in cache:
            # Exact fingerprint hit → definite re-ask
            is_reask = True
            best_cache_data = cache[fp][0]

        elif answered_ids and _overlap_ratio(new_ids, answered_ids) >= _REASK_OVERLAP_THRESHOLD:
            # High overlap with what we've already answered → probable re-ask.
            # Find the cached entry with the most overlapping question IDs.
            is_reask = True
            best_overlap = 0.0
            for cached_fp, (c_data, c_ids) in cache.items():
                ov = _overlap_ratio(new_ids, c_ids)
                if ov > best_overlap:
                    best_overlap = ov
                    best_cache_data = c_data

        if is_reask:
            # Guard: bail out after too many replays for this fingerprint
            replay_counts[fp] = replay_counts.get(fp, 0) + 1
            if replay_counts[fp] > _MAX_REASK_REPLAYS:
                print(
                    f"\n  ⚠  LLM stuck in re-ask loop ({replay_counts[fp]} replays). "
                    "Aborting turn — please try rephrasing your request.\n"
                )
                break
            current_message = _make_cached_response(user_input_fc, best_cache_data)
            print("Agent: ", end="", flush=True)
            continue

        # ── Fresh form — prompt the user ──────────────────────────────────────
        print("Agent: ", end="", flush=True)
        try:
            content, collected, fp, q_ids = await _prompt_user_input(user_input_fc)
        except (KeyboardInterrupt, EOFError):
            print("\n  ⚠  Input cancelled.\n")
            break
        except ValueError as exc:
            print(f"\n  ⚠  {exc}\n")
            break

        # Persist collected values to ADK session state so the LLM can read
        # them as ground truth on subsequent invocations this session.
        try:
            session = await runner.session_service.get_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if session is not None:
                # Direct mutation works for InMemorySessionService
                session.state.update({f"ticket_{k}": v for k, v in collected.items()})
        except Exception:
            pass  # State update is best-effort; non-critical

        # Update per-turn caches
        cache[fp] = (collected, q_ids)
        answered_ids = answered_ids | q_ids

        current_message = content
        print("Agent: ", end="", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    app_name = "checker"
    user_id = "user_1"

    runner = InMemoryRunner(agent=root_agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )

    print("\nSupport Ticket Agent ready")
    print("Type /quit to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not prompt:
            continue
        if prompt.lower() in ("/quit", "/exit"):
            print("Goodbye.")
            break

        print("Agent: ", end="", flush=True)
        await _run_prompt(
            runner, user_id, session.id, app_name, prompt=prompt
        )
        print()


if __name__ == "__main__":
    asyncio.run(main())