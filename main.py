"""
Mercury Agent CLI — improved
=============================
Interactive terminal agent with a robust bash execution harness.

Key improvements over the original:
  • Single unified shell tool  (Python runs via python3 inside bash)
  • Risk-assessed permission gate with Allow-once / Session / Deny
  • Proper timeout handling (SIGTERM → SIGKILL)
  • 1 MB output cap with truncation notice
  • Working directory persists across calls (bare `cd` updates session cwd)
  • Platform-aware shell selection (cmd.exe on Windows, SHELL on POSIX)
  • Interactive-command guard (vim, python REPL, ssh, etc. auto-rejected)
  • Structured output dict the LLM can reason about reliably

Permission levels (configure via PermissionConfig):
  SAFE     — auto-allow  (ls, cat, echo, grep, …)
  MODERATE — ask user    (rm, mv, git push, pip install, …)
  HIGH     — ask user    (sudo, dd, killall, force-push, …)
  CRITICAL — ask user*   (rm -rf, curl|sh, fork bomb, mkfs, …)
             *set deny_critical=True to auto-deny without asking
"""

from __future__ import annotations

import asyncio
import os
import logging
import sys

from dotenv import load_dotenv
import warnings

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


# ── Logging — only show LLM Request/Response blocks ───────────────────────────
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


# ── Imports ───────────────────────────────────────────────────────────────────
from local_executor import LocalExecutorToolkit, PermissionConfig, RiskLevel
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types


# ══════════════════════════════════════════════════════════════════════════════
# PERMISSION CONFIG
# ══════════════════════════════════════════════════════════════════════════════
#
# Adjust these to match your risk tolerance:
#
#   auto_approve=True           — never ask, allow everything  (CI/testing only)
#   min_ask_level=MODERATE      — ask for MODERATE, HIGH, CRITICAL
#   min_ask_level=HIGH          — silently allow MODERATE, only ask for HIGH+
#   deny_critical=True          — auto-deny CRITICAL without asking
#
PERMISSION_CFG = PermissionConfig(
    auto_approve=False,
    min_ask_level=RiskLevel.MODERATE,
    deny_critical=False,
)


# ══════════════════════════════════════════════════════════════════════════════
# TOOLKIT SETUP
# ══════════════════════════════════════════════════════════════════════════════

toolkit = LocalExecutorToolkit(
    workspace_dir=".",          # cwd of the agent session
    permission_config=PERMISSION_CFG,
    # shell=None,               # auto-detects: /bin/sh on POSIX, cmd.exe on Windows
)

# Primary tool: handles ALL shell commands.
# Python code → python3 -c "..." or run_python_code below.
execute_shell_command = toolkit.get_tool_function()

# Optional convenience wrapper for multi-line Python scripts.
# The agent can also just use execute_shell_command with python3 -c.
run_python_code = toolkit.get_python_tool()


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
    name="mercury_agent",
    model=model,
    instruction="""
You are a helpful assistant with access to shell execution tools.

## execute_shell_command — PRIMARY TOOL
Use this for all shell operations:
  • File operations:     ls, cat, cp, find, grep, sed, awk, …
  • Git:                 git status, git diff, git log --oneline, …
  • Python:              python3 -c "print('hello')" or python3 script.py
  • Package management:  pip install X, npm install, …
  • Build systems:       make, npm run build, cargo build, …
  • Data processing:     jq, curl (to get data), xargs, …
  • System inspection:   ps, df, du, env, uname, …

## run_python_code — FOR MULTI-LINE SCRIPTS
Use this when you need to run a multi-line Python program without
escaping. Write clean Python and pass it as the `code` argument.

## Working directory
A bare `cd /path` command updates the session working directory.
Subsequent commands automatically run from the new location.
Use the `workdir` argument for a one-off directory override.

## Long-running commands
Default timeout is 2 minutes. For builds or downloads that take
longer, pass `timeout_ms=600000` (10 minutes maximum).
If a command times out, retry with a larger timeout.

## Output truncation
Very large outputs are capped at 1 MB. The `truncated` field in the
result tells you if this happened. Pipe through `head`, `tail`, or
`grep` to get the relevant section.

## Permission denials
If the user denies a command, do not retry it automatically.
Explain what you were trying to do and ask the user for guidance.

## Rules
- Always check `success` and `exit_code` in the result before proceeding.
- Show relevant output to the user; do not just say "it worked".
- For errors (exit_code != 0), show the full output so we can debug.
- Break complex tasks into steps and confirm each step succeeded.
""",
    tools=[
        execute_shell_command,
        run_python_code,
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

async def _run_prompt(
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    prompt: str,
) -> None:
    """Run one conversation turn and stream the response to the terminal."""
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    print(f"Agent: ", end="", flush=True)
    saw_partial = False

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            text = getattr(part, "text", None)
            if not text:
                continue
            if event.partial:
                print(text, end="", flush=True)
                saw_partial = True
            elif not saw_partial:
                print(text, end="", flush=True)

    print()
    print("─" * 66)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    app_name = "mercury_agent_cli"
    user_id  = "user_1"

    runner  = InMemoryRunner(agent=root_agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id,
    )

    print("═" * 66)
    print("  Mercury Agent  —  Shell + Python Executor")
    print("═" * 66)
    print(f"  Workspace  : {toolkit.workspace_dir}")
    print(f"  Shell      : {toolkit._runner.shell}")
    print(f"  Ask level  : {PERMISSION_CFG.min_ask_level.name}+")
    print(f"  Auto-deny  : {'CRITICAL' if PERMISSION_CFG.deny_critical else 'off'}")
    print("─" * 66)
    print("  Tools  : execute_shell_command  |  run_python_code")
    print("  Type 'exit' or 'quit' to stop.")
    print("─" * 66)

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "/quit", "/exit"):
            print("Goodbye!")
            break

        try:
            await _run_prompt(runner, user_id, session.id, prompt)
        except Exception as exc:
            print(f"\n[Runner error] {exc}")
            print("─" * 66)


if __name__ == "__main__":
    asyncio.run(main())