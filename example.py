"""
example_usage.py
================
Demonstrates every integration pattern for local_executor.py.

Covers:
  1. Direct tool execution (sync + async)
  2. ADK agent integration
  3. LangChain integration hint
  4. Permission configuration variants
  5. Rule persistence
  6. Mercury agent with local executor tools (streaming & non-streaming)

Run standalone to smoke-test without an agent:
    python example_usage.py
"""

from __future__ import annotations

import asyncio
import sys
import os, logging, warnings
from dotenv import load_dotenv

load_dotenv()

# ─── import the harness ────────────────────────────────────────────────────────
from local_executor import (
    LocalExecutorToolkit,
    PythonExecutorTool,
    ShellExecutorTool,
    PermissionSystem,
    PermissionConfig,
    PermissionRule,
    PermissionEffect,
    CodeInput,
    ShellInput,
    Outcome,
)

# ─── Mercury agent imports ─────────────────────────────────────────────────────
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 0 — Smoke test (no agent, no permission prompts)
# ══════════════════════════════════════════════════════════════════════════════


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


def demo_direct_execution():
    """Run both tools directly, permission auto-approved (like CI/testing)."""
    print("\n" + "═" * 66)
    print("  DEMO: Direct execution with auto_approve=True")
    print("═" * 66)

    toolkit = LocalExecutorToolkit(
        permission_config=PermissionConfig(auto_approve=True),
    )

    # ── Python code executor ──────────────────────────────────────────────────
    result = toolkit.python_tool.execute(
        CodeInput(code="import math; print(math.factorial(10))")
    )
    print(f"\n[Python] outcome : {result.outcome.value}")
    print(f"[Python] output  : {result.output.strip()}")
    assert result.outcome == Outcome.OK
    assert "3628800" in result.output

    # ── Shell executor ────────────────────────────────────────────────────────
    cmd = (
        "echo Hello from the shell"
        if sys.platform != "win32"
        else "echo Hello from the shell"
    )
    result = toolkit.shell_tool.execute(ShellInput(command=cmd))
    print(f"\n[Shell] outcome  : {result.outcome.value}")
    print(f"[Shell] output   : {result.output.strip()}")
    assert result.outcome == Outcome.OK

    # ── Timeout test ──────────────────────────────────────────────────────────
    result = toolkit.python_tool.execute(
        CodeInput(code="import time; time.sleep(60)", timeout_ms=500)
    )
    print(f"\n[Timeout] outcome : {result.outcome.value}")
    print(f"[Timeout] timed_out: {result.timed_out}")
    assert result.outcome == Outcome.TIMEOUT

    print("\n  ✓ All direct-execution assertions passed.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 1 — ADK integration (drop-in tools=[...])
# ══════════════════════════════════════════════════════════════════════════════


def demo_adk_tool_functions():
    """
    Shows how to get plain callables for ADK.
    No ADK import needed here — just shows the function signatures.
    """
    print("═" * 66)
    print("  DEMO: ADK tool functions")
    print("═" * 66)

    toolkit = LocalExecutorToolkit(
        permission_config=PermissionConfig(auto_approve=True),
    )

    execute_python_code, execute_shell_command = toolkit.get_tool_functions()

    # Calling them exactly like ADK would call them
    result = execute_python_code(code="print((5 + 7) * 3)")
    print(f"\n[ADK Python] {result}")
    assert result["outcome"] == "OUTCOME_OK"
    assert "36" in result["output"]

    result = execute_shell_command(command="echo shell works")
    print(f"[ADK Shell]  {result}")
    assert result["outcome"] == "OUTCOME_OK"

    print("""
  ── To wire into a real ADK agent ──────────────────────────────────
  from local_executor import LocalExecutorToolkit, PermissionConfig
  from google.adk.agents.llm_agent import LlmAgent

  toolkit = LocalExecutorToolkit(require_permission=True)
  execute_python_code, execute_shell_command = toolkit.get_tool_functions()

  root_agent = LlmAgent(
      name        = "my_agent",
      model       = model,
      instruction = "You can run Python code and shell commands.",
      tools       = [execute_python_code, execute_shell_command],
  )
  ────────────────────────────────────────────────────────────────────
""")


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 2 — Permission system: manual rules + persistence
# ══════════════════════════════════════════════════════════════════════════════


def demo_permission_rules():
    """Shows manual rule injection and JSON persistence."""
    print("═" * 66)
    print("  DEMO: Permission rules and persistence")
    print("═" * 66)

    perm = PermissionSystem(
        PermissionConfig(
            # Shell commands blocked completely
            denied_actions=["bash"],
            # Python always allowed silently
            allowed_actions=["python_executor"],
        )
    )

    python_tool = PythonExecutorTool(permission_system=perm, require_permission=True)
    shell_tool = ShellExecutorTool(permission_system=perm, require_permission=True)

    # Python → allowed without prompting
    r = python_tool.execute(CodeInput(code="print('auto allowed')"))
    print(f"\n[Python - auto allowed] outcome: {r.outcome.value}")
    assert r.outcome == Outcome.OK

    # Shell → denied without prompting
    r = shell_tool.execute(ShellInput(command="ls"))
    print(f"[Shell  - denied      ] outcome: {r.outcome.value}")
    assert r.outcome == Outcome.DENIED

    # Manually inject a saved "always" rule (simulates user clicking "Always")
    perm2 = PermissionSystem()  # fresh system, no config
    shell_cmd = "ls *" if sys.platform != "win32" else "dir *"
    perm2.load_saved_rules_direct(
        [
            PermissionRule(
                action="bash", resource=shell_cmd, effect=PermissionEffect.ALLOW
            ),
        ]
    )
    shell2 = ShellExecutorTool(permission_system=perm2, require_permission=True)
    actual_cmd = "ls ." if sys.platform != "win32" else "dir ."
    r = shell2.execute(ShellInput(command=actual_cmd))
    print(f"[Shell  - saved allow  ] outcome: {r.outcome.value}")
    assert r.outcome == Outcome.OK

    # Persist rules to JSON
    import tempfile

    perm_file = os.path.join(tempfile.gettempdir(), "_demo_perm_rules.json")
    perm2.save_rules(perm_file)
    perm3 = PermissionSystem()
    perm3.load_rules(perm_file)
    print(f"\n  Restored {len(perm3.saved_rules)} rule(s) from JSON.")
    assert len(perm3.saved_rules) == 1

    print("\n  ✓ Permission rule assertions passed.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 3 — Async usage
# ══════════════════════════════════════════════════════════════════════════════


async def demo_async():
    """Async execute (non-blocking for async agent loops)."""
    print("═" * 66)
    print("  DEMO: Async execution")
    print("═" * 66)

    toolkit = LocalExecutorToolkit(
        permission_config=PermissionConfig(auto_approve=True),
    )

    result = await toolkit.python_tool.execute_async(
        CodeInput(code="import sys; print(sys.version.split()[0])")
    )
    print(f"\n[Async Python] Python version: {result.output.strip()}")
    assert result.outcome == Outcome.OK

    result = await toolkit.shell_tool.execute_async(
        ShellInput(command="echo async shell ok")
    )
    print(f"[Async Shell ] {result.output.strip()}")
    assert result.outcome == Outcome.OK

    print("\n  ✓ Async assertions passed.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 4 — Advisory external directory warning
# ══════════════════════════════════════════════════════════════════════════════


def demo_advisory_warnings():
    """
    Shows the advisory external-directory scan.
    Accessing /etc/hosts generates a warning but does NOT block.
    """
    print("═" * 66)
    print("  DEMO: Advisory external directory warnings")
    print("═" * 66)

    toolkit = LocalExecutorToolkit(
        permission_config=PermissionConfig(auto_approve=True),
    )

    if sys.platform == "win32":
        cmd = r"type C:\Windows\System32\drivers\etc\hosts"
    else:
        cmd = "cat /etc/hosts"

    result = toolkit.shell_tool.execute(ShellInput(command=cmd))
    print(f"\n  Warnings ({len(result.warnings)}):")
    for w in result.warnings:
        print(f"    ⚠  {w}")
    print(f"  Outcome  : {result.outcome.value}  (not blocked — advisory only)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 5 — Interactive terminal permission (the HITL flow)
# ══════════════════════════════════════════════════════════════════════════════


def demo_interactive_permission():
    """
    Demonstrates the live terminal permission prompt.
    Skipped when stdin is not a tty (CI environments).
    """
    if not sys.stdin.isatty():
        print("  [Skipped: interactive permission demo requires a terminal]\n")
        return

    print("═" * 66)
    print("  DEMO: Interactive HITL permission prompt")
    print("  (You will be asked to approve a Python execution)")
    print("═" * 66)

    toolkit = LocalExecutorToolkit(require_permission=True)
    result = toolkit.python_tool.execute(CodeInput(code="print(42)"))
    print(f"\n  Result: {result.outcome.value} → {result.output.strip()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Patch PermissionSystem with a helper method used in demo above
# ══════════════════════════════════════════════════════════════════════════════


def _load_saved_rules_direct(self, rules: list[PermissionRule]) -> None:
    """Directly inject pre-built rules (for testing)."""
    import threading

    with self._lock:
        self._saved_rules = list(rules)


PermissionSystem.load_saved_rules_direct = _load_saved_rules_direct


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 6 — Mercury Agent with Local Executor Tools
# ══════════════════════════════════════════════════════════════════════════════

# Mercury model configuration (same as streaming examples)
mercury_model = LiteLlm(
    model="openai/mercury-2",
    api_key=os.getenv("MERCURY_API_KEY"),
    api_base="https://api.inceptionlabs.ai/v1",
    max_tokens=8000,
    stream=False,
    extra_body={"reasoning_effort": "high"},
)


def create_mercury_agent(auto_approve: bool = False) -> LlmAgent:
    """
    Create a Mercury agent with local executor tools.

    Args:
        auto_approve: If True, skip permission prompts (for testing/CI)

    Returns:
        LlmAgent configured with Mercury model and local executor tools
    """
    toolkit = LocalExecutorToolkit(
        permission_config=PermissionConfig(auto_approve=auto_approve),
    )
    execute_python_code, execute_shell_command = toolkit.get_tool_functions()

    agent = LlmAgent(
        name="mercury_agent",
        model=mercury_model,
        instruction=(
            "You are a helpful assistant with access to Python code execution "
            "and shell command tools. Use them when needed to help the user. "
            "Always explain what you're doing and show results clearly."
        ),
        tools=[execute_python_code, execute_shell_command],
    )
    return agent


async def demo_mercury_agent_chat():
    """Interactive chat with Mercury agent using local executor tools."""
    print("═" * 66)
    print("  DEMO: Mercury Agent with Local Executor Tools")
    print("═" * 66)

    agent = create_mercury_agent(auto_approve=True)

    app_name = "mercury_local_executor_demo"
    user_id = "user_1"

    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )

    print("Mercury Agent ready! Type 'exit' or 'quit' to stop.")
    print("Try: 'Calculate factorial of 20', 'List files in current directory'")
    print("─" * 66)

    loop = asyncio.get_event_loop()

    while True:
        try:
            prompt = await loop.run_in_executor(None, input, "You: ")
            prompt = prompt.strip()

            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )

            print("Agent: ", end="", flush=True)

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=content,
            ):
                if not event.content or not event.content.parts:
                    continue
                text = "".join(part.text for part in event.content.parts if part.text)
                if text:
                    print(text, end="", flush=True)

            print()
            print("─" * 66)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("─" * 66)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("\n🔧  local_executor — integration demos\n")

    demo_direct_execution()
    demo_adk_tool_functions()
    demo_permission_rules()
    asyncio.run(demo_async())
    demo_advisory_warnings()
    demo_interactive_permission()

    # Run Mercury agent chat demo
    print("\n" + "═" * 66)
    print("  Starting Mercury Agent chat demo...")
    print("═" * 66)
    asyncio.run(demo_mercury_agent_chat())

    print("═" * 66)
    print("  ✅  All demos complete.")
    print("═" * 66 + "\n")


if __name__ == "__main__":
    main()
