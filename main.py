"""
Mercury Agent CLI
=================
Interactive terminal chat with Mercury model and local executor tools.
"""

from __future__ import annotations

import asyncio
import os
import logging

from dotenv import load_dotenv
import warnings

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


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

# ── Imports ───────────────────────────────────────────────────────────────────
from local_executor import LocalExecutorToolkit, PermissionConfig
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types


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

toolkit = LocalExecutorToolkit(
    permission_config=PermissionConfig(auto_approve=True),
)
execute_python_code, execute_shell_command = toolkit.get_tool_functions()

root_agent = LlmAgent(
    name="mercury_agent",
    model=model,
    instruction=(
        "You are a helpful assistant with access to Python code execution "
        "and shell command tools. Use them when needed to help the user. "
        "Always explain what you're doing and show results clearly."
    ),
    tools=[execute_python_code, execute_shell_command],
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
    """Run one turn and stream response to terminal."""
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    print("Agent: ", end="", flush=True)
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
    user_id = "user_1"

    runner = InMemoryRunner(agent=root_agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )

    print("═" * 66)
    print("  Mercury Agent — Python & Shell Executor")
    print("═" * 66)
    print("  Type 'exit' or 'quit' to stop.")
    print("─" * 66)

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            await _run_prompt(runner, user_id, session.id, prompt)
        except Exception as e:
            print(f"\nError: {e}")
            print("─" * 66)


if __name__ == "__main__":
    asyncio.run(main())
