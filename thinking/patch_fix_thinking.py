from __future__ import annotations

import asyncio
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.genai import types

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
import warnings
import logging

# ---------------------------------------------------------------------------
# Monkey-patch: Map Groq's "reasoning" field to litellm's "reasoning_content".
#
# Groq API returns:  {"delta": {"reasoning": "...", "channel": "reasoning"}}
# litellm Delta expects: {"delta": {"reasoning_content": "..."}}
#
# The patch target is OpenAIChatCompletionStreamingHandler.chunk_parser
# (litellm.llms.openai.chat.gpt_transformation) — this is the actual
# handler used for Groq streaming via BaseLLMHTTPHandler.
# ---------------------------------------------------------------------------

from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler as _OrigHandler,
)

if not getattr(_OrigHandler.chunk_parser, "_groq_reasoning_patched", False):
    _orig_chunk_parser = _OrigHandler.chunk_parser

    def _patched_chunk_parser(self, chunk: dict):
        # Groq sends "reasoning" in delta, but litellm Delta expects "reasoning_content"
        try:
            choices = chunk.get("choices", [])
            for choice in choices:
                delta = choice.get("delta", {})
                if "reasoning" in delta and "reasoning_content" not in delta:
                    delta["reasoning_content"] = delta.pop("reasoning")
        except Exception:
            pass
        return _orig_chunk_parser(self, chunk)

    _patched_chunk_parser._groq_reasoning_patched = True
    _OrigHandler.chunk_parser = _patched_chunk_parser
# ---------------------------------------------------------------------------

load_dotenv()

DEBUG_RAW_CHUNKS = os.getenv("DEBUG_RAW_CHUNKS") == "1"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s -\n%(message)s",
)
_adk_log = logging.getLogger("google_adk")
_adk_log.setLevel(logging.DEBUG if DEBUG_RAW_CHUNKS else logging.WARNING)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_model = LiteLlm(
    model="groq/qwen/qwen3-32b",
    api_key=GROQ_API_KEY,
    stream=True,
    extra_body={
        # "reasoning_effort": "high",
    },
)

root_agent = LlmAgent(
    name="root_agent",
    model=groq_model,
    instruction="You are a helpful assistant with access to tools.",
)


async def _run_prompt(
    *,
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    prompt: str,
) -> None:
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    thinking_open = False
    answer_open = False
    saw_any_thought = False
    saw_any_text = False

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            if not part.text:
                continue

            if part.thought:
                if not thinking_open:
                    if answer_open:
                        print("\033[0m")
                        answer_open = False
                    print("\033[33m\033[1mThinking:\033[0m ", end="", flush=True)
                    thinking_open = True
                print("\033[33m" + part.text + "\033[0m", end="", flush=True)
                saw_any_thought = True
            else:
                if not answer_open:
                    if thinking_open:
                        print("\033[0m")
                        thinking_open = False
                    print("Agent: ", end="", flush=True)
                    answer_open = True
                print(part.text, end="", flush=True)
                saw_any_text = True

    if thinking_open or answer_open:
        print()

    if not saw_any_thought and not saw_any_text:
        print("(no response)")
    elif not saw_any_thought:
        print("(no thinking tokens were returned by the model/provider for this turn)")

    print("------------------------------------")


async def main() -> None:
    app_name = "litellm_streaming_demo"
    user_id = "user_1"
    runner = InMemoryRunner(agent=root_agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )

    print('Interactive chat started. Type "exit" or "quit" to stop.')
    print("------------------------------------")

    loop = asyncio.get_event_loop()

    while True:
        prompt = await loop.run_in_executor(None, input, "You: ")
        prompt = prompt.strip()

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        await _run_prompt(
            runner=runner,
            user_id=user_id,
            session_id=session.id,
            prompt=prompt,
        )


if __name__ == "__main__":
    asyncio.run(main())
from __future__ import annotations

import asyncio
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.genai import types

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
import warnings
import logging

# ---------------------------------------------------------------------------
# Monkey-patch: Map Groq's "reasoning" field to litellm's "reasoning_content".
#
# Groq API returns:  {"delta": {"reasoning": "...", "channel": "reasoning"}}
# litellm Delta expects: {"delta": {"reasoning_content": "..."}}
#
# The patch target is OpenAIChatCompletionStreamingHandler.chunk_parser
# (litellm.llms.openai.chat.gpt_transformation) — this is the actual
# handler used for Groq streaming via BaseLLMHTTPHandler.
# ---------------------------------------------------------------------------

from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler as _OrigHandler,
)

if not getattr(_OrigHandler.chunk_parser, "_groq_reasoning_patched", False):
    _orig_chunk_parser = _OrigHandler.chunk_parser

    def _patched_chunk_parser(self, chunk: dict):
        # Groq sends "reasoning" in delta, but litellm Delta expects "reasoning_content"
        try:
            choices = chunk.get("choices", [])
            for choice in choices:
                delta = choice.get("delta", {})
                if "reasoning" in delta and "reasoning_content" not in delta:
                    delta["reasoning_content"] = delta.pop("reasoning")
        except Exception:
            pass
        return _orig_chunk_parser(self, chunk)

    _patched_chunk_parser._groq_reasoning_patched = True
    _OrigHandler.chunk_parser = _patched_chunk_parser
# ---------------------------------------------------------------------------

load_dotenv()

DEBUG_RAW_CHUNKS = os.getenv("DEBUG_RAW_CHUNKS") == "1"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s -\n%(message)s",
)
_adk_log = logging.getLogger("google_adk")
_adk_log.setLevel(logging.DEBUG if DEBUG_RAW_CHUNKS else logging.WARNING)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_model = LiteLlm(
    model="groq/qwen/qwen3-32b",
    api_key=GROQ_API_KEY,
    stream=True,
    extra_body={
        # "reasoning_effort": "high",
    },
)

root_agent = LlmAgent(
    name="root_agent",
    model=groq_model,
    instruction="You are a helpful assistant with access to tools.",
)


async def _run_prompt(
    *,
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    prompt: str,
) -> None:
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    thinking_open = False
    answer_open = False
    saw_any_thought = False
    saw_any_text = False

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            if not part.text:
                continue

            if part.thought:
                if not thinking_open:
                    if answer_open:
                        print("\033[0m")
                        answer_open = False
                    print("\033[33m\033[1mThinking:\033[0m ", end="", flush=True)
                    thinking_open = True
                print("\033[33m" + part.text + "\033[0m", end="", flush=True)
                saw_any_thought = True
            else:
                if not answer_open:
                    if thinking_open:
                        print("\033[0m")
                        thinking_open = False
                    print("Agent: ", end="", flush=True)
                    answer_open = True
                print(part.text, end="", flush=True)
                saw_any_text = True

    if thinking_open or answer_open:
        print()

    if not saw_any_thought and not saw_any_text:
        print("(no response)")
    elif not saw_any_thought:
        print("(no thinking tokens were returned by the model/provider for this turn)")

    print("------------------------------------")


async def main() -> None:
    app_name = "litellm_streaming_demo"
    user_id = "user_1"
    runner = InMemoryRunner(agent=root_agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )

    print('Interactive chat started. Type "exit" or "quit" to stop.')
    print("------------------------------------")

    loop = asyncio.get_event_loop()

    while True:
        prompt = await loop.run_in_executor(None, input, "You: ")
        prompt = prompt.strip()

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        await _run_prompt(
            runner=runner,
            user_id=user_id,
            session_id=session.id,
            prompt=prompt,
        )


if __name__ == "__main__":
    asyncio.run(main())
