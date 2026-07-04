from __future__ import annotations

import asyncio
import os
import re
import sys
import io
import logging
import warnings

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
import litellm
from litellm.exceptions import BadRequestError

from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner

load_dotenv()

# Alias Cloudflare Workers AI vars to what LiteLLM expects
if os.getenv("CLOUDFLARE_WORKERS_AI_API_KEY") and not os.getenv("CLOUDFLARE_API_KEY"):
    os.environ["CLOUDFLARE_API_KEY"] = os.environ["CLOUDFLARE_WORKERS_AI_API_KEY"]
if os.getenv("CLOUDFLARE_WORKERS_AI_ACCOUNT_ID") and not os.getenv(
    "CLOUDFLARE_ACCOUNT_ID"
):
    os.environ["CLOUDFLARE_ACCOUNT_ID"] = os.environ["CLOUDFLARE_WORKERS_AI_ACCOUNT_ID"]

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Blanket safety net: any provider-unsupported kwarg gets silently dropped
# instead of raising. This does NOT fix wrong *values* (e.g. reasoning_effort
# ="high" on a model that only accepts "none"/"default") -- only fixes
# params the provider doesn't recognize *at all*. Wrong values still need
# the retry-fallback below.
litellm.drop_params = True


# ---------------------------------------------------------------------------
# 1. OPTIONAL presets: LiteLLM normalizes the reasoning OUTPUT (reasoning_
#    content/thinking_blocks) the same way for every provider -- that part
#    is genuinely generic. It does NOT normalize the reasoning INPUT knob:
#    every provider/model family has its own accepted param + values, and
#    there's no API to ask "what values does this model accept" ahead of
#    time. This table is a convenience for models you use often. It is
#    matched by substring against the model string, checked top to bottom.
#    Anything not matched here just runs with no reasoning override, which
#    is fine -- most reasoning models reason by default anyway.
# ---------------------------------------------------------------------------
REASONING_PRESETS: list[tuple[str, dict]] = [
    # Groq gpt-oss family: low / medium / high (default medium)
    (
        "groq/openai/gpt-oss",
        {"extra_body": {"reasoning_effort": "high", "reasoning_format": "parsed"}},
    ),
    # Groq qwen3 family: only none / default
    (
        "groq/qwen/qwen3",
        {"extra_body": {"reasoning_effort": "default", "reasoning_format": "parsed"}},
    ),
    # Groq deepseek-r1-distill family: always reasons, no effort param needed
    (
        "groq/deepseek-r1-distill",
        {"extra_body": {"reasoning_effort": "default", "reasoning_format": "parsed"}},
    ),
    # OpenAI o-series / gpt-5.x reasoning models
    ("openai/o1", {"reasoning_effort": "medium"}),
    ("openai/o3", {"reasoning_effort": "medium"}),
    ("openai/gpt-5", {"reasoning_effort": "medium"}),
    # Anthropic extended thinking (via LiteLLM)
    ("anthropic/claude", {"thinking": {"type": "enabled", "budget_tokens": 4096}}),
    # Gemini thinking budget
    ("gemini/", {"thinking": {"type": "enabled", "budget_tokens": 4096}}),
    # Mistral Magistral reasoning models
    ("mistral/magistral-medium", {"reasoning_effort": "medium"}),
    # Cerebras reasoning models (reasoning disabled by default, must enable via extra_body)
    (
        "cerebras/gemma-4",
        {"extra_body": {"reasoning_effort": "medium", "reasoning_format": "parsed"}},
    ),
    (
        "cerebras/zai-glm",
        {"extra_body": {"reasoning_effort": "medium", "reasoning_format": "parsed"}},
    ),
    # Cloudflare Workers AI reasoning models (via OpenAI-compat endpoint)
    ("openai/@cf/moonshotai/kimi-k2.6", {"reasoning_effort": "medium"}),
    (
        "openai/@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        {"reasoning_effort": "medium"},
    ),
    ("openai/@cf/qwen/qwen3-30b-a3b-fp8", {"reasoning_effort": "medium"}),
]


def get_reasoning_kwargs(model_name: str) -> dict:
    """Best-effort preset lookup. Returns {} if nothing matches (that's fine)."""
    for pattern, kwargs in REASONING_PRESETS:
        if pattern in model_name:
            return kwargs
    return {}


def _is_reasoning_param_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "reasoning" in msg or "thinking" in msg


def build_model(
    model_name: str,
    api_key: str | None,
    reasoning_kwargs: dict,
    api_base: str | None = None,
) -> LiteLlm:
    extra = dict(reasoning_kwargs)
    if api_base:
        extra["api_base"] = api_base
    return LiteLlm(
        model=model_name,
        api_key=api_key,
        stream=True,
        **extra,
    )


async def run_prompt_with_fallback(
    *,
    model_name: str,
    api_key: str | None,
    app_name: str,
    user_id: str,
    prompt: str,
    api_base: str | None = None,
) -> None:
    """Runs one prompt. If the provider rejects our reasoning kwargs (wrong
    value for this specific model), automatically retries once with no
    reasoning override at all -- this is what makes it work for models you
    haven't added to REASONING_PRESETS, without you having to know their
    exact accepted values in advance.
    """
    reasoning_kwargs = get_reasoning_kwargs(model_name)

    for attempt, kwargs in enumerate([reasoning_kwargs, {}]):
        if attempt == 1 and not reasoning_kwargs:
            # nothing to fall back from, first attempt already had no overrides
            break

        model = build_model(model_name, api_key, kwargs, api_base=api_base)
        agent = LlmAgent(
            name="root_agent",
            model=model,
            instruction="You are a helpful assistant with access to tools.",
            tools=[],
        )
        runner = InMemoryRunner(agent=agent, app_name=app_name)
        session = await runner.session_service.create_session(
            app_name=app_name, user_id=user_id
        )

        content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

        LIGHT_GRAY = "\033[90m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        thinking_seen_partial = False
        answer_seen_partial = False
        thinking_final_text = ""
        answer_final_text = ""
        thinking_open = False
        answer_open = False
        saw_anything = False

        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=content,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
                if not event.content or not event.content.parts:
                    continue

                for part in event.content.parts:
                    if not part.text:
                        continue
                    is_thought = bool(getattr(part, "thought", False))

                    if event.partial:
                        if is_thought:
                            if not thinking_open:
                                if answer_open:
                                    print(RESET, end="", flush=True)
                                print(
                                    f"\n{LIGHT_GRAY}[THINK] Thinking: ",
                                    end="",
                                    flush=True,
                                )
                                thinking_open, answer_open = True, False
                            print(part.text, end="", flush=True)
                            thinking_seen_partial = True
                        else:
                            if not answer_open:
                                if thinking_open:
                                    print(RESET)
                                print(f"\n{GREEN}[ANSWER] Answer: ", end="", flush=True)
                                answer_open, thinking_open = True, False
                            print(part.text, end="", flush=True)
                            answer_seen_partial = True
                        saw_anything = True
                    else:
                        if is_thought:
                            thinking_final_text += part.text
                        else:
                            answer_final_text += part.text

            if not thinking_seen_partial and thinking_final_text:
                print(
                    f"\n{LIGHT_GRAY}[THINK] Thinking: {thinking_final_text}",
                    end="",
                    flush=True,
                )
                saw_anything = True
            if not answer_seen_partial and answer_final_text:
                if thinking_open or thinking_final_text:
                    print(RESET)
                print(
                    f"\n{GREEN}[ANSWER] Answer: {answer_final_text}", end="", flush=True
                )
                saw_anything = True

            print(RESET)
            if not saw_anything:
                print("(no text response)")
            print("------------------------------------")
            return  # success, don't fall through to retry

        except BadRequestError as e:
            if attempt == 0 and kwargs and _is_reasoning_param_error(e):
                print(
                    f"\n[warning] '{model_name}' rejected reasoning kwargs "
                    f"({kwargs}); retrying without them...\n"
                )
                continue  # retry with {} on next loop iteration
            raise


async def run_model_test(
    model_name: str,
    api_key: str,
    app_name: str,
    user_id: str,
    prompt: str,
    api_base: str | None = None,
) -> None:
    """Run a single model test with colored thinking output."""
    LIGHT_GRAY = "\033[90m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    print(f"\n{'=' * 60}")
    print(f"{CYAN}[MODEL] Model: {model_name}{RESET}")
    print(f"{'=' * 60}")

    reasoning_kwargs = get_reasoning_kwargs(model_name)

    for attempt, kwargs in enumerate([reasoning_kwargs, {}]):
        if attempt == 1 and not reasoning_kwargs:
            break

        model = build_model(model_name, api_key, kwargs, api_base=api_base)
        agent = LlmAgent(
            name="root_agent",
            model=model,
            instruction="You are a helpful assistant.",
            tools=[],
        )
        runner = InMemoryRunner(agent=agent, app_name=app_name)
        session = await runner.session_service.create_session(
            app_name=app_name, user_id=user_id
        )

        content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

        thinking_seen_partial = False
        answer_seen_partial = False
        thinking_final_text = ""
        answer_final_text = ""
        thinking_open = False
        answer_open = False
        saw_anything = False

        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=content,
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
                if not event.content or not event.content.parts:
                    continue

                for part in event.content.parts:
                    if not part.text:
                        continue
                    is_thought = bool(getattr(part, "thought", False))

                    if event.partial:
                        if is_thought:
                            if not thinking_open:
                                if answer_open:
                                    print(RESET, end="", flush=True)
                                print(
                                    f"\n{LIGHT_GRAY}[THINK] Thinking: ",
                                    end="",
                                    flush=True,
                                )
                                thinking_open, answer_open = True, False
                            print(part.text, end="", flush=True)
                            thinking_seen_partial = True
                        else:
                            if not answer_open:
                                if thinking_open:
                                    print(RESET)
                                print(f"\n{GREEN}[ANSWER] Answer: ", end="", flush=True)
                                answer_open, thinking_open = True, False
                            print(part.text, end="", flush=True)
                            answer_seen_partial = True
                        saw_anything = True
                    else:
                        if is_thought:
                            thinking_final_text += part.text
                        else:
                            answer_final_text += part.text

            if not thinking_seen_partial and thinking_final_text:
                print(
                    f"\n{LIGHT_GRAY}[THINK] Thinking: {thinking_final_text}",
                    end="",
                    flush=True,
                )
                saw_anything = True
            if not answer_seen_partial and answer_final_text:
                if thinking_open or thinking_final_text:
                    print(RESET)
                print(
                    f"\n{GREEN}[ANSWER] Answer: {answer_final_text}", end="", flush=True
                )
                saw_anything = True

            print(RESET)
            if not saw_anything:
                print("(no text response)")
            return

        except BadRequestError as e:
            if attempt == 0 and kwargs and _is_reasoning_param_error(e):
                print(
                    f"\n[warning] '{model_name}' rejected reasoning kwargs ({kwargs}); retrying..."
                )
                continue
            raise


async def main() -> None:
    app_name = "litellm_thinking_demo"
    user_id = "user_1"

    _cf_account = os.getenv("CLOUDFLARE_WORKERS_AI_ACCOUNT_ID", "")
    _cf_base = (
        f"https://api.cloudflare.com/client/v4/accounts/{_cf_account}/ai/v1"
        if _cf_account
        else None
    )

    MODELS = [
        ("groq/openai/gpt-oss-120b", os.getenv("GROQ_API_KEY"), None),
        ("groq/qwen/qwen3-32b", os.getenv("GROQ_API_KEY"), None),
        ("groq/llama-3.3-70b-versatile", os.getenv("GROQ_API_KEY"), None),
        ("mistral/magistral-medium-latest", os.getenv("MISTRAL_API_KEY"), None),
        ("gemini/gemini-2.5-flash", os.getenv("GOOGLE_API_KEY"), None),
        ("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",os.getenv("OPENROUTER_API_KEY"),None,),
        ("cerebras/gemma-4-31b", os.getenv("CEREBRAS_API_KEY"), None),
        ("cerebras/zai-glm-4.7", os.getenv("CEREBRAS_API_KEY"), None),
        ("openai/@cf/moonshotai/kimi-k2.6", os.getenv("CLOUDFLARE_API_KEY"), _cf_base),
        ("openai/@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",os.getenv("CLOUDFLARE_API_KEY"),_cf_base,),
        ("openai/@cf/qwen/qwen3-30b-a3b-fp8",os.getenv("CLOUDFLARE_API_KEY"),_cf_base,),
    ]

    prompt = " tell me a perfect joke"

    print("\n" + "=" * 60)
    print("[TEST] MULTI-MODEL TEST: Sending 'tell me a joke' to all models")
    print("=" * 60)

    for model_name, api_key, api_base in MODELS:
        if not api_key:
            print(f"\n[SKIP] {model_name} - API key not found")
            continue
        try:
            await run_model_test(
                model_name, api_key, app_name, user_id, prompt, api_base=api_base
            )
        except Exception as e:
            print(f"\n[ERROR] {model_name}: {e}")

    print("\n" + "=" * 60)
    print("[DONE] All models tested!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
