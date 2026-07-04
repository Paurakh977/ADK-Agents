"""
Runs a prompt through Google ADK's LlmAgent + LiteLlm, streaming back
(kind, text) chunks where kind is "think" or "answer".

This is your original script's logic, generalized so it works for ANY
provider based on its credential schema, instead of one hardcoded MODELS list.
"""

from __future__ import annotations

import logging
import tempfile
import warnings
from pathlib import Path
from typing import Any, AsyncIterator

import litellm
from litellm.exceptions import BadRequestError

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from provider_registry import normalize_provider_id, is_openai_compat

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Any provider-unsupported kwarg gets dropped instead of raising. Doesn't
# fix wrong *values* (handled by the reasoning-retry fallback below).
litellm.drop_params = True

# ---------------------------------------------------------------------------
# Optional reasoning-effort presets, matched by substring against the final
# "provider/model" string. Not required -- anything unmatched just runs with
# no override, which is fine for most models.
# ---------------------------------------------------------------------------
REASONING_PRESETS: list[tuple[str, dict]] = [
    (
        "groq/openai/gpt-oss",
        {"extra_body": {"reasoning_effort": "high", "reasoning_format": "parsed"}},
    ),
    (
        "groq/qwen/qwen3",
        {"extra_body": {"reasoning_effort": "default", "reasoning_format": "parsed"}},
    ),
    (
        "groq/deepseek-r1-distill",
        {"extra_body": {"reasoning_effort": "default", "reasoning_format": "parsed"}},
    ),
    ("openai/o1", {"reasoning_effort": "medium"}),
    ("openai/o3", {"reasoning_effort": "medium"}),
    ("openai/gpt-5", {"reasoning_effort": "medium"}),
    ("anthropic/claude", {"thinking": {"type": "enabled", "budget_tokens": 4096}}),
    ("gemini/", {"thinking": {"type": "enabled", "budget_tokens": 4096}}),
    ("mistral/magistral", {"reasoning_effort": "medium"}),
    (
        "cerebras/gemma-4",
        {"extra_body": {"reasoning_effort": "medium", "reasoning_format": "parsed"}},
    ),
    (
        "cerebras/zai-glm",
        {"extra_body": {"reasoning_effort": "medium", "reasoning_format": "parsed"}},
    ),
]


def get_reasoning_kwargs(model_string: str) -> dict:
    for pattern, kwargs in REASONING_PRESETS:
        if pattern in model_string:
            return kwargs
    return {}


def _is_reasoning_param_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "reasoning" in msg or "thinking" in msg


def build_call_kwargs(
    provider_id: str, model_id: str, creds: dict[str, Any]
) -> dict[str, Any]:
    """Turns (provider_id, model_id, saved creds) into kwargs for LiteLlm(),
    handling the providers that need more than a bare api_key.
    """
    internal_id = normalize_provider_id(provider_id)
    model_string = f"{provider_id}/{model_id}"
    kwargs: dict[str, Any] = {"model": model_string}

    if internal_id == "cloudflare":
        kwargs["api_key"] = creds["api_key"]
        kwargs["api_base"] = (
            f"https://api.cloudflare.com/client/v4/accounts/{creds['account_id']}/ai/v1"
        )
        # LiteLLM expects Cloudflare models as openai/@cf/... (OpenAI-compat)
        if model_id.startswith("@cf/"):
            model_string = f"openai/{model_id}"
            kwargs["model"] = model_string
    elif internal_id == "azure":
        kwargs["api_key"] = creds["api_key"]
        kwargs["api_base"] = creds["api_base"]
        kwargs["api_version"] = creds["api_version"]
    elif internal_id == "bedrock":
        kwargs["aws_access_key_id"] = creds["aws_access_key_id"]
        kwargs["aws_secret_access_key"] = creds["aws_secret_access_key"]
        kwargs["aws_region_name"] = creds["aws_region_name"]
    elif internal_id == "vertex_ai":
        kwargs["vertex_project"] = creds["vertex_project"]
        kwargs["vertex_location"] = creds["vertex_location"]
        sa_path = creds.get("service_account_json_path")
        if sa_path:
            kwargs["vertex_credentials"] = sa_path
    elif internal_id == "watsonx":
        kwargs["api_key"] = creds["api_key"]
        kwargs["extra_body"] = {"project_id": creds["watsonx_project_id"]}
    elif internal_id == "ollama":
        if creds.get("api_base"):
            kwargs["api_base"] = creds["api_base"]
    elif is_openai_compat(provider_id):
        # OpenAI-compatible providers: use openai/ prefix + user-provided api_base
        kwargs["api_key"] = creds.get("api_key")
        api_base = creds.get("api_base")
        # Default api_base for known OpenAI-compat providers
        if not api_base and internal_id == "opencode":
            api_base = "https://opencode.ai/zen/v1"
        kwargs["api_base"] = api_base
        model_string = f"openai/{model_id}"
        kwargs["model"] = model_string
    else:
        if creds.get("api_key"):
            kwargs["api_key"] = creds["api_key"]
        if creds.get("api_base"):
            kwargs["api_base"] = creds["api_base"]

    kwargs["_model_string"] = model_string  # convenience for callers
    return kwargs


async def stream_chat(
    *,
    provider_id: str,
    model_id: str,
    creds: dict[str, Any],
    prompt: str,
    app_name: str = "ai-tui",
    user_id: str = "local-user",
) -> AsyncIterator[tuple[str, str]]:
    """Yields ("think", text) / ("answer", text) chunks as they stream in.
    On the provider rejecting our reasoning kwargs, retries once with none --
    this is what makes unlisted models work without you knowing their exact
    accepted reasoning-param values in advance.
    """
    call_kwargs = build_call_kwargs(provider_id, model_id, creds)
    model_string = call_kwargs.pop("_model_string")
    call_kwargs.pop("model", None)
    api_key = call_kwargs.pop("api_key", None)
    reasoning_kwargs = get_reasoning_kwargs(model_string)

    for attempt, extra in enumerate([reasoning_kwargs, {}]):
        if attempt == 1 and not reasoning_kwargs:
            break

        model = LiteLlm(
            model=model_string, api_key=api_key, stream=True, **call_kwargs, **extra
        )
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
                    kind = (
                        "think" if bool(getattr(part, "thought", False)) else "answer"
                    )
                    yield kind, part.text
            return  # success
        except BadRequestError as e:
            if attempt == 0 and extra and _is_reasoning_param_error(e):
                yield (
                    "answer",
                    f"\n[retrying '{model_string}' without reasoning override]\n",
                )
                continue
            raise
