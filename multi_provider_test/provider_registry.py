"""
Provider + model registry.

- Model/provider *listing* comes from models.dev (https://models.dev/api.json),
  cached locally and refreshed periodically. This is the thing that grows to
  75+ providers / thousands of models without you writing any code for it.
- Credential *shape* (what fields does provider X need) is NOT something
  models.dev standardizes well enough to trust blindly, so it lives in the
  small static CREDENTIAL_SCHEMA table below. This table rarely changes --
  most providers just need "api_key". Only a handful need more.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

MODELS_DEV_URL = "https://models.dev/api.json"
CACHE_DIR = Path.home() / ".cache" / "ai-tui"
CACHE_FILE = CACHE_DIR / "models_registry.json"
CACHE_TTL_SECONDS = 24 * 60 * 60  # refresh once a day

# ---------------------------------------------------------------------------
# Credential schema: provider_id -> list of fields required to call it.
# Each field: key (kwarg name passed to LiteLLM), label (shown in UI),
# secret (mask input), optional (not required to save).
# Anything NOT listed here falls back to DEFAULT_SCHEMA (just an api_key).
# ---------------------------------------------------------------------------
DEFAULT_SCHEMA: list[dict[str, Any]] = [
    {"key": "api_key", "label": "API Key", "secret": True},
]

CREDENTIAL_SCHEMA: dict[str, list[dict[str, Any]]] = {
    "cloudflare": [
        {"key": "api_key", "label": "API Key", "secret": True},
        {"key": "account_id", "label": "Account ID", "secret": False},
    ],
    "azure": [
        {"key": "api_key", "label": "API Key", "secret": True},
        {"key": "api_base", "label": "Endpoint URL (api_base)", "secret": False},
        {"key": "api_version", "label": "API Version", "secret": False},
    ],
    "bedrock": [
        {"key": "aws_access_key_id", "label": "AWS Access Key ID", "secret": True},
        {
            "key": "aws_secret_access_key",
            "label": "AWS Secret Access Key",
            "secret": True,
        },
        {"key": "aws_region_name", "label": "AWS Region", "secret": False},
    ],
    "vertex_ai": [
        {"key": "vertex_project", "label": "GCP Project ID", "secret": False},
        {"key": "vertex_location", "label": "GCP Region", "secret": False},
        {
            "key": "service_account_json_path",
            "label": "Path to service-account JSON",
            "secret": False,
        },
    ],
    "watsonx": [
        {"key": "api_key", "label": "API Key", "secret": True},
        {"key": "watsonx_project_id", "label": "Project ID", "secret": False},
    ],
    "ollama": [
        {
            "key": "api_base",
            "label": "Local server URL (e.g. http://localhost:11434)",
            "secret": False,
            "optional": True,
        },
    ],
    "huggingface": [
        {"key": "api_key", "label": "API Key", "secret": True},
        {
            "key": "api_base",
            "label": "Custom endpoint (optional)",
            "secret": False,
            "optional": True,
        },
    ],
}

# ---------------------------------------------------------------------------
# Provider ID aliases: models.dev uses different IDs than what LiteLLM /
# our credential schema expects. This mapping normalizes them.
# ---------------------------------------------------------------------------
PROVIDER_ALIASES: dict[str, str] = {
    "cloudflare-workers-ai": "cloudflare",
    "cloudflare-ai-gateway": "cloudflare",
    "amazon-bedrock": "bedrock",
}


def normalize_provider_id(provider_id: str) -> str:
    """Map a models.dev provider ID to the internal ID used by credentials/LiteLLM."""
    return PROVIDER_ALIASES.get(provider_id, provider_id)


# ---------------------------------------------------------------------------
# OpenAI-compatible providers: these aren't direct LiteLLM providers.
# They need the model string prefixed with "openai/" and an api_base URL.
# Users must provide the api_base in the credential screen.
# ---------------------------------------------------------------------------
OPENAI_COMPAT_PROVIDERS: set[str] = {
    "github-copilot",
    "helicone",
    "llmgateway",
    "merge-gateway",
    "302ai",
    "jiekou",
    "privatemode-ai",
    "tinfoil",
    "vivgrid",
    "dinference",
    "cloudferro-sherlock",
    "clarifai",
}

# OpenAI-compat providers all need api_key + api_base
_OPENAI_COMPAT_SCHEMA: list[dict[str, Any]] = [
    {"key": "api_key", "label": "API Key", "secret": True},
    {
        "key": "api_base",
        "label": "API Base URL (e.g. https://api.githubcopilot.com/v1)",
        "secret": False,
    },
]
for _pid in OPENAI_COMPAT_PROVIDERS:
    CREDENTIAL_SCHEMA.setdefault(_pid, _OPENAI_COMPAT_SCHEMA)


def is_openai_compat(provider_id: str) -> bool:
    """Check if a provider needs the openai/ prefix + api_base."""
    return (
        normalize_provider_id(provider_id) in OPENAI_COMPAT_PROVIDERS
        or provider_id in OPENAI_COMPAT_PROVIDERS
    )


# Minimal built-in fallback so the app still works with zero network access.
# Real usage will overwrite this via the models.dev fetch.
_FALLBACK_REGISTRY: dict[str, Any] = {
    "openai": {"name": "OpenAI", "models": {"gpt-5": {}, "gpt-4.1": {}, "gpt-4o": {}}},
    "anthropic": {
        "name": "Anthropic",
        "models": {"claude-sonnet-4-6": {}, "claude-opus-4-8": {}},
    },
    "groq": {
        "name": "Groq",
        "models": {"llama-3.3-70b-versatile": {}, "openai/gpt-oss-120b": {}},
    },
    "google": {
        "name": "Google",
        "models": {"gemini-2.5-flash": {}, "gemini-2.5-pro": {}},
    },
    "openrouter": {"name": "OpenRouter", "models": {"mistralai/mistral-large": {}}},
    "mistral": {
        "name": "Mistral",
        "models": {"mistral-large-latest": {}, "magistral-medium-latest": {}},
    },
    "cerebras": {"name": "Cerebras", "models": {"llama-3.3-70b": {}}},
    "cloudflare": {
        "name": "Cloudflare Workers AI",
        "models": {"@cf/meta/llama-3.1-8b-instruct": {}},
    },
    "deepseek": {
        "name": "DeepSeek",
        "models": {"deepseek-chat": {}, "deepseek-reasoner": {}},
    },
    "xai": {"name": "xAI", "models": {"grok-4": {}}},
}


def _load_cache() -> dict[str, Any] | None:
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if time.time() - data.get("_fetched_at", 0) < CACHE_TTL_SECONDS:
            return data["registry"]
        return data["registry"]  # stale but usable while we try to refresh
    except Exception:
        return None


def _save_cache(registry: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps({"_fetched_at": time.time(), "registry": registry}),
        encoding="utf-8",
    )


def fetch_registry(force_refresh: bool = False) -> dict[str, Any]:
    """Returns {provider_id: {"name": ..., "models": {model_id: {...meta}}}}.

    Order of preference: fresh network fetch -> stale cache -> bundled fallback.
    """
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            # Kick off a background-ish refresh is left to the caller (async);
            # here we just return what we have instantly.
            return cached

    try:
        resp = httpx.get(MODELS_DEV_URL, timeout=10.0)
        resp.raise_for_status()
        raw = resp.json()
        registry = {
            pid: {"name": pdata.get("name", pid), "models": pdata.get("models", {})}
            for pid, pdata in raw.items()
        }
        _save_cache(registry)
        return registry
    except Exception:
        cached = _load_cache()
        if cached is not None:
            return cached
        return _FALLBACK_REGISTRY


def list_providers(registry: dict[str, Any]) -> list[tuple[str, str]]:
    """Returns [(provider_id, display_name), ...] sorted by display name."""
    items = [(pid, pdata.get("name", pid)) for pid, pdata in registry.items()]
    return sorted(items, key=lambda x: x[1].lower())


def list_models(registry: dict[str, Any], provider_id: str) -> list[str]:
    pdata = registry.get(provider_id, {})
    return sorted(pdata.get("models", {}).keys())


def schema_for(provider_id: str) -> list[dict[str, Any]]:
    return CREDENTIAL_SCHEMA.get(normalize_provider_id(provider_id), DEFAULT_SCHEMA)
