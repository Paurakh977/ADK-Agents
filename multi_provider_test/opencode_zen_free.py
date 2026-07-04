#!/usr/bin/env python3
"""
opencode_chat.py
=================

An interactive terminal chat client for OpenCode Zen models, powered by
LiteLLM.

OpenCode Zen (https://opencode.ai/docs/zen) exposes an OpenAI-compatible
Chat Completions endpoint at:

    https://opencode.ai/zen/v1/chat/completions

and a model listing endpoint at:

    https://opencode.ai/zen/v1/models

This script:

  * Discovers available models at runtime (no hardcoded model list).
  * Lets the user pick a model from an interactive menu.
  * Streams chat responses token-by-token through LiteLLM, using the
    generic OpenAI-compatible provider (``openai/<model-id>``) pointed at
    OpenCode Zen's custom ``api_base``.
  * Detects and separately renders "thinking" / "reasoning" content when a
    model streams it, using LiteLLM's normalized ``reasoning_content`` /
    ``thinking_blocks`` fields (falling back to a couple of known
    provider-specific field names for extra robustness). Models that don't
    expose reasoning are handled gracefully -- nothing is fabricated.
  * Supports mid-conversation commands: exit, quit, clear, /models, /help.
  * Retries transient network errors with backoff; never retries
    authentication errors.

Run with:

    python opencode_chat.py
    uv run opencode_chat.py

Requirements: litellm, python-dotenv, requests (see comments below).
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
import dataclasses
from typing import Any, Optional

import requests
from dotenv import load_dotenv

import litellm
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

# ---------------------------------------------------------------------------
# Terminal / encoding setup
# ---------------------------------------------------------------------------


def _setup_utf8_stdio() -> None:
    """Force UTF-8 stdout/stderr so the CLI behaves the same on Windows,
    macOS, and Linux (Windows consoles default to a legacy codepage that
    mangles box-drawing characters, emoji, etc.).
    """
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )


def _enable_windows_ansi() -> None:
    """Enable ANSI/VT100 escape sequence processing on Windows consoles.

    Modern Windows 10+ terminals support ANSI colors, but the feature must
    be turned on explicitly for classic ``cmd.exe`` / older PowerShell
    hosts. The empty ``os.system("")`` call is a well known trick that
    forces the console to initialize its VT100 support.
    """
    if sys.platform == "win32":
        os.system("")


_setup_utf8_stdio()
_enable_windows_ansi()


class Color:
    """ANSI color codes used throughout the CLI."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    LIGHT_GRAY = "\033[90m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENCODE_API_BASE = "https://opencode.ai/zen/v1"
OPENCODE_MODELS_URL = f"{OPENCODE_API_BASE}/models"
ENV_VAR_NAME = "OPENCODE_API_KEY"

MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 30

# Exceptions considered transient / worth retrying.
TRANSIENT_EXCEPTIONS = (
    APIConnectionError,
    BadGatewayError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)

# Exceptions that should never be retried (the credentials / request are
# simply wrong, retrying wastes time and hammers the API).
NON_RETRYABLE_EXCEPTIONS = (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BadRequestError,
)

litellm.drop_params = True  # silently drop kwargs a given model doesn't accept


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelInfo:
    """A single model advertised by the OpenCode Zen /models endpoint."""

    id: str
    display_name: str


@dataclasses.dataclass
class SessionState:
    """Mutable state for the currently selected model / conversation."""

    model: ModelInfo
    history: list[dict[str, str]] = dataclasses.field(default_factory=list)
    reasoning_supported: Optional[bool] = None  # None = not yet observed


# ---------------------------------------------------------------------------
# Friendly display names
# ---------------------------------------------------------------------------

# The /models endpoint only returns raw model IDs (e.g. "big-pickle",
# "deepseek-v4-flash-free"). This lookup table gives a handful of common
# ones a nicer display name identical to OpenCode's own docs. It is purely
# cosmetic -- the model LIST ITSELF always comes from the live API call
# below, never from this table, so newly added/removed models are picked
# up automatically without any code changes.
_KNOWN_DISPLAY_NAMES: dict[str, str] = {
    "big-pickle": "Big Pickle",
    "deepseek-v4-flash-free": "DeepSeek V4 Flash Free",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "mimo-v2.5-free": "MiMo-V2.5 Free",
    "north-mini-code-free": "North Mini Code Free",
    "nemotron-3-ultra-free": "Nemotron 3 Ultra Free",
    "grok-build-0.1": "Grok Build 0.1",
    "kimi-k2.7-code": "Kimi K2.7 Code",
    "kimi-k2.6": "Kimi K2.6",
    "kimi-k2.5": "Kimi K2.5",
    "minimax-m3": "MiniMax M3",
    "minimax-m2.7": "MiniMax M2.7",
    "minimax-m2.5": "MiniMax M2.5",
    "qwen3.7-max": "Qwen3.7 Max",
    "qwen3.7-plus": "Qwen3.7 Plus",
    "qwen3.6-plus": "Qwen3.6 Plus",
    "qwen3.5-plus": "Qwen3.5 Plus",
    "glm-5.2": "GLM 5.2",
    "glm-5.1": "GLM 5.1",
    "glm-5": "GLM 5",
}

# Acronyms that should stay fully upper-cased when auto-generating a
# display name for a model ID this script doesn't recognize.
_ACRONYMS = {"gpt", "glm", "m3", "v4", "v2.5"}


def humanize_model_id(model_id: str) -> str:
    """Turn a raw model id like ``deepseek-v4-flash-free`` into a
    human-friendly display name. Falls back to a light heuristic for any
    model id not present in ``_KNOWN_DISPLAY_NAMES`` (e.g. brand new
    models OpenCode adds after this script was written).
    """
    if model_id in _KNOWN_DISPLAY_NAMES:
        return _KNOWN_DISPLAY_NAMES[model_id]

    parts = re.split(r"[-_]", model_id)
    words = []
    for part in parts:
        if part.lower() in _ACRONYMS or (part.isalpha() and len(part) <= 3):
            words.append(part.upper())
        else:
            words.append(part[:1].upper() + part[1:])
    return " ".join(words)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def fetch_available_models(api_key: str) -> list[ModelInfo]:
    """Fetch the live list of models from OpenCode Zen.

    Raises ``RuntimeError`` with a friendly message on failure -- the
    caller is responsible for deciding whether to retry or exit.
    """
    headers = {"Authorization": f"Bearer {api_key}"}

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                OPENCODE_MODELS_URL,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
            _warn_and_backoff(attempt, exc)
            continue

        if response.status_code == 401:
            raise RuntimeError(
                "Authentication failed while fetching models (HTTP 401). "
                f"Check that {ENV_VAR_NAME} in your .env file is correct."
            )
        if response.status_code == 429:
            last_error = RuntimeError("Rate limited while fetching models.")
            _warn_and_backoff(attempt, last_error)
            continue
        if response.status_code >= 500:
            last_error = RuntimeError(
                f"OpenCode Zen returned a server error (HTTP {response.status_code})."
            )
            _warn_and_backoff(attempt, last_error)
            continue
        if not response.ok:
            raise RuntimeError(
                f"Failed to fetch models: HTTP {response.status_code} - {response.text[:300]}"
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Model list response was not valid JSON: {exc}") from exc

        raw_models = payload.get("data", [])
        if not raw_models:
            raise RuntimeError("OpenCode Zen returned an empty model list.")

        models = [
            ModelInfo(id=m["id"], display_name=humanize_model_id(m["id"]))
            for m in raw_models
            if "id" in m
        ]
        models.sort(key=lambda m: m.display_name.lower())
        return models

    raise RuntimeError(f"Could not fetch model list after {MAX_RETRIES} attempts: {last_error}")


def _warn_and_backoff(attempt: int, exc: Exception) -> None:
    delay = RETRY_BASE_DELAY_SECONDS * attempt
    print(
        f"{Color.YELLOW}[warning] Transient error ({exc}). "
        f"Retrying in {delay:.1f}s (attempt {attempt}/{MAX_RETRIES})...{Color.RESET}"
    )
    time.sleep(delay)


# ---------------------------------------------------------------------------
# Reasoning / thinking extraction
# ---------------------------------------------------------------------------


def _extract_thinking_text(blocks: Any) -> str:
    """Pull plain text out of a ``thinking_blocks``-style structure, which
    can be a list of dicts with a ``thinking`` or ``text`` key.
    """
    if not blocks:
        return ""
    text = ""
    try:
        for block in blocks:
            if isinstance(block, dict):
                text += block.get("thinking") or block.get("text") or ""
            else:
                text += getattr(block, "thinking", "") or getattr(block, "text", "") or ""
    except TypeError:
        return ""
    return text


def extract_delta_parts(delta: Any) -> tuple[str, str]:
    """Given a streamed chunk's ``delta`` object, return
    ``(answer_text, reasoning_text)``.

    LiteLLM normalizes most providers' reasoning output into
    ``delta.reasoning_content`` (and, for extended-thinking style models,
    ``delta.thinking_blocks``). Since OpenCode Zen proxies several
    different upstream providers behind one OpenAI-compatible endpoint,
    this function also checks a couple of raw provider-specific field
    names (``reasoning``, ``thinking``) as a defensive fallback in case a
    given upstream model isn't fully normalized by LiteLLM. Nothing is
    ever invented -- if none of these fields are present, reasoning_text
    is simply an empty string.
    """
    answer_text = getattr(delta, "content", None) or ""

    reasoning_text = getattr(delta, "reasoning_content", None) or ""

    if not reasoning_text:
        reasoning_text = getattr(delta, "reasoning", None) or ""

    if not reasoning_text:
        thinking_attr = getattr(delta, "thinking", None)
        if isinstance(thinking_attr, str):
            reasoning_text = thinking_attr

    if not reasoning_text:
        reasoning_text = _extract_thinking_text(getattr(delta, "thinking_blocks", None))

    if not reasoning_text:
        provider_fields = getattr(delta, "provider_specific_fields", None)
        if isinstance(provider_fields, dict):
            reasoning_text = (
                provider_fields.get("reasoning_content")
                or provider_fields.get("reasoning")
                or _extract_thinking_text(provider_fields.get("thinking_blocks"))
                or ""
            )

    return answer_text, reasoning_text


# ---------------------------------------------------------------------------
# Chat completion (streaming) with retry
# ---------------------------------------------------------------------------


class StreamInterrupted(Exception):
    """Raised when the stream dies partway through a response."""


def stream_chat_response(session: SessionState, api_key: str) -> None:
    """Send ``session.history`` to the model and stream the response,
    printing [THINK] / [ANSWER] sections as content arrives. Updates
    ``session.history`` with the assistant's final answer and updates
    ``session.reasoning_supported`` based on what was actually observed.
    """
    litellm_model = f"openai/{session.model.id}"

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        thinking_open = False
        answer_open = False
        saw_reasoning = False
        answer_accum = ""

        try:
            response_stream = litellm.completion(
                model=litellm_model,
                api_key=api_key,
                api_base=OPENCODE_API_BASE,
                messages=session.history,
                stream=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            for chunk in response_stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta
                if delta is None:
                    continue

                answer_text, reasoning_text = extract_delta_parts(delta)

                if reasoning_text:
                    saw_reasoning = True
                    if not thinking_open:
                        print(f"\n{Color.LIGHT_GRAY}[THINK]{Color.RESET}")
                        thinking_open = True
                        answer_open = False
                    print(f"{Color.LIGHT_GRAY}{reasoning_text}{Color.RESET}", end="", flush=True)

                if answer_text:
                    if not answer_open:
                        if thinking_open:
                            print()  # newline after thinking block
                        print(f"\n{Color.GREEN}[ANSWER]{Color.RESET}")
                        answer_open = True
                        thinking_open = False
                    print(f"{Color.GREEN}{answer_text}{Color.RESET}", end="", flush=True)
                    answer_accum += answer_text

            print()  # trailing newline once the stream finishes

            if not answer_accum:
                print(f"{Color.YELLOW}(model returned no text content){Color.RESET}")

            session.reasoning_supported = saw_reasoning or bool(session.reasoning_supported)
            session.history.append({"role": "assistant", "content": answer_accum})
            return

        except NON_RETRYABLE_EXCEPTIONS as exc:
            _print_friendly_error(exc)
            # Drop the un-answered user turn so the conversation stays valid.
            if session.history and session.history[-1]["role"] == "user":
                session.history.pop()
            return

        except TRANSIENT_EXCEPTIONS as exc:
            last_error = exc
            if answer_accum:
                # We already streamed a partial answer; don't silently
                # retry and duplicate it -- surface the interruption.
                print(
                    f"\n{Color.RED}[error] Stream interrupted mid-response: {exc}{Color.RESET}"
                )
                session.history.append({"role": "assistant", "content": answer_accum})
                return
            if attempt < MAX_RETRIES:
                _warn_and_backoff(attempt, exc)
                continue
            _print_friendly_error(exc)
            if session.history and session.history[-1]["role"] == "user":
                session.history.pop()
            return

        except Exception as exc:  # noqa: BLE001 - final safety net for unknown errors
            _print_friendly_error(exc)
            if session.history and session.history[-1]["role"] == "user":
                session.history.pop()
            return


def _print_friendly_error(exc: Exception) -> None:
    if isinstance(exc, AuthenticationError):
        print(
            f"{Color.RED}[error] Authentication failed. Double-check "
            f"{ENV_VAR_NAME} in your .env file.{Color.RESET}"
        )
    elif isinstance(exc, RateLimitError):
        print(f"{Color.RED}[error] Rate limited by OpenCode Zen: {exc}{Color.RESET}")
    elif isinstance(exc, NotFoundError):
        print(f"{Color.RED}[error] Model not found: {exc}{Color.RESET}")
    elif isinstance(exc, BadRequestError):
        print(f"{Color.RED}[error] Invalid request: {exc}{Color.RESET}")
    elif isinstance(exc, Timeout):
        print(f"{Color.RED}[error] Request timed out: {exc}{Color.RESET}")
    elif isinstance(exc, (APIConnectionError, requests.exceptions.ConnectionError)):
        print(f"{Color.RED}[error] Network connection failed: {exc}{Color.RESET}")
    else:
        print(f"{Color.RED}[error] {type(exc).__name__}: {exc}{Color.RESET}")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def print_banner() -> None:
    print(f"\n{Color.CYAN}{'=' * 25}")
    print("OpenCode Zen CLI")
    print(f"{'=' * 16}{Color.RESET}")


def print_models(models: list[ModelInfo]) -> None:
    print(f"\n{Color.CYAN}{'=' * 24}")
    print("Available Models")
    print(f"{'=' * 16}{Color.RESET}")
    for idx, model in enumerate(models, start=1):
        print(f"{idx}. {model.display_name}")


def prompt_choose_model(models: list[ModelInfo]) -> ModelInfo:
    """Interactively prompt the user to choose a model by number,
    re-prompting on invalid input.
    """
    while True:
        raw = input(f"\n{Color.YELLOW}Choose model: {Color.RESET}").strip()
        if not raw:
            continue
        if not raw.isdigit():
            print(f"{Color.RED}Please enter a number from the list above.{Color.RESET}")
            continue
        choice = int(raw)
        if 1 <= choice <= len(models):
            return models[choice - 1]
        print(f"{Color.RED}Invalid choice: {choice}. Pick 1-{len(models)}.{Color.RESET}")


def print_model_info(model: ModelInfo, reasoning_supported: Optional[bool]) -> None:
    if reasoning_supported is None:
        reasoning_str = "Unknown (auto-detected from the first response)"
    else:
        reasoning_str = "Yes" if reasoning_supported else "No (not observed)"

    print(f"\n{Color.MAGENTA}Selected:{Color.RESET} {model.display_name}")
    print(f"  Provider           : OpenCode Zen")
    print(f"  Model ID            : {model.id}")
    print(f"  Reasoning Supported : {reasoning_str}")
    print(f"  Streaming Enabled   : Yes")


def print_help() -> None:
    print(f"\n{Color.CYAN}Commands{Color.RESET}")
    print("  exit, quit   Leave the program")
    print("  clear        Clear the screen and reset the current conversation")
    print("  /models      Switch to a different model")
    print("  /help        Show this help message")
    print("  Ctrl+C       Exit immediately")


def clear_screen() -> None:
    os.system("cls" if sys.platform == "win32" else "clear")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful assistant."


def new_session(model: ModelInfo) -> SessionState:
    return SessionState(
        model=model,
        history=[{"role": "system", "content": SYSTEM_PROMPT}],
    )


def main() -> None:
    load_dotenv()
    api_key = os.getenv(ENV_VAR_NAME)

    if not api_key:
        print(
            f"{Color.RED}[error] {ENV_VAR_NAME} is not set. "
            f"Create a .env file with:\n\n  {ENV_VAR_NAME}=your_key_here\n{Color.RESET}"
        )
        sys.exit(1)

    print_banner()

    try:
        models = fetch_available_models(api_key)  # cached for the whole run
    except RuntimeError as exc:
        print(f"{Color.RED}[error] {exc}{Color.RESET}")
        sys.exit(1)

    print_models(models)
    selected = prompt_choose_model(models)
    session = new_session(selected)
    print_model_info(session.model, session.reasoning_supported)

    print(f"\n{Color.LIGHT_GRAY}Type '/help' for commands. Type 'exit' or 'quit' to leave.{Color.RESET}")

    while True:
        try:
            user_input = input(f"\n{Color.YELLOW}You:{Color.RESET}\n").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Color.CYAN}Goodbye!{Color.RESET}")
            return

        if not user_input:
            continue

        lowered = user_input.lower()

        if lowered in ("exit", "quit"):
            print(f"{Color.CYAN}Goodbye!{Color.RESET}")
            return

        if lowered == "clear":
            clear_screen()
            session = new_session(session.model)
            print_banner()
            print_model_info(session.model, session.reasoning_supported)
            continue

        if lowered == "/help":
            print_help()
            continue

        if lowered == "/models":
            print_models(models)
            selected = prompt_choose_model(models)
            session = new_session(selected)
            print_model_info(session.model, session.reasoning_supported)
            continue

        session.history.append({"role": "user", "content": user_input})
        try:
            stream_chat_response(session, api_key)
        except KeyboardInterrupt:
            print(f"\n{Color.YELLOW}[interrupted] Response cancelled.{Color.RESET}")
            if session.history and session.history[-1]["role"] == "user":
                session.history.pop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Color.CYAN}Goodbye!{Color.RESET}")
        sys.exit(0)