"""
Local storage for the API keys / creds the user types into the TUI.

Stored at ~/.config/ai-tui/credentials.json, chmod 600. This is a local,
single-user desktop TUI (like OpenCode itself) -- if you later turn this
into a multi-user web backend, swap this module for a DB table keyed by
user_id and encrypt values at rest; the rest of the app doesn't need to change.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".config" / "ai-tui"
CRED_FILE = CONFIG_DIR / "credentials.json"


def _load() -> dict[str, dict[str, Any]]:
    if not CRED_FILE.exists():
        return {}
    try:
        return json.loads(CRED_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict[str, dict[str, Any]]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CRED_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(CRED_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except Exception:
        pass


def list_configured_providers() -> list[str]:
    return sorted(_load().keys())


def get_credentials(provider_id: str) -> dict[str, Any] | None:
    return _load().get(provider_id)


def save_credentials(provider_id: str, creds: dict[str, Any]) -> None:
    data = _load()
    data[provider_id] = creds
    _save(data)


def remove_provider(provider_id: str) -> None:
    data = _load()
    data.pop(provider_id, None)
    _save(data)


def is_configured(provider_id: str) -> bool:
    return provider_id in _load()